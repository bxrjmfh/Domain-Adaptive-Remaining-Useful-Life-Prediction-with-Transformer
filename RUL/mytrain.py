from mydataset import TRANSFORMER_DATA_MINDS,TRANSFORMER_ALL_DATA_MINDS
from mymodel import mymodel, Discriminator, backboneDiscriminator
from myloss import advLoss
from tqdm import tqdm
import random
import numpy
import os 
import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Model,ParameterTuple

target= 'FD002'
source = 'FD003'
epoches = 240
os.chdir('/Domain-Adaptive-Remaining-Useful-Life-Prediction-with-Transformer/')
batch_size = 1000
a = 0.1
b = 0.5

class OneStepCell(nn.Cell):
    def __init__(self, network, optimizer):
        super(OneStepCell, self).__init__(auto_prefix=False)
        self.net, self.D1, self.D2 = network
        self.weights = ParameterTuple(self.net.trainable_params()
                                      ,self.D1.trainable_params()
                                      ,self.D2.trainable_params())
        self.optimizer = optimizer
        self.Loss = nn.MSELoss()
        self.grad = ops.GradOperation(get_by_list=True)
        self.grad2 = ops.GradOperation(get_by_list=True)
    def construct(self, s_d, t_d):
        s_input, s_lb, s_msk = s_d[0], s_d[1], s_d[2]
        t_input, t_msk = t_d[0], t_d[2]
        s_features, s_out = net(s_input, s_msk)
        t_features, t_out = net(t_input, t_msk) # [bts, seq_len, feature_num]
        s_out.squeeze_(2)
        t_out.squeeze_(2)
        loss1 = Loss(s_out, s_lb)
        loss1_sum += loss1
        s_domain_bkb = D2(s_features)
        t_domain_bkb = D2(t_features)
        s_domain_out = D1(s_out)
        t_domain_out = D1(t_out)
        fea_loss = advLoss(s_domain_bkb.squeeze(1), t_domain_bkb.squeeze(1))
        out_loss = advLoss(s_domain_out.squeeze(1), t_domain_out.squeeze(1))
        bkb_sum +=fea_loss
        out_sum +=out_loss
        loss = loss1 + a*fea_loss + b*out_loss
        # how to grad ?????
        # grad what?????
        grads = self.grad(self.network, weights)(data, label)
        return ops.depend(loss, self.optimizer(grads))

def train(source_list, target_list,Loss,Opt):
    minn = 999
    net.set_train()
    

    for e in tqdm(range(epoches)):
        al, tot = 0, 0
        random.shuffle(source_list)
        random.shuffle(target_list)
        source_iter, target_iter = iter(source_list), iter(target_list)
        loss2_sum, loss1_sum = 0, 0
        bkb_sum, out_sum = 0, 0
        cnt = 0
        # 数据集的差异：
        # https://www.mindspore.cn/docs/migration_guide/zh-CN/r1.5/api_mapping/pytorch_diff/DataLoader.html
        # 实现采样器，随后数据集准备，最后得到iterator
        sampler = ds.RandomSampler()
        # prepare data 
        t_dataset = ds.GeneratorDataset(t_data,sampler=sampler, random=True)
        s_dataset = ds.GeneratorDataset(s_data,sampler=sampler, random=True)
        s_dataset = s_dataset.batch(batch_size=batch_size)
        s_iter = s_dataset.create_dict_iterator()
        t_dataset = t_dataset.batch(batch_size=batch_size)
        t_iter = t_dataset.create_dict_iterator()
        l = min(len(s_iter), len(t_iter))
        for _ in range(l):
            s_d, t_d = next(s_iter), next(t_iter)
            s_input, s_lb, s_msk = s_d[0], s_d[1], s_d[2]
            t_input, t_msk = t_d[0], t_d[2]
            s_features, s_out = net(s_input, s_msk)
            t_features, t_out = net(t_input, t_msk) # [bts, seq_len, feature_num]
            s_out.squeeze_(2)
            t_out.squeeze_(2)
            loss1 = Loss(s_out, s_lb)
            loss1_sum += loss1
            cnt += 1
            # jump if type
            s_domain_bkb = D2(s_features)
            t_domain_bkb = D2(t_features)
            s_domain_out = D1(s_out)
            t_domain_out = D1(t_out)
            fea_loss = advLoss(s_domain_bkb.squeeze(1), t_domain_bkb.squeeze(1))
            out_loss = advLoss(s_domain_out.squeeze(1), t_domain_out.squeeze(1))
            bkb_sum +=fea_loss
            out_sum +=out_loss
            loss = loss1 + a*fea_loss + b*out_loss




seq_len = 70
source_list = numpy.loadtxt("save/"+source+"/train"+source+".txt", dtype=str).tolist()
target_list = numpy.loadtxt("save/"+target+"/train"+target+".txt", dtype=str).tolist()
valid_list = numpy.loadtxt("save/"+target+"/test"+target+".txt", dtype=str).tolist()
a_list = numpy.loadtxt("save/"+target+"/valid"+target+".txt", dtype=str).tolist()
target_test_names = valid_list + a_list
minl = min(len(source_list), len(target_list))
net = mymodel(max_len=seq_len,batch_size=batch_size) 
D1 = Discriminator(seq_len)
D2 = backboneDiscriminator(seq_len)
Loss = nn.MSELoss()
Opt = nn.SGD([net.trainable_params,D1.trainable_params,D2.trainable_params]
             ,learning_rate=0.02)
train(source_list=source_list,target_list=target_list,Loss=Loss,Opt=Opt)
s_data = TRANSFORMER_ALL_DATA_MINDS(source_list, seq_len)
t_data = TRANSFORMER_ALL_DATA_MINDS(target_list, seq_len)
t_data_test = TRANSFORMER_ALL_DATA_MINDS(target_test_names, seq_len)

# prepare model





