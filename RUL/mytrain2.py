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

import numpy as np

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Model
from mindspore import dataset as ds
from mindspore.nn import LossBase
from mindspore.common.initializer import Normal
from mindspore.train.callback import LossMonitor

seq_len = 70
target= 'FD002'
source = 'FD003'
epoches = 240
os.chdir('/Domain-Adaptive-Remaining-Useful-Life-Prediction-with-Transformer/')
batch_size = 1000
a = 0.1
b = 0.5


def prepareData(source_list,target_list,target_test_names):
    s_data = TRANSFORMER_ALL_DATA_MINDS(source_list, seq_len)
    t_data = TRANSFORMER_ALL_DATA_MINDS(target_list, seq_len)
    t_data_test = TRANSFORMER_ALL_DATA_MINDS(target_test_names, seq_len)
    return s_data,t_data,t_data_test


# class MultipleLoss(LossBase):
#     def __init__(self,source, target, reduction='mean'):
#         super(MultipleLoss, self).__init__(reduction)
#         self.mseLoss = nn.MSELoss()
#         self.feaLoss = advLoss(source, target)
#         self.outLoss = advLoss(source, target)
#         self.a = 0.1
#         self.b = 0.5
#         self.allLoss = 0.0

#     def construct(self, s_r, s_lb, s_bkb, t_bkb, s_out, t_out):
#         loss1 = self.mseLoss(s_r, s_lb)
#         loss2 = self.feaLoss(s_bkb, t_bkb)
#         loss3 = self.outLoss(s_out, t_out)
#         return loss1 + self.a*loss2 + self.b*loss3
    
class MywithLossCell(nn.Cell):
    def __init__(self,net, D1, D2,loss_fn, auto_prefix=False):
        super(MywithLossCell, self).__init__()
        self._net = net
        self._D1 = D1
        self._D2 = D2
        self._loss_fn = loss_fn

    def construct(self,s_input, s_msk, t_input, t_msk,s_lb):
        s_features, s_out = self.net(s_input, s_msk)
        t_features, t_out = self.net(t_input, t_msk)
        s_out.squeeze(2)
        t_out.squeeze(2)
        s_domain_bkb = self.D2(s_features)
        t_domain_bkb = self.D2(t_features)
        s_domain_out = self.D1(s_out)
        t_domain_out = self.D1(t_out)
        return self._loss_fn(s_out,s_lb
                            ,s_domain_bkb.squeeze(1), t_domain_bkb.squeeze(1)
                            ,s_domain_out.squeeze(1), t_domain_out.squeeze(1))
    

seq_len = 70
target= 'FD002'
source = 'FD003'
epoches = 240
os.chdir('/Domain-Adaptive-Remaining-Useful-Life-Prediction-with-Transformer/')
batch_size = 1000
a = 0.1
b = 0.5
source_list = numpy.loadtxt("save/"+source+"/train"+source+".txt", dtype=str).tolist()
target_list = numpy.loadtxt("save/"+target+"/train"+target+".txt", dtype=str).tolist()
valid_list = numpy.loadtxt("save/"+target+"/test"+target+".txt", dtype=str).tolist()
a_list = numpy.loadtxt("save/"+target+"/valid"+target+".txt", dtype=str).tolist()
target_test_names = valid_list + a_list
minl = min(len(source_list), len(target_list))
s_data,t_data,t_data_test = prepareData(source_list,target_list,target_test_names)

class MERGED_DATA():
    def __init__(self,s_data,t_data) -> None:
        self.s_data = s_data
        self.t_data = t_data
    
    def __len__(self):
        return min(len(self.s_data),len(self.s_data))
    
    def __getitem__(self,index):
        return self.s_data[index]+self.t_data[index]
    
sampler = ds.RandomSampler()
all_data = MERGED_DATA(s_data,t_data)
# t_dataset = ds.GeneratorDataset(t_data,sampler=sampler,
#                                 column_names=['t_input', 't_nouse', 't_msk'])
# s_dataset = ds.GeneratorDataset(s_data,sampler=sampler,
#                                 column_names=['s_input', 's_lb', 's_msk'])

dataset = ds.GeneratorDataset(all_data,sampler=sampler,column_names=['s_input', 's_lb', 's_msk','t_input', 't_nouse', 't_msk'])
dataset.batch(batch_size)

class MultipleLoss(LossBase):
    def __init__(self, reduction='mean'):
        super(MultipleLoss, self).__init__(reduction)
        self.mseLoss = nn.MSELoss()
        self.feaLoss = advLoss()
        self.outLoss = advLoss()
        self.a = 0.1
        self.b = 0.5
        self.allLoss = 0.0

    def construct(self, s_r, s_lb, s_bkb, t_bkb, s_out, t_out):
        loss1 = self.mseLoss(s_r, s_lb)
        loss2 = self.feaLoss(s_bkb, t_bkb)
        loss3 = self.outLoss(s_out, t_out)
        return loss1 + self.a*loss2 + self.b*loss3

loss_func = MultipleLoss()
net = mymodel(max_len=seq_len,batch_size=1000)
D1 = Discriminator(seq_len)
D2 = backboneDiscriminator(seq_len)
loss_net = MywithLossCell(net,D1,D2,loss_func)
opt = nn.SGD(net.trainable_params()+D1.trainable_params()+D2.trainable_params()
             ,learning_rate=0.02)
model = Model(network=loss_net, optimizer=opt)
# FORMAT two dataset into one.
model.train(epoch=10, train_dataset=dataset, callbacks=[LossMonitor()])