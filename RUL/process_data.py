#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import math
import numpy as np
from mindspore import dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Model
from mindspore import dataset as ds
from mindspore.nn import LossBase
from mindspore.common.initializer import Normal
from mindspore.train.callback import LossMonitor
from mindspore import set_context,context
from mindspore.train.callback import Callback
from mindspore import Tensor

set_context(mode=context.GRAPH_MODE,device_target='GPU', device_id=0, save_graphs=True,
            save_graphs_path="/Domain-Adaptive-Remaining-Useful-Life-Prediction-with-Transformer/RUL/ir_files",
            save_graph_dot=True)
# set_context(mode=context.PYNATIVE_MODE,device_target="GPU")

# In[ ]:





# In[ ]:


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

    
class MywithLossCell(nn.Cell):
    def __init__(self,net, D1, D2,loss_fn, auto_prefix=False):
        super(MywithLossCell, self).__init__()
        self.net = net
        self.D1 = D1
        self.D2 = D2
        self.loss_fn = loss_fn

    def construct(self,s_input, s_lb,s_msk, t_input=None, t_msk=None):
        s_features, s_out = self.net(s_input, s_msk)
        t_features, t_out = self.net(t_input, t_msk)
        s_out = s_out.squeeze(2)
        t_out = t_out.squeeze(2)
        s_domain_bkb = self.D2(s_features[0])
        t_domain_bkb = self.D2(t_features[0])
        s_domain_out = self.D1(s_out)
        t_domain_out = self.D1(t_out)
        return self.loss_fn(s_out,s_lb
                            ,s_domain_bkb.squeeze(1), t_domain_bkb.squeeze(1)
                            ,s_domain_out.squeeze(1), t_domain_out.squeeze(1))
    


class EvalCallBack(Callback):
    # 进行评估的类，实例化后传入训练的callback中进行计算
    # ref: https://www.cnblogs.com/skytier/p/16638657.html
    # result_eval = {"epoch": [], "acc": [], "loss": []}
    # eval_cb = EvalCallBack(net, ds_val, EVAL_PER_EPOCH, result_eval)
    def __init__(self, net, eval_per_epoch, epoch_per_eval):
        self.net = net
        self.eval_per_epoch = eval_per_epoch
        self.epoch_per_eval = epoch_per_eval
        # epoch_per_eval：记录验证模型的精度和相应的epoch数，其数据形式为{"epoch": [], "acc": []}

    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_per_epoch == 0:
            # begin eval:
            tot = 0
            for i in target_test_names:
                pred_sum, pred_cnt = Tensor(np.zeros(800)), Tensor(np.zeros(800))
                # 记录结果 （800维？）
                valid_data = TRANSFORMER_DATA_MINDS(i, seq_len)
                # 变换一项结果
                data_len = len(valid_data)
                sampler = ds.RandomSampler()
                valid_dataset = ds.GeneratorDataset(valid_data,sampler=sampler,column_names=['input','label','msk'])
                valid_dataset = valid_dataset.batch(batch_size=len(valid_data),drop_remainder=True)
                vd_iter = valid_dataset.create_tuple_iterator()
                d = next(vd_iter)
                input, lbl, msk = d[0], d[1], d[2]
                outputs = self.net(input,msk)
                # return of WithEvalCell:
                #   return loss, (net)outputs, label
                out = outputs[1]
                out = out.squeeze(2)
                for j in range(data_len):
                    if j < seq_len-1:
                        pred_sum[:j+1] += out[j, -(j+1):]
                        pred_cnt[:j+1] += 1
                    elif j <= data_len-seq_len:
                        pred_sum[j-seq_len+1:j+1] += out[j]
                        pred_cnt[j-seq_len+1:j+1] += 1
                    else:
                        pred_sum[data_len-seq_len+1-(data_len-j):data_len-seq_len+1] += out[j, :(data_len-j)]
                        pred_cnt[data_len-seq_len+1-(data_len-j):data_len-seq_len+1] 
                
                truth = Tensor([lbl[j,-1].asnumpy() for j in range(len(lbl)-seq_len+1)], dtype=mstype.float32)
                pred_sum, pred_cnt = pred_sum[:data_len-seq_len+1], pred_cnt[:data_len-seq_len+1]
                pred = pred_sum/pred_cnt
                mse = (ops.pow(pred-truth, 2)).sum()
                rmse = math.sqrt(mse/data_len)
                tot += rmse
            Rc = 130
            mse_res = tot*Rc/len(valid_list)
            self.epoch_per_eval["mse"].append(mse_res)
            print('eval score: ' + str(mse_res))
            
            

# 以下cell是model的组成成分，用于在callback 中调用，定义行为
# https://www.hiascend.com/app-forum/topic-detail/0224103731274466014
class CustomWithEvalCell(nn.Cell):
    """自定义多标签评估网络"""

    def __init__(self, network):
        super(CustomWithEvalCell, self).__init__(auto_prefix=False)
        self.network = network
        # network is mymodel, receive src as fellow:
        '''
        def construct(self, src, attn_msk=None):
            src = self.pos_encoder(src)
            # output1 = self.transformer_encoder(src, attn_msk, key_msk)
            # problem？
            attn_msk = self.trans_mask_generator(attn_msk)
            output1 = self.transformer_encoder(src, attn_msk)
            # return the multiple result..
            output2 = self.decoder(output1[0])
            return output1, output2
        '''
    def construct(self, target_test_names):
        output = self.network(data)
        return output, label1, label2
    

# In[ ]:


seq_len = 70
target= 'FD002'
source = 'FD003'
epoches = 240
os.chdir('/Domain-Adaptive-Remaining-Useful-Life-Prediction-with-Transformer/')
batch_size = 100
a = 0.1
b = 0.5
source_list = numpy.loadtxt("save/"+source+"/train"+source+".txt", dtype=str).tolist()
target_list = numpy.loadtxt("save/"+target+"/train"+target+".txt", dtype=str).tolist()
valid_list = numpy.loadtxt("save/"+target+"/test"+target+".txt", dtype=str).tolist()
# len(v) = 47
a_list = numpy.loadtxt("save/"+target+"/valid"+target+".txt", dtype=str).tolist()
# len(a_list) 31
target_test_names = valid_list + a_list
# 78 files
target_test_names = target_test_names[:10]
minl = min(len(source_list), len(target_list))
s_data,t_data,t_data_test = prepareData(source_list,target_list,target_test_names)


# In[ ]:




# In[ ]:


class MERGED_DATA():
    def __init__(self,s_data,t_data) -> None:
        self.s_data = s_data
        self.t_data = t_data
    
    def __len__(self):
        return min(len(self.s_data),len(self.t_data))
    
    def __getitem__(self,index):
        # return self.s_data[index]+(self.t_data[index][0],self.t_data[index][2])
        return (self.s_data[index][0],self.s_data[index][1],self.s_data[index][2],self.t_data[index][0],self.t_data[index][2])
        # ['s_input', 's_lb', 's_msk','t_input', 't_msk']


# In[ ]:


sampler = ds.RandomSampler()
# all_data = MERGED_DATA(s_data,t_data)
# t_dataset = ds.GeneratorDataset(t_data,sampler=sampler,
#                                 column_names=['t_input', 't_nouse', 't_msk'])
# s_dataset = ds.GeneratorDataset(s_data,sampler=sampler,
#                                 column_names=['s_input', 's_lb', 's_msk'])

dataset = ds.GeneratorDataset(MERGED_DATA(s_data,t_data),sampler=sampler,column_names=['s_input', 's_lb', 's_msk','t_input', 't_msk'])
dataset = dataset.batch(batch_size=batch_size,drop_remainder=True)
# In[]

from mindspore.dataset import TupleIterator
rtl = TupleIterator(dataset)
# In[]
# for it in rtl:
#     for i in it:
#         print(i.shape)
# In[]

# In[]

# In[]

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
#In[]:
net = mymodel(max_len=seq_len,batch_size=batch_size)
# print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
print(id(net))
D1 = Discriminator(in_features=seq_len)
D2 = backboneDiscriminator(seq_len)
loss_net = MywithLossCell(net,D1,D2,loss_func)
opt = nn.SGD(net.trainable_params()+D1.trainable_params()+D2.trainable_params()
             ,learning_rate=0.02)
result_eval = {"mse": []}
eval_cb = EvalCallBack(net, 1, result_eval)
print("before model")

model = Model(network=loss_net, optimizer=opt)
# FORMAT two dataset into one.

# In[ ]:
print("start training")
import time
time.sleep(3)

model.train(epoch=10, train_dataset=dataset, callbacks=[LossMonitor(),eval_cb])


# 

# In[ ]:


# input_shape = self.shape(input_mask)
# shape_right = (input_shape[0], 1, input_shape[1])
# shape_left = input_shape + (1,)

# input_mask = self.cast(input_mask, ms.float32)
# mask_left = self.reshape(input_mask, shape_left)
# mask_right = self.reshape(input_mask, shape_right)
# attention_mask = self.batch_matmul(mask_left, mask_right)


# In[9]:


# In[ ]:




