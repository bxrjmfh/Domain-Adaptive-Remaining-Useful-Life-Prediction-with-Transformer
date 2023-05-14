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

import numpy as np

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Model
from mindspore import dataset as ds
from mindspore.nn import LossBase
from mindspore.common.initializer import Normal
from mindspore.train.callback import LossMonitor
from mindspore import set_context,context
set_context(mode=context.GRAPH_MODE,device_target='GPU', device_id=0, save_graphs=True,
            save_graphs_path="/Domain-Adaptive-Remaining-Useful-Life-Prediction-with-Transformer/RUL/ir_files",
            save_graph_dot=True)

# In[ ]:


# class L1LossForMultiLabel(LossBase):
#     def __init__(self, reduction="mean"):
#         super(L1LossForMultiLabel, self).__init__(reduction)
#         self.abs = ops.Abs()

#     def construct(self, base, target1, target2):
#         x1 = self.abs(base - target1)
#         x2 = self.abs(base - target2)
#         return self.get_loss(x1)/2 + self.get_loss(x2)/2
# # reformat the loss class
        

# class CustomWithLossCell(nn.Cell):
#     def __init__(self, backbone, loss_fn):
#         super(CustomWithLossCell, self).__init__(auto_prefix=False)
#         self._backbone = backbone
#         self._loss_fn = loss_fn

#     def construct(self, data, label1, label2):
#         output = self._backbone(data)
#         return self._loss_fn(output, label1, label2)

# def get_multilabel_data(num, w=2.0, b=3.0):
#     for _ in range(num):
#         x = np.random.uniform(-10.0, 10.0)
#         noise1 = np.random.normal(0, 1)
#         noise2 = np.random.normal(-1, 1)
#         y1 = x * w + b + noise1
#         y2 = x * w + b + noise2
#         yield np.array([x]).astype(np.float32), np.array([y1]).astype(np.float32), np.array([y2]).astype(np.float32)

# def create_multilabel_dataset(num_data, batch_size=16):
#     dataset = ds.GeneratorDataset(list(get_multilabel_data(num_data)), column_names=['data', 'label1', 'label2'])
#     dataset = dataset.batch(batch_size)
#     return dataset

# # initial net


# # Set up loss
# loss = L1LossForMultiLabel()
# # build loss network
# loss_net = CustomWithLossCell(net, loss)

# opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)
# model = Model(network=loss_net, optimizer=opt)
# ds_train = create_multilabel_dataset(num_data=160)
# model.train(epoch=1, train_dataset=ds_train, callbacks=[LossMonitor()], dataset_sink_mode=False)


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
        self.net = net
        self.D1 = D1
        self.D2 = D2
        self.loss_fn = loss_fn

    def construct(self,s_input, s_lb,s_msk, t_input, t_msk):
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
a_list = numpy.loadtxt("save/"+target+"/valid"+target+".txt", dtype=str).tolist()
target_test_names = valid_list + a_list
minl = min(len(source_list), len(target_list))
s_data,t_data,t_data_test = prepareData(source_list,target_list,target_test_names)


# In[ ]:


s_data.data[0].shape


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

D1 = Discriminator(in_features=seq_len)
D2 = backboneDiscriminator(seq_len)
loss_net = MywithLossCell(net,D1,D2,loss_func)
opt = nn.SGD(net.trainable_params()+D1.trainable_params()+D2.trainable_params()
             ,learning_rate=0.02)
print("before model")

model = Model(network=loss_net, optimizer=opt)
# FORMAT two dataset into one.

# In[ ]:
print("start training")
import time
time.sleep(3)

model.train(epoch=10, train_dataset=dataset, callbacks=[LossMonitor()])


# 

# In[ ]:
import mindspore as ms
a = ms.Tensor([1])

# input_shape = self.shape(input_mask)
# shape_right = (input_shape[0], 1, input_shape[1])
# shape_left = input_shape + (1,)

# input_mask = self.cast(input_mask, ms.float32)
# mask_left = self.reshape(input_mask, shape_left)
# mask_right = self.reshape(input_mask, shape_right)
# attention_mask = self.batch_matmul(mask_left, mask_right)


# In[9]:


# In[ ]:




