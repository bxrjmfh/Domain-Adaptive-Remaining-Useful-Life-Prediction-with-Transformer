# Inpliment of TRANSFORMER_ALL_DATA
# init
import os
import numpy
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import dtype as mstype
import mindspore.numpy as np
from tqdm import tqdm
Rc = 130
class TRANSFORMER_ALL_DATA_MINDS():
    def __init__(self, name, seq_len) -> None:
        data_root = "CMAPSS/units/"
        label_root = "CMAPSS/new_labels/"
        lis = os.listdir(data_root)
        data_list = [i for i in lis if i in name]
        self.data, self.label, self.padding = [], [], []
        
        for n in tqdm(name):
            raw = numpy.loadtxt(data_root+n,dtype='float32')[:,2:]
            # jump id and time cycle
            lbl = numpy.loadtxt(label_root+n,dtype='float32')/Rc
            l = len(lbl)
            if l<seq_len:
                raise RuntimeError("seq_len {} is too big for file '{}' with length {}".format(seq_len, n, l))
            raw, lbl = Tensor(raw), Tensor(lbl)
            lbl_pad_0 = [np.ones([seq_len-i-1],dtype=mstype.float32) for i in range(seq_len-1)]
            data_pad_0 = [np.zeros([seq_len-i-1,24],dtype=mstype.float32) for i in range(seq_len-1)]
            lbl_pad_1 = [np.zeros([i+1],dtype=mstype.float32) for i in range(seq_len-1)]
            data_pad_1 = [np.zeros([i+1,24],dtype=mstype.float32) for i in range(seq_len-1)]
            
            self.data += [ops.concat([data_pad_0[i],raw[:i+1]],0) for i in range(seq_len-1)] 
            self.data += [raw[i-seq_len+1:i+1] for i in range(seq_len-1, l)]
            self.data += [ops.concat([raw[l-seq_len+i+1:], data_pad_1[i]],0) for i in range(seq_len-1)]
            self.label += [ops.concat([lbl_pad_0[i],lbl[:i+1]],0) for i in range(seq_len-1)] 
            self.label += [lbl[i-seq_len+1:i+1] for i in range(seq_len-1, l)]
            self.label += [ops.concat([lbl[l-seq_len+i+1:], lbl_pad_1[i]],0) for i in range(seq_len-1)]
            # temp1 = ops.zeros(1,mstype.float32)
            # temp2 = ops.ones(seq_len-1,type=mstype.float32)
            self.padding += [ops.concat([ops.ones(seq_len-i-1,type=mstype.float32), ops.zeros(i+1,mstype.float32)],0) for i in range(seq_len-1)]   # 1 for ingore
            self.padding += [ops.zeros(seq_len,mstype.float32) for i in range(seq_len-1, l)]
            self.padding += [ops.concat([ops.zeros(seq_len-i-1,mstype.float32), ops.ones(i+1,type=mstype.float32)],0) for i in range(seq_len-1)]
            # todo:debug
            if len(self.data) > 300:
                break

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        return self.data[index], self.label[index], self.padding[index]
        
    
# Inpliment of TRANSFORMER_DATA
class TRANSFORMER_DATA_MINDS():
    def __init__(self, name, seq_len, root='new') -> None:
        data_root = "CMAPSS/units/"
        if root == 'old':
            label_root = "CMAPSS/labels/"
        elif root == 'new':
            label_root = "CMAPSS/new_labels/"
        else:
            raise RuntimeError("got invalid parameter root='{}'".format(root))
        
        raw = numpy.loadtxt(data_root+name,dtype='float32')[:,2:]
        lbl = numpy.loadtxt(label_root+name,dtype='float32')/Rc
        
        
        l = len(lbl)
        if l<seq_len:
            raise RuntimeError("seq_len {} is too big for file '{}' with length {}".format(seq_len, name, l))
        raw, lbl = Tensor(raw), Tensor(lbl)
        lbl_pad_0 = [np.ones([seq_len-i-1],dtype=mstype.float32) for i in range(seq_len-1)]
        data_pad_0 = [np.zeros([seq_len-i-1,24],dtype=mstype.float32) for i in range(seq_len-1)]
        lbl_pad_1 = [np.zeros([i+1],dtype=mstype.float32) for i in range(seq_len-1)]
        data_pad_1 = [np.zeros([i+1,24],dtype=mstype.float32) for i in range(seq_len-1)]
        
        self.data = [ops.concat([data_pad_0[i],raw[:i+1]],0) for i in range(seq_len-1)] 
        self.data += [raw[i-seq_len+1:i+1] for i in range(seq_len-1, l)]
        self.data += [ops.concat([raw[l-seq_len+i+1:], data_pad_1[i]],0) for i in range(seq_len-1)]
        self.label = [ops.concat([lbl_pad_0[i],lbl[:i+1]],0) for i in range(seq_len-1)] 
        self.label += [lbl[i-seq_len+1:i+1] for i in range(seq_len-1, l)]
        self.label += [ops.concat([lbl[l-seq_len+i+1:], lbl_pad_1[i]],0) for i in range(seq_len-1)]
        self.padding = [ops.concat([ops.ones(seq_len-i-1,type=mstype.float32), ops.zeros(i+1,mstype.float32)],0) for i in range(seq_len-1)]   # 1 for ingore
        self.padding += [ops.zeros(seq_len,mstype.float32) for i in range(seq_len-1, l)]
        self.padding += [ops.concat([ops.zeros(seq_len-i-1,mstype.float32), ops.ones(i+1,type=mstype.float32)],0) for i in range(seq_len-1)]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        return self.data[index], self.label[index], self.padding[index]

class MERGED_DATA():
    def __init__(self,s_data,t_data) -> None:
        self.s_data = s_data
        self.t_data = t_data
    
    def __len__(self):
        return min(len(self.s_data),len(self.s_data))
    
    def __getitem__(self,index):
        return self.s_data[index]+self.t_data[index]