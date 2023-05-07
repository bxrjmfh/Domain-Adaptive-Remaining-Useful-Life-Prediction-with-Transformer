# %%
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
from mindspore.ops import composite as C

# %%
dis = Discriminator()

from mindspore import grad
class BabyModel(nn.Cell ):
    def __init__(self, auto_prefix=True, flags=None, in_features=24):
        super(BabyModel,self).__init__(auto_prefix, flags)
    
    def construct(self, x):
        return 2*x
    

class BabyModel1(nn.Cell ):
    def __init__(self, auto_prefix=True, flags=None, in_features=24):
        super(BabyModel1,self).__init__(auto_prefix, flags)
    
    def construct(self, x):
        return 2*x
    
    def bprop(self,x,out,dout):
        return (-dout,)

class ReverseGrad(nn.Cell):
    def __init__(self,alpha, auto_prefix=True, flags=None):
        super().__init__(auto_prefix, flags)
        self.alpha = alpha
    
    def construct(self,x):
        x = x+1 
        x = x-1
        return x
    
    def bprop(self,x,out,dout):
        return self.alpha*dout
    
def getBackward(net,x):
    grad_all = C.GradOperation(get_all=True)
    res = grad_all(net)(x)
    return res



# %%
from mindspore import Tensor
from mindspore import dtype 
a = Tensor(np.random.randn(200,24),dtype=dtype.float32)
net = BabyModel()
net1 = BabyModel1()
net2 = ReverseGrad(11)
# getBackward(net,a)
print('11111')
net2.bprop_debug = True
print(getBackward(net2,a))

# %%
df = Discriminator()
getBackward(df,a)


# %%
a_res = dis(a)

# %%
a_res

# %%
dis_grad = ops.GradOperation(get_all = True)

# %%
dis_grader = dis_grad(dis)

# %%
dis_grader(a)

# %%


# %%
from mindspore import Parameter
from mindspore.common import ParameterTuple
from mindspore.ops import GradOperation
from mindspore import dtype as mstype
from mindspore.ops import operations as P
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = P.MatMul()
        self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')
    def construct(self, x, y):
        # x = x * self.z
        out = self.matmul(x, y)
        return out

class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net
        self.grad_op = GradOperation()
    def construct(self, x, y):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, y)

x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
output = GradNetWrtX(Net())(x, y)

# %%
x

# %%
y

# %%
output

# %%
# https://gitee.com/mindspore/mindspore/blob/master/tests/ut/python/pynative_mode/test_hook.py#
import numpy as np
# import pytest

import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import context, Tensor, ParameterTuple
from mindspore.common.initializer import TruncatedNormal
from mindspore.nn import WithLossCell, Momentum
from mindspore.ops import composite as C

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

grad_all = C.GradOperation(get_all=True)


class MulAdd(nn.Cell):
    def __init__(self):
        super(MulAdd, self).__init__()

    def construct(self, x):
        return 2*x

    def bprop(self, x, out, dout):
        return (-dout, )

def test_custom_bprop():
    mul_add = MulAdd()
    x = Tensor(np.array([1, 2, 3]).astype(np.int32))
    ret = grad_all(mul_add)(x)
    print(ret) 

test_custom_bprop()




# %%
