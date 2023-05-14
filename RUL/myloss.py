import mindspore.nn as nn
import mindspore.numpy as np
from mindspore.nn import LossBase
from mindspore.ops import functional as F
from mindspore import Parameter

class advLoss(LossBase):
    def __init__(self, reduction='mean'):
        super(advLoss,self).__init__(reduction)
        self.Loss = nn.BCELoss()
        # self.lossVal = None
        # self.x = Parameter(0.0)
    
    def construct(self, source, target):
        sourceLabel = np.ones((len(source)))
        targetLabel = np.zeros((len(target)))
        lossVal = F.add(self.Loss(source,sourceLabel), self.Loss(target, targetLabel))
        x = F.mul(lossVal,0.5)
        return self.get_loss(x)
    
        
