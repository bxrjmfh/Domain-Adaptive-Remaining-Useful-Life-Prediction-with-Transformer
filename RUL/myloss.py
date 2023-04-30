import mindspore.nn as nn
import mindspore.numpy as np
from mindspore.nn import LossBase
from mindspore.ops import functional as F

class advLoss(LossBase):
    def __init__(self, reduction='mean'):
        super(advLoss,self).__init__(reduction)
        self.Loss = nn.BCELoss()
        self.lossVal = None
        self.x = 0.0
    
    def construct(self, source, target):
        sourceLabel = np.ones((len(source)))
        targetLabel = np.zeros((len(target)))
        self.lossVal = F.add(self.Loss(source,sourceLabel), self.Loss(target, targetLabel))
        self.x = F.mul(self.lossVal,0.5)
        return self.get_loss(self.x)
    
        
