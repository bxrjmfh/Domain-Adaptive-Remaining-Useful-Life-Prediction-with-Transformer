import mindspore.nn as nn
import mindspore.numpy as np
from mindspore.nn import LossBase

class advLoss(LossBase):
    def __init__(self, reduction='mean'):
        super(advLoss,self).__init__(reduction)
        self.Loss = nn.BCELoss()
        self.lossVal = None
    
    def construct(self, source, target):
        sourceLabel = np.ones((len(source)))
        targetLabel = np.zeros((len(target)))
        self.lossVal = self.Loss(source,sourceLabel) + self.Loss(target, targetLabel)
        return 0.5*self.lossVal
    
        
