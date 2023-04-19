import mindspore.nn as nn
import mindspore.numpy as np
from mindspore.nn import LossBase

class advLoss(LossBase):
    def __init__(self,source,target, reduction='mean'):
        super(advLoss,self).__init__(reduction)
        self.sourceLabel = np.ones((len(source)))
        self.targetLabel = np.zeros((len(target)))
        self.Loss = nn.BCELoss()
        self.lossVal = None
    
    def construct(self, source, target):
        self.lossVal = self.Loss(source,self.sourceLabel) + self.Loss(target, self.targetLabel)
        return 0.5*self.lossVal
    
        
