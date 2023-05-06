import math

import mindspore.nn as nn
import mindspore.nn.transformer as trans
from mindspore import Tensor
from mindspore import numpy as np
from mindspore import ops as ops
from mindspore.common import initializer
from mindspore.ops import operations as P


class PositionalEncoding(nn.Cell):

    def __init__(self,d_model, dropout=0.1, max_len=500) -> None:
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(keep_prob=1-dropout)
        # https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/note/api_mapping/pytorch_diff/mindspore.nn.Dropout.html?highlight=Dropout
        position = np.arange(max_len).reshape(max_len,1)
        div_term = ops.exp(np.arange(0, d_model, 2)*(-math.log(10000.0) / d_model))
        pe = np.zeros((max_len,d_model))
        pe[:, 0::2] = ops.sin(position * div_term)
        pe[:, 1::2] = ops.cos(position * div_term)
        self.pe=pe

    def construct(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, feature_num]
        """
        # pe with shape: 
        x = ops.add(x, self.pe[:x.shape[0]].unsqueeze(0))
        # x is single data
        # 所有的操作都必须ops来搞
        return self.dropout(x)
    

class Discriminator(nn.Cell): #D_y
    def __init__(self, auto_prefix=True, flags=None, in_features=24):
        super(Discriminator,self).__init__(auto_prefix, flags)
        self.in_features = in_features
        self.li = nn.SequentialCell(
            nn.Dense(in_features,512),
            # to config the detail
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dense(512,1),
            nn.Sigmoid()
        )
        self.alpha = 1 
    
    def construct(self, x):
        # 由于实现了自定义的算子，因此需要进行迁移
        # https://zhuanlan.zhihu.com/p/548702030 -> 直接在forward处进行修改
        # 前向传播的过程中使用了梯度翻转层的
        # 确定需求所在：https://www.mindspore.cn/tutorials/experts/zh-CN/r1.10/network/custom_cell_reverse.html
        # jump forward
        if x.shape[0] == 1:
            pad = np.zeros((1,self.in_features))
            # cuda?
            x = ops.concat((x,pad))
            # diff in concat https://www.mindspore.cn/docs/zh-CN/r1.9/note/api_mapping/pytorch_diff/Concat.html
            y = self.li(x)[0].unsqueeze(0)
            return y
        return self.li(x)

class backboneDiscriminator(nn.Cell): #D_f
    def __init__(self, seq_len, d=24):
        super(backboneDiscriminator,self).__init__()
        self.seq_len = seq_len 
        
        self.li1 = nn.Dense(d, 1)
        self.li2 = nn.SequentialCell(
            nn.Dense(seq_len,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dense(512,1),
            nn.Sigmoid()
        )

    def construct(self,x):
        # jump forward
        out1 = self.li1(x).squeeze(2)
        if x.shape[0] ==1:
            pad = np.zeros((1,self.seq_len))
            out1 = ops.concat((out1,pad))
            out2 = self.li2(out1)[0].unsqueeze(0)
            return out2
        out2 = self.li2(out1)
        return out2


class mymodel(nn.Cell):

    def __init__(self, d_model = 24, dropout=0.1, nhead=8, nlayers=2, max_len=500, batch_size=1):
        super(mymodel,self).__init__()
        self.max_len = max_len
        self.pos_encoder = PositionalEncoding( d_model=d_model, dropout=dropout, max_len=max_len)
        # transformer_encoder_layer = nn.TransformerEncoderLayer()
        self.transformer_encoder = nn.TransformerEncoder(seq_length=max_len,batch_size=batch_size,hidden_size=d_model,num_heads=nhead,ffn_hidden_size=512,post_layernorm_residual=True,num_layers=nlayers )
        self.d_model = d_model
        self.dropout = nn.Dropout(1-dropout)
        self.decoder = nn.Dense(d_model, 1 ,weight_init = initializer.Uniform(scale=0.1) )
        # bias default is zeros
        self.trans_mask_generator = trans.AttentionMask(seq_length=max_len)
        
    def construct(self, src, attn_msk=None):
        src = self.pos_encoder(src)
        # output1 = self.transformer_encoder(src, attn_msk, key_msk)
        # problem？
        attn_msk = self.trans_mask_generator(attn_msk)
        output1 = self.transformer_encoder(src, attn_msk)
        # return the multiple result..
        output2 = self.decoder(output1[0])
        return output1, output2