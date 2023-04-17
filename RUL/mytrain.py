from mydataset import TRANSFORMER_DATA_MINDS,TRANSFORMER_ALL_DATA_MINDS
from mymodel import mymodel, Discriminator, backboneDiscriminator
from tqdm import tqdm
import random
import numpy
import os 
import mindspore.dataset as ds
target= 'FD002'
source = 'FD003'
epoches = 240
os.chdir('/Domain-Adaptive-Remaining-Useful-Life-Prediction-with-Transformer/')
batch_size = 1000
def train(source_list, target_list):
    minn = 999
    for e in tqdm(range(epoches)):
        al, tot = 0, 0
        net.train()
        random.shuffle(source_list)
        random.shuffle(target_list)
        source_iter, target_iter = iter(source_list), iter(target_list)
        loss2_sum, loss1_sum = 0, 0
        bkb_sum, out_sum = 0, 0
        cnt = 0
        # 数据集的差异：
        # https://www.mindspore.cn/docs/migration_guide/zh-CN/r1.5/api_mapping/pytorch_diff/DataLoader.html
        # 实现采样器，随后数据集准备，最后得到iterator
        sampler = ds.RandomSampler()
        s_dataset = ds.GeneratorDataset(s_data,sampler=sampler, random=True)
        # prepare data 
        s_dataset = s_dataset.batch(batch_size=batch_size)
        s_iter = s_dataset.create_dict_iterator()
        t_dataset = ds.GeneratorDataset(t_data,sampler=sampler, random=True)
        t_dataset = t_dataset.batch(batch_size=batch_size)
        t_iter = t_dataset.create_dict_iterator()
        l = min(len(s_iter), len(t_iter))



seq_len = 70
source_list = numpy.loadtxt("save/"+source+"/train"+source+".txt", dtype=str).tolist()
target_list = numpy.loadtxt("save/"+target+"/train"+target+".txt", dtype=str).tolist()
valid_list = numpy.loadtxt("save/"+target+"/test"+target+".txt", dtype=str).tolist()
a_list = numpy.loadtxt("save/"+target+"/valid"+target+".txt", dtype=str).tolist()
target_test_names = valid_list + a_list
minl = min(len(source_list), len(target_list))
s_data = TRANSFORMER_ALL_DATA_MINDS(source_list, seq_len)
t_data = TRANSFORMER_ALL_DATA_MINDS(target_list, seq_len)
t_data_test = TRANSFORMER_ALL_DATA_MINDS(target_test_names, seq_len)

# prepare model
net = mymodel(max_len=seq_len) 
D1 = Discriminator(seq_len)
D2 = backboneDiscriminator(seq_len)

train()



