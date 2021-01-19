# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:19:44 2020

@author: 11029
"""
import pandas as pd
import numpy as np
import torch
from torch import nn
dimension = 270
train = pd.read_csv(r'./420USEFUL\B\Train_Amti.csv', header=0)
test = pd.read_csv(r'./420USEFUL\C\Test_Amti.csv', header=0)
train_data = train.iloc[:, 0:dimension]  #
train_target = train.iloc[:, -2:]  # 必须保证原数据是正确的

test_data = test.iloc[:, 0:dimension]  #
test_target = test.iloc[:, -2:]

    # 转换为数组:中转
x_data = np.array(train_data)
y_data = np.array(train_target)

x_data1 = np.array(test_data)
y_data1 = np.array(test_target)


    ## reshape：1 x 270
data = []
for i in range(np.shape(x_data)[0]):
    data.append( np.reshape( x_data[i, :], (1, dimension)) )  # 6个位置6个动作：99.1
x_data = np.asarray(data)
data = []
for i in range(np.shape(x_data1)[0]):
    data.append( np.reshape( x_data1[i, :], (1, dimension)) )  # 6个位置6个动作：99.1
x_data1 = np.asarray(data)


    #转换为张量
x_data = torch.from_numpy(x_data).type(torch.FloatTensor)   # torch.Size([3000, dimension])
y_data = torch.from_numpy(y_data).type(torch.FloatTensor)
x_data1 = torch.from_numpy(x_data1).type(torch.FloatTensor)
y_data1 =torch.from_numpy(y_data1).type(torch.FloatTensor)

'''
print(x_data)
print(x_data.shape)
print(x_data.size())

print(y_data)
print(y_data.size())
print(x_data1)
print(x_data1.size())
print(y_data1)
print(y_data1.size())
'''
num_train_sample=np.shape(x_data)[0]
print(num_train_sample)
num_test_sample = np.shape(x_data1)[0]
print(num_test_sample)

display_step=5

N_EPOCH = 10000
    #训练，返回训练的模型

temp_count=0
batch_size =10
intial_ref = 200
auto_save_model_ref = intial_ref
'''    # 借助该切片思想
for epoch in range(1, N_EPOCH):
    total_batch = int(num_train_sample / batch_size)
    batch_total_loss=0.0
    # ITERATION
for i in range(5):
    rand_train_indexes = np.random.randint(50, size=batch_size)#对于训练数据，取随机
    print(rand_train_indexes)
print(rand_train_indexes.shape)
'''


print(y_data)
t1 = torch.FloatTensor([[1,2],[5,6]])
t2 = torch.FloatTensor([[2,3],[6,7]])



#print(y_datata)
loss_fn = nn.MSELoss(reduce=False, size_average=False)
train_loss = torch.Tensor.mean(torch.Tensor.sqrt(torch.Tensor.sum(loss_fn(t1, t2), 1)))
print(train_loss)
#print(x_data.shape[0])
#print(train.shape[1])


