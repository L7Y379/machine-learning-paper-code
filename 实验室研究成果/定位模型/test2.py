import pandas as pd
import numpy as np
import torch
from torch import nn
dimension = 270
x_data =np.array( (pd.read_csv(r'./420USEFUL\B\50x270\Train_Amti.csv', header=0)).iloc[:, 0:dimension] ) #
print(x_data.shape)#(60000, 270)#50个序列数据，每个序列数据有50组，每个组有270个数据
y_data = np.array((pd.read_csv(r'./420USEFUL\B\Train_Amti.csv', header=0)).iloc[:, -2:])
print(y_data.shape)#(2400, 2)
#print(y_data)
Num = np.size(y_data, axis=0) // 100
print(Num)#24
a = []
for i in range(Num):
    for j in range(50):
        a.append(i * 100 + j)
#print(a)
y_data = y_data[a, :]
print(y_data.shape)#(1200, 2)#总共有1200个序列数据，让每一个序列数据对应一个标签

x_data1 =np.array( (pd.read_csv(r'./420USEFUL\B\50x270\Test_Amti.csv', header=0)).iloc[:, 0:dimension] ) #
print(x_data1.shape)#(30000, 270)
y_data1 = np.array((pd.read_csv(r'./420USEFUL\B\Test_Amti.csv', header=0)).iloc[:, -2:])
print(y_data1.shape)#(1200, 2)
#print(y_data1)
Num = np.size(y_data1, axis=0) // 100
print(Num)#12
a = []
for i in range(Num):
    for j in range(50):
        a.append(i * 100 + j)
y_data1 = y_data1[a, :]
print(y_data1.shape)#(600, 2)
#print(y_data1)#(600, 2)

data = []
for i in range( int(np.shape(x_data)[0]) // 50):
    data.append(np.reshape(x_data[i*50:(i+1)*50, :], (50, 270)))  # 6个位置6个动作：99.1
x_data = np.asarray(data)

data = []
for i in range( int(np.shape(x_data1)[0]) // 50):
    data.append(np.reshape(x_data1[i*50:(i+1)*50, :], (50, 270)))  # 6个位置6个动作：99.1
x_data1 = np.asarray(data)

print(np.shape(x_data))#(1200, 50, 270)
print(np.shape(x_data1))#(600, 50, 270)