import pandas as pd
import sys
import time
import matplotlib.pyplot as plt
#防止编译"输出"报错

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# np.set_printoptions(threshold=np.NaN)


import torch
import torch.nn as nn
import torch.optim as optim
# from tqdm import tqdm
from torch.autograd import Variable
import random

from sklearn import preprocessing
import torch.nn.functional as f
# import DaNN

from torch import nn


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #0.4版本以上才行

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()


## 一维卷积：1*270
        self._k_size=5

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=self._k_size, stride=1 ,padding=0,dilation=1, groups=1, bias=True) #默认padding=0,即0填充
        self.maxp1 = nn.MaxPool1d(kernel_size=self._k_size, stride=1)

        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=self._k_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.maxp2 = nn.MaxPool1d(kernel_size=self._k_size, stride=1)
        #
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=self._k_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.maxp3 = nn.MaxPool1d(kernel_size=self._k_size, stride=1)

## 全连接层

        self.FC1 = nn.Linear(7872,512)
        self.FC2 = nn.Linear(512, 128)  # 全链接层
        self.FC3 = nn.Linear(128, 2)  # 全链接层
        # self.FC4 = nn.Linear(128, 2)  # 全链接层


## 激活函数
        self.act_func = nn.RReLU()  # yes

        # nn.RReLU()  # yes
        # # nn.ReLU(), #no
        # nn.SELU()

        # # nn.LeakyReLU(), # yes
        # # nn.ELU(), #no
        # # nn.ReLU6(),   # no
        # # nn.PReLU(), #no
## Normalizer
        # self.Batch_N_2d=nn.BatchNorm2d(num_features=1,affine=True)
        # self.Batch_N_1d=nn.BatchNorm1d(num_features=1,affine=True)
        #
        # self.Instance_N_2d=nn.InstanceNorm2d(num_features=1,affine=True)
        # self.Instance_N_1d=nn.InstanceNorm1d(num_features=1,affine=True)

        self.Layer_N=nn.LayerNorm(x_data.size()[1:])



    def forward(self, x):

        x = self.Layer_N(x)

        # print("X:",x.shape)
        x = self.act_func(self.conv1(x))
        # print("C1:",x.shape)
        x = self.maxp1(x)
        # print("M1:",x.shape)

        x = self.act_func(self.conv2(x))
        # print("C2:",x.shape)
        x = self.maxp2(x)
        # print("M2:",x.shape)
        #
        x = self.act_func(self.conv3(x))
        # print("C3:",x.shape)
        x = self.maxp3(x)
        # print("M3",x.shape)

        # print("before x: ",x.shape)
        x = x.view(x.shape[0], -1)  # 展开
        # print("extend x: ",x.shape)

        x = self.FC1(x)
        x=self.act_func(x)


        x = self.FC2(x)
        x=self.act_func(x)

        x = self.FC3(x)
        x=self.act_func(x)

        return x

RESULT_TRAIN = [] #
RESULT_TEST = []  #

if __name__ == "__main__":
    start = time.clock()   #返回调用时间

    dimension = 270

### CNN: 1 * 270 or 15*18
    # train= pd.read_csv(r'./420USEFUL/D/Train_Amti.csv', header=0) #DNN_148
    # test= pd.read_csv(r'./420USEFUL/D/Test_Amti.csv', header=0) #DNN_148
    #
    # train= pd.read_csv(r'./New_420_Parrel/B/Train_Amti.csv', header=0) #DNN_148
    # test= pd.read_csv(r'./New_420_Parrel/B/Test_Amti.csv', header=0) #DNN_160
    #
    # train= pd.read_csv(r'./New_5_Stair_Test_Trans/B/Train_Amti.csv', header=0) #DNN_148
    # test= pd.read_csv(r'./New_5_Stair_Test_Trans/B/Test_Amti.csv', header=0) #DNN_160

    train = pd.read_csv(r'./420USEFUL\B\Train_Amti.csv', header=0)
    test = pd.read_csv(r'./420USEFUL\B\Test_Amti.csv', header=0)

    # Seperate_Point_Train = a0 = [2, 4, 9, 12, 14, 17, 19, 22]  #420
    # Seperate_Point_Train = a0 = [2, 5, 8, 14, 17, 21, 24, 28,12,25]  # 410
    # Seperate_Point_Train = a0 = [1, 4, 6, 8, 11,26, 16, 18,20,22]  # 410

    # Seperate_Point_Train = a0 = [1, 4, 8, 9, 12, 16, 19, 23,25,27,31,34,36,40,43]  # 5Floor
    # Seperate_Point_Train = 不需要  # 416


    # Decrease_Num = 10
    # Decrease_Num = 100 - Decrease_Num
    # result = train.iloc[(Seperate_Point_Train[0] * 100 - 100):Seperate_Point_Train[0] * 100 - Decrease_Num]#??
    # for i in range(1, np.size(Seperate_Point_Train)):
    #     result = pd.concat([result, train.iloc[(Seperate_Point_Train[i] * 100 - 100):Seperate_Point_Train[i] * 100 - Decrease_Num]])
    # train = result

    #
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

    CNN_Net = CNN()
    CNN_Net = CNN_Net.to(DEVICE)
    # print(CNN_Net)

    loss_fn = nn.MSELoss(reduce=False, size_average=False)

    optimizer = torch.optim.Adamax(CNN_Net.parameters(), lr=0.001, weight_decay=0.0001)

    model_path = r'Model/test.pkl'
    # model_path = r'./Model/H_416_Source_CNN_2.pkl'

    batch_size =10
    #####################################   因为各自的数量不一致，故随机取值训练时，需要分开取值
    num_train_sample=np.shape(x_data)[0]     #样本数据量
    num_test_sample = np.shape(x_data1)[0]

    display_step=5

    N_EPOCH = 50
    #训练，返回训练的模型

    temp_count=0

    intial_ref = 200
    auto_save_model_ref = intial_ref

    # 借助该切片思想
    for epoch in range(1, N_EPOCH):
        total_batch = int(num_train_sample / batch_size)
        batch_total_loss=0.0
        # ITERATION
        for i in range(total_batch):
            rand_train_indexes = np.random.randint(num_train_sample, size=batch_size)#对于训练数据，取随机
            ################  train_batch
            train_batch_xs = x_data[rand_train_indexes, :,:]
            train_batch_ys = y_data[rand_train_indexes,:]

            train_batch_xs, train_batch_ys = train_batch_xs.to(DEVICE), train_batch_ys.to(DEVICE)
            train_batch_xs, train_batch_ys=Variable(train_batch_xs),Variable(train_batch_ys)

            CNN_Net.train()  # 开始训练正常的网络模型

            #计算 Jnn

            y_src= CNN_Net(train_batch_xs)  #

            # print("BB:",np.shape(y_src))

            batch_loss_c = torch.Tensor.mean(torch.Tensor.sqrt(torch.Tensor.sum(loss_fn(y_src, train_batch_ys), 1)))  # 返回距离损失

            batch_loss = batch_loss_c
            optimizer.zero_grad()  # 参数梯度值初始化为0
            batch_loss.backward()  # 总误差反相传播，计算新的更新参数值
            optimizer.step()  # 更新参数：将计算得到的更新值赋给net.parameters()
            # if ((epoch + 1) % display_step == 0 and i==(total_batch-1)):#每次循环的最后一次的batch
            #     # print("batch_loss_Jnn: %.5f        batch_loss_mmd:%.5f    batch_loss:%f" % (batch_total_loss/total_batch, batch_loss_mmd, batch_loss))
            #     print("batch_loss_Jnn: %0.3f" % (batch_loss_c))

        # DISPLAY
        if (epoch + 1) % display_step == 0:

            print("Epoch: %d(%d)/%d " % (epoch + 1, (epoch + 1) / 30, N_EPOCH))

            ###***************************   依旧是这样，此时不参与训练  ***********************
            with torch.no_grad():
                xs, ys = x_data.to(DEVICE), y_data.to(DEVICE)
                x1s, y1s = x_data1.to(DEVICE), y_data1.to(DEVICE)

                CNN_Net.eval()
                yspred = CNN_Net(xs)
                y1spred = CNN_Net(x1s)

                train_loss = torch.Tensor.mean(torch.Tensor.sqrt(torch.Tensor.sum(loss_fn(yspred, ys), 1)))
                test_loss = torch.Tensor.mean(torch.Tensor.sqrt(torch.Tensor.sum(loss_fn(y1spred, y1s), 1)))

                RESULT_TRAIN.append(train_loss)
                RESULT_TEST.append(test_loss)

                print("train: %.2f   test:%.2f" % (train_loss, test_loss))

                print(
                    '-----------------------------------------------------------------------------------------------')

                # if test_loss<=150:
                #     optimizer = torch.optim.Adamax(model.parameters(), lr=0.0001, weight_decay=0.001)


                if test_loss <= auto_save_model_ref and train_loss <= 50:
                    torch.save(CNN_Net.state_dict(), model_path)
                    auto_save_model_ref = test_loss
                    print("    Save model: ", auto_save_model_ref)

                if test_loss < 110:  # 设定一个经验上限
                    torch.save(CNN_Net.state_dict(), model_path)
                    auto_save_model_ref = test_loss
                    break

                    # if test_loss<120 and train_loss <=20:
                    #     temp_count = temp_count + 1
                    #     if temp_count >=1:
                    #         # save model
                    #         torch.save(model.state_dict(), r'./Model/420_HouTrain_LxyTest.pkl')# 420_DNN_params.pkl-123cm
                    #         # _,x_src_mse, _ = model(xs, xs)
                    #         # _, x1_src_mse, _ = model(x1s, x1s)
                    #         # np.savetxt(r'DNN_MSE_Data\420_A_Train_H3.csv', x_src_mse.detach().numpy(), fmt='%.18f', delimiter=',')
                    #         # np.savetxt(r'DNN_MSE_Data\420_A_Test_H3.csv', x1_src_mse.detach().numpy(), fmt='%.18f',delimiter=',')
                    #
                    #         break
                    #     else:
                    #         pass
                    #
                    # else:
                    #     # temp_count=0
                    #     pass

    print("    Save model: ", auto_save_model_ref)

    fig = plt.figure(1)
    ax1 = fig.add_subplot(111)
    res_train = np.asarray(RESULT_TRAIN)
    res_test = np.asarray(RESULT_TEST)

    plt.plot(res_train, 'k')
    plt.plot(res_test, 'r')
    plt.xlabel("Train_Epoch")
    plt.ylabel("Loss(cm)")
    plt.grid()  # 生成网格
    plt.show()

    # Four_Result = np.concatenate((res_train,res_test), axis=1)  # 数组或者列表拼接,axis=0表示沿着纵轴
    # Six_Result=np.column_stack((res_train,res_test,res_trans1,res_trans2,res_trans1_B2,res_trans2_B2))
    # np.savetxt('Backup_Result.csv', Six_Result, fmt='%.3f', delimiter=',')
    # np.savetxt('Six_Result.csv', Six_Result, fmt='%.3f', delimiter=',')


    end = time.clock()
    print((end - start) / 60)














