import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable

import torch.nn.functional as f

import time
import torch
import torch.utils.data as Data
import torchvision

BATCH_SIZE = 50
# 全局变量
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #0.4版本以上才行


class RNN(torch.nn.Module):
    def __init__(self,input_size=None,hidden_size=None,num_layers=None,bidirectional=None,batch_first=None,out_size_fc1=None,out_size=None):
        super().__init__()
        self.rnn=torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional = bidirectional, #True:双向RNN、False:单向RNN
            batch_first=batch_first
        )

        if self.rnn.bidirectional is True:
            self.FC1=torch.nn.Linear(in_features=hidden_size*2,out_features=out_size_fc1 )
            self.out = torch.nn.Linear(in_features=out_size_fc1, out_features=out_size)
        else:
            self.FC1 = torch.nn.Linear(in_features=hidden_size, out_features=out_size_fc1)
            self.out = torch.nn.Linear(in_features=out_size_fc1, out_features=out_size)

        self.act_func = nn.ReLU()  # yes

        # nn.RReLU()  # yes
        # # nn.ReLU(), #no
        # nn.SELU()

        # # nn.LeakyReLU(), # yes
        # # nn.ELU(), #no
        # # nn.ReLU6(),   # no
        # # nn.PReLU(), #no
        self.dropout = nn.Dropout(p=0.3)


    def forward(self,x):
        # 一下关于shape的注释只针对单项
        # output: [batch_size, time_step, hidden_size]
        # h_n: [num_layers,batch_size, hidden_size] # 虽然LSTM的batch_first为True,但是h_n/c_n的第一维还是num_layers
        # c_n: 同h_n

        # - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
        #   containing the initial hidden state for each element in the batch.
        # - **c_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
        #   containing the initial cell state for each element in the batch.
        #
        #   If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.

        output,(h_n,c_n)=self.rnn(x)

        ## 单向RNN是一致的：
        # output_in_last_timestep=output[:,-1,:] # 也是可以的
        #output_in_last_timestep=h_n[-1,:,:]

        ## 方案0：只取最后一个步长的输出
        # 双向RNN:只有output[:,-1,:]正常，因此选择output[:,-1,:]
        output_in_last_timestep = output[:, -1, :]




        x=self.act_func( self.FC1(output_in_last_timestep) )

        x=self.act_func( self.out(x) )

        return x

dimension=270
RESULT_TRAIN = [] #
RESULT_TEST = []  #

if __name__ == "__main__":
    start = time.clock()

## 410_B1
    # x_data = np.array( (pd.read_csv(r'./410hou_second/100/50x270/Train_Amti.csv', header=0)).iloc[:, 0:dimension]  )#
    # y_data= np.array( (pd.read_csv(r'./410hou_second/100/Train_Amti.csv', header=0)).iloc[:, -2:] )
    # Num = np.size(y_data,axis=0)//100
    # a = []
    # for i in range(Num):
    #     for j in range(50):
    #         a.append(i * 100 + j)
    # y_data=y_data[a,:]
    #
    # x_data1 =np.array ( (pd.read_csv(r'./410hou_second/100/50x270/Test_Amti.csv', header=0)).iloc[:, 0:dimension] ) #
    # y_data1= np.array( (pd.read_csv(r'./410hou_second/100/Test_Amti.csv', header=0)).iloc[:, -2:] )
    # Num = np.size(y_data1,axis=0)//100
    # a = []
    # for i in range(Num):
    #     for j in range(50):
    #         a.append(i * 100 + j)
    # y_data1=y_data1[a,:]

# ## 420_B
#     x_data =np.array( (pd.read_csv(r'./420USEFUL/B/50x270/Train_Amti.csv', header=0)).iloc[:, 0:dimension] ) #
#     y_data = np.array((pd.read_csv(r'./420USEFUL/B/Train_Amti.csv', header=0)).iloc[:, -2:])
#     Num = np.size(y_data, axis=0) // 100
#     a = []
#     for i in range(Num):
#         for j in range(50):
#             a.append(i * 100 + j)
#     y_data = y_data[a, :]
#
#     x_data1 =np.array( (pd.read_csv(r'./420USEFUL/B/50x270/Test_Amti.csv', header=0)).iloc[:, 0:dimension]  )#
#     y_data1 = np.array((pd.read_csv(r'./420USEFUL/B/Test_Amti.csv', header=0)).iloc[:, -2:])
#     Num = np.size(y_data1, axis=0) // 100
#     a = []
#     for i in range(Num):
#         for j in range(50):
#             a.append(i * 100 + j)
#     y_data1 = y_data1[a, :]
#
# ## 5Floor_A
    x_data =np.array( (pd.read_csv(r'./420USEFUL\B\50x270\Train_Amti.csv', header=0)).iloc[:, 0:dimension] ) #
    y_data = np.array((pd.read_csv(r'./420USEFUL\B\Train_Amti.csv', header=0)).iloc[:, -2:])
    Num = np.size(y_data, axis=0) // 100
    a = []
    for i in range(Num):
        for j in range(50):
            a.append(i * 100 + j)
    y_data = y_data[a, :]

    x_data1 =np.array( (pd.read_csv(r'./420USEFUL\B\50x270\Test_Amti.csv', header=0)).iloc[:, 0:dimension] ) #
    y_data1 = np.array((pd.read_csv(r'./420USEFUL\B\Test_Amti.csv', header=0)).iloc[:, -2:])
    Num = np.size(y_data1, axis=0) // 100
    a = []
    for i in range(Num):
        for j in range(50):
            a.append(i * 100 + j)
    y_data1 = y_data1[a, :]



## reshape: x * 50 *270

    data = []
    for i in range( int(np.shape(x_data)[0]) // 50):
        data.append(np.reshape(x_data[i*50:(i+1)*50, :], (50, 270)))  # 6个位置6个动作：99.1
    x_data = np.asarray(data)

    data = []
    for i in range( int(np.shape(x_data1)[0]) // 50):
        data.append(np.reshape(x_data1[i*50:(i+1)*50, :], (50, 270)))  # 6个位置6个动作：99.1
    x_data1 = np.asarray(data)

    print(np.shape(x_data))
    print(np.shape(x_data1))

    ##归一化
    # 方法一
    # scaler = preprocessing.StandardScaler().fit(x_data)#保存训练集的mean和std：假若测试集的分布不同于训练集，借此归一化的结果也会不同
    # x_data=scaler.transform(x_data)
    # x_data1=scaler.transform(x_data1)
    # # train与test分别scale,可能会有出入:若不是同分布的话
    # #
    # # 方法二：0均值，单位方差
    # x_data=preprocessing.scale(x_data)
    # x_data1=preprocessing.scale(x_data1)
    #
    # trans1_data=preprocessing.scale(trans1_data)
    # trans2_data=preprocessing.scale(trans2_data)

    # np.savetxt(r'.\CSI_Classifer\Binary_x.csv', x_data, fmt='%.18f',delimiter=',')
    # np.savetxt(r'.\CSI_Classifer\Binary_x1.csv', x_data1, fmt='%.18f',delimiter=',')
    #
    # 方法三
    # min_max_scaler=preprocessing.MinMaxScaler()#MinMaxScaler是一个类
    # x_data=min_max_scaler.fit_transform(x_data)
    # x_data1=min_max_scaler.fit_transform(x_data1)
    # # np.savetxt(r'.\CSI_Classifer\Min_max_x.csv', x_data, fmt='%.18f',delimiter=',')
    # np.savetxt(r'.\CSI_Classifer\Min_max_x1.csv', x_data1, fmt='%.18f',delimiter=',')
    # }
    # #正则化
    # from sklearn import preprocessing #标准化数据模块
    # x_data = preprocessing.normalize(x_data, norm='l2')
    # x_data1 = preprocessing.normalize(x_data1, norm='l2')
    #
    # # 使用processing.Normalizer()类实现对训练集和测试集的拟合和转换：
    # normalizer=preprocessing.Normalizer().fit(x_data)
    # x_data=normalizer.transform(x_data)
    # x_data1=normalizer.transform(x_data1)


    # 转换为张量
    x_data = torch.from_numpy(x_data).type(torch.FloatTensor)  # torch.Size([3000, dimension])
    y_data = torch.from_numpy(y_data).type(torch.FloatTensor)
    x_data1 = torch.from_numpy(x_data1).type(torch.FloatTensor)
    y_data1 = torch.from_numpy(y_data1).type(torch.FloatTensor)


    # trans_data = torch.from_numpy(trans_data).type(torch.FloatTensor)
    # trans_label = torch.from_numpy(trans_label).type(torch.LongTensor)

    # LSTM网络
    net = RNN(input_size=dimension,hidden_size=128,num_layers=1,bidirectional=True,batch_first=True,out_size_fc1=128, out_size=2)
    net = net.to(DEVICE)

    loss_fn = nn.MSELoss(reduce=False, size_average=False)

    optimizer = torch.optim.Adamax(net.parameters(), lr=0.001, weight_decay=0.0001)

    model_path = r'./Model/TEST-LSTM.pkl'

    batch_size =50
    #####################################   因为各自的数量不一致，故随机取值训练时，需要分开取值
    num_train_sample=np.shape(x_data)[0]
    num_test_sample = np.shape(x_data1)[0]

    display_step=10

    N_EPOCH = 500
    #训练，返回训练的模型

    temp_count=0

    intial_ref = 170
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

            # print("WHAT:",train_batch_xs.size())

            train_batch_xs, train_batch_ys = train_batch_xs.to(DEVICE), train_batch_ys.to(DEVICE)
            # train_batch_xs, train_batch_ys=Variable(train_batch_xs),Variable(train_batch_ys)



            net.train()  # 开始训练正常的网络模型

            #计算 Jnn

            y_src= net(train_batch_xs)  #

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

                net.eval()
                yspred = net(xs)
                y1spred = net(x1s)

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
                    torch.save(net.state_dict(), model_path)
                    auto_save_model_ref = test_loss
                    print("    Save model: ", auto_save_model_ref)

                if test_loss < 110:  # 设定一个经验上限
                    torch.save(net.state_dict(), model_path)
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


