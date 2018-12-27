import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler

def get_amended_data(d):
    dataSet = pd.read_csv('../studentdataset.csv')
    dataSetArr = np.array(dataSet)
    boys=[];girls=[]
    # sc_X = StandardScaler()#归一化
    # train_std = sc_X.fit_transform(dataSetArr[:,0:3])
    #
    # labels=dataSetArr[:,-1]
    # labels=np.mat(labels).T
    # labels =np.array(labels)
    # dataSetArr= np.hstack((train_std,labels))
    # sumtrain = sum(dataSetArr)/len(dataSetArr)
    # print(sumtrain)
    min0=np.min(dataSetArr[:,0])
    min1=np.min(dataSetArr[:,1])
    min2=np.min(dataSetArr[:,2])
    max0=np.max(dataSetArr[:,0])
    max1=np.max(dataSetArr[:,1])
    max2=np.max(dataSetArr[:,2])

    dataSetArr[:,0]=(dataSetArr[:,0]-np.mat(min0))/(max0-min0)
    dataSetArr[:,1]=(dataSetArr[:,1]-np.mat(min1))/(max1-min1)
    dataSetArr[:,2]=(dataSetArr[:,2]-np.mat(min2))/(max2-min2)

    print(dataSetArr)

    for x in dataSetArr:
        if x[3] == 1:
            boys.append(x)
        else:
            girls.append(x)

    boys = np.array(boys)
    girls = np.array(girls)
    trainDataBoys = boys[:, 0:d]
    trainDataGirls = -girls[:, 0:d]

    temp = np.ones((len(boys), 1), int)
    trainDataBoys = np.hstack((trainDataBoys,temp))

    temp = np.ones((len(girls), 1), int)
    temp=-temp
    trainDataGirls = np.hstack((trainDataGirls,temp))

    return np.append(trainDataBoys,trainDataGirls,axis=0),trainDataBoys,trainDataGirls

def draw(trainDataGirls,trainDataBoys,w,d):
    color = {0: 'r', 1: 'b'}
    plt.scatter(-trainDataGirls[:, 0], -trainDataGirls[:, 1], color=color[0])
    plt.scatter(trainDataBoys[:, 0], trainDataBoys[:, 1], color=color[1])
    x = np.linspace(0, 1, 50)
    plt.xlabel('height')
    plt.ylabel('weight')
    y0 = (-float(w[0]) * x - float(w[d])) / float(w[1])

    plt.plot(x, y0, color='green', label='w0')
    plt.show()

'''只能画2d图'''
def train(d,lr):
    amended_data,trainDataBoys,trainDataGirls=get_amended_data(d)
    amended_data=np.mat(amended_data)
    # plt.scatter(-trainDataGirls[:, 0], -trainDataGirls[:, 1], color=color[0])
    # plt.scatter(trainDataBoys[:, 0], trainDataBoys[:, 1], color=color[1])
    length=len(amended_data)

    w = np.mat(np.ones((d+1, 1), int))
    w_list=[]
    for i in range(10000):
        result=amended_data*w

        judge = np.all(np.less(result, 0), axis=1)
        err_count = np.sum(judge.astype(int))
        print("错分个数:",err_count)
        if i%10==0:
            draw(trainDataGirls,trainDataBoys,w,d)
        if err_count<120:
            draw(trainDataGirls, trainDataBoys, w, d)
            break
        w=w.T
        w_list.append(np.array(w)[0])
        for i in range(length):
            if judge[i]:
                w=w+lr*amended_data[i]
        w=w.T
    return w_list



# w_list=np.array(w_list)
def train3D(lr):
    amended_data,trainDataBoys,trainDataGirls=get_amended_data(3)
    amended_data=np.mat(amended_data)
    length=len(amended_data)

    # draw3D(trainDataGirls,trainDataBoys,w)
    w = np.mat(np.ones((4, 1), int))
    for i in range(10000):
        result = amended_data * w
        judge = np.all(np.less(result, 0), axis=1)
        err_count = np.sum(judge.astype(int))
        print("错分个数:", err_count)
        if i%20==0:
            draw3D(trainDataGirls,trainDataBoys,w)
        if err_count<35:
            draw3D(trainDataGirls, trainDataBoys, w)
            break

        w = w.T
        for i in range(length):
            if judge[i]:
                w = w + lr * amended_data[i]
        w = w.T

def draw3D(trainDataGirls,trainDataBoys,w):

    boys_height, boys_weight, boys_size = trainDataBoys[:, 0], trainDataBoys[:, 1], trainDataBoys[:, 2]
    girls_height, girls_weight, girls_size = trainDataGirls[:, 0], trainDataGirls[:, 1], trainDataGirls[:, 2]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(boys_height, boys_weight, boys_size, c='b')
    ax.scatter(-girls_height, -girls_weight,- girls_size, c='r')
    ax.set_zlabel('size')  # 坐标轴
    ax.set_ylabel('weight')
    ax.set_xlabel('height')

    X = np.arange(0, 1, 0.1)
    Y = np.arange(0, 1, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = (-float(w[0]) * X - float(w[1])*Y-w[3]) / float(w[2])
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

    plt.show()

train3D(0.01)
train(2,0.01)