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
    sc_X = StandardScaler()#归一化
    train_std = sc_X.fit_transform(dataSetArr[:,0:3])

    labels=dataSetArr[:,-1]
    labels=np.mat(labels).T
    labels =np.array(labels)
    dataSetArr= np.hstack((train_std,labels))

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
    x = np.linspace(-2, 2, 50)
    plt.xlabel('height')
    plt.ylabel('weight')
    y0 = (-float(w[0]) * x - float(w[d])) / float(w[1])

    plt.plot(x, y0, color='green', label='w0')
    plt.show()

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
        if i%5==0:
           draw(trainDataGirls,trainDataBoys,w,d)
        if err_count<100:
            draw(trainDataGirls, trainDataBoys, w, d)
            break
        w=w.T
        w_list.append(np.array(w)[0])
        for i in range(length):
            if judge[i]:
                w=w+lr*amended_data[i]
        w=w.T
    return w_list



w_list=train(2,0.1)
w_list=np.array(w_list)
