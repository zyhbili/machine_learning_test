import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def gaussian_window(train_mat, mu, h):
    N, d = train_mat.shape
    a = (train_mat - mu) / h
    b=[]
    for i in a:
        i=np.mat(i)
        b.append(1/np.sqrt(np.pi * 2)*np.exp(-0.5*int(i*i.T)))
    kn=sum(b)
    Vn=h**d
    return kn / (N * Vn)

def hypercube_kernel(train_mat, mu, h):
    N, d = train_mat.shape
    a = (train_mat - mu) / h
    b = np.all(np.less(np.abs(a), 1 / 2), axis=1)
    kn = np.sum(b.astype(int))
    Vn=h**d
    return kn / (N * Vn)

def draw(func,h):
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(140, 200, 1)
    Y = np.arange(30, 90, 1)
    XX, YY = np.meshgrid(X, Y)
    ZB = []
    ZG = []
    for i in X:
        bbb = []
        ggg = []
        for j in Y:
            xxxx = np.mat((i, j))
            bbb.append(func(trainDataBoy,xxxx,h))
            ggg.append(func(trainDataGirl,xxxx,h))
        ZB.append(bbb)
        ZG.append(ggg)

    ZB = np.array(ZB)
    ZG = np.array(ZG)
    ax.plot_surface(XX, YY, ZB, rstride=1, cstride=1, cmap='Blues')
    ax.plot_surface(XX, YY, ZG, rstride=1, cstride=1, cmap='Oranges')
    plt.show()

def judge_1(trainDataBoy,trainDataGirl,X,h,p0,p1):
    g1=hypercube_kernel(trainDataBoy,X,h)*p1
    g0=hypercube_kernel(trainDataGirl,X,h)*p0
    if(g1>g0):
        return 1
    return 0

def judge_2(trainDataBoy,trainDataGirl,X,h,p0,p1):
    g1=gaussian_window(trainDataBoy,X,h)*p1
    g0=gaussian_window(trainDataGirl,X,h)*p0
    if(g1>g0):
        return 1
    return 0

dataSet = pd.read_csv('studentdataset.csv')
dataSetNP = np.array(dataSet)
trains = dataSetNP[:, 0:dataSetNP.shape[1] - 1]
labels = dataSetNP[:, dataSetNP.shape[1] - 1]
labels = list(labels)
Px = {}
Px[0] = labels.count(0) / len(labels)
Px[1] = labels.count(1) / len(labels)
# print(Px[0], Px[1])
boys = []
girls = []
for x in dataSetNP:
    if x[3] == 1:
        boys.append(x)
    else:
        girls.append(x)
boys = np.array(boys)
girls = np.array(girls)
# 只取前两维便于画图
trainDataBoy = boys[:, 0:2]
trainDataGirl = girls[:, 0:2]
trainData=trains[:,0:2]
h=5
draw(hypercube_kernel,h)
T_ratio=[]
kf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=0)
for train_index, test_index in kf.split(trains):
    # print('train_index', train_index, 'test_index', test_index)
    girls=[];boys=[]
    TP=0;FP=0;FN=0;TN=0#P:男，N:女
    train_len=len(train_index)
    for i in train_index:
        if(labels[i]==1):
            boys.append(trains[i])
        else:
            girls.append(trains[i])

    boys=np.array(boys);girls=np.array(girls);
    Px={}
    Px[0]=len(girls)/train_len
    Px[1]=len(boys)/train_len
    for i in test_index:
        c=judge_1(boys,girls,trains[i],h,Px[0],Px[1])
        if(labels[i]==1 and 1==c):
            TP+=1
        if(labels[i]==1 and 0==c):
            FP+=1
        if(labels[i]==0 and 0==c):
            TN+=1
        if(labels[i]==0 and 0==c):
            FN+=1
    ratio=(TP+TN)/len(test_index)
    T_ratio.append(ratio)
print(sum(T_ratio)/len(T_ratio))







# for h in range(1,25):
#     for i in range(num):
#         if(labels[i]==judge(trainData[i],h)):
#             success_num+=1
#     ratio.append(success_num/num)
#     success_num=0
# print(ratio)
