import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#txt
# def loadDataSet(fileName):
#     numFeat = len(open(fileName).readline().split('\t')) - 1
#     dataMat = []; labelMat = []
#     fr = open(fileName)
#     for line in fr.readlines():
#         lineArr =[]
#         curLine = line.strip().split('\t')
#         for i in range(numFeat):
#             lineArr.append(float(curLine[i]))
#         dataMat.append(lineArr)
#         labelMat.append(float(curLine[-1]))
#     return dataMat,labelMat


#csv
def getTrainSet(filename):
    dataSet = pd.read_csv(filename)
    # print(dataSet)
    dataSetNP = np.array(dataSet)  #将数据由dataframe类型转换为数组类型
    trainData = dataSetNP[:,0:dataSetNP.shape[1]-1]   #训练数据x1,x2
    labels = dataSetNP[:,dataSetNP.shape[1]-1]        #训练数据所对应的所属类型Y
    return trainData, labels

def classify(trainData, labels, features):
    labels = list(labels)    #转换为list类型
    #求先验概率
    prior_P = {}
    labels_set=set(labels)
    for label in labels_set:
        prior_P[label] = labels.count(label)/len(labels)
    #求条件概率
    conditional_P = {}

    for label in prior_P.keys():
        y_index = [i for i, y in enumerate(labels) if y == label]   # y在labels中的所有下标
        y_count = labels.count(label)     # y在labels中出现的次数
        for j in range(len(features)):
            pkey = str(features[j]) + '|' + str(label)
            x_index = [i for i, x in enumerate(trainData[:,j]) if x == features[j]]   # x在trainData[:,j]中的所有下标
            xy_count = len(set(x_index) & set(y_index))   #x y同时出现的次数
            conditional_P[pkey] = (xy_count) / y_count  #条件概率
    #features所属类
    print(conditional_P)
    F = {}
    # print(P)
    for label in prior_P.keys():
        F[label] = prior_P[label]
        for x in features:
            F[label] = F[label] * conditional_P[str(x)+'|'+str(label)]
            print(F)

    features_y = max(F, key=F.get)  # 概率最大值对应的类别
    return features_y

trainData, labels=getTrainSet('studentdataset.csv')
# # print(classify(trainData, labels,[180,50,40]))
# m=np.mat(trainData)
# print(m.shape)
# print(m[:2,:2])
# a=np.ones()
# print(np.ones(2))
# print(np.sum(m[:2,:2]))
#
# class kernal:
#     def gaussian(self,x):
#     pass
#
#
# def parzen(train_mat,hn,X):
#     pass


def m_pdf(train_array,xfc,average):

    d=train_array.shape[1]

    temp=1/((2*math.pi)**(d/2)*(np.linalg.det(xfc)**0.5))

    t=-1/2*(train_array-average).T*xfc.I*(train_array-average)

    return temp*np.exp(t)


x=[[180,70]]
x=np.mat(x)

# a = np.expand_dims(trainData, axis=1)
# print(a)

# b=np.expand_dims(x, axis=0)
# print("b",b)
# print(b-a)
# print(trainData-x)
# b = np.all(np.less(np.abs(trainData-x), 1/2), axis=-1)
# print(b)
# average= sum(trainData)/trainData.shape[0]
# average=average[0:2]
# xfc=np.cov(trainData.T)
# print(xfc[0:100])
#
# print(m_pdf(x,np.mat(xfc),np.mat(average).T))



def gaussian_window(train_mat, mu, h):
    N, d = train_mat.shape
    a = (train_mat - mu) / h
    b=[]
    for i in a:
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



dataSet = pd.read_csv('studentdataset.csv')
dataSetNP = np.array(dataSet)
trains = dataSetNP[:, 0:dataSetNP.shape[1] - 1]
labels = dataSetNP[:, dataSetNP.shape[1] - 1]
labels = list(labels)
Px = {}
Px[0] = labels.count(0) / len(labels)
Px[1] = labels.count(1) / len(labels)
print(Px[0], Px[1])
boy = []
girl = []
for x in dataSetNP:
    if x[3] == 1:
        boy.append(x)
    else:
        girl.append(x)
boy = np.array(boy)
girl = np.array(girl)
trainDataBoy = boy[:, 0:2]
trainDataGirl = girl[:, 0:2]
trainData=trains[:,0:2]

fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(140, 200, 1)
Y = np.arange(30, 90, 1)
xx = np.hstack((X, Y))
XX, YY = np.meshgrid(X, Y)

ZB = []
ZG = []
for i in X:
    bbb = []
    ggg = []
    for j in Y:
        xxxx = np.mat((i, j))
        bbb.append(hypercube_kernel(trainDataBoy,xxxx,10))
        ggg.append(hypercube_kernel(trainDataGirl,xxxx,10))
    ZB.append(bbb)
    ZG.append(ggg)

ZB = np.array(ZB)
ZG = np.array(ZG)
ax.plot_surface(XX, YY, ZB, rstride=1, cstride=1, cmap='rainbow')
ax.plot_surface(XX, YY, ZG, rstride=1, cstride=1, cmap='rainbow')
plt.show()


# print(gaussian_window(trainData,x,8))
# print(hypercube_kernel(trainData,x,1))

def judge(xxxx,h):
    g1=hypercube_kernel(trainDataBoy,xxxx,h)*Px[1]
    g0=hypercube_kernel(trainDataGirl, xxxx, h)*Px[0]
    if(g1>g0):
        return 1
    return 0
success_num=0
num=len(labels)
print(num)
ratio=[]
for h in range(1,25):
    for i in range(num):
        # print(trainData[i])
        # print(i)
        # print(hypercube_kernel(trainDataBoy,trainData[i],40))
        # print(hypercube_kernel(trainDataGirl,trainData[i],40))
        if(labels[i]==judge(trainData[i],h)):
            success_num+=1
    ratio.append(success_num/num)
    success_num=0

print(ratio)
