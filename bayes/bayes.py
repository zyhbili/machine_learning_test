import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

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
print(classify(trainData, labels,[180,50,40]))