import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_curve, auc


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






dataSet = pd.read_csv('studentdataset.csv')

dataSetNP = np.array(dataSet)
np.random.shuffle(dataSetNP)
testData=dataSetNP[0:500]
boy_dataNP = []
girl_dataNP = []
for x in dataSetNP:
    if x[3] == 1:
        boy_dataNP.append(x)
    else:
        girl_dataNP.append(x)
boy_dataNP = np.array(boy_dataNP)
girl_dataNP = np.array(girl_dataNP)
# print(boy_dataNP.shape[0])
# print(girl_dataNP.shape[0])
np.random.shuffle(boy_dataNP)
np.random.shuffle(girl_dataNP)

dataSetNP = np.vstack((boy_dataNP[0:250], girl_dataNP[0:250]))

trains = dataSetNP[:, 0:dataSetNP.shape[1] - 1]
labels = dataSetNP[:, dataSetNP.shape[1] - 1]
labels = list(labels)
Px = {}
# 先验概率控制为0.5对0.5#
Px[0] = labels.count(0) / len(labels)
Px[1] = labels.count(1) / len(labels)
boy_sum = 0
girl_sum = 0
for x in dataSetNP:
    if x[3] == 1:
        boy_sum += x[0]
    else:
        girl_sum += x[0]
boy_average = boy_sum / labels.count(1)
girl_average = girl_sum / labels.count(0)
# print(boy_average, girl_average)
boy_sig = 0
girl_sig = 0
for x in dataSetNP:
    if x[3] == 1:
        boy_sig += (x[0] - boy_average) ** 2
    else:
        girl_sig += (x[0] - girl_average) ** 2
boy_sig = (boy_sig / labels.count(1)) ** 0.5
girl_sig = (girl_sig / labels.count(0)) ** 0.5

def bayes_2(test_sample):
    test_sample = np.mat(test_sample).T
    pb_test_sample = m_pdf(test_sample, np.mat(xfcBoy), np.mat(averageBoy).T)
    pg_test_sample = m_pdf(test_sample, np.mat(xfcGirl), np.mat(averageGirl).T)
    ppb_test_sample = pb_test_sample * Px[1] / (pb_test_sample * Px[1] + pg_test_sample * Px[0])
    ppg_test_sample = pg_test_sample * Px[0] / (pb_test_sample * Px[1] + pg_test_sample * Px[0])
    return (ppb_test_sample - ppg_test_sample)

def bayes_1_draw():
    x = np.linspace(140, 190, 50)
    by = np.exp(-(x - boy_average) ** 2 / (2 * (boy_sig ** 2))) / (math.sqrt(2 * math.pi) * boy_sig)
    gy = np.exp(-(x - girl_average) ** 2 / (2 * (girl_sig ** 2))) / (math.sqrt(2 * math.pi) * girl_sig)
    byy = by * Px[1] / (by * Px[1] + gy * Px[0])
    gyy = gy * Px[0] / (by * Px[1] + gy * Px[0])
    plt.plot(x, by, 'r', label='boy')
    plt.plot(x, gy, 'b', label='girl')
    plt.legend(loc='center right')
    plt.xlabel('Height')
    plt.ylabel('Probability Density')
    plt.title('Class conditional probability')
    plt.show()

    plt.plot(x, byy, 'r', label='boy')
    plt.plot(x, gyy, 'b', label='girl')
    plt.legend(loc='center right')
    plt.xlabel('Height')
    plt.ylabel('Probability')
    plt.title('Posterior probability')
    plt.show()


def bayes_height(test_sample):
    pb_test_sample = np.exp(-(test_sample - boy_average) ** 2 / (2 * (boy_sig ** 2))) / (
            math.sqrt(2 * math.pi) * boy_sig)
    pg_test_sample = np.exp(-(test_sample - girl_average) ** 2 / (2 * (girl_sig ** 2))) / (
            math.sqrt(2 * math.pi) * girl_sig)
    ppb_test_sample = pb_test_sample * Px[1] / (pb_test_sample * Px[1] + pg_test_sample * Px[0])
    ppg_test_sample = pg_test_sample * Px[0] / (pb_test_sample * Px[1] + pg_test_sample * Px[0])
    return (ppb_test_sample - ppg_test_sample)


def m_pdf(train_array, xfc, average):
    d = train_array.shape[1]
    temp = 1 / ((2 * math.pi) ** (d / 2) * (np.linalg.det(xfc) ** 0.5))
    t = -1 / 2 * (train_array - average).T * xfc.I * (train_array - average)
    return float(temp * np.exp(t))


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
averageBoy = sum(trainDataBoy) / trainDataBoy.shape[0]
averageBoy = averageBoy[0:2]
# print(averageBoy)
xfcBoy = np.cov(trainDataBoy.T)

averageGirl = sum(trainDataGirl) / trainDataGirl.shape[0]
averageGirl = averageGirl[0:2]
# print(averageGirl)
xfcGirl = np.cov(trainDataGirl.T)


def bayes_2_draw():
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
            xxxx = np.mat((i, j)).T
            bbb.append(m_pdf(xxxx, np.mat(xfcBoy), np.mat(averageBoy).T))
            ggg.append(m_pdf(xxxx, np.mat(xfcGirl), np.mat(averageGirl).T))
        ZB.append(bbb)
        ZG.append(ggg)
    ZB = np.array(ZB)
    ZG = np.array(ZG)
    ax.plot_surface(XX, YY, ZB, rstride=1, cstride=1, cmap='Blues')
    ax.plot_surface(XX, YY, ZG, rstride=1, cstride=1, cmap='Oranges')
    plt.show()


def bayes2_test():
    right = 0
    print("二维（身高，体重）：")
    real = []
    predict = []
    for x in testData:
        c = bayes_2((float(x[0]), float(x[1])))
        real.append(x[3])
        predict.append(c)
        if c > 0:
            c = 1
        else:
            c = 0
        if c == x[3]: right += 1
    print(right / dataSetNP.shape[0])
    # print(ratio)
    fpr, tpr, thresholds = roc_curve(real, predict, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 1
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()

def bayes1_test():
    TP = 0
    FP = 0
    FN = 0
    TN = 0  # P:男，N:女
    real = []
    predict = []
    for x in testData:
        c = bayes_height(float(x[0]))
        real.append(x[3])
        predict.append(c)
        if c > 0:
            c = 1
        else:
            c = 0
        if (x[3] == 1 and 1 == c):
            TP += 1
        if (x[3] == 1 and 0 == c):
            FP += 1
        if (x[3] == 0 and 0 == c):
            TN += 1
        if (x[3] == 0 and 0 == c):
            FN += 1
    ratio = (TP + TN) / dataSetNP.shape[0]
    print("一维（身高）：")
    # print(wrong / dataSetNP.shape[0])
    print(ratio)
    fpr, tpr, thresholds = roc_curve(real, predict, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 1
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()


bayes_1_draw()
bayes_2_draw()
bayes1_test()
bayes2_test()
