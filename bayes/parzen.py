import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_curve, auc
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

def draw(func,mtype,h):
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(140, 200, 1)
    Y = np.arange(30, 90, 1)
    XX, YY = np.meshgrid(X, Y)
    ZB = []
    ZG = []
    ax.set_title(mtype+" "+" h:"+str(h))
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

def drawROC(trains):
    h=5
    kf = RepeatedKFold(n_splits=3, n_repeats=10, random_state=0)
    for train_index, test_index in kf.split(trains):
        girls = [];boys = [];predict=[];real=[]
        train_len = len(train_index)
        for i in train_index:
            if (labels[i] == 1):
                boys.append(trains[i])
            else:
                girls.append(trains[i])

        boys = np.array(boys);
        girls = np.array(girls);
        Px = {}
        Px[0] = len(girls) / train_len
        Px[1] = len(boys) / train_len
        for i in test_index:
            c = f(boys, girls, trains[i], h, Px[0], Px[1])
            predict.append(c)
            real.append(labels[i])
        break
    fpr, tpr, thresholds = roc_curve(real,predict,pos_label=1)
    roc_auc = auc(fpr,tpr)

    plt.figure()
    lw = 1
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='3d ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    # plt.show()


    trains=trains[:,0:2]
    kf = RepeatedKFold(n_splits=3, n_repeats=10, random_state=0)
    for train_index, test_index in kf.split(trains):
        girls = [];boys = [];predict=[];real=[]
        train_len = len(train_index)
        for i in train_index:
            if (labels[i] == 1):
                boys.append(trains[i])
            else:
                girls.append(trains[i])

        boys = np.array(boys);
        girls = np.array(girls);
        Px = {}
        Px[0] = len(girls) / train_len
        Px[1] = len(boys) / train_len
        for i in test_index:
            c = f(boys, girls, trains[i], h, Px[0], Px[1])
            predict.append(c)
            real.append(labels[i])
        break
    fpr, tpr, thresholds = roc_curve(real,predict,pos_label=1)
    roc_auc = auc(fpr,tpr)

    # plt.figure()
    lw = 1
    # plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='Blue',
             lw=lw, label='2d ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC')
    plt.legend(loc="lower right")
    # plt.show()

    trains = trains[:, 0:1]
    kf = RepeatedKFold(n_splits=3, n_repeats=10, random_state=0)
    for train_index, test_index in kf.split(trains):
        girls = [];
        boys = [];
        predict = [];
        real = []
        train_len = len(train_index)
        for i in train_index:
            if (labels[i] == 1):
                boys.append(trains[i])
            else:
                girls.append(trains[i])

        boys = np.array(boys);
        girls = np.array(girls);
        Px = {}
        Px[0] = len(girls) / train_len
        Px[1] = len(boys) / train_len
        for i in test_index:
            c = f(boys, girls, trains[i], h, Px[0], Px[1])
            predict.append(c)
            real.append(labels[i])
        break
    fpr, tpr, thresholds = roc_curve(real, predict, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # plt.figure()
    lw = 1
    # plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='Red',
             lw=lw, label='1d ROC curve (area = %0.2f)' % roc_auc)
    plt.legend(loc="lower right")

    plt.show()





def judge_1(trainDataBoy,trainDataGirl,X,h,p0,p1):
    g1=hypercube_kernel(trainDataBoy,X,h)*p1
    g0=hypercube_kernel(trainDataGirl,X,h)*p0
    if(g1>g0):
        return 1
    return 0
def f(trainDataBoy,trainDataGirl,X,h,p0,p1):
    g1 = hypercube_kernel(trainDataBoy, X, h) * p1
    g0 = hypercube_kernel(trainDataGirl, X, h) * p0
    return g1-g0

def judge_2(trainDataBoy,trainDataGirl,X,h,p0,p1):
    g1=gaussian_window(trainDataBoy,X,h)*p1
    g0=gaussian_window(trainDataGirl,X,h)*p0
    if(g1>g0):
        return 1
    return 0


def ten_fold_verify(h):
    T_ratio=[]
    kf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=0)
    for train_index, test_index in kf.split(trains):
        # print('train_index', train_index, 'test_index', test_index)
        girls=[];boys=[];TPR_list=[];FPR_list=[]
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
            if(labels[i]==0 and 1==c):
                FN+=1
        ratio=(TP+TN)/len(test_index)
        TPR=TP/(TP+FN)
        FPR=FP/(TN+FP)
        TPR_list.append(TPR)
        FPR_list.append(FPR)
        T_ratio.append(ratio)

    print(sum(T_ratio)/len(T_ratio))
    return sum(T_ratio)/len(T_ratio)


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
# draw(hypercube_kernel,"hypercube_kernel",1)
acc=[]
for h in range(1,30):
    acc.append(ten_fold_verify(h))
# drawROC(trains)
temp=[i for i in range(1,30)]
plt.figure()
plt.plot(temp,acc,'r',label='acc')
plt.legend(loc='center right')
plt.xlabel('h-range')
plt.ylabel('acc')
plt.title('acc-h')
plt.show()



# draw(gaussian_window,"gaussian",1)




