import csv
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math

datapath='../studentdataset.csv'

student_list = []
male_list = []
female_list = []


def loadDataset():
    with open(datapath, 'r', newline="")as f:
        rows = csv.reader(f)
        cnt = 0
        for row in rows:
            if cnt == 0:
                cnt += 1
                continue
            if row[3] == '1':
                male_list.append(row[0:1]+row[2:])
            elif row[3] == '0':
                female_list.append(row[0:1]+row[2:])
            student_list.append(row[0:1]+row[2:])

    return np.array(student_list).astype(dtype=np.float32),np.array(male_list).astype(dtype=np.float32),np.array(female_list).astype(dtype=np.float32)


def distance(lst1,lst2):
    ans = 0
    for i in range(len(lst1)):
        ans += (lst1[i]-lst2[i])**2
    return math.sqrt(ans)


def take_dis(elem):
    return elem[0]


def compute_k_nn(train,lst,k):
    distances = []
    for x in train:
        # print(x)
        dis = distance(lst,x)
        distances.append((dis,x[2]))
    distances.sort(key=take_dis)
    # print(distances)

    cnt_1 = 0
    cnt_0 = 0

    for i in range(k):
        if distances[i][1]==1.0:
            cnt_1+=1
        elif distances[i][1]==0.0:
            cnt_0+=1
    # print(cnt_1,cnt_0)
    if cnt_1>cnt_0:
        return 1
    else: return 0

def draw(X,K,boy_test,girl_test,accuracy):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # 生成随机数据来做测试集，然后作预测
    # x_min--x_max,步长为 0.02----等差数列
    # xx,yy分别是X的两个特征的其中一个
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                         np.arange(y_min, y_max, 1))

    # np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等。
    # np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。
    # Z为测试集的数据
    # Z=compute_k_nn(train,lst,K)


    Z = np.c_[xx.ravel(), yy.ravel()]
    zz=[]
    for i in Z:
        zz.append(compute_k_nn(train,i,K))
    # 画出测试集数据
    zz=np.array(zz)
    Z = zz.reshape(xx.shape)
    plt.figure()

    # plt.pcolormesh(xx, yy, y_predict, cmap=cmap_light)
    # 作用：画出不同类型数据的色彩范围--区域
    # xx,yy：图像区域内的采样点--组织成一个点
    # y_predict：根据采样点计算出的每个点所属的类别
    # camp：将相应的值映射到颜色
    plt.pcolormesh(xx, yy, Z, cmap='rainbow')

    markers = ['o', 's', '^']
    colors = ['b', 'r', 'c']
    labels = np.unique(X[:,2])


    # 也画出所有的训练集数据
    plt.scatter(girl_test[:, 0], girl_test[:, 1], c='r')
    plt.scatter(boy_test[:, 0], boy_test[:, 1], c='b')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("2-Class classification k = "+str(K)+" accuracy:"+str(accuracy))
    plt.show()


if __name__ == '__main__':
    np_student, np_male, np_female = loadDataset()
    ks = [1,3,5,7,9,11]


    train,test = train_test_split(np_student,train_size=0.8,test_size=0.2,random_state=0)
    # print(train)
    # print(test)
    boy_test=[]
    girl_test=[]
    length=len(test)
    for i in test:
        if i[2]==1:
            boy_test.append(i)
        else:
            girl_test.append(i)
    for K in ks:

        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        predict=[]
        accuracy=0
        for lst in test:

            pred_class = compute_k_nn(train,lst,K)
            predict.append(pred_class)

            obj_class = lst[2]
            # print(clss)
            if pred_class==1 and obj_class==1:
                true_positive+=1
            elif pred_class==1 and obj_class==0:
                false_negative+=1
            elif pred_class==0 and obj_class==0:
                true_negative+=1
            elif pred_class==0 and obj_class==1:
                false_positive+=1

        accuracy=(true_positive+true_negative)/length
        print("K:",K)
        print("true_positive:",true_positive)
        print("true_negative:",true_negative)
        print("false_positive:",false_positive)
        print("false_negative:",false_negative)
        draw(test,K,np.array(boy_test),np.array(girl_test),accuracy)