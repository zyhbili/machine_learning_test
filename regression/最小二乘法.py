import numpy as np
import matplotlib.pyplot as plt
import torch

def standRegression(xMat,yMat):
    xTx=xMat.T*xMat
    xTx=xTx
    if np.linalg.det(xTx)==0:#矩阵行列式为0
        print('无逆矩阵')
        return
    ws=xTx.I*(xMat.T*yMat)
    return ws
'''fake data'''
x=torch.linspace(-1,1,50)
x=torch.unsqueeze(x,dim=1)
y=x+0.5*torch.rand(x.size())
x,y=x.numpy(),y.numpy()
t = np.ones((50, 1), int)
rx = np.hstack((t,x))#增加一维为1，相当于y=kx+b中的b
'''fake data'''

ws=standRegression(np.mat(rx),np.mat(y))


y_predict=rx*ws
plt.scatter(x,y)   #原始数据图
plt.plot(x,y_predict)   #预测数据图
plt.show()