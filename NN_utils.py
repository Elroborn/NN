'''
Created on 2018年12月24日

@author: coderwangson
'''
"#codeing=utf-8"
import numpy as np
def tanh(z):
    r = (np.exp(z)-np.exp(-z))/(np.exp(z) + np.exp(-z))
    return r

def tanh_prime(z):
    return 1- tanh(z)**2

def relu(z):
    return np.maximum(z,0)

def relu_prime(z):
    return z>0
def softmax(z):
    # 因为有ez的操作，避免溢出
    return np.exp(z)/np.sum(np.exp(z),axis = 0,keepdims = True)
# 把x分成一个个小batch，存入batches，并且把数据集进行打乱返回
def get_bathces(x,y,batch_size):
    train_data = np.vstack((x,y))    
    batches = [train_data[:,k:k + batch_size] for k in range(0, train_data.shape[1], batch_size)]
    # 只对第一个维度操作
    np.random.shuffle(train_data.T)
    x = train_data[0:784,:]
    y = train_data[784:,]
    return x,y,batches
def one_hot(y):
    y_onehot = np.zeros((10,y.shape[1]))
    for i in range(y.shape[1]):
        y_onehot[y[0,i]][i] = 1
    return y_onehot
