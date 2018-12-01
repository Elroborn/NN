import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
def createData():
    df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                     header=None)  # 加载Iris数据集作为DataFrame对象
    X = df.iloc[:, [0, 2]].values  # 取出2个特征，并把它们用Numpy数组表示
    X = X[:]  # 复制一个切片取出后面100个用例
    X = np.c_[X, np.ones(150)]  # 矩阵增加一列，常数列
    # print(X)
    y = np.zeros((150, 2))
    for i in range(100):
        if i < 50:
            y[i][0] = 1
        else:
            y[i][1] = 1
    # print(y)
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')  # 前50个样本的散点图
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')  # 中间50个样本的散点图
    plt.scatter(X[100:, 0], X[100:, 1], color='green', marker='+', label='Virginica')  # 后50个样本的散点图
    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    plt.legend(loc=2)  # 把说明放在左上角，具体请参考官方文档
    return X,y
 
def h(lableMatrix, weights, dataMatrix):
    m, n = np.shape(lableMatrix)
    divided = np.ones(m)
    error = np.zeros((m, n))
    # print(m, n)
    for i in range(m):
        for j in range(n):
            # print(weights[:, j])
            # print(dataMatrix[:, i])
            divided[i] = divided[i] + np.exp(weights[:, j].transpose() * dataMatrix[:, i])
    # print(divided[0])
    for i in range(m):
        for j in range(n):
            error[i][j] = np.exp(weights[:, j].transpose() * dataMatrix[:, i]) / divided[i]
    error = lableMatrix - error
    # print(error)
    return error
 
def gradAscent(dataMat, classLabels):
    dataMatrix = np.mat(dataMat).transpose()
    labelMatrix = np.mat(classLabels)
    # print(dataMatrix)
    # print(labelMatrix)
    n, m = np.shape(dataMatrix)
    # print(n, m)
    alpha = 0.01
    maxCycles = 2000
    weights = np.ones((n, 2))
 
    for i in range(maxCycles):
        error = h(labelMatrix, weights, dataMatrix)
        weights = weights + alpha * dataMatrix * error
        print(dataMatrix * error)
        print(np.dot(dataMatrix, error))
        # weights = weights + alpha * np.dot(dataMatrix, error)
        # print(dataMatrix.shape)
        # print(error.shape)
        # print(dataMatrix * error)
 
    # for k in range(maxCycles):  # 随机梯度下降
    #     for i in range(m):
    #         error = h(labelMatrix[i], weights, dataMatrix[:, i])
    #         weights = weights + alpha * dataMatrix[:, i] * error
    #         print(dataMatrix[:, i] * error)
 
    return weights
 
 
if __name__ == "__main__":
    X, y = createData()
    weights = gradAscent(X, y)
    print(weights)
 
    yy1 = np.zeros(40)
    xx1 = np.arange(4, 8, 0.1)  # 定义x的范围，像素为0.1
    for i in range(len(xx1)):
        yy1[i] = (- weights[:, 0][2] - weights[:, 0][0] * xx1[i]) / weights[:, 0][1]
    plt.plot(xx1, yy1, color='yellow')
 
    yy2 = np.zeros(40)
    xx2 = np.arange(4, 8, 0.1)  # 定义x的范围，像素为0.1
    for i in range(len(xx2)):
        yy2[i] = (- weights[:, 1][2] - weights[:, 1][0] * xx2[i]) / weights[:, 1][1]
    plt.plot(xx2, yy2, color='red')
 
    plt.show()
 
