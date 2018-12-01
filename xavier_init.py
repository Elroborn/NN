'''
Created on 2018年11月19日

@author: coderwangson
'''
"#codeing=utf-8"

import numpy as np
import matplotlib.pyplot as plt
# xavier 进行参数初始化 node_in代表左边的 node_out代表右边的
def xavier_init(node_in, node_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (node_in + node_out))
    high = constant * np.sqrt(6.0 / (node_in + node_out))
    return np.random.uniform(low,high,(node_in, node_out))

# 生成权重以及偏执项layers_dim代表每层的神经元个数，
#比如[2,3,1]代表一个三成的网络，输入为2层，中间为3层输出为1层
def init_parameters(layers_dim):
    
    L = len(layers_dim)
    parameters ={}
    for i in range(1,L):
        parameters["w"+str(i)] = xavier_init(layers_dim[i],layers_dim[i-1])
        parameters["b"+str(i)] = np.zeros((layers_dim[i],1))
    return parameters

def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z) + np.exp(-z))

def tanh_prime(z):
    return 1- tanh(z)**2

# 前向传播，需要用到一个输入x以及所有的权重以及偏执项，都在parameters这个字典里面存储
# 最后返回会返回一个caches里面包含的 是各层的a和z，a[layers]就是最终的输出
def forward(x,parameters,keep_prob = 0.5):
    a = []
    z = []
    d = []
    caches = {}
    a.append(x)
    z.append(x)
    # 输入层不用删除
    d.append(np.ones(x.shape))
    layers = len(parameters)//2
    # 前面都要用sigmoid
    for i in range(1,layers):
        z_temp =parameters["w"+str(i)].dot(a[i-1]) + parameters["b"+str(i)]
        a_temp = tanh(z_temp)
        # 生成drop的结点
        d_temp = np.random.rand(z_temp.shape[0],z_temp.shape[1])
        d_temp = d_temp < keep_prob
        a_temp = (a_temp * d_temp)/keep_prob
        z.append(z_temp)
        a.append(a_temp)
        d.append(d_temp)
        
    # 最后一层不用sigmoid,也不用dropout
    z_temp = parameters["w"+str(layers)].dot(a[layers-1]) + parameters["b"+str(layers)]
    z.append(z_temp)
    a.append(z_temp)
    d.append(np.ones(z_temp.shape))
    
    caches["z"] = z
    caches["a"] = a
    caches["d"] = d
    caches["keep_prob"] = keep_prob    
    return  caches,a[layers]

# 反向传播，parameters里面存储的是所有的各层的权重以及偏执，caches里面存储各层的a和z
# al是经过反向传播后最后一层的输出，y代表真实值 
# 返回的grades代表着误差对所有的w以及b的导数
def backward(parameters,caches,al,y):
    layers = len(parameters)//2
    grades = {}
    m = y.shape[1]
    # 假设最后一层不经历激活函数
    # 就是按照上面的图片中的公式写的
    grades["dz"+str(layers)] = al - y
    grades["dw"+str(layers)] = grades["dz"+str(layers)].dot(caches["a"][layers-1].T) /m
    grades["db"+str(layers)] = np.sum(grades["dz"+str(layers)],axis = 1,keepdims = True) /m
    # 前面全部都是sigmoid激活
    for i in reversed(range(1,layers)):
        da_temp = parameters["w"+str(i+1)].T.dot(grades["dz"+str(i+1)])
        da_temp = (caches["d"][i] * da_temp)/caches["keep_prob"]
        grades["dz"+str(i)] = da_temp * tanh_prime(caches["z"][i])
        grades["dw"+str(i)] = grades["dz"+str(i)].dot(caches["a"][i-1].T)/m
        grades["db"+str(i)] = np.sum(grades["dz"+str(i)],axis = 1,keepdims = True) /m
    return grades

# 就是把其所有的权重以及偏执都更新一下
def update_grades(parameters,grades,learning_rate):
    layers = len(parameters)//2
    for i in range(1,layers+1):
        parameters["w"+str(i)] -= learning_rate * grades["dw"+str(i)]
        parameters["b"+str(i)] -= learning_rate * grades["db"+str(i)]
    return parameters
# 计算误差值
def compute_loss(al,y):
    return np.mean(np.square(al-y))

# 加载数据
def load_data():
    """
    加载数据集
    """
    x = np.arange(0.0,1.0,0.01)
    y =20* np.sin(2*np.pi*x)
    # 数据可视化
    plt.scatter(x,y)
    return x,y
#进行测试
x,y = load_data()
x = x.reshape(1,100)
y = y.reshape(1,100)
plt.scatter(x,y)
parameters = init_parameters([1,25,1])
al = 0
for i in range(4000):
    caches,al = forward(x, parameters,keep_prob=.9)
    grades = backward(parameters, caches, al, y)
    parameters = update_grades(parameters, grades, learning_rate= 0.1)
    if i %100 ==0:
        print(compute_loss(al, y))
#预测时候不进行dropout
caches,al = forward(x, parameters,keep_prob=1)
plt.scatter(x,al)
plt.show()
