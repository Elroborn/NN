'''
Created on 2018年11月19日

@author: coderwangson
'''
"#codeing=utf-8"

import numpy as np
import matplotlib.pyplot as plt
import data_load
import NN_utils
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
def initialize_velocity(parameters):
    L = len(parameters) // 2 #神经网络的层数
    v = {}
    for i in range(1,L+1):
        v["dw" + str(i)] = np.zeros_like(parameters["w" + str(i)])
        v["db" + str(i)] = np.zeros_like(parameters["b" + str(i)]) 
    return v

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
        a_temp = NN_utils.tanh(z_temp)
        # 生成drop的结点
        d_temp = np.random.rand(z_temp.shape[0],z_temp.shape[1])
        d_temp = d_temp < keep_prob
        a_temp = (a_temp * d_temp)/keep_prob
        z.append(z_temp)
        a.append(a_temp)
        d.append(d_temp)
    # 最后一层不用sigmoid,也不用dropout
    z_temp = parameters["w"+str(layers)].dot(a[layers-1]) + parameters["b"+str(layers)]
    a_temp = NN_utils.softmax(z_temp)
    z.append(z_temp)
    a.append(a_temp)
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
        grades["dz"+str(i)] = da_temp * NN_utils.tanh_prime(caches["z"][i])
        grades["dw"+str(i)] = grades["dz"+str(i)].dot(caches["a"][i-1].T)/m
        grades["db"+str(i)] = np.sum(grades["dz"+str(i)],axis = 1,keepdims = True) /m
    return grades
# 就是把其所有的权重以及偏执都更新一下
def update_grades(parameters,grades,v,learning_rate,beta = .9):
    layers = len(parameters)//2
    for i in range(1,layers+1):
        # 更新v
        v["dw"+str(i)] = beta * v["dw"+str(i)] +(1-beta)*grades["dw"+str(i)]
        v["db"+str(i)] = beta * v["db"+str(i)] +(1-beta)*grades["db"+str(i)]
#         print(v["dw"+str(i)])
        parameters["w"+str(i)] -= learning_rate * v["dw"+str(i)]
        parameters["b"+str(i)] -= learning_rate * v["db"+str(i)]
    return parameters
# 计算误差值
def compute_loss(al,y):
    return  -np.sum(np.sum(y * np.log(al), axis=0))/(y.shape[1]) 
#进行测试
# x 784*None(60000) y :1* None(60000)
train_x,train_y,test_x,test_y = data_load.load_data()
train_x  = train_x/255.0
test_x = test_x/255.0
train_y = NN_utils.one_hot(train_y)
test_y = NN_utils.one_hot(test_y)
# 初始化参数
parameters = init_parameters([28*28,16,8,16,10])
# 动量法使用
v = initialize_velocity(parameters)
batch_size = 64
cost = []
for i in range(500):
    train_x,train_y,batches = NN_utils.get_bathces(train_x, train_y, batch_size)
    for batch in batches:
        # 因为batches是把x，y进行了堆叠，为了好进行shuffle
        train_x_tmp = batch[0:784,]
        train_y_tmp = batch[784:,]
        caches,al = forward(train_x_tmp, parameters,keep_prob=1)
        grades = backward(parameters,caches, al, train_y_tmp)
        parameters = update_grades(parameters, grades,v, learning_rate= .3)
    if i %1 ==0:
        # 随机抽取数据进行测试
        k = np.random.randint(0,len(batches))
        batch = batches[k]
        train_x_tmp = batch[0:784,]
        train_y_tmp = batch[784:,]
        caches,al = forward(train_x_tmp, parameters,keep_prob=1)
        print(np.mean(np.argmax(al,axis = 0)==np.argmax(train_y_tmp,axis = 0)))
        cost_ =compute_loss(al, train_y_tmp)
        print(cost_)
        cost.append(cost_)
#预测时候不进行dropout
caches,al = forward(test_x, parameters,keep_prob=1)
print(np.mean(np.argmax(al,axis = 0)==np.argmax(test_y,axis = 0)))
plt.plot(cost)
plt.show()