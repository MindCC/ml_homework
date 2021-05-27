'''
多元线性回归与梯度下降法原理及公式推导
See:
https://blog.csdn.net/weixin_44344462/article/details/88989450
'''

import matplotlib.pyplot as plt
import numpy as np
import random

def predict(theta, X):
    '''
    根据当前的theta求Y的估计值
    传入的X的最左侧列为全1，即设x[i,0] = 1
    假设有n个样本、m个参数（含theta_0），则X为n*m, theta为m*1，Y为n*1
    '''
    # 确保theta为列向量
    theta = theta.reshape(-1, 1)
    Y = np.dot(X, theta)

    return Y



def gradient(theta, X, Y):
    '''
    求当前theta的梯度
    传入的X的最左侧列为全1，即设x[i,0] = 1
    '''
    # 调用predict函数根据当前theta估计Y
    P = predict(theta, X)
    # 共有n组数据
    n = X.shape[0]
    # 求解的theta个数
    m = X.shape[1]
    # 构建
    grad = np.zeros([m, 1])
    
    for i in range(m):
        grad[i, 0] = np.dot((P - Y).T, X[:, i]) / n
    
    return grad

def cost(theta, X, Y):
    '''
    求当前theta的残差平方和
    传入的X的最左侧列为全1，即设x[i,0] = 1
    '''
    P = predict(theta, X)
    n = X.shape[0]
    temp = Y - P
    
    return np.dot(temp.T, temp) / n



def gradient_descent(X, Y, Learning_rate = 0.01, ER = 1e-10, MAX_LOOP = 1e5):
    '''
    梯度下降法求解线性回归
    X的一行为一组数据
    Y为列向量，每一行对应X一行的计算结果
    学习率默认为0.01
    误差默认为1e-10
    默认最大迭代次数为1e5
    '''

    # 样本个数为
    n = X.shape[0]
    # 在X的最左侧拼接全1列
    X_0 = np.ones([n, 1])
    X = np.column_stack((X_0, X))
    print(X.shape)
    print(X)
    # 确保Y为列向量
    Y = Y.reshape(-1, 1)
    # 求解的未知元个数为
    m = X.shape[1]
    # 初始化theta向量
    theta = np.random.rand(m, 1)
    er = ER + 1 # 用来连续2次迭代目标函数的差
    ct = 0  	# 用来计算迭代次数
    current_cost = cost(theta, X, Y)
    while er > ER and ct < MAX_LOOP:
        # 更新theta
        grad =  gradient(theta, X, Y)
        theta = theta - Learning_rate * grad
        # 计算残差平方和最差
        last_cost = current_cost
        current_cost = cost(theta, X, Y)
        # print(last_cost, current_cost)
        er = abs(last_cost - current_cost)
        ct += 1
        # print(er,theta)
        
    return theta, ct


def stochastic_gradient_descent(X, Y, Learning_rate = 0.01, ER = 1e-10, MAX_LOOP = 1e5):
    '''
    随机梯度下降法求解线性回归
    X的一行为一组数据
    Y为列向量，每一行对应X一行的计算结果
    学习率默认为0.01
    误差默认为1e-10
    默认最大迭代次数为1e5

    由于每次只用一个样本计算梯度，而gradient默认X是2维的，如果X是1维的，需要用reshape进行处理
    '''

    # 样本个数为
    n = X.shape[0]
    # 在X的最左侧拼接全1列
    X_0 = np.ones([n, 1])
    X = np.column_stack((X_0, X))
    # 确保Y为列向量
    Y = Y.reshape(-1, 1)
    # 求解的未知元个数为
    m = X.shape[1]
    # 初始化theta向量
    theta = np.random.rand(m, 1)
    er = ER + 1 # 用来连续2次迭代目标函数的差
    ct = 0  	# 用来计算迭代次数
    current_cost = cost(theta, X, Y)
    while er > ER and ct < MAX_LOOP:
        # 更新theta
        i = np.random.randint(n)
        grad =  gradient(theta, X[i].reshape((1,m)), Y[i]) #Y[i].reshape((1,1)))
        theta = theta - Learning_rate * grad
        # 计算残差平方和最差
        last_cost = current_cost
        current_cost = cost(theta, X, Y)
        er = abs(last_cost - current_cost)
        ct += 1
        # print(er,theta)
        
    return theta, ct

def random_batch(m, size):
    '''
    从集合[0,1,...,m-1]中随机选择size个元素
    '''
    a = [i for i in range(m)]
    random.shuffle(a)
    return a[0:size]

def mini_batch_gradient_descent(X, Y, batch_size = 4, Learning_rate = 0.01, ER = 1e-10, MAX_LOOP = 1e5):
    '''
    小批量梯度下降法求解线性回归
    X的一行为一组数据
    Y为列向量，每一行对应X一行的计算结果
    小批量大小默认为4
    学习率默认为0.01
    误差默认为1e-10
    默认最大迭代次数为1e5
    '''

    # 样本个数为
    n = X.shape[0]
    # 在X的最左侧拼接全1列
    X_0 = np.ones([n, 1])
    X = np.column_stack((X_0, X))
    # 确保data_y为列向量
    Y = Y.reshape(-1, 1)
    # 求解的未知元个数为
    m = X.shape[1]
    # 初始化theta向量
    theta = np.random.rand(m, 1)
    er = ER + 1 # 用来连续2次迭代目标函数的差
    ct = 0  	# 用来计算迭代次数
    current_cost = cost(theta, X, Y)
    while er > ER and ct < MAX_LOOP:
        # 更新theta
        batch_index = random_batch(n, batch_size)
        grad =  gradient(theta, X[batch_index], Y[batch_index])
        theta = theta - Learning_rate * grad
        # 计算残差平方和最差
        last_cost = current_cost
        current_cost = cost(theta, X, Y)
        er = abs(last_cost - current_cost)
        ct += 1
        # print(er,theta)
        
    return theta, ct
          

def test_random_data():
    # =================== 样本数据生成 =======================
    # 生成数据以1元为例,要估计的theta数为1个
    num_of_features = 1
    num_of_samples = 2000
    # 设置噪声系数
    rate = 0
    X = []
    
    for i in range(num_of_features):
        X.append(np.random.random([1, num_of_samples]) * 10)
        
    X = np.array(X).reshape(num_of_samples, num_of_features)
    print("X的数据规模为 ： ", X.shape)
    
    # 利用方程生成X对应的Y
    Y = []

    for i in range(num_of_samples):
        Y.append(3 + 3.27 * X[i][0] + np.random.rand() * rate)

    Y = np.array(Y).reshape(-1, 1)
    print("Y的数据规模为 ： ", Y.shape)
    print("系数为3和3.27")
    # ======================================================
    
    # 计算并打印结果
    print("Gradient Descent:\n", gradient_descent(X, Y))

    print("Stochastic Gradient Descent:\n", stochastic_gradient_descent(X, Y))

    print("Mini-batch Gradient Descent:\n", mini_batch_gradient_descent(X, Y, 5))



if __name__ == '__main__':
    
    print(random_batch(10,3))
    
    test_random_data()

