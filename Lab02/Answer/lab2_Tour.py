import math
#import random
from os import path
import numpy as np
import matplotlib.pyplot as plt
import TSPInstance

class Tour:
##    inst = None
    def __init__(self, inst, tour = None):
        self.__inst = inst
        self.__city_number = inst.citynum
        if tour == None:
            tour = list(range(inst.citynum))
            np.random.shuffle(tour)

        self.__tour = tour
        self.__tour_length = inst.evaluate(tour)
            
    @property
    def instance(self):
        return self.__inst;

    @property
    def citynum(self):
        return self.__city_number

    @property
    def tourlen(self):
        return self.__tour_length

    @property
    def tour(self):
        return self.__tour

##    @classmethod
##    def set_instance(cls, inst):
##        cls.inst = inst

    def random_inverse(self):
        '''
        Inverse the visiting order of cities in tour[i] to tour[j]
        '''
        i = np.random.randint(self.citynum)
        j = np.random.randint(self.citynum)
        while j == i:
            j = np.random.randint(self.citynum)
        if i > j:
            i, j = j, i
        
        tour = self.tour[:]
        temp = tour[i:j]
        temp.reverse()
        tour[i:j] = temp 
                
        neighbor = Tour(self.instance, tour)
        return neighbor


    def random_insert(self):
        '''
        Move the city in position j to position i
        '''
        i = np.random.randint(self.citynum)
        j = np.random.randint(self.citynum)
        while j == i:
            j = np.random.randint(self.citynum)

        tour = self.tour[:]
        city = tour[j]
        if i < j:
            tour[i+1:j+1] = tour[i:j]
        else:
            tour[j:i] = tour[j+1:i+1]
        tour[i] = city
        
        neighbor = Tour(self.instance, tour)
        return neighbor

    def random_swap(self):
        '''
        Move the city in position j to position i
        '''
        i = np.random.randint(self.citynum)
        j = np.random.randint(self.citynum)
        while j == i:
            j = np.random.randint(self.citynum)

        tour = self.tour[:]
        tour[i], tour[j] = tour[j], tour[i]
        
        neighbor = Tour(self.instance, tour)
        return neighbor    

    

    def __str__(self):
        return "Tour length:" + str(self.tourlen)

def random_search(solution, nt = "swap", iteration_times=1000):
    '''
    随机贪婪搜索，返回找到的最优解及搜索过程中解的变化
    只有新解比原来的解好才接受新解
    
    返回：
    solution  最优解
    lengths   解的变化
    '''
    lengths = np.zeros(iteration_times)
    for i in range(iteration_times):
        for j in range(len(solution.tour)):
            if nt == "swap":
                neighbor = solution.random_swap()
            elif nt == "insert":
                neighbor = solution.random_insert()
            else:
                neighbor = solution.random_inverse()
                
            if neighbor.tourlen < solution.tourlen:
                solution = neighbor
                
        lengths[i] = solution.tourlen

    return solution, lengths

def simulated_annealing(solution, nt = "swap", iteration_times=1000, t0=1000, alpha=0.99):
    '''
    模拟退火算法，返回找到的最优解及搜索过程中解的变化
    如果新解比原来的解好，则接受新解，否则概率接收，接收概率为：
    d = 新解 - 原解
    1.0 / math.exp(math.fabs(d)/t)
    see: https://baike.baidu.com/item/%E6%A8%A1%E6%8B%9F%E9%80%80%E7%81%AB%E7%AE%97%E6%B3%95/355508?fr=aladdin
    
    返回：
    best  最优解
    lengths   解的变化
    '''
    t = t0
    best = solution
    lengths = np.zeros(iteration_times)
    for i in range(iteration_times):
        for j in range(len(solution.tour)):
            if nt == "swap":
                neighbor = solution.random_swap()
            elif nt == "insert":
                neighbor = solution.random_insert()
            else:
                neighbor = solution.random_inverse()
            
            d = neighbor.tourlen - solution.tourlen
            if d <= 0:
                solution = neighbor
            else:
                try:
                    p = 1.0 / math.exp(math.fabs(d)/t)
                except:
                    p = 0
                if np.random.random() < p:
                    solution = neighbor

            if solution.tourlen < best.tourlen:
                best = solution
                    
        t *= alpha
        lengths[i] = solution.tourlen

    return best, lengths

def threshold_accepting(solution, nt = "swap", iteration_times=1000, t0=1000, alpha=0.99):
    '''
    阈值接收算法，返回找到的最优解及搜索过程中解的变化
    如果新解比原来的解不差于阈值t，则接受新解
    
    返回：
    best  最优解
    lengths   解的变化
    '''
    
    t = t0
    best = solution
    lengths = np.zeros(iteration_times)
    for i in range(iteration_times):
        for j in range(len(solution.tour)):
            if nt == "swap":
                neighbor = solution.random_swap()
            elif nt == "insert":
                neighbor = solution.random_insert()
            else:
                neighbor = solution.random_inverse()
            
            d = neighbor.tourlen - solution.tourlen
            if d < t:
                solution = neighbor
            if solution.tourlen < best.tourlen:
                best = solution
                    
        t *= alpha
        lengths[i] = solution.tourlen

    return best, lengths


def observe_convergence(inst, method, t):
    '''
    对指定方法method和邻域解生成方式t，观察搜索到的最优解及解的变化过程
    '''
    
    solution = Tour(instance)
    best, lengths = method(solution, t)
    print(method.__name__, t, best)
    inst.plot_tour(best.tour)

    x = np.array(range(len(lengths)))
    plt.plot(x, lengths, linewidth=2)
    plt.show()    


# def test_performance(inst, m, t, times = 25):
#     '''
#     对指定方法m和邻域解生成方式t进行性能测定
#     inst  TSP instance
#     m 指定的方法，randdom_search, threshold_accepting, or simulated_annealing
#     t: 邻域解生成方式，"swap","insert",or "inverse"
#     times: 重复运行方法m次数
#
#     将方法在每种邻域解生成方式下重复times次的最优解、最差解和平均解输出
#
#     '''
#
#     lengths = np.zeros(times)
#     for i in range(times):
#         solution = Tour(instance)
#         solution, lens = m(solution, t)
#         lengths[i] = solution.tourlen
#         print(m.__name__, t, i, solution.tourlen)
#
#     print(m.__name__, t, np.max(lengths), np.min(lengths), np.average(lengths))
#
#     return lengths


def compare_neighbor_type(inst, method, types, times = 25):
    '''
    对指定方法method，比较在types里面指定的邻域解生成方式算法的性能
    inst  TSP instance
    method 指定的方法，randdom_search, threshold_accepting, or simulated_annealing
    types: 邻域结构的元组，("swap","insert","inverse")的子元组

    将方法在每种邻域解生成方式下重复times次的最优解、最差解和平均解输出

    '''

    lengths = np.zeros((len(types),times))
    for t in range(len(types)):
        for i in range(times):
            solution = Tour(instance)
            solution, lens = method(solution, types[t])
            lengths[t,i] = solution.tourlen
            print(method.__name__, types[t], i, solution.tourlen)

    for t in range(len(types)):
        print(method.__name__, types[t], np.max(lengths[t]), np.min(lengths[t]), np.average(lengths[t]))

    return lengths

def compare_method_type(inst, methods, t = "inverse", times = 25):
    '''
    对指定邻域类型t，比较在methods里面指定的算法的性能
    inst  TSP instance
    methods 方法的元组，（randdom_search, threshold_accepting, simulated_annealing）的子元组
    t: 邻域解生成方式，"swap","insert",or "inverse"

    将各种方法在指定邻域解生成方式下重复times次的最优解、最差解和平均解输出

    '''    
    lengths = np.zeros((len(methods),times))
    for m in range(len(methods)):
        for i in range(times):
            solution = Tour(instance)
            solution, lens = methods[m](solution, t)
            lengths[m,i] = solution.tourlen
            print(methods[m].__name__, t, i, solution.tourlen)

    for m in range(len(methods)):
        print(methods[m].__name__, t, np.max(lengths[m]), np.min(lengths[m]), np.average(lengths[m]))

    return lengths

if __name__=="__main__":
    file_name = path.dirname(__file__) + "/01eil51.txt"
    instance = TSPInstance.TSPInstance(file_name)

    #观察搜索到的最优解及算法的收敛过程
    observe_convergence(instance, random_search, "inverse")
    observe_convergence(instance, threshold_accepting, "inverse")
    observe_convergence(instance, simulated_annealing, "inverse")

    #对指定的算法和邻域解生成方式进行性能测试
    # test_performance(instance, random_search, "swap", 5)

    #对指定的算法，比较不同的邻域结构对算法性能的影响
    types = ("swap", "insert", "inverse")
    compare_neighbor_type(instance, simulated_annealing, types, times= 5)

    #对指定的邻域结构，比较不同算法的性能
    methods = (random_search, threshold_accepting, simulated_annealing)
    compare_method_type(instance, methods, "inverse", times = 5)
    

    
    
        
