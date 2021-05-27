import math
from os import path
import numpy as np
import matplotlib.pyplot as plt

class TSPInstance:
    '''
    file format:
    city number
    best known tour length
    list of city position (index x y)

    best known tour (city index starts from 1) 
    '''

    def __init__(self, file_name):
        self.__file_name = file_name
        with open(file_name, mode='r') as file_handle:
            d = file_handle.readline()
            self.__city_number = int(d)
            d = file_handle.readline()
            self.__best_known_tour_length = int(d)
            self.__city_position = []
            for city in range(self.citynum):
                 d = file_handle.readline()
                 d = d.split()
                 self.__city_position.append((int(d[1]),int(d[2])))

            tour = []
            d = file_handle.readline()
            while d == "":    #skip empty lines
                d = file_handle.readline()
            for city in range(self.citynum):
                 d = file_handle.readline()
                 tour.append(int(d)-1)
            self.__best_tour = tuple(tour)

    @property
    def citynum(self):
        return self.__city_number

    @property
    def optimalval(self):
        return self.__best_known_tour_length

    @property
    def optimaltour(self):
        return self.__best_tour

    
    def __getitem__(self, n):
        return self.__city_position[n]

    def get_distance(self, n, m):
        '''
        返回城市n和城市m之间的距离
        '''
        c1 = self[n]
        c2 = self[m]
        dist = (c1[0]-c2[0])**2 + (c1[1]-c2[1])**2
        return int(math.sqrt(dist)+0.5)

    def evaluate(self, tour):
        '''
        返回访问路径tour的路径长度
        '''
        dist = []
        for i in range(len(tour)-1):
            dist.append(self.get_distance(tour[i],tour[i+1]))
        else:
            dist.append(self.get_distance(tour[-1],tour[0]))
        return sum(dist) 

    def plot_tour(self, solution):
        '''
        画出访问路径solution
        '''
        x = np.zeros(self.citynum+1)
        y = np.zeros(self.citynum+1)
        
        for i, city in enumerate(solution):
            x[i] = self[city][0]
            y[i] = self[city][1]

        city = solution[0]
        x[-1] = self[city][0]
        y[-1] = self[city][1]
        plt.plot(x, y, linewidth = 2)
        ##plt.scatter(x, y)
        
        font = {'family': 'serif',
            'color':  'black',
            'weight': 'normal',
            'size': 9,
            }
        for i, city in enumerate(solution):
            plt.text(x[i],y[i],str(int(city+1)), fontdict=font)
            
        plt.show()



if __name__=="__main__":
    file_name = path.dirname(__file__) + "/01eil51.txt"
    instance = TSPInstance(file_name)
    print(instance.citynum)
    print(instance.optimalval)
    print(instance.evaluate(instance.optimaltour))
    print(instance.optimaltour)
    print(instance[0])
    print(instance.get_distance(0,1))
    instance.plot_tour(instance.optimaltour)                       
