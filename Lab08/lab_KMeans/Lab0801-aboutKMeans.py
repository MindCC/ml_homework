# Lab0801 Unsupervised Learning：K-means
'''
本实验探索k-means算法的实现及两种初始化簇中心的方法!

你需要修改的文件:

    clustering.py   Implementation of K-means (and variants)

clustering.py使用了下述文件中的数据或者方法:

    datasets.py     Some simple toy data sets
    digits          Digits data
    util.py         Utility functions, plotting, etc.
'''
import clustering
import datasets

## Part 1 修改clustering.py中的kmeans方法，实现K-Means Clustering 
'''
实现clustering.py中的kmeans方法
实现后你可以通过下面代码测试你实现的结果:

>>> mu0 = clustering.initialize_clusters(datasets.X2d, 2, 'determ')
>>> (mu,z,obj) = clustering.kmeans(datasets.X2d, mu0)
>>> mu
array([[ 2.31287961,  1.51333813],
       [-2.13455999, -2.15661017]])
>>> obj
array([ 1.91484251,  1.91484251])
'''
mu0 = clustering.initialize_clusters(datasets.X2d, 2, 'determ')
(mu, z, obj) = clustering.kmeans(datasets.X2d, mu0)
print(mu)
print(obj)

'''
或者：
>>> mu0 = clustering.initialize_clusters(datasets.X2d2, 4, 'determ')
>>> (mu,z,obj) = clustering.kmeans(datasets.X2d2, mu0)
Iteration 0, objective=5.84574
Iteration 1, objective=4.3797
Iteration 2, objective=3.06938
Iteration 3, objective=2.45218
Iteration 4, objective=2.34795
Iteration 5, objective=2.34795
>>> mu
array([[ 3.06150611, -1.07977065],
       [-3.92433223,  1.99052827],
       [ 0.87252863,  4.63384851],
       [-3.17087245, -4.10528255]])
'''

##mu0 = clustering.initialize_clusters(datasets.X2d2, 4, 'determ')
##(mu,z,obj) = clustering.kmeans(datasets.X2d2, mu0)
##print(mu)


##Part 2 修改clustering.py中的initialize_clusters方法，完成其中method == 'ffh'分支，
##       实现最远优先启发式（furthest first heuristic, ffh）算法初始化簇中心
'''
假设D(x)表示数据x到已有簇中最近簇中心的距离 ，ffh算法描述如下：  
1a. T从所有数据X中随机选择一个数据点作为第1个簇的中心c1.  
1b. 计算所有数据点的D(x)，从中选择具有最大值的数据点为新的簇中心ci.   
1c. 反复执行1b，直到已经选择了K个簇中心  
'''
from pylab import *
import util

##(X,Y) = datasets.loadDigits()
##mu0 = clustering.initialize_clusters(X, 10, 'ffh')
##(mu,z,obj) = clustering.kmeans(X, mu0, doPlot=False)
##plot(obj)
##show()
##util.drawDigits(mu, arange(10))
##show()

'''
**Question 1** Run kmeans with ffh.  How many iterations does it seem to
take for kmeans to converge using ffh?  Do the resulting cluster means
look like digits for most of these runs?  Pick the "best" run (i.e.,
the one with the lowest final objective) and plot the digits (include
the plot in the writeup).  How many of the digits 0-9 are represented?
Which ones are missing?  Try both with ffh and with random
initialization: how many iterations does it take for kmeans to
converge (on average) for each setting?
用ffh初始化簇中心运行kmeans。
当使用ffh，kmeans需要多少次迭代才能收敛？就多数而言，生成的簇中心是否看起来像数字？
将10次运行的最佳结果画出来，结果中包含0-9中的多少个数字？哪一些不见了？
对使用random和ffh初始化方法进行比较。
'''
##(X,Y) = datasets.loadDigits()
##bestObj = None
##bestZ = None
##bestMu = None
##for rep in range(10):
##    np.random.seed(1234 + rep)
##    mu0 = clustering.initialize_clusters(X, 10, 'ffh')
##    (mu,z,obj) = clustering.kmeans(X, mu0, doPlot=False)
##    if rep == 0 or obj[-1] < bestObj[-1]:
##        bestObj = obj
##        bestZ = z
##        bestMu = mu
##
##plot(bestObj)
##show()
##util.drawDigits(bestMu, arange(10))
##show()    

'''

**Question 2** Repeat Question 1, but for k=25.  Pick the best run, and
plot the digits.  Are you able to see all digits here?
'''
##(X,Y) = datasets.loadDigits()
##bestObj = None
##bestZ = None
##bestMu = None
##for rep in range(10):
##    np.random.seed(1234 + rep)
##    mu0 = clustering.initialize_clusters(X, 25, 'ffh')
##    (mu,z,obj) = clustering.kmeans(X, mu0, doPlot=False)
##    if rep == 0 or obj[-1] < bestObj[-1]:
##        bestObj = obj
##        bestZ = z
##        bestMu = mu
##
##plot(bestObj)
##show()
##util.drawDigits(bestMu, arange(25))
##show()    

##Part 3 修改clustering.py中的initialize_clusters方法，完成其中method == 'km++'分支，
##       实现kmeans++启发式算法初始化簇中心
'''
假设D(x)表示数据x到已有簇中最近簇中心的距离 ，kmeans++启发式算法描述如下：  
1a. T从所有数据X中随机选择一个数据点作为第1个簇的中心c1.  
1b. 计算所有数据点的D(x)，用轮盘法选出下一个聚类中心ci,选择概率为D(x)^2 /sum{D(x)^2 }.   
1c. 反复执行1b，直到已经选择了K个簇中心  
'''
(X,Y) = datasets.loadDigits()
mu0 = clustering.initialize_clusters(X, 25, 'km++')
(mu,z,obj) = clustering.kmeans(X, mu0, doPlot=False)
plot(obj)
show()
util.drawDigits(mu, arange(25))
show()
print("=============")
