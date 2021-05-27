from numpy import *
import numpy as np
from util import *
from pylab import *
import random


def kmeans(X, mu0, doPlot=True):
    '''
    X is an N*D matrix of N data points in D dimensions.
    mu is a K*D matrix of initial cluster centers, K is
    the desired number of clusters.
    this function should return a tuple (mu, z, obj) where mu is the
    final cluster centers, z is the assignment of data points to
    clusters, and obj[i] is the kmeans objective function:
      (1/N) sum_n || x_n - mu_{z_n} ||^2
    at iteration [i].
    mu[k,:] is the mean of cluster k
    z[n] is the assignment (number in 0...K-1) of data point n
    you should run at *most* 100 iterations, but may run fewer
    if the algorithm has converged
    '''

    mu = mu0.copy()  # for safety

    N, D = X.shape
    K = mu.shape[0]

    # initialize assignments and objective list
    z = zeros((N,), dtype=int)
    obj = []

    # run at most 100 iterations
    for it in range(100):
        print(it)
        # store the old value of z so we can check convergence
        z_old = z.copy()

        # recompute the assignment of points to centers（重新确定数据所在的簇）
        for n in range(N):
            # 计算每个点与中心点的距离
            distances = np.linalg.norm(X[n] - mu, axis=1)
            mindist = np.argmin(distances)
            z[n] = mindist

        # recompute means （重新计算各个簇的中心（均值））
        for k in range(K):
            mu[k, :] = mean(X[z == k, :], axis=0)

        # compute the objective
        currentObjective = 0
        for n in range(N):
            currentObjective = currentObjective + linalg.norm(X[n, :] - mu[z[n], :]) ** 2 / float(N)
        obj.append(currentObjective)

        print('Iteration %d, objective=%g' % (it, currentObjective))
        if doPlot:
            plotDatasetClusters(X, mu, z)
            show()

        # check to see if we've converged
        if all(z == z_old):
            break

    if doPlot and D == 2:
        plotDatasetClusters(X, mu, z)
        show()

    # return the required values
    return (mu, z, array(obj))


def initialize_clusters(X, K, method):
    '''
    X is N*D matrix of data
    K is desired number of clusters (>=1)
    method is one of:
      determ: initialize deterministically (for comparitive reasons)
      random: just initialize randomly
      ffh   : use furthest-first heuristic
    returns a matrix K*D of initial means.
    you may assume K <= N
    '''

    N, D = X.shape
    mu = zeros((K, D))

    if method == 'determ':
        # just use the first K points as centers
        mu = X[0:K, :].copy()  # be sure to copy otherwise bad things happen!!!

    elif method == 'random':
        # pick K random centers
        dataPoints = range(N)
        permute(dataPoints)
        mu = X[dataPoints[0:K], :].copy()  # ditto above

    elif method == 'ffh':
        # pick the first center randomly and each subsequent
        # subsequent center according to the furthest first
        # heuristic

        # pick the first center totally randomly
        mu[0, :] = X[int(rand() * N), :].copy()  # be sure to copy!

        # pick each subsequent center by ldh
        for k in range(1, K):
            # find m such that data point n is the best next mean, set
            # this to mu[k,:]
            # farthest from previously
            # 计算每个点与上个点的距离，选择最远的
            distances = np.linalg.norm(X[k] - mu[k - 1], axis=1)
            maxdist = np.argmax(distances)
            mu[k, :] = maxdist

    elif method == 'km++':
        # pick the first center randomly and each subsequent
        # subsequent center according to the kmeans++ method
        # HINT: see numpy.random.multinomial

        # pick the first center totally randomly
        mu[0, :] = X[int(rand() * N), :].copy()  # be sure to copy!
        d = [0 for _ in range(len(X))]
        for k in range(1, K):
            # print("k:",k)
            total = 0.0
            for i, point in enumerate(X):
                # print("i:",i)
                d[i] = get_closest_dist(point, mu)
                total += d[i]
            total *= random.random()
            for i, di in enumerate(d):
                total -= di
                if total > 0:
                    continue
                mu[k, :] = X[i]
                break

    else:
        print("Initialization method not implemented")
        sys.exit(1)

    return mu


def euler_distance(point1: list, point2: list) -> float:
    """
    计算两点之间的欧拉距离，支持多维
    """
    distance = 0.0
    for a, b in zip(point1, point2):
        distance += math.pow(a - b, 2)
    return math.sqrt(distance)


def get_closest_dist(point, centroids):
    min_dist = math.inf  # 初始设为无穷大
    for i, centroid in enumerate(centroids):
        dist = euler_distance(centroid, point)
        if dist < min_dist:
            min_dist = dist
    return min_dist
