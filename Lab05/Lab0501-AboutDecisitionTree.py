# TASK 1: UNDERSTANDING WHAT DECISION TREES ARE DOING

from data import *

X, Y, dictionary = loadTextDataBinary('data/sentiment.tr')
print(X.shape)
print(Y.shape)
print(dictionary[:10])

from sklearn.tree import DecisionTreeClassifier

dt1 = DecisionTreeClassifier(max_depth=1)
dt1.fit(X, Y)
showTree(dt1, dictionary)
print(np.mean(dt1.predict(X) == Y))

# Convince yourself whether or not it is useful to go from depth one
# to depth two on this data. How do you know?
# （在该数据集上，将决策树的max_depth从1提高到2是否有助于提高决策树性能？为什么？）
# 可以询问两个问题 准确率提高
dt2 = DecisionTreeClassifier(max_depth=2)
dt2.fit(X, Y)
showTree(dt2, dictionary)
print(np.mean(dt2.predict(X) == Y))
#   if   !bad and !worst then return POSITIVE
#   elif  !bad and  worst then return NEGATIVE
#   elif  bad and  !stupid then return NEGATIVE
#   elif  bad and  stupid then return NEGATIVE
#   elif  ...
dt3 = DecisionTreeClassifier(max_depth=3)
dt3.fit(X, Y)
showTree(dt3, dictionary)
print(np.mean(dt3.predict(X) == Y))

# TASK 2: UNDERFITTING AND OVERFITTING

Xde, Yde, _ = loadTextDataBinary('data/sentiment.de', dictionary)

Xte, Yte, _ = loadTextDataBinary('data/sentiment.te', dictionary)

import numpy as np

accs = np.zeros((20, 3))
for depth in range(20):
    dt_depth = DecisionTreeClassifier(max_depth=depth + 1)
    dt_depth.fit(X, Y)
    accs[depth, 0] = np.mean(dt_depth.predict(X) == Y)
    accs[depth, 1] = np.mean(dt_depth.predict(Xde) == Yde)
    accs[depth, 2] = np.mean(dt_depth.predict(Xte) == Yte)

import matplotlib.pyplot as plt

plt.title('DecisionTree')
plt.plot(accs)
plt.ylabel('pre')
plt.xlabel('depth')
plt.show()
