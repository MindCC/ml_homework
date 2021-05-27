from os import path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def load_data():
    '''
    SMSSpamCollection.txt数据集
    第一列是短信的label
    ham：非垃圾短信
    spam：垃圾短信
    \t键后面是短信的正文
    '''
    file_name = path.dirname(__file__) + "/SMSSpamCollection.txt"
    X, y = [], []
    spam_count = 0
    with open(file_name, 'r', encoding='UTF-8') as file:
        line = file.readline()
        while line:
            d = line.split("\t")
            X.append(d[1])  # 短信正文
            y.append(d[0])  # label
            if d[0] == 'spam':
                spam_count += 1
            line = file.readline()

    print('Total samples: {}, the number of spam: {}'.format(len(y), spam_count))

    return X, y


'''
class sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, *, weights='uniform', algorithm='auto',
                                             leaf_size=30, p=2, metric='minkowski', metric_params=None,
                                             n_jobs=None, **kwargs)
parameters:  n_neighbors：就是选取最近的点的个数k

             weights{‘uniform’, ‘distance’} or callable, default=’uniform’
                   weight function used in prediction. Possible values:
                 ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
                 ‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
                  [callable] : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights.

            algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
                 Algorithm used to compute the nearest neighbors:
                ‘ball_tree’ will use BallTree
                ‘kd_tree’ will use KDTree
                ‘brute’ will use a brute-force search.
                ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.
'''


# Part 1 观察KNeighborsClassifier在不同参数n_neighbors和weights下的性能

def aboutKNeighborsClassifier():
    '''
    '''
    # step 1 调用load_data方法读入数据
    X, y = load_data()

    # step 2 利用缺省参数调用train_test_split将数据划分为训练集和测试集 
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y)

    # step 3 创建TfidfVectorizer对象vectorizer, 用训练集的短信文本训练vectorizer，
    #        利用训练好的vectorizer抽取训练集的短信文本和测试集的短信文本的tfidf
    vectorizer = TfidfVectorizer()
    vectorizer.fit(X_train_raw)
    X_train = vectorizer.transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    # 请修改下面参数，观察参数对结果的影响----------------------------
    neighbor_num = 1
    weight_type = 'uniform'
    # -----------------------------------------------------------------

    # step 4 用指定的参数创建并训练KNeighborsClassifier对象classifier
    classifier = KNeighborsClassifier(n_neighbors=neighbor_num, weights=weight_type)
    classifier.fit(X_train, y_train)

    # step 5 预测测试集上的结果
    y_pred = classifier.predict(X_test)

    # step 6 输出结果
    print('neighbor number: %d, weight type: %s' % (neighbor_num, weight_type))
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred, pos_label="spam"))
    print('Recall:', recall_score(y_test, y_pred, pos_label="spam"))
    print('f1 score:', f1_score(y_test, y_pred, pos_label="spam"))


# Part 2 在weights='distance'，针对不同的性能指标，查找最佳的参数n_neighbors

def compareKNeighborsClassifier():
    '''
    '''
    # step 1 调用load_data方法读入数据
    X, y = load_data()

    # step 2 设置重复次数times, 比较的n_neighbors的范围
    times = 10
    neighbor_tuple = (1, 3, 5, 15, 30, 50)

    metrics = ('accuracy', 'precision', 'recall', 'f1 score')

    # results[m, n, t]存储第m个metric、第n个K数和第t+1次实验的结果
    # 比如results[0,1,0]存储accuracy、K=neighbor_tuple[1]=3和第1次实验的结果
    results = np.zeros((4, len(neighbor_tuple), times))
    for i in range(times):
        # 利用缺省参数调用train_test_split将数据划分为训练集和测试集 
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y)

        # 创建TfidfVectorizer对象vectorizer, 用训练集的短信文本训练vectorizer，
        # 利用训练好的vectorizer抽取训练集的短信文本和测试集的短信文本的tfidf
        vectorizer = TfidfVectorizer()
        vectorizer.fit(X_train_raw)
        X_train = vectorizer.transform(X_train_raw)
        X_test = vectorizer.transform(X_test_raw)
        weight_type = 'uniform'
        # 创建不同参数n_neighbors的KNeighborsClassifier，
        # 对分类器进行训练、预测（预测结果放如y_pred）、并记录分类性能
        for j in range(len(neighbor_tuple)):
            # add your code here ----------------------------
            # step 4 用指定的参数创建并训练KNeighborsClassifier对象classifier
            classifier = KNeighborsClassifier(n_neighbors=neighbor_tuple[j], weights=weight_type)
            classifier.fit(X_train, y_train)

            # step 5 预测测试集上的结果
            y_pred = classifier.predict(X_test)
            # end your code here ----------------------------
            results[0, j, i] = accuracy_score(y_test, y_pred)
            results[1, j, i] = precision_score(y_test, y_pred, pos_label="spam")
            results[2, j, i] = recall_score(y_test, y_pred, pos_label="spam")
            results[3, j, i] = f1_score(y_test, y_pred, pos_label="spam")

    for i in range(4):
        print('\n', metrics[i])
        for j in range(len(neighbor_tuple)):
            print('K is ', neighbor_tuple[j], ': ', np.mean(results[i, j]))
            # print(results[j])


'''
class sklearn.neighbors.RadiusNeighborsClassifier(radius=1.0, *, weights='uniform',
       algorithm='auto', leaf_size=30, p=2, metric='minkowski', outlier_label=None,
       metric_params=None, n_jobs=None, **kwargs)
'''

from sklearn.neighbors import RadiusNeighborsClassifier


# Part 3 观察RadiusNeighborsClassifier在不同参数radius和weights下的性能

def aboutRadiusNeighborsClassifier():
    '''
    '''
    # step 1 调用load_data方法读入数据
    X, y = load_data()

    # step 2 利用缺省参数调用train_test_split将数据划分为训练集和测试集
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y)

    # step 3 创建TfidfVectorizer对象vectorizer, 用训练集的短信文本训练vectorizer，
    #        利用训练好的vectorizer抽取训练集的短信文本和测试集的短信文本的tfidf
    vectorizer = TfidfVectorizer()
    vectorizer.fit(X_train_raw)
    X_train = vectorizer.transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    # 请修改下面参数，观察参数对结果的影响----------------------------
    r = 1.2
    weight_type = 'uniform'
    # ----

    # step 4 创建并训练RadiusNeighborsClassifier对象classifier

    classifier = RadiusNeighborsClassifier(radius=r, weights=weight_type)
    classifier.fit(X_train, y_train)

    # step 5 预测测试集上的结果
    y_pred = classifier.predict(X_test)

    # step 6 输出结果
    print('Radius: %.1f, weight type: %s' % (r, weight_type))
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred, pos_label="spam"))
    print('Recall:', recall_score(y_test, y_pred, pos_label="spam"))
    print('f1 score:', f1_score(y_test, y_pred, pos_label="spam"))


# Part 4 在weights='distance'，针对不同的性能指标，查找最佳的参数radius

def compareRadiusNeighborsClassifier():
    '''
    '''
    # step 1 调用load_data方法读入数据
    X, y = load_data()

    # step 2 设置重复次数times, 比较的radius的范围
    times = 10
    radius_tuple = (1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45)
    weight_type = 'distance'
    metrics = ('accuracy', 'precision', 'recall', 'f1 score')

    # results[m, n, t]存储第m个metric、第n个radius数和第t次实验的结果
    # 比如results[0,1,0]存储accuracy、radius=radius_tuple[1]=1.1和第1次实验的结果
    results = np.zeros((4, len(radius_tuple), times))
    for i in range(times):
        # 利用缺省参数调用train_test_split将数据划分为训练集和测试集 
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y)

        # 创建TfidfVectorizer对象vectorizer, 用训练集的短信文本训练vectorizer，
        # 利用训练好的vectorizer抽取训练集的短信文本和测试集的短信文本的tfidf
        vectorizer = TfidfVectorizer()
        vectorizer.fit(X_train_raw)
        X_train = vectorizer.transform(X_train_raw)
        X_test = vectorizer.transform(X_test_raw)

        # 创建不同参数radius的RadiusNeighborsClassifier
        # 对分类器进行训练、预测、并记录分类性能
        for j in range(len(radius_tuple)):
            # add your code here ----------------------------
            classifier = RadiusNeighborsClassifier(radius=radius_tuple[j], weights=weight_type)
            # 训练
            classifier.fit(X_train, y_train)
            # 预测测试集上的结果
            y_pred = classifier.predict(X_test)
            results[0, j, i] = accuracy_score(y_test, y_pred)
            results[1, j, i] = precision_score(y_test, y_pred, pos_label="spam")
            results[2, j, i] = recall_score(y_test, y_pred, pos_label="spam")
            results[3, j, i] = f1_score(y_test, y_pred, pos_label="spam")
            # end your code here ----------------------------

    for i in range(4):
        print('\n', metrics[i])
        for j in range(len(radius_tuple)):
            print('Radius is ', radius_tuple[j], ': ', np.mean(results[i, j]))
            # print(results[j])


if __name__ == "__main__":
    print('\nRunning aboutKNeighborsClassifier() ...')
    aboutKNeighborsClassifier()

    print('\nRunning compareKNeighborsClassifier() ...')
    compareKNeighborsClassifier()

    print('\nRunning aboutRadiusNeighborsClassifier() ...')
    aboutRadiusNeighborsClassifier()

    print('\nRunning compareRadiusNeighborsClassifier() ...')
    compareRadiusNeighborsClassifier()
