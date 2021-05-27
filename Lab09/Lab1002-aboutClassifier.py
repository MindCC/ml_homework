'''
python中matplotlib.pyplot使用简介, see:
https://blog.csdn.net/feng98ren/article/details/79392747
'''
import os
import math
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics 
from sklearn.preprocessing import scale
from defectData import DefectData

from sklearn.feature_selection import f_classif, chi2

import numpy as np
import warnings
warnings.filterwarnings("ignore")



#Part 1 
def soft_defect_prediction(clf, metric = 'auc'):
    '''
    对指定分类器clf，测试在SDD问题集上的性能.
    SDD测试数据保存在'\SDData\'文件夹中，共有12个测试数据，本使用仅用其中2个

    参数意义：
    clf: 分类器
    metric: evaluation metric

    返回：r_lists,在每个测试集上每次运行的结果
         其中r_lists[i][j]是在第i+1个测试数据上第j+1次评价的结果
         file_list: 测试数据的文件名列表
    '''
    file_dir = os.path.abspath('.')+'\\SDData\\' ##测试数据集文件所在文件夹
    file_list = os.listdir(file_dir)
##    file_list = ["KC1.arff", "KC3.arff"] #测试数据集文件名列表
    times = 15  #在每个测试数据重复评估次数
    r_lists = []
    for file_name in file_list:
        defect_data = DefectData(file_dir + file_name)
        print(defect_data.get_relation(), end='\t')
        X = defect_data.get_data(f_classif, 50) #特征选择
        if not isinstance(clf, ComplementNB):
            X = scale(X)
        y = defect_data.get_target()
        
        r = []
        for i in range(times):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            if metric == 'accuracy':
                r.append( metrics.accuracy_score(y_test, y_pred))
            elif metric == 'precision':
                r.append( metrics.precision_score(y_test, y_pred))
            elif metric == 'recall':
                r.append(metrics.recall_score(y_test, y_pred))
            elif metric == 'f1':
                r.append( metrics.f1_score(y_test, y_pred))
            elif metric == 'auc':
                false_positive_rate, recall, thresholds = metrics.roc_curve(y_test,y_pred)
                roc_auc = metrics.auc(false_positive_rate, recall)
                r.append(roc_auc)
            else:
                print('Not support metric: ' + metric)
  
        print("Mean %s = %.5f"%(metric, np.mean(r)))     
        r_lists.append(r)
    print("Mean of all test instance: ", np.mean(r_lists))
    return r_lists, file_list #np.mean(r_list)



# Part 2
import matplotlib.pyplot as plt
def compare_classifiers(metric = 'auc'):
    logis = LogisticRegression(solver='lbfgs')
    dt = DecisionTreeClassifier()
    knn = KNeighborsClassifier(weights='distance')
    nb = ComplementNB()
    svm =NuSVC(nu = 0.01, gamma = 'scale')
    rf = RandomForestClassifier()
    
    clf_tuple = (logis, dt, knn, nb, svm, rf) #
    clf_name_tuple = ("LR", "DT", "KNN", "NB", "SVM", "RF")
    results = []
    fs = []
    # results[i] ith clf, results[][j] jth file, results[][][k] kth times
    for idx, clf in enumerate(clf_tuple):
        print(clf_name_tuple[idx])
        rs, fs = soft_defect_prediction(clf, metric)
        results.append(rs)

    print('\n', results)

    # 找到并输出在所有测试数据上平均性能最好的分类器及其性能
    mean_results = np.mean(results, 1)
    mean_results = np.mean(mean_results, 1)
    pos = np.argmax(mean_results)
    print("Best classifer: ", clf_name_tuple[pos])
    print("Best performance: ", mean_results[pos])

    # 柱状图显示位置
    x = [(0.2 + i*1.2) for i in range(len(results))]

    # 显示比较各分类器在所有数据上平均性能的柱状图
    plt.bar(x, mean_results)
    plt.title("mean " + metric)
    plt.xticks(x,clf_name_tuple)
    plt.show()

    # 每行4个图，显示每个数据文件上各分类器比较的柱状图
    num_col = 4
    num_row = math.ceil(len(results[0]) / num_col)
    for j in range(len(results[0])):
        plt.subplot(num_row, num_col, j+1)
        y=[np.nanmean(results[i][j],0) for i in range(len(results))]
        plt.bar(x, y)
        plt.title(fs[j])
        plt.xticks(x,clf_name_tuple)
    plt.show()

    # 将各分类器在各个测试数据上的性能用分组柱状图进行比较
    x = [(0.2 + i * 1.2) for i in range(len(results[0]))]
    x = np.array(x)
    width = 1 / len(results)
    fig, ax = plt.subplots()
    for i in range(len(results)):
        ax.bar(x + (i - len(results) / 2) * width, [np.nanmean(results[i][j],0) for j in range(len(results[i]))], width, label=clf_name_tuple[i])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(metric)
    ax.set_title(metric + ' by different classifiers')
    ax.set_xticks(x)
    ax.set_xticklabels(fs)
    ax.legend()
    plt.show()

    # 每行4个图，显示每个数据文件上各分类器比较的box图
    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.boxplot.html
    # https://blog.csdn.net/weixin_40683253/article/details/87857194
    fig, ax = plt.subplots()
    for i in range(len(results)):
        ax.boxplot([np.nanmean(results[i][j],0) for j in range(len(results[i]))],showmeans=True)
    plt.show()
        

if __name__ == '__main__':
    compare_classifiers('auc')
