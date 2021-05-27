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
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_selection import f_classif, chi2

import numpy as np



#Part 1 
def soft_defect_prediction(clf, metric = 'auc'):
    '''
    对指定分类器clf，测试在SDD问题集上的性能.
    SDD测试数据保存在'\SDData\'文件夹中，共有12个测试数据，本使用仅用其中2个

    参数意义：
    clf: 分类器
    metric: evaluation metric

    返回：
    r_list,在每个测试集上多次运行的平均结果
         其中r_list[i]是在第i+1个测试数据上多次运行的平均结果
    file_list: 测试数据的文件名列表
    '''
    file_dir = os.path.abspath('.')+'\\SDData\\' ##测试数据集文件所在文件夹
    file_list = os.listdir(file_dir)
##    file_list = ["KC1.arff", "KC3.arff"] #测试数据集文件名列表
    times = 15  #在每个测试数据重复评估次数
    r_list = []
    for file_name in file_list:
        defect_data = DefectData(file_dir + file_name)
        print(defect_data.get_relation(), end='\t')
        X = defect_data.get_data(f_classif, 50) #特征选择
        # ComplementNB表示朴素贝叶斯
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
        r_list.append(np.mean(r))
    print("Mean of all test instance: ", np.mean(r_list))
    return r_list, file_list #np.mean(r_list)

# Part 2 选择1、2个分类器，为他们找到合适的参数，比如KNN的n_neighbors和weights, NuSVC的nu，及MPL的隐层参数
from sklearn.model_selection import GridSearchCV
def observe_parameters(metric = 'auc'):
    file_dir = os.path.abspath('.') + '\\SDData\\'  ##测试数据集文件所在文件夹
    file_list = os.listdir(file_dir)
    ##    file_list = ["KC1.arff", "KC3.arff"] #测试数据集文件名列表
    times = 15  # 在每个测试数据重复评估次数
    nu_values = (0.001, 0.010, 0.03, 0.04, 0.05, 0.1, 0.2)
    r_list = []
    for file_name in file_list:
        defect_data = DefectData(file_dir + file_name)
        print(defect_data.get_relation(), end='\t')
        X = defect_data.get_data(f_classif, 50)  # 特征选择

        # X = scale(X)
        y = defect_data.get_target()
        # 创建nuSVC对象
        nusvc = NuSVC()
        # 构造参数字典
        parameters = {'nu': (0.001, 0.010, 0.03, 0.04, 0.05, 0.1, 0.2)}


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        rsCV = GridSearchCV(nusvc, parameters)
        # 调用fit方法搜索最佳参数
        rsCV.fit(X_train, y_train)
        # 输出结果
        print('filename: %s' % file_name)
        print('Best score: %0.3f' % rsCV.best_score_)
        print('Best parameters:')
        best_parameters = rsCV.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print('\t%s: %r' % (param_name, best_parameters[param_name]))



# Part 3 比较各种分类器的性能，利用Part 2选择的参数构造分类器对象
import matplotlib.pyplot as plt
def compare_classifiers(metric = 'auc'):
    # step 1 分别构造LogisticRegression等7种分类器对象
    logis = LogisticRegression(solver='lbfgs')
    dt = DecisionTreeClassifier()
    knn = KNeighborsClassifier(weights='distance')
    nb = ComplementNB()
    svm =NuSVC(nu = 0.01, gamma = 'scale')
    rf = RandomForestClassifier()

    # step 2 用step 1构造的7个分类器生成元组clf_tuple
    clf_tuple = (logis, dt, knn, nb, svm, rf) #
    
    clf_name_tuple = ("LR", "DT", "KNN", "NB", "SVM", "RF")
    results = [] # results[i] ith clf, results[][j] mean performance of jth file
    fs = [] # data file name list
    for idx, clf in enumerate(clf_tuple):
        print(clf_name_tuple[idx])
        # step 3 调用test_soft_defect_prediction, 将结果返回给变量rs和fs
        rs, fs = soft_defect_prediction(clf, metric)
        # step 4 将rs追加到results中
        results.append(rs)

    print('\n', results)

    # 找到并输出在所有测试数据上平均性能最好的分类器及其性能
    mean_results = np.mean(results, 1)
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
        y = [results[i][j] for i in range(len(results))]
        plt.bar(x, y)
        plt.title(fs[j])
        plt.xticks(x,clf_name_tuple)
    plt.show()

    # 将各分类器在各个测试数据上的性能用分组柱状图进行比较
    x = [(0.2 + i*1.2) for i in range(len(results[0]))]
    x = np.array(x)
    width = 1 / len(results)
    fig, ax = plt.subplots()
    for i in range(len(results)):
        ax.bar(x + (i-len(results)/2) * width, results[i], width, label=clf_name_tuple[i])
 
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(metric)
    ax.set_title(metric + ' by different classifiers')
    ax.set_xticks(x)
    ax.set_xticklabels(fs)
    ax.legend()
    plt.show()
        

if __name__ == '__main__':
    # logis = LogisticRegression(solver='lbfgs')
    # r, fs = soft_defect_prediction(logis, 'auc')
    # print( r )

    # observe_parameters('auc')

    compare_classifiers('auc')
