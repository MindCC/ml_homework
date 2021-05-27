from os import path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings("ignore")

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
    with open(file_name,'r', encoding='UTF-8') as file:
        line = file.readline()
        while line:
            d = line.split("\t")
            X.append(d[1])  #短信正文
            y.append(d[0])  #label
            line = file.readline()
        
##    print(X[:10])
##    print(y[:10])

    return X, y


'''
class sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0,
                      shrinking=True, probability=False, tol=0.001, cache_size=200,
                      class_weight=None, verbose=False, max_iter=-1,
                      decision_function_shape='ovr', break_ties=False, random_state=None)

    parmeters:
      kernel in {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
      class_weight dict or ‘balanced’, default=None
      
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

class sklearn.svm.NuSVC(*, nu=0.5, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True,
                        probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False,
                        max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
                        
class sklearn.svm.LinearSVC(penalty='l2', loss='squared_hinge', *, dual=True, tol=0.0001,
                           C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1,
                           class_weight=None, verbose=0, random_state=None, max_iter=1000)

'''

# Part 1 观察SVMs在不同参数下的性能

def aboutSVMs():
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

    # 请修改下面参数，观察参数对结果的影响,哪个（哪些）参数影响大？----------------------------
    nu_value = 0.02
    kernel_type = 'rbf' #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
    degree_num = 1
    class_weight_type = 'balanced' #None  'balanced'
    # -----------------------------------------------------------------

    # 使用不同的SVM分类器，观察性能
##    classifier = LinearSVC()
##    classifier = SVC(kernel = kernel_type, degree = degree_num, class_weight = class_weight_type, gamma='scale')
    classifier = NuSVC(nu = nu_value, kernel = kernel_type, degree = degree_num, class_weight = class_weight_type, gamma='scale')
    # -----------------------------------------------
    
    # step 4 训练
    classifier.fit(X_train, y_train)

    # step 5 预测测试集上的结果
    y_pred = classifier.predict(X_test)

    # step 6 输出结果
    print('kernel: %s, degree: %d, class weight type: %s' % (kernel_type, degree_num, class_weight_type))
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:',precision_score(y_test, y_pred, pos_label = "spam"))
    print('Recall:',   recall_score(y_test, y_pred, pos_label = "spam"))
    print('f1 score:', f1_score(y_test, y_pred, pos_label = "spam"))



# Part 2 为‘linear’、‘rbf’和‘sigmoid’寻找合适的nu参数

def searchForNu(kernel_type):
    '''
    '''
    # step 1 调用load_data方法读入数据
    X, y = load_data()

    # step 2 设置重复次数times, 比较的n_neighbors的范围
    times = 25
    nu_values = (0.001, 0.010, 0.03, 0.04, 0.05, 0.1, 0.2)
    metrics = ('accuracy', 'precision', 'recall', 'f1 score')

    # results[m, n, t]存储第m个metric、第n个nu数和第t+1次实验的结果
    # 比如results[0,1,0]存储accuracy、K=neighbor_tuple[1]=3和第1次实验的结果
    results = np.zeros((4,len(nu_values),times))
    for i in range(times):
        # 利用缺省参数调用train_test_split将数据划分为训练集和测试集 
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y)

        # 创建TfidfVectorizer对象vectorizer, 用训练集的短信文本训练vectorizer，
        # 利用训练好的vectorizer抽取训练集的短信文本和测试集的短信文本的tfidf
        vectorizer = TfidfVectorizer()
        vectorizer.fit(X_train_raw)
        X_train = vectorizer.transform(X_train_raw)
        X_test = vectorizer.transform(X_test_raw)

        # 创建不同参数n_neighbors的KNeighborsClassifier，并记录分类性能
        for j in range(len(nu_values)):
            classifier = NuSVC(nu = nu_values[j], kernel = kernel_type, degree = 1, class_weight='balanced', gamma='scale')
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            results[0,j,i] = accuracy_score(y_test, y_pred)
            results[1,j,i] = precision_score(y_test, y_pred, pos_label = "spam")
            results[2,j,i] = recall_score(y_test, y_pred, pos_label = "spam")
            results[3,j,i] = f1_score(y_test, y_pred, pos_label = "spam")

    print('\n', kernel_type)       
    for i in range(4):
        print('\n', metrics[i])
        for j in range(len(nu_values)):
            print('nu is ', nu_values[j], ': ', np.mean(results[i,j]))
            #print(results[j])    


# Part 3 针对性能指标f1，利用GridSearchCV查找最佳的参数组合kernel和nu
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform
from sklearn.metrics import classification_report

def aboutGridSearchCV():
    '''
    class sklearn.model_selection.GridSearchCV(estimator, param_grid, *, scoring=None,
        n_jobs=None, iid='deprecated', refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs',
        error_score=nan, return_train_score=False)

    示例见：https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV

    可用scoring参数见https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    '''
    # step 1 调用load_data方法读入数据
    X, y = load_data()
    a = {'ham':0,'spam':1}
    y = [ a[s] for s in y]  #将label转换为0和1

    # step 2 创建TfidfVectorizer对象vectorizer进行特征提取
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y)
    vectorizer = TfidfVectorizer()
    vectorizer.fit(X_train_raw)
    X_train = vectorizer.transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    # step 3 创建NuSVC分类器对象

    nusvc = NuSVC()

    # step 4 构造参数字典，其中nu的取值可以参考上题，kernel为'linear'、'rbf'或者'sigmoid'
    parameters = {'kernel': ('linear','rbf','sigmoid'), 'nu': (0.001, 0.010, 0.03, 0.04, 0.05, 0.1, 0.2)}
    
    # step 5 构造GridSearchCV对象
    rsCV = GridSearchCV(nusvc,parameters)

    # step 6 调用fit方法搜索最佳参数组合
    rsCV.fit(X_train,y_train)

    # step 7 输出结果
    print('Best score: %0.3f' % rsCV.best_score_)
    print('Best parameters set:')
    best_parameters = rsCV.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name]))

    # step 8, 进行测试并输出结果
    results = np.zeros(4)
    y_pred = rsCV.predict(X_test)
    results[0] = accuracy_score(y_test, y_pred)
    results[1] = precision_score(y_test, y_pred)
    results[2] = recall_score(y_test, y_pred)
    results[3] = f1_score(y_test, y_pred)

    print("accuracy_score",results[0])
    print("precision_score", results[1])
    print("recall_score", results[2])
    print("f1_score", results[3])
    

# Part 4 针对性能指标f1，利用RandomizedSearchCV查找最佳的参数组合kernel和nu
from sklearn.model_selection import RandomizedSearchCV

def aboutRandomizedSearchCV():
    '''
   class sklearn.model_selection.RandomizedSearchCV(estimator, param_distributions, *,
       n_iter=10, scoring=None, n_jobs=None, iid='deprecated', refit=True, cv=None,
       verbose=0, pre_dispatch='2*n_jobs', random_state=None, error_score=nan,
       return_train_score=False)

    示例见：https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV

    可用scoring参数见https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    '''
    # step 1 调用load_data方法读入数据
    X, y = load_data()
    a = {'ham':0,'spam':1}
    y = [ a[s] for s in y]  #将label转换为0和1

    # step 2 创建TfidfVectorizer对象vectorizer进行特征提取
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y)
    vectorizer = TfidfVectorizer()
    vectorizer.fit(X_train_raw)
    X_train = vectorizer.transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    # step 3 创建NuSVC分类器对象
    nusvc = NuSVC()
    # step 4 构造参数分布字典，其中nu是均匀分布（0.001到0.2），kernel为'linear'、'rbf'或者'sigmoid'
    distributions = {'kernel': ('linear','rbf','sigmoid'), 'nu': [0.001, 0.2]}
    
    # step 5 构造RandomizedSearchCV对象，可以比较不同的n_iter参数
    rsCV=RandomizedSearchCV(nusvc,distributions, random_state=0)

    # step 6 调用fit方法搜索最佳参数组合
    rsCV.fit(X_train,y_train)

    # step 7 输出结果
    print('Best score: %0.3f' % rsCV.best_score_)
    print('Best parameters set:')
    best_parameters = rsCV.best_estimator_.get_params()
    for param_name in sorted(distributions.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name]))

    # step 8, 进行测试并输出结果
    results = np.zeros(4)
    y_pred = rsCV.predict(X_test)
    results[0] = accuracy_score(y_test, y_pred)
    results[1] = precision_score(y_test, y_pred)
    results[2] = recall_score(y_test, y_pred)
    results[3] = f1_score(y_test, y_pred)

    print("accuracy_score", results[0])
    print("precision_score", results[1])
    print("recall_score", results[2])
    print("f1_score", results[3])


    

if __name__=="__main__":
    print('\nRunning aboutSVMs ...')
    aboutSVMs()

    print('\nRunning searchForNu ...')
    searchForNu('poly') #linear, rbf, or sigmoid

    print('\nRunning aboutGridSearchCV ...')
    aboutGridSearchCV()

    print('\nRunning aboutRandomizedSearchCV ...')
    aboutRandomizedSearchCV()


 

