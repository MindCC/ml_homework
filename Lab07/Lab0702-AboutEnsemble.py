from os import path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

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
class sklearn.ensemble.AdaBoostClassifier(base_estimator=None, *, n_estimators=50,
                                          learning_rate=1.0, algorithm='SAMME.R', random_state=None)


class sklearn.ensemble.BaggingClassifier(base_estimator=None, n_estimators=10, *, max_samples=1.0,
                                         max_features=1.0, bootstrap=True, bootstrap_features=False,
                                         oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0)
'''

# Part 1 观察AdaBoostClassifier和BaggingClassifier在不同base_estimator的性能

def aboutAdaBoostAndBaggingClassifier():
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
    X_train = vectorizer.transform(X_train_raw).toarray()
    X_test = vectorizer.transform(X_test_raw).toarray()

    # 请选择基分类器----------------------------
    base_classifier = LogisticRegression(solver='lbfgs', class_weight = 'balanced')
##    base_classifier = DecisionTreeClassifier()
##    base_classifier = KNeighborsClassifier(n_neighbors = 15)
##    base_classifier = ComplementNB()
##    base_classifier = SVC(kernel = 'linear', class_weight = 'balanced', probability=True, gamma='scale')
    # -----------------------------------------------------------------
 
    # step 4 请选择集成分类器，并进行训练
    classifier = AdaBoostClassifier(base_estimator = base_classifier, n_estimators=50)
##    classifier = BaggingClassifier(base_estimator = base_classifier, n_estimators=5, )

    classifier.fit(X_train, y_train)

    # step 5 预测测试集上的结果
    y_pred = classifier.predict(X_test)

    # step 6 输出结果
    print('base classifier: ', base_classifier)
    print('Ensemble: ', classifier)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:',precision_score(y_test, y_pred, pos_label = "spam"))
    print('Recall:',   recall_score(y_test, y_pred, pos_label = "spam"))
    print('f1 score:', f1_score(y_test, y_pred, pos_label = "spam"))




'''
class sklearn.ensemble.RandomForestClassifier(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2,
                                              min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                              max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                              bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0,
                                              warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
parameters:
    criterion {“gini”, “entropy”}, default=”gini”
    class_weight {“balanced”, “balanced_subsample”}, dict or list of dicts, default=None
'''

# Part 2 利用RandomForestClassifier对SMSSpamCollection.txt数据集进行分类
#        采用train_test_split把数据划分为训练集和测试集
#        输出模型在测试集上的性能：Accuracy，Precision，Recall，f1 score

def aboutRandomForestClassifier():
    '''
    基本步骤：
        step 1 调用load_data方法读入数据
        step 2 利用缺省参数调用train_test_split将数据划分为训练集和测试集 
        step 3 创建TfidfVectorizer对象vectorizer, 用训练集的短信文本训练vectorizer，
               利用训练好的vectorizer抽取训练集的短信文本和测试集的短信文本的tfidf
        step 4 创建RandomForestClassifier对象并训练
        step 5 预测测试集上的结果
        step 6 输出结果
    '''
    # step 1
    X, y = load_data()
    # step 2
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y)
    # step 3
    vectorizer = TfidfVectorizer()
    vectorizer.fit(X_train_raw)
    X_train = vectorizer.transform(X_train_raw).toarray()
    X_test = vectorizer.transform(X_test_raw).toarray()
    # step 4
    rfclassifier = RandomForestClassifier()
    rfclassifier.fit(X_train,y_train)
    # step 5
    y_pred = rfclassifier.predict(X_test)
    # step 6
    print('base classifier: ', "RandomForeastClassifier")
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:',precision_score(y_test, y_pred, pos_label = "spam"))
    print('Recall:',   recall_score(y_test, y_pred, pos_label = "spam"))
    print('f1 score:', f1_score(y_test, y_pred, pos_label = "spam"))



if __name__=="__main__":
    print('\nRunning aboutAdaBoostAndBaggingClassifier ...')
    aboutAdaBoostAndBaggingClassifier()

    print('\nRunning aboutRandomForestClassifier ...')
    aboutRandomForestClassifier()


 

