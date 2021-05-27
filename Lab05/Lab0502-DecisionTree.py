from os import path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

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


def about_train_test_split():
    '''
    X_train, X_test, y_train, y_test =sklearn.model_selection.train_test_split(train_data,train_target,
    test_size=0.4, random_state=0,stratify=y_train)
    参数说明：
    # train_data：所要划分的样本特征集
    # train_target：所要划分的样本结果
    # test_size：样本占比，如果是整数的话就是样本的数量
    # random_state：是随机数的种子。
    # stratify是为了保持split前类的分布。比如有100个数据，80个属于A类，20个属于B类。
    如果train_test_split(... test_size=0.25, stratify = y_all), 那么split之后数据如下： 
    training: 75个数据，其中60个属于A类，15个属于B类。 
    testing: 25个数据，其中20个属于A类，5个属于B类。 

    用了stratify参数，training集和testing集的类的比例是 A：B= 4：1，等同于split前的比例（80：20）。
    通常在这种类分布不平衡的情况下会用到stratify。
    将stratify=X就是按照X中的比例分配 
    将stratify=y就是按照y中的比例分配 
    '''
    # step 1 调用load_data方法读入数据
    X, y = load_data()
    a = {'ham': 0, 'spam': 1}
    y = [a[s] for s in y]

    # step 2 利用缺省参数调用train_test_split将数据划分为训练集和测试集(X_train, X_test, y_train, y_test)
    X_train_raw, X_test_raw, y_train, y_test=train_test_split(X,y)

    # step 3 创建TfidfVectorizer对象vectorizer, 用训练集的短信文本训练vectorizer，
    #        利用训练好的vectorizer抽取训练集的短信文本和测试集的短信文本的tfidf
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    print(vectorizer.get_feature_names())
    print("\n抽取训练集的短信文本", X_train)
    print("\n抽取测试集的短信文本", X_test)

    # step 4 创建DecisionTreeClassifier对象classifier, 用训练集训练classifier
    #        利用训练好的classifier对测试集进行预测
    #        输出前10个测试样本的预测值、实际值和短信文本
    #        输出分类准确率（利用accuracy_score函数）
    #        返回混淆矩阵（利用confusion_matrix函数）
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    for i, prediction in enumerate(y_pred[:10]):
        print('Prediction: %s. Real: %s. Message: %s' % (prediction, y_test[i], X_test_raw[i]))

    print('Accuracy:', accuracy_score(y_test, y_pred))
    
    return confusion_matrix(y_test, y_pred)



def show_confusion_matrix(c_mat):
    print(c_mat)
    plt.matshow(c_mat)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def about_cross_val_score():
    '''
    cross_val_score(estimator, X, y=None, groups=None,
                    scoring=None, cv=’warn’, n_jobs=None, verbose=0, fit_params=None,
                    pre_dispatch=‘2*n_jobs’, error_score=’raise-deprecating’)
    参数：
    estimator： 需要使用交叉验证的算法
    X： 输入样本数据
    y： 样本标签
    groups： 将数据集分割为训练/测试集时使用的样本的组标签（一般用不到）
    scoring： 交叉验证最重要的就是他的验证方式，选择不同的评价方法，会产生不同的评价结果。
    cv： 交叉验证折数或可迭代的次数
    n_jobs： 同时工作的cpu个数（-1代表全部）
    verbose： 详细程度
    fit_params： 传递给估计器（验证算法）的拟合方法的参数
    pre_dispatch： 控制并行执行期间调度的作业数量。
    error_score： 如果在估计器拟合中发生错误，要分配给该分数的值（一般不需要指定）
    '''
    X, y = load_data()
    a = {'ham':0,'spam':1}
    y = [ a[s] for s in y]  #将label转换为0和1
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    classifier = DecisionTreeClassifier()
    

    # step 1 利用5折交叉验证对classifier进行训练，输出accuracy
    scores = cross_val_score(classifier, X, y, cv = 5, scoring = 'accuracy')
    print('Accuracy:', np.mean(scores), scores)

    # step 2 利用5折交叉验证对classifier进行训练，输出precision
    precision = cross_val_score(classifier, X, y, cv=5, scoring='precision')
    print('Precision:', np.mean(precision), precision)

    # step 3 利用5折交叉验证对classifier进行训练，输出recall
    recall = cross_val_score(classifier, X, y, cv=5, scoring='recall')
    print('Recall:', np.mean(recall), recall)

    # step 4 利用5折交叉验证对classifier进行训练，输出f1
    f1 = cross_val_score(classifier, X, y, cv=5, scoring='f1')
    print('F1:', np.mean(f1), f1)

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

def about_P_R_curve():
    '''
    sklearn.metrics.precision_recall_curve(y_true, probas_pred, *, pos_label=None, sample_weight=None)
    Compute precision-recall pairs for different probability thresholds

    Note: this implementation is restricted to the binary classification task.
    The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
    The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
    The last precision and recall values are 1. and 0. respectively and do not have a corresponding threshold. This ensures that the graph starts on the y axis.

    Parameters
    y_true array, shape = [n_samples]
    True binary labels. If labels are not either {-1, 1} or {0, 1}, then pos_label should be explicitly given.

    probas_pred array, shape = [n_samples]
    Estimated probabilities or decision function.

    pos_label int or str, default=None
    The label of the positive class. When pos_label=None, if y_true is in {-1, 1} or {0, 1}, pos_label is set to 1, otherwise an error will be raised.

    sample_weight array-like of shape (n_samples,), default=None
    Sample weights.

    Returns
    precision array, shape = [n_thresholds + 1]
    Precision values such that element i is the precision of predictions with score >= thresholds[i] and the last element is 1.

    recall array, shape = [n_thresholds + 1]
    Decreasing recall values such that element i is the recall of predictions with score >= thresholds[i] and the last element is 0.

    thresholds array, shape = [n_thresholds <= len(np.unique(probas_pred))]
    Increasing thresholds on the decision function used to compute precision and recall.
    '''
    X, y = load_data()
    a = {'ham':0,'spam':1}
    y = [ a[s] for s in y]  #将label转换为0和1
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y)
    
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)

    # step 1 调用predict_proba方法得到测试集的预测概率
    y_pred = classifier.predict_proba(X_test)
    
    average_precision = average_precision_score(y_test, y_pred[:,1])

    # step 2 调用precision_recall_curve计算P-R曲线上各点的precision和recall
    precision,recall,thresholds=precision_recall_curve(y_test,y_pred[:,1])

    # step 3 调用matplotlib.pyplot的plot方法画P-R曲线
    plt.plot(recall, precision)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()

from sklearn.metrics import roc_curve, auc
def about_ROC_curve():
    '''
    sklearn.metrics.roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)

    主要参数：
    y_true：真实的样本标签，默认为{0，1}或者{-1，1}。如果要设置为其它值，则 pos_label 参数要设置为特定值。
            例如要令样本标签为{1，2}，其中2表示正样本，则pos_label=2
    y_score：对每个样本的预测结果。
    pos_label：正样本的标签

    返回值：
    fpr：False positive rate
    tpr：True positive rate (Recall)
    thresholds
    '''
    X, y = load_data()
    a = {'ham':0,'spam':1}
    y = [ a[s] for s in y]
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y)

    # step 1 创建TfidfVectorizer对象vectorizer, 用训练集的短信文本训练vectorizer，
    #        利用训练好的vectorizer抽取训练集的短信文本和测试集的短信文本的tfidf
    vectorizer=TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    print(vectorizer.get_feature_names())
    print("\n抽取训练集的短信文本",X_train)
    print("\n抽取测试集的短信文本", X_test)

    # step 2 创建DecisionTreeClassifier()对象classifier, 用训练集训练classifier
    #        利用训练好的classifier对测试集进行预测,得到预测概率值
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict_proba(X_test)


    # step 3 根据真实标签和预测概率值调用roc_curve计算false_positive_rate、recall和thresholds
    #        调用auc计算ROC曲线下方的面积大小roc_auc
    false_positive_rate,recall,thresholds=roc_curve(y_test,y_pred[:,1])
    roc_auc=auc(false_positive_rate,recall)

    # step 4 画ROC曲线
    plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' %roc_auc)

    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('Recall')
    plt.xlabel('Fall-out')
    plt.show()


if __name__=="__main__":
    c_m = about_train_test_split()
    show_confusion_matrix(c_m)
    about_cross_val_score()
    about_P_R_curve()
    about_ROC_curve()

