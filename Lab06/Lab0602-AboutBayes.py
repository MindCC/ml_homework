from os import path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


'''

naive_bayes.BernoulliNB(*[, alpha, …])  Naive Bayes classifier for multivariate Bernoulli models.

naive_bayes.CategoricalNB(*[, alpha, …])  Naive Bayes classifier for categorical features

naive_bayes.ComplementNB(*[, alpha, …])   The Complement Naive Bayes classifier described in Rennie et al.

naive_bayes.GaussianNB(*[, priors, …])   Gaussian Naive Bayes (GaussianNB)

naive_bayes.MultinomialNB(*[, alpha, …])  Naive Bayes classifier for multinomial models

'''

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
    with open(file_name,'r', encoding='UTF-8') as file:
        line = file.readline()
        while line:
            d = line.split("\t")
            X.append(d[1])  #短信正文
            y.append(d[0])  #label
            if d[0] == 'spam':
                spam_count += 1
            line = file.readline()

    print('Total samples: {}, the number of spam: {}'.format(len(y), spam_count)) 

    return X, y


'''
class sklearn.naive_bayes.GaussianNB(*, priors=None, var_smoothing=1e-09)
'''

# Part 1 观察GaussianNB的性能

def aboutGaussianNB():
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

    # step 4 创建并训练GaussianNB对象classifier, GaussianNB对象的训练需要dense data
    classifier = GaussianNB()
    classifier.fit(X_train.toarray(), y_train)

    # step 5 预测测试集上的结果
    y_pred = classifier.predict(X_test.toarray())

    # step 6 输出结果
    print('GaussianNB')
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:',precision_score(y_test, y_pred, pos_label = "spam"))
    print('Recall:',   recall_score(y_test, y_pred, pos_label = "spam"))
    print('f1 score:', f1_score(y_test, y_pred, pos_label = "spam"))


# Part 2 比较GaussianNB、BernoulliNB、MultinomialNB和ComplementNB
#        （比较MultinomialNB和ComplementNB的性能，为什么又这样的结果？）


def compareNaiveBayesClassifier():
    '''
    '''
    # step 1 调用load_data方法读入数据
    X, y = load_data()

    # step 2 设置重复次数times
    times = 10
    metrics = ('accuracy', 'precision', 'recall', 'f1 score')
     
    nb_classifiers = (GaussianNB(), BernoulliNB(), MultinomialNB(), ComplementNB())

    # results[m, n, t]存储第m个metric、第n个分类算法和第t次实验的结果
    results = np.zeros((4,len(nb_classifiers),times))
    '''
    for i in range(times):
        # 利用缺省参数调用train_test_split将数据划分为训练集和测试集 
        # 创建TfidfVectorizer对象vectorizer, 用训练集的短信文本训练vectorizer，
        # 利用训练好的vectorizer抽取训练集的短信文本和测试集的短信文本的tfidf
        # 使用不同的Naive Bayes分类器，对分类器进行训练、预测、并记录分类性能
        for j in range(len(nb_classifiers)):
            ...
    '''
    #start your code here----------------------------------
    for i in range(times):
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y)

        vectorizer = TfidfVectorizer()
        vectorizer.fit(X_train_raw)
        X_train = vectorizer.transform(X_train_raw)
        X_test = vectorizer.transform(X_test_raw)
        for j in range(len(nb_classifiers)):
            classifier = nb_classifiers[j]
            classifier.fit(X_train.toarray(), y_train)
            y_pred = classifier.predict(X_test.toarray())

            results[0, j, i] = accuracy_score(y_test, y_pred)
            results[1, j, i] = precision_score(y_test, y_pred, pos_label="spam")
            results[2, j, i] = recall_score(y_test, y_pred, pos_label="spam")
            results[3, j, i] = f1_score(y_test, y_pred, pos_label="spam")
    #end your code here-------------------------------------
            
    for i in range(4):
        print('\n', metrics[i])
        for j in range(len(nb_classifiers)):
            print('分类器 ', nb_classifiers[j], ': ', np.mean(results[i,j]))
            #print(results[j])
    

if __name__=="__main__":
    aboutGaussianNB()
    compareNaiveBayesClassifier()


