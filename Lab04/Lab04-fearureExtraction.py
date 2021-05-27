#-*- coding:utf-8 -*-

# Part 1， about class DictVectorizer
from sklearn.feature_extraction import DictVectorizer

'''
The class DictVectorizer can be used to convert feature arrays represented as
lists of standard Python dict objects to the NumPy/SciPy representation used by scikit-learn estimators.

While not particularly fast to process, Python’s dict has the advantages of being convenient to use,
being sparse (absent features need not be stored) and storing feature names in addition to values.

DictVectorizer implements what is called one-of-K or “one-hot” coding for categorical (aka nominal, discrete) features.
Categorical features are “attribute-value” pairs where the value is restricted to a list of discrete of possibilities
without ordering (e.g. topic identifiers, types of objects, tags, names…).

In the following, “city” is a categorical attribute while “temperature” is a traditional numerical feature:

>>>
>>> measurements = [
...     {'city': 'Dubai', 'temperature': 33.},
...     {'city': 'London', 'temperature': 12.},
...     {'city': 'San Francisco', 'temperature': 18.},
... ]

>>> from sklearn.feature_extraction import DictVectorizer
>>> vec = DictVectorizer()

>>> vec.fit_transform(measurements).toarray()
array([[ 1.,  0.,  0., 33.],
       [ 0.,  1.,  0., 12.],
       [ 0.,  0.,  1., 18.]])

>>> vec.get_feature_names()
['city=Dubai', 'city=London', 'city=San Francisco', 'temperature']
'''

def about_DictVectorizer():
    '''
    利用DictVectorizer对数据data进行特征提取
    1. 创建DictVectorizer对象vec
    2. 用data训练vec (fit方法)
    3. 输出所有特征的名称 (get_feature_names方法)
    4. 输出vec对data转换的结果(transform方法)，理解为什么得到这样的结果

    '''
    data = [ {'city':'福州', 'income':12},
             {'city':'厦门', 'income':15},
             {'city':'泉州', 'income':10},]

    vec = DictVectorizer()
    vec.fit(data)
    print("\ndata_feature_name:",vec.get_feature_names())
    print("\ndata_transform",vec.transform(data).toarray())

    '''
    5. 输出vec对new_data转换的结果,理解为什么得到这样的结果
    '''
    new_data = [ {'city':'福州', 'income':13},
             {'city':'厦门', 'income':16},
             {'city':'泉州', 'income':11},
                 {'city':'宁德', 'income':9},]
    print("\nnew_data_transform",vec.transform(new_data).toarray())


from sklearn.feature_extraction.text import CountVectorizer
'''
class sklearn.feature_extraction.text.CountVectorizer(*, input='content', encoding='utf-8',
decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None,
stop_words=None, token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1), analyzer='word', max_df=1.0,
min_df=1, max_features=None, vocabulary=None, binary=False, dtype=<class 'numpy.int64'>)

https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

'''
def about_CountVectorizer():
    '''
    对data，比较analyzer='char'和analyzer='word'输出结果的不同
    在analyzer='char'的情况下，比较ngram_range=(1, 1)，ngram_range=(2, 2)和ngram_range=(1, 2)的结果
    如何让CountVectorizer在analyzer='word'的情况下正常工作？手工断词或者使用jieba等库

    '''
    data = ["我爱北京，我爱北京天安门", "天安门上太阳升"]
    #data = ["我 爱 北京 天安门", "天安门 上 太阳 升"]
    # 1、比较analyzer='char'和analyzer='word'输出结果的不同
    analyzer_tuple = ('word', 'char')
    for a in analyzer_tuple:
        print('\nanalyzer=', a)
        transfer = CountVectorizer(analyzer = a)   
        # 2、调用fit和transform，输出特征名称和数据转换结果
        transfer.fit(data)
        print("\ndata_feature_name",transfer.get_feature_names())
        print("\ndata_transform",transfer.transform(data).toarray())
        print("====================================")

    # 3、比较在analyzer='char'的情况下,ngram_range=(1, 1)，ngram_range=(2, 2)和ngram_range=(1, 2)
    ngram_tuple = ((1, 1), (2, 2), (1, 2))
    for n in ngram_tuple:
        transfer=CountVectorizer(analyzer='char',ngram_range=n)
        transfer.fit(data)
        print('\nngram_range=', n)
        print("\ndata_feature_name",transfer.get_feature_names())
        print("\ndata_transform",transfer.transform(data).toarray())
        print("====================================")



from sklearn.feature_extraction.text import TfidfVectorizer

'''
class sklearn.feature_extraction.text.TfidfVectorizer(*, input='content', encoding='utf-8',
decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None,
analyzer='word', stop_words=None, token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1),
max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=<class 'numpy.float64'>,
norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)

https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer
'''
def about_TfidfVectorizer():
    '''
    在analyzer='char'的情况下，比较ngram_range=(1, 1)，ngram_range=(2, 2)和ngram_range=(1, 2)的结果

    '''
    data = ["我爱北京天安门", "天安门上太阳升"]
    ngram_tuple = ((1, 1), (2, 2), (1, 2))
    print("Tfid")
    for ngram in ngram_tuple:
        tfid = TfidfVectorizer(analyzer='char',ngram_range=ngram)
        tfid.fit(data)
        print('\nngram_range=', ngram)
        print("\ndata_feature_name", tfid.get_feature_names())
        print("\ndata_transform", tfid.transform(data).toarray())
        print("====================================")


'''
Standardization of datasets is a common requirement for many machine learning estimators implemented in scikit-learn;
they might behave badly if the individual features do not more or less look like standard normally distributed data:
Gaussian with zero mean and unit variance.

An alternative standardization is scaling features to lie between a given minimum and maximum value,
often between zero and one, or so that the maximum absolute value of each feature is scaled to unit size.
This can be achieved using MinMaxScaler or MaxAbsScaler, respectively.

Normalization is the process of scaling individual samples to have unit norm.
This process can be useful if you plan to use a quadratic form such as the dot-product
or any other kernel to quantify the similarity of any pair of samples.
'''
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
import numpy as np

def about_preprocessing():
    '''
    对数据X_train，分别用StandardScaler、MinMaxScaler和 Normalizer对它进行预处理，
    并输出处理的结果
    '''
    X_train = np.array([[ 1., -1.,  2.],
                        [ 2.,  0.,  0.],
                     [ 0.,  1., -1.]])
    print(X_train)
    scaler_list = (StandardScaler(), MinMaxScaler(), Normalizer())
    for scaler in scaler_list:
        print(scaler)
        scaler.fit(X_train)
        print(scaler.transform(X_train))

    

if __name__=="__main__":
    about_DictVectorizer()
    
    about_CountVectorizer()

    about_TfidfVectorizer()

    about_preprocessing()
