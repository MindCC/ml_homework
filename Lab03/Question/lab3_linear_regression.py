from os import path
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split


## Part 1
## About PolynomialFeatures
def about_PolynomialFeatures():
    '''
    使用PolynomialFeatures 这个类可以进行特征的构造，构造的方式就是特征与特征相乘（自己与自己，自己与其他人），这种方式叫做使用多项式的方式。
    例如：有 a、b 两个特征，那么它的 2 次多项式的次数为 [1,a,b,a^2,ab,b^2]。

    PolynomialFeatures 这个类有 3 个参数：
    degree：控制多项式的次数；
    interaction_only：默认为 False，如果指定为 True，那么就不会有特征自己和自己结合的项，组合的特征中没有 a^2 和 b^2；
    include_bias：默认为 True 。如果为 True 的话，那么结果中就会有 0 次幂项，即全为 1 这一列。

    对数据：
    [[0 1]
     [2 3]
     [4 5]]
    输出PolynomialFeatures所有参数组合下的转换结果，比如：
    Degree = 2, interaction_only = True, include_bias = True
    [[ 1.  0.  1.  0.]
    [ 1.  2.  3.  6.]
    [ 1.  4.  5. 20.]]
    ...
   
    '''

    X = np.arange(6).reshape(3, 2)
    print(X)
    degree_list = [2, 3]
    interaction_only_list = [True, False]
    include_bias_list = [True, False]
    # start your code here
    '''
    for each deg in degree_list
        for each inter in interaction_only_list
           for each incl in include_bias_list
              print the three parameters (deg, inter, and incl)
              use deg, inter, and incl to create PolynomialFeatures object
              use the created object to fit and transform data X
              print the transofrmed result
    '''
    for i, degree in enumerate(degree_list):
        for i, interaction_only in enumerate(interaction_only_list):
            for i, include_bias in enumerate(include_bias_list):
                print("Degree = {}, interaction_only = {}, include_bias = {}".format(degree, interaction_only, include_bias))
                poly = PolynomialFeatures(degree, interaction_only, include_bias)
                new_X = poly.fit_transform(X)
                print(new_X)

    # end your code here


# residual sum of squares
def error(y, y_pred):
    return sum((y_pred - y) ** 2)


'''
对Web应用来说，什么时候需要增加部署资源是个决策问题，
如果增加不及时，影响用户体验，如果太早增加，则浪费资金。
假设目前资源能够应对的服务是每小时100,000 个请求，我们需要预测什么时候应该购买新的资源。

假设最近1个月的数据保存在文件web_traffic.tsv中（(tsv because it contains tab separated values).
每行数据表示时间和点击数，如果数据不存在，表示为nan.
'''


def load_data():
    file_name = path.dirname(__file__) + "/web_traffic.tsv"
    data = np.genfromtxt(file_name, delimiter="\t")
    print(data)
    print(data.shape)

    # Preprocessing and cleaning the data
    x = data[:, 0]
    y = data[:, 1]
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]

    return x, y


def show_data(x, y, is_show):
    # Visualize the data
    plt.scatter(x, y)
    plt.title("Web traffic over the last month")
    plt.xlabel("Time")
    plt.ylabel("Hits/hour")
    plt.xticks([w * 7 * 24 for w in range(10)], \
               ['week %i' % w for w in range(10)])
    plt.autoscale(tight=True)
    plt.grid()
    if is_show:
        plt.show()


## Part 2
## About Simple LinearRegression
def simple_linear_regression(X, y):
    '''
    使用sklearn的LinearRegression进行简单线性回归
    输出residual sum of squares，R-squared及模型的系数
    画出拟合的曲线
    '''
    if X.ndim == 1:
        x = X.reshape(-1, 1)
    else:
        x = X
    show_data(x, y, False)
    # start your code here
    '''
    Create the LinearRegression object model
    Use the data to fit the model
    Use the learned model to predict the y_pred of x
    '''
    model = LinearRegression()
    # 训练模型
    model.fit(x, y)
    y_pred = model.predict(x)
    # end your code here
    print(error(y, y_pred), model.score(x, y), model.intercept_, model.coef_)
    plt.plot(x, y_pred, color='blue', linewidth=3)
    plt.show()


def compare_LinearRegression(X, y):
    '''
    使用sklearn的LinearRegression进行1、2和3阶多项式回归
    Use PolynomialFeatures to generate polynomial and interaction features
    输出residual sum of squares，R-squared及模型的系数
    画出拟合的曲线
    '''
    if X.ndim == 1:
        x = X.reshape(-1, 1)
    else:
        x = X
    show_data(x, y, False)
    colors = ['blue', 'green', 'red']
    orders = [1, 2, 3]
    for index in range(len(orders)):
        # start your code here
        model = LinearRegression()
        poly = PolynomialFeatures(orders[index])
        X = poly.fit_transform(x)
        model.fit(X, y)
        y_pred = model.predict(X)
        # end your code here
        print("Degree={},".format(orders[index]), error(y, y_pred), model.score(X, y), model.intercept_, model.coef_)
        plt.plot(x, y_pred, color=colors[index], linewidth=3)
    plt.show()


def polynomial_regression_in_scipy(x, y):
    '''
    使用scipy (numpy) 的polyfit方法进行1、2和3阶多项式回归
    see:
    https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html

    numpy.polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False)[source]
        Least squares polynomial fit.

        Fit a polynomial p(x) = p[0] * x**deg + ... + p[deg] of degree deg to points (x, y). Returns a vector of coefficients p that minimises the squared error in the order deg, deg-1, … 0.

        The Polynomial.fit class method is recommended for new code as it is more stable numerically. See the documentation of the method for more information.

        Parameters
        x: array_like, shape (M,)
        x-coordinates of the M sample points (x[i], y[i]).

        y: array_like, shape (M,) or (M, K)
        y-coordinates of the sample points. Several data sets of sample points sharing the same x-coordinates can be fitted at once by passing in a 2D-array that contains one dataset per column.

        deg: int
        Degree of the fitting polynomial

        rcondfloat, optional
        Relative condition number of the fit. Singular values smaller than this relative to the largest singular value will be ignored. The default value is len(x)*eps, where eps is the relative precision of the float type, about 2e-16 in most cases.

        full: bool, optional
        Switch determining nature of return value. When it is False (the default) just the coefficients are returned, when True diagnostic information from the singular value decomposition is also returned.

        w: array_like, shape (M,), optional
        Weights to apply to the y-coordinates of the sample points. For gaussian uncertainties, use 1/sigma (not 1/sigma**2).

        cov: bool or str, optional
        If given and not False, return not just the estimate but also its covariance matrix. By default, the covariance are scaled by chi2/sqrt(N-dof), i.e., the weights are presumed to be unreliable except in a relative sense and everything is scaled such that the reduced chi2 is unity. This scaling is omitted if cov='unscaled', as is relevant for the case that the weights are 1/sigma**2, with sigma known to be a reliable estimate of the uncertainty.

        Returns
        p: ndarray, shape (deg + 1,) or (deg + 1, K)
        Polynomial coefficients, highest power first. If y was 2-D, the coefficients for k-th data set are in p[:,k].

        residuals, rank, singular_values, rcond
        Present only if full = True. Residuals is sum of squared residuals of the least-squares fit, the effective rank of the scaled Vandermonde coefficient matrix, its singular values, and the specified value of rcond. For more details, see linalg.lstsq.

        V: ndarray, shape (M,M) or (M,M,K)
        Present only if full = False and cov`=True. The covariance matrix of the polynomial coefficient estimates. The diagonal of this matrix are the variance estimates for each coefficient. If y is a 2-D array, then the covariance matrix for the `k-th data set are in V[:,:,k]

    
    输出residual sum of squares和模型的系数
    画出拟合的曲线 (你需要用class numpy.poly1d来生成曲线对应的函数对象)
    '''
    # start your code here
    show_data(x, y,False)
    colors = ['red', 'blue', 'green']
    orders = [1, 2, 3]
    for index in range(len(orders)):
        fp, residuals, rank, sv, rcond = np.polyfit(x, y, orders[index], full=True)

        f = sp.poly1d(fp)
        # print(residuals)
        print("Degree={},".format(orders[index]), error(y, f(x)), fp)
        plt.plot(x, f(x), color=colors[index], linewidth=3)
    plt.show()
    # end your code here


if __name__ == "__main__":
    about_PolynomialFeatures()

    x, y = load_data()

    show_data(x, y, True)

    simple_linear_regression(x, y)

    compare_LinearRegression(x, y)

    polynomial_regression_in_scipy(x, y)
