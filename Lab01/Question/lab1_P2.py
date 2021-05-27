# -*- coding: cp936 -*-
'''
ID:
Author:
'''


def example(a, b):
    '''
    返回大于等于a且小于b的所有3的倍数的数之和
    Example:example(1,10)的值是18 = 3 + 6 + 9
    '''
    answer = 0
    for i in range(a, b):
        if i % 3 == 0:
            answer += i

    return answer


def p2a(a, b):
    '''
    返回大于等于a且小于b的所有偶数之和
    Example:p2a(1,10)的值是20
    '''
    # START CODE
    answer = 0
    for i in range(a, b):
        if not i & 1:
            answer += i
    return answer
    # STOP CODE


def p2b(a, b):
    '''
    返回大于等于a且小于b的所有平方数之和
    Example:p2b(1,10)的值是14
    '''
    # START CODE
    answer = 0
    for i in range(a, b):
        if i ** 2 in range(a, b):
            answer += i ** 2
    return answer
    # STOP CODE


def p2c(a, b):
    '''
    返回大于等于a且小于b的所有奇数之和
    Example:p2c(1,10)的值是25
    '''
    # START CODE
    answer = 0
    for i in range(a, b):
        if i & 1:
            answer += i
    return answer
    # STOP CODE


def p2d(n):
    '''
    返回参数n的所有数字中是奇数的数字之和
    Example:p2c(3245)的值是3+5=8
    '''
    # START CODE
    answer = 0
    while n % 10:
        if n % 10 & 1:
            answer += n % 10
        n //= 10
    return answer
    # STOP CODE


if __name__ == "__main__":
    assert example(1, 10) == 18
    print(p2a(1, 10))
    assert p2a(1, 10) == 20
    print(p2b(1, 10))
    assert p2b(1, 10) == 14
    print(p2c(1, 10))
    assert p2c(1, 10) == 25
    print(p2d(3245))
    assert p2d(3245) == 8
