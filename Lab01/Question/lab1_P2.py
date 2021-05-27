# -*- coding: cp936 -*-
'''
ID:
Author:
'''


def example(a, b):
    '''
    ���ش��ڵ���a��С��b������3�ı�������֮��
    Example:example(1,10)��ֵ��18 = 3 + 6 + 9
    '''
    answer = 0
    for i in range(a, b):
        if i % 3 == 0:
            answer += i

    return answer


def p2a(a, b):
    '''
    ���ش��ڵ���a��С��b������ż��֮��
    Example:p2a(1,10)��ֵ��20
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
    ���ش��ڵ���a��С��b������ƽ����֮��
    Example:p2b(1,10)��ֵ��14
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
    ���ش��ڵ���a��С��b����������֮��
    Example:p2c(1,10)��ֵ��25
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
    ���ز���n������������������������֮��
    Example:p2c(3245)��ֵ��3+5=8
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
