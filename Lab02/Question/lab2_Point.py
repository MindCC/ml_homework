# -*- coding: cp936 -*-
'''
ID:
Name:
'''

import math


class Point:
    '''
    Define a class of objects called Point (in the two-dimensional plane).
    A Point is thus a pair of two coordinates (x and y).
    Every Point should be able to calculate its distance
    to any other Point once the second point is specified. ��
    �����άƽ���ϵ�Point���㣩��Point����2������x��y��
    ��Point����ʵ�ּ���2��֮�����ķ���distanceTo( Point )����
    '''

    def __init__(self, x, y):
        self.__x = x
        self.__y = y

    def distance_to(self, other):
        '''
        ���㲢����self��other֮��ľ���
        '''
        # sqrt(x^2+y^2)
        return math.sqrt((self.__x - other.__x) ** 2 + (self.__y - other.__y) ** 2)

    def __str__(self):
        '''
        ���ص���ַ�����ʾ��(x,y)
        '''
        return '(%d,%d)' % (self.__x, self.__y)


class Line:
    '''
    Define a class of objects called Line. ������Line(�ߣ�
    Every Line is a pair of two Points. Line����2����
    Lines are created by passing two Points to the Line constructor.
    A Line object must be able to report its length,
    which is the distance between its two end points.
    '''

    def __init__(self, p1, p2):
        self.__p1 = p1
        self.__p2 = p2

    def length(self):
        '''
        ��ʾ�ߵ�2��֮��ľ���
        '''
        return self.__p1.distance_to(self.__p2)

    def __str__(self):
        '''
        �����ߵ��ַ�����ʾ��(x1,y1)--(x2,y2)
        '''
        return str(self.__p1) + '--' + str(self.__p2)


if __name__ == "__main__":
    p1 = Point(0, 3)
    print(p1)
    assert str(p1) == "(0,3)"
    p2 = Point(4, 0)
    print(p2)
    assert str(p2) == "(4,0)"
    print(p1.distance_to(p2))  # should be 5.0

    line1 = Line(p1, p2)
    print(line1)
    assert str(line1) == "(0,3)--(4,0)"
    print(line1.length())  # should be 5.0

    print(Line(Point(0, 0), Point(1, 1)).length())
