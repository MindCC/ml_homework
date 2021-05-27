# -*- coding: cp936 -*-
'''
ID:
Author:
'''
import random
import lab1_sudoku_matrix as lsm
import numpy as np


def construct_matrix_1(n):
    '''
    ����˵������list��append��������һ��n*n����
    '''
    m = []
    # ������n��
    for i in range(n):
        # ��i��
        row = []
        # ÿ����n��
        for j in range(n):
            # ���һ��0��10����������֮�����
            row.append(random.randint(0, 10))

        m.append(row)

    return m


def construct_matrix_2(n):
    '''
    ����˵������list���±깹��һ��n*n����
    '''
    # �ȹ���һ��Ԫ��ֵΪ0��n*n����
    m = [[0 for i in range(n)] for j in range(n)]

    # ��ÿ��row
    for row in range(n):
        # �Ե�row�е�ÿһ��col
        for col in range(n):
            # ���һ��0��10����������֮�����
            m[row][col] = random.randint(0, 10)

    return m


def sum_matrix(m):
    '''
    ����˵����η���list�е�ÿ��Ԫ��
    '''
    total = 0
    # ��ÿ��row
    for row in range(len(m)):
        # �Ե�row�е�ÿһ��col
        for col in range(len(m[row])):
            # �ۼӵ�row�е�col�е�Ԫ��
            total += m[row][col]

    return total


def is_magic(m):
    '''
    ���һ�������ǲ��ǻ÷�����ÿ�С�ÿ�к�2���Խ���֮�Ͷ���ͬ
    ����ǣ�����True��������ǣ�����False
    ��������������󣨿�����list��ʾ���ǻ÷�����
    16  3  2 13
     5 10 11  8
     9  6  7 12
     4 15 14  1
    '''
    # START CODE
    answer = 0
    answer1 = 0
    answer2 = 0
    answer3 = 0
    # �����ά��
    k = len(m[0])
    for i in range(k):
        answer += int(m[i][i])
        answer1 += int(m[i][k - i - 1])
    if answer1 != 0 and answer != 0:
        for i in range(k):
            answer2, answer3 = 0, 0
            for j in range(k):
                answer2 += int(m[i][j])
                answer3 += int(m[j][i])
            if not (answer2 & answer3):
                return False
    return answer == answer1 == answer2 == answer3
    # STOP CODE


def read_matrix():
    '''
    ���û�����16��������input���������������ǹ���ɾ�����list��ʾ��
    �������
    16  3  2 13
     5 10 11  8
     9  6  7 12
     4 15 14  1
    ������list��ʾΪ[[16,3,2,13],[5,10,11,8],[9,6,7,12],[4,15,14,1]]
    '''
    # START CODE
    i = 0
    list = []
    answer = []
    while True:
        i += 1
        a = input("input %d numbers:" % i)
        list.append(a)
        if not i % 4:
            answer.append(list.copy())
            list.clear()
        if i == 16:
            break
    return answer
    # STOP CODE


def construct_magic(n):
    '''
    ����n*n��nΪ����
    �㷨α���룺
    Set row = n - 1, column = n / 2.
    For k = 1 ... n * n
        Place k at [row][column].
        Increment row and column.
        If the row or column is n, replace it with 0.
        If the element at [row][column] has already been filled
            Set row and column to their previous values.
            Decrement row.
    '''
    # START CODE
    row = n - 1
    column = int(n / 2)
    matrix = np.zeros((n, n), dtype=np.int)
    # matrix = [[0 for i in range(3)] for j in range(3)]
    for k in range(1, n * n + 1):
        matrix[row, column] = k
        row += 1
        column += 1
        if row == n:
            row = 0
        if column == n:
            column = 0
        if not matrix[row, column] == 0:
            row -= 2
            column -= 1

    return matrix


def is_sudoku_matrix(m):
    '''
    https://www.jianshu.com/p/53d1cab0f2f5
    ������sudoku�������Ǹ��Ź���ÿһ���ַ�Ϊ�Ÿ�С��
    ���ʮһ���и���һ������֪���ֺͽ��������������߼��������������Ŀո�������1-9�����֡�
    ʹ1-9ÿ��������ÿһ�С�ÿһ�к�ÿһ���ж�ֻ����һ�Σ������ֳơ��Ź��񡱡�

    This method is used to check whether a 9*9 matrix is a sudoku matrix
    Sudoku is a popular puzzle consisting of a grid of 9 by 9 numbers.
    Each grid number is in the range 1-9 which can only occur once in any row/column combination
    and the 3x3 square in which it resides.
    An example:
    [1, 3, 2, 9, 7, 4, 6, 5, 8]
    [5, 6, 4, 8, 3, 1, 9, 7, 2]
    [8, 7, 9, 5, 2, 6, 3, 1, 4]
    [2, 5, 8, 7, 6, 9, 1, 4, 3]
    [9, 4, 6, 3, 1, 2, 7, 8, 5]
    [3, 1, 7, 4, 8, 5, 2, 6, 9]
    [7, 2, 5, 6, 4, 3, 8, 9, 1]
    [6, 9, 1, 2, 5, 8, 4, 3, 7]
    [4, 8, 3, 1, 9, 7, 5, 2, 6]

    To create a sudoku matrix, see lab1_sudoku_matri.py.
    You can use the functions provided in lab1_sudoku_matrix.py
    '''
    # START CODE
    # ����˫���飬�����
    # for i in range(0, 3):
    #     for j in range(0, 3):
    #         # �Ź������Ƿ����
    #         # matrix.flatten()
    #         list = matrix[3 * i:3 * (i + 1), 0 + 3 * j:3 * (j + 1)].tolist()
    #         # ��listչ��һά��
    #         list = [i for j in list for i in j]
    #         if not isrepeat(list):
    #             return False
    # for i in range(9):
    #     list = matrix[i, :].tolist()
    #     if not isrepeat(list):
    #         return False
    #     list = matrix[:, i].tolist()
    #     if not isrepeat(list):
    #         return False
    # return True
    for i in range(9):
        for j in range(9):
            a = lsm.get_enable_arr(m, i, j)
            if len(a) != 0:
                return False
    return True
    # START CODE


# def isrepeat(list):
#     dict = {i: True for i in range(1,9)}
#     for k in range(9):
#         if list[k] in dict.keys():
#             dict[list[k]] = False
#     # ����������δ�޸ĵ�˵���ظ���
#
#     if True in dict.values():
#         return False
#     else:
#         return True


if __name__ == "__main__":
    a = construct_matrix_1(3)
    print(a)
    print(sum_matrix(a))

    a = construct_matrix_2(3)
    print(a)
    print(sum_matrix(a))

    a = [[16, 3, 2, 13], \
         [5, 10, 11, 8], \
         [9, 6, 7, 12], \
         [4, 15, 14, 1]]
    print(is_magic(a))  # should output True
    a[0][0] = 12
    print(is_magic(a))  # should output False

    # a = read_matrix()
    # print(a)
    # print(is_magic(a))

    a = construct_magic(3)
    print(a)
    print(is_magic(a))

    a, c = lsm.create_sudoku_matrix()
    print(a)
    print(is_sudoku_matrix(a))  # should output True
