# -*- coding: cp936 -*-
'''
ID:
Author:
'''
import math
def example():
    answer = ""
    i = 0
    while i < 10:
        i += 1
        answer += str(i) + " "
    #to remove the space in right most，移去右边的空格
    answer = answer.rstrip(" ")
    return answer

def p1a( n ):
    '''
    return all squares less than n,返回所有小于n的平方数
    Example: n=30 -> "1 4 9 16 25"
    '''
    #START CODE
    k = round(math.sqrt(n))
    answer =""
    i=0
    while i<k:
        i+=1

        answer += str(i**2) + " "
    answer = answer.rstrip(" ")
    return answer
    #STOP CODE


def p1b( n ):
    '''
    return all positive numbers that are divisible by 10 and less than n
    返回所有能被10整除且小于n的正数
    Example: n=30 -> "10 20"
    '''
    #START CODE
    answer = ""
    i=0
    while i < n-1:
        i += 1
        if(i%10==0):

            answer += str(i) + " "
    answer = answer.rstrip(" ")
    return answer
    #STOP CODE

def p1c( n ):
    '''
    return all powers of two less than n
    返回所有小于n的2的幂
     Example: n=30 -> "1 2 4 8 16"
    '''
    #START CODE
    i=1
    answer = ""
    while True:
        if i>n:
            break
        answer += str(i) + " "
        i = i << 1
    answer = answer.rstrip(" ")
    return answer

    #STOP CODE

if __name__=="__main__":
    assert example() == "1 2 3 4 5 6 7 8 9 10"
    print(p1a(30))
    assert p1a(30) == "1 4 9 16 25"
    print(p1b(30))
    assert p1b(30) == "10 20"
    print(p1c(30))
    assert p1c(30) == "1 2 4 8 16"
