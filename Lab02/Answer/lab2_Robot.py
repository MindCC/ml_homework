# -*- coding: cp936 -*-
'''
ID:
Name:
'''

class Robot:
    '''
    You have to define a class Robot that has a name,
    a location and a direction which it faces.
    Robot��s direction may be represented by E, S, W or N,
    which means east, south, west, or north respectively.
    Robot can turnLeft or moveForward.
    
    �����˰������ơ�λ�ú���Եķ��򣬷������ַ���ʾ��
    E��S��W��N�ֱ��ʾ�����ϡ����ͱ���

    4������x, y, name, and direction�ֱ��ʾx����, y����, ���ֺ���Եķ���

    �����˿���turn_left����ת���� move_forward����ǰ�ƶ�1����

    When you create a Robot one need specify:
        the location (x, y), a name for the robot,
        the direction the robot is facing

    The robot also needs to be able to:
        report the direction() it's facing (N, S, W, E)
        produce a complete report() (x, y, name, direction)
    '''

    def __init__(self, x, y, name, direc):
        self.__x = x
        self.__y = y
        self.__name = name
        self.__direc = direc

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def direction(self):
        return self.__direc

    @property
    def name(self):
        return self.__name

    def turn_left(self):
        '''
        Robot����ת�������޸���Եķ�λ
        '''
        if self.__direc == "E":
            self.__direc = "N"
        elif self.__direc == "N":
            self.__direc = "W"
        elif self.__direc == "W":
            self.__direc = "S"
        else:
            self.__direc = "E"

    def move_forward(self):
        '''
        Robot��ǰ��һ������������Եķ�λ��ȷ���޸�x����y���޸�ʱ���ܼ�1���߼�1
        '''
        if self.__direc == "E":
            self.__x += 1
        elif self.__direc == "N":
            self.__y -= 1
        elif self.__direc == "W":
            self.__x -= 1
        else:
            self.__y += 1
            
    def __str__(self):
        '''
        ����Robot���ַ�����ʾ��(x,y,name,direction)
        '''
        return "("+str(self.__x)+"," + \
               str(self.__y) + "," + \
               self.__name + "," + self.__direc + ")"

class SuperRobot(Robot):
    '''
    SuperRobot��Robot�����࣬���˼̳�Robot�ķ����⣬�����ܹ���
    move_forward��ǰ�ƶ����ⲽ��
    turn_right����ת
    turn_back���ת
    '''
    def __init__(self, x, y, name, direc):
        Robot.__init__(self, x,y,name,direc)

    def move_forward(self, step=1):
        '''
        Robot��ǰ��step��
        '''
        while step > 0:
            Robot.move_forward(self)
            step -= 1

    def turn_right(self):
        '''
        Robot����ת
        '''
        self.turn_left()
        self.turn_left()
        self.turn_left()

    def turn_back(self):
        '''
        Robot���ת
        '''
        self.turn_left()
        self.turn_left()

if __name__=="__main__":
    a = Robot(1,1,"Tigger","E")
    print(a)
    a.turn_left()
    print(a)

    b = SuperRobot(2,2,"Bank", "S")
    print(b)
    b.turn_right()
    print(b)
    b.move_forward(10)
    print(b)

    b.turn_back()
    print(b)
    b.move_forward(10)
    print(b)
    
    
