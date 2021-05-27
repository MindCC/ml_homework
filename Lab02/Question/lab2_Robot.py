# -*- coding: cp936 -*-
'''
ID:
Name:
'''

direc = ['N', 'W', 'S', 'E']


class Robot:
    '''
    You have to define a class Robot that has a name,
    a location and a direction which it faces.
    Robot’s direction may be represented by E, S, W or N,
    which means east, south, west, or north respectively.
    Robot can turnLeft or moveForward.
    
    机器人包括名称、位置和面对的方向，方向用字符表示，
    E、S、W和N分别表示东、南、西和北，

    4个属性x, y, name, and direction分别表示x坐标, y坐标, 名字和面对的方向

    机器人可以turn_left（左转）和 move_forward（向前移动1步）

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
        Robot向左转，它将修改面对的方位
        '''

        index = (direc.index(self.__direc) + 1) % 4
        self.__direc = direc[index]

    def move_forward(self):
        '''
        Robot向前走一步，它根据面对的方位，确定修改x或者y，修改时可能加1或者减1
        '''
        # direc = ['N', 'W', 'S', 'E']
        index = direc.index(self.__direc)
        if index == 0:
            self.__y += 1
        elif index == 1:
            self.__x -= 1
        elif index == 2:
            self.__y -= 1
        elif index == 3:
            self.__x += 1
        else:
            pass

    def __str__(self):
        '''
        返回Robot的字符串表示：(x,y,name,direction)
        '''
        return "(" + str(self.__x) + "," + \
               str(self.__y) + "," + \
               self.__name + "," + self.__direc + ")"


class SuperRobot(Robot):
    '''
    SuperRobot是Robot的子类，除了继承Robot的方法外，它还能够：
    move_forward向前移动任意步；
    turn_right向右转
    turn_back向后转
    '''

    def __init__(self, x, y, name, direc):
        Robot.__init__(self, x, y, name, direc)

    def move_forward(self, step=1):
        '''
        Robot向前走step步
        '''
        while step:
            Robot.move_forward(self)
            step -= 1

    def turn_right(self):
        '''
        Robot向右转
        '''
        Robot.turn_left(self)
        Robot.turn_left(self)
        Robot.turn_left(self)

    def turn_back(self):
        '''
        Robot向后转
        '''
        Robot.turn_left(self)
        Robot.turn_left(self)


if __name__ == "__main__":
    a = Robot(1, 1, "Tigger", "E")
    print(a)
    a.turn_left()
    print(a)

    b = SuperRobot(2, 2, "Bank", "S")
    print(b)
    b.turn_right()
    print(b)
    b.move_forward(10)
    print(b)

    b.turn_back()
    print(b)
    b.move_forward(10)
    print(b)
