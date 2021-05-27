'''
Created on 2014-11-9

@author: Administrator
'''
import numpy as np
import os

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif, chi2

class DefectData(object):
    '''
    classdocs
    '''

    def __init__(self, file_name):
        self.relation = ''
        self.attributes = []
        self.data = []
        self.target = []
        
        f = open(file_name, 'r')
        rlen = len('@relation ')
        alen = len('@attribute ')
        dlen = len('@data')
        for line in f :
            if line[0:rlen] == '@relation ':
                self.relation = line[rlen:-1]
            elif line[0:alen] == '@attribute ':
                line = line[alen:-1]
                self.attributes.append(line.split(' ')[0])
            elif line[0:dlen] == '@data':
                pass 
            else:
                if not line[0].isdigit():
                    continue
                s = line.split(',')
                if len(s) > 1:
                    if s[-1][0] == 'Y' or s[-1][0] == 'y':
                        self.target.append(1)
                    else:
                        self.target.append(0)
                    del s[-1]
                    features = [float(d) for d in s]
                    self.data.append(features)
                    #print(features)
        #delete the last element of attributes, it is the name of target
        del self.attributes[-1]
        
    def get_data(self, score_func = f_classif, percentile=10):
        data = self.data
        target = self.target
        print(len(data[0]), end = '\t')
        data_new = SelectPercentile(score_func, percentile).fit_transform(data, target)
        print(len(data_new[0]), end = '\t')
        return data_new
    
    def get_target(self):
        return np.array(self.target)  
    
    def get_relation(self):
        return self.relation
    
    def get_attributes(self):
        return np.array(self.attributes)


                    
if __name__ == '__main__':
    from os import listdir
    file_dir = os.path.abspath('.')+'\\SDData\\'
    print(file_dir)
    file_list = listdir(file_dir)
    for file_name in file_list:
        defect_data = DefectData(file_dir + file_name)
        print(defect_data.relation)
        a = defect_data.get_attributes()
        print(a.shape)
        print(a)
        
        d = defect_data.get_data()
        print(d.shape)
        print(d) 
           
        t = defect_data.get_target()
        print(t.shape)
        print(t) 
        print(np.count_nonzero(t))        
        
