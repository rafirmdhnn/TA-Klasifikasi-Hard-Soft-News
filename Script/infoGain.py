# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 21:09:26 2021

@author: rafir
"""

import pandas as pd
from pandas import DataFrame
from collections import Counter
import numpy as np


class infoGain(object):
    
    def __init__(self, data):
        self.data = data
        
    def countEntropyP(self):
        self.label = self.data['label'].tolist()
        self.numClass = Counter(self.label)
        self.total = len(self.label)
        
        self.entropiParent = -sum(count/self.total * np.log2(count/self.total) for count in self.numClass.values())
        print('Parent entropy:', self.entropiParent)

    def countIG(self):
        self.fitur = self.data.iloc[:, 1:30].columns.tolist()
        self.fiturIG = []
        
        for i in self.fitur:
            self.classEntropy = 0
            for c, att in self.data.groupby([i]):
                self.attTotal = att.count().values[0]
                self.p = Counter(att['label'].tolist())
                self.value = []
                for v in self.p.values():
                    self.value.append(-v/self.attTotal * np.log2(v/self.attTotal))
    
                self.classEntropy += self.attTotal/self.data.count().values[0] * sum(self.value)
            
            self.infoGain = (self.entropiParent - self.classEntropy)
            self.fiturIG.append([i, self.entropiParent, self.classEntropy, self.infoGain])
    
    def saveCsv(self, threshold, outIg, dtIg):
        self.outIg = outIg
        self.dtIg = dtIg
        
        # membuat dataframe
        self.df = DataFrame(self.fiturIG)
        self.df.columns = ['Fitur', 'Parent Entropy', 'Class Entropy', 'IG']
        self.df.index = self.df.index + 1
        self.df.sort_values("IG", ascending = False , inplace = True)
        
        self.df.to_csv(self.outIg, index=None)
        
        # get nilai threshold
        self.threshold = threshold
        # seleksi berdasarkan nilai IG sesuai threshold yg ditentukan
        self.result = self.df[self.df.IG <= self.threshold]
        # memisahkan fitur yang memiliki nilai IG <= nilai threshold
        self.wastedFeature = list(self.result['Fitur'])
        # delete column atau fitur dari data sesuai hasil yg telah diseleksi
        self.goodFeature = self.data.drop(self.wastedFeature, axis=1)
        # save data to csv
        self.goodFeature.to_csv(self.dtIg, index = None)
        print(self.wastedFeature)
        
#%%

#import dataset
file = open(r'C:\\Skripsi\Skripsi\Sidang\Dataset\single-dataset.csv','r')
dataset = pd.read_csv(file, delimiter = ',',  encoding='cp1252')

#set nilai threshold, dan path untuk menyimpan file csv output
threshold = 0.01
pathOutIg = r'C:\Skripsi\Skripsi\Sidang\Dataset\OutIG({})-result.csv'.format(threshold)
pathDtIg = r'C:\Skripsi\Skripsi\Sidang\Dataset\DtIG({})-dataset.csv'.format(threshold)

#initiallize object
ig = infoGain(dataset)
entropi = ig.countEntropyP()
out_ig = ig.countIG()
ig.saveCsv(threshold,pathOutIg, pathDtIg)