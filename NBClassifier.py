import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import warnings
import time
warnings.filterwarnings("ignore")

class NBClassifier:
    bayes_matrix= []
    SpamLabel = 0
    HamLabel = 0
    Spam=0
    Ham=0
    P_Spam=0
    P_Ham=0
    result=[]
    def __init__(self):
        self.SpamLabel = 0
        self.HamLabel = 0
        self.Spam=0
        self.Ham=0
        self.P_Spam=0
        self.P_Ham=0
        self.result=[]


    def fit (self, vectors, label):
        size= len(vectors[0])
        self.bayes_matrix = np.zeros((size,2))
        def smoothing(a, b, n):
            return float((a + 1)/(b + n))

        for j, vector in enumerate(vectors): 
            vector=csr_matrix(vector)
            indexes=vector.indices
            row =vector.data
            #print(indexes)
            #print (row)
            if label[j] == 1:
                for i,index in enumerate(indexes):
                    self.bayes_matrix[index][0] +=row[i]
                    self.Spam+= row[i]
                self.SpamLabel+=1
            else:
                for i,index in enumerate(indexes):
                    self.bayes_matrix[index][1] += row[i]
                    self.Ham+= row[i]
                self.HamLabel+=1

        self.P_Spam = self.SpamLabel/(self.SpamLabel+self.HamLabel)
        self.P_Ham =self.HamLabel/(self.SpamLabel+self.HamLabel)
        for i in range(size): 
            self.bayes_matrix[i][0] = smoothing(self.bayes_matrix[i][0], self.Spam,size) 
            self.bayes_matrix[i][1] = smoothing(self.bayes_matrix[i][1], self.Ham,size)

    def NBcalculate(self,vector):
        vector=csr_matrix(vector)
        pspam=np.log(self.P_Spam)
        pham=np.log(self.P_Ham)
        indexes=vector.indices
        row=vector.data
        for i,index in enumerate(indexes):
            pspam+=np.log(self.bayes_matrix[index][0])*row[i]
            pham+=np.log(self.bayes_matrix[index][1])*row[i]
        if pspam>pham:
            return 1
        else :
            return 0

    def predict(self,testvecs):
        result=testvecs.copy()
        for i,vec in enumerate(testvecs['Mail']):
            result['Label'][i] = self.NBcalculate(vec)
        self.result=result
        return self.result
    def score(self,testvecs):
        score=0
        for i,vec in enumerate(testvecs['Mail']):
             if( self.result['Label'][i]==testvecs['Label'][i]):
                 score=score+1
        return score/testvecs.shape[0]
