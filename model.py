import numpy as np
import collections


class knn(object):
    
    def __init__(self,k=1): # default k = 1
        self.k = k
    
    def fit(self,X,y):
        self.X = X
        self.y = y
        return None
    
    def predict(self,X_test):
        pred = []
        for test in X_test:
            test_matrix = np.array([test]*3) # transform test vector into N rows
            dist = np.sqrt(np.sum(abs((test_matrix-self.X)**2),axis = 1)) # calculate the distance 
            sorted_index = np.argsort(dist)  # get index sorted by the values increasingly 
            respones = self.y[sorted_index][0:self.k]  # take the K nearest neighbor 
            pred.append(collections.Counter(respones).most_common()[0][0]) # take the voted class
        return pred
    

