import numpy as np
import math
from collections import Counter
def distance_of_p(x1,x2,p):
    diff = x1 - x2
    tmp =  math.pow(diff,p)
    return math.pow(np.sum(tmp),1 / 1.0*p)
def euclidean_distance(x1,x2):
    return distance_of_p(x1,x2,2.0)

def manhattan_distance(x1,x2):
    diff = x1 - x2
    tmp = math.pow(math.pow(diff,2),0.5))
    return np.sum(tmp)
class Data:
    def __init__(self,idx,dis):
        self.idx = idx
        self.dis = dis
    def __lt__(self,other):
        return self.dis < other.dis
class KNN():
    def __init__(self,k=10):
        self.k = k
    def fit(self,x,y):
        #lazy learning process,just store the data
        self.train_x = x
        self.train_y = y
    def predict(self,x):
        idxs = self.find_k_nearest_data(x,self.k)
        y_labels = [self.train_y for idx in idxs]
        counter = Counter(y_labels)
        label,num = max(counter,lambda x:x[1])
        return label
    def _find_k_nearest_data(self,x,k):
        k_nearest_data = []
        l = k + 1
        for i in range(l):
            k_nearest_data.append(Data(i,euclidean_distance(self.train_x[i],x)))
        k_nearest_data.sort(lambda x:x.dis)
        for idx in range(l+1,len(self.train_x)):
            d = Data(idx,euclidean_distance(self.train_x[idx],x))
            for i in range(l,-1,-1):
                if d < self.k_nearest_data[i]:
                    k_nearset_data.insert(i,d)
                    break
            del k_nearest_data[-1]
        return [d.idx for d in k_nearset_data]

