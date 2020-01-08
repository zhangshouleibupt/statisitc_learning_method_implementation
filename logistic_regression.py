@author zhangshoulei@bupt
"""
实现了二分类的逻辑斯特回归
利用批量梯度下降(有别于课本上的梯度下降)
"""
#todo:多分类任务
import numpy as np
import random
import math
def logistic(x,miu=0,gamma=1.0):
    return 1.0 / (1 + math.exp(-(x-miu)/gamma))
def derivation_of_logistic(x,miu=0,gamma=1.0):
    t = logistic(x,miu=miu,gamma=gamma)
    return  (1 - t) * t / gamma
def softmax(x):
    sigmax = 1 + np.sum(np.exp(x))
    return x / sigma 

class LogisticRegression():
    #just like the book,we turn the x in to [x,1]
    #e.g,if x is original [1.1,2,3],then turn the x into [1.1,2,3,1]
    #and the w,b could compacte into one vector w^ = [w,b]
    #w*x + b == [w,b] * [x,1] == w * x + b*1 == w*x + b
    def __init__(self,lr=0.001,max_iter=None):
        self.lr = lr
        self.max_iter = 6 if max_iter is None else max_iter
        self.batch_size = 16
    def fit(self,train_x,train_y):
        self.train_y = train_y
        n,dim = train_x.shape
        self.w = np.zeros(dim+1)
        ones = np.ones(n,1)
        self.train_x = np.cat((train_x,ones),dim=-1)
        for i in range(self.max_iter):
            self._update_para()
    def _cal_prob(self,x):
        ones = np.ones(x.shape[0],1)
        x = np.cat((x,ones),dim=1)
        t = np.exp(np.sum(self.w * self.x))
        p_of_one = t / (1.0 + t)
        return np.array([1.0-p_of_one,p_of_one])
    def predict(self,test_x):
        pass
    def _odds(self,x):
        t = np.exp(self.w * x)
        return t / (1 + t)
    def _update_para(self):
        idxs = [i for  i in range(self.n)]
        random.shuffle(idxs)
        iters = self.n // self.batch_size
        for i in range(iters):
            idx = idxs[i*self.batch_size:(i+1)*batch_size]
            batch_x = self.train_x[idx]
            batch_y = self.train_y[idx]
            odds = [self._odds(x) for x in batch_x]
            gradient = - [batch_y - odds] * batch_x
            gradient = np.sum(gradient,dim=0)
            self.w = self.w - self.lr * gradent

