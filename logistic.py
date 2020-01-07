import numpy as np
def softmax(x):
    sigmax = 1 + np.sum(np.exp(x))
    return x / sigma 

class Logistic():
    def __init__(self,lr=0.001,max_iter=None):
        self.lr = lr
        self.max_iter = max_iter
    def fit(self,train_x,train_y):
        self.train_x = train_x
        self.train_y = train_y
        n,dim = train_x.shape
        self.w = np.zeros(dim)
        self.n = n
        self.b = b
        if self.max_iter is None:
            self.max_iter = n * 3
        for i in range(self.max_iter):
            self._update_para()
    def _cal_prob(self,x):
        t = np.exp(np.sum(self.w * self.x) + self.b)
        p_of_one = t / (1.0 + t)
        return np.array([1.0-p_of_one,p_of_one])
    def predict(self,test_x):
        return [np.argmax(self._cal_prob(x)) for x in test_x]
    def _sample_one_batch(self):
        return [random.randint(0,self.b) for i in range(self.batch_size)] 
    def _update_para(self):
        batch_idx = self._sample_one_batch(self)
        x = self.train_x[batch_idx]
        y = self.train_y[batch_idx]
        .
