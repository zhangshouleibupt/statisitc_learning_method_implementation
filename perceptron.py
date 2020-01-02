import numpy as np
import random
from tqdm import tqdm
class Perceptron():
    def __init__(self,shuffle=True,lr=0.01,max_iter=100):
        self.shuffle = shuffle
        self.max_iter = max_iter
        self.lr = lr
    def _choose_one_data(self):
        l = self.x_data.shape[0] - 1
        i = 0
        while i <= 10:
            idx = random.randint(0,l)
            x,y = self.x_data[idx],self.y_data[idx]
            pred = self._predict(x)
            if pred == y:
                i+=1
            else:
                return x,y
        return None,None
    def _update(self):
        x,y = self._choose_one_data()
        if x is not None:
            self.weight += self.lr * y * x
            self.b += self.lr * y 
    def _predict(self,x):
        if np.sum(self.weight * x + self.b) < 0:
            return -1
        else:
            return 1
    def fit(self,x_data,y_data):
        self.weight = np.ones(x_data.shape[1],dtype=np.float32)
        self.b = 0.0
        self.x_data = x_data
        self.y_data = y_data
        for epoch in tqdm(range(self.max_iter)):
            self._update()

    def predict(self,x):
        ans = [self._predict(x) for each_x in x]
        return np.array(ans)
def main():
    x = [[3,3],[4,3],[1,1]]
    y = [1,1,-1]
    x = np.array(x,dtype=np.float32)
    y = np.array(y)
    clf = Perceptron()
    clf.fit(x,y)
    test_x = [[0,0],[1,1]]
    text_x = np.array(test_x)
    pred = clf.predict(test_x)
    print(pred)
if __name__ == "__main__":
    main()
