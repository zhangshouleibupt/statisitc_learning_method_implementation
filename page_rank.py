import numpy as np
class PageRank():
    def __init__(self,trans_matrix,threshold = 0.001,max_iter=None,d = None,method  = 'naive'):
        self.trans_matrix = trans_matrix
        self.threshold = threshold
        self.max_iter = max_iter
        self.node_nums = self.trans_matrix.shape[0]
        self.init_distribution = np.array([1.0 / self.node_nums]* self.node_nums)
        self.d = 1 if d is None else d
        self.method = method
        self.ones = np.array([1]*self.node_nums)
        #print(self.trans_matrix * self.d)
    def get_stationary_distribution(self):
        coef = 0 if self.max_iter is None else 1
        max_iter = 10 if self.max_iter is None else self.max_iter
        i = 0
        not_stop = True
        while not_stop:
            i = i + 1 * coef
            if self.method == "naive":
                next_distribution = self.trans_matrix.dot(self.init_distribution)
            elif self.method == "smooth_way":
                next_distribution = self.d * self.trans_matrix.dot(self.init_distribution) + (1-self.d) / self.node_nums * self.ones
            else:
                raise ValueError("none implementation method")
            error = np.sum(np.abs(next_distribution-self.init_distribution))
            not_stop = not(i > max_iter or error < self.threshold)
            self.init_distribution = next_distribution
        return self.init_distribution
def main():
    trans_matrix = [[0,1/2,0,0],
                    [1/3,0,0,1/2],
                    [1/3,0,1,1/2],
                    [1/3,1/2,0,0]]
    pr = PageRank(np.array(trans_matrix),threshold=0.000001,method="smooth_way",d=0.8)
    dis = pr.get_stationary_distribution()
    print(dis)
if __name__ == "__main__":
    main()
