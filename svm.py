import numpy as np
class SVM():
    def __init__(self):
        pass
    def _poly_kernel(self,x,z,p):
        return np.pow((np.sum(x*z) + 1),p)
    def _rbf_kernel(self,x,z,sigma):
        return np.exp(-np.sum(np.pow(x-z,2)) / (2 * sigma * sigma))
    def _linear_kernel(self,x,z):
        pass
    def _update_alpha_step(self,i,j):
        """
        if the two variable of alpha has been choosed
        the step is by the smo algorithm to update
        """
        while not self._alpha_stop_condition():
            i,j = self._choose_one_alpha_pair()
            self._update_one_pair_alpha(i,j)
    def _choose_one_alpha_pair(self):
        pass
    def _alpha_stop_condition(self):
        pass
    def _get_kernel_dot_product(self,i,j):
        x1 = self.get_x(i)
        x2 = self.get_x(j)
        y1 = self.get_y(i)
        y2 = self.get_y(j)

    def _get_error(self,i):
        #the error is differ between real value
        #and the prediction 
        y_i = self._get_y(i)
        error = 0
        for j in range(self.N):
            error += self._get_alpha(j) \
                     *self._get_y(j) \  
                     *self._get_kernel_dot_product(j,i) 
        error += self.bias
        return error - y_i
    def _get_alpha_new(self,i,j):
        a_i_old = self._get_alpha_old(i)
        y_i = self._get_y(i)
        e_i = self._get_error(i)
        e_j = self._get_error(j)
        k_ii = self._get_kernel_dot_product(i,i)
        k_jj = self._get_kernel_dot_product(j,j)
        k_ij = self._get_kernel_dot_product(i,j)
        eta = k_ii + k_jj - 2 * k_ij
        a_i_newunc = a_i_old + y_i (e_i - e_j) / eta
        H = self._get_H(i,j)
        L = self._get_L(i,j)
        if a_i_newunc > H:
            a_i_new = H
        elif a_i_newunc < L:
            a_i_new = L
        else:
            a_i_new = a_i_newunc
        return a_i_new
    def _update_one_pair_alpha(self,i,j):
        a_i_new = self._get_alpha_new(i,j)
        y_i = self._get_y(i)
        y_j = self._get_y(j)
        a_i_new = self._get_alpha_new(i)
        a_j_new = self._get_alpha(j) + y_i * y_j*(self._get_alpha(i - a_i_new))
        return (a_i_new,a_j_new)

