
import sys
import numpy as np

import qbmmlib.src.advancer as advancer
import qbmmlib.src.inversion as inv
import qbmmlib.src.config_manager as cfg
import qbmmlib.utils.stats_util as stats

class TestInversion: 

    def __init__(self) -> None:

        self.inversion = inv.chyqmom4
        # self.indices = np.array(
        #     [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2], [3, 0], [0, 3], [4, 0], [0, 4]])
        self.indices = np.array(
            [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]])
    def test_one(self) -> None:
        # Initial condition
        moments = stats.raw_gaussian_moments_bivar(self.indices, 
                    1.00, 1.00,
                    0.1, 0.1)    
        xi, weight = self.inversion(moments, self.indices)
        print(moments)
        print(weight)
        print(xi[1])
    
    
    def init_batch_input(self, n: int) -> np.ndarray:
        '''
        initialize an Nx5 numpy array containing N moments
        All moments are the same, initialized with raw_gaussian_moments_bivar
        '''
        moments = []            
        mu = [1.0, 1.0]
        sig = [0.1, 0.1]
        one_moment = stats.raw_gaussian_moments_bivar(self.indices,
                mu[0], mu[1],
                sig[0], sig[1])
        for i in range(n):
            moments.append(one_moment)

        return np.asarray(moments)
    
    def compute_batch(self, moments: np.ndarray, n: int) -> tuple: 
        weight = np.zeros([n, 4])
        x = np.zeros([n, 4])
        y = np.zeros([n, 4])
        for i in range(n):
            xi, w = self.inversion(moments[i], self.indices)
            weight[i, :] = w
            x[i, :] = xi[0]
            y[i, :] = xi[1]
        return weight, x, y

if __name__ == "__main__":
    TI = TestInversion()
    N = 3
    moments = TI.init_batch_input(N)
    # print(moments)
    TI.test_one()
    # weight, x, y = TI.compute_batch(moments, N)
    # print(weight)


    