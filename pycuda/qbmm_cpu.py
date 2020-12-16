from advancer import *
from config_manager import *
import sys

sys.path.append("../utils/")
from stats_util import *
from euler_util import *
from jets_util import *
from pretty_print_util import *
import cProfile

import pycuda.autoinit
import pycuda.driver as drv
import numpy


class TestInversion: 

    def __init__(self, config_file: str) -> None:

        config_mgr = config_manager(config_file)
        config = config_mgr.get_config()
        self.qbmm_mgr = qbmm_manager(config)

        # Initial condition
        mu = [1.0, 1.0]
        sig = [0.1, 0.1]
        self.indices = self.qbmm_mgr.indices
        self.moments = raw_gaussian_moments_bivar(self.indices, 
                    mu[0], mu[1],
                    sig[0], sig[1])
    
    def run(self) -> None:
        xi, weight = self.qbmm_mgr.moment_invert(self.moments, self.indices)
        print(xi, weight)

KERNEL = 


def TestCudaInvert:

    def __init__(self) -> None:
    


if __name__ == "__main__":
    config_file = '../inputs/example_2d.yaml'
    TI = TestInversion(config_file)
    TI.run()



    