
import sys
import numpy

import qbmmlib.src.advancer as advancer
import qbmmlib.src.qbmm_manager as qbmm
import qbmmlib.src.config_manager as cfg
import qbmmlib.utils.stats_util as stats

class TestInversion: 

    def __init__(self, config_file: str) -> None:

        config_mgr = cfg.config_manager(config_file)
        config = config_mgr.get_config()
        self.qbmm_mgr = qbmm.qbmm_manager(config)

    def test_one(self) -> None:
        # Initial condition
        mu = [1.0, 1.0]
        sig = [0.1, 0.1]
        indices = self.qbmm_mgr.indices
        moments = stats.raw_gaussian_moments_bivar(indices, 
                    mu[0], mu[1],
                    sig[0], sig[1])
    
        xi, weight = self.qbmm_mgr.moment_invert(moments, indices)
        print(xi, weight)
    


if __name__ == "__main__":
    # file path relative to package source directory
    config_file = 'inputs/example_2d.yaml'
    TI = TestInversion(config_file)
    TI.test_one()



    