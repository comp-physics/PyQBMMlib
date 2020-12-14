import sys

sys.path.append("../src/")
from qbmm_manager import *

if __name__ == "__main__":

    np.set_printoptions(formatter={"float": "{: 0.4E}".format})

    ###
    ### Say hello
    print("test_wheeler: Testing CQMOM for moment inversion")

    ###
    ### QBMM Configuration
    print("test_wheeler: Configuring and initializing qbmm")

    config = {}
    config["governing_dynamics"] = " dx + x = 1"
    config["num_internal_coords"] = 2
    config["num_quadrature_nodes"] = 4
    config["method"] = "cqmom"
    config["adaptive"] = False
    config["max_skewness"] = 30

    ###
    ### QBMM
    qbmm_mgr = qbmm_manager(config)

    indices = np.ones([4, 2])
    moments = np.ones([4, 2])
    qbmm_mgr.moment_invert(moments, indices)

    exit
