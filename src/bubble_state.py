import bubble_model as bm
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
from sys import exit
from moments import get_G

class bubble_state:
    def __init__(self, pop_config={}, model_config={}):

        self.model_config = model_config
        self.pop_config = pop_config
        self.set_defaults()
        self.parse_config()
        self.get_bubbles()

        # Assume all bubbles have the same model
        self.num_RV_dim = self.bubble.num_RV_dim

    def set_defaults(self):
        self.shape = ["lognormal","normal"]
        self.mu = [1.0,0.0]
        self.sig = [1.0,0.0]
        self.moments = [[0, 0]]
        self.Nmom = 1

    def parse_config(self):
        if "shape" in self.pop_config:
            self.shape = self.pop_config["shape"]

        if "sig" in self.pop_config:
            self.sig = self.pop_config["sig"]

        if "mu" in self.pop_config:
            self.mu = self.pop_config["mu"]

        if "moments" in self.pop_config:
            self.moments = self.pop_config["moments"]
            self.Nmom = len(self.moments)

    def get_bubbles(self):
        self.bubble = bm.bubble_model(config=self.model_config, R0=1.)

    # def get_rhs(self, state, p):
    #     self.vals[:, :] = state
    #     for i in range(self.NR0):
    #         self.rhs[i, :] = self.bubble[i].rhs(p)
    #     return self.rhs

    def get_quad(self, vals=None, filt=False, Nfilt=0, Tfilt=False, shifts=0):
        ret = np.zeros(self.Nmom)
        if filt:
            if Nfilt > 0:
                for k, mom in enumerate(self.moments):
                    if self.num_RV_dim == 2 and len(mom) == 2:
                        G = np.zeros(self.NR0)
                        for q in range(Nfilt):
                            G += vals[q, :, 0] ** mom[0] * vals[q, :, 1] ** mom[1]
                        G /= float(Nfilt)
                        ret[k] = np.sum(self.w[:] * G[:])
                    else:
                        raise Exception('I cant handle a requested moment...')
            elif Tfilt:
                for k, mom in enumerate(self.moments):
                    if self.num_RV_dim == 2 and len(mom) == 2:
                        # vals[ time, R0_node, int_coordinate ]
                        # Get max number of times (going backward)
                        Nt = len(vals[:,0,0])
                        G = get_G(vals=vals,
                                mom=np.array(mom),
                                Nt=Nt,
                                shifts=shifts,
                                NR0=self.NR0)

                        ret[k] = np.sum(self.w[:] * G[:])
                    else:
                        raise Exception('I cant handle a requested moment...')
            else:
                raise Exception('Need one of Nfilt or Tfilt')
        else:
            for k, mom in enumerate(self.moments):
                if self.num_RV_dim == 2 and len(mom) == 2:
                    ret[k] = np.sum(
                        self.w[:] * vals[:, 0] ** mom[0] * vals[:, 1] ** mom[1]
                    )
                elif self.num_RV_dim == 2 and len(mom) == 3:
                    ret[k] = np.sum(
                        self.w[:]
                        * vals[:, 0] ** mom[0]
                        * vals[:, 1] ** mom[1]
                        * self.R0[:] ** mom[2]
                    )
                else:
                    raise Exception

        return ret
