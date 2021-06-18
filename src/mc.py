import bubble_model as bm
import bubble_state as bs
import numpy as np
import matplotlib.pyplot as plt
import waveforms as wf

import scipy.io as sio

class mc:
    def __init__(self, config=None):
        self.adv_config = config["advancer"]
        self.mc_config = config["mc"]
        self.pop_config = config["pop"]
        self.model_config = config["model"]
        self.wave_config = config["wave"]

        if "Ntimes" in self.mc_config:
            self.Nt = self.mc_config["Ntimes"]
        else:
            raise Exception("No number of output times")

        if "Nsamples" in self.mc_config:
            self.Nmc = self.mc_config["Nsamples"]
        else:
            raise Exception("No number of samples")

        if "final_time" in self.adv_config:
            self.T = self.adv_config["final_time"]
        else:
            raise Exception("No final_time")

        self.state = bs.bubble_state(
            pop_config=self.pop_config, model_config=self.model_config
        )
        self.wave = wf.waveforms(config=self.wave_config)

    def get_sample(self):
        self.sample = []
        for i,shape in enumerate(self.state.shape):
            if shape == "lognormal":
                self.sample.append(
                    np.random.lognormal(
                        np.log(self.state.mu[i]), self.state.sig[i], self.Nmc
                        )
                    )
            elif shape == "normal":
                self.sample.append( 
                    np.random.normal(self.state.mu[i], self.state.sig[i], self.Nmc)
                    )
            else:
                raise NotImplementedError

        self.sample = np.asarray(self.sample)
        self.sample = np.array(list(zip(self.sample[0],self.sample[1])))

    def moment(self, sample=[]):
        ret = np.zeros((self.state.Nmom, self.Nt))
        for k, mom in enumerate(self.state.moments):
            if self.state.num_RV_dim == 2:
                for samp in sample:
                    ret[k, :] += samp.y[0] ** mom[0] * samp.y[1] ** mom[1]
            else:
                raise NotImplementedError

        return ret / len(sample)

    def run(self):
        T = self.T
        ts = np.linspace(0, T, num=self.Nt)
        p = self.wave.p
        R0 = 1

        self.get_sample()
        # print(self.sample)
        # raise NotImplementedError
        sols = []
        counter = int(0)
        for s in self.sample:
            bubble = bm.bubble_model(config=self.model_config, R0=R0)
            sol = bubble.solve(T=T, p=p, Ro=s[0], Vo=s[1], ts=ts)
            sols.append(sol)
            counter = counter+1
            print(counter)
        
        R_samples = np.zeros((len(self.sample),self.Nt),dtype=float)
        Rd_samples = np.zeros((len(self.sample),self.Nt),dtype=float)
        for ii in range(0,len(self.sample)):
            R_samples[ii,:]  = sols[ii].y[0,:]
            Rd_samples[ii,:] = sols[ii].y[1,:]
        Nmom = len(self.state.moments)
        moments = self.moment(sols)
        pressure = np.zeros(self.Nt,dtype=float)
        if (self.wave_config["form"] == "constant"):
            for tt in range(0,self.Nt):
                pressure[tt] = self.wave_config["amplitude"]
        else:
            for tt in range(0,self.Nt):
                pressure[tt] = self.wave.ambient
                for ii in range(0,np.size(self.wave_config["amplitude"])):
                    pressure[tt] = pressure[tt] +self.wave_config["amplitude"][ii]*np.sin(2.0*np.pi*sols[1].t[tt]/self.wave_config["period"][ii] +self.wave_config["phase"][ii] )
        sio.savemat(self.adv_config["output_dir"]+"MC_HM_"+self.adv_config["output_id"]+".mat" ,{"moments":moments,"T":sols[1].t,
                    "p_amp":self.wave_config["amplitude"],"p_phase":self.wave_config["phase"],"p_period":self.wave_config["period"],"pressure":pressure,'R_samples':R_samples,'Rd_samples':Rd_samples})
        fig, ax = plt.subplots(1, Nmom)
        # fig, ax = plt.subplots(1, self.state.Nmom)
        # for i in range(self.state.Nmom):
        for i in range(Nmom):
            ax[i].plot(sols[i].t, moments[i])
            # ax[i].set(xlabel="$t$", ylabel="$M$" + str(self.state.moments[i]))
        # plt.show()
