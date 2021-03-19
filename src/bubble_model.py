import numpy as np
import scipy.integrate as sp
import matplotlib.pyplot as plt
from sys import exit


class bubble_model:
    def __init__(self, config={}, R0=1.0):

        self.config = config
        self.R0 = R0
        self.set_defaults()
        self.parse_config()
        self.check_inputs()

        if self.model == "RPE" or self.model == "KM" or self.model == "Linear":
            self.num_RV_dim = 2
            self.state = np.array([self.R, self.V])
        else:
            raise NotImplementedError

    def set_defaults(self):

        self.model = "RPE"
        self.R = self.R0
        self.V = 0.0
        self.gamma = 1.4
        self.Ca = 1.0
        self.viscosity = False
        self.Re_inv = 0
        self.tension = False
        self.Web = 0
        self.c = 0.0

    def parse_config(self):

        if "model" in self.config:
            self.model = self.config["model"]

        if "R" in self.config:
            self.R = self.config["R"]

        if "V" in self.config:
            self.V = self.config["V"]

        if "gamma" in self.config:
            self.gamma = self.config["gamma"]

        if "Ca" in self.config:
            self.Ca = self.config["Ca"]

        if "Re_inv" in self.config:
            self.viscosity = True
            self.Re_inv = self.config["Re_inv"]

        if "Web" in self.config:
            self.tension = True
            self.Web = self.config["Web"]

        if "c" in self.config:
            self.c = self.config["c"]
        elif self.model == "KM":
            raise Exception("need c")

    def check_inputs(self):

        if self.Web <= 0.0 and self.tension:
            raise ValueError(self.Web)

        if self.Re_inv <= 0.0 and self.viscosity:
            raise ValueError(self.Re_inv)

    def get_cpbw(self):
        self.cpbw = self.Ca * ((self.R0 / self.R) ** (3.0 * self.gamma)) - self.Ca + 1.0
        if self.tension:
            self.cpbw -= (
                2.0
                / (self.Web * self.R0)
                * ((self.R0 / self.R) - (self.R0 / self.R) ** (3.0 * self.gamma))
            )

    def get_dpbdt(self):
        self.dpbdt = (
            -3.0
            * self.gamma
            * self.Ca
            * self.R0
            * self.V
            * (self.R0 / self.R) ** (3.0 * self.gamma - 1.0)
            / self.R ** 2.0
        )
        if self.tension:
            self.dpbdt += (
                2.0
                * self.V
                / (self.R0 * self.Web * self.R ** 2.0)
                * (
                    self.R0
                    - 3.0
                    * self.gamma
                    * self.R
                    * (self.R0 / self.R) ** (3.0 * self.gamma)
                )
            )
        if self.viscosity:
            self.dpbdt += 4.0 * self.Re_inv * (self.V / self.R) ** 2.0

    def km(self, p):
        pressure = p[0]
        dpdt = p[1]
        self.get_cpbw()
        self.get_dpbdt()
        dpwdt = self.dpbdt - dpdt
        # dpwdt = 0.0

        rhs = (
            (1.0 + self.V / self.c) * (self.cpbw - pressure)
            + self.R / self.c * dpwdt
            - (1.5 - self.V / (2.0 * self.c)) * self.V ** 2.0
        )

        if self.viscosity:
            # associated with LHS
            rhs -= (1 + self.V / self.c) * 4.0 * self.Re_inv * self.V / self.R
            # associated with LHS (top) and dpbwdt in RHS (bottom)
            rhs /= self.R * (1.0 - self.V / self.c) + 4.0 * self.Re_inv / self.c
        else:
            rhs /= self.R * (1.0 - self.V / self.c)

        return [self.V, rhs]

    def rpe(self, p):
        pressure = p[0]
        self.get_cpbw()
        rhs = -1.5 * self.V ** 2.0 + (self.cpbw - pressure)
        if self.viscosity:
            rhs -= 4.0 * self.Re_inv * self.V / self.R
        rhs /= self.R
        return [self.V, rhs]

    def lin(self, p):
        pressure = p[0]
        Cp = (pressure - 1.0) / 1.0
        rhs = -1.0 * Cp
        rhs -= self.R * 3.0 * self.gamma * self.Ca / (self.R0 ** 2.0)
        if self.viscosity:
            rhs -= self.V * 4.0 * self.Re_inv / (self.R0 ** 2.0)
        if self.tension:
            rhs -= self.V * 2.0 * (3.0 * self.gamma - 1.0) / (self.Web * self.R0 ** 3.0)
        return [self.V, rhs]

    def rhs(self, p):
        self.update_state()
        if self.model == "RPE":
            rhs = self.rpe(p)
        elif self.model == "KM":
            rhs = self.km(p)
        elif self.model == "Linear":
            rhs = self.lin(p)
        else:
            raise NotImplementedError
        return rhs

    def update_state(self):
        if self.model == "RPE" or self.model == "KM" or self.model == "Linear":
            self.R = self.state[0]
            self.V = self.state[1]
        else:
            raise NotImplementedError

    def wrap(self, t, y):
        self.R = y[0]
        self.V = y[1]
        if self.model == "RPE":
            return np.array(self.rpe(self.p(t)))
        elif self.model == "KM":
            return np.array(self.km(self.p(t)))
        elif self.model == "Linear":
            return np.array(self.lin(self.p(t)))
        else:
            raise NotImplementedError

    def solve(self, T=0, Ro=1.0, Vo=0.0, p=1.0, ts=None):
        self.p = p
        y0 = np.array([Ro, Vo])
        if ts is None:
            ret = sp.solve_ivp(self.wrap, (0.0, T), y0, method="LSODA", rtol=1e-3)
        else:
            ret = sp.solve_ivp(
                self.wrap, (0.0, T), y0, method="LSODA", rtol=1e-3, t_eval=ts
            )

        return ret
