import numpy as np


class waveforms:
    def __init__(self, config={}):
        self.config = config
        self.parse_config()

        if self.form == "constant":
            self.p = self.p_constant
        elif self.form == "sine":
            self.p = self.p_sine
        elif self.form == "square":
            self.p = self.p_square
        else:
            NotImplementedError

    def parse_config(self):
        if "form" in self.config:
            self.form = self.config["form"]
        else:
            self.form = "constant"

        if "amplitude" in self.config:
            self.amplitude = self.config["amplitude"]
        else:
            raise Exception("No amplitude")

        if "period" in self.config:
            self.period = self.config["period"]
        else:
            if self.form == "sine" or self.form == "square":
                raise Exception("Need period")
            else:
                self.period = 1.0

        if "cycles" in self.config:
            self.cycles = self.config["cycles"]
        else:
            self.cycles = 1.0

        if "ambient" in self.config:
            self.ambient = self.config["ambient"]
        else:
            self.ambient = 1.0

    def p_constant(self, t):
        return [self.amplitude, 0.0]

    def p_sine(self, t):
        if t <= self.period * self.cycles:
            f = 2.0 * np.pi / self.period
            return [self.ambient + self.amplitude * np.sin(f * t), self.amplitude * np.cos(f * t) * f]
        else:
            return [self.ambient, 0.0]

    def p_square(self, t):
        if t <= self.period:
            return [self.amplitude, 0.0]
        else:
            return [self.ambient, 0.0]
