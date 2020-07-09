import Neuron

import math
import numpy as np
import matplotlib.pyplot as plt


class LIF(Neuron):
    def __init__(self):
        self.tmax = 100
        self.dt = 0.5
        self.steps = math.ceil(self.tmax / self.dt)

        # setting constants
        self.R = 5  # this values are "random"
        self.C = 3
        self.I = 10  # for now, it will be constant
        self.uR = -40
        self.thrs = -20

        self.u = np.zeros(self.steps, dtype=float)
        self.u[0] = -65.0
        self.time = 0

    def stimulation(self):
        for t in range(1, self.steps):
            if self.u[t - 1] >= self.thrs:
                self.u[t] = self.uR
            else:
                du = (-self.u[t - 1] + self.R * self.I) * self.dt / (self.R * self.C)
                self.time += self.dt
                self.u[t] = self.u[t - 1] + du

    def plot(self):
        vTime = np.arange(0, self.tmax, self.dt, dtype=None)
        plt.plot(vTime, self.u, color='b')
        plt.title("LIF neuron")
        plt.xlabel("Time [ms]")
        plt.ylabel("Voltage [mv]")
        plt.show()

    """
    This function changes the constants associated with the neuron.
    r -> Resistence;    c -> Capacitor;     i -> Intensity of current;      
    u_r -> value of U after spike;      thrs -> value of threshold;     initial_u -> initial value of U
    """
    def set_constants(self, r="", c="", i="", u_r="", thrs="", initial_u=""):
        if r != "":
            self.R = r
        if c != "":
            self.C = c
        if i != "":
            self.I = i
        if u_r != "":
            self.uR = u_r
        if thrs != "":
            self.thrs = thrs
        if initial_u != "":
            self.u[0] = initial_u

    def set_time_interval(self, tmax="", dt=""):
        if tmax != "":
            self.tmax = tmax
        if dt != "":
            self.dt = dt


if __name__ == '__main__':
    n = LIF()
    n.stimulation()
    n.plot()
