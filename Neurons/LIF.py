from Neuron import Neuron as Neuron_

import math
import numpy as np
import matplotlib.pyplot as plt


class LIF(Neuron_):
    def __init__(self):

        # setting constants
        super().__init__()
        self.R = 5  # this values are "random"
        self.C = 3
        self.I = 10  # for now, it will be constant
        self.uR = -40
        self.thrs = -20

    def stimulation(self, tmax, I, dt):
        steps = math.ceil(tmax / dt)

        u = np.zeros(steps, dtype=float)
        u[0] = -65.0

        for t in range(1, steps):
            if u[t - 1] >= self.thrs:
                u[t] = self.uR
            else:
                du = (-u[t - 1] + self.R * I) * dt / (self.R * self.C)
                u[t] = u[t - 1] + du
        return u

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


def plot(u, tmax, dt):
    vTime = np.arange(0, tmax, dt, dtype=None)
    plt.plot(vTime, u, color='b')
    plt.title("LIF neuron")
    plt.xlabel("Time [ms]")
    plt.ylabel("Voltage [mv]")
    plt.show()


if __name__ == '__main__':
    n = LIF()
    plot(n.stimulation(100, 10, 0.5), 100, 0.5)
