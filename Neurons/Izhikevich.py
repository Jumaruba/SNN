import math
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    def __init__(self):
        self.tmax = 1000
        self.dt = 0.5
        self.steps = math.ceil(self.tmax / self.dt)
        self.i = 10

        # setting parameters with the default value
        self.a = .02
        self.b = .2
        self.c = -65
        self.d = 8

        # input
        self.v = np.zeros(self.steps, dtype=float)
        self.u = np.zeros(self.steps, dtype=float)
        self.v[0] = -65  # resting potential
        self.u[0] = -14

    def stimulation(self):
        for t in range(self.steps - 1):
            if self.v[t] >= -30:  # spike
                self.v[t + 1] = self.c
                self.u[t + 1] = self.u[t] + self.d
            else:
                dv = 0.04 * self.v[t] * self.v[t] + 5 * self.v[t] + 140 - self.u[t] + self.i
                du = self.a * (self.b * self.v[t] - self.u[t])
                self.u[t + 1] = self.u[t] + du
                self.v[t + 1] = self.v[t] + dv

    def plot(self):
        vTime = np.arange(0, 1000, 0.5, dtype=None)
        plt.plot(vTime, self.v, color='b')
        plt.plot(vTime, self.u, linestyle='dashdot')
        plt.title("Single neuron stimulation")
        plt.xlabel("Time [ms]")
        plt.ylabel("Voltage [mv]")
        plt.show()


if __name__ == '__main__':
    n = Neuron()
    n.stimulation()
    n.plot()
