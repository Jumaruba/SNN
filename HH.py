import math
import numpy as np
import matplotlib.pyplot as plt


def alpha_n(v):
    return 0.01 * (v + 10) / (math.exp((v + 10) / 10) - 1)


def beta_n(v):
    return 0.125 * math.exp(v / 80)


def alpha_m(v):
    return 0.1 * (v + 25) / (math.exp((v + 25) / 10) - 1)


def beta_m(v):
    return 4 * math.exp(v / 18)


def alpha_h(v):
    return 0.07 * math.exp(v / 20)


def beta_h(v):
    return 1 / (math.exp((v + 30) / 10) + 1)


class Neuron:
    def __init__(self):
        self.CM = 1  # membrane capacitance
        self.VNa = -115
        self.VK = 12
        self.VL = -10613
        self.GNa = 120  # sodium conductance
        self.GK = 36  # potassium conductance
        self.GL = 0.3  # leak conductance
        self.n = 0
        self.m = 0
        self.h = 0

        self.time = 1000
        self.dt = 0.5  # in ms
        self.steps = math.ceil(1000 / 0.5)

        self.v = np.zeros(self.steps, dtype=float)

    def step(self, t):
        self.n += alpha_n(self.v[t]) * (1 - self.n) - beta_n(self.v[t]) * self.n
        self.m += alpha_m(self.v[t]) * (1 - self.m) - beta_m(self.v[t]) * self.m
        self.h += alpha_h(self.v[t]) * (1 - self.h) - beta_h(self.v[t]) * self.h

        dv = -1 / self.CM * (
                self.GK * self.n ** 4 * (self.v[t] - self.VK) + self.GNa * self.m ** 3 * self.h * (
                self.v[t] - self.VNa) + self.GL * (
                        self.v[t] - self.VL))

        self.v[t + 1] += dv

    def stimulation(self):
        for t in range(self.steps - 1):
            self.step(t)

    def plot(self):
        vTime = np.arange(0, 1000, 0.5, dtype=None)
        plt.plot(vTime, self.v, color='b')
        plt.title("Single neuron stimulation")
        plt.xlabel("Time [ms]")
        plt.ylabel("Voltage [mv]")
        plt.show()


if __name__ == '__main__':
    neuron = Neuron()
    neuron.stimulation()
    neuron.plot()
