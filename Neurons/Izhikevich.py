import math
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    def __init__(self):
        # setting parameters with the default value
        self.a = .02
        self.b = .2
        self.c = -65
        self.d = 8

    def stimulation(self, time, I, dt):
        steps = math.ceil(time / dt)

        # input
        v = np.zeros(steps, dtype=float)
        u = np.zeros(steps, dtype=float)
        v[0] = -65  # resting potential
        u[0] = -14

        for t in range(steps - 1):
            if v[t] >= -30:  # spike
                v[t + 1] = self.c
                u[t + 1] = u[t] + self.d
            else:
                dv = (0.04 * v[t] * v[t] + 5 * v[t] + 140 - u[t] + I[t])
                du = (self.a * (self.b * v[t] - u[t]))
                u[t + 1] = u[t] + du
                v[t + 1] = v[t] + dv
        return v


def plot(time, dt, v, I):
    vTime = np.arange(0, time, dt, dtype=None)
    plt.plot(vTime, v, color='b')
    plt.plot(vTime, I, color='r')
    plt.title("Single neuron stimulation")
    plt.xlabel("Time [ms]")
    plt.ylabel("Voltage [mv]")
    plt.show()


if __name__ == '__main__':
    n = Neuron()

    time = 100
    dt = 0.05
    steps = math.ceil(time / dt)
    I = [10 if (10 / 0.05 <= i <= 30 / 0.05 or 60 / 0.05 <= i <= 80 / 0.05) else 0 for i in range(steps)]
    v = n.stimulation(time, I, dt)
    plot(time, dt, v, I)
