import math
import numpy as np
import matplotlib.pyplot as plt
from Neuron import Neuron as Neuron_

def f_v(i, u, v): 
    return (0.04 * v * v + 5 * v + 140 - u + i)

def f_u(a,b,v,u): 
    return a * (b * v - u)

class Neuron(Neuron_):
    def __init__(self):
        # setting parameters with the default value
        self.a = .1             # fast spiking
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
                dv1 = f_v(I[t], u[t], v[t])*dt 
                dv2 = f_v(I[t], u[t], v[t] + dv1*0.5)*dt
                dv3 = f_v(I[t], u[t], v[t] + dv2*0.5)*dt 
                dv4 = f_v(I[t], u[t], v[t] + dv3)*dt 
                dv  = 1/6*(dv1 + dv2*2 + dv3*2 + dv4) 

                du1 = f_u(self.a, self.b, v[t], u[t])*dt 
                du2 = f_u(self.a, self.b, v[t], u[t] + du1*0.5)*dt
                du3 = f_u(self.a, self.b, v[t], u[t] + du2*0.5)*dt 
                du4 = f_u(self.a, self.b, v[t], u[t] + du3)*dt 
                du  = 1/6*(du1 + du2*2 + du3*2 + du4) 

                u[t + 1] = u[t] + du
                v[t + 1] = v[t] + dv
        return v


def plot(time, dt, v, I):
    vTime = np.arange(0, time, dt, dtype=None)
    plt.plot(vTime, v, color='b', label = "potential")
    plt.plot(vTime, I, color='r', label = "current")
    plt.title("Single neuron stimulation")
    plt.legend(loc=(0.01, 0.68))
    plt.xlabel("Time [ms]")
    plt.ylabel("Voltage [mv]")
    plt.show()


if __name__ == '__main__':
    n = Neuron()

    time = 500
    dt = 0.01
    steps = math.ceil(time / dt)
    I = [0 if 200 / dt <= i <= 300 / dt else 10 for i in range(steps)]
    v = n.stimulation(time, I, dt)
    plot(time, dt, v, I)
