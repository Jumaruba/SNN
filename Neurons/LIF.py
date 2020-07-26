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
        self.uR = -40
        self.thrs = -20
        self.initial_u = -65

    def stimulation(self, tmax, I, dt):
        steps = math.ceil(tmax / dt)

        u = np.zeros(steps, dtype=float)
        u[0] = self.initial_u
        spike_time = []
        time = 0
        if self.method == "el":
            self.do_el(I, dt, spike_time, steps, time, u)
        else:
            self.do_rk4(I, dt, spike_time, steps, time, u)

        return u

    def do_rk4(self, I, dt, spike_time, steps, time, u):
        for t in range(1, steps):
            if u[t - 1] >= self.thrs:
                u[t] = self.uR
                spike_time.append(time)
            else:
                du1 = self.fu(u[t - 1], I[t - 1]) * dt
                du2 = self.fu(u[t - 1] + du1 * 0.5, I[t - 1]) * dt
                du3 = self.fu(u[t - 1] + du2 * 0.5, I[t - 1]) * dt
                du4 = self.fu(u[t - 1] + du3, I[t - 1]) * dt
                du = 1 / 6 * (du1 + du2 * 2 + du3 * 2 + du4)
                u[t] = u[t - 1] + du
                time += dt

    def fu(self, u, I):
        return (-u + self.R * I) / (self.R * self.C)

    def do_el(self, I, dt, spike_time, steps, time, u):
        for t in range(1, steps):
            if u[t - 1] >= self.thrs:
                u[t] = self.uR
                spike_time.append(time)
            else:
                du = (-u[t - 1] + self.R * I[t]) * dt / (self.R * self.C)
                u[t] = u[t - 1] + du

    def create_copy(self):
        n = LIF()
        n.set_constants(self.R, self.C, self.I, self.uR, self.thrs, self.initial_u)
        return n

        
    """
    This function changes the constants associated with the neuron.
    r -> Resistence;    c -> Capacitor;     i -> Intensity of current;      
    u_r -> value of U after spike;      thrs -> value of threshold;     initial_u -> initial value of U
    """
    def set_constants(self, r=None, c=None, i=None, u_r=None, thrs=None, initial_u=None):
        if r:
            self.R = r
        if c:
            self.C = c
        if i:
            self.I = i
        if u_r:
            self.uR = u_r
        if thrs:
            self.thrs = thrs
        if initial_u:
            self.initial_u = initial_u

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
    n.change_method("rk")
    I = 5*np.ones(math.ceil(100 / 0.5))
    plot(n.stimulation(100, I, 0.5), 100, 0.5)
