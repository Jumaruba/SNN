from Neurons.Neuron import Neuron

import math as m
import numpy as np
import matplotlib.pyplot as plt


class LIFS(Neuron):
    def __init__(self):

        # setting constants
        super().__init__()
        self.el = -70
        self.es = 0         # 0 excitatory; -80 inhibitory
        self.tauM = 20
        self.rmie = 25
        self.rmgs = 0.05
        self.uR = -80
        self.thrs = -54
        self.initial_u = -100
        self.P = 0
        self.u = np.zeros(1, dtype=float)

    def stimulation(self, tmax, I, dt):
        steps = m.ceil(tmax / dt)

        u = np.zeros(steps, dtype=float)
        u[0] = self.initial_u
        spike_time = []
        time = 0
        if self.method == "el":
            self.do_el(dt, steps, u)
        else:
            self.do_rk4(dt, spike_time, steps, time, u)

        return u

    def reset(self, steps=1):
        self.u = np.zeros(steps, dtype=float)
        self.u[0] = self.initial_u

    def nextIteration(self, t, dt):
        if self.u[t - 1] >= self.thrs:
            self.u[t] = self.uR
            return True
        else:
            du = (self.el - self.u[t - 1] - self.rmgs * self.P * (self.u[t - 1] - self.es) + self.rmie) * dt / self.tauM
            self.u[t] = self.u[t - 1] + du
            return False


    """
    for t in range(m.ceil(1 / dt)):
        if u[t - 1] >= self.thrs:
            u[t] = self.uR
        else:
            du = (self.el - u[t - 1] - self.rmgs * self.P * (u[t - 1] - self.es) + self.rmie) * dt / self.tauM
            u[t] = u[t - 1] + du
    """

    def do_rk4(self, dt, spike_time, steps, time, u):
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

    def do_el(self, dt, steps, u):
        for t in range(1, steps):
            self.nextIteration(u, dt, t)

    def create_copy(self):
        n = LIFS()
        n.set_constants(self.R, self.C, self.I, self.uR, self.thrs, self.initial_u)
        return n

    def update_P(self, P):
        self.P += P

    def get_u(self):
        return self.u

    def set_constants(self, r=None, c=None, i=None, u_r=None, thrs=None, initial_u=None):
        """
            This function changes the constants associated with the neuron.
            r -> Resistence;    c -> Capacitor;     i -> Intensity of current;
            u_r -> value of U after spike;      thrs -> value of threshold;     initial_u -> initial value of U
        """
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

def plot(u, tmax, dt):
    vTime = np.arange(0, tmax, dt, dtype=None)
    plt.plot(vTime, u, color='b')
    plt.title("LIF neuron")
    plt.xlabel("Time [ms]")
    plt.ylabel("Voltage [mv]")
    plt.show()


if __name__ == '__main__':
    n = LIFS()
    # changeParameters(n)
    # I = np.concatenate( (10*np.ones(math.ceil(100 / 0.5 / 2)), 0*np.ones(math.ceil(100 / 0.5 / 2))))
    I = 5 * np.ones(m.ceil(100 / 0.5))
    plot(n.stimulation(100, I, 0.5), 100, 0.5)
