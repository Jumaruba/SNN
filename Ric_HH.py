import math
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    def __init__(self):
        self.tmax = 1000
        self.dt = 0.5
        self.steps = math.ceil(self.tmax / self.dt)
        self.i = 10

        # setting constants
        self.Cm = 1.0
        self.VNa = -115
        self.Vk = 12
        self.Vl = -10.613
        self.GNa = 120
        self.Gk = 36
        self.Gl = 0.3

        #auxiliary functions
        self.alphaN = lambda v : 0.01 * ( v + 10 ) / ( math.exp( ( v + 10) / 10 ) - 1 )
        self.alphaM = lambda v : 0.1 * ( v + 25 ) / ( math.exp( ( v + 25) / 10 ) - 1 )
        self.alphaH = lambda v : 0.07 * math.exp(v / 20)
        self.betaN = lambda v : 0.125 * math.exp(v / 80)
        self.betaM = lambda v : 4 * math.exp(v / 18)
        self.betaH = lambda v : 1 / (math.exp((v + 30)/ 10) + 1)
        

        self.v = np.zeros(self.steps, dtype=float)
        self.v[0] = -65  # resting potential

    def stimulation(self):
        
        n,m,h = 0,0,0
        for t in range(1, self.steps):
            

            dV = (self.i - self.Gk * n**4 * (self.v[t-1] - self.Vk) - self.GNa * m**3 * h * (self.v[t - 1] - self.VNa) + self.Gl * (self.v[t-1] - self.Vl)) * self.dt / self.Cm
            self.v[t] = self.v[t-1] + dV
            
            dn = (self.alphaN(self.v[t]) * (1 - n) - self.betaN(self.v[t]) * n)*self.dt
            dm = (self.alphaM(self.v[t]) * (1 - m) - self.betaM(self.v[t]) * m)*self.dt
            dh = (self.alphaH(self.v[t]) * (1 - h) - self.betaH(self.v[t]) * h)*self.dt

            n += dn
            m += dm
            h += dh

    def plot(self):
        vTime = np.arange(0, 1000, 0.5, dtype=None)
        plt.plot(vTime, self.v, color='b')
        plt.title("HH")
        plt.xlabel("Time [ms]")
        plt.ylabel("Voltage [mv]")
        plt.show()


if __name__ == '__main__':
    n = Neuron()
    n.stimulation()
    n.plot()
