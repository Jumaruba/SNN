import math
import numpy as np
import matplotlib.pyplot as plt
from Neuron import Neuron as Neuron_

class Neuron(Neuron_):
    def __init__(self):
        self.tmax = 100
        self.dt = 0.5
        self.steps = math.ceil(self.tmax / self.dt)

        # setting parameters with the default value
        self.R = 5                                  # this values are "random"
        self.C = 3
        self.tao = self.R * self.C
        self.I = 10                                 # for now, it will be constant
        self.uR = -40
        self.thrs = -20

        self.u = np.zeros(self.steps, dtype=float)
        self.u[0] = -65.0
        self.time = 0
    
    def fu(self, u):
        return (-u + self.R * self.I)/ self.tao
        
    def stimulation(self):
        for t in range(1, self.steps):
            if self.u[t - 1] >= self.thrs:
                self.u[t] = self.uR
            else:
                du1 = self.fu(self.u[t-1])*self.dt 
                du2 = self.fu(self.u[t-1] + du1*0.5)*self.dt 
                du3 = self.fu(self.u[t-1] + du2*0.5)*self.dt 
                du4 = self.fu(self.u[t-1] + du3)*self.dt 
                du = 1/6*(du1 + du2*2 + du3*2 + du4) 
                self.u[t] = self.u[t - 1] + du
            

    def plot(self):
        vTime = np.arange(0, self.tmax, self.dt, dtype=None)
        plt.plot(vTime, self.u, color='b')
        plt.title("LIF neuron")
        plt.xlabel("Time [ms]")
        plt.ylabel("Voltage [mv]")
        plt.show()


if __name__ == '__main__':
    n = Neuron()
    n.stimulation()
    n.plot()
