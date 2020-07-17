import math
import numpy as np
import matplotlib.pyplot as plt
from Neuron import Neuron as Neuron_

class LIF_RK4(Neuron_):
    def __init__(self):
        # setting parameters with the default value
        self.R = 5                                  # this values are "random"
        self.C = 3
        self.tao = self.R * self.C
        self.uR = -40
        self.thrs = -20

    def fu(self, u, I):
        return (-u + self.R * I) / self.tao
        
    def stimulation(self, tmax, I, dt):
        steps = math.ceil(tmax / dt)
        u = np.zeros(steps, dtype=float)
        u[0] = -65.0
        spike_time = []
        time = 0
        self.do_rk4(I, dt, spike_time, steps, time, u)
        print(spike_time)
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

    def plot(self):
        vTime = np.arange(0, self.tmax, self.dt, dtype=None)
        plt.plot(vTime, self.u, color='b')
        plt.title("LIF neuron")
        plt.xlabel("Time [ms]")
        plt.ylabel("Voltage [mv]")
        plt.show()


# Let the user choose the parameters
def changeParameters(neuron): 
    while(1): 
        print("--------------------CHOOSE PARAMETER--------------------")
        print()
        print("Would you to change any constant?")
        print()
        print("{:47}".format("R [Resistence]  (actual = %.2f)" %neuron.R) + "[1]")
        print("{:47}".format("C [Capacitor]   (actual = %.2f mV)" %neuron.C) + "[2]")
        #print("{:47}".format("i [Current]     (actual = %.2f mV)" %neuron.I) + "[3]")
        print("{:47}".format("u_r [U after spike] (actual = %.2f mV)" %neuron.uR) + "[4]")
        print("{:47}".format("thrs [Threshold value] (actual = %.2f mV)" %neuron.thrs) + "[5]")
        print()
        print("{:47}".format("Show Graph") + "[6]")
        print()
        option = float(input("Option: ")) 
        if option < 1 or option > 6: 
            print("Invalid option!")
        elif option == 6:
            break 
        else: 
            value = float(input("Type the value: "))
            
            if option == 1: 
                neuron.R = value 
            elif option == 2: 
                neuron.C = value  
            elif option == 3: 
                neuron.I = value 
            elif option == 4: 
                neuron.uR = value 
            elif option == 5: 
                neuron.thrs = value 

            print()

def plot(u, tmax, dt):
    vTime = np.arange(0, tmax, dt, dtype=None)
    plt.plot(vTime, u, color='b')
    plt.title("LIF neuron")
    plt.xlabel("Time [ms]")
    plt.ylabel("Voltage [mv]")
    plt.show()


if __name__ == '__main__':
    n = LIF_RK4()
    #changeParameters(n)
    I = np.concatenate((10*np.ones(math.ceil(100 / 0.5 / 2)), 0*np.ones(math.ceil(100 / 0.5 / 2))))
    plot(n.stimulation(100, I, 0.5), 100, 0.5)
