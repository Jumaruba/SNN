from Neuron import Neuron as Neuron_

import math
import numpy as np
import matplotlib.pyplot as plt


class Izh_Neuron(Neuron_):
    def __init__(self):
        # setting parameters with the default value
        super().__init__()
        self.a = .1             # fast spiking
        self.b = .2
        self.c = -65
        self.d = 8

    def stimulation(self, tmax, I, dt):
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
                dv = (0.04 * v[t] * v[t] + 5 * v[t] + 140 - u[t] + I[t]) * dt
                du = (self.a * (self.b * v[t] - u[t])) * dt
                u[t + 1] = u[t] + du
                v[t + 1] = v[t] + dv
        return v


# Let the user choose the parameters
def changeParameters(neuron): 
    while(1): 
        print("--------------------CHOOSE PARAMETER--------------------")
        print()
        print("Would you to change any constant?")
        print()
        print("{:47}".format("a  (actual = %.2f)" %neuron.a) + "[1]")
        print("{:47}".format("b  (actual = %.2f mV)" %neuron.b) + "[2]")
        print("{:47}".format("c  (actual = %.2f mV)" %neuron.c) + "[3]")
        print("{:47}".format("d  (actual = %.2f mV)" %neuron.d) + "[4]")
        print()
        print("{:47}".format("Show Graph") + "[5]")
        print()
        option = float(input("Option: ")) 
        if option < 1 or option > 5: 
            print("Invalid option!")
        elif option == 5:
            break 
        else: 
            value = float(input("Type the value: "))
            
            if option == 1: 
                neuron.a = value 
            elif option == 2: 
                neuron.b = value  
            elif option == 3: 
                neuron.c = value 
            elif option == 4: 
                neuron.d = value 
 

            print("The new value is", value) 
            print()

def plot(time, dt, v, I):
    vTime = np.arange(0, time, dt, dtype=None)
    plt.plot(vTime, v, color='b', label = "potential")
    plt.plot(vTime, I, color='r', label = "current")
    plt.title("Single neuron stimulation")
    plt.xlabel("Time [ms]")
    plt.ylabel("Voltage [mv]")
    plt.savefig("Fast spiking")
    plt.show()


if __name__ == '__main__':
    n = Izh_Neuron()
    changeParameters(n)

    time = 500
    dt = 0.01
    steps = math.ceil(time / dt)
    I = [0 if 200 / dt <= i <= 300 / dt else 10 for i in range(steps)]
    v = n.stimulation(time, I, dt)
    plot(time, dt, v, I)
