from Neuron import Neuron as Neuron_

import math
import numpy as np
import matplotlib.pyplot as plt


"""
Constants indication:
r -> Resistence;   
c -> Capacitor;     
i -> Intensity of current;      
u_r -> value of U after spike;      
thrs -> value of threshold;     
initial_u -> initial value of U
"""
class LIF(Neuron_):
    def __init__(self):

        # setting constants
        super().__init__()
        self.R = 5  # this values are "random"
        self.C = 3
        self.I = 10  # for now, it will be constant
        self.uR = -40
        self.thrs = -20

    def stimulation(self, tmax, I, dt):
        steps = math.ceil(tmax / dt)

        u = np.zeros(steps, dtype=float)
        u[0] = -65.0

        for t in range(1, steps):
            if u[t - 1] >= self.thrs:
                u[t] = self.uR
            else:
                du = (-u[t - 1] + self.R * I) * dt / (self.R * self.C)
                u[t] = u[t - 1] + du
        return u


# Let the user choose the parameters
def changeParameters(neuron): 
    while(1): 
        print("--------------------CHOOSE PARAMETER--------------------")
        print()
        print("Would you to change any constant?")
        print()
        print("{:47}".format("R [Resistence]  (actual = %.2f)" %neuron.R) + "[1]")
        print("{:47}".format("C [Capacitor]   (actual = %.2f mV)" %neuron.C) + "[2]")
        print("{:47}".format("i [Current]     (actual = %.2f mV)" %neuron.I) + "[3]")
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
    n = LIF()
    changeParameters(n)
    plot(n.stimulation(100, 10, 0.5), 100, 0.5)
