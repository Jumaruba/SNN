
import numpy as np
import math 
import matplotlib.pyplot as plt
from Neuron import Neuron as Neuron_

# constants http://www.math.pitt.edu/~bdoiron/assets/ermentrout-and-terman-ch-1.pdf
# paper https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1392413/pdf/jphysiol01442-0106.pdf

# Gate n 
alpha_n = lambda V: 0.01 * (V + 55) / (1 - np.exp(-(V + 55) * 0.1) )
beta_n  = lambda V: 0.125 * np.exp(-(V + 65)/80)
inf_n   = lambda V: alpha_n(V)/(alpha_n(V) + beta_n(V))

# Gate m 
alpha_m = lambda V: 0.1*(V + 40)/(1- np.exp(-(V + 40)*0.1))
beta_m  = lambda V: 4*np.exp(-(V+65)/18)
inf_m   = lambda V: alpha_m(V)/(alpha_m(V) + beta_m(V)) 

# Gate h 
alpha_h = lambda V: 0.07*math.exp(-(V+65)/20)
beta_h  = lambda V: 1/(math.exp(-(V + 35) * 0.1) + 1)
inf_h   = lambda V: alpha_h(V)/(alpha_h(V) + beta_h(V))



class HH_Neuron(Neuron_):
    #Constans 

    Cm      =   1.0         # uF/cm^2 
    VNa     =   50          # mV
    VK      =   -77         # mV
    Vl      =   -54.4       # mV
    gNa     =   120         # ms/cm^2
    gK      =   36          # ms/cm^2 
    gl      =   0.3         # ms/cm^2
    restV   =   -65         # mV

    def __init__(self): 
        # Init channels
        self.n = inf_n(self.restV)
        self.m = inf_m(self.restV) 
        self.h = inf_h(self.restV)
        
        
    def getDv(self, iteration, INa, IK, Il):
        dv1 = (I[iteration] - INa - IK - Il)/self.Cm * dt 
        dv2 = (I[iteration] - INa - IK - Il + dv1*0.5)/self.Cm * dt 
        dv3 = (I[iteration] - INa - IK - Il + dv2*0.5)/self.Cm * dt 
        dv4 = (I[iteration] - INa - IK - Il + dv3)/self.Cm * dt 
        dv  = 1/6*(dv1 + dv2*2 + dv3*2 + dv4) 
        
        return dv 

    def getDn(self, iteration, dt, v): 
        dn1 = (alpha_n(v[iteration])*(1-self.n)-beta_n(v[iteration])*self.n)*dt
        dn2 = (alpha_n(v[iteration])*(1-self.n + dn1*0.5)-beta_n(v[iteration])*(self.n+dn1*0.5))*dt
        dn3 = (alpha_n(v[iteration])*(1-self.n + dn2*0.5)-beta_n(v[iteration])*(self.n+dn2*0.5))*dt
        dn4 = (alpha_n(v[iteration])*(1-self.n + dn3)-beta_n(v[iteration])*(self.n+dn3))*dt
        dn = 1/6*(dn1 + dn2*2 + dn3*2 + dn4) 
        
        return dn 

    def getDm(self, iteration, dt, v):
        dm1 = (alpha_m(v[iteration])*(1-self.m)-beta_m(v[iteration])*self.m)*dt
        dm2 = (alpha_m(v[iteration])*(1-self.m + dm1*0.5)-beta_m(v[iteration])*(self.m + dm1*0.5))*dt
        dm3 = (alpha_m(v[iteration])*(1-self.m + dm2*0.5)-beta_m(v[iteration])*(self.m + dm2*0.5))*dt
        dm4 = (alpha_m(v[iteration])*(1-self.m + dm3)-beta_m(v[iteration])*(self.m + dm3))*dt
        dm = 1/6*(dm1 + dm2*2 + dm3*2 + dm4) 

        return dm 

    def getDh(self, iteration, dt, v):
        dh1 = (alpha_h(v[iteration])*(1-self.h)-beta_h(v[iteration])*self.h)*dt
        dh2 = (alpha_h(v[iteration])*(1-self.h + dh1*0.5)-beta_h(v[iteration])*(self.h + dh1*0.5))*dt
        dh3 = (alpha_h(v[iteration])*(1-self.h + dh2*0.5)-beta_h(v[iteration])*(self.h + dh2*0.5))*dt
        dh4 = (alpha_h(v[iteration])*(1-self.h + dh3)-beta_h(v[iteration])*(self.h + dh3))*dt
        dh = 1/6*(dh1 + dh2*2 + dh3*2 + dh4)

        return dh 

    def stimulation(self, tmax, I, dt):
        steps = math.ceil(time/dt)
        v = np.zeros(steps)
        v[0] = self.restV 
        

        for t in range(steps-1): 
            # Conductances 
            NaCond = self.m**3*self.h*self.gNa 
            KCond  = self.n**4*self.gK 
            gCond  = self.gl 

            # Current 
            INa = NaCond*(v[t]-self.VNa) 
            IK  = KCond *(v[t]-self.VK)
            Il  = gCond *(v[t]-self.Vl)

            # Update membrane voltage 
            v[t+1] = v[t] + self.getDv(t, INa, IK, Il)

            # Update Channels            
            self.n += self.getDn(t, dt, v)
            self.m += self.getDm(t, dt, v)
            self.h += self.getDh(t, dt, v)

        return v 


# Let the user choose the parameters
def changeParameters(neuron): 
    while(1): 
        print("--------------------CHOOSE PARAMETER--------------------")
        print()
        print("Would you to change any constant?")
        print()
        print("{:47}".format("Cm  (actual = %.2f uF/cm^2)" %neuron.Cm) + "[1]")
        print("{:47}".format("VNa (actual = %.2f mV)" %neuron.VNa) + "[2]")
        print("{:47}".format("VK  (actual = %.2f mV)" %neuron.VK) + "[3]")
        print("{:47}".format("Vl  (actual = %.2f mV)" %neuron.Vl) + "[4]")
        print("{:47}".format("gNa (actual = %.2f mV)" %neuron.gNa) + "[5]")
        print("{:47}".format("gK  (actual = %.2f ms/cm^2)" %neuron.gK) + "[6]")
        print("{:47}".format("gl  (actual = %.2f ms/cm^2)" %neuron.gl) + "[7]")
        print("{:47}".format("restV (actual = %.2f mV)" %neuron.restV) + "[8]")
        print()
        print("{:47}".format("Show Graph") + "[9]")
        print()
        option = int(input("Option: ")) 
        if option < 1 or option > 9: 
            print("Invalid option!")
        elif option == 9:
            break 
        else: 
            value = int(input("Type the value: "))
            
            if option == 1: 
                neuron.Cm = value 
            elif option == 2: 
                neuron.VNa = value  
            elif option == 3: 
                neuron.VK = value 
            elif option == 4: 
                neuron.Vl = value 
            elif option == 5: 
                neuron.gNa = value 
            elif option == 6: 
                neuron.gK = value 
            elif option == 7: 
                neuron.gl = value 
            elif option == 8: 
                neuron.restV = value  

            print("The new value is", value) 
            print()


if __name__ == '__main__': 

    # Init neuron 
    neuron = HH_Neuron()
    changeParameters(neuron)

    # Select time of iteraction 
    time = 100      # ms 
    dt   = 0.01      # ms

    # Select time of stimulation 
    steps = math.ceil(time / dt)
    I = np.zeros(steps)
    I[1*math.ceil(steps / 4):3*math.ceil(steps / 4)] = 10

    # Run 
    v = neuron.stimulation(time, I, dt)

    # Plot
    vTime = np.arange(0, time, dt, dtype=float)
    plt.plot(vTime, v, color='r')
    plt.plot(vTime, I, color='b')
    plt.title('Hodgkin Huxel Model')
    plt.xlabel("Time [ms]")
    plt.ylabel("Voltage [mV]")
    plt.show() 
