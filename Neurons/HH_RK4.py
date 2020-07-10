
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


# Constants 
Cm      =   1.0         # uF/cm^2 
VNa     =   50          # mV
VK      =   -77         # mV
Vl      =   -54.4       # mV
gNa     =   120         # ms/cm^2
gK      =   36          # ms/cm^2 
gl      =   0.3         # ms/cm^2
restV   =   -65         # mV

# Init channels 
n = inf_n(restV)
m = inf_m(restV)
h = inf_h(restV) 


class HH_Neuron(Neuron_):
    def __init__(self): 
        self.n = n
        self.m = m 
        self.h = h 
        
    def stimulation(self, tmax, I, dt):
        steps = math.ceil(time/dt)
        v = np.zeros(steps)
        v[0] = restV 

        for t in range(steps-1): 
            # Conductances 
            NaCond = self.m**3*self.h*gNa 
            KCond  = self.n**4*gK 
            gCond  = gl 

            # Corrent 
            INa = NaCond*(v[t]-VNa) 
            IK  = KCond *(v[t]-VK)
            Il  = gCond *(v[t]-Vl)

            # Update membrane voltage 
            dv1 = (I[t] - INa - IK - Il)/Cm * dt 
            dv2 = (I[t] - INa - IK - Il + dv1*0.5)/Cm * dt 
            dv3 = (I[t] - INa - IK - Il + dv2*0.5)/Cm * dt 
            dv4 = (I[t] - INa - IK - Il + dv3)/Cm * dt 
            dv  = 1/6*(dv1 + dv2*2 + dv3*2 + dv4) 
            v[t+1] = v[t] + dv

            # Update Channels
            # Update Channel n 
            dn1 = (alpha_n(v[t])*(1-self.n)-beta_n(v[t])*self.n)*dt
            dn2 = (alpha_n(v[t])*(1-self.n + dn1*0.5)-beta_n(v[t])*(self.n+dn1*0.5))*dt
            dn3 = (alpha_n(v[t])*(1-self.n + dn2*0.5)-beta_n(v[t])*(self.n+dn2*0.5))*dt
            dn4 = (alpha_n(v[t])*(1-self.n + dn3)-beta_n(v[t])*(self.n+dn3))*dt
            dn = 1/6*(dn1 + dn2*2 + dn3*2 + dn4) 
            self.n += dn 

            # Update Channel m 
            dm1 = (alpha_m(v[t])*(1-self.m)-beta_m(v[t])*self.m)*dt
            dm2 = (alpha_m(v[t])*(1-self.m + dm1*0.5)-beta_m(v[t])*(self.m + dm1*0.5))*dt
            dm3 = (alpha_m(v[t])*(1-self.m + dm2*0.5)-beta_m(v[t])*(self.m + dm2*0.5))*dt
            dm4 = (alpha_m(v[t])*(1-self.m + dm3)-beta_m(v[t])*(self.m + dm3))*dt
            dm = 1/6*(dm1 + dm2*2 + dm3*2 + dm4) 
            self.m += dm

            # Update Channel h 
            dh1 = (alpha_h(v[t])*(1-self.h)-beta_h(v[t])*self.h)*dt
            dh2 = (alpha_h(v[t])*(1-self.h + dh1*0.5)-beta_h(v[t])*(self.h + dh1*0.5))*dt
            dh3 = (alpha_h(v[t])*(1-self.h + dh2*0.5)-beta_h(v[t])*(self.h + dh2*0.5))*dt
            dh4 = (alpha_h(v[t])*(1-self.h + dh3)-beta_h(v[t])*(self.h + dh3))*dt
            dh = 1/6*(dh1 + dh2*2 + dh3*2 + dh4)
            self.h += dh 

        return v 


if __name__ == '__main__': 

    # Init neuron 
    neuron = HH_Neuron()

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
    plt.ylabel("Time [mV]")
    plt.show() 
