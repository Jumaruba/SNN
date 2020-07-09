import numpy as np 
import math 
import matplotlib.pyplot as plt

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

class HH_Neuron:
    def __init__(self): 
        self.n = n
        self.m = m 
        self.h = h 
        
    def stimulate(self, I, time, dt): 
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
            dv1 = (I[t] - INa - IK - Il)*dt/Cm
            dv2 = ((I[t] - INa - IK - Il)/Cm + dv1*0.5)*dt 
            dv3 = ((I[t] - INa - IK - Il)/Cm + dv2*0.5)*dt 
            dv4 = ((I[t] - INa - IK - Il)/Cm + dv3)*dt 
            dv = dv1/6 + dv2/3 + dv3/3 + dv4/6  
            v[t+1] = v[t] + dv

            # Update Channels ------------------------

            # Update n Channel
            dn1 = dt*(alpha_n(v[t+1])*(1-n)-beta_n(v[t+1])*n)
            dn2 = dt*(alpha_n(v[t+1])*(1-n + dn1*0.5)-beta_n(v[t+1])*(n+dn1*0.5))
            dn3 = dt*(alpha_n(v[t+1])*(1-n + dn2*0.5)-beta_n(v[t+1])*(n+dn2*0.5))
            dn4 = dt*(alpha_n(v[t+1])*(1-n + dn3)-beta_n(v[t+1])*(n+dn3))
            dn  = dn1/6 + dn2/3 + dn3/3 + dn4/6 
            self.n += dn 

            # Update m Channel 
            dm1 = dt*(alpha_m(v[t+1])*(1-m)-beta_m(v[t+1])*m)
            dm2 = dt*(alpha_m(v[t+1])*(1-m + dm1*0.5)-beta_m(v[t+1])*(m+dm1*0.5))
            dm3 = dt*(alpha_m(v[t+1])*(1-m + dm2*0.5)-beta_m(v[t+1])*(m+dm2*0.5))
            dm4 = dt*(alpha_m(v[t+1])*(1-m + dm3)-beta_m(v[t+1])*(m+dm3))
            dm  = dm1/6 + dm2/3 + dm3/3 + dm4/6 
            self.m += dm

            # Update h Channel 
            dh1 = dt*(alpha_h(v[t+1])*(1-h)-beta_h(v[t+1])*h)
            dh2 = dt*(alpha_h(v[t+1])*(1-h + dh1*0.5)-beta_h(v[t+1])*(h+dh1*0.5))
            dh3 = dt*(alpha_h(v[t+1])*(1-h + dh2*0.5)-beta_h(v[t+1])*(h+dh2*0.5))
            dh4 = dt*(alpha_h(v[t+1])*(1-h + dh3)-beta_h(v[t+1])*(h+dh3))
            dh  = dh1/6 + dh2/3 + dh3/3 + dh4/6 
            self.h += dh

            print(self.h, self.m, self.n, v[t])
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
    v = neuron.stimulate(I, time, dt)

    # Plot
    vTime = np.arange(0,time,dt, dtype = float)
    plt.plot(vTime, v, color = 'b', label="potential")
    plt.plot(vTime, I, color = 'r', label="current")
    plt.title('Hodgkin Huxel Model')
    plt.legend(loc=(0.01, 0.44))
    plt.xlabel("Time [ms]")
    plt.ylabel("Voltage [mV]")
    plt.savefig("HH.png")
    plt.show() 

        
