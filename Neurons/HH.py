import numpy as np
import math 
import matplotlib.pyplot as plt

# constants http://www.math.pitt.edu/~bdoiron/assets/ermentrout-and-terman-ch-1.pdf
# paper https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1392413/pdf/jphysiol01442-0106.pdf


class HH():
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
        self.n = self.inf_n(self.restV)
        self.m = self.inf_m(self.restV) 
        self.h = self.inf_h(self.restV)
        
    def stimulation(self, tmax, I, dt):
        steps = math.ceil(time/dt)
        v = np.zeros(steps)
        v[0] = self.restV 

        for t in range(steps-1): 
            # Conductances 
            NaCond = self.m**3*self.h*self.gNa 
            KCond  = self.n**4*self.gK 
            gCond  = self.gl 

            # Corrent 
            INa = NaCond*(v[t]-self.VNa) 
            IK  = KCond *(v[t]-self.VK)
            Il  = gCond *(v[t]-self.Vl)

            # Update membrane voltage 
            dv = (I[t] - INa - IK - Il)*dt/self.Cm 
            v[t+1] = v[t] + dv

            # Update Channels
            self.n += (self.alpha_n(v[t])*(1-self.n)-self.beta_n(v[t])*self.n)*dt
            self.m += (self.alpha_m(v[t])*(1-self.m)-self.beta_m(v[t])*self.m)*dt
            self.h += (self.alpha_h(v[t])*(1-self.h)-self.beta_h(v[t])*self.h)*dt

        return v

    # Gate n 
    def alpha_n(self, v):
        return 0.01 * (v + 55) / (1 - np.exp(-(v + 55) * 0.1) )
    def beta_n(self, v):
        return 0.125 * np.exp(-(v + 65)/80) 
    def inf_n(self, v): 
        return self.alpha_n(v)/(self.alpha_n(v) + self.beta_n(v)) 

    # Gate m 
    def alpha_m(self, v): 
        return 0.1*(v + 40)/(1- np.exp(-(v + 40)*0.1))
    def beta_m(self, v): 
        return 4*np.exp(-(v+65)/18) 
    def inf_m(self,  v): 
        return self.alpha_m(v)/(self.alpha_m(v) + self.beta_m(v)) 

    # Gate h 
    def alpha_h(self, v): 
        return 0.07*math.exp(-(v+65)/20) 
    def beta_h(self, v):
        return 1/(math.exp(-(v + 35) * 0.1) + 1)
    def inf_h(self,v):
        return self.alpha_h(v)/(self.alpha_h(v) + self.beta_h(v))













if __name__ == '__main__': 

    # Init neuron 
    neuron = HH()

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
