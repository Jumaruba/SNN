
import math
import numpy as np
import matplotlib.pyplot as plt

global tau_m, El, rm_gs, Rm_Ie, Es, V_trhs, dt, T, Pmax
tau_m = 20                      # [ms] How fast all the action happens
tau_s = 10                      # [ms] Peak lag from the beginning of the curve
El = -70                        # [mV] Initial voltage
rm_gs = 0.05                    # [mA] Resistence * Synapse conductance
Rm_Ie = 25                      # [mV] Resistence * Externa current
V_trhs = -54                    # [mV] Threshold voltage
dt = 0.05                       # [ms] Step time
T = 4000                        # [mS] total time of analyses
Pmax = 1                        # Max value the Ps can reach
V_reset = -80                   # [mV] Reset voltage

steps = math.ceil(T/dt)

'''A class to keep track of the spikes transmited by a synapse 
Parameters
-------
time_spikes: float array
Array responsible for store the time the neuron had a spike  
len_time_spikes: int
Size of the time array. Yeah, I could do len(time), but for python it costs O(n) (which is too much) 
max_len: int
Max number of neurons to be considered at the calculation of Ps_num 
'''
class Spikes:

    def __init__(self, max_len=20):
        self.time_spikes = []
        self.len_time_spike = 0
        self.max_len = max_len

    def add_spike(self, current_time):
        if self.len_time_spike == self.max_len:
            self.time_spikes.pop(0)
            self.len_time_spike -= 1

        self.time_spikes.append(current_time)
        self.len_time_spike += 1


'''LIF neuron with synapse implemented 
Parameters
-------
In: boolean 
	True if the neuron is inhibitory and False if it's excitatory 
dt: float
	Delta time 
T: float 
	Total time to be analysed 
'''


class LIF:
    def __init__(self, Exc=True):

        self.Es = 0 if Exc else -80

        self.v = El 
        self.pre_neuron = None                      # Pre-synaptic neurons connected to it

        max_spikes_anal = 100                       # Max spikes to be considered by the algorithm to calculate Ps
        self.spikes = Spikes(max_spikes_anal)

        self.ps_sum = 0

    def step(self, i, t):

        self.Ps(t)

        if self.ps_sum > Pmax:
            self.ps_sum = Pmax

        if self.v > V_trhs:
            self.v = V_reset
        else:
            self.euler(i)

        if self.v >= V_trhs:
            self.spikes.add_spike(t)
            self.v = 0


    def euler(self, i):
        dv = self.fv(self.v)* dt
        self.v += dv 

    def rk4(self, i):
        dv1 = self.fv(self.v) * dt
        dv2 = self.fv(self.v + dv1 * 0.5) * dt
        dv3 = self.fv(self.v + dv2 * 0.5) * dt
        dv4 = self.fv(self.v + dv3) * dt
        dv = 1 / 6 * (dv1 + dv2 * 2 + dv3 * 2 + dv4)
        self.v = self.v + dv

    def fv(self, v):
        return (El - v - self.ps_sum * rm_gs * (v - self.Es) + Rm_Ie) / tau_m

    def Ps(self,t):
        self.ps_sum = 0
        for ti in self.pre_neuron.spikes.time_spikes:
            delta_t = t - ti
            self.ps_sum += Pmax * delta_t / tau_s * np.exp(1 - delta_t / tau_s)         # apply ps formula for each spike



if __name__ == '__main__':

#   INIBITORIO -------------------
    n1 = LIF(False)
    n2 = LIF(False)

    neurons = [n1, n2]
    n1.pre_neuron = neurons[1]
    n2.pre_neuron = neurons[0]

    n1.v = El
    n2.v = -54

    v1 = np.zeros(steps)
    v2 = np.zeros(steps)

    # RUN
    for i in range(1, steps):
        for j,neuron in enumerate(neurons):
            neuron.step(i, i*dt)
            if j == 0: 
                v1[i] = neuron.v 
            if j == 1: 
                v2[i] = neuron.v 
            

    # PLOT RESULTS
    begin = 3800
    begin_step = math.ceil(begin/dt)
    time = np.arange(begin, T, dt)


    plt.title("Inhibitory neurons")
    plt.plot(time, v1[begin_step:], color='b', label="neuron 1")
    plt.plot(time, v2[begin_step:], color='r', label="neuron 2")
    plt.legend(loc="upper left")
    plt.xlabel("t (ms)")
    plt.ylabel("V (mV)")
    plt.show()

#   EXCITATORIO -------------------------------
    n1 = LIF()
    n2 = LIF()

    neurons = [n1, n2]
    n1.pre_neuron = neurons[1]
    n2.pre_neuron = neurons[0]

    n1.v = El
    n2.v = -54

    v3 = np.zeros(steps)
    v4 = np.zeros(steps)

    # RUN
    for i in range(1, steps):
        for j,neuron in enumerate(neurons):
            neuron.step(i, i*dt)
            if j == 0: 
                v3[i] = neuron.v 
            if j == 1: 
                v4[i] = neuron.v 

    begin = 3800
    begin_step = math.ceil(begin/dt)
    time = np.arange(begin, T, dt)


    plt.title("Excitatory neurons")
    plt.plot(time, v3[begin_step:], color='b', label="neuron 1")
    plt.plot(time, v4[begin_step:], color='r', label="neuron 2")
    plt.xlabel("t (ms)")
    plt.ylabel("V (mV)")
    plt.legend(loc="upper left")
    plt.show()