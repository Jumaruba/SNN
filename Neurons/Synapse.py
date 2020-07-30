from Neuron import Neuron as Neuron_

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
dt = 0.05                       # [mS] Step time
T = 2000                        # [mS] total time of analyses
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

        self.v = np.zeros(steps)                    # Voltage historic so as to plot results

        self.actualTime = dt
        self.pre_neuron = None                      # Pre-synaptic neurons connected to it

        max_spikes_anal = 100                       # Max spikes to be considered by the algorithm to calculate Ps
        self.spikes = Spikes(max_spikes_anal)

    def step(self, i):
        self.actualTime += dt

        Ps_sum = self.Ps_sum()

        if Ps_sum > Pmax: Ps_sum = Pmax

        if self.v[i - 1] > V_trhs:
            self.v[i - 1] = 0  # setting last voltage to spike value
            self.v[i] = V_reset
        else:
            self.rk4(Ps_sum, i)

        if self.v[i] >= V_trhs: self.spikes.add_spike(self.actualTime)

    def euler(self, Ps_sum, i):
        dv = (El - self.v[i - 1] - Ps_sum * rm_gs * (self.v[i - 1] - self.Es) + Rm_Ie) / tau_m * dt
        self.v[i] = dv + self.v[i - 1]

    def rk4(self, Ps_sum, i):
        dv1 = self.fu(self.v[i - 1], Ps_sum) * dt
        dv2 = self.fu(self.v[i - 1] + dv1 * 0.5, Ps_sum) * dt
        dv3 = self.fu(self.v[i - 1] + dv2 * 0.5, Ps_sum) * dt
        dv4 = self.fu(self.v[i - 1] + dv3, Ps_sum) * dt
        dv = 1 / 6 * (dv1 + dv2 * 2 + dv3 * 2 + dv4)
        self.v[i] = self.v[i - 1] + dv

    def fu(self, v, Ps_sum):
        return (El - v - Ps_sum * rm_gs * (v - self.Es) + Rm_Ie) / tau_m

    def Ps_sum(self):
        ps_sum = 0
        for ti in self.spikes.time_spikes:
            ps_sum += self.Ps(self.actualTime - ti)

        return ps_sum

    def Ps(self, t):
        return Pmax * t / tau_s * np.exp(1 - t / tau_s)


if __name__ == '__main__':

    n1 = LIF()
    n2 = LIF()

    neurons = [n1, n2]
    n1.pre_neuron = neurons[1]
    n2.pre_neuron = neurons[0]

    n1.v[0] = El + Rm_Ie
    n2.v[0] = El

    # RUN
    for i in range(1, steps):
        for neuron in neurons:
            neuron.step(i)

        # PLOT RESULTS
    time = np.arange(0, T, dt)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time, n1.v)
    plt.xlabel("t (ms)")
    plt.ylabel("V1 (mV)")

    plt.subplot(2, 1, 2)
    plt.plot(time, n2.v)
    plt.xlabel("t (ms)")
    plt.ylabel("V2 (mV)")
    plt.show()
