
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
T = 4000                        # [mS] total time of analyses
Pmax = 1                        # Max value the Ps can reach
V_reset = -80                   # [mV] Reset voltage
Rm = 2.5                        # [mOhm] Membrane resistence 




'''LIF neuron with synapse implemented 
Parameters
-------
Exc: boolean 
	True if the neuron is excitatory and False if it's inhibitory 
'''


class LIF:
    def __init__(self, Exc=True):

        self.Es = 0 if Exc else -80

        self.v = El 
        self.pre_neuron = []                        # Pre-synaptic neurons connected to it

        self.last_spike = None                      # Time of the last spike 
        self.ps_sum = 0
        self.current_time = 0
    def step(self, dt = 0.05, I = 0,  method=0):
        self.current_time += dt 
        self.Ps()

        if self.ps_sum > Pmax: 
            self.ps_sum = Pmax

        if self.v > V_trhs: 
            self.v = V_reset
        else:
            if method == 0: 
                self.euler(I, dt)
            if method == 1: 
                self.rk4(I, dt) 

        if self.v >= V_trhs:
            self.last_spike = self.current_time  
            self.v = 0



    def euler(self, I, dt):
        dv = self.fv(self.v, I)* dt
        self.v += dv 

    def rk4(self, I, dt):
        dv1 = self.fv(self.v, I) * dt
        dv2 = self.fv(self.v + dv1 * 0.5, I) * dt
        dv3 = self.fv(self.v + dv2 * 0.5, I) * dt
        dv4 = self.fv(self.v + dv3, I) * dt
        dv = 1 / 6 * (dv1 + dv2 * 2 + dv3 * 2 + dv4)
        self.v += dv

    def fv(self, v, I):
        return  (El - v - self.ps_sum * rm_gs * (v - self.Es) + Rm * I) / tau_m

    def Ps(self):
        self.ps_sum = 0
        for neuron in self.pre_neuron: 
            if neuron.last_spike: 
                delta_t = self.last_spike - neuron.last_spike
                self.ps_sum += Pmax * delta_t / tau_s * np.exp(1 - delta_t / tau_s)         # apply ps formula for each spike


