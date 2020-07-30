import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Neurons"))

from Neuron import Neuron as Neuron_

import math
import numpy as np
import matplotlib.pyplot as plt
import random

global tau_m, El, rm_gs, Es, V_trhs, dt, T, Pmax
tau_m = 30              # [ms] How fast all the action happens
tau_s = 10              # [ms]
El = -70                # [mV]
rm = 0.1
V_trhs = -54            # [mV]
dt = 0.05               # [mS]
T = 1000                 # [mS] total time of analyses
Pmax = 1
V_reset = -80

# STDP 
gsMax = 0.01            # Max value to gs
forgetTime = 100        # [mS]  might be necessary to change
A_plus = 0.01
A_neg = 0.02
tau_plus = 20           # [mS]
tau_neg = 100           # [mS]

'''A class to keep track of the spikes transmited by a synapse 

Parameters
-------
time: float array
Array responsible for store the time the neuron had a spike  

lenTime: int
Size of the time array. Yeah, I could do len(time), but for python it costs O(n) (which is too much)  

max_len: int 
Maximum number of neurons to be analyzed 

'''


class Synapse:

    def __init__(self, max_len=100):
        self.max_len = max_len
        self.time = []
        self.lenTime = 0
        self.last_spike_checked = False

    def add_spike(self, time):
        if self.lenTime == self.max_len:
            self.time.pop(0)
            self.lenTime -= 1
        self.time.append(time)
        self.lenTime += 1
        self.last_spike_checked = False


    # Checks if the last spike was at the specific time given
    def previous_spike_time(self, time):
        if self.lenTime - 1 < 0: return False
        return self.time[self.lenTime - 1] == time

    # get the spikes from the parameter time until the current time
    def get_spikes_delta_time(self, time):
        counter = self.lenTime - 1
        spikes = []
        while counter >= 0:
            if self.time[counter] <= time:
                spikes.append(self.time[counter])
            counter -= 1
        return spikes


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
    def __init__(self, T, dt, Exc=False):

        # setting constants
        self.Es = 0 if Exc else -80  # [mV]

        self.Rm_Ie = 100                        # [mV]
        self.steps = math.ceil(T / dt)
        self.v = np.zeros(self.steps)           # voltage historic
        self.v[0] = El
        self.pre_neuron = []                    # pre-synaptic neurons connected to it
        self.actualTime = dt
        self.ps = 0

        # STDP
        self.gs_list = []                       # list weights of pre-sytnaptic connections
        self.gs_ps = 0                          # ps * gs
        self.gs_ps_total = 0

        # to be deleted
        self.gs_list2 = [0]

        # Synapse
        self.max_spikes_anal = 10               # max spikes to be considered by the algorithm to calculate Ps
        self.synapse = Synapse(self.max_spikes_anal)
        self.synaptic_components = []
        self.sum_syn_comps = 0

        self.last_spike = 0

    def step(self, i):
        self.actualTime += dt
        print("i=", i)
        print()
        if self.v[i - 1] > V_trhs:
            self.v[i - 1] = 0                   # setting last voltage to spike value
            self.v[i] = V_reset
        else:
            self.rk4(i)

        if self.v[i] >= V_trhs:
            self.synapse.add_spike(self.actualTime)
            self.last_spike = self.actualTime

        self.STDP()  # udpate all the weights
        # if i == 4: exit(0)

    def STDP(self):
        gs_list2 = 0        # to be deleted
        self.gs_ps_total = 0

        # update the value of gs for each neuron
        for i, neuron in enumerate(self.pre_neuron):
            f_t = 0

            # calculates the f_t
            if self.last_spike != 0:
                if neuron.synapse.last_spike_checked: pass

                neuron.synapse.last_spike_checked = True
                delta_t = self.last_spike - neuron.synapse.time[self.synapse.lenTime-1]
                f_t += A_plus * np.exp(delta_t / tau_plus) if delta_t > 0 else -A_neg * np.exp(delta_t / tau_neg)



            print("F_T", f_t)
            # calculates the ps
            self.ps = self.Ps_sum(neuron)
            print("PS:", self.ps)

            self.gs_list[i] += gsMax * f_t
            print("GS:", self.gs_list[i])

            # Upper and lower bound
            if self.gs_list[i] > 1:
                self.gs_list[i] = 1
            if self.gs_list[i] < 0:
                self.gs_list[i] = 0

            gs_list2 = self.gs_list[i]      # to be deleted

            self.gs_ps_total += self.ps * self.gs_list[i]
            print("GS*PS", self.gs_list[i])

            self.gs_ps = self.ps * self.gs_list[i]
            self.synaptic_components.append(self.gs_ps * neuron.Es)



        self.gs_list2.append(gs_list2)
        self.sum_syn_comps = sum(self.synaptic_components)
        self.synaptic_components = []

    def euler(self, i):
        dv = self.fu(self.v[i - 1]) * dt
        self.v[i] = dv + self.v[i - 1]

    # to fix this one
    def rk4(self, i):

        dv1 = self.fu(self.v[i - 1]) * dt
        dv2 = self.fu(self.v[i - 1] + dv1 * 0.5) * dt
        dv3 = self.fu(self.v[i - 1] + dv2 * 0.5) * dt
        dv4 = self.fu(self.v[i - 1] + dv3) * dt
        dv = 1 / 6 * (dv1 + dv2 * 2 + dv3 * 2 + dv4)
        self.v[i] = self.v[i - 1] + dv

    def fu(self, v):
        return (El - v - rm * self.gs_ps_total * v + self.sum_syn_comps + self.Rm_Ie) / tau_m

    def Ps_sum(self, neuron):
        ps_sum = 0
        for ti in neuron.synapse.time:
            ps_sum += self.Ps(self.actualTime - ti)

        return ps_sum

    def add_pre_neuron(self, neuron):
        self.pre_neuron.append(neuron)
        self.gs_list.append(random.random())  # give a weight for the connection

    def Ps(self, t):
        return Pmax * t / tau_s * np.exp(1 - t / tau_s)


class Network:
    def __init__(self, layers):
        self.layers = layers
        self.neurons = []
        self.choice = [True]
        for i, lay in enumerate(layers):
            layer = []
            for j in range(lay):
                type = random.choice(self.choice)
                neuron = LIF(T, dt, type)
                print(type)
                if i != 0:
                    neuron.Rm_Ie = 0
                    for pre_n in self.neurons[i - 1]:
                        neuron.add_pre_neuron(pre_n)

                layer.append(neuron)

            self.neurons.append(layer)

    def run(self):
        global tau_m
        steps = math.ceil(T / dt)
        for i in range(1, steps):
            for t, l in enumerate(self.neurons):
                for j, neuron in enumerate(self.neurons[t]):
                    if i == math.ceil(300/dt):
                        self.neurons[1][0].Rm_Ie = 20
                        self.neurons[0][0].Rm_Ie = 5
                        tau_m = 50
                    if i == math.ceil(700/dt):
                        self.neurons[0][0].Rm_Ie = 25
                        self.neurons[1][0].Rm_Ie = 40
                        tau_m = 20
                    neuron.step(i)


n = Network([1, 1])

n.run()

time = np.arange(0, T, dt)
plt.figure()
plt.title("Neuron n1")
plt.plot(time[0:math.ceil(300/dt)], n.neurons[0][0].v[0:math.ceil(300/dt)])
plt.plot(time[math.ceil(300/dt):math.ceil(700/dt)], n.neurons[0][0].v[math.ceil(300/dt):math.ceil(700/dt)], color = 'r')
plt.plot(time[math.ceil(700/dt): ], n.neurons[0][0].v[math.ceil(700/dt): ], color ='g')
plt.xlabel("t (ms)")
plt.ylabel("V1 (mV)")

plt.figure()
plt.title("Neuron n2")
plt.plot(time[0:math.ceil(300/dt)], n.neurons[1][0].v[0:math.ceil(300/dt)])
plt.plot(time[math.ceil(300/dt):math.ceil(700/dt)], n.neurons[1][0].v[math.ceil(300/dt):math.ceil(700/dt)], color = 'r')
plt.plot(time[math.ceil(700/dt): ], n.neurons[1][0].v[math.ceil(700/dt): ], color ='g')
plt.xlabel("t (ms)")
plt.ylabel("V2 (mV)")

plt.figure()
plt.title("n2 gs")
plt.plot(time[0:math.ceil(300/dt)], n.neurons[1][0].gs_list2[0:math.ceil(300/dt)])
plt.plot(time[math.ceil(300/dt):math.ceil(700/dt)], n.neurons[1][0].gs_list2[math.ceil(300/dt):math.ceil(700/dt)], color = 'r')
plt.plot(time[math.ceil(700/dt): ], n.neurons[1][0].gs_list2[math.ceil(700/dt): ], color ='g')
plt.xlabel("t (ms)")
plt.ylabel("Conductance (ms/m)")

plt.figure()
plt.title("Normalized")
gs = n.neurons[1][0].gs_list2
n1 = n.neurons[0][0].v
n2 = n.neurons[1][0].v
print(gs)
plt.plot(time, (gs - np.min(gs))/(np.max(gs) - np.min(gs)))
plt.plot(time, (n1 - np.min(n1))/(np.max(n1) - np.min(n1)))
plt.plot(time, (n2 - np.min(n2))/(np.max(n2) - np.min(n2)))
plt.xlabel("t (ms)")

plt.show()

