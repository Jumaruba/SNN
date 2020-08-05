from __init__ import * 
from LIF_synapse import * 
import numpy as np 
import matplotlib.pyplot as plt 

dt = 0.05
T = 4000
steps = math.ceil(T/dt)
I = 10

#   INIBITORIO -------------------
n1 = LIF(False)
n2 = LIF(False)

neurons = [n1, n2]
n1.pre_neuron = [neurons[1]]
n2.pre_neuron = [neurons[0]]

n1.v = El
n2.v = -54

v1 = np.zeros(steps)
v2 = np.zeros(steps)

# RUN
for i in range(1, steps):
    for j,neuron in enumerate(neurons):
        neuron.step(dt, I)
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
n1.pre_neuron = [neurons[1]]
n2.pre_neuron = [neurons[0]]

n1.v = El
n2.v = -54

v3 = np.zeros(steps)
v4 = np.zeros(steps)

v3[0] = n1.v
v4[0] = n2.v

# RUN
for i in range(1, steps):
    for j,neuron in enumerate(neurons):
        neuron.step(dt, I)
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


