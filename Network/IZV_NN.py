import numpy as np

from __init__ import *
from IZHI import IZHI as Izhi
import random as rand
import math as m
import matplotlib.pyplot as plt

'''Simple network for neurons with one layer from network explained at izhikevich paper
Izhikevich paper: https://www.izhikevich.org/publications/spikes.pdf 

Parameters 
-------
numNeurons: 
    Number of neurons 
'''

class Network:
    def __init__(self, num_neurons):
        self.Ni = m.ceil(num_neurons / 4)  # number of inhibitory neurons
        self.Ne = num_neurons - self.Ni  # number of excitatory neurons
        self.numNeurons = num_neurons
        self.neurons = [Izhi() for i in range(num_neurons)]

        self.randomize_neurons()
        self.generate_weights()

    def randomize_neurons(self):
        for i in range(self.numNeurons):
            temp = rand.uniform(0, 1)   # random constant

            if i < self.Ne:
                self.neurons[i].a = 0.02
                self.neurons[i].b = 0.2
                self.neurons[i].c = -65 + 15 * temp ** 2
                self.neurons[i].d = 8 - 6 * temp ** 2
            else:
                self.neurons[i].a = 0.02 + 0.08 * temp
                self.neurons[i].b = 0.25 - 0.05 * temp
                self.neurons[i].c = -65
                self.neurons[i].d = 2

    def generate_weights(self):
        self.weights = np.random.rand(self.numNeurons, self.numNeurons)
        self.weights[:, self.Ne:] *= -1
        self.weights[:, :self.Ne] *= 0.5

    def fire(self):
        time = 1000
        dt = 0.5

        firings = []
        for t in range(time):
            I = np.concatenate((np.random.normal(1, 1, self.Ne) * 5, np.random.normal(1, 1, self.Ni) * 2), axis=0)
            fired = [i for i in range(self.Ne + self.Ni) if self.neurons[i].v >= 30]
            len_fired = len(fired)
            len_firings = len(firings)

            if len_fired == 0: 
                pass 
            elif len_firings == 0:
                firings = [[neuronNumber, t] for neuronNumber in fired]
            else:
                fire_time = [[neuronNumber, t] for neuronNumber in fired]
                firings = np.concatenate((firings, fire_time), axis=0)

            # update U and V to the fired ones
            for k in range(len(fired)):
                for i in range(m.ceil(1/dt)):
                    self.neurons[fired[k]].step(dt, I[fired[k]], 1)

            # update I
            I += np.sum(self.weights[:, fired], axis=1)

            for k in range(self.numNeurons):
                for i in range(m.ceil(1/dt)):
                    self.neurons[k].step(dt, I[k], 1)

        return firings 



n = Network(1000)
firings = n.fire()

y = [firings[i][0] for i in range(len(firings))]  # neurons
x = [firings[i][1] for i in range(len(firings))]  # time

plt.title("Spikes in a SNN")
plt.ylabel("Neurons")
plt.xlabel("Time [ms]")
plt.scatter(x, y, s=0.1, color='black')
plt.show()
