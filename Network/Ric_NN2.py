import math as m
from Neurons.LIFS import LIFS
from Network.Synapse import Synapse
import matplotlib.pyplot as plt
import numpy as np


class Network:
    def __init__(self):
        # Neurons of the network
        n1 = LIFS()
        n2 = LIFS()
        self.neurons = (n1, n2)

        # Synapse that connects neuron 1 to neuron 2
        self.syn12 = Synapse()
        self.syn12.add_postsyn_neuron(self.neurons[1])

        # Synapse that connects neuron 2 to neuron 1
        self.syn21 = Synapse()
        self.syn21.add_postsyn_neuron(self.neurons[0])

        self.synapses = {self.neurons[0]: self.syn12, self.neurons[1]: self.syn21}

    def fire(self, dur, dt):
        steps = m.ceil(dur / dt)
        time = 0

        # Preparing the neurons
        self.neurons[0].reset(steps)
        self.neurons[1].reset(steps)
        for i in range(1, steps):
            time += dt
            spiked1 = self.neurons[0].nextIteration(i, dt)
            spiked2 = self.neurons[1].nextIteration(i, dt)
            if spiked1:
                self.synapses[self.neurons[0]].transmit(time)

            if spiked2:
                self.synapses[self.neurons[1]].transmit(time)

        u1 = self.neurons[0].get_u()
        u2 = self.neurons[1].get_u()
        plot(u1, u2, dur, dt)


def plot(u1, u2, tmax, dt):
    vTime = np.arange(0, tmax, dt, dtype=None)
    plt.plot(vTime, u1, color='b')
    plt.plot(vTime, u2, color='r')
    plt.title("LIF neuron")
    plt.xlabel("Time [ms]")
    plt.ylabel("Voltage [mv]")
    plt.show()


net = Network()
net.fire(100, 0.5)
