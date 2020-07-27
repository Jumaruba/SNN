import numpy as np
from Neurons.Synapse import LIF


class Network:
    def __init__(self, layers):
        self.layers = layers
        self.neurons = []
        for i, lay in enumerate(layers):
            layer = []
            for j in range(lay):
                neuron = LIF(1, 1)
                if i != 0:
                    for pre_n in self.neurons[i - 1]:
                        neuron.add_pre_neuron(pre_n)

                layer.append(neuron)     # a remover parametros

            self.neurons.append(layer)
        x = 1


Network([2, 3, 1])