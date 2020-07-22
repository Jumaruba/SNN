import numpy as np

class Network(object):
    def __init__(self, neuron, sizes):

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.neurons = []
        for s in sizes:
            for i in range(s):
                self.neurons.append(neuron.create_copy())

        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[:-1], sizes[1:])]

    def STDP(self):
        
        pass