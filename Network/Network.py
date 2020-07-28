from Neurons.Synapse import LIF


class Network:
    """
    Network with several layers and neurons
    The learning method choosen was STDP
    """
    def __init__(self, layers):
        """
        Constructs the network
        :param layers: the layers with the number of neurons, respective
        """

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

    def STDP(self):
        pass


Network([2, 3, 1])
