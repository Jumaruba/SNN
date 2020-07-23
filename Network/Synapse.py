import math as m

class Synapse:
    def __init__(self):
        self.postNeurons = []
        self.last_t = 0
        self.tauS = 10
        self.Pmax = 1

    def add_postsyn_neuron(self, neuron):
        self.postNeurons.append(neuron)

    def transmit(self, t):
        Ps = self.Pmax/self.tauS * m.exp(1 - (self.last_t - t)/self.tauS)
        self.update_all_postNeurons(Ps)
        self.last_t = t

    def update_all_postNeurons(self, Ps):
        for n in self.postNeurons:
            n.update_P(Ps)
