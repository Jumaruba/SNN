import sys
import os
import numpy as np
from enum import Enum

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "Neurons"))
from Izhikevich import Izhi
import random as rand
import math as m
import matplotlib.pyplot as plt
import networkx as nx

'''
@def A customizable network 
@param Ni: inhibitory neurons 
@param Ne: excitatory neurons 
@param weights: matrix with weigths for each connection. The default  is to generate random weights. 
@param time: total time of the action 
@param dt: variation of time
'''


class CustomNetwork:
    def __init__(self, Ne=800, Ni=200, weights=None, time=1000, dt=0.5):
        self.Ne = Ne
        self.Ni = Ni
        self.time = time
        self.dt = dt

        if weights:
            self.weigths = weights
        else:
            self.generate_weights()

        # the weight matrix must be (ne + ni) x (ne + ni )
        if len(self.weights[0]) != self.Ne + self.Ni or len(self.weights) != self.Ne + self.Ni:
            print("Invalid Matrix")
            raise ValueError

        self.neurons = [Izhi() for i in range(self.Ne + self.Ni)]
        self.randomizeNeurons()

    '''
    @def Set random constants for the neurons 
    '''

    def randomizeNeurons(self):
        for i in range(self.Ni + self.Ne):
            temp = rand.uniform(0, 1)  # random constant
            if i < self.Ne:
                self.neurons[i].set_constants(0.02, 0.2, -0.65 + 15 * temp ** 2, 8 - 6 * temp ** 2)
            else:
                self.neurons[i].set_constants(0.02 + 0.08 * temp, 0.25 - 0.05 * temp, -65, 2)

            self.neurons[i].V = -65
            self.neurons[i].U = self.neurons[i].V * self.neurons[i].b

    '''
    @def Generate the weights randomly 
    '''

    def generate_weights(self):
        self.weights = np.random.rand(self.Ni + self.Ne, self.Ni + self.Ne)
        self.weights[:, self.Ne:] *= -1
        self.weights[:, :self.Ne] *= 0.5

    def fire(self):
        # parsing to avoid big expressions
        time = self.time
        dt = self.dt

        steps = m.ceil(time / dt)
        firings = []  # all occurrences of firing (neuron, time)
        for t in range(time):
            I = np.concatenate((np.random.normal(1, 1, self.Ni) * 5, np.random.normal(1, 1, self.Ne) * 2), axis=0)
            fired = [i for i in range(self.Ni + self.Ne) if self.neurons[i].V >= 30]

            if len(firings) == 0:
                firings = [[neuronNumber, t] for neuronNumber in fired]
            else:
                firings = np.concatenate((firings, [[neuronNumber, t] for neuronNumber in fired]), axis=0)

            # update U and V to the fired ones
            for k in range(len(fired)):
                self.neurons[fired[k]].nextIteration(0.5, I[fired[k]])

            # update I
            I = I + np.sum(self.weights[:, fired], axis=1)

            for k in range(self.Ne + self.Ni):
                self.neurons[k].nextIteration(0.5, I[k])

    '''
    @def Method responsible to draw the structure of the network 
    '''

    def drawGraph(self):

        graph = nx.DiGraph()
        # add nodes to the graph
        for i in range(self.Ne + self.Ni): graph.add_node(i)

        # set red color for inhibitory neurons and blue for excitatory ones
        colors = []
        for i in range(self.Ne): colors.append((0.5, 0.4, 1))  # excitatory
        for i in range(self.Ne, self.Ne + self.Ni): colors.append((1, 0.5, 0.6))  # inhibitory

        # add the edge list
        edgeColor = []
        node_label = {}
        for i in range(self.Ne + self.Ni):
            for j in range(self.Ne + self.Ni):
                weight_value = round(self.weights[i][j], 3)
                if i < j:
                    graph.add_edge(i, j, label=str(weight_value) + " (" + str(i) + "->" + str(j) + ")")
                    edgeColor.append('r')
                elif j < i:
                    graph.add_edge(i, j, label=str(weight_value) + " (" + str(i) + "->" + str(j) + ")")
                    edgeColor.append('b')
                else:
                    node_label[i] = str(i) + ":" + str(weight_value)

        graph = nx.relabel_nodes(graph, node_label, copy=False)
        # set the edges between neurons
        pos = nx.circular_layout(graph)
        nx.draw_networkx_nodes(graph, pos, node_size=800, node_color=colors, with_labels=True)
        nx.draw_networkx_edges(graph, pos, node_size=800, arrowstyle='->',
                               arrowsize=10, edge_cmap=plt.cm.Blues, width=1, connectionstyle='arc3,rad=0.1',
                               edge_color=edgeColor)

        nx.draw_networkx_labels(graph, pos, font_size=6, font_weight="bold")
        labels = nx.get_edge_attributes(graph, 'label')

        nx.draw_networkx_edge_labels(graph, pos, label_pos=0.8, font_size=7, edge_labels=labels, labels=True)

        plt.show()


nn = CustomNetwork(3, 2)
nn.drawGraph()
