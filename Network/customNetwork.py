import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Neurons"))

from Izhikevich import Izhi
import random as rand
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd 

''' A customizable network based on the izhikevich neuron 
Parameters
-------
Ni: integer 
    Number of inhibitory neurons 
Ne: integer 
    Number of excitatory neurons 
weights: matrix of floats or None 
    Matrix with weights for each connection of neurons. The default is to generate random weights. 
    Attention, the matrix must be of the size of (Ne+Ni)x(Ne+Ni). Otherwise it will lead to an error.
time: float
    Total amount of time for studying the network
dt: float
    Variation of time
    
Examples
-------
nn = CustomNetwork(Ne= 3, Ni=4) 
nn.drawGraph()                      # for drawing the graph 
nn.fire(verbose=True)             
nn.plot()                           # show the result. It's just possible to plot after nn.fire 
nn.draw() 
nn.export_csv()                     # export the result of the fires as csv format 
'''


class CustomNetwork:
    def __init__(self, Ne=800, Ni=200, weights=None, time=1000, dt=0.5):
        self.Ne = Ne
        self.Ni = Ni
        self.time = time
        self.dt = dt
        self.firings = [] 

        if weights: self.weigths = weights
        else:       self.generate_weights()

        # the weight matrix must be (ne + ni) x (ne + ni )
        if len(self.weights[0]) != self.Ne + self.Ni or len(self.weights) != self.Ne + self.Ni:
            print("Invalid Matrix")
            raise ValueError

        self.neurons = [Izhi() for i in range(self.Ne + self.Ni)]
        self.randomize_neurons()

    def randomize_neurons(self):
        for i in range(self.Ni + self.Ne):
            temp = rand.uniform(0, 1)   # random constant

            if i < self.Ne:
                self.neurons[i].set_constants(0.02, 0.2, -0.65 + 15 * temp ** 2, 8 - 6 * temp ** 2)
            else:
                self.neurons[i].set_constants(0.02 + 0.08 * temp, 0.25 - 0.05 * temp, -65, 2)

            self.neurons[i].V = -65
            self.neurons[i].U = self.neurons[i].V * self.neurons[i].b

    def generate_weights(self):
        self.weights = np.random.rand(self.Ni + self.Ne, self.Ni + self.Ne)
        self.weights[:, self.Ne:] *= -1
        self.weights[:, :self.Ne] *= 0.5

    '''Starts the process of excitation for a given time self.time and for intervals of time self.dt

    Parameters
    -------

    verbose: bool 
        True to show extra information about the process  
        False otherwise 

    '''
    def fire(self, verbose=False):
        fire_time = []  # all occurrences of firing (neuron, time)

        for t in range(self.time):
            I = np.concatenate((np.random.normal(1, 1, self.Ni) * 5, np.random.normal(1, 1, self.Ne) * 2), axis=0)
            fired = [i for i in range(self.Ne + self.Ni) if self.neurons[i].V >= 30]
            len_fired = len(fired)
            len_firings = len(self.firings)

            if verbose: 
                print("FIRED NEURONS NOW        -------", fired)
                print("NUMBER OF FIRED NEURONS  --------", len_fired)
                print("TIME PASSED              -------- %d ms" %(t+1))
                print()

            if len_fired == 0: 
                pass 
            elif len_firings == 0:
                self.firings = [[neuronNumber, t] for neuronNumber in fired]
            else:
                fire_time = [[neuronNumber, t] for neuronNumber in fired]
                self.firings = np.concatenate((self.firings, fire_time), axis=0)

            # update U and V to the fired ones
            for k in range(len_fired):
                self.neurons[fired[k]].nextIteration(0.5, I[fired[k]])

            # update I
            I = I + np.sum(self.weights[:, fired], axis=1)

            for k in range(self.Ne + self.Ni):
                self.neurons[k].nextIteration(self.dt, I[k])

            if verbose: 
                print("TOTAL NUMBER OF FIRES", len_firings) 

    ''' Method responsible for drawing the structure of the network 
   
    Parameters 
    -------
    edgeLabel: bool 
        False case the edges weights are not wanted, True otherwise 
        
        
    '''

    def drawGraph(self, edgeLabel=True):

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

        # relable the nodes with the weight
        graph = nx.relabel_nodes(graph, node_label, copy=False)

        # set the edges between neurons
        pos = nx.circular_layout(graph)
        nx.draw_networkx_nodes(graph, pos, node_size=800, node_color=colors, with_labels=True)
        nx.draw_networkx_edges(graph, pos, node_size=800, arrowstyle='->',
                               arrowsize=10, edge_cmap=plt.cm.Blues, width=1, connectionstyle='arc3,rad=0.1',
                               edge_color=edgeColor)

        nx.draw_networkx_labels(graph, pos, font_size=6, font_weight="bold")
        if edgeLabel:
            labels = nx.get_edge_attributes(graph, 'label')
            nx.draw_networkx_edge_labels(graph, pos, label_pos=0.8, font_size=7, edge_labels=labels, labels=True)

        plt.show()

    '''Plot the relation time x neuron excitation
    '''
    def plot(self):
        firings_size = len(self.firings)
        x = [self.firings[i][0] for i in range(firings_size)]  # neurons
        y = [self.firings[i][1] for i in range(firings_size)]  # time
        plt.title("Spikes in a SNN")
        plt.xlabel("Neurons")
        plt.ylabel("Time [ms]")
        plt.scatter(x, y, s=0.1, color='black')
        plt.show()

    def get_dataframe(self): 
        df = pd.DataFrame(self.firings)
        df.rename(columns = {0: 'neurons', 1: 'time[ms]'}, inplace=True)
        return df 

    def export_csv(self): 
        self.get_dataframe().to_csv('out.csv')


