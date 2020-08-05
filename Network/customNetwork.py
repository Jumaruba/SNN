from __init__ import * 

from IZHI import IZHI
import random as rand
import numpy as np 
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
I: array of floats 
    Initial input. Default = random input 
    
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
    def __init__(self, Ne=4, Ni=2, weights=None, time=1000, dt=0.5, I=None):
        self.Ne = Ne
        self.Ni = Ni
        self.time = time
        self.dt = dt
        self.firings = []
        if I: self.I = np.array(I)
        else:
            self.I = np.concatenate((np.random.normal(1, 1, self.Ni) * 5, np.random.normal(1, 1, self.Ne) * 2), axis=0)

        if weights: self.weights = np.array(weights)
        else:       self.generate_weights()

        # the weight matrix must be (ne + ni) x (ne + ni )
        if len(self.weights[0]) != self.Ne + self.Ni or len(self.weights) != self.Ne + self.Ni:
            print("Invalid Matrix")
            raise ValueError

        self.neurons = [IZHI() for i in range(self.Ne + self.Ni)]
        self.randomize_neurons()

    def randomize_neurons(self):
        for i in range(self.Ni + self.Ne):
            temp = rand.uniform(0, 1)   # random constant

            if i < self.Ne:
                self.neurons[i].a = 0.02
                self.neurons[i].b = 0.2
                self.neurons[i].c = -0.65 + 15 * temp ** 2
                self.neurons[i].d = 8 - 6 * temp ** 2
            else:
                self.neurons[i].a = 0.02 + 0.08 *temp
                self.neurons[i].b = 0.25 - 0.05 * temp
                self.neurons[i].c = -65
                self.neurons[i].d = 2

            self.neurons[i].v = -65
            self.neurons[i].u = self.neurons[i].v * self.neurons[i].b


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
            
            fired = [i for i in range(self.Ne + self.Ni) if self.neurons[i].v >= 30]
            len_fired = len(fired)
            len_firings = len(self.firings)

            if verbose: 
                #print("FIRED NEURONS NOW        -------", fired)
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
                self.neurons[fired[k]].step(0.5, self.I[fired[k]])

            # update I
            self.I = self.I + np.sum(self.weights[:, fired], axis=1)

            for k in range(self.Ne + self.Ni):
                self.neurons[k].step(self.dt, self.I[k])

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
        y = [self.firings[i][0] for i in range(firings_size)]  # neurons
        x = [self.firings[i][1] for i in range(firings_size)]  # time
        plt.title("Spikes in a SNN")
        plt.xlabel("Time [ms]")
        plt.ylabel("Neurons")
        plt.scatter(x, y, s=0.1, color='black')
        plt.show()

    def get_dataframe(self): 
        df = pd.DataFrame(self.firings)
        df.rename(columns = {0: 'neurons', 1: 'time[ms]'}, inplace=True)
        return df 

    def export_csv(self): 
        self.get_dataframe().to_csv('out.csv')


if __name__ == '__main__': 
    n = CustomNetwork()
    n.fire()
    n.drawGraph(True)
    n2 = CustomNetwork(80, 20, time=100, dt=0.05)
    n2.fire()
    n2.plot()