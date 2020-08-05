

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def generate_weights(Ne, Ni):
    weights = np.random.rand(Ne + Ni, Ne + Ni)
    weights[:, Ne:] *= -1
    weights[:, :Ne] *= 0.5
    return weights


''' Method responsible for drawing the structure of the network as a graph 

Parameters 
-------
Ne: int
    Number of excitatory neurons
Ni: int
    Number of inhibitory neurons 
weights: matrix of floats size [Ne+Ni]x[Ne+Ni] 
    Weights between neurons 
edgeLabel: bool 
    False case the edges weights are not wanted, True otherwise 
    
    
'''


def drawGraph(Ne, Ni, weights=None, edgeLabel=True):
    graph = nx.DiGraph()
    # add nodes to the graph
    for i in range(Ne + Ni): graph.add_node(i)

    # set red color for inhibitory neurons and blue for excitatory ones
    colors = []
    for i in range(Ne): colors.append((0.5, 0.4, 1))  # excitatory
    for i in range(Ne, Ne + Ni): colors.append((1, 0.5, 0.6))  # inhibitory

    # add the edge list
    edgeColor = []
    node_label = {}
    for i in range(Ne + Ni):
        for j in range(Ne + Ni):
            weight_value = round(weights[i][j], 3)
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


# Example of usage
if __name__ == '__main__':
    Ne = 3
    Ni = 3
    drawGraph(Ne, Ni, generate_weights(Ne, Ni))
