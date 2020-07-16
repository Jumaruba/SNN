import sys
import os 
import numpy as np
from enum import Enum
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "Neurons"))
from Izhikevich import Izhi 
import random as rand  
import math as m 
import matplotlib.pyplot as plt


'''
@def Simple network for neurons with one layer of neurons explained at izhikevich paper 
@param numNeurons: number of neurons 
'''


class Network: 
	def __init__(self, num_neurons):
		self.Ni = m.ceil(num_neurons / 4)   				# number of inhibitory neurons
		self.Ne = num_neurons - self.Ni        				# number of excitatory neurons
		self.numNeurons = num_neurons
		self.neurons = [Izhi() for i in range(num_neurons)]

		self.randomize_neurons()
		self.generate_weights()

	def randomize_neurons(self):
		for i in range(self.numNeurons):  
			temp = rand.uniform(0, 1) 		# random constant
			if i < self.Ne:				
				self.neurons[i].set_constants(0.02, 0.2, -0.65 + 15*temp**2, 8 - 6*temp**2)
			else: 
				self.neurons[i].set_constants(0.02 + 0.08*temp, 0.25 - 0.05*temp, -65, 2)

			self.neurons[i].V = -65 
			self.neurons[i].U = self.neurons[i].V * self.neurons[i].b

	def generate_weights(self):
		self.weights = np.random.rand(self.numNeurons, self.numNeurons) 
		self.weights[:, self.Ne:] *= -1
		self.weights[:, :self.Ne] *= 0.5
	
	def fire(self):
		time = 1000 
		dt = 0.5
		steps = m.ceil(time/dt)
		firings = []				# all occurrences of firing (neuron, time)
		for t in range(time): 
			I = np.concatenate((np.random.normal(1, 1, self.Ni)*5, np.random.normal(1, 1, self.Ne)*2), axis=0)
			fired = [i for i in range(self.numNeurons) if self.neurons[i].V >= 30]

			if len(firings) == 0: 
				firings = [[neuronNumber, t] for neuronNumber in fired]
			else:  
				firings = np.concatenate((firings, [[neuronNumber, t] for neuronNumber in fired]), axis=0)

			# update U and V to the fired ones 
			for k in range(len(fired)):
				self.neurons[fired[k]].nextIteration(0.5, I[fired[k]])

			# update I
			I = I + np.sum(self.weights[:, fired], axis=1)
			
			for k in range(self.numNeurons): 
				self.neurons[k].nextIteration(0.5, I[k])

		return firings 


n = Network(4)

firings = n.fire() 

x = [firings[i][0] for i in range(len(firings))]		# neurons 
y = [firings[i][1] for i in range(len(firings))]		# time 
plt.title("Spikes in a SNN")
plt.xlabel("Neurons")
plt.ylabel("Time [ms]")
plt.scatter(x, y, s=0.05, color='black')
plt.show()
