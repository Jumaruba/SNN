import sys
import os 
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "Neurons"))

import numpy as np 
from enum import Enum 
from Izhikevich import Izhi 
import random as rand  
import math as m 
import matplotlib.pyplot as plt

'''
@def Simple network for neurons with one layer of neurons 
@param numNeurons: number of neurons 
'''
class Network: 
	def __init__(self, numNeurons):
			self.Ni = m.ceil(numNeurons/4)   				# number of inhibitory neurons 
			self.Ne = numNeurons - self.Ni        			# number of excitatory neurons 
			self.numNeurons = numNeurons  
			self.neurons = [Izhi() for i in range(numNeurons)]

			self.randomizeNeurons() 
			self.generateWeights() 

	def randomizeNeurons(self): 
		for i in range(self.numNeurons):  
			temp = rand.uniform(0,1) 		# random constant
			if i < self.Ne:				
				self.neurons[i].set_constants(0.02, 0.2, -0.65+15*temp**2, 8 - 6*temp**2)
			else: 
				self.neurons[i].set_constants(0.02 + 0.08*temp, 0.25 - 0.05*temp, -65, 2)

			self.neurons[i].V = -65 
			self.neurons[i].U = self.neurons[i].V * self.neurons[i].b 

	
	def generateWeights(self): 
		self.weights = np.random.rand(self.numNeurons, self.numNeurons) 
		self.weights[:,self.Ni: ]*= -1
	
	def fire(self):
		time = 1000 
		dt = 0.5
		steps = m.ceil(time/dt)
		firings = []				# all occurrences of firing (neuron, time)
		for t in range(time): 
			I = np.concatenate((np.random.normal(1,1,self.Ni)*5, np.random.normal(1,1,self.Ne)*2), axis = 0)
			fired = [i for i in range(self.numNeurons) if self.neurons[i].V >= 30]

			if len(firings) == 0: 
				firings = [[neuronNumber, t] for neuronNumber in fired]
			else:  
				firings = np.concatenate((firings, [[neuronNumber, t] for neuronNumber in fired]), axis = 0)

			# update U and V to the fired ones 
			for i in range(len(fired)): self.neurons[fired[i]].nextIteration(0.5, I[fired[i]])

			# update I
			I = I + np.sum(self.weights[:, fired], axis = 1) 
			
			for i in range(self.numNeurons): 
				self.neurons[i].nextIteration(0.5, I[i]) 


		return firings 


n = Network(1000)  

firings = n.fire() 

x = [firings[i][0] for i in range(len(firings))]		# neurons 
y = [firings[i][1] for i in range(len(firings))]		# time 
plt.xlabel("Neurons")
plt.ylabel("Time [ms]")
plt.scatter(x,y, s= 0.05, color = 'black')
plt.show()