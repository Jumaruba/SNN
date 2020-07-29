

import sys 
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Neurons"))

from Neuron import Neuron as Neuron_

import math
import numpy as np
import matplotlib.pyplot as plt
import random 


global tau_m, El, rm_gs, Rm_Ie, Es, V_trhs, dt, T, Pmax 
tau_m = 20 			# [ms] how fast all the action happens  
tau_s = 10 			# [ms] pyke 
El = -70			# [mV]
rm = 0.01			
Rm_Ie = 25			# [mV]
V_trhs = -54			# [mV]
dt = 0.05			# [mS]  
T = 100				# [mS] total time of analyses  
Pmax = 1 
V_reset = -80 

# STDP 
gsMax = 10 				# might be necessary to change
forgetTime = 40 		# [mS]  
A_plus = 0.01 
A_neg = 0.005			
tau_plus = 20			# [mS]
tau_neg = 100 			# [mS]


'''A class to keep track of the spikes transmited by a synapse 

Parameters
-------
time: float array
Árray responsible for store the time the neuron had a spike  

lenTime: int
Size of the time array. Yeah, I could do len(time), but for python it costs O(n) (which is too much) 

'''
class Synapse: 

	def __init__(self): 
		self.time = []						
		self.lenTime = 0 				# size of the time 

	def add_spike(self, time):
		self.time.append(time) 
		self.lenTime += 1
	
	# Checks if a spike occurred by in a specific time  
	def check_spike(self,time): 
		b = time in self.time 
		return b 

	# Checks if the last spike was at the specific time given 
	def previous_spike_time(self, time): 
		if self.lenTime - 1 < 0 : return False 
		return self.time[self.lenTime -1] == time

	def get_last_spike(self): 
		return self.time[self.lenTime - 1]

	# get the spikes from the parameter time to the current time 
	def get_spikes_delta_time(self, time): 
		counter = self.lenTime - 1 
		spikes = []
		while (counter != 0): 
			if self.time[counter] <= time: spikes.append(self.time[counter])
			counter -= 1 

		return spikes 

'''LIF neuron with synapse implemented 

Parameters
-------
In: boolean 
	True if the neuron is inhibitory and False if it's excitatory 
dt: float
	Delta time 
T: float 
	Total time to be analysed 

'''


class LIF:
	def __init__(self, T, dt,  In = False):

		# setting constants
		self.Es = -80 if In else 0  # [mV]	
		
		self.steps = math.ceil(T/dt)
		self.v = np.zeros(self.steps) 				# voltage historic 
		self.pre_neuron	= []		 				# pre-synaptic neurons connected to it
		
		self.actualTime = dt 						
		
		# STDP 
		self.gs_list = []							# list weights of pre-sytnaptic connections 
		self.gs = 0 

		# Synapse 
		self.synapse = Synapse() 
		self.max_spikes_anal = 10 					# max spikes to be considered by the algorithm to calculate Ps

	def step(self, i):
		self.actualTime += dt 
		

		Ps_sum = self.Ps_sum() 

		if self.v[i-1] > V_trhs: 
			self.v[i-1] = 0			# setting last voltage to spike value 
			self.v[i] = V_reset 
		else: 
			self.rk4(Ps_sum, i)

		if self.v[i] >= V_trhs: 
			self.synapse.add_spike(self.actualTime) 

		self.STDP()  # udpate all the weights 



	def STDP(self): 
		# update the value of gs for each neuron 
		for i,neuron in enumerate(self.pre_neuron): 
			gs = 0
			f_t = 0
			spikes = neuron.synapse.get_spikes_delta_time(self.actualTime)
			# for each spike in the range of time forgetTime, get f(delta_t)
			for spike in spikes: 
				delta_t = self.actualTime - preTime 
				f_t += A_plus*np.exp(delta_t/tau_plus) if delta_t > 0 else  A_neg*np.exp(delta_t/tau_neg) 
			gs_list[i] += gsMax*f_t



	def euler(self, Ps_sum, i):
		dv = (El - self.v[i - 1] + Ps_sum * rm * self.gs * (self.v[i - 1] - self.Es) + Rm_Ie) / tau_m * dt
		self.v[i] = dv + self.v[i - 1]


	def rk4(self, Ps_sum, i):
		dv1 = self.fu(self.v[i - 1], Ps_sum) * dt
		dv2 = self.fu(self.v[i - 1] + dv1 * 0.5, Ps_sum) * dt
		dv3 = self.fu(self.v[i - 1] + dv2 * 0.5, Ps_sum) * dt
		dv4 = self.fu(self.v[i - 1] + dv3, Ps_sum) * dt
		dv = 1 / 6 * (dv1 + dv2 * 2 + dv3 * 2 + dv4)
		self.v[i] = self.v[i - 1] + dv 


	def fu(self, v, Ps_sum):
		return (El - v + Ps_sum * rm * self.gs * ps * (v - self.Es) + Rm_Ie) / tau_m

	def Ps_sum(self):
		counter = 0  
		ps_sum = 0 
		for ti in self.synapse.time: 
			if counter == self.max_spikes_anal: 
				return ps_sum  
			else: 
				ps_sum+= self.Ps(self.actualTime - ti) 
		counter += 1
		return ps_sum 

	def add_pre_neuron(self, neuron):
		self.pre_neuron.append(neuron)  
		self.gs_list.append(random.random()) 			# give a weight for the connection 

	def Ps(self, t): 
		return Pmax*t/tau_s*np.exp(1-t/tau_s) 



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

				layer.append(neuron)     						# a remover parametros

			self.neurons.append(layer)

	def run(self): 

		steps = np.ceil(T/dt)
		for i in range(steps): 
			for t in range(len(layers)): 
				for j in range(layers[t]): 
					self.neurons[t][j].step(i)  


					

Network([2, 3, 1])