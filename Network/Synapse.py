from Neuron import Neuron as Neuron_

import math
import numpy as np
import matplotlib.pyplot as plt


global tau_m, El, rm_gs, Rm_Ie, Es, V_trhs, dt, T, Pmax 
tau_m = 20 			# [ms] how fast all the action happens  
tau_s = 10 			# [ms] pyke 
El = -70			# [mV]
rm_gs = 0.05			
Rm_Ie = 25			# [mV]
V_trhs = -54		# [mV]
dt = 0.05			# [mS]  
T = 100				# [mS] total time of analyses  
Pmax = 1 
V_reset = -80 

'''LIF neuron with synapse implemented 

Parameters
-------
In: boolean 
	True if the neuron is inhibitory and False if it's excitatory 

'''
class LIF:
	def __init__(self, T, dt, In = False):

		# setting constants
		super().__init__()
		self.Es = 0 if In else -80  # [mV]	
		
		self.steps = math.ceil(T/dt)
		self.v = np.zeros(self.steps) 				# voltage historic 
		self.dirac = np.zeros(self.steps)			# dirac function 
		self.time = np.arange(0, T, dt)				# vector representing the actual time 
		self.spikes = np.zeros(self.steps)			# own spikes stored
		self.pre_neurons = []		 				# pre-synaptic neurons connected to it 
		
	def step(self, i):
		# update the dirac function 
		for k in self.pre_neurons: 
			if k.spikes[i-1]: self.dirac[i-1] = 1 

		Ps_sum = 0
		for ti in range(i):
			if self.dirac[ti]: 
				Ps_sum += self.Ps(self.time[i] - self.time[ti])

		if self.v[i-1] > V_trhs: 
			self.v[i-1] = 0			# setting last voltage to spike value 
			self.v[i] = V_reset 
		else: 
			dv = (El - self.v[i-1] + Ps_sum* rm_gs * (self.v[i-1] - self.Es) + Rm_Ie)/tau_m*dt 
			self.v[i] = dv + self.v[i-1] 
			if self.v[i] >= V_trhs: 
				self.spikes[i] = 1 

	def Ps(self, t): 
		return Pmax*t/tau_s*np.exp(1-t/tau_s) 

	
if __name__ == '__main__':
	steps = math.ceil(T/dt)
	
	n1 = LIF(T, dt, True) 
	n2 = LIF(T, dt, True) 

	neurons = [n1, n2]
	n1.pre_neurons = [neurons[1]]
	n2.pre_neurons = [neurons[0]] 
	n1.v[0] = El + Rm_Ie
	n2.v[0] = El + Rm_Ie
 

	# spikes = dirac(t).timeV 
	for i in range(1,steps): 
		for neuron in neurons: 
			if i == 0: 
				neuron.v[0] = El
			else:  
				neuron.step(i) 
	l1 = [n1.time[i] for i in range(steps) if n1.dirac[i] == 1]
	l2 = [n1.time[i] for i in range(steps) if n2.dirac[i] == 1]
	print(l1)
	print(l2)
	time = np.arange(0,T,dt)
	plt.figure()
	plt.subplot(3,1,1)
	plt.plot(time, n1.v) 
	plt.xlabel("t (ms)")
	plt.ylabel("V1 (mV)")

	plt.subplot(3,1,3)
	plt.plot(time, n2.v)
	plt.xlabel("t (ms)")
	plt.ylabel("V2 (mV)")
	plt.show()