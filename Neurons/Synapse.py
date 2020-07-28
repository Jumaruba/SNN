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
V_trhs = -54			# [mV]
dt = 0.05			# [mS]  
T = 1000				# [mS] total time of analyses  
Pmax = 1 
V_reset = -80 

'''A class to keep track of the spikes 


'''
class Synapse: 

	def __init__(self): 
		self.time = []						# time where spikes happened 
		self.lenTime = 0 					# important cause python has time O(N) to find out the size of an array 	 

	def add_spike(self, time):
		self.time.append(time) 
		self.lenTime += 1
	
	def check_spike(self,time): 
		b = time in self.time 
		return b 

	def previous_spike_time(self, time): 
		if self.lenTime - 1 < 0 : return False 
		return self.time[self.lenTime -1] == time




	


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
	def __init__(self, T, dt, In = False):

		# setting constants
		self.Es = 0 if In else -80  # [mV]	
		
		self.steps = math.ceil(T/dt)
		self.v = np.zeros(self.steps) 				# voltage historic 
		self.pre_neuron	= None		 				# pre-synaptic neurons connected to it
		self.actualTime = dt 

		self.synapse = Synapse() 
		self.max_spikes_anal = 10 

		
	def step(self, i):
		self.actualTime += dt 
		
		Ps_sum = self.Ps_sum() 

		if self.v[i-1] > V_trhs: 
			self.v[i-1] = 0			# setting last voltage to spike value 
			self.v[i] = V_reset 
		else: 
			self.rk4(Ps_sum, i)

		if self.v[i] >= V_trhs: self.synapse.add_spike(self.actualTime) 

	def euler(self, Ps_sum, i):
		dv = (El - self.v[i - 1] + Ps_sum * rm_gs * (self.v[i - 1] - self.Es) + Rm_Ie) / tau_m * dt
		self.v[i] = dv + self.v[i - 1]


	def rk4(self, Ps_sum, i):
		dv1 = self.fu(self.v[i - 1], Ps_sum) * dt
		dv2 = self.fu(self.v[i - 1] + dv1 * 0.5, Ps_sum) * dt
		dv3 = self.fu(self.v[i - 1] + dv2 * 0.5, Ps_sum) * dt
		dv4 = self.fu(self.v[i - 1] + dv3, Ps_sum) * dt
		dv = 1 / 6 * (dv1 + dv2 * 2 + dv3 * 2 + dv4)
		self.v[i] = self.v[i - 1] + dv 


	def fu(self, v, Ps_sum):
		return (El - v + Ps_sum * rm_gs * (v - self.Es) + Rm_Ie) / tau_m

	def Ps_sum(self):
		counter = 0  
		ps_sum = 0 
		for ti in self.synapse.time: 
			if counter == self.max_spikes_anal: 
				return ps_sum  
			else: 
				ps_sum+= self.Ps(self.actualTime - ti) 
		return ps_sum 

	def Ps(self, t): 
		return Pmax*t/tau_s*np.exp(1-t/tau_s) 

	
if __name__ == '__main__':
	steps = math.ceil(T/dt)
	
	n1 = LIF(T, dt, True) 
	n2 = LIF(T, dt, True) 

	neurons = [n1, n2]
	n1.pre_neuron = neurons[1]
	n2.pre_neuron = neurons[0]
	n1.v[0] = n2.v[0] = El + Rm_Ie 

	# spikes = dirac(t).timeV 
	for i in range(1,steps): 
		for neuron in neurons: 
			neuron.step(i) 
				

	time = np.arange(0, T, dt)
	plt.figure()
	plt.subplot(3, 1, 1)
	plt.plot(time, n1.v) 
	plt.xlabel("t (ms)")
	plt.ylabel("V1 (mV)")

	plt.subplot(3, 1, 3)
	plt.plot(time, n2.v)
	plt.xlabel("t (ms)")
	plt.ylabel("V2 (mV)")
	plt.show()
