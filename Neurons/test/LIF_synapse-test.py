from __init__ import *
from LIF_synapse import *  
from plot import plot

'''Test EULER METHOD

There's unity test, it's necessary to compare the output image with the image from the folder expected-results

Paramter for test: 

    self.R = 5  
    self.C = 3
    self.uR = -40
    self.thrs = 30
    self.maxV = 90
    self.v = -65
'''
 
neuron = LIF() 
time = 500
dt = 0.01
steps = math.ceil(time / dt)
I = [0 if 200/dt <= i <= 300/dt  else 10 for i in range(steps)]

# Comment the test you do not wish to do 

# TEST : EULER
#plot(neuron, time, dt, I, 0)

#TEST : RK4
plot(neuron, time, dt, I, 1)