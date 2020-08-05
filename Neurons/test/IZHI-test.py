from __init__ import * 
from IZHI import IZHI 
from plot import plot

'''Test EULER METHOD

There's unity test, it's necessary to compare the output image with the image from the folder expected-results

Parameters for test: 

    a = .1            
    b = .2
    c = -65
    d = 8
    
    thrs = 30 
    v = -65
    u = -14

'''


neuron = IZHI() 
time = 500
dt = 0.01
steps = math.ceil(time / dt)
I = [0 if 200/dt <= i <= 300/dt  else 10 for i in range(steps)]

# Comment the test you do not wish to do 

# TEST : EULER
plot(neuron, time, dt, I, 0)

#TEST : RK4
# plot(neuron, time, dt, I, 1)