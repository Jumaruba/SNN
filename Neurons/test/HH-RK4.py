from __init__ import * 
from plot import plot

'''Test RUNGE KUTTA 4 ORDER METHOD

There's unity test, it's necessary to compare the output image with the image from the folder expected-results
'''


neuron = HH() 
time = 500
dt = 0.01
steps = math.ceil(time / dt)
I = [0 if 200/dt <= i <= 300/dt  else 10 for i in range(steps)]

# TEST -- 1 
plot(neuron, time, dt, I, 1)