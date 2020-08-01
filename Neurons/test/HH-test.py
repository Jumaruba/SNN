from __init__ import * 
from plot import plot

'''Test EULER METHOD

There's unity test, it's necessary to compare the output image with the image from the folder expected-results

Parameters for test: 

    Cm      =   1.0         # uF/cm^2 
    VNa     =   50          # mV
    VK      =   -77         # mV
    Vl      =   -54.4       # mV
    gNa     =   120         # ms/cm^2
    gK      =   36          # ms/cm^2 
    gl      =   0.3         # ms/cm^2
    restV   =   -65         # mV
'''

neuron = HH() 
time = 500
dt = 0.01
steps = math.ceil(time / dt)
I = [0 if 200/dt <= i <= 300/dt  else 10 for i in range(steps)]

# Comment the test you do not wish to do 

# TEST : EULER
plot(neuron, time, dt, I, 0)

#TEST : RK4
# plot(neuron, time, dt, I, 1)