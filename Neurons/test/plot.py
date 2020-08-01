from __init__ import * 

def plot(neuron, time, dt, I, method):

    # build the v and u vector
    steps = math.ceil(time/dt)
    v = np.zeros(steps)

    v[0] = neuron.v
 
    for i in range(steps): 
        neuron.step(dt, I[i], method)
        v[i] = neuron.v 

    vTime = np.arange(0, time, dt, dtype=None)
    plt.plot(vTime, v, color='b', label="potential")
    plt.plot(vTime, I, color='r', label="current")
    plt.title("Single neuron stimulation")
    plt.xlabel("Time [ms]")
    plt.ylabel("Voltage [mv]")
    plt.savefig("Fast spiking")
    plt.show()

