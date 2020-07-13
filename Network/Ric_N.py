import numpy as np


class Network:
    def __init__(self, size, neuron):
        self.neuron = neuron
        self.size = size

    def fire(self):
        ne = 800
        ni = 200
        re = np.random.rand(ne, 1)
        ri = np.random.rand(ni, 1)
        a = [0.02 * np.ones((ne, 1)), 0.02 + 0.08 * ri]
        b = [0.2 * np.ones((ne, 1)), 0.25 - 0.05 * ri]
        c = np.array([-65 + 15 * re**2, -65 * np.ones((ni, 1))])
        d = np.array([8 - 6 * re**2, 2 * np.ones((ni, 1))])
        S = [0.5 * np.random.rand(ne + ni, ne), -np.random.rand(ne + ni, ni)]
        v = -65 * np.ones((ne + ni, 1))                         # Initial values of v
        u = np.multiply(b, v)                                   # Initial values of u
        firings = []                                            # Spike timings

        for t in range(1, 1001):                                                     # Simulation of 1000 ms
            i = [5 * np.random.randn(ne, 1), 2 * np.random.randn(ni, 1)]            # Thalamic input
            fired = np.array([i_v for i_v, vi in enumerate(v) if vi >= 30])                   # indices of spikes
            firings = [firings,   t + 0*fired , fired]                                 # ????
            v[fired] = c[fired]
            u[fired] = u[fired] + d[fired]
            i = i + sum(S[:, fired], 2)
            v = v + 0.5 * (0.04 * v ** 2 + 5 * v + 140 - u + i)        # step 0.5ms
            v = v + 0.5 * (0.04 * v ** 2 + 5 * v + 140 - u + i)       # for numerical
            u = u + np.multiply(a, np.multiply(b, v) - u)             # stability
