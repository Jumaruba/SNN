# SNN
Repository with several implementations of neural network models. 

## Neurons implemented 
Variation of voltage can be obtained with [Runge Kutta Fourth Method](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) or [Euler Method](https://en.wikipedia.org/wiki/Euler_method).  

|Neuron |  Implementation |Tests |Results | 
|---|---|---|---|
|Hodking Huxley|[HH.py](Neurons/HH.py) | [HH-test.py](Neurons/test/HH-test.py) | [HH](Neurons/test/expected-results/HH/)  |   
|Izhikevich |  [IZHI.py](Neurons/IZHI.py) | [IZHI-test.py](Neurons/test/IZHI-test.py) | [IZHI](Neurons/test/expected-results/IZHI/) | 
|Leaky Integrate and Fire |  [LIF.py](Neurons/LIF.py)  | [LIF-test.py](Neurons/test/LIF-test.py )|[LIF](Neurons/test/expected-results/LIF) |
|Leaky Integrate and Fire - Synapses |  [LIF_Synapse.py](Neurons/LIF_synapse.py)  | [LIF_Synapse-test.py](Neurons/test/LIF_synapse-test.py) |[LIF_SYN](Neurons/test/expected-results/LIF_SYN/) |

## Demonstration of result - Izhikevich Neuron 
The code for this model was developed using the [paper](https://www.izhikevich.org/publications/spikes.pdf) written by Izhikevich with the constant variables described at the document. This is the [Code](Neurons/Izhikevich.py) for a single neuron. The threshold used was -30 mV. 
 
The parameter used to generate these graphs were:         
```python
        a = 0.1 or 0.02   # How fast the recovery is. It's proportional to the frequency of spikes for a constant input. 
        b = .2            # Sensitivity of the variable u to the subthreshold membrane fluctuation 
        c = -65           # After spike reset value of the voltage
        d = 8             # After spike reset value of the u variable 
```

This graph shows the fast and regular spiking with constant `a = 0.1`
![](https://i.imgur.com/bylBPGF.png)

This graph show the slow and regular spiking with constant `a = 0.02`
![](https://i.imgur.com/0QSoXWK.png)

# Izhikevich Random Network 
[Code](https://github.com/Jumaruba/SNN/blob/master/Network/IZV_NN.py)   
[Paper](https://www.izhikevich.org/publications/spikes.pdf)  
![](https://i.imgur.com/bYqNj2L.png)

## References: 

http://worldcomp-proceedings.com/proc/p2013/BIC3207.pdf  
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1392413/pdf/jphysiol01442-0106.pdf  
http://www.math.mcgill.ca/gantumur/docs/reps/RyanSicilianoHH.pdf    
http://www.math.pitt.edu/~bdoiron/assets/ermentrout-and-terman-ch-1.pdf
https://neuronaldynamics.epfl.ch/online/Ch1.S3.html
[Izhikevich Paper](https://www.izhikevich.org/publications/spikes.pdf)  
[Hodgkin Huxel Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1392413/pdf/jphysiol01442-0106.pdf)   
