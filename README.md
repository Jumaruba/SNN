# SNN
Repository with several implementations of neural network models. 


## Eugene M. Izhikevich 
The code for this model was developed using the [paper](https://www.izhikevich.org/publications/spikes.pdf) written by Izhikevich with the constant variables described at the document. This is the [Code](Neurons/Izhikevich.py) for a single neuron. 
 
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




## Hodgkin Huxel 

The code for this model was developed using the [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1392413/pdf/jphysiol01442-0106.pdf) and also equations and constants from [this pdf](http://www.math.pitt.edu/~bdoiron/assets/ermentrout-and-terman-ch-1.pdf).  
This is a [code](Neurons/HH.py) for a single neuron.  

![](https://i.imgur.com/GKcwAQL.png)

## LIF

The code for this model was developed using this [website](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html), with constants choosed by us.

This is a [code](Neurons/Ric_LIF.py) for a single neuron.

![](https://i.imgur.com/vh4uGSQ.png)

## References: 

http://worldcomp-proceedings.com/proc/p2013/BIC3207.pdf  
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1392413/pdf/jphysiol01442-0106.pdf  
http://www.math.mcgill.ca/gantumur/docs/reps/RyanSicilianoHH.pdf    
http://www.math.pitt.edu/~bdoiron/assets/ermentrout-and-terman-ch-1.pdf
https://neuronaldynamics.epfl.ch/online/Ch1.S3.html
[Izhikevich Paper](https://www.izhikevich.org/publications/spikes.pdf)  
[Hodgkin Huxel Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1392413/pdf/jphysiol01442-0106.pdf)   
