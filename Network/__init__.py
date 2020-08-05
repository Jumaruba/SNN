try : 
    import os 
    import sys 
    sys.path.append((os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Neurons"))) 
except Exception as e: 
    print("Exception of module: \n {}".format(e)) 