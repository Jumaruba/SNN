try: 
    import sys 
    import os 
    import math 
    import numpy as np 
    import matplotlib.pyplot as plt
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except Exception as e: 
    print("Some Modules are missing -- \n{}".format(e)) 