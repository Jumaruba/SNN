

class LIF():
    def __init__(self):
        self.R = 5  
        self.C = 3
        self.uR = -40
        self.thrs = 30
        self.maxV = 90
        self.v = -65

    def step(self, I, dt, method=0): 
        
        if self.v >= self.thrs:
            self.v = self.uR
        else: 
            if method == 1: 
                self.solve_rk4(dt, I)
            elif method == 0: 
                self.solve_euler(dt, I)

            if self.v >= self.thrs: 
                self.v = self.maxV 

    def solve_euler(self, I, dt):        
        dv = self.fu(self.v, I) * dt 
        self.v += dv 

    def solve_rk4(self, I, dt):
        dv1 = self.fu(self.v, I) * dt
        dv2 = self.fu(self.v + dv1 * 0.5, I) * dt
        dv3 = self.fu(self.v + dv2 * 0.5, I) * dt
        dv4 = self.fu(self.v + dv3, I) * dt
        dv = 1 / 6 * (dv1 + dv2 * 2 + dv3 * 2 + dv4)
        self.v += dv


    def fu(self, v, I):
        return (-v + self.R * I) / (self.R * self.C)



