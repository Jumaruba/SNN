
'''Izhikevich neuron 

font: https://www.izhikevich.org/publications/spikes.pdf 
'''

class IZHI():
    def __init__(self):
        self.a = .1            
        self.b = .2
        self.c = -65
        self.d = 8

        self.thrs = 30 
        self.v = -65
        self.u = -14

    '''Step to get the next voltage 

    Parameters
    -------

    dt: float 
        Time variantion [mS]

    I: float 
        Actual current [mA]

    method: integer
        Choose the diferentiation solve method  
        0 - Euler Method
        1 - Runge Kutta 4th Order method  
    '''
    def step(self, dt, I, method=0): 

        if method != 1 and method != 0:
            print("Invalid method\n0 - EULER\n1 - RK4\n")

        if self.v >= self.thrs:
            self.v = self.c
            self.u += self.d
        else: 
            if method == 1: 
                self.solve_rk4(dt, I)
            elif method == 0: 
                self.solve_euler(dt, I)

    def solve_euler(self,dt,I): 
        dv = self.f_v(I,self.v,dt) 
        du = self.f_u(self.u,dt) 
        self.v += dv
        self.u += du

    def solve_rk4(self, dt, I):
        dv1 = self.f_v(I, self.v,dt) 
        dv2 = self.f_v(I, self.v + dv1*0.5,dt)
        dv3 = self.f_v(I, self.v + dv2*0.5,dt) 
        dv4 = self.f_v(I, self.v + dv3,dt) 
        dv  = 1/6*(dv1 + dv2*2 + dv3*2 + dv4) 
        

        du1 = self.f_u(self.u,dt) 
        du2 = self.f_u(self.u + du1*0.5,dt)
        du3 = self.f_u(self.u + du2*0.5,dt) 
        du4 = self.f_u(self.u + du3,dt) 
        du  = 1/6*(du1 + du2*2 + du3*2 + du4) 

        self.v += dv 
        self.u += du 
            
    def f_v(self, i, v, dt): 
        return (0.04 * v * v + 5 * v + 140 - self.u + i) *dt

    def f_u(self, u,dt): 
        return self.a * (self.b * self.v - u) *dt

