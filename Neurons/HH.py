import numpy as np

''' Hodkin Huxley Neuron 

constants http://www.math.pitt.edu/~bdoiron/assets/ermentrout-and-terman-ch-1.pdf
paper https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1392413/pdf/jphysiol01442-0106.pdf
'''

class HH:
    #Constans 

    Cm      =   1.0         # uF/cm^2 
    VNa     =   50          # mV
    VK      =   -77         # mV
    Vl      =   -54.4       # mV
    gNa     =   120         # ms/cm^2
    gK      =   36          # ms/cm^2 
    gl      =   0.3         # ms/cm^2
    restV   =   -65         # mV

    def __init__(self): 
        # Init channels
        self.n = self.inf_n(self.restV)
        self.m = self.inf_m(self.restV) 
        self.h = self.inf_h(self.restV)
        self.v = self.restV 
        
    def step(self, dt, I, method=0):
        
        # Conductance
        NaCond = self.m**3*self.h*self.gNa 
        KCond  = self.n**4*self.gK 
        gCond  = self.gl 

        # Current 
        INa = NaCond * (self.v-self.VNa)
        IK  = KCond  * (self.v-self.VK)
        Il  = gCond  * (self.v-self.Vl)

        if method == 0: 
            self.solve_euler(dt, I, INa, IK, Il)
        if method == 1: 
            self.solve_rk4(dt, I, INa, IK, Il)

    # EULER --------------------------

    def solve_euler(self, dt, I, INa, IK, Il):
        dv = (I - INa - IK - Il)*dt/self.Cm 
        self.v += dv

        self.n += (self.alpha_n(self.v)*(1-self.n)-self.beta_n(self.v)*self.n)*dt
        self.m += (self.alpha_m(self.v)*(1-self.m)-self.beta_m(self.v)*self.m)*dt
        self.h += (self.alpha_h(self.v)*(1-self.h)-self.beta_h(self.v)*self.h)*dt


    # RK4 ----------------------------- 

    def solve_rk4(self, dt, I, INa, IK, Il):
        dv = self.getDv(I, dt, INa, IK, Il)
        self.v += dv 
           
        self.n += self.getDn(I, dt)
        self.m += self.getDm(I, dt)
        self.h += self.getDh(I, dt)

    def getDv(self, I, dt, INa, IK, Il):
        dv1 = (I - INa - IK - Il)/self.Cm * dt 
        dv2 = (I - INa - IK - Il + dv1*0.5)/self.Cm * dt 
        dv3 = (I - INa - IK - Il + dv2*0.5)/self.Cm * dt 
        dv4 = (I - INa - IK - Il + dv3)/self.Cm * dt 
        dv  = 1/6*(dv1 + dv2*2 + dv3*2 + dv4) 
        
        return dv 

    def getDn(self, dt):
        dn1 = (self.alpha_n(self.v)*(1-self.n)-self.beta_n(self.v)*self.n)*dt
        dn2 = (self.alpha_n(self.v)*(1-(self.n + dn1*0.5))-self.beta_n(self.v)*(self.n+dn1*0.5))*dt
        dn3 = (self.alpha_n(self.v)*(1-(self.n + dn2*0.5))-self.beta_n(self.v)*(self.n+dn2*0.5))*dt
        dn4 = (self.alpha_n(self.v)*(1-(self.n + dn3))-self.beta_n(self.v)*(self.n+dn3))*dt
        dn = 1/6*(dn1 + dn2*2 + dn3*2 + dn4) 
        
        return dn 

    def getDm(self, dt):
        dm1 = (self.alpha_m(self.v)*(1-self.m)-self.beta_m(self.v)*self.m)*dt
        dm2 = (self.alpha_m(self.v)*(1-(self.m + dm1*0.5))-self.beta_m(self.v)*(self.m + dm1*0.5))*dt
        dm3 = (self.alpha_m(self.v)*(1-(self.m + dm2*0.5))-self.beta_m(self.v)*(self.m + dm2*0.5))*dt
        dm4 = (self.alpha_m(self.v)*(1-(self.m + dm3))-self.beta_m(self.v)*(self.m + dm3))*dt
        dm = 1/6*(dm1 + dm2*2 + dm3*2 + dm4) 

        return dm 

    def getDh(self, dt):
        dh1 = (self.alpha_h(self.v)*(1-self.h)-self.beta_h(self.v)*self.h)*dt
        dh2 = (self.alpha_h(self.v)*(1-(self.h + dh1*0.5))-self.beta_h(self.v)*(self.h + dh1*0.5))*dt
        dh3 = (self.alpha_h(self.v)*(1-(self.h + dh2*0.5))-self.beta_h(self.v)*(self.h + dh2*0.5))*dt
        dh4 = (self.alpha_h(self.v)*(1-(self.h + dh3))-self.beta_h(self.v)*(self.h + dh3))*dt
        dh = 1/6*(dh1 + dh2*2 + dh3*2 + dh4)

        return dh    

    # GATES ---------------------- 
         
    # Gate n 
    def alpha_n(self, v):
        return 0.01 * (v + 55) / (1 - np.exp(-(v + 55) * 0.1) )
    def beta_n(self, v):
        return 0.125 * np.exp(-(v + 65)/80) 
    def inf_n(self, v): 
        return self.alpha_n(v)/(self.alpha_n(v) + self.beta_n(v)) 

    # Gate m 
    def alpha_m(self, v): 
        return 0.1*(v + 40)/(1- np.exp(-(v + 40)*0.1))
    def beta_m(self, v): 
        return 4*np.exp(-(v+65)/18) 
    def inf_m(self,  v): 
        return self.alpha_m(v)/(self.alpha_m(v) + self.beta_m(v)) 

    # Gate h 
    def alpha_h(self, v): 
        return 0.07*math.exp(-(v+65)/20) 
    def beta_h(self, v):
        return 1/(math.exp(-(v + 35) * 0.1) + 1)
    def inf_h(self,v):
        return self.alpha_h(v)/(self.alpha_h(v) + self.beta_h(v))



