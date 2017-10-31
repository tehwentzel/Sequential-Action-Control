import scipy.io as sio
import scipy.integrate
import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt
import math
import time

m = 1
g = 9.8
height = 1
a_d_init = -25000
beta = .6 #weight for calculating J_tau
t_start = 0
x_start = np.array([np.pi,0,0,0])
w = 1.2
T = .3
t_calc = .003
ts = .1
default_duration = T/100
dJ_min = a_d_init/10
k_max = 50
duration_max = 2*T

integration_steps = 25 #default steps to plug into odeint, odeint default-defaults to 50

u2_max = 50
u2_min = -50

#default weights
q_default = np.diag([1000,100,0,0])
r_default = np.array([.3])
p_default = np.diag([500,0,0,0])

class Weights:
###Class to store the weight matricies that I'll be using
###requires and input x position for getting Q
    def __init__(self, p = p_default, r = r_default, q = q_default):
        self.p = p
        self.r = r
        self.q = q
        
    def getP(self):
        return(self.p)

    def getQ(self):
        return(self.q)
    
    def getR(self):
        return(self.r)
    
test = [x_start]
class FreeDynamics:

    def __init__(self,initial_position  = x_start , initial_time = t_start, final_time = t_start + T, nsolve_steps = integration_steps):
        self.initial_position = initial_position
        self.initial_time = initial_time
        self.final_time = final_time
        self.nsolve_steps = nsolve_steps #number of descrete time steps we'll find x and rho at
        self.t_array = np.linspace(self.initial_time, self.final_time, self.nsolve_steps) #get an array of time steps to evaluate x and rho at
        self.pos = self.simulateX() #gives an array of x values at discrete points in nd-array
        self.pos_func = scipy.interpolate.interp1d(self.t_array,self.pos,axis=0,fill_value='extrapolate') #gives an interpolation of the function of x(t) for use in getting rho *using default interpolation, but cubic should be better for more course meshes
        self.rho = self.simulateRho() #gives rho in disrecte steps as an nd-array
        
    def get_t0(self):
        return(self.initial_time)
        
    def get_tf(self):
        return(self.final_time)
    
    def get_x0(self):
        return(self.initial_position)
        
    def get_times(self):
        return(self.t_array)
    
    def f1(self, pos_cur, t0): ##f1 function for use with odeint and solving for x
        f1 = np.array([pos_cur[1], (g/height)*math.sin(pos_cur[0]), pos_cur[3], 0])
        return(f1)

    def simulateX(self): #get free dynamics of f in time points of ts (for now)
        #Returns the free dynamics overs the given time horizon, in discrete steps given by integartion steps       
        #variable as an (steps,len(x)) size ndarray
        pos = scipy.integrate.odeint(self.f1, self.initial_position, self.t_array)
        return(pos)

    def X(self,t): #because the internet tells me to allways use getters instead of accessing attributes
        #will return X(t) as an interpolated function
        return(self.pos_func(t))
    
    def rhodot(self, rho, t):
        rho = rho.reshape(4,1)#reshpae rho so the matrix math works -> broadcasting casues issues
        pos_cur = self.X(t)
        test.append(pos_cur)
        rhodot_1 = np.transpose( np.array([-weights.getQ()[0][0]*pos_cur[0], -weights.getQ()[1][1]*pos_cur[1], -weights.getQ()[2][2]*pos_cur[2], -weights.getQ()[3][3]*pos_cur[3] ]) )
        rhodot_1 = rhodot_1.reshape(4,1)
        rhodot_2 = np.matrix( [[0, 1, 0, 0], [(g/height)*math.cos(pos_cur[0]), 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]] )
        rhodot_2 = np.transpose(rhodot_2)
        rhodot_3 = np.matmul(-rhodot_2,rho)
        rhodot_3 = rhodot_3
        rhodot_final = np.add(rhodot_1, rhodot_3)
        rhodot_final = np.ravel(rhodot_final)
        return(rhodot_final)

    def simulateRho(self):
        #calculate rho for a given x
        #I'm going to assume p(0) = Px(0) instead of p(tf) = Px(tf)?
        p = weights.getP()
        rhof = np.dot(p,self.pos[-1]) #using a final value contraint for rho
        t_reversed = np.flipud(self.t_array) #reverse t points so we can do backwards integration
        rho = scipy.integrate.odeint(self.rhodot, rhof, t_reversed)
        assert(rho.shape == (self.pos).shape)
        return(rho)
        
class ControlledDynamics:
    
    def __init__(self, u, initial_position  = x_start , initial_time = t_start, final_time = t_start + T, nsolve_steps = integration_steps):
        self.u = u
        self.initial_position = initial_position
        self.initial_time = initial_time
        self.final_time = final_time
        self.nsolve_steps = nsolve_steps #number of discrete time steps we'll find x and rho at
        self.t_array = np.linspace(self.initial_time, self.final_time, self.nsolve_steps) #get an array of time steps to evaluate x and rho at
        self.pos = self.simulateX() #gives an array of x values at discrete points in nd-array
        self.pos_func = scipy.interpolate.interp1d(self.t_array,self.pos,axis=0,fill_value='extrapolate') #gives an interpolation of the function of x(t) for use in getting rho *using default interpolation, but cubic should be better for more course meshes
        
    def get_t0(self):
        return(self.initial_time)
        
    def get_tf(self):
        return(self.final_time)
    
    def get_x0(self):
        return(self.initial_position)
        
    def get_times(self):
        return(self.t_array)

    def f1(self, pos_cur, t0): ##f1 function for use with odeint and solving for x
        f1 = np.array([pos_cur[1], (g/height)*math.sin(pos_cur[0]) + (self.u/height)*math.cos(pos_cur[0]), pos_cur[3], self.u])
        return(f1)

    def simulateX(self): #get free dynamics of f in time points of ts (for now)
        #Returns the free dynamics overs the given time horizon, in discrete steps given by integartion steps       
        #variable as an (steps,len(x)) size ndarray
        global whatup
        pos, whatup = scipy.integrate.odeint(self.f1, self.initial_position, self.t_array, full_output = 1)
        return(pos)

    def X(self,t): #because the internet tells me to always use getters instead of accessing attributes
        #will return X(t) as an interpolated function
        return(self.pos_func(t))
    
    
class SAC:

    def __init__(self, free_system): #takes a FreeDynamics class as an argument
        self.initial_time = free_system.initial_time
        self.final_time = free_system.final_time
        self.positions = free_system.pos
        self.rhos = free_system.rho
        self.times = free_system.t_array
        self.pos_func = free_system.pos_func
        self.J1_init = self.calc_J1(self.positions, self.initial_time, self.final_time, self.pos_func)
        if(dJ_ref == 0):
            self.a_d = a_d_init
        else:
            self.a_d = max(-abs(a_d_init*((self.J1_init/dJ_ref))),a_d_init)
        self.u2s = self.schedule_u2_opt()
        (self.u2_opt, self.tau_opt) = self.get_optimal_action()
        self.u2_opt = self.saturate_u2(self.u2_opt)
        self.control_duration = self.find_Duration(default_duration)
        self.action_result = self.simulate_X_controlled(self.u2_opt, self.control_duration, self.tau_opt, False)

    def getH(self,theta, u_optional = 1): #takes the single scalar variable theta, returns (4,) np array
        h = np.array([0,math.cos(theta)/height,0,1])
        h = u_optional*h    #u_optional is an optional control for getting f2-f1 when calculating dJ/d(lambda)
        return(h)

    def getA(self,rho,pos):
        #takes a full rho vector and full position vector (4,) as arguments at a single time step<- second one can be reordered to take theta or straight up h
        h = self.getH(pos[0])
        a = np.dot(np.dot(np.dot(h.T,rho),rho.T),h)
        return(a)

    def calc_u2_opt(self,rho,pos, u_nom = 0): #calculate u2 at a single point, takes rho and X as (4,) np arrays
        alpha = self.a_d
        r = weights.getR()
        a = self.getA(rho,pos)
        h = self.getH(pos[0])
        u2 = (1/(a + r.T))*( a*u_nom + np.dot((h.T),rho)*alpha)
        return(u2)

    def schedule_u2_opt(self):
        schedule = np.zeros(self.times.shape) #array of all the u2 points along the discrete time interval we have for x and rho
        for idx in range(0,len(schedule)):
            this_pos = self.positions[idx]
            this_rho = self.rhos[idx]
            u2 = self.calc_u2_opt(this_rho,this_pos)
            schedule[idx] = u2
        return(schedule)

    def schedule_J_tau(self):
        #returns an array of J_tau values from points in our discrete region between t_calc and tf
        after_t_calc = np.where(self.times >= t_calc)[0] # gets an array of indices where we can get a valid control input
        J_taus = np.zeros(after_t_calc.shape) #J_tau will be offset in size from everything else
        for point in range(0,J_taus.size):
            idx = after_t_calc[point] #we're getting J_tau at an offset, so this is the index for all the datasets starting at t0
            this_u2 = self.u2s[idx]
            this_rho = self.rhos[idx]
            this_t = self.times[idx]
            this_theta = self.positions[idx][0] #we'll only need theta to plug into getH()
            J_tau = abs(this_u2) + (this_t - self.initial_time)**beta + np.dot(this_rho, self.getH(this_theta, this_u2))
            J_taus[point] = J_tau
        return(J_taus)

    def saturate_u2(self,u2, u2_min = u2_min, u2_max = u2_max):
        if(u2 >= 0):
            u2 = min(u2,u2_max)
        else:
            u2 = max(u2_min,u2)
        return(u2)

    def get_optimal_action(self): #get the u2 and t value to act optimally once we have u2*(t) and J_tau
        #returns a tuple of u2 and tau
        J_taus = self.schedule_J_tau() #calculate J_tau at each discretized point in our mesh
        min_point = np.argmin(J_taus) #get the point in our mesh where J_tau is smallest -> will give us the point where t = tau
        tau = self.times[min_point]
        u2_tau = self.u2s[min_point]
        return((u2_tau,tau))

    def J1_integrand(self,t,pos_func): #the callable function to integrate when finding J1
        x = pos_func(t)
        x = np.reshape(x,(4,1)) # reshape to avoid all the badness with dot products and broadcasting
        q = weights.getQ()
        term = .5*np.dot(np.dot(x.T, q), x)
        assert(term.size == 1)
        return(term)

    def calc_J1(self,dynamics,t0,tf, pos_func): #this will need an input x trajectory so we can calculate J1 when we update with controlled dynamics
        x_tf = dynamics[-1]
        x_tf = np.reshape(x_tf,(4,1))
        p = weights.getP()
        tf_term = .5 * np.dot(np.dot(x_tf.T, p), x_tf)
        integral_term, err = scipy.integrate.quad(self.J1_integrand,t0,tf,args=(pos_func),epsrel = .01) #epsrel sets the error tollerance to 1%, otherwise it maxes out the step number and returns and error message (but err ~= .5 which is more than needed)
        J1 = integral_term + tf_term
        return(J1)

    def simulate_X_controlled(self,u, control_duration, tau, include_after = True): #will calculate a new trajectory when control is applied for a given duration
        ct_start = max(tau - (control_duration/2), self.initial_time + .001)
        ct_end = min(tau + (control_duration/2), self.final_time - .001)
        before = FreeDynamics(self.positions[0], self.initial_time, ct_start)
        tbefore = before.t_array
        xbefore = before.pos
        during = ControlledDynamics(self.u2_opt, xbefore[-1], ct_start, ct_end)
        tduring = during.t_array
        xduring = during.pos
        after = FreeDynamics(xduring[-1], ct_end, self.final_time)
        tafter = after.t_array
        xafter = after.pos
        if(include_after == True):
            new_dynamics = np.vstack((xbefore,xduring,xafter))
            time_points = np.concatenate((tbefore,tduring,tafter))
        else:
            new_dynamics = np.vstack((xbefore,xduring))
            time_points = np.concatenate((tbefore,tduring))
        return((time_points, new_dynamics))

    def find_Duration(self, duration, dJ_min = dJ_min, k_max = k_max):
        k = 0
        J_new = np.inf
        while((J_new - self.J1_init > -self.J1_init*.1) and (duration < 2*T) ):
            duration = (w**k)*duration
            # if(duration > T):
            #     duration = T
            #     break
            dJ_prev = (J_new - self.J1_init)
            (new_times, new_x) = self.simulate_X_controlled(self.u2_opt, duration, self.tau_opt)
            x_func = scipy.interpolate.interp1d(new_times, new_x.T) #this is confusing because x is the y input to interp1d
            J_new = self.calc_J1(new_x, self.initial_time, self.final_time, x_func)
            if(J_new - self.J1_init > dJ_prev):
                duration = duration/(w**k)
                break
            if(J_new - self.J1_init > 0):
                duration = default_duration
                break
            k += 1
        self.J1_f = J_new
        return(duration)

def run(a_d_init,beta,w,T,default_duration,aJ_min,k_max,duration_max,integration_steps,q_default,r_default,p_default)
    m = 1
    g = 9.8
    height = 1
    # a_d_init = -25000
    # beta = .6 #weight for calculating J_tau
    t_start = 0
    x_start = np.array([np.pi,0,0,0])
    # w = 1.2
    # T = .3
    t_calc = .003
    ts = .1
    # default_duration = T/100
    # dJ_min = a_d_init/10
    # k_max = 50
    # duration_max = 2*T

    # integration_steps = 25 #default steps to plug into odeint, odeint default-defaults to 50

    u2_max = 50
    u2_min = -50

    #default weights
    # q_default = np.diag([1000,100,0,0])
    # r_default = np.array([.3])
    # p_default = np.diag([500,0,0,0])


    weights = Weights()
    system = FreeDynamics()
    # t_hist = []
    # x_hist = []
    J_tot = 0
    t0 = system.get_t0()
    x0 = system.initial_position
    dJ_ref = 0
    while t0 < 15:
        freeSys = FreeDynamics(x0,t0,t0+T)
        control = SAC(freeSys)
        if(t0 == t_start):
            dJ_ref = control.J1_init
        t_curr = control.action_result[0]
        x_curr = control.action_result[1]
        # t_hist.append(t_curr)
        # x_hist.append(x_curr)
        J_tot += control.J1_f
        t0 = t_curr[-1]
        x0 = x_curr[-1]
    return(J_tot,control.J1_f)

def timecost(min_J):
    t0 = time()
    J,Jf = run(ta_d_init,tbeta,tw,tT,tdefault_duration,taJ_min,tk_make,tduration_max,tintegration_steps,tq_default,tr_default,tp_default)
    tf = time() - t0
    return(J*tf*Jf)
