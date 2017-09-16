# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sm
import control as ct
from sympy import *
from sympy.physics.mechanics import *

#cart setup
m = 1 #kg
g = 9.81 #m/s2
h = 2 #meters, length of the arm


###set system constants for the cart:
alpha_d = -1000 #desired sensitivity of cost function
delta_J_min = 0 #minimum change in cost 
tcurr = 0 #current time, starting at 0 (seconds)
delta_t_init = .04 #default control duration (seconds), .1 is arbitrary
u_nom = 0 #nominal control
w = .5 #scale factor?
pred_horizon = 1.5 #predicition horizon
t_s = .1 #sampling time, I think this corresponds to 10Hz
t_calc = .5 #max time to do calculations
k_max = 10 #max backtracking iterations
i = 0 #action iteration

###Control weight matrixes and symbolic things
u_opt = symbols("u*")
theta = symbols("theta")
theta_dot = symbols("w")
theta_dot_dot = symbols("alpha")
p = symbols("p") #cart position
p_dot = symbols("v")
p_dot_dot = symbols("a")
t_opt = symbols("t_opt")
t_f = symbols("tf")
u_curr = symbols("u_curr") #the current control, for use during simlulations?
J_1 = symbols("J_1", cls=Function) #original cost funtional
J_2 = symbols("J_2", cls=Function) #cost functional for something maybe
J_t = symbols("J_t", cls=Function) #cost funtional for 
h_1 = symbols("h", cls=Function)
rho = symbols("rho", cls=Function) #adjoint variable
x = symbols("X", cls=Function) #system dynamics
f_1 = symbols("f_1", cls=Function) #bascially x_dot after control input
f_2 = symbols("f_2",cls=Function)
triangle = symbols("A", cls=Function)

##Decide on control weight matrices
beta = 1.6 #weight factor for importance of time calculating J_tau
q_mat = diag(200,0,(u_curr/2)**8,5)
p_mat = zeros(4,4)
r_mat = Matrix([.3])

#put stuff that you can solve for at this point here
f_2 = Matrix([theta_dot, (g/h)*sin(theta)+u_curr*cos(theta)/h, p_dot, u_curr]) #obtain system dynamics
f_1 = f_2.subs({u_curr:u_nom}) #swap in for f_1 with nominal control
x = Matrix([theta, theta_dot, p, p_dot])
rho = pdsolve( diff(rho, t_curr) = (q_mat*x).T - diff(f_1,p).T * rho, rho )###this doesnt work - how to make derivative of f1 wrt x (matrix by vector)

print(rho)