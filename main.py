# This code uses the various functions defined in euler_1d_weno.py to compute the RHS
# Then it time steps it using TVD

#============================
#   Terminology:
#
#   q       -> Physical state variables
#   f       -> Physical flux terms
#   q_char  -> Charateristic form of the state variables
#   f_char  -> Charateristic form of the flux terms
#============================

# Import the libraries that you need
import numpy as np
import scipy.linalg as la 
import matplotlib.pyplot as plt
import sys,os

# Import the helper functions
from euler_1d_weno import *

os.system('clear')

# Specify the number of points in the domain (includes ghost points)
N = 100

# Specify the domain size
X_min = -17
X_max = 2

# Step size
dx = (X_max-X_min)/(N-1)

# Choose CFL
CFL = 0.5

# Select the time step
dt = CFL*dx

# Select total test time
Total_Test_Time = 1e-3

# Number of time steps
NT = (Total_Test_Time)/dt

# Specify the Pressure and temperature values of the initial condition
P1 = 1e5
P4 = 70e5
T1 = 300
T4 = 300

# [STEP 1]: Assign the initial condition (diaphram at x = 0; Ghost cells populated with large discontinuity)
q,X = init_cond(X_min,X_max,N,P4,T4,P1,T1)

# --------------------------------------------------
plt.figure(1)
plt.plot(X[3:N-3],q[3:N-3,0],'-b',linewidth=3.5)
#plt.show()
# --------------------------------------------------


# [STEP 3]: Compute the numerical charecteristic flux at the half points
f_char_i_p_half = char_numerical_flux(q)


# [STEP 4]: Transform the characteristic flux to numerical flux
qdot_cons = spatial_rhs(f_char_i_p_half, q, dx)

# --------------------------------------------------

plt.figure(2)
plt.plot(X[3:N-3],qdot_cons[:,0],'-ob',linewidth=3.5)
plt.show()

# --------------------------------------------------

# Test derivative approximation

# plt.figure(3)
# plt.plot(x_i_p_half,f_i_p_half[:,0],'-b',linewidth=3.5)
# plt.show()

# for i in range(0,NT)

    # # [STEP 2]: TVD time stepping
    # q = TVD_Third_Order_Time_Stepper(q, dt)
    
    # #assign the ghost cell values
    # for i in range(3):
        # q_init[i,0] =  q_init[6-i,0]
        # q_init[i,1] = -q_init[6-i,1]
        # q_init[i,2] =  q_init[6-i,2]
        # # q_init[N-1-i,0] =  q_init[N-6+i,0]
        # # q_init[N-1-i,1] = -q_init[N-6+i,1]
        # # q_init[N-1-i,2] =  q_init[N-6+i,2]
    
    # # #assign the ghost cell values
    # # q_init[0,:] = ((10*3)**10)
    # # q_init[1,:] = ((10*2)**10)
    # # q_init[2,:] = ((10*1)**10)
    
    # q_init[N-1,:] = ((10*3)**10)
    # q_init[N-2,:] = ((10*2)**10)
    # q_init[N-3,:] = ((10*1)**10)


