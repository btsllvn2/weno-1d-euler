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

# --------------------------------------------------

x_i_p_half = (X[2:X.shape[0]-3] + X[3:X.shape[0]-2])*0.5

plt.figure(2)
plt.plot(x_i_p_half[0:-1],f_char_i_p_half[0:-1,2],'-b',linewidth=3.5)
plt.show()

# --------------------------------------------------

# [STEP 4]: Transform the characteristic flux to numerical flux
f_i_p_half = proj_to_cons(f_char_i_p_half, q)

# Test derivative approximation

# plt.figure(3)
# plt.plot(x_i_p_half,f_i_p_half[:,0],'-b',linewidth=3.5)
# plt.show()

# for i in range(0,NT)

    # # [STEP 2]: TVD time stepping
    # q = TVD_Third_Order_Time_Stepper(q, dt)
    



