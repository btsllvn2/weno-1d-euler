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
import matplotlib.animation as animation
import sys,os

# Import the helper functions
from euler_1d_weno import *

os.system('clear')

# Specify the number of points in the domain (includes ghost points)
N = 300

# Specify the domain size
X_min,X_max = -2,2

# Step size
dx = (X_max-X_min)/(N-1)

# Choose CFL
CFL = 0.0005

# Select the time step
dt = CFL*dx

# Select total test time
Total_Test_Time = 5e-3

# Number of time steps
#Nt = round((Total_Test_Time)/dt)
Nt = 1200

# Specify the Pressure and temperature values of the initial condition
P1 = 1e5
P4 = 70e5
T1 = 300
T4 = 300

# [STEP 1]: Assign the initial condition (diaphram at x = 0; Ghost cells populated with large discontinuity)
q_init,X = init_cond(X_min,X_max,N,P4,T4,P1,T1)
i_perturb = 0
for i in range(N):
    if(X[i]>-0.015 and X[i]<0.015):
        i_perturb = i

print('intiial condition generated')
q = np.copy(q_init)
Q = np.zeros((q.shape[0]-6,q.shape[1],Nt+1))            #<-stored history
state_variable_of_interest = np.zeros((q.shape[0]-6,Nt+1)) 
Q[:,:,0] = q[3:-3,:]
q1,q2 = np.zeros(q.shape),np.zeros(q.shape)
print('Entering time loop...')
for i in range(1,Nt):

    
    print('n = %d, t = %2.6f [ms]' % (i,i*dt))
    
    # Third-order TVD Scheme (Shu '01)
    q = update_ghost_pts(X,q)
    L0 = spatial_rhs(char_numerical_flux(q), q, dx, X)
    q1[3:-2,:] = q[3:-2,:] + dt*L0
    
    q1 = update_ghost_pts(X,q1)
    L1 = spatial_rhs(char_numerical_flux(q1), q1, dx, X)
    q2[3:-2,:] = (3/4)*q[3:-2,:] + (1/4)*q1[3:-2,:] + (1/4)*dt*L1
    
    q2 = update_ghost_pts(X,q2)
    L2 = spatial_rhs(char_numerical_flux(q2), q2, dx, X)
    q[3:-2,:] = (1/3)*q[3:-2,:] + (2/3)*q2[3:-2,:] + (2/3)*dt*L2    

    #update the stored history
    Q[:,:,i] = q[3:-3,:]
    state_variable_of_interest[:,i] = eval_dp(q[3:-3,:])

fig, ax = plt.subplots()
line, = ax.plot(X[3:N-3],Q[:,0,0], color='b', marker='o', linewidth=2)
plt.xlim(-2.0,2.0)

def animate(n):
    line.set_ydata(Q[:,0,n])
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(1, Nt),interval=20, blit=True)
plt.show()  
    
    
    
    
    
    
    
    