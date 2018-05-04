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
from time import sleep
import sys,os

# Import library functions
from euler_1d_weno import *

os.system('clear')
eps = np.finfo('float').eps
gam = 7.0/5.0
R = 286.9

# Specify the number of points in the domain (includes ghost points)
N = 500

# Specify the domain size
X_min,X_max = -17.0,3.0

# Step size
dx = (X_max-X_min)/(N-1)

# Choose CFL
CFL = 5e-4

# Select the time step
dt = CFL*dx

# Select total test time
#Total_Test_Time = 5e-3

# Number of time steps
#Nt = round((Total_Test_Time)/dt)
Nt = 600

#final test time
T_final = Nt*dt 

# Specify the Pressure and temperature values of the initial condition
P1 = 1e4
P4 = 1e5
T1 = 300
T4 = 300

#example initial conditions (Sod '78)
rho4 = 1
P4 = 1e5
T4 = P4/(R*rho4)
rho1 = 0.125
P1 = 1e4
T1 = P1/(R*rho1)

#f_num = 1
#q1d_afunc(1,1,True,True)


# [STEP 1]: Assign the initial condition (diaphram at x = 0; Ghost cells populated with large discontinuity)
q_init,X = init_cond(X_min,X_max,N,P4,T4,P1,T1)

#compute the exact solution for the 1D shock-tube problem
t_exact = np.linspace(0,T_final,Nt+1)
Q_exact = Shock_Tube_Exact(X_min,X_max,X.shape[0],P4,T4,P1,T1,t_exact)

q = np.copy(q_init)
Q = np.zeros((q.shape[0]-6,q.shape[1],Nt+1))            #<-stored history
Q[:,:,0] = q[3:-3,:]
q1,q2 = np.zeros(q.shape),np.zeros(q.shape)
print('Starting time integration...')

#real-time animation  
plt.ion()
plt.figure()
plt.show()
line1, = plt.plot(X[3:N-3],Q_exact[3:N-3,0,0],'-k',linewidth=1.0,label='Exact Solution')
line2, = plt.plot(X[3:N-3],q_init[3:N-3,0],'ob',label='WENO-JS')
plt.title('Quasi-1D Euler Equations Using WENO-JS (t=%2.3f[ms])' % 0.0)
plt.xlabel('x')
plt.ylabel('rho')
plt.xlim(-3,3)
plt.legend()
plt.draw()
plt.pause(eps)

#start the time integration
plot_freq = 10 
for i in range(1,Nt+1):

    print('n = %d,    t = %2.6f [ms]' % (i,float(1000*i*dt)))
    
    # Third-order TVD Scheme (Shu '01)
    q = update_ghost_pts(X,q)
    L0 = spatial_rhs(char_numerical_flux(q), q, dx)
    q1[3:-3,:] = q[3:-3,:] + dt*L0
    
    q1 = update_ghost_pts(X,q1)
    L1 = spatial_rhs(char_numerical_flux(q1), q1, dx)
    q2[3:-3,:] = (3/4)*q[3:-3,:] + (1/4)*q1[3:-3,:] + (1/4)*dt*L1
    
    q2 = update_ghost_pts(X,q2)
    L2 = spatial_rhs(char_numerical_flux(q2), q2, dx)
    q[3:-3,:] = (1/3)*q[3:-3,:] + (2/3)*q2[3:-3,:] + (2/3)*dt*L2    

    #update the stored history
    Q[:,:,i] = q[3:-3,:]

    #real-time animation   
    if(i%plot_freq==0):
        line1.set_ydata(Q_exact[3:N-3,0,i])
        line2.set_ydata(q[3:N-3,0])
        plt.title('Quasi-1D Euler Equations Using WENO-JS (t=%2.3f[ms])' % float(1000*i*dt))
        plt.draw()
        plt.pause(eps)

fig, ax = plt.subplots()
line, = ax.plot(X[3:N-3],Q[:,0,0], color='b', marker='o', linewidth=2)

#ax.grid(ydata=[0], color='b', linestyle='-', linewidth=1)
plt.xlabel(r'$x$')
plt.ylabel(r'$\rho(x,t)$')
plt.xlim(-2.0,2.0)
plt.title(r'Density Animation')
def animate(n):
    line.set_ydata(Q[:,0,n])
    return line,
ani = animation.FuncAnimation(fig, animate, np.arange(1, Nt),interval=20, blit=True)
plt.show()  
    
    
    
    
    
    
    
    
