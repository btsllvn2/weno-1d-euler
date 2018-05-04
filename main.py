# This code uses the various functions defined in euler_1d_weno.py to compute the RHS
# Then it time steps it using TVD

#============================
#   Terminology:
#  --------------
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

#options for running the code
saveFrames = True
runQuasi1D = False

#use LaTeX formatting for titles and axes
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['figure.figsize'] = (10.0,5.625)

os.system('clear')
eps = np.finfo('float').eps
gam = 1.40
R = 286.9

# Specify the number of points in the domain (includes ghost points)
N = 500

# Specify the domain size
X_min,X_max = -17.0,3.0

# Specify the Pressure and temperature values of the initial condition
P1 = 1e4
P4 = 1e5
T1 = 300
T4 = 300

#example initial conditions (Sod '78)
rho4 = 1.0
P4 = 1e5
T4 = P4/(R*rho4)
rho1 = 0.125
P1 = 1e4
T1 = P1/(R*rho1)

# [STEP 1]: Assign the initial condition (diaphram at x = 0; Ghost cells populated with large discontinuity)
q_init,X = init_cond(X_min,X_max,N,P4,T4,P1,T1)
#run the code either in quasi-1D mode or in true 1D mode
if (runQuasi1D):

    #define the shock-tube and nozzle geometry
    Geom_Dat = np.array([[-17.00000,  0.0568613],
                         [-0.445792,  0.0568613],
                         [-0.100000,  0.0568613],
                         [0.0295679,  0.0568613],
                         [0.0384761,  0.0369598],
                         [0.0538287,  0.0233131],
                         [0.0828279,  0.0159212],  #<----throat location
                         [0.131160,   0.0233131],
                         [0.203942,   0.0363912],
                         [0.292646,   0.0471948],
                         [0.405800,   0.0557240],
                         [0.543973,   0.0579985],
                         [0.700000,   0.0579985],
                         [2.000000,   0.0579985]])

    #compute scaling factor (vector) for Quasi-1D source term on the grid "X"
    F_vec = areaFunc(Geom_Dat[:,0],Geom_Dat[:,1],X[3:-3],True)
    print('Q1D shape(F_vec) = ',F_vec.shape)
else:

    #force the Q1D source term to be identically zero
    F_vec = np.zeros(X[3:-3].shape)
    print('1D shape(F_vec) = ',F_vec.shape)

#determine the time parameters
CFL = 5e-4
dx = X[1]-X[0]
dt = CFL*dx
Nt = 600
T_final = Nt*dt 

#compute the exact solution for the 1D shock-tube problem
t_exact = np.linspace(0,T_final,Nt+1)
Q_exact = Shock_Tube_Exact(X_min,X_max,X.shape[0],P4,T4,P1,T1,t_exact)

#allocate arrays for updating the solution
q = np.copy(q_init)
Q = np.zeros((q.shape[0]-6,q.shape[1],Nt+1))            #<-stored history
Q[:,:,0] = q_init[3:-3,:]
q1,q2 = np.zeros(q.shape),np.zeros(q.shape)

#real-time animation  
plt.ion()
plt.figure()
line1, = plt.plot(X[3:N-3],Q_exact[3:N-3,0,0],'-k',linewidth=1.0,label='Exact Solution')
line2, = plt.plot(X[3:N-3],q_init[3:N-3,0],'ob',label='WENO-JS')
plt.title('Quasi-1D Euler Equations Using WENO-JS (t=%2.3f[ms])' % 0.0)
plt.xlabel('x')
plt.ylabel('rho')
plt.xlim(-3,3)
plt.legend()
plt.draw()
plt.savefig('frames/frame%08d.png' % 0)
plt.show()
plt.pause(eps)

#time integration
plot_freq = 1
print('Starting time integration...')
for i in range(1,Nt+1):

    #display to terminal
    print('n = %d,    t = %2.3f[ms]' % (i,float(1000*i*dt)))
    
    # Third-order TVD Scheme (Shu '01)
    q = update_ghost_pts(X,q)
    L0 = spatial_rhs(char_numerical_flux(q),q,dx) + quasi1D_rhs(F_vec,q[3:-3,:])
    q1[3:-3,:] = q[3:-3,:] + L0*dt
    
    q1 = update_ghost_pts(X,q1)
    L1 = spatial_rhs(char_numerical_flux(q1),q1,dx) + quasi1D_rhs(F_vec,q1[3:-3,:])
    q2[3:-3,:] = (3/4)*q[3:-3,:] + (1/4)*q1[3:-3,:] + (1/4)*L1*dt
    
    q2 = update_ghost_pts(X,q2)
    L2 = spatial_rhs(char_numerical_flux(q2),q2,dx) + quasi1D_rhs(F_vec,q2[3:-3,:])
    q[3:-3,:] = (1/3)*q[3:-3,:] + (2/3)*q2[3:-3,:] + (2/3)*L2*dt    

    #update the stored history
    Q[:,:,i] = q[3:-3,:]

    #real-time animation   
    if(i%plot_freq==0):
        line1.set_ydata(Q_exact[3:N-3,0,i])
        line2.set_ydata(q[3:N-3,0])
        plt.title('Quasi-1D Euler Equations Using WENO-JS (t=%2.3f[ms])' % float(1000*i*dt))
        plt.draw()
        if (saveFrames): plt.savefig('frames/frame%08d.png' % int(i/plot_freq))
        plt.pause(eps)

plt.show(block=False)

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
    
    
    
    
    
    
    
    
