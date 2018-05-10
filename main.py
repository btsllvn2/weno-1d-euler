#======================================================================
#
#   A Python code which utilizes the various functions supplied in
#   "euler_1d_weno.py" to numerically solve the unsteady, Quasi-1D
#   Euler equations using a finite-difference formulation of the 5th-
#   order WENO scheme by Jiang and Shu[1]. Boundary conditions are 
#   of the characteristic type by Thompson[2]. Time-stepping is via
#   the third-order TVD Runge-Kutta scheme suggested by Shu[3].
#
#   B. Sullivan and S. R. Murthy, 2018
#
#   [1] "Efficient Implementation of Weighted ENO Schemes"
#   Guang-Shan Jiang and Chi-Wang Shu
#   JCP, 126(1): 202-228, 1996
#
#   [2] "Time-dependent Boundary Conditions for Hyperbolic Systems II"
#   Kevin M. Thompson
#   JCP, 89(2): 439-461, 1990
#
#   [3] "Essentially Non-Oscillatory and Weighted Essentially Non-
#    Oscillatory Schemes for Hyperbolic Conservation Laws"
#   Chi-Wang Shu
#   NASA/CR-97-206253, ICASE Report No. 97-65, 1997
#
#   Terminology:
#   q       = Physical state variables
#   f       = Physical flux terms
#   q_char  = Charateristic form of the state variables
#   f_char  = Charateristic form of the flux terms
#
#======================================================================

#Import essential library functions
from euler_1d_weno import *
import scipy.linalg as la 
import numpy as np
import sys,os

#============================================================
#
#   General options for running the code:
#   
#   Advection  = Run using either WENO or 5th-order linear
#   runQuasi1D = Run with Q1D rhs source term enabled
#   saveFrames = Frames from solution are saved to disk
#   fixedCFL   = Run the code with a constant fixed CFL value
#   plot_freq  = Frequency of making/saving plots
#
#============================================================
Adv_Options = ['LINEAR-FD','WENO']
Advection  = Adv_Options[1]
runQuasi1D = True
saveFrames = False
fixedCFL   = False
plot_freq  = 1

#============================================================
#
#   Set the left and right boundary conditions:
# 
#============================================================
BC_Options = ['Non-Reflecting','Neumann','Wall','Force-Free']
left_bc  = BC_Options[0]
right_bc = BC_Options[0]

#supress the display output so code runs faster
if (saveFrames):
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

#create a 'frames' directory if one does not already exist
if not os.path.exists('frames'):
    os.makedirs('frames')

#use LaTeX formatting for titles and axes
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['figure.figsize'] = (10.0,5.625)
os.system('clear')
eps = np.finfo('float').eps

# Specify the number of points in the domain (includes ghost points)
N = 100

# Specifiy target CFL and total number of steps
CFL = 0.5
Nt = 700

# Assign fixed, user-specified dt if not in CFL mode
if(not fixedCFL):
    dt = float(input('What fixed timestep dt should be used? (s) '))
    print('Using a fixed timestep of dt = %1.6e seconds.' % dt)

# initial conditions for specific run mode
gam = 1.4; R = 286.9
if (runQuasi1D):

    # Specify the overall domain size
    X_min,X_max = -0.40,0.60

    #initial location of the discontinuity [m]
    x0 = 0.0

    #set the initial condition
    rho4 = 1.0
    P4 = 1e6
    T4 = P4/(R*rho4)
    rho1 = 0.125
    P1 = 1e4
    T1 = P1/(R*rho1)
    q_init,X,dx = init_cond(X_min,X_max,N,P4,T4,P1,T1,x0)

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
    F_vec,x_throat,Mach_vec = areaFunc(Geom_Dat[:,0],Geom_Dat[:,1],X[3:-3],True)
    
else:

    # Specify the overall domain size
    X_min,X_max = -0.60,0.60
    x0 = 0.0

    #set the initial condition
    rho4 = 1.0
    P4 = 1e6
    T4 = P4/(R*rho4)
    rho1 = 0.125
    P1 = 1e4
    T1 = P1/(R*rho1)
    q_init,X,dx = init_cond(X_min,X_max,N,P4,T4,P1,T1,x0)

    #force Q1D source term to be identically zero
    F_vec = np.zeros(X[3:-3].shape)


#allocate arrays for computing and storing the solution
q = np.copy(q_init)
Q = np.zeros((q.shape[0]-6,q.shape[1],Nt+1))
state_variable_of_interest = np.zeros((q.shape[0]-6,Nt+1)) 
Q[:,:,0] = q_init[3:-3,:]
q1,q2 = np.zeros(q.shape),np.zeros(q.shape)
rho_plt = q[3:-3,0]
u_plt = q[3:-3,1]/rho_plt
e_plt = q[3:-3,2]
p_plt = (gam-1.0)*(e_plt-0.5*rho_plt*u_plt**2)
M_plt = np.sqrt(rho_plt*u_plt**2/(gam*p_plt))
P_41 = int(P4/P1)

#initialize frame for real-time animation  
plt.ion()
plt.figure()
if(runQuasi1D):
    print('Shock tube operating pressure ratio P_41 = %d' % P_41)
    plt.title('Solution to GALCIT Nozzle Flow Using WENO-JS ($P_{41}$=%d, t=%2.3f[ms])' %(P_41,0.0),fontsize=15)
    line1, = plt.plot(X[3:N-3],Mach_vec,'--k',label='Isentropic Solution',linewidth=1.0)
    line2, = plt.plot(X[3:N-3],M_plt,'-b',label='WENO-JS',linewidth=3.0)
    plt.legend(loc=2,fontsize=12)
    plt.ylabel('Mach',fontsize=15)
    plt.ylim(0,5.0)
else:
    Q_exact,M_sh = Shock_Tube_Exact(X,P4,T4,P1,T1,0.0,x0,1.0)
    line1, = plt.plot(X[3:N-3],Q_exact[3:-3,0],'-k',linewidth=1.0,label='Exact Solution')
    line2, = plt.plot(X[3:N-3],q_init[3:N-3,0],'ob',label='WENO-JS')
    plt.title("Sod's Shock Tube Problem ($P_{41}$=%d, t=%2.3f[ms])" % (P_41,0.0),fontsize=15)
    plt.ylabel('Density',fontsize=15)
    plt.legend(fontsize=12)
plt.xlim(X_min,X_max)
plt.xlabel(r'$x$[m]',fontsize=15)
plt.draw()
plt.savefig('frames/frame%08d.png' % 0)
plt.pause(eps)

#perform the time integration
t_vec = np.zeros(Nt+1)
print('\n=====================================================')
print('   Selected advection scheme is: %s' % Advection)
print('   Left-end boundary condition is: %s' % left_bc)
print('   Right-end boundary condition is: %s' % right_bc)
print('=====================================================\n')
print('Performing time integration with Nt=%d total steps...' % Nt)
for i in range(1,Nt+1):

    #set timestep based on target CFL
    ws_max = np.max(euler_1d_wavespeed(q))
    if(fixedCFL):
        dt = CFL*dx/ws_max
    else:
        CFL = ws_max*dt/dx

    #update the time history
    t_vec[i] = t_vec[i-1] + dt

    #display to terminal
    print('n = %d,  CFL = %1.2f,  dt = %1.2es,  t = %1.2es' % (i,CFL,dt,t_vec[i]))
    
    # Third-order TVD RK Scheme (Shu '97)
    #======================================
    q = update_ghost_pts(q,left_bc,right_bc)
    L0 = spatial_rhs(char_numerical_flux(q,Advection),q,dx,left_bc,right_bc) + \
         q1d_rhs(F_vec,q[3:-3,:],left_bc,right_bc)
    q1[3:-3,:] = q[3:-3,:] + L0*dt
    
    q1 = update_ghost_pts(q1,left_bc,right_bc)
    L1 = spatial_rhs(char_numerical_flux(q1,Advection),q1,dx,left_bc,right_bc) + \
         q1d_rhs(F_vec,q1[3:-3,:],left_bc,right_bc)
    q2[3:-3,:] = (3/4)*q[3:-3,:] + (1/4)*q1[3:-3,:] + (1/4)*L1*dt
    
    q2 = update_ghost_pts(q2,left_bc,right_bc)
    L2 = spatial_rhs(char_numerical_flux(q2,Advection),q2,dx,left_bc,right_bc) + \
         q1d_rhs(F_vec,q2[3:-3,:],left_bc,right_bc)
    q[3:-3,:] = (1/3)*q[3:-3,:] + (2/3)*q2[3:-3,:]  + (2/3)*L2*dt    
    #======================================

    #update the stored history
    Q[:,:,i] = q[3:-3,:]

    #compute and store the primitive variables
    rho_plt = q[3:-3,0]
    u_plt = q[3:-3,1]/rho_plt
    e_plt = q[3:-3,2]
    p_plt = (gam-1.0)*(e_plt-0.5*rho_plt*u_plt**2)
    M_plt = np.sqrt(rho_plt*u_plt**2/(gam*p_plt))

    #real-time animation 
    if(i%plot_freq==0):
        if(runQuasi1D): 
            plt.title('Solution to GALCIT Nozzle Flow Using WENO-JS ($P_{41}$=%d, t=%2.3fms)' % (P_41,1e3*t_vec[i]),fontsize=15)
            line2.set_ydata(M_plt)
        else:
            #compute the exact solution for the 1D shock-tube problem
            Q_exact,M_sh = Shock_Tube_Exact(X,P4,T4,P1,T1,t_vec[i],x0,M_sh)
            rho_ex = Q_exact[3:-3,0]
            u_ex = Q_exact[3:-3,1]/rho_ex
            e_ex = Q_exact[3:-3,2]
            p_ex = (gam-1.0)*(e_ex-0.5*rho_ex*u_ex**2)
            M_ex = np.sqrt(rho_ex*u_ex**2/(gam*p_ex))
            line1.set_ydata(rho_ex)
            line2.set_ydata(q[3:N-3,0])
            plt.title("Sod's Shock Tube Problem ($P_{41}$=%d, t=%2.3fms)" % (P_41,1e3*t_vec[i]),fontsize=15)
        if (saveFrames): plt.savefig('frames/frame%08d.png' % int(i/plot_freq))
        plt.pause(eps)

print('\nSolution computed. Building xt-plots...')

#compute XT-flow variables for plotting
plt.ioff()
R = 286.9
RHO = Q[:,0,:].T
U = Q[:,1,:].T/RHO
E = Q[:,2,:].T
P = (gam-1)*(E-0.5*RHO*U**2)
M = np.sqrt(RHO*U**2/(gam*P))
ENTROPY = (P/P4)/(RHO/rho4)**gam
TEMP = P/(R*RHO)
f_num = 3

#generate XT-plots
x_vec = X[3:-3]
X,T = np.meshgrid(x_vec,1e3*t_vec)
def make_XT_plot(var,v_label):

    #used outside the function
    global f_num

    #conditional formatting
    if (runQuasi1D):
        title_str = r'GALCIT Ludwieg Tube ($P_{41}$=%d)' % P_41
    else:
        title_str = r"Sod's Shock Tube Problem Tube ($P_{41}$=%d)" % P_41

    plt.figure(f_num)
    plt.contourf(X,T,var,300,cmap='jet')
    plt.title(title_str,fontsize=12)
    plt.xlabel(r'$x$ [m]',fontsize=12)
    plt.ylabel(r'Time [ms]',fontsize=12)
    cb = plt.colorbar()
    cb.set_label(v_label, labelpad=20, rotation=-90)
    plt.savefig('fig_%d.png' % f_num)
    f_num += 1

    return

#make an xt-plot for each variable
var_lst = [RHO,P,TEMP,M,U,1e-6*E,ENTROPY]
label_lst = [r'Density [kg/$m^3$]',r'Pressure [Pa]',r'Temperature [K]',r'Mach [-]',r'Velocity [m/s]',r'Specific Energy [MJ/kg]',r'Measure of Entropy']
for i in range(len(var_lst)): make_XT_plot(var_lst[i],label_lst[i])

plt.show()  
print('Program complete.\n')
