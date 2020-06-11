#======================================================================
#
#   A Python code which utilizes the various functions supplied in
#   "euler_1d_weno.py" to numerically solve the unsteady, Quasi-1D
#   Euler equations using a finite-difference formulation of the 5th-
#   order WENO scheme by Jiang and Shu[1]. Boundary conditions are 
#   of the characteristic type by Thompson[2]. Time-stepping is via
#   the third-order TVD Runge-Kutta scheme suggested by Shu[3]. 
#   Tested only in Python3.
#
#   B. Sullivan and S. R. Murthy
#   University of Illinois at Urbana-Champaign
#   May 2018
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
from scipy.io import FortranFile
from euler_1d_weno import *
import scipy.linalg as la 
import numpy as np
import sys,os

#============================================================
#
#   General options for running the code:
#   
#   Advection  = Run using either WENO or 5th-order linear
#   runMode    = Run in 1D, Q1D, or Axisymmetric modes
#   saveBCData = Save the solution to use as an imposed bc
#   saveFrames = Frames from solution are saved to disk
#   fixedCFL   = Run the code with a constant fixed CFL value
#   plot_freq  = Frequency of making/saving plots
#
#============================================================
Run_Mode_Options = ['1D','Quasi-1D','Axisymmetric']
Adv_Options = ['WENO','LINEAR-FD']
runMode     = Run_Mode_Options[0]
Advection   = Adv_Options[0]
runQuasi1D  = True
saveBCData  = False
saveFrames  = False #True
fixedCFL    = True
useLaTeX    = True
plot_freq   = 1

# Specify the number of points in the domain (includes ghost points)
Nx = 300

# Specifiy target CFL and total number of steps
CFL = 0.5; Nt = 1000+2

#============================================================
#
#   Set the left and right boundary conditions:
# 
#============================================================
BC_Options = ['Non-Reflecting','Neumann','Wall','Force-Free']
left_bc  = BC_Options[0]; right_bc = BC_Options[0]

#supress the display output so code runs faster
if (saveFrames):
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

os.system('clear')

#create a 'frames' directory if one does not already exist
if not os.path.exists('frames'):
    os.makedirs('frames')

#use LaTeX formatting for titles and axes
if (useLaTeX):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
plt.rcParams['figure.figsize'] = (10,6)
plt.rcParams.update({'font.size': 23})
os.system('clear')
eps = np.finfo('float').eps

# Assign fixed, user-specified dt if not in CFL mode
if(not fixedCFL):
    dt = float(input('What fixed timestep dt should be used? (s) '))
    print('Using a fixed timestep of dt = %1.6e seconds.' % dt)

# initial conditions for specific run mode
gam = 1.4; R = 286.9
if (runMode == 'Quasi-1D'):

    # Specify the overall domain size
    X_min,X_max = -0.50,0.5

    #initial location of the discontinuity [m]
    x0 = 0.0

    #set the initial condition
    rho4 = 1.0
    P4 = 1e6
    T4 = P4/(R*rho4)
    rho1 = 0.125
    P1 = 1e3
    T1 = P1/(R*rho1)
    q_init,X,dx = init_cond(X_min,X_max,Nx,P4,T4,P1,T1,x0)

    #define the shock-tube and nozzle geometry R(x)
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

    #save the grid indices which bound the test-section location
    x_ts = 0.98*1.435
    if((X.max()-x_ts)*(X.min()-x_ts)>0):
        print('Warning: Test section is not within computational domain.')
    else:
        for i in range(X.shape[0]-1):
            if ((X[i]-x_ts)*(X[i+1]-x_ts)<0):
                I1,I2 = i-3,i-2
                break

    #compute scaling factor (vector) for Quasi-1D source term on the grid "X"
    F_vec,x_throat,Mach_vec = areaFunc(Geom_Dat,X[3:-3],True)

elif (runMode == 'Axisymmetric'):

    # Specify the overall domain size
    X_min,X_max = 1e-3,1
    x0 = 0.5

    #defensive programming
    if (X_min<0 or X_max<0):
        print('Initial and final radii must be positive for axisymmetric solver!!! Exting...')
        sys.exit()

    #set the initial condition
    rho4 = 1.0
    P4 = 1e5
    T4 = P4/(R*rho4)
    rho1 = 0.125
    P1 = 1e4
    T1 = P1/(R*rho1)
    q_init,X,dx = init_cond(X_min,X_max,Nx,P4,T4,P1,T1,x0)

    #set Q1D source term for axisymmetric flow (1/r)*F(Q)
    F_vec = 1.0/X[3:-3]

elif (runMode == '1D'):

    # Specify the overall domain size
    X_min,X_max = -0.10,0.10
    x0 = 0.0

    #set the initial condition
    rho4 = 1.0
    P4 = 1e5
    T4 = P4/(R*rho4)
    rho1 = 0.125
    P1 = 1e4
    T1 = P1/(R*rho1)
    q_init,X,dx = init_cond(X_min,X_max,Nx,P4,T4,P1,T1,x0)

    #force Q1D source term to be identically zero for 1D flow
    F_vec = np.zeros(X[3:-3].shape)

else:

    print('Run mode "%s" not recognized! Exiting...' % runMode)
    sys.exit()
    
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
P_51 = int(P4/P1)
fac,exp = exp_format(P_51)

#conditional formatting
if (fac==1):
    if (exp==1):
        pr_str = '$P_{51}=10$'
    else:
        pr_str = '$P_{51}=10^%d$' % exp
else:
    if (exp<=2):
        pr_str = '$P_{51}=%d$' % P_51
    else:
        pr_str = '$P_{51}=%d\\!\\times\\!10^%d$' % (fac,exp)

#initialize frame for real-time animation  
plt.ion()
plt.figure()
if (runMode == 'Quasi-1D'):
    if (fac==1):
        p_str = 'P_51 = 10^%d' % exp
    else:
        p_str = 'P_51 = %dx10^%d' % (fac,exp)
    print('Shock tube operating pressure ratio %s' % p_str)
    plt.title('Solution to GALCIT Nozzle Flow (%s, t=%2.3f[ms])' %(pr_str,0.0))
    line1, = plt.plot(X[3:-3],Mach_vec,'--k',label='Isentropic Solution (Steady)',linewidth=1.0)
    line2, = plt.plot(X[3:-3],M_plt,'-b',label='WENO-JS',linewidth=3.0)
    plt.legend(loc=2)
    plt.ylabel('Mach')
    plt.ylim(0,5.0)
elif (runMode == '1D'):
    Q_exact,M_sh = Shock_Tube_Exact(X,P4,T4,P1,T1,0.0,x0,1.0)
    line1, = plt.plot(X[3:-3],Q_exact[3:-3,0],'-k',linewidth=1.0,label='Exact Solution')
    line2, = plt.plot(X[3:-3],q_init[3:-3,0],'ob',label='WENO-JS')
    plt.title("Sod's Shock Tube Problem (%s, t=%2.3f[ms])" % (pr_str,0.0))
    plt.ylabel('Density')
    plt.legend()
elif (runMode == 'Axisymmetric'):
    if (fac==1):
        p_str = 'P_51 = 10^%d' % exp
    else:
        p_str = 'P_51 = %dx10^%d' % (fac,exp)
    print('Shock tube operating pressure ratio %s' % p_str)
    plt.title('Axisymmetric Shock Tube Problem (%s, t=%2.3f[ms])' %(pr_str,0.0))
    line2, = plt.plot(X[3:-3],q_init[3:-3,0],'-b',label='WENO-JS',linewidth=3.0)
    plt.legend(loc=2)
    plt.ylabel('Density')
plt.tight_layout()
plt.xlim(X_min,X_max)
plt.xlabel(r'$x$[m]')
plt.draw()
plt.savefig('frames/frame%08d.png' % 0)
plt.pause(eps)

#perform the time integration
t_vec = np.zeros(Nt+1)
print('\n=====================================================')
print('   Selected advection scheme is: %s' % Advection)
print('   Selected problem definition is: %s' % runMode)
print('   Left-end boundary condition is: %s' % left_bc)
print('   Right-end boundary condition is: %s' % right_bc)
print('=====================================================\n')
print('Performing time integration with Nt = %d total steps...\n' % Nt)
for i in range(1,Nt+1):

    #set compute timestep or CFL
    ws_max = np.max(euler_1d_wavespeed(q))
    if(fixedCFL):
        dt = CFL*dx/ws_max
    else:
        CFL = ws_max*dt/dx

    #update the time history
    t_vec[i] = t_vec[i-1] + dt

    #if (runMode == 'Quasi-1D') and (t_vec[i] == 1e-4):

    #display to terminal
    print('n = %d,  CFL = %1.2f,  dt = %1.2es,  t = %1.2es' % (i,CFL,dt,t_vec[i]))
    
    # Third-order TVD RK Scheme (Shu '97)
    #======================================
    q = update_ghost_pts(q,left_bc,right_bc)
    L0 = spatial_rhs(q,dx,Advection,left_bc,right_bc) + q1d_rhs(F_vec,q[3:-3,:],left_bc,right_bc)
    q1[3:-3,:] = q[3:-3,:] + L0*dt
    
    q1 = update_ghost_pts(q1,left_bc,right_bc)
    L1 = spatial_rhs(q1,dx,Advection,left_bc,right_bc) + q1d_rhs(F_vec,q1[3:-3,:],left_bc,right_bc)
    q2[3:-3,:] = (3/4)*q[3:-3,:] + (1/4)*q1[3:-3,:] + (1/4)*L1*dt
    
    q2 = update_ghost_pts(q2,left_bc,right_bc)
    L2 = spatial_rhs(q2,dx,Advection,left_bc,right_bc) + q1d_rhs(F_vec,q2[3:-3,:],left_bc,right_bc)
    q[3:-3,:] = (1/3)*q[3:-3,:] + (2/3)*q2[3:-3,:]  + (2/3)*L2*dt    
    #======================================

    #update the stored history
    Q[:,:,i] = q[3:-3,:]

    #real-time animation 
    if(i%plot_freq==0):
        if (runMode == 'Quasi-1D'):
            plt.title('Solution to GALCIT Nozzle Flow (%s, t=%2.3f[ms])' %(pr_str,1e3*t_vec[i]))
            q_p = q[3:-3,:]
            line2.set_ydata((gam*(gam-1)*(q_p[:,0]*q_p[:,2]*(q_p[:,1]+eps)**(-2)-0.5))**(-0.5))
        elif (runMode == '1D'):
            #compute the exact solution for the 1D shock-tube problem
            Q_exact,M_sh = Shock_Tube_Exact(X,P4,T4,P1,T1,t_vec[i],x0,M_sh)
            #rho_ex = Q_exact[3:-3,0]
            #u_ex = Q_exact[3:-3,1]/rho_ex
            #e_ex = Q_exact[3:-3,2]
            #p_ex = (gam-1.0)*(e_ex-0.5*rho_ex*u_ex**2)
            #M_ex = np.sqrt(rho_ex*u_ex**2/(gam*p_ex))
            line1.set_ydata(Q_exact[3:-3,0])
            line2.set_ydata(q[3:-3,0])
            plt.title("Sod's Shock Tube Problem (%s, t=%2.3fms)" % (pr_str,1e3*t_vec[i]))
        elif (runMode == 'Axisymmetric'):
            line2.set_ydata(q[3:-3,0])
            plt.title("Axisymmetric Shock Tube Problem (%s, t=%2.3fms)" % (pr_str,1e3*t_vec[i]))
        if (saveFrames): plt.savefig('frames/frame%08d.png' % int(i/plot_freq))
        plt.pause(eps)

print('\nSolution computed. Building xt-plots...')


#save the boundary condition data for 2D/3D simulations 
if(saveBCData):

    #decide thich values to extract
    index = np.arange(-6,-2)

    #constants
    sizeof_int = 4;
    sizeof_dbl = 8;

    #define the (4) points to extract data from

    f_name = 'GALCIT_1E6.txt'
    #fid = 
    #a = np.fromfile(f_name,dtype=np.float32)

#compute variables for XT-plots
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

#make plot showing final state(s)
if (runMode == '1D'):
    Q_exact,M_sh = Shock_Tube_Exact(X,P4,T4,P1,T1,t_vec[-1],x0,M_sh)
    RHO_exact = Q_exact[3:-3,0]
    U_exact = Q_exact[3:-3,1]/RHO_exact
    E_exact = Q_exact[3:-3,2]
    P_exact = (gam-1)*(E_exact-0.5*RHO_exact*U_exact**2)
    M_exact = np.sqrt(RHO_exact*U_exact**2/(gam*P_exact))
    ENTR_exact = (P_exact/P4)/(RHO_exact/rho4)**gam
    T_exact = P_exact/(R*RHO_exact)
    varex_lst = [RHO_exact,1e-3*P_exact,T_exact,M_exact,U_exact,1e-6*E_exact,ENTR_exact]

#generate snapshots
def make_var_plot(var,var_exact,v_label):

    #used outside the function
    global f_num

    #conditional formatting
    if (runMode == 'Quasi-1D'):
        title_str = 'GALCIT Nozzle Flow (%s)' % pr_str
    elif (runMode == '1D'):
        title_str = "Sod's Shock Tube Problem Tube (%s)" % pr_str
    elif (runMode == 'Axisymmetric'):
        title_str = 'Axisymmetric Shock Tube Problem (%s)' % pr_str

    #make the plot
    plt.figure(f_num)
    plt.scatter(X[3:-3],var[-1,:],s=80,facecolors='none',edgecolors='b',label='WENO-JS',zorder=2)
    plt.plot(X[3:-3],var_exact,'-r',linewidth=1.0,label='Exact Solution',zorder=1)
    plt.title("Sod's Shock Tube Problem (%s, t=%2.3fms)" % (pr_str,1e3*t_vec[-1]))
    plt.xlabel(r'$x$ [m]')
    plt.ylabel(v_label,labelpad=5)
    plt.xlim(X_min,X_max)
    plt.tight_layout()
    plt.grid()
    plt.savefig('fig_%d.png' % f_num)
    f_num += 1

    return

#generate XT-plots
def make_XT_plot(var,v_label):

    #used outside the function
    global f_num

    #conditional formatting
    if (runMode == 'Quasi-1D'):
        title_str = 'GALCIT Ludwieg Tube (%s)' % pr_str
    elif (runMode == '1D'):
        title_str = "Sod's Shock Tube Problem Tube (%s)" % pr_str
    elif (runMode == 'Axisymmetric'):
        title_str = 'Axisymmetric Shock Tube Problem (%s)' % pr_str

    #make the plot
    X_msh,T_msh = np.meshgrid(X[3:-3],1e3*t_vec)
    plt.figure(f_num)
    plt.contourf(X_msh,T_msh,var,300,cmap='jet')
    plt.title(title_str)
    plt.xlabel(r'$x$ [m]')
    plt.ylabel(r'Time [ms]')
    plt.tight_layout()
    cb = plt.colorbar()
    cb.set_label(v_label, labelpad=35, rotation=-90)
    plt.savefig('fig_%d.png' % f_num)
    f_num += 1

    return

#make an xt-plot for each variable
var_lst = [RHO,1e-3*P,TEMP,M,U,1e-6*E,ENTROPY]
label_lst = [r'Density [kg/$m^3$]',r'Pressure [kPa]',r'Temperature [K]',r'Mach [-]',r'Velocity [m/s]',r'Specific Energy [MJ/kg]',r'Measure of Entropy']
for i in range(len(var_lst)): 
    if (runMode == '1D'): make_var_plot(var_lst[i],varex_lst[i],label_lst[i])
    make_XT_plot(var_lst[i],label_lst[i])

plt.show()  
print('Program complete.\n')
