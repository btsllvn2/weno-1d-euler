# N is all the points including ghosts
def init_cond(X_min,X_max,N,P4,T4,P1,T1,x_bnd=0.0):
#compute the initial condition on the grid for Reimann problem
#coordinate system centered at fluid interface (default is x=0, otherwise x=x_bnd)

    import numpy as np

    #define constants
    gam = 1.4
    R = 286.9

    #allocate space for variables
    q_init = np.zeros((N,3))
    rho = np.zeros(N)
    u = np.zeros(N)
    P = np.zeros(N)
    T = np.zeros(N)

    #var_lst = [rho,u,p,T]
    #for var in var_lst: 
    #    var = np.zeros(N)

    #initialize primative variables
    X = np.linspace(X_min,X_max,N)
    for i in range(N):
            if(X[i]<=x_bnd):
                    P[i] = P4
                    T[i] = T4
            else:
                    P[i] = P1
                    T[i] = T1
            
            rho[i] = P[i]/(R*T[i])
            u[i] = 0.0

    #define the initial condition vector
    q_init[:,0] = rho
    q_init[:,1] = rho*u
    q_init[:,2] = P/(gam-1.0) + 0.5*rho*u**2

    return q_init,X

'''	
#Test init_cond
q,x = init_cond(-17.0,2.0,100,70e5,300,1e5,300)

import matplotlib.pyplot as plt
import sys,os


plt.figure()
plt.plot(x,q[:,0],'-b',linewidth=3.5)
plt.show()
sys.exit()
'''

def phys_flux(q):
#q is an Nxnv matrix with N grid points and nv variables   

    import numpy as np

    #primitive variables
    gam = 1.4
    rho = q[:,0]
    u = q[:,1]/q[:,0]
    e = q[:,2]
    p = (gam-1.0)*(e-0.5*rho*u**2)

    #compute the physical flux
    f = np.zeros(q.shape)
    f[:,0] = rho*u
    f[:,1] = p+rho*u**2
    f[:,2] = u*(p+e)

    return f
	
def euler_1d_wavespeed(q):
#q is an Nxnv matrix with N grid points and nv variables   

    import scipy.linalg as la
    import numpy as np

    #primitive variables
    gam = 1.4
    rho = q[:,0]
    u = q[:,1]/q[:,0]
    e = q[:,2]
    p = (gam-1.0)*(e-0.5*rho*u**2)
    c = np.sqrt(gam*p/rho) 

    #define max wavespeed(s) on the grid for global LF splitting
    ws = np.zeros(q.shape[1])
    for j in range(q.shape[1]):
        ws[j] = la.norm(u+(j-1.)*c,np.inf) 

    return ws

import numpy as np

	
def proj_to_char(q,f,q_i_ip1):
    '''
    q is a nsxnv matrix of conservative variables (ns = num pts in current stencil)  
    f is a nsxnv matrix of conservative fluxes (ns = num pts in current stencil)  
    q_i_ip1 is a 2xnv matrix with nv variables to compute average state 

    '''
    import numpy as np

    #approximate state at x_{i+1/2}
    q_st = 0.5*(q_i_ip1[0,:]+q_i_ip1[1,:]) 

    #primitive variables at x_{i+1/2}
    gam = 1.4
    rho = q_st[0]
    u = q_st[1]/q_st[0]
    e = q_st[2]
    p = (gam-1)*(e-0.5*rho*u**2)
    c = np.sqrt(gam*p/rho) 

    #matrix of left eigenvectors of A (eigenvalues in order u-c, u, and u+c)
    L = np.zeros((3,3))
    L[0,0] = 0.5*(0.5*(gam-1.0)*(u/c)**2+(u/c))
    L[1,0] = 1.0-0.5*(gam-1.0)*(u/c)**2
    L[2,0] = 0.5*(0.5*(gam-1.0)*(u/c)**2-(u/c))
    L[0,1] = -(0.5/c)*((gam-1.0)*(u/c)+1.0)
    L[1,1] = (gam-1.0)*u/c**2
    L[2,1] = -(0.5/c)*((gam-1.0)*(u/c)-1.0)
    L[0,2] = L[2,2] = (gam-1.0)/(2*c**2)
    L[1,2] = -(gam-1.0)/c**2

    #project solution/flux into characteristic space for each point in stencil
    q_char = np.zeros(q.shape)
    f_char = np.zeros(f.shape)
    for i in range(q.shape[0]):
        q_char[i,:] = (L.dot(q[i,:].T)).T
        f_char[i,:] = (L.dot(f[i,:].T)).T

    return q_char,f_char

def proj_to_cons(f_char,q_cons):
    '''
    f_char is a Nxnv matrix of the characteristic flux for all grid points
    q_cons is a Nxnv matrix of the conservative variables for all grid points

    '''
    import numpy as np

    #compute the (conservative) flux at each point in the grid
    f_cons = np.zeros(f_char.shape)
    N = f_char.shape[0]
    for i in range(N-1):

        #approximate state at x_{i+1/2}
        q_st = 0.5*(q_cons[i,:]+q_cons[i+1,:]) 

        #primitive variables at x_{i+1/2}
        gam = 1.4
        rho = q_st[0]
        u = q_st[1]/q_st[0]
        e = q_st[2]
        p = (gam-1)*(e-0.5*rho*u**2)
        c = np.sqrt(gam*p/rho) 

        #matrix of right eigenvectors of A (eigenvalues in order u-c, u, and u+c)
        R = np.zeros((3,3))
        R[0,:] = 1.0
        R[1,0] = u-c
        R[1,1] = u
        R[1,2] = u+c
        R[2,0] = c**2/(gam-1.0)+0.5*u**2-u*c
        R[2,1] = 0.5*u**2
        R[2,2] = c**2/(gam-1.0)+0.5*u**2+u*c
       
        #update the boundary condition mask M
        M = np.eye(3)
        
        # Solid Wall
        if(i==3):
                M = np.array([[1,0,0],
                              [0,1,0],
                              [1,0,0]])
        # Non-Reflecting Outflow	
        elif(i==N-4):
                M = np.array([[0,0,0],
                              [0,1,0],
                              [0,0,1]])
	   
        #project flux back into conservative space (no need to project solution)
        f_cons[i,:] = (R.dot(M.dot(f_char[i,:].T))).T

    return f_cons

def q1d_afunc(x,r,makePlot=False,demo=False):
#computes 1/A*dA/dx based on a provided geometry R(x)

    from scipy.interpolate import splrep, splev
    import scipy.interpolate as interp
    import matplotlib.pyplot as plt
    import numpy as np
    import sys,os

    global f_num

    #constants
    pi = 4*np.arctan(1)

    #defensive programming
    for vec in (x,r):
        vec = np.array(vec)
    #if len(x) != len(r):
    #    print('\tProvided xy-data is inconsistent! Exiting...\n')
    #    sys.exit()

    #demo inputs
    if (makePlot):
  
        # GALGIT Ludwieg tube geometry + Mach 5 nozzle
        if (demo): 
            X = np.array([[-17.00000,  0.0568613],
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
            x,r = X[:,0],X[:,1]

        #use LaTeX formatting for titles and axes
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rcParams['figure.figsize'] = (10.0,6.0)

        #plot the discrete points and the phcip spline
        plt.figure()
        x_plt = np.linspace(x.min(),x.max(),int(1e4))
        r_func = interp.pchip(x,r)
        plt.plot(x_plt,r_func.__call__(x_plt),'-b',linewidth=3.0,label='PCHIP Interpolant',zorder=1)
        plt.plot(x_plt,-r_func.__call__(x_plt),'-b',linewidth=3.0,zorder=2)
        plt.scatter(x,r,c='r',label='Imported Data Points',zorder=3)
        plt.scatter(x,-r,c='r',zorder=4)
        plt.plot([-1,3],[0,0],'-.k',linewidth=2,zorder=3)
        plt.xlim(-0.1,0.7)
        plt.ylim(-0.1,0.1)
        plt.xlabel(r'$x\;[m]$',fontsize=15)
        plt.ylabel(r'$r\;[m]$',fontsize=15)
        plt.title(r'GALCIT Ludwieg Tube - Nozzle',fontsize=17)
        plt.legend(fontsize=12)
        plt.show()
        f_num += 1
        #sys.exit()

    #create an interpolant for log[A(x)] 
    l_func = interp.pchip(x,np.log(pi*r**2))

    #output the derivative function
    return l_func.__call__(x,1)

#def Shock_Tube_Exact(q_L,q_R,x,t,makePlots=False):
def Shock_Tube_Exact(makePlots=False):
    '''
    ================================================================
                                                                    
       Function which computes the analytical solution              
       to the 1D shock-tube problem given left and right            
       initial conditions (assumes no shock reflections)            
                                                                    
       Usage: q_an = Shock_Tube_Exact_ND(q_L,q_R,x,t,n_plot), where 
                                                                    
       q_an = Nx3 array of the analytical solution q(x) at time t   
       q_L  = 3x1 vector of the initial driver section conditions   
       q_R  = 3x1 vector of the initial driven section conditions   
       x    = Nx1 vector of x-locations of computatonal gridpoints               
       t    = current solution time [s]                             
                                                                    
    ================================================================
    
    '''

    global f_num

    import matplotlib.pyplot as plt
    import numpy as np
    import sys, os

    #define constants
    eps = np.finfo(float).eps
    pi = np.pi
    gam = 1.4
    R = 286.9
    alpha  = (gam+1)/(gam-1)
    f_num = 1 

    #verify input arrays
    # var_list = [q_L, q_R, x]
    # for var in var_list:
        # var = np.array(var)

    #normal operation or demo mode
    #if len(sys.argv) == 0:

    #example initial conditions (Sod '78)
    q_L = np.array([1, 0, 10**5/(gam-1)])
    q_R = np.array([0.125, 0, 10**4/(gam-1)])

    #set domain limits
    x_min = 0.0; x_max = 10.0
    x = np.linspace(x_min,x_max,501)
    x0 = (max(x)+min(x))/2
    t = 0.0061

    '''
    elif len(sys.argv) == 4:
        #set domain limits for provided x-vector
        x_min,x_max = min(x),max(x)
        x0 = 0
    else:
        print('\tUnrecognized number of inputs! Exiting...\n')
        sys.exit()
    '''

    #total number of points on the grid
    N = x.shape[0] 

    #compute the acoustic speeds in the left- and right-states
    c_L = np.sqrt(gam*(gam-1)*q_L[2]/q_L[0])
    c_R = np.sqrt(gam*(gam-1)*q_R[2]/q_R[0])
    print('c_L = %3.3f' % c_L)
    print('c_R = %3.3f' % c_R)

    #compute the shock Mach number Ms using Newton-Raphson + Complex-Step Derivative
    P_41 = q_L[2]/q_R[2]
    err = 1.0; cntr = 0; h=1e-30; M_sh = 1.5
    print('P_41 = %d' % P_41)
    Res = lambda M_sh: -P_41*(1-(gam-1)/(gam+1)*(c_R/c_L)*(M_sh**2-1)/M_sh)**(2*gam/(gam-1))+2*gam/(gam+1)*(M_sh**2-1)+1
    print('Res(1.5) = %1.6e' % Res(1.5))
    print('Solving for shock Mach number (Ms) based on PR = P_l/P_r:')
    while (err>eps): 
        cntr += 1; Msh_p = M_sh
        M_sh -= h*Res(M_sh)/np.imag(Res(complex(M_sh,h)))
        err = M_sh/Msh_p-1
        print('    N = %d, e = %1.6e' % (cntr,err))

        #limit total number of iterations
        if (cntr == 15):
            print('\n  **Iteration limit exceeded (e = %1.6e)\n' % err)
            break

    #compute shock velocity [m/s] and static pressure-jump P_21 = P2/P1
    v_sh = M_sh*c_R
    x_sh = x0 + v_sh*t
    P_21 = 1.0+2*gam/(gam-1)*(M_sh**2-1)
    #if len(sys.argv) == 0:
    print('\n\n  M_shock = %2.3f\n\n' % M_sh)

    #compute velocity of the contact discontinuity [m/s]
    v_ct = 2*c_L/(gam-1)*(1-(P_21/P_41)**((gam-1)/(2*gam)))
    x_ct = x0 + v_ct*t
    u3 = v_ct; u2 = v_ct

    #compute sound speeds in Regions 2 and 3
    c2 = c_R*np.sqrt(P_21*(2+(gam-1)*M_sh**2)/((gam+1)*M_sh**2))
    c3 = c_L*(P_21/P_41)**((gam-1)/(2*gam))

    #compute velocity of left and right sides of the expansion fan
    v_fL = -c_L
    x_fL = x0 + v_fL*t
    v_fR = u2-c3
    x_fR = x0 + v_fR*t

    #write out the solution
    q_an = np.zeros((N,3))
    for i in range(N):


        #undisturbed driven (right) state 
        if (x[i]>x_sh):
            rho = q_R[0]
            u   = 0.0
            p   = (gam-1)*q_R[2]

        #undisturbed driver (left) state
        elif (x[i]<x_fL):
            rho = q_L[0]
            u   = 0.0
            p   = (gam-1)*q_L[2]

        #between the head of the expansion fan the the shock
        else:

            #define a locator variable
            phi = (x[i]-x_fR)/(x_ct-x_fR)
            
            #x between shock and contact discontinuity
            if (phi>1.0):
                rho = ((1+alpha*P_21)/(alpha+P_21))*q_R[0]
                u   = u2
                p   = P_21*((gam-1)*q_R[2])

            #x within the expansion fan
            elif (phi<0.0):
                u   = 2/(gam+1)*(c_L+(x[i]-x0)/t)
                p   = (gam-1)*q_L[2]*(1-(gam-1)*u/(2*c_L))**(2*gam/(gam-1))
                rho = q_L[0]*(p/((gam-1)*q_L[2]))**(1/gam)

            #x between contact disconinuity and expansion fan
            else:
                rho = q_L[0]*(P_21/P_21)**(1/gam)
                u   = u3
                p   = P_21*((gam-1)*q_R[2])

        #construct solution vector from the primitive variables
        q_an[i,0] = rho
        if (abs(t)>eps):
            q_an[i,1] = rho*u
        q_an[i,2] = p/(gam-1)+0.5*rho*u**2


    #plot the solution if run without arguments 
    #if len(sys.argv) == 0:

    #compute the variables
    P_plot = (gam-1)*(q_an[:,2]-q_an[:,1]**2/(2*q_an[:,0]))
    U = q_an[:,1]/q_an[:,0]
    Mach = np.sqrt(q_an[:,0]*U**2/(gam*P_plot))
    T_plot = P_plot/(R*q_an[:,0])
    c_plot = np.sqrt(gam*R*T_plot)
    Entropy = P_plot/(q_an[:,0])**(gam)
    e = P_plot/(gam-1)+0.5*U**2
    str_t = 'Sod''s Shock Tube Problem (t=%1.3f)' % float(1000*t)

    #make the figures
    #plt.clf()
    fig = plt.figure(f_num)
    ax1 = fig.add_subplot(221)
    ax1.plot(x,q_an[:,0],'-b',linewidth=3.0)
    ax1.set_title(str_t)
    ax1.grid()
    #ax1.set(xlabel='r$x$', ylabel='r$\rho$')
    ax2 = fig.add_subplot(222)
    ax2.plot(x,P_plot,'-b',linewidth=3.0)
    ax2.set_title(str_t)
    ax2.grid()
    ax3 = fig.add_subplot(223)
    ax3.plot(x,U,'-b',linewidth=3.0)
    ax3.set_title(str_t)
    ax3.grid()
    ax4 = fig.add_subplot(224)
    ax4.plot(x,Mach,'-b',linewidth=3.0)
    ax4.set_title(str_t)
    ax4.grid()
    fig.tight_layout()
    plt.savefig('fig_%d.pdf' % f_num)
    f_num += 1

    #show the plot(s)
    plt.show()
     
    return q_an

def phi_weno5():
    '''
    Function which computes the 5th order WENO reconstruction for the flux at x_{i+1/2}

    q is a nsxnv matrix of conservative variables (ns = num pts in current stencil)  
    f is a nsxnv matrix of conservative fluxes (ns = num pts in current stencil)  
    q_i_ip1 is a 2xnv matrix with nv variables to compute average state 

    '''



    return f_hat_plus_half

## Test wave speed function 
#q,x = init_cond(-17.0,2.0,100,70e5,300,1e5,300)
#ws = euler_1d_wavespeed(q)
#print("ws_max = ", ws)
#
##test area ratio function
f_num=5
q1d_afunc(1,1,True,True)

#test exact solution function
Shock_Tube_Exact(True)
