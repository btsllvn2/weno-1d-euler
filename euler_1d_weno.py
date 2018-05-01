def init_cond(X_min,X_max,N,P4,T4,P1,T1,x_bnd=0.0):
#compute the initial condition on the grid for Reimann problem
#coordinate system centered at fluid interface (default is x=0, otherwise x=x_bnd)

    import numpy as np
   
    #define constants
    gam = 1.4
    R = 286.9
 
    #allocate space for variables
    q_init = np.zeros((N+6,3))
    var_lst = [rho,u,p,T]
    for var in var_lst: 
        var = np.zeros(N)

    #initialize primative variables
    x = np.linspace(X_min,X_max,N)
    for i in range(3,N-3):
        if (X[i]<=x_bnd):
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
    q_init[:,2] = p/(gam-1.0) + 0.5*rho*u**2

    return q_init

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
    f[:,2] = u*(p+rho*e)

    return f

def num_flux(q):
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
    f[:,2] = u*(p+rho*e)

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
    ws = np.zeros(q.shape)
    for j in range(q.shape[1]):
        ws[:,j] = la.norm(u+(j-1)*c,np.inf) 

    return ws

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
    for i in range(f_char.shape[0]-1):

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
       
        #project flux back into conservative space (no need to project solution)
        f_cons[i,:] = (R.dot(f_char[i,:].T)).T

    return f_cons

def q1d_afunc(x,r,demo=False):
#computes 1/A*dA/dx based on a provided geometry R(x)

    from scipy.interpolate import splrep, splev
    import scipy.interpolate as interp
    import matplotlib.pyplot as plt
    import numpy as np
    import sys,os

    #constants
    pi = 4*np.arctan(1)

    #demo procedure
    if (demo):
    
        # GALGIT Ludwieg tube geometry + Mach 5 nozzle
        X = np.array([[-17.00000,  0.0568613],
                      [-0.445792,  0.0568613],
                      [0.0295679,  0.0568613],
                      [0.0384761,  0.0369598],
                      [0.0538287,  0.0233131],
                      [0.0828279,  0.0159212],
                      [0.131160,   0.0233131],
                      [0.203942,   0.0363912],
                      [0.292646,   0.0471948],
                      [0.405800,   0.0557240],
                      [0.543973,   0.0579985],
                      [2.000000,   0.0579985]])

        #use LaTeX formatting for titles and axes
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rcParams['figure.figsize'] = (10.0,6.0)

        #plot the discrete points and the phcip spline
        plt.figure()
        x,r = X[:,0],X[:,1]
        x_plt = np.linspace(x.min(),x.max(),int(1e4))
        r_func = interp.pchip(x,r)
        plt.plot(x_plt,r_func.__call__(x_plt),'-b',linewidth=3.0,label='PCHIP Spline',zorder=1)
        #r_func = interp.Akima1DInterpolator(x,r)
        #plt.plot(x_plt,r_func.__call__(x_plt),'-b',linewidth=3.0,label='Akima Spline',zorder=1)
        plt.scatter(x,r,c='r',label='Imported Data Points',zorder=2)
        plt.plot([-1,3],[0,0],'-.k',linewidth=2,zorder=3)
        plt.xlim(-0.1,0.75)
        plt.ylim(-0.06,0.1)
        plt.xlabel(r'$x\;[m]$',fontsize=15)
        plt.ylabel(r'$r\;[m]$',fontsize=15)
        plt.title(r'GALCIT Ludwieg Tube - Nozzle',fontsize=15)
        plt.legend()
        plt.show()
        sys.exit()

    #defensive programming
    if (len(x) != len(r)):
        print('\tXY data is not consistent')
        sys.exit()

    #create an interpolant for log[A(x)] 
    l_func = interp.pchip(x,np.log(pi*r**2))

    #output the derivative function
    return l_func.__call__(x,1)

def phi_weno5():
    '''
        Function which computes the 5th order WENO reconstruction for the flux at x_{i+1/2}

        q is a nsxnv matrix of conservative variables (ns = num pts in current stencil)  
        f is a nsxnv matrix of conservative fluxes (ns = num pts in current stencil)  
        q_i_ip1 is a 2xnv matrix with nv variables to compute average state 

    '''



    return f_hat_plus_half

#attempt to plot the shoc tube geometry
import numpy as np
q1d_afunc(1,1,True)
