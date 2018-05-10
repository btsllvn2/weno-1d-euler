# N is all the points including ghosts
def init_cond(X_min,X_max,N,P4,T4,P1,T1,x0=0.0):
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

    #define the grid with N total points (ghost points outside domain)
    X = np.zeros(N)
    X[3:-3] = np.linspace(X_min,X_max,N-6)
    dx = X[4]-X[3]
    for i in range(3):
        X[i] = X_min+(i-3)*dx
        X[-(i+1)] = X_max+(3-i)*dx 

    #initialize primative variables
    for i in range(N):
        if(X[i]<=x0):
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
    
    print('Intiial condition generated successfully.')
    
    return q_init, X, dx

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
    import sys,os

    #primitive variables
    gam = 1.4
    R = 286.9
    rho = q[:,0]
    u = q[:,1]/q[:,0]
    e = q[:,2]
    p = (gam-1.0)*(e-0.5*rho*u**2)
    t = p/(R*rho)
    if (t.min()<0.0):
        print('\n============================================================')
        print('     Warning: Negative temperature detected!!!')
        print('     Solution is numerically unstable. Now exiting....' % t.min())
        print('============================================================\n\n')
        sys.exit()
    c = np.sqrt(gam*p/rho) 

    #define max wavespeed(s) on the grid for LF splitting
    ws = np.zeros(q.shape[1])
    for j in range(q.shape[1]):
        ws[j] = la.norm(u+(j-1)*c,np.inf) 

    return ws

def langrange_extrap(x_in,q_in,x_ext):
# Function which computes the 5th-order Lagrange extrapolated solution 
# x_in is an Npx1 vector of Np grid points points
# q_in is an NpxNv matrix with Np points and Nv solution variables   
# x_ext is an 1x1 scalar of the desired point to evaluate L(x)
# q_ext is a 1xNv vector of the extrapolation solution 

    import numpy as np

    #perform the extrapolation component-wise
    m,n = q_in.shape
    q_ext = np.zeros(n)
    for i in range(m):

        #compute the current Lagrange factor l_i(x_ext)
        l_i = 1.0
        for j in range(m):
            if (i!=j):    
                l_i *= (x_ext-x_in[j])/(x_in[i]-x_in[j]) 

        #update the running sum
        q_ext += q_in[i,:]*l_i

    return q_ext

def update_ghost_pts(q,left_bc,right_bc):

    ##assign left-end ghost cell values
    #if (left_bc == 'Wall'):
    #    for i in range(3):
    #        q[(2-i),0] =  q[(i+4),0]
    #        q[(2-i),1] = -q[(i+4),1]
    #        q[(2-i),2] =  q[(i+4),2]
    if (left_bc == 'Neumann'):
        for i in range(3):
            q[(2-i),:] =  q[3,:]
    else:
        for i in range(3):
            q[(2-i),:]  = (10*(i+1))**10

    #assign right-end ghost cell values
    #if (right_bc == 'Wall'):
    #    for i in range(3):
    #        q[-(3-i),0] =  q[-(i+5),0]
    #        q[-(3-i),1] = -q[-(i+5),1]
    #        q[-(3-i),2] =  q[-(i+5),2]
    if (right_bc == 'Neumann'):
        for i in range(3):
            q[-(3-i),:] =  q[-4,:]
    else:
        for i in range(3):
            q[-(3-i),:] = (10*(i+1))**10

    return(q)

def char_numerical_flux(q,adv):

    import numpy as np
    
    # Compute the fluxes on the entire grid
    f = phys_flux(q)

    # Compute the state vector at the x_{1+1/2} points
    q_i_p_half = (q[2:q.shape[0]-3,:] + q[3:q.shape[0]-2,:])*0.5
    
    # -------------------------------------------------------------------------
    
    # Number of x_{i+1/2} points on the domain at which the flux is computed
    N_x_p_half = q_i_p_half.shape[0]
    
    # Number of state variables
    Nvar = q.shape[1]
    
    # WENO full stencil size 
    stencil_size = 5
    
    # Number of ghost points at a boundary
    Ng = 3
    
    # -------------------------------------------------------------------------

    # Compute the max wavespeeds on the entire grid
    ws = euler_1d_wavespeed(q[Ng:q.shape[0]-Ng,:])

    # Initialize the arrays
    f_char_p = np.zeros((Nvar, stencil_size))
    f_char_m = np.zeros((Nvar, stencil_size))
    f_char_i_p_half = np.zeros((N_x_p_half, Nvar))
    
    # Loop through each x_{i+1/2} point on the grid
    # Compute the f_char_p and f_char_m terms for phi_weno5
    # Compute the fifth order accurate weno flux-split terms
    # Add them together to obatin to find f_char_i_p_half
    
    for i in range(N_x_p_half):
        #ws = euler_1d_wavespeed(q[i:i+stencil_size+1,:])
        qi, fi = proj_to_char(q[i:i+stencil_size+1,:], f[i:i+stencil_size+1,:], q_i_p_half[i])
        
        for j in range(stencil_size):
            f_char_p[:,j] = (0.5*( (fi[j,:]).T + (np.diag(ws)).dot((qi[j,:]).T) )).T
            f_char_m[:,j] = (0.5*( (fi[j+1,:]).T - (np.diag(ws)).dot((qi[j+1,:]).T) )).T

        # Compute the i + 1/2 points flux
        for k in range(0, Nvar):
            f_char_i_p_half[i,k] = phi_weno5(f_char_p[k,:],adv) + phi_weno5(f_char_m[k,::-1],adv)    
    
    return f_char_i_p_half


def phi_weno5(f_char_p_s,adv):
    '''
    Function which computes a 5th-order WENO reconstruction of the numerical
    flux at location x_{i+1/2}, works regardless of the sign of f'(u)
    '''

    import numpy as np

    #assign the fluxes at each point in the full stencil 
    f_i_m_2 = f_char_p_s[0]
    f_i_m_1 = f_char_p_s[1]
    f_i     = f_char_p_s[2]
    f_i_p_1 = f_char_p_s[3]
    f_i_p_2 = f_char_p_s[4]
    
    #estimate of f_{i+1/2} for each substencil
    f0 = (1/3)*f_i_m_2 - (7/6)*f_i_m_1 + (11/6)*f_i
    f1  = (-1/6)*f_i_m_1 + (5/6)*f_i + (1/3)*f_i_p_1
    f2  = (1/3)*f_i + (5/6)*f_i_p_1 - (1/6)*f_i_p_2
    
    #smoothness indicators for the solution on each substencil 
    beta_0 = (13/12)*(f_i_m_2 - 2*f_i_m_1 + f_i)**2 + (1/4)*(f_i_m_2 - 4*f_i_m_1 + 3*f_i)**2
    beta_1 = (13/12)*(f_i_m_1 - 2*f_i + f_i_p_1)**2 + (1/4)*(f_i_m_1 - f_i_p_1)**2
    beta_2 = (13/12)*(f_i - 2*f_i_p_1 + f_i_p_2)**2 + (1/4)*(3*f_i - 4*f_i_p_1 + f_i_p_2)**2

    #unscaled nonlinear weights 
    epsilon = 1e-6
    w0_tilde = 0.1/(epsilon + beta_0)**2
    w1_tilde = 0.6/(epsilon + beta_1)**2
    w2_tilde = 0.3/(epsilon + beta_2)**2
    
    #scaled nonlinear weights
    w0 = w0_tilde/(w0_tilde + w1_tilde + w2_tilde)
    w1 = w1_tilde/(w0_tilde + w1_tilde + w2_tilde)
    w2 = w2_tilde/(w0_tilde + w1_tilde + w2_tilde)
    
    #overwrite WENO nonlinear weights with optimal linear weights
    if (adv=='LINEAR-FD'): w0 = 0.1; w1 = 0.6; w2 = 0.3;
   
    #linear convex combination of (3) substencil reconstructions
    f_char_i_p_half_p_s = w0*f0 + w1*f1 + w2*f2

    return f_char_i_p_half_p_s
    
def proj_to_char(q,f,q_st):
    '''
    q is a nsxnv matrix of conservative variables (ns = num pts in current stencil)  
    f is a nsxnv matrix of conservative fluxes (ns = num pts in current stencil)  
    q_st is a 1xnv vector with nv variables of average state 

    '''
    import numpy as np

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

def spatial_rhs(f_char,q_cons,dx,left_bc,right_bc):

    '''
    f_char is a Ni x nv matrix of the characteristic flux only at interior adjacent flux interfaces
    q_cons is a Np x nv matrix of the conservative variables full domain

    '''
    import numpy as np
    import sys,os
    
    # Compute the state vector at the x_{1+1/2} points
    q_i_p_half = (q_cons[2:q_cons.shape[0]-3,:] + q_cons[3:q_cons.shape[0]-2,:])*0.5

    # Initialize arrays
    N = f_char.shape[0]
    R = np.zeros((N,3,3))

    # Compute the R matrix at every half point flux location
    for i in range(N):

        #approximate state at x_{i+1/2}
        q_st = q_i_p_half[i,:]

        #primitive variables at x_{i+1/2}
        gam = 1.4
        rho = q_st[0]
        u = q_st[1]/q_st[0]
        e = q_st[2]
        p = (gam-1)*(e-0.5*rho*u**2)
        c = np.sqrt(gam*p/rho) 

        #matrix of right eigenvectors of A (eigenvalues in order u-c, u, and u+c)    
        R[i,0,:] = 1.0
        R[i,1,0] = u-c
        R[i,1,1] = u
        R[i,1,2] = u+c
        R[i,2,0] = c**2/(gam-1.0)+0.5*u**2-u*c
        R[i,2,1] = 0.5*u**2
        R[i,2,2] = c**2/(gam-1.0)+0.5*u**2+u*c

    # Initialize rhs
    rhs = np.zeros((N-1,f_char.shape[1]))

    # Compute qdot at left and right boundaries
    rhs[0,:] = left_b_qdot(np.array(q_cons[3:8,:]),dx,left_bc)
    rhs[-1,:] = right_b_qdot(np.array(q_cons[-8:-3,:]),dx,right_bc)

    #update the rhs values on the interior (and possibly the boundary)
    i_start = 1; i_end = rhs.shape[0]-1
    if (left_bc == 'Neumann'): i_start -= 1
    if (right_bc == 'Neumann'): i_end += 1
    for i in range(i_start,i_end):   
        
        # Local Right Eigen Matrices
        R_p_half = R[i+1,:,:]
        R_m_half = R[i,:,:]
 
        # The local qdot
        rhs[i,:] = (-1/dx)*(R_p_half.dot((f_char[i+1,:]))-R_m_half.dot((f_char[i,:])))
 
    return rhs
    
def left_b_qdot(q,h,bc_type):
    
    import numpy as np
    
    #primitive variables
    gam = 1.4
    rhoarr = q[:,0]
    uarr = q[:,1]/q[:,0]
    earr = q[:,2]
    parr = (gam-1.0)*(earr-0.5*rhoarr*uarr**2)
    carr = np.sqrt(gam*parr/rhoarr) 

    R = np.zeros((3,3))
    
    u = uarr[0]
    rho = rhoarr[0]
    c = carr[0] 
    
    #matrix of right eigenvectors of A (eigenvalues in order u-c, u, and u+c)
    R[0,:] = 1.0
    R[1,0] = u-c
    R[1,1] = u
    R[1,2] = u+c
    R[2,0] = c**2/(gam-1.0)+0.5*u**2-u*c
    R[2,1] = 0.5*u**2
    R[2,2] = c**2/(gam-1.0)+0.5*u**2+u*c    
    
    # Compute spatial gradients
    drhodx = (-25*rhoarr[0]+48*rhoarr[1]-36*rhoarr[2]+16*rhoarr[3]-3*rhoarr[4])/(12*h)
    dpdx =   (-25*parr[0]  +48*parr[1]  -36*parr[2]  +16*parr[3]  -3*parr[4]  )/(12*h)
    dudx =   (-25*uarr[0]  +48*uarr[1]  -36*uarr[2]  +16*uarr[3]  -3*uarr[4]  )/(12*h)
    
    # Apply the NRBC
    f = np.ones(3)
    for j in range(3): 
        if(R[1,j]>0): f[j]=0
    
    # Compute the wave amplitudes
    L1 =  f[0]*(u-c)*(dpdx-rho*c*dudx)/(2*c**2)
    L2 =  f[1]*u*(drhodx-dpdx/c**2)
    L3 =  f[2]*(u+c)*(dpdx+rho*c*dudx)/(2*c**2)
    
    # Apply the Wall BC
    if (bc_type == 'Wall'): 
        L3 = L1
    elif (bc_type == 'Force-Free'):
        L3 = L1+2*rho*c*(u*dudx)

    # Transform back to conservative form
    qdot = -R.dot(np.array([L1,L2,L3]))

    # Bypass characteristic BC if 'Neumann' is selected
    if (bc_type == 'Neumann'): qdot = np.zeros(qdot.shape) 

    return qdot
    
def right_b_qdot(q,h,bc_type):
    
    import scipy.interpolate as interp
    import numpy as np
    
    #primitive variables
    gam = 1.4
    rhoarr = q[:,0]
    uarr = q[:,1]/q[:,0]
    earr = q[:,2]
    parr = (gam-1.0)*(earr-0.5*rhoarr*uarr**2)
    carr = np.sqrt(gam*parr/rhoarr) 

    R = np.zeros((3,3))
    
    u = uarr[-1]
    rho = rhoarr[-1]
    c = carr[-1] 
    
    #matrix of right eigenvectors of A (eigenvalues in order u-c, u, and u+c)
    R[0,:] = 1.0
    R[1,0] = u-c
    R[1,1] = u
    R[1,2] = u+c
    R[2,0] = c**2/(gam-1.0)+0.5*u**2-u*c
    R[2,1] = 0.5*u**2
    R[2,2] = c**2/(gam-1.0)+0.5*u**2+u*c    
    
    # Compute spatial gradients
    drhodx = (3*rhoarr[-5]-16*rhoarr[-4]+36*rhoarr[-3]-48*rhoarr[-2]+25*rhoarr[-1])/(12*h)
    dpdx =   (3*parr[-5]  -16*parr[-4]  +36*parr[-3]  -48*parr[-2]  +25*parr[-1]  )/(12*h)
    dudx =   (3*uarr[-5]  -16*uarr[-4]  +36*uarr[-3]  -48*uarr[-2]  +25*uarr[-1]  )/(12*h)
    
    # Apply the NRBC
    f = np.ones(3)
    for j in range(3):
        if(R[1,j]<0): f[j]=0
    
    # Compute the wave amplitudes
    L1 =  f[0]*(u-c)*(dpdx-rho*c*dudx)/(2*c**2)
    L2 =  f[1]*u*(drhodx-dpdx/c**2)
    L3 =  f[2]*(u+c)*(dpdx+rho*c*dudx)/(2*c**2)

    # Apply the Wall BC
    if (bc_type == 'Wall'): 
        L1 = L3
    elif (bc_type == 'Force-Free'):
        L1 = L3-2*rho*c*(u*dudx)

    # Transform back to conservative form
    qdot = -R.dot(np.array([L1,L2,L3]))

    # Bypass characteristic BC if 'Neumann' is selected
    if (bc_type == 'Neumann'): qdot = np.zeros(qdot.shape)

    return qdot

#source term which accounts for quasi-1D area variation
def q1d_rhs(f_vec,q,left_bc,right_bc): 

    import numpy as np

    #primitive variables
    gam = 1.4
    rho = q[:,0]
    u = q[:,1]/q[:,0]
    e = q[:,2]
    p = (gam-1.0)*(e-0.5*rho*u**2)

    #compute the unscaled Q1D source term
    flux = np.zeros(q.shape)
    flux[:,0] = rho*u
    flux[:,1] = rho*u**2
    flux[:,2] = u*(p+e)

    #compute the rhs matrix
    rhs = np.zeros(flux.shape)
    for i in range(q.shape[0]):
        rhs[i,:] = -f_vec[i]*flux[i,:]

    #BC handling
    #rhs[0,:] = 0.0
    #rhs[-1,:] = 0.0

    return rhs
    
f_num = 1
def areaFunc(x,r,X_vec,makePlot=False,demo=False):
#computes 1/A*dA/dx based on a provided geometry R(x)

    from scipy.interpolate import splrep, splev
    import scipy.interpolate as interp
    import matplotlib.pyplot as plt
    import numpy as np
    import sys,os

    global f_num

    #relevant constants
    eps = np.finfo(float).eps
    gam = 1.4
    R = 286.9
    h = 1e-30

    #defensive programming
    for vec in (x,r):
        vec = np.array(vec)

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

    #define a function which interpolates the given radius data points
    r_func = interp.pchip(x,r)

    #demo inputs
    if (makePlot):
  
        #plot the discrete points and the phcip spline
        plt.figure(f_num)
        x_plt = np.linspace(x.min(),x.max(),int(1e4))
        plt.plot(x_plt,r_func.__call__(x_plt),'-b',linewidth=3.0,label='PCHIP Interpolant',zorder=1)
        plt.plot(x_plt,-r_func.__call__(x_plt),'-b',linewidth=3.0,zorder=2)
        plt.scatter(x,r,c='r',label='Imported Data Points',zorder=3)
        plt.scatter(x,-r,c='r',zorder=4)
        plt.plot([-1,3],[0,0],'-.k',linewidth=2,zorder=3)
        plt.xlim(-0.1,0.7)
        plt.ylim(-0.1,0.1)
        plt.xlabel(r'$x\;[m]$',fontsize=15)
        plt.ylabel(r'$r\;[m]$',fontsize=15)
        plt.title(r'GALCIT Ludwieg Tube Mach 4 Nozzle',fontsize=17)
        plt.legend(fontsize=12)
        plt.savefig('nozzle.pdf')
        plt.savefig('nozzle.png')
        f_num += 1

    #create and differentiate the PCHIP interpolant for log[A(x)] 
    l_func = interp.pchip(x,np.log(np.pi*r**2))
    F_vec = l_func.__call__(X_vec,1)

    #compute the exact x-location of the throat
    def Res(x): return l_func.__call__(x,1)
    print('Computing exact location of the throat...')
    x_th = 0.05; cntr=0; err=1.0;
    x_l = 0.05; x_r = 0.20
    while (abs(err)>eps):

        #update the counter
        cntr += 1 
    
        #use false-position method to estimate the root
        x_new = (x_l*Res(x_r)-x_r*Res(x_l))/(Res(x_r)-Res(x_l))
        if (Res(x_new)*Res(x_r)>0):
            x_r = x_new
        else:
            x_l = x_new
        err = abs(x_l-x_r)

        #limit total number of iterations
        if (cntr == 100):
            print('\n  **Iteration limit exceeded (e = %1.6e)\n' % err)
            break
    x_th = x_new
    print('Location of the throat is x_th = %2.8f[m]' % x_th)
   

    #Compute the isentropic solution for the nozzle flow
    print('Computing the isentropic nozzle solution...')
    Mach_vec = np.zeros(F_vec.shape)
    def Res(M,AR): return (2/(gam+1)*(1+0.5*(gam-1)*M**2))**((gam+1)/(2*(gam-1)))-M*AR
    for i in range(X_vec.shape[0]):

        #area ratio for current location
        AR = (r_func.__call__(X_vec[i])/r_func.__call__(x_th))**2

        #subsonic or supersonic target
        if (X_vec[i]<=x_th):

            #subsonic branch
            M_l = 0.0; M_r = 1.0; cntr=0; err=1.0
            while ((err>1e-12) and (cntr<=1500)):
                cntr+=1; 
                M_new = (M_l*Res(M_r,AR)-M_r*Res(M_l,AR))/(Res(M_r,AR)-Res(M_l,AR))
                if (Res(M_new,AR)*Res(M_r,AR)>0):
                    M_r = M_new
                else:
                    M_l = M_new
                err = abs(Res(M_new,AR))

        else:

            #supersonic branch
            M_l = 1.0; M_r = 5.0; cntr=0; err=1.0
            while ((err>1e-12) and (cntr<=1500)):
                cntr+=1; 
                M_new = (M_l*Res(M_r,AR)-M_r*Res(M_l,AR))/(Res(M_r,AR)-Res(M_l,AR))
                if (Res(M_new,AR)*Res(M_r,AR)>0):
                    M_r = M_new
                else:
                    M_l = M_new
                err = abs(Res(M_new,AR))
    
        #store the converged Mach number    
        Mach_vec[i] = M_new

    print('Finished. Max isentropic Mach number on the grid is M = %2.5f' % Mach_vec.max())

    return F_vec,x_th,Mach_vec

def Shock_Tube_Exact(X,P4,T4,P1,T1,time,x0=0.0,M_sh=1.0,mode='data'):
    '''
    ================================================================
                                                                    
        Function which computes the analytical solution              
        to the 1D shock-tube problem given left and right            
        initial conditions (assumes no shock reflections)            
                                                                    
        Usage: Shock_Tube_Exact(x_min,x_max,N,P4,T4,P1,T1,demo=False), where 
                                                                    
        X       = vector of all grid points on the domain [m]
        time    = solution time(s) when exact solution is desired
        x0      = initial location of the discontinuity [m]
        M_sh    = shock Mach number if already computed [-]
        P4      = Driver pressure [Pa]
        T4      = Driver temperature [K]
        P1      = Driven pressure [Pa]
        T1      = Driven temperature [K]
        time    = Solution time (vector or scalar) [s]
        mode    = Option for running the code in demo mode
                                                                    
    ================================================================
    '''
    global f_num

    import matplotlib.pyplot as plt
    import numpy as np
    import sys, os
    from time import sleep

    #define constants
    eps = np.finfo(float).eps
    pi = 4.0*np.arctan(1.0)
    gam = 1.4
    R = 286.9

    #set the run parameters
    if (mode == 'demo'):

        #example initial conditions (Sod '78)
        q_L = np.array([1, 0, 1e5/(gam-1)])
        q_R = np.array([0.125, 0, 1e4/(gam-1)])

        #set domain limits
        x = np.linspace(-1.0,1.0,501)
        x0 = 0.0
        t_vec = np.array([1e-3])

    else:

        #left and right state vectors
        rho_1 = P1/(R*T1)
        rho_4 = P4/(R*T4)
        q_L = np.array([rho_4, 0, P4/(gam-1)])
        q_R = np.array([rho_1, 0, P1/(gam-1)])

        #set domain limits
        x = X
        #defensive programming
        if np.isscalar(time):
            t_vec = np.array([time])
        else:
            t_vec = np.array(time)

    #total number of points on the grid
    N = x.shape[0] 

    #compute the acoustic speeds in the left- and right-states
    c_L = np.sqrt(gam*(gam-1)*q_L[2]/q_L[0])
    c_R = np.sqrt(gam*(gam-1)*q_R[2]/q_R[0])

    #solve the shock-tube equation using Newton-Raphson + Complex-Step Derivative
    P_41 = q_L[2]/q_R[2]
    if ((t_vec.shape[0]>1)or(np.isscalar(time) and abs(time)<eps)):
        err = 1.0; cntr = 0; h=1e-30; M_sh = 1.5
        Res = lambda M_sh: -P_41*(1-(gam-1)/(gam+1)*(c_R/c_L)*(M_sh**2-1)/M_sh)**(2*gam/(gam-1))+2*gam/(gam+1)*(M_sh**2-1)+1
        print('Solving the shock tube equation for P_41=%3.1f:' % P_41)
        while (err>eps): 
            cntr += 1; Msh_p = M_sh
            M_sh -= h*Res(M_sh)/np.imag(Res(complex(M_sh,h)))
            err = abs(M_sh/Msh_p-1)
            print('    N = %d, e = %1.6e' % (cntr,err))

            #limit total number of iterations
            if (cntr==15):
                print('\n  **Iteration limit exceeded (e = %1.6e)\n' % err)
                break

    #compute the exact solution for each time in t_vec
    Nt = t_vec.shape[0]
    alpha  = (gam+1)/(gam-1)
    Q_exact = np.zeros((N,3,Nt))
    for k in range(Nt):

        #current solution time
        t = t_vec[k]

        #compute shock velocity [m/s] and static pressure-jump P_21 = P2/P1
        v_sh = M_sh*c_R
        x_sh = x0 + v_sh*t
        P_21 = 1.0+2*gam/(gam+1)*(M_sh**2-1)

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
                xi = (x[i]-x_fR)/(x_ct-x_fR+eps)
                
                #x between shock and contact discontinuity
                if (xi>1.0):
                    rho = ((1+alpha*P_21)/(alpha+P_21))*q_R[0]
                    u   = u2
                    p   = P_21*((gam-1)*q_R[2])

                #x within the expansion fan
                elif (xi<0.0):
                    u   = 2/(gam+1)*(c_L+(x[i]-x0)/t)
                    p   = (gam-1)*q_L[2]*(1-(gam-1)*u/(2*c_L))**(2*gam/(gam-1))
                    rho = q_L[0]*(p/((gam-1)*q_L[2]))**(1/gam)

                #x between contact disconinuity and expansion fan
                else:
                    rho = q_L[0]*(P_21/P_41)**(1/gam)
                    u   = u3
                    p   = P_21*((gam-1)*q_R[2])

            #construct solution vector from the primitive variables
            Q_exact[i,0,k] = rho
            Q_exact[i,1,k] = rho*u
            Q_exact[i,2,k] = p/(gam-1)+0.5*rho*u**2

        #plot the solution if run without arguments 
        if (mode=='demo' or (mode=='testing' and k==(Nt-1))):

            #compute the variables
            P_plot = (gam-1)*(Q_exact[:,2,k]-Q_exact[:,1,k]**2/(2*Q_exact[:,0,k]))
            U = Q_exact[:,1,k]/Q_exact[:,0,k]
            Mach = np.sqrt(Q_exact[:,0,k]*U**2/(gam*P_plot))
            T_plot = P_plot/(R*Q_exact[:,0,k])
            c_plot = np.sqrt(gam*R*T_plot)
            Entropy = P_plot/(Q_exact[:,0,k])**(gam)
            e = Q_exact[:,2,k]/Q_exact[:,0,k]-0.5*U**2
            t_plt = float(1e3*t)

            #make the figures
            fig = plt.figure(f_num)
            ax1 = fig.add_subplot(221)
            ax1.plot(x,Q_exact[:,0,k],'-b',linewidth=3.0)
            ax1.set_title('Density (t=%1.3f[ms])' % t_plt)
            ax1.set(xlabel=r'$x\;[m]$', ylabel=r'$\rho\;[kg/m^3]$')
            ax2 = fig.add_subplot(222)
            ax2.plot(x,(1e-3)*P_plot,'-b',linewidth=3.0)
            ax2.set_title('Pressure (t=%1.3f[ms])' % t_plt)
            ax2.set(xlabel=r'$x\;[m]$', ylabel=r'$p\;[kPa]$')
            ax3 = fig.add_subplot(223)
            ax3.plot(x,U,'-b',linewidth=3.0)
            ax3.set_title('Velocity (t=%1.3f[ms])' % t_plt)
            ax3.set(xlabel=r'$x\;[m]$', ylabel=r'$V\;[m/s]$')
            ax4 = fig.add_subplot(224)
            ax4.plot(x,Mach,'-b',linewidth=3.0)
            ax4.set_title('Mach (t=%1.3f[ms])' % t_plt)
            ax4.set(xlabel=r'$x\;[m]$', ylabel=r'$M$')
            fig.tight_layout()
            #plt.savefig('fig_%d.pdf' % f_num)
            f_num += 1

            fig = plt.figure(f_num)
            ax1 = fig.add_subplot(221)
            ax1.plot(x,(1e-3)*e,'-b',linewidth=3.0)
            ax1.set_title('Specific Energy (t=%1.3f[ms])' % t_plt)
            ax1.set(xlabel=r'$x\;[m]$', ylabel=r'$e\;[kJ/kg]$')
            ax2 = fig.add_subplot(222)
            ax2.plot(x,c_plot,'-b',linewidth=3.0)
            ax2.set_title('Speed of Sound (t=%1.3f[ms])' % t_plt)
            ax2.set(xlabel=r'$x\;[m]$', ylabel=r'$c\;[m/s]$')
            ax3 = fig.add_subplot(223)
            ax3.plot(x,T_plot,'-b',linewidth=3.0)
            ax3.set_title('Temperature (t=%1.3f[ms])' % t_plt)
            ax3.set(xlabel=r'$x\;[m]$', ylabel=r'$T\;[K]$')
            ax4 = fig.add_subplot(224)
            ax4.plot(x,(1e-3)*Entropy,'-b',linewidth=3.0)
            ax4.set_title('Entropy (t=%1.3f[ms])' % t_plt)
            ax4.set(xlabel=r'$x\;[x]$', ylabel=r'$s\;[kJ/kgK]$')
            fig.tight_layout()
            #plt.savefig('fig_%d.pdf' % f_num)
            f_num += 1

            #show the plot(s)
            plt.show()
            plt.pause(3.0)

    #shrink data to R^2 if time input is a scalar
    if np.isscalar(time):
        Q_exact = Q_exact[:,:,0]
 
    if ((t_vec.shape[0]>1)or(np.isscalar(time) and abs(time)<eps)):
        print('Exact solution for 1D shock-tube has been generated.')

    return Q_exact,M_sh
