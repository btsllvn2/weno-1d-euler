def euler_1d_flux(q):
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

    #define wavespeed(s) on the grid
    ws = np.zeros(q.shape)
    for j in range(3):
        ws[:,j] = u+(j-1)*c 

    #maximum wavespeed on the domain for global LF splitting
    for j in range(q.shape[1]):
        ws[:,j] = la.norm(ws[:,j],np.inf)

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


def phi_weno5():
'''
    Function which compute the 5th order WENO reconstruction

    q is a nsxnv matrix of conservative variables (ns = num pts in current stencil)  
    f is a nsxnv matrix of conservative fluxes (ns = num pts in current stencil)  
    q_i_ip1 is a 2xnv matrix with nv variables to compute average state 

'''











    return f_hat_plus_half

