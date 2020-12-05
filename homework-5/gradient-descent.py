import numpy as np
import matplotlib.pyplot as plt

thresh = 1e-14
rate = 0.1

# Error function
def err(u, v):
    return (u * np.exp(v) - 2 * v * np.exp(-u))**2

# Gradient function of u 
def dE_u(u, v):
    return (2 * (np.exp(v) + 2 * v * np.exp(-u)) * (u *  np.exp(v) - 2 * v * np.exp(-u))) 

# Gradient function of v
def dE_v(u, v):
    return (2 * (u * np.exp(v) - 2 * np.exp(-u)) * (u *  np.exp(v) - 2 * v * np.exp(-u)))

def part_1(iter):

    ut = u = 1.0
    vt = v = 1.0

    # initialize error function
    E_uv = err(u, v) 

    # while error is greater than thresh
    while E_uv >= thresh: 
        
        ut1 = ut - rate * dE_u(ut, vt) # compute u at t + 1
        vt1 = vt - rate * dE_v(ut, vt) # compute v at t + 1

        ut = ut1 # u of t + 1 now becomes u of t
        vt = vt1 # v of t + 1 now becomes v of t

        # compute new error
        E_uv = err(ut1, vt1) 
        
        iter += 1

    print("iteration: {} ".format(iter))
    print("(u,v) = ({},{})".format(round(ut, 3), round(vt, 3)))

def part_2(iter):
    
    ut = u = 1.0
    vt = v = 1.0

    # initialize error function
    E_uv = err(u, v) 

    # while error is greater than thresh
    for i in range(iter):

        ut1 = ut - rate * dE_u(ut, vt) # compute u at t + 1
        ut = ut1 # u of t + 1 now becomes u of t

        vt1 = vt - rate * dE_v(ut, vt) # compute v at t + 1
        vt = vt1 # v of t + 1 now becomes v of t

        # compute new error
        E_uv = err(ut1, vt1) 

    print("E(u,v) = {:01.0e}".format(E_uv))

part_1(0)
part_2(15)