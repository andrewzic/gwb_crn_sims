import numpy as np

def ptasim_spec(f, fc, P0, alpha):

    return P0*(1.0 + (f/fc)**2.0)**(-alpha/2.0)

def gw_spec(f, A, gamma):
    return A**2.0 / (12.0 * np.pi**2.0)*(f)**-gamma

