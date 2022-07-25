import numpy as np

def P02A(P0, fc, gamma):

    logA = np.log10(np.sqrt(12.0*np.pi**2.0 * P0 * (1.0/fc)**-gamma)) #in general f_1yr/f_c but assume that f_c is in units of 1/yr

    return logA

def alpha2gamma(alpha):
    return alpha

def A2P0(logA, fc, gamma):

    #logA = np.log10(np.sqrt(12.0*np.pi**2.0 * P0 * (1.0/fc)**-gamma)) #in general f_1yr/f_c but assume that f_c is in units of 1/yr
    #10**logA = np.sqrt(12.0*np.pi**2.0 * P0 * (1.0/fc)**-gamma)
    #(10**logA)**2.0 = 12.0*np.pi**2.0 * P0 * (1.0/fc)**-gamma
    #P0 = A**2.0 / (12.0 * np.pi**2.0) * (1.0/fc)**gamma
    A = 10**logA
    P0 = A**2.0 / (12.0 * np.pi**2.0) * (1.0/fc)**gamma
    

    return P0


# alpha=-6.0
# P0=1.0e-25
# fc=0.01

# #print(P02A_2(P0, fc, alpha2gamma(alpha)))

# print(P02A(P0, fc, alpha2gamma(alpha)))
# print(alpha2gamma(alpha))
