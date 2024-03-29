import numpy as np
from scipy.stats import skew, normaltest
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import os
import pickle
import ptasim2enterprise as p2e

import argparse
import glob

import matplotlib.colors
from matplotlib.ticker import LogLocator
import cmasher as cmr

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
})

font = {'family' : 'serif',
        'size'   : 17}


psr_list = np.loadtxt('psrs.dat', dtype = 'str')
N_psr = len(psr_list)


realisations = np.arange(0, 100, 1)

np.random.seed(seed = 20210524)

alpha_uppers = np.linspace(0, -4, 41)[::-1]
alpha_lowers = np.linspace(-8, -4, 41)[::-1]

p0_lowers = np.linspace(-30, -23, 41)[::-1]
p0_uppers = np.linspace(-16, -23, 41) [::-1]

Alpha_lowers, P0_lowers = np.meshgrid(alpha_lowers, p0_lowers)
Alpha_uppers, P0_uppers = np.meshgrid(alpha_uppers, p0_uppers)

N = np.arange(0, p0_lowers.shape[0])

dalphas = []
dP0s = []
dlog10_As = []
for y in N:

    alphas = np.random.uniform(alpha_lowers[y], alpha_uppers[y], size = N_psr)
    dalpha = alpha_uppers[y] - alpha_lowers[y]
    #toaerrs = np.random.uniform(9e-8, 5e-7, size = N_psr)
    dalphas.append(dalpha)

dalphas = np.array(sorted(dalphas))
log10A_lowers = []
log10A_uppers = []
for x in N:

    p0s = 10**(np.random.uniform(p0_lowers[x], p0_uppers[x], size = N_psr))
    log10As = p2e.P02A(p0s, 0.01, -alphas)
    #print(alpha_lowers[x])
    log10A_lower = p2e.P02A(10**p0_lowers[x], 0.01, -1.0*np.array([alpha_lowers[x], alpha_uppers[x]]))
    log10A_upper = p2e.P02A(10**p0_uppers[x], 0.01, -1.0*np.array([alpha_lowers[x], alpha_uppers[x]]))
    log10A_lowers.append(log10A_lower)
    log10A_uppers.append(log10A_upper)
    dP0 = p0_uppers[x] - p0_lowers[x]
    dP0s.append(dP0)
    dlog10_A = np.max(log10A_upper) - np.min(log10A_lower)
    dlog10_As.append(dlog10_A)

dP0s = np.array(sorted(dP0s))
dlog10_As = np.array(dlog10_As)

log10A_central = p2e.P02A(10**-23.0, 0.01, 4.0)
log10A_uppers = np.array(log10A_uppers)
log10A_lowers = np.array(log10A_lowers)

for i in range(len(alpha_lowers)):
    print('{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(np.min(log10A_lowers[i]) , np.max(log10A_uppers[i]), np.abs(alpha_uppers[i]), np.abs(alpha_lowers[i]), dP0s[i], dalphas[i]))
