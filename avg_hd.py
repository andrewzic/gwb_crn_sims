import numpy as np

import astropy
from astropy.coordinates import SkyCoord as s, Angle as a
import astropy.units as u

import  matplotlib.pyplot as plt

def hellingsdowns(zeta):
    x = (1.0 - np.cos(zeta))/2.0
    gamma = 0.5 - 0.25 * x + 1.5 * x * np.log(x)
    return gamma

coordstrings = np.loadtxt('psr_coords.dat', dtype = 'str')
print(coordstrings)

for r in coordstrings:
    print(r[0], r[1])
coords = s([s(a(r[0], unit = u.hourangle), a(r[1], unit = u.deg)) for r in coordstrings])

print(coords)
dist_matrix = np.zeros(shape=(coords.shape[0], coords.shape[0]))*np.nan
corr_matrix = np.zeros_like(dist_matrix)*np.nan


for p1_ind in range(coords.shape[0]-1):
    dist_matrix[p1_ind, p1_ind] = 0.0
    corr_matrix[p1_ind, p1_ind] = hellingsdowns(0.0*u.rad).value
    for p2_ind in range(p1_ind + 1, coords.shape[0]):
        print(p1_ind, p2_ind)
        #print(coords[p1_ind])
        #print(coords[p1_ind].separation(coords[p2_ind]))
        dist_matrix[p1_ind, p2_ind] = coords[p1_ind].separation(coords[p2_ind]).to(u.rad).value
        corr_matrix[p1_ind, p2_ind] = hellingsdowns(dist_matrix[p1_ind, p2_ind]*u.rad).value


#corr_matrix[corr_matrix == 0 
#print(dist_matrix)



plt.imshow(corr_matrix)



plt.show()


mean_hd = np.nanmean(corr_matrix)
var_hd = np.nanvar(corr_matrix)

print(f'{mean_hd:.4f}, {var_hd:.4f}')

size = corr_matrix[~np.isnan(dist_matrix)].flatten().size
plt.plot(dist_matrix[~np.isnan(dist_matrix)].flatten(), corr_matrix[~np.isnan(dist_matrix)].flatten() + np.random.normal(0, 0.1, size = size), '.')
plt.show()
