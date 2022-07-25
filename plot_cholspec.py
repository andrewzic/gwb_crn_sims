import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import gw_spectra_analytic as gwspec
from ptasim2enterprise import P02A
import astropy.units as u

import sys


plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
})

font = {'family' : 'serif',
        'size'   : 17}

spec_globstr = sys.argv[1]

spec_files = sorted(glob.glob(spec_globstr))

spec_dir = os.path.dirname(spec_files[0]) + '/'

fig, (ax1) = plt.subplots(1,1, figsize = (6,4))# ,figsize=(10,7))#, xscale = 'log', yscale = 'log')

plt.sca(ax1)

spec_ch0s = [] #list for lowest frequency channel - storing for help setting ylims

for specf in spec_files:
    print(specf)
    spec = np.loadtxt(specf)
    freq = spec[:, 0]/86400.0
    psd = spec[:, 1]*((1.0*u.year).to(u.s).value)**3.0
    spec_ch0s.append(psd[0])
    plt.plot(freq, psd, label = os.path.splitext(os.path.basename(specf))[0], color = '#9E9E9E', linewidth = 0.8, alpha = 0.5)

#ax1.legend()

max_spec_ch0 = np.amax(spec_ch0s)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'$\mathrm{Frequency~[Hz]}$', fontdict=font)
ax1.set_ylabel(r'$P~\mathrm{{[s}}^3\mathrm{{]}}$', fontdict = font)
ax1.tick_params(axis='y', labelsize = font['size'])
ax1.tick_params(axis='x', labelsize = font['size'])
ax1.set_ylim(10**(int(np.log10(ax1.get_ylim()[0]))), 10**(int(np.log10(max_spec_ch0))))
ax1.set_xlim(10**(int(np.log10(ax1.get_xlim()[0]) - 1)),1E-7)

fig.tight_layout()
plt.savefig('{}_cholspec_similar.png'.format(spec_dir), dpi = 300, bbox_inches = 'tight', facecolor='white')
#plt.show()
