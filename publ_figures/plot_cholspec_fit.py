import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from gw_spectra_analytic import gw_spec
from ptasim2enterprise import P02A
import astropy.units as u

import json

import sys


plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
})

font = {'family' : 'serif',
        'size'   : 14}

#spec_globstr = sys.argv[1]

noise_samps = ['0.00_0.00', '1.40_1.60', '2.80_3.20', '2.80_4.00']
DP0s = [float(i.split('_')[0]) for i in noise_samps]
DALPHAs = [float(i.split('_')[1]) for i in noise_samps]

spec_dir_fmt = '/DATA/CETUS_3/zic006/ssb/ptasim/gwb_crn_sims/data/100r/regsamp_{}/output/real_0/'

spec_dirs = [spec_dir_fmt.format(i) for i in noise_samps]

spec_file_sets = [sorted(glob.glob('{}/*.spec'.format(dir))) for dir in spec_dirs]
noisefiles = [json.load(open('{}/noisefiles/_noise.json'.format(s), 'r')) for s in spec_dirs]


print(len(spec_file_sets))
fig, axs = plt.subplots(4, 1, figsize = (3.3, 3.0*2.04), sharex = True, sharey = True)

spec_ch0s_all = []

for a, spec_file_set, noisefile, DP0, DALPHA in zip(axs, spec_file_sets, noisefiles, DP0s, DALPHAs):
    print(len(spec_file_set))
    
    print(noisefile)
    gw_log10_A = noisefile['gw_log10_A']
    gw_gamma = noisefile['gw_gamma']
    print('"FSDFHDSAFSAIF', gw_log10_A, gw_gamma)
    
    for spec_files in spec_file_set:
        
#fig, (ax1) = plt.subplots(1,1, figsize = (6,4))# ,figsize=(10,7))#, xscale = 'log', yscale = 'log')

        #plt.sca(ax)

        spec_ch0s = [] #list for lowest frequency channel - storing for help setting ylims
        #        print(spec_files[0])
        #        for specf in spec_files:
        #print(specf)
        spec = np.loadtxt(spec_files)
        freq = spec[:, 0]/86400.0
        psd = spec[:, 1]*((1.0*u.year).to(u.s).value)**3.0
        spec_ch0s.append(psd[0])
        if spec_files == spec_files[0]:
            label = os.path.splitext(os.path.basename(spec_files))[0]
        else:
            label = None
                
        a.plot(freq, psd, label = label, color = 'C0', linewidth = 0.8, alpha = 0.5) #9E9E9E

            #ax.legend()
    
    x_freq = np.logspace(np.log10(freq[0]/10.0), np.log10(freq[-1]*10.0), 10000)
    print(x_freq[0], x_freq[-1])
    model_spec = gw_spec(x_freq*86400.0*365.25, 10.0**gw_log10_A, gw_gamma)*((1.0*u.year).to(u.s).value)**3.0
    #model_spec +=

    a.plot(x_freq, model_spec, color = 'C1', linewidth = 1.2)
    max_spec_ch0 = np.amax(spec_ch0s)
    spec_ch0s_all.append(max_spec_ch0)
    a.text(9E-8, 1E-2, r'$\Delta P_0 = {:.1f}$, $\Delta \alpha = {:.1f}$'.format(DP0, DALPHA), fontsize = 10, horizontalalignment = 'right')
    #a.text(
    a.set_xscale('log')
    a.set_yscale('log')
    if DALPHA == DALPHAs[-1]:##a == axs[-1]:
        a.set_xlabel(r'Frequency$\,[\mathrm{{Hz}}]$', fontdict=font)
    a.set_ylabel(r'$\mathcal{P}\,[\mathrm{{s}}^{{3}}]$', fontdict = font)
    a.tick_params(axis='y', labelsize = 12)
    a.tick_params(axis='x', labelsize = 12)#font['size'])
    
    a.set_xlim(1E-9,1E-7)

for ax in axs:
    ax.minorticks_on()
    ax.set_ylim(10**-10, 10**(int(np.log10(np.amax(spec_ch0s_all)))+2))
    #plt.legend()
#(int(np.log10(ax.get_ylim()[0])))
    
fig.tight_layout()
plt.savefig('_cholspec_range_fit.png', dpi = 300, bbox_inches = 'tight', facecolor='white')
plt.savefig('_cholspec_range_fit.pdf',  bbox_inches = 'tight', facecolor='white')
#plt.show()
