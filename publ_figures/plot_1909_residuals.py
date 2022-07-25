import matplotlib.pyplot as plt
import numpy as np
import glob
import os

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
})

font = {'family' : 'serif',
        'size'   : 14}

#spec_globstr = sys.argv[1]

noise_samps = ['0.00_0.00', '1.40_1.60', '2.80_3.20', '2.80_4.00'] #
DP0s = [float(i.split('_')[0]) for i in noise_samps]
DALPHAs = [float(i.split('_')[1]) for i in noise_samps]

toa_dir_fmt = '/DATA/CETUS_3/zic006/ssb/ptasim/gwb_crn_sims/data/100r/regsamp_{}/output/real_0/'

toa_dirs = [toa_dir_fmt.format(i) for i in noise_samps]

res_files = ['{}/J1909-3744.resmjd'.format(dir) for dir in toa_dirs]

print(res_files)
resids = [np.loadtxt(i) for i in res_files]

fig, axs = plt.subplots(4, 1, figsize = (3.3, 4.0*2.04), sharex=True, sharey = True)

spec_ch0s_all = []

for a, resid, DP0, DALPHA in zip(axs, resids, DP0s, DALPHAs):
    print(resid.shape)
    #print(len(spec_file_set))
    mjds = resid[:, 3]
    rs = resid[:, 1] * 1E6
    r_err = resid[:, 2] * 1E6
    a.errorbar(mjds, rs, yerr = r_err, fmt = 'o', capsize = 0, linewidth = 0.8) #9E9E9E
    if a == axs[-1]:
        a.set_xlabel('Time (MJD)', fontdict = font)
    a.set_ylabel(r'Residual ($\mu$s)', fontdict = font)
    a.tick_params(axis='y', labelsize = 12)
    a.tick_params(axis='x', labelsize = 12)#font['size'])    
    

    
plt.savefig('J1909-3744_resid_samp.pdf', bbox_inches = 'tight', facecolor = 'white')
plt.savefig('J1909-3744_resid_samp.png', bbox_inches = 'tight', dpi = 300, facecolor = 'white')
plt.show()
    
folders = ['regsamp_1.40_1.60/', 'regsamp_2.80_3.20/', 'regsamp_2.80_4.00/']

#resid_files = ['{}/output/real_0/J1909-3744.resmjd'.format(i) for i in folders]







