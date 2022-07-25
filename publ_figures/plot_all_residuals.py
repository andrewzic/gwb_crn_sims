import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import glob
import os

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
})

font = {'family' : 'serif',
        'size'   : 12}

tickfont = {'family' : 'serif',
        'size'   : 12}

#spec_globstr = sys.argv[1]

noise_samps = ['2.80_4.00'] #
DP0s = [float(i.split('_')[0]) for i in noise_samps]
DALPHAs = [float(i.split('_')[1]) for i in noise_samps]

psrs = np.loadtxt('/DATA/CETUS_3/zic006/ssb/ptasim/gwb_crn_sims/psrs.dat', dtype = str)

toa_dir = '/DATA/CETUS_3/zic006/ssb/ptasim/gwb_crn_sims/data/100r/regsamp_2.80_4.00/output/real_0/'



res_files = ['{}/{}.resmjd'.format(toa_dir, psr) for psr in psrs]

print(res_files)
resids = [np.loadtxt(i) for i in res_files]

fig, axs = plt.subplots(13, 2, figsize = (2.0*3.3, 2.0*3.5*2.04), sharex=True)#, sharey = True)

spec_ch0s_all = []

_idx = 0
for a, resid, psr in zip(axs.flatten(),  resids, psrs):

    
    
    print(resid.shape)
    #print(len(spec_file_set))
    mjds = resid[:, 3]
    rs = resid[:, 1] * 1E6
    r_err = resid[:, 2] * 1E6
    trans = transforms.blended_transform_factory(a.transAxes, a.transData)
    _title = a.text(0.5, 0.99, psr, fontsize = 10, ha = 'center', va = 'center', transform = a.transAxes)
    
    
    a.errorbar(mjds, rs, yerr = r_err, fmt = 'o', capsize = 0, linewidth = 0.3, markersize = 0.5, ecolor = '0.5') #9E9E9E
    _titlecoords = a.transAxes.transform(_title.get_position())
    print(_titlecoords)
    yt = _titlecoords - 0.1
    a.plot([+0.01, +0.01], [-0.5, 0.5], linewidth = 0.8, c = 'k', alpha = 1, zorder = 9999, transform = trans)
    a.plot([+0.015, +0.005], [-0.5, -0.5], linewidth = 0.8, c = 'k', alpha = 1, zorder = 9999, transform = trans)
    a.plot([+0.015, +0.005], [0.5, 0.5], linewidth = 0.8, c = 'k', alpha = 1, zorder = 9999, transform = trans)
    #a.errorbar(0.1, 0, yerr = 0.5, markersize = 0, capsize = 0.8, linewidth = 0.8, c = '0.0', zorder = 99999, alpha = 0.5) #9E9E9E

    a.tick_params(axis='y', labelsize = 12)
    a.tick_params(axis='x', labelsize = 12)#font['size'])    
    max_ylims = np.amax(np.abs(a.get_ylim()))
    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)
    for pos in ['left', 'right', 'bottom', 'top']:
        a.spines[pos].set_visible(False)
    #a.axis('off')

    if _idx > 23:
    #if psr == psrs[24] or psr == psrs[25]:
        a_pos = a.get_position().bounds
        a_mid_x = (a_pos[0] + a_pos[2]/2.0)
        fig.text(a_mid_x, a_pos[1] - 1.08*a_pos[3], 'Time (MJD)', fontdict = font, ha = 'center')
        a.get_xaxis().set_visible(True)
        a.spines['bottom'].set_visible(True)
        a.minorticks_on()
        for label in a.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')
            label.set_fontsize(10)
        #a.set_xticklabels(a.get_xticklabels(), rotation = 45, ha = 'right')
#        a.set_xlabel('Time (MJD)', fontdict = font)
    #if _idx%2 == 0 :
    #    a.set_ylabel(r'Residual ($\mu$s)', fontdict = font)

    _idx += 1    
    a.set_ylim(-1.05*max_ylims, 1.05*max_ylims) #make y-axis symmetric and give a little bit more range

#x1 = axs.flatten()[24].text(0.5, 0, 'Time (MJD)', fontdict = font)
#x2 = axs.flatten()[25].text(0.5, 0, 'Time (MJD)', fontdict = font)

#ax

#x1 = axs.flatten()[24].set_xlabel('Time (MJD)', fontdict = font)
#x2 = axs.flatten()[25].set_xlabel('Time (MJD)', fontdict = font)

#print(x1)
#fig.text(0.25, 0.04, 'Time (MJD)', fontdict = font, ha='center')
#fig.text(0.75, 0.04, 'Time (MJD)', fontdict = font, ha='center')
fig.text(0.08, 0.5, r'Residual ($\mu$s)', fontdict = font, va='center', rotation='vertical')
plt.subplots_adjust(wspace = 0.02, hspace = 0.02)
plt.savefig('all_resid_samp.pdf', bbox_inches = 'tight', facecolor = 'white')
plt.savefig('all_resid_samp.png', bbox_inches = 'tight', dpi = 300, facecolor = 'white')
#plt.show()
    
folders = ['regsamp_1.40_1.60/', 'regsamp_2.80_3.20/', 'regsamp_2.80_4.00/']

#resid_files = ['{}/output/real_0/J1909-3744.resmjd'.format(i) for i in folders]







