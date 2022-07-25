import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
import os
import ptasim2enterprise as p2e

import argparse

psr_list = np.loadtxt('psrs.dat', dtype = 'str')

N_psr = len(psr_list)

np.random.seed(seed = 20210524)


dataset_vol_fn = 'params/prior_range_test/dataset_ranges.csv'
dataset_vol = np.loadtxt(dataset_vol_fn, delimiter = ',')

prior_vol_fn = 'params/prior_range_test/prior_ranges.csv'
prior_vol = np.loadtxt(prior_vol_fn, delimiter = ',')

dP0s = dataset_vol[:,4]
dalphas = dataset_vol[:,5]
logA_lower_inj = dataset_vol[:, 0]
logA_upper_inj = dataset_vol[:, 1]


logA_lower = prior_vol[:, 0]
logA_upper = prior_vol[:, 1]
alpha_lower = prior_vol[:, 0]
alpha_upper = prior_vol[:, 1]
dP0s_prior = prior_vol[:, 4]
dalphas_prior = prior_vol[:, 5]
dlogA_prior = logA_upper - logA_lower


dP0prior, _ = np.meshgrid( logA_lower, dP0s)
print(dP0prior.shape)
#bf_matrix = np.nan*np.zeros_like(Alpha_lowers)
#N = np.arange(0, p0_lowers.shape[0])


# for dP0, dalpha in zip(dP0s, dalphas):


#     for logA_low, logA_up, alpha_low, alpha_up in zip(logA_lower, logA_upper, alpha_lower, alpha_upper):



inj_volumes = (logA_upper_inj - logA_lower_inj) * dalphas
prior_volumes = dlogA_prior * dalphas_prior
volume_ratio = (inj_volumes[:, None]) @ (1.0/prior_volumes)[None, :]

print(volume_ratio)


bf_matrix_list = []
for realisation_ind in range(0,10):
    bf_matrix = np.nan*np.zeros_like(dP0prior)
    incremental_result_file = 'result_logs/r{}_results_incremental_fin.txt'.format(realisation_ind)
    incremental_results = open(incremental_result_file, 'r')
    for line in incremental_results.readlines():
        result_dict = {'0': 1, '1': 1}
        l = line.split('_')
        result = l[-1]
        result = result[result.index('{')+1:result.index('}')]
        result = result.split(',')
        result_models = [r[0] for r in result]
        result_nsamp = [float(r.split(' ')[-1]) for r in result]
        if result_models == ['0'] and result_nsamp == [8250.0]:
            print(line)
            continue
        
        for r, n in zip(result_models, result_nsamp):
            #print(r, n)
            result_dict[r] = n
            _bf = result_dict['1'] / result_dict['0']
            dP0_ = float(l[0])
            dalpha_ = float(l[1])
            logA_low = float(l[4])
            
            dP0_ind = np.where(np.isclose(dP0s, dP0_))
            prior_ind = np.where(np.isclose(logA_lower , logA_low))
            bf_matrix[dP0_ind, prior_ind] = _bf
    bf_matrix_list.append(bf_matrix)

np.save('bf_prior_matrix_all.npy', np.array(bf_matrix_list))
bf_matrix = 10.0**(np.nanmean(np.log10(np.array(bf_matrix_list)), axis = 0))
np.save('bf_prior_matrix.npy', bf_matrix)
            
plt.imshow(np.nanstd(np.array(bf_matrix_list), axis = 0), origin = 'lower')
plt.savefig('stdbf.png')
plt.close()

print(bf_matrix.shape)

import matplotlib.colors
from matplotlib.ticker import LogLocator

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
})

font = {'family' : 'serif',
        'size'   : 14}
tickfont = {'family' : 'serif',
            'size'   : 12}

fig, ax = plt.subplots(1,1, figsize = (2*3.3, 2*2.04))

print(len(bf_matrix_list))
print(bf_matrix_list[0].shape)
print(np.array(bf_matrix_list).shape)
bf_errs = np.nanstd(np.array(bf_matrix_list), axis = 0)
print(bf_errs.shape)
#ax.plot([], [], label = r'$(\Delta \log_{{10}} P_0, \Delta \gamma) = $', alpha = 0.0)
#for realisation_ in range(0,10):
for ind in range(0, 2):#bf_matrix.shape[0]):
    #ax.plot(volume_ratio[ind,:], bf_matrix_list[realisation_][ind, :], label = '{} {}'.format(dP0s[ind], dalphas[ind]), color = 'C{}'.format(ind))
    #, yerr = bf_errs[ind, :], 
    ax.plot(volume_ratio[ind,:], bf_matrix[ind, :], label = r'$({}, {})$'.format(dP0s[ind], dalphas[ind]), color = 'C{}'.format(ind), linewidth = 1.5)
#for r in range(0, 10):
#    for ind in range(0, 2):#bf_matrix.shape[0]):
        #ax.plot(volume_ratio[ind,:], bf_matrix_list[realisation_][ind, :], label = '{} {}'.format(dP0s[ind], dalphas[ind]), color = 'C{}'.format(ind))
        #, yerr = bf_errs[ind, :], 
#        ax.plot(volume_ratio[ind,:], bf_matrix_list[r][ind, :], color = 'C{}'.format(ind), alpha = 0.2)#ecolor = '0.5', fmt = 'o-')

        
ax.set_xlabel(r'$\Phi$', fontdict = font)#(\Delta \log_{{10}} P_0 \times \Delta \gamma)_{{\mathrm{inj}}} / (\Delta \log_{{10}} A \times \Delta \gamma)_{{\mathrm{prior}}}$', fontsize = 12)
ax.set_ylabel(r'$\mathcal{{B}}^{{\mathrm{{CP1}}}}_{{\mathrm{{TN}}}}$', fontdict = font)
ax.legend(fontsize = 12, title = r'$(\Delta \log_{{10}} P_0, \Delta \gamma)$', title_fontsize = font['size'])
ax.tick_params(axis='y', labelsize = tickfont['size'])
ax.tick_params(axis='x', labelsize = tickfont['size'])

ax.set_ylim(1E-6, 1E6)
ax.axhline(1.0, linestyle = '--', c = 'k', zorder = 1, linewidth = 0.8)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim(0.05, 15)
plt.minorticks_on()
plt.savefig('prior_bf_vol.pdf', bbox_inches = 'tight')
plt.savefig('prior_bf_vol.png', bbox_inches = 'tight', dpi = 300)    

plt.close()
fig, ax = plt.subplots(1,1)
#'coolwarm'
im = plt.imshow(bf_matrix, cmap = "coolwarm", norm = matplotlib.colors.LogNorm(), origin = 'lower', extent = [*dalphas[[0,-1]], *dP0s[[0,-1]]], aspect = 'auto', clim = [1E-6, 1E6], interpolation = 'nearest') #
#im = plt.imshow(bf_matrix, cmap = 'coolwarm', norm = matplotlib.colors.LogNorm(), origin = 'lower', extent = [*dalphas[[0,-1]], *dlog10_As[[0,-1]]], aspect = 'auto', clim = [1E-6, 1E6], interpolation = 'none') #

cb = plt.colorbar(im)#, ticks = LogLocator(subs=range(10)))
cb.set_label(r'$\mathcal{{B}}^{{\mathrm{{CP2}}}}_{{\mathrm{{TN}}}}$', fontsize = font['size'])
cb.ax.minorticks_on()
#minorticks = im.norm(np.arange(1E-6, 1E6, 1))
#cb.ax.yaxis.set_ticks(minorticks, minor = True)
ax.set_xlabel(r'$\Delta \gamma$', fontdict = font)
ax.set_ylabel(r'$\Delta \log_{{10}}(P_0)$', fontdict = font)
ax.tick_params(axis='y', labelsize = tickfont['size'])
ax.tick_params(axis='x', labelsize = tickfont['size'])
plt.minorticks_on()
plt.savefig('prior_bf_matrix.pdf', bbox_inches = 'tight')
plt.savefig('prior_bf_matrix.png', bbox_inches = 'tight', dpi = 300)
# plt.show()
