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
        'size'   : 14}
tickfont = {'family' : 'serif',
        'size'   : 12}


psr_list = np.loadtxt('psrs.dat', dtype = 'str')
N_psr = len(psr_list)


realisations = np.arange(0, 100, 1)

np.random.seed(seed = 20210524)

alpha_uppers = np.linspace(0, -4, 11)[::-1]
alpha_lowers = np.linspace(-8, -4, 11)[::-1]

p0_lowers = np.linspace(-30, -23, 11)[::-1]
p0_uppers = np.linspace(-16, -23, 11) [::-1]

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
ddalphas = np.median(np.abs(np.diff(dalphas)))
for x in N:

    p0s = 10**(np.random.uniform(p0_lowers[x], p0_uppers[x], size = N_psr))
    log10As = p2e.P02A(p0s, 0.01, -alphas)
    #print(alpha_lowers[x])
    log10A_lower = p2e.P02A(10**p0_lowers[x], 0.01, -1.0*np.array([alpha_lowers[x], alpha_uppers[x]]))
    log10A_upper = p2e.P02A(10**p0_uppers[x], 0.01, -1.0*np.array([alpha_lowers[x], alpha_uppers[x]]))

    dP0 = p0_uppers[x] - p0_lowers[x]
    dP0s.append(dP0)
    dlog10_A = np.max(log10A_upper) - np.min(log10A_lower)
    dlog10_As.append(dlog10_A)

dP0s = np.array(sorted(dP0s))
ddP0s = np.median(np.abs(np.diff(dP0s)))
dlog10_As = np.array(dlog10_As)

log10A_central = p2e.P02A(10**-23.0, 0.01, 4.0)

os_matrix =          np.nan*np.zeros((dP0s.shape[0],
                                      dalphas.shape[0],
                                      realisations.shape[0]))
os_marg_matrix =     np.nan*np.zeros((dP0s.shape[0],
                                      dalphas.shape[0],
                                      realisations.shape[0]))
os_snr_matrix =      np.nan*np.zeros((dP0s.shape[0],
                                      dalphas.shape[0],
                                      realisations.shape[0]))
os_marg_snr_matrix = np.nan*np.zeros((dP0s.shape[0],
                                      dalphas.shape[0],
                                      realisations.shape[0]))


corr_os_matrix  = pickle.load(open('corr_os_matrix.pkl', 'rb'))
corr_os_marg_matrix = pickle.load(open('corr_os_marg_matrix.pkl', 'rb'))
corr_sqrtos_matrix = pickle.load(open('corr_sqrtos_matrix.pkl', 'rb'))
corr_sqrtos_marg_matrix = pickle.load(open('corr_sqrtos_marg_matrix.pkl', 'rb'))
corr_os_snr_matrix = pickle.load(open('corr_os_snr_matrix.pkl', 'rb'))
corr_os_marg_snr_matrix = pickle.load(open('corr_os_marg_snr_matrix.pkl', 'rb'))

corr_labels = {'hd': 'hd', 'dipole': 'dp', 'monopole': 'mp'}

def plot_matrix(matrix, norm = None, type = 'os', measure = 'A', label = 'hd', dalphas = dalphas, dP0s = dP0s, close = True, cb_label = None, ax_title = None):

  measure_dict = {'A': r'$\hat{{A}}_{{\mathrm{{{}}}}}^{{2}}$', 'sA': r'$|\hat{{A}}_{{\mathrm{{{}}}}}|$', 'S/N': r'$\rho(\mathrm{{{}}})$', 'SNR': '$\rho(\mathrm{{{}}})$', 'N': '$N_\mathrm{{{}}}$'}

  measure_fignames = {'sA': 'A', 'A': 'A2', 'S/N': 'snr', 'SNR': 'snr', 'N': 'N'}

  measure_str = measure_dict[measure]
  figname_pref = 'os_{}_{}'.format(measure_fignames[measure], label)
  fig, ax = plt.subplots(1,1, figsize = (2.0*3.3, 1.6*3.3))
  im = plt.imshow(np.abs(matrix), cmap = cmr.ocean_r, norm = norm, origin = 'lower', extent = [dalphas[0], dalphas[-1] + ddalphas, dP0s[0], dP0s[-1] + ddP0s], aspect = (8.0 + ddalphas)/(14.0 + ddP0s), interpolation = 'nearest')#, clim = [1E-6, 1E6]) #
  #im = plt.imshow(bf_matrix, cmap = 'inferno', norm = matplotlib.colors.LogNorm(), origin = 'lower', extent = [*dalphas[[0,-1]], *dlog10_As[[0,-1]]], aspect = 'auto', clim = [1E-6, 1E6], interpolation = 'none') #

  if cb_label == None:
      cb_label = measure_str.format(label.replace('marg_', ''))

  cb = plt.colorbar(im)#, ticks = LogLocator(subs=range(10)))
  cb.set_label(cb_label, fontsize = font['size'])
  cb.ax.minorticks_on()
  #minorticks = im.norm(np.arange(1E-6, 1E6, 1))
  #cb.ax.yaxis.set_ticks(minorticks, minor = True)
  ax.set_xlabel(r'$\Delta \alpha$', fontdict = font)
  ax.set_ylabel(r'$\Delta \log_{{10}}(P_0)$', fontdict = font)
  ax.tick_params(axis='y', labelsize = tickfont['size'])
  ax.tick_params(axis='x', labelsize = tickfont['size'])
  if ax_title != None:
      ax.set_title(ax_title, fontsize = font['size'])
  plt.minorticks_on()

  bf_matrix = np.load('bf_matrix_vg.npy', allow_pickle = True)
  #X,Y = np.meshgrid(zoom(dalphas, 3), zoom(dP0s, 3))
  X,Y = np.meshgrid(dalphas, dP0s)
  bf_matrix_smooth = bf_matrix#zoom(bf_matrix, 3)
  ax.contour(X,Y,np.log10(bf_matrix_smooth), cmap = 'inferno', levels = [ 0, 2, 4, 5, 6])
  
  plt.savefig('{}.png'.format(figname_pref), dpi = 300, bbox_inches = 'tight')
  plt.savefig('{}.pdf'.format(figname_pref), bbox_inches = 'tight')
  if close:
    plt.close()
  else:
    plt.show()
# os_pickles = []
# dP0_inds = []
# dalpha_inds = []
# result_realisations = []
# for realisation_ind in realisations:
#   result_dirs = sorted(glob.glob('enterprise_out/100r/spincommonfixgamma/pe_array_spincommonfixgamma_*_20210803_r{}/*/'.format(realisation_ind))) #parent dirs
#   result_dP0s = [float(i.split('/')[-2].split('_')[-3]) for i in result_dirs]
#   result_dalphas = [float(i.split('/')[-2].split('_')[-2]) for i in result_dirs]
#   os_pickle_files = [i + '_os_results.pkl' for i in result_dirs]
#   _os_pickles = [pickle.load(open(i, 'rb')) if os.path.exists(i) else None for i in os_pickle_files]
#   os_pickles.extend(_os_pickles)
#   _dP0_inds = [np.argmin(np.abs(dP0s - i)) for i in result_dP0s]
#   dP0_inds.extend(_dP0_inds)
#   _dalpha_inds = [np.argmin(np.abs(dalphas - i)) for i in result_dalphas]
#   dalpha_inds.extend(_dalpha_inds)
#   chain_files = [i + '/chain_1.txt' for i in result_dirs]
#   result_realisations.extend([realisation_ind for i in _os_pickles])

#print(len(os_pickles), len(dP0_inds), len(dalpha_inds), len(result_realisations))

mean_corr_os_matrix          = dict()
mean_corr_os_marg_matrix     = dict()
mean_corr_sqrtos_matrix          = dict()
mean_corr_sqrtos_marg_matrix     = dict()
max_corr_os_matrix           = dict()
max_corr_os_marg_matrix      = dict()
std_corr_os_matrix           = dict()
std_corr_os_marg_matrix      = dict()
skew_corr_os_matrix          = dict()
skew_corr_os_marg_matrix     = dict()
normal_corr_os_matrix         = dict()
normal_corr_os_marg_matrix    = dict()
mean_corr_os_snr_matrix      = dict()
mean_corr_os_marg_snr_matrix = dict()
max_corr_os_snr_matrix       = dict()
max_corr_os_marg_snr_matrix  = dict()
std_corr_os_snr_matrix       = dict()
std_corr_os_marg_snr_matrix  = dict()
skew_corr_os_snr_matrix      = dict()
skew_corr_os_marg_snr_matrix = dict()
normal_corr_os_snr_matrix     = dict()
normal_corr_os_marg_snr_matrix= dict()
N3sig_corr_os_matrix         = dict()
N3sig_corr_os_marg_matrix    = dict()
N2sig_corr_os_matrix         = dict()
N2sig_corr_os_marg_matrix    = dict()

_, bins, _ = plt.hist(np.concatenate([corr_os_snr_matrix[i] for i in ['hd', 'dipole', 'monopole']]).flatten(), bins = 1000)
#_, marg_bins, _ = plt.hist(np.concatenate([corr_os_marg_snr_matrix[i] for i in ['hd', 'dipole', 'monopole']]).flatten(), bins = 1000)
plt.close()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (2.0*3.3, 4.0*2.2), sharex = True, sharey = True)
labels = {'hd': 'HD', 'dipole': 'Dipole', 'monopole': 'Monopole'}
zorders = {'hd': 100, 'dipole': 50, 'monopole': 10}
colors = {'hd': 'C0', 'dipole': 'C1', 'monopole': 'C2'}
for corr in ['hd', 'dipole', 'monopole']:
    ax1.hist(corr_os_snr_matrix[corr].flatten(), label = labels[corr], density = True, bins = bins, histtype = 'step', zorder = zorders[corr])
    #ax1.hist(corr_os_marg_snr_matrix[corr].flatten(), label = labels[corr], density = True, bins = bins, histtype = 'step', zorder = zorders[corr] - 5, linestyle = 'dashed', color = colors[corr])
    ax1.set_title('Maximum-likelihood', fontdict = font)
    ax2.hist(corr_os_marg_snr_matrix[corr].flatten(), label = labels[corr], density = True, bins = bins, histtype = 'step', zorder = zorders[corr])
    ax2.set_title('Noise-marginalized', fontdict = font)


ax1.axvline(3, linestyle = '--', c = 'k', zorder= 1000)
ax2.axvline(3, linestyle = '--', c = 'k', zorder = 1000)
ax1.set_xlim(-10,10)
#ax2.set_xlim(-8,8)
ax1.tick_params(axis='y', labelsize = tickfont['size'])
ax1.tick_params(axis='x', labelsize = tickfont['size'])
ax2.tick_params(axis='y', labelsize = tickfont['size'])
ax2.tick_params(axis='x', labelsize = tickfont['size'])
ax2.set_xlabel(r'$\rho$', fontdict = font)
ax1.set_ylabel(r'Density', fontdict = font)
ax2.set_ylabel(r'Density', fontdict = font)
ax1.set_yscale('log')
ax2.set_yscale('log')
ax1.legend()
fig.subplots_adjust(hspace = 0.3)
plt.savefig('./os_snr_hist.pdf', bbox_inches = 'tight')




# for corr in ['hd', 'dipole', 'monopole']:

#   corr_label = corr_labels[corr]
#   print(corr_label)
        
#   mean_corr_os_matrix[corr] = np.nanmean(corr_os_matrix[corr], axis = 2)
#   mean_corr_os_marg_matrix[corr] = np.nanmean(corr_os_marg_matrix[corr], axis = 2)
#   mean_corr_sqrtos_matrix[corr] = np.nanmean(corr_sqrtos_matrix[corr], axis = 2)
#   mean_corr_sqrtos_marg_matrix[corr] = np.nanmean(corr_sqrtos_marg_matrix[corr], axis = 2)  
#   max_corr_os_matrix[corr] = np.nanmax(corr_os_matrix[corr], axis = 2)
#   max_corr_os_marg_matrix[corr] = np.nanmax(corr_os_marg_matrix[corr], axis = 2)
#   std_corr_os_matrix[corr] = np.nanstd(corr_os_matrix[corr], axis = 2)
#   std_corr_os_marg_matrix[corr] = np.nanstd(corr_os_marg_matrix[corr], axis = 2)
#   skew_corr_os_matrix[corr] = skew(corr_os_matrix[corr], axis = 2, nan_policy = 'omit')
#   skew_corr_os_marg_matrix[corr] = skew(corr_os_marg_matrix[corr], axis = 2, nan_policy = 'omit')
#   #normal_corr_os_matrix[corr] = normaltest(corr_os_matrix[corr], axis = 2, nan_policy = 'omit')[1]
#   #normal_corr_os_marg_matrix[corr] = normaltest(corr_os_marg_matrix[corr], axis = 2, nan_policy = 'omit')[1]
#   mean_corr_os_snr_matrix[corr] = np.nanmean(corr_os_snr_matrix[corr], axis = 2)
#   mean_corr_os_marg_snr_matrix[corr] = np.nanmean(corr_os_marg_snr_matrix[corr], axis = 2)
#   max_corr_os_snr_matrix[corr] = np.nanmean(corr_os_snr_matrix[corr], axis = 2)
#   max_corr_os_marg_snr_matrix[corr] = np.nanmean(corr_os_marg_snr_matrix[corr], axis = 2)
#   std_corr_os_snr_matrix[corr] = np.nanmean(corr_os_snr_matrix[corr], axis = 2)
#   std_corr_os_marg_snr_matrix[corr] = np.nanmean(corr_os_marg_snr_matrix[corr], axis = 2)
#   skew_corr_os_snr_matrix[corr] = skew(corr_os_snr_matrix[corr], axis = 2, nan_policy = 'omit')
#   skew_corr_os_marg_snr_matrix[corr] = skew(corr_os_marg_snr_matrix[corr], axis = 2, nan_policy = 'omit')
#   #normal_corr_os_snr_matrix[corr] = normaltest(corr_os_snr_matrix[corr], axis = 2, nan_policy = 'omit')[1]
#   #normal_corr_os_marg_snr_matrix[corr] = normaltest(corr_os_marg_snr_matrix[corr], axis = 2, nan_policy = 'omit')[1]
#   N3sig_corr_os_matrix[corr] = np.nansum(corr_os_snr_matrix[corr] > 3, axis = 2)
#   print('N > 3sigma for {} OS: {:.5f}'.format(corr,np.nansum(corr_os_snr_matrix[corr] > 3)/np.count_nonzero(~np.isnan(corr_os_snr_matrix[corr]))))
#   N3sig_corr_os_marg_matrix[corr] = np.nansum(corr_os_marg_snr_matrix[corr] > 3, axis = 2)
#   print('N > 3sigma for {} OS: {:.5f}'.format(corr,np.nansum(corr_os_marg_snr_matrix[corr] > 3)/np.count_nonzero(~np.isnan(corr_os_snr_matrix[corr]))))
#   N2sig_corr_os_matrix[corr] = np.nansum(corr_os_snr_matrix[corr] > 2, axis = 2)
#   N2sig_corr_os_marg_matrix[corr] = np.nansum(corr_os_marg_snr_matrix[corr] > 2, axis = 2)

#   plot_matrix(mean_corr_os_matrix[corr], norm = matplotlib.colors.LogNorm(), measure = 'A', label = 'mean_{}'.format(corr_label))
#   plot_matrix(mean_corr_os_marg_matrix[corr], norm = matplotlib.colors.LogNorm(), measure = 'A', label = 'mean_marg_{}'.format(corr_label))
#   plot_matrix(np.log10(mean_corr_sqrtos_matrix[corr]), measure = 'sA', label = 'sqrt_mean_{}'.format(corr_label))
#   plot_matrix(np.log10(mean_corr_sqrtos_marg_matrix[corr]), measure = 'sA', label = 'sqrt_mean_marg_{}'.format(corr_label))
#   print('sqrt done')
#   plot_matrix(np.abs(mean_corr_sqrtos_matrix[corr] - log10A_central), measure = 'sA', label = 'resid_sqrt_mean_{}'.format(corr_label))
#   plot_matrix(np.abs(np.log10(mean_corr_sqrtos_marg_matrix[corr]) - log10A_central), measure = 'sA', label = 'resid_sqrt_mean_marg_{}'.format(corr_label)) 
#   plot_matrix(mean_corr_os_matrix[corr], norm = matplotlib.colors.LogNorm(), measure = 'A', label = 'mean_{}'.format(corr_label))
#   plot_matrix(mean_corr_os_marg_matrix[corr], norm = matplotlib.colors.LogNorm(), measure = 'A', label = 'mean_marg_{}'.format(corr_label))  
#   plot_matrix(max_corr_os_matrix[corr], norm = matplotlib.colors.LogNorm(), measure = 'A', label = 'max_{}'.format(corr_label))
#   plot_matrix(max_corr_os_marg_matrix[corr], norm = matplotlib.colors.LogNorm(), measure = 'A', label = 'max_marg_{}'.format(corr_label))
#   plot_matrix(std_corr_os_matrix[corr], norm = matplotlib.colors.LogNorm(), measure = 'A', label = 'std_{}'.format(corr_label))
#   plot_matrix(std_corr_os_marg_matrix[corr], norm = matplotlib.colors.LogNorm(), measure = 'A', label = 'std_marg_{}'.format(corr_label))
#   plot_matrix(skew_corr_os_matrix[corr], norm = matplotlib.colors.LogNorm(), measure = 'A', label = 'skew_{}'.format(corr_label))
#   plot_matrix(skew_corr_os_marg_matrix[corr], norm = matplotlib.colors.LogNorm(), measure = 'A', label = 'skew_marg_{}'.format(corr_label))
#   #plot_matrix(normal_corr_os_matrix[corr], norm = matplotlib.colors.LogNorm(), measure = 'A', label = 'Pnorm_{}'.format(corr_label))
#   #plot_matrix(normal_corr_os_marg_matrix[corr], norm = matplotlib.colors.LogNorm(), measure = 'A', label = 'Pnorm_marg_{}'.format(corr_label))
#   plot_matrix(mean_corr_os_snr_matrix[corr], measure = 'SNR', label = 'mean_{}'.format(corr_label))
#   plot_matrix(mean_corr_os_marg_snr_matrix[corr], measure = 'SNR', label = 'mean_marg_{}'.format(corr_label))
#   plot_matrix(max_corr_os_snr_matrix[corr], measure = 'SNR', label = 'max_{}'.format(corr_label))
#   plot_matrix(max_corr_os_marg_snr_matrix[corr], measure = 'SNR', label = 'max_marg_{}'.format(corr_label))
#   plot_matrix(std_corr_os_snr_matrix[corr], measure = 'SNR', label = 'std_{}'.format(corr_label))
#   plot_matrix(std_corr_os_marg_snr_matrix[corr], measure = 'SNR', label = 'std_marg_{}'.format(corr_label))
#   plot_matrix(skew_corr_os_snr_matrix[corr], measure = 'SNR', label = 'skew_{}'.format(corr_label))
#   plot_matrix(skew_corr_os_marg_snr_matrix[corr], measure = 'SNR', label = 'skew_marg_{}'.format(corr_label))
#   #plot_matrix(normal_corr_os_snr_matrix[corr], measure = 'SNR', label = 'Pnorm_{}'.format(corr_label))
#   #plot_matrix(normal_corr_os_marg_snr_matrix[corr], measure = 'SNR', label = 'Pnorm_{marg_}'.format(corr_label))
#   plot_matrix(N3sig_corr_os_matrix[corr], measure = 'N', label = 'N_3sig_{}'.format(corr_label), cb_label = r'$N(\\rho > 3)$', ax_title = '{}'.format(corr.replace('hd', 'HD').capitalize().replace('Hd', 'HD')))
#   plot_matrix(N3sig_corr_os_marg_matrix[corr], measure = 'N', label = 'N_3sig_marg_{}'.format(corr_label), cb_label = r'$N(\\rho > 3)$', ax_title = '{}, marginalised'.format(corr.replace('hd', 'HD').capitalize().replace('Hd', 'HD')))
#   plot_matrix(N2sig_corr_os_matrix[corr], measure = 'N', label = 'N_2sig_{}'.format(corr_label), cb_label = r'$N(\\rho > 2)$', ax_title = '{}'.format(corr.replace('hd', 'HD').capitalize().replace('Hd', 'HD')))
#   plot_matrix(N2sig_corr_os_marg_matrix[corr], measure = 'N', label = 'N_2sig_marg_{}'.format(corr_label), cb_label = r'$N(\\rho  > 2)$', ax_title = '{}, marginalised'.format(corr.replace('hd', 'HD').capitalize().replace('Hd', 'HD')))


