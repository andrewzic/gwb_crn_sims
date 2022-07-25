import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
import os
import ptasim2enterprise as p2e

import argparse

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

np.random.seed(seed = 20210524)

ptasim_inp_template_fn = 'ptasim_input_files/ptasim_all_similar_26_N_template.inp'
ptasim_inp_template_f = open(ptasim_inp_template_fn, 'r')
ptasim_inp_template_str = ptasim_inp_template_f.read()

#print(ptasim_inp_template.format('test', 'ts', '2ff'))

alpha_uppers = np.linspace(0, -4, 11)[::-1]
alpha_lowers = np.linspace(-8, -4, 11)[::-1]

p0_lowers = np.linspace(-30, -23, 11)[::-1]
p0_uppers = np.linspace(-16, -23, 11) [::-1]

Alpha_lowers, P0_lowers = np.meshgrid(alpha_lowers, p0_lowers)
Alpha_uppers, P0_uppers = np.meshgrid(alpha_uppers, p0_uppers)

bf_matrix = np.nan*np.zeros_like(Alpha_lowers)
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
    print(alpha_lowers[x])
    log10A_lower = p2e.P02A(10**p0_lowers[x], 0.01, -1.0*np.array([alpha_lowers[x], alpha_uppers[x]]))
    log10A_upper = p2e.P02A(10**p0_uppers[x], 0.01, -1.0*np.array([alpha_lowers[x], alpha_uppers[x]]))
    if x == 0:
        print('HFWBFKNJKFAD', log10A_lower, log10A_upper)
    dP0 = p0_uppers[x] - p0_lowers[x]
    dP0s.append(dP0)
    dlog10_A = np.max(log10A_upper) - np.min(log10A_lower)
    dlog10_As.append(dlog10_A)

dP0s = np.array(sorted(dP0s))
ddP0s = np.median(np.abs(np.diff(dP0s)))
dlog10_As = np.array(dlog10_As)

bf_matrix_list = []
for realisation_ind in range(0, 10):
  bf_matrix = np.nan*np.zeros_like(Alpha_lowers)
  #print(realisation_ind)
  incremental_result_file = 'result_logs/r{}_fixgam_results_incremental_fin.txt'.format(realisation_ind)
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

      #print(result)
      dP0 = float(l[7])
      dalpha = float(l[8])
      #print(dP0, dalpha)
      #print(_bf, result_dict['1'], result_dict['0'], result)

      dP0_ind = np.where(np.isclose(dP0s, dP0))
      dalpha_ind = np.where(np.isclose(dalphas , dalpha))
      bf_matrix[dP0_ind, dalpha_ind] = _bf
  bf_matrix_list.append(bf_matrix)

np.save('bf_matrix_all.npy', np.array(bf_matrix_list))
bf_matrix = 10.0**np.nanmean(np.log10(np.array(bf_matrix_list)), axis = 0)
#bf_matrix[-1,-1] = np.nan
#bf_matrix[-2,-1] = np.nan
#bf_matrix[-1,-2] = np.nan
#bf_matrix[-2,-2] = np.nan

print([i[-2,-2] for i in bf_matrix_list])
np.save('bf_matrix.npy', bf_matrix)

print(bf_matrix.shape)

import matplotlib.colors
from matplotlib.ticker import LogLocator

def plot_matrix(bf_matrix, dalphas, dP0s):

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
    })
    
    font = {'family' : 'serif',
            'size'   : 14}
    tickfont = {'family' : 'serif',
                'size'   : 12}

    fig, ax = plt.subplots(1,1, figsize = (2.0*3.3,1.6*3.3))
    #'coolwarm'

    im = plt.imshow(bf_matrix, cmap = "coolwarm", norm = matplotlib.colors.LogNorm(), origin = 'lower', extent = [dalphas[0], dalphas[-1] + ddalphas, dP0s[0], dP0s[-1] + ddP0s], aspect = (8.0 + ddalphas)/(14.0 + ddP0s), clim = [1E-6, 1E6], interpolation = 'nearest') #
    #im = plt.imshow(bf_matrix, cmap = 'coolwarm', norm = matplotlib.colors.LogNorm(), origin = 'lower', extent = [*dalphas[[0,-1]], *dlog10_As[[0,-1]]], aspect = 'auto', clim = [1E-6, 1E6], interpolation = 'none') #
    #ax.text(1.0, 3.0, 'CRN favoured', c = 'white', fontsize = font['size'])
    #ax.text(4.0, 10.0, 'CRN disfavoured', c = 'white', fontsize = font['size'])
    ax.scatter(ddalphas/2.0 + np.array([0.0, 1.6, 3.2, 4.0]), ddP0s/2.0 + np.array([0.0, 1.4, 2.8, 2.8]), marker = '*', c = 'white')
    cb = plt.colorbar(im)#, ticks = LogLocator(subs=range(10)))
    cb.set_label(r'$\mathcal{{B}}^{{\mathrm{{CP1}}}}_{{\mathrm{{TN}}}}$', fontsize = font['size'])
    cb.ax.minorticks_on()
    #minorticks = im.norm(np.arange(1E-6, 1E6, 1))
    #cb.ax.yaxis.set_ticks(minorticks, minor = True)
    ax.set_xlabel(r'$\Delta \gamma$', fontdict = font)
    ax.set_ylabel(r'$\Delta \log_{{10}}(P_0)$', fontdict = font)
    ax.tick_params(axis='y', labelsize = tickfont['size'])
    ax.tick_params(axis='x', labelsize = tickfont['size'])
    plt.minorticks_on()
    return fig


# plt.rcParams.update({
#   "text.usetex": True,
#   "font.family": "serif",
# })

# font = {'family' : 'serif',
#         'size'   : 14}
# tickfont = {'family' : 'serif',
#             'size'   : 12}

# fig, ax = plt.subplots(1,1, figsize = (2.0*3.3,1.6*3.3))
# #'coolwarm'
# im = plt.imshow(bf_matrix, cmap = "coolwarm", norm = matplotlib.colors.LogNorm(), origin = 'lower', extent = [dalphas[0], dalphas[-1] + ddalphas, dP0s[0], dP0s[-1] + ddP0s], aspect = (8.0 + ddalphas)/(14.0 + ddP0s), clim = [1E-6, 1E6], interpolation = 'nearest') #
# #im = plt.imshow(bf_matrix, cmap = 'coolwarm', norm = matplotlib.colors.LogNorm(), origin = 'lower', extent = [*dalphas[[0,-1]], *dlog10_As[[0,-1]]], aspect = 'auto', clim = [1E-6, 1E6], interpolation = 'none') #
# #ax.text(1.0, 3.0, 'CRN favoured', c = 'white', fontsize = font['size'])
# #ax.text(4.0, 10.0, 'CRN disfavoured', c = 'white', fontsize = font['size'])
# ax.scatter(ddalphas/2.0 + np.array([0.0, 1.6, 3.2, 4.0]), ddP0s/2.0 + np.array([0.0, 1.4, 2.8, 2.8]), marker = '*', c = 'white')
# cb = plt.colorbar(im)#, ticks = LogLocator(subs=range(10)))
# cb.set_label(r'$\mathcal{{B}}^{{\mathrm{{CP2}}}}_{{\mathrm{{TN}}}}$', fontsize = font['size'])
# cb.ax.minorticks_on()
# #minorticks = im.norm(np.arange(1E-6, 1E6, 1))
# #cb.ax.yaxis.set_ticks(minorticks, minor = True)
# ax.set_xlabel(r'$\Delta \gamma$', fontdict = font)
# ax.set_ylabel(r'$\Delta \log_{{10}}(P_0)$', fontdict = font)
# ax.tick_params(axis='y', labelsize = tickfont['size'])
# ax.tick_params(axis='x', labelsize = tickfont['size'])
# plt.minorticks_on()


fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize = (2.0*3.3,1.8*1.6*3.3), sharex = True, sharey  = True)
indices = np.where((np.log10(bf_matrix)) < -2)[0]
_, bins, _ = ax1.hist(np.log10(np.array(bf_matrix_list))[:, indices].flatten(), bins = 30, density = True, histtype = 'stepfilled')
indices = np.where(np.logical_and(np.log10(bf_matrix) > -2, np.log10(bf_matrix) < 2))[0]
ax2.hist(np.log10(np.array(bf_matrix_list))[:, indices].flatten(), bins = bins, density = True, histtype = 'stepfilled')
indices = np.where((np.log10(bf_matrix)) > 2)[0]
ax3.hist(np.log10(np.array(bf_matrix_list))[:, indices].flatten(), bins = bins, density = True, histtype = 'stepfilled')

ax1.text(0.1, 0.85, s = r'$\langle \mathcal{{B}}^{{\mathrm{{CP}}}}_{{\mathrm{{TN}}}} \rangle < 10^{{-2}}$', fontsize = font['size'], transform = ax1.transAxes)
ax2.text(0.1, 0.85, s = r'$10^{{-2}} < \langle \mathcal{{B}}^{{\mathrm{{CP}}}}_{{\mathrm{{TN}}}} \rangle < 10^{{2}}$', fontsize = font['size'], transform = ax2.transAxes)
ax3.text(0.1, 0.85, s = r'$\langle \mathcal{{B}}^{{\mathrm{{CP}}}}_{{\mathrm{{TN}}}} \rangle > 10^{{2}}$', fontsize = font['size'], transform = ax3.transAxes) 
x
#ax3.set_xlabel(r'$\mathcal{{B}}^{{\mathrm{{CP}}}}_{{\mathrm{{TN}}}}$', fontsize = font['size'])
#ax1.set_ylabel('Frequency', fontsize = font['size'])
for ax in [ax1, ax2, ax3]:
    ax.tick_params(axis='y', labelsize = tickfont['size'])
    ax.tick_params(axis='x', labelsize = tickfont['size'])
    ax.set_ylabel('Frequency', fontsize = font['size'])
    ax.set_yscale('log')
    ax.set_xlim(-6,6)
    if ax == ax3:
        ax.set_xlabel(r'$\log_{{10}} \mathcal{{B}}^{{\mathrm{{CP}}}}_{{\mathrm{{TN}}}}$', fontsize = font['size'])

fig.subplots_adjust(hspace=0.05)
        
plt.savefig('BF_distributions.pdf', bbox_inches = 'tight')
plt.savefig('BF_distributions.png', bbox_inches = 'tight', dpi = 300)
plt.close()

fig = plot_matrix(bf_matrix, dalphas, dP0s)
plt.savefig('fixgam_bf_matrix.pdf', bbox_inches = 'tight')
plt.savefig('fixgam_bf_matrix.png', bbox_inches = 'tight', dpi = 300)
# plt.show()
bf_matrix = 10.0**np.nanmax(np.log10(np.array(bf_matrix_list)), axis = 0)
fig = plot_matrix(bf_matrix, dalphas, dP0s)
plt.savefig('max_fixgam_bf_matrix.pdf', bbox_inches = 'tight')
plt.savefig('max_fixgam_bf_matrix.png', bbox_inches = 'tight', dpi = 300)

bf_matrix = np.nanstd(np.log10(np.array(bf_matrix_list)), axis = 0)
bf_matrix = np.nansum(np.array(bf_matrix_list) > 100, axis = 0)
print(bf_matrix[0,0])
#fig = plot_matrix(bf_matrix, dalphas, dP0s)
plt.close()
plt.close()
plt.imshow(bf_matrix, origin = 'lower', extent = [dalphas[0], dalphas[-1] + ddalphas, dP0s[0], dP0s[-1] + ddP0s], aspect = (8.0 + ddalphas)/(14.0 + ddP0s), cmap = 'inferno', interpolation = 'nearest')
plt.savefig('far_fixgam_bf_matrix.pdf', bbox_inches = 'tight')
plt.savefig('far_fixgam_bf_matrix.png', bbox_inches = 'tight', dpi = 300)
plt.colorbar()
plt.close()


         
