import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
import os
import ptasim2enterprise as p2e

import argparse

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
dlog10_As = np.array(dlog10_As)

bf_matrix_list = []
for realisation_ind in range(0, 10):
  bf_matrix = np.nan*np.zeros_like(Alpha_lowers)
  #print(realisation_ind)
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

      #print(result)
      dP0 = float(l[7])
      dalpha = float(l[8])
      #print(dP0, dalpha)
      #print(_bf, result_dict['1'], result_dict['0'], result)

      dP0_ind = np.where(np.isclose(dP0s, dP0))
      dalpha_ind = np.where(np.isclose(dalphas , dalpha))
      bf_matrix[dP0_ind, dalpha_ind] = _bf
  bf_matrix_list.append(bf_matrix)
bf_matrix = np.nanmean(np.array(bf_matrix_list), axis = 0)

print(bf_matrix.shape)

import matplotlib.colors
from matplotlib.ticker import LogLocator

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
})

font = {'family' : 'serif',
        'size'   : 17}

fig, ax = plt.subplots(1,1)
#'coolwarm'
np.save('bf_matrix_vg.npy', bf_matrix)
np.save('bf_matrix_vg_all.npy', np.array(bf_matrix_list))
im = plt.imshow(bf_matrix, cmap = "coolwarm", norm = matplotlib.colors.LogNorm(), origin = 'lower', extent = [*dalphas[[0,-1]], *dP0s[[0,-1]]], aspect = 'auto', clim = [1E-6, 1E6], interpolation = 'nearest') #
#im = plt.imshow(bf_matrix, cmap = 'coolwarm', norm = matplotlib.colors.LogNorm(), origin = 'lower', extent = [*dalphas[[0,-1]], *dlog10_As[[0,-1]]], aspect = 'auto', clim = [1E-6, 1E6], interpolation = 'none') #

cb = plt.colorbar(im)#, ticks = LogLocator(subs=range(10)))
cb.set_label(r'$\mathcal{{B}}^{{\mathrm{{CP1}}}}_{{\mathrm{{TN}}}}$', fontsize = font['size'])
cb.ax.minorticks_on()
#minorticks = im.norm(np.arange(1E-6, 1E6, 1))
#cb.ax.yaxis.set_ticks(minorticks, minor = True)
ax.set_xlabel(r'$\Delta \alpha$', fontdict = font)
ax.set_ylabel(r'$\Delta \log_{{10}}(P_0)$', fontdict = font)
ax.tick_params(axis='y', labelsize = font['size'])
ax.tick_params(axis='x', labelsize = font['size'])
plt.minorticks_on()
plt.savefig('bf_matrix.pdf', bbox_inches = 'tight')
plt.savefig('bf_matrix.png', bbox_inches = 'tight', dpi = 300)
# plt.show()
