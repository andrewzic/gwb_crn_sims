import numpy as np
import json
import matplotlib.pyplot as plt
import glob
import ptasim2enterprise as p2e
import cmasher as cmr

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
})

font = {'family' : 'serif',
        'size'   : 14}
tickfont =  {'family' : 'serif',
             'size'   : 12}

N_psr = 26

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
    print(alpha_lowers[x])
    log10A_lower = p2e.P02A(10**p0_lowers[x], 0.01, -1.0*np.array([alpha_lowers[x], alpha_uppers[x]]))
    log10A_upper = p2e.P02A(10**p0_uppers[x], 0.01, -1.0*np.array([alpha_lowers[x], alpha_uppers[x]]))

    dP0 = p0_uppers[x] - p0_lowers[x]
    dP0s.append(dP0)
    dlog10_A = np.max(log10A_upper) - np.min(log10A_lower)
    dlog10_As.append(dlog10_A)

dP0s = np.array(sorted(dP0s))
ddP0s = np.median(np.abs(np.diff(dP0s)))
dlog10_As = np.array(dlog10_As)

gw_log10_A_matrix =          np.nan*np.zeros((dP0s.shape[0],
                                      dalphas.shape[0],
                                      realisations.shape[0]))

gw_gamma_matrix = np.nan * np.zeros_like(gw_log10_A_matrix)

print(dP0s[4], dalphas[4])
dP0_inds = []
dalpha_inds = []
result_realisations = []
gw_log10_As_all = []
gw_gammas_all = []

for realisation_ind in realisations:
    noisefiles = sorted(glob.glob('pe_array_spincommon*_r{}/*/noisefiles/_noise.json'.format(realisation_ind)))
    result_dP0s = np.array([float(i.split('/')[0].split('_')[3]) for i in noisefiles])
    result_dalphas = np.array([float(i.split('/')[0].split('_')[4]) for i in noisefiles])
    #reals = np.array([float(i.split('/')[0].split('_')[-1].replace('r', '')) for i in noisefiles])
    
    #print(DP0s)
    # print(len(noisefiles))
    # for n in noisefiles:
    #     print(n)
    #     #noise0
    #     noise0 = json.load(open(n, 'r'))
    noise = [json.load(open(n, 'r')) for n in noisefiles]
    gw_log10_As = [n['gw_log10_A'] for n in noise]
    gw_log10_As_all.extend(gw_log10_As)
    gw_gammas = [n['gw_gamma'] for n in noise]
    gw_gammas_all.extend(gw_gammas)
    #print(gw_log10_As)

    _dP0_inds = [np.argmin(np.abs(dP0s - i)) for i in result_dP0s]
    dP0_inds.extend(_dP0_inds)
    _dalpha_inds = [np.argmin(np.abs(dalphas - i)) for i in result_dalphas]
    dalpha_inds.extend(_dalpha_inds)

    for gw_gamma, gw_log10_A, dP0_ind, dalpha_ind in zip(gw_gammas, gw_log10_As,_dP0_inds, _dalpha_inds):
        #print(gw_log10_A)
        if dP0_ind == 4 and dalpha_ind == 4:
            print('test', gw_log10_A)
        gw_log10_A_matrix[dP0_ind, dalpha_ind, realisation_ind] = gw_log10_A
        gw_gamma_matrix[dP0_ind, dalpha_ind, realisation_ind] = gw_gamma

mean_gw_log10_A_matrix = np.nanmean(gw_log10_A_matrix, axis = 2)
mean_gw_gamma_matrix = np.nanmean(gw_gamma_matrix, axis = 2)

gw_log10_A_95 = np.percentile(gw_log10_A_matrix, 95.0, axis = 2)
gw_gamma_95 = np.percentile(gw_gamma_matrix, 95.0, axis = 2)

def plot_matrix(matrix, norm = None, type = 'os', measure = 'A', label = 'cp', dalphas = dalphas, dP0s = dP0s, close = True, cb_label = None, ax_title = None, cmap = 'cmr.ocean', vmin = None):



  measure_fignames = {'A': r'log10_A', 'gamma': 'gamma', 'alpha': 'alpha', 'dA': r'dlog10_A', 'dgamma': 'dgamma'}
  measure_dict = {'A': r'$\log_{{10}} A$', 'gamma': r'$\gamma$', 'alpha': '$\alpha$', 'dA': r'$\log_{{10}} (A / A_{{m}})$', 'dgamma': r'$\gamma - \gamma_{{m}}$'}
  measure_str = measure_dict[measure]
  
  figname_pref = '{}_{}'.format(measure_fignames[measure], label)
  fig, ax = plt.subplots(1,1, figsize = (2.0*3.3, 1.6*3.3 ))
  im = plt.imshow((matrix), cmap = cmap, norm = norm, origin = 'lower', extent = [dalphas[0], dalphas[-1] + ddalphas, dP0s[0], dP0s[-1] + ddP0s], aspect = (8.0 + ddalphas)/(14.0 + ddP0s), interpolation = 'nearest')#, clim = [1E-6, 1E6]) #
  #im = plt.imshow(bf_matrix, cmap = 'inferno', norm = matplotlib.colors.LogNorm(), origin = 'lower', extent = [*dalphas[[0,-1]], *dlog10_As[[0,-1]]], aspect = 'auto', clim = [1E-6, 1E6], interpolation = 'none') #

  if cb_label == None:
      cb_label = measure_str.format(label.replace('marg_', ''))

  cb = plt.colorbar(im)#, ticks = LogLocator(subs=range(10)))
  cb.set_label(cb_label, fontsize = font['size'])
  cb.ax.minorticks_on()
  #minorticks = im.norm(np.arange(1E-6, 1E6, 1))
  #cb.ax.yaxis.set_ticks(minorticks, minor = True)
  ax.set_xlabel(r'$\Delta \gamma$', fontdict = font)
  ax.set_ylabel(r'$\Delta \log_{{10}}(P_0)$', fontdict = font)
  ax.tick_params(axis='y', labelsize = tickfont['size'])
  ax.tick_params(axis='x', labelsize = tickfont['size'])
  if ax_title != None:
      ax.set_title(ax_title)
  plt.minorticks_on()

  bf_matrix = np.load('../bf_matrix.npy', allow_pickle = True)
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

log10A_central = p2e.P02A(10**-23.0, 0.01, 4.0)
print(log10A_central)    
plot_matrix(mean_gw_log10_A_matrix, measure = 'A', label = 'mean')
plot_matrix(mean_gw_gamma_matrix, measure = 'gamma', label = 'mean')

plot_matrix(gw_log10_A_95, measure = 'A', label = '95')
plot_matrix(gw_gamma_95, measure = 'gamma', label = '95')

#dlog10_As[0] = 1
#dalphas[0] = 1
print(np.sum(np.isnan(mean_gw_log10_A_matrix)))
print(np.where(np.isnan(mean_gw_log10_A_matrix)))
#print(
plot_matrix((mean_gw_log10_A_matrix - log10A_central), measure = 'dA', label = 'mean', vmin = 0)
plot_matrix((mean_gw_gamma_matrix - 4.0), measure = 'dgamma', label = 'mean', cmap = 'bwr', vmin = 0)
#/(dalphas[None,:] * np.ones_like(mean_gw_log10_A_matrix))

# print(len(dP0_inds), len(dalpha_inds), len(dP0_inds[0]))#, len(result_realisations))

# for gw_log10_As, gw_gammas, dP0_ind, dalpha_ind, realisation_ind in zip(gw_log10_As_all, gw_gammas_all, dP0_inds, dalpha_inds, result_realisations):

#     print(dP0_ind, dalpha_ind, realisation_ind)
#     gw_log10_A_matrix[dP0_ind, dalpha_ind, realisation_ind] = 
