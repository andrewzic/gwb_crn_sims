import os
import numpy as np
import glob
import matplotlib
#matplotlib.
import matplotlib.pyplot as plt
import gw_spectra_analytic as gwspec
import astropy.units as u
from ptasim2enterprise import P02A
import json
print(10**P02A(8.0E-23, 0.01, 5.5))
spec_files = sorted(glob.glob('J[012]*.spec'))


fig, (ax1) = plt.subplots(1,1,figsize=(6,4))#, xscale = 'log', yscale = 'log')

plt.sca(ax1)
#ax1.set_title('All pulsars with similar noise')
for specf in spec_files:
    print(specf)
    spec = np.loadtxt(specf)
    freq = spec[:, 0]/86400.0
    psd = spec[:, 1]*((1.0*u.year).to(u.s).value)**3.0
    plt.plot(freq, psd, c = '0.4', alpha = 0.4)#label = os.path.splitext(os.path.basename(specf))[0])

# plt.sca(ax2)
# ax2.set_title('J1713+0747 with weaker, shallower spin noise')
# spec_files = sorted(glob.glob('all-1_common_regsamp/timfiles/J[012]*.spec'))
# for specf in spec_files:
#     spec = np.loadtxt(specf)
#     freq = spec[:, 0]
#     psd = spec[:, 1]
#     plt.plot(freq, psd)#, label = os.path.splitext(os.path.basename(specf))[0])

    
ax1.legend()
for ax in [ax1]:#, ax2]:
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Frequency (s$^{{-1}}$)')
    ax.set_ylabel('PSD (s$^{{3}}$)')
    #ax.set_ylim(1E-33, 1E-24)
    ax.set_xlim(10**-9, 10**-7)#10**(int(np.log10(freq[0]))-1.0), 10**(int(np.log10(freq[-1]))+1.0))#/86400.0)#freq[-1])
    #ax.axvline(1.0/40.0/2.0, ls = '--', c = 'k', alpha = 0.5)

#    "gw_gamma": 4.655765635261815,
#    "gw_log10_A": -15.176018932965718,
#    "nmodel": 0.2599923902636294
    
#ax1.plot(freq, gwspec.gw_spec(freq*365.25, 10**-15.176018932965718, 4.655765635261815), c = 'k', linewidth = 2.0)
psrs = ["J0437-4715", "J1600-3053", "J1022+1001", "J1713+0747", "J1744-1134", "J1909-3744", "J2145-0750", "J2241-5236"]

psr_noise_dict = json.load(open('./noisefiles_3/_credlvl.json', 'r'))

rednoise_log10_As = []
rednoise_gammas = []
for key, item in psr_noise_dict.items():
    if "red_noise" in key:
        if "gamma" in key:
            rednoise_gammas.append(item['50'])
        if "log10_A" in key:
            rednoise_log10_As.append(item['50'])

gw_gamma = psr_noise_dict["gw_gamma"]['50']
gw_log10_A = psr_noise_dict["gw_log10_A"]['50']

# i_ = 0
# for psr, gamma, log10_A in zip(psrs, rednoise_gammas, rednoise_log10_As):
#     color_ = 'C{}'.format(i_)
#     rn_sp = gwspec.gw_spec(freq*365.25, 10.0**log10_A, gamma)#*((1.0*u.year).to(u.s).value)**3.0
#     ax1.plot(freq, rn_sp, linestyle = '--', label = '{} fit'.format(psr), color = color_, linewidth = 0.5)
#     i_+=1

freq = np.logspace(np.log10(freq[0]/10.0), np.log10(freq[-1]*10.0), 1000)
crn_sp = gwspec.gw_spec(freq*86400.0*365.25, 10.0**gw_log10_A, gw_gamma)*((1.0*u.year).to(u.s).value)**3.0
ax1.plot(freq, crn_sp, linestyle = '--', label = '$\log_{{10}} \widehat{{A}}_{{}} = {:.2f}$\n $\widehat{{\gamma}}_{{}} = {:.2f}$'.format(gw_log10_A, gw_gamma), color = 'C0', linewidth = 3)

    
ax1.legend()
ax1.set_xlim(10**-9, 10**-7)
ax1.set_ylim(10**-10, 10**-2)
#
#common noise params in SN + CRN model
#ax2.plot(freq, gwspec.gw_spec(freq*365.25, 10.0**-15.455484441085986, 5.194386924508992), color = 'k', linestyle = '--', linewidth = 2.0, label = 'CRN + SN model')


"""    
    "gw_gamma": 5.194386924508992,
    "gw_log10_A": -15.455484441085986,
"""

#ax2.plot(freq, gwspec.gw_spec(freq*365.25, 10.0**-15.176018932965718, 4.655765635261815), color = 'b', linestyle = '--', linewidth = 2.0, label = 'CRN-only model')

"""
"gw_gamma": 4.655765635261815,
    "gw_log10_A": -15.176018932965718,
"""
print(10**P02A(8.0E-23, 0.01, 5.5))
print(10**P02A(1.5E-28, 0.01, 1.5))


input_noise_pars = np.loadtxt('similar_params4_smp.dat')
input_gamms = input_noise_pars[:, 0]
input_as = input_noise_pars[:, 1]
input_wns = input_noise_pars[:, 2]
avg_gamma = -np.average(input_gamms, weights = 1.0/input_wns**2.0)
avg_as = np.average(input_as, weights = 1.0/input_wns**2.0)
avg_log10A = P02A(avg_as, 0.01, avg_gamma)

freq_ptasim = np.logspace(np.log10(spec[0,0]/10.0), np.log10(spec[-1,0]*10.0), 1000)*((1.0*u.year).to(u.day).value)#freq*86400.0*365.25
print(avg_gamma, avg_as)

print(avg_as, avg_gamma)
print(freq_ptasim[0])
inp_psds = [(gwspec.ptasim_spec(freq_ptasim, 0.01, a, g))*((1.0*u.year).to(u.s).value)**3.0 +w for a, g, w in zip(input_as, -input_gamms, input_wns)]
avg_inp_psd = np.average(inp_psds, weights = 1.0/input_wns**1.0, axis = 0)#*((1.0*u.year).to(u.s).value)**3.0

#avg_inp_psd = gwspec.ptasim_spec(freq_ptasim, 0.01, avg_as, avg_gamma)*((1.0*u.year).to(u.s).value)**3.0

ax1.plot(freq, avg_inp_psd, linestyle = '--', label = '$\overline{{P}}(f | A, \gamma) + \overline{{\sigma}}$', color = 'C1', linewidth = 1.5) #$\log_{{10}}\overline{{A}}_{{\mathrm{{SN,inj}}}} = {:.2f}$\n $\overline{{\gamma}}_{{\mathrm{{SN,inj}}}} = {:.2f}$'.format(avg_log10A, avg_gamma)



#ax2.plot(freq, gwspec.ptasim_spec(freq*365.25, 0.01, 1.5E-28, 1.5), ls = '-', linewidth = 3.0, c = 'C3', label = 'Simulated J1713+0747 PSD')

#ax2.plot(freq, gwspec.ptasim_spec(freq*365.25, 0.01, 8E-23, 5.5), ls = '-', linewidth = 4.0, c = 'k', label = 'Simulated CRN PSD')

#ax2.plot(freq, gwspec.gw_spec(freq*365.25, 10**P02A(8E-23, 0.01, 5.5), 5.5), ls = '--', linewidth = 4.0, c = 'k')
#ax1.plot(freq, gwspec.ptasim_spec(freq*365.25, 0.01, 8E-23, 5.5), ls = '-', linewidth = 4.0, c = 'k')
ax1.legend()

#ax2.legend()
fig.tight_layout()
plt.minorticks_on()
plt.savefig('cholspec_similar.png', dpi = 300, bbox_inches = 'tight', facecolor='white')
plt.savefig('cholspec_similar.pdf', dpi = 300, bbox_inches = 'tight', facecolor='white')
plt.show()
