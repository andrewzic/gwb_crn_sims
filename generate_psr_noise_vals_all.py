import numpy as np
import matplotlib.pyplot as plt
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


N = np.arange(0, p0_lowers.shape[0])
NN, _ = np.meshgrid(N, N)

for y in N:

    alphas = np.random.uniform(alpha_lowers[y], alpha_uppers[y], size = N_psr)
    dalpha = alpha_uppers[y] - alpha_lowers[y]
    toaerrs = np.random.uniform(9e-8, 5e-7, size = N_psr)

    fmt_str_tnoise = "tnoise: psr={:s},alpha={:.3f},p0={:.3e},fc=0.01\n"
    fmt_str_obs = "observe: psr={:s},toaerr={:.3e},freq=1400\n"
    fmt_str_dat = "{:.4f}\t{:.4e}\t{:.4e}\n"

    for x in N:
        p0s = 10**(np.random.uniform(p0_lowers[x], p0_uppers[x], size = N_psr))
        log10As = p2e.P02A(p0s, 0.01, -alphas)
        log10A_lower = p2e.P02A(10**p0_lowers[x], 0.01, -1.0*np.array([alpha_lowers[y], alpha_uppers[y]]))
        log10A_upper = p2e.P02A(10**p0_uppers[x], 0.01, -1.0*np.array([alpha_lowers[y], alpha_uppers[y]]))
        dP0 = p0_uppers[x] - p0_lowers[x]

        str_tnoise = ''
        for psr, alpha, p0 in zip(psr_list, alphas, p0s):
            str_tnoise += fmt_str_tnoise.format(psr, alpha, p0)
            #print(fmt_str_tnoise.format(psr, alpha, p0))

        str_obs = ''
        for psr, toaerr in zip(psr_list, toaerrs):
            str_obs += fmt_str_obs.format(psr, toaerr)
            #print(fmt_str_obs.format(psr, toaerr))


        if True: #not os.path.exists('ptasim_input_files//{:.2f}_{:.2f}.inp'.format(dP0, dalpha)):
            with open('ptasim_input_files/all_mc_array_spin_v_spincommon/{:.2f}_{:.2f}.inp'.format(dP0, dalpha), 'w') as ptasim_inp_f:
                ptasim_inp_f.write(ptasim_inp_template_str.format('{:.2f}'.format(dP0), '{:.2f}'.format(dalpha), str_tnoise, str_obs))
                ptasim_inp_f.close()

            with open('psr_noise_vals/{:.2f}_{:.2f}.dat'.format(dP0, dalpha), 'w') as psr_datf:
                for alpha, p0, toaerr in zip(alphas, p0s, toaerrs):
                    psr_datf.write(fmt_str_dat.format(alpha, p0, toaerr))

                psr_datf.close()
