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

log10A_central = p2e.P02A(10**-23.0, 0.01, 4.0)
print(log10A_central)
