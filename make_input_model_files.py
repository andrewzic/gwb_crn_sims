import os
import shutil
import numpy as np
import sys
import glob

noiseval_fname = sys.argv[1] #.dat file containing alpha, p0, and toa errbar for each psr
outdir = sys.argv[2]

template_str = """MODEL 1
ALPHA {:.2f}
FC 0.01
AMP {:.2e}"""


with open(noiseval_fname, 'r') as noise_f:
    with open("psrs.dat", 'r') as psr_f:
        lines = noise_f.readlines()
        psrs = psr_f.readlines()
        for line, psr in zip(lines, psrs):
            
            print(psr.strip('\n'))
            
            row = np.array(line.strip().split('\t'), dtype = np.float64)
            alpha, p0, dt = row
            out_fname = "{}/{}_input.model".format(outdir, psr.strip())
            if os.path.exists(out_fname):
                print('{} EXISTS'.format(psr))
                continue
            with open(out_fname, 'w') as out_f:
                out_f.write(template_str.format(-1.0*alpha, p0)) #negating alpha because it should be negative in dat file
                out_f.close()

            #if we specify where output model files should go
            if len(sys.argv) > 2:
                
                out_dirs = sorted(glob.glob(sys.argv[2]))
                
                for out_dir in out_dirs:
                    copy_fname = '{}/{}'.format(out_dir, os.path.basename(out_fname))
                    if copy_fname == out_fname:
                        continue
                    shutil.copyfile(out_fname, copy_fname)
                
                
