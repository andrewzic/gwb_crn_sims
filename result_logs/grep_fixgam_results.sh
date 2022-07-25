#!/bin/bash

for n in `seq 0 9`; do grep 'Samples in favor of'  *fixgamma*r${n}.result > r${n}_fixgam_results_incremental_fin.txt; done
