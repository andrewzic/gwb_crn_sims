#!/bin/bash

for n in `seq 0 9`; do grep 'Samples in favor of'  *r${n}.result > r${n}_results_incremental_fin.txt; done
