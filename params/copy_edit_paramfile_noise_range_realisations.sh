#!/bin/bash



for N in `ls ../psr_noise_vals/[0-9]*.dat`; 
do
    delta=`basename $N .dat`
    echo $delta ;
    for real in `seq 0 9`; 
    do 
	echo $r ;
	newpf=params_all_mc_array_spin_v_spincommon_${delta}_20210803_r${real}.dat
	cp params_all_mc_array_spin_v_spincommon_DP0_DALPHA_20210803_rN.dat $newpf;
	sed -i 's|DP0_DALPHA|'${delta}'|g' $newpf;
	sed -i 's|N|'${real}'|g' $newpf;
	#sed -i 's|zic006/ssb/ptasim/|zic006/pptadr2_gwb_crn_sims/|g' $newpf
    done
done

