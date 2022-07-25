#!/bin/bash
#SBATCH --job-name=sn_pe
#SBATCH --output=/flush5/zic006/gwb_crn_sims/slurm_logs/pe_spincommonfixgamma__20210803_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time=0-03:00:00
#SBATCH --mem=4G
#SBATCH --array=0-4


module load singularity

singularity exec /home/zic006/psr_gwb.sif which python3
singularity exec /home/zic006/psr_gwb.sif echo $TEMPO2
singularity exec /home/zic006/psr_gwb.sif echo $TEMPO2_CLOCK_DIR

echo $SLURM_ARRAY_TASK_ID

paramfiles=( /flush5/zic006/gwb_crn_sims/params/all_pe_array_spincommon/params_all_pe_array_spincommon_0.00_0.00_20210803_r0.dat /flush5/zic006/gwb_crn_sims/params/all_pe_array_spincommon/params_all_pe_array_spincommon_1.40_1.60_20210803_r0.dat /flush5/zic006/gwb_crn_sims/params/all_pe_array_spincommon/params_all_pe_array_spincommon_2.80_3.20_20210803_r0.dat /flush5/zic006/gwb_crn_sims/params/all_pe_array_spincommon/params_all_pe_array_spincommon_2.80_4.00_20210803_r0.dat )
#($(ls -v /flush5/zic006/gwb_crn_sims/params/all_pe_array_spincommon/params_all_pe_array_spincommon_*_r[0-9]*.dat))
IFS=$'\n' paramfiles=($(sort -k1.127r <<<"${paramfiles[*]}")); unset IFS
if (( ${SLURM_ARRAY_TASK_ID} > ${#paramfiles[@]} ))
then
    echo "${SLURM_ARRAY_TASK_ID} exceeds number of paramfiles ${#paramfiles[@]}"
    exit 2
fi

PRFILE=${paramfiles[$SLURM_ARRAY_TASK_ID]}


if [ -z "${PRFILE}" ]
then
    echo "paramfile not found"
    exit 2
fi

echo "processing paramfile ${PRFILE}"
if [ ! -f ${PRFILE} ]
then
   echo "${PRFILE} does not exist. exiting"
   exit 2
fi

outdir=`grep "out: " ${PRFILE} | awk '{print $2}'`
prfile_label=`grep "paramfile_label: " ${PRFILE} | awk '{print $2}'`
noisemodel_file=`grep "noise_model_file: " ${PRFILE} | awk '{print $2}'`
noisemodel_name=$( echo `grep "\"model_name\": " ${noisemodel_file} | awk '{print $2}'` | sed -r 's/"//g' | sed -r 's/,//g' )
dirname=${outdir}/${noisemodel_name}_${prfile_label}

if [ -z "${noisemodel_name}" ];
then
    echo "invalid noise model name. exiting... "
    exit 2
fi

if [ -z "${prfile_label}" ];
then
    echo "invalid parmafile_label. exiting... "
    exit 2
fi

chain_file=$( echo $(find ${dirname} -name "chain*.txt" ) )
os_pickle_file=$( echo $(find ${dirname} -name "_os_result*.pkl" ) )

chain_minlines=20000

if [ -z "$chain_file" ];
then
    chain_file="~/NULL.NULL"
    nlines_chain=0
else
    nlines_chain=$( cat ${chain_file} | wc -l )
fi

if [ -z "$os_pickle_file" ];
then
    os_pickle_file="~/NULL.NULL"
fi

#doing the main run
#if [ ! -f "${chain_file}"  ] && [ ! -f "${os_pickle_file}" ]

#IF ( THE OPTIMAL STATISTIC RESULTS AREN't THERE ) AND ( EITHER THE CHAIN ISN'T THERE OR THE CHAIN IS NOT BIG ENOUGH )
if [ ! -f "${os_pickle_file}" ] && [ ! -f ${outdir}/OS_SUCCESS ] && [ ! -f ${outdir}/CHAIN_SUCCESS ] && [[ ! -f "${chain_file}"  || ${nlines_chain} < "${chain_minlines}" ]]
then
    
    singularity exec /home/zic006/psr_gwb.sif python3 /flush5/zic006/gwb_crn_sims/run_enterprise_simple.py --prfile ${PRFILE}
    
    newchain=$( echo $(find ${outdir} -name "chain*.txt" ) )
    if [ -z "${newchain}" ]
    then
	nlines_newchain=0
    else
	nlines_newchain=$( cat ${newchain} | wc -l )
    fi
    
    #NOW CHECK IF CHAIN GOT STUCK AND RE-RUN UNTIL IT WORKS (UP TO 5 ITERS)
    niter=0
    while (( ${nlines_newchain}<${chain_minlines} )) && (( ${niter}<5 ))
    do
    #if (( ${nlines_newchain} < ${chain_minlines} ))
	#then
	echo "CHAIN FAILED. RE-SAMPLING"
	bad_prfile=$(echo ${PRFILE} | sed -r 's|params/all_pe_array_spincommonfixgamma/all|params/all_pe_array_spincommonfixgamma/bad|g')
	cp ${PRFILE} ${bad_prfile}
	
	singularity exec /home/zic006/psr_gwb.sif python3 /flush5/zic006/gwb_crn_sims/run_enterprise_simple.py --prfile ${bad_prfile}
	
	nlines_newchain=$( cat ${newchain} | wc -l )
	if (( ${nlines_newchain} > ${chain_minlines} ));
	then
	    echo "CHAIN SAMPLED SUCCESSFULLY"
	    touch ${outdir}/CHAIN_SUCCESS
	    #cp ${dirname}/chain_1.txt
	    #rm ${bad_prfile} 
	fi
	niter=$(( ${niter} + 1 ))
    done
    #IF THE CHAIN STILL FAILED AFTER 5 TRIES THEN EXIT SCRIPT
    if (( ${nlines_newchain}<${chain_minlines} ))
    then
	echo "TOO MANY FAILED ATTEMPTS TO SAMPLE. EXITING"
	exit 2
	
    #IF THE CHAIN SUCCEEDED I.E. THE CHAIN HAS ENOUGH SAMPLES AND EXISTS
    elif (( ${nlines_newchain} > ${chain_minlines} )) && [ -f ${newchain} ]
    then
	echo "CHAIN SAMPLED SUCCESSFULLY"
	touch ${outdir}/CHAIN_SUCCESS

	
	#doing the results run
	echo "COMPUTING RESULTS SUMMARIES"
	singularity exec /home/zic006/psr_gwb.sif python3 -m enterprise_warp.results --result ${PRFILE} --info 1 -c 2 -p "gw" -f 1 -l 1 -m 1 > /flush5/zic006/gwb_crn_sims/result_logs/`basename ${PRFILE} .dat`.result
	
	#Calculating optimal statistic
	echo "COMPUTING OPTIMAL STATISTICS"
	singularity exec /home/zic006/psr_gwb.sif python3 -m enterprise_warp.results --result ${PRFILE} -o 1 -g "hd,dipole,monopole" -N 2000 -I 0
	if [ -f ${dirname}/_os_results.pkl ]
	then
	    touch ${outdir}/OS_SUCCESS
	    echo "OPTIMAL STATISTIC SUCCESSFULLY CALCULATED"
	fi
    fi
    
fi

#IF THE CHAIN EXISTS BUT THE OPTIMAL STATISTIC RESULTS DON'T
nlines_chain=$( cat ${dirname}/chain_1.txt | wc -l )
if [ -f "${dirname}/chain_1.txt"  ] && [ -f "${outdir}/CHAIN_SUCCESS" ] && [ ! -f "${dirname}/_os_results.pkl" ] && (( ${nlines_chain} > ${chain_minlines} ))
then
    echo "CHAIN EXISTS BUT NOT OPTIMAL STATISTIC. CALCULATING..."
    #calculating optimal statistic
    singularity exec /home/zic006/psr_gwb.sif python3 -m enterprise_warp.results --result ${PRFILE} -o 1 -g "hd,dipole,monopole" -N 2000 -I 0
    if [ -f ${dirname}/_os_results.pkl ]
    then
	touch ${outdir}/OS_SUCCESS
    fi
fi

#IF THE RESULT NOISEFILE DOESN'T EXIST OR IF THE CHAIN IS MORE RECENT THAN THE NOISEFILE
if [ ! -f ${dirname}/noisefiles/_noise.json ] || [[ ${dirname}/chain_1.txt  -nt ${dirname}/noisefiles/_noise.json ]]
then
    if [ -f ${dirname}/chain_1.txt ]
    then
	echo "FOUND THAT CHAIN IS NEWER THAN RESULT NOISEFILES, OR NOISEFILES DON'T EXIST. CALCULATING RESULT SUMMARIES"
	singularity exec /home/zic006/psr_gwb.sif python3 -m enterprise_warp.results --result ${PRFILE} --info 1 -c 2 -p "gw" -f 1 -l 1 -m 1 > /flush5/zic006/gwb_crn_sims/result_logs/`basename ${PRFILE} .dat`.result
    fi
fi



#/flush5/zic006/gwb_crn_sims/tidy_result.sh $( realpath ${outdir} )

#CLEAN UP
if [ -f ${outdir}/OS_SUCCESS ] && [ -f ${dirname}/_os_results.pkl ]
then

    if [ ! -z ${dirname} ] && [ ! -z ${prfile_label} ] && [ ! -z ${noisemodel_name} ]
    then
	echo "RUN SUCCESSFUL. CLEANING UP CHAINS AND OTHER INTERMEDIATE FILES"
	rm ${dirname}/cov*.txt
	rm ${dirname}/draw*.txt
	tar -cvf ${dirname}/corner_plots.tar.gz ${dirname}/*corner.png
	rm ${dirname}/*corner.png
	rm -rf ${dirname}/chain*.txt
	if [ -f ${PRFILE} ];
	then
	    cp ${PRFILE} $(echo ${PRFILE} | sed -r 's|params/all_pe_array_spincommonfixgamma/all|params/all_pe_array_spincommonfixgamma/done|g')
	fi
    fi
fi


###SBATCH --tmp=8G
#4235-8468%121 
 #8470-16939%121
# pyv="$(python -c 'import sys; print(sys.version_info[0])')"
# if [ "$pyv" == 2 ]
# then
#     echo "$pyv"
#     module load numpy/1.16.3-python-2.7.14
# fi


# chain_file=$( echo $(find ${outdir} -name "chain*.txt" ) )
    # chain_dir=$( echo $( dirname ${chain_file} ) )

    # DP0=$( echo ${chain_dir} | awk -F['/'] '{print $7}' | awk -F['_'] '{print $4}' )
    # DALPHA=$( echo ${chain_dir} | awk -F['/'] '{print $7}' | awk -F['_'] '{print $5}' )
    # REALIS=$( echo ${chain_dir} | awk -F['/'] '{print $7}' | awk -F['_'] '{print $7}' | awk -F['r'] '{print $2}' )
    #mv ${outdir}/__corner.png ${outdir}/_gw_corner.png
#grep -Fxq "Run complete with 11000 effective samples" /flush5/zic006/gwb_crn_sims/slurm_logs/gwb_crn_sims_pe_spincommonfixgamma_noise_range_20210803_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log;

	# for psr in `cat /flush5/zic006/gwb_crn_sims/psrs.dat`; do 
	#     echo $psr;
	#     singularity exec /home/zic006/psr_gwb.sif python3 -m enterprise_warp.results --result ${PRFILE} -c 2 -p $psr

	# done
