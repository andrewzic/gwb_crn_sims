#!/bin/bash
#SBATCH --job-name=gwb_crn_sims_mc_spin_v_spincommon_noise_range_allreal
#SBATCH --output=/flush5/zic006/gwb_crn_sims/slurm_logs/gwb_crn_sims_mc_spin_v_spincommon_noise_range_20210803_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=4G
#SBATCH --array=0-268


###SBATCH --tmp=8G
# pyv="$(python -c 'import sys; print(sys.version_info[0])')"
# if [ "$pyv" == 2 ]
# then
#     echo "$pyv"
#     module load numpy/1.16.3-python-2.7.14
# fi


module load singularity

singularity exec /home/zic006/psr_gwb.sif which python3
singularity exec /home/zic006/psr_gwb.sif echo $TEMPO2
singularity exec /home/zic006/psr_gwb.sif echo $TEMPO2_CLOCK_DIR

echo $SLURM_ARRAY_TASK_ID

paramfiles=( $( cat /flush5/zic006/gwb_crn_sims/params/all_mc_array_spin_v_spincommon/unfinished_params.dat ) )
IFS=$'\n' paramfiles=($(sort -k1.127r <<<"${paramfiles[*]}")); unset IFS
echo "processing paramfile ${paramfiles[$SLURM_ARRAY_TASK_ID]}"

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
noisemodel_name=$( echo `grep -h "\"model_name\": " ${noisemodel_file} | awk '{print $2}'` | sed -r 's/"//g' | sed -r 's/,//g' | sed -r 's/ /_/g' )
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

chain_minlines=50000

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
if [[ ! -f "${chain_file}"  || ${nlines_chain} < "${chain_minlines}" ]]
then

    
    #doing the main run
    singularity exec /home/zic006/psr_gwb.sif python3 /flush5/zic006/gwb_crn_sims/run_enterprise_simple.py --prfile ${paramfiles[$SLURM_ARRAY_TASK_ID]}

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
	#bad_prfile=$(echo ${PRFILE} | sed -r 's|params/all_pe_array_spincommonfixgamma/all|params/all_pe_array_spincommonfixgamma/bad|g')
	#cp ${PRFILE} ${bad_prfile}
	
	singularity exec /home/zic006/psr_gwb.sif python3 /flush5/zic006/gwb_crn_sims/run_enterprise_simple.py --prfile ${PRFILE}
	
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
	singularity exec /home/zic006/psr_gwb.sif python3 -m enterprise_warp.results --result ${PRFILE}  --info 1 -c 2 -p "gw" -f 1 -l 1 -m 1 -b 1 > /flush5/zic006/gwb_crn_sims/result_logs/`basename ${PRFILE} .dat`.result


	#Calculating optimal statistic
	echo "COMPUTING OPTIMAL STATISTICS"
	singularity exec /home/zic006/psr_gwb.sif python3 -m enterprise_warp.results --result ${PRFILE} -o 1 -g "hd,dipole,monopole" -N 2000 -I 1
	if [ -f ${dirname}/_os_results.pkl ]
	then
	    touch ${outdir}/OS_SUCCESS
	    echo "OPTIMAL STATISTIC SUCCESSFULLY CALCULATED"
	fi
    fi
    
fi


nlines_chain=$( cat ${dirname}/chain_1.txt | wc -l )
if [ -f "${dirname}/chain_1.txt"  ] && [ ! -f "${dirname}/_os_results.pkl" ] && (( ${nlines_chain} > ${chain_minlines} ))
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


	


