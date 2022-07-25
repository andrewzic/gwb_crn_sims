#!/bin/bash
#SBATCH --job-name=gwb_crn_sims_pe_spin_noise_range_allreal
#SBATCH --output=/flush5/zic006/gwb_crn_sims/slurm_logs/gwb_crn_sims_pe_spinnoise_range_20210803_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=0-06:00:00
#SBATCH --mem=32G
#SBATCH --tmp=8G
#SBATCH --array=605-1210%121

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

paramfiles=(/flush5/zic006/gwb_crn_sims/params/all_pe_array_spin/params_all_pe_array_spin_[0-9]*.dat)
echo "processing paramfile ${paramfiles[$SLURM_ARRAY_TASK_ID]}"


#doing the main run
singularity exec /home/zic006/psr_gwb.sif python3 /flush5/zic006/gwb_crn_sims/run_enterprise_simple.py --prfile ${paramfiles[$SLURM_ARRAY_TASK_ID]}

#doing the results run
singularity exec /home/zic006/psr_gwb.sif python3 -m enterprise_warp.results --result ${paramfiles[$SLURM_ARRAY_TASK_ID]} --info 1 -c 2 -p "gw" -f 1 -l 1 -m 1 
