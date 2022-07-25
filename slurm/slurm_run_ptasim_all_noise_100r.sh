#!/bin/bash
#SBATCH --job-name=run_ptasim_all_noise_100r
#SBATCH --output=/flush5/zic006/gwb_crn_sims/slurm_logs/run_ptasim_all_noise_100r_%A_%a.log
#SBATCH --ntasks-per-node=10
#SBATCH --nodes=1
#SBATCH --time=0-05:00:00
#SBATCH --mem=16G
#SBATCH --tmp=8G
#SBATCH --array=0-121%4

module load singularity

singularity exec /home/zic006/psr_gwb.sif /flush5/zic006/gwb_crn_sims/run_ptasim_all_noise_100r_array.csh $SLURM_ARRAY_TASK_ID
