#gwb_crn_sims

Simulated PTA datasets, processing, and analysis scripts for "Evaluating the prevalence of spurious correlations in pulsar timing array datasets" (Zic et al., 2022)

A singularity container is provided: `psr_gwb.sif`. Activate using `singularity shell psr_gwb.sif` or execute using `singularity exec psr_gwb.sif python3 <scriptname>.py`. This package has all software and python libraries required to run the analyses. Occasionally, slurm jobs may fail when astropy attempts to download updates to IERS parameters due to permissions issues on the job nodes. To fix this, enter the singularity shell interactively from a login node (`singularity shell psr_gwb.sif`), then issue `python3 -c "import enterprise_warp"`: this should result in the appropriate astropy tables being downloaded. 

PTASimulate input files are in `ptasim_input_files/`
scripts to run PTASimulate on generated input files: `run_ptasim_all_noise.csh` or `run_ptasim_all_noise_100r.csh `

Slurm batch scripts are located in `slurm/`

Simulated datasets will be accessible via the CSIRO DAP, and should be downloaded into the `data/` directory in this repo.

enterprise_warp params are located in `params/` (subdirectories are stored as tarballs and will need inflating to access individual files)

enterprise_warp noisemodel json files are in `noisemodels/`

Model comparison result summaries are located in `result_logs/` (for standard spin vs spin_common model comparison, and for experiments with the spin noise priors)
Optimal statistic result dictionaries and parameter estimation results are will be accessible in tarballs via the CSIRO DAP.