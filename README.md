#gwb_crn_sims

Simulated PTA datasets, processing, and analysis scripts for "Evaluating the prevalence of spurious correlations in pulsar timing array datasets" (Zic et al., 2022)

A singularity container is provided: `psr_gwb.sif`. Activate using `singularity shell psr_gwb.sif` or execute using `singularity exec psr_gwb.sif python3 <scriptname>.py`. This package has all software and python libraries required to run the analyses. Occasionally, slurm jobs may fail when astropy attempts to download updates to IERS parameters. To fix this, enter the singularity shell interactively from a login node (`singularity shell psr_gwb.sif`), then issue `python3 -c "import enterprise_warp"`: this should result in the appropriate astropy being downloaded. 

PTASimulate input files are in `ptasim_input_files/`
scripts to run PTASimulate on generated input files: `run_ptasim_all_noise.csh` or `run_ptasim_all_noise_100r.csh `

Slurm batch scripts are located in `slurm/`

Simulated datasets are in `data/`
Note: `regsamp.tar.gz` and `100r.tar.gz` should be (recursively) inflated to access all realisations for each timing noise setup. Warning: these contain A LOT of files.

enterprise_warp params are located in `params/` (subdirectories are stored as tarballs and will need inflating to access individual files)

enterprise_warp noisemodel json files are in `noisemodels/`

Model comparison result summaries are located in `result_logs/` (for standard spin vs spin_common model comparison, and for experiments with the spin noise priors)
Optimal statistic result dictionaries are stored as pickles in `enterprise_out/100r/spincommonfixgamma.tar.gz`. Parameter estimation results are in `enterprise_out/spincommon.tar.gz`.