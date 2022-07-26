INFORMATION ON DATASET

The data are stored in tarballs (100r.tar.gz and regsamp.tar.gz). The tarball "100r.tar.gz" contains simulated datasets with 100 realisations per timing noise setup, and will inflate into a subdirectory "100r/". These data were primarily used for parameter estimation and optimal statistic analyses. The tarball "regsamp.tar.gz" contains simulated datasets with only 10 realisations per timing noise setup, which was used for model selection purposes (with 10x as many posterior samples as was used for the parameter estimation runs in the 100-realisation dataset). This tarball will inflate into the top level of the data/ directory.

The enterprise_warp parameter files within the params/ directory are named in a similar fashion (with the DP0, DALPHA, and realisation number indicated in that order) - each parameter file corresponds to a single timing noise data directory and a certain enterprise job (e.g. model comparison for spin noise vs spin + common noise, parameter estimation for spin noise + common noise with fixed spectral index, etc.).



