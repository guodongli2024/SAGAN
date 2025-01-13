# SAGAN-Modified

This is a modified version of the [SAGAN](https://github.com/jyshangguan/SAGAN/tree/main) Python package.  
It includes additional functionality for **host galaxy decomposition** and the ability to use MCMC fitting.

---

## New Features
### 1. Host Galaxy Decomposition
This version includes a simple and feasible method to decompose the host galaxy's components.  
The decomposition functionality allows users to:
- Isolate the stellar continuum from the AGN, using linear combinations of stellar spectra.
- Please refer to the demo jupyter notebooks in the example folder to use the code.
### 2. MCMC Fitting for Parameter Estimation
In addition to the original least-squares fitting, this version adds MCMC fitting to allow for robust posterior sampling and uncertainty estimation for model parameters.
MCMC fitting is implemented using the emcee package and can be enabled without changing the original usage workflow.
