HMC_GPU
=======

This code can be used to replicate the results found in:

Beam, A.L., Ghosh, S.J., Doyle, J. Fast Hamiltonian Monte Carlo Using GPU Computing. 

To get started you will need to make sure you have the following depedencies installed:

GPU Computing Environment:
- Cuda SDK 5.0: https://developer.nvidia.com/cuda-toolkit

Python Environment
- Python 2.7: http://www.python.org/
- Numpy - http://www.numpy.org/
- SciPy - http://www.scipy.org/
- PyCuda - http://mathema.tician.de/software/pycuda/
- CUDA SciKit: http://scikits.appspot.com/cuda

Additionally, if you would like to run the glmnet comparison, you will need to install the following:
- R: http://cran.us.r-project.org/
- glmnet: http://cran.r-project.org/web/packages/glmnet/index.html
- doMC (for parallel processing in R - Linux): http://cran.r-project.org/web/packages/doMC/index.html
- doSNOW (for parallel processing in R - Windows): http://cran.r-project.org/web/packages/doSNOW/index.html

Each script is a self-contained file that corresponds to one set of results in the manuscript.
