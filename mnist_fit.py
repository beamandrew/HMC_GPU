# -*- coding: utf-8 -*-
"""
@author: Andrew Beam
This file can be used to replicate the multinomial regression example using 
the MNIST data set.

Code to accompany the manuscript:
    Beam, A.L., Ghosh, S.J., Doyle, J. Fast Hamiltonian Monte Carlo Using GPU Computing.
"""

# Load the required libraries #
import numpy as np
import pycuda.driver as cuda
from pycuda import gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.cumath as cumath
import pycuda.curandom as curandom
import time as t

import scikits.cuda.linalg as linalg
linalg.init()

''' 
CPU version of the softmax function
Used only at end of simulation for prediction
'''
def softmax_cpu(w):
    dist = np.zeros(w.shape)
    for i in range(0,dist.shape[0]):
        dist[i] = np.exp(w[i])/(np.exp(w[i]).sum()) 
    return dist

########## BEGIN *GPU* CODE ##########
# Create a softmax kernel to be used for the GPU-version of softmax(XB)
gpu_kernel = SourceModule("""
// M and N are dimensions of input matrix (rows by columns)
__global__ void softmax(float *output, int M, int N)
{
	 #include <math.h>
      int row = blockIdx.y*blockDim.y + threadIdx.y;
      float sum = 0;      
      if(row < M) {// && col < N) {
          // This is done to ensure numerical stability
          float max = output[row*N];
          for(int i=0;i<N;i++){
             float val = output[row*N + i];
             if(val > max) {max = val;}
          }          
          for(int i=0;i<N;i++){
             sum += exp(output[row*N + i]-max);
          }
   	    for(int i=0;i<N;i++){
             output[row*N + i] = exp(output[row*N + i]-max)/sum;
	    }	           
        }                 
                  
}
""")

'''
*GPU VERSION*
Calculates the gradient of the log-likelihood with respect to each beta
'''
def grad_log_like_beta(softmax_vals,X,Y):
    diff = Y-softmax_vals
    return(linalg.dot(X,diff,transa='T'))

'''
*GPU VERSION*
Calculates the gradient of the log-likelihood with respect to each beta
'''
def grad_beta_prior(beta):
    gB_prior = -(2.0*beta)/(beta*beta + 1)
    return(gB_prior)

'''
*GPU VERSION*
Calculates the density value of the log-likelihood
'''
def multinomial_log_likelihood(softmax_vals,Y,one_n_trans,one_c):
    # add small amount to protect against log(0)
    small_val = 1e-9
    prod = Y*cumath.log(softmax_vals+small_val)
    prod = linalg.dot(one_n_trans,prod)
    prod = linalg.dot(prod,one_c)
    return(prod.get())

'''
*GPU VERSION*
Calculates the log prior density on beta
'''
def cauchy_prior_log_den(beta):
    log_beta_den_vals = -cumath.log(1 + beta*beta)    
    return(gpuarray.sum(log_beta_den_vals).get())

'''
*GPU VERSION*
Compute the softmax transformation
This function modifies the argument, i.e. XB. The softmax transformation will
be stored in the argument passed to the function.
'''
def softmax(XB):
        grid2 = (XB.shape[0]+32-1)/32
        M = np.int32(XB.shape[0])       
        N = np.int32(XB.shape[1])
        #Perform softmax using GPU      
        softmax_kernel(XB, M, N, block=(1,32,1),grid=( 1,grid2) )

'''
*GPU* VERSION
Generates one MCMC sample via HMC simulation
'''
def HMC_sample(X,Y,beta,grad_beta,one_n_trans,one_c,beta_k_mask,momentum,L,eps,T,verbose):
    
    # Fill exisitng GPU object to initialize # 
    rng.fill_normal(momentum) 
    momentum = momentum * beta_k_mask
    
    softmax_vals = linalg.dot(X,beta)
    softmax(softmax_vals)
    init_ll = multinomial_log_likelihood(softmax_vals,Y,one_n_trans,one_c)
    init_prior_val = cauchy_prior_log_den(beta)
    current_k = gpuarray.sum(momentum*momentum).get()/2.0
    
    # Posterior is log-like + log_prior 
    current_u = init_ll + init_prior_val    
        
    #Keep copy of initial parameter values in case proposal is rejected
    beta_old = beta.copy()    
        
    # Compute the intial gradient 
    grad_beta = grad_log_like_beta(softmax_vals,X,Y) + grad_beta_prior(beta)
    grad_beta = grad_beta*beta_k_mask
    
    # Take an initial half-step
    momentum += eps*grad_beta/2.0
    
    # Perform L-1 leapfrog steps
    for step in range(0,L):
        beta += eps*momentum
        #Update the gradient
        softmax_vals = linalg.dot(X,beta)
        softmax(softmax_vals)
        grad_beta = grad_log_like_beta(softmax_vals,X,Y) + grad_beta_prior(beta)
        grad_beta = grad_beta*beta_k_mask
        if step != L:
            momentum += eps*grad_beta
    
    # Take a final half-step
    momentum += eps*grad_beta/2.0
    softmax_vals = linalg.dot(X,beta)
    softmax(softmax_vals)   
    
    final_ll = multinomial_log_likelihood(softmax_vals,Y,one_n_trans,one_c)
    proposed_u = final_ll + cauchy_prior_log_den(beta)
    proposed_k = gpuarray.sum(momentum*momentum).get()/2.0
    diff = ((proposed_u-proposed_k) - (current_u-current_k))/T
    
    u = np.log(np.random.random(1)[0])
    alpha = np.min([0,diff])
    vals = list()
    if u < alpha:
        msg = 'Accept!'
        vals.append(beta)
        vals.append(1)
    else:
        msg = 'Reject!'
        vals.append(beta_old)
        vals.append(0)
    
    if verbose:
        print 'Current value of log-kernel: ' + str(current_u)
        print 'Proposed value of log-kernel: ' + str(proposed_u)      
        print 'Current momentum: ' + str(current_k)
        print 'Proposed momentum: ' + str(proposed_k)      
        print 'Total diff: ' + str(diff)
        print 'Current log-like: ' + str(init_ll)
        print 'Proposed log-like: ' + str(final_ll)
        print 'Comparing alpha of: ' + str(alpha) + ' to uniform of: ' + str(u)
        print msg
    
    return(vals)

########## END *GPU* CODE ##########

'''
This function performs the HMC simulation. It takes the X and Y matricies,
creates GPU objects, and runs the simulation under using the specified simulation
parameters. 

Returns: List object containing posterior samples for regression coefficients 
Note beta_post is returned as a CPU object to avoid eating up GPU memory which may be limited
'''
def HMC_simulation(X_cpu,Y_cpu,n_samples,n_burnin,L,eps_burnin,eps_final,T,anneal_rate,verbose):
        
    #Store X and Y as GPU arrays
    X = gpuarray.to_gpu(X_cpu.astype(np.float32).copy())
    Y = gpuarray.to_gpu(Y_cpu.astype(np.float32).copy())
        
    N = X.shape[0]
    p = X.shape[1]
    c = Y.shape[1]
    
    #Define a pycuda based random number generator
    rng = curandom.XORWOWRandomNumberGenerator()    
    
    #Define beta as a gpuarray and fill it with random normal data
    beta = rng.gen_normal((p,c),np.float32)/100.0
    grad_beta = gpuarray.zeros((p,c),np.float32)
        
    # Create a mask that keeps beta_k = 0 for identifiability purposes #
    beta_k_mask_cpu = np.ones(beta.shape).astype(np.float32)
    # Set last index to zero #
    beta_k_mask_cpu[:,-1] = 0    
    # Create a gpu-array for the mask #
    beta_k_mask = gpuarray.to_gpu(beta_k_mask_cpu)
    
    # Zero out beta_k #
    beta = beta*beta_k_mask
        
    #Define the momentum variables as a gpuarray
    momentum = gpuarray.empty_like(beta)*beta_k_mask
    
    one_n_trans = gpuarray.zeros((1,N),np.float32) + 1.0
    one_c = gpuarray.zeros((c,1),np.float32) + 1.0
    
    # This touches the code to ensure it is compiled before we start timing
    beta = HMC_sample(X,Y,beta,grad_beta,one_n_trans,one_c,beta_k_mask,momentum,L,eps_burnin,T,verbose=False)[0]
    t0 = t.time()
    total_accepts = 0.0
    for i in range(0,n_burnin):
        print '----------------------------'
        print 'Burnin Iteration: ' + str(i)
        beta, accept = HMC_sample(X,Y,beta,grad_beta,one_n_trans,one_c,beta_k_mask,momentum,L,eps_burnin,T,verbose=True)
        total_accepts += accept
        print 'T: ' + str(T)
        print 'Acceptance rate: ' + str(total_accepts/(i+1))
        print '----------------------------'
        T = 1.0 + T*anneal_rate
    
    total_accepts = 0.0
    T = 1.0
    beta_post = list()
    for i in range(0,n_samples):
        print '----------------------------'
        print 'Sampling Iteration: ' + str(i)
        beta, accept = HMC_sample(X,Y,beta,grad_beta,one_n_trans,one_c,beta_k_mask,momentum,L,eps_final,T,verbose=True)
        beta_post.append(beta.get())
        total_accepts += accept
        print 'Acceptance rate: ' + str(total_accepts/(i+1))
        print '----------------------------'
    
    t1 = t.time()
    print 'Time taken: ' + str(t1-t0)
    return beta_post


# This GPU kernel was created at the begining of the source file #
softmax_kernel = gpu_kernel.get_function("softmax")

#data_path = '/home/albeam/manuscripts/HMC_GPU/data/minist/'
data_path = '/home/andy/Documents/Research/BNN/data/'

rng = curandom.XORWOWRandomNumberGenerator()    

# Read in the data
X_cpu = np.loadtxt(data_path+'train_x.csv',delimiter=',',dtype=np.float32)
Y_cpu = np.loadtxt(data_path+'train_y.csv',delimiter=',',dtype=np.float32)

X_test_cpu = np.loadtxt(data_path+'test_x.csv',delimiter=',',dtype=np.float32)
Y_test_cpu = np.loadtxt(data_path+'test_y.csv',delimiter=',',dtype=np.float32)


# Add a column of ones for the intercept
X_cpu = np.hstack( (np.ones((len(X_cpu),1)),X_cpu) )
X_test_cpu = np.hstack( (np.ones((len(X_test_cpu),1)),X_test_cpu) )

# Sim parameters
n_burnin = 100 # Burnin iterations before sampling starts
n_samples = 100 # Number of posterior samples

# HMC parameters
L = 100
eps_burnin = 1e-4 # Step size during burning, typically larger than during sampling
eps_final = 1e-5
T = 1000.0
anneal_rate = 0.9
# Print progress? #
verbose = True

beta_posterior = HMC_simulation(X_cpu,Y_cpu,n_samples,n_burnin,L,eps_burnin,eps_final,T,anneal_rate,verbose)

## Compute training error using posterior mean ##
train_errors = 0.0
train_preds = np.zeros(Y_cpu.shape)
for i in range(0,len(beta_posterior)):
    XB = np.dot(X_cpu,beta_posterior[i])
    sm = softmax_cpu(XB)
    train_preds += sm

train_preds = train_preds/len(beta_posterior)

for i in range(0,len(train_preds)):
    train_errors += 1-Y_cpu[i,train_preds[i].argmax()]

print 'Train accuracy: ' + str(1-(train_errors/len(Y_cpu)))

# Compute test error using posterior mean 
test_errors = 0.0
test_preds = np.zeros(Y_test_cpu.shape)
for i in range(0,len(beta_posterior)):
    XB = np.dot(X_test_cpu,beta_posterior[i])
    sm = softmax_cpu(XB)
    test_preds += sm

test_preds = test_preds/len(beta_posterior)

for i in range(0,len(test_preds)):
    test_errors += 1-Y_test_cpu[i,test_preds[i].argmax()]

print 'Test accuracy: ' + str(1-(test_errors/len(Y_test_cpu)))


