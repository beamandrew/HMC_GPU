# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 11:39:25 2013

@author: Andrew Beam
"""

########## BEGIN CPU CODE ##########
'''
CPU VERSION
Calculates the gradient of the log-likelihood with respect to each beta
'''
def grad_log_like_beta_cpu(softmax_vals,X,Y):
    diff = Y-softmax_vals
    return(np.dot(X.T,diff))

'''
CPU VERSION
Calculates the gradient of the log-prior density with respect to each beta
'''
def grad_beta_prior_cpu(beta):
    gB_prior = -(2.0*beta)/(beta*beta + 1)
    return(gB_prior) 

'''
CPU VERSION
Calculates the density value of the log-likelihood
'''
def multinomial_log_likelihood_cpu(softmax_vals,Y,one_n_trans,one_c):
    prod = Y*np.log(softmax_vals)
    prod = np.dot(one_n_trans,prod)
    prod = np.dot(prod,one_c)
    return(prod)

'''
CPU VERSION
Calculates the density value of the log-prior
'''
def cauchy_prior_log_den_cpu(beta):
    log_beta_den_vals = -np.log(1 + beta*beta)    
    return(np.sum(log_beta_den_vals))

'''
CPU VERSION
Compute the softmax transformation using the CPU
'''
def softmax_cpu(w):
    dist = np.zeros(w.shape)
    for i in range(0,dist.shape[0]):
        dist[i] = np.exp(w[i])/(np.exp(w[i]).sum()) 
    return dist

'''
CPU VERSION
Generates one MCMC sample via HMC simulation - Identical representation as GPU version
This version is missing the return statement that return the new (if accepted)
or old (if rejects) sample since it is only being used for timing purposes
'''
def HMC_sample_cpu(X,Y,grad_beta,beta,one_n_trans,one_c,momentum,L,eps):    
    #Initialize the momentum
    momentum = np.random.normal(size=momentum.shape)
    softmax_vals = softmax_cpu(np.dot(X,beta))
    init_ll = multinomial_log_likelihood_cpu(softmax_vals,Y,one_n_trans,one_c)
    init_prior_val = cauchy_prior_log_den_cpu(beta)
    current_k = np.sum(momentum*momentum)/2.0
    current_u = init_ll + init_prior_val
        
    #Compute the intial gradient
    grad_beta = grad_log_like_beta_cpu(softmax_vals,X,Y) + grad_beta_prior_cpu(beta)
    #take an initial half-step
    momentum += eps*grad_beta/2.0
    
    #Perform L-1 leapfrog steps
    for step in range(0,L):
        beta += eps*momentum
        softmax_vals = softmax_cpu(np.dot(X,beta))
        #Update the gradient
        grad_beta = grad_log_like_beta_cpu(softmax_vals,X,Y) + grad_beta_prior_cpu(beta)
        if step != L:
            momentum += eps*grad_beta
    
    #take a final half-step
    momentum += eps*grad_beta/2.0
    
    final_ll = multinomial_log_likelihood_cpu(softmax_vals,Y,one_n_trans,one_c)
    proposed_u =  final_ll + cauchy_prior_log_den_cpu(beta)
    proposed_k = np.sum(momentum*momentum)/2.0 
    diff = ((proposed_u-proposed_k) - (current_u-current_k)) 

########## END CPU CODE ##########

########## BEGIN *GPU* CODE ##########
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
    prod = Y*cumath.log(softmax_vals)
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
Generates one MCMC sample via HMC simulation - Identical representation as GPU version
This version is missing the return statement that return the new (if accepted)
or old (if rejects) sample since it is only being used for timing purposes
'''
def HMC_sample(X,Y,beta,beta_mask,grad_beta,one_n_trans,one_c,momentum,L,eps):
    
    # Fill exisitng GPU object to initialize # 
    rng.fill_normal(momentum) 
    momentum = momentum*beta_mask
    
    softmax_vals = linalg.dot(X,beta)
    softmax(softmax_vals)
    init_ll = multinomial_log_likelihood(softmax_vals,Y,one_n_trans,one_c)
    init_prior_val = cauchy_prior_log_den(beta)
    current_k = gpuarray.sum(momentum*momentum).get()/2.0
    # Posterior is log-like + log_prior 
    current_u = init_ll + init_prior_val    
    
    # Compute the intial gradient 
    grad_beta = grad_log_like_beta(softmax_vals,X,Y) + grad_beta_prior(beta)
    grad_beta = grad_beta * beta_mask
    # Take an initial half-step
    momentum += eps*grad_beta/2.0
    # Perform L-1 leapfrog steps
    for step in range(0,L):
        beta += eps*momentum
        #Update the gradient
        softmax_vals = linalg.dot(X,beta)
        softmax(softmax_vals)
        grad_beta = grad_log_like_beta(softmax_vals,X,Y) + grad_beta_prior(beta)
        grad_beta = grad_beta*beta_mask
        if step != L:
            momentum += eps*grad_beta
    
    # Take a final half-step
    momentum += eps*grad_beta/2.0
    softmax_vals = linalg.dot(X,beta)
    softmax(softmax_vals)   
    
    final_ll = multinomial_log_likelihood(softmax_vals,Y,one_n_trans,one_c)
    proposed_u = final_ll + cauchy_prior_log_den(beta)
    proposed_k = gpuarray.sum(momentum*momentum).get()/2.0
    diff = ((proposed_u-proposed_k) - (current_u-current_k))

########## END *GPU* CODE ##########

########## BEGIN Timing Demonstration Code ##########

import numpy as np
import pycuda.driver as cuda
from pycuda import gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel
import pycuda.cumath as cumath
import pycuda.curandom as curandom
import time as t

import scikits.cuda.linalg as linalg
import scikits.cuda.misc as cumisc
linalg.init()

# Load the softmax kernel to be used for the GPU-version of softmax(XB)
gpu_kernel = SourceModule("""
__global__ void softmax(float *output, int M, int N)
{
	#include <math.h>
      int row = blockIdx.y*blockDim.y + threadIdx.y;
      float sum = 0;     
      if(row < M) {// && col < N) {
          for(int i=0;i<N;i++){
             sum += exp(output[row*N + i]);
           }
        	for(int i=0;i<N;i++){
             output[row*N + i] = exp(output[row*N + i])/sum;
		}	           
        }                 
                  
}
""")
# Register the kernel as a callable python funtion
softmax_kernel = gpu_kernel.get_function("softmax")

rng = curandom.XORWOWRandomNumberGenerator()    

## Time gradients ##
reps = 5
N = ([100,1000,5000,10000])
p = ([10,50,100,500,1000,5000,10000,20000])
k = ([2,3,4,5,10,15,20])

#Arrays used to save timings
grad_times = np.zeros(shape=(len(N)*len(p)*len(k),6))
sample_times = np.zeros(shape=(len(N)*len(p)*len(k),6))

# HMC parameters
L = 1
eps = 1e-4

index = 0
# Begin timing 
# Loop over sample sizes
for i in range(0,len(N)):
    # Loop over predictor dimensionality
    for j in range(0,len(p)):
        # Loop overnumber of classes
        for l in range(0,len(k)):
            
            # Create some random data for demo based on N,p,k
            X = np.random.normal(scale=1,size=(N[i],p[j])).astype(np.float32)
            # Create a valid Y matrix using 1-hot coding
            Y = np.zeros((N[i],k[l])).astype(np.float32)
            for s in range(0,len(Y)):
                Y[s,np.random.randint(0,k[l]-1)] = 1 
            
            beta = np.random.normal(size=(p[j],k[l]),scale=0.001).astype(np.float32)
            
            # Create GPU objects for X,Y,beta and calculate the gradient of the log-likelihood
            X_gpu = gpuarray.to_gpu(X)
            Y_gpu = gpuarray.to_gpu(Y)
            beta_gpu = gpuarray.to_gpu(beta)
            
            # Create a mask to ensure identifiability #            
            beta_mask = np.ones(beta_gpu.shape).astype(np.float32)
            beta_mask[:,beta_mask.shape[1]-1] = 0
            beta_mask_gpu = gpuarray.to_gpu(beta_mask)
            
            # Compute the gradient log-likelihood and time on CPU
            mean_cpu_grad = 0.0
            
            # Update only identifiable components of beta #
            X_id = X[:,0:(p[j]-1)].copy()
            beta_id = beta[0:(p[j]-1)].copy()
            for x in range(0,reps):
                t0 = t.time()
                sm = softmax_cpu(np.dot(X_id,beta_id))
                g_cpu = grad_log_like_beta_cpu(sm,X_id,Y)
                t1 = t.time()
                mean_cpu_grad += t1-t0
            
            # Take average
            mean_cpu_grad = mean_cpu_grad/reps            
            
            # Now do the gradient timings with GPU
            # Touch the code once to make sure everything is compiled 
            sm_gpu = linalg.dot(X_gpu,beta_gpu)
            softmax(sm_gpu)
            g_gpu = grad_log_like_beta(sm_gpu,X_gpu,Y_gpu)
            g_gpu = g_gpu*beta_mask_gpu
            
            mean_gpu_grad = 0.0
            for x in range(0,reps):
                t0 = t.time()
                sm_gpu = linalg.dot(X_gpu,beta_gpu)
                softmax(sm_gpu)
                g_gpu = grad_log_like_beta(sm_gpu,X_gpu,Y_gpu)
                #g_gpu = g_gpu*beta_mask_gpu
                t1 = t.time()
                mean_gpu_grad += t1-t0
            
            mean_gpu_grad = mean_gpu_grad/reps
            
            # Compute HMC sample for L=1 leap-frog update and record time
            # Setup momement and ones vectors needed to do HMC
            momentum = np.zeros(beta.shape).astype(np.float32)
            one_n_trans = np.ones((1,N[i])).astype(np.float32)
            one_c = np.ones((k[l],1)).astype(np.float32)
            
            # Create and transfer corresponding GPU objects
            momentum_gpu = gpuarray.to_gpu(momentum)
            one_n_gpu = gpuarray.to_gpu(one_n_trans)
            one_c_gpu = gpuarray.to_gpu(one_c)
            
            # Set up identifiability momenmtum #
            momentum_id = momentum[0:(p[j]-1)].copy()          
            
            mean_cpu_sample = 0.0
            for x in range(0,reps):
                t0 = t.time()
                # Note we pass g_cpu to avoid any time penalty assocaited with array creation
                HMC_sample_cpu(X_id,Y,g_cpu,beta_id,one_n_trans,one_c,momentum_id,L,eps)
                t1 = t.time()
                mean_cpu_sample += t1-t0
            
            mean_cpu_sample = mean_cpu_sample/reps
            
            # Now tuime HMC_sample with GPU
            # Touch the code once to make sure everything is compiled 
            HMC_sample(X_gpu,Y_gpu,beta_gpu,beta_mask_gpu,g_gpu,one_n_gpu,one_c_gpu,momentum_gpu,L,eps)
            
            mean_gpu_sample = 0.0
            for x in range(0,reps):
                t0 = t.time()
                HMC_sample(X_gpu,Y_gpu,beta_gpu,beta_mask_gpu,g_gpu,one_n_gpu,one_c_gpu,momentum_gpu,L,eps)
                t1 = t.time()
                mean_gpu_sample += t1-t0           
            
            mean_gpu_sample = mean_gpu_sample/reps
            
            # Store the gradient and sample timing results
            grad_times[index] = [N[i],p[j],k[l],mean_cpu_grad,mean_gpu_grad,mean_cpu_grad/mean_gpu_grad]
            sample_times[index] = [N[i],p[j],k[l],mean_cpu_sample,mean_gpu_sample,mean_cpu_sample/mean_gpu_sample]
            index += 1
            #print 'Configuration ' + str(i) + ',' + str(j) + ',' + str(l) + ' timings: CPU - ' + str(mean_cpu) + ' GPU - ' + str(mean_gpu) + ' Speedup: ' + str(mean_cpu/mean_gpu)

np.savetxt('/home/albeam/manuscripts/HMC_GPU/timings/timings_grad.csv',grad_times,delimiter=',')
np.savetxt('/home/albeam/manuscripts/HMC_GPU/timings/timings_sample.csv',sample_times,delimiter=',')
