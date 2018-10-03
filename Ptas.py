#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sept  1 14:05:20 2018

@author: aaouad
"""

from Numericals import Numerical
import numpy as np
#import math
from numba import jitclass,jit,njit,prange         # import the decorator
from numba.extending import overload
from numba import int64, float64, boolean    # import the types
#import numba as nb


spec = [
    ('n', int64), 
    ('n_samples', int64),              
    ('capacity', int64),          
    ('max_M', int64),          
    ('eps', float64),          
    ('weight_scale', float64),    
    ('weight_level', float64),          
    ('price_param', float64),      
    ('rate_M', float64),
    ('pdf_M', float64[:]),
    ('rates_M', float64[:]),
    ('prices', float64[:]),
    ('weights', float64[:]),
    ('avg', float64),
    ('std', float64)          
    
]

@jit(parallel = True, nopython=True)
def copyto(A,B):
    '''
    Numba transcription of numpy function
    '''
    for i in prange(A.shape[0]):
        A[i] =B[i]
    return()
    
@jit(parallel = True, nopython=True)
def tile(A,n):
    '''
    Numba transcription of numpy function
    '''
    B = np.zeros((A.shape[0],A.shape[1]*n))
    for i in prange(n):
        B[:,i*A.shape[1]:(i+1)*A.shape[1]] = A
    return(B)
    
    
@jit(parallel = True, nopython=True)
def repeat(A,n):
    '''
    Numba transcription of numpy function
    '''
    B = np.zeros((A.shape[0],A.shape[1]*n))
    for i in prange(A.shape[1]):
        for j in prange(A.shape[0]):
            B[j,i*n:(i+1)*n] = A[j,i]
    return(B)
    
    
@jit(parallel = True, nopython=True)
def cartesian_prod(A,B):
    '''
    A is Q by n
    B is Q by m
    
    Returns the cartesian product matrix that is 2Q by nm    
    '''
    aleph = tile(A,B.shape[1])
    bae = repeat(B,A.shape[1])    
    concatenated = np.concatenate((aleph,bae),axis = 0)
    return concatenated


@jit("float64[:, :](int64, int64)",parallel = True, nopython=True)
def test_rec(L,ub):
    '''
    L is the recursion horizon
    ub is the integral upper bound on the values tested at each step
    
    Compiler test recursive function (auxiliary)
    '''
    if L > 1:        
        R = np.zeros((L,1), dtype = float64)
        for i in prange(ub+1):
            A = test_rec(L-1,i)            
            B = np.zeros((L,A.shape[1]), dtype = float64)
            S = np.concatenate((R,B),axis = 1)
        return S[:,1:]
    else:
        V = np.arange(ub+1).astype(float64)        
        Q = V.reshape((1,V.shape[0]))
        return Q


@overload(np.unique)
def np_unique(a):
        def np_unique_impl(a):
            b = np.sort(a.flatten())
            unique = b[:1]
            for x in b[1:]:
                if x != unique[-1]:
                    unique = np.concatenate((unique,np.array([x]))) 
            return unique
        return np_unique_impl


@jit("float64[:, :](int64, int64,int64)",parallel = True, nopython=True)
def vector_enumeration(L,T1,T2):
    '''
    L is # of classes
    T1 is the integral sum value lower bound
    T2 is the integral sum value upper bound    
    
    Returns L by m matrix where m are the number of vector combinations
    '''
    if L > 1:        
        S = np.zeros((L,1)).astype(float64)
        lb = 0
        ub = T2
        for i in prange(lb,ub+1):
            T12 = max(0,T1-i)
            T22 = max(0,T2-i)
            A = vector_enumeration(L-1,T12,T22)            
            B = np.zeros((L,A.shape[1])).astype(float64)
            for b in prange(B.shape[1]): 
                B[0,b] = float64(i)
                for a in prange(1,B.shape[0]):
                    B[a,b] = A[a-1,b]
            S = np.concatenate((S,B),axis = 1)
        return S[:,1:]
    else:
        V = np.arange(T1,T2+1).astype(float64)        
        Q = V.reshape((1,V.shape[0]))
        return Q


@jit("float64[:, :](int64, int64,int64,int64,float64)",parallel = True, nopython=True)
def vector_enumeration_monotone_log(L,T1,T2,U_b,eps):
    '''
    L is # of classes
    T1 is the integral sum value lower bound
    T2 is the integral sum value upper bound    
    U_b is the upper bound on the value (invariant: U_b*L > T1)
    
    Returns L by m matrix where m are the number of vector combinations
    '''
    if T1 > 0:
        lb = int(np.log(np.ceil(float(T1)/L))/np.log(1+eps))
    else:     
        lb = -1
    if T2 > 0:
        ub = min(max(int(np.floor(np.log(T2)/np.log(1+eps))),
                     int(np.ceil(np.log(T1)/np.log(1+eps)))),
                 U_b) 
    else:
        ub = -1

    if L > 1:        
        S = np.zeros((L,1)).astype(float64)       
        for i in prange(lb,ub+1):
            if (i> lb) & (
                    np.ceil(np.power(1+eps,i)) == np.ceil(np.power(1+eps,i-1))):
                continue 
            if i == -1:
                q_val = 0
            else:
                q_val = int(np.ceil(np.power(1+eps,i)))
            T12 = max(0,T1-q_val)
            T22 = max(0,T2-q_val)
            A = vector_enumeration_monotone_log(L-1,T12,T22,i,eps)
            B = np.zeros((L,A.shape[1])).astype(float64)
            for b in prange(B.shape[1]): 
                B[0,b] = float64(q_val)
                for a in prange(1,B.shape[0]):
                    B[a,b] = A[a-1,b]
            S = np.concatenate((S,B),axis = 1)
        return S[:,1:]
    else:
        V = np.arange(lb,ub+1)
        G = np.ceil(np.power(1+eps,V)).astype(float64)
        if lb == -1:
            G[0] = 0
        M = np.unique(G)
        Q = M.reshape((1,M.shape[0]))
        return Q
    
    
@jit("float64[:, :](int64,int64[:])",parallel = True, nopython=True)
def capped_vector_enumeration(L,cap):
    '''
    L is # of classes  
    cap is the integral (X +) L lenght vector containing the integral caps
    
    Returns L by m matrix where m are the number of vector combinations
    '''
    if L > 1:        
        S = np.zeros((L,1)).astype(float64)
        lb = 0
        ub = cap[-L]
        for i in prange(lb,ub+1):
#            A = vector_enumeration(L-1,T12,T22,i)
            A = capped_vector_enumeration(L-1,cap)            
            B = np.zeros((L,A.shape[1])).astype(float64)
            for b in prange(B.shape[1]): 
                B[0,b] = float64(i)
                for a in prange(1,B.shape[0]):
                    B[a,b] = A[a-1,b]
            S = np.concatenate((S,B),axis = 1)
        return S[:,1:]
    else:
#        V = np.arange(T1,min(T2,U_b)+1).astype(float64)
        V = np.arange(0,cap[-1]).astype(float64)        
        Q = V.reshape((1,V.shape[0]))
        return Q
    
    
@jit("float64[:, :](int64, int64)",parallel = True, nopython=True)
def sequence_enumeration(L,ub):
    '''
    L is the lenght of the sequence
    ub is the upper bound on the value (the lower bound is 0) 
    
    Returns L by m matrix where m are the number of vector combinations
    '''
    if L > 1:        
        S = np.zeros((L,1), dtype = float64)
        for i in prange(ub+1):
            A = sequence_enumeration(L-1,i)            
            B = np.zeros((L,A.shape[1]), dtype = float64)
            for b in prange(B.shape[1]): 
                B[0,b] = float64(i)
                for a in prange(1,B.shape[0]):
                    B[a,b] = A[a-1,b]
            S = np.concatenate((S,B),axis = 1)
        return S[:,1:]
    else:
        V = np.arange(ub+1).astype(float64)        
        Q = V.reshape((1,V.shape[0]))
        return Q
            
    
@jitclass(spec)
class PtasSimulation(object):
    def __init__(self,n,capacity,max_M,eps,weight_scale,weight_level,
                 price_param,rate_M, 
                 n_samples):
        '''
        Instantiates the generative parameters
        '''
        # set params
        self.n = n 
        self.capacity = capacity
        self.max_M = max_M
        self.eps = eps
        self.weight_scale = weight_scale
        self.weight_level = weight_level        
        self.price_param = price_param
        self.rate_M = rate_M
        self.n_samples = n_samples
    
    def generate_data(self):
        '''
        Generates the random data given instance parameters
        TODO Products should be always numbered by decreasing prices
        '''
        # compute initial params
        self.prices = np.concatenate(
                                     (
                                      np.exp(self.price_param*np.sort(
                                              np.random.randn(self.n)
                                              .astype(float64)
                                                     )[::-1]
                                            )
                                      ,np.zeros((1,),dtype = float64))
                                      ,axis = 0)
        
        self.weights = (
                       np.concatenate(
                                      (self.weight_level*
                                       np.exp(self.weight_scale*
                                               np.sort(np.random.randn(self.n)
                                               .astype(float64))
                                              )
                                       ,np.ones((1,),dtype = float64))
                                      ,axis = 0)
                        )
                                               
        self.rates_M = 1.-self.rate_M*np.random.random(self.max_M)
        self.rates_M[-1] =0
        rates = np.cumprod(self.rates_M)
        final_rate = np.ones(self.max_M)
        final_rate[1:] = rates[:-1]
        self.pdf_M = final_rate * (1-self.rates_M)
        
        self.avg = np.sum(
                            np.multiply(np.arange(self.max_M).astype(float64)
                                        ,self.pdf_M)
                        )
        self.std = np.sum(
                            np.multiply(np.power(np.arange(self.max_M).astype(
                                                                    float64),2)
                                        ,self.pdf_M)
                            )                                 
        self.std = np.sqrt(self.std -self.avg*self.avg)
    
    def evaluate_revenue(self,inventory):
        '''
        Sample-based estimator of the expected revenue
        '''
        rev =0.0
        inventory_vec = np.copy(inventory)
        assortment = (inventory_vec > 0).astype(float64)
        assortment_vec = np.copy(assortment)
        probas = self.weights[:-1]*assortment_vec/(
                          np.sum(self.weights[:-1]*assortment_vec)+1.)
        cumprobas = np.cumsum(probas)
        cumprobas_vec = np.copy(cumprobas) 
        for i in prange(self.n_samples):
            copyto(inventory_vec,inventory)
            copyto(cumprobas_vec,cumprobas)
            copyto(assortment_vec,assortment)
            M = np.argmax(np.random.rand() < np.cumsum(self.pdf_M))            
            for m in prange(M):                  
                  trial = np.random.random()< cumprobas_vec
                  i_max = np.argmax(trial)
                  if trial[i_max]:
                      rev += self.prices[i_max]
                      inventory_vec[i_max] = inventory_vec[i_max] -1                      
                      if inventory_vec[i_max] == 0:
                          assortment_vec[i_max] = 0.  
                          probas = self.weights[:-1]*assortment_vec/(
                          np.sum(self.weights[:-1]*assortment_vec)+1.)
                          cumprobas_vec = np.cumsum(probas)
        return rev/float(self.n_samples)
        
    def solve(self):
        '''
        Run the modified PTAS algorithm
        '''
        combinations = vector_enumeration_monotone_log(self.n
                                                       ,int(np.ceil((1-self.eps)
                                                               *self.capacity))
                                                       ,self.capacity
                                                       ,int(np.ceil(np.log(
                                                        self.capacity)/np.log(
                                                        1+self.eps)))
                                                       ,self.eps
                                                       )
        revenues = np.zeros(combinations.shape[1])                                                                                                          
        for i in prange(0,combinations.shape[1]):
            revenues[i] = self.evaluate_revenue(combinations[:,i])
        i_max = np.argmax(revenues)                
        return(revenues[i_max],combinations[:,i_max],combinations.shape[1])


