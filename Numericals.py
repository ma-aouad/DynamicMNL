'''
Created on Dec 8, 2015

@author: mohammedaliaouad
'''
import pandas as pd
import numpy as np
import math
import time
from random import *
from gurobipy import Model,GRB,quicksum
from scipy.stats import poisson,norm
import scipy.misc as sc

#NOTE: CAPACITY  EXCEEDS THE NUMBER OF PRODUCTS

#Example of parameters
parameters = {
              "n_product":20,
              "capacity":30,
              "max_M": 20,
              "rate_M": 0.2,
              "eps": 0.1,
              "weight": 1,
                "price": 1,
                "model": "nonparametric", #or Poisson
                "max_time": 1000
              }

class Numerical():
    '''
    Optimization object - random instance simulator + algorithm + heuristics ()
    '''
    def __init__(self,params = parameters):
        '''
        Set the desired parameters
        '''
        self.n = params["n_product"]
        self.capacity = params["capacity"]
        self.max_M = params["max_M"]
        self.eps = params["eps"]
        self.weight_param = params["weight"]
        self.price_param = params["price"]
        self.rate_M = params["rate_M"]
        self.model = params["model"]
        self.max_time = params["max_time"]
        self.refresh_data()


    def refresh_data(self):
        '''
        Generates a new model
        '''
        if self.weight_param == 1:
            self.weights = np.concatenate((np.sort(np.random.random(self.n))[::-1],[1]),axis = 0)
        elif self.weight_param >= 2:
            self.weights = np.concatenate((0.5*np.exp(np.sort(np.random.normal(scale = 1.0,size = self.n)))[::-1],[1]),axis = 0)

        if self.price_param == 1:
            #self.prices = np.concatenate((np.sort(np.random.random(self.n)),[0]),axis = 0)
            self.prices = np.concatenate((np.exp(np.sort(np.random.normal(scale = 1.0,size = self.n))),[0]),axis = 0)
        elif self.price_param == 2:
            self.prices = np.concatenate((np.exp(np.sort(np.random.normal(scale = 2.0,size = self.n))),[0]),axis = 0)
        elif self.price_param == 3:
            self.prices = np.concatenate((np.exp(np.sort(np.random.normal(scale = 3.0,size = self.n))),[0]),axis = 0)
        #Distribution of M (non parametric)
        if self.model == "nonparametric":
            #survival rate
            self.M_dis = np.sort(1-self.rate_M*np.random.random(self.max_M))[::-1]
            self.M_dis[-1] =0
            #self.M_dis = np.array(reduce(lambda x,y: x+ [x[-1]*y], self.M_dis,[1]))
        elif self.model == "poisson":
            #self.M_dis = np.array(reduce(lambda x,y: x+ [x[-1]*y], self.M_dis,[1]))
            self.M_dis = np.divide(np.power((1-self.rate_M)*self.max_M,np.arange(self.max_M)),sc.factorial(np.arange(self.max_M)))*np.exp(-(1-self.rate_M)*self.max_M)
            #print self.M_dis
            self.M_dis = np.divide(self.M_dis, np.array([np.sum(self.M_dis[i:]) for i in range(self.max_M)]))
            self.M_dis = 1 - self.M_dis
        elif self.model == "geometric":
            self.M_dis = 1-self.rate_M*np.ones(self.max_M)
        else:
            raise ValueError("Unknown model")
        print self.M_dis
        self.avg = np.sum(np.cumprod(self.M_dis))
        self.std = np.sqrt(np.sum(np.multiply(np.multiply(np.cumprod(self.M_dis)[:-1],1-self.M_dis[1:]),
                                  np.power(np.arange(1,self.max_M),2))
                          ) - np.power(self.avg,2))
        print "Avg demand ",self.avg,"Std demand ",self.std
        
        return()


    def one_sample(self,solution):
        '''
        Generates a single path sample and returns the corresponding revenues
        '''
        revenue = 0
        Assortment = np.where(solution > 0)[0].tolist()+ [self.n]
        inventory = np.copy(solution)
        z= 0
        while z < self.max_M + 1:
            if len(Assortment) ==  1:
                break
            elif np.random.rand() > 1-self.M_dis[z]:
                prod = Assortment[np.argmax(np.cumsum(self.weights[np.ix_(Assortment)]) > np.sum(self.weights[np.ix_(Assortment)])*np.random.rand())]
                revenue = revenue + self.prices[prod]
                if prod < self.n:
                    inventory[prod] = inventory[prod] - 1
                    if inventory[prod] == 0:
                        Assortment = Assortment[:Assortment.index(prod)] + Assortment[Assortment.index(prod)+1:]
                z =  z + 1
                #print z,revenue,prod
            else:
                break
        return(revenue)


    def sample_revenue(self,solution,n_samples = 500):
        '''
        Average-sample estimator
        '''
        return(reduce(lambda x,y: x + self.one_sample(solution),range(n_samples),0)/n_samples)


    def light_alg(self,indices):
        '''
        Implements the light-algorithm through dynamic programming
        '''
        #Indices of light products
        indices = sorted(indices)
        #DP data tables
        DP_val = np.zeros((self.capacity+1,len(indices)))
        DP_arg = np.zeros((self.capacity+1,len(indices)))

        for i,j in enumerate(indices):
            reward = np.zeros(self.capacity+1)
            for c in range(1,self.capacity+1):
                #Calculates the reward for each inventory level
                alloc = np.zeros(c+1)
                #Distribution of remaining units
                alloc[c] = 1.0
                proba = 1

                for m in range(self.max_M):
                    proba = proba*(self.M_dis[m])
                    zero_prob = alloc[0]
                    reward[c] = reward[c] + proba*np.sum(self.weights[j]*alloc[1:])*self.prices[j]
                    alloc[:-1] = alloc[1:]*self.weights[j] + (1-self.weights[j])*alloc[:-1]
                    #boundaries
                    alloc[0] = alloc[0] + zero_prob*self.weights[j]
                    alloc[-1] = (1-self.weights[j])*alloc[-1]

            for c in range(self.capacity+1):
                if i > 0:
                    ## cc / c is how many units were used above
                    DP_val[c,i] = max([DP_val[cc,i-1] + reward[cc-c] for cc in range(c,self.capacity+1)])
                    DP_arg[c,i] = np.argmax([DP_val[cc,i-1] + reward[cc-c] for cc in range(c,self.capacity+1)])
                else:
                    DP_val[c,i] = reward[self.capacity-c]
                    DP_arg[c,i] = self.capacity-c
        solution = reduce(lambda x,y: x + [DP_arg[int(sum(x)),y]], range(len(indices)-2,-1,-1),[DP_arg[0,-1]])[::-1]

        return np.array(solution),DP_val[0,-1]


    def algorithm(self):
        #Candidate light
        indices = np.where(self.weights < self.eps)[0].tolist()
        if len(indices) > 0:
            candidate_light = self.light_alg(indices)
            val_light = candidate_light[1]
            candidate_light = candidate_light[0]
            val_light = self.sample_revenue(candidate_light)

        else:
            #prod = np.argmax(np.multiply(self.weights,self.prices))
            #candidate_light = np.zeros(self.n)
            #candidate_light[prod] = self.capacity
            val_light = 0

        #Candidate heavy
        indices = np.where(self.weights > 1/self.eps)[0].tolist()
        if len(indices) > 0:
            candidate_heavy = np.zeros(self.n)
            candidate_heavy[max(indices)] = self.capacity
            val_heavy = self.sample_revenue(candidate_heavy)
        else:
            #prod = np.argmax(np.multiply(self.weights,self.prices))
            #candidate_heavy = np.zeros(self.n)
            #candidate_heavy[prod] = self.capacity
            val_heavy = 0

        #Solve static problem (capacity > n)
        opt_ass = []
        incr = 0
        val_locale = 0
        denom = 1
        ind = self.n -1

        while incr >= 0 and ind >= 0:
            new_val = val_locale*denom/(denom + self.weights[ind]) + self.weights[ind]*self.prices[ind]/(denom + self.weights[ind])
            incr = new_val - val_locale
            if incr >= 0:
                val_locale =  new_val
                denom =  denom + self.weights[ind]
                opt_ass.append(ind)
                ind = ind -1

        #Core greedy algorithm
        solution = np.zeros(self.n)
        for i in range(self.capacity):
            comparatif = 0
            prod_incr = 0
            for j in opt_ass:
                solution[j] = solution[j] + 1
                val_locale = self.sample_revenue(solution)
                if val_locale > comparatif:
                    prod_incr = j
                    comparatif = val_locale
                solution[j] = solution[j] - 1
            solution[prod_incr] = solution[prod_incr] + 1

#         if val_light == max([val_light,val_heavy,comparatif]):
#             print "Light", candidate_light
#         elif val_heavy == max([val_light,val_heavy,comparatif]):
#             print "Heavy", candidate_heavy
#         else:
#             print "Medium", solution

        #The scale-contribution vector to compete with light products
        contrib = np.zeros(len(opt_ass))

        for i,j in enumerate(opt_ass):
            contrib[i] = self.weights[j]*self.prices[j]/denom

        scale = self.capacity/np.sum(contrib)*contrib
        solution_cheap = np.floor(scale)

        #Rounding
        ordered_list = np.argsort(scale - solution_cheap)[::-1]

        for i in range(int(self.capacity - np.sum(solution_cheap))):
            solution_cheap[ordered_list[i]] = solution_cheap[ordered_list[i]] + 1

        solution = np.zeros(self.n)

        for i,j in enumerate(opt_ass):
            solution[j] = solution_cheap[i]

        comparatif_cheap = self.sample_revenue(solution)

        return (max([val_light,val_heavy,comparatif,comparatif_cheap]),comparatif_cheap)
        #index_max = np.argmax([val_light,val_heavy,comparatif,comparatif_cheap])
        #sol_end = [candidate_light,candidate_light,solution,solution_cheap][index_max]
        #return  "-".join(map(lambda x: str(x),self.weights)) + "_" + "-".join(map(lambda x: str(x),self.prices)) + "_" + "-".join(map(lambda x: str(x),sol_end)) + "-" + str(max([val_light,val_heavy,comparatif,comparatif_cheap]))


    def greedy(self):
        '''
        Implements the greedy algorithm, adding units one at a time
        '''
        solution_greedy = np.zeros(self.n)
        t = time.time()
        while np.sum(solution_greedy) < self.capacity and (time.time() - t < self.max_time):
            comparatif_greedy = 0
            prod_incr = -1
            for j in range(self.n):
                solution_greedy[j] = solution_greedy[j] + 1
                val_locale = self.sample_revenue(solution_greedy)
                if val_locale > comparatif_greedy:
                    prod_incr = j
                    comparatif_greedy = val_locale
                solution_greedy[j] = solution_greedy[j] - 1
            if prod_incr > -1:
                solution_greedy[prod_incr] = solution_greedy[prod_incr] + 1

        return(comparatif_greedy)
        #return "-".join(map(lambda x: str(x),self.weights)) + "_" + "-".join(map(lambda x: str(x),self.prices)) + "_" + "-".join(map(lambda x: str(x),solution_greedy))+ "-" + str(comparatif_greedy)


    def local_search(self):
        '''
        Local swaps & local search
        '''
        prod = np.argmax(np.multiply(self.weights,self.prices))
        solution_local = np.zeros(self.n)
        solution_local[prod] = self.capacity
        comparatif = self.sample_revenue(solution_local)
        incr = comparatif
        old_comparatif = comparatif
        iterations = 0
        t = time.time()
        while (incr/old_comparatif) > 0.01 and (iterations< 250) and (time.time() - t < self.max_time):
            iterations += 1
            old_comparatif = comparatif
            assortment = np.where(solution_local > 0)[0].tolist()
            prod = 0
            incr = 0
            for i in assortment:
                for j in range(self.n):
                    solution_local[j] = solution_local[j] + 1
                    solution_local[i] = solution_local[i] - 1
                    val_locale = self.sample_revenue(solution_local)
                    if val_locale > comparatif:
                        incr = incr +  val_locale - comparatif
                        comparatif = val_locale
                        prod = (i,j)

                    solution_local[j] = solution_local[j] - 1
                    solution_local[i] = solution_local[i] + 1

            if prod <> 0:
                solution_local[prod[1]] = solution_local[prod[1]] + 1
                solution_local[prod[0]] = solution_local[prod[0]] - 1

        return(comparatif)
        #return "-".join(map(lambda x: str(x),self.weights)) + "_" + "-".join(map(lambda x: str(x),self.prices)) + "_" + "-".join(map(lambda x: str(x),solution_local)) + "-" + str(comparatif)


    def naming(self,l):
        return ",".join(map(lambda x: str(x),l))


    def lookup_sample(self,sol):
        name = self.naming(sol.tolist())
        if name not in self.val_dict:
            self.val_dict[name] = self.sample_revenue(sol)
        return(name)


    def lovasz_greedy(self):
        '''
        Continuous greedy algorithm
        '''

        solution_local = np.zeros(self.n)

        self.val_dict = {self.naming(solution_local.tolist()): 0 }

        eps = 0.05*self.capacity

        e ={i:np.zeros(self.n) for i in range(self.n)}

        for i in range(self.n):
            e[i][i] = 1.0

        marginal_increase = 1.0
        iteration = 0
        t = time.time()
        
        while (sum(solution_local) < self.capacity or marginal_increase > 0.005) and (iteration < 250) and (time.time() - t < self.max_time):
            # print iteration
            iteration = iteration + 1
            solution_base = np.floor(solution_local)
            ordered_set = np.argsort(solution_local-solution_base)
            # computing the nested ordered solutions
            vectors = map(lambda x: reduce(lambda z,t: z + e[t],ordered_set[x:], 
                                           np.copy(solution_base)
                                           ),range(self.n)
                          ) + [solution_base]
            # naming the new samples
            names = map(lambda x: self.lookup_sample(x),vectors)
            # computing the gradient values
            gradients = map(lambda x: self.val_dict[names[x]] - self.val_dict[names[x+1]],
                            range(self.n)
                            )
            # ordering the gradient values in the gradient vector
            gradient = np.array(map(lambda x: gradients[np.where(ordered_set==x)[0][0]],range(self.n)))
            # computing the step size
            eps = max((self.capacity - np.sum(solution_local))/2.,0.05*self.capacity)
            solution_local = np.maximum(solution_local + eps*1./np.sum(np.abs(gradient))*gradient,0)
            name_new = self.lookup_sample(np.floor(solution_local))
            marginal_increase = eps/np.sum(np.abs(gradient))/(0.001+self.val_dict[name_new])*np.dot(gradient,gradient)
            #marginal_increase = self.val_dict[name_new]/self.val_dict[names[-1]] - 1.
            if np.sum(solution_local) > self.capacity:
                solution_local = self.capacity/np.sum(solution_local)*solution_local
                marginal_increase = self.capacity/np.sum(solution_local)*marginal_increase
            
        solution_local = self.capacity/np.sum(solution_local)*solution_local
        solution = np.floor(solution_local)

        #Use spare capacity due to rounding to improve the expected revenue
        if np.sum(solution) < self.capacity - 1:
            indices = np.argsort(solution_local - solution)[::-1]
            for i in range(int(self.capacity - np.sum(solution))):
                ind = indices[i]
                solution[ind] = solution[ind] + 1

        return self.sample_revenue(solution)
        #return "-".join(map(lambda x: str(x),self.weights)) + "_" + "-".join(map(lambda x: str(x),self.prices)) + "_" + "-".join(map(lambda x: str(x),solution))


    def relaxation_topaloglu(self,mode ="normal"):
        '''
        Computes the continuous relaxation according to Topaloglu (2013) 
        Note that we should have capacity << 5*max_M 
        IMPORTANT: bisection does not seem to converge 
        '''
        c_1 = 0
        c_2 = np.max(self.prices)
        #step size of discretization
        eps = 0.02
        if mode == "normal":
            eps = 0.02
        M_eps = int(1/eps)
        solution = np.zeros(self.n)
        best_val = 0
        t = time.time()
        while (np.sum(solution) <> self.capacity) and (time.time() - t < self.max_time) and (c_2-c_1)/np.max(self.prices)> 10e-4:
            #print(c_1,c_2,np.sum(solution))
            c_value = (c_1 + c_2)/2
            # value function of DP
            DP_vals = np.zeros((M_eps+1,self.n,M_eps+1))
            # best n_i <-> w_i in the discretization
            DP_argmax = np.zeros((M_eps+1,self.n,M_eps+1))
            # inventory level
            DP_inv = np.zeros((M_eps+1,self.n,M_eps+1))
            for n_0 in range(1,M_eps+1):
                for i in range(self.n):
                    for n_tot in range(n_0,M_eps+1):
                        qtys = {}
                        vals = {}
                        # n_i should not violate n_tot and the ratio constraint on w_i/v_i
                        ni_max = int(np.floor(n_0*self.weights[i]))
                        if ni_max + n_0 <= n_tot:
                            if mode == "normal":
                                subset = [0,ni_max]
                            else:
                                subset = range(ni_max+1)
                        else:
                            if mode == "normal":
                                subset = [0]
                            else:
                                ni_max = min(n_tot - n_0,ni_max)
                                subset = range(ni_max+1)
                            
                        for n_i in subset:
                            # finding the right w_i in discretization
                            if n_i==0:
                                probas = np.zeros(self.capacity + 1)
                            elif mode == "poisson":
                                rv = poisson(self.avg*(n_i+1)*eps)
                                probas = rv.pmf(range(self.max_M+1))
                                probas[-1] = 1 - np.sum(probas[:-1])
#                                 print "avg",np.dot(probas,np.arange(self.max_M+1)),self.avg*(n_i+1)*eps
#                                 print "std",np.sqrt(np.dot(probas,np.power(np.arange(self.max_M+1),2))-
#                                                     np.power(np.dot(probas,np.arange(self.max_M+1)),2)
#                                                     ),self.std*np.sqrt((n_i+1)*eps)
#                                 print probas
                                
                            elif mode == "normal":
                                rv = norm(self.avg*(n_i+1)*eps,self.std*np.sqrt((n_i+1)*eps))
                                probas = rv.cdf(np.arange(0,self.max_M + 1))   
                                low = np.zeros(self.max_M+1)
                                low[1:] = probas[:-1]
                                probas = probas - low
                                probas[-1] = 1 - np.sum(probas[:-1])
#                                 print probas                        
                            else:
                                raise ValueError("Unknown model!")
                                
                            #if n_i == ni_max:
                            #    print probas
                            values_array = self.prices[i]*np.cumsum(np.array([0]+[np.sum(probas[g:]) for g in range(1,self.capacity + 1)])) - c_value*np.arange(0,self.capacity + 1)
                            #print(values_array)
                            qtys[n_i] = np.argmax(values_array)
                            vals[n_i] = np.max(values_array)
                            #if qtys[n_i] > 0:
                            #    print("qty",qtys[n_i],i)
                        choice = np.array([DP_vals[n_0,i-1,n_tot - n_i] + vals[n_i] for n_i in subset])
                        DP_vals[n_0,i,n_tot] = np.max(choice)
                        DP_argmax[n_0,i,n_tot] = subset[np.argmax(choice)]
                        DP_inv[n_0,i,n_tot] = qtys[DP_argmax[n_0,i,n_tot]]
                            
            # reconstructing the inventory vector
            arg_tot_rev = np.unravel_index(np.argmax(DP_vals),DP_vals.shape)
            best_val_now = np.max(DP_vals)
            solution_now = [DP_inv[arg_tot_rev]]
            for i in range(arg_tot_rev[1]-1,-1,-1):
                arg_tot_rev = (arg_tot_rev[0],i,int(arg_tot_rev[2]-DP_argmax[arg_tot_rev]))
                solution_now = [DP_inv[arg_tot_rev]] + solution_now
            # filling the solution with zeros on the right of the maximal product
            solution_now = np.array(solution_now + [0. for i in range(self.n-len(solution_now))])
            #updating the per-unit cost
            if np.abs(np.sum(solution) - self.capacity) > np.abs(np.sum(solution_now) - self.capacity) or np.sum(solution) == 0:
                solution = solution_now
                best_val = best_val_now
                 
            if np.sum(solution_now) > self.capacity:
                c_1 = c_value
            elif np.sum(solution_now) < self.capacity:
                c_2 = c_value
            else:
                break
            
        solution_local = self.capacity/np.sum(solution)*solution
        best_val = self.capacity/np.sum(solution)*best_val
        solution = np.floor(solution_local)

        #Use spare capacity due to rounding to improve the expected revenue
        if np.sum(solution) < self.capacity - 1:
            indices = np.argsort(solution_local - solution)[::-1]
            for i in range(int(self.capacity - np.sum(solution))):
                ind = indices[i]
                solution[ind] = solution[ind] + 1
        
        #print np.sum(solution)
        
        return(self.sample_revenue(solution),best_val + c_value*self.capacity)
                
    
    def relaxation_deterministic(self):
        '''
        Computes the deterministic relaxation according to Honhon et al. (2010) 
        Note that we should have capacity <= max_M 
        '''
        M = Model()
        M.Params.OutputFlag = 0
        M.Params.TimeLimit = self.max_time
        M.Params.MIPGap = 0.005
        
        Inf_val = 1000.
        
        survival_rate = np.cumprod(self.M_dis)
        
        no_purchase_rate = M.addVars(self.n,lb =0,vtype = GRB.CONTINUOUS) # total mass of no purchase at each period
        product_rates = M.addVars(self.n,self.n,lb =0,vtype = GRB.CONTINUOUS) # total mass of product purchase at each period
        assortments  = M.addVars(self.n,self.n,vtype = GRB.BINARY) # assortment at each period
        tau = M.addVars(self.n,lb =0,vtype = GRB.CONTINUOUS) # nominal total mass (time)
        tau_decomposed = M.addVars(self.n,self.max_M,lb =0,vtype = GRB.CONTINUOUS) # fractional part of the arrival rank 
        tau_decomposed_bin = M.addVars(self.n,self.max_M,vtype = GRB.BINARY) # binary part of arrival rank
        product_discounts = M.addVars(self.n,self.n,lb =0,vtype = GRB.CONTINUOUS) # discounted time or mass of each product at each period  
        no_purchase_discount = M.addVars(self.n,lb =0,vtype = GRB.CONTINUOUS) # discounted time or mass of no purchase  at each period
        Y = M.addVars(self.n,vtype = GRB.INTEGER)
        
        # Setting the start value at single-product stocking
        for i in range(self.n):
            no_purchase_rate[i].start = 0
            tau[i].start = 0
            no_purchase_discount[i].start = 0
            Y[i].start = 0
            for j in range(self.n):
                product_rates[i,j].start = 0
                assortments[i,j].start = 0
                product_discounts[i,j].start = 0
            for m in range(self.max_M):
                tau_decomposed[i,m].start = 0
                tau_decomposed_bin[i,m].start = 0
                
        M.update()
        
        prod = np.argmax(np.multiply(self.weights,self.prices))
        Y[prod].start = self.capacity
        tau_start = self.capacity/self.weights[prod]*(self.weights[self.n]+self.weights[prod])
        tau[0].start = tau_start
        product_rates[0,prod].start  = self.capacity
        no_purchase_rate[0].start  = self.capacity*self.weights[self.n]/self.weights[prod]
        assortments[0,prod].start  = 1 
        vec = np.zeros(self.max_M)
        for m in range(self.max_M):
            if m < tau_start - 1 and m < self.max_M-1:
                for i in range(self.n):
                    tau_decomposed_bin[i,m].start  = 1
                vec[m] = 1
            elif m < tau_start:
                for i in range(self.n):
                    tau_decomposed[i,m].start  = tau_start - m
                vec[m] = tau_start - m
        tau_discount = np.dot(survival_rate,vec)/tau_start
        product_discounts[0,prod].start  = tau_discount*self.capacity
        no_purchase_discount[0].start  = tau_discount*self.capacity*self.weights[self.n]/self.weights[prod]

        # Capacity 
        M.addConstr(quicksum(Y[i] for i in range(self.n)) <= self.capacity)
    
        for period in range(self.n):
            #Assortment constraint
            M.addConstrs((assortments[period,i] >=  Y[i]/Inf_val - 
                          1/Inf_val*quicksum(product_rates[p,i] for p in range(period)) 
                          for i in range(self.n)), name = "Residual assortment"
                         )
            
            M.addConstrs((Y[i] >= 
                          quicksum(product_rates[p,i] for p in range(period+1)) 
                          for i in range(self.n)), name = "Max time mass does not violate inventory"
                         )
            
            if period > 0:
                M.addConstr( quicksum(assortments[period,i] for i in range(self.n)) <= 
                              (1-1./Inf_val)*quicksum(assortments[period-1,i] for i in range(self.n))
                            )
                
            M.addConstrs((product_rates[period,i] >= self.weights[i]*no_purchase_rate[period] -
                          Inf_val*(1-assortments[period,i])
                         for i in range(self.n)), name = "lower bound on rate"
                         )
            
            M.addConstrs((product_rates[period,i] <= Inf_val*assortments[period,i]
                         for i in range(self.n)), name = "upper bound on rate 1"
                         )
            
            M.addConstrs((product_rates[period,i] <= self.weights[i]*no_purchase_rate[period]
                         for i in range(self.n)), name = "upper bound on rate 2"
                         )
            
            # stopping time
            M.addConstr( tau[period] == quicksum(product_rates[period,i] for i in range(self.n))+
                         no_purchase_rate[period], name = "definition of tau")
            
            # orderings between the decomposed tau
            M.addConstrs((tau_decomposed_bin[period,m+1]<= tau_decomposed_bin[period,m]for m in range(self.max_M-1)), 
                         name = "monotony of tau bin")
            M.addConstrs((tau_decomposed_bin[period,m]>= tau_decomposed[period,m+1]/Inf_val for m in range(self.max_M-1)),
                         name = "consistency of tau bin")
            M.addConstrs((tau_decomposed_bin[period,m] + tau_decomposed[period,m] <= 1 for m in range(self.max_M-1)),
                         name = "tau + tau bin < 1")
            M.addConstr(tau_decomposed_bin[period,self.max_M-1] ==0, name = "ending the tau bin at zero")
            
            # definition of decomposed tau
            M.addConstr(quicksum(tau[p] for p in range(period+1)) == 
                        quicksum(tau_decomposed_bin[period,m] + tau_decomposed[period,m]
                                 for m in range(self.max_M)), name = "definition of tau decomposition"
                        )
            
            # definition of discount
            if period > 0:
                M.addConstr(quicksum(survival_rate[m]*(tau_decomposed_bin[period,m] + 
                                                       tau_decomposed[period,m] -
                                                       tau_decomposed_bin[period-1,m] -
                                                       tau_decomposed[period-1,m]
                                                       ) 
                                     for m in range(self.max_M)
                                     ) ==  quicksum(product_discounts[period,i] for i in range(self.n)) +
                                            no_purchase_discount[period], 
                            name ="renault"
                            )
            else:
                M.addConstr(quicksum(survival_rate[m]*(tau_decomposed_bin[period,m] + 
                                                       tau_decomposed[period,m]
                                                       ) 
                                     for m in range(self.max_M)
                                     ) ==  quicksum(product_discounts[period,i] for i in range(self.n)) +
                                            no_purchase_discount[period]
                            , name ="peugeot"
                            )
            
            M.addConstrs((product_discounts[period,i] >= self.weights[i]*no_purchase_discount[period] -
                          Inf_val*(1-assortments[period,i])
                         for i in range(self.n)), name = "lower bound on discount"
                         )
            
            M.addConstrs((product_discounts[period,i] <= Inf_val*assortments[period,i]
                         for i in range(self.n)), name = "upper bound on discount 1"
                         )
            
            M.addConstrs((product_discounts[period,i] <= self.weights[i]*no_purchase_discount[period]
                         for i in range(self.n)), name = "upper bound on discount 2"
                         )
            
            
        M.setObjective( quicksum(self.prices[i]*quicksum(product_discounts[period,i] for period in range(self.n))
                                 for i in range(self.n)
                                 ),
                        GRB.MAXIMIZE)
        
        M.optimize()
        
        solution = np.array([Y[i].x for i in range(self.n)])
        
        solution_local = self.capacity/np.sum(solution)*solution
        solution = np.floor(solution_local)

        #Use spare capacity due to rounding to improve the expected revenue
        if np.sum(solution) < self.capacity - 1:
            indices = np.argsort(solution_local - solution)[::-1]
            for i in range(int(self.capacity - np.sum(solution))):
                ind = indices[i]
                solution[ind] = solution[ind] + 1
        
        #print M.objVal,self.sample_revenue(solution),solution
        #print self.weights, self.prices
        
        return self.sample_revenue(solution),M.objVal,M.MIPGap
        
        
if __name__ == '__main__':
    pass

    parameters = {
                "n_product":20,
                "capacity":30,
                "max_M": 20,
                "rate_M": 0.02,
                "eps": 0.1,
                "weight": 1,
                "price": 1,
                "model": "nonparametric", #or Poisson
                "max_time": 1000
              }

    #params = [(20,100,100,3,2),(20,100,100,2,2),(20,100,100,1,1),(20,50,100,2,2),(20,50,100,3,2),(20,50,100,1,1),(20,30,20,1,1),(20,30,20,2,2),(20,30,20,3,2),(20,20,20,1,1),(20,20,20,2,2),(20,20,20,3,2)]
    #params = [(20,30,20,1,1),(20,30,20,2,2),(20,20,20,1,1),(20,20,20,2,2)]
    #(20,25,100,1,1,0.66),(20,50,100,1,1,0.66)
    #params = [(20,25,100,2,2,0.66),(20,50,100,2,2,0.66),(20,100,100,2,2,0.66)]
    params = [(20,25,100,1,1,0.04),(20,50,100,1,1,0.04),(20,100,100,1,1,0.04),(20,25,100,2,2,0.04),(20,50,100,2,2,0.04),(20,100,100,2,2,0.04)]
    #params = [(3,3,50,3,2,0.5)]
    for s in params:
        sol = {
            'alg_norm':[],
            'B_norm':[],
            'alg_top':[],
            'B_top':[],
            'lovasz':[],
            'greedy':[],
            'alg': [],
            'local':[],
            'cheap':[],
            'alg_det':[],
            'B_det':[],
            'MIP gap':[],
            't_top':[],
            't_det':[],
            't_norm':[],
            't_lovasz':[],
            't_greedy':[],
            't_alg': [],
            't_local':[]
           }

        parameters["n_product"] = s[0]
        parameters["capacity"] = s[1]
        parameters["max_M"] = s[2]
        parameters["price"] = s[3]
        parameters["weight"] = s[4]
        parameters["rate_M"] = s[5]

        Obj = Numerical(params = parameters)

        for i in range(10):

            Obj.refresh_data()
            
            t = time.time()
            a,b,c = Obj.relaxation_deterministic()
            print a,b,c
            try:
                sol["alg_det"].append(a)
                sol["B_det"].append(b)
                sol["MIP gap"].append(c)
            except:
                sol["alg_det"].append("ERROR")
                sol["B_det"].append("ERROR")
            t = time.time() - t
            sol["t_det"].append(t)
            
            t = time.time()
            a,b = Obj.relaxation_topaloglu("poisson")
            try:
                sol["alg_top"].append(a)
                sol["B_top"].append(b)
            except:
                sol["alg_top"].append("ERROR")
                sol["B_top"].append("ERROR")
            t = time.time() - t
            sol["t_top"].append(t)
            
            t = time.time()
            a,b = Obj.relaxation_topaloglu("normal")
            try:
                sol["alg_norm"].append(a)
                sol["B_norm"].append(b)
            except:
                sol["alg_norm"].append("ERROR")
                sol["B_norm"].append("ERROR")
            t = time.time() - t
            sol["t_norm"].append(t)
            
            t = time.time()
            try:
                sol["lovasz"].append(Obj.lovasz_greedy())
            except:
                sol["lovasz"].append("ERROR")
            t = time.time() - t
            sol["t_lovasz"].append(t)

            t = time.time()
            try:
                sol["greedy"].append(Obj.greedy())
            except:
                sol["greedy"].append("ERROR")
            t = time.time() - t
            sol["t_greedy"].append(t)

            t = time.time()
            try:
                a,b = Obj.algorithm()
                sol["alg"].append(a)
                sol["cheap"].append(b)
            except:
                sol["alg"].append("ERROR")
                sol["cheap"].append("ERROR")
            t = time.time() - t
            sol["t_alg"].append(t)

            t = time.time()
            try:
                sol["local"].append(Obj.local_search())
            except:
                #raise
                sol["local"].append("ERROR")
            t = time.time() - t
            sol["t_local"].append(t)

            print sol

            pd.DataFrame(sol).to_csv("Dynamic/" + ",".join(map(lambda x: str(x),s))+ parameters["model"]+"_last" + ".csv")
