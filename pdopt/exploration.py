# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 15:42:04 2021

@author: s345001
"""

__author__ = 'Andrea Spinelli'
__copyright__ = 'Copyright 2021, all rights reserved'
__status__ = 'Development'

# Standard Library Imports
from time import time
from itertools import count, product
from os.path import exists
from warnings import warn


# Third-party imports
import numpy as np
import pandas as pd
import multiprocess as mp
import psutil

from scipy.stats.qmc import LatinHypercube, Sobol
from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, RBF
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm

# Local imports
from .data import DesignSpace, Model, ContinousParameter

# Module Constants
N_PROC = int(psutil.cpu_count(logical = False)/2)


# Module Functions
def generate_surrogate_training_data(parameters_list, model, n_train_points, save_dir=None):
    # If factorial sampling has too many points, we switch to a LHS to save time
    
    #levels_list = [parameter.ranges for parameter in parameters_list]
    
    # Generate factorial samples and evaluate them
    #factorial_sampling = [x for x in product(*levels_list)]
    
    #if len(factorial_sampling) > 100:
    samples = LatinHypercube(len(parameters_list)).random(n_train_points)
    
    for i_par in range(len(parameters_list)):
        tmp = samples[:,i_par]
        
        if isinstance(parameters_list[i_par], ContinousParameter):
            #Continous Parameter
            tmp *= abs(parameters_list[i_par].ub - parameters_list[i_par].lb)
            tmp += parameters_list[i_par].lb
        else:
            #Discrete Parameter
            tmp *= (parameters_list[i_par].n_levels)
            np.trunc(tmp, tmp)
    
    factorial_sampling = samples
            
    # Parallelized sampling
    pool = mp.Pool(N_PROC)
    responses = pool.starmap(model.run, tqdm(factorial_sampling, desc='Generating Training Data'))
    
    pool.close()
    
    # responses          = [model.run(*sample) for sample in tqdm(factorial_sampling, 
    #                                                             desc='Response Sampling')]
    
    # Build a dataframe with input and outputs
    out_keys = list(responses[0].keys())
    in_keys  = [parameter.name for parameter in parameters_list]
    
    temp = {key : [] for key in (in_keys + out_keys)}
    
    for des in factorial_sampling:
        for i in range(len(in_keys)):
            temp[in_keys[i]].append(des[i])
            
    for res in responses:
        for i in range(len(out_keys)):
            temp[out_keys[i]].append(res[out_keys[i]])
            
    training_data = pd.DataFrame(temp)
    
    if save_dir:
        training_data.to_csv(save_dir, index=False)
    
    return training_data


def generate_surrogate_test_data(n_points, parameters_list, model, save_dir=None):  
    # Generate random samples via Latin Hypercube and evaluate them
    samples = LatinHypercube(len(parameters_list)).random(n_points)
    

    # The samples are normalized in (0,1), we need to scale them
    # by iterating on each column
    for i_par in range(len(parameters_list)):
        tmp = samples[:,i_par]
        
        if isinstance(parameters_list[i_par], ContinousParameter):
            #Continous Parameter
            tmp *= abs(parameters_list[i_par].ub - parameters_list[i_par].lb)
            tmp += parameters_list[i_par].lb
        else:
            #Discrete Parameter
            tmp *= (parameters_list[i_par].n_levels)
            np.trunc(tmp, tmp)
            
    # Parallelized sampling
    pool = mp.Pool(N_PROC)
    responses = pool.starmap(model.run, tqdm(samples, desc='Generating Testing Data'))
    
    pool.close()
    
    #responses = [model.run(*list(sample)) for sample in samples]

    
    # Build a dataframe with input and outputs
    out_keys = list(responses[0].keys())
    in_keys  = [parameter.name for parameter in parameters_list]
    
    temp = {key : [] for key in (in_keys + out_keys)}
    
    for des in samples:
        for i in range(len(in_keys)):
            temp[in_keys[i]].append(des[i])
            
    for res in responses:
        for i in range(len(out_keys)):
            temp[out_keys[i]].append(res[out_keys[i]])
            
    test_data = pd.DataFrame(temp)
    
    if save_dir:
        test_data.to_csv(save_dir, index=False)
        
    return test_data

def generate_input_samples(n_points, parameters_list, rule='lhs'):
    '''
    Grandom samples via Latin Hypercube and evaluate them
    Rules:
        - 'lhs'   : Latin Hypercube
        - 'sobol' : Sobol Rules
        - 'grid'  : Grid Hypercube
    '''
    
    if rule =='lhs':
        samples = LatinHypercube(len(parameters_list)).random(n_points)
        
        for i_par in range(len(parameters_list)):
            tmp = samples[:,i_par]
            
            if isinstance(parameters_list[i_par], ContinousParameter):
                #Continous Parameter
                tmp *= abs(parameters_list[i_par].ub - parameters_list[i_par].lb)
                tmp += parameters_list[i_par].lb
            else:
                #Discrete Parameter
                tmp *= (parameters_list[i_par].n_levels)
                np.trunc(tmp, tmp)
        
    elif rule == 'sobol':
        samples = Sobol(len(parameters_list)).random(n_points)
        
        for i_par in range(len(parameters_list)):
            tmp = samples[:,i_par]
            
            if isinstance(parameters_list[i_par], ContinousParameter):
                #Continous Parameter
                tmp *= abs(parameters_list[i_par].ub - parameters_list[i_par].lb)
                tmp += parameters_list[i_par].lb
            else:
                #Discrete Parameter
                tmp *= (parameters_list[i_par].n_levels)
                np.trunc(tmp, tmp)        
    else:
        N_lev = round(n_points ** (1/len(parameters_list)))
        
        lev_list = []
                      
        for parm in parameters_list:
    
            if isinstance(parm, ContinousParameter):
                #Continous Parameter
                tmp = np.linspace(parm.lb, parm.ub, N_lev)
                lev_list.append(list(tmp))
            else:
                #Discrete Parameter
                lev_list.append(list(range(parm.n_levels))) 
                
        factorial_sampling = [x for x in product(*lev_list)]
        samples = np.array(factorial_sampling)
        
    return samples



# Abstract Classes

# Concrete Classes
class SurrogateResponse:
    # Automatically generate data and train a gaussian process
    # on the selected response.
    
    _ids = count(0)

    def __init__(self, response_name, parameters_list, model, 
                 train_data=None, test_data=None):
        self.id = next(self._ids)
        self.name = response_name
        self.score = 0
        self.train_time = 0
        
        #Measure elasped time
        t0 = time()
        
        # Load or generate training data
        if train_data is not None:
            training_data = train_data
        else:
            training_data = generate_surrogate_training_data(parameters_list, 
                                                             model, 100)

        # Generate testing data
        if test_data is not None:
            testing_data = test_data
        else:
            testing_data = generate_surrogate_test_data(int(0.25*len(training_data)), 
                                                    parameters_list, 
                                                    model)
        
        
        self.par_names = [parameter.name for parameter in parameters_list]

        # Scale input
        self.x_scaler = MinMaxScaler()

        
        X_train = self.x_scaler.fit_transform(training_data[self.par_names].to_numpy())
        
        Y_train = training_data[self.name].to_numpy().reshape(-1, 1)
        
        
        X_test = self.x_scaler.transform(testing_data[self.par_names].to_numpy())
        Y_test = testing_data[self.name].to_numpy().reshape(-1, 1)
        
        kern = Matern() #ConstantKernel() * RBF()
        
        self.f = GPR(kernel=kern, normalize_y=True).fit(X_train, Y_train)
        self.score = self.f.score(X_test, Y_test)
        
        if self.score < 0.6:
            warn('Warning: Surrogate {} score is low, r={:.4f}'.format(self.name, self.score))
        elif self.score < 0:
            raise('Errore: Surroate {} score is negative, r={:.4f}'.format(self.name, self.score))
        
        t1 = time()
        self.train_time = t1 - t0

    def __repr__(self):
        return f'SurrogateResponse of {self.name}, score = {self.score}'
        
    def predict(self, x):
        
        X_norm = self.x_scaler.transform(np.atleast_2d(x))
        mu, sigma = self.f.predict(X_norm, return_std=True)
        
        return mu, sigma
        
        
class ProbabilisticExploration:
    # Evaluate the design space and calculate the probability to satisfy
    # the requirements and constraints.
    
    def __init__(self, design_space, model, 
                 surrogate_training_data_file=None,
                 n_train_points=100):
        self.design_space   = design_space
        self.parameters     = design_space.parameters
        self.objectives     = design_space.objectives
        self.constraints    = design_space.constraints
        self.run_time       = 0
        
        #Load Samples or Generate Samples
        if surrogate_training_data_file and exists(surrogate_training_data_file):
            self.surrogate_train_data = pd.read_csv(surrogate_training_data_file)
        else:
            self.surrogate_train_data = generate_surrogate_training_data(self.parameters, 
                                                                    model, n_train_points,
                                                                    save_dir=surrogate_training_data_file)
        
        self.surrogate_test_data = generate_surrogate_test_data(30, 
                                                    self.parameters, 
                                                    model)
        
        #Build the list of responses to construct surrogate models of
        surrogate_responses = []
        self.requirements   = {}
        
        # Add objectives only if they have minimum requirements defined
        for objective in self.objectives:
            surrogate_responses.append(objective.name)
            if objective.hasRequirement:
                op, val = objective.get_requirement()
                self.requirements.update({objective.name :
                                          (op, val, 
                                           objective.p_satisfaction)
                                          }
                    )
                
        for constraint in self.constraints:
            surrogate_responses.append(constraint.name)
            op, val = constraint.get_constraint()
            self.requirements.update({constraint.name :
                                      (op, val,
                                       constraint.p_satisfaction)
                                      }
                    )
                
        # Eliminate duplicates
        surrogate_responses = list(set(surrogate_responses))
        
        # Train the surrogates
        self.surrogates = {}
        
        for response in tqdm(surrogate_responses, desc='Training Surrogate Responses'):
            self.surrogates.update({
                response :
                SurrogateResponse(response, 
                                  self.parameters, 
                                  model,
                                  train_data=self.surrogate_train_data, 
                                  test_data=self.surrogate_test_data)
                }
                )
                
        ## Build data structure with responses and their operands
        ## required for the PDOPT space exploration tool
        self.responses = {}
        
        for objective in self.objectives:
            self.responses.update({objective.name : objective.operand})
        
        for constraint in self.constraints:
            self.responses.update({constraint.name : constraint.operand})
        
        
    def run(self, n_samples=100, p_discard=0.5):
        # Run the Probabilistic design exploration and evaluate sets.
        t0 = time()
        for design_set in tqdm(self.design_space.sets, desc='Exploring the Design Space'):
            # Generate samples within the set
            samples = design_set.sample(n_samples, self.parameters)
            
            for requirement_name in self.requirements:
                
                operand, value, p_sat = self.requirements[requirement_name]
                mu, sigma = self.surrogates[requirement_name].predict(samples)
                
                #Calculate P of mu < value
                if operand == 'lt' or operand == 'let':
                    z = (value - mu)/np.atleast_2d(sigma).T
                
                #Calculate P of mu > value
                else:
                    z = (mu - value)/np.atleast_2d(sigma).T
                
                P_samples = norm.cdf(z)
                
                #Now calculate how many samples satisfy the requirement P
                #That is we estimate P(req | set)
                P_req = (P_samples > p_sat).sum() / n_samples
                
                design_set.set_responses_P(requirement_name, P_req)
            
            # Discard the set if the total probability is lower than the
            # specified one
            if design_set.P < p_discard:
                design_set.set_as_discarded()
        self.run_time = time() - t0
        
    def run_surrogate(self, X):
        means, deviations = {}, {}
        for requirement_name in self.requirements:
            mu, sigma = self.surrogates[requirement_name].predict(X)
            means.update({requirement_name:mu})
            deviations.update({requirement_name:sigma})
        
        return means, deviations

# Code if imported
    
    