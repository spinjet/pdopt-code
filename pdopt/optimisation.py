# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 12:34:30 2021

@author: s345001
"""

__author__ = 'Andrea Spinelli'
__copyright__ = 'Copyright 2021, all rights reserved'
__status__ = 'Development'

# Standard Library Imports
import multiprocess as mp
import psutil 

# Third-party imports
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.factory import get_reference_directions
from pymoo.util.termination.default import MultiObjectiveDefaultTermination
from pymoo.optimize import minimize

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, RBF
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor

from tqdm import tqdm

import pandas as pd

# Local imports
from .data import DesignSpace, Model

# Module Constants
N_PROC = int(psutil.cpu_count(logical = False)*(3/4))


# Module Functions

# Abstract Classes

# Concrete Classes
class Surrogate_Model:
    def __init__(self, parameters_list, samples_file):
        self.parameters = parameters_list
        
    
    def predict(self, x):
        pass
    
    def add_sample(self, x):
        pass

class Optimisation:

    def __init__(self, design_space, model, save_history=False, **kwargs):
        
        # Construct the PyMOO problems for surviving design spaces
        self.design_space   = design_space
        self.model          = model
        self.valid_sets_idx = []
        
        n_proc = kwargs['n_proc'] if 'n_proc' in kwargs else N_PROC
        self.pool           = mp.Pool(n_proc)
        
        
        for i_set in range(len(design_space.sets)):
            if not design_space.sets[i_set].is_discarded:
                design_space.sets[i_set].set_optimisation_problem(self.model, 
                                                        design_space.parameters, 
                                                        design_space.objectives, 
                                                        design_space.constraints,
                                                        self.pool)
                    
                self.valid_sets_idx.append(i_set)
        
        n_partitions = kwargs['n_partitions'] \
            if 'n_partitions' in kwargs else 12
        
        self.ref_dirs = get_reference_directions("das-dennis", 
                                                  len(design_space.objectives), 
                                                  n_partitions=n_partitions)
        
        # Define the algorithm hyperparameters
        pop_size = kwargs['pop_size'] \
            if 'pop_size' in kwargs else 10 + self.ref_dirs.shape[0]

        # # Define the algorithm hyperparameters
        # s_history = kwargs['save_history'] if 'save_history' in kwargs else False


        self.algorithm = UNSGA3(pop_size=pop_size,
                                ref_dirs=self.ref_dirs,
                                eliminate_duplicates=True,
                                save_history=save_history,
                                )
        
        # Define the termination hyperparameters
        x_tol = kwargs['x_tol'] \
            if 'x_tol' in kwargs else 1e-25
                
        cv_tol = kwargs['cv_tol'] \
            if 'cv_tol' in kwargs else 1e-25
                
        f_tol = kwargs['f_tol'] \
            if 'f_tol' in kwargs else 1e-25
        
        n_max_gen = kwargs['n_max_gen'] \
            if 'n_max_gen' in kwargs else 10000
        
        n_max_evals = kwargs['n_max_evals'] \
            if 'n_max_evals' in kwargs else 1000000
        
        
        self.termination = MultiObjectiveDefaultTermination(x_tol=x_tol,
                                                            cv_tol=cv_tol,
                                                            f_tol=f_tol,
                                                            nth_gen=5,
                                                            n_last=10,
                                                            n_max_gen=n_max_gen,
                                                            n_max_evals=n_max_evals)
        
    def run(self):
        for i_set in tqdm(self.valid_sets_idx, desc='Searching in the Design Space'):
            res = minimize(self.design_space.sets[i_set].optimisation_problem, 
                           self.algorithm, 
                           termination=self.termination,
                           verbose=True)
            
            self.design_space.sets[i_set].optimisation_results = res
            
        self.pool.close()

class Surrogate_Robust_Optimisation:
    
    def __init__(self, design_space, exploration, model, P_g = 0.9,
                 k_sigma = 3, decoupled_sigma=False, save_history=False,
                 **kwargs):
        
        # Construct the PyMOO problems for surviving design spaces
        self.design_space   = design_space
        self.exploration    = exploration
        self.model          = model
        self.valid_sets_idx = []
        self.decoupled_sigma = decoupled_sigma
        
        n_proc = kwargs['n_proc'] if 'n_proc' in kwargs else N_PROC
        self.pool           = mp.Pool(n_proc)
        
        
        for i_set in range(len(design_space.sets)):
            if not design_space.sets[i_set].is_discarded:
                design_space.sets[i_set].set_robust_optimisation_problem(self.model, 
                                                        self.exploration.surrogates,
                                                        design_space.parameters, 
                                                        design_space.objectives, 
                                                        design_space.constraints,
                                                        P_g,
                                                        k_sigma,
                                                        self.pool,
                                                        decoupled_sigma=decoupled_sigma)
                    
                self.valid_sets_idx.append(i_set)
        
        n_partitions = kwargs['n_partitions'] \
            if 'n_partitions' in kwargs else 12
        
        if decoupled_sigma:
            self.ref_dirs = get_reference_directions("das-dennis", 
                                                      2*len(design_space.objectives), 
                                                      n_partitions=n_partitions)
        else:
            self.ref_dirs = get_reference_directions("das-dennis", 
                                                      len(design_space.objectives), 
                                                      n_partitions=n_partitions)
        
        # Define the algorithm hyperparameters
        pop_size = kwargs['pop_size'] \
            if 'pop_size' in kwargs else 10 + self.ref_dirs.shape[0]

        # Define the algorithm hyperparameters
        #s_history = kwargs['save_history'] if 'save_history' in kwargs else False


        self.algorithm = UNSGA3(pop_size=pop_size,
                                ref_dirs=self.ref_dirs,
                                eliminate_duplicates=True,
                                save_history=save_history,
                                )
        
        # Define the termination hyperparameters
        x_tol = kwargs['x_tol'] \
            if 'x_tol' in kwargs else 1e-25
                
        cv_tol = kwargs['cv_tol'] \
            if 'cv_tol' in kwargs else 1e-25
                
        f_tol = kwargs['f_tol'] \
            if 'f_tol' in kwargs else 1e-25
        
        n_max_gen = kwargs['n_max_gen'] \
            if 'n_max_gen' in kwargs else 10000
        
        n_max_evals = kwargs['n_max_evals'] \
            if 'n_max_evals' in kwargs else 1000000
        
        
        self.termination = MultiObjectiveDefaultTermination(x_tol=x_tol,
                                                            cv_tol=cv_tol,
                                                            f_tol=f_tol,
                                                            nth_gen=10,
                                                            n_last=5,
                                                            n_max_gen=n_max_gen,
                                                            n_max_evals=n_max_evals)
        
    def run(self):
        for i_set in tqdm(self.valid_sets_idx, desc='Searching in the Design Space'):
            res = minimize(self.design_space.sets[i_set].rbo_problem, 
                           self.algorithm, 
                           termination=self.termination,
                           verbose=True)
            
            self.design_space.sets[i_set].rbo_results_raw = res
            
            X_rbo = res.X
            x_df = pd.DataFrame(X_rbo, columns=self.design_space.par_names)
            
            self.design_space.sets[i_set].rbo_results = pd.concat([
                x_df, pd.DataFrame(self.design_space.sets[i_set].rbo_problem.postprocess(X_rbo))], axis=1)
            
            
        self.pool.close()

# Code if imported
