# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 11:31:14 2021

Module that contains all the data structures used within PDOPT.

@author: Andrea Spinelli
"""

__author__ = 'Andrea Spinelli'
__copyright__ = 'Copyright 2021, all rights reserved'
__status__ = 'Development'

# Standard Library Imports
from itertools import count, product
from abc import ABC, abstractmethod
from time import time
import sys
import multiprocessing as mp
import pickle

# Third-party imports
import numpy  as np
import pandas as pd

from scipy.stats.qmc import LatinHypercube, Sobol
from scipy.stats import norm, triang, uniform
from pymoo.core.problem import Problem, ElementwiseProblem, starmap_parallelized_eval

#from mpire import WorkerPool

# Local imports

# Module Constants

# Module Functions


# Abstract Classes
class Base:
    _ids = count(0)
    
    def __init__(self):
        pass


class Parameter:
    _ids = count(0)
    
    
    def get_id(self):
        return self.id
    
    
    def get_name(self):
        return self.name
    
    
    def __init__(self, name):
        self.id   = next(self._ids)
        self.name = name
        
        
    def __repr__(self):
        pass

class Response:
    _ids = count(0)
    
    def get_id(self):
        return self.id
    
    
    def get_name(self):
        return self.name
    
    
    def __init__(self, name, operand, value, p_satisfaction):
        self.id             = next(self._ids)
        self.name           = name
        self.operand        = operand
        self.value          = value
        #Min P to satisfy the response, for the sample
        self.p_satisfaction = p_satisfaction

class ExtendableModel:
    # Model Object that can be extended and used by the library
    def __init__(self):
        pass
    
    def run(self):
        pass
        
# Concrete Classes
class ContinousParameter(Parameter):
    
    def get_bounds(self):
        return self.lb, self.ub
    
    
    def get_level_bounds(self, level):
        assert level < self.n_levels, "Selected level is above total number of levels"
        return self.ranges[level], self.ranges[level+1]
    
    
    def __init__(self, name, lb, ub, n_levels, uq_dist, uq_var_l, uq_var_u):
        super().__init__(name)
        
        assert lb < ub, "Lower Bound value higher than Upper Bound value"
        self.lb, self.ub = lb, ub
        self.n_levels = n_levels
        
        self.ranges  = tuple(np.linspace(lb, ub, n_levels+1))
        self.uq_dist = uq_dist
        self.uq_var_l = uq_var_l
        self.uq_var_u = uq_var_u
        
    def __repr__(self):
        # Print information on the Parameter
        
        s1 = 'id:{:d} - Continous Parameter "{}"\n{:d} Levels, '.format(self.id, 
                                                                        self.name, 
                                                                        self.n_levels)
        s2 = 'Bounds: ('
        s3 = ', '.join(['{:.3f}'.format(x) for x in self.ranges])
        
        return s1 + s2 + s3 + ')\n'
    
    
    def sample(self, n_samples, level=None):  
        #Sample within a level or on the entire parameter bounds
        
        if level:
            assert level < self.n_levels, "Selected level is above total number of levels"
        
        left  = self.ranges[level] if level else self.lb
        right = self.ranges[level+1] if level else self.ub

        return np.random.uniform(left, right, n_samples)
    
    def ppf(self, quantile, x0):
        # Obtain the random values if the parameter has uncertainty
        
        # Variability bounds
        # Check if symmetric or asymmetric
        
        if np.isnan(self.uq_var_u):
            lb = x0*(1-self.uq_var_l) if x0*(1-self.uq_var_l) > self.lb else self.lb
            ub = x0*(1+self.uq_var_l) if x0*(1+self.uq_var_l) < self.ub else self.ub            
        else:
            lb = x0*(1-self.uq_var_l) if x0*(1-self.uq_var_l) > self.lb else self.lb
            ub = x0*(1+self.uq_var_u) if x0*(1+self.uq_var_u) < self.ub else self.ub

     
        if self.uq_dist == 'uniform':
            return uniform.ppf(quantile, lb, ub)
            
        elif self.uq_dist == 'triang':
            scale = ub - lb
            c = (x0 - lb)/scale
            return triang.ppf(quantile, c=c, loc=lb, scale=scale)
            
        elif self.uq_dist == 'norm':
            # Ensure it is simmetric
            A = x0*(1-self.uq_var_l) if x0*(1-self.uq_var_l) > self.lb else self.lb
            B = x0*(1+self.uq_var_l) if x0*(1+self.uq_var_l) < self.ub else self.ub
            scale = (B-A)/4
            return norm.ppf(quantile, loc=x0, scale=scale)
        else:
            #Constant deterministic value
            return x0 * np.ones(len(quantile))

class DiscreteParameter(Parameter):
    # Discrete Parameters work essentially as an ENUM
    
    
    def get_n_levels(self):
        return self.n_levels
    
    
    def __init__(self, name, n_levels):
        super().__init__(name)
        self.n_levels = n_levels
        self.ranges = tuple(range(n_levels))
    
    def __repr__(self):
        # Print information on the Parameter
        s1 = 'id:{:d} - Discrete Parameter "{}"\n{:d} Levels\n'.format(self.id, 
                                                                       self.name, 
                                                                       self.n_levels)
        return s1


class Objective(Response):
    def __init__(self, name, operand, min_requirement=None, p_sat=0.5):
        assert operand == 'max' or operand == 'min', "Objective operand is not max or min"
        super().__init__(name, operand, min_requirement, p_sat)
        
        # Bool to store if this objective has a requirement
        # for the probabilistic filter
        if min_requirement:
            self.hasRequirement = True
        else:
            self.hasRequirement = False
    
    def __repr__(self):
        s1 = f'Objective {self.name} : {self.operand}'
        
        if self.hasRequirement:
            s2 = f': Requirement: {self.get_requirement()} : P_sat = {self.p_satisfaction}'
            return s1 + s2 + '\n'
        else:
            return s1 + '\n'
    
    def get_requirement(self):
        # Get the disequation that defines the soft constraint
        if self.hasRequirement:
            if self.operand == 'min':
                return ('lt', self.value)
            else:
                return ('gt', self.value)
        else:
            return None

    def get_operand(self):
        # PyMoo is set by default to minimise, any maximisation just requires
        # the objective quantity to be flipped
        if self.operand == 'min':
            return 1
        else:
            return -1

class Constraint(Response):
    def __init__(self, name, operand, value, p_sat=0.5):
        assert operand == 'lt' \
            or operand == 'gt' \
            or operand == 'let' \
            or operand == 'get', 'Constraint operand is not lt, gt, let, get'
            
        super().__init__(name, operand, value, p_sat)
      
    def __repr__(self):
        s = f'Constraint {self.name} : {self.get_constraint()} : P_sat = {self.p_satisfaction}\n'
        return s
        
    def get_constraint(self):
        return (self.operand, self.value)
    


class DesignSet:
    _ids = count(0)
    
    def __init__(self, input_parameter_levels, response_parameters):
        self.id   = next(self._ids)
        
        self.parameter_levels_dict  = input_parameter_levels
        self.parameter_levels_list  = [input_parameter_levels[k] for k in input_parameter_levels]
        
        # List of variables
        self.response_parameters    = response_parameters
        
        self.is_discarded = False
        
        self.P = None
        self.P_responses = {} # { k : 0 for k in self.response_parameters }
        
        # Pymoo Optimisation Problem
        self.optimisation_problem   = None
        self.surrogate_optimisation_problem = None
        
        # Pymoo Data
        self.optimisation_results   = None
        self.surrogate_optimisation_results = None
        #self.pareto_points         = None #Possibly a pandas Dataframe

        # UQ Optimisation
        self.rbo_problem            = None
        self.rbo_results_raw        = None # Results with P(constraint) and f = mu + k*sigma
        self.rbo_results            = None # Post-processed RBDO results
        
        # Data on Opt
        self.opt_done      = None
        self.opt_dt        = None
        self.use_surrogate = None
        
    def __repr__(self):
        
        discarded = 'X' if self.is_discarded else ' '
        
        s1 = f'DesignSet {self.id} [{discarded}], Parameters: ' + str(self.parameter_levels_dict)
        if self.P is not None:
            return s1 + f' P = {self.P:.3f}'
        else:
            return s1
    
    
    def get_discarded_status(self):
        return self.is_discarded
    
    
    def get_P(self):
        return self.P
    
    
    def get_response_P(self, response_id=None):
        if response_id:
            return self.P_responses[response_id]
        else:
            return self.P_responses
    
    
    def set_responses_P(self, response_name, P_response):
        # Add the new response result, and updates the total probability
        self.P_responses.update({response_name : P_response})
        
        self.P = 1
        for k in self.P_responses:
            self.P *= self.P_responses[k]
        
    
    def set_as_discarded(self):
        self.is_discarded = True
    
    
    # def set_optimisation_problem(self, model, parameters, objectives, constraints):
        
    #     self.optimisation_problem = OptimisationProblem(model, 
    #                                                     parameters, 
    #                                                     objectives, 
    #                                                     constraints, 
    #                                                     self.parameter_levels_list)#,
    #                                                     #runner=pool.starmap,
    #                                                     #func_eval=starmap_parallelized_eval)
    
    def set_optimisation_problem(self, opt_problem):
        
        self.optimisation_problem = opt_problem
                                                        
    def set_surrogate_optimisation_problem    (self, model, parameters, objectives, constraints):
        
        self.surrogate_optimisation_problem = SurrogateOptimisationProblem(model, 
                                                        parameters, 
                                                        objectives, 
                                                        constraints, 
                                                        self.parameter_levels_list)
    
    def set_robust_optimisation_problem(self, model, surrogates, parameters, objectives, constraints,
                                        P_g, k_sigma, pool, decoupled_sigma=False):
        
        if decoupled_sigma:
            self.rbo_problem =  RobustOptimisationProblem_Decoupled(model, 
                                                            surrogates,   
                                                            parameters, 
                                                            objectives, 
                                                            constraints, 
                                                            self.parameter_levels_list,
                                                            runner=pool.starmap,
                                                            func_eval=starmap_parallelized_eval,
                                                            P_constraints=P_g)
        
        else:
            self.rbo_problem = RobustOptimisationProblem(model, 
                                                            surrogates,   
                                                            parameters, 
                                                            objectives, 
                                                            constraints, 
                                                            self.parameter_levels_list,
                                                            runner=pool.starmap,
                                                            func_eval=starmap_parallelized_eval,
                                                            P_constraints=P_g,
                                                            k_sigma=3)
        

    def sample(self, n_samples, parameters_list):
        # Sample a n_amount within the design space using Latin Hypercube
        samples = LatinHypercube(len(self.parameter_levels_list)).random(n_samples)
        
        # Extract the bounds for each parameter level and scale samples
        for i_par in range(len(parameters_list)):
            #tmp = samples[:,i_par]
            
            if isinstance(parameters_list[i_par], ContinousParameter):
                #Continous Parameter
                lb, ub = parameters_list[i_par].get_level_bounds(self.parameter_levels_list[i_par])
                samples[:,i_par] = lb + samples[:,i_par]*(ub - lb)
            else:
                #Discrete Parameter
                samples[:,i_par] = self.parameter_levels_list[i_par]
                
        return samples
    
    
    def get_surrogate_optimum(self):
        # Construct a Pandas Dataframe with the optimisation results
        result = {}
        
        # Check if the optimisation returned any succesful results
        if self.surrogate_optimisation_results.X is not None:
            # Reconstruct optimal input
            mask  = self.surrogate_optimisation_problem.x_mask
            var   = self.surrogate_optimisation_problem.var
            x_old = self.surrogate_optimisation_results.X
            x_new = {}
            
            i_tmp = 0
            
            for i_var in range(len(mask)):
                if mask[i_var] == 'c':
                    x_new.update({
                        var[i_var].name : x_old[:, i_var]
                        })
                else:
                    x_new.update({
                        var[i_var].name : mask[i_var] * np.ones(x_old.shape[0])
                        })
            
            result.update(x_new)
            
            # Reonstruct F, flip sign for max problems
            obj = self.surrogate_optimisation_problem.obj
            f_old = self.surrogate_optimisation_results.F
            f_new = {}
            
            for i_obj in range(len(obj)):
                sign = obj[i_obj].get_operand() #To flip sign
                
                f_new.update(
                    {obj[i_obj].name : sign * f_old[:,i_obj]}
                    )
            
            result.update(f_new)
            
            # Reconstruct G
            constr = self.surrogate_optimisation_problem.cst
            g_old = self.surrogate_optimisation_results.G
            g_new = {}
            
            for i_con in range(len(constr)):
                # Add values only if not duplicates
                if constr[i_con].name not in f_new.keys():
                    op, val = constr[i_con].get_constraint()
                    
                    if op == 'lt' or op == 'let':
                        # g < val ->  g - val = G < 0 -> g = val + G
                        g_new.update(
                            {constr[i_con].name : val + g_old[:,i_con]}
                            )
                    else:
                        # g > val -> val - g = G < 0 -> g = val - G
                        g_new.update(
                            {constr[i_con].name : val - g_old[:,i_con]}
                            )
                    
            result.update(g_new)
        
        return pd.DataFrame(result)
    
    def get_optimum(self):
        # Construct a Pandas Dataframe with the optimisation results
        # Check if the optimisation returned any succesful results
        if self.optimisation_results is not None:
            if self.use_surrogate:
                if len(self.optimisation_results) > 0:
                    col_names = [p.name for p in self.optimisation_problem.var] \
                        + [o.name for o in self.optimisation_problem.obj] \
                        + [c.name for c in self.optimisation_problem.cst]
                
                    return pd.DataFrame(self.optimisation_results,
                                        columns=col_names)
                else:
                    return None
            else:
                result = {}
                
                # Reconstruct optimal input
                mask  = self.optimisation_problem.x_mask
                var   = self.optimisation_problem.var
                x_old = self.optimisation_results.X
                x_new = {}
                
                i_tmp = 0
                
                for i_var in range(len(mask)):
                    if mask[i_var] == 'c':
                        x_new.update({
                            var[i_var].name : x_old[:, i_var]
                            })
                    else:
                        x_new.update({
                            var[i_var].name : mask[i_var] * np.ones(x_old.shape[0])
                            })
                
                result.update(x_new)
                
                # Reonstruct F, flip sign for max problems
                obj = self.optimisation_problem.obj
                f_old = self.optimisation_results.F
                f_new = {}
                
                for i_obj in range(len(obj)):
                    sign = obj[i_obj].get_operand() #To flip sign
                    
                    f_new.update(
                        {obj[i_obj].name : sign * f_old[:,i_obj]}
                        )
                
                result.update(f_new)
                
                # Reconstruct G
                constr = self.optimisation_problem.cst
                g_old = self.optimisation_results.G
                g_new = {}
                
                for i_con in range(len(constr)):
                    # Add values only if not duplicates
                    if constr[i_con].name not in f_new.keys():
                        op, val = constr[i_con].get_constraint()
                        
                        if op == 'lt' or op == 'let':
                            # g < val ->  g - val = G < 0 -> g = val + G
                            g_new.update(
                                {constr[i_con].name : val + g_old[:,i_con]}
                                )
                        else:
                            # g > val -> val - g = G < 0 -> g = val - G
                            g_new.update(
                                {constr[i_con].name : val - g_old[:,i_con]}
                                )
                        
                result.update(g_new)
                return pd.DataFrame(result)
            
        else:
            return None
    
    def get_robust_optimum(self):
        return self.rbo_results
    

class SurrogateOptimisationProblem(Problem):
    def __init__(self, model, parameters, objectives, constraints, set_levels,
                 **kwargs):
        
        self.model = model #store reference to model
        self.var = parameters
        self.obj = objectives
        self.cst = constraints
         
        # Construct the array with lower bounds and upper bounds for each 
        # input variable. Discrete variables are removed as they are fixed.
        self.x_mask = []
        self.l, self.u = [], []
        
        for i_par in range(len(self.var)):
            
            if isinstance(self.var[i_par], ContinousParameter):
                #Continous Parameter
                self.x_mask.append('c')    
                lb, ub = self.var[i_par].get_level_bounds(set_levels[i_par])
                
                self.l.append(lb)
                self.u.append(ub)
                
            else:
                # Discrete parameters are fixed within the set
                self.x_mask.append(set_levels[i_par]) 
            
        
        super().__init__(n_var=len(self.l),
                         n_obj=len(self.obj),
                         n_constr=len(self.cst),
                         xl=np.array(self.l),
                         xu=np.array(self.u),
                         elementwise_evaluation=False,
                         **kwargs)
    
        ##Sample the input to construct a local surrogate model
        n_pts = 10 * len(self.var)
        

    
    def _evaluate(self, X, out, *args, **kwargs):
        #print(f"Running with {X}")
        #sys.stdout.flush()
        #t0 = time()
        X_in = []
        F_list = []
        G_list = []
        
        
        for x in X:
            
            f_list = []
            g_list = []
            in_x = []
            
            i_tmp = 0
            
            for par in self.x_mask:
                if par == 'c':
                    in_x.append(x[i_tmp])
                    i_tmp += 1
                else:
                    in_x.append(par)
            
            #X_in.append(in_x)
            
            Y = self.model.run(*in_x)

            # Objectives must be of the minimise form
            
            
            for objective in self.obj:
                f = Y[objective.name] * objective.get_operand()
                f_list.append(f)
                
            # Constraints must be of the less than 0 form
            
            
            for constraint in self.cst:
                tmp = Y[constraint.name]
                op, val = constraint.get_constraint()
                
                # g(x) < K ->  g(x) - K < 0
                if op == 'lt' or op == 'let':
                    g = tmp - val
                
                # g(x) > K ->  0 > K - g(x)
                else:
                    g = val - tmp
                
                g_list.append(g)
            
            F_list.append(f_list)
            G_list.append(g_list)
            
        #test_out = single_run(X[0])


        #dt = time() - t0
        
        #sys.stdout.flush()
        out['F'] = np.array(F_list)
        out['G'] = np.array(G_list)
        #print(f"Finished {out}, time {dt:.2f} s")

# class LocalSurrogateOptimisationProblem(Problem):
#     def __init__(self, model, parameters, objectives, constraints, set_levels,
#                  **kwargs):
        
#         self.model = model #store reference to model
#         self.var = parameters
#         self.obj = objectives
#         self.cst = constraints
         
#         # Construct the array with lower bounds and upper bounds for each 
#         # input variable. Discrete variables are removed as they are fixed.
#         self.x_mask = []
#         self.l, self.u = [], []
        
#         for i_par in range(len(self.var)):
            
#             if isinstance(self.var[i_par], ContinousParameter):
#                 #Continous Parameter
#                 self.x_mask.append('c')    
#                 lb, ub = self.var[i_par].get_level_bounds(set_levels[i_par])
                
#                 self.l.append(lb)
#                 self.u.append(ub)
                
#             else:
#                 # Discrete parameters are fixed within the set
#                 self.x_mask.append(set_levels[i_par]) 
            
        
#         super().__init__(n_var=len(self.l),
#                          n_obj=len(self.obj),
#                          n_constr=len(self.cst),
#                          xl=np.array(self.l),
#                          xu=np.array(self.u),
#                          elementwise_evaluation=False,
#                          **kwargs)
    
#         ##Sample the input to construct a local surrogate model
#         n_pts = 10 * len(self.var)
        
        
#     def setup_pool(self, n_proc, use_dill):
#         print(f"Pool started with {n_proc} processes")
#         self.pool = WorkerPool(n_proc, use_dill=use_dill)        
        
    
#     def _evaluate(self, X, out, *args, **kwargs):
#         #print(f"Running with {X}")
#         #sys.stdout.flush()
#         #t0 = time()
#         X_in = []
#         F_list = []
#         G_list = []
        
        
#         for x in X:
            
#             f_list = []
#             g_list = []
#             in_x = []
            
#             i_tmp = 0
            
#             for par in self.x_mask:
#                 if par == 'c':
#                     in_x.append(x[i_tmp])
#                     i_tmp += 1
#                 else:
#                     in_x.append(par)
            
#             #X_in.append(in_x)
            
#             Y = self.model.run(*in_x)

#             # Objectives must be of the minimise form
            
            
#             for objective in self.obj:
#                 f = Y[objective.name] * objective.get_operand()
#                 f_list.append(f)
                
#             # Constraints must be of the less than 0 form
            
            
#             for constraint in self.cst:
#                 tmp = Y[constraint.name]
#                 op, val = constraint.get_constraint()
                
#                 # g(x) < K ->  g(x) - K < 0
#                 if op == 'lt' or op == 'let':
#                     g = tmp - val
                
#                 # g(x) > K ->  0 > K - g(x)
#                 else:
#                     g = val - tmp
                
#                 g_list.append(g)
            
#             F_list.append(f_list)
#             G_list.append(g_list)
            
#         #test_out = single_run(X[0])


#         #dt = time() - t0
        
#         #sys.stdout.flush()
#         out['F'] = np.array(F_list)
#         out['G'] = np.array(G_list)
#         #print(f"Finished {out}, time {dt:.2f} s")

# class OptimisationProblem(Problem):
#     def __init__(self, model, parameters, objectives, constraints, set_levels,
#                  **kwargs):
        
#         self.model = model #store reference to model
#         self.var = parameters
#         self.obj = objectives
#         self.cst = constraints
        
#         self.pool = None
        
#         # Construct the array with lower bounds and upper bounds for each 
#         # input variable. Discrete variables are removed as they are fixed.
#         self.x_mask = []
#         self.l, self.u = [], []
        
#         for i_par in range(len(self.var)):
            
#             if isinstance(self.var[i_par], ContinousParameter):
#                 #Continous Parameter
#                 self.x_mask.append('c')    
#                 lb, ub = self.var[i_par].get_level_bounds(set_levels[i_par])
                
#                 self.l.append(lb)
#                 self.u.append(ub)
                
#             else:
#                 # Discrete parameters are fixed within the set
#                 self.x_mask.append(set_levels[i_par]) 
            
        
#         super().__init__(n_var=len(self.l),
#                          n_obj=len(self.obj),
#                          n_constr=len(self.cst),
#                          xl=np.array(self.l),
#                          xu=np.array(self.u),
#                          elementwise_evaluation=False,
#                          **kwargs)
    
#     def setup_pool(self, n_proc, use_dill):
#         print(f"Pool started with {n_proc} processes")
#         self.pool = WorkerPool(n_proc, 
#                                use_dill=use_dill, 
#                                keep_alive=True)
    
#     # def stop_pool(self):
#     #     print("Pool terminated")
#     #     self.pool.terminate()
    
#     def _evaluate(self, X, out, *args, **kwargs):
#         #print(f"Running with {X}")
#         #sys.stdout.flush()
#         #t0 = time()
        
        
#         def single_run(x):
#             if len(x) < 2:
#                 x = x[0]
            
#             in_x = []
#             i_tmp = 0
            
#             # Reconstruct the full input with the discrete variables
#             for par in self.x_mask:
#                 if par == 'c':
#                     in_x.append(x[i_tmp])
#                     i_tmp += 1
#                 else:
#                     in_x.append(par)
            
#             Y = self.model.run(*in_x)

#             # Objectives must be of the minimise form
#             f_list = []
            
#             for objective in self.obj:
#                 f = Y[objective.name] * objective.get_operand()
#                 f_list.append(f)
                
#             # Constraints must be of the less than 0 form
#             g_list = []
            
#             for constraint in self.cst:
#                 tmp = Y[constraint.name]
#                 op, val = constraint.get_constraint()
                
#                 # g(x) < K ->  g(x) - K < 0
#                 if op == 'lt' or op == 'let':
#                     g = tmp - val
                
#                 # g(x) > K ->  0 > K - g(x)
#                 else:
#                     g = val - tmp
                
#                 g_list.append(g)
#             return f_list, g_list
        
#         #test_out = single_run(X[0])
        
#         output = self.pool.map(single_run, X, chunk_size=1,
#                                progress_bar=False)
        
#         f_list = [output[i][0] for i in range(len(output))]
#         g_list = [output[i][1] for i in range(len(output))]
        
#         #dt = time() - t0
        
#         #sys.stdout.flush()
#         out['F'] = np.array(f_list)
#         out['G'] = np.array(g_list)
#         #print(f"Finished {out}, time {dt:.2f} s")

class RobustOptimisationProblem(ElementwiseProblem):
    def __init__(self, model, surrogates, parameters, objectives, constraints, set_levels,
                 P_constraints = 0.9, k_sigma = 3, **kwargs):
        
        self.model = model #store reference to model
        self.surrogates = surrogates #store the surrogate model for UQ
        self.var = parameters
        self.obj = objectives
        self.cst = constraints
        self.P_cons = P_constraints
        self.k_sigma = k_sigma
        
        # Construct the array with lower bounds and upper bounds for each 
        # input variable. Discrete variables are removed as they are fixed.
        self.x_mask = []
        self.l, self.u = [], []
        
        for i_par in range(len(self.var)):
            
            if isinstance(self.var[i_par], ContinousParameter):
                #Continous Parameter
                self.x_mask.append('c')    
                lb, ub = self.var[i_par].get_level_bounds(set_levels[i_par])
                
                self.l.append(lb)
                self.u.append(ub)
                
            else:
                # Discrete parameters are fixed within the set
                self.x_mask.append(set_levels[i_par]) 
            
        
        super().__init__(n_var=len(self.l),
                         n_obj=len(self.obj),
                         n_constr=len(self.cst),
                         xl=np.array(self.l),
                         xu=np.array(self.u),
                         **kwargs)
    
    def _run_surrogate(self, X):
        means = []
        for name in self.surrogates:
            mu, sigma = self.surrogates[name].predict(X)
            means.append(mu)
        
        return means
    
    def _evaluate(self, x, out, *args, **kwargs):
        
        in_x = []
        i_tmp = 0
        
        for par in self.x_mask:
            if par == 'c':
                in_x.append(x[i_tmp])
                i_tmp += 1
            else:
                in_x.append(par)
        
        # Build samples
        x0 = in_x
        p  = LatinHypercube(i_tmp).random(10000)
        
        x_in = []
        i_cont = 0
        for i_par in range(len(self.var)):
            
            if isinstance(self.var[i_par], ContinousParameter):
                #Continous Parameter
                x_in.append(self.var[i_par].ppf(p[:, i_cont], x0[i_par]))
                i_cont += 1
                
            else:
                # Discrete parameters are fixed within the set
                x_in.append(x0[i_par] * np.ones(10000))

        x_in = np.vstack(x_in).T


        # Propagate samples with the surrogate model
        Y, Y_mu, Y_std = {}, {}, {}
        for sur_name in self.surrogates:
            y, _ = self.surrogates[sur_name].predict(x_in)
            Y_mu.update({sur_name : y.mean()})
            Y_std.update({sur_name: y.std()})
            Y.update({sur_name : y})
        
        # Objectives must be of the minimise form
        f_list = []
        
        for objective in self.obj:
            if objective.get_operand() < 0:
                # Maximise
                f = -Y_mu[objective.name] + self.k_sigma * Y_std[objective.name]
            else:
                # Minimise
                f = Y_mu[objective.name] + self.k_sigma * Y_std[objective.name]

            f_list.append(f)
            
            
        # Constraints must be of the less than 0 form
        g_list = []
        
        for constraint in self.cst:
            tmp = Y[constraint.name]
            op, val = constraint.get_constraint()
            
            # g(x) < K ->  g(x) - K < 0
            # P[g(x) < K] > 0.9
            if op == 'lt' or op == 'let':
                #g = tmp - val
                P = (tmp < val).sum()/len(tmp)
                g = self.P_cons - P
            # g(x) > K ->  0 > K - g(x)
            # P[g(x) > K] > 0.9
            else:
                #g = val - tmp
                P = (tmp > val).sum()/len(tmp)
                g = self.P_cons - P         
                
            g_list.append(g)
        
        out['F'] = np.array(f_list)
        out['G'] = np.array(g_list)    
    
    def postprocess(self, X):
        
        output = dict()
        
        for objective in self.obj:
            name = objective.name
            output.update({name: []})
            output.update({name + '_mu': []})
            output.update({name + '_sigma': []})
        
        for constraint in self.cst:
            name = constraint.name
            output.update({name: []})
            output.update({name + '_mu': []})
            output.update({name + '_sigma': []})
            output.update({'P_' + name: []})

        
        for x in X:
            in_x = []
            i_tmp = 0
            
            for par in self.x_mask:
                if par == 'c':
                    in_x.append(x[i_tmp])
                    i_tmp += 1
                else:
                    in_x.append(par)
            
            # Build samples
            x0 = in_x
            p  = LatinHypercube(i_tmp).random(10000)
            
            x_in = []
            i_cont = 0
            for i_par in range(len(self.var)):
                
                if isinstance(self.var[i_par], ContinousParameter):
                    #Continous Parameter
                    x_in.append(self.var[i_par].ppf(p[:, i_cont], x0[i_par]))
                    i_cont += 1
                    
                else:
                    # Discrete parameters are fixed within the set
                    x_in.append(x0[i_par] * np.ones(10000))
    
            x_in = np.vstack(x_in).T
            
    
            # Propagate samples with the surrogate model
            Y, Y_mu, Y_std = {}, {}, {}
            for sur_name in self.surrogates:
                y, _ = self.surrogates[sur_name].predict(x_in)
                Y_mu.update({sur_name : y.mean()})
                Y_std.update({sur_name: y.std()})
                Y.update({sur_name : y})
            
            # Calculate the deterministic output
            Y0 = self.model.run(*x0)
            
            # Objectives
            for objective in self.obj:
                name = objective.name
                output[name].append(Y0[name])
                output[name + '_mu'].append(Y_mu[name])
                output[name + '_sigma'].append(Y_std[name])


            # Constraints must be of the less than 0 form

            
            for constraint in self.cst:
                tmp = Y[constraint.name]
                op, val = constraint.get_constraint()
                
                # g(x) < K ->  g(x) - K < 0
                # P[g(x) < K] > 0.9
                if op == 'lt' or op == 'let':
                    #g = tmp - val
                    P = (tmp < val).sum()/len(tmp)
                # g(x) > K ->  0 > K - g(x)
                # P[g(x) > K] > 0.9
                else:
                    #g = val - tmp
                    P = (tmp > val).sum()/len(tmp)      
                
                output[constraint.name].append(Y0[constraint.name])
                output[constraint.name + '_mu'].append(Y_mu[constraint.name])
                output[constraint.name + '_sigma'].append(Y_std[constraint.name])
                output['P_' + constraint.name].append(P)

            
        return output  

class RobustOptimisationProblem_Decoupled(ElementwiseProblem):
    def __init__(self, model, surrogates, parameters, objectives, constraints, set_levels,
                 P_constraints = 0.9, **kwargs):
        
        self.model = model #store reference to model
        self.surrogates = surrogates #store the surrogate model for UQ
        self.var = parameters
        self.obj = objectives
        self.cst = constraints
        self.P_cons = P_constraints
        
        # Construct the array with lower bounds and upper bounds for each 
        # input variable. Discrete variables are removed as they are fixed.
        self.x_mask = []
        self.l, self.u = [], []
        
        for i_par in range(len(self.var)):
            
            if isinstance(self.var[i_par], ContinousParameter):
                #Continous Parameter
                self.x_mask.append('c')    
                lb, ub = self.var[i_par].get_level_bounds(set_levels[i_par])
                
                self.l.append(lb)
                self.u.append(ub)
                
            else:
                # Discrete parameters are fixed within the set
                self.x_mask.append(set_levels[i_par]) 
            
        
        super().__init__(n_var=len(self.l),
                         n_obj=2*len(self.obj),
                         n_constr=len(self.cst),
                         xl=np.array(self.l),
                         xu=np.array(self.u),
                         **kwargs)
    
    def _run_surrogate(self, X):
        means = []
        for name in self.surrogates:
            mu, sigma = self.surrogates[name].predict(X)
            means.append(mu)
        
        return means
    
    def _evaluate(self, x, out, *args, **kwargs):
        
        in_x = []
        i_tmp = 0
        
        for par in self.x_mask:
            if par == 'c':
                in_x.append(x[i_tmp])
                i_tmp += 1
            else:
                in_x.append(par)
        
        # Build samples
        x0 = in_x
        p  = LatinHypercube(i_tmp).random(10000)
        
        x_in = []
        i_cont = 0
        for i_par in range(len(self.var)):
            
            if isinstance(self.var[i_par], ContinousParameter):
                #Continous Parameter
                x_in.append(self.var[i_par].ppf(p[:, i_cont], x0[i_par]))
                i_cont += 1
                
            else:
                # Discrete parameters are fixed within the set
                x_in.append(x0[i_par] * np.ones(10000))
                
        x_in = np.vstack(x_in).T


        # Propagate samples with the surrogate model
        Y, Y_mu, Y_std = {}, {}, {}
        for sur_name in self.surrogates:
            y, _ = self.surrogates[sur_name].predict(x_in)
            Y_mu.update({sur_name : y.mean()})
            Y_std.update({sur_name: y.std()})
            Y.update({sur_name : y})
        
        # Objectives must be of the minimise form
        f_list = []
        
        # First the mu to be minimized/maximised, 
        # then its variation to be always minimised
        for objective in self.obj:
            if objective.get_operand() < 0:
                # Maximise
                f = -Y_mu[objective.name] #+ self.k_sigma * Y_std[objective.name]
            else:
                # Minimise
                f = Y_mu[objective.name] #+ self.k_sigma * Y_std[objective.name]

            f_list.append(f)
            f_list.append(Y_std[objective.name])
            
            
        # Constraints must be of the less than 0 form
        g_list = []
        
        for constraint in self.cst:
            tmp = Y[constraint.name]
            op, val = constraint.get_constraint()
            
            # g(x) < K ->  g(x) - K < 0
            # P[g(x) < K] > 0.9
            if op == 'lt' or op == 'let':
                #g = tmp - val
                P = (tmp < val).sum()/len(tmp)
                g = self.P_cons - P
            # g(x) > K ->  0 > K - g(x)
            # P[g(x) > K] > 0.9
            else:
                #g = val - tmp
                P = (tmp > val).sum()/len(tmp)
                g = self.P_cons - P         
                
            g_list.append(g)
        
        print(out)
        
        out['F'] = np.array(f_list)
        out['G'] = np.array(g_list) 
        
    def postprocess(self, X):
        
        output = dict()
        
        for objective in self.obj:
            name = objective.name
            output.update({name: []})
            output.update({name + '_mu': []})
            output.update({name + '_sigma': []})
        
        for constraint in self.cst:
            name = constraint.name
            output.update({name: []})
            output.update({name + '_mu': []})
            output.update({name + '_sigma': []})
            output.update({'P_' + name: []})

        
        for x in X:
            in_x = []
            i_tmp = 0
            
            for par in self.x_mask:
                if par == 'c':
                    in_x.append(x[i_tmp])
                    i_tmp += 1
                else:
                    in_x.append(par)
            
            # Build samples
            x0 = in_x
            p  = LatinHypercube(i_tmp).random(10000)
            
            x_in = []
            i_cont = 0
            for i_par in range(len(self.var)):
                
                if isinstance(self.var[i_par], ContinousParameter):
                    #Continous Parameter
                    x_in.append(self.var[i_par].ppf(p[:, i_cont], x0[i_par]))
                    i_cont += 1
                    
                else:
                    # Discrete parameters are fixed within the set
                    x_in.append(x0[i_par] * np.ones(10000))
    
            x_in = np.vstack(x_in).T
            
    
            # Propagate samples with the surrogate model
            Y, Y_mu, Y_std = {}, {}, {}
            for sur_name in self.surrogates:
                y, _ = self.surrogates[sur_name].predict(x_in)
                Y_mu.update({sur_name : y.mean()})
                Y_std.update({sur_name: y.std()})
                Y.update({sur_name : y})
            
            # Calculate the deterministic output
            Y0 = self.model.run(*x0)
            
            # Objectives
            for objective in self.obj:
                name = objective.name
                output[name].append(Y0[name])
                output[name + '_mu'].append(Y_mu[name])
                output[name + '_sigma'].append(Y_std[name])


            # Constraints must be of the less than 0 form

            
            for constraint in self.cst:
                tmp = Y[constraint.name]
                op, val = constraint.get_constraint()
                
                # g(x) < K ->  g(x) - K < 0
                # P[g(x) < K] > 0.9
                if op == 'lt' or op == 'let':
                    #g = tmp - val
                    P = (tmp < val).sum()/len(tmp)
                # g(x) > K ->  0 > K - g(x)
                # P[g(x) > K] > 0.9
                else:
                    #g = val - tmp
                    P = (tmp > val).sum()/len(tmp)      
                
                output[constraint.name].append(Y0[constraint.name])
                output[constraint.name + '_mu'].append(Y_mu[constraint.name])
                output[constraint.name + '_sigma'].append(Y_std[constraint.name])
                output['P_' + constraint.name].append(P)
                
        return output  

class Model:
    def __init__(self, model_fun):
        # Model function has to be designed such that
        # what it returns is the same as the response list
        self.run = model_fun
        
class DesignSpace:
    def __init__(self, csv_parameters, csv_responses): #, model):
        df_var  = pd.read_csv(csv_parameters, delimiter=',')
        df_resp = pd.read_csv(csv_responses, delimiter=',')
        
        self.parameters = []
        
        # Build the list of parameters of the Design Space
        for i, row in df_var.iterrows():
            if row.type == 'continous':
                # Add continous parameter
                self.parameters.append(ContinousParameter(row['name'], 
                                                          row.lb, 
                                                          row.ub, 
                                                          int(row.levels),
                                                          row['uq_dist'],
                                                          row['uq_var_l'],
                                                          row['uq_var_u'])
                                       )
                
            elif row.type == 'discrete':
                # Add discrete parameter
                self.parameters.append(DiscreteParameter(row['name'], 
                                                         row.levels)
                                       )
                
        
        
        self.n_par = len(self.parameters)
        self.par_names = list(df_var['name'])
    
    
        #Construct the list of Objectives and Constraints
        self.objectives  = []
        self.constraints = []
        
        for i, row in df_resp.iterrows():
            if row.type == 'objective':
                # Add objective
                req_val = None if np.isnan(row.val) else row.val
                p_sat   = 0.5 if np.isnan(row.pSat) else row.pSat
                self.objectives.append(Objective(row['name'], 
                                                 row.op,
                                                 min_requirement=req_val,
                                                 p_sat=p_sat)
                                       )
                
            else:
                # Add constraint
                p_sat   = 0.5 if np.isnan(row.pSat) else row.pSat
                self.constraints.append(Constraint(row['name'], 
                                                   row.op,
                                                   row.val,
                                                   p_sat=p_sat)
                                        )
                
        self.obj_names = [x.name for x in self.objectives]
        self.con_names = [x.name for x in self.constraints]
    
        # Build the list of DesignSets that compose the DS
        
        # Construct the list that contains each possible level
        # of each parameter
        list_of_levels = []
        for i_par in range(self.n_par):
            list_of_levels.append([x for x in range(self.parameters[i_par].n_levels)])
        
        # Generator object with all possible combinations of levels
        combinations = product(*list_of_levels)
        
        self.sets = []
        
        for c in combinations:
            self.sets.append(
                DesignSet({ self.par_names[i] : c[i] for i in range(self.n_par)}, 
                    list(set(self.con_names + self.obj_names))
                    )
                )
    
    def __repr__(self):
        
        s_out = "== Design Problem ==" + "\n" 
        
        s_out += f"\nNumber of Sets: {len(self.sets)}" 
        n_disc = sum([1 for s in self.sets if s.is_discarded])
        s_out += f"\nDiscarded Sets: {n_disc} ({int(n_disc/len(self.sets)):.2%})"
        
        is_exp_done = True if self.sets[-1].P is not None else False
        is_opt_done = True if self.sets[-1].opt_done else False
        
        s_out += f"\nExploration Completed: {is_exp_done}"
        s_out += f"\nSearch      Completed: {is_opt_done}"
        
        
        s_out += "\n\n== Input Parameters ==\n"
        for p in self.parameters:
            s_out += str(p) 
            
        s_out += "\n== Constraints ==\n"
        for p in self.constraints:
            s_out += str(p)
            
        s_out += "\n== Objectives ==\n"
        for p in self.objectives:
            s_out += str(p) 


        return s_out
        

    def get_exploration_results(self):
        # construct a dataframe with the exploration results
        tmp_table = []
        
        for Design_Set in self.sets:
            tmp_table.append(
                [Design_Set.id, int(Design_Set.is_discarded)]
                + Design_Set.parameter_levels_list
                + [Design_Set.P]
                + [Design_Set.P_responses[k] for k in Design_Set.P_responses])
        
        columns = ['set_id', 'is_discarded'] \
                + self.par_names \
                + ['P'] \
                + [f'P_{response}' for response in Design_Set.P_responses]
                

        return pd.DataFrame(tmp_table, columns=columns)
        
        
    def get_optimum_results(self):
        # construct a big dataframe with the set_id column and results
        df_list = []
        
        for i_set in range(len(self.sets)):
            design_set = self.sets[i_set]
            # Check if the set has been optimised and has valid results
            if (not design_set.is_discarded) and (design_set.optimisation_results is not None):
                temp_df = design_set.get_optimum()
                if temp_df is not None:
                    set_id = i_set * np.ones(len(temp_df))
                    temp_df.insert(0, column='set_id', value=set_id)
                    df_list.append(temp_df)
                
        return pd.concat(df_list, ignore_index=True)
    
    def get_optimum_surrogate_results(self):
        # construct a big dataframe with the surrogate set_id column and results
        df_list = []
        
        for i_set in range(len(self.sets)):
            design_set = self.sets[i_set]
            # Check if the set has been optimised and has valid results
            if (not design_set.is_discarded) and (design_set.surrogate_optimisation_results is not None) and (design_set.surrogate_optimisation_results.X is not None):
                temp_df = design_set.get_surrogate_optimum()
                set_id = i_set * np.ones(len(temp_df))
                temp_df.insert(0, column='set_id', value=set_id)
                df_list.append(temp_df)
                
        return pd.concat(df_list, ignore_index=True)
    
    def get_robust_optimum_results(self):
        # Construct a Pandas Dataframe with the robust optimisation results
        result = []
        for i, Set in enumerate(self.sets):
            if Set.rbo_results is not None:
                tmp_df = Set.rbo_results
                cols = tmp_df.columns
                
                tmp_df['set_id'] = i
                tmp_df = tmp_df[['set_id'] + list(cols)[:-1]]
                result.append(tmp_df)
        
        out_col = tmp_df.columns
        df_out = pd.concat(result).reset_index()[out_col]
        
        return df_out #result #pd.concat(result)
    
    def set_discard_status(self, set_id, status):
        # User input for changing the discard status.
        self.sets[set_id].is_discarded = status
    
# Direct Code if imported

# Reset the ID counters of the Designs

if __name__ == "__main__":
    print("pdopt.Data")
