# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 08:23:35 2024

@author: s345001
"""

# Standard Library Imports
from time import time
from itertools import count, product
from os.path import exists
from warnings import warn
from math import ceil
import pickle as pk
import contextlib

# Third-party imports
import numpy as np
import pandas as pd

from scipy.stats.qmc import LatinHypercube, Sobol
from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import (
    Matern,
    ConstantKernel,
    RBF,
    RationalQuadratic,
)
from sklearn.preprocessing import MinMaxScaler

import joblib
from joblib import Parallel, delayed
from tqdm import tqdm


import pdopt.data as data 
import pdopt.exploration as exploration
from pdopt.exploration import generate_input_samples, generate_surrogate_test_data, generate_surrogate_training_data

from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.utils import get_example_model
from pgmpy.estimators import BayesianEstimator

#from pdopt.data import DesignSpace, ExtendableModel, ContinousParameter

## Discretiser based on the discretisation of design space
## Structure definition and parameter names

par1 = data.ContinousParameter("par1", 0, 1, 4, None, None, None)
par2 = data.DiscreteParameter("par2", 4)

obj1 = data.Objective("obj", "min")
con1 = data.Constraint("con1", "lt", 1)
con2 = data.Constraint("con2", "gt", 0)
con3 = data.Constraint("con3", "lt", 0.5, uq_dist='uniform',
                       uq_var_l=0.1, uq_var_u=0.1)


my_design_set = data.DesignSet(
    {"par1": 0, "par2": 0}, {"obj": obj1, "con1": con1, "con2": con2,
                             "con3" : con3}
)


def discretize(data, cardinality, labels=dict(), precision=5):
    df_copy = data.copy()
    for column in cardinality.keys():
        df_copy[column] = pd.cut(
            df_copy[column],
            bins=cardinality[column],
            include_lowest=True,
            labels=labels.get(column),
            precision=precision
        )
    return df_copy

# Utilities

def get_interval(df, column_name, numerical_value):
    categories = df[column_name].cat.categories
    for interval in categories:
        if numerical_value in interval:
            return interval
    return min(categories, key=lambda iv: min(abs(numerical_value - iv.left), abs(numerical_value - iv.right)))

def get_state_name(interval, precision=4):
    return f"{round(interval.left, precision)}-{round(interval.right, precision)}"

def get_separators(interval_list):
    separators = [interval.left for interval in interval_list]
    separators.append(interval_list[-1].right)
    return separators

def query2df(query):
    import pandas as pd
    state_combinations = pd.MultiIndex.from_product(
        [query.state_names[var] for var in query.variables], names=query.variables
    )
    return pd.DataFrame({"probability": query.values.flatten()}, index=state_combinations).reset_index()

test_DS = data.DesignSpace([par1, par2], [obj1], [con1, con2, con3])


def test_fun(par1, par2):
    obj = par1**2 - par1*5 + par2
    con1 = par1*3 - par2
    con2 = par1*par2*0.01
    con3 = par1*2 - par2*0.5
    
    return {"obj": obj, "con1": con1, "con2": con2, "con3": con3}


my_model = data.Model(test_fun)

test_inp = exploration.generate_input_samples(100, [par1, par2])
test_out = [my_model.run(*inputs) for inputs in test_inp]

obj_A  = [val['obj'] for val in test_out]
con1_A = [val['con1'] for val in test_out]
con2_A = [val['con2'] for val in test_out] 



## object for discretisation
class UniformDiscretiser1D:
    def __init__(self, data, n_levels, decimals=2):
        
        self.n_levels = n_levels
        x = data.flatten()
        
        self.level_width  = (x.max() - x.min())/self.n_levels
        self.split_points = [
                x.min() + self.level_width * (n + 1) for n in range(self.n_levels - 1)
            ]
        
        self.ranges = [data.min()] + list(self.split_points) + [data.max()]
        self.range_map = { i-1 : f'[{i-1}]:({self.ranges[i-1]:.{decimals}f}-{self.ranges[i]:.{decimals}f})'  for i in range(1,len(self.ranges))}

    
    def transform(self, data):
        return np.digitize(data, self.split_points)
    
    def transform_map(self, data):
        tf_data  = self.transform(data)
        map_data = np.vectorize(self.range_map.get)(tf_data)
        
        return map_data
        
    def map_2_range(self, value):
        range_idx =  list(self.range_map.keys())[list(self.range_map.values()).index(value)]
        return (self.ranges[range_idx], self.ranges[range_idx+1])
 
        
# ## object for mapping 
# class BN_data_transform:
#     def __init__(self, data):
#         pass
    
#     def 
    
class BayesianNetworkModel:
    def __init__(self, data: pd.DataFrame, structure: list):
        self.data = data
        self.structure = structure
        self.model = DiscreteBayesianNetwork(self.structure)
        self.infer = None

    def fit(self, prior_type="K2"):
        self.model.fit(self.data, estimator=BayesianEstimator, prior_type=prior_type)
        self.infer = VariableElimination(self.model)

    def query(self, variables, evidence=None, joint=True):
        return self.infer.query(variables, evidence=evidence, joint=joint)

    def get_model(self):
        return self.model


## BN Representation Class
earthquake = get_example_model('earthquake')
samples = earthquake.simulate(n_samples=100)

class BayesianNetworkModel:
    def __init__(self, data: pd.DataFrame, structure: list):
        self.data = data
        self.structure = structure
        self.model = DiscreteBayesianNetwork(self.structure)
        self.infer = None

    def fit(self, prior_type="K2"):
        self.model.fit(self.data, estimator=BayesianEstimator, prior_type=prior_type)
        self.infer = VariableElimination(self.model)

    def query(self, variables, evidence=None, joint=True):
        return self.infer.query(variables, evidence=evidence, joint=joint)

    def get_model(self):
        return self.model


class BN_Exploration:
    def __init__(self, design_space, model, 
                 surrogate_training_data_file, 
                 surrogate_testing_data_file=None,
                n_train_points=120, debug=False):
        
        self.design_space = design_space
        self.parameters = design_space.parameters
        self.objectives = design_space.objectives
        self.constraints = design_space.constraints
        self.model = model

        self.debug = debug
        self.run_time = 0
        
        
        
        # Perform the generation of test data
        self.__doe_train_test_data(
            n_train_points, surrogate_training_data_file, surrogate_testing_data_file
        )

        # Train the GPRs if augmentation is introduced
        # self.__surrogates_training()

        # Build data structure with responses and their operands
        # required for the PDOPT space exploration tool
        self.responses = {}

        for objective in self.objectives:
            self.responses.update({objective.name: objective.operand})

        for constraint in self.constraints:
            self.responses.update({constraint.name: constraint.operand})
        
        # Train the Bayesnet
        
        # if the structure is provided use it, otherwise assume full
        # connectivity between inputs/outputs
        
        if self.design_space.graph is None:
            # No pre-set graph, generate one from inp to outputs
            
            graph = []
            
            # Paramters to objectives
            for obj in self.objectives:
                for inp in self.parameters:
                    # Tuples go from parent -> child nodes
                    edge = (inp.name, obj.name)
                    graph.append(edge)
            
            # Parameters to contraints
            for con in self.constraints:
                
                # Check if constraint has uncertainty or not
                if con.uq_dist is None or np.isnan(con.uq_var_u):
                    # Deterministic constraint
                    for inp in self.parameters:
                        # Tuples go from parent -> child nodes
                        edge = (inp.name, f"R_{con.name}")
                        graph.append(edge)    
                else:
                    # Constraint with uncertainty
                    for inp in self.parameters:
                        # Tuples go from parent -> child nodes
                        edge = (inp.name, f"R_{con.name}")
                        graph.append(edge)   
                    
                    #Constraint value edge
                    edge = (f"cv_{con.name}", f"R_{con.name}")
                    graph.append(edge)  
                
            self.design_space.graph = graph

        else:
            # This requires from the user to build a connectivity matrix
            # this would be best expressed as a .csv file with 1 and 0s
            # between all the variables at play

            pass
        # Generate the train/test data
        # If augmentation is true, train GPRs
        # Train the BN model
        
        # Data discretisation for training the BN
        # Start with the input parameters
        
        discretisation_dict = {}
        
        for inp in self.parameters:
            if type(inp) is data.ContinousParameter:
                discretisation_dict.update({inp.name : inp.n_levels})
                
        for obj in self.objectives:
            discretisation_dict.update({obj.name : obj.n_levels})
            #discretise them
            
        for con in self.constraints:
            #discretise them
            
            if con.uq_dist is None or np.isnan(con.uq_var_u):
                # Deterministic constraint, only boolean
                
                if con.get_constraint()[0] == 'lt':
                    self.surrogate_train_data[f"R_{con.name}"] = (self.surrogate_train_data[con.name] < con.get_constraint()[1]).astype(int)
                else:
                    self.surrogate_train_data[f"R_{con.name}"] = (self.surrogate_train_data[con.name] > con.get_constraint()[1]).astype(int)  
                    
            else:
                # Constraint with uncertainty
                # discretisation_dict.update({f"R_{con.name}" : con.n_levels})
                
                discretisation_dict.update({f"cv_{con.name}" : con.n_levels})
                self.surrogate_train_data[f"cv_{con.name}"] = con.sample_cv(len(self.surrogate_train_data))
                
                if con.get_constraint()[0] == 'lt':
                    self.surrogate_train_data[f"R_{con.name}"] = (self.surrogate_train_data[con.name] < self.surrogate_train_data[f"cv_{con.name}"]).astype(int)
                else:
                    self.surrogate_train_data[f"R_{con.name}"] = (self.surrogate_train_data[con.name] > self.surrogate_train_data[f"cv_{con.name}"]).astype(int)  
                   

        self.bn_nodes = list(set([x for xs in self.design_space.graph for x in xs]))
        self.bn_dataset = self.surrogate_train_data[self.bn_nodes]
    
        self.bn_dataset = discretize(self.bn_dataset, discretisation_dict)
        
        # Train the BN
        self.bn_model = BayesianNetworkModel(self.bn_dataset, self.design_space.graph)
        self.bn_model.fit()
    
    
    def __doe_train_test_data(
        self, n_train_points, surrogate_training_data_file, surrogate_testing_data_file
    ):
        # Perform the creation of train and test data or load from file
        # Load Samples or Generate Samples
        if surrogate_training_data_file and exists(surrogate_training_data_file):
            self.surrogate_train_data = pd.read_csv(surrogate_training_data_file)
        else:
            self.surrogate_train_data = generate_surrogate_training_data(
                self.parameters,                self.model,
                n_train_points,
                save_dir=surrogate_training_data_file,
                debug=self.debug,
            )

        if surrogate_testing_data_file and exists(surrogate_training_data_file):
            self.surrogate_test_data = pd.read_csv(surrogate_testing_data_file)
        else:
            self.surrogate_test_data = generate_surrogate_test_data(
                30, self.parameters, self.model, debug=self.debug
            )

    def run(self, variables=None, evidence=None,  p_discard=0.5):
        
        # Perform the BN pass marking each design space value
        
        if not variables:
            # build the standard query: all inputs
 
            variables = [par.name for par in self.parameters]
        
        if not evidence:
            # standard set of evidence: satisfy all requirements
            evidence = {}
            for con in self.constraints:
                evidence.update({f"R_{con.name}" : 1})
        
        result = self.bn_model.query(variables, evidence=evidence)
        discard_threshold = np.quantile(result.values, p_discard)
        # pass every combination of the INP to find the highest values.
        # get all the input 
        # add the option for rapid filtering perhaps
        
        # Run the Probabilistic design exploration and evaluate sets.
        t0 = time()
        for design_set in tqdm(
            self.design_space.sets, desc="Exploring the Design Space"
        ):
            vals = design_set.parameter_levels_dict
            
            for var in variables:
                vals[var] = self.bn_model.model.states[var][vals[var]]
                
                
            design_set.P = result.get_value(**vals)
            
            # Discard the set if the total probability is lower than the
            # specified one
            if design_set.P < discard_threshold:
                design_set.set_as_discarded()
        self.run_time = time() - t0
        
#testBN = BN_Exploration(test_DS, my_model, 'asd.csv',None)
#testBN.run()