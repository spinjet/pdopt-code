# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 08:23:35 2024

@author: s345001
"""

import pdopt.data as data 
import pdopt.exploration as exploration
import pandas as pd
import numpy as np

from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.models import BayesianNetwork
from pgmpy.utils import get_example_model
from pgmpy.estimators import MaximumLikelihoodEstimator

#from pdopt.data import DesignSpace, ExtendableModel, ContinousParameter

## Discretiser based on the discretisation of design space
## Structure definition and parameter names

par1 = data.ContinousParameter("par1", 0, 1, 4, None, None, None)
par2 = data.DiscreteParameter("par2", 4)

obj = data.Objective("obj", "min")
con1 = data.Constraint("con1", "lt", 1)
con2 = data.Constraint("con2", "gt", 0.1)

my_design_set = data.DesignSet(
    {"par1": 0, "par2": 0}, {"obj": obj, "con1": con1, "con2": con2}
)


def test_fun(par1, par2):
    obj = par1**2 - par1*5 + par2
    con1 = par1*3 - par2
    con2 = par1*par2*0.01
    
    return {"obj": obj, "con1": con1, "con2": con2}


my_model = data.Model(test_fun)

test_inp = exploration.generate_input_samples(100, [par1, par2])
test_out = [my_model.run(*inputs) for inputs in test_inp]

obj  = [val['obj'] for val in test_out]
con1 = [val['con1'] for val in test_out]
con2 = [val['con2'] for val in test_out] 

## object for discretisation
class UniformDiscretiser1D:
    def __init__(self, data, n_levels):
        
        self.n_levels = n_levels
        x = data.flatten()
        
        self.level_width  = (x.max() - x.min())/self.n_levels
        self.split_points = [
                x.min() + self.level_width * (n + 1) for n in range(self.n_levels - 1)
            ]
    
    def transform(self, data):
        return np.digitize(data, self.split_points)
        
        
## object for mapping 
    
    



## BN Representation Class
earthquake = get_example_model('earthquake')
samples = earthquake.simulate(n_samples=100)

class BayesianNetworkModel:
    def __init__(self, structure_dict):
        self._bn = BayesianNetwork(structure_dict)
        self.nodes = self._bn.nodes()
        self.edges = self._bn.edges()
    
    def fit_sates(self, df_data):
        pass
    

# Initialize the estimator object.
bn = BayesianNetwork(earthquake.edges())
mle_est = MaximumLikelihoodEstimator(model=bn, data=samples)
all_cpds = mle_est.get_parameters(n_jobs=1)


## Extension of Parameter classes to support BN sampling/discretisation

## Exploration class using BN

## 