# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:35:33 2024

@author: Andrea Spinelli
"""

import pytest
import pdopt.data as data
import pdopt.exploration as exploration
import numpy as np
import pandas as pd


@pytest.fixture
def my_DesignSpace():
    return data.DesignSpace('test_input.csv', 'test_response.csv')

def dummy_function(x, y):
    return {'obj'  : x+y,
            'con1' : y*6,
            'con2' : x*2}

@pytest.fixture
def my_Model():
    return data.Model(dummy_function)


## Test generate input samples
@pytest.mark.parametrize('rule, expected_output',
                          [('lhs', np.array([[3.75859889, 0.        ],
                                             [1.31815468, 1.        ],
                                             [7.53810097, 1.        ],
                                             [8.28743567, 0.        ]])),
                           ('sobol', np.array([[4.87926527, 1.        ],
                                               [9.65420914, 0.        ],
                                               [6.08078728, 1.        ],
                                               [1.31542589, 0.        ]])),
                           ('grid', np.array([[ 1.,  0.],
                                                   [ 1.,  1.],
                                                   [10.,  0.],
                                                   [10.,  1.]]))])
def test_generate_input_samples(my_DesignSpace, rule, expected_output):
    
    out = exploration.generate_input_samples(4, my_DesignSpace.parameters, 
                                       rule=rule, debug=True)
    
    assert (expected_output.round(4) == out.round(4)).all()
    

## Test training data generation
def test_generate_surrogate_training_data(my_DesignSpace, my_Model):
    
    out = exploration.generate_surrogate_training_data(my_DesignSpace.parameters, 
                                                   my_Model, 4, debug=True)
    
    par1 = np.array([4.87926527, 9.65420914, 6.08078728, 1.31542589])
    par2 = np.array([1., 0., 1., 0.])
    
    # Check all columns of the output dataframe
    
    assert (out['par1'].to_numpy().round(4) == par1.round(4)).all()
    assert (out['par2'].to_numpy().round(4) == par2.round(4)).all()
    assert (out['obj'].to_numpy().round(4) == (par1 + par2).round(4)).all()
    assert (out['con1'].to_numpy().round(4) == (par2 * 6).round(4)).all()
    assert (out['con2'].to_numpy().round(4) == (par1 * 2).round(4)).all()
    

## Test testing data generation
def test_generate_surrogate_test_data(my_DesignSpace, my_Model):
    out = exploration.generate_surrogate_test_data(4, my_DesignSpace.parameters, 
                                                   my_Model, debug=True)
    
    par1 = np.array([3.75859889, 1.31815468, 7.53810097, 8.28743567])
    par2 = np.array([0., 1., 1., 0.])
    
    # Check all columns of the output dataframe
    
    assert (out['par1'].to_numpy().round(4) == par1.round(4)).all()
    assert (out['par2'].to_numpy().round(4) == par2.round(4)).all()
    assert (out['obj'].to_numpy().round(4) == (par1 + par2).round(4)).all()
    assert (out['con1'].to_numpy().round(4) == (par2 * 6).round(4)).all()
    assert (out['con2'].to_numpy().round(4) == (par1 * 2).round(4)).all()
    
## Test surrogate response
@pytest.mark.parametrize('response',['obj','con1','con2'])
def test_SurrogateResponse_predict(my_DesignSpace, my_Model, response):
    my_SurrogateResponse = exploration.SurrogateResponse(response, 
                                                         my_DesignSpace.parameters, 
                                                         my_Model)

    samples = exploration.generate_input_samples(4, my_DesignSpace.parameters, 
                                                 'lhs', debug=True)
    
    mu, sigma = my_SurrogateResponse.predict(samples)
    y_exact   = [my_Model.run(*samples[i,:])[response] for i in range(len(samples))]
    
    # Check that the prediction is actually correct
    assert (abs(y_exact - mu) < 2*sigma).all()



## Test probabilistic exploration
@pytest.fixture
def my_ProbabilisticExploration(my_DesignSpace, my_Model):
    return exploration.ProbabilisticExploration(my_DesignSpace, 
                                                my_Model,
                                                debug=True)

