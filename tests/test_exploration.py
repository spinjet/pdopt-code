# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:35:33 2024

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
    return {'obj'  : (x-5)**2 - y*5,
            'con1' : (x-5)**2 - y*5,
            'con2' :  11 - 0.5*x**2}

@pytest.fixture
def my_Model():
    return data.Model(dummy_function)


## Test generate input samples
@pytest.mark.parametrize('rule, expected_output',
                          [('lhs', np.array([[3.06510988, 0.        ],
                                             [0.3535052 , 1.        ],
                                             [7.26455663, 1.        ],
                                             [8.09715075, 0.        ]])),
                           ('sobol', np.array([[4.31029474, 1.        ],
                                              [9.61578793, 0.        ],
                                              [5.6453192 , 1.        ],
                                              [0.35047322, 0.        ]])),
                           ('grid', np.array([[ 0.,  0.],
                                              [ 0.,  1.],
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
    
    par1 = np.array([4.31029474, 9.61578793, 5.6453192 , 0.35047322])
    par2 = np.array([1., 0., 1., 0.])
    
    # Check all columns of the output dataframe
    
    assert (out['par1'].to_numpy().round(4) == par1.round(4)).all()
    assert (out['par2'].to_numpy().round(4) == par2.round(4)).all()
    assert (out['obj'].to_numpy().round(4) == ((par1 - 5)**2 - par2*5).round(4)).all()
    assert (out['con1'].to_numpy().round(4) == ((par1 - 5)**2 - par2*5).round(4)).all()
    assert (out['con2'].to_numpy().round(4) == (11 - 0.5*par1**2 ).round(4)).all()
    

## Test testing data generation
def test_generate_surrogate_test_data(my_DesignSpace, my_Model):
    out = exploration.generate_surrogate_test_data(4, my_DesignSpace.parameters, 
                                                   my_Model, debug=True)
    
    par1 = np.array([3.06510988, 0.3535052 , 7.26455663, 8.09715075])
    par2 = np.array([0., 1., 1., 0.])
    
    # Check all columns of the output dataframe
    
    assert (out['par1'].to_numpy().round(4) == par1.round(4)).all()
    assert (out['par2'].to_numpy().round(4) == par2.round(4)).all()
    assert (out['obj'].to_numpy().round(4) == ((par1 - 5)**2 - par2*5).round(4)).all()
    assert (out['con1'].to_numpy().round(4) == ((par1 - 5)**2 - par2*5).round(4)).all()
    assert (out['con2'].to_numpy().round(4) == (11 - 0.5*par1**2 ).round(4)).all()
    

## Test surrogate response
@pytest.mark.filterwarnings("ignore: lbfgs failed")
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

@pytest.mark.filterwarnings("ignore: lbfgs failed")
def test_my_ProbabilisticExploration_on_creation(my_DesignSpace, my_Model):
    # Check if it is created without any errors and the internal methods 
    # work without failing.
    my_ProbabilisticExploration = exploration.ProbabilisticExploration(my_DesignSpace, 
                                                                       my_Model,
                                                                       debug=True) 
    
    expected_train_data = pd.read_csv('test_train_exploration_data.csv').to_numpy().round(5)
    expected_test_data  = pd.read_csv('test_validation_exploration_data.csv').to_numpy().round(5)
    
    ## This checks the __doe_train_test_data() method
    assert (my_ProbabilisticExploration.surrogate_train_data.to_numpy().round(5) == expected_train_data).all()
    assert (my_ProbabilisticExploration.surrogate_test_data.to_numpy().round(5) == expected_test_data).all()

    ## This checks the __surrogates_training() method
    ## mainly that it built the correct requirements
    
    expected_requirements = {'obj': ('lt', 5, 0.25), 
                             'con1': ('gt', 0, 0.5), 
                             'con2': ('lt', 10, 0.5)}
    
    assert my_ProbabilisticExploration.requirements == expected_requirements
    
    expected_surrogates = ['obj', 'con2', 'con1']
    
    assert all([ surrogate in my_ProbabilisticExploration.surrogates.keys() \
                for surrogate in expected_surrogates])
    

@pytest.mark.filterwarnings("ignore: lbfgs failed")
def test_ProbabilisticExploration_run(my_DesignSpace, my_Model):
    my_ProbabilisticExploration = exploration.ProbabilisticExploration(my_DesignSpace, 
                                                                       my_Model,
                                                                       debug=True)
    # Run the probabilistic exploration
    my_ProbabilisticExploration.run(debug=True)
    
    # Compare the calculated probabilities of each set with the expected values
    
    exp_discarded = np.array([1, 1, 0, 1, 0, 1, 1, 1], dtype='int64')
    exp_P      = np.array([0.    , 0.1161, 0.89  , 0.11  , 0.9   , 0.1   , 0.    , 0.26  ])
    exp_P_obj  = np.array([0.  , 0.27, 0.89, 1.  , 0.9 , 1.  , 0.  , 0.26])
    exp_P_con1 = np.array([1.  , 1.  , 1.  , 0.11, 1.  , 0.1 , 1.  , 1.  ])
    exp_P_con2 = np.array([0.43, 0.43, 1.  , 1.  , 1.  , 1.  , 1.  , 1.  ])
    
    out = my_DesignSpace.get_exploration_results()
    
    assert (out['is_discarded'].to_numpy() == exp_discarded).all()
    assert (out['P'].to_numpy().round(4)      ==  exp_P.round(4)).all()
    assert (out['P_obj'].to_numpy().round(4)  ==  exp_P_obj.round(4)).all()
    assert (out['P_con1'].to_numpy().round(4) ==  exp_P_con1.round(4)).all()
    assert (out['P_con2'].to_numpy().round(4) ==  exp_P_con2.round(4)).all()
    
    
    