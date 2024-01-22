# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:12:31 2024
"""

import pytest
import pdopt.data as data
import pdopt.exploration as exploration
import pdopt.optimisation as optimisation
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


## Test NNSurrogate
@pytest.fixture
def my_NNSurrogate(my_Model, my_DesignSpace):
    return optimisation.NNSurrogate(my_Model, my_DesignSpace, 
                                              set_id=0, debug=True)

## The debug flag fixes all the random states
def test_NNSUrrogate_sample(my_NNSurrogate):
    # Check that it correctly sampled the same things
    my_NNSurrogate.sample()
    
    expected_X_test = np.array([[0.66897452, 1.        ],
       [0.1876068 , 1.        ],
       [0.91404481, 1.        ],
       [0.50982793, 0.        ],
       [0.08327291, 1.        ],
       [0.26699526, 0.        ],
       [0.7688592 , 0.        ],
       [0.13703354, 0.        ],
       [0.48450821, 0.        ],
       [0.63535057, 0.        ],
       [0.09508085, 0.        ],
       [0.94989628, 1.        ],
       [0.15825541, 1.        ],
       [0.35966759, 1.        ],
       [0.01861538, 0.        ],
       [0.79755058, 0.        ],
       [0.3003415 , 1.        ],
       [0.55104381, 1.        ],
       [0.0513041 , 1.        ],
       [0.7183425 , 0.        ],
       [0.93501122, 1.        ],
       [0.7351483 , 0.        ],
       [0.37991396, 1.        ],
       [0.45792965, 1.        ],
       [0.57615978, 0.        ],
       [0.33627077, 0.        ],
       [0.60473845, 1.        ],
       [0.86263638, 1.        ],
       [0.42373889, 0.        ],
       [1.00832237, 0.        ],
       [0.8318984 , 0.        ],
       [0.22887941, 1.        ]])
    
    expected_X_train = np.array([[0.43177599, 1.        ],
           [0.97354942, 0.        ],
           [0.56810277, 1.        ],
           [0.02741659, 0.        ],
           [0.23473156, 1.        ],
           [0.64798802, 0.        ],
           [0.76601971, 1.        ],
           [0.35210548, 0.        ],
           [0.25348936, 1.        ],
           [0.86058461, 0.        ],
           [0.75536051, 1.        ],
           [0.14736584, 0.        ],
           [0.06747669, 1.        ],
           [0.54804943, 0.        ],
           [0.94025123, 1.        ],
           [0.46102296, 0.        ],
           [0.48315498, 1.        ],
           [0.89817551, 0.        ],
           [0.52582001, 1.        ],
           [0.10964979, 0.        ],
           [0.15765488, 1.        ],
           [0.70119249, 0.        ],
           [0.85044749, 1.        ],
           [0.30750544, 0.        ],
           [0.32986534, 1.        ],
           [0.80820547, 0.        ],
           [0.67013855, 1.        ],
           [0.19263538, 0.        ],
           [0.0172668 , 1.        ],
           [0.62212957, 0.        ],
           [0.98385891, 1.        ],
           [0.37758947, 0.        ],
           [0.3933585 , 1.        ],
           [1.        , 0.        ],
           [0.60748626, 1.        ],
           [0.        , 0.        ],
           [0.20728356, 1.        ],
           [0.68740243, 0.        ],
           [0.79243938, 1.        ],
           [0.31371939, 0.        ],
           [0.2918581 , 1.        ],
           [0.83418279, 0.        ],
           [0.71596332, 1.        ],
           [0.17479612, 0.        ],
           [0.09487605, 1.        ],
           [0.50868365, 0.        ],
           [0.913818  , 1.        ],
           [0.49942262, 0.        ],
           [0.44575054, 1.        ],
           [0.92361354, 0.        ],
           [0.56219601, 1.        ],
           [0.08524021, 0.        ],
           [0.13321438, 1.        ],
           [0.7375999 , 0.        ],
           [0.87585411, 1.        ],
           [0.27013192, 0.        ],
           [0.3672522 , 1.        ],
           [0.78278504, 0.        ],
           [0.63371769, 1.        ],
           [0.21708982, 0.        ],
           [0.04168983, 1.        ],
           [0.58573963, 0.        ],
           [0.95840755, 1.        ],
           [0.41500774, 0.        ]])
    
    assert (my_NNSurrogate.X_train.round(6) == expected_X_train.round(6)).all()
    assert (my_NNSurrogate.X_test.round(6)  == expected_X_test.round(6)).all()


def test_NNSurrogate_train_model_predict(my_NNSurrogate):
    my_NNSurrogate.sample()
    my_NNSurrogate.train_model()
    
    #Test the prediction    
    expected_output = {'F': np.array([[22.396518358255157],
                                       [19.882521893338144],
                                       [17.810392630436965]]),
                     'G': np.array([[-22.78313438982879  ,   0.9745664293583953],
                                    [-20.197853335738785 ,   0.8558771222671897],
                                    [-18.054481467775297 ,   0.7055509759847869]])}
    
    x_in     = np.array([[0.25, 0],
                         [0.5,  0],
                         [0.75, 0]])
    out_dict = {}
    
    my_NNSurrogate._evaluate(x_in, out_dict)
    
    # There is some small numerical error that I cannot control caused by the
    # type of machine used for testing by GitHub. Hence assert is written 
    # to assert this precision
    assert (abs(out_dict['F'] - expected_output['F']) < 1e-4).all()
    assert (abs(out_dict['G'] - expected_output['G']) < 1e-4).all()


def test_NSSurrogate_reconstruct_recover_pts(my_NNSurrogate):
    my_NNSurrogate.sample()
    my_NNSurrogate.train_model()
    
    x_in     = np.array([[0.25, 0],
                         [0.5,  0],
                         [0.75, 0]])

    expected_output = np.array([[ 0.25000,  0.00000, 22.56250, 22.56250, 10.96875],
                               [ 0.50000,  0.00000, 20.25000, 20.25000, 10.87500],
                               [ 0.75000,  0.00000, 18.06250, 18.06250, 10.71875]])
                            
    out = my_NNSurrogate.recover_pts(x_in)
    
    assert (out.round(6) == expected_output.round(6)).all()
    

## Test KrigingSurrogate
@pytest.fixture
def my_KrigingSurrogate(my_Model, my_DesignSpace):
    return optimisation.KrigingSurrogate(my_Model, my_DesignSpace, 
                                              set_id=0, kernel='rbf',
                                              debug=True)

## The debug flag fixes all the random states
def test_KrigingSurrogate_sample(my_KrigingSurrogate):
    # Check that it correctly sampled the same things
    my_KrigingSurrogate.sample()
    
    expected_X_test = np.array([[0.66897452, 1.        ],
                               [0.1876068 , 1.        ],
                               [0.91404481, 1.        ],
                               [0.50982793, 0.        ],
                               [0.08327291, 1.        ],
                               [0.26699526, 0.        ],
                               [0.7688592 , 0.        ],
                               [0.13703354, 0.        ],
                               [0.48450821, 0.        ],
                               [0.63535057, 0.        ],
                               [0.09508085, 0.        ],
                               [0.94989628, 1.        ],
                               [0.15825541, 1.        ],
                               [0.35966759, 1.        ],
                               [0.01861538, 0.        ],
                               [0.79755058, 0.        ],
                               [0.3003415 , 1.        ],
                               [0.55104381, 1.        ],
                               [0.0513041 , 1.        ],
                               [0.7183425 , 0.        ],
                               [0.93501122, 1.        ],
                               [0.7351483 , 0.        ],
                               [0.37991396, 1.        ],
                               [0.45792965, 1.        ],
                               [0.57615978, 0.        ],
                               [0.33627077, 0.        ],
                               [0.60473845, 1.        ],
                               [0.86263638, 1.        ],
                               [0.42373889, 0.        ],
                               [1.00832237, 0.        ],
                               [0.8318984 , 0.        ],
                               [0.22887941, 1.        ]])
    
    expected_X_train = np.array([[0.43177599, 1.        ],
                               [0.97354942, 0.        ],
                               [0.56810277, 1.        ],
                               [0.02741659, 0.        ],
                               [0.23473156, 1.        ],
                               [0.64798802, 0.        ],
                               [0.76601971, 1.        ],
                               [0.35210548, 0.        ],
                               [0.25348936, 1.        ],
                               [0.86058461, 0.        ],
                               [0.75536051, 1.        ],
                               [0.14736584, 0.        ],
                               [0.06747669, 1.        ],
                               [0.54804943, 0.        ],
                               [0.94025123, 1.        ],
                               [0.46102296, 0.        ],
                               [0.48315498, 1.        ],
                               [0.89817551, 0.        ],
                               [0.52582001, 1.        ],
                               [0.10964979, 0.        ],
                               [0.15765488, 1.        ],
                               [0.70119249, 0.        ],
                               [0.85044749, 1.        ],
                               [0.30750544, 0.        ],
                               [0.32986534, 1.        ],
                               [0.80820547, 0.        ],
                               [0.67013855, 1.        ],
                               [0.19263538, 0.        ],
                               [0.0172668 , 1.        ],
                               [0.62212957, 0.        ],
                               [0.98385891, 1.        ],
                               [0.37758947, 0.        ],
                               [0.3933585 , 1.        ],
                               [1.        , 0.        ],
                               [0.60748626, 1.        ],
                               [0.        , 0.        ],
                               [0.20728356, 1.        ],
                               [0.68740243, 0.        ],
                               [0.79243938, 1.        ],
                               [0.31371939, 0.        ],
                               [0.2918581 , 1.        ],
                               [0.83418279, 0.        ],
                               [0.71596332, 1.        ],
                               [0.17479612, 0.        ],
                               [0.09487605, 1.        ],
                               [0.50868365, 0.        ],
                               [0.913818  , 1.        ],
                               [0.49942262, 0.        ],
                               [0.44575054, 1.        ],
                               [0.92361354, 0.        ],
                               [0.56219601, 1.        ],
                               [0.08524021, 0.        ],
                               [0.13321438, 1.        ],
                               [0.7375999 , 0.        ],
                               [0.87585411, 1.        ],
                               [0.27013192, 0.        ],
                               [0.3672522 , 1.        ],
                               [0.78278504, 0.        ],
                               [0.63371769, 1.        ],
                               [0.21708982, 0.        ],
                               [0.04168983, 1.        ],
                               [0.58573963, 0.        ],
                               [0.95840755, 1.        ],
                               [0.41500774, 0.        ]])
                        
    assert (my_KrigingSurrogate.X_train.round(6) == expected_X_train.round(6)).all()
    assert (my_KrigingSurrogate.X_test.round(6)  == expected_X_test.round(6)).all()

@pytest.mark.filterwarnings("ignore: lbfgs failed")
def test_KrigingSurrogate_train_model_predict(my_KrigingSurrogate):
    my_KrigingSurrogate.sample()
    my_KrigingSurrogate.train_model()
    
    #Test the prediction    
    expected_output = {'F': np.array([[22.562512195037218],
                            [20.250022440354236],
                            [18.062502637980753]]),
                     'G': np.array([[-22.5625121950372183,   0.9687476713857812],
                                    [-20.2500224403542362,   0.8749941943046551],
                                    [-18.0625026379807530,   0.7187488119140824]])}
    
    x_in     = np.array([[0.25, 0],
                         [0.5,  0],
                         [0.75, 0]])
    out_dict = {}
    
    my_KrigingSurrogate._evaluate(x_in, out_dict)
    
    # There is some small numerical error that I cannot control caused by the
    # type of machine used for testing by GitHub. Hence assert is written 
    # to assert this precision
    assert (abs(out_dict['F'] - expected_output['F']) < 1e-4).all()
    assert (abs(out_dict['G'] - expected_output['G']) < 1e-4).all()

@pytest.mark.filterwarnings("ignore: lbfgs failed")
def test_KrigingSurrogate_reconstruct_recover_pts(my_KrigingSurrogate):
    my_KrigingSurrogate.sample()
    my_KrigingSurrogate.train_model()
    
    x_in     = np.array([[0.25, 0],
                         [0.5,  0],
                         [0.75, 0]])

    expected_output = np.array([[ 0.25000,  0.00000, 22.56250, 22.56250, 10.96875],
                               [ 0.50000,  0.00000, 20.25000, 20.25000, 10.87500],
                               [ 0.75000,  0.00000, 18.06250, 18.06250, 10.71875]])
                            
    out = my_KrigingSurrogate.recover_pts(x_in)
    
    assert (out.round(6) == expected_output.round(6)).all()
    

## Test DirectOpt
@pytest.fixture
def my_DirectOpt(my_Model, my_DesignSpace):
    return optimisation.DirectOpt(my_Model, my_DesignSpace, 0)

def test_DirectOpt(my_Model, my_DirectOpt):
    
    x_in     = np.array([[0.25, 0],
                         [0.5,  0],
                         [0.75, 0]])
    
    expected_out = {
        'F' : np.atleast_2d([my_Model.run(*X)['obj'] for X in x_in]).T,
        'G' : np.atleast_2d(([[0 - my_Model.run(*X)['con1'], my_Model.run(*X)['con2'] - 10] for X in x_in ]))
        }
    
    out_dict = {}
    
    my_DirectOpt._evaluate(x_in, out_dict)
    
    assert (out_dict['F'].round(6) == expected_out['F'].round(6)).all()
    assert (out_dict['G'].round(6) == expected_out['G'].round(6)).all()

# Test Optimisation
@pytest.mark.filterwarnings("ignore: lbfgs failed")
@pytest.mark.filterwarnings("ignore: invalid")
@pytest.mark.parametrize('use_surrogate, use_nn, gp_kern, expected_output',
                          [(True, False, 'matern', 
                            np.array([[ 2.0000000000000000e+00,  4.9776166868405136e+00,
                                     0.0000000000000000e+00,  5.0101270799563630e-04,
                                     5.0101270799563630e-04, -1.3883339405565653e+00],
                                   [ 4.0000000000000000e+00,  5.0153314976974972e+00,
                                     0.0000000000000000e+00,  2.3505482164836190e-04,
                                     2.3505482164836190e-04, -1.5767750158983098e+00]])),
                          (True, False, 'rbf', 
                           np.array([[ 2.0000000000000000e+00,  4.9930717283301140e+00,
                                    0.0000000000000000e+00,  4.8000948331744473e-05,
                                    4.8000948331744473e-05, -1.4653826421247356e+00],
                                  [ 4.0000000000000000e+00,  5.0052569547100640e+00,
                                    0.0000000000000000e+00,  2.7635572823664532e-05,
                                    2.7635572823664532e-05, -1.5262985913367313e+00]])),
                          (True, True,  None, 
                            np.array([[ 2.0000000000000000,  4.8424522210498493,  0.0000000000000000,
                                     0.0248213026521256,  0.0248213026521256, -0.7246717565753098],
                                   [ 4.0000000000000000,  5.1613830538504022,  0.0000000000000000,
                                     0.0260444900700818,  0.0260444900700818, -2.3199375142870515]])),
                          (False, False, None, 
                            np.array([[ 2.0000000000000000e+00,  4.9997891005059012e+00,
                                     0.0000000000000000e+00,  4.4478596611131480e-08,
                                     4.4478596611131480e-08, -1.4989455247688035e+00],
                                   [ 4.0000000000000000e+00,  5.0000005522770889e+00,
                                     0.0000000000000000e+00,  3.0500998297896795e-13,
                                     3.0500998297896795e-13, -1.5000027613855966e+00]]))])
def test_Optimisation_run(my_DesignSpace, my_Model, 
                    use_surrogate, use_nn, gp_kern, expected_output):
    
    my_DesignSpace = data.DesignSpace('test_input.csv', 'test_response.csv')
    my_Exploration = exploration.ProbabilisticExploration(my_DesignSpace, 
                                                my_Model,
                                                debug=True)
    
    my_Exploration.run()
    
    # Fix the number of iterations to reduce variability
    my_Optimisation = optimisation.Optimisation(my_DesignSpace, my_Model, 
                                                use_surrogate=use_surrogate,
                                                use_nn=use_nn,
                                                gp_kern=gp_kern,
                                                debug=True,
                                                n_max_evals=100,
                                                pop_size=10)
    
    my_Optimisation.run('.')
    
    # There is some small numerical error that I cannot control caused by the
    # type of machine used for testing by GitHub. Hence assert is written 
    # to assert this precision
    assert (abs(my_DesignSpace.get_optimum_results().to_numpy() - expected_output) < 1e-2).all()
    
    
