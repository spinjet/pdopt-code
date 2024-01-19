# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:45:20 2024

"""

import pytest
import pdopt.data as data
import numpy as np
import pandas as pd
import os

# Test the continous parameter data structure works
@pytest.fixture
def my_continous_parameter():
    return data.ContinousParameter('test_continous', 
                                        0, 1, 4, 
                                        'uniform', 0.1, 0.1)


def test_continous_parameter_repr(my_continous_parameter):
    test_string = f'id:{my_continous_parameter.id} - Continous Parameter "{my_continous_parameter.name}"\n{my_continous_parameter.n_levels} Levels, Bounds: (0.000, 0.250, 0.500, 0.750, 1.000)\n'
    assert str(my_continous_parameter) == test_string


def test_continous_parameter_bounds(my_continous_parameter):
    assert my_continous_parameter.get_bounds() == (0, 1)
    
@pytest.mark.parametrize("level, expected_output", [(0, (0.0, 0.25)), (1, (0.25, 0.50)), 
                                   (2, (0.50, 0.75)), (3, (0.75, 1.0))])
def test_continous_parameter_get_level_bounds(my_continous_parameter, level, expected_output):
    assert my_continous_parameter.get_level_bounds(level) == expected_output
    

# Test the continous parameter percentile works for the supported distributions
@pytest.mark.parametrize("distribution, expected_output", 
                         [('uniform', np.array([0.475, 0.5, 0.525])),
                          ('triang', np.array([0.48535534, 0.5, 0.51464466])),
                          ('norm', np.array([0.48313776, 0.5, 0.51686224])),
                          (None, np.array([0.5, 0.5, 0.5]))])
def test_continous_parameter_distributions_ppf(distribution, expected_output):
    my_parameter = data.ContinousParameter('test_continous', 
                                        0, 1, 4, 
                                        distribution, 0.1, 0.1)
    
    assert (my_parameter.ppf([0.25, 0.5, 0.75], 0.5).round(3) == expected_output.round(3)).all()


# Test the discrete parameter data structure works
@pytest.fixture
def my_discrete_parameter():
    return data.DiscreteParameter('test_discrete', 5)

def test_discrete_parameter_repr(my_discrete_parameter):
    assert str(my_discrete_parameter) == f'id:{my_discrete_parameter.id} - Discrete Parameter "{my_discrete_parameter.name}"\n{my_discrete_parameter.n_levels} Levels\n'
    
def test_discrete_parameter_get_n_levels(my_discrete_parameter):
    assert my_discrete_parameter.get_n_levels() == 5
    
    
# Test the objective object
def test_objective_repr():
    my_objective = data.Objective('test_objective', 'min', min_requirement=1, p_sat=0.3)
    assert str(my_objective) == f"Objective {my_objective.name} : {my_objective.operand}: Requirement: {my_objective.get_requirement()} : P_sat = {my_objective.p_satisfaction}\n"

@pytest.mark.parametrize("operand, has_requirement, expected_output", 
                         [('min', None, None), ('max', None, None),
                          ('min', 1, ('lt', 1)), ('max', 1, ('gt', 1)),])
def test_objective_get_requirement(operand, has_requirement, expected_output):
    my_objective = data.Objective('test_objective', operand, min_requirement=has_requirement)
    
    assert my_objective.get_requirement() == expected_output
    
@pytest.mark.parametrize("operand, expected_output", 
                         [('min', 1), ('max', -1)])
def test_objective_get_operand(operand, expected_output):
    my_objective = data.Objective('test_objective', operand)
    assert my_objective.get_operand() == expected_output
    

# Test the constraint object
def test_constraint_repr():
    my_constraint = data.Constraint('test_constraint', 'lt', 1, p_sat=0.25)
    assert str(my_constraint) == f'Constraint {my_constraint.name} : {my_constraint.get_constraint()} : P_sat = {my_constraint.p_satisfaction}\n'

@pytest.mark.parametrize("operand, expected_output", 
                         [('lt', ('lt', 1)),
                          ('gt', ('gt', 1)),
                          ('let', ('let', 1)),
                          ('get', ('get', 1))])
def test_constraint_get_constraint(operand, expected_output):
    my_constraint = data.Constraint('test_constraint', operand, 1, p_sat=0.25)
    assert my_constraint.get_constraint() == expected_output
    
    
# Test the DesignSet object
@pytest.fixture
def my_DesignSet():
    par1 = data.ContinousParameter('par1', 0, 1, 4, None, None, None) 
    par2 = data.DiscreteParameter('par2', 4)
    
    obj  = data.Objective('obj', 'min')
    con1  = data.Constraint('con1', 'lt', 1)
    con2  = data.Constraint('con2', 'gt', 0.1)
    
    my_design_set = data.DesignSet({'par1' : 0, 'par2': 0}, 
                                   {'obj': obj, 
                                    'con1': con1,
                                    'con2': con2})
    
    return my_design_set

    
    return my_design_set

def test_DesignSet_response_getting_setting_P(my_DesignSet):
    my_DesignSet.set_responses_P('con1', 0.5)
    assert my_DesignSet.get_P() == 0.5 
    
    my_DesignSet.set_responses_P('con2', 0.3)
    assert my_DesignSet.get_response_P('con1') == 0.5
    assert my_DesignSet.get_response_P('con2') == 0.3
    assert my_DesignSet.get_P() == 0.5*0.3 
    
def test_DesignSet_samples(my_DesignSet):
    par1 = data.ContinousParameter('p1', 0, 1, 4, None, None, None) 
    par2 = data.DiscreteParameter('p2', 4)
    
    expected_output = np.array([[0.0556511 , 0.        ],
           [0.22853505, 0.        ],
           [0.02264557, 0.        ],
           [0.20597151, 0.        ],
           [0.12179716, 0.        ],
           [0.04073005, 0.        ],
           [0.15890337, 0.        ],
           [0.18891465, 0.        ],
           [0.08613538, 0.        ],
           [0.12930922, 0.        ]])
    
    assert (my_DesignSet.sample(10, [par1, par2], debug=True).round(4) == expected_output.round(4)).all()
    
    
class Dummy_Optimisation_Problem:
    var = [data.ContinousParameter('par1', 0, 1, 4, None, None, None),
                data.DiscreteParameter('par2', 4)]
    
    obj = [data.Objective('obj', 'min')]
    
    cst = [data.Constraint('con1', 'lt', 1),
           data.Constraint('con2', 'gt', 0.1)]
    
    x_mask = ['c',0]
    
class Dummy_Optimisation_Results:
    X = np.array([[0.2,0],
                  [0.04,0]])
    
    F = np.array([1.0,2.0]).reshape(-1,1)
    G = np.array([[-0.5, 0.1],
                  [0.5, -0.1]])
    
    
def test_DesignSet_get_optimum(my_DesignSet):
    my_DesignSet.set_optimisation_problem(Dummy_Optimisation_Problem())
    my_DesignSet.optimisation_results = Dummy_Optimisation_Results()
    
    test_opt = my_DesignSet.get_optimum()
    
    expected_output = pd.DataFrame({'par1':   [0.20, 0.04],
                                    'par2':   [0.0, 0.0],
                                    'obj':  [1.0, 2.0],
                                    'con1': [0.5, 1.5],
                                    'con2': [0.0, 0.2]})
    
    assert expected_output.equals(test_opt)
    
# Test Model object
def test_Model():
    def foo():
        pass
    
    assert data.Model(foo).run == foo
    
    
# Test DesignSpace object
@pytest.fixture
def my_DesignSpace():
    return data.DesignSpace('test_input.csv', 'test_response.csv')


def test_DesignSpace_get_exploration_results(my_DesignSpace):
    # Simulate exploration run
    for design_set in my_DesignSpace.sets:
        design_set.set_responses_P('con1', 0.8)
        design_set.set_responses_P('con2', 0.35)
        design_set.set_responses_P('obj', 0.7)
    
    output = my_DesignSpace.get_exploration_results()
    
    # Check that all the data has been put in the correct columns
    assert (output['set_id']  == np.array([s.id for s in my_DesignSpace.sets])).all()
    assert (output['is_discarded']  == np.zeros(8)).all()
    assert (output['par1']      == np.array([0,0,1,1,2,2,3,3])).all()
    assert (output['par2']      == np.array([0,1,0,1,0,1,0,1])).all()
    assert (output['P_con1']    == np.ones(8)*(0.8)).all()
    assert (output['P_con2']    == np.ones(8)*(0.35)).all()
    assert (output['P_obj']     == np.ones(8)*(0.7)).all()
    
    
def test_DesignSpace_get_optimum_results(my_DesignSpace):
    # Add dummy optimisation results to the design space
    for design_set in my_DesignSpace.sets[:-2]:
        # Check that it skips the unresolved optimisation problems
        design_set.set_optimisation_problem(Dummy_Optimisation_Problem())
        design_set.optimisation_results = Dummy_Optimisation_Results()
        
    
    output = my_DesignSpace.get_optimum_results()
    
    # Check that all the data has been put in the correct columns
    
    s_id_list = [0., 0., 1., 1., 2., 2., 3., 3., 4., 4., 5., 5.]
    
    assert (output['set_id']  == np.array(s_id_list)).all()
    assert (output['par1']    == np.array([0.20, 0.04]*6)).all()
    assert (output['par2']    == np.array([0.0, 0.0]*6)).all()
    assert (output['obj']     == np.array([1.0, 2.0]*6)).all()
    assert (output['con1']    == np.array([0.5, 1.5]*6)).all()
    assert (output['con2']    == np.array([0.0, 0.2]*6)).all()
    
def test_DesignSpace_set_discard_status(my_DesignSpace):
    pattern = [True, False, False, True, True, False, True, False]
    
    for i,p in enumerate(pattern):
        my_DesignSpace.set_discard_status(i, p)
        
    # Check the correct value was added
    output = my_DesignSpace.get_exploration_results()
    
    assert (output['is_discarded']  == np.array(pattern).astype(int)).all()
    