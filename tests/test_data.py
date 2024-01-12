# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:45:20 2024

@author: s345001
"""

import pytest
import pdopt.data as data


@pytest.fixture
def my_continous_parameter():
    return data.data.ContinousParameter('test_continous', 
                                        0, 1, 5, 
                                        uq_dist, uq_var_l, uq_var_u)



        
        