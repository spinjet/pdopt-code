# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 10:34:27 2022

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


# Local imports
from .data import DesignSpace, Model

