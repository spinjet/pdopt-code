# -*- coding: utf-8 -*-
"""
This module contains the functions and objects required to carry out the exploration analysis.

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

# Local imports
from .data import DesignSpace, Model, ContinousParameter

# Module Constants
__author__ = "Andrea Spinelli"
__copyright__ = "Copyright 2021, all rights reserved"
__status__ = "Development"

# Module Functions


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    #Context manager to patch joblib to report into tqdm progress bar given as argument

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def generate_surrogate_training_data(parameters_list, model, n_train_points, save_dir=None, debug=False):
    """
    Generates the training data from the evaluation function  for 
    the probabilistic surrogate model, by sampling in the entire design space 
    using a Sobol sequence.

    Args:
        parameters_list (list[pdopt.Parameter]): 
            List containing the input parameter objects.
        model (pdopt.data.Model): 
            Model object containing the evaluation function.
        n_train_points (int): 
            Number of training datapoints to generate.
        save_dir (str, optional): 
            Filename where to save the generated data into a .csv file. If set to None, no file is generated.
        debug (bool, optional): Fix random generator for testing purposes. Defaults to False.

    Returns:
        training_data (pandas.DataFrame): Dataframe with the generated datapoints, 
        with input and output columns as defined in the PDOPT case.

    """
    
    # If factorial sampling has too many points, we switch to a LHS to save time

    # levels_list = [parameter.ranges for parameter in parameters_list]

    # Generate factorial samples and evaluate them
    # factorial_sampling = [x for x in product(*levels_list)]

    # if len(factorial_sampling) > 100:
    if debug:
        samples = Sobol(len(parameters_list), seed=42).random_base2(
            ceil(np.log2(n_train_points))
        )
    else:
        samples = Sobol(len(parameters_list)).random_base2(
            ceil(np.log2(n_train_points))
        )

    for i_par in range(len(parameters_list)):
        tmp = samples[:, i_par]

        if isinstance(parameters_list[i_par], ContinousParameter):
            # Continous Parameter
            tmp *= abs(parameters_list[i_par].ub - parameters_list[i_par].lb)
            tmp += parameters_list[i_par].lb
        else:
            # Discrete Parameter
            tmp *= parameters_list[i_par].n_levels
            np.trunc(tmp, tmp)

    factorial_sampling = samples
    cols = factorial_sampling.shape[1]

    inputs_for_mp = zip(*[factorial_sampling[:, i] for i in range(cols)])

    # Parallelized sampling
    try:
        with Parallel(n_jobs=-1, timeout=20 * 60) as parallel:
            with tqdm_joblib(
                tqdm(desc="Generating Train Data", total=n_train_points)
            ) as progress_bar:
                responses = parallel(delayed(model.run)(*X) for X in inputs_for_mp)

    except TimeoutError as e:
        print(e)
        print(responses)

    # responses          = [model.run(*sample) for sample in tqdm(factorial_sampling,
    #                                                           desc='Response Sampling')]

    # Build a dataframe with input and outputs
    out_keys = list(responses[0].keys())
    in_keys = [parameter.name for parameter in parameters_list]

    temp = {key: [] for key in (in_keys + out_keys)}

    for des in factorial_sampling:
        for i in range(len(in_keys)):
            temp[in_keys[i]].append(des[i])

    for res in responses:
        for i in range(len(out_keys)):
            temp[out_keys[i]].append(res[out_keys[i]])

    training_data = pd.DataFrame(temp)

    if save_dir:
        training_data.to_csv(save_dir, index=False)

    return training_data


def generate_surrogate_test_data(n_points, parameters_list, model, save_dir=None, debug=False):
    """
    Generates the testing data from the evaluation function for 
    the probabilistic surrogate model, by sampling in the entire design space 
    using latin hybercube sampling.

    Args:
        n_points (int): 
            Number of training datapoints to generate.
        parameters_list (list[pdopt.Parameter]): 
            List containing the input parameter objects.
        model (pdopt.data.Model): 
            Model object containing the evaluation function.
        save_dir (str, optional): 
            Filename where to save the generated data into a .csv file. If set to None, no file is generated.
        debug (bool, optional): 
            Fix random generator for testing purposes. Defaults to False.

    Returns:
        test_data (pandas.DataFrame): Dataframe with the generated datapoints, 
        with input and output columns as defined in the PDOPT case.

    """
    
    # Generate random samples via Latin Hypercube and evaluate them
    if debug:
        samples = LatinHypercube(len(parameters_list), seed=42).random(n_points)
    else:
        samples = LatinHypercube(len(parameters_list)).random(n_points)

    # The samples are normalized in (0,1), we need to scale them
    # by iterating on each column
    for i_par in range(len(parameters_list)):
        tmp = samples[:, i_par]

        if isinstance(parameters_list[i_par], ContinousParameter):
            # Continous Parameter
            tmp *= abs(parameters_list[i_par].ub - parameters_list[i_par].lb)
            tmp += parameters_list[i_par].lb
        else:
            # Discrete Parameter
            tmp *= parameters_list[i_par].n_levels
            np.trunc(tmp, tmp)

    # Parallelized sampling
    cols = samples.shape[1]
    inputs_for_mpire = zip(*[samples[:, i] for i in range(cols)])

    with Parallel(n_jobs=-1, timeout=20 * 60) as parallel:
        with tqdm_joblib(
            tqdm(desc="Generating Test Data", total=n_points)
        ) as progress_bar:
            responses = parallel(delayed(model.run)(*X) for X in inputs_for_mpire)

    # Build a dataframe with input and outputs
    out_keys = list(responses[0].keys())
    in_keys = [parameter.name for parameter in parameters_list]

    temp = {key: [] for key in (in_keys + out_keys)}

    for des in samples:
        for i in range(len(in_keys)):
            temp[in_keys[i]].append(des[i])

    for res in responses:
        for i in range(len(out_keys)):
            temp[out_keys[i]].append(res[out_keys[i]])

    test_data = pd.DataFrame(temp)

    if save_dir:
        test_data.to_csv(save_dir, index=False)

    return test_data


def generate_input_samples(n_points, parameters_list, rule="lhs", debug=False):
    """
    Auxiliary function for sampling the full design space.
    The default rule is Latin Hypercube sampling (lhs), but Sobol and 
    factorial grid sampling are available.


    Args:
        n_points (int): 
            Number of samples to generate.
        parameters_list (list[pdopt.data.Parameter]): 
            List containing the input parameter objects.
        rule (str, optional): 
            Method used for sampling. Available methods are LatinHypercube (’lhs’), Sobol sequence (’sobol’) and full factorial sampling (’grid’). Defaults to "lhs".
        debug (TYPE, optional): 
            Fix random generator for testing purposes. Defaults to False.

    Returns:
        samples (numpy.ndarray): Numpy array containing the input samples, columns ordered as the parameters in parameters list..

    """
    
    # Random samples via Latin Hypercube and evaluate them
    # Rules:
    #     - 'lhs'   : Latin Hypercube
    #     - 'sobol' : Sobol Rules
    #     - 'grid'  : Grid Hypercube

    if rule == "lhs":
        if debug:
            samples = LatinHypercube(len(parameters_list), seed=42).random(n_points)
        else:
            samples = LatinHypercube(len(parameters_list)).random(n_points)

        for i_par in range(len(parameters_list)):
            tmp = samples[:, i_par]

            if isinstance(parameters_list[i_par], ContinousParameter):
                # Continous Parameter
                tmp *= abs(parameters_list[i_par].ub - parameters_list[i_par].lb)
                tmp += parameters_list[i_par].lb
            else:
                # Discrete Parameter
                tmp *= parameters_list[i_par].n_levels
                np.trunc(tmp, tmp)

    elif rule == "sobol":
        if debug:
            samples = Sobol(len(parameters_list), seed=42).random(n_points)
        else:
            samples = Sobol(len(parameters_list)).random(n_points)

        for i_par in range(len(parameters_list)):
            tmp = samples[:, i_par]

            if isinstance(parameters_list[i_par], ContinousParameter):
                # Continous Parameter
                tmp *= abs(parameters_list[i_par].ub - parameters_list[i_par].lb)
                tmp += parameters_list[i_par].lb
            else:
                # Discrete Parameter
                tmp *= parameters_list[i_par].n_levels
                np.trunc(tmp, tmp)
    else:
        N_lev = round(n_points ** (1 / len(parameters_list)))

        lev_list = []

        for parm in parameters_list:
            if isinstance(parm, ContinousParameter):
                # Continous Parameter
                tmp = np.linspace(parm.lb, parm.ub, N_lev)
                lev_list.append(list(tmp))
            else:
                # Discrete Parameter
                lev_list.append(list(range(parm.n_levels)))

        factorial_sampling = [x for x in product(*lev_list)]
        samples = np.array(factorial_sampling)

    return samples


# Abstract Classes


# Concrete Classes
class SurrogateResponse:
    '''
    Class that encapsulates the Gaussian Process Regressor to be used as probabilistic surrogate model.
    
    Attributes:
        id (int): 
            Unique id of the surrogate response.
        name (str): 
            Name of the surrogate response.
        n_levels (int): 
            Number of levels of the continous parameter.
        par_names (list[str]): 
            List containing the parameter names
        score (float):
            r^2 score of the trained Gaussian Process
        train_time (float):
            Total training time (seconds).
        x_scaler (sklearn.MixMaxScaler):
            Object used for normalising input data between [0,1].
        f (sklearn.GaussianProcessRegressor):
            Trained Gaussian Process function.
    ''' 
    
    # Automatically generate data and train a gaussian process
    # on the selected response.

    _ids = count(0)

    def __init__(self, response_name, parameters_list, model, train_data=None, test_data=None):
        """
        Initialise and train the SurrogateResponse object. 

        Args:
            response_name (str): 
                Name of the response.
            parameters_list (list[pdopt.data.Parameter]): 
                List containing the input parameter objects.
            model (pdopt.data.Model): 
                Model object containing the evaluation function.
            train_data (str, optional): 
                Path to training data .csv to be loaded. If set to None, it will generate a new set of training data.
            test_data (str, optional): 
                Path to testing data .csv to be loaded. If set to None, it will generate a new set of testing data.

        Returns:
            None.

        """
        
        
        self.id = next(self._ids)
        self.name = response_name
        self.score = 0
        self.train_time = 0

        # Measure elasped time
        t0 = time()

        # Load or generate training data
        if train_data is not None:
            training_data = train_data
        else:
            training_data = generate_surrogate_training_data(
                parameters_list, model, 128
            )

        # Generate testing data
        if test_data is not None:
            testing_data = test_data
        else:
            testing_data = generate_surrogate_test_data(
                int(0.25 * len(training_data)), parameters_list, model
            )

        self.par_names = [parameter.name for parameter in parameters_list]

        # Scale input
        self.x_scaler = MinMaxScaler()

        X_train = self.x_scaler.fit_transform(training_data[self.par_names].to_numpy())

        Y_train = training_data[self.name].to_numpy().reshape(-1, 1)

        X_test = self.x_scaler.transform(testing_data[self.par_names].to_numpy())
        Y_test = testing_data[self.name].to_numpy().reshape(-1, 1)

        kern = (
            Matern()
        )  # * ConstantKernel()# + ConstantKernel()  #RationalQuadratic() # + ConstantKernel()

        self.f = GPR(kernel=kern, normalize_y=True, n_restarts_optimizer=10).fit(
            X_train, Y_train
        )
        self.score = self.f.score(X_test, Y_test)

        if self.score < 0.6:
            warn(
                "Warning: Surrogate {} score is low, r={:.4f}".format(
                    self.name, self.score
                )
            )
        elif self.score < 0:
            raise (
                "Errore: Surrogate {} score is negative, r={:.4f}".format(
                    self.name, self.score
                )
            )

        t1 = time()
        self.train_time = t1 - t0

    def __repr__(self):
        return f"SurrogateResponse of {self.name}, score = {self.score}"

    def predict(self, x):
        """
        

        Args:
            x (np.narray): Array of input points where to evaluate the surrogate.

        Returns:
            mu (np.narray): Expected response of the input points.
            sigma (np.narray): Standard deviation of the expected response.

        """
        
        X_norm = self.x_scaler.transform(np.atleast_2d(x))
        mu, sigma = self.f.predict(X_norm, return_std=True)

        return mu, sigma


class ProbabilisticExploration:
    '''
    Class that represents the object performing the probabilistic exploration.
    
    Attributes:
        design_space (pdopt.data.DesignSpace): 
            The design space object of the problem.
        parameters (list[pdopt.data.Parameter]): 
            List containing the input parameter objects.
        objectives (list[pdopt.data.Objective]): 
            List containing the objective objects.
        constraints (list[pdopt.data.Constraint]): 
            List containing the constraint objects.
        model (pdopt.data.Model):
            Model object containing the evaluation function.
        run_time (float):
            Total running time (seconds).
        surrogate_train_data (pandas.DataFrame):
            Training data for the surrogates.
        surrogate_test_data (pandas.DataFrame):
            Testing data for the surrogates.
        surrogates (dict[str, pdopt.exploration.SurrogateResponse]):
            The trained surrogate objects, keywords are the response names.
    ''' 
    
    
    # Evaluate the design space and calculate the probability to satisfy
    # the requirements and constraints.

    def __init__(self, design_space, model, surrogate_training_data_file=None, surrogate_testing_data_file=None, n_train_points=128, debug=False):
        """
        Initialise the Probabilistic Exploration object.

        Args:
            design_space (pdopt.data.DesignSpace): 
                The design space object of the problem.
            model (pdopt.data.Model):
                Model object containing the evaluation function.
            surrogate_training_data_file (str, optional): 
                Path to the training data .csv file. If None, generates data from scratch.
            surrogate_testing_data_file (str, optional): 
                Path to the testing data .csv file. If None, generates data from scratch.
            n_train_points (float, optional): 
                Number of points for training (must be a power of 2). Defaults to 128.
            debug (bool, optional): 
                Fix random generator for testing purposes. Defaults to False.

        Returns:
            None.

        """
        
        
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

        # Build responses and train surrogates
        self.__surrogates_training()

        # Build data structure with responses and their operands
        # required for the PDOPT space exploration tool
        self.responses = {}

        for objective in self.objectives:
            self.responses.update({objective.name: objective.operand})

        for constraint in self.constraints:
            self.responses.update({constraint.name: constraint.operand})

    def __doe_train_test_data(
        self, n_train_points, surrogate_training_data_file, surrogate_testing_data_file
    ):
        # Perform the creation of train and test data or load from file
        # Load Samples or Generate Samples
        if surrogate_training_data_file and exists(surrogate_training_data_file):
            self.surrogate_train_data = pd.read_csv(surrogate_training_data_file)
        else:
            self.surrogate_train_data = generate_surrogate_training_data(
                self.parameters,
                self.model,
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

    def __surrogates_training(self):
        # Build the list of responses to construct surrogate models of

        surrogate_responses = []
        self.requirements = {}

        # Add objectives only if they have minimum requirements defined
        for objective in self.objectives:
            surrogate_responses.append(objective.name)
            if objective.hasRequirement:
                op, val = objective.get_requirement()
                self.requirements.update(
                    {objective.name: (op, val, objective.p_satisfaction)}
                )

        for constraint in self.constraints:
            surrogate_responses.append(constraint.name)
            op, val = constraint.get_constraint()
            self.requirements.update(
                {constraint.name: (op, val, constraint.p_satisfaction)}
            )

        # Eliminate duplicates
        surrogate_responses = list(set(surrogate_responses))

        # Train the surrogates
        self.surrogates = {}

        for response in tqdm(surrogate_responses, desc="Training Surrogate Responses"):
            self.surrogates.update(
                {
                    response: SurrogateResponse(
                        response,
                        self.parameters,
                        self.model,
                        train_data=self.surrogate_train_data,
                        test_data=self.surrogate_test_data,
                    )
                }
            )

    @classmethod
    def from_pickle(cls, filepath):
        """
        Helper function to load the ProbabilisticExploration object from a pickle file.

        Args:
            filepath (str): Path to the .pk file.

        Returns:
            pdopt.exploration.ProbabilisticExploration: The loaded exploration object.

        """
        
        # Load from pickle a ProbabilisticExploration object

        return pk.load(open(filepath, "rb"))

    def save_to_pickle(self, filepath):
        """
        Save the ProbabilisticExploration object to a pickle file.

        Args:
            filepath (str): Path where to save the .pk file.

        Returns:
            None.

        """
        
        # Save as pickle a ProbabilisticExploration object

        pk.dump(self, open(filepath, "wb"))

    def run(self, n_samples=100, p_discard=0.5, debug=False):
        """
        Perform the probabilistic exploration procedure

        Args:
            n_samples (int, optional): 
                The number of samples used to evaluate the satisfaction probability of a set. Defaults to 100.
            p_discard (float, optional): 
                Probability threshold under which a design set is discarded. Defaults to 0.5.
            debug (bool, optional): 
                Fix the random generator for testing purposes. Defaults to False.

        Returns:
            None.

        """
        
        # Run the Probabilistic design exploration and evaluate sets.
        t0 = time()
        for design_set in tqdm(
            self.design_space.sets, desc="Exploring the Design Space"
        ):
            # Generate samples within the set
            samples = design_set.sample(n_samples, self.parameters, debug=debug)

            for requirement_name in self.requirements:
                operand, value, p_sat = self.requirements[requirement_name]
                mu, sigma = self.surrogates[requirement_name].predict(samples)

                # Calculate P of mu < value
                if operand == "lt" or operand == "let":
                    z = (value - np.atleast_1d(mu)) / np.atleast_1d(sigma)

                # Calculate P of mu > value
                else:
                    z = (np.atleast_1d(mu) - value) / np.atleast_1d(sigma)

                P_samples = norm.cdf(z)

                # Now calculate how many samples satisfy the requirement P
                # That is we estimate P(req | set)
                P_req = (P_samples > p_sat).sum() / n_samples

                design_set.set_responses_P(requirement_name, P_req)

            # Discard the set if the total probability is lower than the
            # specified one
            if design_set.P < p_discard:
                design_set.set_as_discarded()
        self.run_time = time() - t0

    def run_surrogate(self, X):
        """
        Run all the surrogate models for a given input.

        Args:
            X (numpy.ndarray): Array of input parameters to be evaluated with the surrogate.

        Returns:
            means (dict[str, numpy.ndarray]): Dictionary containing the expected value for each surrogate response.
            deviations (dict[str, numpy.ndarray]): Dictionary containing the standard deviation for each surrogate response.

        """
        
        means, deviations = {}, {}
        for requirement_name in self.requirements:
            mu, sigma = self.surrogates[requirement_name].predict(X)
            means.update({requirement_name: mu})
            deviations.update({requirement_name: sigma})

        return means, deviations


# Code if imported
if __name__ == "__main__":
    print("pdopt.Exploration")
