# -*- coding: utf-8 -*-
"""
This module contains classes and functions used for the search phase.

"""

from time import sleep, time
import json
import pickle as pk
import contextlib
from math import ceil

# Third-party imports
import numpy as np
import pandas as pd

from scipy.stats import randint, uniform, gamma, expon
from scipy.stats.qmc import Sobol, LatinHypercube

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.optimize import minimize

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, RBF
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


from tqdm import tqdm
from joblib import parallel_backend, Parallel, delayed

import joblib
import pandas as pd

# Local imports
from .data import DesignSpace, Model, ContinousParameter

# Module Constants
__author__ = "Andrea Spinelli"
__copyright__ = "Copyright 2022, all rights reserved"
__status__ = "Development"

t_max = 60 * 60

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


# Abstract Classes

# Concrete Classes


class NNSurrogate(Problem):
    '''
    Class that represents a deterministic optimisation problem, using a 
    neural-network surrogate model for function evaluation. 
    It is locally trained before use on the DesignSet it is part of. 
    This class wraps the pymoo.core.problem.Problem class.
    
    Attributes:
        design_space (pdopt.data.DesignSpace): 
            The design space object of the problem.
        var (list[pdopt.data.Parameter]): 
            List containing the input parameter objects.
        obj (list[pdopt.data.Objective]): 
            List containing the objective objects.
        cst (list[pdopt.data.Constraint]): 
            List containing the constraint objects.
        model (pdopt.data.Model):
            Model object containing the evaluation function.
        set_id (int):
            id of the set which the problem is defined on.
        x_mask (list[float]):
            Mask to filter out the discrete paramters, as these are
            constant in the set and not modified by the optimiser.
        l (list[float]):
            Lower bounds of the continous input parameters.
        u (list[float]):
            Upper bounds of the continous input parameters.
    '''     
    
    def __init__(self, model, design_space, set_id, debug=False, **kwargs):
        """
        Initialise the NeuralNetwork Surrogate optimisation problem object.

        Args:
            model (pdopt.data.Model):
                Model object containing the evaluation function.
            design_space (pdopt.data.DesignSpace): 
                The design space object of the problem.
            set_id (int):
                id of the set which the problem is defined on.
            debug (bool, optional): 
                Fix the random generator seed for testing purposes. Defaults to False.

        Returns:
            None.

        """
        
        self.model = model  # store reference to model
        self.design_space = design_space

        self.var = self.design_space.parameters
        self.obj = self.design_space.objectives
        self.cst = self.design_space.constraints

        self.set_id = set_id
        set_levels = self.design_space.sets[self.set_id].parameter_levels_list

        # Construct the array with lower bounds and upper bounds for each
        # input variable. Discrete variables are removed as they are fixed.
        self.x_mask = []
        self.l, self.u = [], []

        for i_par in range(len(self.var)):
            if isinstance(self.var[i_par], ContinousParameter):
                # Continous Parameter
                self.x_mask.append("c")
                lb, ub = self.var[i_par].get_level_bounds(set_levels[i_par])

                self.l.append(lb)
                self.u.append(ub)

            else:
                # Discrete parameters are fixed within the set
                self.x_mask.append(set_levels[i_par])

        super().__init__(
            n_var=len(self.l),
            n_obj=len(self.obj),
            n_constr=len(self.cst),
            xl=np.array(self.l),
            xu=np.array(self.u),
            elementwise_evaluation=False,
            **kwargs,
        )

        self.debug = debug
        # Sample the data required to train the network

        # Train the network

        # Test, report if it has not been effective

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluation function for the pymoo optimisation algorithms using the Neural Network Surrogate

        Args:
            X (numpy.ndarray): Input values to be evaluated.
            out (dict[str, numpy.ndarray]): A dictionary containing the evaluated objectives and constraint function.

        Returns:
            None.

        """
        
        X_in = []
        for x in X:
            in_x = []
            i_tmp = 0
            for par in self.x_mask:
                if par == "c":
                    in_x.append(x[i_tmp])
                    i_tmp += 1
                else:
                    in_x.append(par)
            X_in.append(in_x)

        X_tr = self.X_scaler.transform(np.array(X_in))
        Y_tr = self.sr_model.predict(X_tr)

        Y = self.y_scaler.inverse_transform(Y_tr)

        F_list = []
        G_list = []

        for y in Y:
            f_list = []
            g_list = []

            for i_o, objective in enumerate(self.obj):
                f = y[i_o] * objective.get_operand()
                f_list.append(f)

            # Constraints must be of the less than 0 form
            for i_c, constraint in enumerate(self.cst):
                i_c += len(self.obj)
                tmp = y[i_c]
                op, val = constraint.get_constraint()

                # g(x) < K ->  g(x) - K < 0
                if op == "lt" or op == "let":
                    g = tmp - val

                # g(x) > K ->  0 > K - g(x)
                else:
                    g = val - tmp

                g_list.append(g)

            F_list.append(f_list)
            G_list.append(g_list)

        out["F"] = np.array(F_list)
        out["G"] = np.array(G_list)

    def recover_pts(self, X):
        """
        Reconstruct the true output from the evaluation function of the optimal points X.

        Args:
            X (numpy.ndarray): Input values to be evaluated.

        Returns:
            numpy.ndarray: Array containing the input values with the true objective and constraint values.

        """
        
        # Reconstruct the full input with the discrete variables
        X_in = []

        for x in np.atleast_2d(X):
            in_x = []
            i_tmp = 0
            for par in self.x_mask:
                if par == "c":
                    in_x.append(x[i_tmp])
                    i_tmp += 1
                else:
                    in_x.append(par)
            X_in.append(in_x)
        X_in = np.array(X_in)

        recover_inp = zip(*[X_in[:, i] for i in range(X_in.shape[1])])

        # Parallelized reconstruction
        with Parallel(n_jobs=-1, timeout=t_max) as parallel:
            with tqdm_joblib(
                tqdm(desc="Recovering Solutions", total=X_in.shape[0])
            ) as progress_bar:
                recover_resp = parallel(
                    delayed(self.model.run)(*X) for X in recover_inp
                )

        F, G = [], []

        for resp in recover_resp:
            f, g = [], []
            for k_f in self.obj:
                f.append(resp[k_f.name])

            for k_g in self.cst:
                g.append(resp[k_g.name])

            F.append(f)
            G.append(g)

        F, G = np.array(F), np.array(G)

        self.recovered_out = np.hstack([X_in, F, G])

        return self.recovered_out

    def sample(self):
        """
        Sample to train the surrogate model. 

        Returns:
            None.

        """
        
        set_levels = self.design_space.sets[self.set_id].parameter_levels_list

        self.n_train_samples = max([int(10 * len(self.l)), 64])
        self.n_test_samples = max([int(0.20 * self.n_train_samples), 32])

        self.train_samples = Sobol(
            len(self.var), seed=42 if self.debug else None
        ).random_base2(ceil(np.log2(self.n_train_samples)))
        self.test_samples = LatinHypercube(
            len(self.var), seed=42 if self.debug else None
        ).random(self.n_test_samples)

        # setup the training samples
        for i_par in range(len(self.var)):
            tmp = self.train_samples[:, i_par]

            if isinstance(self.var[i_par], ContinousParameter):
                # Continous Parameter
                lb, ub = self.var[i_par].get_level_bounds(set_levels[i_par])

                tmp *= abs(ub - lb)
                tmp += lb
            else:
                # Discrete Parameter
                tmp *= self.var[i_par].n_levels
                np.trunc(tmp, tmp)

        # setup the test samples
        for i_par in range(len(self.var)):
            tmp = self.test_samples[:, i_par]

            if isinstance(self.var[i_par], ContinousParameter):
                # Continous Parameter
                lb, ub = self.var[i_par].get_level_bounds(set_levels[i_par])

                tmp *= abs(ub - lb)
                tmp += lb
            else:
                # Discrete Parameter
                tmp *= self.var[i_par].n_levels
                np.trunc(tmp, tmp)

        cols = self.train_samples.shape[1]

        train_inp = zip(*[self.train_samples[:, i] for i in range(cols)])
        test_inp = zip(*[self.test_samples[:, i] for i in range(cols)])

        # Parallelized sampling

        with Parallel(n_jobs=-1, timeout=t_max) as parallel:
            with tqdm_joblib(
                tqdm(desc="Generating Train Data", total=self.n_train_samples)
            ) as progress_bar:
                train_resp = parallel(delayed(self.model.run)(*X) for X in train_inp)

            with tqdm_joblib(
                tqdm(desc="Generating Test Data", total=self.n_test_samples)
            ) as progress_bar:
                test_resp = parallel(delayed(self.model.run)(*X) for X in test_inp)

        # Responses are ordered in OBJ first and CON second, in the order
        # they are stored in the list.
        self.order_resp = [o.name for o in self.obj] + [c.name for c in self.cst]

        self.train_Y_samples = np.array(
            [[out[k] for k in self.order_resp] for out in train_resp]
        )
        self.test_Y_samples = np.array(
            [[out[k] for k in self.order_resp] for out in test_resp]
        )

        # setup the scaler
        self.X_scaler = MinMaxScaler().fit(self.train_samples)
        self.y_scaler = MinMaxScaler().fit(self.train_Y_samples)

        self.X_train = self.X_scaler.transform(self.train_samples)
        self.Y_train = self.y_scaler.transform(self.train_Y_samples)

        self.X_test = self.X_scaler.transform(self.test_samples)
        self.Y_test = self.y_scaler.transform(self.test_Y_samples)

    def train_model(self):
        """
        Train the surrogate model.

        Returns:
            None.

        """
        
        tmp_model = MLPRegressor(
            solver="adam",
            max_iter=1000,
            early_stopping=True,
            learning_rate="adaptive",
            random_state=42 if self.debug else None,
        )

        units = np.arange(50, 200, 5)  # np.logspace(1.1,2.2,20)

        hidden_layer_sizes = [(int(x), int(x), int(x), int(x)) for x in units]
        hidden_layer_sizes.extend(
            [
                (
                    int(x),
                    int(x),
                    int(x),
                )
                for x in units
            ]
        )
        hidden_layer_sizes.extend(
            [
                (
                    int(x),
                    int(x),
                )
                for x in units
            ]
        )
        hidden_layer_sizes.extend([(int(x),) for x in units])

        params = [
            {
                "hidden_layer_sizes": hidden_layer_sizes,
                "learning_rate_init": expon(scale=0.01),
            }
        ]
        mape, r2 = 1, 0
        it = 0

        while mape > 0.10 and r2 < 0.9:
            with parallel_backend("loky", n_jobs=-1):
                rs = RandomizedSearchCV(
                    tmp_model,
                    params,
                    scoring=["r2", "neg_mean_absolute_percentage_error"],
                    # 'neg_mean_squared_error'],
                    refit="neg_mean_absolute_percentage_error",
                    n_iter=100,
                    random_state=42 if self.debug else None,
                )

                rs.fit(self.X_train, self.Y_train)

            self.sr_model = rs.best_estimator_

            y_pred = self.sr_model.predict(self.X_test)

            self.mse = mean_squared_error(self.Y_test, y_pred)
            self.mape = mean_absolute_percentage_error(self.Y_test, y_pred)
            self.r2 = r2_score(self.Y_test, y_pred)

            mape = self.mape
            r2 = self.r2

            it += 1
            if it > 20:
                break

        print(
            f"Hidden Layers: {rs.best_params_['hidden_layer_sizes']}, Learning Rate {rs.best_params_['learning_rate_init']:.2e}"
        )
        print(
            f"Model Trained: val_mse {self.mse:.3e}, val_mape {100*self.mape:.2f}%, r2 {self.r2:.3f}"
        )


class KrigingSurrogate(Problem):
    '''
    Class that represents a deterministic optimisation problem, using a 
    Kriging surrogate model for function evaluation. 
    It is locally trained before use on the DesignSet it is part of. 
    This class wraps the pymoo.core.problem.Problem class.
    
    Attributes:
        design_space (pdopt.data.DesignSpace): 
            The design space object of the problem.
        var (list[pdopt.data.Parameter]): 
            List containing the input parameter objects.
        obj (list[pdopt.data.Objective]): 
            List containing the objective objects.
        cst (list[pdopt.data.Constraint]): 
            List containing the constraint objects.
        model (pdopt.data.Model):
            Model object containing the evaluation function.
        set_id (int):
            id of the set which the problem is defined on.
        kernel (str):
            Type of Gaussian Process kernel. 
        x_mask (list[float]):
            Mask to filter out the discrete paramters, as these are
            constant in the set and not modified by the optimiser.
        l (list[float]):
            Lower bounds of the continous input parameters.
        u (list[float]):
            Upper bounds of the continous input parameters.
    '''    
    
    def __init__(self, model, design_space, set_id, kernel, debug=False, **kwargs):
        """
        Initialise the Kriging Surrogate optimisation problem.

        Args:
            model (pdopt.data.Model):
                Model object containing the evaluation function.
            design_space (pdopt.data.DesignSpace): 
                The design space object of the problem.
            set_id (int):
                id of the set which the problem is defined on.
            kernel (str):
                Type of Gaussian Process kernel. 
            debug (bool, optional): 
                Fix the random generator seed for testing purposes. Defaults to False.

        Returns:
            None.

        """
        
        self.model = model  # store reference to model
        self.design_space = design_space

        self.var = self.design_space.parameters
        self.obj = self.design_space.objectives
        self.cst = self.design_space.constraints
        self.kernel = kernel

        self.set_id = set_id
        set_levels = self.design_space.sets[self.set_id].parameter_levels_list

        # Construct the array with lower bounds and upper bounds for each
        # input variable. Discrete variables are removed as they are fixed.
        self.x_mask = []
        self.l, self.u = [], []

        for i_par in range(len(self.var)):
            if isinstance(self.var[i_par], ContinousParameter):
                # Continous Parameter
                self.x_mask.append("c")
                lb, ub = self.var[i_par].get_level_bounds(set_levels[i_par])

                self.l.append(lb)
                self.u.append(ub)

            else:
                # Discrete parameters are fixed within the set
                self.x_mask.append(set_levels[i_par])

        super().__init__(
            n_var=len(self.l),
            n_obj=len(self.obj),
            n_constr=len(self.cst),
            xl=np.array(self.l),
            xu=np.array(self.u),
            elementwise_evaluation=False,
            **kwargs,
        )

        self.debug = debug

        # Sample the data required to train the network

        # Train the network

        # Test, report if it has not been effective

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluation function for the pymoo optimisation algorithms using the Kriging Surrogate

        Args:
            X (numpy.ndarray): Input values to be evaluated.
            out (dict[str, numpy.ndarray]): A dictionary containing the evaluated objectives and constraint function.

        Returns:
            None.

        """
        X_in = []
        for x in X:
            in_x = []
            i_tmp = 0
            for par in self.x_mask:
                if par == "c":
                    in_x.append(x[i_tmp])
                    i_tmp += 1
                else:
                    in_x.append(par)
            X_in.append(in_x)
        0
        X_tr = self.X_scaler.transform(np.array(X_in))
        Y_tr = self.sr_model.predict(X_tr)

        Y = self.y_scaler.inverse_transform(Y_tr)

        F_list = []
        G_list = []

        for y in Y:
            f_list = []
            g_list = []

            for i_o, objective in enumerate(self.obj):
                f = y[i_o] * objective.get_operand()
                f_list.append(f)

            # Constraints must be of the less than 0 form
            for i_c, constraint in enumerate(self.cst):
                i_c += len(self.obj)
                tmp = y[i_c]
                op, val = constraint.get_constraint()

                # g(x) < K ->  g(x) - K < 0
                if op == "lt" or op == "let":
                    g = tmp - val

                # g(x) > K ->  0 > K - g(x)
                else:
                    g = val - tmp

                g_list.append(g)

            F_list.append(f_list)
            G_list.append(g_list)

        out["F"] = np.array(F_list)
        out["G"] = np.array(G_list)

    def recover_pts(self, X):
        """
        Reconstruct the true output from the evaluation function of the optimal points X.

        Args:
            X (numpy.ndarray): Input values to be evaluated.

        Returns:
            numpy.ndarray: Array containing the input values with the true objective and constraint values.

        """
        # Reconstruct the full input with the discrete variables
        X_in = []

        for x in np.atleast_2d(X):
            in_x = []
            i_tmp = 0
            for par in self.x_mask:
                if par == "c":
                    in_x.append(x[i_tmp])
                    i_tmp += 1
                else:
                    in_x.append(par)
            X_in.append(in_x)
        X_in = np.array(X_in)

        recover_inp = zip(*[X_in[:, i] for i in range(X_in.shape[1])])

        # Parallelized reconstruction
        with Parallel(n_jobs=-1, timeout=t_max) as parallel:
            with tqdm_joblib(
                tqdm(desc="Recovering Solutions", total=X_in.shape[0])
            ) as progress_bar:
                recover_resp = parallel(
                    delayed(self.model.run)(*X) for X in recover_inp
                )

        F, G = [], []

        for resp in recover_resp:
            f, g = [], []
            for k_f in self.obj:
                f.append(resp[k_f.name])

            for k_g in self.cst:
                g.append(resp[k_g.name])

            F.append(f)
            G.append(g)

        F, G = np.array(F), np.array(G)

        self.recovered_out = np.hstack([X_in, F, G])

        return self.recovered_out


    def sample(self):
        """
        Sample to train the surrogate model. 

        Returns:
            None.

        """
        set_levels = self.design_space.sets[self.set_id].parameter_levels_list

        self.n_train_samples = max([int(10 * len(self.l)), 64])
        self.n_test_samples = max([int(0.20 * self.n_train_samples), 32])

        self.train_samples = Sobol(
            len(self.var), seed=42 if self.debug else None
        ).random_base2(ceil(np.log2(self.n_train_samples)))
        self.test_samples = LatinHypercube(
            len(self.var), seed=42 if self.debug else None
        ).random(self.n_test_samples)

        # setup the training samples
        for i_par in range(len(self.var)):
            tmp = self.train_samples[:, i_par]

            if isinstance(self.var[i_par], ContinousParameter):
                # Continous Parameter
                lb, ub = self.var[i_par].get_level_bounds(set_levels[i_par])

                tmp *= abs(ub - lb)
                tmp += lb
            else:
                # Discrete Parameter
                tmp *= self.var[i_par].n_levels
                np.trunc(tmp, tmp)

        # setup the test samples
        for i_par in range(len(self.var)):
            tmp = self.test_samples[:, i_par]

            if isinstance(self.var[i_par], ContinousParameter):
                # Continous Parameter
                lb, ub = self.var[i_par].get_level_bounds(set_levels[i_par])

                tmp *= abs(ub - lb)
                tmp += lb
            else:
                # Discrete Parameter
                tmp *= self.var[i_par].n_levels
                np.trunc(tmp, tmp)

        cols = self.train_samples.shape[1]

        train_inp = zip(*[self.train_samples[:, i] for i in range(cols)])
        test_inp = zip(*[self.test_samples[:, i] for i in range(cols)])

        # Parallelized sampling

        with Parallel(n_jobs=-1, timeout=t_max) as parallel:
            with tqdm_joblib(
                tqdm(desc="Generating Train Data", total=self.n_train_samples)
            ) as progress_bar:
                train_resp = parallel(delayed(self.model.run)(*X) for X in train_inp)

            with tqdm_joblib(
                tqdm(desc="Generating Test Data", total=self.n_test_samples)
            ) as progress_bar:
                test_resp = parallel(delayed(self.model.run)(*X) for X in test_inp)

        # Responses are ordered in OBJ first and CON second, in the order
        # they are stored in the list.
        self.order_resp = [o.name for o in self.obj] + [c.name for c in self.cst]

        self.train_Y_samples = np.array(
            [[out[k] for k in self.order_resp] for out in train_resp]
        )
        self.test_Y_samples = np.array(
            [[out[k] for k in self.order_resp] for out in test_resp]
        )

        # setup the scaler
        self.X_scaler = MinMaxScaler().fit(self.train_samples)
        self.y_scaler = MinMaxScaler().fit(self.train_Y_samples)

        self.X_train = self.X_scaler.transform(self.train_samples)
        self.Y_train = self.y_scaler.transform(self.train_Y_samples)

        self.X_test = self.X_scaler.transform(self.test_samples)
        self.Y_test = self.y_scaler.transform(self.test_Y_samples)

    def train_model(self):
        """
        Train the surrogate model.

        Returns:
            None.

        """
        if self.kernel == "matern":
            kern = Matern()
        else:
            kern = ConstantKernel() * RBF() + ConstantKernel()

        base_model = GPR(
            kernel=kern,
            n_restarts_optimizer=20,
            random_state=42 if self.debug else None,
        )

        self.sr_model = MultiOutputRegressor(base_model, n_jobs=-1)
        self.sr_model.fit(self.X_train, self.Y_train)

        y_pred = self.sr_model.predict(self.X_test)

        self.mse = mean_squared_error(self.Y_test, y_pred)
        self.mape = mean_absolute_percentage_error(self.Y_test, y_pred)
        self.r2 = r2_score(self.Y_test, y_pred)

        print(
            f"Model Trained: val_mse {self.mse:.3e}, val_mape {100*self.mape:.2f}%, r2 {self.r2:.3f}"
        )


class DirectOpt(Problem):
    '''
    Class that represents a deterministic optimisation problem, with direct function evaluation. 
    This class wraps the pymoo.core.problem.Problem class.
    
    Attributes:
        design_space (pdopt.data.DesignSpace): 
            The design space object of the problem.
        var (list[pdopt.data.Parameter]): 
            List containing the input parameter objects.
        obj (list[pdopt.data.Objective]): 
            List containing the objective objects.
        cst (list[pdopt.data.Constraint]): 
            List containing the constraint objects.
        model (pdopt.data.Model):
            Model object containing the evaluation function.
        set_id (int):
            id of the set which the problem is defined on.
        x_mask (list[float]):
            Mask to filter out the discrete paramters, as these are
            constant in the set and not modified by the optimiser.
        l (list[float]):
            Lower bounds of the continous input parameters.
        u (list[float]):
            Upper bounds of the continous input parameters.
    '''    
    def __init__(self, model, design_space, set_id, **kwargs):
        """
        Initialise the DirectOpt optimisation problem object.

        Args:
            model (pdopt.data.Model):
                Model object containing the evaluation function.
            design_space (pdopt.data.DesignSpace): 
                The design space object of the problem.
            set_id (int):
                id of the set which the problem is defined on.
            debug (bool, optional): 
                Fix the random generator seed for testing purposes. Defaults to False.

        Returns:
            None.

        """
        self.model = model  # store reference to model
        self.design_space = design_space

        self.var = self.design_space.parameters
        self.obj = self.design_space.objectives
        self.cst = self.design_space.constraints

        self.set_id = set_id
        set_levels = self.design_space.sets[self.set_id].parameter_levels_list

        # Construct the array with lower bounds and upper bounds for each
        # input variable. Discrete variables are removed as they are fixed.
        self.x_mask = []
        self.l, self.u = [], []

        for i_par in range(len(self.var)):
            if isinstance(self.var[i_par], ContinousParameter):
                # Continous Parameter
                self.x_mask.append("c")
                lb, ub = self.var[i_par].get_level_bounds(set_levels[i_par])

                self.l.append(lb)
                self.u.append(ub)

            else:
                # Discrete parameters are fixed within the set
                self.x_mask.append(set_levels[i_par])

        super().__init__(
            n_var=len(self.l),
            n_obj=len(self.obj),
            n_constr=len(self.cst),
            xl=np.array(self.l),
            xu=np.array(self.u),
            elementwise_evaluation=False,
            **kwargs,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluation function for the pymoo optimisation algorithms using the full evaluation function.

        Args:
            X (numpy.ndarray): Input values to be evaluated.
            out (dict[str, numpy.ndarray]): A dictionary containing the evaluated objectives and constraint function.

        Returns:
            None.

        """
        def single_run(x):
            # if len(x) < 2:
            #     x = x[0]

            in_x = []
            i_tmp = 0

            # Reconstruct the full input with the discrete variables
            for par in self.x_mask:
                if par == "c":
                    in_x.append(x[i_tmp])
                    i_tmp += 1
                else:
                    in_x.append(par)

            Y = self.model.run(*in_x)

            # Objectives must be of the minimise form
            f_list = []

            for objective in self.obj:
                f = Y[objective.name] * objective.get_operand()
                f_list.append(f)

            # Constraints must be of the less than 0 form
            g_list = []

            for constraint in self.cst:
                tmp = Y[constraint.name]
                op, val = constraint.get_constraint()

                # g(x) < K ->  g(x) - K < 0
                if op == "lt" or op == "let":
                    g = tmp - val

                # g(x) > K ->  0 > K - g(x)
                else:
                    g = val - tmp

                g_list.append(g)
            return f_list, g_list

        output = []

        with Parallel(n_jobs=-1, timeout=t_max) as parallel:
            output = parallel(delayed(single_run)(x) for x in X)

        f_list = [output[i][0] for i in range(len(output))]
        g_list = [output[i][1] for i in range(len(output))]

        out["F"] = np.array(f_list)
        out["G"] = np.array(g_list)


class Optimisation:
    '''
    Class for the object that performs the search within the surviving design sets. 
    Keyword arguments that can be passed are the termination criteria hyperparmeters 
    used in the pymoo library, along with the population size argument of the UNSGA3 algorithm.
    
    Attributes:
        design_space (pdopt.data.DesignSpace): 
            The design space object of the problem.
        model (pdopt.data.Model):
            Model object containing the evaluation function.
        valid_sets_idx (list[int]):
            List of the ids of the sets that were not discarded.
        use_surrogate (bool):
            If a surrogate model has been used in the search phase.
        ref_dirs (numpy.ndarray):
            Reference directions for NSGA3. See: https://pymoo.org/misc/reference_directions.html.
        algorithm (pymoo.UNSGA3):
            The UNSGA3 Algorithm object.
        termination (pymoo.DefaultMultiObjectiveTermination):
            The termination criterion used.
        set_id (int):
            id of the set which the problem is defined on.
        x_mask (list[float]):
            Mask to filter out the discrete paramters, as these are
            constant in the set and not modified by the optimiser.
        l (list[float]):
            Lower bounds of the continous input parameters.
        u (list[float]):
            Upper bounds of the continous input parameters.
    ''' 
    
    
    def __init__(
        self,
        design_space,
        model,
        save_history=False,
        use_surrogate=True,
        use_nn=False,
        gp_kern="matern",
        debug=False,
        **kwargs,
    ):
        """
        Initialise the Optimisation object.

        Args:
            design_space (pdopt.data.DesignSpace): 
                The design space object of the problem.
            model (pdopt.data.Model):
                Model object containing the evaluation function.
            save_history (bool, optional): 
                Save history of the evolution. Defaults to False.
            use_surrogate (bool, optional): 
                Use a surrogate model in the optimisation. Defaults to True.
            use_nn (bool, optional): 
                If to use the neural network surrogate. Defaults to False.
            gp_kern (str, optional): 
                Type of Gaussian Process kernel. Available modes are "matern" and "rbf". Defaults to "matern".
            debug (bool, optional): 
                Fix the random generator seed for testing purposes. Defaults to False.
            **kwargs:
                Optional arguments for introducing termination criteria.

        Returns:
            None.

        """        
        # Construct the PyMOO problems for surviving design spaces
        self.design_space = design_space
        self.model = model
        self.valid_sets_idx = []

        self.use_surrogate = use_surrogate
        self.debug = debug

        if use_surrogate:
            print("Using Local Surrogate Optimisation")
        else:
            print("Using Direct Function Evaluation")

        for i_set in range(len(design_space.sets)):
            if not design_space.sets[i_set].is_discarded:
                if use_surrogate:
                    if use_nn:
                        opt_prob = NNSurrogate(
                            self.model, design_space, i_set, debug=self.debug
                        )
                    else:
                        opt_prob = KrigingSurrogate(
                            self.model,
                            design_space,
                            i_set,
                            kernel=gp_kern,
                            debug=self.debug,
                        )
                else:
                    opt_prob = DirectOpt(
                        self.model, design_space, i_set, debug=self.debug
                    )

                self.design_space.sets[i_set].set_optimisation_problem(opt_prob)

                if self.design_space.sets[i_set].opt_done is None:
                    self.design_space.sets[i_set].opt_done = False

                self.valid_sets_idx.append(i_set)

        n_partitions = kwargs["n_partitions"] if "n_partitions" in kwargs else 12

        self.ref_dirs = get_reference_directions(
            "das-dennis", len(design_space.objectives), n_partitions=n_partitions
        )

        # Define the algorithm hyperparameters
        pop_size = (
            kwargs["pop_size"] if "pop_size" in kwargs else 10 + self.ref_dirs.shape[0]
        )

        # # Define the algorithm hyperparameters
        # s_history = kwargs['save_history'] if 'save_history' in kwargs else False

        self.algorithm = UNSGA3(
            pop_size=pop_size,
            ref_dirs=self.ref_dirs,
            eliminate_duplicates=True,
            save_history=save_history,
            seed=42 if self.debug else None,
        )

        # Define the termination hyperparameters
        x_tol = kwargs["x_tol"] if "x_tol" in kwargs else 1e-16

        cv_tol = kwargs["cv_tol"] if "cv_tol" in kwargs else 1e-16

        f_tol = kwargs["f_tol"] if "f_tol" in kwargs else 1e-16

        n_max_gen = kwargs["n_max_gen"] if "n_max_gen" in kwargs else 1e3

        n_max_evals = kwargs["n_max_evals"] if "n_max_evals" in kwargs else 1e6

        self.termination = DefaultMultiObjectiveTermination(
            xtol=x_tol,
            cvtol=cv_tol,
            ftol=f_tol,
            period=5,
            n_max_gen=n_max_gen,
            n_max_evals=n_max_evals,
        )

    def run(self, folder=None):
        """
+        Run the search phase.

        Args:
            folder (str, optional): 
                Path where to save temporarely the DesignSpace object between set optimisation runs. Defaults to None.

        Returns:
            None.

        """

        
        nopts = len(self.valid_sets_idx)
        print(
            f"Beginning of Optimisation Run of {nopts} sets out of {len(self.design_space.sets)}"
        )

        # tqdm(self.valid_sets_idx, desc='Searching in the Design Space'):
        for n_set, i_set in enumerate(self.valid_sets_idx):
            print(f"\nRunning in set {i_set} ({n_set} of {nopts})")

            if self.design_space.sets[i_set].opt_done:
                print("Opt Done, skipping")

            else:
                t0 = time()

                if self.use_surrogate:
                    self.design_space.sets[i_set].optimisation_problem.sample()
                    self.design_space.sets[i_set].optimisation_problem.train_model()
                    self.design_space.sets[i_set].use_surrogate = True

                    res = minimize(
                        self.design_space.sets[i_set].optimisation_problem,
                        self.algorithm,
                        termination=self.termination,
                        verbose=True,
                    )

                    self.design_space.sets[i_set].optimisation_results_raw = res
                    # Reconstruct the full input with the discrete variables

                    if res.X is not None:
                        recover_out = self.design_space.sets[
                            i_set
                        ].optimisation_problem.recover_pts(res.X)
                    else:
                        recover_out = np.array([])

                    self.design_space.sets[i_set].optimisation_results = recover_out

                else:
                    self.design_space.sets[i_set].use_surrogate = False
                    res = minimize(
                        self.design_space.sets[i_set].optimisation_problem,
                        self.algorithm,
                        termination=self.termination,
                        verbose=True,
                    )

                    self.design_space.sets[i_set].optimisation_results_raw = res
                    self.design_space.sets[i_set].optimisation_results = res

                dt = time() - t0

                self.design_space.sets[i_set].opt_done = True
                self.design_space.sets[i_set].opt_dt = dt

                print(f"Total Time {dt:.3f} s")
                
                if folder:
                    pk.dump(self.design_space, open(folder + "/design_space.pk", "wb"))


# Code if imported
if __name__ == "__main__":
    print("pdopt.Optimisation")
