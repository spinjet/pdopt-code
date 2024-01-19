# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 14:52:21 2021

Functions used in PDOPT for different scopes

"""

# Standard Library Imports
from itertools import product

# Third-party imports
import os
import numpy as np
import pandas as pd

# Local imports

# Module Constants
__author__ = 'Andrea Spinelli'
__copyright__ = 'Copyright 2021, all rights reserved'
__status__ = 'Development'

# Module Functions


def generate_surrogate_training_data(par_csv, fun):

    par_df = pd.read_csv(par_csv)
    N_in = len(par_df)
    lev_list = [np.linspace(par_df['lb'][i], par_df['ub']
                            [i], par_df['levels'][i]) for i in range(N_in)]
    doe = [x for x in product(*lev_list)]
    resp = [fun(*des) for des in doe]

    out_keys = list(resp[0].keys())
    in_keys = list(par_df['name'])

    temp = {key: [] for key in (in_keys + out_keys)}

    for des in doe:
        for i in range(len(in_keys)):
            temp[in_keys[i]].append(des[i])

    for res in resp:
        for i in range(len(out_keys)):
            temp[out_keys[i]].append(res[out_keys[i]])

    return pd.DataFrame(temp)


def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep any point with a lower cost
            is_efficient[is_efficient] = np.any(
                costs[is_efficient] < c, axis=1)
            is_efficient[i] = True  # And keep self
    return is_efficient


def generate_run_report(file_directory, design_space, optimisation, exploration):
    with open(file_directory, 'w') as f:
        surrogate_time = {}
        surrogate_score = {}

        for k in exploration.surrogates:
            surrogate_time.update({k: exploration.surrogates[k].train_time})
            surrogate_score.update({k: exploration.surrogates[k].score})

        exp_time = exploration.run_time

        optimisation_time = {}
        for i_set in optimisation.valid_sets_idx:
            if design_space.sets[i_set].optimisation_results is not None:
                optimisation_time.update(
                    {i_set: design_space.sets[i_set].opt_dt})

        total_surrogate_time = sum([surrogate_time[k] for k in surrogate_time])
        total_optimisation_time = sum(
            [optimisation_time[k] for k in optimisation_time])

        out1 = '''Total Number of Sets      : {}
Number of Surviving Sets  : {}\n
Total Surrogate Train Time : {:>12.3f} s
Total Exploration Time     : {:>12.3f} s
Total Search Time          : {:>12.3f} s
Number of Cores Used       : {:>12d}
'''.format(len(design_space.sets), len(optimisation.valid_sets_idx),
           total_surrogate_time, exp_time, total_optimisation_time, os.cpu_count())

        out2 = '\n'.join(['Train time and score of each Surrogate:'] +
                         ['{:>10}{:>10.4f}(s) {:>10.4f}'.format(k, surrogate_time[k], surrogate_score[k]) for k in surrogate_time])

        out3 = '\n'.join(['Search time and f_evals of each Set:'] +
                         ['{:>10}{:>10.4f}(s)'.format(k, optimisation_time[k]) for k in optimisation_time])

        out = out1 + '\n' + out2 + '\n\n' + out3

        print(out)
        f.write(out)
        
        return out
