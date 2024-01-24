# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:26:24 2024

@author: s345001
"""

import pytest
import pdopt.data as data
import pdopt.exploration as exploration
import pdopt.optimisation as optimisation
import pdopt.tools as tools
import numpy as np
import pandas as pd
import os
import re


def dummy_function(x, y):
    return {
        "obj": (x - 5) ** 2 - y * 5,
        "con1": (x - 5) ** 2 - y * 5,
        "con2": 11 - 0.5 * x**2,
    }


def test_is_pareto_efficient():
    costs = np.random.default_rng(42).random(size=(10, 2))

    expected_pareto = np.array(
        [
            [0.09417735, 0.97562235],
            [0.12811363, 0.45038594],
            [0.4434142, 0.22723872],
            [0.55458479, 0.06381726],
        ]
    )

    pareto = costs[tools.is_pareto_efficient(costs)]

    assert (expected_pareto.round(5) == pareto.round(5)).all()


@pytest.mark.filterwarnings("ignore: lbfgs failed")
@pytest.mark.filterwarnings("ignore: invalid")
def test_generate_run_report():
    my_DesignSpace = data.DesignSpace.from_csv(
        "test_files/test_input.csv", "test_files/test_response.csv"
    )
    my_Model = data.Model(dummy_function)

    my_Exploration = exploration.ProbabilisticExploration(
        my_DesignSpace, my_Model, debug=True
    )

    my_Exploration.run()
    my_Optimisation = optimisation.Optimisation(
        my_DesignSpace,
        my_Model,
        use_surrogate=False,
        use_nn=False,
        gp_kern="gpr",
        debug=True,
    )

    my_Optimisation.run(".")

    ## Extract the information used in the report for cross-checking

    surrogate_time = {}
    surrogate_score = {}

    for k in my_Exploration.surrogates:
        surrogate_time.update({k: my_Exploration.surrogates[k].train_time})
        surrogate_score.update({k: my_Exploration.surrogates[k].score})

    exp_time = my_Exploration.run_time

    optimisation_time = {}

    for i_set in my_Optimisation.valid_sets_idx:
        if my_DesignSpace.sets[i_set].optimisation_results is not None:
            optimisation_time.update({i_set: my_DesignSpace.sets[i_set].opt_dt})

    total_surrogate_time = sum([surrogate_time[k] for k in surrogate_time])
    total_optimisation_time = sum([optimisation_time[k] for k in optimisation_time])

    out = tools.generate_run_report(
        "test_files/test_report.txt", my_DesignSpace, my_Optimisation, my_Exploration
    )

    expected1 = f"""Total Number of Sets      : {len(my_DesignSpace.sets)}
Number of Surviving Sets  : {len(my_Optimisation.valid_sets_idx)}\n
Total Surrogate Train Time : {total_surrogate_time:>12.3f} s
Total Exploration Time     : {exp_time:>12.3f} s
Total Search Time          : {total_optimisation_time:>12.3f} s
Number of Cores Used       : {os.cpu_count():>12d}
"""

    expected2 = "\n".join(
        ["Train time and score of each Surrogate:"]
        + [
            "{:>10}{:>10.4f}(s) {:>10.4f}".format(
                k, surrogate_time[k], surrogate_score[k]
            )
            for k in surrogate_time
        ]
    )

    expected3 = "\n".join(
        ["Search time and f_evals of each Set:"]
        + [
            "{:>10}{:>10.4f}(s)".format(k, optimisation_time[k])
            for k in optimisation_time
        ]
    )

    assert expected1 in out
    assert expected2 in out
    assert expected3 in out
