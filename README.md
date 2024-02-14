# Probabilistic Design and OPTimisation framework (PDOPT)

A python framework for set-based design space exploration without explicit elimination rules. 
It impements a set-based approach for mapping the requirements on the design space, using a probabilistic surrogate model trained on the provided design model. 
This procedure ensures to identify the best candidate areas of the design space with the minimum number of assumptions of the design of the system.

The framework process follows two steps: 

- **Exploration Phase**. After breaking down the design space into sets (i.e. sub-spaces), the code evaluates the probability of satisfying the constraints there. Sets that have a low probability are eliminated.
- **Search Phase**. Surviving sets are further explored by introducing a local optimisation algorithm (GA-based), and recovering the locally optimal design points.

The final output is an aggregation of locally optimal points which constitute a rich database of design alternatives capable to minimally satisfy the requirements set by the user.

## Requisites

The following python libraries are required for installing `PDOPT`. 
These can be automatically installed with the provided setup script.

- numpy
- scipy
- matplotlib
- pandas
- scikit-learn
- pymoo
- joblib

The optional visualisation tool requires the `dash-plotly` package to be installed.

## Installation

The reccomended method is through pyPI by running the command:

```
pip install pdopt
```

Alternatevly it is possible to donwload the release, or `git clone` the repository. Then run the setup from the root folder:

```
pip install -e .
```

Use the `--user` option if not running in admin mode.


## Documentation

Online API documentation with example usage can be found [here](https://pdopt-code.readthedocs.io/en/latest/).

An outdated PDF document with some explaination of the theory behind PDOPT can be found in the docs folder.
Additional information can be found in the following paper: [Application of Probabilistic Set-Based Design Exploration on the Energy Management of a Hybrid-Electric Aircraft
](https://www.mdpi.com/2226-4310/9/3/147)

## Community Guidelines

This software is currently being maintained by me @spin-jet. If you find any bugs, want to contribute or have any questions, you can either open a ticket here on GitHub or send me an email at andrea.spinelli@cranfield.ac.uk 


## Acknowledgements

This software was developed within Project ![FutPrInt50](https://futprint50.eu/), with EU Horizon 2020 Grant No. 875551.
The authors wants to thank all the researchers in the project who cotnributed with their input to shape the framework.

## Licensing

Copyright (c) 2021 Cranfield University. This software is released under the permissive MIT License.

