# PDOPT: Probabilistic Design space exploration and OPTimisation

A `python` framework for set-based design space exploration without explicit elimination rules. It impements a set-based approach for mapping the requirements on the design space, using a probabilistic surrogate model trained on the provided design model. This procedure ensures to identify the best candidate areas of the design space with the minimum number of assumptions of the design of the system.

The framework process follows two steps: 

- **Exploration Phase**. After breaking down the design space into sets (i.e. sub-spaces), the code evaluates the probability of satisfying the constraints there. Sets that have a low probability are eliminated.
- **Search Phase**. Surviving sets are further explored by introducing a local optimisation algorithm (GA-based), and recovering the locally optimal design points.

The final output is an aggregation of locally optimal points which constitute a rich database of design alternatives capable to minimally satisfy the requirements set by the user. Engineers can use this framework for analysing the behavior of complex simulation models for the purpose of designing them, leveraging on set-based design for reducing the design space to the feasible and desirable one.

This online user guide provides a description of the usage, an example analysis, and API reference.


## Installation

First `git clone` the repository or download the zip file.
Then run the setup from the root folder:

```
pip install -e .
```

Use the `--user` option if not running in admin mode.


## Community Guidelines

This software is released with the MIT license and comes as is, without warranty of any kind. It is maintained by the author, but anyone is free to contribute. Users can open a ticket on GitHub to provide feeback on bugs or contributions.

## Acknowledgements

This software was developed within [Project FutPrInt50](https://futprint50.eu/), with EU Horizon 2020 Grant No. 875551. The authors wants to thank all the researchers in the project who cotnributed with their input to shape the framework.

## Licensing

Copyright (c) 2021 Cranfield University. This software is released under the permissive MIT License.


## References

- Spinelli A., Balaghi Enalou H., Zaghari B., Kipouros T., and Laskaridis P. (2022). "Application of Probabilistic Set-Based Design Exploration on the Energy Management of a Hybrid-Electric Aircraft". Aerospace, vol. 9, no. 3: 147. [link](https://doi.org/10.3390/aerospace9030147)

- Georgiades A., Sharma S., Kipouros T., & Savill M. (2019). "ADOPT: An augmented set-based design framework with optimisation". Design Science, 5, e4. [link](https://doi.org/10.1017/dsj.2019.1)

- Singer D.J., Doerry N. and Buckley M.E. (2009), "What Is Set-Based Design?". Naval Engineers Journal, vol 121: 31-43. [link](https://doi.org/10.1111/j.1559-3584.2009.00226.x)

- Blank J. and Deb K. (2020). "Pymoo: Multi-Objective Optimization in Python," in IEEE Access, vol. 8, pp. 89497-89509. [link](https://doi.org/10.1109/ACCESS.2020.2990567)

