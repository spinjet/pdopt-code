# PDOPT: Probabilistic Design space exploration and OPTimisation

A `python` framework for set-based design space exploration without explicit elimination rules. It impements a set-based approach for mapping the requirements on the design space, using a probabilistic surrogate model trained on the provided design model. This procedure ensures to identify the best candidate areas of the design space with the minimum number of assumptions of the design of the system.


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

## References

- Spinelli A., Balaghi Enalou H., Zaghari B., Kipouros T., and Laskaridis P. (2022). "Application of Probabilistic Set-Based Design Exploration on the Energy Management of a Hybrid-Electric Aircraft". Aerospace, vol. 9, no. 3: 147. [link](https://doi.org/10.3390/aerospace9030147)

- Georgiades A., Sharma S., Kipouros T., & Savill M. (2019). "ADOPT: An augmented set-based design framework with optimisation". Design Science, 5, e4. [link](https://doi.org/10.1017/dsj.2019.1)

- Singer D.J., Doerry N. and Buckley M.E. (2009), "What Is Set-Based Design?". Naval Engineers Journal, vol 121: 31-43. [link](https://doi.org/10.1111/j.1559-3584.2009.00226.x)

- Blank J. and Deb K. (2020). "Pymoo: Multi-Objective Optimization in Python," in IEEE Access, vol. 8, pp. 89497-89509. [link](https://doi.org/10.1109/ACCESS.2020.2990567)

