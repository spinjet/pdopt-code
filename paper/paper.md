---
title: 'PDOPT: A Python library for Probabilistic Design space exploration and OPTimisation.'
tags:
  - Python
  - Computational Engineering
  - Design Space Exploration
  - Set-Based Design
  - Design Uncertainty
  - Multi-Objective Optimization
authors:
  - name: Andrea Spinelli
    orcid: 0000-0002-3064-3387
	  corresponding: true
    affiliation: 1
  - name: Timoleon Kipouros
    orcid: 0000-0003-3392-283X
    affiliation: 1
affiliations:
 - name: Centre for Propulsion and Thermal Power, Cranfield University, MK430AL, UK
   index: 1
date: 22 August 2023
bibliography: paper.bib
---

# Summary

Contemporary engineering systems are charaterised by many components and complex interactions between them. The design of such systems entails high uncertainty due to the large number of parameters defining it. An approach to manage it is exploring and evaluating as many alternatives as possible, before committing to a specific solution. The Python package `PDOPT` aims to provide this capability without the high computational cost associated with factorial-based design of experiments methods. By exploiting a probabilistic machine learning model, the code identifies the areas of the design space which are most promising for the requirements provided by the user. The result is a large amount of feasible design points, aiding the designer in understanding the behaviour of the system under design and selecting the desired configuration for furhter development. 

# Statement of need

`PDOPT`, short for Probabilistic Design and OPTimisation, is a Python package for design space exploration of systems under design. It impements a set-based approach for mapping the requirements on the design space, using a probabilistic surrogate model trained on the provided design model. This procedure ensures to identify the best candidate areas of the design space with the minimum number of assumptions of the design of the system. 

The API of `PDOPT` was designed as a library with class-based interfaces between the  components of the framework. This ensures both flexibility and transparency, as the user can inspect the main data structure between the phases of the framework. A full `PDOPT` test case consists of two phases: the Exploration phase and the Search phase. The first one surveys the design space to identify the areas that are most likely to satisfy the constraints over the quantities of interest of the model. This is carried out by breaking down the design space into an hypercube of parameters's levels (named sets), and rapidly  evaluating them with the probabilistic surrogate model. The second phase introduces a multi-objective optimisation problem in each surviving design space area for recovering  the individual design points. The result is multiple local Pareto fronts, one for each set. 
The aggregation of these design points yields the global Pareto front with feasible  suboptimal points. Interactive visualisation tools can be used to analyse the results and proceed with design selection. Thanks to the Exploration phase, the computational cost for design space exploration can be reduced up to 80% [@SpinelliEASN:2021].

`PDOPT` is intended to be used by researchers and engineers alike in developing complex  engineering systems. It has been developed within the FutPrint50 project [@fp50] andreleased as open source software under the MIT license. The software has been used in several scientific publications regarding the design of hybrid-electric aircraft [@SpinelliMDPI:2022], and the effects of operating conditions [@SpinelliEASN:2022] and technological uncertainty [@SpinelliAIAA:2023] on the design. 

# Availability

`PDOPT` can be found on GitHub [@pdopt_repo] and is compatible with the latest Python release. The release includes a PDF manual as user guide and API reference. An example test set-up is also provided in the GitHub repository. Dependencies include the standard Python scientific stack (`numpy`, `scipy`, `pandas`, `matplotlib`) with the addition of the `scikit-learn` machine learning library, the `pymoo` multi-objective optimisation framework, and the `joblib` parallelisation library. As an optional feature, `plotly` can be installed to take advantage of the prototypical decision making environment packaged with the library.   

# Acknowledgements

The authors thank the FutPrInt50 researchers for their collaboration and contributions
to the software and test case applications. This project has received funding from the European Union’s Horizon 2020 Research and Innovation program under Grant Agreement No 875551.

# References