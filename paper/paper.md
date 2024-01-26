
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

Contemporary engineering systems are characterised by many components and complex interactions between them. The design of such systems entails high uncertainty due to the large number of parameters defining it. An approach to manage it is exploring and evaluating as many alternatives as possible, before committing to a specific solution. The Python package `PDOPT` aims to provide this capability without the high computational cost associated with the factorial-based design of experiments methods. By exploiting a probabilistic machine learning model, the code identifies the areas of the design space which are most promising for the requirements provided by the user. The result is a large number of feasible design points, aiding the designer in understanding the behaviour of the system under design and selecting the desired configuration for further development.

# State of the field

Engineering design is a process where needs, quantitative and qualitative, are transformed into a set of specifications (e.g. geometry, materials, component list, etc) such that they can be sufficiently satisfied. The procedure involves performing several analyses of the system under design through suitable modelling tools, which map the design parameters to the quantified needs, and carrying out decisions to narrow and identify the range of desirable parameters. To aid this procedure, the designer relies on computer simulations through optimisation methods or the design of experiments in the design space.

Historically, optimisation has been used in the late stages of design, to refine a solution already in the ballpark of the requirements. However, with the development of Multi-disciplinary Optimisation (MDO), there has been increasing interest in bringing optimisation in earlier stages of the design process. Searching for optimal designs with concurrent analysis of different disciplines allows one to obtain better results than with sequential optimisation alone, reducing the time and cost of the development cycle [@MartinsLambe2013].
In the conceptual phase, optimisation can be employed to understand which combination of input design parameters enables to satisfy the requirements. The algorithm acts by changing the input quantities iteratively to find the ideal combination(s) that minimise/maximise quantities of interest (often performance targets) without violating the problem constraints (which can be technological, geometrical, economical, and so on). Population-based solvers such as Genetic Algorithm [@DebNSGA:2002] or Particle Swarm [@Kennedy1995] are best suited for this task as they evolve multiple design points in parallel, rather than iteratively improving a single solution [@Deb2008]. The result is a family of points, covering the design space which are optimal. Surrogate modelling [@Forrester2009] is often introduced to reduce the computational cost of each functional evaluation and correct with high-fidelity data fast low-fidelity models often used in the conceptual design phase [@Kontogiannis2020].

Software libraries developed for optimisation in Python are the OpenMDAO framework, which focuses on multi-disciplinary optimisation [@Gray2019a], the Object-Oriented `pyOpt` [@Pyopt], the `DEAP` evolutionary optimisation library [@DEAP], and `pymoo` multi-objective optimisation library [@pymoo]. These libraries are open-source and general-purpose.

While effective in finding the desired set of design points, optimisation methods are required to properly set up the problem to deliver meaningful results. This may be difficult at the beginning of the design procedure, as needs may be not fully defined, and the behaviour of the model not be fully understood. Design of Experiments methods can assist in initiating this process through sampling and evaluation of the design space to obtain an understanding of the system response. Traditionally this approach is carried out through factorial sampling, where all possible combinations of extreme parameter values are tested [@MontgomeryDE]. A response surface would then be built for analysing the response of the system inside the interval. This approach is feasible when the number of variables is low, as the number of experiments to be carried out grows exponentially. In the computational engineering field, quasi-random sampling is preferred, with Latin Hypercube sampling and Sobol sequences being most used [@Yondo2018]. These allow for training surrogate models for design exploration with a lower number of points in cases where factorial design would require an unfeasible amount.

Examples of libraries useful for performing the design of experiments are the Quasi-Monte Carlo sampling module `scipy.qmc` and the University of Michigan’s Surrogate Modelling Toolbox [@saves2024smt]. Unlike an optimisation problem, there is no fixed procedure to obtain the mapping of the requirements on the input parameters. The designer is free to play with the model, with the aid of visualisation methods to understand the problem at hand.

The framework presented in this paper attempts to combine the two approaches in a single framework through the application of the principles of Set-Based Design and Bayesian probability. Set-based design is a design practice that focuses on narrowing down the input design space by eliminating the candidate designs that do not satisfy the needs and requirements [@singer2009set].
(WIP: add some background on SBD and PDOPT development).

# Statement of need

`PDOPT`, short for Probabilistic Design and OPTimisation, is a Python package for design space exploration of systems under design. It implements a set-based approach for mapping the requirements to the design space, using a probabilistic surrogate model trained on the provided design model. This procedure ensures to identification of the best candidate areas of the design space with the minimum number of assumptions of the design of the system.

The API of `PDOPT` was designed as a library with class-based interfaces between the components of the framework. This ensures both flexibility and transparency, as the user can inspect the main data structure between the phases of the framework. A full `PDOPT` analysis consists of two phases: the Exploration phase and the Search phase. The first one surveys the design space to identify the areas that are most likely to satisfy the constraints over the quantities of interest of the model. This is carried out by breaking down the design space into a hypercube of parameters’ levels (named sets), and rapidly evaluating them with the probabilistic surrogate model. The second phase introduces a multi-objective optimisation problem in each surviving design space area for recovering the individual design points. The result is multiple local Pareto fronts, one for each set.

The aggregation of these design points yields the global Pareto front with feasible suboptimal points. Interactive visualisation tools can be used to analyse the results and proceed with design selection. Thanks to the probabilistic mapping of the requirements to the design space, the computational cost for design space exploration can be reduced by up to 80% [@SpinelliEASN:2021].

`PDOPT` is intended to be used by researchers and engineers alike in developing complex engineering systems. It has been developed within the FutPrint50 project [@fp50] and released as open-source software under the MIT license. The software has been used in several scientific publications regarding the design of hybrid-electric aircraft [@SpinelliMDPI:2022], and the effects of operating conditions [@SpinelliEASN:2022] and technological uncertainty [@SpinelliAIAA:2023] on the design.


# Theoretical Background

The framework presented in this is built upon Set-Based Design and ... (WIP)

# Availability

`PDOPT` can be found on GitHub [@pdopt_repo] and is compatible with the latest Python release. The release includes a PDF manual as a user guide and API reference. An example test set-up is also provided in the GitHub repository. Dependencies include the standard Python scientific stack (`numpy`, `scipy`, `pandas`, `matplotlib`) with the addition of the `scikit-learn` machine learning library, the `pymoo` multi-objective optimisation framework [@pymoo], and the `joblib` parallelisation library. As an optional feature, `plotly` can be installed to take advantage of the prototypical decision-making environment packaged with the library.   

# Acknowledgements

The authors thank the FutPrInt50 researchers for their collaboration and contributions
to the software and test case applications. This project has received funding from the European Union’s Horizon 2020 Research and Innovation program under Grant Agreement No 875551.

# References