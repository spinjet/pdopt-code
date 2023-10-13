# Example PDOPT Case

The example case is one of the experiments carried out in the paper !["Application of Probabilistic Set-Based Design Exploration on the Energy Management of a Hybrid-Electric Aircraft
"](https://www.mdpi.com/2226-4310/9/3/147). It explores the energy management strategy of a parallel hybrid-electric aircraft, defined as the degree of hybridisation at four points in the mission: beginning of climb, end of climb, beginning of cruise, and end of cruise. These are shown in the following table:

| Parameter | Lower Bound | Upper Bound | Number of levels |
|-----------|-------------|-------------|------------------|
| climb_h0  | 0           | 1           | 4                |
| climb_h1  | 0           | 1           | 4                |
| cruise_h0 | 0           | 1           | 4                |
| cruise_h1 | 0           | 1           | 4                |

Objective of the probelem is minimising fuel consumption and NOx emissions. This improvement is caused by the adoption of batteries as source of electrical power, which cause an increment of take-off mass.
As the aeroplane is not re-designed (the reference is an ATR-42), the take-off muss must be constrainted to the maximum allowable of the original design. In this problem, this has been set to 20'000 kg. 

Input parameters are specified in the `input.csv` file, while the definition of the quantities of interest (objectives and constrained quantities) are in the `response.csv` file.

The example can be run by executing the script in `energy_management_experiment.py`. User should expect the exploration phase to last a few minutes, while the search phase last about 20 minutes or longer (depending on the number of cores).
The output will be stored in two .csv files, `exp_results.csv` for the exploration phase and `opt_results.csv` for the optimisation phase. A `report.txt` file is produced documenting the number of surviving sets and the time required to perform the process. 
Refer to the tutorial in the documentation for details on the output and setting up the run script.

Two variants are provided: `example_1` uses direct functional evaluation in the search phase, while `example_2` employs surrogate modelling to speed up the search operation.
