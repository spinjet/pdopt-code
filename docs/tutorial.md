
# Example PDOPT Case

The files of the example described in this chapter can be found in the [GitHub repoistory](https://github.com/spinjet/pdopt-code/tree/main/example).

## Problem Setup

The problem is a design space exploration of hybrid-electric aircraft energy management strategies, where the goal is to minimise fuel consumption and NOx emissions under a maximum take-off mass constraint. After understanding the problem to analyse, the user should identify the input parameters of the model and the responses he desires to optimise or impose constraints on. For this example, the following input parameters have been identified:

|    Name    |    Type    | Range | Description                                    |
|:----------:|:----------:|:-----:|------------------------------------------------|
|  climb_h0 | continuous |       | The DOH at the beginning of the climb segment  |
|  climb_h1 | continuous |       | The DOH at the end of the climb segment        |
| cruise_h0 | continuous |       | The DOH at the beginning of the cruise segment |
| cruise_h1 | continuous |       | The DOH at the end of the cruise segment       |

The user should decide if these parameters are continuous or discrete, i.e. describe a range of possible values or a set of discrete choices. In the first case, the user should also fix the boundaries of this range. This information will be used in defining the `input.csv` file and the arguments of the evaluation function.

Then the user should list the responses they wish to analyse and define them as objectives or constraints. Special care should be given to objectives, considered "soft constraints" for the exploration phase. This is useful if the user wishes to understand which areas of the design space are guaranteed to provide at most (or least) a certain value. For instance, the user wishes to see which areas of the design space are more likely to have a minimum value of NOx emissions under a certain figure. For the example problem, the following requirements are identified:

|  Name  |    Type    |    Operator    | Value | Description                                 |
|:------:|:----------:|:--------------:|:-----:|---------------------------------------------|
|   Mf   |  objective | min (minimise) |  nan  | Fuel Burnt during the mission               |
| M_NOx  |  objective | min (minimise) |  nan  | NOx Emission during the mission             |
|   TOM  | constraint | lt (less than) | 20000 | Take-off mass, constrainted to 20000 kg     |

These response parameters will then define the output of the evaluation function and the `responses.csv` file.

### Definition of the input .csv files

With the problem broken down into its PDOPT components, it is possible to define the `input.csv` file is as follows:

```csv
name,type,lb,ub,levels,uq_dist,uq_var_l,uq_var_u
climb_h0,continous,0,1,4,nan,nan,nan
climb_h1,continous,0,1,4,nan,nan,nan
cruise_h0,continous,0,1,4,nan,nan,nan
cruise_h1,continous,0,1,4,nan,nan,nan
```

The number of levels for each parameter should be selected with caution. Few levels make a faster analysis, especially during the search phase; however, the exploration phase would likely average the probability, leading to a potential loss of resolution of the decision boundary. 

On the other hand, a high number of levels leads to an unnecessary number of surviving sets, which would considerably slow down the search process. It is advised to use 3 or 4 levels when starting and adjusting it by performing a few exploration steps.

Along the input file, the `response.csv` file is defined as follows:

```csv
name,type,op,val,pSat
Mf,objective,min,nan,nan
M_NOx,objective,min,nan,nan
TOM,constraint,lt,20000,0.5
```

It is possible to include "soft constraints" on the objectives for the exploration phase. It is suggested not to do so at the first run of the problem unless the user is already aware of the range of response values. Soft constraints present a phenomenon of "leakage", whereas some sub-optimal sets are kept as they might have a few points satisfying the constraint.

### Definition of the Evaluation Function

In this case, the evaluation function was developed by extending the `data.ExtendableModel()` class. This was preferred over defining just a Python function for two reasons. First, it is possible to load in the `ExtendableModel()` object external parametric data (for instance, model parameters that are fixed during each run of `PDOPT`). Second, it allows to package with the model `run` function other routines, such as a post-processing function. This is the case for this example problem. It must contain the `run()` method, which must output the data required from the `response.csv` file as a Python dictionary. 

The model class is, therefore, defined as:

```pyhton
class Experiment(ExtendableModel):

def __init__(self, input_parameters, architecture, mission_file):

## This constructor method allows to store in the Experiment
## object some information regarding the aircraft and mission
self.inp        = list(pd.read_csv(input_parameters)['name'])
self.arch       = architecture
self.arch0      = architecture
self.mission    = mission_file

def run(self, *args, **kwargs):

## Omitted code for performing the analysis

output = {
	'TOM'      : analysis.iloc[-1].mass,
	'Mf'       : analysis.iloc[-1].m_fl,
	'M_NOx'    : analysis.iloc[-1].m_NOx,
}

return output

def postprocess_analysis(self, *args, **kwargs):

## Omitted post-processing code, runs the same analysis
## but returns more information

return results
```

### Definition of the Runfile

Now that all the ingredients required to perform the analysis are present, it is possible to construct the Python script to run it.

First a `DesignSpace()` object is created, which will store all the information regarding the problem. It contains a list of all the sets (which are instances of the `DesignSet()` object) and the Pandas DataFrames containing the exploration and search phases results. To ease the creation of the input parameters and responses, the helper functions will be used to load in the two .csv files. To avoid loss of data, the `save_to_pickle()` helper function is used.

```python
architecture = json.load(open("data/architecture.json", "r"))
mission = "data/mission_original.csv"
experiment = Experiment(folder + "/input.csv", architecture, mission)

# Check if a design space is already present otherwise create it
if exists(folder + "/design_space.pk") and restart:
	design_space = DesignSpace.from_pickle(folder + "/design_space.pk")
else:
	design_space = DesignSpace.from_csv(
		folder + "/input.csv", folder + "/response.csv"
	)
	design_space.save_to_pickle(folder + "/design_space.pk")
```

The next step is defining the exploration phase. The `ProbabilisticExploration()` object is used. It requires as arguments the design space object and the model object. Optional arguments include the definition of a surrogate data file, useful to expedite multiple analyses and the number of samples to use. By default, it will use 100 sampled points in the entire design space (using a Latin Hypercube scheme) and 30 points to validate its regression. These points are sampled from the provided model. Once the exploration object is trained, it is run with the `.run()` method, by passing two arguments: the number of samples for evaluating each set and the minimum satisfaction probability.

The results from the exploration phase are stored in the `DesignSpace()` object; they can be retrieved with the `.get_exploration_results()` method. 

```python
# Check if there is already a trained exploration object
if exists(folder + "/exploration.pk") and restart:
	exploration = ProbabilisticExploration.from_pickle(folder + "/exploration.pk")
else:
	exploration = ProbabilisticExploration(
		design_space,
		experiment,
		surrogate_training_data_file=folder + "/samples.csv",
		n_train_points=n_exp_train,
	)

	for k in exploration.surrogates:
		s = exploration.surrogates[k]
		print(f"Surrogate {s.name} with r = {s.score:.4f}")

	exploration.save_to_pickle(folder + "/exploration.pk")

# Check if exploration has been done already
if exists(folder + "/exp_results.csv") and restart:
	pass
else:
	exploration.run(n_exp_samples, P_exploration)
	design_space.save_exploration_results(folder + "/exp_results.csv")

	# Update the saved design object
	design_space.save_to_pickle(folder + "/design_space.pk")
```

It is possible to perform different exploration analyses by using copies of the `DesignSpace()` object and changing the satisfaction probability. This allows one to study how restrictive the requirements are over the design space. Once complete, the exploration step marks some areas of the design space as "Discarded" by setting the `.isDiscarded` boolean of each set as true. If desired, the user can manually change the discarded status of the sets using the `DesignSpace().set_discarded_status()` method.


The following phase is the search step. A multi-objective optimisation algorithm is introduced in the surviving sets to identify the local Pareto front. The `Optimisation()` object is the standard deterministic multi-objective optimiser. The arguments are the design space, model, and options regarding the stopping criteria or the genetic algorithm population size. For the `PDOPT` analysis, it is recommended to use a fixed number of evaluation functions `n_max_evals` or generations `n_max_gen`. This would allow a fair search in every set, as it would ignore the topology of the evaluation function. Instead, using criteria based on convergence would introduce distortions, as some areas of the design space might be shallow (slow convergence) or have multiple local minima/maxima. 

Once set up is complete, optimisation is started with the `Optimisation.run()` method. Once complete, results can be retrieved from the `.get_optimum_results()` method of the design space object. It returns a Pandas DataFrame containing each set's optimal points and the constrained quantities' return value. The optimisation step code is:

```python
optimisation = Optimisation(
	design_space, experiment, n_max_evals=2000, use_surrogate=False
)

# Check if optimisation has been done already
if exists(folder + "/opt_results.csv") and restart:
	pass
else:
	optimisation.run(folder)
	design_space.save_optimisation_results(folder + "/opt_results.csv")

	# Update the saved design object
	design_space.save_to_pickle(folder + "/design_space.pk")

# Runtime Report
generate_run_report(folder + "/report.txt", design_space, optimisation, exploration)

```

The `generate_run_report()` function is used to produce a text output containing information on the total run time and number of discarded sets. 

## Running the Analysis

The file is run now that the full script is ready with the required input files. While the code is running, it will output to the console the current progress. In particular, after training the Gaussian processes, it will output the R^2 score (Coefficient of Determination) of each trained surrogate model. This gives a good indication of the reliability of the exploration process. Usually, if the model's response is smooth, it is expected to be at least above 0.8. Increasing the number of training points may be necessary for non-smooth models.

The output during the search phase is the underlying U-LSGA-III output from the `pymoo` library. It informs the user about the progress of each generation `n_gen` with the number of function evaluations made so far `n_eval`, the minimum and average constraint violations (`cv (min)` and `cv (avg)`), the number of non-dominated solutions found (`n_nds`) and the change of the Pareto convergence indicator (columns `eps/indicator`). The (pymoo documentation)[https://pymoo.org/interface/display.html] provides more detail about this output.

```console
	Generating Training Data:  100%|**********************|100/100 [01:23<00:00,  1.20it/s]
	Generating Testing Data: 100%|*****************************| 30/30 [00:28<00:00, 1.06it/s]
	Training Surrogate Responses: 100%|*************************| 3/3 [00:00<00:00, 36.59it/s]
	Surrogate M_NOx with r = 0.9931
	Surrogate TOM with r = 1.0000
	Surrogate Mf with r = 0.9999
	Exploring the Design Space: 100%|**********************| 256/256 [00:00<00:00, 639.79it/s]
	Searching in the Design Space:   0%|                            | 0/25 [00:00<?, ?it/s]
	=====================================================================================
	n_gen |  n_eval |   cv (min)   |   cv (avg)   |  n_nds  |     eps      |  indicator
	=====================================================================================
	1 |     101 |  0.00000E+00 |  0.00000E+00 |      14 |            - |            -
	2 |     202 |  0.00000E+00 |  0.00000E+00 |      17 |  0.012666136 |        ideal
	3 |     303 |  0.00000E+00 |  0.00000E+00 |      19 |  0.009879243 |        ideal
	4 |     404 |  0.00000E+00 |  0.00000E+00 |      19 |  0.017813004 |        ideal
	5 |     505 |  0.00000E+00 |  0.00000E+00 |      21 |  0.046526161 |        ideal
	6 |     606 |  0.00000E+00 |  0.00000E+00 |      21 |  0.017590293 |        ideal
	7 |     707 |  0.00000E+00 |  0.00000E+00 |      20 |  0.058322279 |        ideal
	8 |     808 |  0.00000E+00 |  0.00000E+00 |      20 |  0.007842796 |            f
	9 |     909 |  0.00000E+00 |  0.00000E+00 |      20 |  0.001397298 |            f
	10 |    1010 |  0.00000E+00 |  0.00000E+00 |      20 |  0.003076413 |            f
	11 |    1111 |  0.00000E+00 |  0.00000E+00 |      20 |  0.005812082 |            f
	12 |    1212 |  0.00000E+00 |  0.00000E+00 |      20 |  0.046733100 |        ideal
	13 |    1313 |  0.00000E+00 |  0.00000E+00 |      21 |  0.007279993 |            f
	14 |    1414 |  0.00000E+00 |  0.00000E+00 |      21 |  0.003993163 |            f
	15 |    1515 |  0.00000E+00 |  0.00000E+00 |      21 |  0.005201088 |            f
	16 |    1616 |  0.00000E+00 |  0.00000E+00 |      21 |  0.001729105 |            f
	17 |    1717 |  0.00000E+00 |  0.00000E+00 |      21 |  0.000351185 |            f
	18 |    1818 |  0.00000E+00 |  0.00000E+00 |      21 |  0.002476596 |            f
	19 |    1919 |  0.00000E+00 |  0.00000E+00 |      21 |  0.005252863 |            f
	20 |    2020 |  0.00000E+00 |  0.00000E+00 |      21 |  0.003703162 |            f
	Searching in the Design Space:   4%|*                  | 1/25 [20:00<8:00:08, 1200.34s/it]
	=====================================================================================
	n_gen |  n_eval |   cv (min)   |   cv (avg)   |  n_nds  |     eps      |  indicator
	=====================================================================================
	1 |     101 |  0.00000E+00 |  0.00000E+00 |      13 |            - |            -
	2 |     202 |  0.00000E+00 |  0.00000E+00 |      18 |  0.066779342 |        ideal
	3 |     303 |  0.00000E+00 |  0.00000E+00 |      19 |  0.013040164 |            f
	4 |     404 |  0.00000E+00 |  0.00000E+00 |      19 |  0.003240776 |        ideal
	5 |     505 |  0.00000E+00 |  0.00000E+00 |      19 |  0.006436898 |            f
	6 |     606 |  0.00000E+00 |  0.00000E+00 |      18 |  0.013058896 |        ideal
	7 |     707 |  0.00000E+00 |  0.00000E+00 |      18 |  0.008253434 |            f
	8 |     808 |  0.00000E+00 |  0.00000E+00 |      17 |  0.023029914 |        ideal
	9 |     909 |  0.00000E+00 |  0.00000E+00 |      17 |  0.002872064 |            f
	10 |    1010 |  0.00000E+00 |  0.00000E+00 |      18 |  0.008945702 |            f
	11 |    1111 |  0.00000E+00 |  0.00000E+00 |      18 |  0.002731946 |            f
	12 |    1212 |  0.00000E+00 |  0.00000E+00 |      18 |  0.006083088 |        nadir
	13 |    1313 |  0.00000E+00 |  0.00000E+00 |      18 |  0.006646305 |            f
	14 |    1414 |  0.00000E+00 |  0.00000E+00 |      18 |  0.004031693 |            f
	15 |    1515 |  0.00000E+00 |  0.00000E+00 |      18 |  0.002041885 |            f
	16 |    1616 |  0.00000E+00 |  0.00000E+00 |      19 |  0.004039912 |            f
	17 |    1717 |  0.00000E+00 |  0.00000E+00 |      19 |  0.001856166 |            f
	18 |    1818 |  0.00000E+00 |  0.00000E+00 |      19 |  0.004233351 |            f
	19 |    1919 |  0.00000E+00 |  0.00000E+00 |      19 |  0.004437145 |            f
	20 |    2020 |  0.00000E+00 |  0.00000E+00 |      19 |  0.028033354 |        nadir
	Searching in the Design Space:   8%|**                 | 2/25 [39:41<7:35:48, 1189.07s/it]
	=====================================================================================
	n_gen |  n_eval |   cv (min)   |   cv (avg)   |  n_nds  |     eps      |  indicator
	=====================================================================================
	1 |     101 |  0.00000E+00 |  0.148803098 |      15 |            - |            -
	2 |     202 |  0.00000E+00 |  0.00000E+00 |      17 |  0.012496447 |            f
	3 |     303 |  0.00000E+00 |  0.00000E+00 |      18 |  0.017854883 |            f
	4 |     404 |  0.00000E+00 |  0.00000E+00 |      18 |  0.037275050 |        ideal
```

The analysis outputs are saved in the csv files, as specified in the script, alongside the pickled Python objects for inspection and debugging. First, inspect the report file, which provides information about the computational time and the number of discarded sets. This information is useful for tuning the number of levels for each set and the number of samples for training and set evaluation.


```txt
	Total Number of Sets      : 256
	Number of Surviving Sets  : 26
	
	Total Surrogate Train Time :        0.021 s
	Total Exploration Time     :        0.861 s
	Total Search Time          :     5358.473 s
	Number of Cores Used       :           16
	
	Train time and score of each Surrogate:
	TOM    0.0211(s)     0.9998
	
	Search time and f_evals of each Set:
	0  205.0636(s)
	1  206.2956(s)
	4  204.2401(s)
	8  203.9787(s)
	16  203.4907(s)
	17  206.1708(s)
	20  203.5766(s)
	32  204.3334(s)
	33  206.6882(s)
	36  207.6279(s)
	48  206.3984(s)
	64  203.8703(s)
	65  206.5425(s)
	68  206.0475(s)
	80  205.4995(s)
	81  208.0759(s)
	84  209.0785(s)
	96  206.8746(s)
	112  206.2059(s)
	128  206.7444(s)
	129  207.0894(s)
	132  207.0260(s)
	144  207.7437(s)
	160  205.8206(s)
	192  206.0405(s)
	208  207.9498(s)
```

A printout of the Pandas data frame of the exploration results is shown. This data is stored in a csv file. 

```pandas
	set_id  is_discarded  climb_h0  climb_h1  cruise_h0  cruise_h1     P  P_TOM
	0         0             0         0         0          0          0  1.00   1.00
	1         1             0         0         0          0          1  1.00   1.00
	2         2             1         0         0          0          2  0.44   0.44
	3         3             1         0         0          0          3  0.00   0.00
	4         4             0         0         0          1          0  0.98   0.98
	..      ...           ...       ...       ...        ...        ...   ...    ...
	251     251             1         3         3          2          3  0.00   0.00
	252     252             1         3         3          3          0  0.00   0.00
	253     253             1         3         3          3          1  0.00   0.00
	254     254             1         3         3          3          2  0.00   0.00
	255     255             1         3         3          3          3  0.00   0.00
	
	[256 rows x 8 columns]
```

A printout of the Pandas data frame of the search results is shown next. This information is also stored in a csv file for easy interchange with other data analysis software. Unlike the exploration data frame, each row represents a single design point here: filtering by `set_id` is necessary for recovering the Pareto front of each surviving set. 

```pandas
	set_id  climb_h0  climb_h1  cruise_h0  cruise_h1           TOM           Mf     M_NOx
	0       0.0  0.019019  0.042392   0.176987   0.136356  18488.123494  1053.114840  6.390926
	1       0.0  0.248721  0.247232   0.248985   0.246306  19570.910964  1019.602066  6.012993
	2       0.0  0.247531  0.247817   0.228041   0.242325  19497.310632  1021.829023  6.047204
	3       0.0  0.061144  0.103122   0.119441   0.056398  18250.701614  1060.478028  6.395150
	4       0.0  0.242335  0.001371   0.010946   0.010773  17969.991130  1070.045548  6.416366
	..      ...       ...       ...        ...        ...           ...          ...       ...
	590   208.0  0.771168  0.282262   0.089347   0.093017  19435.377825  1025.854261  5.904732
	591   208.0  0.843885  0.322598   0.078514   0.163723  19780.236214  1015.011526  5.812730
	592   208.0  0.975473  0.255174   0.126832   0.076693  19750.269531  1015.474685  5.853289
	593   208.0  0.773123  0.411460   0.108772   0.104280  19701.794994  1017.944584  5.820675
	594   208.0  0.752590  0.252407   0.005897   0.001813  18873.928896  1044.221176  6.062478
	
	[595 rows x 8 columns]
```

The csv format allows for easy interchange between the `PDOPT` results and other visualisation programs. On the other hand, the `pandas` and `matplotlib` Python libraries can be used for visualising the output.


