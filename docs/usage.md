
# User Guide of PDOPT


## Structure of a PDOPT case

A PDOPT analysis is composed of the following elements:

- A design model, i.e. a computational simulation of the object under design that takes in design parameters and returns quantites of interest (QoI) quantifying its performance o behaviour. For instance, a simulation model describing the design of a wing would take as input parameters geometry and materials and return as QoI its lift-to-drag ratio, its maximum lift coefficient, its structural weight, and so on.
- A set of input parameters (Continous or Discrete) which define the design space.
- Quantities of Interest, which are divided into Constraints and Objectives:
  - Constraints are feasibility requirements, i.e. the design must satisfy them to be acceptable.
  - Objectives are desirability requirements, i.e. what is expected by the design to do, and possible the best as possible.
  
The framework provides object-oriented classes to define these items within PDOPT. 
The user has first to identify which variables in the design model are parameters (and therefore define a design space), and which instead describe a behaviour of charateristic of the system that they wish to control (either to optimise or to constraint). Usually the design process consists in a set of performance figures of merit to optimise given specific constraints.  

After the definition of the variables and the set-up of the script, the workflow consists in running the exploration phase with `pdopt.exploration.ProbabilisticExploration` and the search phase with `pdopt.optimisation.Optimisation`.


## Definition of the Simulation/Design Model

The framework is designed to assume the simulation model as a black box. Hence, `PDOPT` only expects the data is presented through the interface. Both the `ProbabilisticExploration()` and `Optimisation()` objects expect the simulation model to have the `.run()` method for running the evaluation function. The framework provides two approaches to connect the simulation model to the API: the class `data.Model()` and the `data.ExtendableModel()` abstract class. The first is a wrapper for the actual evaluation function. The second is a template for defining a custom model object, useful if the simulation model requires a setup or parameters to be changed at the object creation. Examples of their usage is shown here:

```python
# Example using the ExtendableModel abstract class
class My_Model(ExtendableModel):
	
	def __init__(self, model_parameters):
	self.model_parameters = model_parameters
		# Do something with the parameters to setup the simulation
	
	def run(self, *args):
		# Do something with args
	
# Example using the Model object
def my_function(self, *args):
	# Do something with args
	
my_model = data.Model(my_function)
```

The evaluation function must be crafted such as:

- The input arguments are an ordered list following the same order of the parameters specified in the `DesignSpace` obect.
- The output quantities must be returned through a python dictionary whose keywords are the constraint or objective name specified in the `DesignSpace` object.

Here is an example where the parameters `A, B, C` are inputs and `X, Y` are outputs:

```python
def my_function(A, B, C):
	# do something
	return {'X' : X, 'Y' : Y}
```

It is possible to construct an evaluation function with flexible inputs by using the Python star argument (*args). This is useful especially if the same function is being used in different `PDOPT` analyses:

```python
# Example of usage of star argument 
def my_flexible_function(*args):
	# Unpack parameters depending on length
	if len(args) == 3:
		A, B, C = args
	else:
		A, B, C, D = args
	
	## do something
``` 

## Definition of the Input Parameters

The input parameters are distinguished as continuous and discrete. Discrete parameters act as a C language `ENUM`: a list of integers (going from 0 to N), which have to be mapped internally inside the evaluation function. Continous parameters, instad, represent a continous range between the upper and lower bound. Both types can be defined directly in code using `data.DiscreteParamter()` and `data.ContinousParameter()`, or with the helper function `DesignSpace.from_csv()` (along with the responses .csv file).

If using a .csv input file, it hast to be structured with the following column entries:

- **name** : Name of the parameter. It must not contain spaces; use an underscore for it.
- **type** : If the variable is continuous or discrete. If discrete, the lb and ub parameters will be ignored, as only integer levels will be provided.
- **lb** : Lower bound of the continuous parameter.
- **ub** :  Upper bound of the continuous parameter.
- **levels** : The number of parameter levels: it must be an integer.
- **uq_dist** : Type of UQ distribution for this parameter for uncertainty-based optimisation. It can be 'uniform', 'triangular' (triangular), or 'norm' (gaussian). Set it to 'nan' for ignoring. Uniform and Triangular distributions support asymmetric distributions, while Gaussian is only symmetric. 
- **uq_var_l** : Lower variation bound for the distribution, expressed as a decimal percentile variation (i.e. 0.05 for 5% variation).
- **uq_var_u** : Upper variation bound for the distribution, expressed as the decimal percentile variation. Set it to 'nan' if the distribution is symmetric.


Each row in the .csv file must be filled in. If a variable is unused, it must be set to 'nan'. Due to the implementation of the framework, it is fundamental that the input parameters in the `input.csv` or in the paramter list passed to `DesignSpace` are ordered in the same order as the evaluation function arguments. During any functional evaluation, `PDOPT` passes an array of inputs in the specified order assuming that it matches the expected order of the evaluation function arguments.

Some parameters are used to specify an uncertainty quantification distribution to be used in uncertainty-based optimisation. While the bare bones are present in the current version of the codebase, it is not yet fully implemented. Hence, these rows are not used other than to set up the `ContinousParameter()` objects.


## Definition of Requirements: Constraints and Objectives

The two types of requirements, constraints and objectives, are defined with the respective objects `data.Constraint()` and `data.Objective()`. Also in this case it is possible to load the definitions from a .csv file using the helper function `DesignSpace.from_csv()`.

Unlike the input definition, the order of the entries does not affect the code's functionality. Instead, the user must use a unique name for each QoI: these have to match the keyword used in the return Python dictionary of the evaluation function. At least one entry per type (objective and constraint) must be present for the code to function, and every column must be filled. The file is structured as such:

- **name** : Name of the response from the evaluation function. These names must be used in the dict() object returned from the evaluation function. 
- **type** : Type of response. It can be 'constraint' or 'objective'. These are the responses that the optimisation step will handle. In the exploration step, constraints are always active, while objectives can be set up as constraints or not (see next point).
- **op** : Operator on the response. If the type is set to 'constraint', the operator will be 'lt' (less than) or 'gt' (greater than), which represents the inequality of the response with respect to the quantity in the value column (that is, 'TOM, constraint, lt, 2000' corresponds to TOM < 20000). If the type is set to 'objective', the operator is 'min' (minimise) or 'max' (maximise).
- **val** : Value which is going to be used by the operator. In the case of an objective, this value is used to set a constraint in the exploration phase with the following criteria: if 'min' is set as the operator, then it is less than the constraint; if 'max' is set as the operator, then it is greater than the constraint. This is done to drive the exploration phase and to remove areas of the design space that might not satisfy minimum requirements. Set it to 'nan' to disable the constraint.
- **pSat** : The minimum satisfaction probability for the constraint/objective. When evaluating the sets, this is the minimum probability to which samples are tested. Samples that go under pSat are counted as unsatisfactory.


## Structure of the Output

Two Pandas `DataFrame` are generated as output. The first covers the results from the exploration phase, while the second covers the results of the search phase. The results of the exploration phase cover the probabilities of satisfaction of the requirements and the levels of the input parameters that make up each set. Each entry is a set with the following columns:

- **set_id** : The number identifying the set.
- **is_discarded** : A Boolean value of 0/1 indicating whether the set has been discarded.
- The input parameters, each column contains the level selected for each set.
- **P** : The total calculated probability for that set.
- The probabilities for each requirement such that it plays a role in the exploration phase.

For the search phase, the results present the optimal points and their QoI values. Each entry in the table is a design point with these columns:

- **set_id**: The number identifying the set to which this data point belongs.
- Columns of the input parameters as defined in `input.csv`. These values are the actual number and not a level; it is the found optimum value.
- Columns of response quantities, objective and constraints, as defined in `response.csv`. 

These dataframes can be obtainted with the  functions `DesignSpace.get_exploration_results()` and `DesignSpace.get_optimum_results()`, and saved directly as .csv files with `DesignSpace.save_exploration_results()` and `DesignSpace.save_optimisation_results()`.



