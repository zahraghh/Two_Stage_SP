# Two-Stage Stochastic Programming
This repository provides a framework to perform multi-objective two-stage stochastic programming on a district energy system. In this framework, we consider uncertainties in energy demands, solar irradiance, wind speed, and electricity emission factors. This framework optimizes the sizing of energy components to minimize the total cost and operating CO<sub>2</sub> emissions. Natural gas boilers, combined heating and power (CHP), solar photovoltaic (PV), wind turbines, batteries, and the grid are the energy components considered in this repository. 

## How Can I Install this Repository?
To use this repository, you need to use either Python or Anaconda. You can download and install Python using the following link https://www.python.org/downloads/ or Anaconda using the following link https://docs.anaconda.com/anaconda/install/. 

Two packages should be installed using the conda or PyPI.

1. install scikit-learn-extra either in conda environment:
```
conda install -c conda-forge scikit-learn-extra 
```
or from PyPI:
```
pip install scikit-learn-extra
```
2. install a solver that is available for public use either in conda environmnet:
```
conda install glpk --channel conda-forge
```
or from PyPI:
```
pip install glpk
```

Download the ZIP file of this repository from this link: https://github.com/zahraghh/Two_Stage_SP/tree/JOSS.

Unzip the "Two_Stage_SP-JOSS" folder and locally install the package using the pip command. The /path/to/Two_Stage_SP-JOSS is the path to the "Two_Stage_SP-JOSS" folder that contains a setup.py file. 
```
pip install -e /path/to/Two_Stage_SP-JOSS
```

To use this repository, you can directly compile the "main_two_stage_SP.py" code in the tests\test1 folder.

Have a look at the "tests\test1" folder. Four files are needed to compile the "main_two_stage_SP.py" code successfully:
1. "Energy Components" folder containing energy components characteristics
2. "editable_values.csv" file containing variable inputs of the package
3. "total_energy_demands.csv" file containing the aggregated hourly electricity, heating, and cooling demands of a group of buildings
4. "main_two_stage_SP.py" file to be compiled and run the two-stage stochastic programming optimization

## How to Use this Repository?
After the package is installed, we can use Two_Stage_SP-JOSS\tests\Test folder that contains the necessary help files ("Energy Components" folder, "editable_values.csv', "total_energy_demands.csv") to have our main.py code in it. We can first download the weather files, calculate the global titlted irradiance, and quantify distributions of solar irradiance and wind speed by writing a similar code in main.py: 
```
import os
import sys
import pandas as pd
import csv
from Two_Stage_SP import download_windsolar_data, GTI,uncertainty_analysis
if __name__ == "__main__":
    #Reading the data from the Weather Data Analysis section of the editable_values.csv
    editable_data_path =os.path.join(sys.path[0], 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    city_DES =str(editable_data['city'])
    #Downloading the weather data from NSRDB
    download_windsolar_data.download_meta_data(city_DES)
    #Calculating the  global tilted irradiance on a surface in the City
    GTI.GTI_results(city_DES)
    #Calculating the distribution of global tilted irradiance (might take ~5 mins)
    uncertainty_analysis.probability_distribution('GTI',46) #Name and the column number in the weather data
    #Calculating the distribution of wind speed (might take ~5 mins)
    uncertainty_analysis.probability_distribution('wind_speed',8) #Name and the column number in the weather data
```
The outcome of this code is a new folder with the name of the city in  the editable_values.csv. If you haven't change the editable_values.csv, the folder name is Salt Lake City, which contains the needed weather parameters. 

After the weather data is generated, we can perfrom scenario generation using Monte Carlo simulation and scenario reduction using k-median algorithm to reduce the number of scenarios:
```
import os
import sys
import pandas as pd
import csv
from Two_Stage_SP import scenario_generation,clustring_kmediod_PCA
if __name__ == "__main__":
    #Reading the data from the  Scenario Generation/Reduction section of the editable_values.csv
    #We need "total_energy_demands.csv" for scenario generation/reduction
    editable_data_path =os.path.join(sys.path[0], 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    #Generate scenarios for uncertainties in ...
    #energy demands,solar irradiance, wind speed, and electricity emissions
    state = editable_data['State']
    scenario_generation.scenario_generation_results(state)
    #Reduce the number scenarios of scenarios ...
    #using the PCA and k-medoid algorithm
    clustring_kmediod_PCA.kmedoid_clusters()
```
After scenarios are generated and reduced, the selected of representative days are located in Scenario Generation\City\Representative days folder. Then, we perfrom the optimization on these selected representative days:
```
import os
import sys
import pandas as pd
import csv
from platypus import NSGAII, Problem, Real, Integer, InjectedPopulation,GAOperator,HUX, BitFlip, SBX,PM,PCX,nondominated,ProcessPoolEvaluator
from pyomo.opt import SolverFactory
from Two_Stage_SP import NSGA2_design_parallel_discrete
if __name__ == "__main__":
    #Reading the data from the  District Energy System Optimization section of the editable_values.csv
    # We need total_energy_demands.csv and a folder with charectristic of energy components
    editable_data_path =os.path.join(sys.path[0], 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    #Perfrom two-stage stochastic optimization
    problem= NSGA2_design_parallel_discrete.TwoStageOpt()
    #Make the optimization parallel
    with ProcessPoolEvaluator(int(editable_data['num_processors'])) as evaluator: #max number of accepted processors is 61 by program/ I have 8 processor on my PC
        algorithm = NSGAII(problem,population_size=int(editable_data['population_size']) ,evaluator=evaluator,variator=GAOperator(HUX(), BitFlip()))
        algorithm.run(int(editable_data['num_iterations']))
    #Generate a csv file as the result
    NSGA2_design_parallel_discrete.results_extraction(problem, algorithm)
```
After the optimization is performed (migh take a few hours based on the number of iterations), a new folder (City_Discrete_EF_EF value_...)  is generated that contains the two csv files, sizing of energy components and objective values for the Pareto front. 

We can also perfrom the three parts together:
```
import os
import sys
import pandas as pd
import csv
from platypus import NSGAII, Problem, Real, Integer, InjectedPopulation,GAOperator,HUX, BitFlip, SBX,PM,PCX,nondominated,ProcessPoolEvaluator
from pyomo.opt import SolverFactory
from Two_Stage_SP import NSGA2_design_parallel_discrete, scenario_generation,clustring_kmediod_PCA, download_windsolar_data, GTI,uncertainty_analysis
if __name__ == "__main__":
    #Reading the data from the Weather Data Analysis section of the editable_values.csv
    editable_data_path =os.path.join(sys.path[0], 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    city_DES =str(editable_data['city'])
    #Downloading the weather data from NSRDB
    download_windsolar_data.download_meta_data(city_DES)
    #Calculating the  global tilted irradiance on a surface in the City
    GTI.GTI_results(city_DES)
    #Calculating the distribution of global tilted irradiance (might take ~5 mins)
    uncertainty_analysis.probability_distribution('GTI',46) #Name and the column number in the weather data
    #Calculating the distribution of wind speed (might take ~5 mins)')
    uncertainty_analysis.probability_distribution('wind_speed',8) #Name and the column number in the weather data
    #Generate scenarios for uncertainties in ...
    #energy demands,solar irradiance, wind speed, and electricity emissions
    state = editable_data['State']
    scenario_generation.scenario_generation_results(state)
    #Reduce the number scenarios of scenarios ...
    #using the PCA and k-medoid algorithm
    clustring_kmediod_PCA.kmedoid_clusters()
    #Perfrom two-stage stochastic optimization
    problem= NSGA2_design_parallel_discrete.TwoStageOpt()
    #Make the optimization parallel
    with ProcessPoolEvaluator(int(editable_data['num_processors'])) as evaluator: #max number of accepted processors is 61 by program/ I have 8 processor on my PC
        algorithm = NSGAII(problem,population_size=int(editable_data['population_size']) ,evaluator=evaluator,variator=GAOperator(HUX(), BitFlip()))
        algorithm.run(int(editable_data['num_iterations']))
    #Generate a csv file as the result
    NSGA2_design_parallel_discrete.results_extraction(problem, algorithm)

```

## What Can I change?
Three sets of input data are present that a user can change to test a new/modified case study.

### editable_values.csv file
The first and primary input is the "editable_values.csv" file. This CSV file consists of four columns: 

1. The first column is "Names (do not change this column)," which provides the keys used in different parts of the code; therefore, please, leave this column unchanged. 

2. The second column is "Values" that a user can change. The values are yes/no questions, text, or numbers, which a user can modify to make it specific to their case study or leave them as they are. 

3. The third column is "Instruction." This column gives some instructions in filling the "Value" column, and if by changing the "Value," the user must change other rows in the CSV file or not. Please, if you want to change a value, read its instruction. 

4. The fourth column is "Where it's used," which gives the subsection of each value. This column can show the rows that are related to each other. 

The "editable_values.csv" consists of four main sections: 
1. The first section is "Setting Up the Framework." In this section, the user fills the rows from 5 to 11 by answering a series of yes/no questions. If this is the first time a user compiles this program, the answer to all of the questions is 'yes.' A user can change the values to 'no' if they have already downloaded/generated the files for that row. For example, if the weather data is downloaded and Global Tilted Irradiance (GTI) is calculated on flat plates, a user can change the row 5 value to 'no' to skip that part. 

A figure is shown that demonstrates if values of rows from 5 to 11 are 'yes' in the "editable_values.csv" file, what rows are the input, and what would be the results. This figure can help a user to understand when they already downloaded/calculated results from one of the rows. They can change that row's value to 'no' to reduce the computational time of compiling their case study next time.
![alt text](https://github.com/zahraghh/Two_Stage_SP/blob/JOSS/Two_stage_framework.png) 


2. The second section is "Weather Data Analysis." In this section, the user fills the rows from 15 to 28. These rows are used to download the data from the National Solar Radiation Database (NSRDB) and using the available solar irradiance in the NSRDB file to calculate the GTI on a flat solar photovoltaic plate. In this section, probability distribution functions (PDF) of uncertain meteorological inputs are calculated for the wind speed and GTI.

3. The third section is "Scenario Generation/Reduction" that consists of row 32 to 34. This section relates to generating uncertain scenarios of energy demands, solar irradiance, wind speed, and electricity emissions. After scenarios representing the uncertainties are generated in the "Scenarios Generation" folder, Principal component analysis (PCA) is used to extract an optimum number of features for each scenario. Then, the k-medoid algorithm is used to reduce the number of generated scenarios. If rows 8 (Search optimum PCA) and 9 (Search optimum clusters) have 'yes' values, two figures will be generated in the directory. These two figures can help a user familiar with the explained variance and elbow method to select the number of optimum clusters in the k-medoid algorithm and features in PCA. If a user is not familiar with these two concepts, they can select 18 features as a safe number for the optimum number of features. They can select 10 clusters as the optimum number of clusters. For more accuracy, a user can increase the number of clusters, but the computation time increases.

4. The fourth section is "District Energy System Optimization." In this section, the two-stage optimization of a district energy system considering uncertainties is performed to minimize cost and emissions. The rows from 38 to 47 are related to the district energy system's characteristics, input parameters to run the multi-objective optimization, and energy components that can be used in the district energy systems. The user is responsible for including a rational set of energy components to provide the electricity and heating needs of buildings. For example, if values of 'CHP' (row 53)and 'Boiler' are written as 'no,' this means no boiler and CHP system will be used in the district energy system. If no boiler and CHP system is used in the district energy system, the heating demand of buildings cannot be satisfied, and an error would occur.

### total_energy_demands.csv file
The "total_energy_demands.csv" file consists of the aggregated hourly electricity (kWh), heating (kWh), and cooling (kWh) needs of a group of buildings for a base year, representing the demand side. This file contains 8760 (number of hours in a year). A user can change electricity, heating, and cooling values to their own case study's energy demands. 

### Energy Components folder
The "Energy Components" folder consists of the CSV files of the five selected energy components in this repository, which are natural gas boilers, CHP, solar PV, wind turbines, and batteries. These CSV files for each energy component consist of a series of capacities, efficiencies, investment cost, operation & maintenance cost, and life span of the energy components. A user can modify these values or add more options to their CSV files to expand the decision space. 

## What are the Results?
If all parts of the framework are used, which means a user writes 'yes' for values of rows 5 to 11 in the "editable_values.csv" file, a series of CSV files and figures will be generated.
1. Two figures will be generated in the directory related to the optimum number of features in PCA and the optimum number of clusters in the k-medoid algorithm if rows 7, 8, and 9 are 'yes.' 
Suppose a user is familiar with the connection of explained variance and the number of features. In that case, they can use the "Explained variance vs PCA features" figure in the directory to select the optimum number of features. If a user is familiar with the elbow method, they can use the "Inertia vs Clusters" figure in the directory to select the optimum number of clusters. 
2. A folder named 'City_name_Discrete_EF_...' will be generated that contains five files. 
    1. One "ParetoFront.png" figure that shows the total cost and operating CO<sub>2</sub> emissions trade-off for different scenarios to minimize cost and emissions. 
    2. One CSV file, "objectives.csv," that represents the total cost and operating CO<sub>2</sub> emissions trade-off for the different scenarios to minimize cost and emissions. This CSV file contains the values that are shown in the "ParetoFront.png" figure. 
    3. Two parallel coordinates figures, "Parallel_coordinates_cost.png" and "Parallel_coordinates_emissions.png," which show the variation in the optimum energy configurations to minimize the total cost and operating CO<sub>2</sub> emissions. 
    4. One CSV file that contains the optimum sizing of five selected energy components in this repository, which are natural gas boilers, CHP, solar PV, wind turbines, and batteries, to minimize the total cost and operating CO<sub>2</sub> emissions. 
This CSV file contains all of the values that are used in the two parallel coordinates figures.

