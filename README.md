# Two-Stage Stochastic Programming
This repository provides a framework to perform a multi-objective two-stage stochastic programming on a district energy system considering uncertainties in energy demands, solar irradiance, wind speed, and electricity emission factors. This framework optimizes the sizing of energy components to minimze the total cost and operating CO<sub>2</sub> emissions. Natural gas boilers, combined heating and power, solar photovoltaic, wind turbines, batteries, and the grid are the energy components conisdred in this repository. 

## How to Run the File?
To run the file in this repository, we suggest to use a new conda environment. You can downalod and install anaconda using the following link: https://docs.anaconda.com/anaconda/install/

After anaconda is installed, search for anaconda prompt on your system:
- Windows: Click Start, search, or select Anaconda Prompt from the menu.
- macOS: Cmd+Space to open Spotlight Search and type “Navigator” to open the program.
- Linux–CentOS: Open Applications - System Tools - termin
    
Create a new environment for this repository, two_stage_env. We have tested this framework using python=3.7.7 version.
```
conda create -n two_stage_env python=3.7.7
```

Make sure the environment is created. By running the following code, the list of available environments, including two_stage_env, should be shown.
```
conda env list
```
Activate two_stage_env environment. This command should change the environment from base to two_stage_env.
```
conda activate two_stage_env
```
Now a new environmnet, two_stage_env, is ready to test the repository on it. 

Two packages should be installed using the conda command in the two_stage_env environment.

1. install scikit-learn-extra in the conda environment:
```
conda install -c conda-forge scikit-learn-extra
```
2. install a solver that is available for public use:
```
conda install glpk --channel conda-forge
```

Download the ZIP file of this repository from this link: https://github.com/zahraghh/Two_Stage_SP/tree/IMECE.

Unzip the folder and change the directory in the anaconda prompt to the unziped folder. 
```
cd Path_to_the_folder\Two_Stage_SP-IMECE
```
Now, you can install the package using the pip command and use it in your code.
```
pip install .
```
Have a look at "Framework Test_University of Utah" folder. Four files are needed to succesfully compile the "main_two_stage_SP.py" code:
- "Energy Components" folder containing energy components charectristics
- "editable_values.csv' file containing variable inputs of the package
- "total_energy_demands.csv" file containing the aggregated hourly electricity, heating, and cooling demands of a group of buildings
- "main_two_stage_SP.py" file to be compiled and run the two-stage stochastic programming optimization

## What Can I change?
Three sets of input data are present that a user can change to test a new case study.

### editable_values.csv file
The first and primary input is the "editable_values.csv" file. This CSV file consists of four columns: 

1. The first column is "Names (do not change this column)," which provides the keys used in different parts of the code; therefore, please, leave this column unchanged. 

2. The second column is "Values" that a user can change. The values are yes/no questions, text, or numbers, which a user can modify to make it specific to their case study or leave them as they are. 

3. The third column is "Instruction." This column gives some instructions in filling the "Value" column, and if by changing the "Value," the user must change other rows in the CSV file or not. 

4. The fourth column is "Where it's used," which gives the subsection of each value. This column can show the rows related to each other. 

The "editable_values.csv" consists of four main sections: 
1. The first section is "Setting Up the Framework." In this section, the user fills the rows from 5 to 11 by answering a series of yes/no questions. If this is the first time a user compiles this program, the answer to all of the questions is 'yes.' 

2. The second section is "Weather Data Analysis." In this section, the user fills the rows from 15 to 28. These rows are used to download the data from the National Solar Radiation Database (NSRDB), using the available solar irradiance in the NSRDB file to calculate the Global Tilted Irradiance (GTI) on a flat solar photovoltaic plate. In this section, probability distribution functions (PDF) of uncertain meteorological inputs are calculated for the wind speed and GTI in this program. 

3. The third section is "Scenario Generation/Reduction" that consists of row 32 to 34. This section relates to generating uncertain scenarios of energy demands, solar irradiance, wind speed, and electricity emissions. After 81 years of synthetic data is generated in the "Scenarios Generation" folder, Principal component analysis (PCA) is used to extract an optimum number of features for each day. The k-medoid algorithm is used to reduce the number of generated scenarios. If rows 8 (Search optimum PCA) and 9 (Search optimum clusters) have 'yes' values, two figures will be generated in the directory. These two figures can help a user familiar with the explained variance and elbow method to select the number of optimum clusters and features. If a user is not familiar with these two concepts, they can select 18 features as a safe number for the optimum number of features. They can select 10 clusters as the optimum number of clusters. For more accuracy, a user can increase the number of clusters, but the computation time increases.

4. The fourth section is "District Energy System Optimization." In this section, the two-stage optimization of a district energy system considerin uncertainties to minimize cost and emissions. The rows from 38 to 47 are related to the district energy system's characteristics, input parameters to run the multi-objective optimization, and energy components that can be used in the district energy systems. The user is responsible for including a rational set of energy components to provide the electricity and heating needs from the demand side. 

### total_energy_demands.csv file
The "total_energy_demands.csv" file consists of the aggregated hourly electricity (kWh), heating (kWh), and cooling (kWh) needs of a group of buildings for a base year, representing the demand side. The user can change the values of electricity, heating, and cooling to their own case study's energy demands. 

### Energy Compoennts folder
The "Energy Components" folder consists of the CSV files of the five selected energy components in this repository, which are natural gas boilers, combined heating and power (CHP), solar photovoltaic (PV), wind turbines, and batteries. These CSV files for each energy component consist of a series of capacities, efficiencies, investment cost, operation & maintenance cost, and life span of the energy components considered in this discrete optimization repository. A user can modify these values or add more options to their CSV files to expand the decision space. 

## What are the Results?
If all parts of the framework are used, which means a user writes 'yes' values for rows 5 to 11 in the "editable_values.csv" file, a series of CSV files and figures will be generated.
1. Two figures will be generated in the directory related to the optimum number of features in PCA and the optimum number of clusters in the k-medoid algorithm if rows 7, 8, and 9 are 'yes.' 
If a user is familiar with the connection of explained variance and the number of features, can use the "Explained variance vs PCA features" figure in the directory to select the optimum number of features. If a user is familiar with the elbow method, they can use the "Inertia vs Clusters" figure in the directory to select the optimum number of clusters. 
2. A folder named 'City_name_Discrete_EF_...' will be generated that contains five files. 
    1. One "ParetoFront" figure shows the cost and emissions trade-off for the different scenarios to minimize cost and emissions. 
    2. One CSV file, "objectives," that represent the cost and CO<sub>2</sub> emissions trade-off for the different scenarios to minimize cost and emissions. This CSV file contains the values that are shown in the "ParetoFront" figure. 
    3. Two parallel coordinates figures, "Parallel_coordinates_cost" and "Parallel_coordinates_emissions," which show the variation in the optimum energy configurations to minimize the total cost and operating CO<sub>2</sub> emissions. 
    4. One CSV file that contains the optimum sizing of five selected energy components in this repository, which are natural gas boilers, combined heating and power (CHP), solar photovoltaic (PV), wind turbines, and batteries, to minimize the total cost and operating CO<sub>2</sub> emissions. 
This CSV file contains all of the values that are used in the two parallel coordinates figures.
3. A "Scenario Generation" folder will be generated. This folder is not the direct results from compiling repository, but it will be used to generate the final results.  
