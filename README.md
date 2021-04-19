# Two-Stage Stochastic Programming
This repository provides a framework to perform two-stage stochastic programming on a district energy system considering uncertainties in energy demands, solar irradiance, wind speed, and electricity emission factors.

## What Can I change?
Three sets of input data are present that a user can change to test a new case study.

### editable_values.csv file
The first and primary input is the "editable_values.csv" file. This CSV file consists of four columns: 

1. The first column is "Names (do not change this column)," which provides the keys used in different parts of the code; therefore, please, leave this column unchanged. 

2. The second column is "Values" that a user can change. The values are yes/no questions, text, or numbers, which a user can change to make it specific to their case study or leave them as they are. 

3. The third column is "Instruction." This column gives some instructions in filling the "Value" column, and if by changing the "Value," the user should change other rows in the CSV file or not. 

4. The fourth column is "Where it's used," which gives the subsection of each value. This column can show the rows related to each other. 

The "editable_values.csv" consists of four main sections: 
1. The first section is "Setting Up the Framework." In this section, the user fills the rows from 5 to 11 by answering a series of yes/no questions. If this is the first time a user compiles this program, the answer to all of the questions is 'yes.' 

2. The second section is "Weather Data Analysis." In this section, the user fills the rows from 15 to 28. These rows are used to download the data from the National Solar Radiation Database (NSRDB), using the available solar irradiance in the NSRDB file to calculate the Global Tilted Irradiance (GTI) on a flat solar photovoltaic plate. In this section, probability distribution functions (PDF) of uncertain meteorological inputs are calculated for the wind speed and GTI in this program. 

3. The third section is "Scenario Generation/Reduction" that consists of row 32 to 34. This section relates to generating uncertain scenarios of energy demands, solar irradiance, wind speed, and electricity emissions. After 81 years of synthetic data is generated in the "Scenarios Generation" folder, Principal component analysis (PCA) is used to extract an optimum number of features for each day, and the k-medoid algorithm is used to reduce the number of generated scenarios. If rows 8 (Search optimum PCA) and 9 (Search optimum clusters) have 'yes' values, two figures will be generated in the directory. These two figures can help a user familiar with the explained variance and elbow method to select the number of optimum clusters and features. If a user is not familiar with these two concepts, they can select 18 features as a safe number for the optimum number of features. They can select 10 clusters as the optimum number of clusters. For more accuracy, a user can increase the number of clusters but the computation time increases as well.

4. The fourth section is "District Energy System Optimization." In this section, the two-stage optimization of a district energy system considerin uncertainties to minimze cost and emissions. The rows from 38 to 47 are related to district energy system's charectristics, input parameters to run the multi-objective optimization, and energy components that can be used in the district energy systems. The user is responsible to include rational set of energy components to provide the electricity and heating needs from the demand side. 

### total_energy_demands.csv file
The "total_energy_demands.csv" file consists of the aggregated hourly electricity (kWh), heating (kWh), and cooling (kWh) needs of a group of buildings for a base year, representing the demand side. The user can changethevalues of electricity, heating, and cooling to their own case study's enegry demands. 

### Energy Compoennts folder
The "Energy Components" folder consists of the CSV files of the five selected energy components in this repository, which are natural gas boilers, combined heating and power (CHP), solar photovoltaic (PV), wind turbines, and batteries. These CSV files for each energy component consists of a series of capacities, efficiencies, investment cost, operation and maintenece cost, and life span of the energy components that are considered in this discrete optimization repository. 

## What are the Results?

## How to Run the File?
Installing Anaconda

Search for Anaconda Prompt

Create a new environment for the two-stage stochastic programming optimization
```
conda create -n two_stage_env python=3.7.7
```
Make sure the environment is created. By running this code, the list of available environments, including two_stage_env, should be shown.
```
conda env list
```
Active the new environment. This command should change the environment from base to two_stage_env.
```
conda activate two_stage_env
```
Download this repository. 

To install the required packages to run the framework, the requirements are stored in the requirements.txt file in the repository:
```
pip install -r  Path_to_the_folder\Two_Stage_SP-main\requirements.txt
```
and install scikit-learn-extra using the conda environment to use in the k-medoid clustering algorithm
```
conda install -c conda-forge scikit-learn-extra
```
To run the two-stage stochastic optimization, you should directly compile the main_twostage_SP.py file. Run the Python file after making sure the inputs in the editable_values.csv, Total_energy_demands.csv, and CSV files in the Energy Components are based on your need.
```
python Path_to_the_folder\Two_Stage_SP-main\main_twostage_SP.py
```


