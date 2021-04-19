# Two-Stage Stochastic Programming
This repository provides a framework to perform two-stage stochastic programming on a district energy system considering uncertainties in energy demands, solar irradiance, wind speed, and electricity emission factors.

## What Can I change?
Three sets of input data are present that a user can change to test a new case study.

### EditableFile.csv file
The first and main input is "EditableFile.csv" file. This CSV file consists of four columns. The first column is "Names (do not change this column)" that provides the keys used in different parts of the code; therefore, please, leave this column unchanged. The second column is "Values" that a user can change. The values are yes/no questions, text, or numbers, which a user can change to make it specific to their case study or leave them as they are. The third column is "Instruction". This column gives some instructions in filling the "Value" column, and if by changing the "Value", the user should change other rows in the CSV file or not. The forth column is "Where it's used" that gives the subsection of each value. This column can show the rows that are related to each other. 

### total_energy_demands.csv file


### Energy Compoennts folder



## What are the Results?

## How to Run the File?
Installing Anaconda

Search for Anaconda Prompt

Create a new environmnet for the two stage stochastic programming optimization
```
conda create -n two_stage_env python=3.7.7
```
Make sure the environment is created. By running this code, the list of available environments, including two_stage_env, should be shown.
```
conda env list
```
Actiev the new environment. This command should change the environment from base to two_stage_env.
```
conda activate two_stage_env
```
Download this repository. 

To install the required packages to run the framework, the requirements are stored in the requirements.txt file in the repository:
```
pip install -r  Path_to_the_folder\Two_Stage_SP-main\requirements.txt
```
and install scikit-learn-extra uisng the conda environment to use in the k-medoid clustering algorithm
```
conda install -c conda-forge scikit-learn-extra
```
To run the two-stage stochastic optimization, you should directly complile the main_twostage_SP.py file. Run the Python file after making sure the inputs in the EditableFile.csv, Total_energy_demands.csv, and csv files in the Energy Components are based on your need.
```
python Path_to_the_folder\Two_Stage_SP-main\main_twostage_SP.py
```


