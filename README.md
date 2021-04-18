# Two-Stage Stochastic Programming
This repository provides a framework to perform two-stage stochastic programming on a district energy system considering uncertainties in energy demands, solar irradiance, wind speed, and electricity emission factors.

To run the two-stage stochastic optimization, you should directly complile the main_twostage_SP.py file. The input data should be entered using the EditableFile.csv. The EditableFile.csv file has four columns: the names of each row, the value of each row that will be used in the framework, the instruction of each row that helps to undesrtand why the user needs to fill this value, and in what stage of the framework this row is used.

Installing Anaconda

Search for Anaconda Prompt

Creat a new environmnet for the two stage stochastic programming optimization
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
pip install -r  Path_to_the_file/requirements.txt
```
and install scikit-learn-extra uisng the conda environment:
```
conda install -c conda-forge scikit-learn-extra
```
