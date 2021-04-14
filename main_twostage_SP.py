### Performing Two Stage Stochastic Programming for the Design of District Energy system ###
import os
import sys
import pandas as pd
import csv
import download_windsolar_data as download_data
import GTI
import uncertainty_analysis
import scenario_generation
import clustring_kmediod_PCA
import NSGA2_design_parallel_discrete as two_stage
from platypus import NSGAII, Problem, Real, Integer, InjectedPopulation,GAOperator,HUX, BitFlip, SBX,PM,PCX,nondominated,ProcessPoolEvaluator
# I use platypus library to solve the muli-objective optimization problem:
# https://platypus.readthedocs.io/en/latest/getting-started.html
from pyomo.opt import SolverFactory
editable_data_path =os.path.join(sys.path[0], 'EditableFile.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
city_DES ='/'+ str(editable_data['city'])
state = editable_data['State']
if __name__ == '__main__':
    #Do we need to generate the meteorlogical data and their distributions?
    if editable_data['Weather data download and analysis']=='yes':
        download_data.download_meta_data(city_DES)
        #Calculating the  global tilted irradiance on a surface in the City
        GTI.GTI(city_DES)
        #Calculating the distribution of variable inputs: solar irradiance and wind speed
        print('Calculating the distribution of global tilted irradiance (might take ~5 mins)')
        uncertainty_analysis.probability_distribution('GTI',46) #Name and the column number in the weather data
        print('Calculating the distribution of wind speed (might take ~5 mins)')
        uncertainty_analysis.probability_distribution('wind_speed',8) #Name and the column number in the weather data
    #Do we need to generate scenarios for uncertainties in ...
    #energy demands,solar irradiance, wind speed, and electricity emissions?
    if editable_data['Generate Scenarios']=='yes':
        print('Generate scenarios for uncertain variables')
        scenario_generation.scenario_generation(state)
    if editable_data['Perfrom scenario reduction']=='yes':
        print('Perfrom scenarios reduction using k-medoid algorithm')
        clustring_kmediod_PCA.kmedoid_clusters()
    if editable_data['Perform two stage optimization']=='yes':
        print('Perfrom two-stage stochastic optimization')
        problem= two_stage.TwoStageOpt()
        with ProcessPoolEvaluator(int(editable_data['num_processors'])) as evaluator: #max number of accepted processors is 61 by program/ I have 8 processor on my PC
            algorithm = NSGAII(problem,population_size=int(editable_data['population_size']) ,evaluator=evaluator,variator=GAOperator(HUX(), BitFlip()))
            algorithm.run(int(editable_data['num_iterations']))
        two_stage.results_extraction(problem, algorithm)
