### Performing Two Stage Stochastic Programming for the Design of District Energy system ###
import os
import sys
import pandas as pd
import csv
from pathlib import Path
from platypus import NSGAII, Problem, Real, Integer, InjectedPopulation,GAOperator,HUX, BitFlip, SBX,PM,PCX,nondominated,ProcessPoolEvaluator
# I use platypus library to solve the muli-objective optimization problem:
# https://platypus.readthedocs.io/en/latest/getting-started.html
editable_data_path =os.path.join(sys.path[0], 'editable_values.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
path_test =  os.path.join(sys.path[0])
from pyomo.opt import SolverFactory
path_parent= Path(Path(sys.path[0]))
Two_stage_SP_folder = os.path.join(path_parent.parent.absolute())
sys.path.insert(0, Two_stage_SP_folder)
import error_evaluation
error_evaluation.errors(path_test)
import download_windsolar_data, GTI,uncertainty_analysis,scenario_generation,clustring_kmediod_PCA,NSGA2_design_parallel_discrete
sys.path.remove(Two_stage_SP_folder)

if __name__ == "__main__":
    city_DES =str(editable_data['city'])
    state = editable_data['State']
    #Do we need to generate the meteorlogical data and their distributions?
    if editable_data['Weather data download and analysis']=='yes':
        download_windsolar_data.download_meta_data(city_DES)
        #Calculating the  global tilted irradiance on a surface in the City
        GTI.GTI_results(city_DES,path_test)
        #Calculating the distribution of variable inputs: solar irradiance and wind speed
        print('Calculating the distribution of global tilted irradiance (might take ~5 mins)')
        uncertainty_analysis.probability_distribution('GTI',46,path_test) #Name and the column number in the weather data
        print('Calculating the distribution of wind speed (might take ~5 mins)')
        uncertainty_analysis.probability_distribution('wind_speed',8,path_test) #Name and the column number in the weather data
    #Do we need to generate scenarios for uncertainties in ...
    #energy demands,solar irradiance, wind speed, and electricity emissions?
    if editable_data['Generate Scenarios']=='yes':
        print('Generate scenarios for uncertain variables')
        scenario_generation.scenario_generation_results(path_test,state)
    #Do we need to reduce the number scenarios of scenarios in ...
    #using the PCA and k-medoid algorithm?
    if editable_data['Perfrom scenario reduction']=='yes':
        print('Perfrom scenarios reduction using k-medoid algorithm')
        clustring_kmediod_PCA.kmedoid_clusters(path_test)
    #Do we need to perfrom the two stage stochastic programming using NSGA-II?
    if editable_data['Perform two stage optimization']=='yes':
        print('Perfrom two-stage stochastic optimization')
        problem= NSGA2_design_parallel_discrete.TwoStageOpt(path_test)
        with ProcessPoolEvaluator(int(editable_data['num_processors'])) as evaluator: #max number of accepted processors is 61 by program/ I have 8 processor on my PC
            algorithm = NSGAII(problem,population_size=int(editable_data['population_size']) ,evaluator=evaluator,variator=GAOperator(HUX(), BitFlip()))
            algorithm.run(int(editable_data['num_iterations']))
        NSGA2_design_parallel_discrete.results_extraction(problem, algorithm,path_test)
    #Do we need to generate Pareto-front and parallel coordinates plots for the results?
    if editable_data['Visualizing the final results']=='yes':
        file_name = city_DES+'_Discrete_EF_'+str(float(editable_data['renewable percentage']) )+'_design_'+str(editable_data['num_iterations'])+'_'+str(editable_data['population_size'])+'_'+str(editable_data['num_processors'])+'_processors'
        results_path = os.path.join(sys.path[0], file_name)
        print(results_path)
        if not os.path.exists(results_path):
            print('The results folder is not available. Please, generate the results first')
            sys.exit()
        sys.path.insert(0, Two_stage_SP_folder)
        import plot_results_design
        plot_results_design.ParetoFront_EFs(path_test)
        plot_results_design.parallel_plots('cost',path_test)
        plot_results_design.parallel_plots('emissions',path_test)
        file_name = editable_data['city']+'_Discrete_EF_'+str(float(editable_data['renewable percentage']) )+'_design_'+str(editable_data['num_iterations'])+'_'+str(editable_data['population_size'])+'_'+str(editable_data['num_processors'])+'_processors'
        print('Plots are generated in the '+ file_name+' folder')
