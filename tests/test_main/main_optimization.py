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
