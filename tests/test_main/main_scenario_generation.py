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
