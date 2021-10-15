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
    #Calculating the distribution of wind speed (might take ~5 mins)')
    uncertainty_analysis.probability_distribution('wind_speed',8) #Name and the column number in the weather data
