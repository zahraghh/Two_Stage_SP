import numpy as np
import pandas as pd
import csv
from collections import defaultdict
from scipy import stats
import os
import sys

editable_data_path =os.path.join(sys.path[0], 'EditableFile.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
city = '/'+editable_data['city']
folder_path = os.path.join(sys.path[0])+str(city)
save_path = os.path.join(sys.path[0]) + str('\Scenario Generation')
if not os.path.exists(save_path):
    os.makedirs(save_path)
lbstokg_convert = 0.453592 #1 lb = 0.453592 kg
def scenario_generation():
    #Normal distribution for electricity emissions
    city_EF = int(editable_data['city EF'])
    electricity_EF = city_EF*lbstokg_convert/1000
    electricity_EF_high = (city_EF+1.732051*154)*lbstokg_convert/1000
    electricity_EF_low = (city_EF-1.732051*154)*lbstokg_convert/1000
    lat = float(editable_data['Latitude'])
    lon = float(editable_data['Longitude'])
    year = int(editable_data['ending_year']) #Getting the last year of wind and solar data from NSRDB
    info_name =city+'_'+str(lat)+'_'+str(lon)+'_psm3_60_'+str(year)+'.csv'
    weather_data = pd.read_csv(folder_path+info_name, header=None)[2:]
    GTI_distribution = pd.read_csv(folder_path+'/best_fit_GTI.csv')
    wind_speed_distribution = pd.read_csv(folder_path+'/best_fit_wind_speed.csv')
    energy_data = pd.read_csv(os.path.join(sys.path[0])+'/Total_energy_demands.csv')
    weather_data = pd.read_csv(folder_path+info_name, header=None)[2:]
    GTI = [float(i) for i in list(weather_data[46])]
    wind_speed = [float(i) for i in list(weather_data[8])]
    energy_demand_scenario = {}
    solar_scenario = defaultdict(list)
    wind_scenario = defaultdict(list)
    electricity_scenario = defaultdict(list)
    heating_scenario = defaultdict(list)
    cooling_scenario = defaultdict(list)
    electricity_emissions_scenario = defaultdict(list)
    for i in range(8760):
        #Energy demnad uses uniform distribution from AEO 2021 --> 3-point approximation
        ## Energy Demand
        ## range cooling energy = (0.9*i, i)=(0,(i-0.9i)/0.1i) --> low = 0.112702 = (x-0.9i)/0.1i --> x = tick
        ## range heating energy = 0.69*i, i)
        ## range electricity energy = (0.91*i, i)
        electricity_scenario['low'].append(energy_data['Electricity (kWh)'][i]*(0.1*0.112702+0.91))
        heating_scenario['low'].append(energy_data['Heating (kWh)'][i]*(0.1*0.112702+0.69))
        cooling_scenario['low'].append(energy_data['Cooling (kWh)'][i]*(0.1*0.112702+0.90))
        electricity_scenario['medium'].append(energy_data['Electricity (kWh)'][i]*(0.1*0.50+0.91))
        heating_scenario['medium'].append(energy_data['Heating (kWh)'][i]*(0.1*0.50+0.69))
        cooling_scenario['medium'].append(energy_data['Cooling (kWh)'][i]*(0.1*0.50+0.90))
        electricity_scenario['high'].append(energy_data['Electricity (kWh)'][i]*(0.1*0.887298+0.91))
        heating_scenario['high'].append(energy_data['Heating (kWh)'][i]*(0.1*0.887298+0.69))
        cooling_scenario['high'].append(energy_data['Cooling (kWh)'][i]*(0.1*0.887298+0.90))

        if GTI_distribution['Mean'][i] == 0:
            solar_scenario['low'].append(GTI[i])
            solar_scenario['medium'].append(GTI[i])
            solar_scenario['high'].append(GTI[i])
        ## If Solar GTI is normal: from Rice & Miller  low = 0.112702 = (x-loc)/scale -->  =tick
        elif GTI_distribution['Best fit'][i] == 'norm':
            mu = GTI_distribution['Mean'][i]
            sigma = GTI_distribution['STD'][i]
            solar_scenario['low'].append(-1.732051*sigma+mu)
            solar_scenario['medium'].append(mu)
            solar_scenario['high'].append(1.732051*sigma+mu)
        ## If Solar GTI is uniform: from Rice & Miller low = 0.112702 (i - loc)/scale
        elif GTI_distribution['Best fit'][i] == 'uniform':
            loc = GTI_distribution['Mean'][i]
            scale = GTI_distribution['STD'][i]
            solar_scenario['low'].append(0.112702*scale+loc)
            solar_scenario['medium'].append(0.5*scale+loc)
            solar_scenario['high'].append(0.887298*scale+loc)
        ## If Solar GTI is expon: from Rice & Miller low = 0.415775 (i - loc)/scale, scale/scale)
        elif GTI_distribution['Best fit'][i] == 'expon':
            loc = GTI_distribution['Mean'][i]
            scale = GTI_distribution['STD'][i]
            solar_scenario['low'].append(0.415775*scale+loc)
            solar_scenario['medium'].append(2.294280*scale+loc)
            solar_scenario['high'].append(6.289945*scale+loc)
        if wind_speed_distribution['Mean'][i] == 0:
            wind_scenario['low'].append(wind_speed[i])
            wind_scenario['medium'].append(wind_speed[i])
            wind_scenario['high'].append(wind_speed[i])
        ## If Solar GTI is normal: from Rice & Miller  low = 0.112702 = (x-loc)/scale -->  =tick
        elif wind_speed_distribution['Best fit'][i] == 'norm':
            mu = wind_speed_distribution['Mean'][i]
            sigma = wind_speed_distribution['STD'][i]
            wind_scenario['low'].append(-1.732051*sigma+mu)
            wind_scenario['medium'].append(mu)
            wind_scenario['high'].append(1.732051*sigma+mu)
        ## If Solar GTI is uniform: from Rice & Miller low = 0.112702 (i - loc)/scale
        elif wind_speed_distribution['Best fit'][i] == 'uniform':
            loc = wind_speed_distribution['Mean'][i]
            scale = wind_speed_distribution['STD'][i]
            wind_scenario['low'].append(0.112702*scale+loc)
            wind_scenario['medium'].append(0.5*scale+loc)
            wind_scenario['high'].append(0.887298*scale+loc)
        ## If Solar GTI is expon: from Rice & Miller low = 0.415775 (i - loc)/scale, scale/scale)
        elif wind_speed_distribution['Best fit'][i] == 'expon':
            loc = wind_speed_distribution['Mean'][i]
            scale = wind_speed_distribution['STD'][i]
            wind_scenario['low'].append(0.415775*scale+loc)
            wind_scenario['medium'].append(2.294280*scale+loc)
            wind_scenario['high'].append(6.289945*scale+loc)
    electricity_emissions_scenario['low'] = 8760*[electricity_EF_low]
    electricity_emissions_scenario['medium'] = 8760*[electricity_EF]
    electricity_emissions_scenario['high'] = 8760*[electricity_EF_high]
    range_data = ['low','medium','high']
    scenario_genrated = {}
    scenario_genrated_normalized = {}
    for i_demand in range_data:
        for i_solar in range_data:
            for i_wind in range_data:
                for i_emission in range_data:
                    scenario_genrated['D:'+i_demand+'/S:'+i_solar+'/W:'+i_wind+'/C:'+i_emission] = {'Total Electricity (kWh)': [x + y for x, y in zip(cooling_scenario[i_demand],electricity_scenario[i_demand])],
                            'Heating (kWh)':heating_scenario[i_demand],
                            'GTI (Wh/m^2)':solar_scenario[i_solar],
                            'Wind Speed (m/s)': wind_scenario[i_wind],
                            'Electricity (kWh)':electricity_scenario[i_demand],
                            'Cooling (kWh)':cooling_scenario[i_demand],
                            'Electricity Emission Factor': electricity_emissions_scenario[i_emission]
                            }
                    df_scenario_generated=pd.DataFrame(scenario_genrated['D:'+i_demand+'/S:'+i_solar+'/W:'+i_wind+'/C:'+i_emission])
                    df_scenario_generated.to_csv(save_path + '\D_'+i_demand+'_S_'+i_solar+'_W_'+i_wind+'_C_'+i_emission+'.csv', index=False)
