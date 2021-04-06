from SolarIrradiance import aoi, get_total_irradiance
from Solarposition import get_solarposition
from pvlib import atmosphere, solarposition, tools
import csv
from csv import writer, reader
import pandas as pd
import datetime
import os
import sys

editable_data_path =os.path.join(sys.path[0], 'EditableFile.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
lat = float(editable_data['Latitude'])
lon = float(editable_data['Longitude'])
altitude = float(editable_data['Altitude']) #SLC altitude m
surf_tilt = float(editable_data['solar_tilt']) #panels tilt degree
surf_azimuth = float(editable_data['solar_azimuth']) #panels azimuth degree
class gti:
    def __init__(self,_year,_city):
        self.city = _city
        self.year = _year
        self.info_name =self.city+'_'+str(lat)+'_'+str(lon)+'_psm3_60_'+str(self.year)+'.csv'
        self.weather_data = pd.read_csv(folder_path+self.info_name, header=None)[2:]
    def process_gti(self):
        DNI= self.weather_data[5]
        DHI = self.weather_data[6]
        GHI = self.weather_data[7]
        dti = pd.date_range(str(self.year)+'-01-01', periods=365*24, freq='H')
        solar_position = get_solarposition(dti, lat, lon, altitude, pressure=None, method='nrel_numpy', temperature=12)
        solar_zenith = solar_position['zenith']
        solar_azimuth =  solar_position['azimuth']
        poa_components_vector = []
        poa_global = []
        for i in range(len(solar_zenith)):
            poa_components_vector.append(get_total_irradiance(surf_tilt, surf_azimuth,
                                     solar_zenith[i], solar_azimuth[i],
                                    float(DNI[3+i]), float(GHI[3+i]), float(DHI[3+i]), dni_extra=None, airmass=None,
                                     albedo=.25, surface_type=None,
                                     model='isotropic',
                                     model_perez='allsitescomposite1990'))
            poa_global.append(poa_components_vector[i]['poa_global'])
        csv_input = pd.read_csv(folder_path+self.info_name, header=None)[2:]
        poa_global.insert(0,'GTI')
        csv_input['ghi'] = poa_global
        csv_input.to_csv(folder_path+self.info_name, index=False)
        return poa_global
def GTI(city_DES):
    city = city_DES
    global folder_path
    folder_path = os.path.join(sys.path[0])+str(city)
    for year in range(int(editable_data['starting_year']),int(editable_data['ending_year'])+1):
        print('Calculating the global tilted irradiance on a surface in '+city+' in '+str(year))
        weather_year = gti(year,city)
        weather_year.process_gti()
