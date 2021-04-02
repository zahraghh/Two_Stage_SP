from SolarIrradiance import aoi, get_total_irradiance
from Solarposition import get_solarposition
from pvlib import atmosphere, solarposition, tools
import csv
from csv import writer, reader
import pandas as pd
import datetime

folder_path = 'F:/Zahra/Research/Aim 2/salt_lake_city/'
year  = 2018 #getting the data for which year
latitude = 40.7608 #SLC latitude
longitude = 111.8910 #SLC longitude
altitude = 1288 #SLC altitude m
surf_tilt = 35 #panels tilt degree
surf_azimuth = 180 #panels azimuth degree
class gti:
    def __init__(self,_year):
        self.year = _year
        self.weather_data = pd.read_csv(folder_path+'salt_lake_city_40.758478_-111.888142_psm3_60_'+str(_year)+'.csv', header=None)[2:]
    def process_gti(self):
        DNI= self.weather_data[5]
        DHI = self.weather_data[6]
        GHI = self.weather_data[7]
        dti = pd.date_range(str(self.year)+'-01-01', periods=365*24, freq='H')
        solar_position = get_solarposition(dti, latitude, longitude, altitude, pressure=None, method='nrel_numpy', temperature=12)
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
        csv_input = pd.read_csv(folder_path+'salt_lake_city_40.758478_-111.888142_psm3_60_'+str(self.year)+'.csv', header=None)[2:]
        poa_global.insert(0,'GTI')
        csv_input['gti'] = poa_global
        csv_input.to_csv(folder_path+'salt_lake_city_40.758478_-111.888142_psm3_60_'+str(self.year)+'.csv', index=False)
        return poa_global
for year in range(1999,2020):
    print(year)
    weather_year = gti(year)
    weather_year.process_gti()
