import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import scipy.stats as st
import statsmodels as sm
import seaborn as sns
import math
import collections
from collections import Counter
import statistics
import matplotlib
# By state level - Fuels emission factors - 1999 to 2017
HHV_Coal= 19.73 # Coke Coal HHV (mm BTU/ short ton) Source: Emission Factors for Greenhouse Gas Inventories
HHV_Gas= 1.033 # Natural gas HHV (mmBtu/ mcf) Source: Emission Factors for Greenhouse Gas Inventories
HHV_Pet= 0.145*42 # Crude Oil (close to distilled Oil) HHV (mmBtu/ barrel) Source: Emission Factors for Greenhouse Gas Inventories

# Input Files
emissions_regions = pd.read_csv('https://raw.githubusercontent.com/zahraghh/EmissionFactorElectricity/master/emission_annual_state.csv')
generation_regions = pd.read_csv('https://raw.githubusercontent.com/zahraghh/EmissionFactorElectricity/master/annual_generation_state.csv')
consumption_regions = pd.read_csv('https://raw.githubusercontent.com/zahraghh/EmissionFactorElectricity/master/consumption_annual_state.csv')
US_states= pd.read_csv('https://raw.githubusercontent.com/zahraghh/EmissionFactorElectricity/master/US_states.csv')
#  Characteristics of Inputs
states= US_states['States']
emissions_regions_year= emissions_regions['Year']
consumption_regions_year= consumption_regions['YEAR']
generation_regions_year= generation_regions['YEAR']
emissions_regions_state= emissions_regions['State']
consumption_regions_state= consumption_regions['STATE']
generation_regions_state= generation_regions['STATE']
emissions_regions_type= emissions_regions['Producer Type']
consumption_regions_type= consumption_regions['TYPE OF PRODUCER']
generation_regions_type= generation_regions['TYPE OF PRODUCER']
emissions_regions_source= emissions_regions['Energy Source']
consumption_regions_source= consumption_regions['ENERGY SOURCE']
generation_regions_source= generation_regions['ENERGY SOURCE']
emissions_regions_CO2= emissions_regions['CO2']  # metric tones
emissions_regions_SO2= emissions_regions['SO2']  # metric tones
emissions_regions_NOx= emissions_regions['Nox']  # metric tones
consumption_regions_fuels= consumption_regions['CONSUMPTION for ELECTRICITY'] #Short tones, barrel, Mcf
generation_regions_fuels= generation_regions['GENERATION (Megawatthours)'] #Short tones, barrel, Mcf

def EF(year, Fuel, emissions_regions_xxx, Electric_scale, HHV):
    emission= []
    emission_state = []
    consumption= []
    consumption_state= []
    generation= []
    generation_state= []
    for k in range(50):
      for i in range(len(emissions_regions)):
        if emissions_regions_year[i]== year and emissions_regions_state[i]== states[k] and emissions_regions_type[i]== Electric_scale and emissions_regions_source[i]== Fuel:
            emission.append(emissions_regions_xxx[i]*1000) # converting metric ton to kg CO2/SO2/NOx
            emission_state.append(states[k])
      for j in range(len(consumption_regions)):
        if consumption_regions_year[j]== year and consumption_regions_state[j]==states[k] and consumption_regions_type[j]==Electric_scale and consumption_regions_source[j]==Fuel:
            consumption.append(int(consumption_regions_fuels[j])*HHV) # converting original unit to mmBTU
            consumption_state.append(states[k])
      for m in range(len(generation_regions)):
        if generation_regions_year[m]== year and generation_regions_state[m]==states[k] and generation_regions_type[m]==Electric_scale and generation_regions_source[m]==Fuel:
            generation.append(int(generation_regions_fuels[m])) # MWh electricity generation from each fuel type
            generation_state.append(states[k])
    dict_c={} # Dictonary map for  consumption
    dict_e={} # Dictonary map for  emissions
    dict_g={} # Dictonary map for  generation

    for c in range(len(consumption)):
      dict_c[consumption_state[c]]= consumption[c]
    for k in range(50):
      if not states[k] in dict_c.keys():
        dict_c[states[k]]=0
    for e in range(len(emission)):
      dict_e[emission_state[e]]= emission[e]
    for k in range(50):
      if not states[k] in dict_e.keys():
        dict_e[states[k]]=0
    for g in range(len(generation)):
      dict_g[generation_state[g]]= generation[g]
    for k in range(50):
      if not states[k] in dict_g.keys():
        dict_g[states[k]]=0

    EF_st={k: dict_e[k]/ dict_c[k] for k in dict_e.keys() & dict_c if dict_c[k]}
    GE_st={k: dict_e[k]/ dict_g[k] for k in dict_e.keys() & dict_c if dict_c[k]}

    return dict_c, dict_e, dict_g, EF_st, GE_st ## mmBTU, kg CO2, MWh
EF_coal_results =  EF(2017, 'Coal', emissions_regions_CO2, 'Total Electric Power Industry', HHV_Coal)  #kg CO2/ mmBTU
EF_gas_results =  EF(2017, 'Natural Gas', emissions_regions_CO2, 'Total Electric Power Industry', HHV_Gas)  #kg CO2/ mmBTU
EF_pet_results =  EF(2017, 'Petroleum', emissions_regions_CO2, 'Total Electric Power Industry', HHV_Pet)  #kg CO2/ mmBTU
EF_coal_list=list(EF_coal_results[3].values())   #kg CO2/mm BTU
EF_gas_list=list(EF_gas_results[3].values())  #kg CO2/mm BTU
EF_pet_list=list(EF_pet_results[3].values()) #kg CO2/mm BTU
GE_coal_list=list(EF_coal_results[4].values())   #kg CO2/MWh
GE_gas_list=list(EF_gas_results[4].values()) #kg CO2/MWh
GE_pet_list=list(EF_pet_results[4].values())   #kg CO2/MWh
EF_coal_list =  [i for i in EF_coal_list if 85 < i < 120]
EF_gas_list =  [i for i in EF_gas_list if 43 < i < 63]
EF_pet_list =  [i for i in EF_pet_list if 50 < i < 100]

def printinfo(list_EF):
  return print( "/STD: ", round(statistics.stdev(list_EF),2),"/Mean: ",round(statistics.mean(list_EF),2),"/Median: ",round(statistics.median(list_EF),2),
               "/Coef of variation %: ", round(statistics.stdev(list_EF)*100/statistics.mean(list_EF),2),
               "/Relative Range: ", round((max(list_EF)-min(list_EF))/statistics.mean(list_EF),2))

#Electricity CO2 EF
bins=20
def fit_and_plot(dist,data):
    params = dist.fit(data)
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    x = np.linspace( min(data), 150, bins)
    bin_centers = 0.5*(x[1:] + x[:-1])
    x = (x + np.roll(x, -1))[:-1] / 2.0
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    pdf= dist.pdf(bin_centers, loc=loc, scale=scale, *arg)

    return x, y, params, arg, loc, scale,

num_simulations=1
num_reps=10000
coal_params=fit_and_plot(st.levy_stable, EF_coal_list)
gas_params=fit_and_plot(st.lognorm, EF_gas_list)
pet_params=fit_and_plot(st.johnsonsu, EF_pet_list)

def EGEF_state(state):
    state_stats = []
    for i in range(num_simulations):
        # Choose random inputs for the uncertain inputs: Coal, Natural gas, Petroleum.
        coal_EF_rd = st.levy_stable.rvs(alpha=coal_params[2][0], beta=coal_params[2][1], loc= coal_params[2][2] , scale= coal_params[2][3] , size=num_reps)
        gas_EF_rd = st.lognorm.rvs(s=gas_params[2][0], loc= gas_params[2][1] , scale= gas_params[2][2] , size=num_reps)
        pet_EF_rd = st.johnsonsu.rvs(a=pet_params[2][0], b=pet_params[2][1], loc= pet_params[2][2] , scale= pet_params[2][3] , size=num_reps)
        electricty_generation_total_state = generation_regions[generation_regions['STATE']==state][generation_regions['YEAR']==2017][generation_regions['TYPE OF PRODUCER']=='Total Electric Power Industry'][generation_regions['ENERGY SOURCE']=='Total']['GENERATION (Megawatthours)']
        state_stats.append((coal_EF_rd*EF_coal_results[0][state] +  gas_EF_rd*EF_gas_results[0][state] +  pet_EF_rd*EF_pet_results[0][state])*2.20462/float(electricty_generation_total_state)) # EF_Electriicty (lb/MWh) Average distribution of fuels in the U.S.
    data_new= state_stats
    return data_new
matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')
# Create models from data
def best_fit_distribution(data,  ax=None):
  """Model data by finding best fit distribution to data"""
  # Get histogram of original data
  y, x = np.histogram(data, bins='auto', density=True)
  x = (x + np.roll(x, -1))[:-1] / 2.0
  # Distributions to check
  #DISTRIBUTIONS = [
    #st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,st.hypsecant,
    #st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
    #st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
    #st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
    #st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.invgamma,st.invgauss,
    #st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,
    #st.levy,st.levy_l,st.levy_stable,  #what's wrong with these distributions?
    #st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
    #st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
    #st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
    #st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy]
  DISTRIBUTIONS = [st.norm, st.uniform, st.expon]
  # Best holders
  best_distribution = st.norm  # random variables
  best_params = (0.0, 1.0)
  best_sse = np.inf
  # Estimate distribution parameters from data
  for distribution in DISTRIBUTIONS:
      # fit dist to data
      params = distribution.fit(data)
      if len(params)>2:
          print('here',distribution)
      #warnings.filterwarnings("ignore")
      # Separate parts of parameters
      arg = params[:-2]
      loc = params[-2]
      scale = params[-1]
      # Calculate fitted PDF and error with fit in distribution
      pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
      sse = np.sum(np.power(y - pdf, 2.0))
      # if axis pass in add to plot
      try:
          if ax:
              pd.Series(pdf, x).plot(ax=ax)
          end
      except Exception:
          pass
      # identify if this distribution is better
      if best_sse > sse > 0:
          best_distribution = distribution
          best_params = params
          best_sse = sse
  return (best_distribution.name, best_params)
