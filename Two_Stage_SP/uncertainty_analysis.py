import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import datetime as dt
import csv
import re
import math
from collections import defaultdict
from nested_dict import nested_dict
import os
import scipy.stats as st
import numpy as np
from matplotlib.ticker import FuncFormatter
from functools import partial
import warnings
from calendar import monthrange
import os
import sys
import pandas as pd
import csv
import PySAM.ResourceTools as tools
editable_data_path =os.path.join(sys.path[0], 'EditableFile.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
city = '/'+editable_data['city']
folder_path = os.path.join(sys.path[0])+str(city)
#Location Coordinates
lat = float(editable_data['Latitude'])
lon = float(editable_data['Longitude'])

weather_data = {}
def uncertain_input(type_input,number_weatherfile):
    uncertain_dist =  defaultdict(list)
    uncertain_input = {}
    for year in range(int(editable_data['starting_year']),int(editable_data['ending_year'])+1):
        weather_data[year] = pd.read_csv(folder_path+city+'_'+str(lat)+'_'+str(lon)+'_psm3_60_'+str(year)+'.csv', header=None)[2:]
        uncertain_input[year] = weather_data[year][number_weatherfile]
        for index_in_year in range(2,8762):
            uncertain_dist[index_in_year-2].append(float(uncertain_input[year][index_in_year]))
    return uncertain_dist

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
def fit_and_plot(dist,data,min_data, max_data):
    params = dist.fit(data)
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    y, x = np.histogram(data, bins='auto', density=True)
    bin_centers = 0.5*(x[1:] + x[:-1])
    fig_monte=plt.figure('probability distribution', figsize=(8,5), dpi=100)
    nplt, binsplt, patchesplt = plt.hist(data, bins='auto', range=(min_data,max_data), density= True)
    pdf= dist.pdf(bin_centers, loc=loc, scale=scale, *arg)
    percent_formatter = partial(to_percent ,k = sum(nplt))
    formatter = FuncFormatter(percent_formatter)    # Set the formatter
    plt.gca().yaxis.set_major_formatter(formatter)
    ax = fig_monte.add_subplot(111)
    ax.plot(bin_centers, pdf, linestyle ='-', color = "m", linewidth =3)
    ax.set_xlabel('DNI')
    ax.set_ylabel('Probability of DNI')
    plt.show()
    return dist, pdf
def to_percent(nplt,position, k):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(round(100*nplt/k,0))
    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'
# Find best fit distribution
def probability_distribution(name,column_number):
    dict_data = uncertain_input(name,column_number)
    best_fit_input = defaultdict(list)
    df_object = {}
    df_object_all  = pd.DataFrame(columns = ['Index in year','Best fit','Best loc','Best scale','Mean','STD'])
    for key in dict_data.keys():
        data = dict_data[key]
        ax_new = plt.hist(data, bins = 'auto', range=(min(data)*0.8,max(data)*1.1), density= True)
        best_fit_name, best_fit_params = best_fit_distribution(data,ax_new)
        best_fit_input['Name'].append(best_fit_name)
        best_fit_input['Params'].append(best_fit_params)
        data_frame_input  ={'Index in year': key,
        'Best fit': best_fit_name,
        'Best loc': best_fit_params[-2],
        'Best scale': best_fit_params[-1],
        'Mean': np.mean(data),
        'STD': np.std(data)}
        df_object[key] =  pd.DataFrame(data_frame_input,index=[0])
        df_object_all =  df_object_all.append(df_object[key])
    df_object_all.to_csv(folder_path + '/best_fit_'+name+'.csv')
