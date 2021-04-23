import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import csv
import numpy as np
import plotly.express as px
import os
import sys
data_path = {}
editable_data_path =os.path.join(sys.path[0], 'editable_values.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
city =str(editable_data['city'])
file_name = city+'_Discrete_EF_'+str(float(editable_data['renewable percentage']) )+'_design_'+str(editable_data['num_iterations'])+'_'+str(editable_data['population_size'])+'_'+str(editable_data['num_processors'])+'_processors/'
results_path = os.path.join(sys.path[0], file_name)
scatter_data = {}
scatter_data_modified={}
label_data = {}
cost = {}
emissions = {}
scatter_data = pd.read_csv(os.path.join(results_path , 'objectives.csv'))
label_data = pd.read_csv(os.path.join(results_path , 'sizing_all.csv'))
cost = [i/10**6 for i in scatter_data['Cost ($)']]
emissions = [j/10**6 for j in scatter_data['Emission (kg CO2)']]
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.edgecolor'] = 'black'
cmap = plt.cm.RdYlGn
def ParetoFront_EFs():
    fig,ax = plt.subplots()
    c = 'tab:red'
    m = "o"
    plt.figure(figsize=(8, 5))
    plt.scatter(cost,emissions,color=c,marker=m, s=60, cmap=cmap)
    plt.title('Cost and emissions trade-off')
    plt.xlabel("Cost (million $)")
    plt.ylabel("Emissions (million kg $CO_2$)")
    plt.savefig(os.path.join(results_path ,'ParetoFront.png'),dpi=300,facecolor='w')
### Parallel coordinates plot of the sizings
#Ref: https://coderzcolumn.com/tutorials/data-science/how-to-plot-parallel-coordinates-plot-in-python-matplotlib-plotly
def parallel_plots(type_plot):
    label_data['Solar (sq-m)'] = label_data['Solar Area (m^2)']
    label_data['Swept (sq-m)'] = label_data['Swept Area (m^2)']
    label_data['Emissions (kg)'] = label_data['Emission (kg CO2)']
    label_data['Battery (kW)'] = label_data['Battery Capacity (kW)']

    if type_plot == 'cost':
        #cols = ['Solar Area (sq-m)', 'Swept Area (sq-m)', 'Boilers Capacity (kW)', 'CHP Electricty Capacity (kW)', 'Battery Capacity (kW)','Emission (kg CO2)','Cost ($)']
        cols = ['Solar (sq-m)', 'Swept (sq-m)', 'Battery (kW)','Emissions (kg)','Cost ($)']
        color_plot = px.colors.sequential.Blues
    if type_plot == 'emissions':
        cols = ['Solar (sq-m)', 'Swept (sq-m)', 'Battery (kW)','Cost ($)','Emissions (kg)']
        color_plot = px.colors.sequential.Reds
    fig_new = px.parallel_coordinates(label_data, color=cols[-1], dimensions=cols,
                                  color_continuous_scale=color_plot)
    fig_new.update_layout(
        font=dict(
            size=18,
        )
    )
    fig_new.write_image(os.path.join(results_path,'Parallel_coordinates_'+type_plot+'.png'),width=680, height=450,scale=3)
