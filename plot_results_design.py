import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import csv
import numpy as np
import plotly.express as px
data_path = {}
save_path = 'F:/Zahra/Research/Aim 2/Design/'
data_path[1] = 'F:/Zahra/Research/Aim 2/Design/Discrete_EF_1.00_design_2000_50_5cores_112min/'
data_path[0.46] = 'F:/Zahra/Research/Aim 2/Design/Discrete_EF_0.46_design_2000_50_5cores_110min/'
data_path[0.29] = 'F:/Zahra/Research/Aim 2/Design/Discrete_EF_0.29_design_2000_50_5cores_112min/'
data_path[0] = 'F:/Zahra/Research/Aim 2/Design/Discrete_EF_0.0_design_2000_50_5cores_112min/'

EF_electricity = list(data_path.keys())
color_EF = {'0':'tab:red', '0.29':'tab:blue','0.46': 'tab:purple','1':'tab:orange'}
marker_EF = {'0':">" , '0.29':"," ,'0.46': "o" ,'1':"^"}
scatter_data = {}
scatter_data_modified={}
label_data = {}
cost = {}
emissions = {}
for EF_i in EF_electricity:
    scatter_data[EF_i] = pd.read_csv(data_path[EF_i] + 'objectives.csv')
    label_data[EF_i] = pd.read_csv(data_path[EF_i] + 'sizing_all.csv')
for EF in EF_electricity:
    cost[EF] = [i/10**6 for i in scatter_data[EF]['Cost ($)']]
    emissions[EF] = [j/10**6 for j in scatter_data[EF]['Emission (kg CO2)']]
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
def ParetoFront_EFs(EFs_list):
    fig,ax = plt.subplots()
    def plot_EF(EF):
        c = color_EF[str(EF)]
        m = marker_EF[str(EF)]
        cmap = plt.cm.RdYlGn
        plt.scatter(cost[EF],emissions[EF],color=c,marker=m, s=60, cmap=cmap,label='Case '+str(EFs_list.index(EF)+1)+' (EF= '+str(EF)+')')
        plt.xlabel("Cost (million $)")
        plt.ylabel("Emissions (million kg $CO_2$)")
        plt.savefig(save_path + 'newB_All_+'+str(EF)+'_ParetoFront.png',dpi=300)
    for EF in EFs_list:
        plot_EF(EF)
    plt.legend(loc='best',  fontsize=14)
    plt.savefig(save_path + 'newB_All_EFs_ParetoFront.png',dpi=400)
    plt.close()
    for EF in EFs_list:
        plot_EF(EF)
        plt.close()
ParetoFront_EFs([1,0.29,0.46,0])
###Violin plot to show cost-emissions distribution###
def violin_plot(data_list,type_plot,labels):
    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value
    def set_axis_style(ax, labels):
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Modified electricity EF')
        if type_plot=='cost':
            ax.set_ylabel("Cost (million $)")
        if type_plot=='emissions':
            ax.set_ylabel("Emissions (million kg $CO_2$)")
    fig, ax2 = plt.subplots(figsize=(7, 5), sharey=True)
    parts = ax2.violinplot(
            data_list, showmeans=False, showmedians=False,
            showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    quartile1, medians, quartile3 = np.percentile(data_list, [25, 50, 75], axis=1)
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(data_list, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    # set style for the axes
    set_axis_style(ax2, labels)
    plt.subplots_adjust(bottom=0.15, wspace=0.05)
    plt.savefig(save_path + 'new_B_EFs_violin_'+type_plot+'.png',dpi=300)
    plt.close()
# create test data
data_cost = []
data_emissions= []
labels = []
for EF in EF_electricity:
    data_cost.append(cost[EF])
    data_emissions.append(emissions[EF])
    labels.append('EF='+str(EF))
violin_plot(data_cost,'cost',labels)
violin_plot(data_emissions,'emissions',labels)

### Parallel coordinates plot of the sizings
#Ref: https://coderzcolumn.com/tutorials/data-science/how-to-plot-parallel-coordinates-plot-in-python-matplotlib-plotly
def parallel_plots(EF, type_plot):
    label_data[EF]['Solar (sq-m)'] = label_data[EF]['Solar Area (m^2)']
    label_data[EF]['Swept (sq-m)'] = label_data[EF]['Swept Area (m^2)']
    label_data[EF]['Emissions (kg)'] = label_data[EF]['Emission (kg CO2)']
    label_data[EF]['Battery (kW)'] = label_data[EF]['Battery Capacity (kW)']

    if type_plot == 'cost':
        #cols = ['Solar Area (sq-m)', 'Swept Area (sq-m)', 'Boilers Capacity (kW)', 'CHP Electricty Capacity (kW)', 'Battery Capacity (kW)','Emission (kg CO2)','Cost ($)']
        cols = ['Solar (sq-m)', 'Swept (sq-m)', 'Battery (kW)','Emissions (kg)','Cost ($)']
        color_plot = px.colors.sequential.Blues
    if type_plot == 'emissions':
        cols = ['Solar (sq-m)', 'Swept (sq-m)', 'Battery (kW)','Cost ($)','Emissions (kg)']
        color_plot = px.colors.sequential.Reds
    fig_new = px.parallel_coordinates(label_data[EF], color=cols[-1], dimensions=cols,
                                  color_continuous_scale=color_plot)
    fig_new.update_layout(
        font=dict(
            size=18,
        )
    )
    fig_new.write_image(save_path+'newB_parallel_'+type_plot+'_EF_'+str(EF)+'.png',width=680, height=450,scale=3)
for EF in EF_electricity:
    parallel_plots(EF,'cost')
    parallel_plots(EF,'emissions')
