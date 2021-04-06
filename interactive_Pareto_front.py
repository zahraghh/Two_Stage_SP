import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
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
label_data = {}
for EF_i in EF_electricity:
    scatter_data[EF_i] = pd.read_csv(data_path[EF_i] + 'objectives.csv')
    label_data[EF_i] = pd.read_csv(data_path[EF_i] + 'sizing_all.csv')
#Define the EF here
EF =0.29
#Reference code:https://stackoverflow.com/questions/7908636/possible-to-make-labels-appear-when-hovering-over-a-point-in-matplotlib
# define 1 color
labels_color_map = {0: '#20b2aa'}
# set the examples count
no_examples = len(scatter_data[EF]['Emission (kg CO2)'])
# generate the data needed for the scatterplot
generated_data = [(scatter_data[EF]['Cost ($)'][i], scatter_data[EF]['Emission (kg CO2)'][i]) for i in range(no_examples)]
generated_labels = ["A_solar= {a_solar}, A_swept= {a_swept}, \n CHP_CAP= {chp_cap}, Boilers_CAP= {boilers_cap}, \n Battery_CAP= {battery_cap}".format(a_solar=round(label_data[EF]['Solar Area (m^2)'][i],1), a_swept=round(label_data[EF]['Swept Area (m^2)'][i],1), chp_cap=round(label_data[EF]['CHP Electricty Capacity (kW)'][i],1), boilers_cap=round(label_data[EF]['Boilers Capacity (kW)'][i],1),battery_cap=round(label_data[EF]['Battery Capacity (kW)'][i],1)) for i in range(no_examples)]

x = [i/10**6 for i in scatter_data[EF]['Cost ($)']]
y = [j/10**6 for j in scatter_data[EF]['Emission (kg CO2)']]
names = generated_labels
c = np.random.randint(1,2,size=no_examples)
norm = plt.Normalize(1,4)
cmap = plt.cm.RdYlGn
fig,ax = plt.subplots()
sc = plt.scatter(x,y,c=c, s=100, cmap=cmap, norm=norm)
annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)
def update_annot(ind):

    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}".format(" ".join([names[n] for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    annot.get_bbox_patch().set_alpha(0.4)
def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()
fig.canvas.mpl_connect("motion_notify_event", hover)
plt.xlabel("Cost (million-$)")
plt.ylabel("Emissions (million-kg $CO_2$)")
plt.show()
