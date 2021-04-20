from platypus import NSGAII, Problem, Real, Integer, InjectedPopulation,GAOperator,HUX, BitFlip, SBX,PM,PCX,nondominated,ProcessPoolEvaluator
# I use platypus library to solve the muli-objective optimization problem:
# https://platypus.readthedocs.io/en/latest/getting-started.html
from pyomo.opt import SolverFactory
import pyomo.environ as pyo
import pandas as pd
import matplotlib.pyplot as plt
import csv
import math
import datetime as dt
from collections import defaultdict
import os
import sys
from pathlib import Path
components_path = os.path.join(sys.path[0])+'/Energy Components/'
editable_data_path =os.path.join(sys.path[0], 'editable_values.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
city = '/'+editable_data['city']
num_components = 0
use_solar_PV = editable_data['Solar_PV']
use_wind_turbine = editable_data['Wind_turbines']
use_battery = editable_data['Battery']
use_grid = editable_data['Grid']
use_CHP = editable_data['CHP']
use_boilers = editable_data['Boiler']
representative_days_path = os.path.join(sys.path[0])+ '\Scenario Generation'+city+ '\Representative days'
import Two_Stage_SP
from Two_Stage_SP.battery import battery_calc
from Two_Stage_SP.boilers import NG_boiler
from Two_Stage_SP.solar_PV import solar_pv_calc
from Two_Stage_SP.CHP_system import CHP
from Two_Stage_SP.wind_turbine import wind_turbine_calc
renewable_percentage = float(editable_data['renewable percentage'])  #Amount of renewables at the U (100% --> 1,mix of 43% grid--> 0.463, mix of 29% grid--> 0.29, 100% renewables -->0)
if use_boilers=='yes':
    num_components +=1
    boiler_component = pd.read_csv(components_path+'boilers.csv')
if use_CHP=='yes':
    num_components +=1
    CHP_component = pd.read_csv(components_path+'CHP.csv')
if use_solar_PV=='yes':
    num_components +=1
    PV_module = float(editable_data['PV_module']) #area of each commercial PV moduel is 1.7 M^2
    roof_top_area = float(editable_data['roof_top_area']) #60% percentage of the rooftop area of all buildings https://www.nrel.gov/docs/fy16osti/65298.pdf
if use_wind_turbine=='yes':
    num_components +=1
    wind_component = pd.read_csv(components_path+'wind_turbine.csv')
if use_battery=='yes':
    num_components +=1
    battery_component = pd.read_csv(components_path+'battery.csv')
###System Parameters## #
year = int(editable_data['ending_year'])
electricity_prices =  float(editable_data['electricity_price'])/100 #6.8cents/kWh in Utah -->$/kWh CHANGE
lifespan_project = float(editable_data['lifespan_project']) #life span of DES
UPV_elect = float(editable_data['UPV_elect']) #https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.85-3273-34.pdf discount rate =3% page 21 utah
num_clusters = int(editable_data['Cluster numbers'])

class TwoStageOpt(Problem):
    def __init__(self):
        energy_component_type = 0
        self.energy_component_number = {}
        super(TwoStageOpt, self).__init__(int(num_components), 2, 1)  #To create a problem with five decision variables, two objectives, and one constraint
        if use_solar_PV=='yes':
            self.energy_component_number['solar_PV']=energy_component_type
            self.types[energy_component_type] = Integer(0, int(roof_top_area/PV_module)) #Decision space for A_solar
            energy_component_type +=1
        if use_wind_turbine=='yes':
            self.energy_component_number['wind_turbine']=energy_component_type
            self.types[energy_component_type] = Integer(0, len(wind_component['Number'])-1) #Decision space for A_swept
            energy_component_type +=1
        if use_CHP=='yes':
            self.energy_component_number['CHP']=energy_component_type
            self.types[energy_component_type] = Integer(0, len(CHP_component['Number'])-1) #Decision space for CHP capacity
            energy_component_type +=1
        if use_boilers=='yes':
            self.energy_component_number['boilers']=energy_component_type
            self.types[energy_component_type] = Integer(0, len(boiler_component['Number'])-1) #Decision space for boiler capacity
            energy_component_type +=1
        if use_battery=='yes':
            self.energy_component_number['battery']=energy_component_type
            self.types[energy_component_type] = Integer(0, len(battery_component['Number'])-1) #Decision space for battery capacity
            energy_component_type +=1
        self.constraints[0] = ">=0" #Constraint to make sure heating demand can be satisfied using the boiler and CHP etc.
    def evaluate(self, solution):
        represent_day_max_results = self.represent_day_max()
        if use_solar_PV=='yes':
            self.A_solar = solution.variables[self.energy_component_number['solar_PV']]*PV_module
        else:
            self.A_solar = 0
        if use_wind_turbine=='yes':
            self.A_swept = solution.variables[self.energy_component_number['wind_turbine']]
            self.A_swept= wind_component['Swept Area m^2'][self.A_swept]
        else:
            self.A_swept = 0
        if use_CHP=='yes':
            self.CAP_CHP_elect = solution.variables[self.energy_component_number['CHP']]
            self.CAP_CHP_elect= CHP_component['CAP_CHP_elect_size'][self.CAP_CHP_elect]
        else:
            self.CAP_CHP_elect = 0
        if use_boilers=='yes':
            self.CAP_boiler = solution.variables[self.energy_component_number['boilers']]
            self.CAP_boiler= boiler_component['CAP_boiler (kW)'][self.CAP_boiler]
        else:
            self.CAP_boiler = 0
        if use_battery=='yes':
            self.CAP_battery = solution.variables[self.energy_component_number['battery']]
            self.CAP_battery= battery_component['CAP_battery (kWh)'][self.CAP_battery]
        else:
            self.CAP_battery = 0
        operating_cost_initialized = self.operating_cost(self.A_solar,self.A_swept,self.CAP_CHP_elect,self.CAP_boiler,self.CAP_battery)
        solution.objectives[:] = [operating_cost_initialized[0],operating_cost_initialized[1]]
        solution.constraints[0] = Two_Stage_SP.CHP_system.CHP(self.CAP_CHP_elect,0)[6] + self.CAP_boiler - represent_day_max_results[1]
    def operating_cost(self,_A_solar,_A_swept,_CAP_CHP_elect,_CAP_boiler,_CAP_battery):
        A_swept = _A_swept #Swept area of rotor m^2
        A_solar = _A_solar #Solar durface m^2 --> gives 160*A_solar W solar & needs= A_solar/0.7 m^2 rooftop
        CAP_CHP_elect = _CAP_CHP_elect #kW
        CAP_boiler = _CAP_boiler #kW
        CAP_battery = _CAP_battery #kW
        sum_emissions_total = []
        sum_cost_total = []
        sum_emissions = []
        sum_cost = []
        electricity_demand = {}
        heating_demand = {}
        representative_day = {}
        for represent in range(num_clusters):
            E_bat = {}
            representative_day[represent] = pd.read_csv(representative_days_path +'\Represent_days_modified_'+str(represent)+'.csv')
            self.G_T = list(representative_day[represent]['GTI (Wh/m^2)']) #Global Tilted Irradiation (Wh/m^2) in Slat Lake City from TMY3 file for 8760 hr a year on a titled surface 35 deg
            self.V_wind = list(representative_day[represent]['Wind Speed (m/s)']) #Wind Speed m/s in Slat Lake City from AMY file for 8760 hr in 2019
            electricity_demand = representative_day[represent]['Electricity total (kWh)'] #kWh
            heating_demand = representative_day[represent]['Heating (kWh)'] #kWh
            self.electricity_EF = representative_day[represent]['Electricity EF (kg/kWh)'][0]*renewable_percentage #kg/kWh
            probability_represent = representative_day[represent]['Percent %'][0] #probability that representative day happens
            num_days_represent = probability_represent*365/100 #Number of days in a year that representative day represent :)
            for hour in range(0,24):
                V_wind_now = self.V_wind[hour]
                G_T_now = self.G_T[hour]
                if hour==0:
                    E_bat[hour]=0
                battery_results = Two_Stage_SP.battery.battery_calc(electricity_demand[hour],hour,E_bat[hour],A_solar,A_swept,CAP_battery,G_T_now,V_wind_now)
                electricity_demand_new = battery_results[2]
                E_bat[hour+1]= battery_results[0]
                results = self.evaluate_operation(self.electricity_EF,electricity_demand_new,heating_demand[hour],A_solar,A_swept,CAP_CHP_elect,CAP_boiler,CAP_battery)
                sum_cost.append(results[1])
                sum_emissions.append(results[3])
            sum_cost_total.append(sum(sum_cost)*num_days_represent)
            sum_emissions_total.append(sum(sum_emissions)*num_days_represent)
        operating_cost = sum(sum_cost_total)
        operating_emissions = sum(sum_emissions_total)*lifespan_project
        capital_cost = Two_Stage_SP.solar_PV.solar_pv_calc(A_solar,hour,0,0,1)[1] + Two_Stage_SP.wind_turbine.wind_turbine_calc(A_swept,hour,0,0,1)[1] + Two_Stage_SP.CHP_system.CHP(CAP_CHP_elect,1)[2] + Two_Stage_SP.boilers.NG_boiler(1,CAP_boiler)[1] + Two_Stage_SP.battery.battery_calc(electricity_demand[hour],hour,E_bat[hour],A_solar,A_swept,CAP_battery,0,0)[1]
        total_cost = capital_cost+operating_cost
        return total_cost,operating_emissions, operating_cost,Two_Stage_SP.solar_PV.solar_pv_calc(A_solar,hour,0,0,1)[1],Two_Stage_SP.wind_turbine.wind_turbine_calc(A_swept,hour,0,0,1)[1], Two_Stage_SP.CHP_system.CHP(CAP_CHP_elect,results[5])[2],Two_Stage_SP.boilers.NG_boiler(results[7],CAP_boiler)[1],Two_Stage_SP.battery.battery_calc(electricity_demand[hour],hour,E_bat[hour],A_solar,A_swept,CAP_battery,0,0)[1]
    def evaluate_operation(self,_electricity_EF,_electricity_demand,_heating_demand,_A_swept,_A_solar,_CAP_CHP_elec,_CAP_boiler,_CAP_battery):
        A_swept = _A_swept #Swept area of rotor m^2
        A_solar = _A_solar #Solar durface m^2 --> gives 160*A_solar W solar & needs= A_solar/0.7 m^2 rooftop
        CAP_CHP_elect = _CAP_CHP_elec #kW
        CAP_boiler = _CAP_boiler #kW
        CAP_battery = _CAP_battery #kW
        heating_demand= _heating_demand
        model = pyo.ConcreteModel()
        if _electricity_demand==0:
            Cost_minimzed = Two_Stage_SP.boilers.NG_boiler(heating_demand/Two_Stage_SP.boilers.NG_boiler(0,CAP_boiler)[4],CAP_boiler)[2] #$
            emission_objective = Two_Stage_SP.boilers.NG_boiler(heating_demand/Two_Stage_SP.boilers.NG_boiler(0,CAP_boiler)[4],CAP_boiler)[3] #kg CO2
            return 'Cost ($)', Cost_minimzed,'Emissions (kg CO2)', emission_objective ,'CHP',0,'Boiler', heating_demand/Two_Stage_SP.boilers.NG_boiler(0,CAP_boiler)[4],'Grid',0
        else:
            if use_boilers == 'yes':
                F_F_boiler = 1
                model.F_boiler = pyo.Var(bounds=(0,CAP_boiler/Two_Stage_SP.boilers.NG_boiler(0,CAP_boiler)[4])) #Decision space for natural gas fuel rate
                Boiler_model = model.F_boiler
            else:
                F_F_boiler =  0
                Boiler_model = 0
            if use_CHP=='yes':
                F_F_CHP = 1
                model.F_CHP = pyo.Var(bounds=(0,Two_Stage_SP.CHP_system.CHP(CAP_CHP_elect,0)[5])) #Decision space for CHP fuel rate
                CHP_model = model.F_CHP
            else:
                F_F_CHP = 0
                CHP_model = 0
            if use_grid =='yes':
                F_E_grid = 1
                model.E_grid = pyo.Var(bounds=(0,_electricity_demand+1)) #Decision space for grid consumption
                grid_model = model.E_grid
            else:
                F_E_grid = 0
                grid_model = 0
            model.OBJ_cost = pyo.Objective(expr = Two_Stage_SP.CHP_system.CHP(CAP_CHP_elect,CHP_model)[3]*F_F_CHP + Two_Stage_SP.boilers.NG_boiler(Boiler_model,CAP_boiler)[2]*F_F_CHP + grid_model*electricity_prices*UPV_elect*F_E_grid) #$
            model.Constraint_elect = pyo.Constraint(expr = Two_Stage_SP.CHP_system.CHP(CAP_CHP_elect,CHP_model)[0]*F_F_CHP + grid_model*F_E_grid - _electricity_demand>=0) # Electricity balance of demand and supply sides
            model.Constraint_heat = pyo.Constraint(expr = Two_Stage_SP.boilers.NG_boiler(Boiler_model,CAP_boiler)[0]*F_F_boiler + Two_Stage_SP.CHP_system.CHP(CAP_CHP_elect,CHP_model)[1]*F_F_CHP -heating_demand>=0) # Heating balance of demand and supply sides
            #opt = SolverFactory('cplex')
            opt = SolverFactory('glpk')
            results =opt.solve(model,load_solutions=False)
            if len(results.solution)==0:
                return 'Cost ($)', 10000,'Emissions (kg CO2)', 100000 ,'CHP', 0,'Boiler', 0,'Grid', 0
            else:
                model.solutions.load_from(results)
                if use_boilers == 'yes':
                    value_Boiler_model = model.F_boiler.value
                    if CAP_boiler==0:
                        value_Boiler_model=0
                else:
                    value_Boiler_model = 0
                if use_CHP=='yes':
                    value_CHP_model=model.F_CHP.value
                    if CAP_CHP_elect==0:
                        value_CHP_model=0
                else:
                    value_CHP_model = 0
                if use_grid =='yes':
                    value_grid_model = model.E_grid.value
                else:
                    value_grid_model = 0
                Cost_minimzed =Two_Stage_SP.CHP_system.CHP(CAP_CHP_elect,value_CHP_model*F_F_CHP)[3]+Two_Stage_SP.boilers.NG_boiler(value_Boiler_model*F_F_boiler,CAP_boiler)[2]+value_grid_model*electricity_prices*F_E_grid
                emission_objective = Two_Stage_SP.CHP_system.CHP(CAP_CHP_elect,value_CHP_model*F_F_CHP)[4] + Two_Stage_SP.boilers.NG_boiler(value_Boiler_model*F_F_boiler,CAP_boiler)[3] +value_grid_model*_electricity_EF*F_E_grid #kg CO2
                return 'Cost ($)', Cost_minimzed,'Emissions (kg CO2)', emission_objective ,'CHP',value_CHP_model,'Boiler', value_Boiler_model,'Grid', value_grid_model
    def represent_day_max(self):
        electricity_demand_max = []
        heating_demand_max = []
        V_max = []
        GTI_max = []
        representative_day_max = {}
        electricity_demand_total = defaultdict(list)
        heating_demand_total = defaultdict(list)
        for represent in range(num_clusters):
            representative_day_max[represent] = pd.read_csv(representative_days_path +'\Represent_days_modified_'+str(represent)+'.csv')
            probability_represent = representative_day_max[represent]['Percent %'][0] #probability that representative day happens
            num_days_represent = probability_represent*365/100 #Number of days in a year that representative day represent :)
            electricity_demand = representative_day_max[represent]['Electricity total (kWh)'] #kWh
            heating_demand = representative_day_max[represent]['Heating (kWh)'] #kWh
            electricity_demand_total[represent] = electricity_demand*num_days_represent
            heating_demand_total[represent] = heating_demand*num_days_represent
            G_T = list(representative_day_max[represent]['GTI (Wh/m^2)']) #Global Tilted Irradiation (Wh/m^2) in Slat Lake City from TMY3 file for 8760 hr a year on a titled surface 35 deg
            V_wind = list(representative_day_max[represent]['Wind Speed (m/s)']) #Wind Speed m/s in Slat Lake City from AMY file for 8760 hr in 2019
            V_max.append(max(V_wind))
            GTI_max.append(max(G_T))
            electricity_demand_max.append(max(electricity_demand))
            heating_demand_max.append(max(heating_demand))
        sum_electricity = []
        sum_heating = []
        for key in electricity_demand_total.keys():
            sum_electricity.append(sum(electricity_demand_total[key]))
            sum_heating.append(sum(heating_demand_total[key]))
        #print('here',sum(sum_electricity),sum(sum_heating))
        return max(electricity_demand_max), max(heating_demand_max),max(GTI_max),max(V_max)
def results_extraction(problem, algorithm):
    file_name = city+'_Discrete_EF_'+str(float(editable_data['renewable percentage']) )+'_design_'+str(editable_data['num_iterations'])+'_'+str(editable_data['population_size'])+'_'+str(editable_data['num_processors'])+'_processors'
    results_path = os.path.join(sys.path[0]) + file_name
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    i = 0
    df_operation = {}
    df_cost ={}
    df_object = {}
    df_object_all  = pd.DataFrame(columns = ['Pareto number','Cost ($)','Emission (kg CO2)'])
    df_operation_all = pd.DataFrame(columns = ['Pareto number','Solar Area (m^2)','Swept Area (m^2)','CHP Electricty Capacity (kW)','Boilers Capacity (kW)','Battery Capacity (kW)','Cost ($)','Emission (kg CO2)'])
    df_cost_all = pd.DataFrame(columns = ['Pareto number','Solar ($)','Wind ($)','CHP ($)','Boilers ($)','Battery ($)','Operating Cost ($)','Operating Emissions (kg CO2)','Total Capital Cost ($)'])
    i=0
    solar_results = {}
    wind_results = {}
    CHP_results = {}
    boiler_results = {}
    battery_results = {}
    print('Generating the results')
    for s in algorithm.result:
        if use_solar_PV=='yes':
            solar_results[s] = s.variables[problem.energy_component_number['solar_PV']]
            if isinstance(solar_results[s], list):
                solar_results[s] = float(problem.types[problem.energy_component_number['solar_PV']].decode(solar_results[s]))
                solar_results[s]= solar_results[s]*PV_module
        else:
            solar_results[s]=0
        if use_wind_turbine=='yes':
            wind_results[s] = s.variables[problem.energy_component_number['wind_turbine']]
            if isinstance(wind_results[s], list):
                wind_results[s] = float(problem.types[problem.energy_component_number['wind_turbine']].decode(wind_results[s]))
                wind_results[s]=wind_component['Swept Area m^2'][wind_results[s]]
        else:
            wind_results[s]=0
        if use_CHP=='yes':
            CHP_results[s] = s.variables[problem.energy_component_number['CHP']]
            if isinstance(CHP_results[s], list):
                CHP_results[s] = float(problem.types[problem.energy_component_number['CHP']].decode(CHP_results[s]))
                CHP_results[s]=CHP_component['CAP_CHP_elect_size'][CHP_results[s]]
        else:
            CHP_results[s] = 0
        if use_boilers=='yes':
            boiler_results[s] = s.variables[problem.energy_component_number['boilers']]
            if isinstance(boiler_results[s], list):
                boiler_results[s] = float(problem.types[problem.energy_component_number['boilers']].decode(boiler_results[s]))
                boiler_results[s]=boiler_component['CAP_boiler (kW)'][boiler_results[s]]
        else:
            boiler_results[s] = 0
        if use_battery=='yes':
            battery_results[s] = s.variables[problem.energy_component_number['battery']]
            if isinstance(battery_results[s], list):
                battery_results[s] = float(problem.types[problem.energy_component_number['battery']].decode(battery_results[s]))
                battery_results[s]=battery_component['CAP_battery (kWh)'][battery_results[s]]
        else:
            battery_results[s] = 0
        print('Scenario:',i,', Energy components sizing: ',solar_results[s],wind_results[s],CHP_results[s],boiler_results[s],battery_results[s])
        print('Scenario:',i,', Total cost and emissions: ',s.objectives[0],'$',s.objectives[1], 'kg CO2')
        data_object = {'Pareto number':i,
        'Cost ($)':s.objectives[0],
        'Emission (kg CO2)':s.objectives[1]}
        data_operation={'Pareto number':i,
        'Solar Area (m^2)':solar_results[s],
        'Swept Area (m^2)':wind_results[s],
        'CHP Electricty Capacity (kW)':CHP_results[s],
        'Boilers Capacity (kW)':boiler_results[s],
        'Battery Capacity (kW)':battery_results[s],
        'Cost ($)':s.objectives[0],
        'Emission (kg CO2)':s.objectives[1]}
        cost_results = problem.operating_cost(solar_results[s],wind_results[s],CHP_results[s],boiler_results[s],battery_results[s])
        data_cost = {'Pareto number':i,
        'Solar ($)':cost_results[3],
        'Wind ($)':cost_results[4],
        'CHP ($)':cost_results[5],
        'Boilers ($)':cost_results[6],
        'Battery ($)':cost_results[7],
        'Operating Cost ($)':cost_results[2],
        'Operating Emissions (kg CO2)':cost_results[1],
        'Total Capital Cost ($)':cost_results[0]-cost_results[2]}
        print(data_object)
        df_object[i] =  pd.DataFrame(data_object,index=[0])
        df_object_all =  df_object_all.append(df_object[i])
        df_operation[i] = pd.DataFrame(data_operation,index=[0])
        df_operation_all =  df_operation_all.append(df_operation[i])
        df_cost[i] = pd.DataFrame(data_cost,index=[0])
        df_cost_all =  df_cost_all.append(df_cost[i])
        i += 1
    df_object_all.to_csv(results_path + '/objectives.csv')
    df_operation_all.to_csv(results_path + '/sizing_all.csv')
    #df_cost_all.to_csv(results_path + '/cost_all.csv')

    plt.scatter([s.objectives[0] for s in algorithm.result],
                [s.objectives[1] for s in algorithm.result])
    plt.xlabel("Cost ($)")
    plt.ylabel("Emissions kg ($CO_2$)")
    plt.close()
    print('Results are generated in the '+ file_name+' folder')
