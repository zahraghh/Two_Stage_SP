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
editable_data_path =os.path.join(sys.path[0], 'EditableFile.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
representative_days_path = os.path.join(sys.path[0])+str('\Scenario Generation\Representative days')
components_path = os.path.join(sys.path[0])+'/Energy Components Capacities/'
wind_component = pd.read_csv(components_path+'wind_turbine.csv')
boiler_component = pd.read_csv(components_path+'boilers.csv')
CHP_component = pd.read_csv(components_path+'CHP.csv')
battery_component = pd.read_csv(components_path+'battery.csv')
solar_component = pd.read_csv(components_path+'solar_PV.csv')
renewable_percentage = float(editable_data['renewable percentage'])  #Amount of renewables at the U (100% --> 1, 46.3%, and 29%)

###System Parameters## #
city = editable_data['city']
year = int(editable_data['ending_year'])
CP_NG = int(editable_data['CP_ng']) #The heat content of  natural gas is 1,037 Btu per cubic foot
BTUtokWh_convert = 0.000293071 # 1BTU = 0.000293071 kWh
mmBTutoBTU_convert = 10**6
lbstokg_convert = 0.453592 #1 l b = 0.453592 kg
NG_prices = float(editable_data['price_NG'])/293.001 #Natural gas price at UoU $/kWh
electricity_prices =  float(editable_data['electricity_price'])/100 #6.8cents/kWh in Utah -->$/kWh CHANGE
electricity_EF = float(editable_data['city EF'])*lbstokg_convert/1000*renewable_percentage
lifespan_project = float(editable_data['lifespan_project']) #life span of DES
lifespan_solar = int(solar_component['Lifespan (year)'][0]) #lifespan of solar PV System
lifespan_wind = int(wind_component['Lifespan (year)'][0]) #lifespan of wind turbines
lifespan_chp = int(CHP_component['Lifespan (year)'][0])
lifespan_boiler = int(boiler_component['Lifespan (year)'][0])
cut_in_wind_speed = wind_component['Cut-in Speed'][0] #2.5 m/s is the minimum wind speed to run the wind turbines
UPV_maintenance = float(editable_data['UPV_maintenance']) #https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.85-3273-34.pdf discount rate =3% page 7
UPV_NG = float(editable_data['UPV_NG']) #https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.85-3273-34.pdf discount rate =3% page 21 utah
UPV_elect = float(editable_data['UPV_elect']) #https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.85-3273-34.pdf discount rate =3% page 21 utah
deltat = 1 #hour for batteries
num_clusters = int(editable_data['Cluster numbers'])
PV_module = float(editable_data['PV_module']) #area of each commercial PV moduel is 1.7 M^2
roof_top_area = float(editable_data['roof_top_area']) #60% percentage of the rooftop area of all buildings https://www.nrel.gov/docs/fy16osti/65298.pdf

class TwoStageOpt(Problem):
    def __init__(self):
        super(TwoStageOpt, self).__init__(5, 2, 1)  #To create a problem with five decision variables, two objectives, and one constraint
        self.types[0] = Integer(0, int(roof_top_area/PV_module)) #Decision space for A_solar
        self.types[1] = Integer(0, len(wind_component['Number'])-1) #Decision space for A_swept
        self.types[2] = Integer(0, len(CHP_component['Number'])-1) #Decision space for CHP capacity
        self.types[3] = Integer(0, len(boiler_component['Number'])-6) #Decision space for boiler capacity
        self.types[4] = Integer(0, len(battery_component['Number'])-1) #Decision space for battery capacity
        self.constraints[0] = ">=0" #Constraint to make sure heating demand can be satisfied
    def evaluate(self, solution):
        represent_day_max_results = self.represent_day_max()
        self.A_solar = solution.variables[0]*PV_module
        self.A_swept = solution.variables[1]
        self.CAP_CHP_elect = solution.variables[2]
        self.CAP_boiler = solution.variables[3]
        self.CAP_battery = solution.variables[4]
        self.A_swept= wind_component['Swept Area m^2'][self.A_swept]
        self.CAP_CHP_elect= CHP_component['CAP_CHP_elect_size'][self.CAP_CHP_elect]
        self.CAP_boiler= boiler_component['CAP_boiler (kW)'][self.CAP_boiler]
        self.CAP_battery= battery_component['CAP_battery (kWh)'][self.CAP_battery]
        operating_cost_initialized = self.operating_cost(self.A_solar,self.A_swept,self.CAP_CHP_elect,self.CAP_boiler,self.CAP_battery)
        solution.objectives[:] = [operating_cost_initialized[0],operating_cost_initialized[1]]
        solution.constraints[0] = self.CHP(self.CAP_CHP_elect,0)[6] + self.CAP_boiler - represent_day_max_results[1]
        if (self.CHP(self.CAP_CHP_elect,0)[6] + self.CAP_boiler - represent_day_max_results[1])<0:
            print('constraint error')
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
                if hour==0:
                    E_bat[hour]=0
                battery_results = self.battery(electricity_demand[hour],hour,E_bat[hour],A_solar,A_swept,CAP_battery)
                electricity_demand_new = battery_results[2]
                E_bat[hour+1]= battery_results[0]
                results = self.evaluate_operation(self.electricity_EF,electricity_demand_new,heating_demand[hour],A_solar,A_swept,CAP_CHP_elect,CAP_boiler,CAP_battery)
                sum_cost.append(results[1])
                sum_emissions.append(results[3])
            sum_cost_total.append(sum(sum_cost)*num_days_represent)
            sum_emissions_total.append(sum(sum_emissions)*num_days_represent)
        operating_cost = sum(sum_cost_total)
        operating_emissions = sum(sum_emissions_total)*lifespan_project
        capital_cost = self.solar_pv(A_solar,hour,0,1)[1] + self.wind_turbine(A_swept,hour,0,1)[1] + self.CHP(CAP_CHP_elect,1)[2] + self.NG_boiler(1,CAP_boiler)[1] + self.battery(electricity_demand[hour],hour,E_bat[hour],A_solar,A_swept,CAP_battery)[1]
        total_cost = capital_cost+operating_cost
        #print('sizing',_A_solar,_A_swept,_CAP_CHP_elect,_CAP_boiler,_CAP_battery)
        #print('cost', capital_cost,operating_cost,total_cost)
        return total_cost,operating_emissions, operating_cost,self.solar_pv(A_solar,hour,0,1)[1],self.wind_turbine(A_swept,hour,0,1)[1], self.CHP(CAP_CHP_elect,results[5])[2],self.NG_boiler(results[7],CAP_boiler)[1],self.battery(electricity_demand[hour],hour,E_bat[hour],A_solar,A_swept,CAP_battery)[1]
    def evaluate_operation(self,_electricity_EF,_electricity_demand,_heating_demand,_A_swept,_A_solar,_CAP_CHP_elec,_CAP_boiler,_CAP_battery):
        A_swept = _A_swept #Swept area of rotor m^2
        A_solar = _A_solar #Solar durface m^2 --> gives 160*A_solar W solar & needs= A_solar/0.7 m^2 rooftop
        CAP_CHP_elect = _CAP_CHP_elec #kW
        CAP_boiler = _CAP_boiler #kW
        CAP_battery = _CAP_battery #kW
        heating_demand= _heating_demand
        model = pyo.ConcreteModel()
        if _electricity_demand==0:
            Cost_minimzed = self.NG_boiler(heating_demand/self.NG_boiler(0,CAP_boiler)[4],CAP_boiler)[2] #$
            emission_objective = self.NG_boiler(heating_demand/self.NG_boiler(0,CAP_boiler)[4],CAP_boiler)[3] #kg CO2
            return 'Cost ($)', Cost_minimzed,'Emissions (kg CO2)', emission_objective ,'CHP',0,'Boiler', heating_demand/self.NG_boiler(0,CAP_boiler)[4],'Grid',0
        else:
            model.F_boiler = pyo.Var(bounds=(0,CAP_boiler/self.NG_boiler(0,CAP_boiler)[4])) #Decision space for natural gas fuel rate
            model.F_CHP = pyo.Var(bounds=(0,self.CHP(CAP_CHP_elect,0)[5])) #Decision space for CHP fuel rate
            model.E_grid = pyo.Var(bounds=(0,_electricity_demand+1)) #Decision space for grid consumption
            model.OBJ_cost = pyo.Objective(expr = self.CHP(CAP_CHP_elect,model.F_CHP)[3] + self.NG_boiler(model.F_boiler,CAP_boiler)[2] + model.E_grid*electricity_prices*UPV_elect) #$
            model.Constraint_elect = pyo.Constraint(expr = self.CHP(CAP_CHP_elect,model.F_CHP)[0] + model.E_grid - _electricity_demand>=0) # Electricity balance of demand and supply sides
            model.Constraint_heat = pyo.Constraint(expr = self.NG_boiler(model.F_boiler,CAP_boiler)[0] + self.CHP(CAP_CHP_elect,model.F_CHP)[1] -heating_demand>=0) # Heating balance of demand and supply sides
            #opt = SolverFactory('cplex')
            opt = SolverFactory('glpk')
            results =opt.solve(model,load_solutions=False)
            if len(results.solution)==0:
                print('ERROR')
                print('demands',_electricity_demand,_heating_demand)
                print('sizing',A_swept,A_solar,CAP_CHP_elect,CAP_boiler,CAP_battery)
                print(self.CHP(CAP_CHP_elect,0)[6] +self.NG_boiler(0,CAP_boiler)[4]*CAP_boiler)
                #print('results',results.objectives)
                #print('results',results)
                #print('here',self.CHP(self.CAP_CHP_elect,0)[6] ,self.NG_boiler(0,self.CAP_boiler)[4]*self.CAP_boiler)
                return 'Cost ($)', 10000,'Emissions (kg CO2)', 100000 ,'CHP', 0,'Boiler', 0,'Grid', 0
            else:
                model.solutions.load_from(results)
                Cost_minimzed = self.CHP(CAP_CHP_elect,model.F_CHP.value)[3] + self.NG_boiler(model.F_boiler.value,CAP_boiler)[2] + model.E_grid.value*electricity_prices
                emission_objective = self.CHP(CAP_CHP_elect,model.F_CHP.value)[4] + self.NG_boiler(model.F_boiler.value,CAP_boiler)[3] +model.E_grid.value*_electricity_EF #kg CO2
                return 'Cost ($)', Cost_minimzed,'Emissions (kg CO2)', emission_objective ,'CHP', model.F_CHP.value,'Boiler', model.F_boiler.value,'Grid', model.E_grid.value
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
    def solar_pv(self,A_surf_size,hour_of_day,electricity_demand_max,GT_max):
        ###Solar PV###
        IC_solar = solar_component['Investment cost ($/Wdc)'][0] #Solar PV capital investment cost is 1.75$/Wdc
        OM_solar = solar_component['Fixed solar PV O&M cost ($/kW-year)'][0] #fixed solar PV O&M cost 18$/kW-year
        PD_solar = solar_component['Power density of solar PV system W/m^2'][0] #Module power density of solar PV system W/m^2
        eff_module = solar_component['Module efficiency'][0] #Module efficiency
        eff_inverter = solar_component['Inverter efficiency'][0] #Inverter efficiency
        CAP_solar = PD_solar*A_surf_size/1000
        A_surf_max = electricity_demand_max/(GT_max*eff_module*eff_inverter/1000)
        salvage_solar = 1-(lifespan_solar-lifespan_project+lifespan_solar*int(lifespan_project/lifespan_solar))/lifespan_solar
        self.E_solar = A_surf_size*self.G_T[hour_of_day]*eff_module*eff_inverter/1000 #Solar generation from PV system (kWh) CHANGE G_T
        self.invest_solar  = (IC_solar*1000*salvage_solar+OM_solar*UPV_maintenance)*CAP_solar #CAP_solar in kW + investment cost of solar in $
        return self.E_solar, self.invest_solar,A_surf_max
    def wind_turbine(self,A_swept_size,hour_of_day,electricity_demand_max,V_max):
        ###Wind Turbine###
        index_wind = list(wind_component['Swept Area m^2']).index(A_swept_size)
        CAP_wind = wind_component['Rated Power kW'][index_wind]
        IC_wind = wind_component['Investment Cost'][index_wind] #Wind turbine capital cost in Utah 2018 1740$/kW
        rho = 1.2 #air density for wind turbines kg/m^3 CHANGE
        OM_wind = 44 #fixed wind turbines O&M cost 44$/kW-year
        C_p = 0.35 #Power coefficient default value of 0.35 in E+ CHANGE
        if self.V_wind[hour_of_day]<cut_in_wind_speed:
            self.V_wind[hour_of_day] = 0
        self.E_wind = 0.5*C_p*rho*A_swept_size*self.V_wind[hour_of_day]**3/1000 #Wind generation from wind Turbine (kW) CHANGE V_wind
        salvage_wind = 1-(lifespan_wind-lifespan_project+lifespan_wind*int(lifespan_project/lifespan_wind))/lifespan_wind
        self.invest_wind = (IC_wind + OM_wind*UPV_maintenance)*CAP_wind #CAP_wind in kW + investment cost of wind in $
        return self.E_wind, self.invest_wind
    def CHP(self, CAP_CHP_elect_size,F_CHP_size):
        ###CHP system###
        if CAP_CHP_elect_size==0:
            return 0,0,0,0,0,0,0
        else:
            index_CHP = list(CHP_component['CAP_CHP_elect_size']).index(CAP_CHP_elect_size)
            IC_CHP = CHP_component['IC_CHP'][index_CHP] #investment cost for CHP system $/kW
            CAP_CHP_Q= CHP_component['CAP_CHP_Q'][index_CHP] #Natural gas input mmBTU/hr, HHV
            eff_CHP_therm = CHP_component['eff_CHP_therm'][index_CHP] #Thermal efficiency of CHP system Q/F
            eff_CHP_elect = CHP_component['eff_CHP_elect'][index_CHP] #Electricity efficiency of CHP system P/F
            OM_CHP =CHP_component['OM_CHP'][index_CHP] #cent/kWh to $/kWh. now is $/kWh
            gamma_CHP =CHP_component['gamma_CHP'][index_CHP] #cent/kWh to $/kWh. now is $/kWh
            self.E_CHP = F_CHP_size*eff_CHP_elect/100 #Electricty generation of CHP system kWh
            self.Q_CHP = F_CHP_size*eff_CHP_therm/100 #Heat generation of CHP system kWh
            #salvage_CHP = (lifespan_chp-lifespan_project+lifespan_chp*int(lifespan_project/lifespan_chp))/lifespan_chp
            self.invest_CHP = IC_CHP*CAP_CHP_elect_size #Investment cost of the CHP system $
            self.OPC_CHP = (NG_prices)*F_CHP_size*UPV_NG +OM_CHP*UPV_maintenance*self.E_CHP #O&M cost of CHP system $
            self.OPE_CHP = gamma_CHP*self.E_CHP # O&M emission of CHP system kg CO2
            return self.E_CHP,self.Q_CHP,self.invest_CHP,self.OPC_CHP,self.OPE_CHP, CAP_CHP_elect_size/eff_CHP_elect*100,CAP_CHP_elect_size*eff_CHP_therm/eff_CHP_elect
    def NG_boiler(self,F_boilers,_CAP_boiler):
        CAP_boiler = _CAP_boiler
        index_boiler = list(boiler_component['CAP_boiler (kW)']).index(CAP_boiler)
        CAP_boiler = boiler_component['CAP_boiler (kW)'][index_boiler]
        ###Natural gas boiler###
        IC_boiler = float(boiler_component['Investment cost $/MBtu/hour'][index_boiler])/1000/BTUtokWh_convert #Natural gas boiler estimated capital cost of $35/MBtu/hour. now unit is $/kW
        landa_boiler = float(boiler_component['Variabel O&M cost ($/mmBTU)'][index_boiler])/mmBTutoBTU_convert/BTUtokWh_convert #O&M cost of input 0.95 $/mmBTU. now unit is 119 $/kWh
        gamma_boiler = float(boiler_component['Natural gas emission factor (kg-CO2/mmBTU)'][index_boiler])/mmBTutoBTU_convert/BTUtokWh_convert #kg-CO2/kWh is emission factor of natural gas
        self.eff_boiler = float(boiler_component['Boiler Efficiency'][index_boiler]) #efficiency of natural gas boiler
        self.Q_boiler = self.eff_boiler*F_boilers #Net heat generation of NG boilers kWh
        #salvage_boiler = 1-(lifespan_boiler-lifespan_project+lifespan_boiler*int(lifespan_project/lifespan_boiler))/lifespan_boiler
        self.invest_boiler = IC_boiler*CAP_boiler  #Investment cost of boiler in $
        self.OPC_boiler = (landa_boiler*UPV_maintenance+NG_prices*UPV_NG)*F_boilers #O&M cost of boiler $
        self.OPE_boiler = gamma_boiler*F_boilers #O&M emissions of boiler in kg CO2
        return self.Q_boiler,self.invest_boiler,self.OPC_boiler,self.OPE_boiler, self.eff_boiler
    def battery(self,electricity_demand_bat,hour,E_bat_,_A_solar,_A_swept,_CAP_battery):
        A_swept = _A_swept #Swept area of rotor m^2
        A_solar = _A_solar #Solar durface m^2 --> gives 160*A_solar W solar & needs= A_solar/0.7 m^2 rooftop
        CAP_battery= _CAP_battery
        index_battery =  list(battery_component['CAP_battery (kWh)']).index(CAP_battery)
        CAP_battery = battery_component['CAP_battery (kWh)'][index_battery]
        eff_bat_ch = battery_component['Battery efficiency charge'][index_battery]
        eff_bat_disch = battery_component['Battery efficiency discharge'][index_battery]
        bat_dod = battery_component['battery depth of discharge'][index_battery] #battery depth of discharge
        lifespan_battery = battery_component['Lifespan (year)'][index_battery]
        self.E_bat = E_bat_
        renewables_elect =  self.solar_pv(A_solar, hour,0,1)[0] + self.wind_turbine(A_swept, hour,0,1)[0]
        electricity_demand = electricity_demand_bat
        if renewables_elect>=electricity_demand:
            P_ch_dis_old = renewables_elect - electricity_demand
            electricity_demand = 0
            if P_ch_dis_old>CAP_battery/eff_bat_ch -  self.E_bat: #Charging the battery
                P_ch_dis_old = CAP_battery/eff_bat_ch -  self.E_bat
            self.E_bat_new= self.E_bat + eff_bat_ch*P_ch_dis_old*deltat
        elif renewables_elect<electricity_demand: #Diccharging the battery
            P_ch_dis_old = electricity_demand - renewables_elect
            if self.E_bat- (1-bat_dod)*CAP_battery<0:
                self.E_bat_new= self.E_bat
                P_ch_dis_old = 0
            elif self.E_bat- (1-bat_dod)*CAP_battery < 1/eff_bat_disch*P_ch_dis_old*deltat:
                P_ch_dis_old = eff_bat_disch*self.E_bat - (1-bat_dod)*CAP_battery
            electricity_demand = electricity_demand - P_ch_dis_old - renewables_elect
            self.E_bat_new = self.E_bat - 1/eff_bat_disch*P_ch_dis_old*deltat
        IC_battery =  battery_component['Investment cost ($/kW)'][index_battery] #Battery capital investment cost is 2338 $/kW
        OM_battery = battery_component['Fixed O&M cost  $/kW-year'][index_battery]#fixed battery O&M cost 6$/kW-year
        self.invest_battery = (IC_battery*lifespan_project/lifespan_battery +OM_battery*UPV_maintenance)*CAP_battery
        return self.E_bat_new,self.invest_battery,electricity_demand
def results_extraction(problem, algorithm):
    file_name = '/Discrete_EF_'+str(float(editable_data['renewable percentage']) )+'_design_'+str(editable_data['num_iterations'])+'_'+str(editable_data['population_size'])+'_'+str(editable_data['num_processors'])+'_processors'
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
    for s in algorithm.result:
        solar_results[s] = s.variables[0]
        wind_results[s] = s.variables[1]
        CHP_results[s] = s.variables[2]
        boiler_results[s] = s.variables[3]
        battery_results[s] = s.variables[4]
        if isinstance(solar_results[s], list):
            solar_results[s] = float(problem.types[0].decode(solar_results[s]))
            solar_results[s]= solar_results[s]*PV_module
        if isinstance(wind_results[s], list):
            wind_results[s] = float(problem.types[1].decode(wind_results[s]))
            wind_results[s]=wind_component['Swept Area m^2'][wind_results[s]]
        if isinstance(CHP_results[s], list):
            CHP_results[s] = float(problem.types[2].decode(CHP_results[s]))
            CHP_results[s]=CHP_component['CAP_CHP_elect_size'][CHP_results[s]]
        if isinstance(boiler_results[s], list):
            boiler_results[s] = float(problem.types[3].decode(boiler_results[s]))
            boiler_results[s]=boiler_component['CAP_boiler (kW)'][boiler_results[s]]
        if isinstance(battery_results[s], list):
            battery_results[s] = float(problem.types[4].decode(battery_results[s]))
            battery_results[s]=battery_component['CAP_battery (kWh)'][battery_results[s]]
        print(i,'variable',solar_results[s],wind_results[s],CHP_results[s],boiler_results[s],battery_results[s])
        print(i,'objective',s.objectives[0],s.objectives[1])
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
    df_cost_all.to_csv(results_path + '/cost_all.csv')

    plt.scatter([s.objectives[0] for s in algorithm.result],
                [s.objectives[1] for s in algorithm.result])
    plt.xlabel("Cost ($)")
    plt.ylabel("Emissions kg ($CO_2$)")
    plt.savefig(results_path + '/operation.png',dpi=300)
    plt.close()
