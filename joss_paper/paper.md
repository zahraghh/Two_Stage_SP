---
title: 'Two-stage Stochastic Programming of the Sizing of District Energy Systems'
tags:
  - Python
  - district energy system
  - stochastic programming
  - multi-objective optimization
  - uncertainty analysis
authors:
  - name: Zahra Ghaemi^[Corresponding author]
    orcid: 0000-0003-4072-0881
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Amanda D. Smith
    affiliation: "1, 2"

affiliations:
 - name: Department of Mechanical Engineering, University of Utah, Salt Lake City, UT 84112
   index: 1
 - name: Pacific Northwest National Laboratory, Richland, WA, 99352
   index: 2

date: 18 April 2021
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.31224/osf.io/hcnd6 <- update this with the DOI from AAS once you know it.
aas-journal: ASME International Mechanical Engineering Congress and Exposition
---

# Summary
Two_stage_SP package optimizes the sizing of energy components for a district energy system to minimize the total cost and operating  CO<sub>2</sub>  emissions. This package considers five selected energy components: natural gas boilers, combined heating and power, solar photovoltaics, wind turbines, and batteries. These five energy components and the grid can provide the heating, cooling, and electricity needs of a group of buildings. Uncertainties in energy demand, solar irradiance, wind speed, and electricity emissions are considered in this package.

This framework has four main sections. In the first section, actual weather data from National Solar Radiation Database (NSRDB) is downloaded. Global tilted irradiance is quantified per unit of a tilted flat plate, representing solar photovoltaic arrays. In the second section, uncertainties in energy demands, solar irradiance, wind speed, and electricity emissions are considered by generating new scenarios with different values of energy demands, solar irradiance, wind speed, and electricity emissions. In the third section, the k-medoid clustering algorithm is used to reduce the number of scenarios to a selected number defined by the user by removing similar scenarios. In the fourth section, the two-stage stochastic programming is performed by utilizing the nondominated sorting genetic algorithm II (NSGA-II) algorithm in the first stage and the public GNU Linear Programming Kit (GLPK) solver in the second stage. The objectives are to minimize the total cost (investment, operation & maintenance, and operating cost) and operating CO<sub>2</sub> emissions.

A user can change the case study's location, edit energy demands of buildings, modify the energy components' characteristics, change the energy configuration, and/or change the district energy system's charectrsitics. 

# Statement of need
Optimizating sizing of district energy systems is one the present challenges in developing district energy systems [@mahmoud2020recent], especially with the lack of open-source frameworks in the energy research field [@klemm2021modeling]. Therefore, a framwork that can help with different stages of optimizing a district energy system is valuable. 
research problem: 
in the vast  and [@pfenninger2017importance]

real world problem: helping facility managers 



# Acknowledgements


# Reference
