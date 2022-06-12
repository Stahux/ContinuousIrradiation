#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###########################################
#code used to simulate differential quantum yields
#for OCP with single photon and two photon models (simpified toy models)
###########################################


import sys
sys.path.append("../../application/")
from model import Model
from modfit import MParameters, MParameter, ModFit
from kineticdata import LightEvent, KineticData, Experiment
import numpy as np

#to set serif font
from matplotlib import rcParams
rcParams['font.family'] = 'serif'

import matplotlib.pyplot as plt
import lmfit
from lmfit import minimize
import copy
import random
import pickle

#mult to convert mol L-1 s-1 to mol cm-2 s-1
#assumes 1 x 0.2 cm cuvette, 1.5cm solution height in cuvette, and
#irradiation field 1 x 1.5 cm
mol_L_to_mol_m2 = 10000 * 1.5*0.2*1/(1.5*1*1000)

#model names used for simulation
modelname = 'OCP_AB.model'
model2name = 'OCP_2photon.model'

def buildModel(modelname):
#function to rebuild model, note that it will overwrite
#previous version of the model so backup it first
    try:
        test1 = Model.load(modelname)
    except:
        test1 = Model()
        
    test1.manualModelBuild()
    test1.genParameters().pretty_print() #params print
    test1.save(modelname) #save new model

#load models from files
test0 = Model.load(modelname) #simple A->B model
test1 = Model.load(model2name) # A->B->C->A 2 photon model

#assume some epsilons for intermediate product (set red color for intermediate)
test1params = test1.genParameters()
for wavelength in [452,550,580]:
    test1params["OCPI__"+str(wavelength)+"_0"].value = test1params["OCPR__"+str(wavelength)+"_0"].value
test1params["OCPI_OCPR__fi"].value = 0.7
test1params["OCPO_OCPI__fi"].value = 0.005
test1params["OCPI_OCPO__k"].value = 1/0.18
test1.updateParameters(test1params)    

#construct empty datasets to fill with kinetics
data0 = Experiment()
for x in np.logspace(-5.5,-3, 20): #iterate over light intensities
    data0.loadKineticData(None, src="load", probe = 550, t_on = 30, t_off = 1200, 
                          irradiation = 452, intensity = x, absorbance = 0.18506, 
                          probe_length = 1, irradiation_length = 0.2)

#data0.all_data[0].selectTimes(-10, 120)

#SIMULATE DATA WITH SIMPLE A->B MODEL
test0.setInitial("OCPO",1)
test0.setInitial("OCPR",0)

#generate params from model (mechanism) and from empty datasets (experimental conditions)
params0 = data0.genParameters() + test0.genParameters()
params0['OCPR_OCPO__k'].set(value=1/90)

#fit with everything fixed, so it just recreates data for these fixed params
modfit1 = ModFit(test0, data0, params0)
out1 = modfit1.fit()
params0 = out1.params

data0.updateParameters(params0)
test0.updateParameters(params0)

test0.plotYourself(data0,dpi=100,x_min=0, x_max=2000, title="")
test0.plotConcentrations(data0.all_data[-1],[0,1],dpi=100,x_min=0, x_max=2000, title="")
lmfit.report_fit(params0)

#recreate data
simulation_single_photon = test0.recreateExperimentData(data0)

#calculate the differential quantum yelds from the simulated datasets 
slopes_1pmodel = []
effs_1pmodel = []
intensities_1pmodel = []
no = 5
for x in simulation_single_photon:
    tmpkin = x
    slopes_1pmodel.append(tmpkin.getInitialSlope(time_window = 10)[0])
    intensities_1pmodel.append(tmpkin.intensity)
    Ainit = tmpkin.data_a[np.argmin(np.abs(tmpkin.data_t-tmpkin.t_on))]
    Aexc = tmpkin.absorbance
    F = (1 - np.exp(-2.30259 * Aexc)) / Aexc #at2mm, 452nm
    effs_1pmodel.append(slopes_1pmodel[-1]/(tmpkin.intensity*F*21473*1*Aexc)) #1cm, delta eps at 550nm
    no -= 1

#....repeat the same procedure for 2 photon model with two effieicncies
#SIMULATE DATA WITH DOUBLE PHOTON MODEL (1)
data1 = copy.deepcopy(data0)
test1p = copy.deepcopy(test1)
test1p.setInitial("OCPO",1)
test1p.setInitial("OCPI",0)
test1p.setInitial("OCPR",0)

params1 = data1.genParameters() + test1p.genParameters()

params1['OCPI_OCPR__fi'].set(value=1)
params1['OCPR_OCPO__k'].set(value=1/90)

modfit1 = ModFit(test1p, data1, params1)
out1 = modfit1.fit()
params1 = out1.params

data1.updateParameters(params1)
test1p.updateParameters(params1)

test1p.plotYourself(data1,dpi=100,x_min=0, x_max=2000, title="")
test1p.plotConcentrations(data1.all_data[-1],[0,1,2],dpi=100,x_min=0, x_max=2000, title="")
lmfit.report_fit(params1)

simulation_double_photon_1 = test1p.recreateExperimentData(data1)

 
slopes_2pmodel = []
effs_2pmodel = []
intensities_2pmodel = []
no = 5
for x in simulation_double_photon_1:
    tmpkin = x
    slopes_2pmodel.append(tmpkin.getInitialSlope(time_window = 10)[0])
    intensities_2pmodel.append(tmpkin.intensity)
    Ainit = tmpkin.data_a[np.argmin(np.abs(tmpkin.data_t-tmpkin.t_on))]
    Aexc = tmpkin.absorbance
    F = (1 - np.exp(-2.30259 * Aexc)) / Aexc #at2mm, 452nm
    effs_2pmodel.append(slopes_2pmodel[-1]/(tmpkin.intensity*F*21473*1*Aexc)) #1cm, delta eps at 550nm
    no -= 1
    
#SIMULATE DATA WITH DOUBLE PHOTON MODEL (2)
data1 = copy.deepcopy(data0)
test1p = copy.deepcopy(test1)
test1p.setInitial("OCPO",1)
test1p.setInitial("OCPI",0)
test1p.setInitial("OCPR",0)

params1 = data1.genParameters() + test1p.genParameters()
params1['OCPI_OCPR__fi'].set(value=0.1)
params1['OCPR_OCPO__k'].set(value=1/90)

modfit1 = ModFit(test1p, data1, params1)
out1 = modfit1.fit()
params1 = out1.params

data1.updateParameters(params1)
test1p.updateParameters(params1)

test1p.plotYourself(data1,dpi=100,x_min=0, x_max=2000, title="")
test1p.plotConcentrations(data1.all_data[-1],[0,1,2],dpi=100,x_min=0, x_max=2000, title="")
lmfit.report_fit(params1)

simulation_double_photon_1 = test1p.recreateExperimentData(data1)

 
slopes_2pmodel = []
effs_2pmodel_2 = []
intensities_2pmodel = []
no = 5
for x in simulation_double_photon_1:
    tmpkin = x
    slopes_2pmodel.append(tmpkin.getInitialSlope(time_window = 10)[0])
    intensities_2pmodel.append(tmpkin.intensity)
    Ainit = tmpkin.data_a[np.argmin(np.abs(tmpkin.data_t-tmpkin.t_on))]
    Aexc = tmpkin.absorbance
    F = (1 - np.exp(-2.30259 * Aexc)) / Aexc #at2mm, 452nm
    effs_2pmodel_2.append(slopes_2pmodel[-1]/(tmpkin.intensity*F*21473*1*Aexc)) #1cm, delta eps at 550nm
    no -= 1
    
 

#change mol per L intensity to mol per cm2 intensity (commonly used)  
intensities_1pmodel = np.array(intensities_1pmodel)*mol_L_to_mol_m2
intensities_2pmodel = np.array(intensities_2pmodel)*mol_L_to_mol_m2  


#efficiency vs intensity plot
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.plot(intensities_1pmodel, effs_1pmodel, "co-", label = "single h\u03BD model")
ax.plot(intensities_2pmodel, effs_2pmodel, "mo-", label = "two h\u03BD model, \u03c6\u2082=1.0")
ax.plot(intensities_2pmodel, effs_2pmodel_2, "ro-", label = "two h\u03BD model, \u03c6\u2082=0.1")
ax.legend(shadow=False, frameon=True, prop={'size': 14}, labelspacing=0.1, loc="upper left")
ax.set_xlabel("Light intensity I\u2080 (mol m\u207B\u00B2 s\u207B\u00B9)", fontdict={'size': 16})
ax.set_ylabel("Quantum yield $\mathregular{\u03a6_d}$", fontdict={'size': 16})
ax.set_xscale("log")
#ax.set_ylim(0, 0.004)
ax.set_xlim(9E-6, 1E-3)
ax.grid(True, which="major", linestyle='--')
ax.tick_params(which='both', direction="in", bottom=True, top=True, left=True, right=True, labelsize=14)
plt.show()

