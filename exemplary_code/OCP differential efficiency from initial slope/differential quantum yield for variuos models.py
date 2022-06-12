#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###########################################
#this code is loads various sample models of OCP
#simulates photoconversion for the grid of photon flux densities
#then for each experiment (some model with some photon flux density)
#the script fits first 10s of growing kinetic with linear function
#and calculates differential quantum yield
#this procedure is equivalent of experimental one used to determine QY
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

#models used for simulations
modelname = 'OCP_AB.model'
model2name = 'OCP_2photon.model'
model8name = "OCP_2nd_photon_speedup.model"
model4name = "OCP_2photon_dimeric.model"
model7name = "OCP_relaxation_to_prone.model"
model10name = "OCP_maksimovs_model.model"

def buildModel(modelname):
#function that can be called to rebuild model if needed
    
    try:
        test1 = Model.load(modelname)
    except:
        test1 = Model()
        
    test1.manualModelBuild()
    test1.genParameters().pretty_print() #params print
    test1.save(modelname) #save new model


test0 = Model.load(modelname) #simple A->B model
test1 = Model.load(model2name) # A->B->C->A 2 photon model

#assume some epsilons for intermediate product
test1params = test1.genParameters()
for wavelength in [452,550,580]:
    test1params["OCPI__"+str(wavelength)+"_0"].value = test1params["OCPR__"+str(wavelength)+"_0"].value
test1params["OCPI_OCPR__fi"].value = 0.7
test1params["OCPO_OCPI__fi"].value = 0.005
test1params["OCPI_OCPO__k"].value = 1/0.18
test1.updateParameters(test1params)    
    
#create empty datasets for grid of irradiation photon flux densities
data0 = Experiment()
for x in np.logspace(-5.5,-3, 20):
    data0.loadKineticData(None, src="load", probe = 550, t_on = 30, t_off = 1200, 
                          irradiation = 452, intensity = x, absorbance = 0.18506, 
                          probe_length = 1, irradiation_length = 0.2)

#data0.all_data[0].selectTimes(-10, 120)

#SIMULATE DATA WITH SIMPLE A->B MODEL
test0.setInitial("OCPO",1)
test0.setInitial("OCPR",0)

#generates parameters from model and from the "data" (data is now empty grid of NaNs
params0 = data0.genParameters() + test0.genParameters()

#set some params, but here they are already in the model and fixed by default
#params0['_0_absorbance'].set(vary=False)
#params0['_0_intensity'].set(vary=False)
#params0['OCPO_OCPR__fi'].set(value=0.2,vary=True)
params0['OCPR_OCPO__k'].set(value=1/90,vary=False)

#everything is fixed, so it just fills the data with numbers
modfit1 = ModFit(test0, data0, params0)
out1 = modfit1.fit()
params0 = out1.params

data0.updateParameters(params0)
test0.updateParameters(params0)

#test0.plotYourself(data0,dpi=100,x_min=0, x_max=2000, title="")
#test0.plotConcentrations(data0.all_data[0],[0,1],dpi=100,x_min=0, x_max=2000, title="")
#lmfit.report_fit(params0)

simulation_single_photon = test0.recreateExperimentData(data0)

#calc. differential QY from the simulated data (slope fitting)
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


#SIMULATE DATA WITH DOUBLE PHOTON MODEL (1)
data1 = copy.deepcopy(data0)
test1p = copy.deepcopy(test1)
test1p.setInitial("OCPO",1)
test1p.setInitial("OCPI",0)
test1p.setInitial("OCPR",0)

params1 = data1.genParameters() + test1p.genParameters()

#set some params, but here they are already in the model and fixed by default
params1['_0_absorbance'].set(vary=False)
params1['_0_intensity'].set(vary=False)
params1['OCPI_OCPR__fi'].set(value=0.2,vary=False)
params1['OCPR_OCPO__k'].set(value=1/90,vary=False)
params1['OCPI_OCPO__k'].set(value=1/0.18,vary=False)
params1['OCPO_OCPI__fi'].set(value=0.01,vary=False)

#everything is fixed, so it just fills the data with numbers
modfit1 = ModFit(test1p, data1, params1)
out1 = modfit1.fit()
params1 = out1.params

data1.updateParameters(params1)
test1p.updateParameters(params1)

#test1p.plotYourself(data1,dpi=100,x_min=0, x_max=2000, title="")
#test1p.plotConcentrations(data1.all_data[-1],[0,1,2],dpi=100,x_min=0, x_max=2000, title="")
#lmfit.report_fit(params1)

simulation_double_photon_1 = test1p.recreateExperimentData(data1)

#calc. differential QY from the simulated data (slope fitting)
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
    
#another models to demonstrate how they look like for reviewer
    
#now i can load epsilons like that! so cool!
from spectrumdata import SpectrumData
OCPO_Ctag_spectrum = SpectrumData("OCPO_Ctag_spectrum.txt")
OCPR_Ctag_spectrum = SpectrumData("OCPR_Ctag_spectrum.txt")    
    
#SIMULATE DATA WITH 2nd photon speedup model
test8 = Model.load(model8name) # 2nd photon model whith speedup    
data8 = copy.deepcopy(data0)
test8.setInitial("OCPO",1)
test8.setInitial("OCPI",0)
test8.setInitial("OCPR",0)

#load all epsilons from spectrum file
test8["OCPO"].loadEps(OCPO_Ctag_spectrum)
test8["OCPI"].loadEps(OCPR_Ctag_spectrum)
test8["OCPR"].loadEps(OCPR_Ctag_spectrum)

params8 = data8.genParameters() + test8.genParameters()

#set some params, as they are not in model. everything is fixed by default
params8['OCPO_OCPI__fi'].set(value=0.01)
params8['OCPI_OCPR__fi'].set(value=0.2)
params8['OCPI_OCPO__k'].set(value=1/0.18)
params8['OCPR_OCPO__k'].set(value=1/90)
params8['OCPI_OCPR_2__k'].set(value=1)

#everything is fixed, so it just fills the data with numbers
modfit = ModFit(test8, data8, params8)
out = modfit.fit()
params = out.params

data8.updateParameters(params)
test8.updateParameters(params)

simulation = test8.recreateExperimentData(data8) 

#calc. differential QY from the simulated data (slope fitting)
effs_8 = []
intensities_8 = []
no = 5
for x in simulation:
    tmpkin = x
    slope = tmpkin.getInitialSlope(time_window = 10)[0]
    intensities_8.append(tmpkin.intensity)
    Ainit = tmpkin.data_a[np.argmin(np.abs(tmpkin.data_t-tmpkin.t_on))]
    Aexc = tmpkin.absorbance
    F = (1 - np.exp(-2.30259 * Aexc)) / Aexc #at2mm, 452nm
    effs_8.append(slope/(tmpkin.intensity*F*21473*1*Aexc)) #1cm, delta eps at 550nm
    no -= 1
    
    
    
#SIMULATE DATA WITH two photon with orange interm.
test4 = Model.load(model2name) 
data4 = copy.deepcopy(data0)
test4.setInitial("OCPO",1)
test4.setInitial("OCPI",0)
test4.setInitial("OCPR",0)

#load all epsilons from spectrum file
test4["OCPO"].loadEps(OCPO_Ctag_spectrum)
test4["OCPI"].loadEps(OCPO_Ctag_spectrum)
test4["OCPR"].loadEps(OCPR_Ctag_spectrum)

params4 = data4.genParameters() + test4.genParameters()

#set some params, as they are not in model. everything is fixed by default
params4['OCPI_OCPO__k'].set(value=1/0.18)
params4['OCPI_OCPR__fi'].set(value=0.01)
params4['OCPO_OCPI__fi'].set(value=0.2)
params4['OCPR_OCPO__k'].set(value=1/90)

#everything is fixed, so it just fills the data with numbers
modfit = ModFit(test4, data4, params4)
out = modfit.fit()
params = out.params

data4.updateParameters(params)
test4.updateParameters(params)

simulation = test4.recreateExperimentData(data4)
#lmfit.report_fit(params)

#calc. differential QY from the simulated data (slope fitting)
effs_4 = []
intensities_4 = []
no = 5
for x in simulation:
    tmpkin = x
    slope = tmpkin.getInitialSlope(time_window = 10)[0]
    intensities_4.append(tmpkin.intensity)
    Ainit = tmpkin.data_a[np.argmin(np.abs(tmpkin.data_t-tmpkin.t_on))]
    Aexc = tmpkin.absorbance
    F = (1 - np.exp(-2.30259 * Aexc)) / Aexc #at2mm, 452nm
    effs_4.append(slope/(tmpkin.intensity*F*21473*1*Aexc)) #1cm, delta eps at 550nm
    no -= 1
        
    
#SIMULATE DATA WITH relaxation to prone scheme
test7 = Model.load(model7name) # 2nd photon model whith speedup    
data7 = copy.deepcopy(data0)
test7.setInitial("OCPO",1)
test7.setInitial("OCPR",0)
test7.setInitial("OCPOB",0)

#load all epsilons from spectrum file
test7["OCPO"].loadEps(OCPO_Ctag_spectrum)
test7["OCPR"].loadEps(OCPR_Ctag_spectrum)
test7["OCPOB"].loadEps(OCPO_Ctag_spectrum)

params7 = data7.genParameters() + test7.genParameters()

#set some params, as they are not in model. everything is fixed by default
params7['OCPO_OCPR__fi'].set(value=0.0025)
params7['OCPOB_OCPR__fi'].set(value=0.1)
params7['OCPR_OCPOB__k'].set(value=1/90)
params7['OCPOB_OCPO__k'].set(value=1/300)
#everything is fixed, so it just fills the data with numbers

modfit = ModFit(test7, data7, params7)
out = modfit.fit()
params = out.params

data7.updateParameters(params)
test7.updateParameters(params)

simulation = test7.recreateExperimentData(data7)
#lmfit.report_fit(params)

#calc. differential QY from the simulated data (slope fitting)
effs_7 = []
intensities_7 = []
no = 5
for x in simulation:
    tmpkin = x
    slope = tmpkin.getInitialSlope(time_window = 10)[0]
    intensities_7.append(tmpkin.intensity)
    Ainit = tmpkin.data_a[np.argmin(np.abs(tmpkin.data_t-tmpkin.t_on))]
    Aexc = tmpkin.absorbance
    F = (1 - np.exp(-2.30259 * Aexc)) / Aexc #at2mm, 452nm
    effs_7.append(slope/(tmpkin.intensity*F*21473*1*Aexc)) #1cm, delta eps at 550nm
    no -= 1
        
        
    
    
#SIMULATE DATA WITH Maksimovs model
test10 = Model.load(model10name) # 2nd photon model whith speedup    
data10 = copy.deepcopy(data0)
test10.setInitial("OCPO",1)
test10.setInitial("OCPR",0)
test10.setInitial("OCPOI",0)
test10.setInitial("OCPRI",0)

#load all epsilons from spectrum file
test10["OCPO"].loadEps(OCPO_Ctag_spectrum)
test10["OCPR"].loadEps(OCPR_Ctag_spectrum)
test10["OCPOI"].loadEps(OCPO_Ctag_spectrum)
test10["OCPRI"].loadEps(OCPR_Ctag_spectrum)

params10 = data10.genParameters() + test10.genParameters()

#set some params, as they are not in model. everything is fixed by default
params10['OCPO_OCPRI__fi'].set(value=0.005)
params10['OCPRI_OCPO__k'].set(value=1/0.0003)
params10['OCPRI_OCPR__k'].set(value=1/0.0003)
params10['OCPR_OCPOI__k'].set(value=1/90)
params10['OCPOI_OCPO__k'].set(value=1/300)

#everything is fixed, so it just fills the data with numbers
modfit = ModFit(test10, data10, params10)
out = modfit.fit()
params = out.params

data10.updateParameters(params)
test10.updateParameters(params)

simulation = test10.recreateExperimentData(data10)
#lmfit.report_fit(params)

#calc. differential QY from the simulated data (slope fitting)
effs_10 = []
intensities_10 = []
no = 5
for x in simulation:
    tmpkin = x
    slope = tmpkin.getInitialSlope(time_window = 10)[0]
    intensities_10.append(tmpkin.intensity)
    Ainit = tmpkin.data_a[np.argmin(np.abs(tmpkin.data_t-tmpkin.t_on))]
    Aexc = tmpkin.absorbance
    F = (1 - np.exp(-2.30259 * Aexc)) / Aexc #at2mm, 452nm
    effs_10.append(slope/(tmpkin.intensity*F*21473*1*Aexc)) #1cm, delta eps at 550nm
    no -= 1
        
    

#change mol per L intensity to mol per cm2 intensity (commonly used)  
intensities_1pmodel = np.array(intensities_1pmodel)*mol_L_to_mol_m2
intensities_2pmodel = np.array(intensities_2pmodel)*mol_L_to_mol_m2 
intensities_8 = np.array(intensities_8)*mol_L_to_mol_m2  
intensities_4 = np.array(intensities_4)*mol_L_to_mol_m2 
intensities_7 = np.array(intensities_7)*mol_L_to_mol_m2  
intensities_10 = np.array(intensities_10)*mol_L_to_mol_m2  



#efficiency vs intensity plotting
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.plot(intensities_1pmodel, effs_1pmodel, "co-", label = "1\u00B0 model")
ax.plot(intensities_2pmodel, effs_2pmodel, "mo-", label = "2\u00B0 model")
ax.plot(intensities_8, effs_8, "ro-", label = "3\u00B0 model")
ax.plot(intensities_4, effs_4, "go-", label = "4\u00B0 model")
ax.plot(intensities_7, effs_7, "bo-", label = "5\u00B0 model")
ax.plot(intensities_10, effs_10, "yo-", label = "6\u00B0 model")
ax.legend(shadow=False, frameon=True, prop={'size': 15}, labelspacing=0.2, loc=(0.05, 0.15))
ax.set_xlabel("Light intensity I\u2080 (mol m\u207B\u00B2 s\u207B\u00B9)", fontdict={'size': 16})
ax.set_ylabel("Quantum yield $\mathregular{\u03a6_d}$", fontdict={'size': 16})
ax.set_xscale("log")
ax.set_ylim(0, 0.0025)
ax.set_xlim(9E-6, 1E-3)
ax.grid(True, which="major", linestyle='--')
ax.tick_params(which='both', direction="in", bottom=True, top=True, left=True, right=True, labelsize=12)
plt.show()
