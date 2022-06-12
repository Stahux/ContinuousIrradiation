#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###########################################
#this code is used to simulate "action spectrum" for two photon and single photon OCP model
#"action spectrum" refers to the paper of Rakhimberdieva et al. FEBS Letters 574, 2004
#where authors irradiated cyanobacteria solution with variable light and subsequently
#measured fluorescence of the phocobilisomes, induced by another light source to excite PBS
#so the spectrum is constructed as a strength of PBS fluorescence quenching vs wavelength of 
#light used to irradiate the sample (not PBS)
#the constructed spectrum originally resembled OCP stationary absorption spectrum
#simulation is done to investigate effect of used model on the shape of action spectrum
#
#in this simulation only OCP is present, and we used approximation that quenching is
#proportional to amount of OCP accumulated at 10s
#script may throw warnings, because at some wavelengths OCPO absorption is zero
#and signals are lower than solver accuracy
###########################################


import sys
sys.path.append("../../application/")

from model import Model
from modfit import MParameters, MParameter, ModFit
from kineticdata import LightEvent, KineticData, Experiment
from spectrumdata import SpectrumData
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

#model names used for simulations
modelname = 'OCP_AB.model'
model2name = 'OCP_2photon.model'

def buildModel(modelname):
#function that can be called to rebuild model if needed    
    try:
        test1 = Model.load(modelname)
    except:
        test1 = Model()
        
    test1.manualModelBuild()
    test1.genParameters().pretty_print() #params print
    test1.save(modelname) #save new model


#load epsilon spectra of OCPO and OCPR
OCPO_Ctag_epsilons = SpectrumData("OCPO_Ctag_epsilons.txt")
OCPR_Ctag_epsilons = SpectrumData("OCPR_Ctag_epsilons.txt")

#load models
test0 = Model.load(modelname) #simple A->B model
test1 = Model.load(model2name) # A->B->C->A 2 photon model

test0["OCPR"].loadEps(OCPR_Ctag_epsilons)
test0["OCPO"].loadEps(OCPO_Ctag_epsilons)

test1["OCPR"].loadEps(OCPR_Ctag_epsilons)
test1["OCPI"].loadEps(OCPR_Ctag_epsilons)
test1["OCPO"].loadEps(OCPO_Ctag_epsilons)

#construct empty datasets to fill with simulations
#low concentration is assumed
wavelength_grid = np.linspace(350,750,int((750-350)/5+1))
epsilon_max = test0["OCPO"][497]
data0 = Experiment()
for w in wavelength_grid:
    data0.loadKineticData(None, src="load", probe = 580, t_on = 30, t_off = 1200, 
                          irradiation = w, intensity = 100E-06, 
                          concentration = 0.01/(epsilon_max*0.3), 
                          probe_length = 1, irradiation_length = 0.3)

#data0.all_data[0].selectTimes(-10, 120)

#SIMULATE DATA WITH SIMPLE A->B MODEL
test0.setInitial("OCPO",1)
test0.setInitial("OCPR",0)

#load parameters from the model
params0 = data0.genParameters() + test0.genParameters()
params0['OCPR_OCPO__k'].set(value=1/90)

modfit1 = ModFit(test0, data0, params0)
out1 = modfit1.fit()
params0 = out1.params

data0.updateParameters(params0)
test0.updateParameters(params0)

#test0.plotYourself(data0,dpi=100,x_min=0, x_max=2000, title="")
#test0.plotConcentrations(data0.all_data[0],[0,1],dpi=100,x_min=0, x_max=2000, title="")
#lmfit.report_fit(params0)

#generate datasets from simulation
simulation_single_photon = test0.recreateExperimentData(data0)

#build arrays to plot
signals_1pmodel = []
wavelengths_1pmodel = []
no = 5
for x in simulation_single_photon:
    tmpkin = x
    signals_1pmodel.append(tmpkin[10+30])
    wavelengths_1pmodel.append(tmpkin.irradiation)
    no -= 1
signals_1pmodel = np.array(signals_1pmodel)
wavelengths_1pmodel = np.array(wavelengths_1pmodel)


#SIMULATE DATA WITH DOUBLE PHOTON MODEL (1)

#assume some epsilons for intermediate product
test1params = test1.genParameters()
test1params["OCPI_OCPR__fi"].value = 0.7
test1params["OCPO_OCPI__fi"].value = 0.005
test1params["OCPI_OCPO__k"].value = 1/0.18
test1.updateParameters(test1params)   

data1 = copy.deepcopy(data0)
test1p = copy.deepcopy(test1)

#set initial populations
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

#test1p.plotYourself(data1,dpi=100,x_min=0, x_max=2000, title="")
#test1p.plotConcentrations(data1.all_data[-1],[0,1,2],dpi=100,x_min=0, x_max=2000, title="")
#lmfit.report_fit(params1)

simulation_double_photon_1 = test1p.recreateExperimentData(data1)

signals_2pmodel = []
wavelengths_2pmodel = []
no = 5
for x in simulation_double_photon_1:
    tmpkin = x
    signals_2pmodel.append(tmpkin[10+30])
    wavelengths_2pmodel.append(tmpkin.irradiation)
    no -= 1
signals_2pmodel = np.array(signals_2pmodel)
wavelengths_2pmodel = np.array(wavelengths_2pmodel)

   
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

#test1p.plotYourself(data1,dpi=100,x_min=0, x_max=2000, title="")
#test1p.plotConcentrations(data1.all_data[-1],[0,1,2],dpi=100,x_min=0, x_max=2000, title="")
#lmfit.report_fit(params1)

simulation_double_photon_1 = test1p.recreateExperimentData(data1)

signals_2pmodel_2 = []
wavelengths_2pmodel_2 = []
no = 5
for x in simulation_double_photon_1:
    tmpkin = x
    signals_2pmodel_2.append(tmpkin[10+30])
    wavelengths_2pmodel_2.append(tmpkin.irradiation)
    no -= 1
signals_2pmodel_2 = np.array(signals_2pmodel_2)
wavelengths_2pmodel_2 = np.array(wavelengths_2pmodel_2)

#zero at 750nm
signals_1pmodel = signals_1pmodel-signals_1pmodel[-1]
signals_2pmodel = signals_2pmodel-signals_2pmodel[-1]
signals_2pmodel_2 = signals_2pmodel_2-signals_2pmodel_2[-1]

#signal vs irrradtaion wavelength
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.plot(wavelengths_1pmodel, 2*signals_1pmodel/0.0013, "co-", label = "single h\u03BD model")
ax.plot(wavelengths_2pmodel, 2*signals_2pmodel/0.00122, "mo-", label = "two h\u03BD model, \u03c6\u2082=1.0")
ax.plot(wavelengths_2pmodel_2, 10.2*signals_2pmodel_2/0.0012, "ro-", label = "two h\u03BD model, \u03c6\u2082=0.1")
ax.plot(OCPO_Ctag_epsilons.w, OCPO_Ctag_epsilons.a/50000000/0.00125, "b-", label = "stationary spectrum")
ax.legend(shadow=False, frameon=True, prop={'size': 14}, labelspacing=0.1, loc="upper right")
ax.set_xlabel("Irradiation wavelength (nm)", fontdict={'size': 16})
ax.set_ylabel("Normalized action spectrum", fontdict={'size': 16})
#ax.set_ylim(0, 0.004)
#ax.set_xlim(9E-6, 1E-3)
ax.grid(True, which="major", linestyle='--')
ax.tick_params(which='both', direction="in", bottom=True, top=True, left=True, right=True, labelsize=14)
plt.show()


