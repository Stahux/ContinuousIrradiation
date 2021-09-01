#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("../../application/")

from model import Model
from model import Model
from kineticdata import KineticData, Experiment
from modfit import ModFit
import copy
import lmfit


model1 = Model.load('model_ABC_singlephoton.model')

datadir = "../../sample_datasets/naphtopyrans vs temp irr time and irr power/"
experiment1 = Experiment()
experiment1.loadKineticData(
        datadir+"NP chex 11C 30s 3.2 mW.txt",
        probe = 430, t_on = 10, t_off = 60, irradiation = 365,
        intensity = 2.3479e-06, absorbance = 0.32, 
        skip_header = 18, temperature = 11+273.15)
experiment1.loadKineticData(
        datadir+"NP chex 21C 10s 3.2 mW.txt",
        probe = 430, t_on = 10, t_off = 60, irradiation = 365,
        intensity = 2.3479e-06, absorbance = 0.32, 
        skip_header = 18, temperature = 21+273.15)
experiment1.loadKineticData(
        datadir+"NP chex 21C 30s 10mW.txt",
        probe = 430, t_on = 10, t_off = 60, irradiation = 365,
        intensity = 2.3479e-06, absorbance = 0.32, 
        skip_header = 18, temperature = 21+273.15)
experiment1.loadKineticData(
        datadir+"NP chex 21C 30s 1mW.txt",
        probe = 430, t_on = 10, t_off = 60, irradiation = 365,
        intensity = 2.3479e-06, absorbance = 0.32, 
        skip_header = 18, temperature = 21+273.15)
experiment1.loadKineticData(
        datadir+"NP chex 21C 30s 3.15-3.2 mW 4th checkpoint.txt",
        probe = 430, t_on = 10, t_off = 60, irradiation = 365,
        intensity = 2.3479e-06, absorbance = 0.32, 
        skip_header = 18, temperature = 21+273.15)
experiment1.loadKineticData(
        datadir+"NP chex 21C 30s 3.2 mW 1st checkpoint.txt",
        probe = 430, t_on = 10, t_off = 60, irradiation = 365,
        intensity = 2.3479e-06, absorbance = 0.32, 
        skip_header = 18, temperature = 21+273.15)
experiment1.loadKineticData(
        datadir+"NP chex 21C 30s 3.2 mW 2nd checkpoint.txt",
        probe = 430, t_on = 10, t_off = 60, irradiation = 365,
        intensity = 2.3479e-06, absorbance = 0.32, 
        skip_header = 18, temperature = 21+273.15)
experiment1.loadKineticData(
        datadir+"NP chex 21C 30s 3.2 mW 3rd checkpoint.txt",
        probe = 430, t_on = 10, t_off = 60, irradiation = 365,
        intensity = 2.3479e-06, absorbance = 0.32, 
        skip_header = 18, temperature = 21+273.15)
experiment1.loadKineticData(
        datadir+"NP chex 21C 90s 3.2 mW.txt",
        probe = 430, t_on = 10, t_off = 60, irradiation = 365,
        intensity = 2.3479e-06, absorbance = 0.32, 
        skip_header = 18, temperature = 21+273.15)
experiment1.loadKineticData(
        datadir+"NP chex 31C 30s 3.2 mW.txt",
        probe = 430, t_on = 10, t_off = 60, irradiation = 365,
        intensity = 2.3479e-06, absorbance = 0.32, 
        skip_header = 18, temperature = 31+273.15)

model1.setInitial("A",1)
model1.setInitial("B",0)
model1.setInitial("C",0)


def fitModel():
    params1 = experiment1.genParameters() + model1.genParameters()
    
    params1.pretty_print()
    
    params1['fi1__fi'].set(vary=True)
    params1['fi2__fi'].set(vary=True)
    params1['fi3__fi'].set(vary=True)
    params1['fi4__fi'].set(vary=True)
    params1['fisingle__fi'].set(vary=True)
    params1['k1__k'].set(vary=True)  
    
    experiment1.updateParameters(params1)
    model1.updateParameters(params1)
    
    #for i in range(0,4):
     #   test1.plotYourself(data1, i, title="Kinetic " + str(i))
        
    #model1.plotYourself(experiment1,dpi=160,x_min=-10, x_max=200)
    #print(params1)


    #out_data = model1.solveModel(experiment1)

    #model1.plotYourself(experiment1, x_max = 200)
    #model1.plotYourself(out_data, x_max = 200)
    
    modfit1 = ModFit(model1, experiment1, params1)
    out1 = modfit1.fit()
    params2 = out1.params    
    
    experiment1.updateParameters(params2)
    model1.updateParameters(params2)
    
    for i in range(10):
        model1.plotYourself(experiment1, i)
    model1.plotYourself(experiment1)
    
    lmfit.report_fit(params2)    
    
fitModel()









