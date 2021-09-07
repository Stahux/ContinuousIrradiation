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


model1 = Model.load('model_ABC_singlephoton_eyring.model')

datadir = "../../sample_datasets/naphtopyrans vs temp irr time and irr power/"
experiment1 = Experiment()
experiment1.loadKineticData(
        datadir+"NP chex 11C 30s 3.2 mW.txt",
        probe = 430, t_on = 10.5, t_off = 41, irradiation = 365,
        intensity = 2.3479e-06, absorbance = 0.32, 
        skip_header = 18, temperature = 11+273.15)
experiment1.loadKineticData(
        datadir+"NP chex 21C 10s 3.2 mW.txt",
        probe = 430, t_on = 10.5, t_off = 21, irradiation = 365,
        intensity = 2.3479e-06, absorbance = 0.32, 
        skip_header = 18, temperature = 21+273.15)
experiment1.loadKineticData(
        datadir+"NP chex 21C 30s 10mW.txt",
        probe = 430, t_on = 10, t_off = 41, irradiation = 365,
        intensity = 2.3479e-06, absorbance = 0.32, 
        skip_header = 18, temperature = 21+273.15)
experiment1.loadKineticData(
        datadir+"NP chex 21C 30s 1mW.txt",
        probe = 430, t_on = 10.5, t_off = 41.4, irradiation = 365,
        intensity = 2.3479e-06, absorbance = 0.32, 
        skip_header = 18, temperature = 21+273.15)
experiment1.loadKineticData(
        datadir+"NP chex 21C 30s 3.15-3.2 mW 4th checkpoint.txt",
        probe = 430, t_on = 10.5, t_off = 40.5, irradiation = 365,
        intensity = 2.3479e-06, absorbance = 0.32, 
        skip_header = 18, temperature = 21+273.15)
experiment1.loadKineticData(
        datadir+"NP chex 21C 30s 3.2 mW 1st checkpoint.txt",
        probe = 430, t_on = 10.5, t_off = 41.5, irradiation = 365,
        intensity = 2.3479e-06, absorbance = 0.32, 
        skip_header = 18, temperature = 21+273.15)
experiment1.loadKineticData(
        datadir+"NP chex 21C 30s 3.2 mW 2nd checkpoint.txt",
        probe = 430, t_on = 20.5, t_off = 50.5, irradiation = 365,
        intensity = 2.3479e-06, absorbance = 0.32, 
        skip_header = 18, temperature = 21+273.15)
experiment1.loadKineticData(
        datadir+"NP chex 21C 30s 3.2 mW 3rd checkpoint.txt",
        probe = 430, t_on = 10.5, t_off = 40.5, irradiation = 365,
        intensity = 2.3479e-06, absorbance = 0.32, 
        skip_header = 18, temperature = 21+273.15)
experiment1.loadKineticData(
        datadir+"NP chex 21C 90s 3.2 mW.txt",
        probe = 430, t_on = 10.5, t_off = 100.5, irradiation = 365,
        intensity = 2.3479e-06, absorbance = 0.32, 
        skip_header = 18, temperature = 21+273.15)
experiment1.loadKineticData(
        datadir+"NP chex 31C 30s 3.2 mW.txt",
        probe = 430, t_on = 10.5, t_off = 40.5, irradiation = 365,
        intensity = 2.3479e-06, absorbance = 0.32, 
        skip_header = 18, temperature = 31+273.15)

model1.setInitial("A",1)
model1.setInitial("B",0)
model1.setInitial("C",0)


def fitModel():
    params1 = experiment1.genParameters() + model1.genParameters()
    
    params1['fi1__fi'].set(vary=True)
    params1['fi2__fi'].set(vary=True)
    params1['fi3__fi'].set(vary=True)
    params1['fi4__fi'].set(vary=True)
    params1['fisingle__fi'].set(vary=True)
    params1['k1__deltaH'].set(vary=True, value=23)  
    params1['k1__deltaS'].set(vary=True, value=0.01) 

    mJ_per_hv = 5.44231976708963E-019*1000 #one 365nm photon [mJ]
    avogadro = 6.022140857E+023
    mw_to_intensity = ((1*1.5)/(3.14*(1/2)**2))*1000/(1*1.5*1*mJ_per_hv*avogadro)
    
    params1['_0_intensity'].set(value=mw_to_intensity * 3.2, vary = True)
    params1['_1_intensity'].set(value=mw_to_intensity * 3.2, vary = True)
    params1['_2_intensity'].set(value=mw_to_intensity * 10, vary = True)
    params1['_3_intensity'].set(value=mw_to_intensity * 1, vary = True)
    params1['_4_intensity'].set(value=mw_to_intensity * 3.2, vary = True)
    params1['_5_intensity'].set(value=mw_to_intensity * 3.2, vary = True)
    params1['_6_intensity'].set(value=mw_to_intensity * 3.2, vary = True)
    params1['_7_intensity'].set(value=mw_to_intensity * 3.2, vary = True)
    params1['_8_intensity'].set(value=mw_to_intensity * 3.2, vary = True)
    params1['_9_intensity'].set(value=mw_to_intensity * 3.2, vary = True) 
    
    for i in range(10):
       params1['_'+str(i)+'_intensity'].penalty_std = 3.8868e-06/10
       params1['_'+str(i)+'_intensity'].penalty_weight = 10

    params1.pretty_print()
    
    modfit1 = ModFit(model1, experiment1, params1)
    out1 = modfit1.fit(maxfev = 10000)
    params2 = out1.params    
    
    experiment1.updateParameters(params2)
    model1.updateParameters(params2)
    
    for i in range(10):
        model1.plotYourself(experiment1, i, x_max=200)
    model1.plotYourself(experiment1, x_max=200)
    
    modfit1.reportFit(params2)    
    
fitModel()




