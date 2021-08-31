#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("../../application/")

from model import Model
from model import Model
from kineticdata import KineticData, Experiment
from modfit import ModFit
import copy


#performs simple simulation without any data input
test1 = Model.load('model_ABC_singlephoton.model')

data1 = Experiment()
data1.loadKineticData("", probe = 430, t_on = 0, t_off = 60, irradiation = 365, intensity = 2.3479e-06, absorbance = 0.3, src = "NaN")
data1.loadKineticData("", probe = 430, t_on = 0, t_off = 60, irradiation = 365, intensity = 9.2117e-06, absorbance = 0.3, src = "NaN")
data1.loadKineticData("", probe = 430, t_on = 0, t_off = 60, irradiation = 365, intensity = 3.4768e-05, absorbance = 0.3, src = "NaN")
data1.loadKineticData("", probe = 430, t_on = 0, t_off = 60, irradiation = 365, intensity = 0.00013039, absorbance = 0.3, src = "NaN")               

test1.setInitial("A",[1,1,1,1])
test1.setInitial("B",[0,0,0,0])
test1.setInitial("C",[0,0,0,0])
        
    
def simulateIt():
    params1 = data1.genParameters() + test1.genParameters()
    
    data1.updateParameters(params1)
    test1.updateParameters(params1)
    
    #for i in range(0,4):
     #   test1.plotYourself(data1, i, title="Kinetic " + str(i))
        
    test1.plotYourself(data1,dpi=160,x_min=-10, x_max=200)
    #print(params1)

    
simulateIt()
