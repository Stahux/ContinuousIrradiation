from model import Model
from kineticdata import KineticData, Experiment
from modfit import ModFit
import numpy as np
import matplotlib.pyplot as plt
import lmfit
import copy
import random
import pickle



test1 = Model.load('model_NP2P.model')

data1 = Experiment()
data1.loadKineticData("", probe = 450, t_on = 0, t_off = 60, irradiation = 365, intensity = 2.3479e-06, absorbance = 0.3, src = "NaN")
data1.loadKineticData("", probe = 450, t_on = 0, t_off = 60, irradiation = 365, intensity = 9.2117e-06, absorbance = 0.3, src = "NaN")
data1.loadKineticData("", probe = 450, t_on = 0, t_off = 60, irradiation = 365, intensity = 3.4768e-05, absorbance = 0.3, src = "NaN")
data1.loadKineticData("", probe = 450, t_on = 0, t_off = 60, irradiation = 365, intensity = 0.00013039, absorbance = 0.3, src = "NaN")               

test1.setInitial("CF",[1,1,1,1])
test1.setInitial("TC",[0,0,0,0])
test1.setInitial("TT",[0,0,0,0])
        
    
def simulateThisShit(thermal):
    params1 = data1.genParameters() + test1.genParameters()
    params1['eq1__k'].set(thermal)
    
    data1.updateParameters(params1)
    test1.updateParameters(params1)
    
    #for i in range(0,4):
     #   test1.plotYourself(data1, i, title="Kinetic " + str(i))
        
    test1.plotYourself(data1,dpi=160,x_min=-10, x_max=200, title="tau="+str(1/thermal)+"s")
    #print(params1)

    
simulateThisShit(1/8.8339222614841)
simulateThisShit(1/12)
simulateThisShit(1/15)
simulateThisShit(1/18)