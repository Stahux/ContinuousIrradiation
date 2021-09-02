import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import lmfit
import copy

class ModFit(lmfit.Minimizer):    
    def __init__(self, model, experiment, params = None):
        
        if params == None:
            model.genParameters()
            super().__init__(self.residual, model.genParameters() + experiment.genParameters(), nan_policy='propagate')  
        else:
            super().__init__(self.residual, params, nan_policy='propagate')  
          
        self.model = copy.deepcopy(model)
        self.experiment = copy.deepcopy(experiment)
    
    def residual(self, params):
        self.model.updateParameters(params)
        self.experiment.updateParameters(params)
        
        modelled_experiment = self.model.solveModel(self.experiment)
        res_vect = list()
        for i in range(modelled_experiment.count):
            res_vect.extend(modelled_experiment.all_data[i].data_a - self.experiment.all_data[i].data_a)
        return res_vect
    
  
    def fit(self, params = None, **kwargs):
        output = self.minimize(params = params, **kwargs)
        
        return output
        
        
        
        
#how it should work?
#firstly you build model, data objects
#then you generate parameters from them, and unfix some of them before fit
#so you need to feed ModFit with these three objects
#mostly after feeding these objects shouldnt change except fixation and values of some parameters
#so preferably you shuld build object with all three objects
#later eventually replace parameter object, or build function to do it from outside
#still try to stick to the original Minimizer architecture, apply minimal modifications...