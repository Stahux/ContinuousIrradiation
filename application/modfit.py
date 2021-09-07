"""
Contains classes based on lmfit ones, extending their
functionality for the purpose of this package.
"""
__version__ = "v0.2"
__authors__ = ["Stanisław Niziński"]
#__authors__.append("Add yourself my friend...")

import numpy as np
import lmfit
import copy
from numpy import inf
import json
from lmfit.jsonutils import decode4js

#allows to set also penalty_std, which will switch on penalty 
#and allow some swings in given parameter
#setting some penalty_std will setup penalty mechanism, and initial 
#value will be remembered as expected value

"""
 i copied oryginal code to change only Parameter to MParameter.
 not super elegant way, but should work well.
 in case of unintended conversuion from MParameters to Parameters,
 code will still keep information about penalty, etc (it is in user_data), but
 will lose methods used to nicely set up penalty etc.
 """    
class MParameters(lmfit.Parameters):   
    def __deepcopy__(self, memo):  
        _pars = self.__class__(asteval=None)

        unique_symbols = {key: self._asteval.symtable[key]
                          for key in self._asteval.user_defined_symbols()}
        _pars._asteval.symtable.update(unique_symbols)

        parameter_list = []
        for key, par in self.items():
            if isinstance(par, lmfit.Parameter):
                param = MParameter(name=par.name,
                                  value=par.value,
                                  min=par.min,
                                  max=par.max)
                param.vary = par.vary
                param.brute_step = par.brute_step
                param.stderr = par.stderr
                param.correl = par.correl
                param.init_value = par.init_value
                param.expr = par.expr
                param.user_data = par.user_data
                parameter_list.append(param)

        _pars.add_many(*parameter_list)

        return _pars  

    def add(self, name, value=None, vary=True, min=-inf, max=inf, expr=None,
            brute_step=None):

        if isinstance(name, lmfit.Parameter):
            self.__setitem__(name.name, name)
        else:
            self.__setitem__(name, MParameter(value=value, name=name, vary=vary,
                                             min=min, max=max, expr=expr,
                                             brute_step=brute_step))

    def add_many(self, *parlist):
        __params = []
        for par in parlist:
            if not isinstance(par, lmfit.Parameter):
                par = MParameter(*par)
            __params.append(par)
            par._delay_asteval = True
            self.__setitem__(par.name, par)

        for para in __params:
            para._delay_asteval = False
    
    
    def loads(self, s, **kws):
        self.clear()

        tmp = json.loads(s, **kws)
        unique_symbols = {key: decode4js(tmp['unique_symbols'][key]) for key
                          in tmp['unique_symbols']}

        state = {'unique_symbols': unique_symbols, 'params': []}
        for parstate in tmp['params']:
            _par = MParameter(name='')
            _par.__setstate__(parstate)
            state['params'].append(_par)
        self.__setstate__(state)
        return self

"""
Now one can use old Parameter from lmfit or MParameter, which allows
additionally to specify penalty for the parameter when going outside the std.
"""
class MParameter(lmfit.Parameter):
    def __init__(self, *args, penalty_std = None, penalty_weight = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_data = (penalty_std, penalty_weight, None)
    
    def set(self, *args, penalty_std = None, penalty_weight = None, **kwargs):
        if(penalty_std is not None):
            self.penalty_std = penalty_std
        if(penalty_weight is not None):
            self.penalty_weight = penalty_weight
        super().set(*args, **kwargs)

    @property
    def penalty_std(self):
        return self.user_data[0]

    @penalty_std.setter
    def penalty_std(self, val):
        if(val is not None): 
            if(val <= 0):
                raise Exception("Error! penalty_std cannot be negative or zero!")
        self.user_data = (val, self.user_data[1], self.user_data[2])

    @property
    def penalty_weight(self):
        return self.user_data[1]

    @penalty_weight.setter
    def penalty_weight(self, val):
        if(val is not None): 
            if(val <= 0):
                raise Exception("Error! penalty_weight cannot be negative or zero!")
        self.user_data = (self.user_data[0], val, self.user_data[2])

    @property
    def penalty_expected_value(self): #only getter
        return self.user_data[2]

    @property
    def penalty_enabled(self): #True if penalty enabled
        if(self.penalty_std is not None):
            return True
        else:
            return False
            

class ModFit(lmfit.Minimizer):    
    def __init__(self, model, experiment, params = None):
        
        """
        This class is extenson to lmfit.Mizimizer, but specialized in solving
        kinetic models of fotochemical reactions.
        Init ModFit object, with the data to be fitted and the model to be
        applied. Parameters need to be generated from the model and the data,
        then merged and modified by the user accordingly to his fitting strategy.
        One can feed these params here or later just before the fit.
        ModFit will build residual function from above inpust, and feed it
        to the lmfit package.
        
        Parameters
        ----------
        model : Model object (generated from this package, not lmfit one)
            Kinetic model which will be applied to the data.
        experiment : Experiment object
            Set of kinetic traces with specified experimental conditions,
            meant to be modelled together, using model specified.
        params : Parameters (from lmfit) or MParameters object
            Parameters generated from model and experiment, and adjusted by
            the user. They can be feed later just before the fit. If not
            specified at any point, ModFit will generate default ones automatically.

        """        
        
        if(params is not None): 
            params = self.initPenalty(params)
            
        if(params is None):
            model.genParameters()
            super().__init__(self.residual, 
                 model.genParameters() + experiment.genParameters(), 
                 nan_policy="propagate")  
        else:
            super().__init__(self.residual, params, nan_policy="propagate")  
          
        self.model = copy.deepcopy(model)
        self.experiment = copy.deepcopy(experiment)
    
    @staticmethod
    def normalDistribution(x, mean, variance): 
        #just normal dostribution
        return np.exp(-np.square((x-mean)/variance)/2)/(variance*np.sqrt(2*np.pi))

    @staticmethod
    def normalPenalty(value, expected, error, weight):
        #experimental penalty to keep param close within its error
        return weight/ModFit.normalDistribution(value, expected, error)
    
    def residual(self, params):
        """
        TODO:
            1) parallel calc residues for all kineticdata
            2) you can precompile model to avoid building equations every time
            3) preallocate residual vector, not extend
            4) ... probably more things
        """
        self.model.updateParameters(params)
        self.experiment.updateParameters(params)
        
        modelled_experiment = self.model.solveModel(self.experiment)
        
        res_vect = list() #here huge optimization obviously can be made
        
        for i in range(modelled_experiment.count):
            res_vect.extend(modelled_experiment.all_data[i].data_a - self.experiment.all_data[i].data_a)
        
        penalties = []
        for key, param in params.items():
            if(isinstance(param, MParameter)):
                if(param.penalty_enabled):
                    penalties.append(ModFit.normalPenalty(param.value, 
                          param.penalty_expected_value, 
                          param.penalty_std, 
                          param.penalty_weight))
        res_vect.extend(penalties)
            
        return res_vect
    
    def initPenalty(self, params): #just set third val in user_data to expected value
        params = params.copy()
        for key, param in params.items():
            if(isinstance(param, MParameter)):
                param.user_data = (param.user_data[0], param.user_data[1], param.value)
        return params
        
  
    def fit(self, params = None, **kwargs): 
        """
        Print fit report, just like in lmfit but with some additional info.
        
        Parameters
        ----------
        params : Parameters or MParameters
            Parameters consistent with used model and datasets (generated from
            them plus additional ones specified by user may appear. They should
            provide nice starting point por optimization and resaonable constraints.
            
        Returns
        -------
        MinimizerResult
            Object with fit-results returned by lmfit.
        """            
        if(params is not None): 
            params = self.initPenalty(params)
        output = self.minimize(params = params, **kwargs)
        
        return output
    
    def reportFit(self, params):
        """
        Print fit report, just like in lmfit but with some additional info.
        
        Parameters
        ----------
        params : Parameters or MParameters
            Parameters obtained from the last optiumization.
            
        """
        lmfit.report_fit(params) 
        
        for key, param in params.items():
            if(isinstance(param, MParameter)):
                if(param.penalty_enabled):
                    penalty_tmp = ModFit.normalPenalty(param.value, 
                          param.penalty_expected_value, 
                          param.penalty_std, 
                          param.penalty_weight) 
                    deviation = 100*abs(param.penalty_expected_value-param.value)/param.penalty_expected_value
                    sigmas = (param.value-param.penalty_expected_value)/param.penalty_std
                    print("For param: " + key + " penalty equals %.9f, due "
                          "to %.1f%% deviation from expected value (%.1f\u03c3)." 
                          % (penalty_tmp, deviation, sigmas))

        #print total penalty and chi2
        
        

#how it should work?
#firstly you build model, data objects
#then you generate parameters from them, and unfix some of them before fit
#so you need to feed ModFit with these three objects
#mostly after feeding these objects shouldnt change except fixation and values of some parameters
#so preferably you shuld build object with all three objects
#later eventually replace parameter object, or build function to do it from outside
#still try to stick to the original Minimizer architecture, apply minimal modifications...
        