import numpy as np
import lmfit
import copy
from numpy import inf
import json
from lmfit.jsonutils import decode4js

#allows to set also penalty_std, which will switch on penalty and allow some swings in given parameter
#setting some penalty_std will setup penalty mechanism, and initial value will be remembered as expected value

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

class MParameter(lmfit.Parameter):
    def __init__(self, *args, penalty_std = None, penalty_scale = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_data = (penalty_std, penalty_scale, None)
    
    def set(self, *args, penalty_std = None, penalty_scale = None, **kwargs):
        if penalty_std is not None:
            self.penalty_std = penalty_std
        if penalty_scale is not None:
            self.penalty_scale = penalty_scale
        super().set(*args, **kwargs)

    @property
    def penalty_std(self):
        return self.user_data[0]

    @penalty_std.setter
    def penalty_std(self, val):
        if val is not None: 
            if(val <= 0):
                raise Exception("Error! penalty_std cannot be negative or zero!")
        self.user_data = (val, self.user_data[1], self.user_data[2])

    @property
    def penalty_scale(self):
        return self.user_data[1]

    @penalty_scale.setter
    def penalty_scale(self, val):
        if val is not None: 
            if(val <= 0):
                raise Exception("Error! penalty_scale cannot be negative or zero!")
        self.user_data = (self.user_data[0], val, self.user_data[2])


class ModFit(lmfit.Minimizer):    
    def __init__(self, model, experiment, params = None):
        if(params != None): 
            params = self.initPenalty(params)
            
        if params == None:
            model.genParameters()
            super().__init__(self.residual, model.genParameters() + experiment.genParameters(), nan_policy='propagate')  
        else:
            super().__init__(self.residual, params, nan_policy='propagate')  
          
        self.model = copy.deepcopy(model)
        self.experiment = copy.deepcopy(experiment)
    
    @staticmethod
    def normalDistribution(x, mean, variance): #just normal dostribution
        return np.exp(-np.square((x-mean)/variance)/2)/(variance*np.sqrt(2*np.pi))

    @staticmethod
    def normalPenalty(value, expected, error, weight): #experimental penalty to keep param close within its error
        return weight/ModFit.normalDistribution(value, expected, error)
    
    def residual(self, params):
        self.model.updateParameters(params)
        self.experiment.updateParameters(params)
        
        
        modelled_experiment = self.model.solveModel(self.experiment)
        res_vect = list()
        for i in range(modelled_experiment.count):
            res_vect.extend(modelled_experiment.all_data[i].data_a - self.experiment.all_data[i].data_a)
        return res_vect
    
    def initPenalty(self, params): #just set third val in user_data to expected value
        params = params.copy()
        for key, param in params.items():
            if(isinstance(param, MParameter)):
                param.user_data = (param.user_data[0], param.user_data[1], param.value)
        return params
        
  
    def fit(self, params = None, **kwargs):
        if(params != None): 
            params = self.initPenalty(params)
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