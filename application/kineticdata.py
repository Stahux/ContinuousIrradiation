"""
Implements classes used to contain the data, and to do some simple data
manipulations. Especially, Experiment class is intended to contain full
set of data, which posses some integrity (or just shared experimental
conditions) and are meant to be modelled together.
"""
__version__ = "v0.2"
__authors__ = ["Stanisław Niziński"]
#__authors__.append("Add yourself my friend...")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from modfit import MParameter, MParameters
import lmfit
from scipy.signal import savgol_filter
import pickle


class LightEvent: 
    #designates light distribution starting from some moment to the next lightevent
    def __init__(self, wavelength, intensity, start_time):
        #verify if thse values make any sense
       self.wavelength = wavelength
       self.intensity = intensity
       self.start_time = start_time


class KineticData:
    #irr and probe lengths in cm.
    #absorbance is defined at irradiation wavelength and length
    #one can specify absorbance (at irr), or concentration (at beginning in M)
    def __init__(self, filename, probe, irradiation, intensity = 0.0, 
                 absorbance = None, concentration = None,
                 t_on = None, t_off = None, probe_length = 1, 
                 irradiation_length = 1, num = None, zeroed = False, src = None, 
                 skip_header = 0, skip_footer = 0, temperature = None, name = None):    
        if(src is None):
            #inport data
            #data = np.loadtxt(filename) #previously used
            data = np.genfromtxt(filename, skip_header=skip_header, 
                                 skip_footer=skip_footer, dtype=float, 
                                 invalid_raise = False)
            self.data_t = data[:,0]
            self.data_a = data[:,1]
        elif(src == "Lucas"):    #read data of meine beste freunde lucasso  !
            #!!! set the area you want to average (because 
            #in these data wavelengths are densely probed)
            avg_area = 4      
            #two columns are not absorbances (column with numbering + time)
            dead_columns = 2    
            tmp = pd.read_csv(filename)        
            columns = tmp.columns.values[dead_columns:]                    
            columns2 = columns.astype(np.float)
            positions = np.where((columns2 >= probe-avg_area/2) & (columns2 <= probe+avg_area/2))[0]  
            columns_names = [columns[pos] for pos in positions]
            selected_columns = tmp[columns_names]                    
            
            self.data_t = tmp["Time"].to_numpy() 
            self.data_a = selected_columns.mean(axis = 1, skipna = True).to_numpy()       
        else: #just generate some grid and NaNs as values, ignore filename value
            if(t_on is None): raise Exception("Please set some t_on value!")
            if(t_off is None): raise Exception("Please set some t_off value!")
            irr_time = np.abs(t_off-t_on)
            self.data_t = np.linspace(t_on-10, t_off+irr_time*5, num=10000)
            self.data_a = self.data_t * np.nan
            
        #sort the data, in case that they are not sorted!
        order = np.argsort(self.data_t)
        self.data_t = self.data_t[order]
        self.data_a = self.data_a[order]
            
        #self.light = light #later there should be array of lighevents...
        self.probe = probe #probing wavelength
        self.irradiation = irradiation
        self.intensity = intensity
        self.temperature = temperature
        
        if(absorbance is not None):
            #lets say for now that it is absorbance at irradiation wavelength in the ground state
            self.absorbance = absorbance
            self.concentration = None
        if(concentration is not None):
            #at the beginning, in M
            self.concentration = concentration
            self.absorbance = None
        if(concentration is not None and absorbance is not None):
            raise Exception("Concentration and absorbance cannot be defined both in the same time.")
        if(concentration is None and absorbance is None):
            raise Exception("Specify concentration OR absorbance.")
        
        if(t_on is None):
            self.t_on = self.data_t[0]
        else:
            self.t_on = t_on #before t0 there is no irradiation

        if(t_off is None):
            self.t_off = self.data_t[-1]
        else:
            self.t_off = t_off #before t0 there is no irradiation
            
        self.probe_length = probe_length
        self.irradiation_length = irradiation_length
        
        self.num = num
        if(name is None):
            self.name = filename
        else:
            self.name = name
        self.zeroed = zeroed #if true, means that before t_on absorbance was set to zero
        
        #to be sure that after loading we always finish with np.array lists
        #self.data_t = np.array(self.data_t)
        #self.data_a = np.array(self.data_a)
        
    def save(self, filename):
        np.savetxt(filename + ".txt", np.transpose(np.vstack([self.data_t,self.data_a])), delimiter=',')
        
    def selectTimes(self, t_min, t_max): #TODO: checl consistency, written fast
        index_min = np.argmin(np.abs(self.data_t-t_min))
        index_max = np.argmin(np.abs(self.data_t-t_max))
        self.data_t = self.data_t[index_min:index_max-1] #check again!
        self.data_a = self.data_a[index_min:index_max-1]
        if(self.t_off > self.data_t[-1]):
            self.t_off = self.data_t[-1]
    
    def plotYourself(self, dpi=100):
        plt.figure(figsize=(8, 6), dpi=dpi)
        plt.plot(self.data_t, self.data_a, "bo")
        #plt.plot(self.data_t, residual(out2.params) + self.data_a, "r-")
        plt.show()        

    def genParameters(self):
        #parameters are fixed by default, unfix some of them before fitting procedure
        params = MParameters()
        numstring = ""
        if(self.num is not None):
            numstring = "_" + str(self.num) + "_"
        params.add(MParameter(numstring+"probe", value=float(self.probe), vary=False, min=0))
        params.add(MParameter(numstring+"irradiation", value=float(self.irradiation), vary=False, min=0))
        params.add(MParameter(numstring+"intensity", value=self.intensity, vary=False, min=0))
        params.add(MParameter(numstring+"t_on", value=self.t_on, vary=False))
        params.add(MParameter(numstring+"t_off", value=self.t_off, vary=False))
        params.add(MParameter(numstring+"probe_length", value=self.probe_length, vary=False, min=0))
        params.add(MParameter(numstring+"irradiation_length", value=self.irradiation_length, vary=False, min=0)) 
        if(self.absorbance is not None):
            params.add(MParameter(numstring+"absorbance", value=self.absorbance, vary=False, min=0))
        if(self.concentration is not None):
            params.add(MParameter(numstring+"concentration", value=self.concentration, vary=False, min=0))
        params.add(MParameter(numstring+"temperature", value=self.temperature, vary=False, min=0)) 
        return params
    
    def updateParameters(self, params):
        p = params.valuesdict()
        numstring = ""
        if(self.num is not None):
            numstring = "_" + str(self.num) + "_"
        self.probe = p[numstring+"probe"]
        self.irradiation = p[numstring+"irradiation"]
        self.intensity = p[numstring+"intensity"]
        self.t_on = p[numstring+"t_on"]
        self.t_off = p[numstring+"t_off"]
        self.probe_length = p[numstring+"probe_length"]
        self.irradiation_length = p[numstring+"irradiation_length"]
        if(numstring+"absorbance" in p):
            self.absorbance = p[numstring+"absorbance"]
            self.concentration = None
        elif(numstring+"concentration" in p):
            self.concentration = p[numstring+"concentration"]
            self.absorbance = None
        else:
            raise Exception("Faulty params, no concentraion or absorbance is defined!")
        self.temperature = p[numstring+"temperature"]

    #simple exp fitting funcs
    def fitExp(self, guesses = None, a_guesses = None, t0 = 0.0, fix = None, t_min = None, t_max = None):
        #guesses is array with tau guess values, their number determine number of exps to fit
        #a_guesses must contain preexponential values associated with taus, mandatory guesses.shape == a_guesses.shape
        #t_0 is t_0 value (always fixed), fix must stay None or array of booleans True if given tau must be fixed
        #t_min and t_max may specify boundaries of the fit. that's all
        if(guesses == None):
            raise Exception("Give me some guess man!")
        if(a_guesses == None):
            a_guesses = np.array(guesses)*0.0
        if(fix == None):
            fix = [False]*len(guesses)

        if(t_min is None):
            index_min = 0
        else:
            index_min = np.argmin(np.abs(t_min-self.data_t))
            
        if(t_max is None):
            index_max = len(self.data_t)-1
        else:
            index_max = np.argmin(np.abs(t_max-self.data_t))

        p = lmfit.Parameters()
        p.add("nexp", len(guesses), vary=False)
        for component in range(1,len(guesses)+1):
            p.add("t"+ str(component), guesses[component-1], vary = not(fix[component-1]))
            p.add("A"+ str(component), a_guesses[component-1], vary = True)
        p.add("t0", t0, vary = False)
        
        p.add("index_max", index_max, vary = False)
        p.add("index_min", index_min, vary = False)

        p_out = self.optimize(p, self.residual)
        self.plotFit(p_out)
        return p_out
    
    def fitSecondOrder(self, k_guess = 1, a_guess = 1, t0 = 0.0, k_fix = False, offset = 0, t_min = None, t_max = None):
        #the same as above, but for second order reaction - dimerization (simplest case)
        if(t_min is None):
            index_min = 0
        else:
            index_min = np.argmin(np.abs(t_min-self.data_t))

        if(t_max is None):
            index_max = len(self.data_t)-1
        else:
            index_max = np.argmin(np.abs(t_max-self.data_t))

        p = lmfit.Parameters()
        p.add("k", k_guess, vary = not(k_fix), min = 0)
        p.add("A", a_guess, vary = True, min = 0)
        p.add("t0", t0, vary = False)
        p.add("offset", offset, vary = True)
        
        p.add("index_max", index_max, vary = False)
        p.add("index_min", index_min, vary = False)

        p_out = self.optimize(p, self.residualSO)
        self.plotFit(p_out)
        return p_out

    def residual(self, params):
        p = params.valuesdict()
        nexp = int(p["nexp"])
        index_max = int(p["index_max"])
        index_min = int(p["index_min"])

        As = []
        Taus = []
        for component in range(1,nexp+1):
            As.append(p["A"+ str(component)])
            Taus.append(p["t"+ str(component)])
        t0 = p["t0"]
        y_model = self.multipleGaussExp(self.data_t[index_min:index_max+1], t0, As, Taus, 0.0)

        return np.subtract(self.data_a[index_min:index_max+1], y_model)

    def residualSO(self, params):
        p = params.valuesdict()
        index_max = int(p["index_max"])
        index_min = int(p["index_min"])

        k = p["k"]
        A = p["A"]
        t0 = p["t0"]
        offset = p["offset"]
        
        y_model = self.secondOrder(self.data_t[index_min:index_max+1], t0, A, k, offset)

        return np.subtract(self.data_a[index_min:index_max+1], y_model)

    def multipleGaussExp(self, t, t0, As, taus, offset): 
        #As and taus should be lists/tuples of the same length
        return_value = offset
        for i in range(len(taus)):
            return_value += self.gaussExp(t-t0, As[i], taus[i])
        return return_value 
    
    def gaussExp(self, t, A, tau):
        return A * np.exp(- t / tau)
    
    def secondOrder(self, t, t0, A0, k, offset):
        return A0/(A0*k*(t-t0)+1) + offset
    
    def optimize(self, params, residual_func):
        mini = lmfit.Minimizer(residual_func, params, nan_policy='propagate')
        out = mini.leastsq()
        #lmfit.report_fit(out.params) #robi dużo zamieszania
        print("chisquare is " + str(out.chisqr))
        self.last_chisqr = out.chisqr
        return out.params

    def plotFit(self, params, dpi=100):
        p = params.valuesdict()
        index_max = int(p["index_max"])
        index_min = int(p["index_min"]) 
        
        if("nexp" in p): #this is multiexp fit
            nexp = int(p["nexp"])

            As = []
            Taus = []
            for component in range(1,nexp+1):
                As.append(p["A"+ str(component)])
                Taus.append(p["t"+ str(component)])
            t0 = p["t0"]
            y_model = self.multipleGaussExp(self.data_t[index_min:index_max+1], t0, As, Taus, 0.0)  
            
        elif("k" in p): #this is dimerization second order fit (simple one)
            k = p["k"]
            A = p["A"]
            t0 = p["t0"] 
            offset = p["offset"] 
            y_model = self.secondOrder(self.data_t[index_min:index_max+1], t0, A, k, offset)
             
        plt.figure(figsize=(8, 6), dpi=dpi)
        plt.plot(self.data_t, self.data_a, 'bo')
        plt.plot(self.data_t[index_min:index_max+1], y_model, 'r-')
        plt.xlabel("Time")
        plt.ylabel("Signal")
        plt.grid(True, which="major", linestyle='--')
        #plt.xticks(fontsize=12)
        #plt.yticks(fontsize=12)
        plt.title("Fit of kinetic " + self.name)
        
        if("nexp" in p): #this is multiexp fit
            sum_preexps = 0.0
            for i in range(1,nexp+1):
                sum_preexps += p["A"+ str(i)]
            desc = ""
            for i in range(1,nexp+1):
                contribution = 100*p["A"+str(i)]/sum_preexps
                desc += "tau_" + str(i) + " = {a:.2e} ({b:.1f}%)\n".format(a=p["t"+str(i)], b=contribution)
            plt.figtext(0.5, 0.7, desc, fontdict={'size': 12})
        elif("k" in p): #this is dimerization second order fit (simple one)            
            desc = "k = {a:.2e} 1/M/s\nA\u2080 = {b:.2e} M".format(a=k, b=A)
            plt.figtext(0.5, 0.7, desc, fontdict={'size': 12})
            
        plt.show()
        print(params)

    def plotRateVsA(self, power = 1, x_min = None, x_max = None, dpi=100):
        #plot d signal/dt vs signal to test order of reaction
        tmp_a = self.data_a - self.data_a[-1] #zero at end
        rate = -np.gradient(tmp_a, self.data_t)
        plt.figure(figsize=(8, 6), dpi=dpi)
        plt.plot(np.power(tmp_a,power), rate, 'bo')
        plt.xlabel("Signal")
        plt.ylabel("Rate")
        plt.xlim(x_min, x_max)
        plt.grid(True, which="major", linestyle='--')
        #plt.xticks(fontsize=12)
        #plt.yticks(fontsize=12)
        plt.title("Kinetic " + self.name)
        plt.show()

    def plotRateVsLog(self, x_min = None, x_max = None, smooth = None, dpi=100):
        #plot d signal/dt vs signal to test order of reaction

        if(x_min is not None):
            index_min = np.argmin(np.abs(self.data_t-x_min))
        else:
            index_min = 0

        if(x_max is not None):
            index_max = np.argmin(np.abs(self.data_t-x_max))
        else:
            index_max = len(self.data_t)-1

        tmp_a = self.data_a[index_min:index_max-1]
        tmp_t = self.data_t[index_min:index_max-1]

        if(smooth is not None):
            tmp_a_sm = savgol_filter(tmp_a, 3*3, 3)
            #plt.figure(figsize=(8, 6), dpi=dpi)
            #plt.plot(tmp_t, tmp_a, 'bo') 
            #plt.plot(tmp_t, tmp_a_sm, 'ro') 
            #plt.show()
            tmp_a = tmp_a_sm

        rate = -np.gradient(tmp_a, tmp_t)
        x = np.log(tmp_a)
        y = np.log(rate)
        
        new_x = []
        new_y = []
        for i in range(x.shape[0]): #dirty but works. cleans nans
            if(np.isfinite(x[i]) and np.isfinite(y[i])):
                new_x.append(x[i])
                new_y.append(y[i])
        x = np.array(new_x)
        y = np.array(new_y)

        mod = lmfit.models.LinearModel()
        pars = mod.guess(y, x=x)
        out = mod.fit(y, pars, x=x)        
        slope = out.params["slope"].value
        
        plt.figure(figsize=(8, 6), dpi=dpi)
        plt.plot(x, y, 'bo')
        plt.plot(x, out.best_fit, 'r-')    
        plt.ylabel("Log(dA/dt)", fontdict={'size': 16})
        plt.xlabel("Log(\u0394A)", fontdict={'size': 16})
        #plt.xlim(x_min, x_max)
        plt.grid(True, which="major", linestyle='--')
        plt.figtext(0.6, 0.15, "slope = " + "{:.2f}".format(slope), fontdict={'size': 17})
        #plt.xticks(fontsize=12)
        #plt.yticks(fontsize=12)
        plt.grid(True, which="major", linestyle='--')
        plt.tick_params(which='both', direction="in", bottom=True, top=True, left=True, right=True, labelsize=14)
        plt.title("Kinetic: " + self.name + ", intensity: " + str(self.intensity) + "\n")
        plt.show()

    def getInitialSlope(self, points_no = 10, time_window = None, plot = False, dpi=100):
        #used to calc initial slope of growing, starts at t_on
        start_index = np.argmin(np.abs(self.data_t-self.t_on))+1
        
        if(time_window is not None):
            t_end = self.data_t[start_index]+time_window
            end_index = np.argmin(np.abs(self.data_t-t_end))
            points_no = end_index-start_index+1
        
        x = self.data_t[start_index:start_index+points_no]
        y = self.data_a[start_index:start_index+points_no]
        
        mod = lmfit.models.LinearModel()
        pars = mod.guess(y, x=x)
        out = mod.fit(y, pars, x=x)        
        slope = out.params["slope"].value
        slopeerr = out.params["slope"].stderr
        
        if(plot):
            plt.figure(figsize=(8, 6), dpi=dpi)
            plt.plot(x-self.t_on, y, 'bo', label='raw data')
            #plt.plot(x, out.init_fit, 'k--', label='initial fit')
            plt.plot(x-self.t_on, out.best_fit, 'r-', label='best fit')
            plt.legend(loc='best')
            plt.xlabel("Time (s)", fontdict={'size': 14})
            plt.ylabel("A", fontdict={'size': 14})
            plt.grid(True, which="major", linestyle='--')
            plt.figtext(0.6, 0.15, "slope = " + "{:.2e}".format(slope), fontdict={'size': 12})
            if(type(self.name) is str):
                plt.title("Initial slope for kinetic " + self.name + "\n")
            plt.show()
        
        return slope, slopeerr

    def plotAroundPoint(self, time, width = 50, dpi=100):
        #show data around some point to check if this is true t_on or t_off (alignment)
        start_index = np.argmin(np.abs(time-width-self.data_t))
        end_index = np.argmin(np.abs(time+width-self.data_t))
        
        x = self.data_t[start_index:end_index+1]
        y = self.data_a[start_index:end_index+1]

        plt.figure(figsize=(8, 6), dpi=dpi)
        plt.plot(x, y, 'ro--')
        plt.xlim(time-width/2,time+width/2)
        plt.xlabel("Time")
        plt.ylabel("Signal")
        plt.axvline(time, 0, 1, linewidth=1, color='k', linestyle="--")
        plt.grid(True, which="major", linestyle='--')
        plt.title("Initial slope for kinetic " + self.name)
        plt.show()

    def residualLBC(self, params):
        p = params.valuesdict()
        nexp = int(p["nexp"])
        
        residuals = []
        As = []
        Taus = []
        for component in range(1,nexp+1):
            As.append(p["A"+ str(component)])
            Taus.append(p["t"+ str(component)])
        offset = p["offset"]
        t0 = p["t0"]
        y_model = self.multipleGaussExp(self.data_t, t0, As, Taus, offset)
        
        splitpoint = 0
        for i in range(len(self.data_t)-1): #split data between on and off light regime
            if(self.data_t[i] <= self.t_off and self.t_off < self.data_t[i+1]):
                splitpoint = i
                break
        
        residuals.extend(np.subtract(self.data_a[splitpoint+1:], y_model[splitpoint+1:]))
        
        #plt.figure() #for eventual diagnostics
        #plt.plot(self.data_t[splitpoint+1:], y_model[splitpoint+1:], "b-")
        #plt.plot(self.data_t[splitpoint+1:], self.data_a[splitpoint+1:], "ro")
        #plt.show()        
        
        return residuals
    
    def smooth(self, window_length = 3*3, polyorder = 3):
        self.data_a = savgol_filter(self.data_a, window_length, polyorder)
     
    def linearBaselineCorrect(self, exp_no): 
        #it is made to correct roughly data done without reference. 
        #it assumes linear drift during experiment time.
        #so it will make fit of decay after t_off, and find offset. 
        #then it will assume that offset should be equal to absorbance 
        #before t_on, and it will do linear correction. if exp_no == 0, then 
        #the last point at kinetic will be taken as offset
        if(exp_no == 0):
            cfactor = (self.data_a[-1] - self.data_a[0])/(len(self.data_a)-1)
            newdata = [(self.data_a[i] - cfactor*i) for i in range(len(self.data_a))]
            self.data_a = newdata
        else:
            p = lmfit.Parameters()
            p.add("nexp", exp_no, vary=False)
            for component in range(1,exp_no+1):
                p.add("A"+ str(component), max(self.data_a)/exp_no)
                p.add("t"+ str(component), component*(self.data_t[-1] - self.t_off)/(3*exp_no))
            p.add("offset", 0.0)
            p.add("t0", self.t_off, vary=False)      
            
            mini = lmfit.Minimizer(self.residualLBC, p, nan_policy="propagate")
            out = mini.minimize(method="leastsq")
            offset = out.params["offset"]
            lmfit.report_fit(out.params)
            cfactor = (offset - self.data_a[0])/(len(self.data_a)-1)
            newdata = [(self.data_a[i] - cfactor*i) for i in range(len(self.data_a))]
            self.data_a = newdata                 

    def __getitem__(self, t): #return value at time in a portable
        return self.data_a[np.argmin(np.abs(self.data_t - t))]
        
    def zeroAt(self, t): #zero data at and mark it is zeroed
        self.data_a -= self[t]   
        self.zeroed = True
        
    def maximum(self):
        return self.data_t[np.argmax(self.data_a)]

    def minimum(self):
        return self.data_t[np.argmin(self.data_a)]

    def normAt(self, t):
        index = np.argmin(np.abs(self.data_t - t))
        if(self.data_a[index] < 0):
            self.data_a = - self.data_a / self.data_a[index]
        elif(self.data_a[index] > 0):
            self.data_a = self.data_a / self.data_a[index]
        
    def __sub__(self, offset):
        tmp = copy.deepcopy(self)
        tmp.data_a -= offset
        return tmp 
    
    def __isub__(self, offset):
        self.data_a -= offset    
    
    def __add__(self, offset): 
        tmp = copy.deepcopy(self)
        tmp.data_a += offset
        return tmp

    def __iadd__(self, offset):
        self.data_a += offset         

    def __mul__(self, value): 
        tmp = copy.deepcopy(self)
        tmp.data_a *= value
        return tmp      

    def __imul__(self, value): 
        self.data_a *= value  

    def __neg__(self):
        tmp = copy.deepcopy(self)
        tmp.data_a = -tmp.data_a
        return tmp
    
    def __pos__(self):
        tmp = copy.deepcopy(self)
        tmp.data_a = +tmp.data_a
        return tmp    

    def __truediv__(self, o):
        tmp = copy.deepcopy(self)
        tmp.a = tmp.data_a / o #assumes o is number
        return tmp        
    
    def __idiv__(self, o):
        self.a /= o #assumes o is number
    
            
class Experiment: 
    #container for kineticdata or other future data types prepared to fit globally
    def __init__(self, kineticdata = None):
        self.all_data = list()
        #self.count = 0
        if(type(kineticdata) is list): #if it is list of KineticData
            for x in kineticdata:
                self.all_data.append(copy.deepcopy(x))
            self._renumber()
        elif(kineticdata is not None): #if it is single KineticData
            self.all_data.append(copy.deepcopy(kineticdata))
            self._renumber()            
        
    def loadKineticData(self, *args, **kwargs):
        newdata = KineticData(*args, num = len(self.all_data), **kwargs)
        self.all_data.append(newdata)
        #self.count += 1
        
    def addKineticData(self, kineticdata):
        kineticdata.num = len(self.all_data)
        self.all_data.append(kineticdata)
        self._renumber()
        #self.count += 1

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        
    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            loaded = pickle.load(f)
        return loaded
        
    def genParameters(self):
        generated_params = MParameters()
        for i in range(len(self.all_data)):
            generated_params += self.all_data[i].genParameters()
        return generated_params
        
    def updateParameters(self, params): 
        for i in range(len(self.all_data)):
            self.all_data[i].updateParameters(params)
 
    def plotYourself(self, num = None, dpi=100, 
                     x_min = None, x_max = None, y_min = None, y_max = None,
                     zero_at = None, xlabel = "Time", ylabel = "Signal", 
                     labeling = None, intensity_scale = None,
                     loc="best"):
        tmp_data = copy.deepcopy(self.all_data)
        
        if(zero_at is not None):
            for i in range(len(tmp_data)):
                tmp_data[i].zeroAt(zero_at)
        
        labels = []
        if(labeling is None):
            labels = [tmp_data[i].name for i in range(len(tmp_data))]
        elif(labeling == "intensity"):
            labels = ["{:.0f} \u00B5mol L\u207B\u00B9 s\u207B\u00B9".format(tmp_data[i].intensity*10**6) for i in range(len(tmp_data))]            
        elif(labeling == "intensity_flux"):
            labels = ["{:.0f} \u00B5mol m\u207B\u00B2 s\u207B\u00B9".format(tmp_data[i].intensity*intensity_scale*10**6) for i in range(len(tmp_data))]            
        elif(labeling == "temperature"):
            labels = ["{:.0f} \u2103".format(tmp_data[i].temperature-273.15) for i in range(len(tmp_data))]            
                      
        plt.figure(figsize=(8, 6), dpi=dpi)
        if(num is None): #later remove this option, to plot one kinetics this way
            for i in range(len(tmp_data)):
                plt.plot(tmp_data[i].data_t, tmp_data[i].data_a, "-", label=labels[i])
        else:
            plt.plot(tmp_data[num].data_t, tmp_data[num].data_a, "-", label=labels[num])
                
        plt.legend(shadow=False, frameon=True, prop={'size': 16}, labelspacing=0.1, loc=loc)
        plt.xlabel(xlabel, fontdict={'size': 16})
        plt.ylabel(ylabel, fontdict={'size': 16})
        plt.grid(True, which="major", linestyle='--')
        #plt.grid(True, which="minor", linestyle='--')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.tick_params(which='both', direction="in", bottom=True, top=True, left=True, right=True, labelsize=14)
        plt.show()

    def linearBaselineCorrect(self, exp_no):
        for i in range(len(self.all_data)):
            self.all_data[i].linearBaselineCorrect(exp_no)

    def fuseTwoKinetics(self, kinetic1_id, kinetic2_id, delay_between):
        #merge two kinetics into one (not do average, but connect at some point
        #for example connect growing kinetic and decay into one kinetic
        #at the end only merged kinetic stays in the experiment object
        #takes kinetic attributes from kinetic1, does not compare them
        #delay between is value by which kinetic2 is shifted compared to kinetic1
        #delay_between=0 means that there is full temporal overlap between kinetics
        tmp_kinetic = copy.deepcopy(self[kinetic1_id])
        k1 = self[kinetic1_id]
        k2 = self[kinetic2_id]
        tmp_times = np.unique(np.concatenate([k1.data_t,delay_between+k2.data_t]))
        #may be a good idea to project onto new grid later
        k1_indexes = [np.where(k1.data_t==tmp_times[i])[0] for i in range(tmp_times.shape[0])]
        k2_indexes = [np.where(delay_between+k2.data_t==tmp_times[i])[0] for i in range(tmp_times.shape[0])]
        tmp_values = [np.average(np.concatenate([k1.data_a[k1_indexes[i]],k2.data_a[k2_indexes[i]]]))
                                                    for i in range(tmp_times.shape[0])]
        
        tmp_kinetic.data_t = np.array(tmp_times)
        tmp_kinetic.data_a = np.array(tmp_values)

        self.all_data.remove(k1)
        self.all_data.remove(k2)
        self.all_data.append(tmp_kinetic)
        self._renumber()
        
    def splitKineticIntoTwo(self, kinetic_id, split_point): 
        #to be implemented, idea is to take one kinetic, split into two parts
        #delete oryginal kinetic and replace it with two parts
        pass

    #simple exp fitting funcs, but for whole Experiment
    def fitExpParams(self, nexp):
        p = lmfit.Parameters()
        p.add("nexp", nexp, vary=False)
        for component in range(1,nexp+1):
            for kinetic_no in range(len(self.all_data)):
                p.add("t"+ str(component)+"_"+str(kinetic_no), 10, vary=True, min=0)
                if(kinetic_no > 0): #share by default
                    p["t"+ str(component)+"_"+str(kinetic_no)].set(expr="t"+ str(component)+"_0")
                p.add("A"+ str(component)+"_"+str(kinetic_no), 0.01, vary = True)
        
        for kinetic_no in range(len(self.all_data)):
            p.add("t0_"+str(kinetic_no), 0, vary = False)
            p.add("index_max_"+str(kinetic_no), len(self.all_data[kinetic_no].data_t)-1, vary = False)
            p.add("index_min_"+str(kinetic_no), 0, vary = False)
        return p

    def gaussExp(self, t, A, tau):
        return A * np.exp(- t / tau)
    
    def multipleGaussExp(self, t, t0, As, taus, offset): 
        #As and taus should be lists/tuples of the same length
        return_value = offset
        for i in range(len(taus)):
            return_value += self.gaussExp(t-t0, As[i], taus[i])
        return return_value 

    def residual(self, params):
        p = params.valuesdict()
        nexp = int(p["nexp"])
        
        residuals = [0]*len(self.all_data)
        res_counter = 0
        
        for kinetic_no in range(len(self.all_data)):
            index_max = int(p["index_max_"+str(kinetic_no)])
            index_min = int(p["index_min_"+str(kinetic_no)])
    
            As = []
            Taus = []
            for component in range(1,nexp+1):
                As.append(p["A"+ str(component)+"_"+str(kinetic_no)])
                Taus.append(p["t"+ str(component)+"_"+str(kinetic_no)])
            t0 = p["t0_"+str(kinetic_no)]

            y_model = self.multipleGaussExp(self.all_data[kinetic_no].data_t[index_min:index_max+1], t0, As, Taus, 0.0)
            residuals[res_counter]=np.subtract(self.all_data[kinetic_no].data_a[index_min:index_max+1], y_model)
            res_counter += 1

        res = np.hstack(residuals)
        #print(res.shape)
        return res
     
    def fitExpOptimize(self, params):
        mini = lmfit.Minimizer(self.residual, params, nan_policy='propagate')
        out = mini.leastsq()
        lmfit.report_fit(out) #robi dużo zamieszania
        print("chisquare is " + str(out.chisqr))
        self.last_chisqr = out.chisqr
        return out.params

    def fitExpOptimizePlot(self, params, dpi=100):
        p = params.valuesdict()
        nexp = int(p["nexp"])
        
        for kinetic_no in range(len(self.all_data)): 
            index_max = int(p["index_max_"+str(kinetic_no)])
            index_min = int(p["index_min_"+str(kinetic_no)]) 
            
            As = []
            Taus = []
            for component in range(1,nexp+1):
                As.append(p["A"+ str(component)+"_"+str(kinetic_no)])
                Taus.append(p["t"+ str(component)+"_"+str(kinetic_no)])
            t0 = p["t0_"+str(kinetic_no)]
            y_model = self.multipleGaussExp(self.all_data[kinetic_no].data_t[index_min:index_max+1], t0, As, Taus, 0.0)  
    
            plt.figure(figsize=(8, 6), dpi=dpi)
            plt.plot(self.all_data[kinetic_no].data_t, self.all_data[kinetic_no].data_a, 'bo')
            plt.plot(self.all_data[kinetic_no].data_t[index_min:index_max+1], y_model, 'r-')
            plt.xlabel("Time")
            plt.ylabel("Signal")
            plt.grid(True, which="major", linestyle='--')
            #plt.xticks(fontsize=12)
            #plt.yticks(fontsize=12)
            plt.title("Fit of kinetic " + self.all_data[kinetic_no].name)
            
            sum_preexps = 0.0
            for component in range(1,nexp+1):
                sum_preexps += p["A"+ str(component)+"_"+str(kinetic_no)]
            desc = ""
            for component in range(1,nexp+1):
                contribution = 100*p["A"+ str(component)+"_"+str(kinetic_no)]/sum_preexps
                desc += "tau_" + str(component) + " = {a:.2e} ({b:.1f}%)\n".format(a=p["t"+ str(component)+"_"+str(kinetic_no)], b=contribution)
            plt.figtext(0.5, 0.7, desc, fontdict={'size': 12})
            
        plt.show()

    
    def remove(self, kinetic_id):
        #remove selected kinetic from experiment
        kinetic = self[kinetic_id]
        self.all_data.remove(kinetic)
        self._renumber()
    
    def getNamesList(self):
        namelist = []
        for i in range(len(self.all_data)):
            namelist.append(self.all_data[i].name)
        return namelist
    
    def __getitem__(self, i):
        #note that when taking single item it will return reference
        #but when taking a few it will return new separate experiment with copies
        if(type(i) is str):
            for x in self.all_data:
                if(x.name == i):
                    return x
            raise IndexError("kinetic with name: " + i + " not found in experiment")
        elif(type(i) is list):
            tmp = Experiment()
            for x in i:
                tmp.all_data.append(copy.deepcopy(self[x]))
            tmp._renumber()
            return tmp
        else:
            return self.all_data[i]
        #add also posibity to call them by name
        #(if int then by number, if string then by name)
        
    def _renumber(self): #recount numbers in kinetics
        for i in range(len(self.all_data)):
            self.all_data[i].num = i

    def __iter__(self):
        self.nx = 0
        return self
        
    def __next__(self):    
        if self.nx < len(self.all_data):
            self.nx += 1
            return self.all_data[self.nx-1]
        else:
            raise StopIteration
            
#przemyśleć
#    def __add__(self, offset): #to add two experiment objects (merge them)
#        tmp = copy.deepcopy(self)
#        tmp.data_a += offset
#        return tmp          
    
    #def __setitem__
    #def __getslice__
    #def __setslice__
    #def __delslice__
