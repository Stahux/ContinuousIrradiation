import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lmfit
import copy


class LightEvent: #designates light distribution starting from some moment to the next lightevent
    def __init__(self, wavelength, intensity, start_time):
        #verify if thse values make any sense
       self.wavelength = wavelength
       self.intensity = intensity
       self.start_time = start_time
        

class KineticData:
    def __init__(self, filename, probe, irradiation, intensity = 0.0, absorbance = 0.0, t_on = None, t_off = None, probe_length = 1, irradiation_length = 1, num = None, zeroed = False, src = None):    
        
        if(src == None):
            #inport data
            data = np.loadtxt(filename)
            self.data_t = data[:,0]
            self.data_a = data[:,1]
        elif(src == "Lucas"):    #read data of meine beste freunde lucasso  !
            avg_area = 4        #!!! set the area you want to average (because in these data wavelengths are densely probed)
            dead_columns = 2    #two columns are not absorbances (column with numbering + time)
            tmp = pd.read_csv(filename)        
            columns = tmp.columns.values[dead_columns:]                    
            columns2 = columns.astype(np.float)
            positions = np.where((columns2 >= probe-avg_area/2) & (columns2 <= probe+avg_area/2))[0]  
            columns_names = [columns[pos] for pos in positions]
            selected_columns = tmp[columns_names]                    
            
            self.data_t = tmp['Time'].to_numpy() 
            self.data_a = selected_columns.mean(axis = 1, skipna = True).to_numpy()       
        else: #just generate some grid and NaNs as values, ignore filename value
            if(t_on == None): raise Exception("Please set some t_on value!")
            if(t_off == None): raise Exception("Please set some t_off value!")
            irr_time = np.abs(t_off-t_on)
            self.data_t = np.linspace(t_on-10, t_off+irr_time*5, num=10000)
            self.data_a = self.data_t * np.nan
            
        #self.light = light #later there should be array of lighevents...
        self.probe = probe #probing wavelength
        self.irradiation = irradiation
        self.intensity = intensity
        self.absorbance = absorbance #lets say for now that it is absorbance at irradiation wavelength in the ground state
        if t_on == None:
            self.t_on = self.data_t[0]
        else:
            self.t_on = t_on #before t0 there is no irradiation

        if t_off == None:
            self.t_off = self.data_t[-1]
        else:
            self.t_off = t_off #before t0 there is no irradiation
            
        self.probe_length = probe_length
        self.irradiation_length = irradiation_length
        
        self.num = num
        self.zeroed = zeroed #if true, means that before t_on absorbance was set to zero
          
    def selectTimes(self, t_min, t_max): #TODO: checl consistency, written fast
        index_min = np.argmin(np.abs(self.data_t-t_min))
        index_max = np.argmin(np.abs(self.data_t-t_max))
        self.data_t = self.data_t[index_min:index_max-1] #check again!
        self.data_a = self.data_a[index_min:index_max-1]
        if(self.t_off > self.data_t[-1]):
            self.t_off = self.data_t[-1]
    
    def plotYourself(self):
        plt.figure()
        plt.plot(self.data_t, self.data_a, 'bo')
        #plt.plot(self.data_t, residual(out2.params) + self.data_a, 'r-')
        plt.show()        

    def genParameters(self): #parameters are fixed by default, unfix some of them before fitting procedure
        params = lmfit.Parameters()
        numstring = ''
        if self.num != None:
            numstring = '_' + str(self.num) + '_'
        params.add(numstring+'probe', value=float(self.probe), vary=False, min=0)
        params.add(numstring+'irradiation', value=float(self.irradiation), vary=False, min=0) 
        params.add(numstring+'intensity', value=self.intensity, vary=False, min=0) 
        params.add(numstring+'t_on', value=self.t_on, vary=False)
        params.add(numstring+'t_off', value=self.t_off, vary=False)
        params.add(numstring+'probe_length', value=self.probe_length, vary=False, min=0) 
        params.add(numstring+'irradiation_length', value=self.irradiation_length, vary=False, min=0) 
        params.add(numstring+'absorbance', value=self.absorbance, vary=False, min=0) 
        return params
    
    def updateParameters(self, params):
        p = params.valuesdict()
        numstring = ''
        if self.num != None:
            numstring = '_' + str(self.num) + '_'
        self.probe = p[numstring+'probe']
        self.irradiation = p[numstring+'irradiation']
        self.intensity = p[numstring+'intensity']
        self.t_on = p[numstring+'t_on']
        self.t_off = p[numstring+'t_off']
        self.probe_length = p[numstring+'probe_length']
        self.irradiation_length = p[numstring+'irradiation_length']
        self.absorbance = p[numstring+'absorbance']
  
    def gaussExp(self, t, A, tau):
        eksp = np.exp(-t/tau)
        return A * eksp
    
    def multipleGaussExp(self, t, t0, As, taus, offset): #As and taus should be lists/tuples of the same length
        return_value = offset
        for i in range(len(taus)):
            return_value += self.gaussExp(t-t0, As[i], taus[i])
        return return_value 

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
            if self.data_t[i] <= self.t_off and self.t_off < self.data_t[i+1]:
                splitpoint = i
                break
        
        residuals.extend(np.subtract(self.data_a[splitpoint+1:], y_model[splitpoint+1:]))
        
        #plt.figure() #for eventual diagnostics
        #plt.plot(self.data_t[splitpoint+1:], y_model[splitpoint+1:], 'b-')
        #plt.plot(self.data_t[splitpoint+1:], self.data_a[splitpoint+1:], 'ro')
        #plt.show()        
        
        return residuals
     
    def linearBaselineCorrect(self, exp_no): #it is made to correct roughly data done without reference. it assumes linear drift during experiment time.
        #so it will make fit of decay after t_off, and find offset. then it will assume that offset should be equal to absorbance before t_on, and 
        #it will do linear correction. if exp_no == 0, then the last point at kinetic will be taken as offset
        if exp_no == 0:
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
            
            mini = lmfit.Minimizer(self.residualLBC, p, nan_policy='propagate')
            out = mini.minimize(method='leastsq')
            offset = out.params['offset']
            lmfit.report_fit(out.params)
            cfactor = (offset - self.data_a[0])/(len(self.data_a)-1)
            newdata = [(self.data_a[i] - cfactor*i) for i in range(len(self.data_a))]
            self.data_a = newdata            
            
            
class Experiment: #container for kineticdata or other future data types prepared to fit globally
    def __init__(self):
        self.all_data = list()
        self.count = 0
        
    def loadKineticData(self, filename, probe, irradiation, intensity = 0.0, absorbance = 0.0, t_on = None, t_off = None, probe_length = 1, irradiation_length = 1, zeroed = False, src = None):
        newdata = KineticData(filename, probe, irradiation, intensity, absorbance, t_on, t_off, probe_length, irradiation_length, self.count, zeroed, src = src)
        self.all_data.append(newdata)
        self.count += 1
        
    def addKineticData(self, kineticdata):
        kineticdata.num = self.count
        self.all_data.append(kineticdata)
        self.count += 1
        
    def genParameters(self):
        generated_params = lmfit.Parameters()
        for i in range(self.count):
            generated_params += self.all_data[i].genParameters()
        return generated_params
        
    def updateParameters(self, params): 
        for i in range(self.count):
            self.all_data[i].updateParameters(params)
 
    def plotYourself(self, num = None, dpi=150):
        plt.figure(dpi=dpi)
        
        if num == None:
            for i in range(self.count):
                plt.plot(self.all_data[i].data_t, self.all_data[i].data_a, 'b-')
        else:
            plt.plot(self.all_data[num].data_t, self.all_data[num].data_a, 'b-')
                
        plt.show()

    def linearBaselineCorrect(self, exp_no):
        for i in range(self.count):
            self.all_data[i].linearBaselineCorrect(exp_no)

