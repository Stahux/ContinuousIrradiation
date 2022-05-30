"""
Implements spectrum class, it is not used in modelling directly,
only to load epsilons.
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


class SpectrumData():
    def __init__(self, spectra = None):
        #todo ensure that w and a are always in numpy format, not list
        if(spectra == None):
            self.w = np.nan
            self.a = np.nan
        elif(type(spectra) == list): #cos czuje ze to nie zadziala ale dobra, mniejsza
            self.w = sum([x.w for x in spectra])/len(spectra)
            self.a = sum([x.a for x in spectra])/len(spectra)
        elif(type(spectra) == str):
            self.load(spectra)
        else:
            self.w = spectra.w
            self.a = spectra.a
        
        self.delay = None
        self.label = None
    
    def load(self, filename, skip_header=0, skip_footer=0):
        tmp = np.genfromtxt(filename, skip_header=skip_header, skip_footer=skip_footer)
        self.w = tmp[:,0]
        self.a = tmp[:,1]
        
    def save(self, filename, delimiter=None):
        if(delimiter == None):
            np.savetxt(filename, np.transpose(np.vstack([self.w,self.a])))
        else:
            np.savetxt(filename, np.transpose(np.vstack([self.w,self.a])), delimiter=delimiter)
        
    def plotTogether(spectra, x_min = None, x_max = None, y_min = None, y_max = None, dpi = None, title = None):
        if(type(spectra) != list):
            spectra = [spectra,]
            
        if(dpi == None):
            dpi=80
            
        colors = ("b","r","g","c","m","y","C0","C1","C2","C3","C4","C5","C6","C7") * 100  #color order
        description = ""
       
        plt.figure(figsize=(8, 6), dpi=dpi)
        plt.grid(True, linestyle='--')
        count = 0
        for spectrum in spectra:
            if(spectrum.delay != None):
                label = "%.3i" % spectrum.delay+" ps"
            elif(spectrum.label != None):
                label = spectrum.label
            else:
                label = ""
            
            plt.plot(spectrum.w, spectrum.a, colors[count]+'o-', markersize=2, alpha=0.5, label=label)
            count += 1
            
        plt.xlabel("Wavelength (nm)", fontdict={'size': 16})
        plt.ylabel("\u0394A", fontdict={'size': 16})
        plt.figtext(0.14, 0.15, description, fontdict={'size': 18})
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim(left = x_min, right = x_max)
        plt.ylim(bottom = y_min, top = y_max)
        if(title != None):
            plt.title(title, fontdict={'size': 20}, loc="right")
        plt.legend(shadow=False, prop={'size': 11}, fontsize="small", labelspacing=0.1)
        plt.show() 
    
    def subtractRayleigh(self, amp = 1):
        self.a -= amp*0.1*self.w**(-4)/400**(-4)

    def removeWavelengths(self, x_min = -np.inf, x_max = np.inf): #remove in selected w range
        tmpw = self.w
        tmpa = self.a
        
        filterit = ~((self.w > x_min) & (self.w < x_max))
        
        self.w = tmpw[filterit]
        self.a = tmpa[filterit]

    def __getitem__(self, w): #return value at wavelengh in a cool way
        return self.a[np.argmin(np.abs(self.w - w))]
        
    def zeroAt(self, w):
        self.a -= self[w]   
        
    def maximum(self):
        return self.w[np.argmax(self.a)]

    def minimum(self):
        return self.w[np.argmin(self.a)]

    def normAt(self, w):
        index = np.argmin(np.abs(self.w - w))
        if(self.a[index] < 0):
            self.a = - self.a / self.a[index]
        elif(self.a[index] > 0):
            self.a = self.a / self.a[index]
        
    def __sub__(self, o):
        tmp = copy.deepcopy(self)
        tmp.a = tmp.a - o.a #assumes same grid
        return tmp
        
    def __add__(self, o):
        tmp = copy.deepcopy(self)
        tmp.a = tmp.a + o.a #assumes same grid
        return tmp

    def __mul__(self, o):
        tmp = copy.deepcopy(self)
        tmp.a = tmp.a * o #assumes o is number
        return tmp

    def __neg__(self):
        tmp = copy.deepcopy(self)
        tmp.a = -tmp.a
        return tmp

    def __truediv__(self, o):
        tmp = copy.deepcopy(self)
        tmp.a = tmp.a / o #assumes o is number
        return tmp
    
    
    