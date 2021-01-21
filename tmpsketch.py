#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:44:28 2019

@author: staszek
"""

from model import Model
from kineticdata import KineticData, Experiment
from modfit import ModFit
import numpy as np
import matplotlib.pyplot as plt
import lmfit
import copy
import random


#test1 = Model()
test1 = Model.load('model_NP2P.model')
test1.manualModelBuild()
#test1.save('model_NP2P.model')