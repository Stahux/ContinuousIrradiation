#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("../../application/")

from model import Model

#this file is only to convert one model into another one, which will have temperature dependent rates.

#test1 = Model()
test1 = Model.load('model_ABC_singlephoton.model') #load normal model with fixed k arrows
#test1.manualModelBuild()

test1.setInitial("A",1) #set initial conditions
test1.setInitial("B",0)
test1.setInitial("C",0)
test1.genParameters().pretty_print() #params print

test1.turnKsIntoEyrings() #turn fixed k arrows to eyring (T dependent)

print("\n")
test1.genParameters().pretty_print() #print params with eyring


test1.save('model_ABC_singlephoton_eyring.model') #save new model