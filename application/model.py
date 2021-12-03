"""
Implements Model class, which intention is similar to lmfit.Model.
However, this Model class is built separately from the scratch and
applies only to ModFit (has nothing to do with lmfit.Minimizer).
"""
__version__ = "v0.2"
__authors__ = ["Stanisław Niziński"]
#__authors__.append("Add yourself my friend...")

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sys
import keyword
import copy
import math
import pickle
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from kineticdata import Experiment
from modfit import MParameter, MParameters

#firstly create populations and put them i model
#then add arrows. Create arrow binded by two populations and add to model list
#arrow should add itself to both end-populations
#remove arrow by calling kill() in this arrow and removing from model list
#arrow should remove itself from both end-populations
#if you kill population, firstly remove all arrows inside



def isIdentifier(identifier): #check if string is valid python identifier
    if(not(isinstance(identifier, str))):
        return False
    if(not(identifier.isidentifier())):
        return False
    if(keyword.iskeyword(identifier)):
        return False
    return True

class ModPopulation:
    def __init__(self, new_name):
        self.arrows = list() #list of processes associated with this population
        self.name = new_name #name of the population
        self.epsilon = dict() # define epsilons for irr and probe wavelengths
        self.initial = 0.0
        
        self.rect_w = 120
        self.rect_h = 40
        self.rect = QtCore.QRect(150, 20, self.rect_w, self.rect_h)
        
    def remove(self, model): 
        #before calling ensure that all arrows with this population are removed,
        #in other case there will be exception!
        if(len(self.arrows) != 0):
            raise Exception("Attempted to invalid population removal!!")
        n = model.populations.index(self)
        model.populations.pop(n)
        
    def countProcesses(self, second_population):
        #gives numer of the existing arrows between populations
        arrows = 0
        for arr in self.arrows:
            if( ( arr.source is self and arr.target is second_population ) or 
               ( arr.source is second_population and arr.target is self ) ):
                arrows += 1
        return arrows        

class ModProcess:
    def __init__(self, new_name, pop_source, pop_target):
        #firstly check how many parallel arrows already there
        arrow_count = pop_source.countProcesses(pop_target) 
        self.name = new_name #name of the process
        self.source = pop_source #initialize yourself with both neighbour populations
        self.target = pop_target
        self.source.arrows.append(self) #initialize neighbour populations with yourself
        self.target.arrows.append(self)
        self.p1 = QtCore.QPoint()
        self.p2 = QtCore.QPoint()
        self.type = ""
        #number of arrow between some pair of populations.numbering helps to render arrows separately
        self.number = arrow_count + 1 
        self.displacement = 14 #separation distance between arrows
        #if distance from point to crossing point is below treshold function contains return true
        self.dist_treshold = 7 
    
    def remove(self, model): #removes arrow from neighbouring populations and model
        n1 = self.source.arrows.index(self)
        self.source.arrows.pop(n1)
        n2 = self.target.arrows.index(self)
        self.target.arrows.pop(n2) #here you have to recount arrows between populations...
        count = 1
        for arr in self.source.arrows:
            if( arr.source is self.target or arr.target is self.target ):
                arr.number = count
                count += 1
        n3 = model.processes.index(self)
        model.processes.pop(n3)
        
    def getsetLocation(self):
        p1 = QtCore.QPointF(self.source.rect.x()+self.source.rect.width()/2, 
                            self.source.rect.y()+self.source.rect.height()/2)
        p2 = QtCore.QPointF(self.target.rect.x()+self.target.rect.width()/2, 
                            self.target.rect.y()+self.target.rect.height()/2)
        
        diff = p2 - p1 # just make arrow shorter....
        #uzaleznij odjecie od kata...
        correction = abs(diff.x() / math.sqrt(diff.x()*diff.x() + diff.y()*diff.y())) 
        to_substr = (40 * correction + 28) * diff / math.sqrt(diff.x()*diff.x() + diff.y()*diff.y())
        p1 = p1 + to_substr
        p2 = p2 - to_substr
        
        diff = p1 - p2
        difflen = math.sqrt(diff.x()**2 + diff.y()**2)
        #move second, third, ... arrow a little bit to avoid overlap
        if(self.number > 1 and difflen != 0.0):
            sinkat = diff.y() / difflen
            coskat = diff.x() / difflen
            if(sinkat < 0):
                alittle = QtCore.QPointF(sinkat * self.displacement, -coskat * self.displacement)
            elif(sinkat > 0):
                alittle = QtCore.QPointF(-sinkat * self.displacement, coskat * self.displacement)
            elif(coskat > 0):
                alittle = QtCore.QPointF(sinkat * self.displacement, -coskat * self.displacement)
            else:
                alittle = QtCore.QPointF(-sinkat * self.displacement, coskat * self.displacement)
            
            p1 += (-1)**self.number * alittle * math.floor(self.number / 2.0)
            p2 += (-1)**self.number * alittle * math.floor(self.number / 2.0)
        
        self.p1 = p1
        self.p2 = p2
        return (p1, p2)
        
    def contains(self, point):
        #find linear eq for p1 and p2
        a_p1p2 = float(self.p2.y() - self.p1.y()) / float(self.p2.x() - self.p1.x())
        b_p1p2 = float(self.p1.y()) - a_p1p2 * float(self.p1.x())
        
        a_point = -1 / a_p1p2 #find linear eq for point which is perpendicular to p1p2
        b_point = float(point.y()) - a_point * float(point.x())
        
        x_cross = (b_point - b_p1p2) / (a_p1p2 - a_point) #find crossing point
        y_cross = a_p1p2 * x_cross + b_p1p2
        
        if(self.p1.x() >= self.p2.x()): #check if crossing point is between p1 and p2
            if(x_cross <= self.p1.x() and x_cross >= self.p2.x()):
                cond1 = True
            else:
                cond1 = False
        else:
            if(x_cross <= self.p2.x() and x_cross >= self.p1.x()):
                cond1 = True
            else:
                cond1 = False  
                
        if(self.p1.y() >= self.p2.y()): #check if crossing point is between p1 and p2
            if(y_cross <= self.p1.y() and y_cross >= self.p2.y()):
                cond2 = True
            else:
                cond2 = False
        else:
            if(y_cross <= self.p2.y() and y_cross >= self.p1.y()):
                cond2 = True
            else:
                cond2 = False                 
            
        
        dist = math.sqrt(math.pow(float(point.x()) - x_cross,2)+math.pow(float(point.y()) - y_cross,2)) 
        if(dist <= self.dist_treshold):
            cond3 = True
        else:
            cond3 = False
        
        return (cond1 and cond2 and cond3)

class ModThermal(ModProcess):
    def __init__(self, new_name, pop_source, pop_target):
        super().__init__(new_name, pop_source, pop_target)
        self.k = 0
        self.type = "k"
        
    def paintYourself(self, painter):
        p1, p2 = self.getsetLocation()
        
        #firstly draw sinusiodal shape indicating nonradiative process
        fragm_len = 3.0
        modamp = 5.0 #depth of modulation
        diff = p2 - p1
        full_length = math.sqrt(diff.x()*diff.x() + diff.y()*diff.y())
        iters = math.floor(full_length / fragm_len)
        unit_vect = diff * fragm_len / full_length #piece of line used to render whole curve
        if(diff.x() >= 0):
            angle = math.asin(diff.y() / math.sqrt(diff.x()*diff.x() + diff.y()*diff.y()))
        else:
            angle = math.pi - math.asin(diff.y() / math.sqrt(diff.x()*diff.x() + diff.y()*diff.y()))
        perp_vect = QtCore.QPointF(modamp*math.cos(angle+math.pi/2), 
                                   modamp*math.sin(angle+math.pi/2))
        
        path = QtGui.QPainterPath(p1)
        
        for i in range(1,iters+1):
            path.lineTo(p1 + unit_vect * i + perp_vect * math.sin(i * math.pi / iters) * math.sin(i * 1))
        
        path.lineTo(p2)
        painter.drawPath(path)
            
        diff = p1 - p2 #potrzebne do zrobienia grota strzalki  
        if(diff.x() >= 0):
            angle = math.asin(diff.y() / math.sqrt(diff.x()*diff.x() + diff.y()*diff.y()))
        else:
            angle = math.pi - math.asin(diff.y() / math.sqrt(diff.x()*diff.x() + diff.y()*diff.y()))
        
        angle_diff = math.pi / 10.0 #determines shape of the arrow
        length = 10.0 #determines shape of the arrow
        p_arr1 = QtCore.QPointF(length*math.cos(angle+angle_diff), 
                                length*math.sin(angle+angle_diff))
        p_arr2 = QtCore.QPointF(length*math.cos(angle-angle_diff), 
                                length*math.sin(angle-angle_diff))
        
        painter.drawLine(p2, p2 + p_arr1)
        painter.drawLine(p2, p2 + p_arr2)
        
class ModThermalEyring(ModProcess):
    #extension of ModThermal, where k is not const. but depends on temperature
    def __init__(self, new_name, pop_source, pop_target):
        super().__init__(new_name, pop_source, pop_target)
        self.k = None #is recalculated after calling getK()
        self.type = "ke"
        self.kappa = 1 #transmission coefficient, no unit
        self.deltaH = 10 #enthalpy of activation (with two plusses in the upper index), [kcal/mol]
        self.deltaS = 0.001 #entropy of activation (with two plusses in the upper index) [kcal/mol/K]

    def getK(self, temperature): 
        #calc rate constant based on params and temp. (absolute not Celcius!!!)
        R_gas = 0.00198720425864083 #gas constant [kcal/K/mol]
        h_planck = 6.62607015E-34 #planck [J*s]
        kb = 1.380649E-23 #boltzmann constant [J/K]
        self.k = (self.kappa*kb*temperature/h_planck) * np.exp(self.deltaS/R_gas \
                 - self.deltaH/(R_gas*temperature))
        return self.k

    def paintYourself(self, painter):
        p1, p2 = self.getsetLocation()
        
        #firstly draw sinusiodal shape indicating nonradiative process
        fragm_len = 3.0
        modamp = 5.0 #depth of modulation
        diff = p2 - p1
        full_length = math.sqrt(diff.x()*diff.x() + diff.y()*diff.y())
        iters = math.floor(full_length / fragm_len)
        unit_vect = diff * fragm_len / full_length #piece of line used to render whole curve
        if(diff.x() >= 0):
            angle = math.asin(diff.y() / math.sqrt(diff.x()*diff.x() + diff.y()*diff.y()))
        else:
            angle = math.pi - math.asin(diff.y() / math.sqrt(diff.x()*diff.x() + diff.y()*diff.y()))
        perp_vect = QtCore.QPointF(modamp*math.cos(angle+math.pi/2), 
                                   modamp*math.sin(angle+math.pi/2))
        
        path = QtGui.QPainterPath(p1)
        
        for i in range(1,iters+1):
            path.lineTo(p1 + unit_vect * i + perp_vect * math.sin(i * math.pi / iters) * math.sin(i * 1))
        
        path.lineTo(p2)
        painter.drawPath(path)
            
        diff = p1 - p2 #potrzebne do zrobienia grota strzalki  
        if(diff.x() >= 0):
            angle = math.asin(diff.y() / math.sqrt(diff.x()*diff.x() + diff.y()*diff.y()))
        else:
            angle = math.pi - math.asin(diff.y() / math.sqrt(diff.x()*diff.x() + diff.y()*diff.y()))
        
        angle_diff = math.pi / 10.0 #determines shape of the arrow
        length = 10.0 #determines shape of the arrow
        p_arr1 = QtCore.QPointF(length*math.cos(angle+angle_diff), 
                                length*math.sin(angle+angle_diff))
        p_arr2 = QtCore.QPointF(length*math.cos(angle-angle_diff), 
                                length*math.sin(angle-angle_diff))
        
        painter.drawLine(p2, p2 + p_arr1)
        painter.drawLine(p2, p2 + p_arr2)        
        
class ModRadiative(ModProcess):
    def __init__(self, new_name, pop_source, pop_target):
        super().__init__(new_name, pop_source, pop_target)    
        self.fi = 0  
        self.type = "fi"
        
    def paintYourself(self, painter):
        p1, p2 = self.getsetLocation()
            
        diff = p1 - p2 #tym razem potrzebne do zrobienia grota strzalki
        
        if(diff.x() >= 0):
            angle = math.asin(diff.y() / math.sqrt(diff.x()*diff.x() + diff.y()*diff.y()))
        else:
            angle = math.pi - math.asin(diff.y() / math.sqrt(diff.x()*diff.x() + diff.y()*diff.y()))
        
        angle_diff = 3.14 / 10.0
        length = 10.0
        p_arr1 = QtCore.QPointF(length*math.cos(angle+angle_diff),
                                length*math.sin(angle+angle_diff))
        p_arr2 = QtCore.QPointF(length*math.cos(angle-angle_diff),
                                length*math.sin(angle-angle_diff))
        
        painter.drawLine(p1, p2)
        painter.drawLine(p2, p2 + p_arr1)
        painter.drawLine(p2, p2 + p_arr2)

class ModelWindow(QWidget):

    def __init__(self, model_ref):
        super().__init__()
        self.model = model_ref
        self.title = "Model Editor"
        self.left = 10
        self.top = 35
        self.width = 800
        self.height = 600

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.show()
        
        ## Create some widgets to be placed inside
        self.label1 = QtWidgets.QLabel("Population name:", self)
        self.text1 = QtWidgets.QLineEdit("", self)
        self.button1 = QtWidgets.QPushButton("Add population", self)
        self.button1.clicked.connect(self.button1Func)
        self.label5 = QtWidgets.QLabel("Process name:", self)
        self.text5 = QtWidgets.QLineEdit("", self)
        self.select_process = QtWidgets.QComboBox(self)
        self.select_process.addItem("Thermal")
        self.select_process.addItem("Light activated")
        self.button5 = QtWidgets.QPushButton("Add process", self)
        self.button5.clicked.connect(self.button5Func)
        self.label1.show()
        self.text1.show()
        self.button1.show()
        self.label5.show()
        self.text5.show()
        self.select_process.show()
        self.button5.show()        

        self.installEventFilter(self)

        #indicates which population is being moved, or False if none
        self.mousepressed = False 
        #markers of relative mouse move from the moment of click
        self.mouse_dx = 0 
        self.mouse_dy = 0
        #marker of mouse position when population was clicked
        self.ref_mouse = QtCore.QPoint(0,0) 
        
        self.pop_edit = False #to edit populatiions
        self.label2 = QtWidgets.QLabel("", self)
        self.eps_table = QtWidgets.QTableWidget(self)
        self.eps_table.setColumnCount(2)
        self.eps_table.setColumnWidth(0,49);
        self.eps_table.setColumnWidth(1,49);
        self.eps_table.setHorizontalHeaderLabels(("lambda", "epsilon"));
        self.eps_table.verticalHeader().setVisible(False);
        self.button2 = QtWidgets.QPushButton("Add eps", self)
        self.button2.clicked.connect(self.button2Func)        
        self.button3 = QtWidgets.QPushButton("Save!", self)
        self.button3.clicked.connect(self.button3Func)
        self.button31 = QtWidgets.QPushButton("Delete", self)
        self.button31.clicked.connect(self.button31Func)    
        
        self.proc_edit = False #to edit processes
        self.text_proc = QtWidgets.QLineEdit("", self) 
        self.button4 = QtWidgets.QPushButton("Save!", self)
        self.button4.clicked.connect(self.button4Func)
        self.button41 = QtWidgets.QPushButton("Delete", self)
        self.button41.clicked.connect(self.button41Func)
        
        #indicates that process arrow is being added (select populations)
        self.process_adding = False

    def button1Func(self): #creates new population
        found = False #ensure that new name is unique
        for elem in self.model.populations:
            if(elem.name == self.text1.text()):
                found = True
                
        for elem in self.model.processes:
            if(elem.name == self.text1.text()):
                found = True
                
        if(not(isIdentifier(self.text1.text()))): #it has to be a valid python id
            found = True
            
        if(found == False and len(self.text1.text()) > 0):
            self.model.addPopulation(ModPopulation(self.text1.text()))
            self.text1.setText("")
            self.repaint()

    def button2Func(self): #adds new epsilon entry to population
        if(self.pop_edit != False):
            added_row = self.eps_table.rowCount()
            self.eps_table.setRowCount(added_row + 1)
            tmp1 = QtWidgets.QTableWidgetItem("")
            tmp2 = QtWidgets.QTableWidgetItem("")
            #set flags?
            self.eps_table.setItem(added_row,0,tmp1)
            self.eps_table.setItem(added_row,1,tmp2)
        
    def button3Func(self): #saves epsilons to population
        #in future ensure that values are not rounded during this process (dict->txt->dict)
        num_rows = self.eps_table.rowCount() 
        new_dict = dict()
        for row in range(num_rows):
            tmp_item1 = self.eps_table.item(row, 0).text()
            tmp_item2 = self.eps_table.item(row, 1).text()
            if(not(self.isStrNumber(tmp_item1) and self.isStrNumber(tmp_item2))):
                continue
            new_dict[float(tmp_item1)] = float(tmp_item2)
        self.pop_edit.epsilon = new_dict
        self.pop_edit = False
        self.repaint()
        
    def button31Func(self): #deletes population if possible
        if(len(self.pop_edit.arrows) == 0):
            self.pop_edit.remove(self.model)
            self.pop_edit = False
        self.repaint()
    
    def button4Func(self): #finished edition of process
        if(self.isStrNumber(self.text_proc.text())):
            if(self.proc_edit.type == "k"):
                self.proc_edit.k = float(self.text_proc.text())
            elif(self.proc_edit.type == "fi"):
                self.proc_edit.fi = float(self.text_proc.text())    
            self.proc_edit = False
            self.text_proc.setText("")
        self.repaint()
            
    def button41Func(self): #deletes arrow
        self.proc_edit.remove(self.model)
        self.proc_edit = False
        self.text_proc.setText("")
        self.repaint()
            
    def button5Func(self): #adds process and starts selection of connected populations
        found = False #ensure that new name is unique
        for elem in self.model.processes:
            if(elem.name == self.text5.text()):
                found = True
            
        for elem in self.model.populations:
            if(elem.name == self.text5.text()):
                found = True
                
        if(not(isIdentifier(self.text5.text()))): #it has to be a valid python id
            found = True
            
        if(found == False and len(self.text5.text()) > 0):
            self.process_adding = True
            self.repaint()
        
    def isStrNumber(self,s):
        try:
            float(s)
            return True
        except ValueError:
            return False    
        
    def paintEvent(self,event): 
        self.label1.setGeometry(10, 10, 100, 25) #standard gui configuration
        self.text1.setGeometry(10, 40, 100, 25)
        self.button1.setGeometry(10, 70, 100, 30)
        self.label5.setGeometry(10, 10+100, 100, 25)
        self.text5.setGeometry(10, 40+100, 100, 25)
        self.select_process.setGeometry(10, 70+100, 100, 25)
        self.button5.setGeometry(10, 70+130, 100, 30)        
        self.label2.setGeometry(10, 100+150, 100, 25)
        self.eps_table.setGeometry(10, 130+150, 100, 200)
        self.button2.setGeometry(10, 350+150, 50, 28)
        self.button3.setGeometry(10, 380+150, 100, 30)
        self.button31.setGeometry(10, 410+150, 100, 30)
        self.text_proc.setGeometry(10, 130+150, 100, 25)
        self.button4.setGeometry(10, 160+150, 100, 30)
        self.button41.setGeometry(10, 190+150, 100, 30)
        
        if(self.pop_edit == False): #population edit menu
            self.label2.setVisible(False)
            self.eps_table.setVisible(False)
            self.button2.setVisible(False)
            self.button3.setVisible(False)
            self.button31.setVisible(False)
        else:
            self.label2.show()
            self.eps_table.show()
            self.button2.show()
            self.button3.show()
            self.button31.show()
            
        if(self.proc_edit == False):
            self.button4.setVisible(False)
            self.button41.setVisible(False)
            self.text_proc.setVisible(False)
        else:
            self.button4.show()
            self.button41.show()
            self.text_proc.show()   
            self.label2.show()
            
        
        painter = QtGui.QPainter(self)
        painter.begin(self)
        
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        #painter.setWorldMatrixEnabled(False)
        
        rmpen = QtGui.QPen()
        rmpen.setWidth(3)
        colorum = QtGui.QColor(100,200,0)
        if(self.process_adding != False):
            colorum = QtGui.QColor(0,200,255)
        rmpen.setBrush(colorum)
        painter.setPen(rmpen)
        marg = 5
        
        #should also call paintYourself function of the population, 
        #just like in case of processss
        for r in self.model.populations: 
            if(self.mousepressed is r):
                tmprect = copy.deepcopy(r.rect)
                tmprect.setX(tmprect.x() + self.mouse_dx)
                tmprect.setY(tmprect.y() + self.mouse_dy)
                tmprect.setWidth(r.rect_w)
                tmprect.setHeight(r.rect_h)  
            else:
                tmprect = r.rect   
            painter.drawRoundedRect(tmprect,10,10)
            painter.drawText(tmprect.x()+marg, tmprect.y()+marg, tmprect.width()-2*marg, 
                             tmprect.height()-2*marg, QtCore.Qt.AlignCenter, r.name)

        colorum = QtGui.QColor(200, 100, 0)
        rmpen.setBrush(colorum)
        painter.setPen(rmpen)
        
        for r in self.model.processes:
            r.paintYourself(painter)

        painter.end()
        
    def eventFilter(self, obj, event):
        #TODO: change and check type to isinstance!
        if(event.type() == QtCore.QEvent.MouseMove):
            mouse_position = event.pos()
            if(self.mousepressed != False):
                self.mouse_dx = mouse_position.x() - self.ref_mouse.x()
                self.mouse_dy = mouse_position.y() - self.ref_mouse.y()
                self.repaint()
            
        elif(event.type() == QtCore.QEvent.MouseButtonRelease):
            if(self.mousepressed != False):
                self.mousepressed.rect.setX(self.mousepressed.rect.x() + self.mouse_dx)
                self.mousepressed.rect.setY(self.mousepressed.rect.y() + self.mouse_dy)
                self.mousepressed.rect.setWidth(self.mousepressed.rect_w)
                self.mousepressed.rect.setHeight(self.mousepressed.rect_h)
                self.mouse_dx = 0
                self.mouse_dy = 0
                self.mousepressed = False
                self.repaint()
        elif(event.type() == QtCore.QEvent.MouseButtonPress):
            for r in self.model.populations:
                if(r.rect.contains(event.pos())):
                    self.mousepressed = r
                    self.ref_mouse = event.pos()
                    break
                    
            #finalize arrow adding process
            if(self.process_adding != True and self.process_adding != False):
                for r in self.model.populations:
                    found = False #if user clicked something useful
                    if(r.rect.contains(event.pos())):
                        if(not(r is self.process_adding)):
                            found = r
                            break
                if(found == False):
                    self.process_adding = False
                else: #finalize arrow from  self.process_adding to found
                    if(self.select_process.currentIndex() == 0): # Thermal
                        tmp_arrow = ModThermal(self.text5.text(), self.process_adding, found)
                    elif(self.select_process.currentIndex() == 1): # Light activated
                        #asumes, that there are only 2 kinds of arrows, which can be added!
                        tmp_arrow = ModRadiative(self.text5.text(), self.process_adding, found) 
                    self.model.addProcess(tmp_arrow)
                    self.text5.setText("")     
                    self.process_adding = False

            if(self.process_adding == True): #process arrow adding, first popul needs to be selected
                for r in self.model.populations:
                    found = False #if user clicked something useful
                    if(r.rect.contains(event.pos())):
                        found = True
                        self.process_adding = r
                        break
                if(found == False):
                    self.process_adding = False
                
            self.repaint()
        
        elif(event.type() == QtCore.QEvent.MouseButtonDblClick):
            if(self.proc_edit == False and self.pop_edit == False): #should edit process?
                for p in self.model.processes:
                    if(p.contains(event.pos())):
                        self.proc_edit = p
                        self.label2.setText("Edit " + p.type + " for " + p.name + " :")
                        if(p.type == "k"):
                            self.text_proc.setText(str(p.k))
                        elif(p.type == "fi"):
                            self.text_proc.setText(str(p.fi))  
                        break
            if(self.pop_edit == False and self.proc_edit == False): #should edit population?
                for r in self.model.populations:
                    if(r.rect.contains(event.pos())):
                        self.pop_edit = r
                        self.label2.setText("Edit eps for " + r.name + " :")
                        self.eps_table.setRowCount(len(r.epsilon))
                        ct_tmp = 0
                        for k, v in r.epsilon.items():
                            tmp1 = QtWidgets.QTableWidgetItem(str(k))
                            tmp2 = QtWidgets.QTableWidgetItem(str(v))
                            #set flags?
                            self.eps_table.setItem(ct_tmp,0,tmp1)
                            self.eps_table.setItem(ct_tmp,1,tmp2)
                            ct_tmp += 1
                        self.repaint()
                        break

        return False
        
class Model:
    def __init__(self):    
        self.populations = list()
        self.processes = list()
        self.psplit = False
        
    def addPopulation(self, new_population):
        self.populations.append(new_population)
        
    def addProcess(self, new_process):
        self.processes.append(new_process)
        
    def countProcesses(self, population1, population2):
        #gives numer of the existing arrows between populations
        arrows = 0
        for arr in population1.arrows:
            if( ( arr.source is population1 and arr.target is population2 ) or 
               ( arr.source is population2 and arr.target is population1 ) ):
                arrows += 1
        return arrows        

    def manualModelBuild(self):
        app = QApplication(sys.argv)
        ex = ModelWindow(self)
        app.exec_()
        
    def turnKsIntoEyrings(self):
        #simple helper func to replace constant k thermal arrows with Ering ones
        for arr in self.processes:
            if(arr.type == "k"):
                old_name = arr.name
                source = arr.source
                target = arr.target
                arr.remove(self)
                
                tmp_arrow = ModThermalEyring(old_name, source, target)
                self.addProcess(tmp_arrow) 
        
    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        
    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            loaded = pickle.load(f)
        return loaded
        
    def setInitial(self, name, value): #set some initial population/s manually
        #if you give value as array there will be differnt initial populations for each kinetic
        #however in this case number of kinetics must match number of values in array
        if(isinstance(value,list) or isinstance(value,np.ndarray)): 
            self.psplit = True                                
        elif(isinstance(value,int) or isinstance(value,float)):
            self.psplit = False
        else:                           
            raise Exception("Values given in setInitial() are not correct!")
        found = False
        for elem in self.populations:            
            if(elem.name == name):
                elem.initial = value
                found = True
                break
        if(found is False):
            raise Exception("Cannot find and set initial for requested population!")
        #before fit program must check if all populations have initials of the 
        #same length. if not then exception!
    
    def configureInitial(self): #set some initial populations for populations
        self.psplit = False
        #somehow initial conditions also depend on model (their structure), 
        #however they are not directly determined by model 
        for elem in self.populations: 
            is_sourcethermstate = False
            for arr in elem.arrows:
                if(arr.source is elem and ( arr.type == "k" or arr.type == "ke")):
                    is_sourcethermstate = True
            if(is_sourcethermstate == True):
                elem.initial = 0.0
            else:
                elem.initial = 1.0   
                        
        init_count = 0
        for x in self.populations:
            if(x.initial != 0.0):
                init_count += 1
        if(init_count != 1):
            raise Exception("Cannot determine ground state of the model! Change model"
                            ", or set initial conditions manually (setInitial() function)!")
    
    def genParameters(self):
        #parameters are fixed by default, unfix some of them before fitting procedure
        params = MParameters()
        
        for elem in self.populations:
            for l, eps in elem.epsilon.items():
                params.add(MParameter(elem.name + "__" + str(l).replace(".","_"), 
                                      value=eps, vary=False))
            
        for elem in self.processes:
            if(elem.type == "fi"):
                params.add(MParameter(elem.name + "__fi", value=elem.fi, 
                                      vary=False, min=0, max=1))
                #well, theoretically can be > 1, but not in this meaning i think
            elif(elem.type == "k"):
                params.add(MParameter(elem.name + "__k", value=elem.k, 
                                      vary=False, min=0))   
            elif(elem.type == "ke"): #temperature-dependent rate constant!
                params.add(MParameter(elem.name + "__deltaH", value=elem.deltaH, 
                                      vary=False)) 
                #negative energy barrier makes no sense, but may happen from data noise
                params.add(MParameter(elem.name + "__deltaS", value=elem.deltaS, 
                                      vary=False)) 
                
        if(self.psplit == False):  
            #check if soeme initial populations are determined if no, then do it yourself
            max_initial = 0.0    
            for elem in self.populations:    
                if(elem.initial > max_initial):
                    max_initial = elem.initial
                
            if(max_initial == 0.0):
                self.configureInitial()
                
            #somehow initial conditions also depend on model, however they are 
            #not directly determined by model 
            for elem in self.populations: 
                #initialize only "ground states" as initial states
                params.add(MParameter(elem.name, value=elem.initial, 
                                      vary=False, min=0, max=1)) 
            
        #if there is multi initial populations mode (self.psplit == true),
        #check if all vectors are the same, they must be the same    
        else: 
            checklen = None
            for elem in self.populations:
                if(not(isinstance(elem.initial,list) or isinstance(elem.initial,np.ndarray))):
                    raise Exception("Initial population vectors are not properly set! "
                                    "You have to set them all manually!!!")
                if(checklen is None): checklen = len(elem.initial)
                if(checklen != len(elem.initial)):
                    raise Exception("Initial population vectors are not properly set! "
                                    "They all must have the same length, and you "
                                    "have to set them all manually!!!")
            
            for elem in self.populations:
                for num in range(len(elem.initial)):
                    #initialize only "ground states" as initial states
                    params.add(MParameter("_" + str(num) + "_" + elem.name, 
                                          value=elem.initial[num], 
                                          vary=False, min=0, max=1)) 
            
        return params
        
    def updateParameters(self, params):
    #updates values of existing parameters. does not add new values,
    #does not modify model structure
        p = params.valuesdict()
        for elem in self.populations:
            
            if(self.psplit == False):
                elem.initial = p[elem.name]
            else:
                for num in range(len(elem.initial)):
                    elem.initial[num] = p["_" + str(num) + "_" + elem.name]
            
            for l, eps in elem.epsilon.items():
                elem.epsilon[l] = p[elem.name + "__" + str(l).replace(".","_")]
            
        for elem in self.processes:
            if(elem.type == "fi"):
                elem.fi = p[elem.name + "__fi"]    
            elif(elem.type == "k"):
                elem.k = p[elem.name + "__k"]     
            elif(elem.type == "ke"): #temperature-dependent rate constant!
                #negative energy barrier makes no sense!
                elem.deltaH = p[elem.name + "__deltaH"] 
                elem.deltaS = p[elem.name + "__deltaS"] 
        
    def checkParams(self, experiment):
        #run tests if params are correctly set. 
        #experiment object is to validiate its compatibility with model
        #it should be run after updataParameers for both model and experiment, 
        #and assume that funcs loaded them correctly
        result = True
        #OK, there is PROBLEM. it was supposed to check params which are alrady 
        #loaded to model and experiment   
        #but i wrote like check of params object. and realized that params 
        #itself should also have check func, thbk about it ....               
        return True 
        
        p = params.valuesdict() 
        try:    
            if(self.psplit == False):  
                pop_sum = 0
                for elem in self.populations:
                    pop_sum += p[elem.name]
                if(pop_sum != 1):
                    result = False
                    print("Initial population values does not sum to unity. "
                          "Correct them, their sum needs to be 1!")
                    
                    
                    
            else:
                #TODO: check if number of initial parameters for each population
                #is equal to number of kinetics in experiment!
                
                pop_sum = [0]*len(self.populations[0].initial)
                for elem in self.populations: 
                    for num in range(len(elem.initial)):
                        pop_sum[num] += p["_" + str(num) + "_" + elem.name]
                for i in range(len(pop_sum)):
                    if(pop_sum[i] != 1):
                        result = False
                        print("Initial population values in case of kinetic " + 
                              str(i) + " does not sum to unity (kinetics are "
                              "counted from 0). Correct them, their sum needs to be 1!")
   
                
            #TODO:
            #check if probe and irr in exp has epsilon in model
            #check if epsilon at irr is != than zero (othervise population 
            #calculated from abs will be undetermined, also what is the point 
            #of irr at zero epsilon????)
            #if initials are not fixed, check if they have correct dependence 
            #to force sum to be equal to 1. it may be hard to code in general form....
            #... invent more tests, to have better feedback if something if 
            #wrong before program fails...
            #for now, probe/irr wavelengths should be always fixed (in future, 
            #where epsilon curve will be used it may be ok to vary them)
        except Exception:
            print("Something went wrong during parameter check! This is very "
                  "strange error! Parameters are not validiated! Check "
                  "parameters if everything is all right!")
            return False
        return result   
       
    def _totalA(self, cs, irradiation, length):
        sum = 0.0
        for i in range(len(cs)):
            sum += cs[i] * self.populations[i].epsilon[irradiation]
        
        return sum * length 
    
    def _F(self, cs, irradiation, length):
        tA = self._totalA(cs, irradiation, length)
        return (1 - np.exp(-2.30259 * tA)) / tA
    
    def _derrivt(self, cs, t, irradiation, length, intensity):
        out = list()
        Ftmp = self._F(cs, irradiation, length)
        #for every population add contributions from different populations 
        #if there is some process
        for c in range(len(cs)): 
            ct = 0.0
            for arr in self.populations[c].arrows:
                cont = 0.0
                if(self.populations[c] is arr.source):
                    cont = - cs[c]
                elif(self.populations[c] is arr.target):
                    cont = cs[self.populations.index(arr.source)]
                if(arr.type == "k"):
                    cont *= arr.k
                elif(arr.type == "ke"):
                    #remember to caculate it before otherwise result will be incorrect
                    cont *= arr.k 
                elif(arr.type == "fi"):
                    cont *= arr.fi * Ftmp * intensity * arr.source.epsilon[irradiation] * length 
                ct += cont
            out.append(ct)
        return out #returns list with derrivatives of cs
    
    def _absorbance(self, cs, length, wavelength): 
        #calculate absorbance for some concentrations, wavelength and cuvette length
        out_abs = 0.0
        for x in range(len(cs)):
            out_abs += cs[x] * self.populations[x].epsilon[float(wavelength)]
        return out_abs * length

    def solveModelSingle(self, data, population_num = None): 
        #returns data object with values generated from model. 
        #update model with new params if they were obtained from fit
        # population_num = None will calculate absorbances, but you can give
        #population number to get concentration trace of this population
        initial_conditions = list()
        #there was major mistake, i corrected, but better check again, 
        #because error was really stupid and its late now...
        if(self.psplit == False): 
            weighted_epsilons = [(elem.epsilon[data.irradiation] * elem.initial) \
                                 for elem in self.populations]
            cfactor = data.absorbance/(data.irradiation_length * sum(weighted_epsilons))
            #create initial conditions, initialize all population in the 
            #ground state, if cannot find (or more than 1), raise error
            for elem in self.populations: 
                initial_conditions.append(elem.initial * cfactor)  
        else:
            weighted_epsilons = [(elem.epsilon[data.irradiation] * elem.initial[data.num]) \
                                 for elem in self.populations]
            cfactor = data.absorbance/(data.irradiation_length * sum(weighted_epsilons))            
            #create initial conditions, initialize all population in the
            #ground state, if cannot find (or more than 1), raise error
            for elem in self.populations: 
                initial_conditions.append(elem.initial[data.num] * cfactor)  
            
        splitpoint1 = -1
        for i in range(len(data.data_t)-1): #split data between on and off light regime
            if(data.data_t[i] <= data.t_on and data.t_on < data.data_t[i+1]):
                splitpoint1 = i
                break
            
        splitpoint2 = -1
        for i in range(len(data.data_t)-1): #split data between on and off light regime
            if(data.data_t[i] <= data.t_off and data.t_off < data.data_t[i+1]):
                splitpoint2 = i
                break
       
        if(splitpoint1 == -1 or splitpoint1 == len(data.data_t)-1):
            splitpoint1 = 0
        if(splitpoint2 == -1 or splitpoint2 <= splitpoint1):
            splitpoint2 = len(data.data_t)-1
            
        abs_out = list()    
            
        for arr in self.processes:
            if(arr.type == "ke"):
                #update temperature for temperature dependent process
                arr.getK(data.temperature) 
        
        if(splitpoint1 != 0):
            grid1 = data.data_t[:splitpoint1+1]
            # by adding hmax specify max step
            y1 = odeint(self._derrivt, initial_conditions, grid1, 
                        args=(data.irradiation,data.irradiation_length,0.0)) 
            initial_conditions = y1[-1]
            if(population_num is None):
                abs1 = [self._absorbance(cse, data.probe_length, data.probe) for cse in y1]
            else:
                #mało eleganckie, ale zwykłe y1[:,pop_num] nie dziala...
                abs1 = [cse[population_num] for cse in y1] 
            abs_out += abs1[:-1]
        
        grid2 = data.data_t[splitpoint1:splitpoint2+1]
        # by adding hmax specify max step
        y2 = odeint(self._derrivt, initial_conditions, grid2, 
                    args=(data.irradiation,data.irradiation_length,data.intensity)) 
        initial_conditions = y2[-1]
        if(population_num is None):
            abs2 = [self._absorbance(cse, data.probe_length, data.probe) for cse in y2]
        else:
            abs2 = [cse[population_num] for cse in y2]
        abs_out += abs2
        
        if(splitpoint2 != len(data.data_t)-1):
            grid3 = data.data_t[splitpoint2:]
            # by adding hmax specify max step
            y3 = odeint(self._derrivt, initial_conditions, grid3, 
                        args=(data.irradiation,data.irradiation_length,0.0)) 
            if(population_num is None):
                abs3 = [self._absorbance(cse, data.probe_length, data.probe) for cse in y3]
            else:
                abs3 = [cse[population_num] for cse in y3]
            abs_out += abs3[1:]
            
        output_data = copy.deepcopy(data)    
        abs_out = np.array(abs_out)
        if(data.zeroed):
            abs_out = abs_out - abs_out[0]
        
        output_data.data_a = abs_out
        if(len(abs_out) != len(output_data.data_t)):
            raise Exception("Inconsistent generated data!")
            
        return output_data
    
    def solveModel(self, experiment): 
        #returns experiment object with values generated from model.
        #update model with new params if they were obtained from fit
        if(self.checkParams(experiment) == False):
            raise Exception("Paratemer check directly before fitting process "
                            "failed! Check again if all parameters are set "
                            "correctly and nothing is missing! Note that both "
                            "params from Experiment and Model objects are required!")
        return_experiment = Experiment()
        for i in range(len(experiment.all_data)):
            new_data = self.solveModelSingle(experiment.all_data[i])
            return_experiment.addKineticData(new_data)
        return return_experiment
    
    def plotYourself(self, experiment, num = None, x_min = None, 
                     x_max = None,dpi = 120, title = None):
        colors = ("b","r","g","c","m","y","C0","C1","C2","C3","C4","C5","C6","C7") * 100
        plt.figure(dpi=dpi)
        
        if(num is None):
            tmplist = list()
            for i in range(len(experiment.all_data)):
                tmplist.append(self.solveModelSingle(experiment.all_data[i]))
                plt.plot(experiment.all_data[i].data_t, experiment.all_data[i].data_a, 
                         colors[i]+"-",alpha=0.5)
                plt.plot(tmplist[i].data_t, tmplist[i].data_a, 
                         colors[i]+"-", label = "Kinetic " + str(i) + " irr: " + \
                         str(experiment.all_data[i].irradiation) + " nm, probe: " + \
                         str(experiment.all_data[i].probe) + " nm.")
            plt.legend(fontsize="x-small", frameon=False, labelspacing=0.1)
        else:
            if(num < len(experiment.all_data)): # and num < tmp.count #dokoncz!!!!!!!!!!!!
                tmp = self.solveModelSingle(experiment.all_data[num])
                plt.plot(experiment.all_data[num].data_t, experiment.all_data[num].data_a, "bo")
                plt.plot(tmp.data_t, tmp.data_a, "r-")
        
        if(title is not None):
            plt.title(title, loc="left", fontsize=16)
        plt.xlabel("Time (s)",fontsize=16)
        plt.ylabel("\u0394A or A",fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)        
        if(x_min is not None):
            plt.xlim(left=x_min) 
        if(x_max is not None):
            plt.xlim(right=x_max)             
        plt.show()
    
    def plotConcentrations(self, data, population_nums = [], x_min = None, 
                           x_max = None,dpi = 80, title = None):
        colors = ("b","r","g","c","m","y","C0","C1","C2","C3","C4","C5","C6","C7") * 100
        plt.figure(dpi=dpi)
        
        for num in population_nums:
            tmp = self.solveModelSingle(data,num)
            plt.plot(tmp.data_t, tmp.data_a, colors[num]+"-", 
                     label = "Population " + self.populations[num].name)
        plt.legend(fontsize="x-small", frameon=False, labelspacing=0.1)
        
        if(title is not None):
            plt.title(title, loc="left", fontsize=16)
        plt.xlabel("Time (s)",fontsize=16)
        plt.ylabel("Concentration",fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)        
        if(x_min is not None):
            plt.xlim(left=x_min) 
        if(x_max is not None):
            plt.xlim(right=x_max)             
        plt.show()
        
    #what it should do:
    #1. there should be function which extract required paremeter from params, 
    #if absent then return error
    #1.1. maybe you should check if all params are there at the begining only...
    #2. implement totalA and F based on all populations
    #3. implement derrivative func based on all populations and processes
    #4. solve odeint
    #5. calculate and return kinetic man...



