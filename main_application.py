# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 10:13:56 2018

@author: lenovo_i5
"""
import sys
#import numpy as np
from PyQt5 import QtCore
from cycler import cycler
#from mpldatacursor import datacursor
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit, QMessageBox, QHBoxLayout,
    QTextEdit, QApplication, QPushButton, QVBoxLayout, QGroupBox, QFormLayout, QDialog,
    QCheckBox, QGridLayout)
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
#from functions import line_select_callback
from matplotlib.widgets import RectangleSelector
from subFunctions import ecogTSGUI

class Application(QDialog):
    keyPressed = QtCore.pyqtSignal(QtCore.QEvent)
    def __init__(self, parent = None):
        pathName = '/Users/bendichter/Desktop/Chang/data/EC125/EC125_B22'
        self.temp = []        
        super(Application, self).__init__()
        self.keyPressed.connect(self.on_key)
        self.init_gui()
        self.showMaximized()
        self.setWindowTitle('')          
        self.show()
        '''
        run the main file
        '''
        
        parameters = {}
        parameters['pars'] = {'Axes': [self.axes1, self.axes2, self.axes3], 'Figure': [self.figure1, self.figure2, self.figure3]}
        parameters['editLine'] = {'qLine0': self.qline0, 'qLine1': self.qline1, 'qLine2': self.qline2, 'qLine3': self.qline3, 
                  'qLine4': self.qline4}        
        self.model = ecogTSGUI(pathName, parameters)
        
#        model.channelScrollUp()
    def keyPressEvent(self, event): 
        super(Application, self).keyPressEvent(event)
        self.keyPressed.emit(event) 
        
    def on_key(self, event):
        if event.key() == QtCore.Qt.Key_W:
            self.model.channel_Scroll_Up()
        elif event.key() == QtCore.Qt.Key_S:            
            self.model.channel_Scroll_Down()
        elif event.key() == QtCore.Qt.Key_A:
            self.model.page_back()
            print "Left"
        elif event.key() == QtCore.Qt.Key_D:
            self.model.page_forward()
            print 'Right'
                    
    def init_gui(self):
        vbox = QVBoxLayout()
        
        groupbox1 = QGroupBox('')
        groupbox1.setFixedHeight(50)
        formlayout4 = QFormLayout()
        self.figure3 = Figure()
        self.axes3 = self.figure3.add_subplot(111)
        canvas2 = FigureCanvas(self.figure3)        
        self.axes3.set_axis_off()
        self.figure3.tight_layout(rect = [0, 0, 1, 1])
        formlayout4.addWidget(canvas2)
        groupbox1.setLayout(formlayout4)
        
        groupbox2 = QGroupBox('Channels Plot')
        groupbox2.setFixedHeight(580)
        formlayout5 = QFormLayout() 
        
        self.figure1 = Figure()
        self.axes1 = self.figure1.add_subplot(111)
        plt.rc('axes', prop_cycle = (cycler('color', ['b', 'g'])))
        self.figure1.tight_layout(rect = [0, 0, 1, 1])        
        self.canvas = FigureCanvas(self.figure1)
        formlayout5.addWidget(self.canvas)
        groupbox2.setLayout(formlayout5)        
        
        groupbox3 = QGroupBox('')
        groupbox3.setFixedHeight(40)
        formlayout6 = QFormLayout()
        self.figure2 = Figure()
        self.axes2 = self.figure2.add_subplot(111)
        canvas1 = FigureCanvas(self.figure2)        
        self.axes2.set_axis_off()
        self.figure2.tight_layout(rect = [0, 0, 1, 1])        
        formlayout6.addWidget(canvas1)        
        groupbox3.setLayout(formlayout6) 
        
        vbox.addWidget(groupbox1)
        vbox.addWidget(groupbox2)
        vbox.addWidget(groupbox3)
        
        
        hbox = QHBoxLayout()
        panel1 = QGroupBox('Panel')
        panel1.setFixedHeight(100)
        form1 = QFormLayout()
        self.push1 = QPushButton('Data Cursor On')
        self.push1.setFixedWidth(200)
        self.push1.clicked.connect(self.Data_Cursor)
        self.push2 = QPushButton('Get Ch')
        self.push2.clicked.connect(self.On_Click)
        self.push2.setFixedWidth(200)
        self.push3 = QPushButton('Save Bad Intervals')
        self.push3.clicked.connect(self.SaveBadIntervals)
        self.push3.setFixedWidth(200)
        self.push4 = QPushButton('Select Bad intervals')
        self.push4.clicked.connect(self.SelectBadInterval)
        self.push4.setFixedWidth(200)
        self.push5 = QPushButton('Delete Intervals')
        self.push5.clicked.connect(self.DeleteBadInterval)
        self.push5.setFixedWidth(200)
        form1.addRow(self.push1, self.push2)
        form1.addRow(self.push4, self.push5)
        form1.addWidget(self.push3)
        panel1.setLayout(form1)
        panel2 = QGroupBox('Signal Type')
        panel2.setFixedWidth(200)
        form2 = QFormLayout()
        self.rbtn1 = QCheckBox('raw ECoG')
        self.rbtn1.setChecked(True)
        self.rbtn2 = QCheckBox('High Gamma')
        self.rbtn2.setChecked(False)
        form2.addWidget(self.rbtn1)
        form2.addWidget(self.rbtn2)
        panel2.setLayout(form2)
        
        panel3 = QGroupBox('Plot Controls')
        gridLayout = QGridLayout()
#        gridLayout.setAlignment(Qt.AlignLeft)
        qlabel1 = QLabel('Ch selected #')
        qlabel1.setFixedWidth(70)
        gridLayout.addWidget(qlabel1, 0, 0)
        self.qline1 = QLineEdit('1')        
        self.qline0 = QLineEdit('16')
        self.qline0.returnPressed.connect(self.channelDisplayed)
        self.qline2 = QLineEdit('0.01')
        self.qline2.returnPressed.connect(self.start_location)
        self.qline1.setFixedWidth(40)
        self.qline2.setFixedWidth(40)
        self.pushbtn1 = QPushButton('^')
        self.pushbtn1.clicked.connect(self.scroll_up)
        self.pushbtn1.setFixedWidth(30)
        self.pushbtn2 = QPushButton('v')
        self.pushbtn2.clicked.connect(self.scroll_down)
        self.pushbtn3 = QPushButton('<<')
        self.pushbtn3.clicked.connect(self.page_backward)
        self.pushbtn4 = QPushButton('<')
        self.pushbtn4.clicked.connect(self.scroll_backward)
        self.pushbtn5 = QPushButton('>>')
        self.pushbtn5.clicked.connect(self.page_forward)
        self.pushbtn6 = QPushButton('>')
        self.pushbtn6.clicked.connect(self.scroll_forward)
        self.pushbtn3.setFixedWidth(30)
        self.pushbtn4.setFixedWidth(30)
        self.pushbtn5.setFixedWidth(30)
        self.pushbtn6.setFixedWidth(30)
        self.pushbtn2.setFixedWidth(30)
        
        gridLayout.addWidget(self.qline1, 0, 1)
        gridLayout.addWidget(self.qline0, 0, 2)
        gridLayout.addWidget(self.pushbtn1, 0, 3)
        gridLayout.addWidget(self.pushbtn2, 0, 4)
        gridLayout.addWidget(self.pushbtn3, 0, 5)
        gridLayout.addWidget(self.pushbtn4, 0, 6)
        
        qlabel2 = QLabel('Interval start(s)')
        qlabel2.setFixedWidth(80)
        gridLayout.addWidget(qlabel2, 0, 7)
        gridLayout.addWidget(self.qline2, 0, 8)
        gridLayout.addWidget(self.pushbtn6, 0, 9)
        gridLayout.addWidget(self.pushbtn5, 0, 10)
        qlabel3 = QLabel('Window')
        qlabel3.setFixedWidth(60)
        self.qline3 = QLineEdit('25')
        self.qline3.returnPressed.connect(self.plot_interval)
        self.qline3.setFixedWidth(40)
        
        gridLayout.addWidget(qlabel3, 0, 11)
        gridLayout.addWidget(self.qline3, 0, 12)
        qlabel4 = QLabel('Vertical Scale')
        
        qlabel4.setFixedWidth(60)
        self.qline4 = QLineEdit('1')
        self.qline4.setFixedWidth(40)
        
        gridLayout.addWidget(qlabel4, 0, 13)
        gridLayout.addWidget(self.qline4, 0, 14)
        
        self.pushbtn7 = QPushButton('*2')
        self.pushbtn7.clicked.connect(self.verticalScaleIncrease)
        self.pushbtn8 = QPushButton('/2')
        self.pushbtn8.clicked.connect(self.verticalScaleDecrease)
        self.pushbtn7.setFixedWidth(30)
        self.pushbtn8.setFixedWidth(30)
        gridLayout.addWidget(self.pushbtn7, 0, 15)
        gridLayout.addWidget(self.pushbtn8, 0, 16)
        panel3.setLayout(gridLayout)
        hbox.addWidget(panel1)
        hbox.addWidget(panel2)
        hbox.addWidget(panel3)       
        vbox.addLayout(hbox)       
        self.setLayout(vbox) 
        
     
        
    def SaveBadIntervals(self):
        self.model.pushSave()
    
    def SelectBadInterval(self):        
        self.toggle_selector = RectangleSelector(self.axes1, self.line_select_callback, useblit = True,
                                                 spancoords = 'pixels', drawtype = 'box', 
                                                 minspanx=5, minspany=5, rectprops = dict(facecolor = 'peachpuff', edgecolor = None,
                 alpha = 0.04, fill = True))
        self.toggle_selector.to_draw.set_visible(True)
             
    def On_Click(self):
        self.model.getChannel()
    
    def line_select_callback(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        BadInterval = [x1, x2]
        self.model.addBadTimeSeg(BadInterval)
        self.model.refreshScreen()
                 
    def DeleteBadInterval(self):
        cid = self.figure1.canvas.mpl_connect('button_press_event', self.get_coordinates)
        self.toggle_selector.set_active(False)
    def get_coordinates(self, event):
        x = event.xdata
#        y = event.ydata
        self.model.deleteInterval(x)
    def get_cursor_position(self, event):
        x = event.xdata
        y = event.ydata
        self.model.x_cur = x
        self.model.y_cur = y
        props = dict(boxstyle = 'round', facecolor = 'y',  alpha=0.5)
        text_ = self.axes1.text(x, y, 'x:' + str(round(x,2)) + '\n' + 'y:' + str(round(y, 2)), bbox = props)   
        
        if self.temp == []:
            self.temp = text_
        else:
            self.temp.remove()
            self.temp = text_
            
        self.figure1.canvas.draw()
        
    def Data_Cursor(self):        
        if self.push1.text() == 'Data Cursor On':
            self.push1.setText('Data Cursor Off')
            cid = self.figure1.canvas.mpl_connect('button_press_event', self.get_cursor_position)
           
        elif self.push1.text() == 'Data Cursor Off':
            if self.temp != []:
                self.temp.remove()
                self.temp = []
                self.figure1.canvas.draw()
                
            self.push1.setText('Data Cursor On')
            
    def scroll_up(self):
        self.model.channel_Scroll_Up()
    def scroll_down(self):
        self.model.channel_Scroll_Down()
    def page_backward(self):
        self.model.page_back()
    def scroll_backward(self):
        self.model.scroll_back()
    def page_forward(self):
        self.model.page_forward()
    def scroll_forward(self):
        self.model.scroll_forward()
    def verticalScaleIncrease(self):
        self.model.verticalScaleIncrease()
    def verticalScaleDecrease(self):
        self.model.verticalScaleDecrease()        
    def plot_interval(self):
        self.model.plot_interval()
    def start_location(self):
        self.model.start_location()
    def channelDisplayed(self):        
        self.model.nChannels_Displayed()
    
        
if __name__ == '__main__':
    app = QCoreApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    ex = Application()
    

sys.exit(app.exec_())