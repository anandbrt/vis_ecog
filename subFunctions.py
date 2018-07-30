# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 21:32:20 2018

@author: lenovo_i5
"""
import os
import scipy.io
import numpy as np
import h5py
from read_HTK import readHTK
import math
import matplotlib.patches as patches
from matplotlib.path import Path
from PyQt5.QtWidgets import QInputDialog, QLineEdit
import datetime
import sys
sys.path.insert(0, 'G:\GUI\code\Function')
import functions as f

class ecogTSGUI:
    def __init__(self, pathName, parameters):
        self.pathName = pathName
        self.axesParams = parameters        
        out = self.loadBlock()
        self.ecog = out['ecogDS']
        self.x_cur = []
        self.y_cur = []
        n_ch = np.shape(self.ecog['data'])[1]
        self.channelScrollDown = np.arange(0, int(self.axesParams['editLine']['qLine0'].text()))
        self.indexToSelectedChannels = np.arange(0, int(self.axesParams['editLine']['qLine0'].text()))
        self.selectedChannels = np.arange(0, n_ch)
        self.channelSelector = np.arange(0, n_ch)
#        self.intervalStartGuiUnits = int(self.axesParams['editLine']['qLine2'].text())
        self.rawecog = out['ecogDS']
        self.badIntervals = out['badTimeSegments']
        print self.badIntervals
        self.badChannels = out['badChannels']
#        self.keyPressed = keyPressed
        
        if os.path.exists(os.path.join(pathName, 'Analog', 'ANIN4.htk')):#menu('load audio?','yes','no') == 1
            self.disp_audio = 1
            self.audio = readHTK(os.path.join(pathName, 'Analog', 'ANIN4.htk'))    
            
            self.downsampled = self.downsample(10)
            self.audio['sampling_rate'] = self.audio['sampling_rate']/10000
            self.fs_audio = self.audio['sampling_rate']/10            
            self.taudio = np.arange(1, np.shape(self.downsampled)[0])/self.fs_audio
            
        else:
            self.disp_audio = 0
         
        total_dur = len(self.ecog['data'])/self.ecog['sampFreq'][0][0]
        self.axesParams['pars']['Axes'][2].plot([0, total_dur], [0.5, 0.5], color = 'k', linewidth = 0.5)

#        %plot bad time segments on timeline
        BIs = self.badIntervals   
        self.BIRects = np.array([], dtype = 'object')
        for i in range(np.shape(BIs)[0]):
            verts = [(BIs[i][0], 0),
                     (BIs[i][0] + max([BIs[i][1] - BIs[i][0], 0.01]), 0),
                     (BIs[i][0] + max([BIs[i][1] - BIs[i][0], 0.01]), 1),
                     (BIs[i][0], 1),
                     ]
            codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             ]
            path = Path(verts, codes)
            self.BIRects = np.append(self.BIRects, [patches.PathPatch(path,  facecolor = 'r')])
#            self.BIRects[str(i)] = patches.PathPatch(path,  facecolor = 'r')
            self.axesParams['pars']['Axes'][2].add_patch(self.BIRects[i])
            self.axesParams['pars']['Figure'][2].canvas.draw()
#        BIs(all(~BIs,2),:) = []; % fix row of all zeros
#        for i=1:size(BIs,1)
#            handles.BIRects(i) = rectangle('Position',[BIs(i,1),0,max(BIs(i,2)-BIs(i,1),.01),1],'FaceColor','r');
#        end
        
        verts = [(0, 0),
                 (0, 1),
                 (1, 1),
                 (1, 0),
                 ]
        codes = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.LINETO,
         ]
        path = Path(verts, codes)
        self.current_rect = patches.PathPatch(path, facecolor = 'g', linewidth = 0.5, alpha = 0.4)
        self.axesParams['pars']['Axes'][2].add_patch(self.current_rect)
#        self.current_rect = 
        self.refreshScreen()
       
    def refreshScreen(self):
        self.AR_plotter()
        
    def AR_plotter(self):        
        self.getCurAxisParameters()
        startSamp = int(math.ceil(self.intervalStartSamples))      
        endSamp = int(math.floor(self.intervalEndSamples))       
        
        channelsToShow = self.selectedChannels[self.indexToSelectedChannels]
        self.verticalScaleFactor = float(self.axesParams['editLine']['qLine4'].text())
        scaleFac = np.var(self.ecog['data'][: (endSamp - startSamp), :], axis = 0)/self.verticalScaleFactor #We use on fixed interval for the scaling factor to keep things comparable
       
        scaleVec = np.arange(1, len(channelsToShow) + 1) * max(scaleFac) * 1/50 #The multiplier is arbitrary. Find a better solution
        
        timebaseGuiUnits = np.arange(startSamp - 1, endSamp) * (self.intervalStartGuiUnits/self.intervalStartSamples) 
        print timebaseGuiUnits[0], timebaseGuiUnits[-1]
        scaleV = np.zeros([len(scaleVec), 1])
        scaleV[:, 0] = scaleVec
        try:
            
            data = self.ecog['data'][startSamp - 1 : endSamp, channelsToShow].T
            plotData = data + np.tile(scaleV, (1, endSamp - startSamp + 1)) #data + offset        
        except:
#            #if time segment shorter than window.
            data = self.ecog['data'][:, channelsToShow].T
            plotData = data + np.tile(scaleV, (1, endSamp - startSamp + 1)) # %data + offset

###### Rectangle Plot
        x = float(self.axesParams['editLine']['qLine2'].text())
        w = float(self.axesParams['editLine']['qLine3'].text())
        verts = [(x, 0),
                 (x, 1),
                 (x + w, 1),
                 (x + w, 0),
                 ]
        codes = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.LINETO,
         ]
        path = Path(verts, codes)
#        self.axesParams['pars']['Axes'][2].cla()
        self.current_rect = patches.PathPatch(path, facecolor = 'g', linewidth = 0.5, alpha = 0.4)
        self.axesParams['pars']['Axes'][2].add_patch(self.current_rect)
#        self.axesParams['pars']['Axes'][2].set_axis_off()
#        self.axesParams['pars']['Figure'][2].canvas.draw()
#        self.axesParams['pars']['Figure'][2].canvas.flush_events()
######       Rectangle Plot
        
#        %bad_t=handles.ecog.bad_t;
#        %A line indicating zero for every channel        
        x = np.tile([timebaseGuiUnits[0], timebaseGuiUnits[-1]], (len(scaleVec), 1))
        y = np.hstack((scaleV, scaleV))
        self.axesParams['pars']['Axes'][0].cla()
        self.p = self.axesParams['pars']['Axes'][0].plot(x.T, y.T, color = 'k', linewidth = 0.5)
        self.axesParams['pars']['Axes'][0].set_xlabel('Time (seconds)')
        self.axesParams['pars']['Axes'][0].set_ylabel('Channel #')
        self.axesParams['pars']['Axes'][0].set_yticks(y[:, 0])
        labels = [str(ch + 1) for ch in channelsToShow]        
        self.axesParams['pars']['Axes'][0].set_yticklabels(labels)
        
#        
        
        badch = np.array([])
        for i, channels in enumerate(channelsToShow):
            if np.any(str(channels) in self.badChannels):
                badch[i] = channels
        if not np.empty(badch):
            pass
#            if ~length(find(handles.ecog.badChannels==999))
#                ph=plot(handles.ecg_axes,timebaseGuiUnits,plotData,'LineWidth',1);
#                plot(handles.ecg_axes,timebaseGuiUnits,plotData(badch,:),'r','LineWidth',1);
#            else
#                plotData(badch,:)=NaN;
#                ph=plot(handles.ecg_axes,timebaseGuiUnits,plotData);
                
            
            
        else:
            #PLOT CHANNELS
            
            
            self.axesParams['pars']['Axes'][0].plot(timebaseGuiUnits, plotData.T, linewidth = 0.5, alpha = 0.6)
           
#            self.axesParams['pars']['Axes'][0].set_xlim((timebaseGuiUnits[0], timebaseGuiUnits[-1]))
#            self.axesParams['axes']
#        %MAKE TRANSPARENT BOX AROUND BAD TIME SEGMENTS
        self.plotData = []
        self.plotData = plotData.T
        self.showChan = channelsToShow
#        
#        %axis(handles.ecg_axes,'tight');
#        
        ymin, ymax = self.axesParams['pars']['Axes'][0].get_ylim()
        for i in range(np.shape(self.badIntervals)[0]):
            BI = self.badIntervals[i, :]
            
            verts = [(BI[0], ymin),
                 (BI[0], ymax),
                 (BI[1], ymax),
                 (BI[1], ymin),
                 ]
            codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             ]
            path = Path(verts, codes)
#        self.axesParams['pars']['Axes'][2].cla()
            h = patches.PathPatch(path, facecolor = 'y', 
                                              linewidth = 0.5, alpha = 0.2, edgecolor = None)
            self.axesParams['pars']['Axes'][0].add_patch(h)
            self.axesParams['pars']['Axes'][0].text(BI[0], ymin, str(round(BI[0], 4)), color = 'r')
            self.axesParams['pars']['Axes'][0].text(BI[1], ymin, str(round(BI[1], 4)), color = 'r')
    
        self.axesParams['pars']['Figure'][0].canvas.draw()
        self.axesParams['pars']['Figure'][0].canvas.flush_events()

        if self.disp_audio:#    % PLOT AUDIO            
            begin = self.intervalStartGuiUnits            
            stop = self.intervalStartGuiUnits + self.intervalLengthGuiUnits            
            ind_disp = np.where((self.taudio > begin) & (self.taudio < stop))
            self.axesParams['pars']['Axes'][1].cla()
            self.axesParams['pars']['Axes'][1].plot(self.taudio[ind_disp], \
                           self.downsampled[ind_disp], linewidth = 0.5)
            self.axesParams['pars']['Axes'][1].set_axis_off()
            self.axesParams['pars']['Figure'][1].canvas.draw()
            self.axesParams['pars']['Figure'][1].canvas.flush_events()
#    plot(handles.taudio(ind_disp),handles.audio(ind_disp));
#    
#            self.axesParams['pars']['Axes'][1].set_xlim(timebaseGuiUnits[0], timebaseGuiUnits[-1])
            
        
    def getCurAxisParameters(self):        
        self.getCurXAxisPosition()
        
    def getCurXAxisPosition(self):
        self.intervalStartGuiUnits = float(self.axesParams['editLine']['qLine2'].text())
        
        self.intervalLengthGuiUnits = float(self.axesParams['editLine']['qLine3'].text())
        
        
        total_dur = np.shape(self.ecog['data'])[0] * self.ecog['sampDur'][0][0]/1000        
        if self.intervalLengthGuiUnits > total_dur:
            self.intervalLengthGuiUnits = total_dur
        
        self.intervalLengthSamples = self.intervalLengthGuiUnits * (1000/self.ecog['sampDur'][0][0])
        
        self.intervalStartSamples = self.intervalStartGuiUnits * (1000/self.ecog['sampDur'][0][0]) # assumes seconds in GUI and milliseconds in  ecog.sampDur
        self.intervalEndSamples = self.intervalStartSamples + self.intervalLengthSamples - 1 #We assume that the intervall length is specified in samples
        #%should always work because plausibility has been checked when channels were entered (
        self.indexToSelectedChannels = self.channelScrollDown
        
    def downsample(self, n):
        L = self.audio['num_samples']
        block = round(L/n)
        n_zeros = block * n + n - L       
        self.downsampled = np.append(self.audio['data'], np.zeros([int(n_zeros)]))
        re_shape = np.reshape(self.downsampled, [int((block * n + n)/n), n])[:, 0]
        return re_shape
            
    
    def loadBadTimes(self):
        
        filename = os.path.join(self.pathName, 'Artifacts', 'badTimeSegments.mat')
        
        if os.path.exists(filename):
            loadmatfile = scipy.io.loadmat(filename)
            badTimeSegments = loadmatfile['badTimeSegments']
            print '{} bad time segments loaded '.format(np.shape(loadmatfile['badTimeSegments'])[1])
        else:
            if not os.path.exists(os.path.join(self.pathName, 'Artifacts')):
                os.mkdir(os.path.join(self.pathName, 'Artifacts'))
                
            badTimeSegments = []
            scipy.io.savemat(filename, mdict = {'badTimeSegments': badTimeSegments})
    
        return badTimeSegments
    
    def loadBadCh(self):
        filename = os.path.join(self.pathName, 'Artifacts', 'badChannels.txt')
        if os.path.exists(filename):
            with open(filename)  as f:
                badChannels = f.read()
                print 'Bad Channels : {}'.format(badChannels)
        else:
            os.mkdir(os.path.join(self.pathName, 'Artifcats'))
            with open(filename, 'w') as f:
                f.write('')
                f.close()
            badChannels = []
        return badChannels
    
    
    def fileParts(self):
        parts = self.pathName.split('\\')
        return parts[-1]

#def getEcog(pathName, newfs):
    
    def loadBlock(self, *argv):
        
        try:
            saveopt = argv[0]  
            
        except:
            saveopt = 0
            
            
        try:
            newfs = argv[1]
        except:
            newfs = 400
            
        if saveopt:
            auto = 1
        else:
            auto = 0
        
            
    
    
    #if ~exist(pathName,'dir')
    #    pathName = uigetdir(pwd);
    #end
    
        if os.path.exists(os.path.join(self.pathName, 'RawHTK')) and \
            os.path.exists(os.path.join(self.pathName, 'ecog400')) and \
            os.path.exists(os.path.join(self.pathName, 'ecog600')) and \
            os.path.exists(os.path.join(self.pathName, 'ecog1000')) and \
            os.path.exists(os.path.join(self.pathName, 'ecog2000')):
                print('Please choose a block folder that contains the RawHTK, ecog400 or ecog600 folder, e.g. EC34_B5')
                
        
    
    
        blockName = self.fileParts()
        print blockName
    
    #automatically load bad time segments
    
        badTimeSegments = self.loadBadTimes()
        badChannels = self.loadBadCh()
        
    #load data
    
    #try to find downsampled data
        file_ = os.path.join(self.pathName, 'ecog' + str(newfs), 'ecog.mat')
        out = dict()
        if os.path.exists(file_):
            print 'Loading downsampled ecog....'
            loadmatfile = h5py.File(file_)
            print 'done'
            
        elif os.path.exists(os.path.join(self.pathName, 'RawHTK')):
            print 'Loading downsampled ecog...'
            loadmatfile = h5py.File(file_)
            print 'done'
            ## LEFT to be implemented
        
            
        out['badTimeSegments'] = badTimeSegments
        out['badChannels'] = badChannels
        out['blockName'] = blockName
        out['ecogDS'] = loadmatfile['ecogDS']
        
        return out

    def channel_Scroll_Up(self):
        blockIndices = self.channelScrollDown        
        self.nChannelsDisplayed = int(self.axesParams['editLine']['qLine0'].text())
        nChanToShow = self.nChannelsDisplayed        
        chanToShow = self.channelSelector        
        blockIndices  = blockIndices + nChanToShow        
        if blockIndices[-1] > len(chanToShow):
            blockIndices = np.arange(len(chanToShow) - len(blockIndices) + 1, len(chanToShow))
#        print blockIndices
        self.channelScrollDown = blockIndices        
        self.refreshScreen()
        
    def channel_Scroll_Down(self):
        blockIndices = self.channelScrollDown        
        nChanToShow = int(self.axesParams['editLine']['qLine0'].text())
        blockIndices = blockIndices - nChanToShow
        if blockIndices[0] <= 0:
            blockIndices = np.arange(0, nChanToShow) #first possible block
            
        self.channelScrollDown = blockIndices
        self.refreshScreen()
        
    def page_forward(self):
        n, m = np.shape(self.ecog['data'])       
        self.getCurXAxisPosition()
        # check if inteval length is appropriate        
        if self.intervalLengthSamples > n:
            #set interval length to available data
            self.intervalLengthSamples = n        
        #new interval start
        
        self.intervalStartSamples = self.intervalStartSamples \
        + self.intervalLengthSamples        
       
        #if the last sample of the new interval is beyond is the available data
        # set start such that intserval is in a valid range
        if self.intervalStartSamples + self.intervalLengthSamples > n:
            self.intervalStartSamples = n - self.intervalLengthSamples + 1
        
        self.setXAxisPositionSamples()
        self.refreshScreen()
        
    def setXAxisPositionSamples(self):
        t = 1000/self.ecog['sampDur'][0][0]
        self.axesParams['editLine']['qLine2'].setText(str(self.intervalStartSamples/t))
        self.axesParams['editLine']['qLine3'].setText(str(self.intervalLengthSamples/t))
        
    def page_back(self):
        n, m = np.shape(self.ecog['data'])
        self.getCurXAxisPosition()
        #new interval start
        self.intervalStartSamples = self.intervalStartSamples - self.intervalLengthSamples
        
        # if the last sample of the new interval is beyond is the available data
        # set start such that intserval is in a valid range
        if self.intervalStartSamples < 1:
            self.intervalStartSamples = 1
        
        self.setXAxisPositionSamples()
        self.refreshScreen()
        
    def scroll_back(self):
        n, m = np.shape(self.ecog['data'])
        self.getCurXAxisPosition()
        #new interval start
        self.intervalStartSamples = self.intervalStartSamples - self.intervalLengthSamples/3
        
        # if the last sample of the new interval is beyond is the available data
        # set start such that intserval is in a valid range
        if self.intervalStartSamples < 1:
            self.intervalStartSamples = 1
        
        self.setXAxisPositionSamples()
        self.refreshScreen()

    def scroll_forward(self):
        n, m = np.shape(self.ecog['data'])       
        self.getCurXAxisPosition()
        # check if inteval length is appropriate        
        if self.intervalLengthSamples > n:
            #set interval length to available data
            self.intervalLengthSamples = n        
        #new interval start
        
        self.intervalStartSamples = self.intervalStartSamples \
        + self.intervalLengthSamples/3        
       
        #if the last sample of the new interval is beyond is the available data
        # set start such that intserval is in a valid range
        if self.intervalStartSamples + self.intervalLengthSamples > n:
            self.intervalStartSamples = n - self.intervalLengthSamples + 1
        
        self.setXAxisPositionSamples()
        self.refreshScreen()

    def verticalScaleIncrease(self):
        scaleFac = float(self.axesParams['editLine']['qLine4'].text())
        self.axesParams['editLine']['qLine4'].setText(str(scaleFac * 2))        
        self.refreshScreen()
        
    def verticalScaleDecrease(self):
        scaleFac = float(self.axesParams['editLine']['qLine4'].text())
        self.axesParams['editLine']['qLine4'].setText(str(scaleFac/2.0))        
        self.refreshScreen()
        
    def plot_interval(self):
        plotIntervalGuiUnits = float(self.axesParams['editLine']['qLine3'].text())
        sam = self.ecog['sampDur'][0][0]
        n = np.shape(self.ecog['data'])[0]
        
        if plotIntervalGuiUnits * 1000 < sam:
            plotIntervalGuiUnits = self.ecog['sampDur'][0][0]/1000
            self.axesParams['editLine']['qLine3'].setText(str(plotIntervalGuiUnits))
        elif plotIntervalGuiUnits * 1000/sam > n:
            plotIntervalGuiUnits = (n - 1) * sam/1000
            self.axesParams['editLine']['qLine3'].setText(str(plotIntervalGuiUnits))
        self.refreshScreen()        
        
    def start_location(self):
        self.getCurXAxisPosition()
        #interval at least one sample long
        if self.intervalStartSamples < self.ecog['sampDur'][0][0]:
            self.intervalStartSamples = 1 #set to the first sample
            self.setXAxisPositionSamples()       
        
        if self.intervalStartSamples + self.intervalLengthSamples - 1 > \
        np.shape(self.ecog['data'])[0]:
            self.intervalStartSamples = np.shape(self.ecog['data'])[0] - \
            self.intervalLengthSamples + 1
            self.setXAxisPositionSamples()
        
        self.refreshScreen()
        
    def nChannels_Displayed(self):
        nChanToShow = int(self.axesParams['editLine']['qLine0'].text())        
#        nChanToShow = nChanToShow(1)# %make sure we have a scalar
#         make we have sure indices will be in a valid range
        chanToShow = self.channelSelector        
        if nChanToShow < 1:
            nChanToShow = 1
        elif nChanToShow > len(chanToShow):
            nChanToShow = len(chanToShow)
        self.axesParams['editLine']['qLine0'].setText(str(nChanToShow))
        
        self.channelScrollDown = np.arange(0, nChanToShow)        
        self.refreshScreen()
        
    def addBadTimeSeg(self, BadInterval):
#        axes(handles.timeline_axes);
        x = BadInterval[0]        
        y = 0
        w = np.diff(np.array(BadInterval)) 
        h = 1
        verts = [(x, y),
                     (x + w, y),
                     (x + w, h),
                     (x, h),
                     ]
        codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             ]
        path = Path(verts, codes)
        if np.shape(self.BIRects)[0] != 0:          
            l = len(self.BIRects)
            self.BIRects = np.append(self.BIRects, [patches.PathPatch(path, facecolor = 'r')])
#            self.BIRects[str(l)] = patches.PathPatch(path, facecolor = 'r')
            self.axesParams['pars']['Axes'][2].add_patch(self.BIRects[l])
            self.axesParams['pars']['Figure'][2].canvas.draw()
        else:
            self.BIRects = np.append(self.BIRects, [patches.PathPatch(path, facecolor = 'r')]) 
            self.axesParams['pars']['Axes'][2].add_patch(self.BIRects[0])
            self.axesParams['pars']['Figure'][2].canvas.draw()
#        
        if np.shape(self.badIntervals)[1] == 0:
            self.badIntervals = np.array(BadInterval)
        else:
            self.badIntervals = np.vstack((self.badIntervals, np.array(BadInterval)))
        
            
    def deleteInterval(self, x):
        BIs = self.badIntervals
        di = np.where((x >= BIs[:, 0]) & (x <= BIs[:, 1]))
        self.BIRects = np.delete(self.BIRects, di, axis = 0)
        self.badIntervals = np.delete(self.badIntervals, di, axis = 0)
        self.refreshScreen()
            
    def pushSave(self):
        BAD_INTERVALS = self.badIntervals
        fullfile = os.path.join(self.pathName, 'Artifacts', 'bad_time_segments.lab')
        n_size = np.shape(BAD_INTERVALS)[0]
        one = np.ones((n_size, 1))
        variable = np.hstack((BAD_INTERVALS, one))
        f.BadTimesConverterGUI(variable, fullfile)        
        badTimeSegments = BAD_INTERVALS
        file_name = os.path.join(self.pathName, 'Artifacts', 'badTimeSegments')
        scipy.io.savemat(file_name, {'badTimeSegments': badTimeSegments})
        my_file = os.path.join(self.pathName, 'Artifacts', 'info.txt')
        if not os.path.exists(my_file):
            username, okPressed = QInputDialog.getText(self, 'Enter Text', 'Who is this:', QLineEdit.Normal, '')
            if okPressed and username != '':
                fileid = open(os.path.join(self.pathName, 'Artifacts', 'info.txt'), 'w')
                fileid.write(username + ' ' + datetime.datetime.today().strftime('%Y-%m-%d'))
                fileid.close()
        else:
            fileid = open(os.path.join(self.pathName, 'Artifacts', 'info.txt'), 'a')
            fileid.write(' ' + datetime.datetime.today().strftime('%Y-%m-%d'))
            fileid.close()

    def getChannel(self):
        x = self.x_cur
        y = self.y_cur
        if (x == []):
            pass
        else:
            start_ = float(self.axesParams['editLine']['qLine2'].text())
            end_ = float(self.axesParams['editLine']['qLine3'].text())
            points = np.round(np.linspace(start_, end_, np.shape(self.plotData)[0]), 2)
            index = np.where(points == round(x, 2))
            DataIndex = index
            y_points = self.plotData[DataIndex]
            for values in y_points:
                diff_ = abs(np.array(values) - y)            
                index = np.argmin(diff_)
                chanel = self.showChan + 1
                channel = chanel[index]
                
            self.axesParams['pars']['Axes'][0].set_title('Selected Channel:' + str(channel))
            self.axesParams['pars']['Figure'][0].canvas.draw()
            
            
            
            
        