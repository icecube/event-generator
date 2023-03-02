#!/usr/bin/env python

import os
from I3Tray import *
from os.path import *
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from icecube import tableio,hdfwriter
from icecube.tableio import I3TableWriter
from icecube import icetray, dataclasses, dataio, WaveCalibrator, simclasses
from icecube import photonics_service
from icecube.weighting import get_weighted_primary
from icecube.hdfwriter import I3HDFTableService
from icecube import gulliver
from icecube import recclasses
from icecube.phys_services import I3Calculator
import icecube_labels
import math
import heapq, operator
from itertools import izip
# from icecube import SeededRTCleaning

from .misc import weighted_quantile, weighted_std


@icetray.traysegment
def CheckRawDataTray(tray, name):
    def CheckRawData(frame):
        if frame.Has('InIceRawData'):
            return True
        return False
    tray.AddModule(CheckRawData,
                   'check-raw-data',
                   Streams=[icetray.I3Frame.Physics])
    return


@icetray.traysegment
def getPulseData(tray, name, domPosDict,maxNumberOfPulses=300, nanValue = float('nan')):
    def writePulses(frame):
        pulses = []
        pulseMap = frame['I3SuperDST'].unpack()
        for domPulses in pulseMap:
            for pulse in domPulses[1]:
                omkey = domPulses[0]
                pos = domPosDict[omkey]
                 # 0: omkey, 1-3:pos, 4: time, 5: width, 6: charge
                # heapq.heappush(pulses, (pulse.charge,pos.x,pos.y,pos.z,pulse.time,pulse.width,omkey))
                pulses.append([omkey,pos.x,pos.y,pos.z,pulse.time,pulse.width,pulse.charge])
        noOfPulses = len(pulses)
        noOfHitDoms = len(pulseMap)
        # only choose 300 highest amplitude pulses
        if noOfPulses > maxNumberOfPulses:
            pulses = heapq.nlargest(maxNumberOfPulses,pulses,key=operator.itemgetter(6))
        # sort pulses according to time
        pulses = np.array(pulses)
        pulses = pulses[pulses[:,4].argsort()]
        # write pulses to frame
        pulseData = dataclasses.I3MapStringDouble()
        pulseData['NoOfPulses'] = noOfPulses
        pulseData['NoOfHitDoms'] = noOfHitDoms
        frame.Put('pulseData',pulseData)
        for p,i in zip(pulses,xrange(noOfPulses)):
            container = dataclasses.I3MapStringDouble()
            container['x'] = p[1]
            container['y'] = p[2]
            container['z'] = p[3]
            container['time'] = p[4]
            container['width'] = p[5]
            container['charge'] = p[6]
            frame.Put('Pulse{:03d}'.format(i),container)
        for i in xrange(noOfPulses,maxNumberOfPulses):
            # write nans to missing pulses (potentially waste of computation time and disk space)
            container = dataclasses.I3MapStringDouble()
            container['x'] = -1000#nanValue
            container['y'] = -1000#nanValue
            container['z'] = -1000#nanValue
            container['time'] = -1#nanValue
            container['width'] = -1#nanValue
            container['charge'] = -1#nanValue
            frame.Put('Pulse{:03d}'.format(i),container)
        return

    # def writePulses_old(frame):
    #     atwd = frame['CalibratedWaveformsHLCATWD']
    #     dataDict = dataclasses.I3MapStringDouble()
    #     pulseData = dataclasses.I3MapStringVectorDouble()
    #     for omkey in domPosDict.keys():
    #         # dataDict = dataclasses.I3MapStringDouble()
    #         prefix = "S{:02d}D{:02d}_".format(omkey.string,omkey.om)
    #         dataDict[prefix+'x'] = domPosDict[omkey].x
    #         dataDict[prefix+'y'] = domPosDict[omkey].y
    #         dataDict[prefix+'z'] = domPosDict[omkey].z
    #         pulseData[prefix[:-1]] = dataclasses.I3VectorDouble()
    #         domPulse = dataclasses.I3VectorDouble()
    #         domPulse.append(domPosDict[omkey].x)
    #         domPulse.append(domPosDict[omkey].y)
    #         domPulse.append(domPosDict[omkey].z)
    #         if omkey in atwd.keys():
    #             # ----------------------------------------------- Problem: more than one Waveform per Dom per Event
    #             # if len(atwd[omkey]) != 1:
    #             #     print atwd[omkey]
    #             #     raw_input()
    #                 # raise ValueError('More than one expected waveform in I3WaveformSeries found')
    #             pulseData[prefix[:-1]].append(domPosDict[omkey].x)
    #             pulseData[prefix[:-1]].append(domPosDict[omkey].y)
    #             pulseData[prefix[:-1]].append(domPosDict[omkey].z)

    #             wf = atwd[omkey][0]
    #             dataDict[prefix+'time'] = wf.time
    #             dataDict[prefix+'status'] = wf.status
    #             pulseData[prefix[:-1]].append(wf.time)
    #             pulseData[prefix[:-1]].append(wf.status)
    #             domPulse.append(wf.time)
    #             domPulse.append(wf.status)
    #             for sample in wf.waveform:
    #                 pulseData[prefix[:-1]].append(sample)
    #                 domPulse.append(sample)

    #         # frame[prefix[:-1]] = domPulse

    #             # frame[prefix+'wf'] = dataclasses.I3VectorDouble(wf.waveform)

    #             # for b,sample in zip(range(128),wf.waveform):
    #             #     dataDict[prefix + 'wf_{:03d}'.format(b)] = sample
    #         # else:
    #             # dataDict[prefix+'time'] =  float('nan')
    #             # dataDict[prefix+'status'] = float('nan')
    #             # emptyWF = []
    #             # frame[prefix+'wf'] = dataclasses.I3VectorDouble([])
    #             # for b in range(128):
    #             #     dataDict[prefix + 'wf_{:03d}'.format(b)] = float('nan')
    #         # frame.Put(prefix[:-1],dataDict)
    #     # frame.Put('pulseData',dataDict)
    #     return

    tray.AddModule(writePulses,'writePulses',Streams=[icetray.I3Frame.Physics])
    return

@icetray.traysegment
def getCubePulseData(tray, name, domPosDict,maxNumberOfPulses=300, nanValue = float('nan')):
    NoOfBinsX = 7
    NoOfBinsY = 7
    NoOfBinsZ = 15
    # constants
    xbinEdges = np.linspace(-400,400,NoOfBinsX-1)
    ybinEdges = np.linspace(-350,350,6,NoOfBinsY-1)
    zbinEdges = np.linspace(-444,454,NoOfBinsZ-1)

    def create_empty_array_of_shape(shape):
        if shape:
            return [create_empty_array_of_shape(shape[1:]) for i in xrange(shape[0])]
        else:
            return []

    def getBinFromCoord(coord,binEdges):
        binNumber = 0
        for edge in binEdges:
            if coord < edge:
                break
            binNumber += 1
        return binNumber

    def getBinFromPos(pos):
        xbin = getBinFromCoord(pos.x,xbinEdges)
        ybin = getBinFromCoord(pos.y,ybinEdges)
        zbin = getBinFromCoord(pos.z,zbinEdges)
        return xbin, ybin, zbin

    def writeCubePulses(frame):
        cubePulses = create_empty_array_of_shape((7,7,15))
        pulseMap = frame['I3SuperDST'].unpack()
        for domPulses in pulseMap:
            # collect pulses and put them into the correct bin
            omkey = domPulses[0]
            pos = domPosDict[omkey]
            xbin, ybin, zbin =  getBinFromPos(pos)
            cubePulses[xbin][ybin][zbin].extend([[pulse.time,pulse.charge] for pulse in domPulses[1]])

        for xbin in xrange(NoOfBinsX):
            for ybin in xrange(NoOfBinsY):
                for zbin in xrange(NoOfBinsZ):
                    if cubePulses[xbin][ybin][zbin]:
                        totalCharge = 0
                        noOfPulses = 0
                        maxCharge = 0
                        minTime = float('Inf')
                        maxTime = 0
                        sumOfTimes = 0
                        for pulse in cubePulses[xbin][ybin][zbin]:
                            totalCharge += pulse[1]
                            sumOfTimes += pulse[0]
                            if pulse[1] > maxCharge:
                                maxCharge  = pulse[1]
                            if pulse[0] < minTime:
                                minTime = pulse[0]
                            if pulse[0] > maxTime:
                                maxTime = pulse[0]
                            noOfPulses += 1
                        meanTime = sumOfTimes / noOfPulses
                        # loop again for var of time:
                        stdTime = 0
                        for pulse in cubePulses[xbin][ybin][zbin]:
                            stdTime += (pulse[0] - meanTime)**2
                        stdTime = math.sqrt(stdTime / noOfPulses)
                    else:
                        totalCharge = 0
                        noOfPulses = 0
                        maxCharge = 0
                        minTime = -1
                        maxTime = -1
                        meanTime = -1
                        stdTime = -1

                    # Wrtie Values to frame
                    container = dataclasses.I3MapStringDouble()
                    container['totalCharge'] = totalCharge
                    container['noOfPulses'] = noOfPulses
                    container['maxCharge'] = maxCharge
                    container['minTime'] = minTime
                    container['maxTime'] = maxTime
                    container['meanTime'] = meanTime
                    container['stdTime'] = stdTime
                    frame.Put('Cube{:01d}{:01d}{:02d}'.format(xbin,ybin,zbin),container)

        return

    tray.AddModule(writeCubePulses,'writeCubePulses',Streams=[icetray.I3Frame.Physics])
    return


@icetray.traysegment
def getData(tray, name, KeysToAdd='standard',dataAttributes=[[],[],[],[]]):
    if KeysToAdd == 'standard':
        KeysToAdd = ['CVMultiplicity','CVStatistics','FilterMask']
    # forbiddenKeys = ['id']

    def getOnlineL2(frame):
        onlineL2Keys = [ key for key in frame.keys() if 'OnlineL2' in key or 'PoleMuon' in key or key in KeysToAdd]
        dataDicts = [{},{},{},{}]
        for key in onlineL2Keys:
            if type(frame[key]) == dataclasses.I3Particle:
                dataDictsNew = handleI3Particle(frame,key)
            elif type(frame[key]) == dataclasses.I3FilterResultMap:
                dataDictsNew = handleI3FilterResultMap(frame,key)
            else:
                dataDictsNew = handleStandardType(frame,key)
            # update dataDicts
            for dataDict,dataDictNew in zip(dataDicts,dataDictsNew):
                dataDict.update(dataDictNew)

        # Write valuesToWriteForNan for missing values ----> Potentially dangerous!!
        # Alternative: define all datatypes as floats and save nans
        valuesToWriteForNan = [False,-1,float('nan'),'']
        for keys,dataDict,valueToWriteForNan in zip(dataAttributes,dataDicts,valuesToWriteForNan):
            for key in keys:
                if key not in dataDict.keys():
                    dataDict[key] = valueToWriteForNan

        # write dataDicts to frame [ignore dataString, I3MapStringString doesnt exit, dataString is constant]
        containers = [dataclasses.I3MapStringBool(),dataclasses.I3MapStringInt(),
                      dataclasses.I3MapStringDouble()]
        names = ['dataBool','dataInt','dataDouble']
        for name,dataDict,container in zip(names,dataDicts[:3],containers):
            container.update(dataDict)
            frame.Put(name,container)
        return


    def handleStandardType(frame,key):
        attributes = [att for att in dir(frame[key]) if not '__' in att]
        dataDictBool = {}
        dataDictInt = {}
        dataDictDouble = {}
        dataDictStr = {}
        for attr in attributes:
            if hasattr(frame[key],attr):
                value = getattr(frame[key],attr)
                if type(value) == bool:
                    dataDictBool[key+'_'+attr] = value
                elif type(value) == int:
                    dataDictInt[key+'_'+attr] = value
                elif type(value) == float:
                    dataDictDouble[key+'_'+attr] = value
                elif type(value) == str:
                    dataDictStr[key+'_'+attr] = value
                elif type(value) == dataclasses.I3Position:
                    handleI3Pos(dataDictDouble,key+'_'+attr,value)
                elif attr == 'CramerRaoStatus':
                    #ignore
                    continue
                elif attr == 'status' and issubclass(recclasses.CramerRaoParams.CramerRaoStatus,type(value)):
                    if value == recclasses.CramerRaoParams.CramerRaoStatus.OK:
                        dataDictBool[key+'_'+attr] = True
                    else:
                        dataDictBool[key+'_'+attr] = True
                else:
                    raise ValueError('handleStandardType: uknown value type for: ',value,type(value),'key:',key,'attr:',attr)
            else:
                raise ValueError('handleStandardType: Attribute Missing:', attr)
        return dataDictBool, dataDictInt, dataDictDouble, dataDictStr

    def addAttrToDict(dataDict,key,value,attributes):
        for attr in attributes:
            if hasattr(value,attr):
                dataDict[key+'_'+attr] = getattr(value,attr)
            else:
                raise ValueError('addAttrToDict: Attribute Missing:', attr)

    def handleI3Pos(dataDictDouble,key,value):
        attributes = ['x','y','z','r','phi','theta']
        addAttrToDict(dataDictDouble,key,value,attributes)

    def handleI3Dir(dataDictDouble,key,value):
        attributes = ['azimuth','zenith','x','y','z']
        addAttrToDict(dataDictDouble,key,value,attributes)

    def handleI3Particle(frame,key):
        dataDictBool = {}
        dataDictInt = {}
        dataDictDouble = {}
        dataDictStr = {}

        # bool attributes
        boolAttributes = ['is_cascade', 'is_neutrino', 'is_primary', 'is_top_shower', 'is_track']
        addAttrToDict(dataDictBool,key,frame[key],boolAttributes)
        # double attributes
        doubleAttributes = ['energy', 'speed', 'time', 'kinetic_energy', 'length']
        addAttrToDict(dataDictDouble,key,frame[key],doubleAttributes)
        # string attributes
        stringAttributes = ['shape_string', 'location_type_string', 'type_string']
        addAttrToDict(dataDictStr,key,frame[key],stringAttributes)
        # I3Position
        handleI3Pos(dataDictDouble,key+'_pos',frame[key].pos)
        # I3Dir
        handleI3Dir(dataDictDouble,key+'_dir',frame[key].dir)
        # Fit status
        if frame[key].fit_status_string == 'OK':
            dataDictBool[key+'_fit_status'] = True
        else:
            dataDictBool[key+'_fit_status'] = False

        return dataDictBool, dataDictInt, dataDictDouble, dataDictStr

    def handleI3FilterResultMap(frame,key):
        dataDictBool = {}
        for maskKey in frame[key].keys():
            dataDictBool[key+'_'+maskKey+'_condition_passed'] = frame[key][maskKey].condition_passed
            dataDictBool[key+'_'+maskKey+'_prescale_passed'] = frame[key][maskKey].prescale_passed
        return dataDictBool, {}, {}, {}


    tray.AddModule(getOnlineL2,'getOnlineL2',Streams=[icetray.I3Frame.Physics])
    return




@icetray.traysegment
def getBenchmarkData(tray, name, KeysToAdd='standard',dataAttributes=[[],[],[],[]]):
    if KeysToAdd == 'standard':
        KeysToAdd = ['CVMultiplicity','CVStatistics','FilterMask']
    # forbiddenKeys = ['id']

    def getOnlineL2(frame):
        onlineL2Keys = [ key for key in frame.keys() if 'OnlineL2' in key or 'PoleMuon' in key or key in KeysToAdd]
        dataDicts = [{},{},{},{}]
        for key in onlineL2Keys:
            if type(frame[key]) == dataclasses.I3Particle:
                dataDictsNew = handleI3Particle(frame,key)
            elif type(frame[key]) == dataclasses.I3FilterResultMap:
                dataDictsNew = handleI3FilterResultMap(frame,key)
            else:
                dataDictsNew = handleStandardType(frame,key)
            # update dataDicts
            for dataDict,dataDictNew in zip(dataDicts,dataDictsNew):
                dataDict.update(dataDictNew)

        # Write valuesToWriteForNan for missing values ----> Potentially dangerous!!
        # Alternative: define all datatypes as floats and save nans
        valuesToWriteForNan = [False,-1,float('nan'),'']
        for keys,dataDict,valueToWriteForNan in zip(dataAttributes,dataDicts,valuesToWriteForNan):
            for key in keys:
                if key not in dataDict.keys():
                    dataDict[key] = valueToWriteForNan

        # write dataDicts to frame [ignore dataString, I3MapStringString doesnt exit, dataString is constant]
        containers = [dataclasses.I3MapStringBool(),dataclasses.I3MapStringInt(),
                      dataclasses.I3MapStringDouble()]
        names = ['dataBool','dataInt','dataDouble']
        for name,dataDict,container in zip(names,dataDicts[:3],containers):
            container.update(dataDict)
            frame.Put(name,container)
        return


    tray.AddModule(getOnlineL2,'getOnlineL2',Streams=[icetray.I3Frame.Physics])
    return






@icetray.traysegment
def getDOMPulseData(tray, name,domPosDict, lengths, times, charges,method='Pulses'):

    def writeDOMPulses(frame,method='Pulses',numberOfBins=12,timeRange=[10000,20000]):
        timeBins = np.linspace(timeRange[0],timeRange[1],numberOfBins-1)
        timeBins = np.insert(timeBins,0,0)
        timeBins = np.append(timeBins,1e10)

        DOMPulseDataSummary = dataclasses.I3MapKeyVectorDouble()
        DOMPulseBinIndices = dataclasses.I3MapKeyVectorInt()
        DOMPUlseBinValues = dataclasses.I3MapKeyVectorDouble()

        if method == 'Pulses':
            if 'InIceDSTPulses' in frame.keys():
                pulseMap = frame['InIcePulses'].apply(frame)#.unpack()
                for domPulses in pulseMap:
                    dom_times = [pulse.time for pulse in domPulses[1]]
                    dom_charges = [pulse.charge for pulse in domPulses[1]]
                    hist,bin_edges = np.histogram(dom_times,weights=dom_charges, bins=timeBins)
                    binValuesList = []
                    binIndicesList = []
                    for i,charge in enumerate(hist):
                        if charge != 0:
                            binValuesList.append(charge)
                            binIndicesList.append(i)
                    # write to frame
                    DOMPulseDataSummary[domPulses[0]] = [sum(dom_charges),len(dom_times),max(dom_charges),min(dom_times),max(dom_times)]
                    DOMPUlseBinValues[domPulses[0]] = binValuesList
                    DOMPulseBinIndices[domPulses[0]] = binIndicesList

                frame['DOMPulseDataSummary'] = DOMPulseDataSummary
                frame['DOMPulseBinIndices'] = DOMPulseBinIndices
                frame['DOMPulseBinValues'] = DOMPUlseBinValues
        return


    def test(frame,lengths=[],times=[],charges=[]):
        if ('I3SuperDST' in frame.keys()) != ('InIceDSTPulses' in frame.keys()):
            raise ValueError('Ungliech')
        if 'I3SuperDST' in frame.keys() and 'CalibratedWaveforms' in frame.keys():
            pulseMap = frame['InIcePulses'].apply(frame)
            lengths.append( len(pulseMap) )
            total_dom_times = []
            total_dom_charges = []
            total_wf_charges = []
            total_wf_times = []
            total_wf_charges_atwd = []
            total_wf_times_atwd = []
            total_wf_charges_fadc = []
            total_wf_times_fadc = []
            total_wf_charges_slc = []
            total_wf_times_slc = []
            for i,domPulses in enumerate(pulseMap):
                # if dataclasses.get_most_energetic_muon(frame['I3MCTree']) != None:
                #     if np.log10(dataclasses.get_most_energetic_muon(frame['I3MCTree']).energy) > 6:
                # lengths.append(len(domPulses[1]))
                # print len(domPulses[1]), domPulses[1][0].time,domPulses[1][0].charge
                dom_times = [pulse.time for pulse in domPulses[1]]
                dom_charges = [pulse.charge for pulse in domPulses[1]]
                chargeSum_bin0 = 0
                times.extend(dom_times)
                charges.extend(dom_charges)
                if len(domPulses[1]) > 0:
                    container = dataclasses.I3MapStringDouble()
                    container['max'+str(i)] = 0
                    container['min'+str(i)] =-1

                    for pulse in domPulses[1]:
                        if pulse.time < 10000:
                            chargeSum_bin0 += pulse.charge
                    # print chargeSum_bin0
                    timeRange = [8000,17000]
                    numberOfBins = 1000
                    timeBins = np.linspace(timeRange[0],timeRange[1],numberOfBins+1)
                    timeBins = np.insert(timeBins,0,0)
                    timeBins = np.append(timeBins,1e10)
                    # print min(times),max(times)
                    hist,bin_edges = np.histogram(dom_times,weights=dom_charges, bins=timeBins)
                    # print hist
                    wf_times = []
                    wf_charges = []
                    wf_times_atwd = []
                    wf_times_slc = []
                    wf_times_fadc = []
                    wf_charges_atwd = []
                    wf_charges_slc = []
                    wf_charges_fadc = []
                    for wfs in frame['CalibratedWaveformsHLCATWD']:
                        # check if correct DOM
                        #wfs = [OMKey, ListOfWf]
                        if wfs[0] == domPulses[0]:
                            print len(wfs[1]),len(domPulses[1])
                            for wf in wfs[1]:
                                # check if there are two lists, one for FADC and one for ATWD
                                # wf.source: SLC, FADC, ATWD, ...
                                for j,charge in enumerate(wf.waveform):
                                    wf_times.append(wf.time + wf.bin_width*j)
                                    wf_charges.append(charge*1e10)
                                if wf.source == wf.SLC:
                                    wf_times_slc.append(wf.time + wf.bin_width*j)
                                    wf_charges_slc.append(charge*1e10)
                                if wf.source == wf.ATWD:
                                    wf_times_atwd.append(wf.time + wf.bin_width*j)
                                    wf_charges_atwd.append(charge*1e10)
                                if wf.source == wf.FADC:
                                    wf_times_fadc.append(wf.time + wf.bin_width*j)
                                    wf_charges_fadc.append(charge*1e10)

                    total_dom_times.extend(dom_times)
                    total_dom_charges.extend(dom_charges)
                    total_wf_times.extend(wf_times)
                    total_wf_charges.extend(wf_charges)
                    total_wf_times_atwd.extend(wf_times_atwd)
                    total_wf_charges_atwd.extend(wf_charges_atwd)
                    total_wf_times_fadc.extend(wf_times_fadc)
                    total_wf_charges_fadc.extend(wf_charges_fadc)
                    total_wf_times_slc.extend(wf_times_slc)
                    total_wf_charges_slc.extend(wf_charges_slc)
                    # fig = plt.figure()
                    # # plt.hist(dom_times,alpha=0.5,label='times',bins=timeBins)
                    # plt.hist(dom_times,weights=dom_charges,alpha=0.5,label='charge weighted times',bins=timeBins)
                    # # plt.hist(wf_times,alpha=0.5,label='WF: times',bins=timeBins)
                    # if wf_times:
                    #     plt.hist(wf_times,weights=wf_charges,alpha=0.5,label='WF:charge weighted times',bins=timeBins)
                    # plt.xlim(timeRange[0]-500,timeRange[1]+500)
                    # plt.legend()
                    # plt.yscale('log', nonposy='clip')
                    # plt.show()
                    # plt.close()
                    # fraime['DOMPulseDataTest'+str(i)] = container
            fig = plt.figure()
            plt.hist(total_dom_times,weights=total_dom_charges,alpha=0.5,label='total charge weighted times',bins=timeBins)
            plt.hist(total_wf_times,weights=total_wf_charges,alpha=0.95,label='WF:total charge weighted times',bins=timeBins)
            if total_wf_times_atwd:
                plt.hist(total_wf_times_atwd,weights=total_wf_charges_atwd,alpha=0.99,label='WF ATWD:total charge weighted times',bins=timeBins)
            if total_wf_times_fadc:
                plt.hist(total_wf_times_fadc,weights=total_wf_charges_fadc,alpha=0.99,label='WF FADC:total charge weighted times',bins=timeBins)
            if total_wf_times_slc:
                plt.hist(total_wf_times_slc,weights=total_wf_charges_slc,alpha=0.99,label='WF SLC:total charge weighted times',bins=timeBins)
            plt.xlim(timeRange[0]-500,timeRange[1]+500)
            plt.legend()
            plt.yscale('log', nonposy='clip')
            plt.show()
            plt.close()

        # print min(lengths),np.median(lengths),np.mean(lengths),max(lengths)
        # plt.hist(lengths)
        # plt.show()
        # plt.close()
        return

    if method == 'WF':
    # if True:
        def CheckRawData(frame):
            if frame.Has('InIceRawData'):
                return True
            return False

        tray.AddModule(CheckRawData,
                       'check-raw-data',
                       Streams=[icetray.I3Frame.Physics])

        tray.AddModule('I3WaveCalibrator',
                       'calibrator')(
            ('Launches', 'InIceRawData'),
            ('Waveforms', 'CalibratedWaveforms'),
            ('ATWDSaturationMargin', 123),
            # ('DOMsimulatorWorkArounds', False),
            ('FADCSaturationMargin', 0),
            ('Errata', 'OfflineInIceCalibrationErratas'),
            ('WaveformRange', 'CalibratedWaveformRanges'),
        )

        tray.AddModule('I3WaveformSplitter', 'waveformsplit')(
            ('Input', 'CalibratedWaveforms'),
            ('HLC_ATWD', 'CalibratedWaveformsHLCATWD'),
            ('HLC_FADC', 'CalibratedWaveformsHLCFADC'),
            ('SLC', 'CalibratedWaveformsSLC'),
            ('Force', True),
        )

    # tray.AddModule(test,'test',Streams=[icetray.I3Frame.Physics],lengths=lengths,times=times,charges=charges)
    tray.AddModule(writeDOMPulses,'writeDOMPulses',Streams=[icetray.I3Frame.Physics])
    return









@icetray.traysegment
def getCalibratedWaveforms(tray, name):
    def CheckRawData(frame):
        if frame.Has('InIceRawData'):
            return True
        return False

    tray.AddModule(CheckRawData,
                   'check-raw-data',
                   Streams=[icetray.I3Frame.Physics])

    tray.AddModule('I3WaveCalibrator',
                   'calibrator')(
        ('Launches', 'InIceRawData'),
        ('Waveforms', 'CalibratedWaveforms'),
        ('ATWDSaturationMargin', 123),
        ('FADCSaturationMargin', 0),
        ('Errata', 'OfflineInIceCalibrationErratas'),
        ('WaveformRange', 'CalibratedWaveformRanges'),
    )



class BenchMarkRecos(icetray.I3ConditionalModule):
    ''' Module to add reconstructed I3 particles to frame

    Parameters
    ----------

    Returns
    -------

    '''
    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("OutputKey","Save Data under this name.",'BenchMarkRecos')
        self.AddParameter("Dataset",
                "If ParticleKeys is empty this will load a predefined list of particle keys for the given dataset.",
                '11069')
        self.AddParameter("ParticleKeys","A list of I3Particle keys which will be stored to benchMarkRecos.",[])

    def Configure(self):
        self._output_key = self.GetParameter("OutputKey")
        self._dataset = self.GetParameter("Dataset")
        self._particle_keys = self.GetParameter("ParticleKeys")

        if self._particle_keys == []:
            if self._dataset == '11069':
                # Hardcode keys to add
                # [ Problem: all keys need to be known at first frame in order to save as a dict]
                # ToDo: Make this automatic
                self._particle_keys = [
                                         'PoleMuonLinefit',
                                         'PoleMuonLlhFit',
                                         'OnlineL2_PoleL2MPEFit_TruncatedEnergy_AllDOMS_Muon',
                                         'OnlineL2_PoleL2MPEFit_TruncatedEnergy_BINS_Neutrino',
                                         'OnlineL2_PoleL2BayesianFit_GeoSplit2',
                                         'OnlineL2_PoleL2BayesianFit_TimeSplit2',
                                         'OnlineL2_PoleL2SPE2it_GeoSplit1',
                                         'OnlineL2_PoleL2BayesianFit',
                                         'OnlineL2_LineFitTimeSplit1',
                                         'OnlineL2_PoleL2BayesianFit_GeoSplit1',
                                         'OnlineL2_PoleL2MPEFit_TruncatedEnergy_DOMS_Neutrino',
                                         'OnlineL2_PoleL2IpdfGConvolute_2it',
                                         'OnlineL2_PoleL2MPEFitMuE',
                                         'OnlineL2_PoleL2MPEFit',
                                         'OnlineL2_PoleL2MPEFit_MuEx',
                                         'OnlineL2_PoleL2MPEFit_TruncatedEnergy_AllBINS_Muon',
                                         'OnlineL2_PoleL2MPEFit_TruncatedEnergy_AllBINS_Neutrino',
                                         'OnlineL2_PoleL2MPEFit_TruncatedEnergy_AllDOMS_Neutrino',
                                         'OnlineL2_PoleL2MPEFit_TruncatedEnergy_BINS_Muon',
                                         'OnlineL2_PoleL2MPEFit_TruncatedEnergy_DOMS_Muon',
                                         'OnlineL2_PoleL2MPEFit_TruncatedEnergy_ORIG_Neutrino',
                                         'OnlineL2_PoleL2MPEFit_TruncatedEnergy_ORIG_Muon',

                                         'MPEFit',
                                         'MPEFitMuEX',
                                         'LineFit',
                                         'SPEFit2',
                                         'EHEOpheliaParticleSRT_ImpLF',

                                         # 'LineFitEHE',
                                         # 'SPEFit12EHE',
                                         # 'SPEFitSingleEHE',
                                         # 'EHEOpheliaParticleBTWSRT',
                                         # 'EHEOpheliaParticleSRT',

                                         # 'OnlineL2_PoleL2BayesianFit_TimeSplit1',
                                         # 'OnlineL2_PoleL2SPE2it_GeoSplit2',
                                         # 'OnlineL2_PoleL2SPE2it_TimeSplit1',
                                         # 'OnlineL2_PoleL2SPE2it_TimeSplit2',
                                         # 'OnlineL2_LineFitGeoSplit2',
                                         # 'OnlineL2_LineFitGeoSplit1',
                                         # 'OnlineL2_LineFitTimeSplit2',

                                         # 'CascadeImprovedLineFit_L2',
                                         # 'CascadeLineFit_L2',
                                         # 'CascadeLineFitSplit2_L2',
                                         # 'CascadeDipoleFit_L2',
                                         # 'CascadeToISplit1_L2',
                                         # 'CascadeLlhVertexFit_L2',
                                         # 'CascadeLlhVertexFitSplit1_L2',
                                         # 'CascadeLlhVertexFitSplit2_L2',
                                         # 'CascadeLineFitSplit1_L2',
                                         # 'CascadeLast_L2',
                                         # 'FiniteRecoFit',
                                         # 'HuberFit',
                                         # 'SPEFit2MuEX_FSS',
                                         # 'CascadeToISplit2_L2',
                                         # 'AtmCscdEnergyReco_L2',
                                         # 'SPEFitSingle',
                                         # 'CascadeLast_DC_Azimuth','SPEFit2_DC', 'DipoleFit_DC', 'SPEFitSingle_DC', 'CascadeLast_DC', 'ToI_DC', 'LineFit_DC'
                                         ]
            elif self._dataset in ['NugenPerfect','NugenNormal'] \
                or '2017OnlineL2' in self._dataset:

                self._particle_keys = [
                                         'PoleMuonLinefit',
                                         'PoleMuonLlhFit',
                                         'OnlineL2_MPEFit',
                                         'OnlineL2_SplineMPE',
                                         'OnlineL2_SPE2itFit',
                                         'OnlineL2_BestFit',
                                         'OnlineL2_BestFit_MuEx',
                                         'OnlineL2_SplineMPE_MuE',
                                         'OnlineL2_SplineMPE_MuEx',
                                         'OnlineL2_SplitTime1_Linefit',

                                         'OnlineL2_SplineMPE_TruncatedEnergy_AllBINS_Muon',
                                         'OnlineL2_SplineMPE_TruncatedEnergy_AllBINS_Neutrino',
                                         'OnlineL2_SplineMPE_TruncatedEnergy_AllDOMS_Muon',
                                         'OnlineL2_SplineMPE_TruncatedEnergy_AllDOMS_Neutrino',
                                         'OnlineL2_SplineMPE_TruncatedEnergy_BINS_Muon',
                                         'OnlineL2_SplineMPE_TruncatedEnergy_BINS_Neutrino',
                                         'OnlineL2_SplineMPE_TruncatedEnergy_DOMS_Muon',
                                         'OnlineL2_SplineMPE_TruncatedEnergy_DOMS_Neutrino',
                                         'OnlineL2_SplineMPE_TruncatedEnergy_ORIG_Muon',
                                         'OnlineL2_SplineMPE_TruncatedEnergy_ORIG_Neutrino',

                                         'OnlineL2_SplitGeo1_Linefit',
                                         'OnlineL2_SplitGeo1_SPE2itFit',
                                         'OnlineL2_SplitGeo2_Linefit',
                                         'OnlineL2_SplitGeo2_SPE2itFit',

                                         'OnlineL2_SplitTime1_SPE2itFit',
                                         'OnlineL2_SplitTime2_Linefit',
                                         'OnlineL2_SplitTime2_SPE2itFit',
                                         ]
            elif self._dataset == '11069_PS_final':
                self._particle_keys = [
                                        'SplineMPEmod',
                                        'MuEXAngular4',
                                        'SplineMPETruncatedEnergy_SPICEMie_ORIG_Muon',
                                        'SPEFitSingle_HV',
                                        'SplineMPEBootstrap',
                                        'BestTrack',
                                        'SPEFitSingle_TWHV',
                                        'SPEFit2_HV',
                                        'MPEFitHighNoise',
                                        'SplineMPEParaboloid',
                                        'SplineMPETruncatedEnergy_SPICEMie_AllDOMS_Muon',
                                        'SplineMPETruncatedEnergy_SPICEMie_AllBINS_Muon',
                                        'SplineMPETruncatedEnergy_SPICEMie_AllDOMS_Neutrino',
                                        'SplineMPE',
                                        'SplineMPETruncatedEnergy_SPICEMie_AllBINS_Neutrino',
                                        'LineFit_HV',
                                        'SplineMPETruncatedEnergy_SPICEMie_DOMS_Muon',
                                        'SplineMPETruncatedEnergy_SPICEMie_DOMS_Neutrino',
                                        'SplineMPEMuEXDifferential',
                                        'SplineMPETruncatedEnergy_SPICEMie_BINS_Neutrino',
                                        'SplineMPETruncatedEnergy_SPICEMie_BINS_Muon',
                                        'MPEFit_HV',
                                        'SplineMPETruncatedEnergy_SPICEMie_ORIG_Neutrino',
                                        'MPEFitParaboloid',
                                        'MPEFit_TWHV',
                                        'LineFit_TWHV',
                                        'SplineMPEBootstrapMean',
                                        'SPEFit2_TWHV',
                                        'SplineMPEBootstrapFinal',
                                        ]

            elif self._dataset[:3] == 'NuE' or self._dataset[:7] == 'cascade':
                self._particle_keys = [
                                        'L5MonopodFit4',
                                        'L3_MonopodFit4'
                                        'MCPrimary',
                                        ]

            else:
                raise ValueError('BenchMarkRecos: Benchmark keys for dataset {} are not defined.'.format(self._dataset))



    def Physics(self, frame):
        # # get all I3Particles in frame
        # # particle_keys = [key for key in frame.keys() if key[:2]!='MC' and type(frame[key]) == dataclasses.I3Particle]
        # particle_keys = []
        # for key in frame.keys():
        #     try:
        #         if key[:2]!='MC' and type(frame[key]) == dataclasses.I3Particle:
        #             particle_keys.append(key)
        #     except:
        #         continue

        # create container
        benchMarkRecos = dataclasses.I3MapStringDouble()

        keys_in_frame = set(frame.keys())

        for key in self._particle_keys:
            if key in keys_in_frame:
                benchMarkRecos[key+'_X'] = frame[key].pos.x
                benchMarkRecos[key+'_Y'] = frame[key].pos.y
                benchMarkRecos[key+'_Z'] = frame[key].pos.z
                benchMarkRecos[key+'_Azimuth'] = frame[key].dir.azimuth
                benchMarkRecos[key+'_Zenith'] = frame[key].dir.zenith
                benchMarkRecos[key+'_Energy'] = frame[key].energy
                benchMarkRecos[key+'_Time'] = frame[key].time
                benchMarkRecos[key+'_Length'] = frame[key].length
            else:
                benchMarkRecos[key+'_X'] = float('nan')
                benchMarkRecos[key+'_Y'] = float('nan')
                benchMarkRecos[key+'_Z'] = float('nan')
                benchMarkRecos[key+'_Azimuth'] = float('nan')
                benchMarkRecos[key+'_Zenith'] = float('nan')
                benchMarkRecos[key+'_Energy'] = float('nan')
                benchMarkRecos[key+'_Time'] = float('nan')
                benchMarkRecos[key+'_Length'] = float('nan')


        keys_to_add = []

        # Add Cramer Rao values
        if not (self._dataset[:3] == 'NuE' or self._dataset[:7] == 'cascade'):
            keys_to_add.extend(['OnlineL2_SplineMPE_CramerRao_cr_zenith',
                                'OnlineL2_SplineMPE_CramerRao_cr_azimuth'])

        # Add is_cascade, is_track, is_HESE
        if self._dataset == 'cascade_labels':
            keys_to_add.extend(['IsTrack_true',
                                'IsTrack_reco',
                                'IsCascade',
                                'IsCascade_reco',
                                'IsHESE_ck',
                                'IsUpgoingMuon',
                                'IsHese',
                                'IsCascade_true'])

            # add neutrino type
            if frame['MCPrimary'].type_string[:3] == 'NuE':
                is_nue = True
                is_numu = False
                is_nutau = False
            elif frame['MCPrimary'].type_string[:4] == 'NuMu':
                is_nue = False
                is_numu = True
                is_nutau = False
            elif frame['MCPrimary'].type_string[:5] == 'NuTau':
                is_nue = False
                is_numu = False
                is_nutau = True
            else:
                raise ValueError('Expected one NuE, NuMu or NuTau')

            benchMarkRecos['is_NuE'] = is_nue
            benchMarkRecos['is_NuMu'] = is_numu
            benchMarkRecos['is_NuTau'] = is_nutau


        for key in keys_to_add:
            if key in keys_in_frame:
                benchMarkRecos[key] = frame[key].value
            else:
                benchMarkRecos[key] = float('nan')






        frame.Put(self._output_key,benchMarkRecos)
        self.PushFrame(frame)








class DOMPulseData(icetray.I3ConditionalModule):
    ''' Module to add DOMPulseData to frame

    Parameters
    ----------
        PerformLog : bool.
            True: preforms log10 on relevant data attributes before saving.
                [This seems to increase needed diskspace of file]
    Returns
    -------

    '''
    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("PulseMapString","Name of pulse map to use.",'InIcePulses')
        self.AddParameter("CalibratedWaveforms","Name of CalibratedWaveforms to use.",'CalibratedWaveforms')
        self.AddParameter("TimeRange","TimeRange in which to bin the pulses/waveforms.",[8000,17000])
        self.AddParameter("NumberOfBins","Number of bins for the pulses/waveforms.",12)
        self.AddParameter("NumberOfSummaryBins","Number of summary items for the pulses/waveforms.",5)
        self.AddParameter("SummaryType","Which summary type should be used: 'PulseSummary','TrackCherenkov'.",'PulseSummary')
        self.AddParameter("Method","Method: choose 'Pulses','WF_ALL','WF_ATWD','WF_FADC','WF_SLC'.",'Pulses')
        self.AddParameter("OutputKey","Save Data under this name.",'DOMPulse')
        self.AddParameter("PerformLog","True: perfom log10(charges) and for other relevant data.",True)
        self.AddParameter("TimeWindowSize","If DynamicTimeRange is true a time window of this size [ns] will be calculated",6000)
        self.AddParameter("DynamicTimeRange","If True TimeRange will not be used, instead a dynamic time range will be calculated",True)
        self.AddParameter("ParticleTrackKey","If SummaryType is TrackCherenkov, this track hypothesis will be used. If None the muon from the primary particle will be used.",None)
        self.AddParameter("PrimaryKey","Frame key of the primary particle.",'MCPrimary')
        self.AddParameter("BareMuAmplitudeSpline","BareMuAmplitudeSpline to use if SummaryType is Likelihood",'/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/InfBareMu_mie_abs_z20a10_V2.fits')
        self.AddParameter("BareMuTimingSpline","BareMuTimingSpline to use if SummaryType is Likelihood",'/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/InfBareMu_mie_prob_z20a10_V2.fits')
        self.AddParameter("LikelihoodTMin","Timewindw minimum (cherenkov time - LikelihoodTMin) to use if SummaryType is Likelihood",-200)
        self.AddParameter("LikelihoodTMax","Timewindw maximum (cherenkov time + LikelihoodTMax) to use if SummaryType is Likelihood",1000)
        self.AddParameter("NumberOfLikelihoodBins","Number of Likelihood bins to use if SummaryType is Likelihood",20)

    def Configure(self):
        self._pulse_map_string = self.GetParameter("PulseMapString")
        self._waveforms_string = self.GetParameter("CalibratedWaveforms")
        self._time_range = self.GetParameter("TimeRange")
        self._no_of_bins = self.GetParameter("NumberOfBins")
        self._no_of_summary_bins = self.GetParameter("NumberOfSummaryBins")
        self._method = self.GetParameter("Method")
        self._output_key = self.GetParameter("OutputKey")
        self._perform_log = self.GetParameter("PerformLog")
        self._time_window_size = self.GetParameter("TimeWindowSize")
        self._dynamic_time_range = self.GetParameter("DynamicTimeRange")
        self._summary_type = self.GetParameter("SummaryType")
        self._particle_track_key = self.GetParameter("ParticleTrackKey")
        self._primary_key = self.GetParameter("PrimaryKey")
        self._bareMuAmplitudeSpline = self.GetParameter("BareMuAmplitudeSpline")
        self._bareMuTimingSpline = self.GetParameter("BareMuTimingSpline")

        if self._summary_type in ['Likelihood','PulseSummaryV2_llh']:
            self._ps = photonics_service.I3PhotoSplineService(
                self._bareMuAmplitudeSpline,
                self._bareMuTimingSpline,
                0) #PreJitter
            self._num_likelihood_bins = self.GetParameter("NumberOfLikelihoodBins")
            self._likelihood_t_min = self.GetParameter("LikelihoodTMin")
            self._likelihood_t_max = self.GetParameter("LikelihoodTMax")


    def Geometry(self, frame):
        if self._summary_type in ['TrackCherenkov','Likelihood','PulseSummaryV2_llh']:
            geoMap = frame['I3Geometry'].omgeo
            domPosDict = {i[0]:i[1].position for i in geoMap if i[1].omtype.name == 'IceCube'}
            self._dom_pos_dict = domPosDict

        self.PushFrame(frame)

    def Physics(self, frame):
        self.writeDOMPulses(frame)
        self.PushFrame(frame)

    def getTimeRange(self, charges, times, time_window_size=6000):
        max_charge_sum = 0
        start_t = 9000
        for t in range(int(np.nanmin(times)//1000)*1000,int(np.nanmax(times)//1000)*1000 - time_window_size ,500):
            indices_smaller = t < times
            indices_bigger = times < t + time_window_size
            indices = np.logical_and(indices_smaller,indices_bigger)
            charge_sum = np.sum(charges[indices])
            if charge_sum > max_charge_sum:
                max_charge_sum = charge_sum
                start_t = t
        return [start_t, start_t + time_window_size]

    def get_likelihood_values(self, particle, omkey, min_pulse_time, charge_sum):

        if particle == None:
            return None

        # Tell PhotonicsService the position where we'd like to query the light distributions
        pos = self._dom_pos_dict[omkey]
        self._ps.SelectModuleCoordinates(pos.x, pos.y, pos.z)

        """
        Set up a light source. Photonics deals with a special-purpose type called
        PhotonicsSource, but we can make on from a vanilla I3Particle. In this case,
        we make 0.01 GeV infinete muon track.
        """
        x = particle.pos.x
        y = particle.pos.y
        z = particle.pos.z
        zenith = particle.dir.zenith / I3Units.deg
        azimuth = particle.dir.azimuth / I3Units.deg
        length = 0. # infinite muon track
        energy = 0.01
        sourceType = 0 # muon
        source = photonics_service.PhotonicsSource(
            x, y, z, zenith, azimuth, 1, length, energy, sourceType)

        """
        Select the source, returning 3 numbers, of which only 2 are important.
        mean_npe: the total integrated charge the DOM should collect, given the event hypothesis
        geotime: the minimum time it takes the photon to propagate from the particle track vertex to the DOM
        """
        mean_npe, emission_distance, geotime = self._ps.SelectSource(source)

        # get spline MPE llh
        tres = min_pulse_time - geotime - particle.time

        npe = max(1., np.floor(charge_sum))

        pdf = self._ps.GetProbabilityDensity(tres)
        if pdf <= 0.:
            llh = 0.
        else:
            times_edges = np.array([-10000., tres])
            integrated_pdf = self._ps.GetProbabilityQuantiles(times_edges, 0, False)

            llh = np.log( max( pdf*(1.-integrated_pdf[0])**(npe-1.),1e-8) )
            if not np.isfinite(llh):
                print (1.-integrated_pdf[0])
                print (1.-integrated_pdf[0])**(npe-1.)
                print pdf*(1.-integrated_pdf[0])**(npe-1.)
                print('Setting llh from {} to 0. pdf: {}, npe: {}, integrated_pdf[0]: {}'.format(llh, pdf, npe,integrated_pdf[0]))
                llh = 0.

        # if the coords are out of the table range, it returns meanPEs=-1
        return llh, mean_npe, emission_distance, tres, geotime

    def get_likelihood_quantile_values(self, particle, omkey, bin_edges, min_pulse_time, charge_sum):

        llh, mean_npe, emission_distance, tres, geotime = self.get_likelihood_values(
                                                                particle=particle,
                                                                omkey=omkey,
                                                                min_pulse_time=min_pulse_time,
                                                                charge_sum=charge_sum)

        # if the coords are out of the table range, it returns meanPEs=-1
        if mean_npe >= 0:

            return self._ps.GetProbabilityQuantiles(bin_edges - geotime - particle.time,0, False), llh
        else:
            return None, llh


    def get_cherenkov_information(self, track, omkey):
        dom_position = self._dom_pos_dict[omkey]
        dom_direction = dataclasses.I3Direction(0,0,-1)
        dom_direction_x = dataclasses.I3Direction(0,1,0)

        if track != None:
            cherenkov_distance = I3Calculator.cherenkov_distance(track, dom_position)
            cherenkov_time = I3Calculator.cherenkov_time(track, dom_position)
            approach_angle = I3Calculator.cherenkov_approach_angle(track, dom_position, dom_direction)
            approach_angle_azimuth = I3Calculator.cherenkov_approach_angle(track, dom_position, dom_direction_x)
            abs_cherenkov_time = cherenkov_time + track.time
            cherenkov_position = I3Calculator.cherenkov_position(track, dom_position)
            cherenkov_pos_time = cherenkov_time - cherenkov_distance /(.227201)
        else:
            cherenkov_distance = -1.
            cherenkov_time = -1.
            approach_angle = -1.
            approach_angle_azimuth = -1.
            abs_cherenkov_time = 0.
            cherenkov_position = dataclasses.I3Position(0,0,0)
            cherenkov_pos_time = 0.

        if not np.isfinite(cherenkov_time) or not np.isfinite(cherenkov_distance) or not np.isfinite(approach_angle):
            # DOM is not in range of cherenkov cone
            cherenkov_distance = -1.
            cherenkov_time = -1.
            approach_angle = -1.
            abs_cherenkov_time = 0.
            approach_angle_azimuth = -1.
            cherenkov_pos_time = 0.

        cherenkov_info = {
                        'cherenkov_distance' : cherenkov_distance,
                        'cherenkov_time' : cherenkov_time,
                        'approach_angle' : approach_angle,
                        'approach_angle_azimuth' : approach_angle_azimuth,
                        'abs_cherenkov_time' : abs_cherenkov_time,
                        'cherenkov_position' : cherenkov_position,
                        'cherenkov_pos_time' : cherenkov_pos_time,
                        }
        return cherenkov_info

    def writeDOMPulses(self, frame):

        # get track (I3Particle) if neeeded
        if self._summary_type in ['TrackCherenkov','Likelihood','PulseSummaryV2_llh']:
            if self._particle_track_key != None:
                # take given track hypothesis (I3Particle)
                self._particle_track = frame[self._particle_track_key]
            else:
                # # take MC Primary particle
                # self._particle_track = frame[self._primary_key]

                # take MC muon from primary (I3Particle)
                self._particle_track = icecube_labels.get_next_muon_daughter_of_nu(frame, frame[self._primary_key])

        # calculate time range
        # [This can be added in loop further below in order to reduce redundancy]
        if self._dynamic_time_range:
            if self._method != 'Pulses':
                print '\033[93mWARNING: Using '+self._pulse_map_string+' to calculate time range\033[0m'
                # raise ValueError('Dynamic time range not yet implemented for WF method')

            if self._pulse_map_string in frame.keys():

                # get pulses defined by pulse_map_string
                pulseMap = frame[self._pulse_map_string]
                if isinstance(pulseMap, dataclasses.I3RecoPulseSeriesMapMask):
                    pulseMap = pulseMap.apply(frame)

                charges = []
                times = []
                for key in pulseMap.keys():
                    for pulse in pulseMap[key]:
                        charges.append(pulse.charge)
                        times.append(pulse.time)
                charges = np.asarray(charges)
                times = np.asarray(times)
                self._time_range = self.getTimeRange(charges,times,time_window_size=self._time_window_size)

        if self._no_of_bins > 0:
            timeBins = np.linspace(self._time_range[0],self._time_range[1],self._no_of_bins-1)
            timeBins = np.insert(timeBins,0,0)
            timeBins = np.append(timeBins,1e10)

        DOMPulseDataSummary = dataclasses.I3MapKeyVectorDouble()
        DOMPulseBinIndices = dataclasses.I3MapKeyVectorInt()
        DOMPUlseBinValues = dataclasses.I3MapKeyVectorDouble()

        # Use pulses in pulse_map_string
        if self._method == 'Pulses':
            if self._pulse_map_string in frame.keys():
                # get pulses defined by pulse_map_string
                pulseMap = frame[self._pulse_map_string]
                if isinstance(pulseMap, dataclasses.I3RecoPulseSeriesMapMask):
                    pulseMap = pulseMap.apply(frame)

                for domPulses in pulseMap:
                    dom_times = [pulse.time for pulse in domPulses[1]]

                    # abort early if no pulses exist for DOM key
                    if not dom_times:
                        continue

                    dom_charges = [pulse.charge for pulse in domPulses[1]]
                    binValuesList = []
                    binIndicesList = []
                    if self._no_of_bins > 0:
                        hist,bin_edges = np.histogram(dom_times,weights=dom_charges, bins=timeBins)
                        for i,charge in enumerate(hist):
                            if charge != 0:
                                value = charge
                                if self._perform_log: # --------- Perform log
                                    value = np.log10(1 + value)
                                binValuesList.append(value)
                                binIndicesList.append(i)
                    DOMPUlseBinValues[domPulses[0]] = binValuesList
                    DOMPulseBinIndices[domPulses[0]] = binIndicesList

                    # get summary data
                    if self._no_of_summary_bins > 0:

                        if self._summary_type == 'PulseSummary':
                        #--------------------
                        # Pulse Summary
                        #--------------------
                            rel_dom_times = np.asarray(dom_times) - self._time_range[0]
                            if self._perform_log:
                                summary_data = [np.log10(1e-4 + sum(dom_charges)),
                                                np.log10(1e-4 + len(rel_dom_times)),
                                                np.log10(1e-4 + max(dom_charges)),
                                                min(rel_dom_times),max(rel_dom_times),
                                                np.mean(rel_dom_times),np.std(rel_dom_times)]
                            else:
                                summary_data = [sum(dom_charges),len(rel_dom_times),
                                                max(dom_charges),min(rel_dom_times),
                                                max(rel_dom_times),np.mean(rel_dom_times),
                                                np.std(rel_dom_times)]
                            DOMPulseDataSummary[domPulses[0]] = summary_data[:self._no_of_summary_bins]

                        elif self._summary_type in[
                                                    'PulseSummaryV2',
                                                    'PulseSummaryV2_clipped',
                                                    'PulseSummaryV2_llh',
                                                    ]:
                        #--------------------
                        # Pulse Summary V2
                        #--------------------
                            # assume dom_times are sorted
                            assert (np.argsort(dom_times, kind='mergesort') == range(len(dom_times))).all(), 'dom_times are not ordered!'
                            assert len(dom_times) >= 1, 'Assumes non empty list'

                            # calculate necessary quantities
                            rel_dom_times = np.asarray(dom_times) - self._time_range[0]
                            dom_charges = np.asarray(dom_charges)

                            #--------------------------------------
                            # clip pulses outside of [-5000, 14000]
                            # if PulseSummaryV2_clipped is used
                            #--------------------------------------
                            if self._summary_type == 'PulseSummaryV2_clipped':
                                clip_mask = rel_dom_times >= -5000
                                clip_mask = np.logical_and(clip_mask,
                                                        rel_dom_times <= 14000
                                                        )
                                rel_dom_times = rel_dom_times[clip_mask]
                                dom_charges = dom_charges[clip_mask]

                                # abort early if no pulses exist for DOM key
                                if len(rel_dom_times) == 0:
                                    continue
                            #--------------------------------------

                            dom_charge_sum = sum(dom_charges)
                            rel_dom_times_first = rel_dom_times[0]
                            rel_dom_times_last = rel_dom_times[-1]

                            charge_weighted_mean_time = np.average(rel_dom_times, weights=dom_charges)
                            charge_weighted_std_time = weighted_std(rel_dom_times, weights=dom_charges)
                            charge_weighted_quantile20_time = weighted_quantile(rel_dom_times, weights=dom_charges, quantile=0.2)
                            charge_weighted_quantile50_time = weighted_quantile(rel_dom_times, weights=dom_charges, quantile=0.5)


                            mask_100ns_interval = rel_dom_times - rel_dom_times_first < 100
                            mask_500ns_interval = rel_dom_times - rel_dom_times_first < 500

                            dom_charge_sum_100ns = np.sum(dom_charges[mask_100ns_interval])
                            dom_charge_sum_500ns = np.sum(dom_charges[mask_500ns_interval])

                            # #-------
                            # # DEBUG
                            # #-------
                            # # remove creation of domposdict in Geometry() !!!!!!!!!!!
                            # self._particle_track = icecube_labels.get_next_muon_daughter_of_nu(frame, frame[self._primary_key])
                            # cherenkov_info = self.get_cherenkov_information( self._particle_track, omkey=domPulses[0])

                            # print 't_diff:',rel_dom_times - rel_dom_times_first
                            # print '100ns',dom_charge_sum_100ns
                            # print '500ns',dom_charge_sum_500ns
                            # print 'Totoal',dom_charge_sum
                            # print 'weighted std time', charge_weighted_std_time
                            # print 'weighted mean time', charge_weighted_mean_time
                            # print 'quatnile 20 time', charge_weighted_quantile20_time
                            # print 'quatnile 50 time', charge_weighted_quantile50_time
                            # print 'first pulse time', rel_dom_times_first
                            # print 'last pulse time', rel_dom_times_last
                            # print 'ch time: ', cherenkov_info['abs_cherenkov_time'] - self._time_range[0]



                            # if len(rel_dom_times) > 15:
                            #     plt.hist(rel_dom_times, weights=dom_charges, bins=100)
                            #     plt.plot((rel_dom_times_first, rel_dom_times_first),(0,-0.5),'b-')
                            #     plt.plot((rel_dom_times_last, rel_dom_times_last),(0,-0.5),'b-')
                            #     plt.plot((charge_weighted_quantile20_time, charge_weighted_quantile20_time),(0,-0.3),'g-',label='Quantile 20')
                            #     plt.plot((charge_weighted_quantile50_time, charge_weighted_quantile50_time),(0,-0.3),'g-',label='Quantile 50')
                            #     plt.plot((charge_weighted_mean_time, charge_weighted_mean_time),(0,-0.3),'p-',label='Weighted Mean')
                            #     plt.legend(loc='best')
                            #     plt.show()
                            # #-------

                            if self._perform_log:
                                summary_data = [np.log10(1.0 + dom_charge_sum),
                                                np.log10(1.0 + dom_charge_sum_500ns),
                                                np.log10(1.0 + dom_charge_sum_100ns),
                                                rel_dom_times_first,
                                                charge_weighted_quantile20_time,
                                                charge_weighted_quantile50_time,
                                                rel_dom_times_last,
                                                charge_weighted_mean_time,
                                                charge_weighted_std_time,
                                                ]
                            else:
                                summary_data = [dom_charge_sum,
                                                dom_charge_sum_500ns,
                                                dom_charge_sum_100ns,
                                                rel_dom_times_first,
                                                charge_weighted_quantile20_time,
                                                charge_weighted_quantile50_time,
                                                rel_dom_times_last,
                                                charge_weighted_mean_time,
                                                charge_weighted_std_time,
                                                ]

                            #--------
                            if self._summary_type == 'PulseSummaryV2_llh':
                            #--------
                                #--------
                                # Get likelihood values
                                #--------
                                llh, mean_npe, emission_distance, tres, geotime = self.get_likelihood_values(
                                                                particle=self._particle_track,
                                                                omkey=domPulses[0],
                                                                min_pulse_time=dom_times[0],
                                                                charge_sum=dom_charge_sum)
                                #--------
                                if self._perform_log:
                                    summary_data.extend([np.log10(1.1 + mean_npe), # mean_npe is -1 if out of table range
                                                           emission_distance,
                                                           tres,
                                                           llh
                                                    ])
                                else:
                                    summary_data.extend([  mean_npe,
                                                           emission_distance,
                                                           tres,
                                                           llh
                                                    ])

                            DOMPulseDataSummary[domPulses[0]] = summary_data[:self._no_of_summary_bins]

                        elif self._summary_type == 'Likelihood':
                        #--------------------
                        # Likelihood Summary
                        #--------------------

                            # get cherenkov information
                            cherenkov_info = self.get_cherenkov_information( self._particle_track, omkey=domPulses[0])
                            rel_cherenkov_pos = cherenkov_info['cherenkov_position'] - self._dom_pos_dict[domPulses[0]]
                            rel_cherenkov_dir = dataclasses.I3Direction(rel_cherenkov_pos)

                            #----------------------
                            # make binning
                            #----------------------
                            t_min = cherenkov_info['abs_cherenkov_time'] + self._likelihood_t_min
                            t_max = cherenkov_info['abs_cherenkov_time'] + self._likelihood_t_max
                            bin_edges = np.linspace(t_min, t_max ,self._num_likelihood_bins-1)
                            bin_edges = np.insert(bin_edges,0,-10000)
                            bin_edges = np.append(bin_edges,1e10)

                            #----------------------
                            # summary pulse data
                            #----------------------
                            rel_dom_times = np.asarray(dom_times) - cherenkov_info['abs_cherenkov_time']
                            charge_sum = sum(dom_charges)
                            if self._perform_log:
                                summary_data = [np.log10(1e-4 + charge_sum),
                                                np.log10(1e-4 + len(rel_dom_times)),
                                                np.log10(1e-4 + max(dom_charges)),
                                                min(rel_dom_times),max(rel_dom_times),
                                                np.mean(rel_dom_times),np.std(rel_dom_times)]
                            else:
                                summary_data = [charge_sum,len(rel_dom_times),
                                                max(dom_charges),min(rel_dom_times),
                                                max(rel_dom_times),np.mean(rel_dom_times),
                                                np.std(rel_dom_times)]

                            #----------------------
                            # get likelihood values
                            #----------------------
                            min_pulse_time = min(dom_times)
                            pdf_bin_heights, llh_value = self.get_likelihood_quantile_values(self._particle_track,
                                                                            omkey=domPulses[0],
                                                                            bin_edges=bin_edges,
                                                                            min_pulse_time=min_pulse_time,
                                                                            charge_sum=charge_sum,
                                                                            )
                            if pdf_bin_heights is None:
                                dom_diff = np.zeros(self._num_likelihood_bins)
                            else:
                                dom_bin_heights = np.histogram(dom_times,weights=dom_charges, bins=bin_edges)[0]

                                # # Normalization to max bin height = 1
                                # dom_bin_heights /= max(dom_bin_heights[1:-1])
                                # pdf_bin_heights /= max(pdf_bin_heights[1:-1])

                                # Normalization to sum = 1
                                dom_bin_heights /= sum(dom_bin_heights)
                                pdf_bin_heights /= sum(pdf_bin_heights)

                                # diff
                                dom_diff = dom_bin_heights - pdf_bin_heights

                            #----------------------
                            # Add values to llh data
                            #----------------------

                            llh_data = []

                            # order in importance
                            llh_data.append(llh_value) # spline mpe llh
                            llh_data.append(np.sum(np.abs(dom_diff))) # sum of absolute llh diffs
                            llh_data.append(summary_data[0]) # charge sum
                            llh_data.append(summary_data[3]) # min rel pulse time
                            llh_data.append(min_pulse_time) # min absolute pulse time
                            #----- llh [5 values]

                            llh_data.extend(summary_data[4:7]) # max, mean, std rel. time [3 values]
                            if len(dom_times) > 1:
                                llh_data.append(rel_dom_times[1]) # 2nd min rel pulse time
                                llh_data.append(dom_times[1]) # 2nd min absolute pulse time
                            else:
                                llh_data.append(0.) # 2nd min rel pulse time
                                llh_data.append(0.) # 2nd min absolute pulse time
                            llh_data.append(cherenkov_info['cherenkov_distance'])
                            #----- llh [11 values]
                            llh_data.extend(dom_diff)
                            #----- llh [31 values]

                            llh_data.extend(summary_data[1:3]) # len and max charge [2 values]
                            llh_data.append(rel_cherenkov_dir.x)
                            llh_data.append(rel_cherenkov_dir.y)
                            llh_data.append(rel_cherenkov_dir.z)
                            llh_data.append(cherenkov_info['abs_cherenkov_time'])
                            llh_data.append(cherenkov_info['approach_angle'])
                            llh_data.append(cherenkov_info['approach_angle_azimuth'])
                            llh_data.append(cherenkov_info['cherenkov_pos_time'])
                            #----- llh [40 values]

                            DOMPulseDataSummary[domPulses[0]] = llh_data[:self._no_of_summary_bins]


                        elif self._summary_type == 'TrackCherenkov':
                        #--------------------
                        # Track Cherenkov
                        #--------------------
                            cherenkov_info = self.get_cherenkov_information( self._particle_track, omkey=domPulses[0])
                            cherenkov_position = cherenkov_info['cherenkov_position']

                            summary_data = [
                                            cherenkov_position.x,
                                            cherenkov_position.y,
                                            cherenkov_position.z,
                                            cherenkov_info['cherenkov_pos_time'],
                                            # cherenkov_distance,
                                            # cherenkov_time,
                                            # approach_angle,
                                            # sum(dom_charges),
                                            # len(dom_times),
                                            # approach_angle_azimuth,
                                            ]
                            # if not np.isfinite(summary_data).all():
                            #     print summary_data
                            DOMPulseDataSummary[domPulses[0]] = summary_data[:self._no_of_summary_bins]

                        else:
                            raise ValueError('SummaryType must be "PulseSummary","TrackCherenkov","Likelihood" but is: {}'.format(self._summary_type))

        # use waveforms in CalibratedWaveforms
        elif 'WF' in self._method:
            if self._waveforms_string in frame.keys():
                for omkey in frame[self._waveforms_string].keys():

                    # get waveform lists
                    if self._method == 'WF_FADC':
                        wfs = [wf for wf in frame[self._waveforms_string][key]
                                if wf.source == wf.FADC ]
                    elif self._method == 'WF_ATWD':
                        wfs = [wf for wf in frame[self._waveforms_string][key]
                                if wf.source == wf.ATWD ]
                    elif self._method == 'WF_SLC':
                        wfs = [wf for wf in frame[self._waveforms_string][key]
                                if wf.source == wf.SLC ]
                    elif self._method == 'WF_ALL':
                        wfs = [wf for wf in frame[self._waveforms_string][key] ]
                    else:
                        ValueError('Method:',self._method,'does not exist.')

                    wf_times = []
                    wf_charges = []
                    for wf in wfs:
                        for j,charge in enumerate(wf.waveform):
                            wf_times.append(wf.time + wf.bin_width*j)
                            # FADC 10^-10
                            # ATWD 10^-09
                            # SLC  10^-12
                            # multiply by 1e10 to not loose precision
                            # when saving or handling with float32
                            wf_charges.append(charge*1e10)

                    binValuesList = []
                    binIndicesList = []
                    if self._no_of_bins > 0:
                        hist,bin_edges = np.histogram(wf_times,weights=wf_charges, bins=timeBins)
                        for i,charge in enumerate(hist):
                            if charge != 0:
                                value = charge
                                if self._perform_log: # --------- Perform log
                                    value = np.log10(1 + value)
                                binValuesList.append(value)
                                binIndicesList.append(i)
                    DOMPUlseBinValues[omkey] = binValuesList
                    DOMPulseBinIndices[omkey] = binIndicesList

                    # get summary data
                    if self._no_of_summary_bins > 0:
                        if self._summary_type == 'PulseSummary':
                            if len(wf_charges) != 0:
                                wf_sum = sum(wf_charges)
                                wf_len = len(wf_times)
                                wf_max_charge = max(wf_charges)
                                wf_min_time = min(wf_times)
                                wf_max_time = max(wf_times)
                                wf_mean_time =  np.mean(wf_times)
                                wf_std_time = np.std(wf_times)
                            else:
                                wf_sum = 0.
                                wf_len = 0
                                wf_max_charge = 0.
                                wf_min_time = 0.
                                wf_max_time = 0.
                                wf_mean_time =  0.
                                wf_std_time = 0.
                            # write to frame
                            if self._perform_log:
                                summary_data = [np.log10(1 + wf_sum),
                                                np.log10(1 + wf_len),
                                                np.log10(1 + wf_max_charge),
                                                wf_min_time,wf_max_time,
                                                wf_mean_time,wf_std_time]
                            else:
                                summary_data = [    wf_sum,wf_len,
                                                    wf_max_charge,wf_min_time,
                                                    wf_max_time,wf_mean_time,
                                                    wf_std_time ]
                            DOMPulseDataSummary[omkey] = summary_data[:self._no_of_summary_bins]

                        elif self._summary_type == 'TrackCherenkov':
                            raise NotImplemented('This is not yet implemented for waveform method.')

                        elif self._summary_type == 'Likelihood':
                            raise NotImplemented('This is not yet implemented for waveform method.')

                        else:
                            raise ValueError('SummaryType must be "PulseSummary" or "TrackCherenkov" but is: {}'.format(self._summary_type))
            else:
                print('WARNING:',self._waveforms_string,'does not exist in frame. Returning empty data.')

        if self._no_of_summary_bins > 0:
            frame[self._output_key+'DataSummary'] = DOMPulseDataSummary
        if self._no_of_bins > 0:
            frame[self._output_key+'BinIndices'] = DOMPulseBinIndices
            frame[self._output_key+'BinValues'] = DOMPUlseBinValues

        # write relative time to frame
        frame[self._output_key+'TimeRangeStart'] = \
                                    dataclasses.I3Double(self._time_range[0])