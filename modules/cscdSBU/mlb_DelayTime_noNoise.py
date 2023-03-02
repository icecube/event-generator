from __future__ import print_function, division
from I3Tray import *
from icecube import icetray, dataio, dataclasses
import numpy as n

def getreco(frame, name):
    """
        retrieve I3Particle from frame and check fit_status
    """
    if name in frame:
        reco = frame[name]
        if reco.fit_status == reco.fit_status.OK:
            return reco
        else:
            return None
    else:
        return None

def getobject(frame,name):
    """
        retrieve I3FrameObject from frame
    """
    if name in frame:
        return frame[name]
    else:
        return None

def calc_dt_nearly_ice(frame, name, reconame, pulsemapname):

    try:
        mask = dataclasses.I3RecoPulseSeriesMapMask(frame, pulsemapname)
    except:
        return
    pmap = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, pulsemapname)
    reco  = getreco(frame, reconame)
    geo   = getobject(frame, "I3Geometry")

    if any([i is None for i in [reco, pmap, geo]]):
        return True

    ref_pos = n.array(reco.pos)
    ref_time = reco.time

    c_vac = 2.99792458e8 *1e-9 # m/ns
    n_ice_group = 1.35634
    c_ice = c_vac/n_ice_group  # m/ns

    dt_n_early_vac = n.inf
    dt_n_early_ice = n.inf
    dist = -1e-6
    first_time = -1e-6

    nHitDOMs = 0
    nch = 0

    ############ select early DOMs #########
    early_DOMs = []
    for ss in range (1,87):
        p_list = [(dom, pmap[dom][0].time) for dom in pmap.keys() if dom.string == ss]
        if len(p_list)>=2:

            p_time_tuple=zip(*p_list)[1]
            p_time=n.asarray(p_time_tuple)
            q_time=list(p_time)
            q_time.pop(0)
            q_time.append(0)
            time_diff = n.asarray(p_time) - n.asarray(q_time)

            timeIndex=n.argmin(time_diff)

            if (min(time_diff))< -1000.:
                early_DOMs.append(p_list[timeIndex][0])

    if not frame.Has('cscdSBU_EarlyDOMs'):
        frame['cscdSBU_EarlyDOMs'] = dataclasses.I3Double(len(early_DOMs))

    noisy_early_DOMs=[]
    if len(early_DOMs)>0:
        for om in early_DOMs:
            ww = sum([sum([p.charge for p in pulses if p.flags & p.PulseFlags.ATWD ]) for dom, pulses in pmap if dom==om])
            if ww<2.:
                noisy_early_DOMs.append(om)
            #else:
            #    print '--- charge too big'
    #else:
    #    print 'early_omkey empty!!\n'

    if len(noisy_early_DOMs)>0:
        if not frame.Has('OfflinePulsesHLC_CleanedFirstPulses'):
            frame['OfflinePulsesHLC_CleanedFirstPulses'] = dataclasses.I3RecoPulseSeriesMapMask(frame, "OfflinePulsesHLC_noSaturDOMs", lambda om, idx, pulse: idx==0 and om not in early_DOMs)
    else:
        if not frame.Has('OfflinePulsesHLC_CleanedFirstPulses'):
            frame['OfflinePulsesHLC_CleanedFirstPulses'] = dataclasses.I3RecoPulseSeriesMapMask(frame, "OfflinePulsesHLC_noSaturDOMs")
    pmap_withEarly = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'OfflinePulsesHLC_CleanedFirstPulses')

    ############ calculate delay time for a cleaned pulses
    pmap_Cleaned = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'OfflinePulsesHLC_CleanedFirstPulses')
    for i,(omkey, pseries) in enumerate(pmap_Cleaned.iteritems()):
        if len(pseries) == 0:
            continue
        #if(frame.Has('SaturationWindows') and frame['SaturationWindows'].keys()==omkey):
        #    print 'Saturated DOM!!!!!!!!!!', omkey, '\n'
        else:
            om_pos = n.array(geo.omgeo[omkey].position)
            first_time = min([pls.time for pls in pseries])      # time of first pulse in this DOM
            dist = n.sqrt( n.power(om_pos - ref_pos, 2).sum() )  # distance between dom and reco vertex

            dt_n_early_ice = min(first_time - dist/c_ice - ref_time, dt_n_early_ice) # smallest delay time with c_ice
            dt_n_early_vac = min(first_time - dist/c_vac - ref_time, dt_n_early_vac) # smallest delay time with c_vac

    frame['%s_Delay_ice' % name] = dataclasses.I3Double(dt_n_early_ice)
    frame['%s_Delay_vac' % name] = dataclasses.I3Double(dt_n_early_vac)
    frame['%s_Delay_dist' % name]           = dataclasses.I3Double(dist)
    frame['%s_Delay_firstTime' % name]     = dataclasses.I3Double(first_time)


