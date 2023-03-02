"""
Adopted from:

https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/projects/
level3-filter-cascade/trunk/python/CascadeL3TraySegment.py
"""
from icecube import icetray, dataclasses

from icecube.level3_filter_cascade.level3_TopoSplitter import TopologicalCounter
from icecube.level3_filter_cascade.level3_RunVeto import runVeto_Singles, runVeto_Coinc
from icecube.level3_filter_cascade.level3_MultiCalculator import multiCalculator
from icecube.level3_filter_cascade.level3_Globals import which_split, label
from icecube.level3_filter_cascade.level3_Cuts import tagBranches
from icecube.level3_filter_cascade.level3_Recos import CascadeLlhVertexFit


def cutBranches(frame, year, discard_non_cscdl3):
    if frame['CscdL3_precut'].value:
        contTag = frame['CscdL3_Cont_Tag'].value
        if not int(year) == 2011:
            fill_ratio = frame['CascadeFillRatio_L2'].fill_ratio_from_mean
        else:
            fill_ratio = frame['CascadeFillRatio'].fill_ratio_from_mean

        L3_Branch0 = False
        L3_Branch1 = False
        L3_Branch2 = False
        L3_Branch3 = False
        if contTag == 1:
            nstr = frame['NString_OfflinePulsesHLC_noDC'].value
            rloglCscd = frame['CascadeLlhVertexFit_ICParams'].ReducedLlh
            if (fill_ratio > 0.6 and nstr >= 3 and rloglCscd < 9.0
                    and rloglCscd > 0.0):
                L3_Branch1 = True
            else:
                L3_Branch1 = False
        elif contTag == 0:
            nch = frame['NCh_OfflinePulses'].value
            if fill_ratio > 0.6 and nch >= 120:
                L3_Branch0 = True
            else:
                L3_Branch0 = False
        elif contTag == 2:
            rlogL_0 = frame['CascadeLlhVertexFit_IC_Coincidence0Params'].ReducedLlh
            rlogL_1 = frame['CascadeLlhVertexFit_IC_Coincidence1Params'].ReducedLlh
            if ((rlogL_0 > 0.0 and rlogL_0 < 8.5) or
                    (rlogL_1 > 0.0 and rlogL_1 < 8.5)):
                L3_Branch2 = True
            else:
                L3_Branch2 = False

        else:
            L3_Branch3 = False
        # The actual cut is here
        if L3_Branch0 or L3_Branch1 or L3_Branch2 or L3_Branch3:
            frame['CscdL3'] = icetray.I3Bool(True)
        else:
            frame['CscdL3'] = icetray.I3Bool(False)
    else:
        frame['CscdL3'] = icetray.I3Bool(False)

    if discard_non_cscdl3:
        return frame['CscdL3'].value
    else:
        return True


def get_time_range(frame, pulse_key):
    """Get time range for a given pulse series

    Use with caution: no idea what the I3TimeRange usually defines, here
    it is only the range within which all pulses lie.

    Parameters
    ----------
    frame : I3Frame
        The current frame.
    pulse_key : str
        The name of the pulses.
    """
    if pulse_key + 'TimeRange' not in frame:
        # get pulses
        pulses = frame[pulse_key]

        if isinstance(pulses, dataclasses.I3RecoPulseSeriesMapMask) or \
           isinstance(pulses, dataclasses.I3RecoPulseSeriesMapUnion):
            pulses = pulses.apply(frame)

        min_time = float('inf')
        max_time = -float('inf')

        for om_key, dom_pulses in pulses.items():
            min_time = min(min_time, dom_pulses[0].time)
            max_time = max(max_time, dom_pulses[-1].time)

        frame[pulse_key + 'TimeRange'] = \
            dataclasses.I3TimeWindow(min_time, max_time)


def maskify(frame):
    # In IC86-2013 'SplitInIcePulses' is used as 'OfflinePulses' in IC86-2011
    if frame.Has('SplitInIcePulses'):
        get_time_range(frame, pulse_key='SplitInIcePulses')
        frame['OfflinePulses'] = frame['SplitInIcePulses']
        frame['OfflinePulsesTimeRange'] = frame['SplitInIcePulsesTimeRange']
    else:
        return True
    if frame.Has('SRTInIcePulses'):
        frame['SRTOfflinePulses'] = frame['SRTInIcePulses']
    else:
        return True
    return True


def pre_cuts(frame, split_name):
    passed_pre_cut = frame['CscdL2'].value and which_split(frame, split_name)
    frame['CscdL3_precut'] = icetray.I3Bool(passed_pre_cut)


def discard_events_not_passing_l2_filters(frame):
    """Discard events that did not pass the Cascade or Muon Filter.

    Parameters
    ----------
    frame : I3Frame
        Current I3Frame.

    Returns
    -------
    bool
        True if cascade or muon filter is passed.

    Raises
    ------
    ValueError
        If keys can't be found.
    """
    if 'FilterMask' in frame:
        f = frame['FilterMask']
    elif 'QFilterMask' in frame:
        f = frame['QFilterMask']
    else:
        raise ValueError('No FilterMask in {!r}'.format(frame))

    muon_key = None
    cascade_key = None
    for key in f.keys():
        if 'MuonFilter' == key[:10]:
            muon_key = key
        if 'CascadeFilter' == key[:13]:
            cascade_key = key

    if muon_key is None:
        raise ValueError('Could not find MuonFilter in {!r}'.format(f.keys()))

    if cascade_key is None:
        raise ValueError('Could not find CascadeFilter in {!r}'.format(
                                                                    f.keys()))

    return f[muon_key].condition_passed or f[cascade_key].condition_passed


def discard_events_not_passing_l2_cascade_filter(frame):
    """Discard events that did not pass the Level2 Cascade Filter.

    Parameters
    ----------
    frame : I3Frame
        Current I3Frame.

    Returns
    -------
    bool
        True if cascade or muon filter is passed.

    Raises
    ------
    ValueError
        If keys can't be found.
    """
    if 'FilterMask' in frame:
        f = frame['FilterMask']
    elif 'QFilterMask' in frame:
        f = frame['QFilterMask']
    else:
        raise ValueError('No FilterMask in {!r}'.format(frame))

    cascade_key = None
    for key in f.keys():
        if 'CascadeFilter' == key[:13]:
            cascade_key = key

    if cascade_key is None:
        raise ValueError('Could not find CascadeFilter in {!r}'.format(
                                                                    f.keys()))

    return f[cascade_key].condition_passed


def rename_filter_mask(frame):
    if 'FilterMask' not in frame:
        if 'QFilterMask' in frame:
            f = frame['QFilterMask']
            del frame['QFilterMask']
            frame['FilterMask'] = f


@icetray.traysegment
def CascadeL3Cuts(tray, name, year, discard_non_l2=True,
                  discard_non_cscdl2=False,
                  discard_non_cscdl3=False):

    tray.AddModule(rename_filter_mask, 'rename_filter_mask')

    if discard_non_l2:
        tray.AddModule(discard_events_not_passing_l2_filters,
                       'discard_events_not_passing_l2_filters')
        tray.AddModule('I3OrphanQDropper', 'drop_double_q_frames')

    if discard_non_cscdl2:
        tray.AddModule(discard_events_not_passing_l2_cascade_filter,
                       'discard_events_not_passing_l2_cascade_filter')
        tray.AddModule('I3OrphanQDropper', 'drop_double_q_frames_cscdl2')

    if not int(year) == 2011:
        split_name = 'InIceSplit'
        tray.AddModule(maskify, 'maskify')
    else:
        split_name = 'in_ice'

    icetray.load("DomTools", False)
    tray.AddModule("I3LCPulseCleaning", name+"_I3LCPulseCleaning",
                   Input='OfflinePulses',
                   OutputHLC='OfflinePulsesHLC',
                   OutputSLC="",
                   If=lambda frame: 'OfflinePulsesHLC' not in frame)

    tray.AddModule(label, 'label_CascadeL2Stream', year=year)

    # # selects cascade L2 stream w/o removing IceTop p-frames.
    # tray.AddModule(lambda frame: frame['CscdL2'].value and
    #                which_split(frame, split_name), 'SelecCscdL2')
    tray.AddModule(pre_cuts, 'pre_cuts', split_name=split_name)

    # count multiplicity of the events
    tray.AddSegment(TopologicalCounter, 'CscdL2_Topo_HLC',
                    pulses='OfflinePulsesHLC',
                    InIceCscd=lambda frame: which_split(frame, split_name) and frame['CscdL2'].value
                    )
    # run I3Veto on single events
    tray.AddSegment(runVeto_Singles,
                    '_SRTOfflinePulses',
                    pulses='SRTOfflinePulses',
                    If=lambda frame: which_split(frame, split_name) and frame['CscdL2'].value and frame['CscdL2_Topo_HLCSplitCount'].value==1
                    )
    # run I3Veto on coincident events
    tray.AddSegment(runVeto_Coinc,
                    'CscdL2_Topo_HLC',
                    pulses='CscdL2_Topo_HLC',
                    If=lambda frame: which_split(frame, split_name) and frame['CscdL2'].value and frame['CscdL2_Topo_HLCSplitCount'].value==2
                    )

    # tagging events
    tray.AddModule(tagBranches, 'Branchtagging', InIceCscd=lambda frame: which_split(frame, split_name) and frame['CscdL2'].value)

    # calculating NCh, NStrings for singles branches but run all events (no harm done)
    tray.AddSegment(multiCalculator, 'multiHLC',
                    pulses="OfflinePulsesHLC",
                    InIceCscd=lambda frame: which_split(frame, split_name) and frame['CscdL2'].value,
                    )
    # general cascade llh w/o DC for singles branches but run all events (no harm done)
    tray.AddSegment(CascadeLlhVertexFit, 'CascadeLlhVertexFit_IC',
                    Pulses='OfflinePulsesHLC_noDC',
                    If=lambda frame: which_split(frame, split_name) and frame['CscdL2'].value,
                    )
    # calculating NCh, NStrings for singles branches but run all events (no harm done)
    tray.AddSegment(multiCalculator, 'multi',
                    pulses="OfflinePulses",
                    InIceCscd=lambda frame: which_split(frame, split_name) and frame['CscdL2'].value,
                    )
    # calculating NCh, NStrings QTot etc for doubles branch but only doubles
    tray.AddSegment(multiCalculator, 'multiHLC_TS0',
                    pulses="CscdL2_Topo_HLC0",
                    InIceCscd=lambda frame: which_split(frame, split_name) and frame['CscdL2'].value and frame['CscdL2_Topo_HLCSplitCount'].value==2,
                    )
    tray.AddSegment(multiCalculator, 'multiHLC_TS1',
                    pulses="CscdL2_Topo_HLC1",
                    InIceCscd=lambda frame: which_split(frame, split_name) and frame['CscdL2'].value and frame['CscdL2_Topo_HLCSplitCount'].value==2,
                    )
    # calculate two CscdLlhVertexFits w/o DC for doubles contained branch
    tray.AddSegment(CascadeLlhVertexFit, 'CascadeLlhVertexFit_IC_Coincidence0',
                    Pulses='CscdL2_Topo_HLC0_noDC',
                    If=lambda frame: which_split(frame, split_name) and frame['CscdL2'].value and frame['CscdL2_Topo_HLCSplitCount'].value==2,
                    )
    tray.AddSegment(CascadeLlhVertexFit, 'CascadeLlhVertexFit_IC_Coincidence1',
                    Pulses='CscdL2_Topo_HLC1_noDC',
                    If=lambda frame: which_split(frame, split_name) and frame['CscdL2'].value and frame['CscdL2_Topo_HLCSplitCount'].value==2,
                    )
    # L3 cut happens here
    tray.AddModule(cutBranches, 'Level3_Cut', year=year,
                   discard_non_cscdl3=discard_non_cscdl3)
    tray.AddModule('I3OrphanQDropper', 'drop_unwanted_q')
