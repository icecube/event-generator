#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Helper functions to mask pulses
'''
from icecube import dataclasses
from ic3_data.ext_boost import get_valid_pulse_map as get_valid_pulse_map_cpp


def get_valid_pulse_map(frame, pulse_key, excluded_doms,
                        partial_exclusion,
                        output_key=None,
                        verbose=False):
    """Simple wrapper over c++ version.
    Necessary for I3 Magic...

    Parameters
    ----------
    frame : TYPE
        Description
    pulse_key : TYPE
        Description
    excluded_doms : TYPE
        Description
    partial_exclusion : TYPE
        Description
    output_key : None, optional
        Description
    verbose : bool, optional
        Description

    No Longer Returned
    ------------------
    TYPE
        Description
    """
    if isinstance(excluded_doms, str):
        excluded_doms = [excluded_doms]

    pulses = get_valid_pulse_map_cpp(frame, pulse_key, excluded_doms,
                                     partial_exclusion, verbose)

    if output_key is None:
        frame[pulse_key + '_masked'] = pulses
    else:
        frame[output_key] = pulses


def get_valid_pulse_map_old(frame, pulse_key, excluded_doms,
                            partial_exclusion, verbose=False):
    # -------------------------------
    #   Todo: Only work on
    #         I3RecoPulseSeriesMapMask
    # -------------------------------
    pulses = frame[pulse_key]

    if excluded_doms:
        if isinstance(pulses, dataclasses.I3RecoPulseSeriesMapMask) or \
           isinstance(pulses, dataclasses.I3RecoPulseSeriesMapUnion):
            pulses = pulses.apply(frame)

        pulses = dict(pulses)
        length_before = len(pulses)
        num_rm_pulses = 0

        for exclusion_key in excluded_doms:

            if exclusion_key in frame:

                #-------------------------
                # List of OMkeys to ignore
                #-------------------------
                if isinstance(frame[exclusion_key],
                            dataclasses.I3VectorOMKey) or \
                   isinstance(frame[exclusion_key],list):

                    for key in frame[exclusion_key]:
                        pulses.pop(key, None)

                #-------------------------
                # I3TimeWindowSeriesMap
                #-------------------------
                elif isinstance(frame[exclusion_key],
                            dataclasses.I3TimeWindowSeriesMap):

                    if partial_exclusion:
                        # remove Pulses in exluded time window
                        for key in frame[exclusion_key].keys():

                            # skip this key if it does
                            # not exist in reco pulse map
                            if not key in pulses:
                                continue

                            valid_hits = []

                            # go through each reco pulse
                            for hit in pulses[key]:

                                # assume hit is valid
                                hit_is_valid = True

                                # go through every time window
                                for time_window in frame[exclusion_key][key]:
                                    if hit.time >= time_window.start and \
                                       hit.time <= time_window.stop:

                                       # reco pulse is in exclusion
                                       # time window and therefore
                                       # not valid
                                       hit_is_valid = False
                                       break

                                # append to valid hits
                                if hit_is_valid:
                                    valid_hits.append(hit)

                            # replace old hit
                            num_rm_pulses += len(pulses[key]) - len(valid_hits)
                            pulses.pop(key, None)
                            if valid_hits:
                                pulses[key] = dataclasses.vector_I3RecoPulse(valid_hits)

                    else:
                        # remove whole DOM
                        for key in frame[exclusion_key].keys():
                            pulses.pop(key, None)
                else:
                    msg = 'Unknown exclusion type {} of key {}'
                    raise ValueError(msg.format( type(frame[exclusion_key]),
                                                             exclusion_key))

        pulses = dataclasses.I3RecoPulseSeriesMap(pulses)

        if verbose:
            num_removed = length_before - len(pulses)
            msg = '[DNN_reco] Removed {} DOMs and {} additional pulses from {}'
            # ToDo: use logging
            print(  msg.format( num_removed,
                            num_rm_pulses,
                            pulse_key ))

    frame[pulse_key+'_masked'] = pulses
    # -------------------------------


def get_bright_pulses(frame, pulse_key,
                        bright_doms_key='BrightDOMs',
                        output_name='BrightPulses',
                    ):
    # -------------------------------
    #   Todo: Only work on
    #         I3RecoPulseSeriesMapMask
    # -------------------------------

    if bright_doms_key in frame:

        pulses = frame[pulse_key]

        if isinstance(pulses, dataclasses.I3RecoPulseSeriesMapMask) or \
           isinstance(pulses, dataclasses.I3RecoPulseSeriesMapUnion):
            pulses = pulses.apply(frame)

        pulses = dict(pulses)

        bright_pulses = {}

        for om_key in frame[bright_doms_key]:

            if om_key in pulses:
                bright_pulses[om_key] = pulses[om_key]

        if bright_pulses != {}:
            bright_pulses = dataclasses.I3RecoPulseSeriesMap(bright_pulses)
            frame[output_name] = bright_pulses
    # -------------------------------