#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Helper functions to mask pulses
'''
from icecube import dataclasses
import numpy as np

def exclusion_test(frame, pulse_key, exclusion_variation, 
                    verbose=False):
    # -------------------------------
    #   Todo: Only work on 
    #         I3RecoPulseSeriesMapMask
    # -------------------------------
    pulses = frame[pulse_key]

    if not exclusion_variation is None:
        if isinstance(pulses, dataclasses.I3RecoPulseSeriesMapMask) or \
           isinstance(pulses, dataclasses.I3RecoPulseSeriesMapUnion):
            pulses = pulses.apply(frame)

        pulses = dict(pulses)
        length_before = len(pulses)
        num_rm_pulses = 0


        # if bright_doms_key in frame:

        #     for om_key in frame[bright_doms_key]:

        #         if om_key in pulses:
        #             bright_pulses = pulses[om_key]

        #             # modify pulses
        #             modified_bright_pulses = []
                    
        #             # go through each reco pulse
        #             is_first_pulse = True
        #             for bright_pulse in bright_pulses:

        #                 # modify pulse and append to modified hits   
        #                 modified_bright_pulses.append(modify_pulse(bright_pulse,
        #                                                      brights_variation,
        #                                                      is_first_pulse))
        #                 is_first_pulse = False
                    
        #             # replace old hit
        #             num_rm_pulses += len(pulses[om_key]) - \
        #                              len(modified_bright_pulses)
        #             pulses.pop(om_key, None)
        #             pulses[om_key] = dataclasses.vector_I3RecoPulse(
        #                                                 modified_bright_pulses)
                    

        pulses = dataclasses.I3RecoPulseSeriesMap(pulses)

        if verbose:
            num_removed = length_before - len(pulses)
            msg = '[ExclusionTest] Removed {} DOMs and {} additional pulses from {}'
            # ToDo: use logging
            print(  msg.format( num_removed,
                            num_rm_pulses,
                            pulse_key ))
    del frame[pulse_key]
    frame[pulse_key] = pulses
    # -------------------------------

def modify_pulse(bright_pulse, brights_variation, is_first_pulse):

    modified_bright_pulse = dataclasses.I3RecoPulse(bright_pulse)

    # Variation 1: increase charge by 20%
    if brights_variation == 1:
        modified_bright_pulse.charge *= 1.2 

    # Variation 2: decrease charge by 20%
    elif brights_variation == 2:
        modified_bright_pulse.charge *= 0.8 

    elif brights_variation == 3:
        modified_bright_pulse.charge *= 1.5

    elif brights_variation == 4:
        modified_bright_pulse.charge *= 0.5

    elif brights_variation == 5:
        modified_bright_pulse.charge *= 2

    elif brights_variation == 6:
        modified_bright_pulse.charge /= 2

    elif brights_variation == 7:
        modified_bright_pulse.charge *= 5

    elif brights_variation == 8:
        modified_bright_pulse.charge /= 5

    elif brights_variation == 9:
        modified_bright_pulse.charge *= 10

    elif brights_variation == 10:
        modified_bright_pulse.charge /= 10

    # Randomly sample 20% unc
    elif brights_variation == 11:
        modified_bright_pulse.charge = np.random.normal(
                                        modified_bright_pulse.charge,
                                        0.2*modified_bright_pulse.charge)

    # Randomly sample 50% unc
    elif brights_variation == 12:
        modified_bright_pulse.charge = np.random.normal(
                                        modified_bright_pulse.charge,
                                        0.5*modified_bright_pulse.charge)

    # shift time
    elif brights_variation == 13:
        modified_bright_pulse.time += 5

    # shift time
    elif brights_variation == 14:
        modified_bright_pulse.time -= 5

    # shift time
    elif brights_variation == 15:
        modified_bright_pulse.time += 10

    # shift time
    elif brights_variation == 16:
        modified_bright_pulse.time -= 10

    # shift time
    elif brights_variation == 17:
        modified_bright_pulse.time += 30

    # shift time
    elif brights_variation == 18:
        modified_bright_pulse.time -= 30

    # test if first pulse is relevant
    elif brights_variation == 19:
        if is_first_pulse:
            modified_bright_pulse.time -= 5

    # test if first pulse is relevant
    elif brights_variation == 20:
        if is_first_pulse:
            modified_bright_pulse.time += 5

    # test if first pulse is relevant
    elif brights_variation == 21:
        if is_first_pulse:
            modified_bright_pulse.time -= 20

    # test if first pulse is relevant
    elif brights_variation == 22:
        if is_first_pulse:
            modified_bright_pulse.time += 20

    # test if first pulse is relevant
    elif brights_variation == 23:
        if not is_first_pulse:
            modified_bright_pulse.time += 5

    # test if first pulse is relevant
    elif brights_variation == 24:
        if not is_first_pulse:
            modified_bright_pulse.time += 20

    # test if first pulse is relevant
    elif brights_variation == 25:
        if not is_first_pulse:
            modified_bright_pulse.time += 50

    else:
        raise ValueError('Unknown pulse variation: {}'.format(
                                                            brights_variation))

    return modified_bright_pulse