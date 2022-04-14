from icecube import dataclasses


from ic3_data.ext_boost import combine_exclusions
from ic3_data.ext_boost import merge_pulses
from ic3_data.ext_boost import get_valid_pulse_map as get_valid_pulse_map_cpp


def get_combined_exclusions(
        frame, dom_and_tw_exclusions, partial_exclusion, output_key):
    """Get combined exclusions and write them to the frame.

    Parameters
    ----------
    frame : I3Frame
        The frame to which to write the combined exclusions.
    dom_and_tw_exclusions : list of str
        The list of DOM and time window exclusion frame keys to combine.
    partial_exclusion : bool
        Whether or not to apply partial DOM exclusion.
    output_key : str
        The base output key to which results will be written.
    """
    if isinstance(dom_and_tw_exclusions, str):
        dom_and_tw_exclusions = [dom_and_tw_exclusions]

    excluded_doms, excluded_tws = combine_exclusions(
        frame, dom_and_tw_exclusions, partial_exclusion)

    frame[output_key + 'DOMs'] = excluded_doms
    frame[output_key + 'TimeWindows'] = excluded_tws


def get_valid_pulse_map(frame, pulse_key, dom_and_tw_exclusions,
                        partial_exclusion,
                        output_key=None,
                        verbose=False):
    """Simple wrapper over c++ version.
    Necessary for I3 Magic...

    Parameters
    ----------
    frame : I3Frame
        The current IeFrame.
    pulse_key : str
        The name of the input pulse series map.
    dom_and_tw_exclusions : TYPE
        Description
    partial_exclusion : TYPE
        Description
    output_key : str, optional
        The name to which the masked pulse series map will be written to.
        If none provided, the output name will be `pulse_key`+'_masked'.
    verbose : bool, optional
        Description
    """
    if isinstance(dom_and_tw_exclusions, str):
        dom_and_tw_exclusions = [dom_and_tw_exclusions]

    pulses = get_valid_pulse_map_cpp(
        frame, pulse_key, dom_and_tw_exclusions, partial_exclusion, verbose)

    if output_key is None:
        frame[pulse_key + '_masked'] = pulses
    else:
        frame[output_key] = pulses


def get_merged_pulse_map(frame, pulse_key, time_threshold,
                         output_key=None):
    """Simple wrapper over c++ version.
    Necessary for I3 Magic...

    Parameters
    ----------
    frame : I3Frame
        The current IeFrame.
    pulse_key : str
        The name of the input pulse series map.
    time_threshold : double, optional
        If provided, pulses within this time threshold will be merged together
        into one pulse. The merged pulse obtains the time stamp and properties
        of the first pulse.
    output_key : str, optional
        The name to which the merged pulse series map will be written to.
        If none provided, the output name will be `pulse_key`+'_merged'.
    """
    assert time_threshold > 0., time_threshold

    pulses = merge_pulses(frame, pulse_key, time_threshold)

    if output_key is None:
        frame[pulse_key + '_merged'] = pulses
    else:
        frame[output_key] = pulses
