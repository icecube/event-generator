from icecube import dataclasses


from ic3_data.ext_boost import combine_exclusions
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
    frame : TYPE
        Description
    pulse_key : TYPE
        Description
    dom_and_tw_exclusions : TYPE
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
    if isinstance(dom_and_tw_exclusions, str):
        dom_and_tw_exclusions = [dom_and_tw_exclusions]

    pulses = get_valid_pulse_map_cpp(
        frame, pulse_key, dom_and_tw_exclusions, partial_exclusion, verbose)

    if output_key is None:
        frame[pulse_key + '_masked'] = pulses
    else:
        frame[output_key] = pulses
