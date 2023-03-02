from icecube import icetray
from egenerator.utils.ic3 import exclusions, bright_doms

@icetray.traysegment
def CombineAndApplyExclusions(
        tray, name,
        pulse_key = 'HVInIceDSTPulses',
        dom_and_tw_exclusions = [
            'BadDomsList', 'CalibrationErrata', 'SaturationWindows'],
        partial_exclusion = True,
        exclude_bright_doms=False,
        bright_doms_threshold_fraction=0.4,
        bright_doms_threshold_charge=100.,
        merge_pulses_time_threshold=None,
        ):
    """Combine and Apply DOM and TimeWindow exclusions
    Parameters
    --------------------------------------------------------------------
    tray : icecube.icetray
        The icetray. 
    name : str
        Name of the module.
    pulse_key : str, optional
        The pulses to run the event-generator on. If specified,these pulses
        will be masked to comply with the DOM and time window exclusions. 
    dom_and_tw_exclusions : str or list of str, optional
        A list of frame keys that specify DOM and time window exclusions. 
        If no exclusions are to be applied, pass an empty list or None. 
    partial_exclusion : bool, optional
        If True, partially exclude DOMs, e.g. only omit pulses from excluded 
        TimeWindows defined in 'dom_and_tw_exclusions'.
        If False, all pulses from a DOM will be excluded regardless of how 
        long the excluded time window is. 
    exclude_bright_doms : bool, optional 
        If True, bright DOMs will be excluded. A bright DOM is a DOM that 
        fulfills: 
            DOM charge > 'bright_doms_threshold_fraction' * event charge
            DOM charge > 'bright_doms_threshold_charge'
    bright_doms_threshold_fraction : float, optional
        The threshold fraction of total event charge above which a DOM is 
        considered to be a bright DOM. See exclude_bright_doms.
    bright_doms_threshold_charge : float, optional 
        The threshold charge a DOM must exceed to be considered a bright DOM. 
        See exclude_bright_doms. 
    merge_pulses_time_threshold : double, optional
        If provided, pulses within this time threshold will be merged
        together into one pulse. The merged pulse obtains the time stamp 
        and properties of the first pulse. 

    Returns
    ----------------------------------------------------------------------
    str 
        Frame key of the masked pulses.
    str 
        Frame key of the combined excluded DOMs.
    str 
        Frame key of the combined excluded time windows. 
    list of str
        List of frame keys that were added.
    """

    name = 'Combined_exclusion'
    if dom_and_tw_exclusions is None:
        dom_and_tw_exclusions =[]
    if isinstance(dom_and_tw_exclusions, str):
        dom_and_tw_exclusions = [dom_and_tw_exclusions]

    combined_exclusion_key = name
    bright_dom_key = 'BrightDOMs' 
    excluded_dom_key = combined_exclusion_key + 'DOMs'
    excluded_tw_key = 'time_exclusion'
    masked_pulses_key = combined_exclusion_key + 'Pulses'
    added_keys = [masked_pulses_key, excluded_dom_key, excluded_tw_key]

    if exclude_bright_doms:
        dom_and_tw_exclusions.append(bright_dom_key)
        added_keys.appedn(bright_dom_key)

        # compute and add bright DOMs to frame
        tray.addModule(
                bright_doms.AddBrightDOMs, name + 'BrightDOMs',
                PulseKey = pulse_key,
                BrightThresholdFraction = bright_doms_threshold_fraction, 
                BrghtThresholdCharge = bright_doms_threshold_charge,
                OutputKey = bright_dom_key
                )

    # combine exclusions in a single key for DOMs and TWs
    tray.AddModule(
            exclusions.get_combined_exclusions, combined_exclusion_key,
            dom_and_tw_exclusions = dom_and_tw_exclusions,
            partial_exclusion = partial_exclusion,
            output_key = combined_exclusion_key,
            )

    # mask pulses
    tray.AddModule(
            exclusions.get_valid_pulse_map, name + 'MaskPulses',
            pulse_key= pulse_key,
            output_key = 'masked_pulses',
            dom_and_tw_exclusions = dom_and_tw_exclusions,
            partial_exclusion = partial_exclusion,
            verbose = True,
            )

