from icecube import icetray

from egenerator.ic3.reconstruction import EventGeneratorReconstruction
from egenerator.ic3.visualization import EventGeneratorVisualizeBestFit
from egenerator.utils.ic3 import exclusions, bright_doms


@icetray.traysegment
def ApplyEventGeneratorReconstruction(
    tray,
    name,
    pulse_key="SplitInIceDSTPulses",
    dom_and_tw_exclusions=[
        "BadDomsList",
        "CalibrationErrata",
        "SaturationWindows",
    ],
    partial_exclusion=True,
    exclude_bright_doms=True,
    bright_doms_threshold_fraction=0.4,
    bright_doms_threshold_charge=100.0,
    merge_pulses_time_threshold=None,
    clean_up=True,
    **egenerator_kwargs
):
    """Convenice I3TraySegment to apply Event-Generator reconstruction

    Combines DOM and time window exclusions in addition to computing bright
    DOMs if specified. Pulses are masked and handed to the event-generator
    reconstruction.

    Parameters
    ----------
    tray : icecube.icetray
        The icetry.
    name : str
        Name of module
    pulse_key : str, optional
        The pulses to run the event-generator on. If specified, these pulses
        will be masked to comply with the DOM and time window exclusions.
    dom_and_tw_exclusions : str or list of str, optional
        A list of frame keys that specify DOM and time window exclusions.
        If no exclusions are to be applied, pass an empty list or None.
    partial_exclusion : bool, optional
        If True, partially exclude DOMS, e.g. only omit pulses from excluded
        TimeWindows defined in 'dom_and_tw_exclusions'.
        If False, all pulses from a DOM will be excluded regardless of how
        long the excluded time window is.
    exclude_bright_doms : bool, optional
        If True, bright DOMs will be excluded. A bright DOM is a DOM that
            fulfills:
                DOM charge > `bright_doms_threshold_fraction` * event charge
                DOm charge > `bright_doms_threshold_charge`
    bright_doms_threshold_fraction : float, optional
        The threshold fraction of total event charge above which a DOM is
        considered as a bright DOM. See exclude_bright_doms.
    bright_doms_threshold_charge : float, optional
        The threshold charge a DOM must exceeded to be considered as a
        bright DOM. See exclude_bright_doms.
    merge_pulses_time_threshold : double, optional
        If provided, pulses within this time threshold will be merged together
        into one pulse. The merged pulse obtains the time stamp and properties
        of the first pulse.
    clean_up : bool, optional
        If True, temporarily created frame objects, such as the combined
        exclusions and masked pulses, are deleted from the frame once no
        longer needed.
    **egenerator_kwargs
        Keyword arguments that will be passed on to the Event-Generator
        reconstruction I3Module. See documentation of the I3Module
        `EventGeneratorReconstruction` in egenerator.ic3.reconstruction
        for further details.
    """

    # combine exclusions and mask pulses
    masked_pulses, excluded_dom_k, excluded_tw_k, added_keys = tray.AddSegment(
        CombineAndApplyExclusions,
        name + "CombinedExclusions",
        pulse_key=pulse_key,
        dom_and_tw_exclusions=dom_and_tw_exclusions,
        partial_exclusion=partial_exclusion,
        exclude_bright_doms=exclude_bright_doms,
        bright_doms_threshold_fraction=bright_doms_threshold_fraction,
        bright_doms_threshold_charge=bright_doms_threshold_charge,
        merge_pulses_time_threshold=merge_pulses_time_threshold,
    )

    # apply event-generator reconstruction
    tray.AddModule(
        EventGeneratorReconstruction,
        name + "Reco",
        pulse_key=masked_pulses,
        dom_exclusions_key=excluded_dom_k,
        time_exclusions_key=excluded_tw_k,
        **egenerator_kwargs
    )

    # clean up
    if clean_up:
        tray.AddModule("Delete", name + "CleanUp", Keys=added_keys)


@icetray.traysegment
def ApplyEventGeneratorVisualizeBestFit(
    tray,
    name,
    pulse_key="SplitInIceDSTPulses",
    dom_and_tw_exclusions=[
        "BadDomsList",
        "CalibrationErrata",
        "SaturationWindows",
    ],
    partial_exclusion=True,
    exclude_bright_doms=True,
    bright_doms_threshold_fraction=0.4,
    bright_doms_threshold_charge=100.0,
    merge_pulses_time_threshold=None,
    clean_up=True,
    **egenerator_kwargs
):
    """Convenice I3TraySegment to apply Event-Generator best fit visualization

    Combines DOM and time window exclusions in addition to computing bright
    DOMs if specified. Pulses are masked and handed to the event-generator
    reconstruction.

    Parameters
    ----------
    tray : icecube.icetray
        The icetry.
    name : str
        Name of module
    pulse_key : str, optional
        The pulses to run the event-generator on. If specified, these pulses
        will be masked to comply with the DOM and time window exclusions.
    dom_and_tw_exclusions : str or list of str, optional
        A list of frame keys that specify DOM and time window exclusions.
        If no exclusions are to be applied, pass an empty list or None.
    partial_exclusion : bool, optional
        If True, partially exclude DOMS, e.g. only omit pulses from excluded
        TimeWindows defined in 'dom_and_tw_exclusions'.
        If False, all pulses from a DOM will be excluded regardless of how
        long the excluded time window is.
    exclude_bright_doms : bool, optional
        If True, bright DOMs will be excluded. A bright DOM is a DOM that
            fulfills:
                DOM charge > `bright_doms_threshold_fraction` * event charge
                DOm charge > `bright_doms_threshold_charge`
    bright_doms_threshold_fraction : float, optional
        The threshold fraction of total event charge above which a DOM is
        considered as a bright DOM. See exclude_bright_doms.
    bright_doms_threshold_charge : float, optional
        The threshold charge a DOM must exceeded to be considered as a
        bright DOM. See exclude_bright_doms.
    merge_pulses_time_threshold : double, optional
        If provided, pulses within this time threshold will be merged together
        into one pulse. The merged pulse obtains the time stamp and properties
        of the first pulse.
    clean_up : bool, optional
        If True, temporarily created frame objects, such as the combined
        exclusions and masked pulses, are deleted from the frame once no
        longer needed.
    **egenerator_kwargs
        Keyword arguments that will be passed on to the Event-Generator
        visualization I3Module. See documentation of the I3Module
        `EventGeneratorVisualizeBestFit` in egenerator.ic3.visualization
        for further details.
    """

    # combine exclusions and mask pulses
    masked_pulses, excluded_dom_k, excluded_tw_k, added_keys = tray.AddSegment(
        CombineAndApplyExclusions,
        name + "CombinedExclusions",
        pulse_key=pulse_key,
        dom_and_tw_exclusions=dom_and_tw_exclusions,
        partial_exclusion=partial_exclusion,
        exclude_bright_doms=exclude_bright_doms,
        bright_doms_threshold_fraction=bright_doms_threshold_fraction,
        bright_doms_threshold_charge=bright_doms_threshold_charge,
        merge_pulses_time_threshold=merge_pulses_time_threshold,
    )

    # apply event-generator reconstruction
    tray.AddModule(
        EventGeneratorVisualizeBestFit,
        name + "Visualization",
        pulse_key=masked_pulses,
        dom_exclusions_key=excluded_dom_k,
        time_exclusions_key=excluded_tw_k,
        **egenerator_kwargs
    )

    # clean up
    if clean_up:
        tray.AddModule("Delete", name + "CleanUp", Keys=added_keys)


@icetray.traysegment
def CombineAndApplyExclusions(
    tray,
    name,
    pulse_key="SplitInIceDSTPulses",
    dom_and_tw_exclusions=[
        "BadDomsList",
        "CalibrationErrata",
        "SaturationWindows",
    ],
    partial_exclusion=True,
    exclude_bright_doms=True,
    bright_doms_threshold_fraction=0.4,
    bright_doms_threshold_charge=100.0,
    merge_pulses_time_threshold=None,
):
    """Combine and Apply DOM and TimeWindow exclusions

    Parameters
    ----------
    tray : icecube.icetray
        The icetry.
    name : str
        Name of module
    pulse_key : str, optional
        The pulses to run the event-generator on. If specified, these pulses
        will be masked to comply with the DOM and time window exclusions.
    dom_and_tw_exclusions : str or list of str, optional
        A list of frame keys that specify DOM and time window exclusions.
        If no exclusions are to be applied, pass an empty list or None.
    partial_exclusion : bool, optional
        If True, partially exclude DOMS, e.g. only omit pulses from excluded
        TimeWindows defined in 'dom_and_tw_exclusions'.
        If False, all pulses from a DOM will be excluded regardless of how
        long the excluded time window is.
    exclude_bright_doms : bool, optional
        If True, bright DOMs will be excluded. A bright DOM is a DOM that
            fulfills:
                DOM charge > `bright_doms_threshold_fraction` * event charge
                DOm charge > `bright_doms_threshold_charge`
    bright_doms_threshold_fraction : float, optional
        The threshold fraction of total event charge above which a DOM is
        considered as a bright DOM. See exclude_bright_doms.
    bright_doms_threshold_charge : float, optional
        The threshold charge a DOM must exceeded to be considered as a
        bright DOM. See exclude_bright_doms.
    merge_pulses_time_threshold : double, optional
        If provided, pulses within this time threshold will be merged together
        into one pulse. The merged pulse obtains the time stamp and properties
        of the first pulse.

    Returns
    -------
    str
        Frame key of the masked pulses.
    str
        Frame key of the combined excluded DOMs.
    str
        Frame key of the combined excluded time windows.
    list of str
        List of frame keys that were added.

    No Longer Raises
    ----------------
    NotImplementedError
        Description
    """
    if dom_and_tw_exclusions is None:
        dom_and_tw_exclusions = []
    if isinstance(dom_and_tw_exclusions, str):
        dom_and_tw_exclusions = [dom_and_tw_exclusions]

    combined_exclusion_key = name
    bright_dom_key = name + "BrightDOMs"
    excluded_dom_key = combined_exclusion_key + "DOMs"
    excluded_tw_key = combined_exclusion_key + "TimeWindows"
    masked_pulses_key = combined_exclusion_key + "Pulses"
    added_keys = [masked_pulses_key, excluded_dom_key, excluded_tw_key]

    if exclude_bright_doms:
        dom_and_tw_exclusions.append(bright_dom_key)
        added_keys.append(bright_dom_key)

        # compute and add bright DOMs to frame
        tray.AddModule(
            bright_doms.AddBrightDOMs,
            name + "BrightDOMs",
            PulseKey=pulse_key,
            BrightThresholdFraction=bright_doms_threshold_fraction,
            BrightThresholdCharge=bright_doms_threshold_charge,
            OutputKey=bright_dom_key,
        )

    # combine exclusions in a single key for DOMs and TWs
    tray.AddModule(
        exclusions.get_combined_exclusions,
        combined_exclusion_key,
        dom_and_tw_exclusions=dom_and_tw_exclusions,
        partial_exclusion=partial_exclusion,
        output_key=combined_exclusion_key,
    )

    # mask pulses
    tray.AddModule(
        exclusions.get_valid_pulse_map,
        name + "MaskPulses",
        pulse_key=pulse_key,
        output_key=masked_pulses_key,
        dom_and_tw_exclusions=dom_and_tw_exclusions,
        partial_exclusion=partial_exclusion,
        verbose=False,
    )

    if merge_pulses_time_threshold is not None:
        # merge_pulses
        tray.AddModule(
            exclusions.get_merged_pulse_map,
            name + "MergePulses",
            pulse_key=masked_pulses_key,
            time_threshold=merge_pulses_time_threshold,
            output_key=masked_pulses_key + "Merged",
        )
        masked_pulses_key = masked_pulses_key + "Merged"
        added_keys.append(masked_pulses_key)

    return masked_pulses_key, excluded_dom_key, excluded_tw_key, added_keys
