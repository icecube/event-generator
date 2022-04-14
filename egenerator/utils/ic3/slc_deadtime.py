from icecube import dataclasses, icetray


class FlagSLCDeadTimeWindows(icetray.I3ConditionalModule):
    '''Module to flag deadtime windows after SLC hits

    Decision for hard local coincidence are performed 2450 ns after ATWD and
    FADC have launched. If no local coincidence is found in this time, readout
    is aborted and "ATWD clear" is initiated.
    Additionally, "for SLC pulses, FADC readout is reduced to only 3 samples
    of the *first 25* samples of the FADC (the highest amplitude and its
    two neighbors)"
    [See: https://wiki.icecube.wisc.edu/index.php/Soft_Local_Coincidence]

    This essentially means that pulses happening within this 2500ns time window
    (2450ns + 50ns rearming of ATWD) that aren't contained in the three FADC
    samples are lost. In itself this isn't necessarily an issue. However,
    our current calibration does not flag these deadtimes, which can result
    in problematic behaviour!

    This module attempts to implement an ad. hoc. guess for the deadtime after
    a recorded SLC hit. Note that this is done based on extract pulse series,
    so information on the FADC launch start isn't available. This results in
    an uncertainty on the exact dead times.
    However, since most of these issues will arise from individual early noise
    hits, we will assume here that the recorded and readout FADC SLC hit
    corresponds to the launch of the readout.
    A more proper solution should implement this at calibration and handling
    of the InIceRaw data.

    The DOMs will be searched for SLC pulses. If found, deadtimes will be added
    after each corresponding to the times:
        [SLC hit time + 50 ns, min(SLC hit time + 2500ns, next pulse > 1875ns)]
    '''
    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter(
            "PulseKey",
            "Name of the pulses for which to add SLC Deadtime windows.",
            'InIceDSTPulses')
        self.AddParameter(
            "RunOnPFrames",
            "True: run on P-frames; False: run on Q-Frames.",
            False)
        self.AddParameter(
            "OutputKey",
            "The key to which to save the SLC deadtime windows.",
            'SLCDeadTimeWindows')

    def Configure(self):
        """Configure module.
        """
        self._pulse_key = self.GetParameter("PulseKey")
        self._run_on_pframe = self.GetParameter("RunOnPFrames")
        self._output_key = self.GetParameter("OutputKey")

    def Physics(self, frame):
        """Modifies pulses as specified in modification.

        Parameters
        ----------
        frame : I3Frame
            Current i3 frame.
        """
        if self._run_on_pframe:
            self._add_slc_deadtime_windows(frame)

        self.PushFrame(frame)

    def DAQ(self, frame):
        """Modifies pulses as specified in modification.

        Parameters
        ----------
        frame : I3Frame
            Current i3 frame.
        """
        if not self._run_on_pframe:
            self._add_slc_deadtime_windows(frame)

        self.PushFrame(frame)

    def _add_slc_deadtime_windows(self, frame):

        # get pulses
        pulse_series = frame[self._pulse_key]
        if isinstance(pulse_series, dataclasses.I3RecoPulseSeriesMapMask) or \
           isinstance(pulse_series, dataclasses.I3RecoPulseSeriesMapUnion):
            pulse_series = pulse_series.apply(frame)

        deadtime_map = dataclasses.I3TimeWindowSeriesMap()

        # compute charge for each DOM
        for omkey, pulses in pulse_series:
            tws = dataclasses.I3TimeWindowSeries()
            for i, p in enumerate(pulses):
                # Pulse Flags:
                #   1: LC
                #   2: ATWD
                #   4: FADC
                # ==> 4 =  4 (FADC) [no LC flag means SLC]
                # ==> 7 =  1 + 2 + 4 (LC, ATWD, FADC)
                if p.flags == 4:

                    # found an SLC FADC pulse
                    # FADC readout is this and neighboring bins, each of 25ns
                    # width. So deadtime can start earliest 25ns afterwards.
                    # We'll go with 50 here to account for pulses at edge of
                    # the bins.
                    t_start = p.time + 50.

                    # find next pulse outside of ones that can be contributed
                    # to the same FADC readout. Note that wavedeform can add
                    # some uncertainties here. Thereforew we will start looking
                    # 200ns after the initial hit.
                    index = i
                    while (len(pulses) > index + 1 and
                            pulses[index+1].time < t_start + 200.):
                        index += 1

                    if index > i and pulses[index].time - t_start >= 1875:
                        # found a pulse after the start of our deadtime
                        t_next_pulse = pulses[index].time
                    else:
                        t_next_pulse = float('inf')
                    t_stop = min(p.time + 2500., t_next_pulse)
                    tws.append(dataclasses.I3TimeWindow(t_start, t_stop))

            if len(tws) > 0:
                deadtime_map[omkey] = tws

        # write to frame
        frame[self._output_key] = deadtime_map
