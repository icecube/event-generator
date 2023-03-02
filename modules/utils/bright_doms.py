#!/usr/bin/env python
# -*- coding: utf-8 -*-
from icecube import dataclasses, icetray
import numpy as np


class AddBrightDOMs(icetray.I3ConditionalModule):
    '''Module to add BrightDOMs for a given pulse series.
    '''
    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("PulseKey",
                          "Name of the pulses for which to add BrightDOMs.",
                          'InIceDSTPulses')
        self.AddParameter("BrightThreshold",
                          "The threshold above which a DOM is considered as a "
                          "bright DOM. A bright DOM is a DOM that has a "
                          "charge that is above threshold * average charge.",
                          10)
        self.AddParameter("OutputKey",
                          "The key to which to save the BrightDOMs.",
                          'BrightDOMs')

    def Configure(self):
        """Configure AddBrightDOMs module.
        """
        self._pulse_key = self.GetParameter("PulseKey")
        self._threshold = self.GetParameter("BrightThreshold")
        self._output_key = self.GetParameter("OutputKey")

    def Physics(self, frame):
        """Adds the BrightDOMs to the frame.

        Parameters
        ----------
        frame : I3Frame
            Current pyhsics I3Frame.
        """

        # get pulses
        pulse_series = frame[self._pulse_key]
        if isinstance(pulse_series, dataclasses.I3RecoPulseSeriesMapMask) or \
           isinstance(pulse_series, dataclasses.I3RecoPulseSeriesMapUnion):
            pulse_series = pulse_series.apply(frame)

        dom_charges = {}
        dom_charges_list = []

        # compute charge for each DOM
        for omkey, pulses in pulse_series:
            dom_charge = np.sum([p.charge for p in pulses])
            dom_charges_list.append(dom_charge)
            dom_charges[omkey] = dom_charge

        average_charge = np.mean(dom_charges_list)
        bright_doms = dataclasses.I3VectorOMKey()

        for omkey, dom_charge in dom_charges.items():
            if dom_charge > self._threshold * average_charge:
                bright_doms.append(omkey)

        # write to frame
        frame[self._output_key] = bright_doms

        # push frame
        self.PushFrame(frame)
