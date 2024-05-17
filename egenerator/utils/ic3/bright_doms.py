from icecube import dataclasses, icetray
import numpy as np


class AddBrightDOMs(icetray.I3ConditionalModule):
    """Module to add BrightDOMs for a given pulse series.

    BrightDOMs are DOMs which collect a large fraction of the event charge.
    These DOMs are not saturated and should in principle be correctly
    simulated. However, often when a single or only a few DOMs contain
    almost all of the event charge, this may be caused by an energy deposition
    in the close vicinity of that particular DOM. Systematic uncertainties
    on the hole-ice become more and more important, the closer the energy
    deposition is to the DOM. In addition, we often do not properly simulate
    energy depositions very close to DOMs due to DOM oversizing and an
    effective hole-ice parameterization as an angular acceptance curve.
    """

    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter(
            "PulseKey",
            "Name of the pulses for which to add BrightDOMs.",
            "SplitInIceDSTPulses",
        )
        self.AddParameter(
            "BrightThresholdFraction",
            "The threshold fraction of total event charge above which a DOM "
            "is considered as a bright DOM. A bright DOM is a DOM that "
            "fulfills: DOM charge > threshold fraction * event charge and "
            "which exceeds the threshold charge `BrightThresholdCharge`.",
            0.4,
        )
        self.AddParameter(
            "BrightThresholdCharge",
            "The threshold charge a DOM must exceed to be eligible as a "
            "bright DOM. See also `BrightThresholdFraction`.",
            100,
        )
        self.AddParameter(
            "OutputKey",
            "The key to which to save the BrightDOMs.",
            "BrightDOMs",
        )

    def Configure(self):
        """Configure AddBrightDOMs module."""
        self._pulse_key = self.GetParameter("PulseKey")
        self._threshold_fraction = self.GetParameter("BrightThresholdFraction")
        self._threshold_charge = self.GetParameter("BrightThresholdCharge")
        self._output_key = self.GetParameter("OutputKey")

    def Physics(self, frame):
        """Adds the BrightDOMs to the frame.

        Parameters
        ----------
        frame : I3Frame
            Current pyhsics I3Frame.
        """

        # get pulses
        pulse_series = dataclasses.I3RecoPulseSeriesMap.from_frame(
            frame,
            self._pulse_key,
        )

        dom_charges = {}
        dom_charges_list = []

        # compute charge for each DOM
        for omkey, pulses in pulse_series:
            dom_charge = np.sum([p.charge for p in pulses])
            dom_charges_list.append(dom_charge)
            dom_charges[omkey] = dom_charge

        event_charge = np.sum(dom_charges_list)
        bright_doms = dataclasses.I3VectorOMKey()

        for omkey, dom_charge in dom_charges.items():
            if dom_charge > self._threshold_fraction * event_charge:
                if dom_charge > self._threshold_charge:
                    bright_doms.append(omkey)

        # write to frame
        frame[self._output_key] = bright_doms

        # push frame
        self.PushFrame(frame)
