import numpy as np
from icecube import dataclasses, icetray

from ic3_labels.labels.utils import detector
from ic3_labels.labels.utils import muon as mu_utils
from ic3_labels.labels.modules.event_generator.utils import (
    get_track_energy_depositions,
)


class WriteMuonMCLosses(icetray.I3ConditionalModule):

    """Class to get track labels for Event-Generator

    The computed labels contain information on the n highest charge energy
    losses as well as quantile information of the remaining track energy
    depositions.
    """

    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("ExtendBoundary",
                          "Extend boundary of convex hull around IceCube "
                          "[in meters].",
                          60)
        self.AddParameter("RunOnDAQFrames",
                          "If True, the label module will run on DAQ frames "
                          "instead of Physics frames",
                          False)
        self.AddParameter("OutputBase",
                          "Output base key.",
                          'MCMuonLosses')
        self.AddParameter("PrimaryKey", "Name of the primary.", 'MCPrimary')
        self.AddParameter("I3MCTreeKey", "Name of the I3MCTree.", 'I3MCTree')

    def Configure(self):
        self._extend_boundary = self.GetParameter("ExtendBoundary")
        self._run_on_daq = self.GetParameter("RunOnDAQFrames")
        self._output_base = self.GetParameter("OutputBase")
        self._primary_key = self.GetParameter("PrimaryKey")
        self._mctree_key = self.GetParameter("I3MCTreeKey")

    def DAQ(self, frame):
        """Run on DAQ frames.

        Parameters
        ----------
        frame : I3Frame
            The current DAQ Frame
        """
        if self._run_on_daq:
            self.write_data(frame)

        self.PushFrame(frame)

    def Physics(self, frame):
        """Run on Physics frames.

        Parameters
        ----------
        frame : I3Frame
            The current Physics Frame
        """
        if not self._run_on_daq:
            self.write_data(frame)

        self.PushFrame(frame)

    def write_data(self, frame):
        """Add labels to frame

        Parameters
        ----------
        frame : I3Frame
            The frame to which to add the labels.
        """

        # get muon
        muon = mu_utils.get_muon(
            frame,
            primary=frame[self._primary_key],
            convex_hull=detector.icecube_hull,
        )

        # compute energy updates and high energy losses
        if muon is not None:
            energy_depositions_dict = get_track_energy_depositions(
                    mc_tree=frame[self._mctree_key],
                    track=muon,
                    num_to_remove=1000,
                    correct_for_em_loss=False,
                    extend_boundary=self._extend_boundary)

            cascades = energy_depositions_dict['cascades']
            rel_losses = energy_depositions_dict['relative_energy_losses']
        else:
            cascades = []
            rel_losses = np.array([])

        loss_energies = np.array([c.energy for c in cascades])
        loss_types = [int(c.pdg_encoding) for c in cascades]

        # write to frame
        frame.Put(self._output_base + 'RelLoss', dataclasses.I3VectorDouble(
            rel_losses))

        frame.Put(self._output_base + 'MuonEnergy', dataclasses.I3VectorDouble(
            loss_energies / rel_losses
        ))

        frame.Put(self._output_base + 'LossEnergy', dataclasses.I3VectorDouble(
            loss_energies
        ))

        frame.Put(self._output_base + 'LossType', dataclasses.I3VectorInt(
            loss_types
        ))
