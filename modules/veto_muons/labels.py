from icecube import icetray, dataclasses


class MCVetoMuonLabels(icetray.I3ConditionalModule):

    """Simple MCVetoMuonLabels Generator

    Important: 'MCVetoMuonInjectionInfo' must exist in veto frames!!
    """

    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter('OutputKey', 'Output Key.', 'MCVetoMuonLabels')

    def Configure(self):
        self._output_key = self.GetParameter("OutputKey")

    def Physics(self, frame):
        """Create Labels

        Parameters
        ----------
        frame : I3Frame
            The current physics I3Frame.
        """
        labels = dataclasses.I3MapStringDouble()

        if 'MCVetoMuonInjectionInfo' in frame:
            muon_energy = frame['MCVetoMuonInjectionInfo']['muon_energy']
            if 'is_correlated' in frame['MCVetoMuonInjectionInfo']:
                p_is_veto_event = frame['MCVetoMuonInjectionInfo'][
                    'is_correlated']
            else:
                p_is_veto_event = 1.
        else:
            muon_energy = 0.
            p_is_veto_event = 0.

        labels['muon_energy'] = muon_energy
        labels['p_is_veto_event'] = p_is_veto_event

        frame[self._output_key] = labels

        self.PushFrame(frame)
