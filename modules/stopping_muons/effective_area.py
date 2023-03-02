from icecube import icetray, dataclasses
from icecube import MuonGun


class AddMuonGunEffectiveArea(icetray.I3ConditionalModule):

    """Calculate MuonGun Effective Area

    Note: this assumes that the S-frame exists in the i3-File.
    When combining multiple generated i3-files, one must scale the calculated
    effective area by the number of combined files.
    """

    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter('OutputKey', 'Output Key.', 'MuonGunEffectiveArea')
        self.AddParameter(
            'Generator',
            'Generator object to use. If none is provided, it is assumed that '
            'one exists in a prior S-frame',
            None,
        )

    def Configure(self):
        self._output_key = self.GetParameter("OutputKey")
        self.generator = self.GetParameter("Generator")

        # get generator object from previous S-frame
        if self.generator is None:
            self.generator = frame['GenerateCosmicRayMuons']

    def DAQ(self, frame):
        """Compute effective area

        Parameters
        ----------
        frame : I3Frame
            The current physics I3Frame.
        """
        eff_area = dataclasses.I3MapStringDouble()

        mctree = frame['I3MCTree']
        primary = mctree.primaries[0]
        muon = mctree.get_daughters(primary)[0]

        bundle = MuonGun.BundleConfiguration(
            [MuonGun.BundleEntry(0, muon.energy)])

        area_weight = 1./self.generator.generated_events(primary, bundle)
        frame[self._output_key] = dataclasses.I3Double(area_weight)

        self.PushFrame(frame)
