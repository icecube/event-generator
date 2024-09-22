from icecube import icetray

from ic3_labels.labels.modules.muon_track_labels import (
    get_sphere_inf_track_geometry_values,
)


class SphereInfTrackSeedConverter(icetray.I3ConditionalModule):
    """Class to convert a I3Particle to a seed for the sphere inf track model"""

    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter(
            "seed_key",
            "The I3Particle which will be used as the seed.",
        )
        self.AddParameter(
            "output_key",
            "The output frame key to which the clusters will be written.",
            None,
        )
        self.AddParameter(
            "sphere_radius",
            "The radius of the sphere around IceCube [in meters].",
            750,
        )

    def Configure(self):
        """Configure"""
        self._seed_key = self.GetParameter("seed_key")
        self._output_key = self.GetParameter("output_key")
        self._sphere_radius = self.GetParameter("sphere_radius")

        if self._output_key is None:
            self._output_key = self._seed_key + "_SphereInfTrackSeed"

    def Physics(self, frame):
        """Convert I3Particle to a seed for the sphere inf track model

        Parameters
        ----------
        frame : I3Frame
            The current I3frame.
        """

        # get track
        track = frame[self._seed_key]

        # get geometry values
        geometry_values, _, _ = get_sphere_inf_track_geometry_values(
            muon=track,
            sphere_radius=self._sphere_radius,
        )

        # add energy
        geometry_values["entry_energy"] = track.energy

        frame[self._output_key] = geometry_values
        self.PushFrame(frame)
