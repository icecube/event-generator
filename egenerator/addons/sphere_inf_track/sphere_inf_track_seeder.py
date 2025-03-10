import math
from icecube import icetray, dataclasses

from ic3_labels.labels.modules.event_generator.muon_track_labels import (
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
            "The output frame key to which the seed labels will be written.",
            None,
        )
        self.AddParameter(
            "sphere_radius",
            "The radius of the sphere around IceCube [in meters].",
            750,
        )
        self.AddParameter(
            "replacement_values",
            "Replacement values for non-finite values.",
            {"entry_energy": 10000.0},
        )

    def Configure(self):
        """Configure"""
        self._seed_key = self.GetParameter("seed_key")
        self._output_key = self.GetParameter("output_key")
        self._sphere_radius = self.GetParameter("sphere_radius")
        self._replacement_values = self.GetParameter("replacement_values")

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

        # replace non-finite values
        for key, value in self._replacement_values.items():
            if not math.isfinite(geometry_values[key]):
                geometry_values[key] = value

        frame[self._output_key] = geometry_values
        self.PushFrame(frame)


class SphereInfTrackI3ParticleConverter(icetray.I3ConditionalModule):
    """Class to convert sphere inf track labels to I3Particle"""

    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter(
            "labels_key",
            "The labels which will be used to create the I3Particle.",
        )
        self.AddParameter(
            "output_key",
            "The output frame key to which the I3Particle will be written.",
            None,
        )

    def Configure(self):
        """Configure"""
        self._labels_key = self.GetParameter("labels_key")
        self._output_key = self.GetParameter("output_key")

        if self._output_key is None:
            self._output_key = self._labels_key + "_I3Particle"

    def Physics(self, frame):
        """Convert sphere inf track labels to I3Particle

        Parameters
        ----------
        frame : I3Frame
            The current I3frame.
        """

        particle = dataclasses.I3Particle()
        particle.time = frame[self._labels_key]["entry_t"]
        particle.pos = dataclasses.I3Position(
            frame[self._labels_key]["entry_x"],
            frame[self._labels_key]["entry_y"],
            frame[self._labels_key]["entry_z"],
        )
        particle.dir = dataclasses.I3Direction(
            frame[self._labels_key]["zenith"],
            frame[self._labels_key]["azimuth"],
        )
        particle.energy = frame[self._labels_key]["entry_energy"]
        particle.length = frame[self._labels_key]["finite_length"]

        particle.shape = dataclasses.I3Particle.InfiniteTrack
        particle.fit_status = dataclasses.I3Particle.FitStatus.OK

        frame[self._output_key] = particle
        self.PushFrame(frame)
