import logging
import tensorflow as tf
import numpy as np

from egenerator.model.multi_source.base import MultiSource


class TrackEquidistantCascadeModel(MultiSource):
    """Equidistant Multi-Cascade Model

    Infinite track defined by an anchor point (x, y, z, t) and a direction
    (zenith, azimuth). The track is then composed of equidistantly spaced
    cascades along the track.

    ----The cascades are placed relative to the anchor
    vertex, e.g. the anchor point is also the
    """

    def __init__(self, logger=None):
        """Instantiate Source class

        Parameters
        ----------
        logger : logging.logger, optional
            The logger to use.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(TrackEquidistantCascadeModel, self).__init__(logger=self._logger)

    def get_parameters_and_mapping(self, config, base_models):
        """Get parameter names and their ordering as well as source mapping.

        This is a pure virtual method that must be implemented by
        derived class.

        Parameters
        ----------
        config : dict
            A dictionary of settings.
        base_models : dict of Source objects
            A dictionary of sources. These sources are used as a basis for
            the MultiSource object. The event hypothesis can be made up of
            multiple sources which may be created from one or more
            base source objects.

        Returns
        -------
        list of str
            A list of parameter names of the MultiSource object.
        dict
            This describes the sources which compose the event hypothesis.
            The dictionary is a mapping from source_name (str) to
            base_source (str). This mapping allows the reuse of a single
            source component instance. For instance, a muon can be build up of
            multiple cascades. However, all cascades should use the same
            underlying model. Hence, in this case only one base_source is
            required: the cascade source. The mapping will then map all
            cascades in the hypothesis to this one base cascade source.
        """
        max_num_cascades = config["max_num_cascades"]
        cascade_spacing = config["cascade_spacing"]
        cylinder_radius = config["cylinder_radius"]
        cylinder_extension = config["cylinder_extension"]

        # compute maximum number of required cascades
        # A diagonal track through cylinder
        max_length = 2 * np.sqrt(cylinder_radius**2 + cylinder_extension**2)
        num_cascades = max(int(max_length / cascade_spacing), 1)

        if num_cascades > max_num_cascades:
            msg = "Maximum number of cascades {} ".format(max_num_cascades)
            msg += "might not cover complete track with spacing of {}!".format(
                cascade_spacing
            )
            self._logger.warning(msg)
            num_cascades = max_num_cascades

        self._untracked_data["cascade_spacing"] = cascade_spacing
        self._untracked_data["cylinder_radius"] = cylinder_radius
        self._untracked_data["cylinder_extension"] = cylinder_extension
        self._untracked_data["num_cascades"] = num_cascades

        if sorted(list(base_models.keys())) != ["cascade"]:
            msg = "Expected only ['cascade'] bases, but got {!r}"
            raise ValueError(msg.format(base_models.keys()))

        # gather parameter names and sources
        sources = {}
        parameter_names = [
            "anchor_x",
            "anchor_y",
            "anchor_z",
            "zenith",
            "azimuth",
            "anchor_time",
        ]
        for index in range(num_cascades):
            cascade_name = "cascade_{:05d}".format(index)
            sources[cascade_name] = "cascade"
            parameter_names.append(cascade_name + "_energy")

        return parameter_names, sources

    def get_cylinder_intersection_distance(self, x, y, z, dx, dy, dz):
        """Get distance along track to cylinder intersection

        Parameters
        ----------
        x : array_like
            The x-coordinate of the track anchor point.
            Shape: [n_events]
        y : array_like
            The y-coordinate of the track anchor point.
            Shape: [n_events]
        z : array_like
            The z-coordinate of the track anchor point.
            Shape: [n_events]
        dx : array_like
            The x-component of the normalized track direction vector.
            Shape: [n_events]
        dy : array_like
            The y-component of the normalized track direction vector.
            Shape: [n_events]
        dz : array_like
            The z-component of the normalized track direction vector.
            Shape: [n_events]

        Returns
        -------
        array_like
            The distance along the track to the entry point into the cylinder
            of the infinite track.
            Shape: [n_events]
        """
        cascade_spacing = self._untracked_data["cascade_spacing"]
        num_cascades = self._untracked_data["num_cascades"]
        r = self._untracked_data["cylinder_radius"]
        c_z = self._untracked_data["cylinder_extension"]

        # vector of track vertex to cylinder center (0, 0, 0)
        h_x = 0.0 - x
        h_y = 0.0 - y
        h_z = 0.0 - z

        # distance between track vertex and closest approach of infinite track
        dist_closest = dx * h_x + dy * h_y + dz * h_z

        # distances to points with sqrt(x**2 + y**2) == r**2
        # solution for |dz| != 1:
        # t = -x**2*dy**2 + 2*x*y*dx*dy - y**2*dx**2 + (dx**2 + dy**2)*r**2
        # l = +- (sqrt(t) - x*dx - y*dy) / (dx**2 + dy**2)
        t1 = dx**2 + dy**2
        t2 = -(x**2) * dy**2 + 2 * x * y * dx * dy - y**2 * dx**2 + t1 * r**2
        l_p = (tf.math.sqrt(t2) - x * dx - y * dy) / t1
        l_n = (-tf.math.sqrt(t2) - x * dx - y * dy) / t1

        # distances to points with z == +- cylinder_ext
        l_t = (c_z - z) / dz
        l_b = (-c_z - z) / dz

        # only accept points that are on the edge of the cylinder (plus tol)
        tol = 1  # 1 meter tolerance
        l_p = tf.where(
            tf.math.abs(z + l_p * dz) < c_z + tol,
            l_p,
            tf.ones_like(l_p) * float("inf"),
        )
        l_n = tf.where(
            tf.math.abs(z + l_n * dz) < c_z + tol,
            l_n,
            tf.ones_like(l_n) * float("inf"),
        )

        l_t = tf.where(
            (x + l_t * dx) ** 2 + (y + l_t * dy) ** 2 < (r + tol) ** 2,
            l_t,
            tf.ones_like(l_t) * float("inf"),
        )
        l_b = tf.where(
            (x + l_b * dx) ** 2 + (y + l_b * dy) ** 2 < (r + tol) ** 2,
            l_b,
            tf.ones_like(l_b) * float("inf"),
        )

        # now choose minimum distance since we want the entry, i.e. earliest
        dist_cyl = tf.reduce_min(
            tf.stack([l_p, l_n, l_t, l_b], axis=1), axis=1
        )

        # if track did not hit the cylinder, the distance will not be finite,
        # in this case set the position to the closest approach point minus
        # half of the binned track length
        dist_cyl = tf.where(
            tf.math.is_finite(dist_cyl),
            dist_cyl,
            dist_closest - 0.5 * (cascade_spacing * max(0, num_cascades - 1)),
        )

        return dist_cyl

    def get_model_parameters(self, parameters):
        """Get the input parameters for the individual sources.

        Parameters
        ----------
        parameters : tf.Tensor
            The input parameters for the MultiSource object.
            The input parameters of the individual Source objects are composed
            from these.

        Returns
        -------
        dict of tf.Tensor
            Returns a dictionary of (name: input_parameters) pairs, where
            name is the name of the Source and input_parameters is a tf.Tensor
            for the input parameters of this Source.
            Each input_parameters tensor has shape [..., n_parameters_i].
        """
        c = 0.299792458  # meter / ns
        d_thresh = 700  # meter

        anchor_x = parameters.params["anchor_x"]
        anchor_y = parameters.params["anchor_y"]
        anchor_z = parameters.params["anchor_z"]
        zenith = parameters.params["zenith"]
        azimuth = parameters.params["azimuth"]
        anchor_time = parameters.params["anchor_time"]

        # calculate direction vector
        dx = -tf.sin(zenith) * tf.cos(azimuth)
        dy = -tf.sin(zenith) * tf.sin(azimuth)
        dz = -tf.cos(zenith)

        # compute entry distance of track to cylinder
        dist_cyl = self.get_cylinder_intersection_distance(
            anchor_x, anchor_y, anchor_z, dx, dy, dz
        )

        # compute entry point of track to cylinder
        start_x = anchor_x + dist_cyl * dx
        start_y = anchor_y + dist_cyl * dy
        start_z = anchor_z + dist_cyl * dz
        start_time = anchor_time + dist_cyl / c

        # create a dictionary for the mapping of source_name: input_parameters
        source_parameter_dict = {}

        # -------------------------------
        # Compute parameters for Cascades
        # -------------------------------
        for index in range(self._untracked_data["num_cascades"]):
            cascade_name = "cascade_{:05d}".format(index)
            cascade_energy = parameters.params[cascade_name + "_energy"]

            # calculate position and time of cascade
            dist = (index + 0.5) * self._untracked_data["cascade_spacing"]
            cascade_x = start_x + dist * dx
            cascade_y = start_y + dist * dy
            cascade_z = start_z + dist * dz
            cascade_time = start_time + dist / c

            # make sure cascade energy does not turn negative
            cascade_energy = tf.clip_by_value(
                cascade_energy, 0.0, float("inf")
            )

            # set cascades far out to zero energy
            cascade_energy = tf.where(
                tf.abs(cascade_x) > d_thresh,
                cascade_energy * 0.0,
                cascade_energy,
            )
            cascade_energy = tf.where(
                tf.abs(cascade_y) > d_thresh,
                cascade_energy * 0.0,
                cascade_energy,
            )
            cascade_energy = tf.where(
                tf.abs(cascade_z) > d_thresh,
                cascade_energy * 0.0,
                cascade_energy,
            )

            source_parameter_dict[cascade_name] = tf.stack(
                [
                    cascade_x,
                    cascade_y,
                    cascade_z,
                    zenith,
                    azimuth,
                    cascade_energy,
                    cascade_time,
                ],
                axis=1,
            )

        return source_parameter_dict
