from __future__ import division, print_function
import logging
import tensorflow as tf

from egenerator.model.multi_source.base import MultiSource


class StochasticTrackModel(MultiSource):

    def __init__(self, logger=None):
        """Instantiate Source class

        Parameters
        ----------
        logger : logging.logger, optional
            The logger to use.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(StochasticTrackModel, self).__init__(logger=self._logger)

    def get_parameters_and_mapping(self, config, base_sources):
        """Get parameter names and their ordering as well as source mapping.

        This is a pure virtual method that must be implemented by
        derived class.

        Parameters
        ----------
        config : dict
            A dictionary of settings.
        base_sources : dict of Source objects
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
        num_cascades = config["num_cascades"]
        self._untracked_data["num_cascades"] = num_cascades

        if sorted(list(base_sources.keys())) != ["cascade", "track"]:
            msg = "Expected only ['cascade', 'track'] bases, but got {!r}"
            raise ValueError(msg.format(base_sources.keys()))

        # gather parameter names and sources
        sources = {"track": "track"}
        parameter_names = [
            "zenith",
            "azimuth",
            "track_anchor_x",
            "track_anchor_y",
            "track_anchor_z",
            "track_anchor_time",
            "track_energy",
            "track_distance_start",
            "track_distance_end",
            "track_stochasticity",
        ]
        if num_cascades >= 1:
            parameter_names.append("cascade_0000_energy")
            sources["cascade_0000"] = "cascade"

            if num_cascades > 1:
                for index in range(1, num_cascades):
                    cascade_name = "cascade_{:04d}".format(index)
                    sources[cascade_name] = "cascade"
                    parameter_names.append(cascade_name + "_energy")
                    parameter_names.append(cascade_name + "_distance")

        # add snowstorm parameters
        num_snowstorm_params = 0
        if "snowstorm_parameter_names" in config:
            for param_name, num in config["snowstorm_parameter_names"]:
                num_snowstorm_params += num
                for i in range(num):
                    parameter_names.append(param_name.format(i))
        self._untracked_data["num_snowstorm_params"] = num_snowstorm_params

        return parameter_names, sources

    def get_source_parameters(self, parameters):
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

        """
        c = 0.299792458  # meter / ns
        distance = 500  # meter

        anchor_x = parameters.params["track_anchor_x"]
        anchor_y = parameters.params["track_anchor_y"]
        anchor_z = parameters.params["track_anchor_z"]
        zenith = parameters.params["zenith"]
        azimuth = parameters.params["azimuth"]
        anchor_time = parameters.params["track_anchor_time"]
        d_start = parameters.params["track_distance_start"]
        d_end = parameters.params["track_distance_end"]

        num_snowstorm_params = self._untracked_data["num_snowstorm_params"]
        snowstorm_params = parameters[:, -num_snowstorm_params:]

        # calculate direction vector
        dir_x = -tf.sin(zenith) * tf.cos(azimuth)
        dir_y = -tf.sin(zenith) * tf.sin(azimuth)
        dir_z = -tf.cos(zenith)

        # create a dictionary for the mapping of source_name: input_parameters
        source_parameter_dict = {}

        # -----------------------------------
        # create parameters for track segment
        # -----------------------------------

        # compute vertex of track
        track_x = anchor_x + d_start * dir_x
        track_y = anchor_y + d_start * dir_y
        track_z = anchor_z + d_start * dir_z
        track_time = anchor_time + d_start / c

        track_length = d_end - d_start

        # parameters: x, y, z, zenith, azimuth, energy, time, length, stoch
        track_parameters = tf.concat(
            (
                tf.stack(
                    [
                        track_x,
                        track_y,
                        track_z,
                        zenith,
                        azimuth,
                        parameters.params["track_energy"],
                        track_time,
                        track_length,
                        parameters.params["track_stochasticity"],
                    ],
                    axis=1,
                ),
                snowstorm_params,
            ),
            axis=-1,
        )

        # add to source parameter mapping
        source_parameter_dict["track"] = track_parameters

        # -------------------------------
        # Compute parameters for Cascades
        # -------------------------------
        if self._untracked_data["num_cascades"] >= 1:

            # highest energy cascade that is being used as vertex
            # parameters: x, y, z, zenith, azimuth, energy, time
            source_parameter_dict["cascade_0000"] = tf.concat(
                (
                    tf.stack(
                        [
                            anchor_x,
                            anchor_y,
                            anchor_z,
                            zenith,
                            azimuth,
                            parameters.params["cascade_0000_energy"],
                            anchor_time,
                        ],
                        axis=1,
                    ),
                    snowstorm_params,
                ),
                axis=-1,
            )

            # Now go through the rest of the cascades
            for index in range(1, self._untracked_data["num_cascades"]):

                cascade_name = "cascade_{:04d}".format(index)
                distance = parameters.params[cascade_name + "_distance"]

                # compute vertex
                cascade_x = anchor_x + distance * dir_x
                cascade_y = anchor_y + distance * dir_y
                cascade_z = anchor_z + distance * dir_z
                cascade_time = anchor_time + distance / c

                # Add cascade source
                # parameters: x, y, z, zenith, azimuth, energy, time
                source_parameter_dict[cascade_name] = tf.concat(
                    (
                        tf.stack(
                            [
                                cascade_x,
                                cascade_y,
                                cascade_z,
                                zenith,
                                azimuth,
                                parameters.params[cascade_name + "_energy"],
                                cascade_time,
                            ],
                            axis=1,
                        ),
                        snowstorm_params,
                    ),
                    axis=-1,
                )

        return source_parameter_dict
