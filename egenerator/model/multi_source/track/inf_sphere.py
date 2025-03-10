import tensorflow as tf

from egenerator.model.multi_source.base import MultiSource


class InfSphereTrackVariableMultiCascadeModel(MultiSource):
    """Variable Multi-Cascade Track Model

    Infinite track defined by an anchor point which is defined as the
    intersection with a sphere around the origin (entry_zenith, entry_azimuth)
    and a direction (zenith, azimuth) pulse time and energy at the entry.
    In addition to the track, multiple cascades are added along the track
    at variable distances.
    """

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
        if sorted(list(base_models.keys())) != ["cascade", "track"]:
            msg = "Expected only ['cascade', 'track'] bases, but got {!r}"
            raise ValueError(msg.format(base_models.keys()))

        # gather parameter names and sources
        sources = {"track": "track"}
        parameter_names = [
            "entry_zenith",
            "entry_azimuth",
            "entry_t",
            "entry_energy",
            "zenith",
            "azimuth",
        ]
        for index in range(config["num_cascades"]):
            cascade_name = f"cascade_{index:05d}"
            parameter_names.append(cascade_name + "_energy")
            parameter_names.append(cascade_name + "_distance")
            sources[cascade_name] = "cascade"

        return parameter_names, sources

    def get_model_parameters(self, parameters):
        """Get the input parameters for the individual sources.

        Parameters
        ----------
        parameters : tf.Tensor
            The input parameters for the MultiSource object.
            The input parameters of the individual Source objects
            are composed from these.

        Returns
        -------
        dict of tf.Tensor
            Returns a dictionary of (name: input_parameters) pairs, where
            name is the name of the Source and input_parameters is a
            tf.Tensor for the input parameters of this Source.
            Each input_parameters tensor has shape [..., n_parameters_i].
        """
        c = 0.299792458  # meter / ns
        d_thresh = 700  # meter

        config = self.configuration.config["config"]
        sphere_radius = config["sphere_radius"]

        e_zenith = parameters.params["entry_zenith"]
        e_azimuth = parameters.params["entry_azimuth"]
        entry_t = parameters.params["entry_t"]
        e_energy = parameters.params["entry_energy"]
        zenith = parameters.params["zenith"]
        azimuth = parameters.params["azimuth"]

        # calculate entry point of track to sphere
        e_pos_x = tf.sin(e_zenith) * tf.cos(e_azimuth) * sphere_radius
        e_pos_y = tf.sin(e_zenith) * tf.sin(e_azimuth) * sphere_radius
        e_pos_z = tf.cos(e_zenith) * sphere_radius

        # calculate direction vector
        dx = -tf.sin(zenith) * tf.cos(azimuth)
        dy = -tf.sin(zenith) * tf.sin(azimuth)
        dz = -tf.cos(zenith)

        # create a dictionary for the mapping of source_name: input_parameters
        source_parameter_dict = {
            "track": tf.stack(
                [
                    e_zenith,
                    e_azimuth,
                    entry_t,
                    e_energy,
                    zenith,
                    azimuth,
                ],
                axis=1,
            )
        }

        # -------------------------------
        # Compute parameters for Cascades
        # -------------------------------
        for index in range(config["num_cascades"]):
            cascade_name = f"cascade_{index:05d}"
            cascade_energy = parameters.params[f"cascade_{index:05d}_energy"]
            dist = parameters.params[f"cascade_{index:05d}_distance"]

            # calculate position and time of cascade
            cascade_x = e_pos_x + dist * dx
            cascade_y = e_pos_y + dist * dy
            cascade_z = e_pos_z + dist * dz
            cascade_time = entry_t + dist / c

            # make sure cascade energy does not turn negative
            cascade_energy = tf.clip_by_value(
                cascade_energy, 0.0, float("inf")
            )

            # set cascades far out to zero energy
            cascade_energy = tf.where(
                tf.abs(cascade_x) > d_thresh,
                tf.zeros_like(cascade_energy),
                cascade_energy,
            )
            cascade_energy = tf.where(
                tf.abs(cascade_y) > d_thresh,
                tf.zeros_like(cascade_energy),
                cascade_energy,
            )
            cascade_energy = tf.where(
                tf.abs(cascade_z) > d_thresh,
                tf.zeros_like(cascade_energy),
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


class InfSphereTrackEquidistantCascadeModel(MultiSource):
    """Equidistant Multi-Cascade Model

    Infinite track defined by an anchor point which is defined as the
    intersection with a sphere around the origin (entry_zenith, entry_azimuth)
    and a direction (zenith, azimuth) pulse time and energy at the entry.
    In addition to the track, multiple cascades are added along the track
    in equidistant spacing.
    """

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
        sphere_radius = config["sphere_radius"]
        cascade_spacing = config["cascade_spacing"]

        # compute maximum number of required cascades
        # A diagonal track through cylinder
        num_cascades = max(int(2 * sphere_radius / cascade_spacing), 1)

        self._untracked_data["num_cascades"] = num_cascades

        if sorted(list(base_models.keys())) != ["cascade", "track"]:
            msg = "Expected only ['cascade', 'track'] bases, but got {!r}"
            raise ValueError(msg.format(base_models.keys()))

        # gather parameter names and sources
        sources = {"track": "track"}
        parameter_names = [
            "entry_zenith",
            "entry_azimuth",
            "entry_t",
            "entry_energy",
            "zenith",
            "azimuth",
        ]
        for index in range(num_cascades):
            cascade_name = f"cascade_{index:05d}"
            parameter_name = f"energy_loss_{index:04d}"
            sources[cascade_name] = "cascade"
            parameter_names.append(parameter_name)

        return parameter_names, sources

    def get_model_parameters(self, parameters):
        """Get the input parameters for the individual sources.

        Parameters
        ----------
        parameters : tf.Tensor
            The input parameters for the MultiSource object.
            The input parameters of the individual Source objects
            are composed from these.

        Returns
        -------
        dict of tf.Tensor
            Returns a dictionary of (name: input_parameters) pairs, where
            name is the name of the Source and input_parameters is a
            tf.Tensor for the input parameters of this Source.
            Each input_parameters tensor has shape [..., n_parameters_i].
        """
        c = 0.299792458  # meter / ns
        d_thresh = 700  # meter

        config = self.configuration.config["config"]
        sphere_radius = config["sphere_radius"]
        cascade_spacing = config["cascade_spacing"]

        e_zenith = parameters.params["entry_zenith"]
        e_azimuth = parameters.params["entry_azimuth"]
        entry_t = parameters.params["entry_t"]
        e_energy = parameters.params["entry_energy"]
        zenith = parameters.params["zenith"]
        azimuth = parameters.params["azimuth"]

        # calculate entry point of track to sphere
        e_pos_x = tf.sin(e_zenith) * tf.cos(e_azimuth) * sphere_radius
        e_pos_y = tf.sin(e_zenith) * tf.sin(e_azimuth) * sphere_radius
        e_pos_z = tf.cos(e_zenith) * sphere_radius

        # calculate direction vector
        dx = -tf.sin(zenith) * tf.cos(azimuth)
        dy = -tf.sin(zenith) * tf.sin(azimuth)
        dz = -tf.cos(zenith)

        # create a dictionary for the mapping of source_name: input_parameters
        source_parameter_dict = {
            "track": tf.stack(
                [
                    e_zenith,
                    e_azimuth,
                    entry_t,
                    e_energy,
                    zenith,
                    azimuth,
                ],
                axis=1,
            )
        }

        # -------------------------------
        # Compute parameters for Cascades
        # -------------------------------
        for index in range(self._untracked_data["num_cascades"]):
            cascade_name = f"cascade_{index:05d}"
            cascade_energy = parameters.params[f"energy_loss_{index:04d}"]

            # calculate position and time of cascade
            dist = (index + 0.5) * cascade_spacing
            cascade_x = e_pos_x + dist * dx
            cascade_y = e_pos_y + dist * dy
            cascade_z = e_pos_z + dist * dz
            cascade_time = entry_t + dist / c

            # make sure cascade energy does not turn negative
            cascade_energy = tf.clip_by_value(
                cascade_energy, 0.0, float("inf")
            )

            # set cascades far out to zero energy
            cascade_energy = tf.where(
                tf.abs(cascade_x) > d_thresh,
                tf.zeros_like(cascade_energy),
                cascade_energy,
            )
            cascade_energy = tf.where(
                tf.abs(cascade_y) > d_thresh,
                tf.zeros_like(cascade_energy),
                cascade_energy,
            )
            cascade_energy = tf.where(
                tf.abs(cascade_z) > d_thresh,
                tf.zeros_like(cascade_energy),
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
