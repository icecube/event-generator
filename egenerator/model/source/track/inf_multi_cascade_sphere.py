"""Todo: Refactor code. As is this copies large parts of the
EnteringSphereInfTrackCascade and DefaultCascadeModel classes.
This code duplication is not ideal and should be reduced.
"""

import tensorflow as tf
import numpy as np

from tfscripts import layers as tfs
from tfscripts.weights import new_weights

from egenerator.model.source.base import Source
from egenerator.model.source.cascade.default import setup_cascade_model
from egenerator.utils.cascades import shift_to_maximum
from egenerator.utils.build_components import build_decoder
from egenerator.utils import basis_functions
from egenerator.utils import (
    detector,
    angles,
    dom_acceptance,
    tf_helpers,
)


class EnteringSphereInfTrackMultiCascade(Source):

    def _build_architecture(self, config, name=None):
        """Set up and build architecture: create and save all model weights.

        This is a virtual method which must be implemented by the derived
        source class.

        Parameters
        ----------
        config : dict
            A dictionary of settings that fully defines the architecture.
        name : str, optional
            The name of the source.
            If None is provided, the class name __name__ will be used.

        Returns
        -------
        list of str
            A list of parameter names. These parameters fully describe the
            source hypothesis. The model expects the hypothesis tensor input
            to be in the same order as this returned list.
        """
        self.assert_configured(False)

        if (
            config["scale_charge_by_angular_acceptance"]
            and config["scale_charge_by_relative_angular_acceptance"]
        ):
            raise ValueError(
                "Only one of 'scale_charge_by_angular_acceptance' "
                "and 'scale_charge_by_relative_angular_acceptance' "
                "can be set to True."
            )

        # -----------------------------------------------------------
        # Define input parameters of track + multi-cascade hypothesis
        # -----------------------------------------------------------
        sphere_radius = config["sphere_radius"]
        cascade_spacing = config["cascade_spacing"]

        # compute maximum number of required cascades
        # A diagonal track through cylinder
        n_total_cascades = max(int(2 * sphere_radius / cascade_spacing), 1)

        self._untracked_data["n_total_cascades"] = n_total_cascades

        # gather parameter names and sources
        parameter_names = [
            "entry_zenith",
            "entry_azimuth",
            "entry_t",
            "entry_energy",
            "zenith",
            "azimuth",
        ]

        num_snowstorm_params = 0
        if "snowstorm_parameter_names" in config:
            for param_name, num in config["snowstorm_parameter_names"]:
                num_snowstorm_params += num
                for i in range(num):
                    parameter_names.append(param_name.format(i))
        self.untracked_data["n_parameters_track"] = len(parameter_names)
        self.untracked_data["n_snowstorm_parameters"] = num_snowstorm_params

        # add parameters for energy losses of cascades
        for index in range(n_total_cascades):
            parameter_name = f"energy_loss_{index:04d}"
            parameter_names.append(parameter_name)

        # compute inputs for track
        num_inputs = 21 + num_snowstorm_params

        if config["add_anisotropy_angle"]:
            num_inputs += 2

        if config["add_dom_angular_acceptance"]:
            num_inputs += 1

        if config["add_dom_coordinates"]:
            num_inputs += 3

        if config["num_local_vars"] > 0:
            self._untracked_data["local_vars"] = new_weights(
                shape=[1, 86, 60, config["num_local_vars"]],
                float_precision=config["float_precision"],
                name="local_dom_input_variables",
            )
            num_inputs += config["num_local_vars"]

        # -------------------------------------------
        # convolutional hex3d layers over X_IC86 data
        # -------------------------------------------
        self._untracked_data["conv_hex3d_layer"] = tfs.ConvNdLayers(
            input_shape=[-1, 86, 60, num_inputs],
            filter_size_list=config["filter_size_list"],
            num_filters_list=config["num_filters_list"],
            pooling_type_list=None,
            pooling_strides_list=[1, 1, 1, 1],
            pooling_ksize_list=[1, 1, 1, 1],
            use_dropout_list=config["use_dropout_list"],
            padding_list="SAME",
            strides_list=[1, 1, 1, 1],
            use_batch_normalisation_list=config["use_batch_norm_list"],
            activation_list=config["activation_list"],
            use_residual_list=config["use_residual_list"],
            hex_zero_out_list=False,
            dilation_rate_list=None,
            hex_num_rotations_list=1,
            method_list=config["method_list"],
            float_precision=config["float_precision"],
        )

        # setup CNN layers for cascade model
        if "additional_label_names" in config["config_cascade"]["config"]:
            raise NotImplementedError

        cnn_layers, local_vars, parameter_names_cascade = setup_cascade_model(
            config["config_cascade"]["config"],
        )

        if local_vars is not None:
            self._untracked_data["local_vars_cascade"] = local_vars

        self._untracked_data["conv_hex3d_layer_cascade"] = cnn_layers
        self.untracked_data["n_parameters_cascade"] = len(
            parameter_names_cascade
        )

        # dirty hack to build decoders for cascade
        # Careful: This will only work for static decoders
        # without any trainable parameters!
        self._untracked_data["decoder_cascade"] = build_decoder(
            config["config_cascade"]["decoder_settings"],
            allow_rebuild_base_decoders=True,
        )
        self._untracked_data["decoder_cascade_charge"] = build_decoder(
            config["config_cascade"]["decoder_charge_settings"],
            allow_rebuild_base_decoders=True,
        )

        if isinstance(self._untracked_data["decoder_cascade"], tf.Module):
            if self._untracked_data["decoder_cascade"].trainable_variables:
                raise ValueError("Cascade decoder is not a static decoder!")

        if isinstance(
            self._untracked_data["decoder_cascade_charge"], tf.Module
        ):
            if self._untracked_data[
                "decoder_cascade_charge"
            ].trainable_variables:
                raise ValueError(
                    "Cascade charge decoder is not a static decoder!"
                )

        return parameter_names

    def get_cascade_parameters(self, parameters_all):
        """Get the input parameters for the individual cascade energy losses.

        Parameters
        ----------
        parameters : tf.Tensor
            The input parameters for the entire track + multi-cascade hypothesis.

        Returns
        -------
        tf.Tensor
            The input parameters for the individual cascade energy losses.
            Each DOM may have contributions from different cascade hypotheses.
            Shape: [n_batch * num_cascades, 86, 60, n_params]
        """
        c = 0.299792458  # meter / ns
        d_thresh = 700  # meter

        config = self.configuration.config["config"]
        sphere_radius = config["sphere_radius"]
        cascade_spacing = config["cascade_spacing"]
        n_cascades = self._untracked_data["n_total_cascades"]

        # Shape: [n_batch]
        e_zenith = parameters_all.params["entry_zenith"]
        e_azimuth = parameters_all.params["entry_azimuth"]
        entry_t = parameters_all.params["entry_t"]
        zenith = parameters_all.params["zenith"]
        azimuth = parameters_all.params["azimuth"]

        # calculate entry point of track to sphere
        # Shape: [n_batch]
        e_pos_x = tf.sin(e_zenith) * tf.cos(e_azimuth) * sphere_radius
        e_pos_y = tf.sin(e_zenith) * tf.sin(e_azimuth) * sphere_radius
        e_pos_z = tf.cos(e_zenith) * sphere_radius

        # calculate direction vector
        # Shape: [n_batch]
        dx = -tf.sin(zenith) * tf.cos(azimuth)
        dy = -tf.sin(zenith) * tf.sin(azimuth)
        dz = -tf.cos(zenith)

        # input parameters for all
        cascade_parameters = []

        # -------------------------------
        # Compute parameters for Cascades
        # -------------------------------
        for index in range(self._untracked_data["n_total_cascades"]):
            cascade_energy = parameters_all.params[f"energy_loss_{index:04d}"]

            # calculate position and time of cascade
            # Shape: [n_batch]
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

            parameters_i = [
                cascade_x,
                cascade_y,
                cascade_z,
                zenith,
                azimuth,
                cascade_energy,
                cascade_time,
            ]

            if "snowstorm_parameter_names" in config:
                for param_name, num in config["snowstorm_parameter_names"]:
                    for i in range(num):
                        parameters_i.append(
                            parameters_all.params[param_name.format(i)]
                        )

            cascade_parameters.append(tf.stack(parameters_i, axis=-1))

        # shape of cascade_parameters: [-1, n_params, n_cascades]
        num_features = cascade_parameters[0].get_shape().as_list()[-1]
        cascade_parameters = tf.stack(cascade_parameters, axis=2)
        print(
            "cascade_parameters [-1, n_params, n_cascades]", cascade_parameters
        )

        # parameters: x, y, z, zenith, azimuth, energy, time
        params_reshaped = tf.reshape(
            cascade_parameters, [-1, 1, 1, num_features, n_cascades]
        )
        parameter_list = tf.unstack(params_reshaped, axis=3)

        # Choose the number of cascades to evaluate by choosing
        # based on a simplified guess on the expected light yield
        # by taking into account the energy and distance of the cascade
        # todo: find most generic function for: light_yield(energy, distance)

        # compute distance of DOM to each cascade
        # Shape: [n_batch, 86, 60, n_cascade] = [86, 60, 1] - [n_batch, 1, 1, n_cascade]
        dx = detector.x_coords[..., 0, np.newaxis] - parameter_list[0]
        dy = detector.x_coords[..., 1, np.newaxis] - parameter_list[1]
        dz = detector.x_coords[..., 2, np.newaxis] - parameter_list[2]

        # stabilize with 1e-1 (10cm) when distance approaches DOM radius
        # Shape: [n_batch, 86, 60, n_cascade]
        distance = tf.sqrt(dx**2 + dy**2 + dz**2) + 1e-1

        # ToDo: find most generic function for: light_yield(energy, distance)
        # Shape: [n_batch, 86, 60, n_cascade]
        exp_light_yield = 1.0 / distance

        # Find the top n cascades that are likely to contribute most to DOM
        # Shape: [n_batch, 86, 60, n_sel]
        values, indices = tf.math.top_k(
            exp_light_yield, k=config["num_cascades"]
        )
        print("values", values)
        print("indices", indices)
        # tf.print("values", values[0, 0, 0, :])
        # tf.print("indices", indices[0, 0, 0, :])
        # tf.print("values", values[0, 4, 10, :])
        # tf.print("indices", indices[0, 4, 10, :])

        # select the top n cascades
        # Shape: [None, n_params, 86, 60, n_sel]
        top_cascade_parameters = tf.gather(
            cascade_parameters, indices, axis=2, batch_dims=1
        )
        print(
            "top_cascade_parameters [None, n_params, 86, 60, n_sel]",
            top_cascade_parameters,
        )

        # Shape: [None, n_sel, 86, 60, n_params]
        top_cascade_parameters = tf.transpose(
            top_cascade_parameters, [0, 4, 2, 3, 1]
        )
        print(
            "top_cascade_parameters [None, n_sel, 86, 60, n_params]",
            top_cascade_parameters,
        )

        return top_cascade_parameters

    @tf.function
    def get_tensors_cascade(
        self,
        data_batch_dict,
        is_training,
        parameter_tensor_name="x_parameters",
    ):
        """Get tensors computed from input parameters and pulses.

        Similar method to `get_tensors` but for the embedded cascade model.
        Parameters must be a tensor of shape [-1, 86, 60, n_params] where
        each DOM may have contributions from different cascade hypotheses.
        Result tensors will not have time exclusions applied yet.

        Parameters
        ----------
        data_batch_dict : dict of tf.Tensor
            parameters : tf.Tensor
                A tensor which describes the input parameters of the source.
                This fully defines the source hypothesis. The tensor is of
                shape [-1, 86, 60, n_params] and the last dimension must match the
                order of the parameter names (self.parameter_names),
            pulses : tf.Tensor
                The input pulses (charge, time) of all events in a batch.
                Shape: [-1, 2]
            pulses_ids : tf.Tensor
                The pulse indices (batch_index, string, dom, pulse_number)
                of all pulses in the batch of events.
                Shape: [-1, 4]
        is_training : bool, optional
            Indicates whether currently in training or inference mode.
            Must be provided if batch normalisation is used.
            True: in training mode
            False: inference mode.
        parameter_tensor_name : str, optional
            The name of the parameter tensor to use. Default: 'x_parameters'

        Returns
        -------
        dict of tf.Tensor
            A dictionary of output tensors.
            Similar to the output of `get_tensors
        """
        self.assert_configured(True)

        print("Applying embedded Cascade Model...")
        tensor_dict = {}

        config = self.configuration.config["config"]["config_cascade"][
            "config"
        ]
        n_parameters_track = self.untracked_data["n_parameters_track"]
        n_snowstorm_parameters = self.untracked_data["n_snowstorm_parameters"]
        num_features = self.untracked_data["n_parameters_cascade"]
        num_cascades = self.configuration.config["config"]["num_cascades"]
        sphere_radius = self.configuration.config["config"]["sphere_radius"]

        # parameters: x, y, z, zenith, azimuth, energy, time
        # Shape: [None, n_sel, 86, 60, n_params]
        parameters = data_batch_dict[parameter_tensor_name]
        assert parameters.shape[1:] == (
            num_cascades,
            86,
            60,
            num_features,
        ), parameters.shape

        parameters_trafo_all = data_batch_dict["parameters_trafo_all"]
        snowstorm_trafo = tf.unstack(parameters_trafo_all, axis=-1)[
            n_parameters_track - n_snowstorm_parameters : n_parameters_track
        ]
        pulses = data_batch_dict["x_pulses"]
        pulses_ids = data_batch_dict["x_pulses_ids"][:, :3]

        tensors = self.data_trafo.data["tensors"]
        if (
            "x_time_exclusions" in data_batch_dict
            and data_batch_dict["x_time_exclusions"] is not None
        ):
            time_exclusions_exist = True
            x_time_exclusions = data_batch_dict["x_time_exclusions"]
            x_time_exclusions_ids = data_batch_dict["x_time_exclusions_ids"]
        else:
            time_exclusions_exist = False
        print("\t Applying time exclusions:", time_exclusions_exist)

        # Shape: [None, n_sel, 86, 60]
        parameter_list = tf.unstack(parameters, axis=-1)

        # make sure energy is >= 0
        parameter_list[5] = tf.clip_by_value(
            parameter_list[5], 0.0, float("inf")
        )

        # shift cascade vertex to shower maximum
        if config["shift_cascade_vertex"]:
            x, y, z, t = shift_to_maximum(
                x=parameter_list[0],
                y=parameter_list[1],
                z=parameter_list[2],
                zenith=parameter_list[3],
                azimuth=parameter_list[4],
                ref_energy=parameter_list[5],
                t=parameter_list[6],
                reverse=False,
            )
            parameter_list[0] = x
            parameter_list[1] = y
            parameter_list[2] = z
            parameter_list[6] = t

        # get parameters tensor dtype
        param_dtype_np = tensors[parameter_tensor_name].dtype_np

        # shape: [n_batch, 86, 60, 1]
        dom_charges_true = data_batch_dict["x_dom_charge"]

        pulse_times = pulses[:, 1]
        pulse_batch_id = pulses_ids[:, 0]

        # -----------------------------------
        # Calculate input values for DOMs
        # -----------------------------------
        # cascade_azimuth, cascade_zenith, cascade_energy
        # cascade_x, cascade_y, cascade_z
        # dx, dy, dz, distance
        # alpha (azimuthal angle to DOM)
        # beta (zenith angle to DOM)

        # calculate displacement vector
        # Shape: [n_batch, n_sel, 86, 60] = [86, 60] - [n_batch, n_sel, 86, 60]
        dx = detector.x_coords[..., 0] - parameter_list[0]
        dy = detector.x_coords[..., 1] - parameter_list[1]
        dz = detector.x_coords[..., 2] - parameter_list[2]
        # Shape: [n_batch, n_sel, 86, 60, 1]
        dx = tf.expand_dims(dx, axis=-1)
        dy = tf.expand_dims(dy, axis=-1)
        dz = tf.expand_dims(dz, axis=-1)

        # stabilize with 1e-1 (10cm) when distance approaches DOM radius
        # Shape: [n_batch, n_sel, 86, 60, 1]
        distance = tf.sqrt(dx**2 + dy**2 + dz**2) + 1e-1
        distance_xy = tf.sqrt(dx**2 + dy**2) + 1e-1

        # compute t_geometry: time for photon to travel to DOM
        # Shape: [n_batch, n_sel, 86, 60, 1]
        c_ice = 0.22103046286329384  # m/ns
        dt_geometry = distance / c_ice

        # calculate observation angle
        # Shape: [n_batch, n_sel, 86, 60, 1]
        dx_normed = dx / distance
        dy_normed = dy / distance
        dz_normed = dz / distance

        # angle in xy-plane (relevant for anisotropy)
        # Shape: [n_batch, n_sel, 86, 60, 1]
        dx_normed_xy = dx / distance_xy
        dy_normed_xy = dy / distance_xy

        # calculate direction vector of cascade
        # Shape: [n_batch, n_sel, 86, 60]
        cascade_zenith = parameter_list[3]
        cascade_azimuth = parameter_list[4]
        cascade_dir_x = -tf.sin(cascade_zenith) * tf.cos(cascade_azimuth)
        cascade_dir_y = -tf.sin(cascade_zenith) * tf.sin(cascade_azimuth)
        cascade_dir_z = -tf.cos(cascade_zenith)

        # calculate opening angle of displacement vector and cascade direction
        # Shape: [n_batch, n_sel, 86, 60, 1]
        opening_angle = angles.get_angle(
            tf.stack([cascade_dir_x, cascade_dir_y, cascade_dir_z], axis=-1),
            tf.concat([dx_normed, dy_normed, dz_normed], axis=-1),
        )
        opening_angle = tf.expand_dims(opening_angle, axis=-1)

        # transform input values to correct scale
        norm_const = self.data_trafo.data["norm_constant"]

        # transform distance
        distance_tr = distance / (sphere_radius + norm_const)

        # transform coordinates by approximate size of IceCube
        parameters_trafo = tf.unstack(parameters, axis=-1)
        parameters_trafo[0] /= 500.0 + norm_const
        parameters_trafo[1] /= 500.0 + norm_const
        parameters_trafo[2] /= 500.0 + norm_const

        # transform angle
        opening_angle_traf = opening_angle / (norm_const + np.pi)

        # transform energy roughly to centered around 0 assuming E in 1e2 - 1e6 GeV
        parameters_trafo[5] = (
            tf.math.log(parameters_trafo[5] + 1.0) - 4.0
        ) / 2.0

        # parameters: x, y, z, zenith, azimuth, energy, time
        parameters_trafo[5] /= np.pi + norm_const

        # Shape: [n_batch, n_sel, 86, 60, n_inputs]
        modified_parameters = tf.stack(
            parameters_trafo[:3]
            + [cascade_dir_x, cascade_dir_y, cascade_dir_z]
            + [parameters_trafo[5]]
            + snowstorm_trafo,
            axis=-1,
        )

        # Shape: [n_batch, n_sel, 86, 60, n_inputs_i]
        input_list = [
            modified_parameters,
            dx_normed,
            dy_normed,
            dz_normed,
            distance_tr,
        ]

        if config["add_anisotropy_angle"]:
            input_list.append(dx_normed_xy)
            input_list.append(dy_normed_xy)

        if config["add_opening_angle"]:
            input_list.append(opening_angle_traf)

        if (
            config["add_dom_angular_acceptance"]
            or config["scale_charge_by_angular_acceptance"]
            or config["scale_charge_by_relative_angular_acceptance"]
        ):
            if (
                config["use_constant_baseline_hole_ice"]
                or config["scale_charge_by_relative_angular_acceptance"]
            ):
                # Hole-ice Parameters
                # Shape: [n_batch, n_sel, 86, 60, 1]
                p0_base = (
                    tf.ones_like(dz_normed) * config["baseline_hole_ice_p0"]
                )
                p1_base = (
                    tf.ones_like(dz_normed) * config["baseline_hole_ice_p1"]
                )

                # input tenser: cos(eta), p0, p1 in last dimension
                # Shape: [n_batch, n_sel, 86, 60, 3]
                x_base = tf.concat([dz_normed, p0_base, p1_base], axis=-1)

                # Shape: [n_batch, n_sel, 86, 60, 1]
                angular_acceptance_base = dom_acceptance.get_acceptance(
                    x=x_base,
                    dtype=self.float_precision,
                )[..., tf.newaxis]

            if config["use_constant_baseline_hole_ice"]:
                angular_acceptance = angular_acceptance_base
            else:
                p0 = parameter_list[
                    self.get_index("HoleIceForward_Unified_p0")
                ][..., tf.newaxis]
                p1 = parameter_list[
                    self.get_index("HoleIceForward_Unified_p1")
                ][..., tf.newaxis]

                # input tenser: cos(eta), p0, p1 in last dimension
                # Shape: [n_batch, n_sel, 86, 60, 3]
                x = tf.concat([dz_normed, p0, p1], axis=-1)

                # Shape: [n_batch, n_sel, 86, 60, 1]
                angular_acceptance = dom_acceptance.get_acceptance(
                    x=x,
                    dtype=self.float_precision,
                )[..., tf.newaxis]

            if config["scale_charge_by_relative_angular_acceptance"]:
                # stabilize with 1e-3 when acceptance approaches zero
                relative_angular_acceptance = (angular_acceptance + 1e-3) / (
                    angular_acceptance_base + 1e-3
                )

            if config["add_dom_angular_acceptance"]:
                input_list.append(angular_acceptance)

        if config["add_dom_coordinates"]:

            # transform coordinates to correct scale with mean 0 std dev 1
            # shape: [1, 1, 86, 60, 3]
            dom_coords = np.reshape(
                detector.x_coords.astype(param_dtype_np),
                [1, 1, 86, 60, 3],
            )
            # scale of coordinates is ~-500m to ~500m with std dev of ~ 284m
            dom_coords /= 284.0

            # extend to correct batch shape:
            # Shape: [n_batch, n_sel, 86, 60, 3]
            #    = [n_batch, n_sel, 86, 60, 1] * [1, 1, 86, 60, 3]
            dom_coords = tf.ones_like(dx_normed) * dom_coords
            input_list.append(dom_coords)

        if config["num_local_vars"] > 0:

            # extend to correct shape:
            # shape: [n_batch, n_sel, 86, 60, 1]
            local_vars = (
                tf.ones_like(dx_normed) * self._untracked_data["local_vars"]
            )
            input_list.append(local_vars)

        # shape: [n_batch, n_sel, 86, 60, n_inputs]
        x_doms_input = tf.concat(input_list, axis=-1)

        # shape: [n_batch*, 86, 60, n_inputs]
        x_doms_input = tf.reshape(
            x_doms_input, [-1, 86, 60, x_doms_input.shape[-1]]
        )
        print("\t x_doms_input", x_doms_input)

        # -------------------------------------------
        # convolutional hex3d layers over X_IC86 data
        # -------------------------------------------
        conv_hex3d_layers = self._untracked_data["conv_hex3d_layer_cascade"](
            x_doms_input,
            is_training=is_training,
            keep_prob=config["keep_prob"],
        )
        decoder = self._untracked_data["decoder_cascade"]
        decoder_charge = self._untracked_data["decoder_cascade_charge"]
        n_components = decoder.n_components_total
        n_latent_time = decoder.n_parameters
        n_latent_charge = decoder_charge.n_parameters
        n_latent = n_latent_time + n_latent_charge

        # -------------------------------------------
        # Get times at which to evaluate DOM PDF
        # -------------------------------------------

        # special handling for placements of mixture model compeonents:
        # these will be placed relative to t_geometry,
        # which is the earliest possible time for a DOM to be hit from photons
        # originating from the cherenkov point. Once waveforms are defined
        # relative to t_geometry, their overall features are similar to each
        # other. In particular, there is a peak around 8000ns arising form
        # after pulses and a smaller one at slightly negative times from
        # pre-pulses. We will place the components around these features.
        if "mixture_component_t_seeds" in config:
            t_seed = config["mixture_component_t_seeds"]
            if len(t_seed) != n_components:
                raise ValueError(
                    f"Number of t_seeds {t_seed} does not match number of "
                    f"components {n_components}"
                )
        else:
            t_seed = np.zeros(n_components)
        t_seed = np.reshape(t_seed, [1, 1, 1, 1, n_components])

        # per DOM offset to shift to t_geometry
        # Note that the pulse times are already offset by the cascade vertex
        # time. So we now only need to add dt_geometry.
        # shape: [n_batch, n_sel, 86, 60, n_components] =
        #       [n_batch, n_sel, 86, 60, 1] + [1, 1, 1, 1, n_components]
        #       + [n_batch, n_sel, 86, 60, 1]
        tensor_dict["time_offsets_per_dom"] = (
            dt_geometry + t_seed + parameters[..., 6][..., tf.newaxis]
        )

        # shape: [n_batch, 86, 60, n_sel, n_components]
        tensor_dict["time_offsets_per_dom"] = tf.transpose(
            tensor_dict["time_offsets_per_dom"], [0, 2, 3, 1, 4]
        )

        # shape: [n_events]
        tensor_dict["time_offsets"] = tf.zeros_like(pulse_batch_id)

        # offset PDF evaluation times with per DOM offset
        # shape: [n_pulses, n_sel, n_components] =
        #       [n_pulses, 1, 1] - [n_pulses, n_sel, n_components]
        t_pdf_at_dom = pulse_times[:, tf.newaxis, tf.newaxis] - tf.gather_nd(
            tensor_dict["time_offsets_per_dom"], pulses_ids
        )
        t_pdf_at_dom = tf.ensure_shape(
            t_pdf_at_dom,
            [None, num_cascades, n_components],
        )

        if time_exclusions_exist:
            # offset time exclusions

            # shape: [n_events, n_sel, n_components]
            tw_cascade_t = tf.gather_nd(
                tensor_dict["time_offsets_per_dom"], x_time_exclusions_ids
            )

            # shape: [n_events, 2, n_sel, n_components]
            #      = [n_events, 2, 1, 1] - [n_events, 1, b_sel, n_components]
            t_exclusions = x_time_exclusions[
                ..., tf.newaxis, tf.newaxis
            ] - tf.expand_dims(tw_cascade_t, axis=1)
            t_exclusions = tf.ensure_shape(
                t_exclusions, [None, 2, num_cascades, n_components]
            )

        # scale time range down to avoid big numbers:
        t_scale = 1.0 / self.time_unit_in_ns  # [1./ns]
        t_pdf_at_dom *= t_scale
        dt_geometry *= t_scale
        if time_exclusions_exist:
            t_exclusions *= t_scale

        # -------------------------------------------
        # Gather latent vars of mixture model
        # -------------------------------------------

        # check if we have the right amount of filters in the latent dimension
        if n_latent != config["num_filters_list"][-1]:
            raise ValueError(
                "{!r} != {!r}".format(n_latent, config["num_filters_list"][-1])
            )
        if n_components < 1:
            raise ValueError("{!r} !> 0".format(n_components))

        # print out information about the mixture model components
        print("\t Charge PDF mixture model components:")
        decoder_config = decoder_charge.configuration.config["config"]
        if "decoder_names" in decoder_charge._untracked_data:
            counter = 0
            for name in decoder_charge._untracked_data["decoder_names"]:
                base_name, num, weight = decoder_config["decoder_mapping"][
                    name
                ]
                print(f"\t\t {base_name}: {num} [w={weight}]")
        else:
            print(f"\t\t Decoder name: {decoder_charge.name}")

        print("\t Time PDF mixture model components:")
        decoder_config = decoder.configuration.config["config"]
        counter = 0
        for name in decoder._untracked_data["decoder_names"]:
            base_name, num, weight = decoder_config["decoder_mapping"][name]
            if "mixture_component_t_seeds" in config:
                t_seed_i = config["mixture_component_t_seeds"][
                    counter : counter + num
                ]
            else:
                t_seed_i = np.zeros(num)
            counter += num
            print(f"\t\t {base_name}: {num} [w={weight}, t_seeds={t_seed_i}]")

        # shape: [n_batch*, 86, 60, n_latent_charge + n_latent_time]
        latent_vars = conv_hex3d_layers[-1]

        # shape: [n_batch, n_sel, 86, 60, n_latent_time]
        latent_vars = tf.reshape(
            latent_vars, [-1, num_cascades, 86, 60, n_latent]
        )

        # shape: [n_batch, 86, 60, n_sel, n_latent_time]
        latent_vars = tf.transpose(latent_vars, [0, 2, 3, 1, 4])

        # shape: [n_batch, 86, 60, n_sel, n_latent_time]
        latent_vars_time = latent_vars[..., n_latent_charge:]

        # shape: [n_batch, 86, 60, n_sel, n_latent_charge]
        latent_vars_charge = latent_vars[..., :n_latent_charge]

        tensor_dict["latent_vars"] = latent_vars
        tensor_dict["latent_vars_time"] = latent_vars_time

        # -------------------------
        # Calculate Time Exclusions
        # -------------------------
        if time_exclusions_exist:

            # get latent vars for each time window
            # Shape: [n_tw, n_sel, n_latent_time]
            tw_latent_vars = tf.gather_nd(
                latent_vars_time, x_time_exclusions_ids
            )

            # ensure shape
            tw_latent_vars = tf.ensure_shape(
                tw_latent_vars, [None, num_cascades, n_latent_time]
            )

            # x: [n_tw, n_sel, n_components]
            # latent_vars: [n_tw, n_sel, n_latent_time]
            #   --> [n_tw, n_sel]
            tw_cdf_start = decoder.cdf(
                x=t_exclusions[:, 0],
                latent_vars=tw_latent_vars,
                reduce_components=True,
            )
            tw_cdf_stop = decoder.cdf(
                x=t_exclusions[:, 1],
                latent_vars=tw_latent_vars,
                reduce_components=True,
            )

            # shape: [n_tw, n_sel]
            tw_cdf_exclusion = tf_helpers.safe_cdf_clip(
                tw_cdf_stop - tw_cdf_start
            )
            tw_cdf_exclusion = tf.cast(tw_cdf_exclusion, self.float_precision)

            # accumulate time window exclusions for each DOM and MM component
            # shape: [None, 86, 60, n_sel]
            dom_cdf_exclusion = tf.zeros_like(latent_vars[..., 0])

            dom_cdf_exclusion = tf.tensor_scatter_nd_add(
                dom_cdf_exclusion,
                indices=x_time_exclusions_ids,
                updates=tw_cdf_exclusion,
            )
            dom_cdf_exclusion = tf_helpers.safe_cdf_clip(dom_cdf_exclusion)

            tensor_dict["tw_cdf_exclusion"] = tw_cdf_exclusion
            tensor_dict["dom_cdf_exclusion"] = dom_cdf_exclusion

        # -------------------------------------------
        # Get expected charge at DOM
        # -------------------------------------------

        # get charge scale to apply
        charge_scale = 1.0

        # scale charges by cascade energy
        if config["scale_charge"]:
            # Shape:[n_batch, n_sel, 86, 60]
            e_scale = parameter_list[5] / 1000.0

            # Shape: [n_batch, 86, 60, n_sel]
            charge_scale *= tf.transpose(e_scale, [0, 2, 3, 1])

        # scale charges by relative DOM efficiency
        if config["scale_charge_by_relative_dom_efficiency"]:
            charge_scale *= tf.expand_dims(
                detector.rel_dom_eff.astype(param_dtype_np), axis=-1
            )

        # scale charges by global DOM efficiency
        if config["scale_charge_by_global_dom_efficiency"]:
            charge_scale *= tf.expand_dims(
                parameter_list[self.get_index("DOMEfficiency")], axis=-1
            )

        if config["scale_charge_by_angular_acceptance"]:
            # do not let charge scaling go down to zero
            # Even if cascade is coming from directly above, the photons
            # will scatter and arrive from vaying angles.
            tr_angular_acceptance = tf.transpose(
                tf.squeeze(angular_acceptance, axis=4),
                [0, 2, 3, 1],
            )
            charge_scale *= (
                tf.clip_by_value(tr_angular_acceptance, 0, float("inf")) + 1e-2
            )

        if config["scale_charge_by_relative_angular_acceptance"]:
            tr_relative_angular_acceptance = tf.transpose(
                tf.squeeze(relative_angular_acceptance, axis=4), [0, 2, 3, 1]
            )
            charge_scale *= tf.clip_by_value(
                tr_relative_angular_acceptance,
                1e-2,
                100,
            )

        # check that things are spelled correctly
        for param_name in config["charge_scale_tensors"]:
            if param_name not in decoder_charge.loc_parameters:
                raise ValueError(
                    f"Charge scale tensor {param_name} not found in "
                    f"decoder loc_parameters: {decoder_charge.loc_parameters}!"
                )

        # transform charge scale tensors
        latent_vars_charge_scaled = []
        for idx, param_name in enumerate(decoder_charge.parameter_names):

            # Shape: [n_batch, 86, 60, n_sel]
            charge_tensor = latent_vars_charge[..., idx]

            if param_name in decoder_charge.loc_parameters:

                # make sure no value range mapping is not defined for this tensor
                if param_name in decoder_charge.value_range_mapping:
                    raise ValueError(
                        f"Value range mapping for charge scale tensor "
                        f"{param_name} at index {idx} not allowed since "
                        f"manual mapping in model is done!"
                    )
                else:
                    print(
                        f"\t Applying charge scale to {param_name} at index {idx}"
                    )

                # clip value range for more stability during training
                # apply exponential which also forces positive values
                charge_tensor = tf.exp(
                    tf.clip_by_value(charge_tensor, -20.0, 15)
                )

                if param_name in config["charge_scale_tensors"]:
                    charge_tensor *= charge_scale

                # apply time window exclusions if needed
                if time_exclusions_exist:
                    charge_tensor = charge_tensor * (
                        1.0 - dom_cdf_exclusion + self.epsilon
                    )

                # add small constant to make sure dom charges are > 0:
                charge_tensor += self.epsilon

            latent_vars_charge_scaled.append(charge_tensor)

        # Shape: [n_batch, 86, 60, n_sel, n_latent_charge]
        latent_vars_charge = tf.stack(latent_vars_charge_scaled, axis=-1)
        tensor_dict["latent_vars_charge"] = latent_vars_charge

        # shape: [n_batch, 86, 60, n_sel, n_components]
        dom_charges_component = decoder_charge.expectation(
            latent_vars=latent_vars_charge,
            reduce_components=False,
        )
        if dom_charges_component.shape[1:] == [86, 60, num_cascades]:
            dom_charges_component = dom_charges_component[..., tf.newaxis]
        tensor_dict["dom_charges_component"] = dom_charges_component

        # shape: [n_batch, 86, 60, n_sel]
        tensor_dict["dom_charges"] = decoder_charge.expectation(
            latent_vars=latent_vars_charge,
            reduce_components=True,
        )

        # shape: [n_batch, 86, 60, n_sel, n_components]
        variance_component = decoder_charge.variance(
            latent_vars=latent_vars_charge, reduce_components=False
        )
        if variance_component.shape[1:] == [86, 60, num_cascades]:
            variance_component = variance_component[..., tf.newaxis]
        tensor_dict["dom_charges_variance_component"] = variance_component

        # shape: [n_batch, 86, 60, n_sel]
        tensor_dict["dom_charges_variance"] = decoder_charge.variance(
            latent_vars=latent_vars_charge, reduce_components=True
        )

        # -------------------------
        # Compute charge PDF values
        # -------------------------
        # shape: [n_batch, 86, 60, n_sel]
        dom_charges_pdf = decoder_charge.pdf(
            x=dom_charges_true,
            axis=3,
            latent_vars=latent_vars_charge,
            reduce_components=True,
        )
        # cast back to specified float precision
        dom_charges_pdf = tf.cast(dom_charges_pdf, self.float_precision)
        tensor_dict["dom_charges_pdf"] = dom_charges_pdf

        # --------------------------
        # Calculate Pulse PDF Values
        # --------------------------

        # get latent vars for each pulse
        # Shape: [n_pulses, n_sel, n_latent_time]
        pulse_latent_vars = tf.gather_nd(latent_vars_time, pulses_ids)

        # ensure shapes
        pulse_latent_vars = tf.ensure_shape(
            pulse_latent_vars, [None, num_cascades, n_latent_time]
        )

        # ------------------------
        # Apply Time Mixture Model
        # ------------------------

        # t_pdf_at_dom: [n_events, n_sel, n_components]
        # Shape: [n_pulses, n_sel]
        pulse_pdf_values = decoder.pdf(
            x=t_pdf_at_dom,
            latent_vars=pulse_latent_vars,
            reduce_components=True,
        )
        pulse_cdf_values = decoder.cdf(
            x=t_pdf_at_dom,
            latent_vars=pulse_latent_vars,
            reduce_components=True,
        )

        # cast back to specified float precision
        pulse_pdf_values = tf.cast(pulse_pdf_values, self.float_precision)
        pulse_cdf_values = tf.cast(pulse_cdf_values, self.float_precision)

        tensor_dict["pulse_pdf"] = pulse_pdf_values
        tensor_dict["pulse_cdf"] = pulse_cdf_values
        # ---------------------

        return tensor_dict

    @tf.function
    def get_tensors(
        self,
        data_batch_dict,
        is_training,
        parameter_tensor_name="x_parameters",
    ):
        """Get tensors computed from input parameters and pulses.

        Parameters are the hypothesis tensor of the source with
        shape [-1, n_params]. The get_tensors method must compute all tensors
        that are to be used in later steps. It returns these as a dictionary
        of output tensors.

        Parameters
        ----------
        data_batch_dict : dict of tf.Tensor
            parameters : tf.Tensor
                A tensor which describes the input parameters of the source.
                This fully defines the source hypothesis. The tensor is of
                shape [-1, n_params] and the last dimension must match the
                order of the parameter names (self.parameter_names),
            pulses : tf.Tensor
                The input pulses (charge, time) of all events in a batch.
                Shape: [-1, 2]
            pulses_ids : tf.Tensor
                The pulse indices (batch_index, string, dom, pulse_number)
                of all pulses in the batch of events.
                Shape: [-1, 4]
        is_training : bool, optional
            Indicates whether currently in training or inference mode.
            Must be provided if batch normalisation is used.
            True: in training mode
            False: inference mode.
        parameter_tensor_name : str, optional
            The name of the parameter tensor to use. Default: 'x_parameters'

        Raises
        ------
        ValueError
            Description

        Returns
        -------
        dict of tf.Tensor
            A dictionary of output tensors.
            This  dictionary must at least contain:

                'dom_charges':
                    The predicted charge at each DOM
                    Shape: [n_events, 86, 60, 1]
                'pulse_pdf':
                    The likelihood evaluated for each pulse
                    Shape: [n_pulses]
                'time_offsets':
                    The global time offsets for each event.
                    Shape: [n_events]

            Other relevant optional tensors are:
                'latent_vars_time':
                    Shape: [n_events, 86, 60, n_latent]
                'latent_vars_charge':
                    Shape: [n_events, 86, 60, n_charge]
                'time_offsets_per_dom':
                    The time offsets per DOM (includes global offset).
                    Shape: [n_events, 86, 60, n_components]
                'dom_cdf_exclusion':
                    Shape: [n_events, 86, 60]
                'pulse_cdf':
                    Shape: [n_pulses]
        """
        self.assert_configured(True)

        print("Applying EnteringSphereInfTrackMultiCascade Model...")
        tensor_dict = {}

        config = self.configuration.config["config"]
        parameters_all = data_batch_dict[parameter_tensor_name]
        parameters_all = self.add_parameter_indexing(parameters_all)
        parameters = parameters_all[
            :, : self.untracked_data["n_parameters_track"]
        ]

        # ensure energy is greater or equal zero
        parameters = tf.unstack(parameters, axis=-1)
        parameters[3] = tf.clip_by_value(parameters[3], 0.0, float("inf"))
        parameters = tf.stack(parameters, axis=-1)

        pulses = data_batch_dict["x_pulses"]
        pulses_ids = data_batch_dict["x_pulses_ids"][:, :3]

        tensors = self.data_trafo.data["tensors"]
        if (
            "x_time_exclusions" in data_batch_dict
            and data_batch_dict["x_time_exclusions"] is not None
        ):
            time_exclusions_exist = True
            x_time_exclusions = data_batch_dict["x_time_exclusions"]
            x_time_exclusions_ids = data_batch_dict["x_time_exclusions_ids"]
        else:
            time_exclusions_exist = False
        print("\t Applying time exclusions:", time_exclusions_exist)

        # parameters: e_zenith, e_azimuth, time, energy, zenith, azimuth
        num_features = parameters.get_shape().as_list()[-1]
        params_reshaped = tf.reshape(parameters, [-1, 1, 1, 1, num_features])
        parameter_list = tf.unstack(params_reshaped, axis=-1)

        # get parameters tensor dtype
        param_dtype_np = tensors[parameter_tensor_name].dtype_np

        # shape: [n_batch, 86, 60, 1]
        dom_charges_true = data_batch_dict["x_dom_charge"]

        # DOM coordinates
        # Shape: [1, 86, 60, 1, 3]
        dom_coords = np.reshape(detector.x_coords, [1, 86, 60, 1, 3])

        pulse_times = pulses[:, 1]
        pulse_batch_id = pulses_ids[:, 0]

        # get transformed (unshifted) parameters
        parameters_trafo_all = self.data_trafo.transform(
            parameters_all, tensor_name=parameter_tensor_name
        )
        parameters_trafo = parameters_trafo_all[
            :, : self.untracked_data["n_parameters_track"]
        ]

        # -----------------------------------
        # Calculate input values for DOMs
        # -----------------------------------
        # Globals:
        #   - entry_energy
        #   - e_dir_x, e_dir_y, e_dir_z [normalized]
        #   - dir_x, dir_y, dir_z [normalized]
        #
        # Locals:
        #   - dx, dy, dz, dist, length_along_track [cherenkov]
        #   - opening angle cherenkov cone and PMT direction
        #   - opening angle track direction and displacement vector
        #   - x, y, y [closest approach point to DOM]
        #   - closest approach distance
        #   - dx, dy, dz [closest approach point to DOM]
        #
        # Note: we will try to keep the number of input features
        # to a minimum for now. One could test if adding more
        # (redundant) features could help the model to learn

        # extract parameters
        # shape: [n_batch, 1, 1, 1]
        e_zenith = parameter_list[0]
        e_azimuth = parameter_list[1]
        e_energy = tf.clip_by_value(parameter_list[3], 0.0, float("inf"))
        zenith = parameter_list[4]
        azimuth = parameter_list[5]

        # shape: [n_batch]
        e_time = parameters[:, 2]

        # calculate normalized vector to entry point
        # shape: [n_batch, 1, 1, 1]
        e_dir_x = tf.sin(e_zenith) * tf.cos(e_azimuth)
        e_dir_y = tf.sin(e_zenith) * tf.sin(e_azimuth)
        e_dir_z = tf.cos(e_zenith)

        # calculate entry position on a sphere of the given radius
        # shape: [n_batch, 1, 1, 1]
        e_pos_x = e_dir_x * config["sphere_radius"]
        e_pos_y = e_dir_y * config["sphere_radius"]
        e_pos_z = e_dir_z * config["sphere_radius"]

        # calculate direction vector of track
        # shape: [n_batch, 1, 1, 1]
        dir_x = -tf.sin(zenith) * tf.cos(azimuth)
        dir_y = -tf.sin(zenith) * tf.sin(azimuth)
        dir_z = -tf.cos(zenith)

        # vector of entry point to DOM
        # Shape: [n_batch, 86, 60, 1] = [1, 86, 60, 1] - [n_batch, 1, 1, 1]
        h_x = dom_coords[..., 0] - e_pos_x
        h_y = dom_coords[..., 1] - e_pos_y
        h_z = dom_coords[..., 2] - e_pos_z

        # distance along track between entry point and
        # closest approach point of infinite track
        # Shape: [n_batch, 86, 60, 1]
        length_closest_approach = dir_x * h_x + dir_y * h_y + dir_z * h_z

        # calculate closest approach points of track to each DOM
        # Shape: [n_batch, 86, 60, 1]
        closest_x = e_pos_x + length_closest_approach * dir_x
        closest_y = e_pos_y + length_closest_approach * dir_y
        closest_z = e_pos_z + length_closest_approach * dir_z

        # calculate displacement vectors to closest approach
        # Shape: [n_batch, 86, 60, 1] = [1, 86, 60, 1] - [n_batch, 1, 1, 1]
        dx_inf = dom_coords[..., 0] - closest_x
        dy_inf = dom_coords[..., 1] - closest_y
        dz_inf = dom_coords[..., 2] - closest_z

        # distance from DOM to closest approach point
        # shape: [n_batch, 86, 60, 1]
        distance_closest = tf.sqrt(dx_inf**2 + dy_inf**2 + dz_inf**2) + 1e-1

        # normalize displacement vectors
        # shape: [n_batch, 86, 60, 1]
        dx_inf_normed = dx_inf / distance_closest
        dy_inf_normed = dy_inf / distance_closest
        dz_inf_normed = dz_inf / distance_closest

        # shift to cherenkov emission point
        # calculate distance on track of cherenkov position
        # shape: [n_batch, 86, 60, 1]
        cherenkov_angle = np.arccos(1.0 / 1.3195)
        length_cherenkov_pos = (
            length_closest_approach
            - distance_closest / np.tan(cherenkov_angle)
        )

        # calculate cheerenkov emission point
        # shape: [n_batch, 86, 60, 1]
        cherenkov_x = e_pos_x + length_cherenkov_pos * dir_x
        cherenkov_y = e_pos_y + length_cherenkov_pos * dir_y
        cherenkov_z = e_pos_z + length_cherenkov_pos * dir_z

        # calculate displacement vectors to cherenkov emission
        # Shape: [n_batch, 86, 60, 1] = [1, 86, 60, 1] - [n_batch, 1, 1, 1]
        dx_cherenkov = dom_coords[..., 0] - cherenkov_x
        dy_cherenkov = dom_coords[..., 1] - cherenkov_y
        dz_cherenkov = dom_coords[..., 2] - cherenkov_z

        # distance from DOM to cherenkov emission point
        # shape: [n_batch, 86, 60, 1]
        distance_cherenkov = (
            tf.sqrt(dx_cherenkov**2 + dy_cherenkov**2 + dz_cherenkov**2) + 1e-1
        )
        distance_cherenkov_xy = (
            tf.sqrt(dx_cherenkov**2 + dy_cherenkov**2) + 1e-1
        )

        # calculate observation angle
        # Shape: [n_batch, 86, 60, 1]
        dx_cherenkov_normed = dx_cherenkov / distance_cherenkov
        dy_cherenkov_normed = dy_cherenkov / distance_cherenkov
        dz_cherenkov_normed = dz_cherenkov / distance_cherenkov

        # angle in xy-plane (relevant for anisotropy)
        # shape: [n_batch, 86, 60, 1]
        dx_cherenkov_normed_xy = dx_cherenkov / distance_cherenkov_xy
        dy_cherenkov_normed_xy = dy_cherenkov / distance_cherenkov_xy

        # calculate opening angle of cherenkov light and PMT direction
        # shape: [n_batch, 86, 60, 1]
        opening_angle = angles.get_angle(
            tf.cast(tf.stack([0.0, 0.0, 1.0], axis=-1), self.float_precision),
            tf.concat(
                [
                    dx_cherenkov_normed,
                    dy_cherenkov_normed,
                    dz_cherenkov_normed,
                ],
                axis=-1,
            ),
        )[..., tf.newaxis]

        # calculate opening angle of track direction and displacement vector
        # from the closest approach point to the DOM
        opening_angle_closest = angles.get_angle(
            tf.concat([dir_x, dir_y, dir_z], axis=-1),
            tf.concat([dx_inf, dy_inf, dz_inf], axis=-1),
        )[..., tf.newaxis]

        # compute t_geometry: time for photon to travel to DOM
        # Shape: [n_batch, 86, 60, 1]
        c_ice = 0.22103046286329384  # m/ns
        c = 0.299792458  # m/ns
        dt_geometry = distance_cherenkov / c_ice + length_cherenkov_pos / c

        # transform dx, dy, dz, distance, zenith, azimuth to correct scale
        norm_const = self.data_trafo.data["norm_constant"]

        # transform distances and lengths in detector
        distance_closest_tr = distance_closest / (
            config["sphere_radius"] + norm_const
        )
        distance_cherenkov_tr = distance_cherenkov / (
            config["sphere_radius"] + norm_const
        )
        length_cherenkov_pos_tr = length_cherenkov_pos / (
            config["sphere_radius"] + norm_const
        )

        # transform coordinates by approximate size of IceCube
        closest_x_tr = closest_x / (500.0 + norm_const)
        closest_y_tr = closest_y / (500.0 + norm_const)
        closest_z_tr = closest_z / (500.0 + norm_const)

        # transform angle
        opening_angle_traf = opening_angle / (norm_const + np.pi)
        opening_angle_closest_traf = opening_angle_closest / (
            norm_const + np.pi
        )

        # ----------------------
        # Collect input features
        # ----------------------
        # Shape: [n_batch, 1, 1, 1]
        x_parameters_expanded = tf.unstack(
            tf.reshape(parameters_trafo, [-1, 1, 1, 1, num_features]), axis=-1
        )

        # parameters: e_zenith, e_azimuth, time, energy, zenith, azimuth [+ snowstorm]
        # Shape: [n_batch, 1, 1, num_inputs]
        modified_parameters = tf.concat(
            [e_dir_x, e_dir_y, e_dir_z]
            + [x_parameters_expanded[3]]
            + [dir_x, dir_y, dir_z]
            + x_parameters_expanded[6:],
            axis=-1,
        )

        # put everything together
        # Shape: [n_batch, 86, 60, num_inputs]
        params_expanded = tf.tile(modified_parameters, [1, 86, 60, 1])

        # Now add local features
        #   - dx, dy, dz, dist, length_along_track [cherenkov]
        #   - opening angle cherenkov cone and PMT direction
        input_list = [
            params_expanded,
            dx_cherenkov_normed,
            dy_cherenkov_normed,
            dz_cherenkov_normed,
            distance_cherenkov_tr,
            length_cherenkov_pos_tr,
            opening_angle_traf,
            opening_angle_closest_traf,
            closest_x_tr,
            closest_y_tr,
            closest_z_tr,
            distance_closest_tr,
            dx_inf_normed,
            dy_inf_normed,
            dz_inf_normed,
        ]

        if config["add_anisotropy_angle"]:
            input_list.append(dx_cherenkov_normed_xy)
            input_list.append(dy_cherenkov_normed_xy)

        if (
            config["add_dom_angular_acceptance"]
            or config["scale_charge_by_angular_acceptance"]
            or config["scale_charge_by_relative_angular_acceptance"]
        ):
            if (
                config["use_constant_baseline_hole_ice"]
                or config["scale_charge_by_relative_angular_acceptance"]
            ):
                # Hole-ice Parameters
                # shape: [n_batch, 1, 1, 1]
                p0_base = (
                    tf.ones_like(dz_cherenkov_normed)
                    * config["baseline_hole_ice_p0"]
                )
                p1_base = (
                    tf.ones_like(dz_cherenkov_normed)
                    * config["baseline_hole_ice_p1"]
                )

                # input tenser: cos(eta), p0, p1 in last dimension
                # Shape: [n_batch, 86, 60, 3]
                x_base = tf.concat(
                    [dz_cherenkov_normed, p0_base, p1_base], axis=-1
                )

                # Shape: [n_batch, 86, 60, 1]
                angular_acceptance_base = dom_acceptance.get_acceptance(
                    x=x_base,
                    dtype=self.float_precision,
                )[..., tf.newaxis]

            if config["use_constant_baseline_hole_ice"]:
                angular_acceptance = angular_acceptance_base
            else:
                p0 = tf.tile(
                    parameter_list[
                        self.get_index("HoleIceForward_Unified_p0")
                    ],
                    [1, 86, 60, 1],
                )
                p1 = tf.tile(
                    parameter_list[
                        self.get_index("HoleIceForward_Unified_p1")
                    ],
                    [1, 86, 60, 1],
                )

                # input tenser: cos(eta), p0, p1 in last dimension
                # Shape: [n_batch, 86, 60, 3]
                x = tf.concat([dz_cherenkov_normed, p0, p1], axis=-1)

                # Shape: [n_batch, 86, 60, 1]
                angular_acceptance = dom_acceptance.get_acceptance(
                    x=x,
                    dtype=self.float_precision,
                )[..., tf.newaxis]

            if config["scale_charge_by_relative_angular_acceptance"]:
                # stabilize with 1e-3 when acceptance approaches zero
                relative_angular_acceptance = (angular_acceptance + 1e-3) / (
                    angular_acceptance_base + 1e-3
                )

            if config["add_dom_angular_acceptance"]:
                input_list.append(angular_acceptance)

        if config["add_dom_coordinates"]:

            # transform coordinates to correct scale with mean 0 std dev 1
            dom_coords = np.expand_dims(
                detector.x_coords.astype(param_dtype_np), axis=0
            )
            # scale of coordinates is ~-500m to ~500m with std dev of ~ 284m
            dom_coords /= 284.0

            # extend to correct batch shape:
            dom_coords = tf.ones_like(dz_cherenkov_normed) * dom_coords

            print("\t dom_coords", dom_coords)
            input_list.append(dom_coords)

        if config["num_local_vars"] > 0:

            # extend to correct shape:
            local_vars = (
                tf.ones_like(dz_cherenkov_normed)
                * self._untracked_data["local_vars"]
            )
            print("\t local_vars", local_vars)

            input_list.append(local_vars)

        x_doms_input = tf.concat(input_list, axis=-1)
        print("\t x_doms_input", x_doms_input)

        # -------------------------------------------
        # convolutional hex3d layers over X_IC86 data
        # -------------------------------------------
        conv_hex3d_layers = self._untracked_data["conv_hex3d_layer"](
            x_doms_input,
            is_training=is_training,
            keep_prob=config["keep_prob"],
        )
        n_components = self.decoder.n_components_total
        n_latent_time = self.decoder.n_parameters
        n_latent_charge = self.decoder_charge.n_parameters
        n_latent = n_latent_time + n_latent_charge

        # -------------------------------------------
        # Get times at which to evaluate DOM PDF
        # -------------------------------------------

        # special handling for placements of mixture model compeonents:
        # these will be placed relative to t_geometry,
        # which is the earliest possible time for a DOM to be hit from photons
        # originating from the cherenkov point. Once waveforms are defined
        # relative to t_geometry, their overall features are similar to each
        # other. In particular, there is a peak around 8000ns arising form
        # after pulses and a smaller one at slightly negative times from
        # pre-pulses. We will place the components around these features.
        if "mixture_component_t_seeds" in config:
            t_seed = config["mixture_component_t_seeds"]
            if len(t_seed) != n_components:
                raise ValueError(
                    f"Number of t_seeds {t_seed} does not match number of "
                    f"components {n_components}"
                )
        else:
            t_seed = np.zeros(n_components)
        t_seed = np.reshape(t_seed, [1, 1, 1, n_components])

        # per DOM offset to shift to t_geometry
        # Note that the pulse times are already offset by the vertex
        # time. So we now only need to add dt_geometry.
        # shape: [n_batch, 86, 60, n_components] =
        #       [n_batch, 86, 60, 1] + [1, 1, 1, n_components]
        tensor_dict["time_offsets_per_dom"] = dt_geometry + t_seed

        # shape: [n_events]
        tensor_dict["time_offsets"] = e_time

        # offset PDF evaluation times with vertex time
        # shape: [n_pulses]
        t_pdf = pulse_times - tf.gather(e_time, indices=pulse_batch_id)

        # offset PDF evaluation times additionally with per DOM offset
        # shape: [n_pulses, n_components] =
        #       [n_pulses, 1] - [n_pulses, n_components]
        t_pdf_at_dom = t_pdf[:, tf.newaxis] - tf.gather_nd(
            tensor_dict["time_offsets_per_dom"], pulses_ids
        )
        t_pdf_at_dom = tf.ensure_shape(t_pdf_at_dom, [None, n_components])

        if time_exclusions_exist:
            # offset time exclusions

            # shape: [n_events]
            tw_event_t = tf.gather(e_time, indices=x_time_exclusions_ids[:, 0])

            # shape: [n_events, n_components]
            tw_event_t = tw_event_t[:, tf.newaxis] + tf.gather_nd(
                tensor_dict["time_offsets_per_dom"], x_time_exclusions_ids
            )

            # shape: [n_events, 2, n_components]
            #      = [n_events, 2, 1] - [n_events, 1, n_components]
            t_exclusions = x_time_exclusions[..., tf.newaxis] - tf.expand_dims(
                tw_event_t, axis=1
            )
            t_exclusions = tf.ensure_shape(
                t_exclusions, [None, 2, n_components]
            )

        # new shape: [n_pulses]
        t_pdf = tf.ensure_shape(t_pdf, [None])
        t_pdf_at_dom = tf.ensure_shape(t_pdf_at_dom, [None, n_components])

        # scale time range down to avoid big numbers:
        t_scale = 1.0 / self.time_unit_in_ns  # [1./ns]
        t_pdf *= t_scale
        t_pdf_at_dom *= t_scale
        dt_geometry *= t_scale
        if time_exclusions_exist:
            t_exclusions *= t_scale

        # -------------------------------------------
        # Gather latent vars of mixture model
        # -------------------------------------------

        # check if we have the right amount of filters in the latent dimension
        if n_latent != config["num_filters_list"][-1]:
            raise ValueError(
                "{!r} != {!r}".format(n_latent, config["num_filters_list"][-1])
            )
        if n_components < 1:
            raise ValueError("{!r} !> 0".format(n_components))

        # print out information about the mixture model components
        print("\t Charge PDF mixture model components:")
        decoder_config = self.decoder_charge.configuration.config["config"]
        if "decoder_names" in self.decoder_charge._untracked_data:
            counter = 0
            for name in self.decoder_charge._untracked_data["decoder_names"]:
                base_name, num, weight = decoder_config["decoder_mapping"][
                    name
                ]
                print(f"\t\t {base_name}: {num} [w={weight}]")
        else:
            print(f"\t\t Decoder name: {self.decoder_charge.name}")

        print("\t Time PDF mixture model components:")
        decoder_config = self.decoder.configuration.config["config"]
        counter = 0
        for name in self.decoder._untracked_data["decoder_names"]:
            base_name, num, weight = decoder_config["decoder_mapping"][name]
            if "mixture_component_t_seeds" in config:
                t_seed_i = config["mixture_component_t_seeds"][
                    counter : counter + num
                ]
            else:
                t_seed_i = np.zeros(num)
            counter += num
            print(f"\t\t {base_name}: {num} [w={weight}, t_seeds={t_seed_i}]")

        # shape: [n_batch, 86, 60, n_latent_charge + n_latent_time]
        latent_vars = conv_hex3d_layers[-1]

        # shape: [n_batch, 86, 60, n_latent_time]
        latent_vars_time = latent_vars[..., n_latent_charge:]

        # shape: [n_batch, 86, 60, n_latent_charge]
        latent_vars_charge = latent_vars[..., :n_latent_charge]

        tensor_dict["latent_vars"] = latent_vars
        tensor_dict["latent_vars_time"] = latent_vars_time

        # -------------------------
        # Calculate Time Exclusions
        # -------------------------
        if time_exclusions_exist:

            # get latent vars for each time window
            # Shape: [n_tw, n_latent_time]
            tw_latent_vars = tf.gather_nd(
                latent_vars_time, x_time_exclusions_ids
            )

            # ensure shape
            tw_latent_vars = tf.ensure_shape(
                tw_latent_vars, [None, n_latent_time]
            )

            # x: [n_tw, n_components], latent_vars: [n_tw, n_latent_time]
            #   --> [n_tw]
            tw_cdf_start = self.decoder.cdf(
                x=t_exclusions[:, 0],
                latent_vars=tw_latent_vars,
                reduce_components=True,
            )
            tw_cdf_stop = self.decoder.cdf(
                x=t_exclusions[:, 1],
                latent_vars=tw_latent_vars,
                reduce_components=True,
            )

            # shape: [n_tw]
            tw_cdf_exclusion = tf_helpers.safe_cdf_clip(
                tw_cdf_stop - tw_cdf_start
            )
            tw_cdf_exclusion = tf.cast(tw_cdf_exclusion, self.float_precision)

            # accumulate time window exclusions for each DOM and MM component
            # shape: [None, 86, 60]
            dom_cdf_exclusion = tf.zeros_like(latent_vars[..., 0])

            dom_cdf_exclusion = tf.tensor_scatter_nd_add(
                dom_cdf_exclusion,
                indices=x_time_exclusions_ids,
                updates=tw_cdf_exclusion,
            )
            dom_cdf_exclusion = tf_helpers.safe_cdf_clip(dom_cdf_exclusion)

            tensor_dict["tw_cdf_exclusion"] = tw_cdf_exclusion
            tensor_dict["dom_cdf_exclusion"] = dom_cdf_exclusion

        # -------------------------------------------
        # Get expected charge at DOM
        # -------------------------------------------

        # get charge scale to apply
        charge_scale = 1.0

        # scale charges by energy
        if config["scale_charge"]:
            charge_scale *= e_energy / 1000.0

        # scale charges by relative DOM efficiency
        if config["scale_charge_by_relative_dom_efficiency"]:
            charge_scale *= tf.expand_dims(
                detector.rel_dom_eff.astype(param_dtype_np), axis=-1
            )

        # scale charges by global DOM efficiency
        if config["scale_charge_by_global_dom_efficiency"]:
            charge_scale *= tf.expand_dims(
                parameter_list[self.get_index("DOMEfficiency")], axis=-1
            )

        if config["scale_charge_by_angular_acceptance"]:
            # do not let charge scaling go down to zero
            # Even if light is coming from directly above, the photons
            # will scatter and arrive from vaying angles.
            charge_scale *= (
                tf.clip_by_value(angular_acceptance, 0, float("inf")) + 1e-2
            )

        if config["scale_charge_by_relative_angular_acceptance"]:
            charge_scale *= tf.clip_by_value(
                relative_angular_acceptance,
                1e-2,
                100,
            )

        # check that things are spelled correctly
        for param_name in config["charge_scale_tensors"]:
            if param_name not in self.decoder_charge.loc_parameters:
                raise ValueError(
                    f"Charge scale tensor {param_name} not found in "
                    f"decoder loc_parameters: {self.decoder_charge.loc_parameters}!"
                )

        # transform charge scale tensors
        latent_vars_charge_scaled = []
        for idx, param_name in enumerate(self.decoder_charge.parameter_names):

            charge_tensor = tf.expand_dims(
                latent_vars_charge[..., idx], axis=-1
            )

            if param_name in self.decoder_charge.loc_parameters:

                # make sure no value range mapping is not defined for this tensor
                if param_name in self.decoder_charge.value_range_mapping:
                    raise ValueError(
                        f"Value range mapping for charge scale tensor "
                        f"{param_name} at index {idx} not allowed since "
                        f"manual mapping in model is done!"
                    )
                else:
                    print(
                        f"\t Applying charge scale to {param_name} at index {idx}"
                    )

                # clip value range for more stability during training
                # apply exponential which also forces positive values
                charge_tensor = tf.exp(
                    tf.clip_by_value(charge_tensor, -20.0, 15)
                )

                if param_name in config["charge_scale_tensors"]:
                    charge_tensor *= charge_scale

                # apply time window exclusions if needed
                if time_exclusions_exist:
                    charge_tensor = charge_tensor * (
                        1.0 - dom_cdf_exclusion[..., tf.newaxis] + self.epsilon
                    )

                # add small constant to make sure dom charges are > 0:
                charge_tensor += self.epsilon

            latent_vars_charge_scaled.append(charge_tensor)

        latent_vars_charge = tf.concat(latent_vars_charge_scaled, axis=-1)
        tensor_dict["latent_vars_charge"] = latent_vars_charge

        dom_charges_component = self.decoder_charge.expectation(
            latent_vars=latent_vars_charge,
            reduce_components=False,
        )
        if dom_charges_component.shape[1:] == [86, 60]:
            dom_charges_component = dom_charges_component[..., tf.newaxis]
        tensor_dict["dom_charges_component"] = dom_charges_component

        tensor_dict["dom_charges"] = self.decoder_charge.expectation(
            latent_vars=latent_vars_charge,
            reduce_components=True,
        )
        variance_component = self.decoder_charge.variance(
            latent_vars=latent_vars_charge, reduce_components=False
        )
        if variance_component.shape[1:] == [86, 60]:
            variance_component = variance_component[..., tf.newaxis]
        tensor_dict["dom_charges_variance_component"] = variance_component

        tensor_dict["dom_charges_variance"] = self.decoder_charge.variance(
            latent_vars=latent_vars_charge, reduce_components=True
        )

        # -------------------------
        # Compute charge PDF values
        # -------------------------
        dom_charges_pdf = self.decoder_charge.pdf(
            x=tf.squeeze(dom_charges_true, axis=3),
            latent_vars=latent_vars_charge,
            reduce_components=True,
        )
        # cast back to specified float precision
        dom_charges_pdf = tf.cast(dom_charges_pdf, self.float_precision)
        tensor_dict["dom_charges_pdf"] = dom_charges_pdf

        # --------------------------
        # Calculate Pulse PDF Values
        # --------------------------

        # get latent vars for each pulse
        # Shape: [n_pulses, n_latent_time]
        pulse_latent_vars = tf.gather_nd(latent_vars_time, pulses_ids)

        # ensure shapes
        pulse_latent_vars = tf.ensure_shape(
            pulse_latent_vars, [None, n_latent_time]
        )

        # ------------------------
        # Apply Time Mixture Model
        # ------------------------

        # t_pdf_at_dom: [n_events, n_components]
        # Shape: [n_pulses]
        pulse_pdf_values = self.decoder.pdf(
            x=t_pdf_at_dom,
            latent_vars=pulse_latent_vars,
            reduce_components=True,
        )
        pulse_cdf_values = self.decoder.cdf(
            x=t_pdf_at_dom,
            latent_vars=pulse_latent_vars,
            reduce_components=True,
        )

        # cast back to specified float precision
        pulse_pdf_values = tf.cast(pulse_pdf_values, self.float_precision)
        pulse_cdf_values = tf.cast(pulse_cdf_values, self.float_precision)

        tensor_dict["pulse_pdf"] = pulse_pdf_values  # not re-normalized
        tensor_dict["pulse_cdf"] = pulse_cdf_values  # not re-normalized

        # ----------------------------------
        # Compute contribution from cascades
        # ----------------------------------
        # Shape: [None, n_sel, 86, 60, n_params]
        parameters_cascade = self.get_cascade_parameters(
            parameters_all=parameters_all,
        )
        print("\t parameters_cascade", parameters_cascade)

        data_batch_dict_cascade = {
            parameter_tensor_name: parameters_cascade,
            "parameters_trafo_all": parameters_trafo_all,
            "x_pulses": data_batch_dict["x_pulses"],
            "x_pulses_ids": data_batch_dict["x_pulses_ids"],
            "x_time_exclusions": data_batch_dict["x_time_exclusions"],
            "x_time_exclusions_ids": data_batch_dict["x_time_exclusions_ids"],
            "x_dom_charge": data_batch_dict["x_dom_charge"],
        }

        # Result tensors: [None, 86, 60, n_sel]
        tensor_dict_cascade = self.get_tensors_cascade(
            data_batch_dict=data_batch_dict_cascade,
            is_training=is_training,
            parameter_tensor_name=parameter_tensor_name,
        )

        # -----------------------------------------
        # Combine pulse PDFs from track and cascade
        # -----------------------------------------

        # shape: [n_batch, 86, 60, 1 + n_sel]
        dom_weights = tf.concat(
            [
                tensor_dict["dom_charges"][..., tf.newaxis],
                tensor_dict_cascade["dom_charges"],
            ],
            axis=-1,
        )

        # shape: [n_batch, 86, 60]
        dom_charges = tf.reduce_sum(dom_weights, axis=-1)

        # shape: [n_batch, 86, 60, 1 + n_sel]
        dom_weights /= dom_charges[..., tf.newaxis] + self.epsilon

        # shape: [n_batch, 86, 60]
        dom_charges_variance = tf.reduce_sum(
            tf.concat(
                [
                    tensor_dict["dom_charges_variance"][..., tf.newaxis],
                    tensor_dict_cascade["dom_charges_variance"],
                ],
                axis=-1,
            ),
            axis=-1,
        )

        if time_exclusions_exist:
            # shape: [n_batch, 86, 60]
            dom_cdf_exclusion = tf.reduce_sum(
                tf.concat(
                    [
                        tensor_dict["dom_cdf_exclusion"][..., tf.newaxis],
                        tensor_dict_cascade["dom_cdf_exclusion"],
                    ],
                    axis=-1,
                )
                * dom_weights,
                axis=-1,
            )
            dom_cdf_exclusion = tf_helpers.safe_cdf_clip(dom_cdf_exclusion)

            # shape: [n_pulses, 1 + n_sel]
            tw_cdf_exclusion = tf.concat(
                [
                    tensor_dict["tw_cdf_exclusion"][..., tf.newaxis],
                    tensor_dict_cascade["tw_cdf_exclusion"],
                ],
                axis=-1,
            )

        # shape: [n_pulses, 1 + n_sel]
        pulse_weight = tf.gather_nd(dom_weights, pulses_ids)
        pulse_pdf = tf.reduce_sum(
            tf.concat(
                [
                    tensor_dict["pulse_pdf"][..., tf.newaxis],
                    tensor_dict_cascade["pulse_pdf"],
                ],
                axis=-1,
            )
            * pulse_weight,
            axis=-1,
        )
        pulse_cdf = tf.reduce_sum(
            tf.concat(
                [
                    tensor_dict["pulse_cdf"][..., tf.newaxis],
                    tensor_dict_cascade["pulse_cdf"],
                ],
                axis=-1,
            )
            * pulse_weight,
            axis=-1,
        )

        # -------------------------
        # collect charge components
        # -------------------------

        # Cascade
        n_components = tensor_dict_cascade["dom_charges_component"].shape[-1]

        if n_components != 1:
            raise NotImplementedError(
                "Cascade charge components are not yet implemented for more "
                "than one component! This is also not advisable as the "
                "number of required PDF evaluations grows with "
                "n_components^n_cascades."
            )

        # shape: [n_batch, 86, 60, n_components=1]
        mu_cascade = tf.reduce_sum(
            tensor_dict_cascade["dom_charges"],
            axis=-1,
            keepdims=True,
        )
        var_cascade = tf.reduce_sum(
            tensor_dict_cascade["dom_charges_variance"], axis=-1, keepdims=True
        )
        # weights_cascade = mu_cascade / (
        #     tensor_dict_cascade["dom_charges"][..., tf.newaxis]
        # ) # In the case of 1 component, this is just 1
        weights_cascade = tf.ones_like(mu_cascade)

        # now combine with track components
        mu_track = tensor_dict["dom_charges_component"]
        var_track = tensor_dict["dom_charges_variance_component"]
        weights_track = mu_track / (
            tensor_dict["dom_charges"][..., tf.newaxis]
        )

        n_total = weights_cascade.shape[-1] * mu_track.shape[-1]
        # multiply out components
        # Shape: [n_events, 86, 60, n_total * n_new] =
        #           Reshape(
        #               [n_events, 86, 60, n_total, 1] *
        #               [n_events, 86, 60, 1, n_new]
        #           )
        component_weights = tf.reshape(
            weights_cascade[..., tf.newaxis]
            * tf.expand_dims(weights_track, axis=3),
            [-1, 86, 60, n_total],
        )
        component_mu = tf.reshape(
            mu_cascade[..., tf.newaxis] + tf.expand_dims(mu_track, axis=3),
            [-1, 86, 60, n_total],
        )
        component_var = tf.reshape(
            var_cascade[..., tf.newaxis] + tf.expand_dims(var_track, axis=3),
            [-1, 86, 60, n_total],
        )

        # re-compute PDF values for all sources
        # Shape: [n_events, 86, 60, n_components_accumulated]
        # should already be normalized, but just to be sure
        component_weights /= tf.reduce_sum(
            component_weights,
            axis=3,
            keepdims=True,
        )
        component_alpha = (component_var - component_mu) / component_mu**2
        component_alpha = tf.clip_by_value(component_alpha, 1e-6, float("inf"))
        component_log_pdf = basis_functions.tf_log_negative_binomial(
            x=data_batch_dict["x_dom_charge"],
            mu=component_mu,
            alpha=component_alpha,
            add_normalization_term=True,
            dtype=self.configuration.config["config"]["float_precision"],
        )
        # Shape: [n_events, 86, 60]
        dom_charges_pdf = tf.reduce_sum(
            tf.math.exp(component_log_pdf) * component_weights, axis=3
        )

        # -----------------------------------------------
        # scale up pulse pdf by time exclusions if needed
        # -----------------------------------------------
        if time_exclusions_exist:

            # Shape: [n_pulses, 1 + n_sel]
            pulse_cdf_exclusion_total = tf.gather_nd(
                dom_cdf_exclusion,
                pulses_ids,
            )

            raise NotImplementedError(
                "get_prior_pulse_cdf_exclusion needs to be extended to "
                "support multiple cascades. Alternatively, an outer loop "
                "over cascades can be implemented."
            )

            # subtract excluded regions from cdf values
            pulse_cdf_exclusion = tf_helpers.get_prior_pulse_cdf_exclusion(
                x_pulses=pulses,
                x_pulses_ids=pulses_ids,
                x_time_exclusions=x_time_exclusions,
                x_time_exclusions_ids=x_time_exclusions_ids,
                tw_cdf_exclusion=tw_cdf_exclusion,
            )
            pulse_cdf -= pulse_cdf_exclusion

            # Shape: [n_pulses, n_sel]
            pulse_pdf /= 1.0 - pulse_cdf_exclusion_total + self.epsilon
            pulse_cdf /= 1.0 - pulse_cdf_exclusion_total + self.epsilon

        # ensure proper CDF ranges
        pulse_cdf = tf_helpers.safe_cdf_clip(pulse_cdf)

        # collect result tensors
        result_tensors = {
            "dom_charges": dom_charges,
            "dom_charges_component": component_mu,
            "dom_charges_pdf": dom_charges_pdf,
            "dom_charges_variance": dom_charges_variance,
            "dom_charges_variance_component": component_var,
            "pulse_pdf": pulse_pdf,
            "pulse_cdf": pulse_cdf,
            "nested_results": {
                "cascades": tensor_dict_cascade,  # note: no re-normalization!
                "track": tensor_dict,  # note: no re-normalization!
            },
        }
        if time_exclusions_exist:
            result_tensors["dom_cdf_exclusion"] = dom_cdf_exclusion

        return result_tensors
