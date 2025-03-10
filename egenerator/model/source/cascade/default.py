import tensorflow as tf
import numpy as np

from tfscripts import layers as tfs
from tfscripts.weights import new_weights

from egenerator.model.source.base import Source
from egenerator.utils.cascades import shift_to_maximum
from egenerator.utils import (
    detector,
    angles,
    dom_acceptance,
    tf_helpers,
)


def setup_cascade_model(config):
    """Set up a cascade model.

    Parameters
    ----------
    config : dict
        A dictionary of settings that fully defines the architecture.

    Returns
    -------
    tfscripts.Layers
        The CNN layers of the cascade model.
    tf.Variable
        The local variables tensor.
    list of str
        A list of parameter names. These parameters fully describe the
        source hypothesis. The model expects the hypothesis tensor input
        to be in the same order as this returned list.
    """
    if (
        config["scale_charge_by_angular_acceptance"]
        and config["scale_charge_by_relative_angular_acceptance"]
    ):
        raise ValueError(
            "Only one of 'scale_charge_by_angular_acceptance' "
            "and 'scale_charge_by_relative_angular_acceptance' "
            "can be set to True."
        )

    # ---------------------------------------------
    # Define input parameters of cascade hypothesis
    # ---------------------------------------------
    parameter_names = [
        "x",
        "y",
        "z",
        "zenith",
        "azimuth",
        "energy",
        "time",
    ]
    if "additional_label_names" in config:
        parameter_names += config["additional_label_names"]
        num_add_labels = len(config["additional_label_names"])
    else:
        num_add_labels = 0

    num_snowstorm_params = 0
    if "snowstorm_parameter_names" in config:
        for param_name, num in config["snowstorm_parameter_names"]:
            num_snowstorm_params += num
            for i in range(num):
                parameter_names.append(param_name.format(i))

    num_inputs = 11 + num_add_labels + num_snowstorm_params

    if config["add_anisotropy_angle"]:
        num_inputs += 2

    if config["add_opening_angle"]:
        num_inputs += 1

    if config["add_dom_angular_acceptance"]:
        num_inputs += 1

    if config["add_dom_coordinates"]:
        num_inputs += 3

    if config["num_local_vars"] > 0:
        local_vars = new_weights(
            shape=[1, 86, 60, config["num_local_vars"]],
            float_precision=config["float_precision"],
            name="local_dom_input_variables",
        )
        num_inputs += config["num_local_vars"]
    else:
        local_vars = None

    # -------------------------------------------
    # convolutional hex3d layers over X_IC86 data
    # -------------------------------------------
    cnn_layers = tfs.ConvNdLayers(
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

    return cnn_layers, local_vars, parameter_names


class DefaultCascadeModel(Source):

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

        cnn_layers, local_vars, parameter_names = setup_cascade_model(config)

        if local_vars is not None:
            self._untracked_data["local_vars"] = local_vars

        self._untracked_data["conv_hex3d_layer"] = cnn_layers

        return parameter_names

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
                    Shape: [n_events, 86, 60]
                'dom_charges_component':
                    The predicted charge at each DOM for each component
                    Shape: [n_events, 86, 60, n_components]
                'dom_charges_variance':
                    The predicted charge variance at each DOM
                    Shape: [n_events, 86, 60]
                'dom_charges_variance_component':
                    The predicted charge variance at each DOM for each
                    component of the mixture model.
                    Shape: [n_events, 86, 60, n_components]
                'dom_charges_pdf':
                    The charge likelihood evaluated for each DOM.
                    Shape: [n_events, 86, 60]
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

        print("Applying Default Cascade Model...")
        tensor_dict = {}

        config = self.configuration.config["config"]
        parameters = data_batch_dict[parameter_tensor_name]

        # ensure energy is greater or equal zero
        parameters = tf.unstack(parameters, axis=-1)
        parameters[5] = tf.clip_by_value(parameters[5], 0.0, float("inf"))
        parameters = tf.stack(parameters, axis=-1)

        pulses = data_batch_dict["x_pulses"]
        pulses_ids = data_batch_dict["x_pulses_ids"][:, :3]

        tensors = self.data_handler.tensors
        if (
            "x_time_exclusions" in tensors.names
            and tensors.list[tensors.get_index("x_time_exclusions")].exists
        ):
            time_exclusions_exist = True
            x_time_exclusions = data_batch_dict["x_time_exclusions"]
            x_time_exclusions_ids = data_batch_dict["x_time_exclusions_ids"]
        else:
            time_exclusions_exist = False
        print("\t Applying time exclusions:", time_exclusions_exist)

        # parameters: x, y, z, zenith, azimuth, energy, time
        num_features = parameters.get_shape().as_list()[-1]
        params_reshaped = tf.reshape(parameters, [-1, 1, 1, num_features])
        parameter_list = tf.unstack(params_reshaped, axis=-1)

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

        # get transformed (unshifted) parameters
        parameters_trafo = self.data_trafo.transform(
            parameters, tensor_name=parameter_tensor_name
        )

        # -----------------------------------
        # Calculate input values for DOMs
        # -----------------------------------
        # cascade_azimuth, cascade_zenith, cascade_energy
        # cascade_x, cascade_y, cascade_z
        # dx, dy, dz, distance
        # alpha (azimuthal angle to DOM)
        # beta (zenith angle to DOM)

        # calculate displacement vector
        # Shape: [n_batch, 86, 60] = [86, 60] - [n_batch, 1, 1]
        dx = detector.x_coords[..., 0] - parameter_list[0]
        dy = detector.x_coords[..., 1] - parameter_list[1]
        dz = detector.x_coords[..., 2] - parameter_list[2]
        # Shape: [n_batch, 86, 60, 1]
        dx = tf.expand_dims(dx, axis=-1)
        dy = tf.expand_dims(dy, axis=-1)
        dz = tf.expand_dims(dz, axis=-1)

        # stabilize with 1e-1 (10cm) when distance approaches DOM radius
        # Shape: [n_batch, 86, 60, 1]
        distance = tf.sqrt(dx**2 + dy**2 + dz**2) + 1e-1
        distance_xy = tf.sqrt(dx**2 + dy**2) + 1e-1

        # compute t_geometry: time for photon to travel to DOM
        # Shape: [n_batch, 86, 60, 1]
        c_ice = 0.22103046286329384  # m/ns
        dt_geometry = distance / c_ice

        # calculate observation angle
        # Shape: [n_batch, 86, 60, 1]
        dx_normed = dx / distance
        dy_normed = dy / distance
        dz_normed = dz / distance

        # angle in xy-plane (relevant for anisotropy)
        # Shape: [n_batch, 86, 60, 1]
        dx_normed_xy = dx / distance_xy
        dy_normed_xy = dy / distance_xy

        # calculate direction vector of cascade
        # Shape: [n_batch, 86, 60]
        cascade_zenith = parameter_list[3]
        cascade_azimuth = parameter_list[4]
        cascade_dir_x = -tf.sin(cascade_zenith) * tf.cos(cascade_azimuth)
        cascade_dir_y = -tf.sin(cascade_zenith) * tf.sin(cascade_azimuth)
        cascade_dir_z = -tf.cos(cascade_zenith)

        # calculate opening angle of displacement vector and cascade direction
        opening_angle = angles.get_angle(
            tf.stack([cascade_dir_x, cascade_dir_y, cascade_dir_z], axis=-1),
            tf.concat([dx_normed, dy_normed, dz_normed], axis=-1),
        )
        opening_angle = tf.expand_dims(opening_angle, axis=-1)

        # transform dx, dy, dz, distance, zenith, azimuth to correct scale
        params_mean = self.data_trafo.data[parameter_tensor_name + "_mean"]
        params_std = self.data_trafo.data[parameter_tensor_name + "_std"]
        norm_const = self.data_trafo.data["norm_constant"]

        distance /= np.linalg.norm(params_std[0:3]) + norm_const
        opening_angle_traf = (opening_angle - params_mean[3]) / (
            norm_const + params_std[3]
        )

        x_parameters_expanded = tf.unstack(
            tf.reshape(parameters_trafo, [-1, 1, 1, num_features]), axis=-1
        )

        modified_parameters = tf.stack(
            x_parameters_expanded[:3]
            + [cascade_dir_x, cascade_dir_y, cascade_dir_z]
            + [x_parameters_expanded[5]]
            + x_parameters_expanded[7:],
            axis=-1,
        )

        # put everything together
        params_expanded = tf.tile(modified_parameters, [1, 86, 60, 1])

        input_list = [
            params_expanded,
            dx_normed,
            dy_normed,
            dz_normed,
            distance,
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
                # shape: [n_batch, 1, 1, 1]
                p0_base = (
                    tf.ones_like(dz_normed) * config["baseline_hole_ice_p0"]
                )
                p1_base = (
                    tf.ones_like(dz_normed) * config["baseline_hole_ice_p1"]
                )

                # input tenser: cos(eta), p0, p1 in last dimension
                # Shape: [n_batch, 86, 60, 3]
                x_base = tf.concat([dz_normed, p0_base, p1_base], axis=-1)

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
                    ][..., tf.newaxis],
                    [1, 86, 60, 1],
                )
                p1 = tf.tile(
                    parameter_list[
                        self.get_index("HoleIceForward_Unified_p1")
                    ][..., tf.newaxis],
                    [1, 86, 60, 1],
                )

                # input tenser: cos(eta), p0, p1 in last dimension
                # Shape: [n_batch, 86, 60, 3]
                x = tf.concat([dz_normed, p0, p1], axis=-1)

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
            dom_coords = tf.ones_like(dx_normed) * dom_coords
            input_list.append(dom_coords)

        if config["num_local_vars"] > 0:

            # extend to correct shape:
            local_vars = (
                tf.ones_like(dx_normed) * self._untracked_data["local_vars"]
            )
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
        # Note that the pulse times are already offset by the cascade vertex
        # time. So we now only need to add dt_geometry.
        # shape: [n_batch, 86, 60, n_components] =
        #       [n_batch, 86, 60, 1] + [1, 1, 1, n_components]
        tensor_dict["time_offsets_per_dom"] = dt_geometry + t_seed

        # shape: [n_events]
        cascade_time = parameters[:, 6]
        tensor_dict["time_offsets"] = cascade_time

        # offset PDF evaluation times with cascade vertex time
        # shape: [n_pulses]
        t_pdf = pulse_times - tf.gather(cascade_time, indices=pulse_batch_id)

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
            tw_cascade_t = tf.gather(
                cascade_time, indices=x_time_exclusions_ids[:, 0]
            )

            # shape: [n_events, n_components]
            tw_cascade_t = tw_cascade_t[:, tf.newaxis] + tf.gather_nd(
                tensor_dict["time_offsets_per_dom"], x_time_exclusions_ids
            )

            # shape: [n_events, 2, n_components]
            #      = [n_events, 2, 1] - [n_events, 1, n_components]
            t_exclusions = x_time_exclusions[..., tf.newaxis] - tf.expand_dims(
                tw_cascade_t, axis=1
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

            tensor_dict["dom_cdf_exclusion"] = dom_cdf_exclusion

        # -------------------------------------------
        # Get expected charge at DOM
        # -------------------------------------------

        # get charge scale to apply
        charge_scale = 1.0

        # scale charges by cascade energy
        if config["scale_charge"]:
            # make sure cascade energy does not turn negative
            charge_scale *= tf.expand_dims(parameter_list[5], axis=-1) / 1000.0

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

        # scale up pulse pdf by time exclusions if needed
        if time_exclusions_exist:

            # Shape: [n_pulses]
            pulse_cdf_exclusion_total = tf.gather_nd(
                dom_cdf_exclusion,
                pulses_ids,
            )

            # subtract excluded regions from cdf values
            pulse_cdf_exclusion = tf_helpers.get_prior_pulse_cdf_exclusion(
                x_pulses=pulses,
                x_pulses_ids=pulses_ids,
                x_time_exclusions=x_time_exclusions,
                x_time_exclusions_ids=x_time_exclusions_ids,
                tw_cdf_exclusion=tw_cdf_exclusion,
            )
            pulse_cdf_values -= pulse_cdf_exclusion

            # Shape: [n_pulses]
            pulse_pdf_values /= 1.0 - pulse_cdf_exclusion_total + self.epsilon
            pulse_cdf_values /= 1.0 - pulse_cdf_exclusion_total + self.epsilon

        tensor_dict["pulse_pdf"] = pulse_pdf_values
        tensor_dict["pulse_cdf"] = pulse_cdf_values
        # ---------------------

        return tensor_dict
