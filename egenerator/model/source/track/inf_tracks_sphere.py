import logging
import tensorflow as tf
import numpy as np

from tfscripts import layers as tfs
from tfscripts.weights import new_weights

from egenerator.model.source.base import Source
from egenerator.utils import (
    detector,
    basis_functions,
    angles,
    dom_acceptance,
    tf_helpers,
)


class EnteringSphereInfTrack(Source):

    def __init__(self, logger=None):
        """Instantiate Source class

        Parameters
        ----------
        logger : logging.logger, optional
            The logger to use.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(EnteringSphereInfTrack, self).__init__(logger=self._logger)

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

        # ---------------------------------------------
        # Define input parameters of cascade hypothesis
        # ---------------------------------------------
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

        num_inputs = 13 + num_snowstorm_params

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

                'dom_charges': the predicted charge at each DOM
                               Shape: [-1, 86, 60]
                'dom_charges_variance':
                    the predicted variance on the charge at each DOM.
                    Shape: [-1, 86, 60]
                'pulse_pdf': The likelihood evaluated for each pulse
                             Shape: [-1]
            Optional:

                'pulse_cdf': The cumulative likelihood evaluated
                             for each pulse
                             Shape: [-1]
        """
        self.assert_configured(True)

        print("Applying EnteringSphereInfTrack Model...")
        tensor_dict = {}

        config = self.configuration.config["config"]
        parameters = data_batch_dict[parameter_tensor_name]
        pulses = data_batch_dict["x_pulses"]
        pulses_ids = data_batch_dict["x_pulses_ids"][:, :3]

        tensors = self.data_trafo.data["tensors"]
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
        parameters_trafo = self.data_trafo.transform(
            parameters, tensor_name=parameter_tensor_name
        )

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
        e_dir_x = -tf.sin(e_zenith) * tf.cos(e_azimuth)
        e_dir_y = -tf.sin(e_zenith) * tf.sin(e_azimuth)
        e_dir_z = -tf.cos(e_zenith)

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
            tf.stack([0.0, 0.0, 1.0], axis=-1),
            tf.concat(
                [
                    dx_cherenkov_normed,
                    dy_cherenkov_normed,
                    dz_cherenkov_normed,
                ],
                axis=-1,
            ),
        )[..., tf.newaxis]

        # compute t_geometry: time for photon to travel to DOM
        # Shape: [n_batch, 86, 60, 1]
        c_ice = 0.22103046286329384  # m/ns
        c = 0.299792458  # m/ns
        dt_geometry = distance_cherenkov / c_ice + length_cherenkov_pos / c

        # transform dx, dy, dz, distance, zenith, azimuth to correct scale
        norm_const = self.data_trafo.data["norm_constant"]

        # transform distances and lengths in detector
        distance_cherenkov_tr = distance_cherenkov / (
            config["sphere_radius"] + norm_const
        )
        length_cherenkov_pos_tr = length_cherenkov_pos / (
            config["sphere_radius"] + norm_const
        )

        # transform angle
        opening_angle_traf = opening_angle / (norm_const + np.pi)

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
                    dtype=config["float_precision"],
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
                    dtype=config["float_precision"],
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

        # -------------------------------------------
        # Get times at which to evaluate DOM PDF
        # -------------------------------------------

        # offset PDF evaluation times with cascade vertex time
        tensor_dict["time_offsets"] = e_time
        t_pdf = pulse_times - tf.gather(e_time, indices=pulse_batch_id)
        if time_exclusions_exist:
            # offset time exclusions

            # shape: [n_events]
            tw_cascade_t = tf.gather(
                e_time, indices=x_time_exclusions_ids[:, 0]
            )

            # shape: [n_events, 2, 1]
            t_exclusions = tf.expand_dims(
                x_time_exclusions - tf.expand_dims(tw_cascade_t, axis=-1),
                axis=-1,
            )
            t_exclusions = tf.ensure_shape(t_exclusions, [None, 2, 1])

        # new shape: [None, 1]
        t_pdf = tf.expand_dims(t_pdf, axis=-1)
        t_pdf = tf.ensure_shape(t_pdf, [None, 1])

        # scale time range down to avoid big numbers:
        t_scale = 1.0 / self.time_unit_in_ns  # [1./ns]
        t_pdf = t_pdf * t_scale
        dt_geometry = dt_geometry * t_scale
        if time_exclusions_exist:
            t_exclusions = t_exclusions * t_scale

        # -------------------------------------------
        # Gather latent vars of mixture model
        # -------------------------------------------
        if config["charge_distribution_type"] == "asymmetric_gaussian":
            n_charge = 3
        elif config["charge_distribution_type"] == "negative_binomial":
            n_charge = 2
        elif config["charge_distribution_type"] == "poisson":
            n_charge = 1
        else:
            raise ValueError(
                "Unknown charge distribution type: {!r}".format(
                    config["charge_distribution_type"]
                )
            )

        # check if we have the right amount of filters in the latent dimension
        n_models = config["num_latent_models"]
        if n_models * 4 + n_charge != config["num_filters_list"][-1]:
            raise ValueError(
                "{!r} != {!r}".format(
                    n_models * 4 + n_charge, config["num_filters_list"][-1]
                )
            )
        if n_models <= 1:
            raise ValueError("{!r} !> 1".format(n_models))

        print("\t Charge method:", config["charge_distribution_type"])
        print("\t Number of Asymmetric Gaussian Components:", n_models)

        # shape: [n_batch, 86, 60, n_models * 4 + n_charge]
        out_layer = conv_hex3d_layers[-1]
        latent_mu = out_layer[
            ..., n_charge + 0 * n_models : n_charge + 1 * n_models
        ]
        latent_sigma = out_layer[
            ..., n_charge + 1 * n_models : n_charge + 2 * n_models
        ]
        latent_r = out_layer[
            ..., n_charge + 2 * n_models : n_charge + 3 * n_models
        ]
        latent_scale = out_layer[
            ..., n_charge + 3 * n_models : n_charge + 4 * n_models
        ]

        # add reasonable scaling for parameters assuming the latent vars
        # are distributed normally around zero
        factor_sigma = 1000 * t_scale  # ns
        factor_mu = 500 * t_scale  # ns
        factor_r = 1.0
        factor_scale = 1.0

        # create correct offset and scaling
        latent_sigma = 2 + factor_sigma * latent_sigma
        latent_r = 1 + factor_r * latent_r
        latent_scale = 1 + factor_scale * latent_scale

        # special handling for placements of asymmetric Gaussians via mu:
        # these will be placed relative to t_geometry,
        # which is the earliest possible time for a DOM to be hit from photons
        # originating from the cascade vertex. Once waveforms are defined
        # relative to t_geometry, their overall features are similar to each
        # other. In particular, there is a peak around 8000ns arising form
        # after pulses. We will place the asymmetric Gaussians around these
        # features.
        t_seed = (
            np.r_[
                [0, 100, 8000, 14000, 4000, 800, 300, 1000, 400, 2000],
                np.random.RandomState(42).uniform(0, 14000, max(1, n_models)),
            ][:n_models]
            * t_scale
        )
        t_seed = np.reshape(t_seed, [1, 1, 1, n_models])

        # per DOM offset to shift to t_geometry
        # Note that the pulse times are already offset by the cascade vertex
        # time. So we now only need to  add dt_geometry.
        # shape: [n_batch, 86, 60, n_models] =
        #       [n_batch, 86, 60, 1] + [1, 1, 1, n_models]
        #       + [n_batch, 86, 60, n_models]
        latent_mu = dt_geometry + t_seed + factor_mu * latent_mu

        # force positive and min values
        latent_scale = tf.math.exp(latent_scale)
        latent_r = tf.math.exp(latent_r)
        latent_sigma = tf.math.exp(latent_sigma) + 0.0001

        # normalize scale to sum to 1
        latent_scale /= tf.reduce_sum(latent_scale, axis=-1, keepdims=True)

        # Sort mixture model components in time if desired
        if (
            "prevent_mixture_component_swapping" in config
            and config["prevent_mixture_component_swapping"]
        ):

            # swap latent variables of components, such that these are ordered
            # in time. This puts a constrained on the model and reduces
            # the permutation options and should thus facilitate training.
            # We could keep the latent_mu in place and sort the other
            # components accordingly. An equivalent alternative is to keep
            # the other components in place and to simply sort the latent_mu.
            latent_mu = tf.ensure_shape(
                tf.sort(latent_mu, axis=-1), shape=[None, 86, 60, n_models]
            )

        tensor_dict["latent_var_mu"] = latent_mu
        tensor_dict["latent_var_sigma"] = latent_sigma
        tensor_dict["latent_var_r"] = latent_r
        tensor_dict["latent_var_scale"] = latent_scale

        # -------------------------
        # Calculate Time Exclusions
        # -------------------------
        if time_exclusions_exist:

            # get latent vars for each time window
            tw_latent_mu = tf.gather_nd(latent_mu, x_time_exclusions_ids)
            tw_latent_sigma = tf.gather_nd(latent_sigma, x_time_exclusions_ids)
            tw_latent_r = tf.gather_nd(latent_r, x_time_exclusions_ids)

            # ensure shapes
            tw_latent_mu = tf.ensure_shape(tw_latent_mu, [None, n_models])
            tw_latent_sigma = tf.ensure_shape(
                tw_latent_sigma, [None, n_models]
            )
            tw_latent_r = tf.ensure_shape(tw_latent_r, [None, n_models])

            # [n_tw, 1] * [n_tw, n_models] = [n_tw, n_models]
            tw_cdf_start = basis_functions.tf_asymmetric_gauss_cdf(
                x=t_exclusions[:, 0],
                mu=tw_latent_mu,
                sigma=tw_latent_sigma,
                r=tw_latent_r,
            )
            tw_cdf_stop = basis_functions.tf_asymmetric_gauss_cdf(
                x=t_exclusions[:, 1],
                mu=tw_latent_mu,
                sigma=tw_latent_sigma,
                r=tw_latent_r,
            )

            tw_cdf_exclusion = tf_helpers.safe_cdf_clip(
                tw_cdf_stop - tw_cdf_start
            )

            # accumulate time window exclusions for each DOM and MM component
            # shape: [None, 86, 60, n_models]
            dom_cdf_exclusion = tf.zeros_like(latent_mu)

            dom_cdf_exclusion = tf.tensor_scatter_nd_add(
                dom_cdf_exclusion,
                indices=x_time_exclusions_ids,
                updates=tw_cdf_exclusion,
            )
            dom_cdf_exclusion = tf_helpers.safe_cdf_clip(dom_cdf_exclusion)

            # Shape: [None, 86, 60, 1]
            dom_cdf_exclusion_sum = tf.reduce_sum(
                dom_cdf_exclusion * latent_scale, axis=-1, keepdims=True
            )

            # add safety checks to make sure we aren't clipping too much
            dom_cdf_exclusion_sum = tf_helpers.safe_cdf_clip(
                dom_cdf_exclusion_sum
            )
            tensor_dict["dom_cdf_exclusion_sum"] = dom_cdf_exclusion_sum

        # -------------------------------------------
        # Get expected charge at DOM
        # -------------------------------------------

        # the result of the convolution layers are the latent variables
        # Shape: [n_batch, 86, 60, 1]
        dom_charges_trafo = tf.expand_dims(
            conv_hex3d_layers[-1][..., 0], axis=-1
        )

        # clip value range for more stability during training
        dom_charges_trafo = tf.clip_by_value(
            dom_charges_trafo,
            -20.0,
            15,
        )

        # apply exponential which also forces positive values
        # Shape: [n_batch, 86, 60, 1]
        dom_charges = tf.exp(dom_charges_trafo)

        # scale charges by cascade energy
        if config["scale_charge"]:
            # make sure cascade energy does not turn negative
            # shape: [n_batch, 86, 60, 1] * [n_batch, 1, 1, 1]
            scale_factor = e_energy / 1000.0
            dom_charges *= scale_factor

        # scale charges by relative DOM efficiency
        if config["scale_charge_by_relative_dom_efficiency"]:
            dom_charges *= tf.expand_dims(
                detector.rel_dom_eff.astype(param_dtype_np), axis=-1
            )

        # scale charges by global DOM efficiency
        if config["scale_charge_by_global_dom_efficiency"]:
            dom_charges *= parameter_list[self.get_index("DOMEfficiency")]

        if config["scale_charge_by_angular_acceptance"]:
            # do not let charge scaling go down to zero
            # Even if cascade is coming from directly above, the photons
            # will scatter and arrive from vaying angles.
            dom_charges *= (
                tf.clip_by_value(angular_acceptance, 0, float("inf")) + 1e-2
            )

        if config["scale_charge_by_relative_angular_acceptance"]:
            dom_charges *= tf.clip_by_value(
                relative_angular_acceptance,
                1e-2,
                100,
            )

        # apply time window exclusions if needed
        if time_exclusions_exist:
            dom_charges = dom_charges * (1.0 - dom_cdf_exclusion_sum + 1e-3)

        # add small constant to make sure dom charges are > 0:
        dom_charges += 1e-7

        tensor_dict["dom_charges"] = dom_charges

        # -------------------------------------
        # get charge distribution uncertainties
        # -------------------------------------
        if config["charge_distribution_type"] == "asymmetric_gaussian":
            sigma_scale_trafo = tf.expand_dims(
                conv_hex3d_layers[-1][..., 1], axis=-1
            )
            dom_charges_r_trafo = tf.expand_dims(
                conv_hex3d_layers[-1][..., 2], axis=-1
            )

            # create correct offset and scaling
            sigma_scale_trafo = 0.1 * sigma_scale_trafo - 2
            dom_charges_r_trafo = 0.01 * dom_charges_r_trafo - 2

            # force positive and min values
            # The uncertainty can't be smaller than Poissonian error.
            # However, we are approximating the distribution with an
            # asymmetric Gaussian which might result in slightly different
            # sigmas at low values.
            # We will limit Gaussian sigma to a minimum value of 90% of the
            # Poisson expectation.
            # The Gaussian approximation will not hold for low charge DOMs.
            # We will use a standard poisson likelihood for DOMs with a true
            # detected charge of less than 5.
            # Since the normalization is not correct for these likelihoods
            # we need to keep the choice of llh fixed for an event, e.g.
            # base the decision on the true measured charge.
            # Note: this is not a correct and proper PDF description!
            sigma_scale = tf.nn.elu(sigma_scale_trafo) + 1.9
            dom_charges_r = tf.nn.elu(dom_charges_r_trafo) + 1.9

            # set default value to poisson uncertainty
            dom_charges_sigma = (
                tf.sqrt(tf.clip_by_value(dom_charges, 0.0001, float("inf")))
                * sigma_scale
            )

            # set threshold under which a Poisson Likelihood is used
            charge_threshold = 5

            # Apply Asymmetric Gaussian and/or Poisson Likelihood
            # shape: [n_batch, 86, 60, 1]
            eps = 1e-7
            dom_charges_llh = tf.where(
                dom_charges_true > charge_threshold,
                tf.math.log(
                    basis_functions.tf_asymmetric_gauss(
                        x=dom_charges_true,
                        mu=dom_charges,
                        sigma=dom_charges_sigma,
                        r=dom_charges_r,
                    )
                    + eps
                ),
                dom_charges_true * tf.math.log(dom_charges + eps)
                - dom_charges,
            )

            # compute (Gaussian) uncertainty on predicted dom charge
            dom_charges_unc = tf.where(
                dom_charges_true > charge_threshold,
                # take mean of left and right side uncertainty
                # Note: this might not be correct
                dom_charges_sigma * ((1 + dom_charges_r) / 2.0),
                tf.math.sqrt(dom_charges + eps),
            )

            # add tensors to tensor dictionary
            tensor_dict["dom_charges_sigma"] = dom_charges_sigma
            tensor_dict["dom_charges_r"] = dom_charges_r
            tensor_dict["dom_charges_unc"] = dom_charges_unc
            tensor_dict["dom_charges_variance"] = dom_charges_unc**2
            tensor_dict["dom_charges_log_pdf_values"] = dom_charges_llh

        elif config["charge_distribution_type"] == "negative_binomial":
            """
            Use negative binomial PDF instead of Poisson to account for
            over-dispersion induces by systematic variations.

            The parameterization chosen here is defined by the mean mu and
            the over-dispersion factor alpha.

                Var(x) = mu + alpha*mu**2

            Alpha must be greater than zero.
            """
            alpha_trafo = tf.expand_dims(
                conv_hex3d_layers[-1][..., 1], axis=-1
            )

            # create correct offset and force positive and min values
            # The over-dispersion parameterized by alpha must be greater zero
            dom_charges_alpha = tf.math.exp(alpha_trafo - 5) + 0.000001

            # compute log pdf
            dom_charges_llh = basis_functions.tf_log_negative_binomial(
                x=dom_charges_true,
                mu=dom_charges,
                alpha=dom_charges_alpha,
            )

            # compute standard deviation
            # std = sqrt(var) = sqrt(mu + alpha*mu**2)
            dom_charges_variance = (
                dom_charges + dom_charges_alpha * dom_charges**2
            )
            dom_charges_unc = tf.sqrt(dom_charges_variance)

            # add tensors to tensor dictionary
            tensor_dict["dom_charges_alpha"] = dom_charges_alpha
            tensor_dict["dom_charges_unc"] = dom_charges_unc
            tensor_dict["dom_charges_variance"] = dom_charges_variance
            tensor_dict["dom_charges_log_pdf_values"] = dom_charges_llh

        elif config["charge_distribution_type"] == "poisson":
            # Poisson Distribution: variance is equal to expected charge
            tensor_dict["dom_charges_unc"] = tf.sqrt(dom_charges)
            tensor_dict["dom_charges_variance"] = dom_charges

        else:
            raise ValueError(
                "Unknown charge distribution type: {!r}".format(
                    config["charge_distribution_type"]
                )
            )

        # --------------------------
        # Calculate Pulse PDF Values
        # --------------------------

        # get latent vars for each pulse
        pulse_latent_mu = tf.gather_nd(latent_mu, pulses_ids)
        pulse_latent_sigma = tf.gather_nd(latent_sigma, pulses_ids)
        pulse_latent_r = tf.gather_nd(latent_r, pulses_ids)
        pulse_latent_scale = tf.gather_nd(latent_scale, pulses_ids)

        # ensure shapes
        pulse_latent_mu = tf.ensure_shape(pulse_latent_mu, [None, n_models])
        pulse_latent_sigma = tf.ensure_shape(
            pulse_latent_sigma, [None, n_models]
        )
        pulse_latent_r = tf.ensure_shape(pulse_latent_r, [None, n_models])
        pulse_latent_scale = tf.ensure_shape(
            pulse_latent_scale, [None, n_models]
        )

        # -------------------------------------------
        # Apply Asymmetric Gaussian Mixture Model
        # -------------------------------------------

        # [n_pulses, 1] * [n_pulses, n_models] = [n_pulses, n_models]
        pulse_pdf_values = (
            basis_functions.tf_asymmetric_gauss(
                x=t_pdf,
                mu=pulse_latent_mu,
                sigma=pulse_latent_sigma,
                r=pulse_latent_r,
            )
            * pulse_latent_scale
        )
        pulse_cdf_values = (
            basis_functions.tf_asymmetric_gauss_cdf(
                x=t_pdf,
                mu=pulse_latent_mu,
                sigma=pulse_latent_sigma,
                r=pulse_latent_r,
            )
            * pulse_latent_scale
        )

        # new shape: [n_pulses]
        pulse_pdf_values = tf.reduce_sum(pulse_pdf_values, axis=-1)
        pulse_cdf_values = tf.reduce_sum(pulse_cdf_values, axis=-1)

        # scale up pulse pdf by time exclusions if needed
        if time_exclusions_exist:

            # Shape: [n_pulses, 1] -> squeeze -> [n_pulses]
            pulse_cdf_exclusion = tf.squeeze(
                tf.gather_nd(dom_cdf_exclusion_sum, pulses_ids), axis=1
            )

            # Shape: [n_pulses]
            pulse_pdf_values /= 1.0 - pulse_cdf_exclusion + 1e-3
            pulse_cdf_values /= 1.0 - pulse_cdf_exclusion + 1e-3

        tensor_dict["pulse_pdf"] = pulse_pdf_values
        tensor_dict["pulse_cdf"] = pulse_cdf_values
        # ---------------------

        return tensor_dict
