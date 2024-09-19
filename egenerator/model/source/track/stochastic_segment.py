import logging
import tensorflow as tf
import numpy as np

from tfscripts import layers as tfs
from tfscripts.weights import new_weights

from egenerator.model.source.base import Source
from egenerator.utils import detector, basis_functions, angles


class StochasticTrackSegmentModel(Source):

    def __init__(self, logger=None):
        """Instantiate Source class

        Parameters
        ----------
        logger : logging.logger, optional
            The logger to use.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(StochasticTrackSegmentModel, self).__init__(logger=self._logger)

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

        # -------------------------------------------
        # Define input parameters of track hypothesis
        # -------------------------------------------
        parameter_names = [
            "x",
            "y",
            "z",
            "zenith",
            "azimuth",
            "energy",
            "time",
            "length",
            "stochasticity",
        ]

        num_snowstorm_params = 0
        if "snowstorm_parameter_names" in config:
            for param_name, num in config["snowstorm_parameter_names"]:
                num_snowstorm_params += num
                for i in range(num):
                    parameter_names.append(param_name.format(i))

        num_inputs = 16 + num_snowstorm_params

        if config["add_opening_angle"]:
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

    def get_dep_energy(self, energy, track_length, d1, d2):
        """Compute deposited energy of a track segment part.

        The deposited energy for a part of the track segment specified by the
        two distances d1 and d2 along the track is calculated. This assumes
        a constant dE/dx along the track.

        Parameters
        ----------
        energy : tf.Tensor
            The total deposited energy of the track over its whole length.
        track_length : tf.Tensor
            The total length of the track.
        d1 : tf.Tensor
            The distance along the track defining the beginning of the track
            segment part.
        d2 : tf.Tensor
            The distance along the track defining the end of the track segment
            part.

        Returns
        -------
        tf.Tensor
            The deposited energy within the distances d1 and d2.
        """

        d1_finite = self.shift_distance_on_track(track_length, d1)
        d2_finite = self.shift_distance_on_track(track_length, d2)

        # compute fraction of total track length
        rel_track_length = tf.abs(d2_finite - d1_finite) / track_length

        return energy * rel_track_length

    def shift_distance_on_track(self, track_length, distance):
        """Shift a distance on the finite track.

        Distance before track start or after end of the track will be shifted
        to track start and end, respectively.
        Note: this is done approximately by the use of a softmax function
        in order to maintain continuous gradients.

        Parameters
        ----------
        track_length : tf.Tensor
            The total length of the track.
        distance : tf.Tensor
            The distance along the track which is to be shifted onto the finite
            track.

        Returns
        -------
        tf.Tensor
            The shifted distance on the finite track
        """

        # Shift distances d1 and d2 on actual finite track
        # Note: this might be problematic due to missing gradients if simply
        # applying a cut. This approach in adding deltas with the softplus
        # activation function will help to provide gradients.
        # Using Softplus instead of RELU: this will smear out position, but
        # it will provide smooth gradients. This should be more important,
        # especially since actual track start/end uncertainty of +-1m should
        # be irrelevant. High-energy cascades should be handled separately.

        distance = (
            distance
            + tf.math.softplus(-distance)
            - tf.math.softplus(distance - track_length)
        )
        return distance

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
        """
        self.assert_configured(True)

        tensor_dict = {}

        config = self.configuration.config["config"]
        parameters = data_batch_dict[parameter_tensor_name]
        pulses = data_batch_dict["x_pulses"]
        pulses_ids = data_batch_dict["x_pulses_ids"][:, :3]

        # shape: [n_batch, 86, 60, 1]
        dom_charges_true = data_batch_dict["x_dom_charge"]

        pulse_times = pulses[:, 1]
        pulse_batch_id = pulses_ids[:, 0]

        print("pulses", pulses)
        print("pulses_ids", pulses_ids)
        print("parameters", parameters)

        # get transformed parameters
        parameters_trafo = self.data_trafo.transform(
            parameters, tensor_name=parameter_tensor_name
        )

        num_features = parameters.get_shape().as_list()[-1]

        # get parameters tensor dtype
        tensors = self.data_trafo.data["tensors"]
        param_dtype_np = tensors[parameter_tensor_name].dtype_np

        # -----------------------------------
        # Calculate input values for DOMs
        # -----------------------------------
        # Globals:
        #   track:
        #       energy, stochasticity, track length, dir_x, dir_y, dir_z
        #   SnowStorm parameters
        # Relative:
        #   true closest approach point:
        #       rel_length_on_track, disp_x, disp_y, disp_z, distance
        #       opening angle(track, displacement vector)
        #       E_-100m, E_+100m
        #   infinite track closest approach point:
        #       rel_length_on_track
        #   E_cherenkov_inf
        #   delta_dist_approach_points (between infinite and actual point)

        params_reshaped = tf.reshape(parameters, [-1, 1, 1, num_features])

        # parameters: x, y, z, zenith, azimuth, energy, time, length, stoch
        parameter_list = tf.unstack(params_reshaped, axis=-1)

        zenith = parameter_list[3]
        azimuth = parameter_list[4]
        energy = parameter_list[5] + 1e-1

        if is_training:
            # Ensure positive track length and energy
            assert_op_energy = tf.Assert(
                tf.greater_equal(tf.reduce_min(parameter_list[5]), -1e-3),
                [tf.reduce_min(parameter_list[5])],
            )
            assert_op_length = tf.Assert(
                tf.greater_equal(tf.reduce_min(parameter_list[7]), -1e3),
                [tf.reduce_min(parameter_list[7])],
            )
            with tf.control_dependencies([assert_op_energy, assert_op_length]):
                track_length = parameter_list[7] + 1.0
        else:
            track_length = parameter_list[7] + 1.0

        # calculate direction vector of track
        dir_x = -tf.sin(zenith) * tf.cos(azimuth)
        dir_y = -tf.sin(zenith) * tf.sin(azimuth)
        dir_z = -tf.cos(zenith)

        # vector of track vertex to DOM
        h_x = detector.x_coords[..., 0] - parameter_list[0]
        h_y = detector.x_coords[..., 1] - parameter_list[1]
        h_z = detector.x_coords[..., 2] - parameter_list[2]

        # distance between track vertex and closest approach of infinite track
        # Shape: [-1, 86, 60]
        dist_infinite_approach = dir_x * h_x + dir_y * h_y + dir_z * h_z
        rel_dist_infinite_approach = dist_infinite_approach / track_length

        # this value can get extremely large if track length is ~ 0
        # Therefore: limit it to range -10, 10 and map it onto (-1, 1)
        rel_dist_infinite_approach_trafo = tf.math.tanh(
            rel_dist_infinite_approach / 10.0
        )

        # shift distance of infinite track onto the finite track
        dist_closest_approach = self.shift_distance_on_track(
            track_length, dist_infinite_approach
        )
        rel_dist_closest_approach = dist_closest_approach / track_length

        # compute delta in distance of true closest approach and of closest
        # approach of infinite track
        delta_dist_approach = dist_infinite_approach - dist_closest_approach

        # # debugging
        # tf.print(
        #     'track_length summary',
        #     tf.reduce_min(track_length),
        #     tf.reduce_max(track_length),
        #     tf.reduce_mean(track_length),
        # )
        # tf.print(
        #     'dist_closest_approach summary',
        #     tf.reduce_min(dist_closest_approach),
        #     tf.reduce_max(dist_closest_approach),
        #     tf.reduce_mean(dist_closest_approach),
        # )
        # tf.print('dist_closest_approach', dist_closest_approach)
        # tf.print('dist_infinite_approach', dist_infinite_approach)

        # calculate closest approach points of track to each DOM
        closest_x = parameter_list[0] + dist_closest_approach * dir_x
        closest_y = parameter_list[1] + dist_closest_approach * dir_y
        closest_z = parameter_list[2] + dist_closest_approach * dir_z

        infinite_x = parameter_list[0] + dist_infinite_approach * dir_x
        infinite_y = parameter_list[1] + dist_infinite_approach * dir_y
        infinite_z = parameter_list[2] + dist_infinite_approach * dir_z

        # calculate displacement vectors
        dx_inf = detector.x_coords[..., 0] - infinite_x
        dy_inf = detector.x_coords[..., 1] - infinite_y
        dz_inf = detector.x_coords[..., 2] - infinite_z

        # shape: [-1, 86, 60]
        distance_infinite = tf.sqrt(dx_inf**2 + dy_inf**2 + dz_inf**2)

        dx = detector.x_coords[..., 0] - closest_x
        dy = detector.x_coords[..., 1] - closest_y
        dz = detector.x_coords[..., 2] - closest_z
        dx = tf.expand_dims(dx, axis=-1)
        dy = tf.expand_dims(dy, axis=-1)
        dz = tf.expand_dims(dz, axis=-1)

        # shape: [-1, 86, 60, 1]
        distance = tf.sqrt(dx**2 + dy**2 + dz**2) + 1e-1

        # calculate distance on track of cherenkov position
        cherenkov_angle = np.arccos(1.0 / 1.3195)
        dist_cherenkov_pos = (
            dist_infinite_approach
            - distance_infinite / np.tan(cherenkov_angle)
        )

        # calculate deposited energies between points on track
        energy_before = self.get_dep_energy(
            energy,
            track_length,
            d1=dist_closest_approach - 100,
            d2=dist_closest_approach,
        )
        energy_after = self.get_dep_energy(
            energy,
            track_length,
            d1=dist_closest_approach,
            d2=dist_closest_approach + 100,
        )
        energy_cherenkov = self.get_dep_energy(
            energy,
            track_length,
            d1=dist_cherenkov_pos,
            d2=dist_infinite_approach,
        )

        # calculate observation angle
        dx_normed = dx / distance
        dy_normed = dy / distance
        dz_normed = dz / distance

        # calculate opening angle of displacement vector and cascade direction
        opening_angle = angles.get_angle(
            tf.stack([dir_x, dir_y, dir_z], axis=-1),
            tf.concat([dx_normed, dy_normed, dz_normed], axis=-1),
        )
        opening_angle = tf.expand_dims(opening_angle, axis=-1)

        # transform dx, dy, dz, distance, zenith, azimuth to correct scale
        params_mean = self.data_trafo.data[parameter_tensor_name + "_mean"]
        params_std = self.data_trafo.data[parameter_tensor_name + "_std"]
        tensor = self.data_trafo.data["tensors"][parameter_tensor_name]
        norm_const = self.data_trafo.data["norm_constant"]

        distance /= np.linalg.norm(params_std[0:3]) + norm_const
        delta_dist_approach_trafo = delta_dist_approach / (
            np.linalg.norm(params_std[0:3]) + norm_const
        )
        opening_angle_traf = (opening_angle - params_mean[3]) / (
            norm_const + params_std[3]
        )

        x_parameters_expanded = tf.unstack(
            tf.reshape(parameters_trafo, [-1, 1, 1, num_features]), axis=-1
        )

        # transform energies
        if tensor.trafo_log[5]:
            energy_before_trafo = tf.math.log(1 + energy_before)
            energy_after_trafo = tf.math.log(1 + energy_after)
            energy_cherenkov_trafo = tf.math.log(1 + energy_cherenkov)

        # apply bias correction
        energy_before_trafo -= params_mean[5]
        energy_after_trafo -= params_mean[5]
        energy_cherenkov_trafo -= params_mean[5]

        # apply scaling factor
        energy_before_trafo /= params_std[5] + norm_const
        energy_after_trafo /= params_std[5] + norm_const
        energy_cherenkov_trafo /= params_std[5] + norm_const

        # parameters: x, y, z, zenith, azimuth, energy, time, length, stoch
        modified_parameters = tf.stack(
            [dir_x, dir_y, dir_z]
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
            tf.expand_dims(delta_dist_approach_trafo, axis=-1),
            tf.expand_dims(energy_before_trafo, axis=-1),
            tf.expand_dims(energy_after_trafo, axis=-1),
            tf.expand_dims(energy_cherenkov_trafo, axis=-1),
            tf.expand_dims(rel_dist_closest_approach, axis=-1),
            tf.expand_dims(rel_dist_infinite_approach_trafo, axis=-1),
        ]
        # for i, input_i in enumerate(input_list):
        #     tf.print(
        #         'input summary: {}'.format(i),
        #         tf.reduce_min(input_i),
        #         tf.reduce_max(input_i),
        #         tf.reduce_mean(input_i),
        #     )

        if config["add_opening_angle"]:
            input_list.append(opening_angle_traf)

        if config["add_dom_coordinates"]:

            # transform coordinates to correct scale with mean 0 std dev 1
            dom_coords = np.expand_dims(
                detector.x_coords.astype(param_dtype_np), axis=0
            )
            # scale of coordinates is ~-500m to ~500m with std dev of ~ 284m
            dom_coords /= 284.0

            # extend to correct batch shape:
            dom_coords = tf.ones_like(dx_normed) * dom_coords

            print("dom_coords", dom_coords)
            input_list.append(dom_coords)

        if config["num_local_vars"] > 0:

            # extend to correct shape:
            local_vars = (
                tf.ones_like(dx_normed) * self._untracked_data["local_vars"]
            )
            print("local_vars", local_vars)

            input_list.append(local_vars)

        # # Ensure input is not NaN
        if is_training:
            assert_op = tf.Assert(
                tf.math.is_finite(
                    tf.reduce_mean(tf.concat(input_list, axis=-1))
                ),
                [
                    tf.reduce_min(track_length),
                    tf.reduce_mean(track_length),
                    tf.reduce_mean(delta_dist_approach_trafo),
                    tf.reduce_mean(energy_before_trafo),
                    tf.reduce_mean(energy_after_trafo),
                    tf.reduce_mean(energy_cherenkov_trafo),
                    tf.reduce_mean(rel_dist_closest_approach),
                    tf.reduce_mean(rel_dist_infinite_approach_trafo),
                    tf.reduce_mean(rel_dist_infinite_approach),
                    tf.reduce_mean(dist_closest_approach),
                ],
            )
            with tf.control_dependencies([assert_op]):
                x_doms_input = tf.concat(input_list, axis=-1)
        else:
            x_doms_input = tf.concat(input_list, axis=-1)
        print("x_doms_input", x_doms_input)

        # -------------------------------------------
        # convolutional hex3d layers over X_IC86 data
        # -------------------------------------------
        conv_hex3d_layers = self._untracked_data["conv_hex3d_layer"](
            x_doms_input,
            is_training=is_training,
            keep_prob=config["keep_prob"],
        )

        # -------------------------------------------
        # Get expected charge at DOM
        # -------------------------------------------
        if config["charge_distribution_type"] == "asymmetric_gaussian":
            n_charge = 3
        elif config["charge_distribution_type"] == "negative_binomial":
            n_charge = 2
        elif config["charge_distribution_type"] == "poisson":
            n_charge = 1
        else:
            raise ValueError(
                "Unknown charge distribution type: {}".format(
                    config["charge_distribution_type"]
                )
            )

        # the result of the convolution layers are the latent variables
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
        dom_charges = tf.exp(dom_charges_trafo)

        # scale charges by cascade energy
        if config["scale_charge"]:
            scale_factor = tf.expand_dims(parameter_list[5], axis=-1) / 10000.0
            dom_charges *= scale_factor

        # scale charges by relative DOM efficiency
        if config["scale_charge_by_relative_dom_efficiency"]:
            dom_charges *= tf.expand_dims(
                detector.rel_dom_eff.astype(param_dtype_np), axis=-1
            )

        # scale charges by global DOM efficiency
        if config["scale_charge_by_global_dom_efficiency"]:
            dom_charges *= tf.expand_dims(
                parameter_list[self.get_index("DOMEfficiency")], axis=-1
            )

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

            print("dom_charges_sigma", dom_charges_sigma)
            print("dom_charges_llh", dom_charges_llh)
            print("dom_charges_unc", dom_charges_unc)

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
            dom_charges_alpha = tf.nn.elu(alpha_trafo - 5) + 1.000001

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

            print("dom_charges_llh", dom_charges_llh)

            # tf.print(
            #     'dom_charges_alpha',
            #     tf.reduce_min(dom_charges_alpha),
            #     tf.reduce_mean(dom_charges_alpha),
            #     tf.reduce_max(dom_charges_alpha),
            # )

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
                "Unknown charge distribution type: {}".format(
                    config["charge_distribution_type"]
                )
            )

        # -------------------------------------------
        # Get times at which to evaluate DOM PDF
        # -------------------------------------------

        # offset PDF evaluation times with cascade vertex time
        tensor_dict["time_offsets"] = parameters[:, 6]
        t_pdf = pulse_times - tf.gather(
            parameters[:, 6], indices=pulse_batch_id
        )
        # new shape: [None, 1]
        t_pdf = tf.expand_dims(t_pdf, axis=-1)
        t_pdf = tf.ensure_shape(t_pdf, [None, 1])

        # scale time range down to avoid big numbers:
        t_scale = 1.0 / self.time_unit_in_ns  # [1./ns]
        average_t_dist = 1000.0 * t_scale
        t_pdf = t_pdf * t_scale

        # -------------------------------------------
        # Gather latent vars of mixture model
        # -------------------------------------------
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
        factor_sigma = 1.0  # ns
        factor_mu = 1.0  # ns
        factor_r = 1.0
        factor_scale = 1.0

        # create correct offset and scaling
        latent_mu = average_t_dist + factor_mu * latent_mu
        latent_sigma = 2 + factor_sigma * latent_sigma
        latent_r = 1 + factor_r * latent_r
        latent_scale = 1 + factor_scale * latent_scale

        # force positive and min values
        latent_scale = tf.nn.elu(latent_scale) + 1.00001
        latent_r = tf.nn.elu(latent_r) + 1.001
        latent_sigma = tf.nn.elu(latent_sigma) + 1.001

        # normalize scale to sum to 1
        latent_scale /= tf.reduce_sum(latent_scale, axis=-1, keepdims=True)

        tensor_dict["latent_var_mu"] = latent_mu
        tensor_dict["latent_var_sigma"] = latent_sigma
        tensor_dict["latent_var_r"] = latent_r
        tensor_dict["latent_var_scale"] = latent_scale

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

        print("latent_mu", latent_mu)
        print("pulse_latent_mu", pulse_latent_mu)
        print("latent_scale", latent_scale)
        print("pulse_latent_scale", pulse_latent_scale)

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

        # new shape: [n_pulses]
        pulse_pdf_values = tf.reduce_sum(pulse_pdf_values, axis=-1)
        print("pulse_pdf_values", pulse_pdf_values)

        if is_training:
            # Ensure finite values
            asserts = []
            for name, tensor in sorted(tensor_dict.items()):
                assert_finite = tf.Assert(
                    tf.math.is_finite(tf.reduce_mean(tensor)),
                    [name, tf.reduce_mean(tensor)],
                )
                asserts.append(assert_finite)
            with tf.control_dependencies(asserts):
                tensor_dict["pulse_pdf"] = pulse_pdf_values
        else:
            tensor_dict["pulse_pdf"] = pulse_pdf_values
        # -------------------------------------------

        return tensor_dict
