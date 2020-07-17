from __future__ import division, print_function
import logging
import tensorflow as tf
import numpy as np

from tfscripts import layers as tfs
from tfscripts.weights import new_weights

from egenerator.model.source.base import Source
from egenerator.utils import detector, basis_functions, angles
# from egenerator.manager.component import Configuration, BaseComponent


class ChargeQuantileCascadeModel(Source):

    """This Cascade model predicts quantile time PDFs

    The pdf is dependent on the charge quantile (fraction of total DOM charge)
    of the pulse and the total measured (true) charge D_i at the DOM:
        Pulse PDF: p(t_i | q_i, D_i) for the i-th pulse
    The pulse PDF p(t_i | q_i, D_i) is estimated via a mixture model of
    asymmetric Gaussians.
    Note: for high charge DOMs, the pulse PDF is not dependent on noise hits
    and is therefore very Gaussian shaped and in this case could be
    approximated by a single Gaussian.
    """

    def __init__(self, logger=None):
        """Instantiate Source class

        Parameters
        ----------
        logger : logging.logger, optional
            The logger to use.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(ChargeQuantileCascadeModel, self).__init__(logger=self._logger)

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

        # ---------------------------------------------
        # Define input parameters of cascade hypothesis
        # ---------------------------------------------
        parameter_names = ['x', 'y', 'z', 'zenith', 'azimuth',
                           'energy', 'time']

        num_snowstorm_params = 0
        if 'snowstorm_parameter_names' in config:
            for param_name, num in config['snowstorm_parameter_names']:
                num_snowstorm_params += num
                for i in range(num):
                    parameter_names.append(param_name.format(i))

        num_features = 7 + num_snowstorm_params
        num_inputs = 11 + num_snowstorm_params

        if config['add_opening_angle']:
            num_inputs += 1

        if config['add_dom_coordinates']:
            num_inputs += 3

        if config['num_local_vars'] > 0:
            self._untracked_data['local_vars'] = new_weights(
                    shape=[1, 86, 60, config['num_local_vars']],
                    name='local_dom_input_variables')
            num_inputs += config['num_local_vars']

        # -------------------------------------------
        # convolutional hex3d layers over X_IC86 data
        # -------------------------------------------
        self._untracked_data['conv_hex3d_layer'] = tfs.ConvNdLayers(
            input_shape=[-1, 86, 60, num_inputs],
            filter_size_list=config['filter_size_list'],
            num_filters_list=config['num_filters_list'],
            pooling_type_list=None,
            pooling_strides_list=[1, 1, 1, 1],
            pooling_ksize_list=[1, 1, 1, 1],
            use_dropout_list=config['use_dropout_list'],
            padding_list='SAME',
            strides_list=[1, 1, 1, 1],
            use_batch_normalisation_list=config['use_batch_norm_list'],
            activation_list=config['activation_list'],
            use_residual_list=config['use_residual_list'],
            hex_zero_out_list=False,
            dilation_rate_list=None,
            hex_num_rotations_list=1,
            method_list=config['method_list'],
            )

        # ----------------------------------------------------
        # Fully Connected Layers to comput PDF p(t | q_i, D_i)
        # ----------------------------------------------------

        # minus 1 for charge prediction, plus 2 for q_i, D_i
        num_fc_inputs = config['num_filters_list'][-1] - 1 + 2
        if config['estimate_charge_distribution'] is True:
            num_fc_inputs -= 2
        elif config['estimate_charge_distribution'] == 'negative_binomial':
            num_fc_inputs -= 1

        if config['add_predicted_charge_to_latent_vars']:
            num_fc_inputs += 1

        self._untracked_data['fully_connected_layer'] = tfs.FCLayers(
            input_shape=[-1, num_fc_inputs],
            fc_sizes=config['fc_num_filters_list'],
            use_dropout_list=config['fc_use_dropout_list'],
            activation_list=config['fc_activation_list'],
            use_batch_normalisation_list=config['fc_use_batch_norm_list'],
            use_residual_list=config['fc_use_residual_list'],
        )

        return parameter_names

    def get_tensors(self, data_batch_dict, is_training,
                    parameter_tensor_name='x_parameters'):
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
                The pulse indices (batch_index, string, dom) of all pulses in
                the batch of events.
                Shape: [-1, 3]
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

        config = self.configuration.config['config']
        parameters = data_batch_dict[parameter_tensor_name]
        pulses = data_batch_dict['x_pulses']
        pulses_ids = data_batch_dict['x_pulses_ids']

        # shape: [n_batch, 86, 60, 1]
        dom_charges_true = data_batch_dict['x_dom_charge']

        pulse_quantiles = pulses[:, 2]
        pulse_times = pulses[:, 1]
        pulse_charges = pulses[:, 0]
        pulse_batch_id = pulses_ids[:, 0]

        print('pulses', pulses)
        print('pulses_ids', pulses_ids)
        print('parameters', parameters)

        # get transformed parameters
        parameters_trafo = self.data_trafo.transform(
                                parameters, tensor_name=parameter_tensor_name)

        num_features = parameters.get_shape().as_list()[-1]

        # -----------------------------------
        # Calculate input values for DOMs
        # -----------------------------------
        # cascade_azimuth, cascade_zenith, cascade_energy
        # cascade_x, cascade_y, cascade_z
        # dx, dy, dz, distance
        # alpha (azimuthal angle to DOM)
        # beta (zenith angle to DOM)

        params_reshaped = tf.reshape(parameters, [-1, 1, 1, num_features])

        # parameters: x, y, z, zenith, azimuth, energy, time
        parameter_list = tf.unstack(params_reshaped, axis=-1)

        # calculate displacement vector
        dx = detector.x_coords[..., 0] - parameter_list[0]
        dy = detector.x_coords[..., 1] - parameter_list[1]
        dz = detector.x_coords[..., 2] - parameter_list[2]
        dx = tf.expand_dims(dx, axis=-1)
        dy = tf.expand_dims(dy, axis=-1)
        dz = tf.expand_dims(dz, axis=-1)

        # shape: [-1, 86, 60, 1]
        distance = tf.sqrt(dx**2 + dy**2 + dz**2)

        # calculte time it takes for unscattered light to propagate to DOM
        c_ice = 0.22103046286329384  # meter / ns
        light_propagation_time = distance / c_ice

        # calculate observation angle
        dx_normed = dx / distance
        dy_normed = dy / distance
        dz_normed = dz / distance

        # calculate direction vector of cascade
        cascade_zenith = parameter_list[3]
        cascade_azimuth = parameter_list[4]
        cascade_dir_x = -tf.sin(cascade_zenith) * tf.cos(cascade_azimuth)
        cascade_dir_y = -tf.sin(cascade_zenith) * tf.sin(cascade_azimuth)
        cascade_dir_z = -tf.cos(cascade_zenith)

        # calculate opening angle of displacement vector and cascade direction
        opening_angle = angles.get_angle(tf.stack([cascade_dir_x,
                                                   cascade_dir_y,
                                                   cascade_dir_z], axis=-1),
                                         tf.concat([dx_normed,
                                                    dy_normed,
                                                    dz_normed], axis=-1)
                                         )
        opening_angle = tf.expand_dims(opening_angle, axis=-1)

        # transform dx, dy, dz, distance, zenith, azimuth to correct scale
        params_mean = self.data_trafo.data[parameter_tensor_name+'_mean']
        params_std = self.data_trafo.data[parameter_tensor_name+'_std']
        norm_const = self.data_trafo.data['norm_constant']

        distance /= (np.linalg.norm(params_std[0:3]) + norm_const)
        opening_angle_traf = ((opening_angle - params_mean[3]) /
                              (norm_const + params_std[3]))

        x_parameters_expanded = tf.unstack(tf.reshape(
                                                parameters_trafo,
                                                [-1, 1, 1, num_features]),
                                           axis=-1)

        modified_parameters = tf.stack(x_parameters_expanded[:3]
                                       + [cascade_dir_x,
                                          cascade_dir_y,
                                          cascade_dir_z]
                                       + [x_parameters_expanded[5]]
                                       + x_parameters_expanded[7:],
                                       axis=-1)

        # put everything together
        params_expanded = tf.tile(modified_parameters,
                                  [1, 86, 60, 1])

        input_list = [params_expanded, dx_normed, dy_normed, dz_normed,
                      distance]

        if config['add_opening_angle']:
            input_list.append(opening_angle_traf)

        if config['add_dom_coordinates']:

            # transform coordinates to correct scale with mean 0 std dev 1
            dom_coords = np.expand_dims(detector.x_coords.astype(np.float32),
                                        axis=0)
            # scale of coordinates is ~-500m to ~500m with std dev of ~ 284m
            dom_coords /= 284.

            # extend to correct batch shape:
            dom_coords = (tf.ones_like(dx_normed) * dom_coords)

            print('dom_coords', dom_coords)
            input_list.append(dom_coords)

        if config['num_local_vars'] > 0:

            # extend to correct shape:
            local_vars = (tf.ones_like(dx_normed) *
                          self._untracked_data['local_vars'])
            print('local_vars', local_vars)

            input_list.append(local_vars)

        x_doms_input = tf.concat(input_list, axis=-1)
        print('x_doms_input', x_doms_input)

        # -------------------------------------------
        # convolutional hex3d layers over X_IC86 data
        # -------------------------------------------
        conv_hex3d_layers = self._untracked_data['conv_hex3d_layer'](
                                        x_doms_input, is_training=is_training,
                                        keep_prob=config['keep_prob'])

        # -------------------------------------------
        # Get expected charge at DOM
        # -------------------------------------------
        if config['estimate_charge_distribution'] is True:
            n_charge = 3
        elif config['estimate_charge_distribution'] == 'negative_binomial':
            n_charge = 2
        else:
            n_charge = 1

        # the result of the convolution layers are the latent variables
        dom_charges_trafo = tf.expand_dims(conv_hex3d_layers[-1][..., 0],
                                           axis=-1)

        # apply exponential which also forces positive values
        dom_charges = tf.exp(dom_charges_trafo)

        # scale charges by cascade energy
        if config['scale_charge']:
            scale_factor = tf.expand_dims(parameter_list[5], axis=-1) / 10000.0
            dom_charges *= scale_factor

        # scale charges by realtive DOM efficiency
        if config['scale_charge_by_relative_dom_efficiency']:
            dom_charges *= tf.expand_dims(detector.rel_dom_eff, axis=-1)

        # scale charges by global DOM efficiency
        if config['scale_charge_by_global_dom_efficiency']:
            dom_charges *= tf.expand_dims(
                parameter_list[self.get_index('DOMEfficiency')], axis=-1)

        # add small constant to make sure dom charges are > 0:
        dom_charges += 1e-7

        tensor_dict['dom_charges'] = dom_charges

        # -------------------------------------
        # get charge distribution uncertainties
        # -------------------------------------
        if config['estimate_charge_distribution'] is True:
            sigma_scale_trafo = tf.expand_dims(conv_hex3d_layers[-1][..., 1],
                                               axis=-1)
            dom_charges_r_trafo = tf.expand_dims(conv_hex3d_layers[-1][..., 2],
                                                 axis=-1)

            # create correct offset and scaling
            sigma_scale_trafo = 0.1 * sigma_scale_trafo - 2
            dom_charges_r_trafo = 0.01 * dom_charges_r_trafo - 2

            # force positive and min values
            # The uncertainty can't be smaller than Poissonian error.
            # However, we are approximating the distribution with an
            # asymmetric Gaussian which might result in slightly different
            # sigmas at low values.
            # We will limit Gaussian sigma to a minimum value of 90% ofthe
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
            dom_charges_sigma = tf.sqrt(tf.clip_by_value(
                dom_charges,
                0.0001,
                float('inf'))) * sigma_scale

            # set threshold under which a Poisson Likelihood is used
            charge_threshold = 5

            # Apply Asymmetric Gaussian and/or Poisson Likelihood
            # shape: [n_batch, 86, 60, 1]
            eps = 1e-7
            dom_charges_llh = tf.where(
                dom_charges_true > charge_threshold,
                tf.math.log(basis_functions.tf_asymmetric_gauss(
                    x=dom_charges_true,
                    mu=dom_charges,
                    sigma=dom_charges_sigma,
                    r=dom_charges_r,
                ) + eps),
                dom_charges_true * tf.math.log(dom_charges + eps) - dom_charges
            )

            # compute (Gaussian) uncertainty on predicted dom charge
            dom_charges_unc = tf.where(
                dom_charges_true > charge_threshold,
                # take mean of left and right side uncertainty
                # Note: this might not be correct
                dom_charges_sigma*((1 + dom_charges_r)/2.),
                tf.math.sqrt(dom_charges + eps)
            )

            print('dom_charges_sigma', dom_charges_sigma)
            print('dom_charges_llh', dom_charges_llh)
            print('dom_charges_unc', dom_charges_unc)

            # add tensors to tensor dictionary
            tensor_dict['dom_charges_sigma'] = dom_charges_sigma
            tensor_dict['dom_charges_r'] = dom_charges_r
            tensor_dict['dom_charges_unc'] = dom_charges_unc
            tensor_dict['dom_charges_variance'] = dom_charges_unc**2
            tensor_dict['dom_charges_log_pdf_values'] = dom_charges_llh

        elif config['estimate_charge_distribution'] == 'negative_binomial':
            """
            Use negative binomial PDF instead of Poisson to account for
            over-dispersion induces by systematic variations.

            The parameterization chosen here is defined by the mean mu and
            the over-dispersion factor alpha.

                Var(x) = mu + alpha*mu**2

            Alpha must be greater than zero.
            """
            alpha_trafo = tf.expand_dims(
                conv_hex3d_layers[-1][..., 1], axis=-1)

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
                dom_charges + dom_charges_alpha*dom_charges**2)
            dom_charges_unc = tf.sqrt(dom_charges_variance)

            print('dom_charges_llh', dom_charges_llh)

            # add tensors to tensor dictionary
            tensor_dict['dom_charges_alpha'] = dom_charges_alpha
            tensor_dict['dom_charges_unc'] = dom_charges_unc
            tensor_dict['dom_charges_variance'] = dom_charges_variance
            tensor_dict['dom_charges_log_pdf_values'] = dom_charges_llh

        else:
            # Poisson Distribution: variance is equal to expected charge
            tensor_dict['dom_charges_unc'] = tf.sqrt(dom_charges)
            tensor_dict['dom_charges_variance'] = dom_charges

        # ----------------------------------------------------------
        # Fully Connected Layers to for p(t_i | q_i, D_i) calulation
        # ----------------------------------------------------------

        # get latent dimension prior to p(t_i | q_i, D_i) calulation
        # shape: [-1, 86, 60, n_latent]
        latend_vars = conv_hex3d_layers[-1][..., n_charge:]

        # shape: [n_pulses, n_latent]
        pulse_latent_vars = tf.gather_nd(latend_vars, pulses_ids)

        # Total charge D_i of DOM. Shape: [n_pulses, 1]
        pulse_dom_charge_true = tf.gather_nd(dom_charges_true, pulses_ids)
        pulse_dom_charge = tf.gather_nd(dom_charges, pulses_ids)

        # expanded pulse_quantiles and charges to shape: [n_pulses, 1]
        exp_pulse_quantiles = tf.expand_dims(pulse_quantiles, axis=-1)
        exp_pulse_charges = tf.expand_dims(pulse_charges, axis=-1)

        # transform measured quantiles and total charge
        pulse_dom_charge_true_trafo = 0.1*tf.math.log(pulse_dom_charge_true+1.)
        pulse_dom_charge_trafo = 0.1*tf.math.log(pulse_dom_charge + 1.)

        # put all information together.
        input_list = [
            # ln(quantiles q_i) --> early times are important
            tf.math.log(exp_pulse_quantiles),
            # total (true) DOM charge D_i
            pulse_dom_charge_true_trafo,
            # rel. charge fraction of pulses wrt total DOM charge
            # This should help to estimate how accurate the quantile is
            tf.math.log(exp_pulse_charges / pulse_dom_charge_true),
            pulse_latent_vars,
        ]
        if config['add_predicted_charge_to_latent_vars']:
            # total (predicted) DOM charge D_i
            input_list.append(pulse_dom_charge_trafo)
            # charge fraction of the pulse relative to predicted DOM charge
            input_list.append(
                tf.math.log(exp_pulse_charges / pulse_dom_charge))

        # Shape: [n_pulses, n_latent + (2 or 3)]
        quantile_pdf_input = tf.concat(input_list, axis=-1)
        print('quantile_pdf_input', quantile_pdf_input)

        # now apply fully connected layers to compute p(t | q_i, D_i)
        fc_layers = self._untracked_data['fully_connected_layer'](
            quantile_pdf_input,
            is_training=is_training,
            keep_prob=config['keep_prob'],
        )

        # -------------------------------------------
        # Get times at which to evaluate DOM PDF
        # -------------------------------------------

        # offset PDF evaluation times with cascade vertex time
        t_pdf = pulse_times - tf.gather(parameters[:, 6],
                                        indices=pulse_batch_id)
        # new shape: [None, 1]
        t_pdf = tf.expand_dims(t_pdf, axis=-1)
        t_pdf = tf.ensure_shape(t_pdf, [None, 1])

        # get light propagation time of unscattered light for each pulse
        # shape: [n_pulses, 1]
        pulse_light_propagation_time = tf.gather_nd(light_propagation_time,
                                                    pulses_ids)
        print('pulse_light_propagation_time', pulse_light_propagation_time)

        # scale time range down to avoid big numbers:
        t_scale = 0.001  # unit: 1./ns --> time units in us
        average_t_dist = 1000. * t_scale
        t_pdf = t_pdf * t_scale
        pulse_light_propagation_time *= t_scale

        # -------------------------------------------
        # Gather latent vars of mixture model
        # -------------------------------------------
        # check if we have the right amount of filters in the latent dimension
        n_models = config['num_latent_models']
        if n_models*4 != config['fc_num_filters_list'][-1]:
            raise ValueError('{!r} != {!r}'.format(
                n_models*4, config['fc_num_filters_list'][-1]))
        if n_models < 1:
            raise ValueError('{!r} < 1'.format(n_models))

        out_layer = fc_layers[-1]

        # shape: [n_pulses, n_models]
        latent_mu_offset = out_layer[..., 0*n_models:1*n_models]
        latent_sigma = out_layer[..., 1*n_models:2*n_models]
        latent_r = out_layer[..., 2*n_models:3*n_models]
        latent_scale = out_layer[..., 3*n_models:4*n_models]

        # add reasonable scaling for parameters assuming the latent vars
        # are distributed normally around zero
        factor_sigma = 1.0  # units: 1/t_scale
        factor_mu = 1.0  # units: 1/t_scale
        factor_r = 0.001
        factor_scale = 1.0

        # create correct offset and scaling
        # latent_mu = average_t_dist + factor_mu * latent_mu
        latent_mu = pulse_light_propagation_time + factor_mu * latent_mu_offset
        latent_sigma = factor_sigma * latent_sigma
        latent_r = factor_r * latent_r
        latent_scale = 1 + factor_scale * latent_scale

        # force positive and min values
        latent_scale = tf.nn.elu(latent_scale) + 1.00001
        latent_r = tf.nn.elu(latent_r) + 1.001
        latent_sigma = tf.nn.elu(latent_sigma) + 1.001

        # normalize scale to sum to 1
        latent_scale /= tf.reduce_sum(latent_scale, axis=-1, keepdims=True)

        tensor_dict['quantile_latent_var_mu'] = latent_mu
        tensor_dict['quantile_latent_var_sigma'] = latent_sigma
        tensor_dict['quantile_latent_var_r'] = latent_r
        tensor_dict['quantile_latent_var_scale'] = latent_scale

        # ensure shapes
        pulse_latent_mu = tf.ensure_shape(latent_mu, [None, n_models])
        pulse_latent_sigma = tf.ensure_shape(latent_sigma, [None, n_models])
        pulse_latent_r = tf.ensure_shape(latent_r, [None, n_models])
        pulse_latent_scale = tf.ensure_shape(latent_scale, [None, n_models])

        mu_offset = factor_mu * latent_mu_offset / t_scale
        mask = tf.ones_like(mu_offset)*exp_pulse_quantiles < 0.5
        tf.print('pulse_latent_mu offset',
                 tf.reduce_min(mu_offset[mask]),
                 tf.reduce_mean(mu_offset[mask]),
                 tf.reduce_max(mu_offset[mask]),
                 )
        tf.print('pulse_latent_sigma',
                 tf.reduce_min(pulse_latent_sigma[mask]),
                 tf.reduce_mean(pulse_latent_sigma[mask]),
                 tf.reduce_max(pulse_latent_sigma[mask]),
                 )
        tf.print('pulse_latent_r',
                 tf.reduce_min(pulse_latent_r[mask]),
                 tf.reduce_mean(pulse_latent_r[mask]),
                 tf.reduce_max(pulse_latent_r[mask]),
                 )
        tf.print('pulse_latent_scale',
                 tf.reduce_min(pulse_latent_scale[mask]),
                 tf.reduce_mean(pulse_latent_scale[mask]),
                 tf.reduce_max(pulse_latent_scale[mask]),
                 )
        # -------------------------------------------
        # Apply Asymmetric Gaussian Mixture Model
        # -------------------------------------------

        # [n_pulses, 1] * [n_pulses, n_models] = [n_pulses, n_models]
        quantile_pdf_values = basis_functions.tf_asymmetric_gauss(
                    x=t_pdf, mu=pulse_latent_mu, sigma=pulse_latent_sigma,
                    r=pulse_latent_r) * pulse_latent_scale

        # new shape: [n_pulses]
        quantile_pdf_values = tf.reduce_sum(quantile_pdf_values, axis=-1)

        tensor_dict['pulse_quantile_pdf'] = quantile_pdf_values
        # -------------------------------------------

        return tensor_dict
