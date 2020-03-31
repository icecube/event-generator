from __future__ import division, print_function
import logging
import tensorflow as tf
import numpy as np

from tfscripts import layers as tfs
from tfscripts.weights import new_weights

from egenerator.model.source.base import Source
from egenerator.utils import detector, basis_functions, angles
# from egenerator.manager.component import Configuration, BaseComponent


class DefaultCascadeModel(Source):

    def __init__(self, logger=None):
        """Instantiate Source class

        Parameters
        ----------
        logger : logging.logger, optional
            The logger to use.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(DefaultCascadeModel, self).__init__(logger=self._logger)

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
                'pulse_pdf': The likelihood evaluated for each pulse
                             Shape: [-1]
        """
        self.assert_configured(True)

        tensor_dict = {}

        config = self.configuration.config['config']
        parameters = data_batch_dict[parameter_tensor_name]
        pulses = data_batch_dict['x_pulses']
        pulses_ids = data_batch_dict['x_pulses_ids']

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

        distance = tf.sqrt(dx**2 + dy**2 + dz**2)

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

        # the result of the convolution layers are the latent variables
        dom_charges_trafo = tf.expand_dims(conv_hex3d_layers[-1][..., 0],
                                           axis=-1)

        # -------------------------------------------
        # Get expected charge at DOM
        # -------------------------------------------
        # apply exponential which also forces positive values
        dom_charges = tf.exp(dom_charges_trafo)

        # scale charges by cascade energy
        if config['scale_charge']:
            scale_factor = tf.expand_dims(parameter_list[5], axis=-1) / 10000.0
            dom_charges *= scale_factor

        tensor_dict['dom_charges'] = dom_charges

        # -------------------------------------------
        # Get times at which to evaluate DOM PDF
        # -------------------------------------------

        # offset PDF evaluation times with cascade vertex time
        t_pdf = pulse_times - tf.gather(parameters[:, 6],
                                        indices=pulse_batch_id)
        # new shape: [None, 1]
        t_pdf = tf.expand_dims(t_pdf, axis=-1)
        t_pdf = tf.ensure_shape(t_pdf, [None, 1])

        # scale time range down to avoid big numbers:
        t_scale = 0.001  # 1./ns
        average_t_dist = 1000. * t_scale
        t_pdf = t_pdf * t_scale

        # -------------------------------------------
        # Gather latent vars of mixture model
        # -------------------------------------------
        # check if we have the right amount of filters in the latent dimension
        n_models = config['num_latent_models']
        if n_models*4 + 1 != config['num_filters_list'][-1]:
            raise ValueError('{!r} != {!r}'.format(
                n_models*4 + 1, config['num_filters_list'][-1]))
        if n_models <= 1:
            raise ValueError('{!r} !> 1'.format(n_models))

        out_layer = conv_hex3d_layers[-1]
        latent_mu = out_layer[..., 1 + 0*n_models: 1 + 1*n_models]
        latent_sigma = out_layer[..., 1 + 1*n_models: 1 + 2*n_models]
        latent_r = out_layer[..., 1 + 2*n_models: 1 + 3*n_models]
        latent_scale = out_layer[..., 1 + 3*n_models: 1 + 4*n_models]

        # add reasonable scaling for parameters assuming the latent vars
        # are distributed normally around zero
        factor_sigma = 1.  # ns
        factor_mu = 1.  # ns
        factor_r = 1.
        factor_scale = 1.

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

        tensor_dict['latent_var_mu'] = latent_mu
        tensor_dict['latent_var_sigma'] = latent_sigma
        tensor_dict['latent_var_r'] = latent_r
        tensor_dict['latent_var_scale'] = latent_scale

        # get latent vars for each pulse
        pulse_latent_mu = tf.gather_nd(latent_mu, pulses_ids)
        pulse_latent_sigma = tf.gather_nd(latent_sigma, pulses_ids)
        pulse_latent_r = tf.gather_nd(latent_r, pulses_ids)
        pulse_latent_scale = tf.gather_nd(latent_scale, pulses_ids)

        # ensure shapes
        pulse_latent_mu = tf.ensure_shape(pulse_latent_mu, [None, n_models])
        pulse_latent_sigma = tf.ensure_shape(pulse_latent_sigma,
                                             [None, n_models])
        pulse_latent_r = tf.ensure_shape(pulse_latent_r, [None, n_models])
        pulse_latent_scale = tf.ensure_shape(pulse_latent_scale,
                                             [None, n_models])

        print('latent_mu', latent_mu)
        print('pulse_latent_mu', pulse_latent_mu)
        print('latent_scale', latent_scale)
        print('pulse_latent_scale', pulse_latent_scale)

        # -------------------------------------------
        # Apply Asymmetric Gaussian Mixture Model
        # -------------------------------------------

        # [n_pulses, 1] * [n_pulses, n_models] = [n_pulses, n_models]
        pulse_pdf_values = basis_functions.tf_asymmetric_gauss(
                    x=t_pdf, mu=pulse_latent_mu, sigma=pulse_latent_sigma,
                    r=pulse_latent_r) * pulse_latent_scale

        # new shape: [n_pulses]
        pulse_pdf_values = tf.reduce_sum(pulse_pdf_values, axis=-1)
        print('pulse_pdf_values', pulse_pdf_values)

        tensor_dict['pulse_pdf'] = pulse_pdf_values
        # -------------------------------------------

        return tensor_dict
