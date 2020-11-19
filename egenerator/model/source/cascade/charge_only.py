from __future__ import division, print_function
import logging
import tensorflow as tf
import numpy as np

from tfscripts import layers as tfs
from tfscripts.weights import new_weights

from egenerator.model.source.base import Source
from egenerator.utils import detector, basis_functions, angles


class ChargeOnlyCascadeModel(Source):

    def __init__(self, logger=None):
        """Instantiate Source class

        Parameters
        ----------
        logger : logging.logger, optional
            The logger to use.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(ChargeOnlyCascadeModel, self).__init__(logger=self._logger)

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
        if 'additional_label_names' in config:
            parameter_names += config['additional_label_names']
            num_add_labels = len(config['additional_label_names'])
        else:
            num_add_labels = 0

        num_snowstorm_params = 0
        if 'snowstorm_parameter_names' in config:
            for param_name, num in config['snowstorm_parameter_names']:
                num_snowstorm_params += num
                for i in range(num):
                    parameter_names.append(param_name.format(i))

        num_inputs = 11 + num_add_labels + num_snowstorm_params

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

        return parameter_names

    @tf.function
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

        print('Applying Charge Only Cascade Model...')
        tensor_dict = {}

        config = self.configuration.config['config']
        parameters = data_batch_dict[parameter_tensor_name]

        # shape: [n_batch, 86, 60, 1]
        dom_charges_true = data_batch_dict['x_dom_charge']

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
        params_expanded = tf.tile(modified_parameters, [1, 86, 60, 1])

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

            print('\t dom_coords', dom_coords)
            input_list.append(dom_coords)

        if config['num_local_vars'] > 0:

            # extend to correct shape:
            local_vars = (tf.ones_like(dx_normed) *
                          self._untracked_data['local_vars'])
            print('\t local_vars', local_vars)

            input_list.append(local_vars)

        x_doms_input = tf.concat(input_list, axis=-1)
        print('\t x_doms_input', x_doms_input)

        # -------------------------------------------
        # convolutional hex3d layers over X_IC86 data
        # -------------------------------------------
        conv_hex3d_layers = self._untracked_data['conv_hex3d_layer'](
                                        x_doms_input, is_training=is_training,
                                        keep_prob=config['keep_prob'])

        # -------------------------------------------
        # Get expected charge at DOM
        # -------------------------------------------
        print('\t Charge method:', config['estimate_charge_distribution'])

        # the result of the convolution layers are the latent variables
        dom_charges_trafo = tf.expand_dims(conv_hex3d_layers[-1][..., 0],
                                           axis=-1)

        # apply exponential which also forces positive values
        dom_charges = tf.exp(dom_charges_trafo)

        # scale charges by cascade energy
        if config['scale_charge']:
            # make sure cascade energy does not turn negative
            cascade_energy = tf.clip_by_value(
                parameter_list[5], 0., float('inf'))
            scale_factor = tf.expand_dims(cascade_energy, axis=-1) / 10000.0
            dom_charges *= scale_factor

        # scale charges by relative DOM efficiency
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
        if config['estimate_charge_distribution'] == 'negative_binomial':
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

            # compute standard deviation
            # std = sqrt(var) = sqrt(mu + alpha*mu**2)
            dom_charges_variance = (
                dom_charges + dom_charges_alpha*dom_charges**2)
            dom_charges_unc = tf.sqrt(dom_charges_variance)

            # add tensors to tensor dictionary
            tensor_dict['dom_charges_alpha'] = dom_charges_alpha
            tensor_dict['dom_charges_unc'] = dom_charges_unc
            tensor_dict['dom_charges_variance'] = dom_charges_variance

        else:
            # Poisson Distribution: variance is equal to expected charge
            tensor_dict['dom_charges_unc'] = tf.sqrt(dom_charges)
            tensor_dict['dom_charges_variance'] = dom_charges

        return tensor_dict
