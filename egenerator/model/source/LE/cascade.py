from __future__ import division, print_function
import logging
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from tfscripts import layers as tfs
from tfscripts.weights import new_weights
from egenerator import misc

from egenerator.model.source.base import Source
from egenerator.utils import detector, basis_functions, angles
from egenerator.utils.optical_module.DetectorInfo import DetectorInfoModule
# from egenerator.manager.component import Configuration, BaseComponent


class CascadeLEModel(Source):

    def __init__(self, logger=None):
        """Instantiate Source class

        Parameters
        ----------
        logger : logging.logger, optional
            The logger to use.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(CascadeLEModel, self).__init__(logger=self._logger)

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
          

        # backwards compatibility for models that didn't define precision
        if 'float_precision' in config:
            float_precision = config['float_precision']
        else:
            float_precision = 'float32'
    
        # Define optical module to use
        optical_module = DetectorInfoModule(config['optical_module_key'])
        self.optical_module = optical_module
        num_strings=optical_module.num_strings
        doms_per_string=optical_module.doms_per_string   
        num_pmts = optical_module.num_pmts
    
        # ---------------------------------------------
        # Define input parameters of hypothesis
        # ---------------------------------------------
        parameter_names = ['x', 'y', 'z', 'zenith', 'azimuth', 'cscd_energy',
                           'time', 'trck_energy']
        if 'additional_label_names' in config:
            parameter_names += config['additional_label_names']
            num_add_labels = len(config['additional_label_names'])
        else:
            num_add_labels = 0

        num_inputs = 4 + num_add_labels

        if config['add_opening_angle']:
            num_inputs += 1

        if config['add_dom_coordinates']:
            num_inputs += 3
        
        if config['num_local_vars'] > 0:
            self._untracked_data['local_vars'] = new_weights(
                    shape=[1]+input_shape[1:-1]+[config['num_local_vars']],
                    float_precision=float_precision,
                    name='local_dom_input_variables')
            num_inputs += config['num_local_vars']
            
        if optical_module.dom_name == 'ICU':
            pmt_flat_idx = optical_module.pmt_flat_idx
            input_shape = [-1, len(pmt_flat_idx.keys()), num_inputs]
            s_list = [1, 1, 1]
        else:
            input_shape=[-1, num_strings, doms_per_string*num_pmts, num_inputs]
            s_list = [1, 1, 1, 1]

        # -------------------------------------------
        # convolutional hex3d layers over X_IC86 data
        # -------------------------------------------
        self._untracked_data['conv_hex3d_layer'] = tfs.ConvNdLayers(
            input_shape=input_shape,
            filter_size_list=config['filter_size_list'],
            num_filters_list=config['num_filters_list'],
            pooling_type_list=None,
            pooling_strides_list=s_list,
            pooling_ksize_list=s_list,
            use_dropout_list=config['use_dropout_list'],
            padding_list='SAME',
            strides_list=s_list,
            use_batch_normalisation_list=config['use_batch_norm_list'],
            activation_list=config['activation_list'],
            use_residual_list=config['use_residual_list'],
            hex_zero_out_list=False,
            dilation_rate_list=None,
            hex_num_rotations_list=1,
            method_list=config['method_list'],
            float_precision=float_precision,
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

        print('Applying Default LE Model...')
        tensor_dict = {}

        config = self.configuration.config['config']
        parameters = data_batch_dict[parameter_tensor_name]
        pulses = data_batch_dict['x_pulses']
        pulses_ids = data_batch_dict['x_pulses_ids']
  
        # Define optical module to use
        optical_module = self.optical_module

        num_strings = optical_module.num_strings
        doms_per_string = optical_module.doms_per_string  
        num_pmts = optical_module.num_pmts    
        coordinates_std = optical_module.coordinates_std
        coordinates_mean = optical_module.coordinates_mean    
        dom_coordinates = optical_module.dom_coordinates   
        rel_dom_eff = optical_module.dom_rel_eff
        dom_azimuths = optical_module.dom_azimuths
        dom_zeniths = optical_module.dom_zeniths
        if optical_module.dom_areas is None:
            dom_areas = np.ones(optical_module.dom_zeniths.shape) * 0.0444
        else:
            dom_areas = optical_module.dom_areas
            
        if optical_module.dom_name == 'ICU':
            pmt_flat_idx = optical_module.pmt_flat_idx
            input_shape = [1, len(pmt_flat_idx.keys()), 1]
        else:
            input_shape = [1, num_strings, doms_per_string*num_pmts, 1]
        ndim = len(input_shape) - 2


        tensors = self.data_trafo.data['tensors']
        if ('x_time_exclusions' in tensors.names and
                tensors.list[tensors.get_index('x_time_exclusions')].exists):
            time_exclusions_exist = True
            x_time_exclusions = data_batch_dict['x_time_exclusions']
            x_time_exclusions_ids = data_batch_dict['x_time_exclusions_ids']
        else:
            time_exclusions_exist = False
        print('\t Applying time exclusions:', time_exclusions_exist)

        # get parameters tensor dtype
        param_dtype_np = getattr(np, tensors[parameter_tensor_name].dtype)

        # shape: [n_batch, 86, 60, 1]
        dom_charges_true = data_batch_dict['x_dom_charge']

        pulse_times = pulses[:, 1]
        pulse_charges = pulses[:, 0]
        pulse_batch_id = pulses_ids[:, 0]

        # get transformed parameters
        parameters_trafo = self.data_trafo.transform(
                                parameters, tensor_name=parameter_tensor_name)

        num_features = parameters.get_shape().as_list()[-1]
        
        # -----------------------------------
        # Calculate input values for DOMs
        # -----------------------------------

        params_reshaped = tf.reshape(parameters, [-1]+[1]*ndim+[num_features])

        # parameters: x, y, z, zenith, azimuth, cscd_energy, time, trck_energy, trck_length
        parameter_list = tf.unstack(params_reshaped, axis=-1)

        # calculate displacement vector
        dx = dom_coordinates[..., 0] - parameter_list[0]
        dy = dom_coordinates[..., 1] - parameter_list[1]
        dz = dom_coordinates[..., 2] - parameter_list[2]
        dx = tf.expand_dims(dx, axis=-1)
        dy = tf.expand_dims(dy, axis=-1)
        dz = tf.expand_dims(dz, axis=-1)

        distance = tf.sqrt(dx**2 + dy**2 + dz**2)

        # calculate observation angle
        dx_normed = dx / distance
        dy_normed = dy / distance
        dz_normed = dz / distance

        # calculate direction vector of cascade
        zenith = parameter_list[3]
        azimuth = parameter_list[4]
        dir_x = -tf.sin(zenith) * tf.cos(azimuth)
        dir_y = -tf.sin(zenith) * tf.sin(azimuth)
        dir_z = -tf.cos(zenith)
       
        # calculate direction of pmt orientations
        dom_azimuths_tf = tf.convert_to_tensor(dom_azimuths, dtype=tf.float32)
        dom_zeniths_tf = tf.convert_to_tensor(dom_zeniths, dtype=tf.float32)
        pmt_orientations = angles.sph_to_cart_tf(azimuth=dom_azimuths_tf, zenith=dom_zeniths_tf)
        
        # calculate opening angle of displacement vector and pmt orientation
        orientation_angle = angles.get_angle(pmt_orientations,
                                             tf.concat([dx_normed,
                                                        dy_normed,
                                                        dz_normed], axis=-1)
                                             )
        orientation_angle = tf.expand_dims(tf.cos(orientation_angle), axis=-1)

        # calculate opening angle of displacement vector and cascade direction
        opening_angle = angles.get_angle(tf.stack([dir_x, 
                                                   dir_y, 
                                                   dir_z], axis=-1),
                                         tf.concat([dx_normed,
                                                    dy_normed,
                                                    dz_normed], axis=-1)
                                         )
        opening_angle = tf.expand_dims(tf.cos(opening_angle), axis=-1)

        # transform dx, dy, dz, distance, zenith, azimuth to correct scale
        params_mean = self.data_trafo.data[parameter_tensor_name+'_mean']
        params_std = self.data_trafo.data[parameter_tensor_name+'_std']
        norm_const = self.data_trafo.data['norm_constant']

        distance /= (np.linalg.norm(params_std[0:3]) + norm_const)
        p_dens = tf.clip_by_value(1/distance**2, 0, 10)
        
        orientation_angle_traf = ((orientation_angle + 1) / 2)
        opening_angle_traf = ((opening_angle + 1) / 2)

        x_parameters_expanded = tf.unstack(tf.reshape(
                                                parameters_trafo,
                                                [-1]+[1]*ndim+[num_features]),
                                           axis=-1)

        #modified_parameters = tf.stack(x_parameters_expanded[:3] #pos
        #                               + [dir_x,
        #                                  dir_y,
        #                                  dir_z],
        #                               + [x_parameters_expanded[5]], #cscd energy
        #                               axis=-1)
        modified_parameters = tf.stack([x_parameters_expanded[5]], #cscd energy
                                       axis=-1)

        # put everything together
        params_expanded = tf.tile(modified_parameters, input_shape)
       
        input_list = [params_expanded, #dx_normed, dy_normed, dz_normed,
                      distance, orientation_angle_traf, p_dens]
       
        if config['add_opening_angle']:
            input_list.append(opening_angle_traf)

        if config['add_dom_coordinates']:

            # transform coordinates to correct scale with mean 0 std dev 1
            dom_coords = np.expand_dims(
                dom_coordinates.astype(param_dtype_np), axis=0)
            # scale of coordinates is ~-500m to ~500m with std dev of ~ 284m
            dom_coords -= coordinates_mean
            dom_coords /= coordinates_std

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
        # Get times at which to evaluate DOM PDF
        # -------------------------------------------
        
        # offset PDF evaluation times with cascade vertex time
        tensor_dict['time_offsets'] = parameters[:, 6]
        t_pdf = pulse_times - tf.gather(parameters[:, 6],
                                        indices=pulse_batch_id)
        if time_exclusions_exist:
            # offset time exclusions

            # shape: [n_events]
            tw_cascade_t = tf.gather(
                parameters[:, 6], indices=x_time_exclusions_ids[:, 0])

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
        t_scale = 1. / self.time_unit_in_ns  # [1./ns]
        t_pdf = t_pdf * t_scale
        if time_exclusions_exist:
            t_exclusions = t_exclusions * t_scale

        # -------------------------------------------
        # Gather latent vars of mixture model
        # -------------------------------------------
        if config['estimate_charge_distribution'] is True:
            n_charge = 3
        elif config['estimate_charge_distribution'] == 'negative_binomial':
            n_charge = 2
        else:
            n_charge = 1

        # check if we have the right amount of filters in the latent dimension
        n_models = config['num_latent_models']
        if n_models*4 + n_charge != config['num_filters_list'][-1]:
            raise ValueError('{!r} != {!r}'.format(
                n_models*4 + n_charge, config['num_filters_list'][-1]))
        if n_models <= 1:
            raise ValueError('{!r} !> 1'.format(n_models))

        print('\t Charge method:', config['estimate_charge_distribution'])
        print('\t Number of Asymmetric Gaussian Components:', n_models)

        out_layer = conv_hex3d_layers[-1]
        latent_mu = out_layer[...,
                              n_charge + 0*n_models:n_charge + 1*n_models]
        latent_sigma = out_layer[...,
                                 n_charge + 1*n_models:n_charge + 2*n_models]
        latent_r = out_layer[..., n_charge + 2*n_models:n_charge + 3*n_models]
        latent_scale = out_layer[...,
                                 n_charge + 3*n_models:n_charge + 4*n_models]

        # add reasonable scaling for parameters assuming the latent vars
        # are distributed normally around zero
        factor_sigma = 1. #0.1
        factor_mu = 1.
        factor_r = 1.
        factor_scale = 1.

        # create correct offset and scaling
        latent_mu = 0.7 + factor_mu * latent_mu #0.2 + factor_mu * latent_mu
        latent_sigma = 2 + factor_sigma * latent_sigma #0.08 + factor_sigma * latent_sigma
        latent_r = 1 + factor_r * latent_r #7 + factor_r * latent_r
        latent_scale = 1 + factor_scale * latent_scale

        # force positive and min values
        #latent_mu = tf.nn.elu(latent_mu - 2.) + 1.
        latent_scale = tf.nn.elu(latent_scale) + 1.00001
        latent_r = tf.nn.elu(latent_r) + 1.001 #tf.nn.elu(latent_r - 2.) + 2.
        latent_sigma = tf.nn.elu(latent_sigma) + 1.001 #tf.nn.elu(latent_sigma - 3.) + 1.0001

        # normalize scale to sum to 1
        latent_scale /= tf.reduce_sum(latent_scale, axis=-1, keepdims=True)

        # Sort mixture model components in time if desired
        if ('prevent_mixture_component_swapping' in config and
                config['prevent_mixture_component_swapping']):

            # swap latent variables of components, such that these are ordered
            # in time. This puts a constrained on the model and reduces
            # the permutation options and should thus facilitate training.
            # We could keep the latent_mu in place and sort the other
            # components accordingly. An equivalent alternative is to keep
            # the other components in place and to simply sort the latent_mu.
            latent_mu = tf.ensure_shape(
                tf.sort(latent_mu, axis=-1), shape=[None]+input_shape[1:-1]+[n_models])

        tensor_dict['latent_var_mu'] = latent_mu
        tensor_dict['latent_var_sigma'] = latent_sigma
        tensor_dict['latent_var_r'] = latent_r
        tensor_dict['latent_var_scale'] = latent_scale

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
                tw_latent_sigma, [None, n_models])
            tw_latent_r = tf.ensure_shape(tw_latent_r, [None, n_models])

            # [n_tw, 1] * [n_tw, n_models] = [n_tw, n_models]
            tw_cdf_start = basis_functions.tf_asymmetric_gauss_cdf(
                x=t_exclusions[:, 0], mu=tw_latent_mu, sigma=tw_latent_sigma,
                r=tw_latent_r)
            tw_cdf_stop = basis_functions.tf_asymmetric_gauss_cdf(
                x=t_exclusions[:, 1], mu=tw_latent_mu, sigma=tw_latent_sigma,
                r=tw_latent_r)

            tw_cdf_exclusion = tw_cdf_stop - tw_cdf_start

            # some safety checks to make sure we aren't clipping too much
            asserts = []
            asserts.append(tf.debugging.Assert(
                tf.reduce_all(tf.math.logical_or(
                    tw_cdf_exclusion > -1e-4,
                    ~tf.math.is_finite(tw_cdf_exclusion)
                )),
                ['Cascade TW CDF < 0!', tf.reduce_min(tw_cdf_exclusion)],
            ))
            asserts.append(tf.debugging.Assert(
                tf.reduce_all(tf.math.logical_or(
                    tw_cdf_exclusion < 1.0001,
                    ~tf.math.is_finite(tw_cdf_exclusion)
                )),
                ['Cascade TW CDF > 1!', tf.reduce_max(tw_cdf_exclusion)],
            ))
            with tf.control_dependencies(asserts):
                tw_cdf_exclusion = tfp.math.clip_by_value_preserve_gradient(
                    tw_cdf_exclusion, 0., 1.)

            # accumulate time window exclusions for each DOM and MM component
            # shape: [None, 86, 60, n_models]
            dom_cdf_exclusion = tf.zeros_like(latent_mu)

            dom_cdf_exclusion = tf.tensor_scatter_nd_add(
                dom_cdf_exclusion,
                indices=x_time_exclusions_ids,
                updates=tw_cdf_exclusion,
            )
            # limit to range [0., 1.]
            # Note: we loose gradients if clipping is applied. Maybe
            # tfp.math.clip_value_value_preserve_gradient is a better idea?
            # these values should be close to 0 and 1, keeping gradients
            # for values slightly outside should therefore be more correct
            # add safety checks to make sure we aren't clipping too much
            asserts = []
            asserts.append(tf.debugging.Assert(
                tf.reduce_all(tf.math.logical_or(
                    dom_cdf_exclusion > -1e-4,
                    ~tf.math.is_finite(dom_cdf_exclusion)
                )),
                ['Cascade DOM CDF < 0!', tf.reduce_min(dom_cdf_exclusion)],
            ))
            asserts.append(tf.debugging.Assert(
                tf.reduce_all(tf.math.logical_or(
                    dom_cdf_exclusion < 1.0001,
                    ~tf.math.is_finite(dom_cdf_exclusion)
                )),
                ['Cascade DOM CDF > 1!', tf.reduce_max(dom_cdf_exclusion)],
            ))
            with tf.control_dependencies(asserts):
                dom_cdf_exclusion = tfp.math.clip_by_value_preserve_gradient(
                    dom_cdf_exclusion, 0., 1.)

            # Shape: [None, 86, 60, 1]
            dom_cdf_exclusion_sum = tf.reduce_sum(
                dom_cdf_exclusion * latent_scale, axis=-1, keepdims=True)

            # add safety checks to make sure we aren't clipping too much
            asserts = []
            asserts.append(tf.debugging.Assert(
                tf.reduce_all(tf.math.logical_or(
                    dom_cdf_exclusion_sum > -1e-4,
                    ~tf.math.is_finite(dom_cdf_exclusion_sum)
                )),
                ['Cascade DOM CDF sum < 0!',
                 tf.reduce_min(dom_cdf_exclusion_sum)],
            ))
            asserts.append(tf.debugging.Assert(
                tf.reduce_all(tf.math.logical_or(
                    dom_cdf_exclusion_sum < 1.0001,
                    ~tf.math.is_finite(dom_cdf_exclusion_sum)
                )),
                ['Cascade DOM CDF sum > 1!',
                 tf.reduce_max(dom_cdf_exclusion_sum)],
            ))
            with tf.control_dependencies(asserts):
                dom_cdf_exclusion_sum = (
                    tfp.math.clip_by_value_preserve_gradient(
                        dom_cdf_exclusion_sum, 0., 1.)
                )

            tensor_dict['dom_cdf_exclusion'] = dom_cdf_exclusion
            tensor_dict['dom_cdf_exclusion_sum'] = dom_cdf_exclusion_sum

        # -------------------------------------------
        # Get expected charge at DOM
        # -------------------------------------------

        # the result of the convolution layers are the latent variables
        dom_charges_trafo = tf.expand_dims(conv_hex3d_layers[-1][..., 0],
                                           axis=-1)

        # clip value range for more stability during training
        dom_charges_trafo = tf.clip_by_value(dom_charges_trafo, -20., 15)

        # apply exponential which also forces positive values
        dom_charges = tf.exp(dom_charges_trafo)
        
        # scale charges by DOM area
        dom_charges *= tf.expand_dims(dom_areas.astype(param_dtype_np)/0.0444, axis=-1)

        # scale charges by energy
        if config['scale_charge']:
            # make sure energy does not turn negative
            energy = tf.clip_by_value(parameter_list[5], 0., float('inf'))
            scale_factor = tf.expand_dims(energy, axis=-1) / 10.0
            dom_charges *= scale_factor

        # scale charges by realtive DOM efficiency
        if config['scale_charge_by_relative_dom_efficiency']:
            dom_charges *= tf.expand_dims(
                rel_dom_eff.astype(param_dtype_np), axis=-1)

        # scale charges by global DOM efficiency
        if config['scale_charge_by_global_dom_efficiency']:
            dom_charges *= tf.expand_dims(
                parameter_list[self.get_index('DOMEfficiency')], axis=-1)

        # apply time window exclusions if needed
        if time_exclusions_exist:
            dom_charges = dom_charges * (1. - dom_cdf_exclusion_sum + 1e-3)

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
            dom_charges_sigma = tf.sqrt(
                tfp.math.clip_by_value_preserve_gradient(
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

            # add tensors to tensor dictionary
            tensor_dict['dom_charges_alpha'] = dom_charges_alpha
            tensor_dict['dom_charges_unc'] = dom_charges_unc
            tensor_dict['dom_charges_variance'] = dom_charges_variance
            tensor_dict['dom_charges_log_pdf_values'] = dom_charges_llh

        else:
            # Poisson Distribution: variance is equal to expected charge
            tensor_dict['dom_charges_unc'] = tf.sqrt(dom_charges)
            tensor_dict['dom_charges_variance'] = dom_charges

        # --------------------------
        # Calculate Pulse PDF Values
        # --------------------------

        # get latent vars for each pulse
        pulse_latent_mu = tf.gather_nd(latent_mu, pulses_ids)
        pulse_latent_sigma = tf.gather_nd(latent_sigma, pulses_ids)
        pulse_latent_r = tf.gather_nd(latent_r, pulses_ids)
        pulse_latent_scale = tf.gather_nd(latent_scale, pulses_ids)

        # scale up pulse pdf by time exclusions if needed
        if time_exclusions_exist:
            pulse_cdf_exclusion = tf.gather_nd(dom_cdf_exclusion, pulses_ids)
            pulse_latent_scale /= (1. - pulse_cdf_exclusion + 1e-3)

        # ensure shapes
        pulse_latent_mu = tf.ensure_shape(pulse_latent_mu, [None, n_models])
        pulse_latent_sigma = tf.ensure_shape(pulse_latent_sigma,
                                             [None, n_models])
        pulse_latent_r = tf.ensure_shape(pulse_latent_r, [None, n_models])
        pulse_latent_scale = tf.ensure_shape(pulse_latent_scale,
                                             [None, n_models])

        # -------------------------------------------
        # Apply Asymmetric Gaussian Mixture Model
        # -------------------------------------------

        # [n_pulses, 1] * [n_pulses, n_models] = [n_pulses, n_models]
        pulse_pdf_values = basis_functions.tf_asymmetric_gauss(
                    x=t_pdf, mu=pulse_latent_mu, sigma=pulse_latent_sigma,
                    r=pulse_latent_r) * pulse_latent_scale

        # new shape: [n_pulses]
        if is_training:
            pulse_pdf_values = tf.where(t_pdf > 0,
                                        tf.reduce_sum(pulse_pdf_values, axis=-1),
                                        1e-3
                                       )
        else:
            pulse_pdf_values = tf.reduce_sum(pulse_pdf_values, axis=-1)

        tensor_dict['pulse_pdf'] = pulse_pdf_values

        # ---------------------
        
        return tensor_dict
