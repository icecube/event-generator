from __future__ import division, print_function
import logging
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from tfscripts import layers as tfs
from tfscripts.weights import new_weights

from egenerator.model.source.base import Source
from egenerator.utils import detector, basis_functions, angles
# from egenerator.manager.component import Configuration, BaseComponent
from egenerator.utils.optical_module.DetectorInfo import DetectorInfoModule

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

        # backwards compatibility for models that didn't define precision
        if 'float_precision' in config:
            float_precision = config['float_precision']
        else:
            float_precision = 'float32'

        # Define optical modules to use
        optical_module = DetectorInfoModule(config['optical_module_key1'])
        self.optical_module = optical_module
        num_strings=optical_module.num_strings
        doms_per_string=optical_module.doms_per_string   
        num_pmts = optical_module.num_pmts 

        optical_module_2 = DetectorInfoModule(config['optical_module_key2'])
        self.optical_module_2 = optical_module_2
        num_strings_2=optical_module_2.num_strings
        doms_per_string_2=optical_module_2.doms_per_string   
        num_pmts_2 = optical_module_2.num_pmts 

        # -------------------------------------------
        # Define input parameters of track hypothesis
        # -------------------------------------------
        parameter_names = ['x', 'y', 'z', 'zenith', 'azimuth',
                           'energy', 'time', 'length', 'stochasticity', 'cascade_energy', 'cascade_distance','muon_energy']
        
        # Remove cascade_distance in the future

        num_snowstorm_params = 0
        if 'snowstorm_parameter_names' in config:
            for param_name, num in config['snowstorm_parameter_names']:
                num_snowstorm_params += num
                for i in range(num):
                    parameter_names.append(param_name.format(i))

        num_inputs = 21 + num_snowstorm_params 

        if config['add_opening_angle']:
            num_inputs += 1

        if config['add_dom_coordinates']:
            num_inputs += 3

        #if config['num_local_vars'] > 0: # not in use
        #    self._untracked_data['local_vars'] = new_weights(
        #            shape=[1, 86, 60, config['num_local_vars']],
        #            float_precision=float_precision,
        #            name='local_dom_input_variables')
        #    num_inputs += config['num_local_vars']

        # -------------------------------------------
        # convolutional hex3d layers over X_IC86 data
        # -------------------------------------------
        self._untracked_data['conv_hex3d_layer'] = tfs.ConvNdLayers(
            input_shape=[-1, num_strings, doms_per_string*num_pmts, num_inputs], 
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
            float_precision=float_precision,
           # name="Gen2_conv_{}d_layer",
            )
        
        self._untracked_data['conv_hex3d_layer_icecube_2'] = tfs.ConvNdLayers( # it is important to name this *_2 for the time being
            input_shape=[-1, num_strings_2, doms_per_string_2*num_pmts_2, (num_inputs-1)],# no pmt opening angle needed
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
            float_precision=float_precision,
            name='IceCube_conv_{}d_layer',
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
        Note: this is done approximatively by the use of a softmax function
        in order to maintain continous gradients.

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
        # be irrelevant. High-energy cascades should be handled seperately.
        
        # Softplus works similarly to relu, that is max(0,value)
        distance = (
            distance
            + tf.math.softplus(-distance)
            - tf.math.softplus(distance - track_length)
        )
        return distance

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

        tensor_dict = {}

        config = self.configuration.config['config']
        parameters = data_batch_dict[parameter_tensor_name]
        pulses = data_batch_dict['x_pulses']
        pulses_ids = data_batch_dict['x_pulses_ids']

        # shape: [n_batch, num_strings, doms_per_string*num_pmts, 1]
        dom_charges_true = data_batch_dict['x_dom_charge']

        pulse_times = pulses[:, 1]
        pulse_charges = pulses[:, 0]
        pulse_batch_id = pulses_ids[:, 0]

        print('pulses', pulses)
        print('pulses_ids', pulses_ids)
        print('parameters', parameters)

        # IceCube-86 -------------------------------------------
        pulses_2 = data_batch_dict['x_pulses_2'] # important to call *_2 to all these tensors
        pulses_ids_2 = data_batch_dict['x_pulses_ids_2']

        # shape: [n_batch, 86, 60, 1]
        dom_charges_true_2 = data_batch_dict['x_dom_charge_2']

        pulse_times_2 = pulses_2[:, 1]
        pulse_charges_2 = pulses_2[:, 0]
        pulse_batch_id_2 = pulses_ids_2[:, 0]
        #-----------------------------------------------------------------

        # Define optical module to use
        try:
            optical_module = self.optical_module
        except:
            optical_module = DetectorInfoModule(config['optical_module_key'])
           
        num_strings=optical_module.num_strings
        doms_per_string=optical_module.doms_per_string  
        num_pmts = optical_module.num_pmts    
        coordinates_std=optical_module.coordinates_std
        coordinates_mean=optical_module.coordinates_mean    
        dom_coordinates=optical_module.dom_coordinates   
        rel_dom_eff=optical_module.dom_rel_eff
        dom_azimuths = optical_module.dom_azimuths
        dom_zeniths = optical_module.dom_zeniths

        # IceCube-86 -------------------------------------------
        try:
            optical_module_2 = self.optical_module_2
        except:
            optical_module_2 = DetectorInfoModule(config['optical_module_key2'])

        num_strings_2=optical_module_2.num_strings
        doms_per_string_2=optical_module_2.doms_per_string  
        num_pmts_2 = optical_module_2.num_pmts    
        coordinates_std_2=optical_module_2.coordinates_std
        coordinates_mean_2=optical_module_2.coordinates_mean    
        dom_coordinates_2=optical_module_2.dom_coordinates   
        rel_dom_eff_2=optical_module_2.dom_rel_eff
        dom_azimuths_2 = optical_module_2.dom_azimuths
        dom_zeniths_2 = optical_module_2.dom_zeniths
        #-----------------------------------------------------------------

        
        # get transformed parameters
        parameters_trafo = self.data_trafo.transform(
                                parameters, tensor_name=parameter_tensor_name)

        #parameters_trafo_2 = self.data_trafo.transform( #icecube trafo is going to be different due to different trigger
        #                        parameters, tensor_name=parameter_tensor_name+"_2")
        parameters_trafo_2 = parameters_trafo # Use different trafo if convenient using commented code above

        num_features = parameters.get_shape().as_list()[-1]

        # get parameters tensor dtype
        tensors = self.data_trafo.data['tensors']
        param_dtype_np = getattr(np, tensors[parameter_tensor_name].dtype)

        # -----------------------------------
        # Calculate input values for DOMs
        # -----------------------------------
        # Globals:
        #   track:
        #       energy, stochasticity, track length, dir_x, dir_y, dir_z, cascade_en, cascade_dist
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

        # parameters: x, y, z, zenith, azimuth, energy, time, length, stoch, cascade_energy, cascade_distance
        parameter_list = tf.unstack(params_reshaped, axis=-1)

        zenith = parameter_list[3]
        azimuth = parameter_list[4]
        energy = parameter_list[5] + 1e-1

        if is_training:
            # Ensure positive track length and energy
            assert_op_energy = tf.Assert(
                tf.greater_equal(tf.reduce_min(parameter_list[5]), -1e-3),
                [tf.reduce_min(parameter_list[5])])
            assert_op_length = tf.Assert(
                tf.greater_equal(tf.reduce_min(parameter_list[7]), -1e3),
                [tf.reduce_min(parameter_list[7])])
            with tf.control_dependencies([assert_op_energy, assert_op_length]):
                track_length = parameter_list[7] + 1.
        else:
            track_length = parameter_list[7] + 1.
        track_lstochasticity = parameter_list[8]

        cascade_energy = parameter_list[9] + 1e-7
        cascade_distance = parameter_list[10] + 1e-7

        # calculate direction vector of track
        dir_x = -tf.sin(zenith) * tf.cos(azimuth)
        dir_y = -tf.sin(zenith) * tf.sin(azimuth)
        dir_z = -tf.cos(zenith)

        #Gen2
        # vector of track vertex to DOM
        h_x = dom_coordinates[..., 0] - parameter_list[0]
        h_y = dom_coordinates[..., 1] - parameter_list[1]
        h_z = dom_coordinates[..., 2] - parameter_list[2]
        #vector director of cascade_vertex to DOM
        h_xc = dom_coordinates[..., 0] - (parameter_list[0]-parameter_list[10]*tf.sin(parameter_list[3])*tf.cos(parameter_list[4]))
        h_yc = dom_coordinates[..., 1] - (parameter_list[1]-parameter_list[10]*tf.sin(parameter_list[3])*tf.sin(parameter_list[4]))
        h_zc = dom_coordinates[..., 2] - (parameter_list[2]-parameter_list[10]*tf.cos(parameter_list[3]))
        
        distance_to_cascade_gen2 = tf.sqrt(h_xc**2+h_yc**2+h_zc**2)
        
        # IceCube-86 -------------------------------------------
        # vector of track vertex to DOM
        h_x_ice = dom_coordinates_2[..., 0] - parameter_list[0]
        h_y_ice = dom_coordinates_2[..., 1] - parameter_list[1]
        h_z_ice = dom_coordinates_2[..., 2] - parameter_list[2]
        #vector director of cascade_vertex to DOM
        h_xc_ice = dom_coordinates_2[..., 0] - (parameter_list[0]-parameter_list[10]*tf.sin(parameter_list[3])*tf.cos(parameter_list[4]))
        h_yc_ice = dom_coordinates_2[..., 1] - (parameter_list[1]-parameter_list[10]*tf.sin(parameter_list[3])*tf.sin(parameter_list[4]))
        h_zc_ice = dom_coordinates_2[..., 2] - (parameter_list[2]-parameter_list[10]*tf.cos(parameter_list[3]))
        distance_to_cascade_icecube = tf.sqrt(h_xc_ice**2+h_yc_ice**2+h_zc_ice**2)
        #-----------------------------------------------------------------

        # distance between track vertex and closest approach of infinite track
        # Shape: [-1, num_strings, doms_per_string*num_pmts]
        dist_infinite_approach = dir_x*h_x + dir_y*h_y + dir_z*h_z 
        rel_dist_infinite_approach = dist_infinite_approach / track_length

        # IceCube-86 -------------------------------------------
        dist_infinite_approach_2 = dir_x*h_x_ice + dir_y*h_y_ice + dir_z*h_z_ice
        rel_dist_infinite_approach_2 = dist_infinite_approach_2 / track_length
        #-----------------------------------------------------------------

        # this value can get extremely large if track length is ~ 0
        # Therefore: limit it to range -10, 10 and map it onto (-1, 1)
        rel_dist_infinite_approach_trafo = tf.math.tanh(
            rel_dist_infinite_approach / 10.)
        
        # IceCube-86 -------------------------------------------
        rel_dist_infinite_approach_trafo_2 = tf.math.tanh(
            rel_dist_infinite_approach_2 / 10.)
        #-----------------------------------------------------------------

        # shift distance of infinite track onto the finite track
        dist_closest_approach = self.shift_distance_on_track(
            track_length, dist_infinite_approach)
        rel_dist_closest_approach = dist_closest_approach / track_length

        # IceCube-86 -------------------------------------------
        dist_closest_approach_2 = self.shift_distance_on_track(
            track_length, dist_infinite_approach_2)
        rel_dist_closest_approach_2 = dist_closest_approach_2 / track_length
        #-----------------------------------------------------------------

        # compute delta in distance of true closest approach and of closest
        # approach of infinite track
        delta_dist_approach = dist_infinite_approach - dist_closest_approach

        # IceCube-86 -------------------------------------------------------------
        delta_dist_approach_2 = dist_infinite_approach_2 - dist_closest_approach_2
        # -------------------------------------------------------------------------

        # calculate closest approach points of track to each DOM
        closest_x = parameter_list[0] + dist_closest_approach*dir_x
        closest_y = parameter_list[1] + dist_closest_approach*dir_y
        closest_z = parameter_list[2] + dist_closest_approach*dir_z

        infinite_x = parameter_list[0] + dist_infinite_approach*dir_x
        infinite_y = parameter_list[1] + dist_infinite_approach*dir_y
        infinite_z = parameter_list[2] + dist_infinite_approach*dir_z

        # calculate displacement vectors
        dx_inf = optical_module.dom_coordinates[..., 0] - infinite_x
        dy_inf = optical_module.dom_coordinates[..., 1] - infinite_y
        dz_inf = optical_module.dom_coordinates[..., 2] - infinite_z


        # shape: [-1, num_strings, doms_per_string*num_pmts]
        distance_infinite = tf.sqrt(dx_inf**2 + dy_inf**2 + dz_inf**2)

        dx = optical_module.dom_coordinates[..., 0] - closest_x
        dy = optical_module.dom_coordinates[..., 1] - closest_y
        dz = optical_module.dom_coordinates[..., 2] - closest_z
        dx = tf.expand_dims(dx, axis=-1)
        dy = tf.expand_dims(dy, axis=-1)
        dz = tf.expand_dims(dz, axis=-1)
       
        # shape: [-1, num_strings, doms_per_string*num_pmts, 1]
        distance = tf.sqrt(dx**2 + dy**2 + dz**2) + 1e-1

        # calculate distance on track of cherenkov position
        cherenkov_angle = np.arccos(1./1.3195)
        dist_cherenkov_pos = (
            dist_infinite_approach - distance_infinite/np.tan(cherenkov_angle))

        # calculate deposited energies between points on track
        energy_before = self.get_dep_energy(
            energy, track_length,
            d1=dist_closest_approach - 100,
            d2=dist_closest_approach,
        )
        energy_after = self.get_dep_energy(
            energy, track_length,
            d1=dist_closest_approach,
            d2=dist_closest_approach + 100,
        )
        energy_cherenkov = self.get_dep_energy(
            energy, track_length,
            d1=dist_cherenkov_pos,
            d2=dist_infinite_approach,
        )

        
        dx_normed = dx / distance
        dy_normed = dy / distance
        dz_normed = dz / distance

        #dx_c_normed = h_xc / distance_to_cascade_gen2
        #dy_c_normed = h_yc / distance_to_cascade_gen2
        #dz_c_normed = h_zc / distance_to_cascade_gen2

        # IceCube-86 -------------------------------------------
        # calculate closest approach points of track to each DOM
        closest_x_ice = parameter_list[0] + dist_closest_approach_2*dir_x
        closest_y_ice = parameter_list[1] + dist_closest_approach_2*dir_y
        closest_z_ice = parameter_list[2] + dist_closest_approach_2*dir_z

        infinite_x_ice = parameter_list[0] + dist_infinite_approach_2*dir_x
        infinite_y_ice = parameter_list[1] + dist_infinite_approach_2*dir_y
        infinite_z_ice = parameter_list[2] + dist_infinite_approach_2*dir_z

        # calculate displacement vectors
        dx_inf_ice = optical_module_2.dom_coordinates[..., 0] - infinite_x_ice
        dy_inf_ice = optical_module_2.dom_coordinates[..., 1] - infinite_y_ice
        dz_inf_ice = optical_module_2.dom_coordinates[..., 2] - infinite_z_ice


        # shape: [-1, 86, 60]
        distance_infinite_ice = tf.sqrt(dx_inf_ice**2 + dy_inf_ice**2 + dz_inf_ice**2)

        dx_ice = optical_module_2.dom_coordinates[..., 0] - closest_x_ice
        dy_ice = optical_module_2.dom_coordinates[..., 1] - closest_y_ice
        dz_ice = optical_module_2.dom_coordinates[..., 2] - closest_z_ice
        dx_ice = tf.expand_dims(dx_ice, axis=-1)
        dy_ice = tf.expand_dims(dy_ice, axis=-1)
        dz_ice = tf.expand_dims(dz_ice, axis=-1)
        
        # shape: [-1, 86, 60, 1]
        distance_ice = tf.sqrt(dx_ice**2 + dy_ice**2 + dz_ice**2) + 1e-1

        # calculate distance on track of cherenkov position
        cherenkov_angle = np.arccos(1./1.3195)
        dist_cherenkov_pos_ice = (
            dist_infinite_approach_2 - distance_infinite_ice/np.tan(cherenkov_angle))

        # calculate deposited energies between points on track
        energy_before_ice = self.get_dep_energy(
            energy, track_length,
            d1=dist_closest_approach_2 - 100,
            d2=dist_closest_approach_2,
        )
        energy_after_ice = self.get_dep_energy(
            energy, track_length,
            d1=dist_closest_approach_2,
            d2=dist_closest_approach_2 + 100,
        )
        energy_cherenkov_ice = self.get_dep_energy(
            energy, track_length,
            d1=dist_cherenkov_pos_ice,
            d2=dist_infinite_approach_2,
        )

        # calculate observation angle (not really)
        dx_normed_ice = dx_ice / distance_ice
        dy_normed_ice = dy_ice / distance_ice
        dz_normed_ice = dz_ice / distance_ice
        #-----------------------------------------------------------------
        # calculate direction of pmt orientations
        dom_azimuths_tf = tf.convert_to_tensor(dom_azimuths, dtype=tf.float32) 
        dom_zeniths_tf = tf.convert_to_tensor(dom_zeniths, dtype=tf.float32)
        pmt_orientations = angles.sph_to_cart_tf(azimuth=dom_azimuths_tf,zenith=dom_zeniths_tf)

        # calculate the relative neutrino direction for each PMT
        # Define track_dir
        track_dir = tf.stack([dir_x, dir_y, dir_z], axis=-1)  

        # calculate opening angle of displacement vector and track direction
        opening_angle = angles.get_angle(track_dir,
                                         tf.concat([dx_normed,
                                                    dy_normed,
                                                    dz_normed], axis=-1)
                                         )
        opening_angle = tf.expand_dims(opening_angle, axis=-1)
        
        opening_angle_pmt = angles.get_angle(pmt_orientations,
                                         tf.concat([dx_normed,
                                                    dy_normed,
                                                    dz_normed], axis=-1)
                                         )
        opening_angle_pmt = tf.expand_dims(opening_angle_pmt, axis=-1)

        # transform dx, dy, dz, distance, zenith, azimuth to correct scale
        params_mean = self.data_trafo.data[parameter_tensor_name+'_mean']
        params_std = self.data_trafo.data[parameter_tensor_name+'_std']
        tensor = self.data_trafo.data['tensors'][parameter_tensor_name]
        norm_const = self.data_trafo.data['norm_constant']
        
        distance = distance + np.mean(params_mean[0:3])
        distance /= (np.linalg.norm(params_std[0:3]) + norm_const)

        combined_vector = np.append(params_std[0:3], params_std[-1])
        distance_to_cascade_gen2 /= (np.linalg.norm(combined_vector) + norm_const)

        # IceCube-86 --------------------------------------------------------------
        distance_to_cascade_icecube /= (np.linalg.norm(combined_vector) + norm_const)
        #----------------------------------------------------------------------------

        delta_dist_approach_trafo = delta_dist_approach / (
            np.linalg.norm(params_std[0:3]) + norm_const)
        opening_angle_traf = ((opening_angle - params_mean[3]) /
                              (norm_const + params_std[3]))
        opening_angle_pmt_traf = ((opening_angle_pmt - params_mean[3]) /
                              (norm_const + params_std[3]))

        # IceCube-86 -------------------------------------------
        # all pmts have same orientation in IceCube
        # Normalize the relative neutrino direction vector, it should be normal, but just in case
        track_dir_ice = tf.stack([dir_x, dir_y, dir_z], axis=-1)
        track_dir_norm = tf.linalg.norm(track_dir_ice, axis=-1, keepdims=True)
        track_dir_normalized = track_dir_ice / track_dir_norm

        # calculate opening angle of displacement vector and cascade direction
        # calculate opening angle of displacement vector and cascade direction
        opening_angle_ice = angles.get_angle(track_dir_normalized,
                                         tf.concat([dx_normed_ice,
                                                    dy_normed_ice,
                                                    dz_normed_ice], axis=-1)
                                         )
        opening_angle_ice = tf.expand_dims(opening_angle_ice, axis=-1)

        # transform dx, dy, dz, distance, zenith, azimuth to correct scale
        #params_mean_2 = self.data_trafo.data[parameter_tensor_name+'_2_mean']
        #params_std_2 = self.data_trafo.data[parameter_tensor_name+'_2_std']
        params_mean_2 = params_mean
        params_std_2 = params_std
        tensor = self.data_trafo.data['tensors'][parameter_tensor_name]
        norm_const = self.data_trafo.data['norm_constant']

        distance_ice /= (np.linalg.norm(params_std_2[0:3]) + norm_const)
        delta_dist_approach_ice_trafo = delta_dist_approach_2 / (
            np.linalg.norm(params_std_2[0:3]) + norm_const)
        opening_angle_ice_traf = ((opening_angle_ice - params_mean_2[3]) /
                              (norm_const + params_std_2[3]))
        
        x_parameters_expanded_2 = tf.unstack(tf.reshape(
                                                parameters_trafo_2,
                                                [-1, 1, 1, num_features]),
                                           axis=-1)
        #-----------------------------------------------------------------

        x_parameters_expanded = tf.unstack(tf.reshape(
                                                parameters_trafo,
                                                [-1, 1, 1, num_features]),
                                           axis=-1)

        # transform energies
        if tensor.trafo_log[5]:
            energy_before_trafo = tf.math.log(1 + energy_before)
            energy_after_trafo = tf.math.log(1 + energy_after)
            energy_cherenkov_trafo = tf.math.log(1 + energy_cherenkov)

            energy_before_ice_trafo = tf.math.log(1 + energy_before_ice)
            energy_after_ice_trafo = tf.math.log(1 + energy_after_ice)
            energy_cherenkov_ice_trafo = tf.math.log(1 + energy_cherenkov_ice)

        # apply bias correction
        
        energy_before_trafo -= params_mean[5]
        energy_after_trafo -= params_mean[5]
        energy_cherenkov_trafo -= params_mean[5]

        # apply scaling factor
        energy_before_trafo /= (params_std[5] + norm_const)
        energy_after_trafo /= (params_std[5] + norm_const)
        energy_cherenkov_trafo /= (params_std[5] + norm_const)

        # parameters: x, y, z, zenith, azimuth, energy, time, length, stoch
        modified_parameters = tf.stack([dir_x, dir_y, dir_z]
                                       + [x_parameters_expanded[5]]
                                       + x_parameters_expanded[7:],
                                       axis=-1)

        # put everything together
        params_expanded = tf.tile(modified_parameters, [1, num_strings, doms_per_string*num_pmts, 1])

        input_list = [
            params_expanded, dx_normed, dy_normed, dz_normed,
            (distance),  tf.expand_dims((distance_to_cascade_gen2),axis=-1), # some parameters normalized by hand...
            tf.expand_dims(delta_dist_approach_trafo, axis=-1),
            tf.expand_dims(energy_before_trafo, axis=-1),
            tf.expand_dims(energy_after_trafo, axis=-1),
            tf.expand_dims(energy_cherenkov_trafo, axis=-1),
            tf.expand_dims(rel_dist_closest_approach, axis=-1),
            tf.expand_dims(rel_dist_infinite_approach_trafo, axis=-1),
            opening_angle_pmt_traf,
           
        ]
        # IceCube-86 -------------------------------------------
        # apply bias correction
        modified_parameters_2 = tf.stack([dir_x, dir_y, dir_z]
                                       + [x_parameters_expanded_2[5]]
                                       + x_parameters_expanded_2[7:],
                                       axis=-1)

        energy_before_ice_trafo -= params_mean_2[5]
        energy_after_ice_trafo -= params_mean_2[5]
        energy_cherenkov_ice_trafo -= params_mean_2[5]

        # apply scaling factor
        energy_before_ice_trafo /= (params_std_2[5] + norm_const)
        energy_after_ice_trafo /= (params_std_2[5] + norm_const)
        energy_cherenkov_ice_trafo /= (params_std_2[5] + norm_const)

        distance_to_cascade_icecube /= (np.linalg.norm(params_std_2[0:3]+params_std_2[-1]) + norm_const)


        # put everything together
        params_expanded_ice = tf.tile(modified_parameters_2, [1, num_strings_2, doms_per_string_2*num_pmts_2, 1])

        input_list_ice = [
            params_expanded_ice, dx_normed_ice, dy_normed_ice, dz_normed_ice,
            distance_ice, tf.expand_dims(distance_to_cascade_icecube, axis = -1),
            tf.expand_dims(delta_dist_approach_ice_trafo, axis=-1),
            tf.expand_dims(energy_before_ice_trafo, axis=-1),
            tf.expand_dims(energy_after_ice_trafo, axis=-1),
            tf.expand_dims(energy_cherenkov_ice_trafo, axis=-1),
            tf.expand_dims(rel_dist_closest_approach_2, axis=-1),
            tf.expand_dims(rel_dist_infinite_approach_trafo_2, axis=-1),
            
        ]
        #tf.print("\n")
        #for i, input_i in enumerate(input_list_ice):
        #    tf.print(
        #        'input summary iceeeee: {}'.format(i),
        #         tf.reduce_min(input_i),
        #         tf.reduce_max(input_i),
        #         tf.reduce_mean(input_i),
        #         tf.math.reduce_std(input_i),
        #     )
        #tf.print("\n")
        #-----------------------------------------------------------------

        # for i, input_i in enumerate(input_list):
        #     tf.print(
        #         'input summary: {}'.format(i),
        #         tf.reduce_min(input_i),
        #         tf.reduce_max(input_i),
        #         tf.reduce_mean(input_i),
        #     )

        if config['add_opening_angle']:
            input_list.append(opening_angle_traf)
            input_list_ice.append(opening_angle_ice_traf)

        #if config['add_dom_coordinates'] and False:

            # transform coordinates to correct scale with mean 0 std dev 1
        #   dom_coords = np.expand_dims(
        #        detector.x_coords.astype(param_dtype_np), axis=0)
        #    # scale of coordinates is ~-500m to ~500m with std dev of ~ 284m
        #    dom_coords /= 284.

            # extend to correct batch shape:
        #    dom_coords = (tf.ones_like(dx_normed) * dom_coords)

        #    print('dom_coords', dom_coords)
        #    input_list.append(dom_coords)

        #if config['num_local_vars'] > 0:

            # extend to correct shape:
        #    local_vars = (tf.ones_like(dx_normed) *
        #                  self._untracked_data['local_vars'])
        #    print('local_vars', local_vars)

        #    input_list.append(local_vars)
        
        #tf.print("\n")
        #for i, input_i in enumerate(input_list):
        #    tf.print(
        #        'input summary: {}'.format(i),
        #         #tf.reduce_min(input_i),
        #         #tf.reduce_max(input_i),
        #         tf.reduce_mean(input_i),
        #         tf.math.reduce_std(input_i),
        #     )
        #tf.print("\n")

        # # Ensure input is not NaN
        if is_training:
            assert_op = tf.Assert(
                tf.math.is_finite(tf.reduce_mean(
                    tf.concat(input_list, axis=-1))),
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
                    tf.reduce_mean(delta_dist_approach_ice_trafo),
                    tf.reduce_mean(energy_before_ice_trafo),
                    tf.reduce_mean(energy_after_ice_trafo),
                    tf.reduce_mean(energy_cherenkov_ice_trafo),
                    tf.reduce_mean(rel_dist_closest_approach_2),
                    tf.reduce_mean(rel_dist_infinite_approach_trafo_2),
                    tf.reduce_mean(rel_dist_infinite_approach_2),
                    tf.reduce_mean(dist_closest_approach_2),
                ])
            with tf.control_dependencies([assert_op]):
                x_doms_input = tf.concat(input_list, axis=-1)
                x_doms_input_ice = tf.concat(input_list_ice, axis=-1)
        else:
            x_doms_input = tf.concat(input_list, axis=-1)
            x_doms_input_ice = tf.concat(input_list_ice, axis=-1)
        print('x_doms_input', x_doms_input)

        # -------------------------------------------
        # convolutional hex3d layers over X_IC86 data
        # -------------------------------------------
        conv_hex3d_layers = self._untracked_data['conv_hex3d_layer'](
                                        x_doms_input, is_training=is_training,
                                        keep_prob=config['keep_prob'])
        
        conv_hex3d_layers_icecube = self._untracked_data['conv_hex3d_layer_icecube_2'](
                                        x_doms_input_ice, is_training=is_training,
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
        
        dom_charges_trafo_ice = tf.expand_dims(conv_hex3d_layers_icecube[-1][..., 0],
                                           axis=-1)
        
        shape_diff_2 = tf.shape(dom_charges_trafo)[1] - tf.shape(dom_charges_trafo_ice)[1]
        shape_diff_3 = tf.shape(dom_charges_trafo)[2] - tf.shape(dom_charges_trafo_ice)[2]

        #print("charge_trafo",dom_charges_trafo.shape,dom_charges_trafo_ice.shape)
        padded_dom_charges_trafo_ice = tf.pad(dom_charges_trafo_ice, 
                              [[0, 0],  # No padding for the first dimension
                               [0, shape_diff_2],  # Padding for the second dimension
                               [0, shape_diff_3],  # Padding for the third dimension
                               [0, 0]])

        dom_charges_trafo = dom_charges_trafo #+ padded_dom_charges_trafo_ice*0 # If there is no icecube input pulse uncomment this

        # clip value range for more stability during training (maybe increase the ranges? might be the reason of underestimation for very large true charges)
        dom_charges_trafo = tf.clip_by_value(dom_charges_trafo, -20., 15)
        dom_charges_trafo_ice = tf.clip_by_value(dom_charges_trafo_ice, -20., 15)

        # apply exponential which also forces positive values
        dom_charges = tf.exp(dom_charges_trafo)
        dom_charges_ice = tf.exp(dom_charges_trafo_ice)

        # scale charges by cascade energy
        if config['scale_charge']:
            scale_factor = tf.expand_dims(parameter_list[5], axis=-1) / 10000.0
            dom_charges *= scale_factor
            #think how to correctly implement this... 

        # scale charges by realtive DOM efficiency
        if config['scale_charge_by_relative_dom_efficiency']:
            dom_charges *= tf.expand_dims(
                rel_dom_eff.astype(param_dtype_np), axis=-1)
            dom_charges_ice *= tf.expand_dims(
                rel_dom_eff_2.astype(param_dtype_np), axis=-1)

        # scale charges by global DOM efficiency
        if config['scale_charge_by_global_dom_efficiency']:
            dom_charges *= tf.expand_dims(
                parameter_list[self.get_index('DOMEfficiency')], axis=-1)

        # add small constant to make sure dom charges are > 0:
        dom_charges += 1e-7
        dom_charges_ice += 1e-7

        tensor_dict['dom_charges'] = dom_charges
        tensor_dict['dom_charges_ice'] = dom_charges_ice

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
            
            alpha_trafo_ice = tf.expand_dims(
                conv_hex3d_layers_icecube[-1][..., 1], axis=-1)
            
            padded_alpha_trafo_ice = tf.pad(alpha_trafo_ice, 
                              [[0, 0],  # No padding for the first dimension
                               [0, shape_diff_2],  # Padding for the second dimension
                               [0, shape_diff_3],  # Padding for the third dimension
                               [0, 0]])

            alpha_trafo = alpha_trafo + padded_alpha_trafo_ice*0

            # create correct offset and force positive and min values
            # The over-dispersion parameterized by alpha must be greater zero
            dom_charges_alpha = tf.nn.elu(alpha_trafo - 5) + 1.000001
            dom_charges_alpha_ice = tf.nn.elu(alpha_trafo_ice - 5) + 1.000001

            # compute log pdf
            dom_charges_llh = basis_functions.tf_log_negative_binomial(
                x=dom_charges_true,
                mu=dom_charges,
                alpha=dom_charges_alpha,
            )

            dom_charges_llh_ice = basis_functions.tf_log_negative_binomial(
                x=dom_charges_true_2,
                mu=dom_charges_ice,
                alpha=dom_charges_alpha_ice,
            )

            # compute standard deviation
            # std = sqrt(var) = sqrt(mu + alpha*mu**2)
            dom_charges_variance = (
                dom_charges + dom_charges_alpha*dom_charges**2)
            dom_charges_unc = tf.sqrt(dom_charges_variance)

            dom_charges_variance_ice = (
                dom_charges_ice + dom_charges_alpha_ice*dom_charges_ice**2)
            dom_charges_unc_ice = tf.sqrt(dom_charges_variance_ice)

            print('dom_charges_llh', dom_charges_llh, dom_charges_llh_ice)

            # tf.print(
            #     'dom_charges_alpha',
            #     tf.reduce_min(dom_charges_alpha),
            #     tf.reduce_mean(dom_charges_alpha),
            #     tf.reduce_max(dom_charges_alpha),
            # )

            # add tensors to tensor dictionary
            tensor_dict['dom_charges_alpha'] = dom_charges_alpha
            tensor_dict['dom_charges_unc'] = dom_charges_unc
            tensor_dict['dom_charges_variance'] = dom_charges_variance
            tensor_dict['dom_charges_log_pdf_values'] = dom_charges_llh

            tensor_dict['dom_charges_alpha_ice'] = dom_charges_alpha_ice
            tensor_dict['dom_charges_unc_ice'] = dom_charges_unc_ice
            tensor_dict['dom_charges_variance_ice'] = dom_charges_variance_ice
            tensor_dict['dom_charges_log_pdf_values_ice'] = dom_charges_llh_ice

        else:
            # Poisson Distribution: variance is equal to expected charge
            tensor_dict['dom_charges_unc'] = tf.sqrt(dom_charges)
            tensor_dict['dom_charges_variance'] = dom_charges

            tensor_dict['dom_charges_unc_ice'] = tf.sqrt(dom_charges_ice)
            tensor_dict['dom_charges_variance_ice'] = dom_charges_ice

        # -------------------------------------------
        # Get times at which to evaluate DOM PDF
        # -------------------------------------------

        # offset PDF evaluation times with cascade vertex time
        tensor_dict['time_offsets'] = parameters[:, 6]
        t_pdf = pulse_times - tf.gather(parameters[:, 6],
                                        indices=pulse_batch_id)
        # new shape: [None, 1]
        t_pdf = tf.expand_dims(t_pdf, axis=-1)
        t_pdf = tf.ensure_shape(t_pdf, [None, 1])

        # icecube modification --------------------------------
        t_pdf_ice = pulse_times_2 - tf.gather(parameters[:, 6],
                                        indices=pulse_batch_id_2)
        # new shape: [None, 1]
        t_pdf_ice = tf.expand_dims(t_pdf_ice, axis=-1)
        t_pdf_ice = tf.ensure_shape(t_pdf_ice, [None, 1])
        # -----------------------------------------------------

        # scale time range down to avoid big numbers:
        t_scale = 1. / self.time_unit_in_ns  # [1./ns]
        average_t_dist = 1000. * t_scale # increase this for Gen2 or tracks in General?
        t_pdf = t_pdf * t_scale
        t_pdf_ice = t_pdf_ice * t_scale

        # -------------------------------------------
        # Gather latent vars of mixture model
        # -------------------------------------------
        # check if we have the right amount of filters in the latent dimension
        n_models = config['num_latent_models']
        if n_models*4 + n_charge != config['num_filters_list'][-1]:
            raise ValueError('{!r} != {!r}'.format(
                n_models*4 + n_charge, config['num_filters_list'][-1]))
        if n_models <= 1:
            raise ValueError('{!r} !> 1'.format(n_models))

        out_layer = conv_hex3d_layers[-1]
        latent_mu = out_layer[...,
                              n_charge + 0*n_models:n_charge + 1*n_models]
        latent_sigma = out_layer[...,
                                 n_charge + 1*n_models:n_charge + 2*n_models]
        latent_r = out_layer[..., n_charge + 2*n_models:n_charge + 3*n_models]
        latent_scale = out_layer[...,
                                 n_charge + 3*n_models:n_charge + 4*n_models]
        
        # icecube modification ----------------------------------------------
        out_layer_ice = conv_hex3d_layers_icecube[-1]

        latent_mu_ice = out_layer_ice[...,
                              n_charge + 0*n_models:n_charge + 1*n_models]
       
        latent_sigma_ice = out_layer_ice[...,
                                 n_charge + 1*n_models:n_charge + 2*n_models]
        latent_r_ice = out_layer_ice[..., n_charge + 2*n_models:n_charge + 3*n_models]
        latent_scale_ice = out_layer_ice[...,
                                 n_charge + 3*n_models:n_charge + 4*n_models]
        
        padded_latent_mu_ice = tf.pad(latent_mu_ice, 
                              [[0, 0],  # No padding for the first dimension
                               [0, shape_diff_2],  # Padding for the second dimension
                               [0, shape_diff_3],  # Padding for the third dimension
                               [0, 0]])

        padded_latent_sigma_ice = tf.pad(latent_sigma_ice, 
                              [[0, 0],  # No padding for the first dimension
                               [0, shape_diff_2],  # Padding for the second dimension
                               [0, shape_diff_3],  # Padding for the third dimension
                               [0, 0]])
        padded_latent_r_ice = tf.pad(latent_r_ice, 
                              [[0, 0],  # No padding for the first dimension
                               [0, shape_diff_2],  # Padding for the second dimension
                               [0, shape_diff_3],  # Padding for the third dimension
                               [0, 0]])
        padded_latent_scale_ice = tf.pad(latent_scale_ice, 
                              [[0, 0],  # No padding for the first dimension
                               [0, shape_diff_2],  # Padding for the second dimension
                               [0, shape_diff_3],  # Padding for the third dimension
                               [0, 0]])


        latent_mu = latent_mu #+ padded_latent_mu_ice*0
        latent_sigma = latent_sigma #+ padded_latent_sigma_ice*0
        latent_r = latent_r #+ padded_latent_r_ice*0
        latent_scale = latent_scale #+ padded_latent_scale_ice*0
        # -------------------------------------------------------------------

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

        # IceCube-86 --------------------------------------------
        # create correct offset and scaling
        latent_mu_ice = average_t_dist + factor_mu * latent_mu_ice
        latent_sigma_ice = 2 + factor_sigma * latent_sigma_ice
        latent_r_ice = 1 + factor_r * latent_r_ice
        latent_scale_ice = 1 + factor_scale * latent_scale_ice

        # force positive and min values
        latent_scale_ice = tf.nn.elu(latent_scale_ice) + 1.00001
        latent_r_ice = tf.nn.elu(latent_r_ice) + 1.001
        latent_sigma_ice = tf.nn.elu(latent_sigma_ice) + 1.001

        # normalize scale to sum to 1
        latent_scale_ice /= tf.reduce_sum(latent_scale_ice, axis=-1, keepdims=True)
        # -----------------------------------------------------------------

        tensor_dict['latent_var_mu'] = latent_mu
        tensor_dict['latent_var_sigma'] = latent_sigma
        tensor_dict['latent_var_r'] = latent_r
        tensor_dict['latent_var_scale'] = latent_scale

        tensor_dict['latent_var_mu_ice'] = latent_mu_ice
        tensor_dict['latent_var_sigma_ice'] = latent_sigma_ice
        tensor_dict['latent_var_r_ice'] = latent_r_ice
        tensor_dict['latent_var_scale_ice'] = latent_scale_ice

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

        # IceCube-86 --------------------------------------------
        pulse_latent_mu_ice = tf.gather_nd(latent_mu_ice, pulses_ids_2)
        pulse_latent_sigma_ice = tf.gather_nd(latent_sigma_ice, pulses_ids_2)
        pulse_latent_r_ice = tf.gather_nd(latent_r_ice, pulses_ids_2)
        pulse_latent_scale_ice = tf.gather_nd(latent_scale_ice, pulses_ids_2)

        # ensure shapes
        pulse_latent_mu_ice = tf.ensure_shape(pulse_latent_mu_ice, [None, n_models])
        pulse_latent_sigma_ice = tf.ensure_shape(pulse_latent_sigma_ice,
                                             [None, n_models])
        pulse_latent_r_ice = tf.ensure_shape(pulse_latent_r_ice, [None, n_models])
        pulse_latent_scale_ice = tf.ensure_shape(pulse_latent_scale_ice,
                                             [None, n_models])
        
                # [n_pulses, 1] * [n_pulses, n_models] = [n_pulses, n_models]
        pulse_pdf_values_ice = basis_functions.tf_asymmetric_gauss(
                    x=t_pdf_ice, mu=pulse_latent_mu_ice, sigma=pulse_latent_sigma_ice,
                    r=pulse_latent_r_ice) * pulse_latent_scale_ice

        # new shape: [n_pulses]
        pulse_pdf_values_ice = tf.reduce_sum(pulse_pdf_values_ice, axis=-1)
        print('pulse_pdf_values', pulse_pdf_values_ice)
        # -----------------------------------------------------------------

        if is_training:
            # Ensure finite values
            asserts = []
            for name, tensor in sorted(tensor_dict.items()):
                assert_finite = tf.Assert(
                    tf.math.is_finite(tf.reduce_mean(tensor)),
                    [name, tf.reduce_mean(tensor)])
                asserts.append(assert_finite)
            with tf.control_dependencies(asserts):
                tensor_dict['pulse_pdf'] = pulse_pdf_values
                tensor_dict['pulse_pdf_ice'] = pulse_pdf_values_ice
        else:
            tensor_dict['pulse_pdf'] = pulse_pdf_values
            tensor_dict['pulse_pdf_ice'] = pulse_pdf_values_ice
        # -------------------------------------------

        return tensor_dict
