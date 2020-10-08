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


class DefaultNoiseModel(Source):

    def __init__(self, logger=None):
        """Instantiate Source class

        Parameters
        ----------
        logger : logging.logger, optional
            The logger to use.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(DefaultNoiseModel, self).__init__(logger=self._logger)

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

        # ---------------------------------------
        # Define input parameters of noise source
        # ---------------------------------------
        parameter_names = []

        # get weights for general scaling of mu and over-dispersion
        self._untracked_data['local_vars'] = new_weights(
            shape=[2],
            stddev=1e-5,
            name='noise_scaling',
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
        print('Applying Noise Model...')

        tensor_dict = {}

        config = self.configuration.config['config']
        parameters = data_batch_dict[parameter_tensor_name]

        # get time exclusions
        tensors = self.data_trafo.data['tensors']
        if ('x_time_exclusions' in tensors.names and
                tensors.list[tensors.get_index('x_time_exclusions')].exists):
            time_exclusions_exist = True

            # shape: [n_tw, 2]
            x_time_exclusions = data_batch_dict['x_time_exclusions']

            # shape: [n_tw, 3]
            x_time_exclusions_ids = data_batch_dict['x_time_exclusions_ids']
            x_time_excl_batch_id = x_time_exclusions_ids[:, 0]
        else:
            time_exclusions_exist = False
        print('\t Applying time exclusions:', time_exclusions_exist)

        # shape: [n_pulses, 2]
        pulses = data_batch_dict['x_pulses']

        # shape: [n_pulses, 3]
        pulses_ids = data_batch_dict['x_pulses_ids']

        # shape: [n_batch, 2]
        time_window = data_batch_dict['x_time_window']

        # shape: [n_batch]
        livetime = time_window[:, 1] - time_window[:, 0]

        # shape: [n_batch, 1, 1, 1]
        livetime_exp = tf.reshape(
            time_window[:, 1] - time_window[:, 0], [-1, 1, 1, 1])

        # compute the expected charge at each DOM based off of noise rate
        # shape: [1, 86, 60, 1]
        dom_noise_rates = tf.reshape(
            detector.dom_noise_rates.astype(np.float32), shape=[1, 86, 60, 1])

        # shape: [n_batch, 86, 60, 1]
        dom_charges = dom_noise_rates * livetime_exp

        # shape: [n_batch]
        dom_pdf_constant = 1. / livetime

        # ----------------------------
        # Apply time window exclusions
        # ----------------------------
        if time_exclusions_exist:

            # limit exclusion windows to read out window
            # shape: [n_tw, 2]
            t_min = tf.gather(time_window[:, 0], indices=x_time_excl_batch_id)
            t_max = tf.gather(time_window[:, 1], indices=x_time_excl_batch_id)
            tw_livetime = tf.gather(livetime, indices=x_time_excl_batch_id)
            tw_reduced = tf.clip_by_value(
                x_time_exclusions,
                tf.expand_dims(t_min, axis=-1),
                tf.expand_dims(t_max, axis=-1),
            )

            # now calculate exclusions cdf
            # shape: [n_tw, 1]
            tw_cdf_exclusion = tf.expand_dims(
                (tw_reduced[:, 1] - tw_reduced[:, 0]) / tw_livetime, axis=-1)

            # some safety checks to make sure we aren't clipping too much
            tf.print(x_time_exclusions, 'x_time_exclusions', summarize=1000)
            tf.print(x_time_exclusions_ids, 'x_time_exclusions_ids', summarize=1000)
            tf.print(tw_reduced, 'tw_reduced', summarize=1000)
            tf.print(livetime, 'livetime', summarize=1000)
            tf.print(tw_livetime, 'tw_livetime', summarize=1000)
            tf.print(tw_cdf_exclusion, 'tw_cdf_exclusion', summarize=1000)
            asserts = []
            asserts.append(tf.debugging.Assert(
                tf.reduce_min(tw_cdf_exclusion) > -1e-4,
                ['Noise TW CDF < 0!',
                 tf.reduce_min(tw_cdf_exclusion),
                 tw_cdf_exclusion, x_time_exclusions_ids, x_time_exclusions,
                 tw_reduced],
                summarize=100000,
            ))
            asserts.append(tf.debugging.Assert(
                tf.reduce_max(tw_cdf_exclusion) < 1.0001,
                ['Noise TW CDF > 1!', tf.reduce_max(tw_cdf_exclusion),
                 tw_cdf_exclusion, x_time_exclusions_ids, x_time_exclusions,
                 tw_reduced],
                summarize=100000,
            ))
            with tf.control_dependencies(asserts):
                tw_cdf_exclusion = tfp.math.clip_by_value_preserve_gradient(
                    tw_cdf_exclusion, 0., 1.)

            # accumulate time window exclusions for each event
            # shape: [n_batch, 86, 60, 1]
            dom_cdf_exclusion = tf.tensor_scatter_nd_add(
                tf.zeros_like(data_batch_dict['x_dom_charge']),
                indices=x_time_exclusions_ids,
                updates=tw_cdf_exclusion,
            )

            # -------------
            # HACK TO DEBUG
            # -------------
            # def tf_unique_2d(x):
            #     x_shape = tf.shape(x)  # (3,2)
            #     x1 = tf.tile(x, [1, x_shape[0]])  # [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]
            #     x2 = tf.tile(x, [x_shape[0], 1])  # [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]

            #     x1_2 = tf.reshape(x1, [x_shape[0] * x_shape[0], x_shape[1]])
            #     x2_2 = tf.reshape(x2, [x_shape[0] * x_shape[0], x_shape[1]])
            #     cond = tf.reduce_all(tf.equal(x1_2, x2_2), axis=1)
            #     cond = tf.reshape(cond, [x_shape[0], x_shape[0]])  # reshaping cond to match x1_2 & x2_2
            #     cond_shape = tf.shape(cond)
            #     cond_cast = tf.cast(cond, tf.int32)  # convertin condition boolean to int
            #     cond_zeros = tf.zeros(cond_shape, tf.int32)  # replicating condition tensor into all 0's

            #     # CREATING RANGE TENSOR
            #     r = tf.range(x_shape[0])
            #     r = tf.add(tf.tile(r, [x_shape[0]]), 1)
            #     r = tf.reshape(r, [x_shape[0], x_shape[0]])

            #     # converting TRUE=1 FALSE=MAX(index)+1 (which is invalid by default) so when we take min it wont get selected & in end we will only take values <max(indx).
            #     f1 = tf.multiply(tf.ones(cond_shape, tf.int32), x_shape[0] + 1)
            #     f2 = tf.ones(cond_shape, tf.int32)
            #     cond_cast2 = tf.where(tf.equal(cond_cast, cond_zeros), f1, f2)  # if false make it max_index+1 else keep it 1

            #     # multiply range with new int boolean mask
            #     r_cond_mul = tf.multiply(r, cond_cast2)
            #     r_cond_mul2 = tf.reduce_min(r_cond_mul, axis=1)
            #     r_cond_mul3, unique_idx = tf.unique(r_cond_mul2)
            #     r_cond_mul4 = tf.subtract(r_cond_mul3, 1)

            #     # get actual values from unique indexes
            #     op = tf.gather(x, r_cond_mul4)
            #     return op, r_cond_mul4

            # unique_a_vals, unique_idx = tf_unique_2d(x_time_exclusions_ids)
            # count_a_unique = tf.math.unsorted_segment_sum(
            #     tf.ones_like(x_time_exclusions_ids[:, 0]),
            #      unique_idx,
            #      tf.shape(unique_idx)[0])

            # more_than_one = tf.greater(count_a_unique, 1)
            # more_than_one_idx = tf.squeeze(tf.where(more_than_one))
            # more_than_one_vals = tf.squeeze(tf.gather(unique_a_vals, more_than_one_idx))
            # tf.print(more_than_one_vals, 'more_than_one_vals')
            # -------------

            # some safety checks to make sure we aren't clipping too much
            asserts = []
            asserts.append(tf.debugging.Assert(
                tf.reduce_min(dom_cdf_exclusion) > -1e-4,
                ['Noise DOM CDF < 0!', tf.reduce_min(dom_cdf_exclusion),
                 tw_cdf_exclusion, x_time_exclusions_ids, x_time_exclusions,
                 tw_reduced],
                summarize=100000,
            ))
            asserts.append(tf.debugging.Assert(
                tf.reduce_max(dom_cdf_exclusion) < 1.0001,
                ['Noise DOM CDF > 1!', tf.reduce_max(dom_cdf_exclusion),
                 tw_cdf_exclusion, x_time_exclusions_ids, x_time_exclusions,
                 tw_reduced],
                summarize=100000,
            ))
            with tf.control_dependencies(asserts):
                dom_cdf_exclusion = tfp.math.clip_by_value_preserve_gradient(
                    dom_cdf_exclusion, 0., 1.)

            # we only have one model, so no need to actually sum up components
            # shape: [n_batch, 86, 60, 1]
            dom_cdf_exclusion_sum = dom_cdf_exclusion

        # ----------------------------

        # local scaling vars are initialized around zero with small std dev
        local_vars = tf.nn.elu(self._untracked_data['local_vars']) + 1.01

        # scaling of expected noise hits: ensure positive values.
        # shape: [1, 1, 1, 1]
        mean_scaling = tf.reshape(
            tf.nn.elu(local_vars[0]) + 1.01, [1, 1, 1, 1])

        # scaling of uncertainty. shape: [1, 1, 1, 1]
        # The over-dispersion parameterized by alpha must be greater zero
        # Var(x) = mu + alpha*mu**2
        dom_charges_alpha = tf.reshape(
            tf.nn.elu(local_vars[1] - 5) + 1.000001, [1, 1, 1, 1])

        # scale dom charge and uncertainty by learned scaling
        dom_charges = dom_charges * mean_scaling

        # scale by time exclusions
        if time_exclusions_exist:
            dom_charges *= (1. - dom_cdf_exclusion_sum + 1e-3)

        # add small constant to make sure dom charges are > 0:
        dom_charges += 1e-7

        # compute standard deviation
        # std = sqrt(var) = sqrt(mu + alpha*mu**2)
        dom_charges_variance = (
            dom_charges + dom_charges_alpha*dom_charges**2)
        dom_charges_unc = tf.sqrt(dom_charges_variance)

        # Compute Log Likelihood for pulses
        # PDF is a uniform distribution in the specified time window.
        # The time window is constructed such that every pulse is part of it
        # That means that every pulse of an event has the same likelihood.
        # shape: [n_pulses]
        pulse_pdf = tf.gather(dom_pdf_constant, indices=pulses_ids[:, 0])

        # scale up pulse pdf by time exclusions if needed
        if time_exclusions_exist:
            pulse_cdf_exclusion = tf.gather_nd(dom_cdf_exclusion, pulses_ids)
            pulse_pdf /= tf.squeeze(1. - pulse_cdf_exclusion + 1e-3, axis=-1)

        # add tensors to tensor dictionary
        tensor_dict['dom_charges'] = dom_charges
        tensor_dict['dom_charges_alpha'] = dom_charges_alpha
        tensor_dict['dom_charges_unc'] = dom_charges_unc
        tensor_dict['dom_charges_variance'] = dom_charges_variance
        tensor_dict['pdf_constant'] = dom_pdf_constant
        tensor_dict['pdf_time_window'] = time_window
        tensor_dict['pulse_pdf'] = pulse_pdf

        if time_exclusions_exist:
            tensor_dict['dom_cdf_exclusion'] = dom_cdf_exclusion
            tensor_dict['dom_cdf_exclusion_sum'] = dom_cdf_exclusion_sum
        # -------------------------------------------

        return tensor_dict
