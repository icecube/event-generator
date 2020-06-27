from __future__ import division, print_function
import logging
import tensorflow as tf

from egenerator import misc
from egenerator.utils import basis_functions
from egenerator.manager.component import BaseComponent, Configuration


class DefaultLossModule(BaseComponent):

    """Default loss module that implements some standard loss functions.

    A loss component that is used to compute the loss. The component
    must provide a
    loss_module.get_loss(data_batch_dict, result_tensors, tensors,
                         parameter_tensor_name='x_parameters', **kwargs)
    method.
    """

    @property
    def loss_function(self):
        if (self.untracked_data is not None and
                'loss_function' in self.untracked_data):
            return self.untracked_data['loss_function']
        else:
            return None

    def __init__(self, logger=None):
        """Initializes LossModule object.

        Parameters
        ----------
        logger : logging.logger, optional
            A logging instance.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(DefaultLossModule, self).__init__(logger=self._logger)

    def _configure(self, config):
        """Configure the LossModule component instance.

        Parameters
        ----------
        config : dict
            Configuration settings of the LossModule object.
            Must contain:
                'float_precision': str
                    The float precision to use
                'loss_function_name': str
                    The name of the loss function to use.

        Returns
        -------
        Configuration object
            The configuration object of the newly configured component.
            This does not need to include configurations of sub components
            which are passed directly as parameters into the configure method,
            as these are automatically gathered. Components passed as lists,
            tuples, and dicts are also collected, unless they are nested
            deeper (list of list of components will not be detected).
            The dependent_sub_components may also be left empty for these
            passed and detected sub components.
            Deeply nested sub components or sub components created within
            (and not directly passed as an argument to) this component
            must be added manually.
            Settings that need to be defined are:
                class_string:
                    misc.get_full_class_string_of_object(self)
                settings: dict
                    The settings of the component.
                mutable_settings: dict, default={}
                    The mutable settings of the component.
                check_values: dict, default={}
                    Additional check values.
        dict
            The data of the component.
        dict
            A dictionary of dependent sub components. This is a dictionary
            of sub components that need to be saved and loaded recursively
            when the component is saved and loaded.
            Return None if no dependent sub components exist.
        """

        # choose loss function
        self.untracked_data['loss_function'] = getattr(
                                        self, config['loss_function_name'])

        # create configuration object
        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings=dict(config=config))

        return configuration, {}, {}

    def get_loss(self, data_batch_dict, result_tensors, tensors, model,
                 parameter_tensor_name='x_parameters',
                 reduce_to_scalar=True,
                 normalize_by_total_charge=False,
                 **kwargs):
        """Get the scalar loss for a given data batch and result tensors.

        Parameters
        ----------
        data_batch_dict : dict of tf.Tensor
            parameters : tf.Tensor
                A tensor which describes the input parameters of the source.
                This fully defines the source hypothesis. The tensor is of
                shape [-1, n_params] and the last dimension must match the
                order of the parameter names (model.parameter_names),
            pulses : tf.Tensor
                The input pulses (charge, time) of all events in a batch.
                Shape: [-1, 2]
            pulses_ids : tf.Tensor
                The pulse indices (batch_index, string, dom) of all pulses in
                the batch of events.
                Shape: [-1, 3]
        result_tensors : dict of tf.Tensor
            A dictionary of output tensors.
            This  dictionary must at least contain:

                'dom_charges': the predicted charge at each DOM
                               Shape: [-1, 86, 60, 1]
                'pulse_pdf': The likelihood evaluated for each pulse
                             Shape: [-1]
        tensors : DataTensorList
            The data tensor list describing the input data
        model : Model
            The model object used to calculate the result tensors.
        parameter_tensor_name : str, optional
            The name of the parameter tensor to use. Default: 'x_parameters'.
            This option is ignored here and has no effect.
        reduce_to_scalar : bool, optional
            If True, the individual terms of the log likelihood loss will be
            reduced (aggregated) to a scalar loss.
            If False, a list of tensors will be returned that contain the terms
            of the log likelihood. Note that each of the returend tensors may
            have a different shape.
        normalize_by_total_charge : bool, optional
            If True, the loss will be normalized (divided) by the total charge.
            This will make the loss of events with vastly different amounts of
            detected photons be more comparable.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        tf.Tensor or list of tf.Tensor
            if `reduce_to_scalar` is True:
                Scalar loss
                Shape: []
            else:
                List of tensors defining the terms of the log likelihood
        """
        loss_terms = self.loss_function(data_batch_dict=data_batch_dict,
                                        result_tensors=result_tensors,
                                        tensors=tensors)

        if normalize_by_total_charge:
            total_charge = tf.reduce_sum(data_batch_dict['x_pulses'][:, 0])

            # Note: this depends on the actual loss being used.
            # Here we assume that an additional poisson term for each DOM
            # is used as well. This adds ~5160 (minus exclusions) additional
            # terms to the likelhood.
            # ToDo: properly count number of likelihood terms and also properly
            # normalize if loss is composed from multiple individual losses
            approx_number_of_terms = total_charge + 5160
            loss_terms = [loss / approx_number_of_terms for loss in loss_terms]

        if reduce_to_scalar:
            return tf.math.accumulate_n([tf.reduce_sum(loss_term)
                                         for loss_term in loss_terms])
        else:
            return loss_terms

    def log_faculty(self, x):
        """Get continous log faculty approximation via gamma distribution

        Parameters
        ----------
        x : tf.Tensor
            The tensor for which to compute the log faculty approximation.

        Returns
        -------
        tf.Tensor
            Description
        """
        return tf.math.lgamma(tf.clip_by_value(x + 1, 2, float('inf')))

    def unbinned_extended_pulse_llh(self, data_batch_dict, result_tensors,
                                    tensors):
        """Unbinned extended poisson likelhood for data pulses.

        Pulses must *not* contain any pulses in excluded DOMs or excluded time
        windows. It is assumed that these pulses are already removed, e.g.
        the time pdf is calculated for all pulses.

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
        result_tensors : dict of tf.Tensor
            A dictionary of output tensors.
            This  dictionary must at least contain:

                'dom_charges': the predicted charge at each DOM
                               Shape: [-1, 86, 60, 1]
                'pulse_pdf': The likelihood evaluated for each pulse
                             Shape: [-1]
        tensors : DataTensorList
            The data tensor list describing the input data

        Returns
        -------
        List of tf.tensor
            Poisson Likelihood.
            List of tensors defining the terms of the log likelihood

        Raises
        ------
        NotImplementedError
            Description
        """
        dtype = getattr(
            tf, self.configuration.config['config']['float_precision'])

        # shape: [n_pulses]
        pulse_charges = data_batch_dict['x_pulses'][:, 0]
        pulse_pdf_values = result_tensors['pulse_pdf']

        # shape: [n_batch, 86, 60, 1]
        hits_true = data_batch_dict['x_dom_charge']

        # shape: [n_batch, 86, 60]
        dom_charges_true = tf.squeeze(hits_true, axis=-1)
        dom_charges_pred = tf.squeeze(result_tensors['dom_charges'], axis=-1)

        # throw error if this is being used with time window exclusions
        # one needs to calculate cumulative pdf from exclusion window and
        # reduce the predicted charge by this factor
        if ('x_time_exclusions' in tensors.names and
                tensors.list[tensors.get_index('x_time_exclusions')].exists):
            raise NotImplementedError(
                'Time exclusions are currently not implemented!')

        # mask out dom exclusions
        if ('x_dom_exclusions' in tensors.names and
                tensors.list[tensors.get_index('x_dom_exclusions')].exists):
            mask_valid = tf.cast(
                tf.squeeze(data_batch_dict['x_dom_exclusions'], axis=-1),
                dtype=dtype)
            dom_charges_true = dom_charges_true * mask_valid
            dom_charges_pred = dom_charges_pred * mask_valid

        # prevent log(zeros) issues
        eps = 1e-7
        pulse_log_pdf_values = tf.math.log(pulse_pdf_values + eps)
        # pulse_log_pdf_values = tf.where(hits_true > 0,
        #                                 tf.math.log(pulse_pdf_values + eps),
        #                                 tf.zeros_like(pulse_pdf_values))

        # compute unbinned negative likelihood over pulse times with given
        # time pdf: -sum( charge_i * log(pdf_d(t_i)) )
        time_log_likelihood = -pulse_charges * pulse_log_pdf_values

        # get poisson likelihood over total charge at a DOM for extendended LLH
        llh_poisson = (dom_charges_pred -
                       dom_charges_true * tf.math.log(dom_charges_pred + eps))

        # Poisson loss over total event charge
        event_charges_true = tf.reduce_sum(dom_charges_true, axis=[1, 2])
        event_charges_pred = tf.reduce_sum(dom_charges_pred, axis=[1, 2])
        llh_event = event_charges_pred - event_charges_true * tf.math.log(
                                                    event_charges_pred + eps)

        loss_terms = [llh_poisson, time_log_likelihood, llh_event]

        # Add normalization terms if desired
        # Note: these are irrelevant for the minimization, but will make loss
        # curves more meaningful
        if self.configuration.config['config']['add_normalization_term']:
            norm_pulses = tf.reduce_sum(self.log_faculty(pulse_charges))
            norm_doms = tf.reduce_sum(self.log_faculty(dom_charges_true))
            norm_events = tf.reduce_sum(self.log_faculty(event_charges_true))
            norm = norm_pulses + norm_doms + norm_events
            loss_terms.append(norm)

        return loss_terms

    def unbinned_pulse_time_llh(self, data_batch_dict,
                                result_tensors, tensors):
        """Unbinned Pulse Time likelhood.

        Pulses must *not* contain any pulses in excluded DOMs or excluded time
        windows. It is assumed that these pulses are already removed, e.g.
        the time pdf is calculated for all pulses.

        This is similar to `unbinned_extended_pulse_llh`. Major differences:
            - No additional likelihood terms for DOM or event charge

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
        result_tensors : dict of tf.Tensor
            A dictionary of output tensors.
            This  dictionary must at least contain:
                'pulse_pdf': The likelihood evaluated for each pulse
                             Shape: [-1]
        tensors : DataTensorList
            The data tensor list describing the input data

        Returns
        -------
        List of tf.tensor
            Poisson Likelihood.
            List of tensors defining the terms of the log likelihood

        Raises
        ------
        NotImplementedError
            Description
        """
        dtype = getattr(
            tf, self.configuration.config['config']['float_precision'])

        # shape: [n_pulses]
        pulse_charges = data_batch_dict['x_pulses'][:, 0]
        pulse_pdf_values = result_tensors['pulse_pdf']

        # throw error if this is being used with time window exclusions
        # one needs to calculate cumulative pdf from exclusion window and
        # reduce the predicted charge by this factor
        if ('x_time_exclusions' in tensors.names and
                tensors.list[tensors.get_index('x_time_exclusions')].exists):
            raise NotImplementedError(
                'Time exclusions are currently not implemented!')

        # prevent log(zeros) issues
        eps = 1e-7
        pulse_log_pdf_values = tf.math.log(pulse_pdf_values + eps)

        # compute unbinned negative likelihood over pulse times with given
        # time pdf: -sum( charge_i * log(pdf_d(t_i)) )
        time_loss = -pulse_charges * pulse_log_pdf_values

        loss_terms = [time_loss]

        # Add normalization terms if desired
        # Note: these are irrelevant for the minimization, but will make loss
        # curves more meaningful
        if self.configuration.config['config']['add_normalization_term']:
            loss_terms.append(tf.reduce_sum(self.log_faculty(pulse_charges)))

        return loss_terms

    def unbinned_pulse_and_dom_charge_pdf(self, data_batch_dict,
                                          result_tensors, tensors):
        """Unbinned extended poisson likelhood with DOM charge PDF.

        Pulses must *not* contain any pulses in excluded DOMs or excluded time
        windows. It is assumed that these pulses are already removed, e.g.
        the time pdf is calculated for all pulses.

        This is similar to `unbinned_extended_pulse_llh`. Major differences:
            - No Poisson Likelihood is assumed here. Instead, the model
              estimates the charge PDF for each DOM
            - No additional likelihood term is added for total event charge

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
        result_tensors : dict of tf.Tensor
            A dictionary of output tensors.
            This  dictionary must at least contain:

                'dom_charges': the predicted charge at each DOM
                               Shape: [-1, 86, 60, 1]
                'dom_charges_pdf_values': the likelihood evaluated for each DOM
                               Shape: [-1, 86, 60, 1]
                'pulse_pdf': The likelihood evaluated for each pulse
                             Shape: [-1]
        tensors : DataTensorList
            The data tensor list describing the input data

        Returns
        -------
        List of tf.tensor
            Poisson Likelihood.
            List of tensors defining the terms of the log likelihood

        Raises
        ------
        NotImplementedError
            Description
        """
        dtype = getattr(
            tf, self.configuration.config['config']['float_precision'])

        # shape: [n_pulses]
        pulse_charges = data_batch_dict['x_pulses'][:, 0]
        pulse_pdf_values = result_tensors['pulse_pdf']

        # shape: [n_batch, 86, 60, 1]
        hits_true = data_batch_dict['x_dom_charge']

        # get charge likelihood over total charge at a DOM for extendended LLH
        # shape: [n_batch, 86, 60]
        llh_charge = tf.squeeze(result_tensors['dom_charges_log_pdf_values'],
                                axis=-1)

        # throw error if this is being used with time window exclusions
        # one needs to calculate cumulative pdf from exclusion window and
        # reduce the predicted charge by this factor
        if ('x_time_exclusions' in tensors.names and
                tensors.list[tensors.get_index('x_time_exclusions')].exists):
            raise NotImplementedError(
                'Time exclusions are currently not implemented!')

        # mask out dom exclusions
        if ('x_dom_exclusions' in tensors.names and
                tensors.list[tensors.get_index('x_dom_exclusions')].exists):
            mask_valid = tf.cast(
                tf.squeeze(data_batch_dict['x_dom_exclusions'], axis=-1),
                dtype=dtype)
            llh_charge = llh_charge * mask_valid

        # prevent log(zeros) issues
        eps = 1e-7
        pulse_log_pdf_values = tf.math.log(pulse_pdf_values + eps)
        # pulse_log_pdf_values = tf.where(hits_true > 0,
        #                                 tf.math.log(pulse_pdf_values + eps),
        #                                 tf.zeros_like(pulse_pdf_values))

        # compute unbinned negative likelihood over pulse times with given
        # time pdf: -sum( charge_i * log(pdf_d(t_i)) )
        time_loss = -pulse_charges * pulse_log_pdf_values

        loss_terms = [-llh_charge, time_loss]

        # Add normalization terms if desired
        # Note: these are irrelevant for the minimization, but will make loss
        # curves more meaningful
        if self.configuration.config['config']['add_normalization_term']:
            norm_pulses = tf.reduce_sum(self.log_faculty(pulse_charges))
            norm_doms = tf.reduce_sum(self.log_faculty(hits_true))
            loss_terms.append(norm_pulses + norm_doms)

        return loss_terms

    def unbinned_charge_quantile_pdf(self, data_batch_dict, result_tensors,
                                     tensors):
        """Unbinned Pulse Quantile PDF.

        Pulses must *not* contain any pulses in excluded DOMs or excluded time
        windows. It is assumed that these pulses are already removed, e.g.
        the time pdf is calculated for all pulses.

        This loss will compute the likelihood of
            p(t_i| q_i, D_i)
            t_i: Time of pulse i
            q_i: Cumulative fractional DOM charge (charge quantile) of pulse i
            D_i: Total charge at the DOM at which pulse i was measured

        Note this likelihood does not place any constraints on total measured
        charge. It is therefore not suited to reconstruct Energy.

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
        result_tensors : dict of tf.Tensor
            A dictionary of output tensors.
            This  dictionary must at least contain:
                'pulse_quantile_pdf':
                    The likelihood evaluated for each pulse
                    Shape: [-1]
        tensors : DataTensorList
            The data tensor list describing the input data

        Returns
        -------
        List of tf.tensor
            Quantile PDF Likelihood.
            List of tensors defining the terms of the log likelihood
        """
        dtype = getattr(
            tf, self.configuration.config['config']['float_precision'])

        # shape: [n_pulses]
        pulse_charges = data_batch_dict['x_pulses'][:, 0]
        pulse_quantiles = data_batch_dict['x_pulses'][:, 2]
        pulse_pdf_values = result_tensors['pulse_quantile_pdf']

        # throw error if this is being used with time window exclusions
        # one needs to calculate cumulative pdf from exclusion window and
        # reduce the predicted charge by this factor
        if ('x_time_exclusions' in tensors.names and
                tensors.list[tensors.get_index('x_time_exclusions')].exists):
            self._logger.warning('Pulses in excluded time windows must have '
                                 'already been removed!')

        # mask out dom exclusions
        if ('x_dom_exclusions' in tensors.names and
                tensors.list[tensors.get_index('x_dom_exclusions')].exists):
            self._logger.warning('Pulses at excluded DOMs must have already '
                                 'been removed!')

        # prevent log(zeros) issues
        eps = 1e-7
        pulse_log_pdf_values = tf.math.log(pulse_pdf_values + eps)

        # compute unbinned negative likelihood over pulse times with given
        # time pdf: -sum( log(pdf_d(t_i)) )
        # do not weight with pulse charges here: a very high pulse charge
        # means that multiple pulses were combinend and that the true quantile
        # is not very accurate. Contrary, a low pulse charge
        # (probable single pulse) defines a very accurate time and quantile.
        # Therefore: do not weight log_pdf by charge
        time_loss = -pulse_log_pdf_values

        # only train for certain quantiles if bounds are provided
        low_lim = self.configuration.config['config']['min_charge_quantile']
        max_lim = self.configuration.config['config']['max_charge_quantile']

        if low_lim is not None:
            time_loss = tf.where(
                    pulse_quantiles < low_lim,
                    tf.zeros_like(time_loss),
                    time_loss
                )
        if max_lim is not None:
            time_loss = tf.where(
                    pulse_quantiles > max_lim,
                    tf.zeros_like(time_loss),
                    time_loss
                )

        loss_terms = [time_loss]

        # Add normalization terms if desired
        # Note: these are irrelevant for the minimization, but will make loss
        # curves more meaningful
        if self.configuration.config['config']['add_normalization_term']:
            # the pulse_pdf is already correctly normalized due to the
            # mixture model
            pass

        return loss_terms

    def dom_and_event_charge_pdf(self, data_batch_dict, result_tensors,
                                 tensors):
        """Charge PDF (estimated by Model)

        This is a likelihood over the total event charge in addition to the
        charge measured at each DOM. The PDF and likelihood used is defined
        by the model which must provide the computed log likelihood values
        for the DOMs: `dom_charges_log_pdf_values`.
        The uncertainty on the total event charge is computed by accumulating
        the uncertainties in quadrature.

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
        result_tensors : dict of tf.Tensor
            A dictionary of output tensors.
            This  dictionary must at least contain:

                'dom_charges': the predicted charge at each DOM
                               Shape: [-1, 86, 60, 1]
                'dom_charges_log_pdf_values':
                    The likelihood evaluated for each DOM.
                    Shape: [-1, 86, 60, 1]
                'dom_charges_gaussian_unc':
                    The (Gaussian) uncertainty on the predicted DOM charge
                    Shape: [-1, 86, 60, 1]
        tensors : DataTensorList
            The data tensor list describing the input data

        Returns
        -------
        List of tf.tensor
            Charge PDF Likelihood.
            List of tensors defining the terms of the log likelihood
        """
        eps = 1e-7
        dtype = getattr(
            tf, self.configuration.config['config']['float_precision'])

        # shape: [n_batch, 86, 60, 1]
        hits_true = tf.squeeze(data_batch_dict['x_dom_charge'], axis=-1)
        hits_pred = tf.squeeze(result_tensors['dom_charges'], axis=-1)

        # get charge likelihood over total charge at a DOM for extendended LLH
        # shape: [n_batch, 86, 60]
        llh_charge = tf.squeeze(result_tensors['dom_charges_log_pdf_values'],
                                axis=-1)

        # get variance on DOM charges
        # shape: [n_batch, 86, 60]
        dom_charges_variance = tf.squeeze(
            result_tensors['dom_charges_variance'], axis=-1)

        # mask out dom exclusions
        if ('x_dom_exclusions' in tensors.names and
                tensors.list[tensors.get_index('x_dom_exclusions')].exists):
            mask_valid = tf.cast(
                tf.squeeze(data_batch_dict['x_dom_exclusions'], axis=-1),
                dtype=dtype)
            llh_charge = llh_charge * mask_valid
            hits_true = hits_true * mask_valid
            hits_pred = hits_pred * mask_valid
            dom_charges_variance = dom_charges_variance * mask_valid
        else:
            mask_valid = tf.ones_like(llh_charge)

        # Compute Gaussian likelihood over total event charge
        event_charges_true = tf.reduce_sum(hits_true, axis=[1, 2])
        event_charges_pred = tf.reduce_sum(hits_pred, axis=[1, 2])
        event_charges_unc = tf.sqrt(tf.reduce_sum(dom_charges_variance,
                                                  axis=[1, 2]))
        llh_event = basis_functions.tf_log_gauss(
            x=event_charges_true,
            mu=event_charges_pred,
            sigma=event_charges_unc,
        )

        loss_terms = [-llh_charge, -llh_event]

        # Add normalization terms if desired
        # Note: these are irrelevant for the minimization, but will make loss
        # curves more meaningful
        if self.configuration.config['config']['add_normalization_term']:
            # total event charge is properly normalized due to the used gauss
            pass

        return loss_terms

    def negative_binomial_charge_pdf(self, data_batch_dict, result_tensors,
                                     tensors):
        """Negative Binomial Charge PDF

        This is a likelihood over the total event charge in addition to the
        charge measured at each DOM. A negative binomial distribution
        is used to calculate the charge likelihood. This allows for the
        inclusion of over-dispersion in contrast to the simple Poisson
        Likelihood. The model must provide the dom charges and variances
        thereof: `dom_charges`, `dom_charges_variance`.
        This loss pairs well with a loss for the time PDF such
        as `unbinned_pulse_time_llh`.

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
        result_tensors : dict of tf.Tensor
            A dictionary of output tensors.
            This  dictionary must at least contain:

                'dom_charges': the predicted charge at each DOM
                               Shape: [-1, 86, 60, 1]
                'dom_charges_variance':
                    the predicted variance on the charge at each DOM.
                    This assumes the underlying distribution is a negative
                    binomial distribution.
                    Shape: [-1, 86, 60]
        tensors : DataTensorList
            The data tensor list describing the input data

        Returns
        -------
        List of tf.tensor
            Charge PDF Likelihood.
            List of tensors defining the terms of the log likelihood
        """
        eps = 1e-7
        dtype = getattr(
            tf, self.configuration.config['config']['float_precision'])

        # shape: [n_batch, 86, 60]
        hits_true = tf.squeeze(data_batch_dict['x_dom_charge'], axis=-1)
        hits_pred = tf.squeeze(result_tensors['dom_charges'], axis=-1)
        dom_charges_variance = tf.squeeze(
            result_tensors['dom_charges_variance'], axis=-1)

        # compute over-dispersion factor alpha
        # var = mu + alpha*mu**2
        # alpha = (var - mu) / (mu**2)
        dom_charges_alpha = (dom_charges_variance - hits_pred) / (
            hits_pred**2 + eps)
        # Make sure alpha is positive
        dom_charges_alpha = tf.clip_by_value(
            dom_charges_alpha, eps, float('inf'))

        # compute negative binomial charge likelihood over DOMs
        # shape: [n_batch, 86, 60]
        llh_charge = basis_functions.tf_log_negative_binomial(
            x=hits_true,
            mu=hits_pred,
            alpha=dom_charges_alpha,
        )

        # mask out dom exclusions
        if ('x_dom_exclusions' in tensors.names and
                tensors.list[tensors.get_index('x_dom_exclusions')].exists):
            mask_valid = tf.cast(
                tf.squeeze(data_batch_dict['x_dom_exclusions'], axis=-1),
                dtype=dtype)
            llh_charge = llh_charge * mask_valid
            hits_true = hits_true * mask_valid
            hits_pred = hits_pred * mask_valid
            dom_charges_variance = dom_charges_variance * mask_valid
        else:
            mask_valid = tf.ones_like(hits_true)

        # compute negative binomial charge likelihood over total event charge
        event_charges_true = tf.reduce_sum(hits_true, axis=[1, 2])
        event_charges_pred = tf.reduce_sum(hits_pred, axis=[1, 2])
        event_charges_variance = tf.reduce_sum(hits_pred, axis=[1, 2])

        # compute over-dispersion factor alpha
        # var = mu + alpha*mu**2
        # alpha = (var - mu) / (mu**2)
        event_charges_alpha = (event_charges_variance - event_charges_pred) / (
            event_charges_pred**2)

        # Make sure alpha is positive
        event_charges_alpha = tf.clip_by_value(
            event_charges_alpha, eps, float('inf'))

        llh_event = basis_functions.tf_log_negative_binomial(
            x=event_charges_true,
            mu=event_charges_pred,
            alpha=event_charges_alpha,
        )

        loss_terms = [-llh_charge, -llh_event]

        # Add normalization terms if desired
        # Note: these are irrelevant for the minimization, but will make loss
        # curves more meaningful
        if self.configuration.config['config']['add_normalization_term']:
            # total event charge is properly normalized due to the used gauss
            pass

        return loss_terms
