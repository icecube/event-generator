from __future__ import division, print_function
import logging
import tensorflow as tf

from egenerator import misc
from egenerator.manager.component import BaseComponent, Configuration


class DefaultLossModule(BaseComponent):

    """Default loss module that implements some standard loss functions.

    A loss component that is used to compute the loss. The component
    must provide a
    loss_module.get_loss(data_batch_dict, result_tensors, tensors,
                         parameter_tensor_name='x_parameters')
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
                 parameter_tensor_name='x_parameters'):
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

        Returns
        -------
        tf.Tensor
            Scalar loss
            Shape: []
        """
        return self.loss_function(data_batch_dict=data_batch_dict,
                                  result_tensors=result_tensors,
                                  tensors=tensors)

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
        tf.tensor
            Poisson Likelihood.
            Shape: []

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

        # calculate sum over a whole batch of events
        total_llh_poisson = tf.reduce_sum(llh_poisson)
        total_time_log_likelihood = tf.reduce_sum(time_log_likelihood)
        total_llh_event = tf.reduce_sum(llh_event)

        # average loss over events, such that it does not depend on batch size
        batch_size = tf.cast(tf.shape(llh_event)[0], dtype=dtype)
        average_event_loss = (total_llh_poisson + total_time_log_likelihood
                              + total_llh_event) / batch_size
        return average_event_loss

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
        tf.tensor
            Poisson Likelihood.
            Shape: []

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
        dom_charges_pdf_values = result_tensors['dom_charges_pdf_values']

        # get charge likelihood over total charge at a DOM for extendended LLH
        # shape: [n_batch, 86, 60]
        llh_charge = tf.math.log(tf.squeeze(dom_charges_pdf_values, axis=-1)
                                 + eps)

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

        # calculate sum over a whole batch of events
        total_charge_loss = tf.reduce_sum(-llh_charge)
        total_time_loss = tf.reduce_sum(time_loss)

        # average loss over events, such that it does not depend on batch size
        batch_size = tf.cast(tf.shape(llh_charge)[0], dtype=dtype)
        average_event_loss = (total_charge_loss + total_time_loss
                              ) / batch_size
        return average_event_loss
