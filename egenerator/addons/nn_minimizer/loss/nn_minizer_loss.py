from __future__ import division, print_function
import logging
import tensorflow as tf

from egenerator import misc
from egenerator.utils import basis_functions
from egenerator.manager.component import BaseComponent, Configuration


class NNMinimizerLoss(BaseComponent):

    """NNMinimizer loss module that implements some standard loss functions.

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
        super(NNMinimizerLoss, self).__init__(logger=self._logger)

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
                 parameter_tensor_name='x_parameters', reduce_to_scalar=True):
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

        if reduce_to_scalar:
            return tf.math.accumulate_n([tf.reduce_sum(loss_term)
                                         for loss_term in loss_terms])
        else:
            return loss_terms

    def gaussian_likelihood(self, data_batch_dict, result_tensors, tensors):
        """Gaussian Likelihood

        Computes a gaussian likelihood over predicted parameters and
        estimated uncertainty thereof.

        Parameters
        ----------
        data_batch_dict : dict of tf.Tensor
            x_parameters : tf.Tensor
                A tensor which describes the input parameters of the source.
                This fully defines the source hypothesis. The tensor is of
                shape [-1, n_params] and the last dimension must match the
                order of the parameter names (self.parameter_names),
            x_pulses : tf.Tensor
                The input pulses (charge, time) of all events in a batch.
                Shape: [-1, 2]
            x_pulses_ids : tf.Tensor
                The pulse indices (batch_index, string, dom) of all pulses in
                the batch of events.
                Shape: [-1, 3]
        result_tensors : dict of tf.Tensor
            A dictionary of output tensors.
            This  dictionary must at least contain:

                'parameters': the predicted parameters
                              Shape: [-1, num_parameters]
                'parameters_unc': the estimated uncertainty on the parameters.
                                  Shape: [-1, num_parameters]
                'parameters_trafo': the transformed predicted parameters
                                    Shape: [-1, num_parameters]
                'parameters_unc_trafo': the estimated uncertainty
                                        on the transformed parameters.
                                        Shape: [-1, num_parameters]
        tensors : DataTensorList
            The data tensor list describing the input data

        Returns
        -------
        List of tf.tensor
            Charge PDF Likelihood.
            List of tensors defining the terms of the log likelihood
        """
        y_pred_trafo = result_tensors['parameters_trafo']
        y_pred_unc_trafo = result_tensors['parameters_unc_trafo']
        y_true_trafo = data_batch_dict['x_parameters']

        llh_event = basis_functions.tf_log_gauss(
            x=y_pred_trafo,
            mu=y_true_trafo,
            sigma=y_pred_unc_trafo,
        )

        loss_terms = [-llh_event]

        # Add normalization terms if desired
        # Note: these are irrelevant for the minimization, but will make loss
        # curves more meaningful
        if self.configuration.config['config']['add_normalization_term']:
            # total event charge is properly normalized due to the used gauss
            pass

        return loss_terms
