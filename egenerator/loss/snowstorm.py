import logging
import numpy as np
import tensorflow as tf

from egenerator import misc
from egenerator.manager.component import BaseComponent, Configuration
from egenerator.utils import basis_functions


class SnowstormPriorLossModule(BaseComponent):
    """Calculates a loss for the default snowstorm configuration:

    May also be used to impose uniform priors on any other model parameters.
    Note: this prior is technically not a real uniform prior. First of all,
    only the negative log loss (~ likelihood) is computed. For a uniform prior,
    this should be:
     -LLH = - ln[1./(upper_bound - lower_bound)]
    if inside bounds and +infinity if outside.
    Here, finite values are enforced. The negative log loss is set to zero
    inside the uniform bounds, i.e. there is no effect and also no gradients
    that need to be calculated. If outside the bounds, an exponential function
    is applied. As a result, the prior applied here is not normalized, but
    allows for finite gradients if outside the boundaries.

    Absorption:                 uniform [0.9, 1.1]
    AnisotropyScale             uniform [0., 2.0]
    DOMEfficiency               uniform [0.9, 1.1]
    Scattering:                 uniform [0.9, 1.1]
    HoleIceForward_Unified_00   uniform [-2., 1.]
    HoleIceForward_Unified_01   uniform [-0.2, 0.2]
    IceWavePlusModes_XY         default (e.g. MultivariateNormal)
        Number of amplitudes: 12
        Amplitude sigmas:
            [0.00500100, 0.03900780, 0.04500900, 0.17903581, 0.07101420,
             0.30306061, 0.14502901, 0.09501900, 0.16103221, 0.13302661,
             0.15703141, 0.13302661]
        Phase_sigmas:
            [0.00000001, 0.01664937, 0.02708014, 0.43171273, 0.02351273,
             2.33565571, 0.16767628, 0.05414841, 0.31355088, 0.04227052,
             0.27955606, 4.02237848]

    A loss component that is used to compute the loss. The component
    must provide a
    loss_module.get_loss(data_batch_dict, result_tensors, tensors,
                         parameter_tensor_name='x_parameters', **kwargs)
    method.
    """

    @property
    def sigmas(self):
        if self.untracked_data is not None and "sigmas" in self.untracked_data:
            return self.untracked_data["sigmas"]
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
        super(SnowstormPriorLossModule, self).__init__(logger=self._logger)

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

        # define parameterizations
        if "sigmas" in config:
            self.untracked_data["sigmas"] = config["sigmas"]
        else:
            self.untracked_data["sigmas"] = [
                # Amplitude sigmas
                0.00500100,
                0.03900780,
                0.04500900,
                0.17903581,
                0.07101420,
                0.30306061,
                0.14502901,
                0.09501900,
                0.16103221,
                0.13302661,
                0.15703141,
                0.13302661,
                # Phase sigmas
                0.00000001,
                0.01664937,
                0.02708014,
                0.43171273,
                0.02351273,
                2.33565571,
                0.16767628,
                0.05414841,
                0.31355088,
                0.04227052,
                0.27955606,
                4.02237848,
            ]

        if "uniform_parameters" in config:
            self.untracked_data["uniform_parameters"] = config[
                "uniform_parameters"
            ]
        else:
            self.untracked_data["uniform_parameters"] = {
                "Absorption": [0.9, 1.1],
                "AnisotropyScale": [0.0, 2.0],
                "DOMEfficiency": [0.9, 1.1],
                "Scattering": [0.9, 1.1],
                "HoleIceForward_Unified_00": [-2.0, 1.0],
                "HoleIceForward_Unified_01": [-0.2, 0.2],
            }

        # create configuration object
        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings=dict(config=config),
        )

        return configuration, {}, {}

    def uniform_log_prior_loss(self, values, low, high, eps=0):
        """Computes a loss for a uniform log prior.

        Loss is zero for values within bounds and exponentially grows outside.

        Parameters
        ----------
        values : tf.Tensor
            The values at which to evaluate the loss.
        low : float
            The lower limit of the uniform prior.
        high : TYPE
            The upper limit of the uniform prior.
        eps : float, optional
            This defines the amount before low/high at which the penalty will
            begin.

        Returns
        -------
        TYPE
            Description
        """
        float_precision = self.configuration.config["config"][
            "float_precision"
        ]
        values = tf.cast(values, float_precision)
        low = tf.cast(low, float_precision)
        high = tf.cast(high, float_precision)

        if high <= low:
            msg = "Upper bound [{}] must be greater than lower bound [{}]"
            raise ValueError(msg.format(high, low))

        scale = high - low
        exp_factor = 10
        normalization = np.exp(exp_factor)

        def loss_excess(scaled_excess):
            return tf.exp((scaled_excess + 1) * exp_factor) - normalization

        loss = tf.where(
            values > high - eps,
            loss_excess((values - high) / scale),
            tf.zeros_like(values),
        )
        loss += tf.where(
            values < low + eps,
            loss_excess((low - values) / scale),
            tf.zeros_like(values),
        )
        return loss

    def get_loss(
        self,
        data_batch_dict,
        result_tensors,
        tensors,
        model,
        parameter_tensor_name="x_parameters",
        reduce_to_scalar=True,
        sort_loss_terms=False,
        **kwargs
    ):
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
        reduce_to_scalar : bool, optional
            If True, the individual terms of the log likelihood loss will be
            reduced (aggregated) to a scalar loss.
            If False, a list of tensors will be returned that contain the terms
            of the log likelihood. Note that each of the returned tensors may
            have a different shape.
        sort_loss_terms : bool, optional
            If true, the loss terms will be sorted and aggregated in three
            types of loss terms (this requires `reduce_to_scalar` == False):
                scalar: shape []
                    scalar loss for the whole batch of events
                event: shape [n_batch]
                    vector loss with one value per event
                dom: shape [n_batch, 86, 60]
                    tensor loss with one value for each DOM and event
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

        # get parameters
        parameters = model.add_parameter_indexing(
            data_batch_dict[parameter_tensor_name]
        )

        loss_terms = []

        # compute loss for uniform priors
        for name, bounds in sorted(
            self.untracked_data["uniform_parameters"].items()
        ):
            values = parameters.params[name]
            loss_terms.append(self.uniform_log_prior_loss(values, *bounds))

        # compute loss for Fourier modes
        num_sigmas = len(self.untracked_data["sigmas"])
        if num_sigmas > 0:
            start_index = model.get_index("IceWavePlusModes_00")
            end_index = (
                model.get_index("IceWavePlusModes_{:02d}".format(num_sigmas))
                + 1
            )
            assert end_index - start_index == num_sigmas
            for i, exp_index in enumerate(range(start_index, end_index)):
                index = model.get_index("IceWavePlusModes_{:02d}".format(i))
                assert exp_index == index, "{} != {}".format(exp_index, index)

            # shape: [batch, n_fourier]
            fourier_values = parameters[:, start_index:end_index]

            fourier_sigmas = tf.expand_dims(self.sigmas, axis=0)
            fourier_log_pdf = basis_functions.tf_log_gauss(
                fourier_values,
                mu=tf.zeros_like(fourier_sigmas),
                sigma=fourier_sigmas,
                dtype=self.configuration.config["config"]["float_precision"],
            )

            # we will use the negative log likelihood as loss
            fourier_loss = -fourier_log_pdf
            loss_terms.append(fourier_loss)

        if sort_loss_terms:
            event_loss = None
            for loss_term in loss_terms:
                if (loss_term.shape) > 1:
                    loss_term = tf.reduce_sum(loss_term, axis=1)
                if event_loss is None:
                    event_loss = loss_term
                else:
                    event_loss += loss_term

            dom_tensor = data_batch_dict["x_dom_charge"][..., 0]
            loss_terms = [
                tf.zeros_like(dom_tensor[0, 0, 0]),
                event_loss,
                tf.zeros_like(dom_tensor),
            ]

        if reduce_to_scalar:
            return tf.math.add_n(
                [tf.reduce_sum(loss_term) for loss_term in loss_terms]
            )
        else:
            return loss_terms
