from __future__ import division, print_function
import logging
import numpy as np
import tensorflow as tf

from egenerator import misc
from egenerator.manager.component import BaseComponent, Configuration
from egenerator.utils.basis_functions import tf_gauss


class SnowstormPriorLossModule(BaseComponent):

    """Calculates a loss for the default snowstorm configuration:

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
                         parameter_tensor_name='x_parameters')
    method.
    """

    @property
    def sigmas(self):
        if (self.untracked_data is not None and
                'sigmas' in self.untracked_data):
            return self.untracked_data['sigmas']
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
        self.untracked_data['sigmas'] = [

            # Amplitude sigmas
            0.00500100, 0.03900780, 0.04500900, 0.17903581, 0.07101420,
            0.30306061, 0.14502901, 0.09501900, 0.16103221, 0.13302661,
            0.15703141, 0.13302661,

            # Phase sigmas
            0.00000001, 0.01664937, 0.02708014, 0.43171273, 0.02351273,
            2.33565571, 0.16767628, 0.05414841, 0.31355088, 0.04227052,
            0.27955606, 4.02237848
        ]

        self.untracked_data['uniform_parameters'] = {
            'Absorption': [0.9, 1.1],
            'AnisotropyScale': [0., 2.0],
            'DOMEfficiency': [0.9, 1.1],
            'Scattering': [0.9, 1.1],
            'HoleIceForward_Unified_00': [-2., 1.],
            'HoleIceForward_Unified_01': [-0.2, 0.2],
        }

        # create configuration object
        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings=dict(config=config))

        return configuration, {}, {}

    def uniform_prior_loss(self, values, low, high):
        """Computes a loss for a uniform prior.

        Loss is zero for values within bounds and exponentially grows outside.

        Parameters
        ----------
        values : tf.Tensor
            The values at which to evaluate the loss.
        low : float
            The lower limit of the uniform prior.
        high : TYPE
            The upper limit of the uniform prior.
        """
        scale = high - low
        exp_factor = 5
        normalization = np.exp(exp_factor)

        def loss_excess(scaled_excess):
            return tf.exp((scaled_excess + 1)*exp_factor) - normalization

        loss = tf.where(values > high,
                        loss_excess((values - high) / scale),
                        tf.zeros_like(values))
        loss += tf.where(values < low,
                         loss_excess((low - values) / scale),
                         tf.zeros_like(values))
        return loss

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

        Returns
        -------
        tf.Tensor
            Scalar loss
            Shape: []
        """

        # get parameters
        parameters = model.add_parameter_indexing(
            data_batch_dict[parameter_tensor_name])

        event_loss = None

        # compute loss for uniform priors
        for name, bounds in self.untracked_data['uniform_parameters'].items():
            values = parameters.params[name]
            if event_loss is None:
                event_loss = self.uniform_prior_loss(values, *bounds)
            else:
                event_loss += self.uniform_prior_loss(values, *bounds)

        # compute loss for Fourier modes
        start_index = model.get_index('IceWavePlusModes_00')
        end_index = model.get_index('IceWavePlusModes_23') + 1
        assert end_index - start_index == 24
        for i, exp_index in enumerate(range(start_index, end_index)):
            index = model.get_index('IceWavePlusModes_{:02d}'.format(i))
            assert exp_index == index, '{} != {}'.format(exp_index, index)

        # shape: [batch, n_fourier]
        fourier_values = parameters[:, start_index:end_index]

        fourier_sigmas = tf.expand_dims(self.sigmas, axis=0)
        fourier_pdf = tf_gauss(fourier_values,
                               mu=tf.zeros_like(fourier_sigmas),
                               sigma=fourier_sigmas)

        # we will use the negative log likelihood as loss
        fourier_loss = -tf.math.log(fourier_pdf)
        event_loss += tf.reduce_sum(fourier_loss, axis=1)

        return tf.reduce_sum(event_loss)
