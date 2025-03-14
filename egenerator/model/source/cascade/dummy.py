import tensorflow as tf

from egenerator.model.source.base import Source


class DummyCascadeModel(Source):

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

        parameter_names = [
            "x",
            "y",
            "z",
            "zenith",
            "azimuth",
            "energy",
            "time",
        ]

        self._untracked_data["dummy_var"] = tf.Variable(1.0, name="dummy_var")

        return parameter_names

    @tf.function
    def get_tensors(
        self,
        data_batch_dict,
        is_training,
        parameter_tensor_name="x_parameters",
    ):
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

        Returns
        -------
        dict of tf.Tensor
            A dictionary of output tensors.
            This  dictionary must at least contain:

                'dom_charges':
                    The predicted charge at each DOM
                    Shape: [n_events, 86, 60]
                'dom_charges_component':
                    The predicted charge at each DOM for each component
                    Shape: [n_events, 86, 60, n_components]
                'dom_charges_variance':
                    The predicted charge variance at each DOM
                    Shape: [n_events, 86, 60]
                'dom_charges_variance_component':
                    The predicted charge variance at each DOM for each
                    component of the mixture model.
                    Shape: [n_events, 86, 60, n_components]
                'dom_charges_pdf':
                    The charge likelihood evaluated for each DOM.
                    Shape: [n_events, 86, 60]
                'pulse_pdf':
                    The likelihood evaluated for each pulse
                    Shape: [n_pulses]
                'time_offsets':
                    The global time offsets for each event.
                    Shape: [n_events]
        """
        self.assert_configured(True)

        parameters = data_batch_dict[parameter_tensor_name]
        pulses = data_batch_dict["x_pulses"]

        temp_var_reshaped = tf.reshape(
            tf.reduce_sum(parameters, axis=1), [-1, 1, 1]
        )
        dom_charges = tf.ones([1, 86, 60]) * temp_var_reshaped
        pulse_pdf = (
            tf.reduce_sum(pulses, axis=1) * self._untracked_data["dummy_var"]
        )

        tensor_dict = {
            "dom_charges_pdf": dom_charges,
            "dom_charges": dom_charges,
            "dom_charges_component": dom_charges[..., tf.newaxis],
            "dom_charges_variance": dom_charges,
            "dom_charges_variance_component": dom_charges[..., tf.newaxis],
            "pulse_pdf": pulse_pdf,
            "time_offsets": None,
        }

        return tensor_dict
