from __future__ import division, print_function
import logging
import numpy as np
import tensorflow as tf

from egenerator import misc
from egenerator.utils import basis_functions
from egenerator.model.base import Model
from egenerator.manager.component import Configuration


class InputTensorIndexer(dict):
    """A simple wrapper to easily obtain named tensor slices
    """
    def __init__(self, tensor, names):

        # sanity check
        if tensor.shape[-1] != len(names):
            raise ValueError('Shapes do not match up: {!r} != {!r}'.format(
                tensor.shape[-1], len(names)))

        dictionary = {}
        for i, name in enumerate(names):
            dictionary[name] = tensor[..., i]

        dict.__init__(self, dictionary)
        self.__dict__ = self


class Source(Model):

    """Defines base class for an unbinned Source.

    This is an abstract class for an unbinned source. Unbinned in this context,
    means that an unbinned likelihood is used for the measured pulses.

    A derived class must implement
    ------------------------------
    _build_architecture(self, config, name=name):
        builds architecture and creates all model weights
        returns parameter_names

    get_tensors(self, data_batch_dict, is_training, parameter_tensor_name):
        The data_batch_dict is a dictionary of named inputs. This dictionary
        must contain at least the following keys:
            parameters, pulses, pulses_ids
        Parameters are the hypothesis tensor of the source with
        shape [-1, n_params]. The get_tensors method must compute all tensors
        that are to be used in later steps. It returns these as a dictionary
        of output tensors. This  dictionary must at least contain:

            'dom_charges': the predicted charge at each DOM
                           Shape: [-1, 86, 60, 1]
            'pulse_pdf': The likelihood evaluated for each pulse
                         Shape: [-1]

    Attributes
    ----------
    data_trafo : DataTrafo
        The data transformation object.

    name : str
        The name of the source.

    parameter_names : list of str
        The names of the n_params number of parameters.
    """

    @property
    def time_unit_in_ns(self):
        return 1000.

    @property
    def data_trafo(self):
        if self.sub_components is not None and \
                'data_trafo' in self.sub_components:
            return self.sub_components['data_trafo']
        else:
            return None

    @property
    def name(self):
        if self.untracked_data is not None and \
                'name' in self.untracked_data:
            return self.untracked_data['name']
        else:
            return None

    @property
    def parameter_names(self):
        if self.untracked_data is not None and \
                'parameter_names' in self.untracked_data:
            return self.untracked_data['parameter_names']
        else:
            return None

    @property
    def num_parameters(self):
        if self.untracked_data is not None and \
                'num_parameters' in self.untracked_data:
            return self.untracked_data['num_parameters']
        else:
            return None

    def __init__(self, logger=None):
        """Instantiate Source class

        Parameters
        ----------
        logger : logging.logger, optional
            The logger to use.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(Source, self).__init__(logger=self._logger)

    def _configure_derived_class(self, config, data_trafo, name=None):
        """Setup and configure the Source's architecture.

        After this function call, the sources's architecture (weights) must
        be fully defined and may not change again afterwards.

        Parameters
        ----------
        config : dict
            A dictionary of settings which is used to set up the model
            architecture and weights.
        data_trafo : DataTrafo
            A data trafo object.
        name : str, optional
            The name of the source.

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
            Return None, if the component has no data that needs to be tracked.
        dict
            A dictionary of dependent sub components. This is a dictionary
            of sub components that need to be saved and loaded recursively
            when the component is saved and loaded.
            Return None if no dependent sub components exist.

        Raises
        ------
        ValueError
            Description
        """

        # check that all created variables are part of the module.variables

        """Build the architecture. This must return:

            tf.Module:
                - Must contain all tensorflow variables as class attributes,
                    so that they are found by tf.Module.variables
                - Must implement:
                     __call__(self, input_params)
                    which returns a dictionary with output tensors. This
                    dictionary must at least contain
                        'dom_charges': the predicted charge at each DOM
                                       Shape: [-1, 86, 60, 1]
                        'latent_var_mu': Shape: [-1, 86, 60, n_models]
                        'latent_var_sigma': Shape: [-1, 86, 60, n_models]
                        'latent_var_r': Shape: [-1, 86, 60, n_models]
                        'latent_var_scale': Shape: [-1, 86, 60, n_models]

        """
        if name is None:
            name = __name__

        # # collect all tensorflow variables before creation
        # variables_before = set([
        #     v.ref() for v in tf.compat.v1.global_variables()])

        # build architecture: create and save model weights
        # returns parameter_names
        parameter_names = self._build_architecture(config, name=name)

        # # collect all tensorflow variables after creation and match
        # variables_after = set([
        #     v.ref() for v in tf.compat.v1.global_variables()])

        # set_diff = variables_after - variables_before
        # model_variables = set([v.ref() for v in self.variables])
        # new_unaccounted_variables = set_diff - model_variables
        # if len(new_unaccounted_variables) > 0:
        #     msg = 'Found new variables that are not part of the tf.Module: {}'
        #     raise ValueError(msg.format(new_unaccounted_variables))

        # get names of parameters
        self._untracked_data['name'] = name
        self._untracked_data['num_parameters'] = len(parameter_names)
        self._untracked_data['parameter_names'] = parameter_names
        self._untracked_data['parameter_name_dict'] = \
            {name: index for index, name in enumerate(parameter_names)}
        self._untracked_data['parameter_index_dict'] = \
            {index: name for index, name in enumerate(parameter_names)}

        # Add parameter names to __dict__?

        # create configuration object
        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings=dict(config=config),
            mutable_settings=dict(name=name))

        return configuration, {}, {'data_trafo': data_trafo}

    def get_index(self, param_name):
        """Returns the index of a parameter name

        Parameters
        ----------
        param_name : str
            The name of the input parameter for which to return the index
        """
        self.assert_configured(True)
        return self._untracked_data['parameter_name_dict'][param_name]

    def get_name(self, index):
        """Returns the name of the input parameter input_parameters[:, index].

        Parameters
        ----------
        index : int
            The parameter input index for which to return the name.

        Raises
        ------
        NotImplementedError
            Description
        """
        self.assert_configured(True)
        return self._untracked_data['parameter_index_dict'][index]

    def add_parameter_indexing(self, tensor):
        """Add meta data to a tensor and allow easy indexing via names.

        Parameters
        ----------
        tensor : tf.Tensor
            The input parameter tensor for the Source.
        """
        setattr(tensor, 'params', InputTensorIndexer(
                    tensor, self._untracked_data['parameter_names']))
        return tensor

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
        raise NotImplementedError()

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
        data_batch_dict: dict of tf.Tensor
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
        NotImplementedError
            Description

        Returns
        -------
        dict of tf.Tensor
            A dictionary of output tensors.
            This  dictionary must at least contain:

                'dom_charges': the predicted charge at each DOM
                               Shape: [-1, 86, 60, 1]
                'pulse_pdf': The likelihood evaluated for each pulse
                             Shape: [-1]
        """
        self.assert_configured(True)
        raise NotImplementedError()

    def pdf(self, x, result_tensors, **kwargs):
        """Compute PDF values at x for given result_tensors

        This is a numpy, i.e. not tensorflow, method to compute the PDF based
        on a provided `result_tensors`. This can be used to investigate
        the generated PDFs.

        Note: this function only works for sources that use asymmetric
        Gaussians to parameterize the PDF. The latent values of the AG
        must be included in the `result_tensors`.

        Note: the PDF does not set values inside excluded time windows to zero,
        but it does adjust the normalization. It is assumed that pulses will
        already be masked before evaluated by Event-Generator. Therefore, an
        extra check for exclusions is not performed due to performance issues.

        Parameters
        ----------
        x : array_like
            The times in ns at which to evaluate the result tensors.
            Shape: () or [n_points]
        result_tensors : dict of tf.tensor
            The dictionary of output tensors as obtained from `get_tensors`.
        **kwargs
            Keyword arguments.

        Returns
        -------
        array_like
            The PDF values at times x for the given event hypothesis and
            exclusions that were used to compute `result_tensors`.
            Shape: [n_events, 86, 60, n_points]

        Raises
        ------
        NotImplementedError
            If assymetric Gaussian latent variables are not present in
            `result_tensors` dictionary.
        """

        x = np.atleast_1d(x)
        assert len(x.shape) == 1, x.shape

        # shape: [1, 1, 1, 1, n_points]
        x = np.reshape(x, (1, 1, 1, 1, -1))

        if result_tensors['time_offsets'] is not None:
            # shape: [n_events]
            t_offsets = result_tensors['time_offsets'].numpy()
            # shape: [n_events, 1, 1, 1, 1]
            t_offsets = np.reshape(t_offsets, [-1, 1, 1, 1, 1])
            # shape: [n_events, 1, 1, 1, n_points]
            x = x - t_offsets

        # internally we are working with different time units
        x = x / self.time_unit_in_ns

        # Check if the asymmetric Gaussian latent variables exist
        for latent_name in ['mu', 'scale', 'sigma', 'r']:
            if 'latent_var_' + latent_name not in result_tensors:
                msg = 'PDF evaluation is not supported for this model: {}'
                raise NotImplementedError(msg.format(
                    self._untracked_data['name']))

        # extract values
        # shape: [n_events, 86, 60, n_components, 1]
        mu = result_tensors['latent_var_mu'].numpy()[..., np.newaxis]
        scale = result_tensors['latent_var_scale'].numpy()[..., np.newaxis]
        sigma = result_tensors['latent_var_sigma'].numpy()[..., np.newaxis]
        r = result_tensors['latent_var_r'].numpy()[..., np.newaxis]

        # shape: [n_events, 86, 60, n_components, n_points]
        mixture_pdf = basis_functions.asymmetric_gauss(
            x=x, mu=mu, sigma=sigma, r=r
        )

        # uniformly scale up pdf values due to excluded regions
        if 'dom_cdf_exclusion' in result_tensors:

            # shape: [n_events, 86, 60, n_components]
            dom_cdf_exclusion = result_tensors['dom_cdf_exclusion'].numpy()

            # shape: [n_events, 86, 60, n_components, 1]
            dom_cdf_exclusion = dom_cdf_exclusion[..., np.newaxis]
            scale /= (1. - dom_cdf_exclusion + 1e-3)

        # shape: [n_events, 86, 60, n_points]
        pdf_values = np.sum(mixture_pdf * scale, axis=3)

        return pdf_values

    # def _get_top_node(self):
    #     """Helper function to get the top

    #     Returns
    #     -------
    #     TYPE
    #         Description
    #     """
    #     parent_node = tf.constant(np.ones(shape=[1, self.num_parameters]))
    #     output = self._untracked_data['module'](parent_node)
    #     return _find_top_nodes(output)

    # def _find_top_nodes(self, tensor):
    #     if len(tensor.op.inputs) == 0:
    #         return set([tensor])

    #     tensor_list = set()
    #     for in_tensor in tensor.op.inputs:
    #         tensor_list = tensor_list.union(find_top_nodes(in_tensor))

    #     return tensor_list
