from __future__ import division, print_function
import logging
import tensorflow as tf

from tfscripts import layers as tfs

from egenerator import misc
from egenerator.model.base import Model
from egenerator.model.source.base import InputTensorIndexer
from egenerator.manager.component import Configuration


class NNMinimizerModel(Model):

    """Defines class for an NN minimizer model.

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
    def data_trafo(self):
        if self.sub_components is not None and \
                'data_trafo' in self.sub_components:
            return self.sub_components['data_trafo']
        else:
            return None

    def model_loss_module(self):
        if self.sub_components is not None and \
                'model_loss_module' in self.sub_components:
            return self.sub_components['model_loss_module']
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
    def num_points(self):
        if self.untracked_data is not None and \
                'num_points' in self.untracked_data:
            return self.untracked_data['num_points']
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
        super(NNMinimizerModel, self).__init__(logger=self._logger)

    def _configure_derived_class(self, config, data_trafo,
                                 parameter_names, name=None):
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
        parameter_names : list of string
            A list of parameter names of the Event-Generator model.
        name : str, optional
            The name of the NN minmizer.

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

        """
        if name is None:
            name = __name__

        # build architecture: this must return the tf.Module with an im

        # build architecture: create and save model weights
        # returns parameter_names
        parameter_names = self._build_architecture(config, parameter_names)

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
            settings=dict(config=config, parameter_names=parameter_names),
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

    def _build_architecture(self, config, parameter_names):
        """Set up and build architecture: create and save all model weights.

        Parameters
        ----------
        config : dict
            A dictionary of settings that fully defines the architecture.
        parameter_names : list of str
            A list of parameter names of the Event-Generator model.

        Returns
        -------
        list of str
            A list of parameter names. These parameters fully describe the
            source hypothesis. The model expects the hypothesis tensor input
            to be in the same order as this returned list.

        Raises
        ------
        ValueError
            Description

        """
        self.assert_configured(False)

        parameter_names = list(parameter_names) + [p + '_unc' for
                                                   p in parameter_names]

        num_points = int(
            config['proposal_layers_config']['fc_sizes'][-1] /
            (len(parameter_names) / 2))
        if (num_points * (len(parameter_names) / 2) !=
                config['proposal_layers_config']['fc_sizes'][-1]):
            raise ValueError(
                'Last layer of proposal_layers must have a multitude of n '
                'nodes where n is the number of parameters')
        self._untracked_data['num_points'] = num_points

        # build fully-connected (dense) layers that define at which points
        # to evaluate the SourceModel
        self._untracked_data['proposal_layers'] = tfs.FCLayers(
            input_shape=[-1, len(parameter_names)],
            name='proposal_layer',
            verbose=True,
            **config['proposal_layers_config']
        )

        # build fully-connected (dense) layers that interpret evaluated points
        # and put together a new seed and uncertainty
        self._untracked_data['interpretation_layers'] = tfs.FCLayers(
            input_shape=[-1, num_points + len(parameter_names)],
            name='interpretation_layer',
            verbose=True,
            **config['interpretation_layers_config']
        )

        return parameter_names

    def setup_model_loss_function(self, model_manager):
        """Define and save the model loss function.

        Parameters
        ----------
        model_manager : SourceManager
            The manager of the Event-Generator source for which to define
            the loss function.
        """

        # a very basic sanity check
        assert model_manager.models[0].parameter_names == self.parameter_names

        # get a concrete loss function of Event-Generator model
        data_batch_signature = []
        for tensor in model_manager.data_handler.tensors.list:
            if tensor.exists:
                shape = tf.TensorShape(tensor.shape)
            else:
                shape = tf.TensorShape(None)
            data_batch_signature.append(tf.TensorSpec(
                shape=shape,
                dtype=getattr(tf, tensor.dtype)))

        data_batch_signature = tuple(data_batch_signature)

        get_model_loss = model_manager.get_concrete_function(
            function=model_manager.get_loss,
            input_signature=data_batch_signature,
            loss_module=model_loss_module,
            is_training=False)
        self._untracked_data['get_model_loss'] = get_model_loss

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

                'parameters': the predicted hypothesis parameters.
                              Shape: [num_parameters]
                'parameters_unc': the uncertainty on the predicted
                                  hypothesis parameters.
                                  Shape: [num_parameters]
        """
        self.assert_configured(True)

        if 'get_model_loss' not in self._untracked_data:
            raise ValueError(
                'You must first setup a model loss function via '
                'setup_model_loss_function(model_manager)')

        config = self.configuration.config['config']
        dtype = getattr(tf, config['float_precision'])
        x_dom_charge = data_batch_dict['x_dom_charge']

        # only allow a batch size of 1 for now
        x_dom_charge = tf.ensure_shape(x_dom_charge, [1, 86, 60, 1])

        # num_features = parameters.get_shape().as_list()[-1]
        num_events = x_dom_charge.get_shape().as_list()[0]
        if num_events != 1:
            raise NotImplementedError('Only supports batch size of 1')

        # create initial guess
        parameters_trafo = tf.zeros(
            (num_events, self.num_parameters / 2), dtype=dtype)
        parameters_unc_trafo = tf.ones(
            (num_events, self.num_parameters / 2), dtype=dtype)

        initial_seed = tf.concat((parameters_trafo, parameters_unc_trafo),
                                 axis=-1)
        print('initial_seed', initial_seed)

        # run refinement block
        for i in range(config['num_refinement_blocks']):
            initial_seed = self.refinement_block(
                initial_seed, data_batch_dict, is_training,
                parameter_tensor_name)
        result = initial_seed
        print('result', result)

        # put the result together
        parameters_trafo = result[..., 0:self.num_parameters/2]
        parameters_unc_trafo = result[..., self.num_parameters/2:]

        # undo transformation
        parameters = self.data_trafo.inverse_transform(
            parameters_trafo, tensor_name=parameter_tensor_name)
        parameters_unc = self.data_trafo.inverse_transform(
            parameters_unc_trafo,
            tensor_name=parameter_tensor_name,
            bias_correction=False,
        )

        tensor_dict = {}
        tensor_dict['parameters'] = parameters
        tensor_dict['parameters_unc'] = parameters_unc
        tensor_dict['parameters_trafo'] = parameters_trafo
        tensor_dict['parameters_unc_trafo'] = parameters_unc_trafo

        return tensor_dict

    def refinement_block(self, seed, data_batch_dict, is_training,
                         parameter_tensor_name='x_parameters'):
        """Defines one refinement block to update the input seed.

        Parameters
        ----------
        seed : tf:Tensor
            Shape: [-1, 2*num_parameters]
        data_batch_dict : dict of tf.Tensor
            A dictionary with the tensors from the data batch.
        is_training : bool, optional
            Indicates whether currently in training or inference mode.
            Must be provided if batch normalisation is used.
            True: in training mode
            False: inference mode.
        parameter_tensor_name : str, optional
            The name of the parameter tensor to use. Default: 'x_parameters'

        Returns
        -------
        tf.Tensor
            The refined and updated seed.
            Shape: [-1, 2*num_parameters]
        """
        config = self.configuration.config['config']

        # run proposal layer
        proposals = self._untracked_data['proposal_layers'](
            seed,
            is_training=is_training,
            keep_prob=config['keep_prob'],
        )[-1]

        proposals = tf.reshape(
            proposals, [-1, self.num_points, self.num_parameters / 2])

        print('proposals', proposals)

        # get loss from Event-Generator model for each of these proposals
        # Todo: figure out a proper way to handle tensors without having to
        # loop through individual proposals
        # Todo: make this support a batch size != 1
        loss_results = []
        for index in range(self.num_points):
            data_batch = []
            for i, name in enumerate(self.data_handler.tensors.names):
                if name == parameter_tensor_name:
                    data_batch.append(proposals[:, index, :])
                else:
                    data_batch.append(data_batch_dict[name])

            # Warning: this only works for a batch size of 1, since
            # loss is provided as a scalar!
            loss_results.append(
                self._untracked_data['get_model_loss'](data_batch))

        loss_results = tf.stack(loss_results, axis=1)
        loss_results = tf.ensure_shape(loss_results, [1, self.num_points])

        # stack on initial seed
        interp_input = tf.concat((seed, loss_results), axis=-1)
        print('interp_input', interp_input)

        # run interpretation layer
        refined_seed = self._untracked_data['interpretation_layers'](
            interp_input,
            is_training=is_training,
            keep_prob=config['keep_prob'],
        )[-1]

        print('refined_seed', refined_seed)

        return refined_seed
