from __future__ import division, print_function
import os
import logging
import tensorflow as tf

from egenerator import misc
from egenerator.model.base import Model
from egenerator.model.source.base import Source
from egenerator.manager.component import Configuration


class MultiSource(Source):

    """Defines base class for an unbinned MultiSource.

    This is an abstract class for a combination of unbinned sources.
    Unbinned in this context, means that an unbinned likelihood is used for
    the measured pulses.

    A derived class must implement
    ------------------------------
    get_parameters_and_mapping(self, config, sources):
        Get parameter names and their ordering as well as source mapping.

        This is a pure virtual method that must be implemented by
        derived class.

        Parameters
        ----------
        config : dict
            A dictionary of settings.
        base_sources : dict of Source objects
            A dictionary of sources. These sources are used as a basis for
            the MultiSource object. The event hypothesis can be made up of
            multiple sources which may be created from one or more
            base source objects.

        Returns
        -------
        list of str
            A list of parameters of the MultiSource object.
        dict
            This describes the sources which compose the event hypothesis.
            The dictionary is a mapping from source_name (str) to
            base_source (str). This mapping allows the reuse of a single
            source component instance. For instance, a muon can be build up of
            multiple cascades. However, all cascades should use the same
            underlying model. Hence, in this case only one base_source is
            required: the cascade source. The mapping will then map all
            cascades in the hypothesis to this one base cascade source.

    get_source_parameters(self, parameters):
        Get the input parameters for the individual sources.

        Parameters
        ----------
        parameters : tf.Tensor
            The input parameters for the MultiSource object.
            The input parameters of the individual Source objects are composed
            from these.

        Returns
        -------
        dict of tf.Tensor
            Returns a dictionary of (name: input_parameters) pairs, where
            name is the name of the Source and input_parameters is a tf.Tensor
            for the input parameters of this Source.

    Attributes
    ----------
    name : str
        The name of the source.

    parameter_names: list of str
        The names of the n_params number of parameters.
    """

    def __init__(self, logger=None):
        """Instantiate Source class

        Parameters
        ----------
        logger : logging.logger, optional
            The logger to use.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(MultiSource, self).__init__(logger=self._logger)

    def get_parameters_and_mapping(self, config, base_sources):
        """Get parameter names and their ordering as well as source mapping.

        This is a pure virtual method that must be implemented by
        derived class.

        Parameters
        ----------
        config : dict
            A dictionary of settings.
        base_sources : dict of Source objects
            A dictionary of sources. These sources are used as a basis for
            the MultiSource object. The event hypothesis can be made up of
            multiple sources which may be created from one or more
            base source objects.

        Returns
        -------
        list of str
            A list of parameter names of the MultiSource object.
        dict
            This describes the sources which compose the event hypothesis.
            The dictionary is a mapping from source_name (str) to
            base_source (str). This mapping allows the reuse of a single
            source component instance. For instance, a muon can be build up of
            multiple cascades. However, all cascades should use the same
            underlying model. Hence, in this case only one base_source is
            required: the cascade source. The mapping will then map all
            cascades in the hypothesis to this one base cascade source.
        """
        raise NotImplementedError()

    def get_source_parameters(self, parameters):
        """Get the input parameters for the individual sources.

        Parameters
        ----------
        parameters : tf.Tensor
            The input parameters for the MultiSource object.
            The input parameters of the individual Source objects are composed
            from these.

        Returns
        -------
        dict of tf.Tensor
            Returns a dictionary of (name: input_parameters) pairs, where
            name is the name of the Source and input_parameters is a tf.Tensor
            for the input parameters of this Source.

        """
        raise NotImplementedError()

    def _configure_derived_class(self, base_sources, config,
                                 data_trafo=None,
                                 name=None):
        """Setup and configure the Source's architecture.

        After this function call, the sources's architecture (weights) must
        be fully defined and may not change again afterwards.

        Parameters
        ----------
        base_sources : dict of Source objects
            A dictionary of sources. These sources are used as a basis for
            the MultiSource object. The event hypothesis can be made up of
            multiple sources which may be created from one or more
            base source objects.
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
        NotImplementedError
            Description
        ValueError
            Description
        """
        if name is None:
            name = __name__

        # collect all tensorflow variables before creation
        variables_before = set(tf.compat.v1.global_variables())

        # build architecture: create and save model weights
        # returns parameter_names
        parameter_names, sources = self.get_parameters_and_mapping(
            config, base_sources)

        sub_components = base_sources
        if data_trafo is not None:
            sub_components['data_trafo'] = data_trafo

        # collect all tensorflow variables after creation and match
        variables_after = set(tf.compat.v1.global_variables())
        set_diff = variables_after - variables_before
        if set_diff != set(self.variables):
            msg = 'Found variables that are not part of the tf.Module: '
            msg += '{!r} != {!r}.'
            raise ValueError(msg.format(set_diff, set(self.variables)))

        # get names of parameters
        self._untracked_data['name'] = name
        self._untracked_data['sources'] = sources
        self._untracked_data['num_parameters'] = len(parameter_names)
        self._untracked_data['parameter_names'] = parameter_names
        self._untracked_data['parameter_name_dict'] = \
            {name: index for index, name in enumerate(parameter_names)}
        self._untracked_data['parameter_index_dict'] = \
            {index: name for index, name in enumerate(parameter_names)}

        # create configuration object
        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings=dict(config=config),
            mutable_settings=dict(name=name))

        return configuration, {}, sub_components

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
                               Shape: [-1, 86, 60, 1]
                'pulse_pdf': The likelihood evaluated for each pulse
                             Shape: [-1]

        """
        self.assert_configured(True)

        parameters = data_batch_dict[parameter_tensor_name]
        pulses = data_batch_dict['x_pulses']
        pulses_ids = data_batch_dict['x_pulses_ids']

        parameters = self.add_parameter_indexing(parameters)
        source_parameters = self.get_source_parameters(parameters)

        # -----------------------------------------------
        # get concrete functions of base sources.
        # That way tracing only needs to be applied once.
        # -----------------------------------------------
        input_signature = None

        concrete_get_tensors_funcs = {}
        for base in set(self._untracked_data['sources'].values()):
            base_source = self.sub_components[base]

            @tf.function(input_signature=input_signature)
            def concrete_function(data_batch_dict_i):
                return base_source.get_tensors(
                                data_batch_dict_i,
                                is_training=is_training,
                                parameter_tensor_name='x_parameters')
            concrete_get_tensors_funcs[base] = concrete_function
        # -----------------------------------------------

        dom_charges = None
        pulse_pdf = None
        for name, base in self._untracked_data['sources'].items():

            # get the base source
            sub_component = self.sub_components[base]

            # get input parameters for Source i
            parameters_i = source_parameters[name]
            parameters_i = sub_component.add_parameter_indexing(parameters_i)

            # Get expected DOM charge and Likelihood evaluations for source i
            data_batch_dict_i = {'x_parameters': parameters_i}
            for key, values in data_batch_dict.items():
                if key != 'x_parameters':
                    data_batch_dict_i[key] = values

            result_tensors_i = concrete_get_tensors_funcs[base](
                                                            data_batch_dict_i)

            dom_charges_i = result_tensors_i['dom_charges']
            pulse_pdf_i = result_tensors_i['pulse_pdf']

            if dom_charges_i.shape[1:] != [86, 60, 1]:
                msg = 'DOM charges of source {!r} ({!r}) have an unexpected '
                msg += 'shape {!r}.'
                raise ValueError(msg.format(name, base, dom_charges_i.shape))

            # accumulate charge
            # (assumes that sources are linear and independent)
            if dom_charges is None:
                dom_charges = dom_charges_i
            else:
                dom_charges += dom_charges_i

            # accumulate likelihood values
            # (reweight by fraction of charge of source i vs total DOM charge)
            pulse_weight_i = tf.gather_nd(tf.squeeze(dom_charges_i, axis=3),
                                          pulses_ids)
            if pulse_pdf is None:
                pulse_pdf = pulse_pdf_i * pulse_weight_i
            else:
                pulse_pdf += pulse_pdf_i * pulse_weight_i

        # normalize pulse_pdf values: divide by total charge at DOM
        pulse_weight_total = tf.gather_nd(tf.squeeze(dom_charges, axis=3),
                                          pulses_ids)

        pulse_pdf /= pulse_weight_total

        result_tensors = {
            'dom_charges': dom_charges,
            'pulse_pdf': pulse_pdf,
        }

        return result_tensors

    def save_weights(self, dir_path, max_keep=3, protected=False,
                     description=None, num_training_steps=None):
        """Save the model weights.

        Metadata on the checkpoints is stored in a model_checkpoints.yaml
        in the output directory. If it does not exist yet, a new one will be
        created. Otherwise, its values will be updated
        The file contains meta data on the checkpoints and keeps track
        of the most recents files. The structure  and content of meta data:

            latest_checkpoint: int
                The number of the latest checkpoint.

            unprotected_checkpoints:
                '{checkpoint_number}':
                    'creation_date': str
                        Time of model creation in human-readable format
                    'time_stamp': int
                        Time of model creation in seconds since
                        1/1/1970 at midnight (time.time()).
                    'file_basename': str
                        Path to the model
                    'description': str
                        Optional description of the checkpoint.

            protected_checkpoints:
                (same as unprotected_checkpoints)

        Parameters
        ----------
        dir_path : str
            Path to the output directory.
        max_keep : int, optional
            The maximum number of unprotectd checkpoints to keep.
            If there are more than this amount of unprotected checkpoints,
            the oldest checkpoints will be deleted.
        protected : bool, optional
            If True, this checkpoint will not be considered for deletion
            for max_keep.
        description : str, optional
            An optional description string that describes the checkpoint.
            This will be saved in the checkpoints meta data.
        num_training_steps : int, optional
            The number of training steps with the current training settings.
            This will be used to update the training_steps.yaml file to
            account for the correct number of training steps for the most
            recent training step.

        Raises
        ------
        IOError
            If the model checkpoint file already exists.
        KeyError
            If the model checkpoint meta data already exists.
        ValueError
            If the model has changed since it was configured.

        """
        for name, sub_component in self.sub_components.items():

            # get directory of sub component
            sub_dir_path = os.path.join(dir_path, name)

            if issubclass(type(sub_component), Model):
                # save weights of Model sub component
                sub_component.save_weights(
                    dir_path=sub_dir_path, max_keep=max_keep,
                    protected=protected, description=description,
                    num_training_steps=num_training_steps)

    def load_weights(self, dir_path, checkpoint_number=None):
        """Load the model weights.

        Parameters
        ----------
        dir_path : str
            Path to the input directory.
        checkpoint_number : None, optional
            Optionally specify a certain checkpoint number that should be
            loaded. If checkpoint_number is None (default), then the latest
            checkpoint will be loaded.

        Raises
        ------
        IOError
            If the checkpoint meta data cannot be found in the input directory.
        """
        for name, sub_component in self.sub_components.items():

            # get directory of sub component
            sub_dir_path = os.path.join(dir_path, name)

            if issubclass(type(sub_component), Model):
                # load weights of Model sub component
                sub_component.load_weights(dir_path=sub_dir_path,
                                           checkpoint_number=checkpoint_number)

    def _save(self, dir_path, **kwargs):
        """Virtual method for additional save tasks by derived class

        This is a virtual method that may be overwritten by derived class
        to perform additional tasks necessary to save the component.
        This can for instance be saving of tensorflow model weights.

        The MultiSource only contains weights in its submodules which are
        automatically saved via recursion. Therefore, it does not need
        to explicitly save anything here.

        Parameters
        ----------
        dir_path : str
            The path to the output directory to which the component will be
            saved.
        **kwargs
            Additional keyword arguments that may be used by the derived
            class.
        """
        pass

    def _load(self, dir_path, **kwargs):
        """Virtual method for additional load tasks by derived class

        This is a virtual method that may be overwritten by derived class
        to perform additional tasks necessary to load the component.
        This can for instance be loading of tensorflow model weights.

        The MultiSource only contains weights in its submodules which are
        automatically loaded via recursion. Therefore, it does not need
        to explicitly load anything here.

        Parameters
        ----------
        dir_path : str
            The path to the input directory from which the component will be
            loaded.
        **kwargs
            Additional keyword arguments that may be used by the derived
            class.
        """

        # rebuild the tensorflow graph if it does not exist yet
        self.rebuild_tensorflow_graph()

    def check_source_parameter_creation(self):
        """Check created source input parameters

        This will check the created source input parameters for obvious
        errors. Passing this check does not guarantee correctness, but will
        ensure that all sources obtain input parameters which are only based
        on the MultiSource input parameters.
        """

        # create a dummy input tensor for the MultiSource
        input_tensor = tf.ones([3, self.num_parameters],
                               name='MultiSourceInput')

        # get source parameters
        input_tensor = self.add_parameter_indexing(input_tensor)
        source_parameters = self.get_source_parameters(input_tensor)

        # check if each specified source has the correct amount of input
        # parameters and if these are only based on the MultiSourceInput
        for name, base in self._untracked_data['sources'].items():

            # get base component
            sub_component = self.sub_components[base]

            # get parameters for this source
            source_parameters_i = source_parameters[name]

            # check number of parameters
            if source_parameters_i.shape[-1] != sub_component.num_parameters:
                msg = 'Source {!r} with base component {!r} expected {!r} '
                msg += 'number of parameters but got {!r}'
                raise ValueError(msg.format(
                    name, base, sub_component.num_parameters,
                    source_parameters_i.shape[-1]))

            # check input tensor dependency of input
            try:
                # get parent tensor
                top_nodes = self._find_top_nodes(source_parameters_i,
                                                 input_tensor.name)
                if top_nodes == set([input_tensor.name]):
                    continue

                for i in range(sub_component.num_parameters):
                    tensor_i = source_parameters_i[:, i]

                    # get parent tensor
                    top_nodes = self._find_top_nodes(tensor_i,
                                                     input_tensor.name)

                    if input_tensor.name not in top_nodes:
                        msg = 'Source {!r} with base component {!r} has '
                        msg += 'an input tensor component {!r} ({!r}) that '
                        msg += 'does not depend on MultiSourceInput!'
                        raise ValueError(msg.format(name, base, i,
                                                    sub_component.get_name(i)))

                    for node in top_nodes:
                        if node != input_tensor.name:
                            msg = 'Source {!r} with base component {!r} has '
                            msg += 'an input tensor component {!r} ({!r}) '
                            msg += 'that depends on {!r} instead of the '
                            msg += 'MultiSourceInput {!r}!'
                            raise ValueError(msg.format(
                                name, base, i,
                                sub_component.get_name(i),
                                node, input_tensor.name),
                            )
            except AttributeError as e:
                self._logger.warning(
                    'Can not check inputs since Tensorflow is in eager mode.')

    def print_parameters(self, source=None):
        """Print parameters of the MultiSource model and its components.

        Parameters
        ----------
        source : name, optional
            If provided, only the parameters of this source are printed
        """
        raise NotImplementedError()

    def _find_top_nodes(self, tensor, collect_name=None):
        """Find top nodes of a tensor's computation graph.

        Parameters
        ----------
        tensor : tf.Tensor
            The tensor for which to return the top nodes.
        collect_name : str, optional
            Constant inputs are not collected, unless their names match this
            string.

        Returns
        -------
        set of tf.Tensor
            A set of tensors which make up the top nodes of the tensor's
            computation graph.
        """
        if len(tensor.op.inputs) == 0:

            # ignore constant inputs unless they match the collect_name
            if (tensor.op.node_def.op == 'Const' and
                    tensor.name == collect_name):
                return set([tensor.name])
            else:
                return set()

        tensor_list = set()
        for in_tensor in tensor.op.inputs:
            tensor_list = tensor_list.union(self._find_top_nodes(
                in_tensor, collect_name=collect_name))

        return tensor_list
