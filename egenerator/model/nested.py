import os
import logging
import numpy as np
import tensorflow as tf

from egenerator import misc
from egenerator.model.base import Model
from egenerator.manager.component import Configuration


class NestedModel(Model):
    """Defines base class for a nested Model.

    This is an abstract class for a combination of (nested) models.
    A nested model is one that internally consists of multiple models.
    These internal models are used to build up the final model.
    The benefit of such a construction is that internal models can
    be trained separately and reused in different contexts.
    The `NestedModel` class ensures that the internal models are
    correctly configured, loaded, and saved.

    A derived class must implement
    ------------------------------
    __call__(self, *args, **kwargs) or equivalent (e.g. get_tensors)
        The method that combines the output of the nested models to
        produce the final output of the NestedModel.

    get_parameters_and_mapping(self, config, base_models):
        Get parameter names of the model input tensor models mapping.

        This is a pure virtual method that must be implemented by
        derived class.

        Parameters
        ----------
        config : dict
            A dictionary of settings.
        base_models : dict of Model objects
            A dictionary of models. These models are used as a basis for
            the NestedModel object. The final NestedModel can be made up
            of multiple models which may be created from one or more
            base model objects.

        Returns
        -------
        list of str
            A list of parameter names describing the input tensor to
            the NestedModel object. These parameter names correspond to
            the last dimension of the input tensor.
        dict
            This describes the models which compose the NestedModel.
            The dictionary is a mapping from model_name (str) to
            base_model (str). This mapping allows the reuse of a single
            model component instance. For instance, a muon can be build up
            of multiple cascades. However, all cascades should use the same
            underlying model. Hence, in this case only one base_model is
            required: the cascade model. The mapping will then map all
            cascades in the hypothesis to this one base cascade model.

    get_model_parameters(self, parameters):
        Get the input parameters for the individual models.

        Parameters
        ----------
        parameters : tf.Tensor
            The input parameters for the NestedModel object.
            The input parameters of the individual Model objects are composed
            from these.
            Shape: [..., num_parameters]

        Returns
        -------
        dict of tf.Tensor
            Returns a dictionary of (name: input_parameters) pairs, where
            name is the name of the nested Model and input_parameters
            is a tf.Tensor for the input parameters of that Model.
            Each input_parameters tensor has shape [..., num_parameters_i].

    Attributes
    ----------
    name : str
        The name of the model.

    parameter_names: list of str
        The names of the n_params number of parameters.
    """

    @property
    def epsilon(self):
        model_precision = None

        if "float_precision" in self.configuration.config:
            model_precision = self.configuration.config["float_precision"]
        elif "config" in self.configuration.config:
            config = self.configuration.config["config"]
            if "float_precision" in config:
                model_precision = config["float_precision"]
        elif "models_mapping" in self._untracked_data:
            model_precisions = []
            for _, base_source in config["models_mapping"].items():
                model_precisions.append(
                    self.sub_components[base_source].configuration.config[
                        "config"
                    ]["float_precision"]
                )
            model_precisions = np.unique(model_precisions)
            if len(model_precisions) > 1:
                raise ValueError(
                    "Multiple float precisions in model: {}".format(
                        model_precisions
                    )
                )
            else:
                if model_precision is None:
                    model_precision = model_precisions[0]
                elif model_precision != model_precisions[0]:
                    raise ValueError(
                        "Mismatched float precisions in model: {} and {}".format(
                            model_precision, model_precisions[0]
                        )
                    )
        else:
            raise ValueError(
                f"No float precision found in configuration: {config}"
            )

        if model_precision == "float32":
            return 1e-7
        elif model_precision == "float64":
            return 1e-15
        else:
            raise ValueError(
                "Invalid float precision: {}".format(model_precision)
            )

    def __init__(self, logger=None):
        """Instantiate Model class

        Parameters
        ----------
        logger : logging.logger, optional
            The logger to use.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(NestedModel, self).__init__(logger=self._logger)

    def __call__(self, *args, **kwargs):
        """Call the model

        This is a virtual method that must be implemented by the derived
        class. The derived model must implement the appropriate accumulation
        of the results of the nested models.

        Note: for a `Source` class, the relevant method to implement
        is `get_tensors` and not `__call__`.

        Returns
        -------
        tf.Tensor
            The output tensor of the model.
        """
        raise NotImplementedError()

    def get_parameters_and_mapping(self, config, base_models):
        """Get parameter names of the model input tensor models mapping.

        This is a pure virtual method that must be implemented by
        derived class.

        Parameters
        ----------
        config : dict
            A dictionary of settings.
        base_models : dict of Model objects
            A dictionary of models. These models are used as a basis for
            the NestedModel object. The final NestedModel can be made up
            of multiple models which may be created from one or more
            base model objects.

        Returns
        -------
        list of str
            A list of parameter names describing the input tensor to
            the NestedModel object. These parameter names correspond to
            the last dimension of the input tensor.
        dict
            This describes the models which compose the NestedModel.
            The dictionary is a mapping from model_name (str) to
            base_model (str). This mapping allows the reuse of a single
            model component instance. For instance, a muon can be build up
            of multiple cascades. However, all cascades should use the same
            underlying model. Hence, in this case only one base_model is
            required: the cascade model. The mapping will then map all
            cascades in the hypothesis to this one base cascade model.
        """
        raise NotImplementedError()

    def get_model_parameters(self, parameters):
        """Get the input parameters for the individual models.

        Parameters
        ----------
        parameters : tf.Tensor
            The input parameters for the NestedModel object.
            The input parameters of the individual Model objects are composed
            from these.
            Shape: [..., num_parameters]

        Returns
        -------
        dict of tf.Tensor
            Returns a dictionary of (name: input_parameters) pairs, where
            name is the name of the nested Model and input_parameters
            is a tf.Tensor for the input parameters of that Model.
            Each input_parameters tensor has shape [..., num_parameters_i].
        """
        raise NotImplementedError()

    def _configure_derived_class(
        self,
        base_models,
        config,
        name=None,
    ):
        """Setup and configure the Model's architecture.

        After this function call, the models's architecture (weights) must
        be fully defined and may not change again afterwards.

        Parameters
        ----------
        base_models : dict of Model objects
            A dictionary of models. These models are used as a basis for
            the NestedModel object. The final NestedModel can be made up
            of multiple models which may be created from one or more
            base model objects.
        config : dict
            A dictionary of settings which is used to set up the
            NestedModel's architecture and weights.
        name : str, optional
            The name of the model.

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

        if "float_precision" in config:
            for _, base_model in base_models.items():
                base_config = base_model.configuration.config
                if "float_precision" in base_config:
                    base_precision = base_config["float_precision"]
                elif (
                    "config" in base_config
                    and "float_precision" in base_config["config"]
                ):
                    base_precision = base_config["config"]["float_precision"]
                else:
                    raise ValueError(
                        "No float precision found in configuration: {}".format(
                            base_config
                        )
                    )
                if base_precision != config["float_precision"]:
                    raise ValueError(
                        "Mismatched float precisions in model: {} and {}".format(
                            base_precision,
                            config["float_precision"],
                        )
                    )

        # # collect all tensorflow variables before creation
        # variables_before = set([
        #     v.ref() for v in tf.compat.v1.global_variables()])

        # build architecture: create and save model weights
        # returns parameter_names and models mapping
        parameter_names, models_mapping = self.get_parameters_and_mapping(
            config, base_models
        )

        sub_components = base_models

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
        self._untracked_data["name"] = name
        self._untracked_data["models_mapping"] = models_mapping
        self._set_parameter_names(parameter_names)

        # create configuration object
        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings=dict(config=config),
            mutable_settings=dict(name=name),
        )

        return configuration, {}, sub_components

    def save_weights(
        self,
        dir_path,
        max_keep=3,
        protected=False,
        description=None,
        num_training_steps=None,
    ):
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
                    dir_path=sub_dir_path,
                    max_keep=max_keep,
                    protected=protected,
                    description=description,
                    num_training_steps=num_training_steps,
                )

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
                sub_component.load_weights(
                    dir_path=sub_dir_path, checkpoint_number=checkpoint_number
                )

    def _save(self, dir_path, **kwargs):
        """Virtual method for additional save tasks by derived class

        This is a virtual method that may be overwritten by derived class
        to perform additional tasks necessary to save the component.
        This can for instance be saving of tensorflow model weights.

        The NestedModel only contains weights in its submodules which are
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

        The NestedModel only contains weights in its submodules which are
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
        if not self.is_configured:

            # save temporary values to make sure these aren't modified
            configuration_id = id(self.configuration)
            sub_components_id = id(self.sub_components)
            configuration = Configuration(**self.configuration.dict)
            data = dict(self.data)
            sub_components = dict(self.sub_components)

            # rebuild graph
            config_dict = self.configuration.config
            if "data_trafo" in self._sub_components:
                data_trafo = self._sub_components["data_trafo"]
            else:
                data_trafo = None
            base_models = {}
            for key, sub_component in self._sub_components.items():
                if key != "data_trafo":
                    base_models[key] = sub_component

            self._configure(
                data_trafo=data_trafo, base_models=base_models, **config_dict
            )

            # make sure that no additional class attributes are created
            # apart from untracked ones
            self._check_member_attributes()

            # make sure the other values weren't overwritten
            if (
                not configuration.is_compatible(self.configuration)
                or configuration_id != id(self.configuration)
                or data != self.data
                or sub_components != self.sub_components
                or sub_components_id != id(self.sub_components)
            ):
                raise ValueError("Tracked components were changed!")

    def check_model_parameter_creation(self):
        """Check created model input parameters

        This will check the created model input parameters for obvious
        errors. Passing this check does not guarantee correctness, but will
        ensure that all models obtain input parameters which are only based
        on the NestedModel input parameters.
        """

        # create a dummy input tensor for the NestedModel
        input_tensor = tf.ones(
            [3, self.num_parameters], name="NestedModelInput"
        )

        # get model parameters
        input_tensor = self.add_parameter_indexing(input_tensor)
        model_parameters = self.get_model_parameters(input_tensor)

        # check if each specified model has the correct amount of input
        # parameters and if these are only based on the NestedModelInput
        for name, base in self._untracked_data["models"].items():

            # get base component
            sub_component = self.sub_components[base]

            # get parameters for this model
            model_parameters_i = model_parameters[name]

            # check number of parameters
            if model_parameters_i.shape[-1] != sub_component.num_parameters:
                msg = "Model {!r} with base component {!r} expected {!r} "
                msg += "number of parameters but got {!r}"
                raise ValueError(
                    msg.format(
                        name,
                        base,
                        sub_component.num_parameters,
                        model_parameters_i.shape[-1],
                    )
                )

            # check input tensor dependency of input
            try:
                # get parent tensor
                top_nodes = self._find_top_nodes(
                    model_parameters_i, input_tensor.name
                )
                if top_nodes == set([input_tensor.name]):
                    continue

                for i in range(sub_component.num_parameters):
                    tensor_i = model_parameters_i[:, i]

                    # get parent tensor
                    top_nodes = self._find_top_nodes(
                        tensor_i, input_tensor.name
                    )

                    if input_tensor.name not in top_nodes:
                        msg = "Model {!r} with base component {!r} has "
                        msg += "an input tensor component {!r} ({!r}) that "
                        msg += "does not depend on NestedModelInput!"
                        raise ValueError(
                            msg.format(
                                name, base, i, sub_component.get_name(i)
                            )
                        )

                    for node in top_nodes:
                        if node != input_tensor.name:
                            msg = "Model {!r} with base component {!r} has "
                            msg += "an input tensor component {!r} ({!r}) "
                            msg += "that depends on {!r} instead of the "
                            msg += "NestedModelInput {!r}!"
                            raise ValueError(
                                msg.format(
                                    name,
                                    base,
                                    i,
                                    sub_component.get_name(i),
                                    node,
                                    input_tensor.name,
                                ),
                            )
            except AttributeError:
                self._logger.warning(
                    "Can not check inputs since Tensorflow is in eager mode."
                )


class ConcreteFunctionCache:
    """Concrete Function Container"""

    def __init__(
        self,
        model_parameters,
        sub_components,
        data_batch_dict,
        is_training,
        logger=None,
    ):
        self.model_parameters = model_parameters
        self.sub_components = sub_components
        self.data_batch_dict = data_batch_dict
        self.is_training = is_training
        self.concrete_tensor_funcs = {}
        self._logger = logger or logging.getLogger(__name__)

    def get_or_add_tf_func(
        self,
        model_name,
        base_name,
        func_name="__call__",
    ):
        """Retrieve concrete tf function from cache or add new one.

        Parameters
        ----------
        model_name : str
            The name of the Model object.
        base_name : str
            The name of the base model object.
        func_name : str, optional
            The name of the function to call.

        Returns
        -------
        tf.Function
            The concrete tensorflow function
        """
        if base_name not in self.concrete_tensor_funcs:
            base_model = self.sub_components[base_name]

            @tf.function
            def concrete_function(data_batch_dict_i):
                print(
                    "Tracing multi-model base: {} ({})".format(
                        base_name, base_model
                    )
                )
                func = getattr(base_model, func_name)
                return func(
                    data_batch_dict_i,
                    is_training=self.is_training,
                    parameter_tensor_name="x_parameters",
                )

            # get input parameters for Model i
            parameters_i = self.model_parameters[model_name]
            parameters_i = base_model.add_parameter_indexing(parameters_i)

            # Create data batch for this model
            data_batch_dict_i = {"x_parameters": parameters_i}
            for key, values in self.data_batch_dict.items():
                if key != "x_parameters":
                    data_batch_dict_i[key] = values

            self.concrete_tensor_funcs[base_name] = (
                concrete_function.get_concrete_function(data_batch_dict_i)
            )

        return self.concrete_tensor_funcs[base_name]