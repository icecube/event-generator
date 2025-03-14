import os
import tensorflow as tf
import numpy as np
import glob
from datetime import datetime
import time

from egenerator.settings.yaml import yaml_loader, yaml_dumper
from egenerator.manager.component import BaseComponent, Configuration
from egenerator.utils import tf_helpers


class InputTensorIndexer(dict):
    """A simple wrapper to easily obtain named tensor slices"""

    def __init__(self, tensor, names):

        # sanity check
        if tensor.shape[-1] != len(names):
            raise ValueError(
                "Shapes do not match up: {!r} != {!r}".format(
                    tensor.shape[-1], len(names)
                )
            )

        dictionary = {}
        for i, name in enumerate(names):
            dictionary[name] = tensor[..., i]

        dict.__init__(self, dictionary)
        self.__dict__ = self


class Model(tf.Module, BaseComponent):
    """Model base class

    This is an abstract  base class to create a Model instance. A model
    instance inherits from tf.Module and
    egenerator.manager.component.BaseComponent. As a consequence, settings
    and tensorflow variables are automatically tracked.

    The model architecture (weights) must be defined in the _configure
    (_configure_derived_class) method and only there. After the Model's
    configure() method is called, the trainable variables of the model must
    all be created. Only the variables created at this point, will be saved
    to file in later calls to save() or save_weights().

    A derived class of the abstract Model class must at least implement the
    virtual _configure_derived_class method.
    """

    @property
    def checkpoint(self):
        if (
            self.untracked_data is not None
            and "checkpoint" in self.untracked_data
        ):
            return self.untracked_data["checkpoint"]
        else:
            return None

    @property
    def step(self):
        if self.untracked_data is not None and "step" in self.untracked_data:
            return self.untracked_data["step"]
        else:
            return None

    @property
    def num_variables(self):
        return self._count_number_of_variables()

    @property
    def name(self):
        if self.untracked_data is not None and "name" in self.untracked_data:
            return self.untracked_data["name"]
        else:
            return None

    @property
    def parameter_names(self):
        if (
            self.untracked_data is not None
            and "parameter_names" in self.untracked_data
        ):
            return self.untracked_data["parameter_names"]
        else:
            return None

    @property
    def n_parameters(self):
        if (
            self.untracked_data is not None
            and "n_parameters" in self.untracked_data
        ):
            return self.untracked_data["n_parameters"]
        else:
            return None

    def _set_parameter_names(self, parameter_names):
        """Set the names of the input parameters of the model

        Parameters
        ----------
        parameter_names : list of str
            The names of the input parameters of the model.
        """
        self.assert_configured(False)

        self._untracked_data["n_parameters"] = len(parameter_names)
        self._untracked_data["parameter_names"] = parameter_names
        self._untracked_data["parameter_name_dict"] = {
            name: index for index, name in enumerate(parameter_names)
        }
        self._untracked_data["parameter_index_dict"] = {
            index: name for index, name in enumerate(parameter_names)
        }

    def get_index(self, param_name):
        """Returns the index of a parameter name

        Parameters
        ----------
        param_name : str
            The name of the input parameter for which to return the index
        """
        self.assert_configured(True)
        self.assert_paramerters_set()
        return self._untracked_data["parameter_name_dict"][param_name]

    def get_name(self, index):
        """Returns the name of the input parameter input_parameters[..., index].

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
        self.assert_paramerters_set()
        return self._untracked_data["parameter_index_dict"][index]

    def add_parameter_indexing(self, tensor, parameter_names=None):
        """Add meta data to a tensor and allow easy indexing via names.

        Parameters
        ----------
        tensor : tf.Tensor
            The input parameter tensor for the Source.
        """
        if parameter_names is None:
            self.assert_paramerters_set()
            parameter_names = self.parameter_names

        setattr(
            tensor,
            "params",
            InputTensorIndexer(tensor, parameter_names),
        )
        return tensor

    @property
    def float_precision(self):
        model_precision = None
        if "float_precision" in self.configuration.config:
            model_precision = self.configuration.config["float_precision"]
        elif "config" in self.configuration.config:
            config = self.configuration.config["config"]
            if "float_precision" in config:
                model_precision = config["float_precision"]
        return model_precision

    @property
    def epsilon(self):
        if self.float_precision is None:
            return None

        if self.float_precision == "float32":
            return 1e-7
        elif self.float_precision == "float64":
            return 1e-15
        else:
            raise ValueError(
                f"Invalid float precision: {self.float_precision}"
            )

    def __init__(self, name=None, logger=None):
        """Initializes Model object.

        Parameters
        ----------
        name : str, optional
            Name of the model. This gets passed on to tf.Module.__init__
        logger : logging.logger, optional
            A logging instance.
        """
        if name is None:
            name = self.__class__.__name__
        BaseComponent.__init__(self, logger=logger)
        tf.Module.__init__(self, name=name)

    def _configure(self, **kwargs):
        """Configure and setup the model architecture.

        The model's architecture (weights) must be fully defined at this point
        and must not change at a later stage. Any variables created after
        the _configure() method will not be saved in a call to save() or
        save_weights().

        Parameters
        ----------
        **kwargs
            Keyword arguments that are passed on to virtual method
            _configure_derived_class.

        Returns
        -------
        Configuration object
            The configuration object of the newly configured component.
            This does not need to include configurations of sub components
            which are passed as parameters into the configure method,
            as these are automatically gathered. The dependent_sub_components
            may also be left empty for these passed sub components.
            Nested sub components or sub components created within
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
            This must at least contain the tensor list which must be
            stored under the key 'tensors'.
        dict
            A dictionary of dependent sub components. This is a dictionary
            of sub components that need to be saved and loaded recursively
            when the component is saved and loaded.
            Return None if no dependent sub components exist.
        """
        # get name of model
        if "name" in self._untracked_data:
            name = self._untracked_data["name"]
        else:
            name = self.__class__.__name__

        # collect variables of all sub components
        variables_of_sub_components = list(self.variables)

        # configure and setup model architecture [can add variables]
        configuration, component_data, sub_components = (
            self._configure_derived_class(**kwargs)
        )

        # get newly created variables
        self._untracked_data["variables_top_level"] = [
            v
            for v in self.variables
            if not tf_helpers.is_in(v, variables_of_sub_components)
        ]

        # create step counter for this object
        self._untracked_data["step"] = tf.Variable(
            0, trainable=False, dtype=tf.int64, name=name + "_step"
        )

        # create a tensorflow checkpoint object and keep track of variables
        checkpoint_vars = {"step": self._untracked_data["step"]}
        if len(self._untracked_data["variables_top_level"]) > 0:
            checkpoint_vars["model"] = self._untracked_data[
                "variables_top_level"
            ]
        self._untracked_data["checkpoint"] = tf.train.Checkpoint(
            **checkpoint_vars
        )

        # collect any variables from sub_components if they aren't already
        all_variables = list(self.variables)
        for name, sub_component in sorted(sub_components.items()):
            if issubclass(type(sub_component), tf.Module):
                all_variables.extend(
                    [
                        v
                        for v in sub_component.variables
                        if not tf_helpers.is_in(v, all_variables)
                    ]
                )
        self._untracked_data["variables"] = tuple(all_variables)

        num_vars, num_total_vars = self._count_number_of_variables()
        msg = f"\nNumber of Model Variables for {name}:\n"
        msg = f"\tFree: {num_vars}\n"
        msg += f"\tTotal: {num_total_vars}"
        self._logger.info(msg)

        return configuration, component_data, sub_components

    def _configure_derived_class(self, **kwargs):
        """Setup and configure the model's architecture.

        After this function call, the model's architecture (weights) must
        be fully defined and may not change again afterwards.
        This method needs to be implemented by derived class.

        Parameters
        ----------
        **kwargs
            Keyword arguments that are passed on to virtual method
            _configure_derived_class.

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
        raise NotImplementedError()

    def _update_sub_components(self, names):
        """Update settings which are based on the modified sub component.

        During loading of a component, sub components may be changed with a
        new and modified (but compatible) version. This allows the alteration
        of mutable settings.
        Some settings or data of a component may depend on mutable settings
        of a sub component. If these are not saved and retrieved directly from
        the sub component, they will not be automatically updated.
        This method is triggered when a sub component with the name 'name'
        is updated. It allows to update settings and data that depend on the
        modified sub component.

        Enforcing a derived class to implement this method (even if it is a
        simple 'pass' in the case of no dependent settings and data)
        will ensure that the user is aware of the issue.

        A good starting point to obtain an overview of which settings may need
        to be modified, is to check the _configure method. Any settings and
        data set there might need to be updated.

        Parameters
        ----------
        names : list of str
            The names of the sub components that were modified.
        """
        for name in names:
            if name == "data_trafo":
                # make sure that the only difference is the exists field
                # (currently we only need this, so we will be restrictive
                #  in the future we might need to be more flexible)
                tensors = self.data_trafo.data["tensors"]
                tensors_other = self.sub_components[name].data["tensors"]

                # check if the same tensors are present
                if tensors.names != tensors_other.names:
                    raise ValueError(
                        f"{tensors.names} != {tensors_other.names}"
                    )

                diffs = []
                for tensor, tensors_other in zip(
                    tensors.list, tensors_other.list
                ):
                    diffs.extend(tensor.compare(tensors_other))

                diffs = set(diffs)
                if diffs and diffs != {"exists"}:
                    raise ValueError(f"Unexpected differences: {diffs}")
            elif isinstance(self.sub_components[name], Model):
                pass
            else:
                raise ValueError(f"Can not update {name}.")

    def _count_number_of_variables(self):
        """Counts number of model variables

        Returns
        -------
        int
            The number of trainable variables of the model.
        int
            The total number of variables of the model.
            This includes the non-trainable ones.
        """
        num_trainable = np.sum(
            [
                np.prod(x.get_shape().as_list())
                for x in self.trainable_variables
            ]
        )
        num_total = np.sum(
            [np.prod(x.get_shape().as_list()) for x in self.variables]
        )
        return num_trainable, num_total

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
            if (
                tensor.op.node_def.op == "Const"
                and tensor.name == collect_name
            ):
                return set([tensor.name])
            else:
                return set()

        tensor_list = set()
        for in_tensor in tensor.op.inputs:
            tensor_list = tensor_list.union(
                self._find_top_nodes(in_tensor, collect_name=collect_name)
            )

        return tensor_list

    def assert_paramerters_set(self):
        """Check if the parameters of the model have been set."""
        if "parameter_names" not in self._untracked_data:
            raise ValueError("No parameter names have been set!")

    def assert_configured(self, configuration_state):
        """Checks the model's configuration state.

        Parameters
        ----------
        configuration_state : bool
            The expected configuration state. If the model's actual state
            differs from this, an exception is raised.

        Raises
        ------
        TypeError
            If the passed 'configuration_state' is not a boolean value.
        ValueError
            If the model's configuration state does not match
            the passed boolean value 'configuration_state'.
        """
        if not isinstance(configuration_state, bool):
            raise TypeError(
                "Expected bool, but got {!r}".format(type(configuration_state))
            )

        if configuration_state:
            if not self.is_configured:
                raise ValueError("Model needs to be set up first!")
        else:
            if self.is_configured:
                raise ValueError("Model is already set up!")

    def increment_step(self):
        """Increment the step counter of the model."""
        self.step.assign_add(1)

        for sub_component in self.sub_components.values():
            if issubclass(type(sub_component), Model):
                sub_component.increment_step()

    def save_weights(
        self,
        dir_path,
        max_keep=3,
        protected=False,
        description=None,
        num_training_steps=None,
        recursive=True,
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
        recursive : bool, optional
            If True, the weights of all sub components will be saved
            recursively.

        Raises
        ------
        IOError
            If the model checkpoint file already exists.
        KeyError
            If the model checkpoint meta data already exists.
        ValueError
            If the model has changed since it was configured.

        """
        # make sure that there weren't any additional changes to the model
        # after its configure() call.
        variable_names = [t.name for t in self.variables]
        all_variable_names = [
            t.name for t in self._untracked_data["variables"]
        ]

        if sorted(variable_names) != sorted(all_variable_names):
            msg = "Model has changed since configuration call: {!r} != {!r}"
            raise ValueError(
                msg.format(self.variables, self._untracked_data["variables"])
            )

        checkpoint_pattern = os.path.join(dir_path, "model_checkpoint_{:08d}")

        # Load the model_checkpoints.yaml file if it exists.
        yaml_file = os.path.join(dir_path, "model_checkpoint.yaml")
        if os.path.exists(yaml_file):

            # load checkpoint meta data
            with open(yaml_file, "r") as stream:
                meta_data = yaml_loader.load(stream)
        else:
            # create new checkpoint meta data
            meta_data = {
                "latest_checkpoint": 0,
                "unprotected_checkpoints": {},
                "protected_checkpoints": {},
            }

        checkpoint_index = meta_data["latest_checkpoint"] + 1
        checkpoint_file = checkpoint_pattern.format(checkpoint_index)

        # check if file already exists
        if os.path.exists(checkpoint_file + ".index"):
            raise IOError(
                "Checkpoint file {!r} already exists!".format(checkpoint_file)
            )

        # check if meta data already exists
        if (
            checkpoint_index in meta_data["unprotected_checkpoints"]
            or checkpoint_index in meta_data["protected_checkpoints"]
        ):
            msg = "Checkpoint index {!r} already exists in meta data: {!r}!"
            raise KeyError(msg.format(checkpoint_index, meta_data))

        # update number of training steps
        if num_training_steps is not None:
            self.update_num_training_steps(
                dir_path=dir_path,
                num_training_steps=num_training_steps,
            )

        # update latest checkpoint index
        meta_data["latest_checkpoint"] += 1

        # add entry to meta data
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d %H:%M:%S")

        checkpoint_meta_data = {
            "creation_date": dt_string,
            "time_stamp": time.time(),
            "file_basename": os.path.basename(checkpoint_file),
            "description": description,
        }
        if protected:
            meta_data["protected_checkpoints"][
                checkpoint_index
            ] = checkpoint_meta_data
        else:
            meta_data["unprotected_checkpoints"][
                checkpoint_index
            ] = checkpoint_meta_data

        # save checkpoint
        self._logger.debug(f"[Model] Saving checkpoint: {checkpoint_file}")
        self._untracked_data["checkpoint"].write(checkpoint_file)

        # remove old checkpoints
        if not protected:
            old_checkpoint_numbers = sorted(
                meta_data["unprotected_checkpoints"].keys()
            )
            if len(old_checkpoint_numbers) > max_keep:
                for number in old_checkpoint_numbers[:-max_keep]:

                    # remove entry from tracked checkpoints meta data
                    info = meta_data["unprotected_checkpoints"].pop(number)

                    # remove files
                    pattern = os.path.join(
                        dir_path, info["file_basename"] + ".*"
                    )
                    files_to_remove = glob.glob(pattern)
                    msg = "Removing old checkpoint files: {!r}"
                    self._logger.info(msg.format(files_to_remove))
                    for file in files_to_remove:
                        os.remove(file)

        # save new meta data
        with open(yaml_file, "w") as stream:
            yaml_dumper.dump(meta_data, stream)

        # also save weights of all sub components
        if recursive:
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
                        recursive=recursive,
                    )

    def update_num_training_steps(self, dir_path, num_training_steps):
        """Update the number of training iterations for current training step.

        Parameters
        ----------
        dir_path : str
            Path to the output directory.
        num_training_steps : int
            The number of training steps with the current training settings.
            This will be used to update the training_steps.yaml file to
            account for the correct number of training steps for the most
            recent training step.
        """
        training_dir = os.path.join(dir_path, "training")
        yaml_file = os.path.join(training_dir, "training_steps.yaml")

        # Load the training_steps.yaml file if it exists.
        yaml_file = os.path.join(training_dir, "training_steps.yaml")
        if os.path.exists(yaml_file):

            # load training meta data
            with open(yaml_file, "r") as stream:
                meta_data = yaml_loader.load(stream)
        else:
            msg = "Could not find the training steps meta file: {!r}"
            raise IOError(msg.format(yaml_file))

        # update training steps for current training step
        current_step = meta_data["latest_step"]
        meta_data["training_steps"][current_step][
            "num_training_steps"
        ] = num_training_steps
        meta_data["training_steps"][current_step][
            "num_total_training_steps"
        ] = int(self.step.numpy())

        # save new meta data
        with open(yaml_file, "w") as stream:
            yaml_dumper.dump(meta_data, stream)

    def save_training_settings(self, dir_path, new_training_settings):
        """Save a new training step with its components and settings.

        Parameters
        ----------
        dir_path : str
            Path to the output directory.
        new_training_settings : dict, optional
            If provided, a training step will be created.
            A dictionary containing the settings of the new training step.
            This dictionary must contain the following keys:

                config: dict
                    The configuration settings used to train.
                components: dict
                    The components used during training. These typically
                    include the Loss and Evaluation components.
        """
        training_dir = os.path.join(dir_path, "training")

        # Load the training_steps.yaml file if it exists.
        yaml_file = os.path.join(training_dir, "training_steps.yaml")
        if os.path.exists(yaml_file):

            # load training meta data
            with open(yaml_file, "r") as stream:
                meta_data = yaml_loader.load(stream)
        else:
            # create new training meta data
            meta_data = {
                "latest_step": 0,
                "training_steps": {},
            }

        training_index = meta_data["latest_step"] + 1
        training_step_dir = os.path.join(
            training_dir, "step_{:04d}".format(training_index)
        )

        # check if file already exists
        if os.path.exists(training_step_dir):
            raise IOError(
                "Training directory {!r} already exists!".format(
                    training_step_dir
                )
            )
        else:
            os.makedirs(training_step_dir)
            msg = "Creating directory: {!r}"
            self._logger.info(msg.format(training_step_dir))

        # check if meta data already exists
        if training_index in meta_data["training_steps"]:
            msg = "Training index {!r} already exists in meta data: {!r}!"
            raise KeyError(msg.format(training_index, meta_data))

        # update latest training step
        meta_data["latest_step"] += 1

        # add entry to meta data
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d %H:%M:%S")

        training_meta_data = {
            "creation_date": dt_string,
            "time_stamp": time.time(),
            "num_training_steps": 0,
            "num_total_training_steps": int(self.step.numpy()),
        }
        meta_data["training_steps"][training_index] = training_meta_data

        # save training config
        training_step_config_file = os.path.join(
            training_step_dir, "training_config.yaml"
        )
        with open(training_step_config_file, "w") as stream:
            yaml_dumper.dump(new_training_settings["config"], stream)

        # save components
        for name, component in new_training_settings["components"].items():
            component_dir = os.path.join(training_step_dir, name)
            component.save(component_dir)

        # save new meta data
        with open(yaml_file, "w") as stream:
            yaml_dumper.dump(meta_data, stream)

    def _save(
        self,
        dir_path,
        max_keep=3,
        protected=False,
        description=None,
        new_training_settings=None,
        num_training_steps=None,
    ):
        """Save the model weights.

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
        new_training_settings : dict, optional
            If provided, a training step will be created.
            A dictionary containing the settings of the new training step.
            This dictionary must contain the following keys:

                config: dict
                    The configuration settings used to train.
                components: dict
                    The components used during training. These typically
                    include the Loss and Evaluation components.
        num_training_steps : int, optional
            The number of training steps with the current training settings.
            This will be used to update the training_steps.yaml file to
            account for the correct number of training steps for the most
            recent training step.
        """
        if new_training_settings is not None:
            self.save_training_settings(
                dir_path=dir_path, new_training_settings=new_training_settings
            )

        # Note: if we are saving weights via _save() (BaseComponent.save())
        # then BaseComponent.save() and subsequently BaseComponent._save()
        # is already called for all sub components and we don't need to
        # save weights again. Hence, we set recursive=False.
        self.save_weights(
            dir_path=dir_path,
            max_keep=max_keep,
            protected=protected,
            description=description,
            num_training_steps=num_training_steps,
            recursive=False,
        )

    def _rebuild_computation_graph(self):
        """Rebuild the computation graph of the model.

        When constructing the model via the BaseComponent.load() method,
        the computation graph is not automatically created. This method
        rebuilds the computation graph of the model.
        It is therefore typically only needed when the model is loaded
        via the BaseComponent.load() method.
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
            config_dict.update(self._sub_components)

            self._logger.debug(f"[Model] Rebuilding {self.__class__.__name__}")

            self._configure(**config_dict)

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

    def retrieve_weight_file_path(self, dir_path, checkpoint_number=None):
        """Retrieve the file path of the model weights.

        Parameters
        ----------
        dir_path : str
            Path to the input directory.
        checkpoint_number : None, optional
            Optionally specify a certain checkpoint number that should be
            loaded. If checkpoint_number is None (default), then the latest
            checkpoint will be loaded.

        Returns
        -------
        str
            The file path of the model weights.
        """
        # Load the model_checkpoints.yaml
        yaml_file = os.path.join(dir_path, "model_checkpoint.yaml")
        if os.path.exists(yaml_file):

            # load checkpoint meta data
            with open(yaml_file, "r") as stream:
                meta_data = yaml_loader.load(stream)
        else:
            msg = "Could not find checkpoints meta data {!r}"
            raise IOError(msg.format(yaml_file))

        # retrieve meta data of checkpoint that will be loaded
        if checkpoint_number is None:
            checkpoint_number = meta_data["latest_checkpoint"]

        if checkpoint_number in meta_data["unprotected_checkpoints"]:
            info = meta_data["unprotected_checkpoints"][checkpoint_number]
            file_basename = info["file_basename"]
        else:
            info = meta_data["protected_checkpoints"][checkpoint_number]
            file_basename = info["file_basename"]

        return os.path.join(dir_path, file_basename)

    def load_weights(self, dir_path, checkpoint_number=None, recursive=True):
        """Load the model weights.

        Parameters
        ----------
        dir_path : str
            Path to the input directory.
        checkpoint_number : None, optional
            Optionally specify a certain checkpoint number that should be
            loaded. If checkpoint_number is None (default), then the latest
            checkpoint will be loaded_load.
        recursive : bool, optional
            If True, the weights of all sub components will be loaded
            recursively.

        Raises
        ------
        IOError
            If the checkpoint meta data cannot be found in the input directory.
        """
        # Sub-components need to be rebuilt first, as higher-level
        # components need to know which variables exist when configuring
        for name, sub_component in self.sub_components.items():

            # get directory of sub component
            sub_dir_path = os.path.join(dir_path, name)

            if issubclass(type(sub_component), Model):
                # load weights of Model sub component
                if recursive:
                    sub_component.load_weights(
                        dir_path=sub_dir_path,
                        checkpoint_number=checkpoint_number,
                        recursive=recursive,
                    )
                else:
                    # make sure that the sub component is configured,
                    # i.e. has weights loaded or created from scratch
                    sub_component.assert_configured(True)

        # rebuild the tensorflow graph if it does not exist yet
        # This is the case if the model is configured via the
        # BaseComponent.load() method and not via the configure() method.
        self._rebuild_computation_graph()

        file_path = self.retrieve_weight_file_path(
            dir_path=dir_path,
            checkpoint_number=checkpoint_number,
        )
        self._logger.debug(f"[Model] Loading checkpoint: {file_path}")
        self._untracked_data["checkpoint"].read(
            file_path
        ).assert_existing_objects_matched()

    def _load(self, dir_path, checkpoint_number=None):
        """Load the model weights.

        Parameters
        ----------
        dir_path : str
            Path to the input directory.
        checkpoint_number : None, optional
            Optionally specify a certain checkpoint number that should be
            loaded. If checkpoint_number is None (default), then the latest
            checkpoint will be loaded.
        """
        # Note: if we are loading weights via _load() (BaseComponent.load())
        # then BaseComponent.load() and subsequently BaseComponent._load()
        # is already called for all sub components and we don't need to
        # load weights again. Hence, we set recursive=False.
        self.load_weights(
            dir_path=dir_path,
            checkpoint_number=checkpoint_number,
            recursive=False,
        )
