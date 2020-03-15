from __future__ import division, print_function
import os
import logging
import tensorflow as tf
import ruamel.yaml as yaml
import glob
from datetime import datetime
import time

from egenerator.manager.component import BaseComponent, Configuration


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

    A derived class of the asbtract Model class must at least implement the
    virtual _configure_derived_class method.
    """

    @property
    def checkpoint(self):
        if self.untracked_data is not None and \
                'checkpoint' in self.untracked_data:
            return self.untracked_data['checkpoint']
        else:
            return None

    def __init__(self, name=None, logger=None):
        """Initializes Model object.

        Parameters
        ----------
        name : str, optional
            Name of the model. This gets passed on to tf.Module.__init__
        logger : logging.logger, optional
            A logging instance.
        """
        self._logger = logger or logging.getLogger(__name__)
        BaseComponent.__init__(self, logger=self._logger)
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

        # configure and setup model architecture
        configuration, component_data, sub_components = \
            self._configure_derived_class(**kwargs)

        # create a tensorflow checkpoint object and keep track of variables
        self._untracked_data['step'] = tf.Variable(1)
        self._untracked_data['checkpoint'] = tf.train.Checkpoint(
            step=self._untracked_data['step'], model=self)
        self._untracked_data['variables'] = list(self.variables)

        # collect any variables from sub_components as well
        for name, sub_component in sub_components.items():
            if issubclass(type(sub_component), tf.Module):
                self._untracked_data['variables'].extend(
                    sub_component.variables)
        self._untracked_data['variables'] = \
            tuple(self._untracked_data['variables'])

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
            raise TypeError('Expected bool, but got {!r}'.format(
                type(configuration_state)))

        if configuration_state:
            if not self.is_configured:
                raise ValueError('Model needs to be set up first!')
        else:
            if self.is_configured:
                raise ValueError('Model is already set up!')

    def save_weights(self, dir_path, max_keep=3, protected=False,
                     description=None):
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
        saved_variable_names = [t.name for t
                                in self._untracked_data['variables']]
        if sorted(variable_names) != sorted(saved_variable_names):
            msg = 'Model has changed since configuration call: {!r} != {!r}'
            raise ValueError(msg.format(self.variables,
                                        self._untracked_data['variables']))

        checkpoint_pattern = os.path.join(dir_path, 'model_checkpoint_{:08d}')

        # Load the model_checkpoints.yaml file if it exists.
        yaml_file = os.path.join(dir_path, 'model_checkpoint.yaml')
        if os.path.exists(yaml_file):

            # load checkpoint meta data
            with open(yaml_file, 'r') as stream:
                meta_data = yaml.safe_load(stream)
        else:
            # create new checkpoint meta data
            meta_data = {
                'latest_checkpoint': 0,
                'unprotected_checkpoints': {},
                'protected_checkpoints': {},
            }

        checkpoint_index = meta_data['latest_checkpoint'] + 1
        checkpoint_file = checkpoint_pattern.format(checkpoint_index)

        # check if file already exists
        if os.path.exists(checkpoint_file+'.index'):
            raise IOError('Checkpoint file {!r} already exists!'.format(
                checkpoint_file))

        # check if meta data already exists
        if checkpoint_index in meta_data['unprotected_checkpoints'] or \
                checkpoint_index in meta_data['protected_checkpoints']:
            msg = 'Checkpoint index {!r} already exists in meta data: {!r}!'
            raise KeyError(msg.format(checkpoint_index, meta_data))

        # update latest checkpoint index
        meta_data['latest_checkpoint'] += 1

        # add entry to meta data
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d %H:%M:%S")

        checkpoint_meta_data = {
            'creation_date': dt_string,
            'time_stamp': time.time(),
            'file_basename': os.path.basename(checkpoint_file),
            'description': description,
        }
        if protected:
            meta_data['protected_checkpoints'][checkpoint_index] = \
                checkpoint_meta_data
        else:
            meta_data['unprotected_checkpoints'][checkpoint_index] = \
                checkpoint_meta_data

        # save checkpoint
        self._untracked_data['checkpoint'].write(checkpoint_file)

        # remove old checkpoints
        if not protected:
            old_checkpoint_numbers = \
                sorted(meta_data['unprotected_checkpoints'].keys())
            if len(old_checkpoint_numbers) > max_keep:
                for number in old_checkpoint_numbers[:-max_keep]:

                    # remove entry from tracked checkpoints meta data
                    info = meta_data['unprotected_checkpoints'].pop(number)

                    # remove files
                    pattern = os.path.join(dir_path,
                                           info['file_basename'] + '.*')
                    files_to_remove = glob.glob(pattern)
                    msg = 'Removing old checkpoint files: {!r}'
                    self._logger.info(msg.format(files_to_remove))
                    for file in files_to_remove:
                        os.remove(file)

        # save new meta data
        with open(yaml_file, 'w') as stream:
            yaml.dump(meta_data, stream)

    def _save(self, dir_path, max_keep=3, protected=False, description=None):
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
        """
        self.save_weights(dir_path=dir_path, max_keep=max_keep,
                          protected=protected, description=description)

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
            self._configure(**config_dict)

            # make sure that no additional class attributes are created
            # apart from untracked ones
            self._check_member_attributes()

            # make sure the other values weren't overwritten
            if (not configuration.is_compatible(self.configuration) or
                configuration_id != id(self.configuration) or
                data != self.data or
                sub_components != self.sub_components or
                    sub_components_id != id(self.sub_components)):
                raise ValueError('Tracked components were changed!')

        # Load the model_checkpoints.yaml
        yaml_file = os.path.join(dir_path, 'model_checkpoint.yaml')
        if os.path.exists(yaml_file):

            # load checkpoint meta data
            with open(yaml_file, 'r') as stream:
                meta_data = yaml.safe_load(stream)
        else:
            msg = 'Could not find checkpoints meta data {!r}'
            raise IOError(msg.format(yaml_file))

        # retrieve meta data of checkpoint that will be loaded
        if checkpoint_number is None:
            checkpoint_number = meta_data['latest_checkpoint']

        if checkpoint_number in meta_data['unprotected_checkpoints']:
            info = meta_data['unprotected_checkpoints'][checkpoint_number]
            file_basename = info['file_basename']
        else:
            info = meta_data['protected_checkpoints'][checkpoint_number]
            file_basename = info['file_basename']

        file_path = os.path.join(dir_path, file_basename)
        self._untracked_data['checkpoint'].restore(file_path)

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
        # set counter of checkpoint to counter in file
        self.load_weights(dir_path=dir_path,
                          checkpoint_number=checkpoint_number)
