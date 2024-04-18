from __future__ import division, print_function

import os
import ruamel.yaml as yaml
import pickle
import logging
import inspect
from copy import deepcopy
import tensorflow as tf

import egenerator
from egenerator import misc
from egenerator.utils.getclass import getclass
from egenerator.settings import version_control


class Configuration(object):
    """This class gathers all settings necessary to configure a component or
    to check its compatibility. It does *not* contain data attributes of the
    component. Each component has a configuration.

    Note: a component may require additional components or parameters to
    configure itself. These are *not* stored within the configuration.

    A configuration is made up of four core types of dictionaries:

        - settings:
            These are inmutable settings that may be used to check the
            compatibility of one configuration with another.

        - mutable_settings:
            These are settings that are required to configure the component,
            but they may change between two objects of a component class
            without breaking the compatibility between them.

        - check_values:
            These are values that can be used to check the identity or
            compatibility of a component.
            For example, these could be hashes of the data attributes or the
            time stamp of the creation time of a component.

        - sub_components_configurations:
            A list of configurations of sub modules.

    and the name of the component class:

        - class_string:
            The name of the component class. This is necessary in order to
            load the component from file.

    as well as:

        - dependent_sub_components : list, optional
            A list of dependent sub component names. These are a subset of
            the defined sub components in the sub_component_configurations.
            A dependent sub component is a sub component that must also be
            saved and loaded when the component is being saved or loaded.

        - mutable_sub_components : list, optional
            A list of mutable sub component names. These are a subset of
            the defined sub components in the sub_component_configurations.
            A mutable sub component is a sub component that may change
            completely. When checking compatibility of a Configuration,
            mutable sub components are *not* checked for compatibility.

    The settings and mutable_settings completely define the settings needed
    to configure a component. The attribute check_values is used to apply
    additional checks for compatibility.
    The settings and mutable_settings get combined in a config dictionary
    for convenience.
    """

    @property
    def class_string(self):
        return deepcopy(self._dict["class_string"])

    @property
    def check_values(self):
        return dict(deepcopy(self._dict["check_values"]))

    @property
    def mutable_settings(self):
        return dict(deepcopy(self._dict["mutable_settings"]))

    @property
    def settings(self):
        return dict(deepcopy(self._dict["settings"]))

    @property
    def sub_component_configurations(self):
        return dict(deepcopy(self._dict["sub_component_configurations"]))

    @property
    def dependent_sub_components(self):
        return self._dict["dependent_sub_components"]

    @property
    def mutable_sub_components(self):
        return self._dict["mutable_sub_components"]

    @property
    def dict(self):
        return dict(deepcopy(self._dict))

    @property
    def config(self):
        """The combination of mutable and constant settings.

        Returns
        -------
        dict
            Combined 'settings' and 'mutable_settings'
        """
        return dict(deepcopy(self._config))

    def __init__(
        self,
        class_string,
        settings,
        mutable_settings={},
        check_values={},
        dependent_sub_components=[],
        mutable_sub_components=[],
        sub_component_configurations={},
        event_generator_version=egenerator.__version__,
        event_generator_git_short_sha=None,
        event_generator_git_sha=None,
        event_generator_origin=None,
        event_generator_uncommitted_changes=None,
        logger=None,
    ):
        """Create a configuration object.

        Parameters
        ----------
        class_string : str
            The class string of the component. Example for the BaseComponent:
            'egenerator.manager.component.BaseComponent'
        settings : dict
            These are inmutable settings that may be used to check the
            compatibility of one configuration with another.
        mutable_settings : dict, optional
            These are settings that are required to configure the component,
            but they may change between two objects of a component class
            without breaking the compatibility between them.
        check_values : dict, optional
            These are values that can be used to check the identity or
            compatibility of a component.
            For example, these could be hashes of the data attributes or the
            time stamp of the creation time of a component.
        dependent_sub_components : list, optional
            A list of dependent sub component names. These are a subset of
            the defined sub components in the sub_component_configurations.
            A dependent sub component is a sub component that must also be
            saved and loaded when the component is being saved or loaded.
        mutable_sub_components : list, optional
            A list of mutable sub component names. These are a subset of
            the defined sub components in the sub_component_configurations.
            A mutable sub component is a sub component that may change
            completely. When checking compatibility of a Configuration,
            mutable sub components are *not* checked for compatibility.
        sub_component_configurations : dict, optional
            A dictionary of sub component configurations.
            Additional sub components may be added after instantiation via
            the 'add_sub_components' method.
        event_generator_version : str, optional
            The version string of the Event-Generator package.
        event_generator_git_short_sha : str, optional
            Event-Generator GitHub repository short sha.
        event_generator_git_sha : str, optional
            Event-Generator GitHub repository sha.
        event_generator_origin : str, optional
            Event-Generator GitHub repository origin.
        event_generator_uncommitted_changes : bool, optional
            If true, there are uncommitted changes in the repository.
        logger : logging.logger, optional
            The logger to use.
        """
        short_sha, sha, origin, uncommitted_changes = (
            version_control.get_git_infos()
        )
        if event_generator_git_short_sha is None:
            event_generator_git_short_sha = short_sha
        if event_generator_git_sha is None:
            event_generator_git_sha = sha
        if event_generator_origin is None:
            event_generator_origin = origin
        if event_generator_uncommitted_changes is None:
            event_generator_uncommitted_changes = uncommitted_changes

        self._logger = logger or logging.getLogger(__name__)
        self._dict = {
            "event_generator_version": egenerator.__version__,
            "event_generator_git_sha": event_generator_git_sha,
            "event_generator_origin": event_generator_origin,
            "event_generator_uncommitted_changes": event_generator_uncommitted_changes,
            "class_string": class_string,
            "check_values": dict(deepcopy(check_values)),
            "settings": dict(deepcopy(settings)),
            "mutable_settings": dict(deepcopy(mutable_settings)),
            "sub_component_configurations": dict(
                deepcopy(sub_component_configurations)
            ),
            "dependent_sub_components": list(
                deepcopy(dependent_sub_components)
            ),
            "mutable_sub_components": list(deepcopy(mutable_sub_components)),
        }

        # make sure defined dependen sub components exist in the configuration
        self._check_dependent_names(self.dependent_sub_components)
        # self._check_dependent_names(self.mutable_sub_components)
        self._config = self._combine_settings()

    def update_mutable_settings(self, new_settings):
        """Update mutable settings

        Parameters
        ----------
        new_settings : dict
            The dictionary with the key-value pairs of the mutable settings
            that should be updated.
        """
        for key, value in new_settings.items():
            if key not in self._dict["mutable_settings"]:
                raise KeyError(
                    "Mutable setting: {!r} does not exist".format(key)
                )
            self._dict["mutable_settings"][key] = value

        # update config
        self._config = self._combine_settings()

    def _check_dependent_names(self, dependent_sub_components):
        """Make sure defined dependent sub components exist in the
        configuration

        Raises
        ------
        KeyError
            If defined dependent sub components do not exist in the specified
            configuration of sub components.
        """
        for name in dependent_sub_components:
            if name not in self.sub_component_configurations:
                msg = "Sub component {!r} does not exist in {!r}"
                raise KeyError(
                    msg.format(name, self.sub_component_configurations.keys())
                )

    def add_dependent_sub_components(self, dependent_components):
        self._check_dependent_names(dependent_components)
        self._dict["dependent_sub_components"].extend(dependent_components)

    def add_mutable_sub_components(self, mutable_sub_components):
        self._check_dependent_names(mutable_sub_components)
        self._dict["mutable_sub_components"].extend(mutable_sub_components)

    def _combine_settings(self):
        """Combine the constant and mutable settings in a single config.

        Returns
        -------
        dict
            The combined dictionary.

        Raises
        ------
        ValueError
            If keys are defined multiple times.
        """

        # make sure no options are defined multiple times
        duplicates = set(self.settings.keys()).intersection(
            set(self.mutable_settings.keys())
        )
        if duplicates:
            raise ValueError(
                "Keys are defined multiple times: {!r}".format(duplicates)
            )

        config = dict(self.settings)
        config.update(self.mutable_settings)
        return config

    def add_sub_components(self, sub_components):
        """Add configurations of sub components

        Parameters
        ----------
        sub_components : dict of BaseComponent objects
            A dict of sub components for which to save the configurations.
            The dict key is the name of the sub component and the value is
            the sub component itself.
        """
        for name, component in sub_components.items():
            if not issubclass(type(component), BaseComponent):
                raise TypeError("Incorrect type: {!r}".format(type(component)))
            if name in self.sub_component_configurations:
                raise KeyError(
                    "Sub component {!r} already exists!".format(name)
                )
            if not component.is_configured:
                raise ValueError(
                    "Component {!r} is not configured!".format(name)
                )
            self._dict["sub_component_configurations"][
                name
            ] = component.configuration.dict

    def replace_sub_components(self, new_sub_components):
        """Replace sub components with new sub components.

        Parameters
        ----------
        new_sub_components : dict of BaseComponent objects
            A dict of sub components for which to save the configurations.
            The dict key is the name of the sub component and the value is
            the sub component itself.
        """
        for name, component in new_sub_components.items():
            if not issubclass(type(component), BaseComponent):
                raise TypeError("Incorrect type: {!r}".format(type(component)))
            if name not in self.sub_component_configurations:
                msg = "Sub component {!r} cannot be replaced because it "
                msg += "does not exists!"
                raise KeyError(msg.format(name))
            if not component.is_configured:
                raise ValueError(
                    "Component {!r} is not configured!".format(name)
                )
            # replace sub component's configuration
            self._dict["sub_component_configurations"][
                name
            ] = component.configuration.dict

    def is_compatible(self, other):
        """Check compatibility between two configuration objects.

        Parameters
        ----------
        other : Configuration object
            Another Configuration instance.

        Returns
        -------
        bool
            True if configuration objects are compatible, false if not.
        """
        if (
            self.settings != other.settings
            or self.check_values != other.check_values
            or self.class_string != other.class_string
        ):
            return False

        if set(self.sub_component_configurations.keys()) != set(
            other.sub_component_configurations.keys()
        ):
            return False

        # now check if sub components are also compatible
        for name, sub_conf in self.sub_component_configurations.items():

            # create a Configuration object from sub component
            this_sub_configuration = Configuration(**sub_conf)
            other_sub_configuration = Configuration(
                **other.sub_component_configurations[name]
            )

            if not this_sub_configuration.is_compatible(
                other_sub_configuration
            ):

                # only now check if module is mutable in order to
                # log warnings of modified settings
                if name not in self.mutable_sub_components:
                    return False

        # provide information on mutable settings that changed
        # Note: this can be done more efficiently
        keys_changed = []
        for key, value in self.mutable_settings.items():
            if key not in other.mutable_settings:
                keys_changed.append(key)
            else:
                if value != other.mutable_settings[key]:
                    keys_changed.append(key)
        for key, value in other.mutable_settings.items():
            if key not in self.mutable_settings:
                keys_changed.append(key)
        keys_changed = set(keys_changed)
        if keys_changed:
            msg = "The following mutable settings have changed: {!r}"
            self._logger.warning(msg.format(keys_changed))

        return True


class BaseComponent(object):
    """This is the base class for components used within the event-generator
    project.

    A component can be instantiated without requiring settings that define
    the component. At instantiation the component is unconfigured and cannot
    yet be used. There are two ways how to configure a component:

        - configure(config, *args, **kwargs)
            This configures the component object with a dictionary 'config'
            that contains all necessary settings to create the object or
            alternatively it can be created via a 'Configuration' object.

        - load(file_path)
            This loads the component with all its settings and member variables
            from file. Specified sub components are loaded recursively.

    Once either of these methods is called, the component is configured.
    Attempting to re-configure it will result in an exception.
    The component provides a method to check compatibility with a second
    component (self.is_compatible(component)), as well as a
    method to save the component to file (self.save(file_path)).
    Again, defined dependent sub components are saved recursively to file.

    The BaseComponent class is an abstract class with a pure virtual method
    self._configure(). A derived class must at least implement and override
    this method. A call to self._configure() must return the following:

        Configuration object
            The configuration object of the newly configured component.
            This does not need to include configurations of sub components
            as these are automatically gathered.
        dict
            The data of the component.
            Return None if the component has no data.
        dict
            A dictionary of dependent sub components. This is a dictionary
            of sub components that need to be saved and loaded recursively
            when the component is saved and loaded.
            Return None if no dependent sub components exist.

    Attributes
    ----------
    configuration : Configuration object
        A configuration object that describes the configuration of this class.
    data : dict
        A dictionary with data of the component. This data will be saved to
        file in a self.save() call and will subsequently be loaded in a
        self.load() call.
    is_configured : boolt
        Indicates whether the component is configured.
    sub_components : dict
        This is a dict with sub components this component relies on.
        Sub components are components that will be recursively saved and loaded
        when this component gets saved or loaded.
    untracked_data : dict
        This is a dictionary of data that is not tracked and will not be saved
        to file. Any derived class that needs to store data which is not to be
        saved to file, must store it in this dict.
    """

    @property
    def is_configured(self):
        """Component's configuration status

        Returns
        -------
        bool
            Indicates whether the component is configured.
        """
        return self._is_configured

    @property
    def configuration(self):
        """Configuratoion of component

        Returns
        -------
        Configuration
            The configuration of this component.
        """
        return self._configuration

    @property
    def sub_components(self):
        """Dependent sub components

        These are sub components that will be recursively saved and loaded
        when this component gets saved or loaded.

        Returns
        -------
        dict
            A dictionary with dependent sub components.
        """
        return self._sub_components

    @property
    def data(self):
        """Data of component

        Returns
        -------
        dict
            A dictionary with data of this component.
        """
        return self._data

    @property
    def untracked_data(self):
        """Untracked data of component

        Returns
        -------
        dict
            A dictionary with untracked data of this component.
        """
        return self._untracked_data

    def __init__(self, logger=None):
        """Instantiate BaseComponent Class

        Parameters
        ----------
        logger : logging.logger, optional
            The logger instance to use.
        """
        self._logger = logger or logging.getLogger(__name__)
        self._is_configured = False
        self._configuration = None
        self._data = None
        self._sub_components = {}
        self._untracked_data = {}

    def _get_untracked_member_attributes(self):
        """Get all attributes of class instance that will not be tracked
        and saved by a self.save() call. This method can be used to identify
        if this class is being used in a wrongful way.

        Returns
        -------
        list of str
            A list of untracked attributes

        """
        attributes = inspect.getmembers(
            self, lambda a: not inspect.isroutine(a)
        )

        filtered_attributes = [
            a[0]
            for a in attributes
            if not (a[0].startswith("__") and a[0].endswith("__"))
        ]

        # tracked or knowingly untracked data
        tracked_attributes = [
            "_is_configured",
            "_logger",
            "_configuration",
            "_data",
            "_untracked_data",
            "is_configured",
            "logger",
            "configuration",
            "data",
            "untracked_data",
            "_sub_components",
            "sub_components",
        ]
        untracked_attributes = [
            a for a in filtered_attributes if a not in tracked_attributes
        ]

        # if this is a tf.Module derived class instance, then we need to
        # also remove the following variables:
        if issubclass(type(self), tf.Module):
            tf_module_vars = [
                "_TF_MODULE_IGNORED_PROPERTIES",
                "_name",
                "_scope_name",
                "_setattr_tracking",
                "_tf_api_names",
                "_tf_api_names_v1",
                "name",
                "name_scope",
                "submodules",
                "trainable_variables",
                "variables",
                "_name_scope",
                "_self_name_based_restores",
                "_self_setattr_tracking",
                "_self_saveable_object_factories",
                "_self_unconditional_checkpoint_dependencies",
                "_self_unconditional_deferred_dependencies",
                "_self_unconditional_dependency_names",
                "_self_update_uid",
            ]
            untracked_attributes = [
                a for a in untracked_attributes if a not in tf_module_vars
            ]

        # remove properties:
        object_class = getclass(self)
        untracked_attributes_without_properties = []
        for a in untracked_attributes:
            if not hasattr(object_class, a):
                untracked_attributes_without_properties.append(a)
            elif not isinstance(getattr(object_class, a), property):
                untracked_attributes_without_properties.append(a)

        return untracked_attributes_without_properties

    def _check_member_attributes(self):
        """Check if class instance has any attributes it should not have.

        Raises
        ------
        ValueError
            If member attributes are found that should not be there.
        """
        untracked_attributes = self._get_untracked_member_attributes()
        if len(untracked_attributes) > 0:
            msg = "Class contains member variables that it should not at "
            msg += "this point: {!r}"
            raise ValueError(msg.format(untracked_attributes))

    def configure(self, **kwargs):
        """Configure the BaseComponent instance.

        If additional components are directly passed via the kwargs argument,
        these will be automatically collected and their configurations
        accumulated.
        Note: this will not detect and collect all nested components. Nested
        components other than dictionaries, lists, or tuples must be added
        manually.

        Parameters
        ----------
        **kwargs
            Arbitrary keyword arguments.
        """
        if self.is_configured:
            raise ValueError("Component is already configured!")

        # Get the dictionary format of the passed Configuration object
        if isinstance(kwargs, Configuration):
            kwargs = kwargs.config

        # check if it already has attributes other than the allowed ones
        # This indicates an incorrect usage of the BaseComponent class.
        self._check_member_attributes()

        # collect configurations of sub components from kwargs
        missing_settings = {}
        parameter_sub_components = {}
        for key, value in kwargs.items():

            if issubclass(type(value), BaseComponent):
                # found another component, keep track of it
                parameter_sub_components[key] = value

                # make sure the component is configured
                if not value.is_configured:
                    msg = "Component {!r} is not configured!"
                    raise ValueError(msg.format(key))

            # check if this is a dict of Components
            elif isinstance(value, dict):

                all_components = True
                for name, comp in value.items():
                    if issubclass(type(comp), BaseComponent):
                        # found another component, keep track of it
                        parameter_sub_components[name] = comp

                        # make sure the component is configured
                        if not comp.is_configured:
                            msg = "Component {!r} is not configured!"
                            raise ValueError(msg.format(name))
                    else:
                        all_components = False

                if not all_components:
                    missing_settings[key] = value

            # check if this is a list or tuple of Components
            elif isinstance(value, (list, tuple)):

                all_components = True
                for i, comp in enumerate(value):
                    name = key + "_{:04d}".format(i)
                    if issubclass(type(comp), BaseComponent):
                        # found another component, keep track of it
                        parameter_sub_components[name] = comp

                        # make sure the component is configured
                        if not comp.is_configured:
                            msg = "Component {!r} is not configured!"
                            raise ValueError(msg.format(name))
                    else:
                        all_components = False

                if not all_components:
                    missing_settings[key] = value

            else:
                missing_settings[key] = value
        self._configuration, self._data, self._sub_components = (
            self._configure(**kwargs)
        )

        if self._data is None:
            self._data = {}

        if self._sub_components is None:
            self._sub_components = {}

        if not isinstance(self._sub_components, dict):
            msg = "Sub components must be provided as a dictionary, "
            msg += "but are: {!r}"
            raise TypeError(msg.format(type(self._sub_components)))

        # add sub component configurations to this configuration
        self._configuration.add_sub_components(parameter_sub_components)

        # add dependent sub components to this configuration
        self._configuration.add_dependent_sub_components(
            [component for component in self._sub_components.keys()]
        )

        # check if specified mutable sub components exist in configuration
        self._configuration._check_dependent_names(
            self._configuration.mutable_sub_components
        )

        # check if data has correct type
        if self._data is not None:
            if not isinstance(self._data, dict):
                raise TypeError(
                    "Wrong type {!r}. Expected dict.".format(type(self._data))
                )

        # check if passed settings are all defined in configuration
        # and if they are set to the correct values. A component may have
        # more settings than these, but they must be arguments to the
        # _configure method.
        for key, value in missing_settings.items():

            # setting is missing
            if key not in self._configuration.config:
                msg = "Key {!r} is missing in configuration {!r}"
                raise ValueError(msg.format(key, self._configuration.config))

            # setting has wrong value
            elif value != self._configuration.config[key]:
                msg = "Values of {!r} do not match: {!r} != {!r}"
                raise ValueError(
                    msg.format(key, value, self._configuration.config[key])
                )

        self._is_configured = True

    def _configure(self, **kwargs):
        """Configure Component class instance

        This is an abstract method and must be implemented by derived class.

        Parameters
        ----------
        **kwargs
            Arbitrary keyword arguments.

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
                mutable_sub_components: list, default=[]
                    A list of mutable sub components.
                    Warning: use this with caution as these sub components
                             will not be checked for compatibility!
        dict
            The data of the component.
            Return None if the component has no data.
        dict
            A dictionary of dependent sub components. This is a dictionary
            of sub components that need to be saved and loaded recursively
            when the component is saved and loaded.
            Return None if no dependent sub components exist.
        """
        raise NotImplementedError()

    def is_compatible(self, other):
        """Check the compatibility of this component with another.

        Parameters
        ----------
        other : Component object
            The other component object for which to check the compatibility.

        Returns
        -------
        bool
            True if both components are compatible
        """

        # check if both instances are of the same class
        if getclass(self) != getclass(other):
            return False

        return self.configuration.is_compatible(other.configuration)

    def save(
        self,
        dir_path,
        overwrite=False,
        allow_untracked_attributes=False,
        **kwargs
    ):
        """Save component to file.

        Parameters
        ----------
        dir_path : str
            The path to the output directory to which the component
            configuration will be saved. Two output files will be generated,
            one for the component configuration and one for the data.
        overwrite : bool, optional
            If True, potential existing files will be overwritten.
            If False, an error will be raised.
        allow_untracked_attributes : bool, optional
            If False, an error is raised if there are object attributes
            which are untracked and would not be saved to file.
        **kwargs
            Additional keyword arguments that will be passed on to the
            virtual _save() method, that derived classes my overwrite.

        Raises
        ------
        AttributeError
            If allow_untracked_attributes is False and the component contains
            any unallowed, untracked attributes.
        IOError
            If overwrite is False, but a file already exits.
        ValueError
            If the component is not configured yet.
        """
        if not self.is_configured:
            raise ValueError("Component is not configured!")

        # check if there are any untracked attributes
        untracked_attributes = self._get_untracked_member_attributes()
        if len(untracked_attributes) > 0:
            msg = "Found untracked attributes: {!r}"
            if allow_untracked_attributes:
                self._logger.warning(msg.format(untracked_attributes))
            else:
                raise AttributeError(msg.format(untracked_attributes))

        # split of possible extension
        dir_path = os.path.splitext(dir_path)[0]
        config_file_path = os.path.join(dir_path, "configuration.yaml")
        data_file_path = os.path.join(dir_path, "data.pickle")

        # check if file already exists
        for file in [config_file_path, data_file_path]:
            if os.path.exists(file):
                if overwrite:
                    msg = "Overwriting existing file at: {}"
                    self._logger.info(msg.format(file))
                else:
                    raise IOError("File {!r} already exists!".format(file))

        # check if directory needs to be created
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
            self._logger.info("Creating directory: {}".format(dir_path))

        # save configuration settings
        with open(config_file_path, "w") as yaml_file:
            yaml.dump(self.configuration.dict, yaml_file)

        # write data
        with open(data_file_path, "wb") as handle:
            pickle.dump(self.data, handle, protocol=2)

        # walk through dependent sub components and save these as well
        for name, sub_component in self.sub_components.items():
            sub_component_dir_path = os.path.join(dir_path, name)
            sub_component.save(
                sub_component_dir_path,
                overwrite=overwrite,
                allow_untracked_attributes=allow_untracked_attributes,
                **kwargs
            )

        # call additional tasks of derived class via virtual method
        self._save(dir_path, **kwargs)

    def load(self, dir_path, modified_sub_components={}, **kwargs):
        """Load component from file.

        Parameters
        ----------
        dir_path : str
            The path to the input directory from which the component will be
            loaded.
        modified_sub_components : dict of Components (and nested), optional
            A dictionary of modified sub components with key-value pairs of the
            structure: (sub_component_name: sub_component).
            The modified_sub_components may also be a nested dictionary of
            sub components. In this case, the nested dictionary is passed
            on to the load method of the sub component.
            If modifided sub components are provided, these will be used
            instead of loading the previously saved sub_component.
            A check for compatibility of the old and new sub_component is
            performed. If sub components are compatible, the component's
            configuration.sub_component_configurations will be updated for the
            modified sub_component.
        **kwargs
            Additional keyword arguments that will be passed on to the
            virtual _load() method, that derived classes my overwrite.

        Raises
        ------
        TypeError
            If the component's class does not match the class of the component
            that is to be loaded.
        ValueError
            If the component is already configured.

        """
        if self.is_configured:
            raise ValueError("Component is already configured!")

        # check if it already has attributes other than the allowed ones
        # This indicates an incorrect usage of the BaseComponent class.
        self._check_member_attributes()

        # split of possible extension
        dir_path = os.path.splitext(dir_path)[0]
        config_file_path = os.path.join(dir_path, "configuration.yaml")
        data_file_path = os.path.join(dir_path, "data.pickle")

        # load files
        with open(config_file_path, "r") as stream:
            config_dict = yaml.load(stream, Loader=yaml.Loader)

        with open(data_file_path, "rb") as handle:
            data = pickle.load(handle)

        # get configuration of self:
        self._configuration = Configuration(**config_dict)

        # check if the version is correct
        if config_dict["event_generator_version"] != egenerator.__version__:
            msg = "Event-Generator versions do not match. "
            msg += "Make sure they are still compatible. "
            msg += "Saved component was created with version {!r}, but this is"
            msg += " version {!r}."
            self._logger.info(
                msg.format(
                    config_dict["event_generator_version"],
                    egenerator.__version__,
                )
            )

        # check if this is the correct class
        if (
            self.configuration.class_string
            != misc.get_full_class_string_of_object(self)
        ):
            msg = "The object's class {!r} does not match the saved "
            msg += "class string {!r}"
            raise TypeError(
                msg.format(
                    misc.get_full_class_string_of_object(self),
                    self.configuration.class_string,
                )
            )

        # get data and check if it has correct type
        self._data = data
        if self._data is not None:
            if not isinstance(self._data, dict):
                raise ValueError(
                    "Wrong type {!r}. Expected dict.".format(type(self._data))
                )

        # walk through dependent sub components and load these as well
        updated_sub_components = []
        for name in self.configuration.dependent_sub_components:

            # check if there is a nested modified_sub_components dictionary
            if name in modified_sub_components and isinstance(
                modified_sub_components[name], dict
            ):
                nested_dict = modified_sub_components.pop(name)
                updated_sub_components.append(name)
            else:
                nested_dict = {}

            sub_configuration = (
                self.configuration.sub_component_configurations[name]
            )
            sub_component_dir_path = os.path.join(dir_path, name)

            sub_class = misc.load_class(sub_configuration["class_string"])
            sub_component = sub_class()
            sub_component.load(
                sub_component_dir_path,
                modified_sub_components=nested_dict,
                **kwargs
            )
            self._sub_components[name] = sub_component
            if name in updated_sub_components:
                self.configuration.replace_sub_components(
                    {name: sub_component}
                )

            # check if this sub component is to be replaced with a new
            # (and compatible) version
            if name in modified_sub_components:
                modified_sub_component = modified_sub_components.pop(name)

                # check if configured
                if not modified_sub_component.is_configured:
                    msg = "Sub component {!r} is not configured."
                    raise ValueError(msg.format(name))

                # check if compatible
                if name not in self.configuration.mutable_sub_components:
                    if not sub_component.is_compatible(modified_sub_component):
                        msg = "Sub component {!r} is not compatible."
                        raise ValueError(msg.format(name))

                # replace sub component and its configuration
                self._sub_components[name] = modified_sub_component
                self.configuration.replace_sub_components(
                    {name: modified_sub_component}
                )
                updated_sub_components.append(name)

        if updated_sub_components != []:
            # update components such that settings which are dependent on
            # the modified sub components may be updated
            self._update_sub_components(updated_sub_components)

        if modified_sub_components != {}:
            msg = "The following modified sub components were unused: {!r}"
            raise ValueError(msg.format(modified_sub_components.keys()))

        # call additional tasks of derived class via virtual method
        self._load(dir_path, **kwargs)

        self._is_configured = True

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
        raise NotImplementedError()

    def _save(self, dir_path, **kwargs):
        """Virtual method for additional save tasks by derived class

        This is a virtual method that may be overwritten by derived class
        to perform additional tasks necessary to save the component.
        This can for instance be saving of tensorflow model weights.

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

        Parameters
        ----------
        dir_path : str
            The path to the input directory from which the component will be
            loaded.
        **kwargs
            Additional keyword arguments that may be used by the derived
            class.
        """
        pass
