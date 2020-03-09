from __future__ import division, print_function

import os
import ruamel.yaml as yaml
import pickle
import logging
import inspect
from getclass import getclass
from copy import deepcopy

from egenerator import misc


class Configuration(object):

    """This class gathers all settings necessary to configure a component or
    to check its compatibility. It does *not* contain data attributes of the
    component. Each component has a configuration.

    Note: a component may require additional components or parameters to
    configure itself. These are *not* stored within the configuration.

    A configuration is made up of four types of dictionaries:

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
            The name of the component class. This is necessar in order to
            load the component from file.

    The settings and mutable_settings completely define the settings needed
    to configure a component. The attribute check_values is used to apply
    additional checks for compatibility.
    The settings and mutable_settings get combined in a config dictionary
    for convenience.
    """

    @property
    def class_string(self):
        return self._dict['class_string']

    @property
    def check_values(self):
        return self._dict['check_values']

    @property
    def mutable_settings(self):
        return self._dict['mutable_settings']

    @property
    def settings(self):
        return self._dict['settings']

    @property
    def sub_component_configurations(self):
        return self._dict['sub_component_configurations']

    @property
    def dependent_sub_components(self):
        return self._dict['dependent_sub_components']

    @property
    def dict(self):
        return self._dict

    @property
    def config(self):
        """The combination of mutable and constant settings.

        Returns
        -------
        dict
            Combined 'settings' and 'mutable_settings'
        """
        return self._config

    def __init__(self, class_string, settings, mutable_settings={},
                 check_values={}, dependent_sub_components=[],
                 sub_component_configurations={}, logger=None):
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
        sub_component_configurations : dict, optional
            A dictionary of sub component configurations.
            Additional sub components may be added after instatiation via
            the 'add_sub_components' method.
        logger : logging.logger, optional
            The logger to use.
        """
        self._logger = logger or logging.getLogger(__name__)
        self._dict = {
            'class_string': class_string,
            'check_values': dict(deepcopy(check_values)),
            'settings': dict(deepcopy(settings)),
            'mutable_settings': dict(deepcopy(mutable_settings)),
            'sub_component_configurations':
                dict(deepcopy(sub_component_configurations)),
            'dependent_sub_components':
                list(deepcopy(dependent_sub_components)),
        }

        # make sure defined dependen sub components exist in the configuration
        self._check_dependent_names(self.dependent_sub_components)
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
                msg = 'Sub component {!r} does not exist in {!r}'
                raise KeyError(msg.format(
                    name, self.sub_component_configurations.keys()))

    def add_dependent_sub_components(self, dependet_components):
        self._check_dependent_names(dependet_components)
        self._dict['dependent_sub_components'].extend(dependet_components)

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
                                        set(self.mutable_settings.keys()))
        if duplicates:
            raise ValueError('Keys are defined multiple times: {!r}'.format(
                                                                duplicates))

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
                raise TypeError('Incorrect type: {!r}'.format(type(component)))
            if name in self.sub_component_configurations:
                raise KeyError('Sub component {!r} already exists!'.format(
                                                                        name))
            if not component.is_configured:
                raise ValueError('Component {!r} is not configured!'.format(
                                                                        name))
            self.sub_component_configurations[name] = \
                component.configuration.dict

    def is_compatible(self, other):
        """Check compatibility between two configuration objects.

        Parameters
        ----------
        other : Configuartion object
            Another Configuration instance.

        Returns
        -------
        bool
            True if configuration objects are compatible, false if not.
        """
        if self.settings != other.settings or \
                self.check_values != other.check_values or \
                self.class_string != other.class_string:
            return False

        if set(self.sub_component_configurations.keys()) != \
                set(other.sub_component_configurations.keys()):
            return False

        # now check if sub components are also compatible
        for name, sub_conf in self.sub_component_configurations.items():

            # create a Configuration object from sub component
            this_sub_configuration = Configuration(**sub_conf)
            other_sub_configuration = Configuration(
                **other.sub_component_configurations[name])

            if not this_sub_configuration.is_compatible(
                    other_sub_configuration):
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
            msg = 'The following mutable settings have changed: {!r}'
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
    self._configure(). A derived class must at least implement and overrride
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
            A dictionary with dependent sub components..
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
            self, lambda a: not inspect.isroutine(a))

        filtered_attributes = [a[0] for a in attributes if not
                               (a[0].startswith('__') and a[0].endswith('__'))]

        # tracked or knowingly untracked data
        tracked_attributes = ['_is_configured', '_logger', '_configuration',
                              '_data', '_untracked_data', 'is_configured',
                              'logger', 'configuration', 'data',
                              'untracked_data', '_sub_components',
                              'sub_components']
        untracked_attributes = [a for a in filtered_attributes if a not in
                                tracked_attributes]

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
        """Check if class instance has any attributes it should not after
        instantiation and before configuration.

        Raises
        ------
        ValueError
            If member attributes are found that should not be there.
        """
        untracked_attributes = self._get_untracked_member_attributes()
        if len(untracked_attributes) > 0:
            msg = 'Class contains member variables that it should not at '
            msg += 'this point: {!r}'
            raise ValueError(msg.format(untracked_attributes))

    def configure(self, **kwargs):
        """Configure the BaseComponent instance.

        If additional components are passed via the kwargs argument, these
        will be automatically collected and their configurations accumulated.

        Parameters
        ----------
        **kwargs
            Arbitrary keyword arguments.
        """
        if self.is_configured:
            raise ValueError('Component is already configured!')

        # Get the dictionary format of the passed Configuration object
        if isinstance(kwargs, Configuration):
            kwargs = kwargs.dict

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
                    msg = 'Component {!r} is not configured!'
                    raise ValueError(msg.format(key))
            else:
                missing_settings[key] = value
        self._configuration, self._data, self._sub_components = \
            self._configure(**kwargs)

        if self._sub_components is None:
            self._sub_components = {}

        if not isinstance(self._sub_components, dict):
            msg = 'Sub components must be provided as a dictionary, '
            msg += 'but are: {!r}'
            raise TypeError(msg.format(type(self._sub_components)))

        # add sub component configurations to this configuration
        self._configuration.add_sub_components(parameter_sub_components)

        # add dependent sub components to this configuration
        self._configuration.add_dependent_sub_components(
            [component for component in self._sub_components.keys()])

        # check if data has correct type
        if self._data is not None:
            if not isinstance(self._data, dict):
                raise TypeError('Wrong type {!r}. Expected dict.'.format(
                    type(self._data)))

        # check if passed settings are all defined in configuration
        # and if they are set to the correct values. A component may have
        # more settings than these.
        for key, value in missing_settings.items():

            # setting is missing
            if key not in self._configuration.config:
                msg = 'Key {!r} is missing in configuration {!r}'
                raise ValueError(msg.format(key, self._configuration.config))

            # setting has wrong value
            elif value != self._configuration.config[key]:
                msg = 'Values of {!r} do not match: {!r} != {!r}'
                raise ValueError(msg.format(key, value,
                                            self._configuration.config[key]))

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
            which are passed as parameters into the configure method,
            as these are automatically gathered. The dependent_sub_components
            may also be left empty for these passed sub components.
            Sum components created within a component must be added.
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
            Return None if the component has no data.
        dict
            A dictionary of dependent sub components. This is a dictionary
            of sub components that need to be saved and loaded recursively
            when the component is saved and loaded.
            Return None if no dependent sub components exist.
        """
        raise NotImplementedError()

    def is_compatible(self, other):
        """Check the compatibilty of this component with another.

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

    def save(self, dir_path, overwrite=False,
             allow_untracked_attributes=False):
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
        """
        if not self.is_configured:
            raise ValueError('Component is not configured!')

        # check if there are any untracked attributes
        untracked_attributes = self._get_untracked_member_attributes()
        if len(untracked_attributes) > 0:
            msg = 'Found untracked attributes: {!r}'
            if allow_untracked_attributes:
                self._logger.warning(msg.format(untracked_attributes))
            else:
                raise AttributeError(msg.format(untracked_attributes))

        # split of possible extension
        dir_path = os.path.splitext(dir_path)[0]
        config_file_path = os.path.join(dir_path, 'configuration.yaml')
        data_file_path = os.path.join(dir_path, 'data.pickle')

        # check if file already exists
        for file in [config_file_path, data_file_path]:
            if os.path.exists(file):
                if overwrite:
                    msg = 'Overwriting existing file at: {}'
                    self._logger.info(msg.format(file))
                else:
                    raise IOError('File {!r} already exists!'.format(file))

        # check if directory needs to be created
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
            self._logger.info('Creating directory: {}'.format(dir_path))

        # save configuration settings
        with open(config_file_path, 'w') as yaml_file:
            yaml.dump(self.configuration.dict, yaml_file)

        # write data
        with open(data_file_path, 'wb') as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # walk through dependent sub components and save these as well
        for name, sub_component in self.sub_components.items():
            sub_component_dir_path = os.path.join(dir_path, name)
            sub_component.save(
                sub_component_dir_path, overwrite=overwrite,
                allow_untracked_attributes=allow_untracked_attributes)

        # call additional tasks of derived class via virtual method
        self._save(dir_path)

    def load(self, dir_path):
        """Load component from file.

        Parameters
        ----------
        dir_path : str
            The path to the input directory from which the component will be
            loaded.
        """
        if self.is_configured:
            raise ValueError('Component is already configured!')

        # check if it already has attributes other than the allowed ones
        # This indicates an incorrect usage of the BaseComponent class.
        self._check_member_attributes()

        # split of possible extension
        dir_path = os.path.splitext(dir_path)[0]
        config_file_path = os.path.join(dir_path, 'configuration.yaml')
        data_file_path = os.path.join(dir_path, 'data.pickle')

        # load files
        with open(config_file_path, 'r') as stream:
            config_dict = yaml.load(stream, Loader=yaml.Loader)

        with open(data_file_path, 'rb') as handle:
            data = pickle.load(handle)

        # get configuration of self:
        self._configuration = Configuration(**config_dict)

        # check if this is the correct class
        if self.configuration.class_string != \
                misc.get_full_class_string_of_object(self):
            msg = "The object's class {!r} does not match the saved "
            msg += 'class string {!r}'
            raise TypeError(msg.format(
                misc.get_full_class_string_of_object(self),
                self.configuration.class_string))

        # get data and check if it has correct type
        self._data = data
        if self._data is not None:
            if not isinstance(self._data, dict):
                raise ValueError('Wrong type {!r}. Expected dict.'.format(
                    type(self._data)))

        # walk through dependent sub components and load these as well
        for name in self.configuration.dependent_sub_components:
            sub_configuration = \
                self.configuration.sub_component_configurations[name]
            sub_component_dir_path = os.path.join(dir_path, name)

            sub_class = misc.load_class(sub_configuration['class_string'])
            sub_component = sub_class()
            sub_component.load(sub_component_dir_path)
            self._sub_components[name] = sub_component

        # call additional tasks of derived class via virtual method
        self._load(dir_path)

        self._is_configured = True

    def _save(self, dir_path):
        """Virtual method for additional save tasks by derived class

        This is a virtual method that may be overwritten by derived class
        to perform additional tasks necessary to save the component.
        This can for instance be saving of tensorflow model weights.

        Parameters
        ----------
        dir_path : str
            The path to the output directory to which the component will be
            saved.
        """
        pass

    def _load(self, dir_path):
        """Virtual method for additional load tasks by derived class

        This is a virtual method that may be overwritten by derived class
        to perform additional tasks necessary to load the component.
        This can for instance be loading of tensorflow model weights.

        Parameters
        ----------
        dir_path : str
            The path to the input directory from which the component will be
            loaded.
        """
        pass
