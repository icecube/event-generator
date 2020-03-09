from __future__ import division, print_function
import os
import logging
import numpy as np
import pandas as pd

from egenerator import misc
from egenerator.manager.component import Configuration
from egenerator.data.handler.base import BaseDataHandler
from egenerator.data.tensor import DataTensorList


class ModuleDataHandler(BaseDataHandler):

    """A derived class from the base data handler 'BaseDataHandler'.
    This class uses a modular structure to load a data, label, misc, weight,
    and filter module. These modules are responsible for loading the
    corresponding data tensors.

    Attributes
    ----------
    data_module : DataModule
        The data module which is responsible for loading and creating of
        tensors of type 'data'.
    data_tensors : DataTensorList
        The data tensor list of tensors of type 'data'.
    filter_module : FilterModule
        A filter module that is used to filter certain events from the input
        batch.
    label_module : LabelModule
        The weight module which handles loading of tensors of type 'label'.
    label_tensors : DataTensorList
        The data tensor list of tensors of type 'label'.
    misc_module : MiscModule
        The weight module which handles loading of tensors of type 'misc'.
    misc_tensors : DataTensorList
        The data tensor list of tensors of type 'misc'.
    modules_are_loaded : bool
        Indicates whether the models have been loaded.
        If True, the modules have been loaded.
    weight_module : WeightModule
        The weight module which handles loading of tensors of type 'weight'.
    weight_tensors : DataTensorList
        The data tensor list of tensors of type 'weight'.
    """
    def _get_value(self, name, member_attribute):
        data_dict = getattr(self, member_attribute)
        if data_dict is not None and name in data_dict:
            return data_dict[name]
        else:
            return None

    @property
    def data_module(self):
        return self._get_value('data_module', '_sub_components')

    @property
    def data_tensors(self):
        return self._get_value('data_tensors', '_data')

    @property
    def filter_module(self):
        return self._get_value('filter_module', '_sub_components')

    @property
    def label_module(self):
        return self._get_value('label_module', '_sub_components')

    @property
    def label_tensors(self):
        return self._get_value('label_tensors', '_data')

    @property
    def misc_module(self):
        return self._get_value('misc_module', '_sub_components')

    @property
    def misc_tensors(self):
        return self._get_value('misc_tensors', '_data')

    @property
    def modules_are_loaded(self):
        return self._get_value('modules_are_loaded', '_untracked_data')

    @property
    def weight_module(self):
        return self._get_value('weight_module', '_sub_components')

    @property
    def weight_tensors(self):
        return self._get_value('weight_tensors', '_data')

    def __init__(self, logger=None):
        """Initialize data handler

        Parameters
        ----------
        logger : logging.logger, optional
            A logging instance.
        """
        # self.data['data_module'] = None
        # self._label_module = None
        # self._weight_module = None
        # self._misc_module = None
        # self._filter_module = None

        # self.data['data_tensors'] = None
        # self.data['label_tensors'] = None
        # self.data['misc_tensors'] = None
        # self.data['weight_tensors'] = None

        logger = logger or logging.getLogger(__name__)
        super(ModuleDataHandler, self).__init__(logger=logger)

        self._untracked_data['modules_are_loaded'] = False

    # def _assign_settings_of_derived_class(self, tensors, config,
    #                                       skip_check_keys):
    #     """Perform operations to setup and configure the derived class.
    #     This is only necessary when the data handler is loaded
    #     from file. In this case the setup methods '_setup' and 'setup' have
    #     not run.

    #     Parameters
    #     ----------
    #     tensors : DataTensorList
    #         A list of DataTensor objects. These are the tensors the data
    #         handler will create and load. They must always be in the same order
    #         and have the described settings.
    #     config : dict
    #         Configuration of the DataHandler.
    #     skip_check_keys : list
    #         List of keys in the config that do not need to be checked, e.g.
    #         that may change.
    #     """

    #     self.data['data_tensors'] = DataTensorList([l for l in tensors.list
    #                                         if l.type == 'data'])
    #     self.data['label_tensors'] = DataTensorList([l for l in tensors.list
    #                                          if l.type == 'label'])
    #     self.data['weight_tensors'] = DataTensorList([l for l in tensors.list
    #                                           if l.type == 'weight'])
    #     self.data['misc_tensors'] = DataTensorList([l for l in tensors.list
    #                                         if l.type == 'misc'])

    #     # load modules
    #     self._load_modules(config)

    #     # configure modules if this has not happened yet
    #     data_tensors = self.data_module.configure(self.data_tensors)
    #     label_tensors = self.label_module.configure(self.label_tensors)
    #     weight_tensors = self.weight_module.configure(self.weight_tensors)
    #     misc_tensors = self.misc_module.configure(self.misc_tensors)
    #     self._filter_module.configure()

    #     # check if tensors match
    #     for t1, t2 in ((data_tensors, self.data_tensors),
    #                    (label_tensors, self.label_tensors),
    #                    (weight_tensors, self.weight_tensors),
    #                    (misc_tensors, self.misc_tensors)):
    #         if not t1 == t2:
    #             raise ValueError('{!r} != {!r}'.format(t1, t2))

    #     # check if skip_check_keys match
    #     skip_check_keys = self.get_skip_check_keys()
    #     if self.skip_check_keys != skip_check_keys:
    #         raise ValueError('{!r} != {!r}'.format(self.skip_check_keys,
    #                                                skip_check_keys))

    def _load_modules(self, config):
        """Load modules

        Parameters
        ----------
        config : dict
            Configuration of the DataHandler.

        Returns
        -------
        dict
            A dictionary of the sub components.
        """
        if self.modules_are_loaded:
            raise ValueError('Modules have already been loaded!')

        base = 'egenerator.data.modules.{}.{}'

        sub_components = {}

        # load the data loader module
        sub_components['data_module'] = misc.load_class(
            base.format('data', config['data_module']))()

        # load the label loader module
        sub_components['label_module'] = misc.load_class(
            base.format('labels', config['label_module']))()

        # load the weight loader module
        sub_components['weight_module'] = misc.load_class(
            base.format('weights', config['weight_module']))()

        # load the misc loader module
        sub_components['misc_module'] = misc.load_class(
            base.format('misc', config['misc_module']))()

        # load the filter module
        sub_components['filter_module'] = misc.load_class(
            base.format('filters', config['filter_module']))()

        self._untracked_data['modules_are_loaded'] = True
        return sub_components

    def _configure_derived_class(self, config, config_data=None):
        """Configure the data handler with a test input file.

        Parameters
        ----------
        config : dict
            Configuration of the data handler.
        config_data : str or list of str, optional
            File name pattern or list of file patterns which define the paths
            to input data files. The first of the specified files will be
            read in to obtain meta data.

        Returns
        -------
        Configuration object
            The configuration object of the newly configured component.
            This does not need to include configurations of sub components
            which are passed as parameters into the configure method,
            as these are automatically gathered. The dependent_sub_components
            may also be left empty for these passed sub components.
            Sub components created within (and not passed to) this component
            must be added.
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
            of all sub components that need to be saved and loaded recursively
            when the component is saved and loaded.
            Return None if no dependent sub components exist.
        """

        # load the label loader module
        sub_components = self._load_modules(config)

        # read test file and collect meta data
        data = {}
        configuration_dict = {}
        for name in ['data', 'label', 'weight', 'misc', 'filter']:
            sub_components[name+'_module'].configure(
                    config_data=config_data, **config[name + '_settings'])

            data.update(sub_components[name+'_module'].data)

        # data['data_tensors'] = \
        #     sub_components['data_module'].configure(config_data)
        # data['label_tensors'] = \
        #     sub_components['label_module'].configure(config_data)
        # data['weight_tensors'] = \
        #     sub_components['weight_module'].configure(config_data)
        # data['misc_tensors'] = \
        #     sub_components['misc_module'].configure(config_data)
        # _, filter_data, _ = sub_components['filter_module'].configure()

        # combine tensors
        data['tensors'] = DataTensorList(
            data['data_tensors'].list + data['label_tensors'].list +
            data['weight_tensors'].list + data['misc_tensors'].list)

        # define configuration
        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings={'config': config})

        # add sub component configurations to this configuration
        configuration.add_sub_components(sub_components)

        return configuration, data, sub_components

    # def get_skip_check_keys(self):
    #     """Get a list of config keys which do not have to match

    #     Returns
    #     -------
    #     list of str
    #         A list of config keys which do not have to match.
    #     """
    #     if not self.modules_are_loaded:
    #         raise ValueError('Modules must first be loaded!')

    #     keys = self.data_module.get_skip_check_keys() + \
    #         self._label_module.get_skip_check_keys() +\
    #         self._weight_module.get_skip_check_keys() +\
    #         self._misc_module.get_skip_check_keys() +\
    #         self._filter_module.get_skip_check_keys()
    #     return sorted(keys)

    def _get_data_from_hdf(self, file, *args, **kwargs):
        """Get data from hdf file.

        Parameters
        ----------
        file : str
            The path to the hdf file.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        int
            Number of events.
        tuple of array-like tensors
            The input data (array-like) as specified in the
            DataTensorList (self.tensors).

        """

        # load data
        num_data, data = self.data_module.get_data_from_hdf(
                                                        file, *args, **kwargs)
        num_labels, labels = self.label_module.get_data_from_hdf(
                                                        file, *args, **kwargs)
        num_misc, misc = self.misc_module.get_data_from_hdf(
                                                        file, *args, **kwargs)
        num_weights, weights = self.weight_module.get_data_from_hdf(
                                                        file, *args, **kwargs)

        # check if there were any problems loading files and skip if there were
        found_problem = False
        if self.data_tensors.len > 0 and num_data is None and data is None:
            found_problem = True
        if self.label_tensors.len > 0 and num_labels is None \
                and labels is None:
            found_problem = True
        if self.misc_tensors.len > 0 and num_misc is None and misc is None:
            found_problem = True
        if self.weight_tensors.len > 0 and num_weights is None \
                and weights is None:
            found_problem = True
        if found_problem:
            self.logger.warning('Found missing values, skipping {!r}'.format(
                file))
            return None, None

        # combine data in correct order
        num_events = None

        def check_num_events(num_events, num):
            if num_events is None:
                num_events = num
            elif num_events != num:
                raise ValueError('{!r} != {!r}'.format(num_events, num))
            return num_events

        event_batch = []
        for tensor in self.tensors.list:

            if tensor.exists:
                if tensor.type == 'data':
                    num_events = check_num_events(num_events, num_data)
                    event_batch.append(
                        data[self.data_tensors.get_index(tensor.name)])

                elif tensor.type == 'label':
                    num_events = check_num_events(num_events, num_labels)
                    event_batch.append(
                        labels[self.label_tensors.get_index(tensor.name)])

                elif tensor.type == 'misc':
                    num_events = check_num_events(num_events, num_misc)
                    event_batch.append(
                        misc[self.misc_tensors.get_index(tensor.name)])

                elif tensor.type == 'weight':
                    num_events = check_num_events(num_events, num_weights)
                    event_batch.append(
                        weights[self.weight_tensors.get_index(tensor.name)])
            else:
                event_batch.append(None)

        if num_events is None:
            raise ValueError('Something went wrong!')

        num_events, event_batch = self.filter_module.filter_data(
                                        self.tensors, num_events, event_batch)

        return num_events, event_batch

    def _get_data_from_frame(self, frame, *args, **kwargs):
        """Get data from I3Frame.
        This will only return tensors of type 'data'.

        Parameters
        ----------
        frame : I3Frame
            The I3Frame from which to get the data.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        tuple of array-like tensors
            The input data (array-like) as specified in the
            DataTensorList (self.tensors).
        """
        return self.data_module.get_data_from_frame(frame, *args, **kwargs)

    def _create_data_from_frame(self, frame, *args, **kwargs):
        """Create data from I3Frame.
        This will only return tensors of type 'data'.

        Parameters
        ----------
        frame : I3Frame
            The I3Frame from which to get the data.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        tuple of array-like tensors
            The input data (array-like) as specified in the
            DataTensorList (self.tensors).
        """
        return self.data_module.create_data_from_frame(frame, *args, **kwargs)

    def _write_data_to_frame(self, data, frame, *args, **kwargs):
        """Write data to I3Frame.
        This will only write tensors of type 'data' to frame.

        Parameters
        ----------
        data : tuple of array-like tensors
            The input data (array-like) as specified in the
            DataTensorList (self.tensors).
        frame : I3Frame
            The I3Frame to which the data is to be written to.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        return self.data_module.write_data_to_frame(
            data, frame, *args, **kwargs)
