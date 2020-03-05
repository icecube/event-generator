from __future__ import division, print_function
import os
import logging
import numpy as np
import pandas as pd

from egenerator import misc
from egenerator.data.handler.base import BaseDataHandler


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
    filter_module : FilterModule
        A filter module that is used to filter certain events from the input
        batch.
    label_module : LabelModule
        The weight module which handles loading of tensors of type 'label'.
    misc_module : MiscModule
        The weight module which handles loading of tensors of type 'misc'.
    modules_are_loaded : bool
        Indicates whether the models have been loaded.
        If True, the modules have been loaded.
    weight_module : WeightModule
        The weight module which handles loading of tensors of type 'weight'.
    """

    def __init__(self, logger=None):
        """Initialize data handler

        Parameters
        ----------
        logger : logging.logger, optional
            A logging instance.
        """
        self.modules_are_loaded = False
        self.data_module = None
        self.label_module = None
        self.weight_module = None
        self.misc_module = None
        self.filter_module = None

        self.data_tensors = None
        self.label_tensors = None
        self.weight_tensors = None
        self.misc_tensors = None

        logger = logger or logging.getLogger(__name__)
        super(BaseModuleDataHandler, self).__init__(logger=logger)

    def _configure_settings_of_derived_class(self):
        """Perform any additional operations to setup and configure the
        derived class.
        When this method is called, the member variables:
            'tensors', 'config', and 'skip_check_keys'
        have been set and may be used.
        """
        self.data_tensors = DataTensorList([l for l in self.tensors.list
                                            if l.type == 'data'])
        self.label_tensors = DataTensorList([l for l in self.tensors.list
                                             if l.type == 'label'])
        self.weight_tensors = DataTensorList([l for l in self.tensors.list
                                              if l.type == 'weight'])
        self.misc_tensors = DataTensorList([l for l in self.tensors.list
                                            if l.type == 'misc'])

        # load modules if not loaded yet
        self._load_modules(config)

        # configure modules
        data_tensors = self.data_module.configure(self.data_tensors)
        label_tensors = self.label_module.configure(self.label_tensors)
        weight_tensors = self.weight_module.configure(self.weight_tensors)
        misc_tensors = self.misc_module.configure(self.misc_tensors)
        self.filters.configure()

        # check if tensors match
        for t1, t2 in zip((data_tensors, self.data_tensors),
                          (label_tensors, self.label_tensors),
                          (weight_tensors, self.weight_tensors),
                          (misc_tensors, self.misc_tensors)):
            if not t1 == t2:
                raise ValueError('{!r} != {!r}'.format(t1, t2))

        # check if skip_check_keys match
        skip_check_keys = self.get_skip_check_keys()
        if self.skip_check_keys != skip_check_keys:
            raise ValueError('{!r} != {!r}'.format(self.skip_check_keys,
                                                   skip_check_keys))

    def _load_modules(self, config):
        """Load modules

        Parameters
        ----------
        config : dict
            Configuration of the DataHandler.
        """
        if self.modules_are_loaded:
            raise ValueError('Modules have already been loaded!')

        base = 'egenerator.data.modules.{}.{}'

        # load the data loader module
        if self.data_module is None:
            data_class = misc.load_class(
                base.format('data', config['data_module']))
            self.data_module = data_class(**config['data_settings'])

        # load the label loader module
        if self.label_module is None:
            label_class = misc.load_class(
                base.format('label', config['label_module']))
            self.label_module = label_class(**config['label_settings'])

        # load the weight loader module
        if self.weight_module is None:
            weight_class = misc.load_class(
                base.format('weights', config['weight_module']))
            self.weight_module = weight_class(**config['weight_settings'])

        # load the misc loader module
        if self.misc_module is None:
            misc_class = misc.load_class(
                base.format('misc', config['misc_module']))
            self.misc_module = misc_class(**config['misc_settings'])

        # load the filter module
        if self.filter_module is None:
            filter_class = misc.load_class(
                base.format('filters', config['filter_module']))
            self.filter_module = filter_class(**config['filter_settings'])

        self.modules_are_loaded = True

    def _setup(self, config, test_data=None):
        """Setup the datahandler with a test input file.

        Parameters
        ----------
        config : dict
            Configuration of the DataHandler.
        test_data : list of str, optional
            List of valid file paths to input data files. The first of the
            specified files will be read in to obtain meta data and to setup
            the data handler.

        Returns
        -------
        DataTensorList
            A list of DataTensor objects. These are the tensors the data
            handler will create and load. They must always be in the same order
            and have the described settings.
        dict
            Configuration of the DataHandler.
        list
            List of keys in the config that do not need to be checked, e.g.
            that may change.
        """
        if test_data is None:
            raise ValueError('Must provide a test_data file!')

        # load the label loader module
        self._load_modules(config)

        # read test file and collect meta data
        self.data_tensors = self.data_module.configure(test_data)
        self.label_tensors = self.label_module.configure(test_data)
        self.weight_tensors = self.weight_module.configure(test_data)
        self.misc_tensors = self.misc_module.configure(test_data)
        self.filters.configure()

        # combine tensors
        tensors = DataTensorList(data_tensors.list + label_tensors.list +
                                 weight_tensors.list + misc_tensors.list)

        # get a list of keys whose settings do not have to match
        skip_check_keys = self.get_skip_check_keys()
        return tensors, config, skip_check_keys

    def get_skip_check_keys(self):
        """Get a list of config keys which do not have to match

        Returns
        -------
        list of str
            A list of config keys which do not have to match.
        """
        if not self.modules_are_loaded:
            raise ValueError('Modules must first be loaded!')

        keys = self.data_module.get_skip_check_keys() + \
            self.label_module.get_skip_check_keys() +\
            self.weight_module.get_skip_check_keys() +\
            self.misc_module.get_skip_check_keys() +\
            self.filter_module.get_skip_check_keys()
        return sorted(keys)

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

        # combine data in correct order
        num_events = None

        def check_num_events(num):
            if num_events is None:
                num_events = num
            elif num_events != num:
                raise ValueError('{!r} != {!r}'.format(num_events, num))

        event_batch = []
        for tensor in self.tensors:

            if tensor.type == 'data':
                check_num_events(num_data)
                event_batch.append(
                    data[self.data_tensors.get_index(tensor.name)])

            elif tensor.type == 'label':
                check_num_events(num_labels)
                event_batch.append(
                    labels[self.label_tensors.get_index(tensor.name)])

            elif tensor.type == 'misc':
                check_num_events(num_misc)
                event_batch.append(
                    misc[self.misc_tensors.get_index(tensor.name)])

                check_num_events(num_weights)
            elif tensor.type == 'weight':
                event_batch.append(
                    weights[self.weight_tensors.get_index(tensor.name)])

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
        return self.data_module.write_data_to_frame(frame, *args, **kwargs)
