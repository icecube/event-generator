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
        logger = logger or logging.getLogger(__name__)
        super(ModuleDataHandler, self).__init__(logger=logger)

        self._untracked_data['modules_are_loaded'] = False

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

        # combine tensors
        data['tensors'] = DataTensorList(
            data['data_tensors'].list + data['label_tensors'].list +
            data['weight_tensors'].list + data['misc_tensors'].list)

        # define configuration
        # set config to mutable_settings:
        #   sub components are checked recursively and may specify non-mutable
        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings={},
            mutable_settings={'config': config},
            mutable_sub_components=['weight_module', 'misc_module',
                                    'filter_module', 'label_module'])

        # add sub component configurations to this configuration
        configuration.add_sub_components(sub_components)

        return configuration, data, sub_components

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
            # check if the updated component is allowed to be updated
            if name not in ['data_module', 'label_module', 'weight_module',
                            'misc_module', 'filter_module']:
                msg = 'Can not update {!r} since it does not exist'
                raise ValueError(msg.format(name))

            # update data
            self._data.update(self.sub_components[name].data)

            # update mutable settings
            new_config = self.configuration.mutable_settings
            new_config['config'][name] = misc.get_full_class_string_of_object(
                self.sub_components[name])
            new_config['config'][name.replace('module', 'settings')] = \
                self.sub_components[name].configuration.config
            self.configuration.update_mutable_settings(new_config)

            # replace sub components configuration
            self.configuration.replace_sub_components(
                    {name: self.sub_components[name]})

        # update tensors list
        self._data['tensors'] = DataTensorList(
            self._data['data_tensors'].list +
            self._data['label_tensors'].list +
            self._data['weight_tensors'].list +
            self._data['misc_tensors'].list)

    def _get_data(self, file_or_frame, method, *args, **kwargs):
        """Get data from hdf file or I3Frame.

        Parameters
        ----------
        file_or_frame : str or I3Frame
            The path to the hdf file or the I3Frame from which to get the data.
        method : str
            The method how to obtain the data. Options are:
                'get_from_hdf', 'get_from_frame', 'create_from_frame'
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

        Raises
        ------
        ValueError
            Description
        """
        methods = ['get_from_hdf', 'get_from_frame', 'create_from_frame']
        if method not in methods:
            raise ValueError('Method {} is not in {!r}'.format(
                method, methods))

        # load data
        if method == 'get_from_hdf':
            assert isinstance(file_or_frame, str), 'Expected file path string'

            num_data, data = self.data_module.get_data_from_hdf(
                                                file_or_frame, *args, **kwargs)
            num_labels, labels = self.label_module.get_data_from_hdf(
                                                file_or_frame, *args, **kwargs)
            num_misc, misc = self.misc_module.get_data_from_hdf(
                                                file_or_frame, *args, **kwargs)
            num_weights, weights = self.weight_module.get_data_from_hdf(
                                                file_or_frame, *args, **kwargs)
        elif method == 'get_from_frame':
            num_data, data = self.data_module.get_data_from_frame(
                                                file_or_frame, *args, **kwargs)
            num_labels, labels = self.label_module.get_data_from_frame(
                                                file_or_frame, *args, **kwargs)
            num_misc, misc = self.misc_module.get_data_from_frame(
                                                file_or_frame, *args, **kwargs)
            num_weights, weights = self.weight_module.get_data_from_frame(
                                                file_or_frame, *args, **kwargs)
        elif method == 'create_from_frame':
            num_data, data = self.data_module.create_data_from_frame(
                                                file_or_frame, *args, **kwargs)
            num_labels, labels = self.label_module.create_data_from_frame(
                                                file_or_frame, *args, **kwargs)
            num_misc, misc = self.misc_module.create_data_from_frame(
                                                file_or_frame, *args, **kwargs)
            num_weights, weights = self.weight_module.create_data_from_frame(
                                                file_or_frame, *args, **kwargs)
        # check if there were any problems loading data and skip if there were
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
            msg = 'Found missing values, skipping {!r}'
            self._logger.warning(msg.format(file_or_frame))
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

        # get filter mask for events
        if method == 'get_from_hdf':
            filter_mask = self.filter_module.get_event_filter_mask_from_hdf(
                file_or_frame, self.tensors, num_events, event_batch,
                *args, **kwargs)
        elif method in ['get_from_frame', 'create_from_frame']:
            filter_mask = self.filter_module.get_event_filter_mask_from_frame(
                file_or_frame, self.tensors, num_events, event_batch,
                *args, **kwargs)

        # filter out events, if there is at least one event that is filtered
        if not np.all(filter_mask):
            num_events, event_batch = self._filter_data(filter_mask,
                                                        num_events,
                                                        event_batch)

        if num_events == 0:
            return None, None
        else:
            return num_events, event_batch

    def _filter_data(self, filter_mask, num_events, event_batch):
        """Filter events according to provided filter mask.

        Parameters
        ----------
        filter_mask : array_like
            An array of bool indicating whether an event passed the filter
            (True) or wheter it is filtered out (False).
            Shape: [num_events]
        num_events : int
            The number of loaded events.
        event_batch : tuple of array-like
            The data that needs to be filtered.

        Returns
        -------
        int
            The number of events left after filtering.
        tuple of array-like
            The filtered data.
        """
        filter_mask = np.asarray(filter_mask)
        assert filter_mask.shape == (num_events,)

        filtered_batch = [None for v in self.tensors.names]
        filtered_num_events = np.sum(filter_mask)

        for tensor_index, tensor in enumerate(self.tensors.list):

            # check if data exists
            if event_batch[tensor_index] is None:
                filtered_batch[tensor_index] = None

            if tensor.vector_info is not None:
                if tensor.vector_info['type'] == 'value':
                    # we are collecting value tensors together with
                    # the indices tensors, so skip for now
                    continue
                elif tensor.vector_info['type'] == 'index':
                    # get value tensor
                    value_index = self.tensors.get_index(
                        tensor.vector_info['reference'])
                    values = event_batch[value_index]
                    indices = event_batch[tensor_index]

                    if values is None or indices is None:
                        assert values == indices, '{!r} != {!r}'.format(
                            values, indices)
                        filtered_batch[value_index] = None
                        filtered_batch[tensor_index] = None
                    else:
                        # This data input is a vector type and must be
                        # restructured to event shape, filtered, and
                        # flattened again
                        # shape: [n_events, n_per_event, k] (not a tensor!)
                        values, indices = self.batch_to_event_structure(
                                            values, indices, num_events)

                        # filtered event values and indices
                        # shape: [n_filtered, n_per_event. k] (not a tensor!)
                        event_values_filterd = []
                        event_indices_filtered = []
                        for index, passed_filter in enumerate(filter_mask):
                            if passed_filter:
                                event_values_filterd.append(values[index])
                                event_indices_filtered.append(indices[index])

                        # set new batch indices
                        for batch_index in range(filtered_num_events):
                            event_indices_filtered[batch_index][:, 0] = \
                                batch_index

                        # shape: [n_pulses, k]
                        values, indices = self.event_to_batch_structure(
                            event_values_filterd,
                            event_indices_filtered,
                            filtered_num_events,
                        )

                        filtered_batch[value_index] = values
                        filtered_batch[tensor_index] = indices
            else:
                # This data input is already in event structure and
                # must simply be concatenated along axis 0.
                filtered_batch[tensor_index] = \
                    event_batch[tensor_index][filter_mask]

        return filtered_num_events, filtered_batch

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
        return self._get_data(file, method='get_from_hdf', *args, **kwargs)

    def _get_data_from_frame(self, frame, *args, **kwargs):
        """Get data from I3Frame.

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
        return self._get_data(frame, method='get_from_frame', *args, **kwargs)

    def _create_data_from_frame(self, frame, *args, **kwargs):
        """Create data from I3Frame.

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

        return self._get_data(
            frame, method='create_from_frame', *args, **kwargs)

    def _write_data_to_frame(self, data, frame, *args, **kwargs):
        """Write data to I3Frame.

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

        # loop through each module
        for name in ['data', 'label', 'misc', 'weight']:

            # get module
            module = getattr(self, name + '_module')

            # collect data tensors
            data_batch = []
            for i, tensor in enumerate(self.tensors.list):
                if tensor.type == name:
                    data_batch.append(data[i])

            # write_data to frame
            module.write_data_to_frame(data_batch, frame, *args, **kwargs)
