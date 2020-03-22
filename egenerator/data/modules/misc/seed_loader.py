from __future__ import division, print_function

import logging
import numpy as np
import pandas as pd

from egenerator import misc
from egenerator.manager.component import BaseComponent, Configuration
from egenerator.data.tensor import DataTensorList, DataTensor


class SeedLoaderMiscModule(BaseComponent):

    """This is a misc module that loads seeds for Source Hypotheses.
    """

    def __init__(self, logger=None):
        """Initialize seed loader module

        Parameters
        ----------
        logger : logging.logger, optional
            A logging instance.
        """

        logger = logger or logging.getLogger(__name__)
        super(SeedLoaderMiscModule, self).__init__(logger=logger)

    def _configure(self, config_data, seed_names, seed_parameter_names,
                   float_precision):
        """Configure Module Class
        This is an abstract method and must be implemented by derived class.

        Parameters
        ----------
        config_data : str or DataTensorList
            This is either the path to a test file or a data tensor list
            object. The module will be configured with this.
        seed_names : list of str
            The list of seed names. These must specify data tables in the hdf
            file or keys in the I3Frame.
        seed_parameter_names : list of str
            The column names to extract from the data table in hdf file
            or from the I3Frame object.
            The values will be extracted and concatenated in the same order
            as the seed_parameter_names.
        float_precision : str
            The float precision as a str.

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
            The data of the component. Contains:
                'misc_tensors': DataTensorList
                    The tensors of type 'misc' that will be loaded.
        dict
            A dictionary of dependent sub components. This is a dictionary
            of sub components that need to be saved and loaded recursively
            when the component is saved and loaded.
            Return None if no dependent sub components exist.

        Raises
        ------
        ValueError
            Description
        """

        if not isinstance(config_data, (type(None), str, DataTensorList)):
            raise ValueError('Unknown type: {!r}'.format(type(config_data)))

        num_params = len(seed_parameter_names)

        # create data tensor list
        tensor_list = []
        for seed_name in seed_names:
            tensor_list.append(DataTensor(name=seed_name,
                                          shape=[None, num_params],
                                          tensor_type='misc',
                                          dtype=float_precision))

        data = {
            'misc_tensors': DataTensorList(tensor_list)
        }

        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings={'config_data': config_data},
            mutable_settings={'seed_names': seed_names,
                              'seed_parameter_names': seed_parameter_names,
                              'float_precision': float_precision},
            )
        return configuration, data, {}

    def get_data_from_hdf(self, file):
        """Get misc data from hdf file.

        Parameters
        ----------
        file : str
            The path to the hdf file.

        Returns
        -------
        int
            Number of events.
        tuple of array-like tensors or None
            The input data (array-like) as specified in the
            DataTensorList (self.tensors).
            Returns None if no misc data is loaded.
        """
        # open file
        f = pd.HDFStore(file, 'r')

        seeds = []
        num_events_list = []
        for seed_name in self.configuration.config['seed_names']:

            seed_parameters = []
            try:
                _labels = f[seed_name]
                for l in self.configuration.config['seed_parameter_names']:
                    seed_parameters.append(_labels[l])

            except Exception as e:
                self._logger.warning(e)
                self._logger.warning('Skipping file: {}'.format(file))
                return None, None
            finally:
                f.close()

            # format cascade parameters
            dtype = getattr(np, self.configuration.config['float_precision'])
            seed_parameters = np.array(seed_parameters, dtype=dtype).T

            seeds.append(seed_parameters)
            num_events = len(seed_parameters)
            num_events_list.append(num_events)

        # check if number of events matches accross all events
        if not np.all([n == num_events for n in num_events_list]):
            msg = 'Not all event numbers match: {!r} for seeds: {!r}.'
            raise ValueError(msg.format(
                num_events_list, self.configuration.config['seed_names']))

        return num_events, seeds
