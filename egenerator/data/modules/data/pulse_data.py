from __future__ import division, print_function

import logging
import numpy as np
import pandas as pd

from egenerator import misc
from egenerator.manager.component import BaseComponent, Configuration
from egenerator.data.tensor import DataTensorList, DataTensor


class PulseDataModule(BaseComponent):

    """Pulse data module

    This data module loads unbinned pulse data and total dom charge.
    In addition, dom and time exclusions will be loaded if the keys are
    provided.

    Attributes
    ----------
    dom_exclusions_exist : bool
        Indicates wheter a DOM exclusions key was provided.
    time_exclusions_exist : bool
        Indicates wheter a time exclusions key was provided.
    """

    def __init__(self, logger=None):
        """Initialize pulse data module

        Parameters
        ----------
        logger : logging.logger, optional
            A logging instance.
        """

        logger = logger or logging.getLogger(__name__)
        super(PulseDataModule, self).__init__(logger=logger)

    def _configure(self, config_data, pulse_key, dom_exclusions_key,
                   time_exclusions_key, float_precision):
        """Configure Module Class
        This is an abstract method and must be implemented by derived class.

        Parameters
        ----------
        config_data : None, str, or DataTensorList
            This is either the path to a test file or a data tensor list
            object. The module will be configured with this.
        pulse_key : str
            The key in which the pulse series are saved to.
        dom_exclusions_key : str, optional
            The key in which the dom exclusions are saved to.
        time_exclusions_key : str, optional
            The key in which the time window exclusions are saved to.
        float_precision : str
            The float precision as a str.

        Returns
        -------
        Configuration object
            The configuration object of the newly configured component.
            This does not need to include configurations of sub components
            as these are automatically gathered.The dependent_sub_components
            may also be left empty. This is later filled by the base class
            from the returned sub components dict.
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
                'data_tensors': DataTensorList
                    The tensors of type 'data' that will be loaded.
        dict
            A dictionary of dependent sub components. This is a dictionary
            of sub components that need to be saved and loaded recursively
            when the component is saved and loaded.
            Return None if no dependent sub components exist.
        """
        if not isinstance(pulse_key, str):
            msg = 'Pulse key type: {!r} != str'
            raise TypeError(msg.format(type(pulse_key)))

        time_exclusions_exist = time_exclusions_key is not None
        dom_exclusions_exist = dom_exclusions_key is not None

        x_dom_charge = DataTensor(name='x_dom_charge',
                                  shape=[None, 86, 60, 1],
                                  tensor_type='data',
                                  dtype=float_precision,
                                  trafo=True,
                                  trafo_reduce_axes=(1, 2),
                                  trafo_log=True,
                                  trafo_batch_axis=0)
        x_dom_exclusions = DataTensor(name='x_dom_exclusions',
                                      shape=[None, 86, 60, 1],
                                      tensor_type='data',
                                      dtype='bool',
                                      exists=dom_exclusions_exist)
        x_pulses = DataTensor(name='x_pulses',
                              shape=[None, 2],
                              tensor_type='data',
                              dtype=float_precision)
        x_pulses_ids = DataTensor(name='x_pulses_ids',
                                  shape=[None, 3],
                                  tensor_type='data',
                                  dtype='int32')
        x_time_exclusions = DataTensor(name='x_time_exclusions',
                                       shape=[None, 2],
                                       tensor_type='data',
                                       dtype=float_precision,
                                       exists=time_exclusions_exist)
        x_time_exclusions_ids = DataTensor(name='x_time_exclusions_ids',
                                           shape=[None, 3],
                                           tensor_type='data',
                                           dtype='int32',
                                           exists=time_exclusions_exist)

        data = {}
        data['data_tensors'] = DataTensorList([
            x_dom_charge, x_dom_exclusions, x_pulses, x_pulses_ids,
            x_time_exclusions, x_time_exclusions_ids])

        data['np_float_precision'] = getattr(np, float_precision)

        data['time_exclusions_exist'] = time_exclusions_exist
        data['dom_exclusions_exist'] = dom_exclusions_exist

        if isinstance(config_data, DataTensorList):
            if not config_data == data['data_tensors']:
                raise ValueError('{!r} != {!r}'.format(config_data,
                                                       data['data_tensors']))
        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings=dict(config_data=config_data,
                          float_precision=float_precision),
            mutable_settings=dict(pulse_key=pulse_key,
                                  dom_exclusions_key=dom_exclusions_key,
                                  time_exclusions_key=time_exclusions_key),
            )
        return configuration, data, {}

    def get_data_from_hdf(self, file, *args, **kwargs):
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
            DataTensorList (self.data['data_tensors']).

        """
        if not self.is_configured:
            raise ValueError('Module not configured yet!')

        # open file
        f = pd.HDFStore(file, 'r')

        try:
            pulses = f[self.configuration.config['pulse_key']]
            _labels = f['LabelsDeepLearning']
            if self.data['dom_exclusions_exist']:
                try:
                    dom_exclusions = \
                        f[self.configuration.config['dom_exclusions_key']]
                except KeyError:
                    msg = 'Could not find exclusion key {!r}'
                    self._logger.warning(msg.format(
                        self.configuration.config['dom_exclusions_key']))
                    dom_exclusions = None
            else:
                dom_exclusions = None

        except Exception as e:
            self._logger.warning('Skipping file: {} due to {}'.format(file, e))
            return None, None
        finally:
            f.close()

        # create Dictionary with event IDs
        size = len(_labels['Event'])
        event_dict = {}
        for row in _labels.iterrows():
            event_dict[(row[1][0], row[1][1], row[1][2], row[1][3])] = row[0]

        # create empty array for DOM charges
        x_dom_charge = np.zeros([size, 86, 60, 1],
                                dtype=self.data['np_float_precision'])

        if self.data['dom_exclusions_exist']:
            x_dom_exclusions = np.ones_like(x_dom_charge, dtype=bool)
        else:
            x_dom_exclusions = None

        if self.data['time_exclusions_exist']:
            raise NotImplementedError(
                'Time window exclusions not yet implemented!')
        else:
            x_time_exclusions = None
            x_time_exclusions_ids = None

        num_pulses = len(pulses['Event'])
        x_pulses = np.empty((num_pulses, 2),
                            dtype=self.data['np_float_precision'])
        x_pulses_ids = np.empty((num_pulses, 3), dtype=np.int32)

        # get pulse information
        for pulse_index, row in enumerate(pulses.itertuples()):
            string = row[6]
            dom = row[7]
            if dom > 60:
                self._logger.warning(
                    'skipping pulse: {} {}'.format(string, dom))
                continue
            index = event_dict[(row[1:5])]

            # pulse charge: row[12], time: row[10]
            # accumulate charge in DOMs
            x_dom_charge[index, string-1, dom-1, 0] += row[12]

            # gather pulses (charge, time)
            x_pulses[pulse_index] = [row[12], row[10]]

            # gather pulse ids (batch index, string dom)
            x_pulses_ids[pulse_index] = [index, string-1, dom-1]

        # get dom exclusions
        if dom_exclusions is not None:
            for row in dom_exclusions.itertuples():
                string = row[7]
                dom = row[8]
                if dom > 60:
                    msg = 'skipping exclusion DOM: {!r} {!r}'
                    self._logger.warning(msg.format(string, dom))
                    continue
                index = event_dict[(row[1:5])]
                x_dom_exclusions[index, string-1, dom-1, 0] = False

        # put everything together and make sure the order is correct
        data_dict = {
            'x_dom_charge': x_dom_charge,
            'x_dom_exclusions': x_dom_exclusions,
            'x_pulses': x_pulses,
            'x_pulses_ids': x_pulses_ids,
            'x_time_exclusions': x_time_exclusions,
            'x_time_exclusions_ids': x_time_exclusions_ids,
        }
        event_batch = []
        for tensor in self.data['data_tensors'].list:
            event_batch.append(data_dict[tensor.name])

        return size, event_batch

    def get_data_from_frame(self, frame, *args, **kwargs):
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
            DataTensorList (self.data['data_tensors']).
        """
        if not self.is_configured:
            raise ValueError('Module not configured yet!')

        raise NotImplementedError()

    def create_data_from_frame(self, frame, *args, **kwargs):
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
            DataTensorList (self.data['data_tensors']).
        """
        if not self.is_configured:
            raise ValueError('Module not configured yet!')

        raise NotImplementedError()

    def write_data_to_frame(self, data, frame, *args, **kwargs):
        """Write data to I3Frame.
        This will only write tensors of type 'data' to frame.

        Parameters
        ----------
        data : tuple of array-like tensors
            The input data (array-like) as specified in the
            DataTensorList (self.data['data_tensors']).
        frame : I3Frame
            The I3Frame to which the data is to be written to.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        if not self.is_configured:
            raise ValueError('Module not configured yet!')

        raise NotImplementedError()