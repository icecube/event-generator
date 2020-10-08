from __future__ import division, print_function

import logging
import numpy as np
import pandas as pd

from egenerator import misc
from egenerator.utils import detector
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
                   time_exclusions_key, float_precision, add_charge_quantiles,
                   discard_pulses_from_excluded_doms):
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
        add_charge_quantiles : bool
            If True, the charge quantiles of the data pulses are computed and
            added to the `x_pulses` data tensor.
            A pulse then consists of (charge, time, quantile), where quantile
            is the fraction of cumulative charge collected at the DOM:
                quantile_i = 1./D * sum_{j=0}^{j=i} charge_j
            with the DOM's total charge D.
        discard_pulses_from_excluded_doms : bool, optional
            If True, pulses on excluded DOMs are discarded. The pulses are
            discarded after the charge at the DOM is collected.

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
                'data_tensors': DataTensorList
                    The tensors of type 'data' that will be loaded.
        dict
            A dictionary of dependent sub components. This is a dictionary
            of sub components that need to be saved and loaded recursively
            when the component is saved and loaded.
            Return None if no dependent sub components exist.

        Raises
        ------
        TypeError
            Description
        ValueError
            Description
        """
        if not isinstance(pulse_key, str):
            msg = 'Pulse key type: {!r} != str'
            raise TypeError(msg.format(type(pulse_key)))

        time_exclusions_exist = time_exclusions_key is not None
        dom_exclusions_exist = dom_exclusions_key is not None

        if add_charge_quantiles:
            pulse_dim = 3
        else:
            pulse_dim = 2

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
                              shape=[None, pulse_dim],
                              tensor_type='data',
                              vector_info={'type': 'value',
                                           'reference': 'x_pulses_ids'},
                              dtype=float_precision)
        x_pulses_ids = DataTensor(name='x_pulses_ids',
                                  shape=[None, 3],
                                  tensor_type='data',
                                  vector_info={'type': 'index',
                                               'reference': 'x_pulses'},
                                  dtype='int32')
        x_time_window = DataTensor(name='x_time_window',
                                   shape=[None, 2],
                                   tensor_type='data',
                                   dtype=float_precision)
        x_time_exclusions = DataTensor(
                            name='x_time_exclusions',
                            shape=[None, 2],
                            tensor_type='data',
                            vector_info={'type': 'value',
                                         'reference': 'x_time_exclusions_ids'},
                            dtype=float_precision,
                            exists=time_exclusions_exist)
        x_time_exclusions_ids = DataTensor(
                            name='x_time_exclusions_ids',
                            shape=[None, 3],
                            tensor_type='data',
                            vector_info={'type': 'index',
                                         'reference': 'x_time_exclusions'},
                            dtype='int32',
                            exists=time_exclusions_exist)

        data = {}
        data['data_tensors'] = DataTensorList([
            x_dom_charge, x_dom_exclusions, x_pulses, x_pulses_ids,
            x_time_window, x_time_exclusions, x_time_exclusions_ids])

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
                          float_precision=float_precision,
                          add_charge_quantiles=add_charge_quantiles),
            mutable_settings=dict(
                pulse_key=pulse_key,
                dom_exclusions_key=dom_exclusions_key,
                time_exclusions_key=time_exclusions_key,
                discard_pulses_from_excluded_doms=(
                    discard_pulses_from_excluded_doms
                )),
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

            if self.data['time_exclusions_exist']:
                try:
                    time_exclusions = \
                        f[self.configuration.config['time_exclusions_key']]
                except KeyError:
                    msg = 'Could not find time exclusion key {!r}'
                    self._logger.warning(msg.format(
                        self.configuration.config['time_exclusions_key']))
                    time_exclusions = None
            else:
                time_exclusions = None

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
            bad_doms = np.reshape(detector.bad_doms_mask, [1, 86, 60, 1])
            x_dom_exclusions = (
                np.ones_like(x_dom_charge) * bad_doms).astype(bool)
        else:
            x_dom_exclusions = None

        if self.data['time_exclusions_exist']:
            num_tws = len(time_exclusions['Event'])
            x_time_exclusions = np.empty(
                (num_tws, 2), dtype=self.data['np_float_precision'])
            x_time_exclusions_ids = np.empty((num_tws, 3), dtype=np.int32)
        else:
            x_time_exclusions = None
            x_time_exclusions_ids = None

        add_charge_quantiles = \
            self.configuration.config['add_charge_quantiles']
        if add_charge_quantiles:
            pulse_dim = 3
        else:
            pulse_dim = 2

        num_pulses = len(pulses['Event'])
        x_pulses = np.empty((num_pulses, pulse_dim),
                            dtype=self.data['np_float_precision'])
        x_pulses_ids = np.empty((num_pulses, 3), dtype=np.int32)

        # create array for time data
        x_time_window = np.empty(
            [size, 2],  dtype=self.data['np_float_precision'])
        x_time_window[:, 0] = float('inf')
        x_time_window[:, 1] = -float('inf')

        # ---------------------
        # get pulse information
        # ---------------------
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

            # gather pulses
            if add_charge_quantiles:

                # (charge, time, quantile)
                cum_charge = float(x_dom_charge[index, string-1, dom-1, 0])
                x_pulses[pulse_index] = [row[12], row[10], cum_charge]

            else:
                # (charge, time)
                x_pulses[pulse_index] = [row[12], row[10]]

            # gather pulse ids (batch index, string, dom)
            x_pulses_ids[pulse_index] = [index, string-1, dom-1]

            # update time window
            if row[10] > x_time_window[index, 1]:
                x_time_window[index, 1] = row[10]
            if row[10] < x_time_window[index, 0]:
                x_time_window[index, 0] = row[10]

        # convert cumulative charge to fraction of total charge, e.g. quantile
        if add_charge_quantiles:

            # compute flat indices
            dim1 = x_dom_charge.shape[1]
            dim2 = x_dom_charge.shape[2]
            flat_indices = (x_pulses_ids[:, 0]*dim1*dim2 +  # event
                            x_pulses_ids[:, 1]*dim2 +  # string
                            x_pulses_ids[:, 2])  # DOM

            # flatten dom charges
            flat_charges = x_dom_charge.flatten()

            # calculate quantiles
            x_pulses[:, 2] /= flat_charges[flat_indices]

        # -------------------
        # get time exclusions
        # -------------------
        if time_exclusions is not None:
            for tw_index, row in enumerate(time_exclusions.itertuples()):
                string = row[6]
                dom = row[7]
                if dom > 60:
                    self._logger.warning(
                        'skipping tw: {} {}'.format(string, dom))
                    continue
                index = event_dict[(row[1:5])]

                # t_start (pulse time): row[10], t_end (pulse width): row[11]

                # (t_start, t_end)
                x_time_exclusions[tw_index] = [row[10], row[11]]

                # gather pulse ids (batch index, string, dom)
                x_time_exclusions_ids[tw_index] = [index, string-1, dom-1]

            # ----------------------
            # safety check and hack:
            # ----------------------
            """Some files seem to have duplicate time windows which should
            never happen. Time Windows must not overlap!
            ToDo: Where do these come from?
            """
            unique_vals, indices, counts = np.unique(
                x_time_exclusions,
                axis=0,
                return_index=True,
                return_counts=True,
            )
            if len(unique_vals) < num_tws:

                # IDs that come up more than once
                # These may have potential overlaps
                double_ids = x_time_exclusions_ids[indices[counts > 1]]
                double_tws = x_time_exclusions[indices[counts > 1]]
                found_duplicates = False

                msg = 'Found possible duplicate Time Windows!'
                for double_id, double_tw in zip(double_ids, double_tws):
                    double_count = 0
                    for i, (tw_id, tw) in enumerate(zip(
                            x_time_exclusions_ids, x_time_exclusions)):
                        if ((double_id == tw_id).all()
                                and (double_tw == tw).all()):
                            double_count += 1

                    if double_count > 1:
                        msg += '\n\t {} duplicates for {} with TW: {}'.format(
                            double_count, double_id, double_tw)
                        found_duplicates = True

                if found_duplicates:
                    print(msg)
                    print('Skipping file: {} [Duplicate TimeWindow]'.format(
                        file))
                    return None, None

            def test_for_overlaps(tws, tw_ids):
                found_overlap = False
                overlaps = []
                tw_dict = {}
                for tw_id, tw in zip(tw_ids, tws):
                    tw_id = tuple(tw_id.tolist())
                    tw = tuple(tw.tolist())
                    if tw_id in tw_dict:

                        # check if there are any overlaps
                        for tw_previous in tw_dict[tw_id]:
                            if (
                                    # start in previous tw
                                    (tw[0] < tw_previous[1] and
                                     tw[0] >= tw_previous[0]) or

                                    # end in previous tw
                                    (tw[1] >= tw_previous[0] and
                                     tw[1] < tw_previous[1]) or

                                    # previous tw completely in new tw
                                    (tw[1] >= tw_previous[1] and
                                     tw[0] <= tw_previous[0]) or

                                    # new tw completely in previous tw
                                    (tw[1] <= tw_previous[1] and
                                     tw[0] >= tw_previous[0])
                                    ):

                                # found an overlap!
                                overlaps.append((tw_id, tw_previous, tw))
                                found_overlap = True

                        tw_dict[tw_id].append(tw)
                    else:
                        tw_dict[tw_id] = [tw]
                return found_overlap, overlaps

            found_overlap, overlaps = test_for_overlaps(
                x_time_exclusions, x_time_exclusions_ids)
            if found_overlap:
                print('Found overlaps in exlcusion time windows!')
                print(overlaps)
                print('Skipping file: {} [Overlapping TimeWindow]'.format(
                    file))
                return None, None
            # ----------------------

        # ------------------
        # get dom exclusions
        # ------------------
        if dom_exclusions is not None:
            for row in dom_exclusions.itertuples():
                string = row[7]
                dom = row[8]
                if dom > 60:
                    msg = 'skipping exclusion DOM: {!r} {!r}'
                    self._logger.info(msg.format(string, dom))
                    continue
                index = event_dict[(row[1:5])]
                x_dom_exclusions[index, string-1, dom-1, 0] = False

            if self.configuration.config['discard_pulses_from_excluded_doms']:

                # compute flat indices
                dim1 = x_dom_exclusions.shape[1]
                dim2 = x_dom_exclusions.shape[2]
                flat_indices = (x_pulses_ids[:, 0]*dim1*dim2 +  # event
                                x_pulses_ids[:, 1]*dim2 +  # string
                                x_pulses_ids[:, 2])  # DOM

                # flatten dom charges
                flat_exclusions = x_dom_exclusions.flatten()

                # discard pulses coming from excluded DOMs
                mask = flat_exclusions[flat_indices]
                x_pulses = x_pulses[mask]
                x_pulses_ids = x_pulses_ids[mask]

        # put everything together and make sure the order is correct
        data_dict = {
            'x_dom_charge': x_dom_charge,
            'x_dom_exclusions': x_dom_exclusions,
            'x_pulses': x_pulses,
            'x_pulses_ids': x_pulses_ids,
            'x_time_window': x_time_window,
            'x_time_exclusions': x_time_exclusions,
            'x_time_exclusions_ids': x_time_exclusions_ids,
        }
        event_batch = []
        for tensor in self.data['data_tensors'].list:
            event_batch.append(data_dict[tensor.name])

        return size, event_batch

    def get_data_from_frame(self, frame, *args, **kwargs):
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
            DataTensorList (self.data['data_tensors']).
        """
        from icecube import dataclasses

        if not self.is_configured:
            raise ValueError('Module not configured yet!')

        # get pulses
        pulses = frame[self.configuration.config['pulse_key']]

        if isinstance(pulses, dataclasses.I3RecoPulseSeriesMapMask) or \
           isinstance(pulses, dataclasses.I3RecoPulseSeriesMapUnion):
            pulses = pulses.apply(frame)

        # get DOM exclusions
        if self.data['dom_exclusions_exist']:
            try:
                dom_exclusions = \
                    frame[self.configuration.config['dom_exclusions_key']]
            except KeyError:
                msg = 'Could not find exclusion key {!r}'
                self._logger.warning(msg.format(
                    self.configuration.config['dom_exclusions_key']))
                dom_exclusions = None
        else:
            dom_exclusions = None

        # get time window exclusions
        if self.data['time_exclusions_exist']:
            try:
                time_exclusions = \
                    frame[self.configuration.config['time_exclusions_key']]
            except KeyError:
                msg = 'Could not find time window exclusion key {!r}'
                self._logger.warning(msg.format(
                    self.configuration.config['time_exclusions_key']))
                time_exclusions = None
        else:
            time_exclusions = None

        # number of events in batch (one frame at a time)
        size = 1

        # create empty array for DOM charges
        x_dom_charge = np.zeros([size, 86, 60, 1],
                                dtype=self.data['np_float_precision'])

        if self.data['dom_exclusions_exist']:
            bad_doms = np.reshape(detector.bad_doms_mask, [1, 86, 60, 1])
            x_dom_exclusions = (
                np.ones_like(x_dom_charge) * bad_doms).astype(bool)
        else:
            x_dom_exclusions = None

        if self.data['time_exclusions_exist']:
            x_time_exclusions = []
            x_time_exclusions_ids = []
        else:
            x_time_exclusions = None
            x_time_exclusions_ids = None

        # create array for time data
        x_time_window = np.empty(
            [size, 2],  dtype=self.data['np_float_precision'])
        x_time_window[:, 0] = float('inf')
        x_time_window[:, 1] = -float('inf')

        add_charge_quantiles = \
            self.configuration.config['add_charge_quantiles']

        x_pulses = []
        x_pulses_ids = []

        # ---------------------
        # get pulse information
        # ---------------------
        for omkey, pulse_list in pulses.items():

            string = omkey.string
            dom = omkey.om

            if dom > 60:
                self._logger.warning(
                    'skipping pulse: {} {}'.format(string, dom))
                continue

            for pulse in pulse_list:
                index = 0

                # pulse charge: row[12], time: row[10]
                # accumulate charge in DOMs
                x_dom_charge[index, string-1, dom-1, 0] += pulse.charge

                # gather pulses
                if add_charge_quantiles:

                    # (charge, time, quantile)
                    cum_charge = float(x_dom_charge[index, string-1, dom-1, 0])
                    x_pulses.append([pulse.charge, pulse.time, cum_charge])

                else:
                    # (charge, time)
                    x_pulses.append([pulse.charge, pulse.time])

                # gather pulse ids (batch index, string, dom)
                x_pulses_ids.append([index, string-1, dom-1])

                # update time window
                if pulse.time > x_time_window[index, 1]:
                    x_time_window[index, 1] = pulse.time
                if pulse.time < x_time_window[index, 0]:
                    x_time_window[index, 0] = pulse.time

        x_pulses = np.array(x_pulses, dtype=self.data['np_float_precision'])
        x_pulses_ids = np.array(x_pulses_ids, dtype=np.int32)

        # convert cumulative charge to fraction of total charge, e.g. quantile
        if add_charge_quantiles:

            # compute flat indices
            dim1 = x_dom_charge.shape[1]
            dim2 = x_dom_charge.shape[2]
            flat_indices = (x_pulses_ids[:, 0]*dim1*dim2 +  # event
                            x_pulses_ids[:, 1]*dim2 +  # string
                            x_pulses_ids[:, 2])  # DOM

            # flatten dom charges
            flat_charges = x_dom_charge.flatten()

            # calculate quantiles
            x_pulses[:, 2] /= flat_charges[flat_indices]

        # -------------------
        # get time exclusions
        # -------------------
        if time_exclusions is not None:
            for omkey, tw_list in time_exclusions.items():

                string = omkey.string
                dom = omkey.om

                if dom > 60:
                    self._logger.warning(
                        'skipping time window: {} {}'.format(string, dom))
                    continue

                for tw in tw_list:
                    index = 0

                    # (t_start, t_end)
                    x_time_exclusions.append([tw.start, tw.stop])

                    # gather pulse ids (batch index, string, dom)
                    x_time_exclusions_ids.append([index, string-1, dom-1])

            x_time_exclusions = np.array(
                x_time_exclusions, dtype=self.data['np_float_precision'])
            x_time_exclusions_ids = np.array(
                x_time_exclusions_ids, dtype=np.int32)

        # ------------------
        # get dom exclusions
        # ------------------
        if dom_exclusions is not None:
            for omkey in dom_exclusions:
                string = omkey.string
                dom = omkey.om

                if dom > 60:
                    msg = 'skipping exclusion DOM: {!r} {!r}'
                    self._logger.info(msg.format(string, dom))
                    continue

                index = 0
                x_dom_exclusions[index, string-1, dom-1, 0] = False

            if self.configuration.config['discard_pulses_from_excluded_doms']:

                # compute flat indices
                dim1 = x_dom_exclusions.shape[1]
                dim2 = x_dom_exclusions.shape[2]
                flat_indices = (x_pulses_ids[:, 0]*dim1*dim2 +  # event
                                x_pulses_ids[:, 1]*dim2 +  # string
                                x_pulses_ids[:, 2])  # DOM

                # flatten dom charges
                flat_exclusions = x_dom_exclusions.flatten()

                # discard pulses coming from excluded DOMs
                mask = flat_exclusions[flat_indices]
                x_pulses = x_pulses[mask]
                x_pulses_ids = x_pulses_ids[mask]

        # put everything together and make sure the order is correct
        data_dict = {
            'x_dom_charge': x_dom_charge,
            'x_dom_exclusions': x_dom_exclusions,
            'x_pulses': x_pulses,
            'x_pulses_ids': x_pulses_ids,
            'x_time_window': x_time_window,
            'x_time_exclusions': x_time_exclusions,
            'x_time_exclusions_ids': x_time_exclusions_ids,
        }
        event_batch = []
        for tensor in self.data['data_tensors'].list:
            event_batch.append(data_dict[tensor.name])

        return size, event_batch

    def create_data_from_frame(self, frame, *args, **kwargs):
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
            DataTensorList (self.data['data_tensors']).
        """
        if not self.is_configured:
            raise ValueError('Module not configured yet!')

        return self.get_data_from_frame(frame, *args, **kwargs)

    def write_data_to_frame(self, data, frame, *args, **kwargs):
        """Write data to I3Frame.

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

        pass
