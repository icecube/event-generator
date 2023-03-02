#!/usr/bin/env python
# -*- coding: utf-8 -*-
from icecube import dataclasses, icetray
import numpy as np

from . import misc
from .autoencoder import autoencoder


class CGeneratorData(icetray.I3ConditionalModule):
    '''Module to add CGenerator data to frame
    '''
    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("PulseKey",
                          "Name of pulse map to use.",
                          'InIcePulses')
        self.AddParameter("TimeBins",
                          "List of bin edges for time binning"
                          " (charge_bins data format)",
                          ['-inf', 'inf'])
        self.AddParameter("TimeQuantiles",
                          "List of quantiles for charge weighted time "
                          "quantiles. A quantile of zero will be defined as "
                          "the time of the first light."
                          "(charge_weighted_time_quantiles data format)",
                          [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99])
        self.AddParameter("RelativeTimeMethod",
                          "Time Method: 'cascade_vertex','time_range', None, "
                          "'first_light_at_dom'",
                          'first_light_at_dom')
        self.AddParameter("DataFormat",
                          "Format of data: 'charge_bins',"
                          "'charge_weighted_time_quantiles'",
                          'charge_bins')
        self.AddParameter("CascadeKey",
                          "Frame key for MC cascade",
                          'MCCascade')
        self.AddParameter("TimeRangePulseKey",
                          "If specified, these pulses will be used instead of "
                          "'PulseKey' to calculate the time range",
                          None)
        self.AddParameter("AutoencoderSettings",
                          "Settings: 'homogenous_2_2','inhomogenous_3_3', "
                          'homogenous_2_2')
        self.AddParameter("AutoencoderEncoderName",
                          "Name of encoder to use: e.g. wf_100, wf_1, ...",
                          'wf_100')

    def Configure(self):
        self._pulse_key = self.GetParameter("PulseKey")
        self._time_bins = self.GetParameter("TimeBins")
        self._time_quantiles = self.GetParameter("TimeQuantiles")
        self._relative_time_method = self.GetParameter("RelativeTimeMethod")
        self._data_format = self.GetParameter("DataFormat")
        self._cascade_key = self.GetParameter("CascadeKey")
        self._time_range_pulse_key = self.GetParameter("TimeRangePulseKey")
        self._autoencoder_settings = self.GetParameter("AutoencoderSettings")
        self._encoder_name = self.GetParameter("AutoencoderEncoderName")

        if self._time_range_pulse_key is None:
            self._time_range_pulse_key = self._pulse_key

        if self._data_format == 'charge_weighted_time_quantiles':
            # sanity check
            for q in self._time_quantiles:
                if q > 1. or q < 0.:
                    msg = 'Quantile should be within [0,1]:'
                    msg += ' {!r}'.format(q)
                    raise ValueError(msg)

        if self._autoencoder_settings == \
                'inhomogenous_2_2_calibrated_emd_wf_quantile':
            if self._relative_time_method != 'wf_quantile':
                raise ValueError('Timing method must be wf_quantile!')

        if self._data_format == 'autoencoder':
            self._autoencoder = autoencoder.get_autoencoder(
                                                    self._autoencoder_settings)

    def Geometry(self, frame):
        """Get a dictionary with DOM positions

        Parameters
        ----------
        frame : I3Frame
            Current i3 frame.
        """
        geo_map = frame['I3Geometry'].omgeo
        self._dom_pos_dict = {i[0]: i[1].position for i in geo_map
                              if i[1].omtype.name == 'IceCube'}
        self.PushFrame(frame)

    def Physics(self, frame):
        """Add CGenerator data keys to frame.

        Parameters
        ----------
        frame : I3Frame
            Current i3 frame.
        """
        self.add_dom_response(frame,
                              pulse_key=self._pulse_key,
                              relative_time_method=self._relative_time_method,
                              data_format=self._data_format,
                              bins=self._time_bins,
                              time_quantiles=self._time_quantiles,
                              cascade_key=self._cascade_key,
                              )
        self.PushFrame(frame)

    def add_dom_response(self, frame, pulse_key, relative_time_method,
                         data_format, bins=None, time_quantiles=None,
                         cascade_key='MCCascade'):
        """Add time binned DOM response.

        Adds the following keys to the frame:
            DOMPulseBinIndices : I3MapKeyVectorInt
                List of time bin indices.
            DOMPulseBinValues : I3MapKeyVectorDouble
                List of charges in the time bin.

        Parameters
        ----------
        frame : I3Frame
            Current i3 frame.
        pulse_key : str
            Pulses to use.
        relative_time_method : str
            Defines method to use to calculate the time offset for the time
            binning.
        data_format : str
            Defines the data format:
                charge_weighted_time_quantiles:
                    First bin is total charge at DOM.
                    The rest of the bins will be filled with the times at wich
                    q% of the charge was measured at the DOM, where q stands
                    for the quantiles defined in time_quantiles.
                    Quantile of zero is interpreted as time of first pulse.
                charge_bins:
                    Charges are binned in time.
                    The binning is defined with the bins parameter.
        bins : List of float, optional
            A list defining the bin edges if data_format is 'charge_bins'
        time_quantiles : None, optional
            List of quantiles for charge weighted time quantiles.
            A quantile of zero will be defined as the time of the first pulse.
            This is only needed when data format is
            'charge_weighted_time_quantiles'.
        cascade_key : str, optional
            Key of the I3Particle representing the cascade.
            Binning in time will be relative to the time of the cascade vertex.

        Raises
        ------
        ValueError
            Description
        """

        pulses = frame[pulse_key]

        if isinstance(pulses, dataclasses.I3RecoPulseSeriesMapMask) or \
           isinstance(pulses, dataclasses.I3RecoPulseSeriesMapUnion):
            pulses = pulses.apply(frame)

        bin_indices = dataclasses.I3MapKeyVectorInt()
        bin_values = dataclasses.I3MapKeyVectorDouble()

        if relative_time_method.lower() == 'cascade_vertex':
            time_offset = frame[cascade_key].time

        elif relative_time_method.lower() == 'time_range':

            # choose different pulse series to base time window on?
            if self._time_range_pulse_key != pulse_key:
                time_pulses = frame[self._time_range_pulse_key]

                if isinstance(time_pulses,
                              dataclasses.I3RecoPulseSeriesMapMask) or \
                   isinstance(time_pulses,
                              dataclasses.I3RecoPulseSeriesMapUnion):
                    time_pulses = time_pulses.apply(frame)
            else:
                time_pulses = pulses

            charges = []
            times = []
            for om_key, dom_pulses in time_pulses:
                for pulse in dom_pulses:
                    charges.append(pulse.charge)
                    times.append(pulse.time)
            charges = np.asarray(charges)
            times = np.asarray(times)
            time_offset = self.get_time_range(charges, times,
                                              time_window_size=6000)[0]

        elif relative_time_method is None:
            time_offset = 0.

        elif relative_time_method == 'first_light_at_dom':
            vertex_time = frame[cascade_key].time
            vertex_pos = frame[cascade_key].pos
            time_offset = vertex_time

        elif relative_time_method == 'wf_quantile':
            time_offset = 0.

        else:
            raise ValueError('Option is uknown: {!r}'.format(
                                                        relative_time_method))

        frame['DOMPulseTimeRangeStart'] = dataclasses.I3Double(time_offset)

        for om_key, dom_pulses in pulses:

            dom_charges = [pulse.charge for pulse in dom_pulses]
            dom_times = np.array([pulse.time for pulse in dom_pulses])

            # abort early if no pulses exist for DOM key
            if not dom_charges:
                continue

            if relative_time_method == 'first_light_at_dom':
                time_offset = self.get_time_of_first_light(
                                            dom_pos=self._dom_pos_dict[om_key],
                                            vertex_pos=vertex_pos,
                                            vertex_time=vertex_time,
                                            )

            elif relative_time_method == 'wf_quantile':
                time_offset = self.get_wf_quantile(
                                    times=dom_times,
                                    charges=dom_charges)

            # Calculate times relative to offset
            dom_times -= time_offset

            bin_values_list = []
            bin_indices_list = []

            # Charge Bins: Charges are binned in time
            if data_format.lower() in ['charge_bins', 'charge_bins_and_times']:

                # Add time information
                if data_format.lower() == 'charge_bins_and_times':
                    t_first_pulse = dom_times[0] + time_offset
                    t_wf_quantile = time_offset
                    bin_values_list.append(t_first_pulse)
                    bin_indices_list.append(0)
                    bin_values_list.append(t_wf_quantile)
                    bin_indices_list.append(1)
                    index_offset = 2
                else:
                    index_offset = 0

                hist, bin_edges = np.histogram(dom_times,
                                               weights=dom_charges, bins=bins)
                for i, charge in enumerate(hist):
                    if charge != 0:
                        bin_values_list.append(charge)
                        bin_indices_list.append(i + index_offset)

            elif data_format.lower() == 'autoencoder':

                bin_values_list, bin_indices_list = \
                    autoencoder.get_encoded_data(
                        self._autoencoder,
                        self._encoder_name,
                        dom_times=dom_times,
                        dom_charges=dom_charges,
                        bins=bins,
                        autoencoder_settings=self._autoencoder_settings,
                        time_offset=time_offset)

            # First bin is total charge at DOM
            # Quantile of zero is time of first pulse
            elif data_format.lower() == 'charge_weighted_time_quantiles':

                dom_times = np.array(dom_times)

                # add total charge at DOM
                total_dom_charge = np.sum(dom_charges)
                bin_values_list.append(total_dom_charge)
                bin_indices_list.append(0)

                # compute cumulative sum
                cum_charge = np.cumsum(dom_charges) / total_dom_charge

                # small constant:
                epsilon = 1e-6

                # add time quantiles
                for i, q in enumerate(time_quantiles):

                    # get time of pulse at which the cumulative charge is
                    # first >= q:
                    mask = cum_charge >= q - epsilon
                    q_value = dom_times[mask][0]

                    bin_values_list.append(q_value)
                    bin_indices_list.append(i+1)

            else:
                raise ValueError('Data format is not understood: {!r}'.format(
                                                                data_format))
            bin_values[om_key] = bin_values_list
            bin_indices[om_key] = bin_indices_list

        frame['DOMPulseBinIndices'] = bin_indices
        frame['DOMPulseBinValues'] = bin_values

    def get_wf_quantile(self, times, charges, quantile=0.1):
        """Get charge weighted time quantile of pulses.

        Parameters
        ----------
        times : list or np.ndarray
            Time of pulses.
            Assumed that the times are already SORTED!
            This is usually the case for I3 PulseSeries.
        charges : list or np.ndarray
            Charges of pulses.
        quantile : float, optional
            Cumulative charge quantile.

        Returns
        -------
        float
            The time will be returned at which a fraction of the total charge
            equal to 'quantile' is collected.
        """

        assert 0 <= quantile and quantile <= 1., \
            '{!r} not in [0,1]'.format(quantile)

        # calculate total charge
        total_charge = np.sum(charges)

        # compute cumulative sum
        cum_charge = np.cumsum(charges) / total_charge

        # small constant:
        epsilon = 1e-6

        # get time of pulse at which the cumulative charge is first >= q:
        mask = cum_charge >= quantile - epsilon
        return times[mask][0]

    def get_time_of_first_light(self, dom_pos, vertex_pos, vertex_time):
        """Get the time when unscattered light, emitted at the vertex position
        and time, arrives at the dom position.

        Parameters
        ----------
        dom_pos : I3Position
            I3Position of DOM
        vertex_pos : I3Position
            I3Position of cascade vertex.
        vertex_time : float
            Time of cascade vertex.

        Returns
        -------
        float
            Time when first unscattered light from cascade arrives at DOM
        """
        distance = (dom_pos - vertex_pos).magnitude
        photon_time = distance / dataclasses.I3Constants.c_ice
        return vertex_time + photon_time

    def get_time_range(self, charges, times, time_window_size=6000):
            """Get time range in which the most charge is detected.

            A time window of size 'time_window_size' is shifted across the
            event in 500 ns time steps. The time range, at which the maximum
            charge is enclosed within the time window, is returned

            Parameters
            ----------
            charges : list of float
                List of measured pulse charges.
            times : list of float
                List of measured pulse times
            time_window_size : int, optional
                Defines the size of the time window.

            Returns
            -------
            (float, float)
                Time rane in which the maximum charge is detected.
            """
            max_charge_sum = 0
            start_t = 9000
            for t in range(int(np.nanmin(times)//1000)*1000,
                           int(np.nanmax(times)//1000)*1000 - time_window_size,
                           500):
                indices_smaller = t < times
                indices_bigger = times < t + time_window_size
                indices = np.logical_and(indices_smaller, indices_bigger)
                charge_sum = np.sum(charges[indices])
                if charge_sum > max_charge_sum:
                    max_charge_sum = charge_sum
                    start_t = t
            return [start_t, start_t + time_window_size]
