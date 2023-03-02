from icecube import dataclasses

def create_hdf5_time_window_series_map_hack(frame, tws_map_name, output_key):
    """Hack from Javier Vara to write I3TimeWindowSeriesMap to frame
    Create an I3RecoPulseSeriesMap with information from the I3TimeWindowSeriesMap.
    A tableio converter already exists for the I3RecoPulseSeriesMap, but not for the 
    I3TimeWindowSeriesMap. So we will mis-use the I3RecoPulseSeriesMap one.

    Parameters
    -----------
    frame : I3Frame
        The frame to which to write the combined exclusions.
    tws_map_name : str
        The name of the I3TimeWindowSeriesMap
    output_key : str 
        The key to which the created I3RecoPulseSeriesMap will be written.
    """
#    print(frame.keys())
    if tws_map_name in frame.keys():
        tws_map = frame[tws_map_name]
        pulse_series_map = dataclasses.I3RecoPulseSeriesMap()
        for om_key, time_window_series in tws_map:
            # create an I3RecoPulseSeries to which we will save the time windows
            string = om_key.string
            # if string > 1000:  Gen1 strings
            reco_pulse_series = dataclasses.I3RecoPulseSeries()

            for time_window in time_window_series:
                    # We will mis-use the pulse-time field for the start time
                    # and the pulse-width field for the end time of the I3TimeWindow
                    pulse = dataclasses.I3RecoPulse()
                    pulse.time = time_window.start
                    pulse.width = time_window.stop
                    reco_pulse_series.append(pulse)

            pulse_series_map[om_key] = reco_pulse_series

        frame[output_key] = pulse_series_map

    else:
        frame[output_key] = dataclasses.I3RecoPulseSeriesMap()
