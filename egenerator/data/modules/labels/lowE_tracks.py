from __future__ import division, print_function

import logging
import numpy as np
import pandas as pd

from egenerator import misc
from egenerator.manager.component import BaseComponent, Configuration
from egenerator.data.tensor import DataTensorList, DataTensor
from egenerator.utils.cascades import shift_to_maximum


class LowETrackGeneratorLabelModule(BaseComponent):

    """This is a label module that loads the low energy track labels.
    """

    def __init__(self, logger=None):
        """Initialize track module

        Parameters
        ----------
        logger : logging.logger, optional
            A logging instance.
        """

        logger = logger or logging.getLogger(__name__)
        super(LowETrackGeneratorLabelModule, self).__init__(logger=logger)

    def _configure(self, config_data, shift_cascade_vertex, trafo_log,
                   float_precision,
                   num_cascades=1,
                   label_key='LabelsDeepLearning',
                  ):
        """Configure Module Class
        This is an abstract method and must be implemented by derived class.

        Parameters
        ----------
        config_data : None, str, or DataTensorList
            This is either the path to a test file or a data tensor list
            object. The module will be configured with this.
        shift_cascade_vertex : bool
            Shift cascade vertex to shower maximum instead of interaction
            point.
        trafo_log : None or bool or list of bool
            Whether or not to apply logarithm on parameters.
            If a single bool is given, this applies to all labels. Otherwise
            a list of bools corresponds to the labels in the order:
                zenith, azimuth,
                track_anchor_x, track_anchor_y, track_anchor_z,
                track_anchor_time, track_energy,
                track_distance_start, track_distance_end,
                track_stochasticity,
                cascade_0000_energy,
                cascade_{i:04d}_energy, cascade_{i:04d}_distance,
            Snowstorm parameters must not be defined here. No logarithm will be
            applied to the snowstorm parameters.
        float_precision : str
            The float precision as a str.
        num_cascades : int, optional
            Number of cascades along the track.
        label_key : str, optional
            The name of the key under which the labels are saved.

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
                'label_tensors': DataTensorList
                    The tensors of type 'label' that will be loaded.
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

        # sanity checks:
        if not isinstance(shift_cascade_vertex, bool):
            raise TypeError('{!r} is not a boolean value!'.format(
                shift_cascade_vertex))
        if num_cascades <= 0:
            raise ValueError('Num cascades {} must be >0!'.format(
                num_cascades))

        # compute number of parameters
        if num_cascades == 1:
            num_params = 8
        else:
            raise ValueError('Only 1 cascade supported at the moment')
            #num_params = 8 + (num_cascades - 1) * 2

        # create list of parameter names which is needed for data loading
        parameter_names = [
            'PrimaryPositionX', 'PrimaryPositionY', 'PrimaryPositionZ',
            'PrimaryZenith', 'PrimaryAzimuth',
            'CascadeEnergy', 'Time', 'TrackEnergy',
        ]# 'TrackLength',
        #if num_cascades > 1:
        #    for i in range(1, num_cascades):
        #        parameter_names.append('cascade_{:04d}_energy'.format(i))
        #        parameter_names.append('cascade_{:04d}_distance'.format(i))

        parameter_dict = {}
        for i, parameter_name in enumerate(parameter_names):
            parameter_dict[parameter_name] = i

        # extend trafo log for snowstorm parameters: fill with Flase
        if isinstance(trafo_log, bool):
            trafo_log_ext = [trafo_log] * num_params
        else:
            trafo_log_ext = list(trafo_log)

        data = {
            'parameter_dict': parameter_dict,
            'parameter_names': parameter_names,
        }
        data['label_tensors'] = DataTensorList([DataTensor(
            name='x_parameters',
            shape=[None, num_params],
            tensor_type='label',
            dtype=float_precision,
            trafo=True,
            trafo_log=trafo_log_ext)])

        if isinstance(config_data, DataTensorList):
            if config_data != data['label_tensors']:
                msg = 'Tensors are wrong: {!r} != {!r}'
                raise ValueError(msg.format(config_data,
                                            data['label_tensors']))
        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings=dict(config_data=config_data,
                          shift_cascade_vertex=shift_cascade_vertex,
                          trafo_log=trafo_log,
                          float_precision=float_precision,
                          num_cascades=num_cascades,
                          label_key=label_key,
                          )
                          )
        return configuration, data, {}

    def get_data_from_hdf(self, file, *args, **kwargs):
        """Get label data from hdf file.

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
        tuple of array-like tensors or None
            The input data (array-like) as specified in the
            DataTensorList (self.tensors).
            Returns None if no label data is loaded.

        Raises
        ------
        ValueError
            Description
        """
        if not self.is_configured:
            raise ValueError('Module not configured yet!')

        # open file
        f = pd.HDFStore(file, 'r')

        track_parameters = []
        try:
            _labels = f[self.configuration.config['label_key']]
            for l in self.data['parameter_names']:
                    track_parameters.append(_labels[l])

            num_events = len(track_parameters[0])

        except Exception as e:
            self._logger.warning(e)
            self._logger.warning('Skipping file: {}'.format(file))
            return None, None
        finally:
            f.close()

        # shift cascade vertices to shower maximum?
        if self.configuration.config['shift_cascade_vertex']:
            track_parameters = self._shift_parameters(track_parameters)

        # format track parameters
        dtype = getattr(np, self.configuration.config['float_precision'])
        track_parameters = np.array(track_parameters, dtype=dtype).T
        num_events = len(track_parameters)

        return num_events, (track_parameters,)

    def get_data_from_frame(self, frame, *args, **kwargs):
        """Get label data from frame.

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
        int
            Number of events.
        tuple of array-like tensors or None
            The input data (array-like) as specified in the
            DataTensorList (self.tensors).
            Returns None if no label data is loaded.
        """
        if not self.is_configured:
            raise ValueError('Module not configured yet!')

        track_parameters = []
        try:
            _labels = frame[self.configuration.config['label_key']]
            for l in self.data['parameter_names']:
                track_parameters.append(np.atleast_1d(_labels[l]))

            num_events = len(track_parameters[0])

        except Exception as e:
            self._logger.warning(e)
            self._logger.warning('Skipping frame: {}'.format(frame))
            return None, None

        # shift cascade vertices to shower maximum?
        if self.configuration.config['shift_cascade_vertex']:
            track_parameters = self._shift_parameters(track_parameters)

        # format track parameters
        dtype = getattr(np, self.configuration.config['float_precision'])
        track_parameters = np.array(track_parameters, dtype=dtype).T
        num_events = len(track_parameters)

        return num_events, (track_parameters,)

    def create_data_from_frame(self, frame, *args, **kwargs):
        """Create label data from frame.

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
        int
            Number of events.
        tuple of array-like tensors or None
            The input data (array-like) as specified in the
            DataTensorList (self.tensors).
            Returns None if no label data is created.
        """
        if not self.is_configured:
            raise ValueError('Module not configured yet!')

        return self.get_data_from_frame(frame, *args, **kwargs)

    def write_data_to_frame(self, data, frame, *args, **kwargs):
        """Write label data to I3Frame.

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

    def _shift_parameters(self, parameters):
        """Adjust parameters due to shifting of cascades to shower maximum.

        Parameters
        ----------
        parameters : list of array
            The parameters that should be shifted.

        Returns
        -------
        list of array
            The shifted parameters
        """
        num_cascades = self.configuration.config['num_cascades']
        param_dict = self.data['parameter_dict']

        zenith = parameters[param_dict['PrimaryZenith']]
        azimuth = parameters[param_dict['PrimaryAzimuth']]

        c = 0.299792458  # meter / ns
        dir_x = -np.sin(zenith) * np.cos(azimuth)
        dir_y = -np.sin(zenith) * np.sin(azimuth)
        dir_z = -np.cos(zenith)

        # fix anchor point of track which is the first provided cascade
        # This means that the start and end distance of the track segment
        # must also be adjusted
        if num_cascades > 1:
            shift = self._get_cascade_extension(
                parameters[param_dict['cascade_0001_energy']])

            parameters[param_dict['PrimaryPositionX']] += dir_x * shift
            parameters[param_dict['PrimaryPositionY']] += dir_y * shift
            parameters[param_dict['PrimaryPositionZ']] += dir_z * shift
            parameters[param_dict['Time']] += shift / c

            #parameters[param_dict['track_distance_start']] -= shift
            #parameters[param_dict['track_distance_end']] -= shift

            # Also shift all of the remaining cascades
            for i in range(2, num_cascades):
                shift_i = self._get_cascade_extension(
                    parameters[param_dict['cascade_{:04d}_energy'.format(i)]])

                # get index of cascade distance parameter
                dist_index = param_dict['cascade_{:04d}_distance'.format(i)]

                # we need to compensate for the shift of the anchor point
                parameters[dist_index] -= shift

                # and also for the shift of the ith cascade itself
                parameters[dist_index] += shift_i

        return parameters

    def _get_cascade_extension(self, ref_energy, eps=1e-6):
        """
        PPC does its own cascade extension, leaving the showers at the
        production vertex. Reapply the parametrization to find the
        position of the shower maximum, which is also the best approximate
        position for a point cascade.

        Parameters
        ----------
        ref_energy : array_like
            Energy of cascade in GeV.
        eps : float, optional
            Small constant float.

        Returns
        -------
        array_like
            Distance of shower maximum to cascade vertex in meter.
        """

        # Radiation length in meters, assuming an ice density of 0.9216 g/cm^3
        l_rad = (0.358/0.9216)  # in meter

        """
        Parameters taken from I3SimConstants (for particle e-):
        https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/
        meta-projects/combo/trunk/sim-services/private/
        sim-services/I3SimConstants.cxx
        """
        a = 2.01849 + 0.63176 * np.log(ref_energy + eps)
        b = l_rad/0.63207

        # Mode of the gamma distribution gamma_dist(a, b) is: (a-1.)/b
        length_to_maximum = np.clip(((a-1.)/b)*l_rad, 0., float('inf'))
        return length_to_maximum

    def _shift_to_maximum(self, x, y, z, zenith, azimuth, ref_energy, t,
                          eps=1e-6):
        """
        PPC does its own cascade extension, leaving the showers at the
        production vertex. Reapply the parametrization to find the
        position of the shower maximum, which is also the best approximate
        position for a point cascade.

        Parameters
        ----------
        x : float or np.ndarray of floats
            Cascade interaction vertex x (unshifted) in meters.
        y : float or np.ndarray of floats
            Cascade interaction vertex y (unshifted) in meters.
        z : float or np.ndarray of floats
            Cascade interaction vertex z (unshifted) in meters.
        zenith : float or np.ndarray of floats
            Cascade zenith direction in rad.
        azimuth : float or np.ndarray of floats
            Cascade azimuth direction in rad.
        ref_energy : float or np.ndarray of floats
            Energy of cascade in GeV.
        t : float or np.ndarray of floats
            Cascade interaction vertex time (unshifted) in ns.
        eps : float, optional
            Small constant float.

        Returns
        -------
        Tuple of float or tuple of np.ndarray
            Shifted vertex position (position of shower maximum) in meter and
            shifted vertex time in nano seconds.
        """

        return shift_to_maximum(
            x=x, y=y, z=z, zenith=zenith, azimuth=azimuth,
            ref_energy=ref_energy, t=t, eps=eps, reverse=False,
        )
