from __future__ import division, print_function

import logging
import numpy as np
import pandas as pd

from egenerator import misc
from egenerator.manager.component import BaseComponent, Configuration
from egenerator.data.tensor import DataTensorList, DataTensor


class SnowstormCascadeGeneratorLabelModule(BaseComponent):

    """This is a label module that loads the snowstorm cascade labels.
    """

    def __init__(self, logger=None):
        """Initialize cascade module

        Parameters
        ----------
        logger : logging.logger, optional
            A logging instance.
        """

        logger = logger or logging.getLogger(__name__)
        super(SnowstormCascadeGeneratorLabelModule, self).__init__(
                                                                logger=logger)

    def _configure(self, config_data, shift_cascade_vertex, trafo_log,
                   float_precision, label_key='LabelsDeepLearning',
                   additional_labels=None,
                   snowstorm_key='SnowstormParameters',
                   num_snowstorm_params=30):
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
            Whether or not to apply logarithm on cascade parameters.
            If a single bool is given, this applies to all labels. Otherwise
            a list of bools corresponds to the labels in the order:
                x, y, z, zenith, azimuth, energy, time
            Snowstorm parameters must not be defined here. No logarithm will be
            applied to the snowstorm parameters.
        float_precision : str
            The float precision as a str.
        label_key : str, optional
            The name of the key under which the labels are saved.
        additional_labels : list, optional
            A list of additional labels to load from the label_key.
            These labels are added before the snowstorm keys if provided.
        snowstorm_key : str, optional
            The name of the key under which the snowstorm parameters are saved.
            If `snowstorm_key` is None, no snowstorm parameters will be loaded.
            Instead a default value of 1. will be assigned to each of the
            `num_snowstorm_params` number of snowstorm parameters.
        num_snowstorm_params : int, optional
            The number of varied snowstorm parameters.

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

        # extend trafo log for snowstorm parameters: fill with Flase
        if isinstance(trafo_log, bool):
            trafo_log_ext = [trafo_log] * 7
        else:
            trafo_log_ext = list(trafo_log)
        trafo_log_ext.extend([False]*num_snowstorm_params)

        # create a list of label names to load
        if additional_labels is None:
            additional_labels = []
        label_names = [
            'cascade_x', 'cascade_y', 'cascade_z', 'cascade_zenith',
            'cascade_azimuth', 'cascade_energy', 'cascade_t',
        ] + additional_labels

        data = {'label_names': label_names}
        data['label_tensors'] = DataTensorList([DataTensor(
            name='x_parameters',
            shape=[None, len(label_names) + num_snowstorm_params],
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
                          label_key=label_key,
                          additional_labels=additional_labels,
                          snowstorm_key=snowstorm_key,
                          num_snowstorm_params=num_snowstorm_params))
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

        cascade_parameters = []
        try:
            _labels = f[self.configuration.config['label_key']]
            for l in self.data['label_names']:
                cascade_parameters.append(_labels[l])

            snowstorm_key = self.configuration.config['snowstorm_key']
            num_params = self.configuration.config['num_snowstorm_params']
            num_events = len(cascade_parameters[0])

            if num_params > 0:
                if snowstorm_key is not None:
                    _snowstorm_params = f[snowstorm_key]
                    params = _snowstorm_params['item']
                    index = _snowstorm_params['vector_index']
                    assert max(index) == num_params - 1
                    assert min(index) == 0

                    for i in range(num_params):

                        snowstorm_param = params[index == i]
                        assert len(snowstorm_param) == num_events
                        cascade_parameters.append(snowstorm_param)

                else:
                    # No Snowstorm key is provided: add dummy values
                    for i in range(num_params):
                        cascade_parameters.append(np.ones(num_events))

        except Exception as e:
            self._logger.warning(e)
            self._logger.warning('Skipping file: {}'.format(file))
            return None, None
        finally:
            f.close()

        # shift cascade vertex to shower maximum?
        if self.configuration.config['shift_cascade_vertex']:
            x, y, z, t = self._shift_to_maximum(*cascade_parameters[:7])
            cascade_parameters[0] = x
            cascade_parameters[1] = y
            cascade_parameters[2] = z
            cascade_parameters[6] = t

        # format cascade parameters
        dtype = getattr(np, self.configuration.config['float_precision'])
        cascade_parameters = np.array(cascade_parameters,
                                      dtype=dtype).T
        num_events = len(cascade_parameters)

        return num_events, (cascade_parameters,)

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

        cascade_parameters = []
        try:
            _labels = frame[self.configuration.config['label_key']]
            for l in self.data['label_names']:
                cascade_parameters.append(np.atleast_1d(_labels[l]))

            snowstorm_key = self.configuration.config['snowstorm_key']
            num_params = self.configuration.config['num_snowstorm_params']
            num_events = len(cascade_parameters[0])

            if num_params > 0:
                if snowstorm_key is not None:
                    _snowstorm_params = frame[snowstorm_key]
                    assert len(_snowstorm_params) == num_params

                    for i in range(num_params):

                        snowstorm_param = np.atleast_1d(_snowstorm_params[i])
                        assert len(snowstorm_param) == num_events
                        cascade_parameters.append(snowstorm_param)

                else:
                    # No Snowstorm key is provided: add dummy values
                    for i in range(num_params):
                        cascade_parameters.append(np.ones(num_events))

        except Exception as e:
            self._logger.warning(e)
            self._logger.warning('Skipping frame: {}'.format(frame))
            return None, None

        # shift cascade vertex to shower maximum?
        if self.configuration.config['shift_cascade_vertex']:
            x, y, z, t = self._shift_to_maximum(*cascade_parameters[:7])
            cascade_parameters[0] = x
            cascade_parameters[1] = y
            cascade_parameters[2] = z
            cascade_parameters[6] = t

        # format cascade parameters
        dtype = getattr(np, self.configuration.config['float_precision'])
        cascade_parameters = np.array(cascade_parameters,
                                      dtype=dtype).T
        num_events = len(cascade_parameters)

        return num_events, (cascade_parameters,)

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

        # Radiation length in meters, assuming an ice density of 0.9216 g/cm^3
        l_rad = (0.358/0.9216)  # in meter

        c = 0.299792458  # meter / ns
        dir_x = -np.sin(zenith) * np.cos(azimuth)
        dir_y = -np.sin(zenith) * np.sin(azimuth)
        dir_z = -np.cos(zenith)

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
        x_shifted = x + dir_x * length_to_maximum
        y_shifted = y + dir_y * length_to_maximum
        z_shifted = z + dir_z * length_to_maximum
        t_shifted = t + length_to_maximum / c
        return x_shifted, y_shifted, z_shifted, t_shifted
