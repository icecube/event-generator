from __future__ import division, print_function

import logging
import numpy as np
import pandas as pd

from egenerator import misc
from egenerator.manager.component import BaseComponent, Configuration
from egenerator.data.tensor import DataTensorList, DataTensor


class CascadeGeneratorLabelModule(BaseComponent):

    """This is a label module that loads the cascade labels which are also used
    for the cascade-generator project.
    """

    def __init__(self, logger=None):
        """Initialize cascade module

        Parameters
        ----------
        logger : logging.logger, optional
            A logging instance.
        """

        logger = logger or logging.getLogger(__name__)
        super(CascadeGeneratorLabelModule, self).__init__(logger=logger)

    def _configure(self, config_data, shift_cascade_vertex, trafo_log,
                   float_precision, label_key='LabelsDeepLearning'):
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
        label_key : str, optional
            The name of the key under which the labels are saved.
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
                'label_tensors': DataTensorList
                    The tensors of type 'label' that will be loaded.
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

        # sanity checks:
        if not isinstance(shift_cascade_vertex, bool):
            raise TypeError('{!r} is not a boolean value!'.format(
                shift_cascade_vertex))

        data = {}
        data['label_tensors'] = DataTensorList([DataTensor(
                                        name='x_parameters',
                                        shape=[None, 7],
                                        tensor_type='label',
                                        dtype=float_precision,
                                        trafo=True,
                                        trafo_log=trafo_log)])

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
                          label_key=label_key))
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
        """
        if not self.is_configured:
            raise ValueError('Module not configured yet!')

        # open file
        f = pd.HDFStore(file, 'r')

        cascade_parameters = []
        try:
            _labels = f[self.configuration.config['label_key']]
            for l in ['cascade_x', 'cascade_y', 'cascade_z', 'cascade_zenith',
                      'cascade_azimuth', 'cascade_energy', 'cascade_t']:
                cascade_parameters.append(_labels[l])

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
            for l in ['cascade_x', 'cascade_y', 'cascade_z', 'cascade_zenith',
                      'cascade_azimuth', 'cascade_energy', 'cascade_t']:
                cascade_parameters.append(np.atleast_1d(_labels[l]))

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
