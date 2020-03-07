from __future__ import division, print_function

import logging
import numpy as np
import pandas as pd

from egenerator.data.modules.base import BaseModule
from egenerator.data.tensor import DataTensorList, DataTensor


class CascadeGeneratorLabelModule(BaseModule):

    """This is a label module that loads the cascade labels which are also used
    for the cascade-generator project.
    """

    def _initialize(self, shift_cascade_vertex, trafo_log,
                    label_key='LabelsDeepLearning', *args, **kwargs):
        """Initialize Module class.
        This is an abstract method and must be implemented by derived class.

        If there are skip_check_keys, e.g. config keys that do not have to
        match, they must be defined here.
        Any settings used within the module must be saved to 'self.settings'.

        Parameters
        ----------
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
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        self.logger = logging.getLogger(__name__)
        self._settings['shift_cascade_vertex'] = shift_cascade_vertex
        self._settings['trafo_log'] = trafo_log
        self._settings['label_key'] = label_key

        # sanity checks:
        if not isinstance(self.settings['shift_cascade_vertex'], bool):
            raise ValueError('{!r} is not a boolean value!'.format(
                self.settings['shift_cascade_vertex']))

    def _configure(self, config_data):
        """Configure Module Class
        This is an abstract method and must be implemented by derived class.

        Parameters
        ----------
        config_data : None, str, or DataTensorList
            This is either the path to a test file or a data tensor list
            object. The module will be configured with this.

        Returns
        -------
        DataTensorList
            The tensors of type 'label' that will be loaded.
        """

        self._tensors = DataTensorList([DataTensor(
                                        name='cascade_labels',
                                        shape=[None, 7],
                                        tensor_type='label',
                                        dtype='float32',
                                        trafo=True,
                                        trafo_log=self.settings['trafo_log'])])

        if isinstance(config_data, DataTensorList):
            if not config_data == self.tensors:
                raise ValueError('{!r} != {!r}'.format(config_data,
                                                       self.tensors))
        return self.tensors

    def get_data_from_hdf(self, file):
        """Get label data from hdf file.

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
            Returns None if no label data is loaded.
        """
        if not self.is_configured:
            raise ValueError('Module not configured yet!')

        # open file
        f = pd.HDFStore(file, 'r')

        cascade_parameters = []
        try:
            _labels = f[self.settings['label_key']]
            for l in ['cascade_x', 'cascade_y', 'cascade_z', 'cascade_zenith',
                      'cascade_azimuth', 'cascade_energy', 'cascade_t']:
                cascade_parameters.append(_labels[l])

        except Exception as e:
            self.logger.warning(e)
            self.logger.warning('Skipping file: {}'.format(file))
            return None, None
        finally:
            f.close()

        # shift cascade vertex to shower maximum?
        if self.settings['shift_cascade_vertex']:
            x, y, z = self._shift_to_maximum(*cascade_parameters[:6])
            cascade_parameters[0] = x
            cascade_parameters[1] = y
            cascade_parameters[2] = z

        # format cascade parameters
        cascade_parameters = np.array(cascade_parameters).T
        num_events = len(cascade_parameters)

        return num_events, (cascade_parameters,)

    def _shift_to_maximum(self, x, y, z, zenith, azimuth, ref_energy,
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
        eps : float, optional
            Small constant float.

        Returns
        -------
        (float, float, float) or (np.ndarray, np.ndarray, np.ndarray)
            Shifted vertex poisition (position of shower maximum) in meter.
        """
        dir_x = -np.sin(zenith) * np.cos(azimuth)
        dir_y = -np.sin(zenith) * np.sin(azimuth)
        dir_z = -np.cos(zenith)

        a = 2.03 + 0.604 * np.log(ref_energy + eps)
        b = 0.633
        l_rad = (0.358/0.9216)  # in meter
        length_to_maximum = np.clip(((a-1.)/b)*l_rad, 0., float('inf'))
        x_shifted = x + dir_x * length_to_maximum
        y_shifted = y + dir_y * length_to_maximum
        z_shifted = z + dir_z * length_to_maximum
        return x_shifted, y_shifted, z_shifted
