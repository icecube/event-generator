from __future__ import division, print_function
import logging
import tensorflow as tf
import numpy as np

from egenerator.model.multi_source.base import MultiSource


class DoubleCascadeModel(MultiSource):

    def __init__(self, logger=None):
        """Instantiate Source class

        Parameters
        ----------
        logger : logging.logger, optional
            The logger to use.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(DoubleCascadeModel, self).__init__(
            logger=self._logger)

    def get_parameters_and_mapping(self, config, base_sources):
        """Get parameter names and their ordering as well as source mapping.

        This is a pure virtual method that must be implemented by
        derived class.

        Parameters
        ----------
        config : dict
            A dictionary of settings.
        base_sources : dict of Source objects
            A dictionary of sources. These sources are used as a basis for
            the MultiSource object. The event hypothesis can be made up of
            multiple sources which may be created from one or more
            base source objects.

        Returns
        -------
        list of str
            A list of parameter names of the MultiSource object.
        dict
            This describes the sources which compose the event hypothesis.
            The dictionary is a mapping from source_name (str) to
            base_source (str). This mapping allows the reuse of a single
            source component instance. For instance, a muon can be build up of
            multiple cascades. However, all cascades should use the same
            underlying model. Hence, in this case only one base_source is
            required: the cascade source. The mapping will then map all
            cascades in the hypothesis to this one base cascade source.
        """

        if list(base_sources.keys()) != ['cascade']:
            msg = "Expected only 'cascade' base, but got {!r}"
            raise ValueError(msg.format(base_sources.keys()))

        sources = {'first_cascade': 'cascade',
                   'second_cascade': 'cascade'}
        parameters = ['x', 'y', 'z', 'zenith', 'azimuth', 'energy',
                      'time', 'energy2', 'distance']

        return parameters, sources

    def get_source_parameters(self, parameters):
        """Get the input parameters for the individual sources.

        Parameters
        ----------
        parameters : tf.Tensor
            The input parameters for the MultiSource object.
            The input parameters of the individual Source objects are composed
            from these.

        Returns
        -------
        dict of tf.Tensor
            Returns a dictionary of (name: input_parameters) pairs, where
            name is the name of the Source and input_parameters is a tf.Tensor
            for the input parameters of this Source.

        """
        c = 0.299792458  # meter / ns

        x = parameters.params['x']
        y = parameters.params['y']
        z = parameters.params['z']
        zenith = parameters.params['zenith']
        azimuth = parameters.params['azimuth']
        time = parameters.params['time']
        energy1 = parameters.params['energy']
        energy2 = parameters.params['energy2']
        distance = parameters.params['distance']

        # calculate direction vector
        dir_x = -tf.sin(zenith) * tf.cos(azimuth)
        dir_y = -tf.sin(zenith) * tf.sin(azimuth)
        dir_z = -tf.cos(zenith)

        source_parameter_dict = {}
        for cascade in sorted(self._untracked_data['sources'].keys()):

            if cascade == 'first_cascade':
                source_parameter_dict[cascade] = tf.stack(
                    [x, y, z, zenith, azimuth, energy1, time],
                    axis=1,
                )

            elif cascade == 'second_cascade':
                # calculate position and time of cascade
                x2 = x + distance * dir_x
                y2 = y + distance * dir_y
                z2 = z + distance * dir_z
                time2 = time + distance / c

                source_parameter_dict[cascade] = tf.stack(
                    [x2, y2, z2, zenith, azimuth, energy2, time2],
                    axis=1,
                )
                
            else:
                msg = "Expected only 'first_cascade' and 'second_cascade', but got {!r}"
                raise NameError(msg.format(cascade))

        return source_parameter_dict
