from __future__ import division, print_function
import logging
import tensorflow as tf
import numpy as np

from egenerator.model.multi_source.base import MultiSource


class StartingVariableMultiCascadeModel(MultiSource):

    def __init__(self, logger=None):
        """Instantiate Source class

        Parameters
        ----------
        logger : logging.logger, optional
            The logger to use.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(StartingVariableMultiCascadeModel, self).__init__(
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
        self._untracked_data['num_cascades'] = config['num_cascades']

        if list(base_sources.keys()) != ['cascade']:
            msg = "Expected only 'cascade' base, but got {!r}"
            raise ValueError(msg.format(base_sources.keys()))

        sources = {}
        parameters = ['x', 'y', 'z', 'zenith', 'azimuth', 'energy', 'time']
        for index in range(1, self._untracked_data['num_cascades']):
            cascade_name = 'cascade_{:05d}'.format(index)
            parameters.append(cascade_name + '_energy')
            parameters.append(cascade_name + '_distance')
            sources[cascade_name] = 'cascade'

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

        # calculate direction vector
        dir_x = -tf.sin(zenith) * tf.cos(azimuth)
        dir_y = -tf.sin(zenith) * tf.sin(azimuth)
        dir_z = -tf.cos(zenith)

        source_parameter_dict = {}
        for cascade in self._untracked_data['sources'].keys():

            dist = parameters.params[cascade + '_distance']
            cascade_energy = parameters.params[cascade + '_energy']

            # calculate position and time of cascade
            cascade_x = x + dist * dir_x
            cascade_y = y + dist * dir_y
            cascade_z = z + dist * dir_z
            cascade_time = time + dist / c

            source_parameter_dict[cascade] = tf.stack([
                cascade_x, cascade_y, cascade_z,
                zenith, azimuth, cascade_energy, cascade_time],
                axis=1)

        return source_parameter_dict
