from __future__ import division, print_function
import logging
import tensorflow as tf
import numpy as np

from egenerator.model.multi_source.base import MultiSource


class StartingTrackWithCascadeModel(MultiSource):

    def __init__(self, logger=None):
        """Instantiate Source class

        Parameters
        ----------
        logger : logging.logger, optional
            The logger to use.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(StartingTrackWithCascadeModel, self).__init__(logger=self._logger)

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
        if sorted(list(base_sources.keys())) != ['cascade', 'track']:
            msg = "Expected only ['cascade', 'track'] bases, but got {!r}"
            raise ValueError(msg.format(base_sources.keys()))

        # gather parameter names and sources
        sources = {'track': 'track',
                   'cascade': 'cascade'}
        parameter_names = ['x', 'y', 'z', 'time', 'energy',
                           'zenith', 'azimuth', 'length']

        # add snowstorm parameters
        num_snowstorm_params = 0
        if 'snowstorm_parameter_names' in config:
            for param_name, num in config['snowstorm_parameter_names']:
                num_snowstorm_params += num
                for i in range(num):
                    parameter_names.append(param_name.format(i))
        self._untracked_data['num_snowstorm_params'] = num_snowstorm_params

        return parameter_names, sources

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
        x = parameters.params['x']
        y = parameters.params['y']
        z = parameters.params['z']
        time = parameters.params['time']
        energy = parameters.params['energy']
        zenith = parameters.params['zenith']
        azimuth = parameters.params['azimuth']
        length = parameters.params['length']

        num_snowstorm_params = self._untracked_data['num_snowstorm_params']
        snowstorm_params = parameters[:, -num_snowstorm_params:]

        # calculate direction vector
        dir_x = -tf.sin(zenith) * tf.cos(azimuth)
        dir_y = -tf.sin(zenith) * tf.sin(azimuth)
        dir_z = -tf.cos(zenith)

        # create a dictionary for the mapping of source_name: input_parameters
        source_parameter_dict = {}

        # -----------------------------------
        # create parameters for track segment
        # -----------------------------------

        # parameters: x, y, z, zenith, azimuth, energy, time, length, stoch
        source_parameter_dict['track'] = tf.concat(
            (tf.stack([x, y, z, zenith, azimuth, energy, time, length, 0], axis=1),
             snowstorm_params), axis=-1)

        # -------------------------------
        # Compute parameters for Cascades
        # -------------------------------
        # highest energy cascade that is being used as vertex
        # parameters: x, y, z, zenith, azimuth, energy, time
        source_parameter_dict['cascade'] = tf.concat(
            (tf.stack([x, y, z, zenith, azimuth, energy, time,], axis=1,),
             snowstorm_params), axis=-1)

        return source_parameter_dict
