from __future__ import division, print_function
import logging
import tensorflow as tf
from egenerator.model.multi_source.base import MultiSource


class IndependentMultiSource(MultiSource):

    """This is a MultiSource that assumes that the parameters of each of its
    sources are independent, e.g. there are no constraints on the source
    paramters.
    """

    def __init__(self, logger=None):
        """Instantiate Source class

        Parameters
        ----------
        logger : logging.logger, optional
            The logger to use.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(IndependentMultiSource, self).__init__(logger=self._logger)

    def get_parameters(self, config, base_sources, sources):
        """Get parameter names and their ordering.

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
        sources : dict
            This describes the sources which compose the event hypothesis.
            The dictionary is a mapping from source_name (str) to
            base_source (str). This mapping allows the reuse of a single
            source component instance. For instance, a muon can be build up of
            multiple cascades. However, all cascades should use the same
            underlying model. Hence, in this case only one base_source is
            required: the cascade source. The mapping will then map all
            cascades in the hypothesis to this one base cascade source.

        Returns
        -------
        list of str
            A list of parameter names of the MultiSource object.

        """
        parameters = []
        for cascade in sorted(sources.keys()):
            for variable in sorted(base_sources[cascade].parameter_names):
                parameters.append(cascade + '_' + variable)
        return parameters

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
        source_parameter_dict = {}
        counter = 0
        for cascade in sorted(sources.keys()):
            num = base_sources[cascade].num_parameters
            source_parameter_dict[cascade] = \
                parameters[:, counter:counter + num]
            counter += num

        return source_parameter_dict

