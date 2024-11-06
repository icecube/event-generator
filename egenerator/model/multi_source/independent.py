import logging
from egenerator.model.multi_source.base import MultiSource


class IndependentMultiSource(MultiSource):
    """This is a MultiSource that assumes that the parameters of each of its
    sources are independent, e.g. there are no constraints on the source
    parameters.
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

    def get_parameters_and_mapping(self, config, base_models):
        """Get parameter names and their ordering as well as source mapping.

        This is a pure virtual method that must be implemented by
        derived class.

        Parameters
        ----------
        config : dict
            A dictionary of settings. The config must contain a 'sources'
            dictionary with the following structure:
                'sources': {
                    'independent_source1': base_source_name_a,
                    'independent_source2': base_source_name_b,
                }
        base_models : dict of Source objects
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
        sources = config["sources"]

        parameters = []
        for cascade in sorted(sources.keys()):
            base = sources[cascade]
            for variable in base_models[base].parameter_names:
                parameters.append(cascade + "_" + variable)
        return parameters, sources

    def get_model_parameters(self, parameters):
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
            Each input_parameters tensor has shape [..., num_parameters_i].
        """
        source_parameter_dict = {}
        counter = 0
        for cascade in sorted(self._untracked_data["models_mapping"].keys()):
            base = self._untracked_data["models_mapping"][cascade]
            num = self.sub_components[base].num_parameters
            source_parameter_dict[cascade] = parameters[
                :, counter : counter + num
            ]
            counter += num

        return source_parameter_dict
