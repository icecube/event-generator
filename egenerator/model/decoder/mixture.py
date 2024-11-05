import logging

from egenerator.model.nested import NestedModel
from egenerator.model.decoder.base import LatentToPDFDecoder


class MixtureModel(NestedModel, LatentToPDFDecoder):
    """Compose mixture models from multiple decoders.

    This is a wrapper class that allows to combine multiple decoders into a
    single decoder by composing them as a mixture model. The mixture model
    is defined by a set of weights that determine the contribution of each
    decoder to the final output. The weights are normalized to sum up to one.

    Instances of MixtureModel can be nested, i.e., a MixtureModel can contain
    other MixtureModel instances.

    Note that this class is similar to the implementation in
    `egenerator.source.multi_source.MultiSource`.
    """

    def __init__(self, logger=None):
        """Instantiate Decoder class

        Parameters
        ----------
        logger : logging.logger, optional
            The logger to use.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(MixtureModel, self).__init__(logger=self._logger)

    def get_parameters_and_mapping(self, config, base_models):
        """Get parameter names of the model input tensor models mapping.

        This is a pure virtual method that must be implemented by
        derived class.

        Parameters
        ----------
        config : dict
            A dictionary of settings.
        base_models : dict of Model objects
            A dictionary of models. These models are used as a basis for
            the NestedModel object. The final NestedModel can be made up
            of multiple models which may be created from one or more
            base model objects.

        Returns
        -------
        list of str
            A list of parameter names describing the input tensor to
            the NestedModel object. These parameter names correspond to
            the last dimension of the input tensor.
        dict
            This describes the models which compose the NestedModel.
            The dictionary is a mapping from model_name (str) to
            base_model (str). This mapping allows the reuse of a single
            model component instance. For instance, a muon can be build up
            of multiple cascades. However, all cascades should use the same
            underlying model. Hence, in this case only one base_model is
            required: the cascade model. The mapping will then map all
            cascades in the hypothesis to this one base cascade model.
        """
        raise NotImplementedError()

    def get_model_parameters(self, parameters):
        """Get the input parameters for the individual models.

        Parameters
        ----------
        parameters : tf.Tensor
            The input parameters for the NestedModel object.
            The input parameters of the individual Model objects are composed
            from these.
            Shape: [..., num_parameters]

        Returns
        -------
        dict of tf.Tensor
            Returns a dictionary of (name: input_parameters) pairs, where
            name is the name of the nested Model and input_parameters
            is a tf.Tensor for the input parameters of that Model.
            Each input_parameters tensor has shape [..., num_parameters_i].
        """
        raise NotImplementedError()

    def _configure_derived_class(
        self,
        base_models,
        config,
        name=None,
    ):
        """Setup and configure the Decoder's architecture.

        After this function call, the decoders's architecture (weights) must
        be fully defined and may not change again afterwards.

        Parameters
        ----------
        base_models : dict of LatentToPDFDecoder
            A dictionary of decoders. These decoders are used as a basis for
            the MixtureModel object. The mixture model can be made up of
            multiple components (decoders) which may be created from
            one or more `base_models` objects.
        config : dict
            A dictionary of settings which is used to set up the model
            architecture and weights.
        name : str, optional
            The name of the decoder.

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
            The data of the component.
            Return None, if the component has no data that needs to be tracked.
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
        if name is None:
            name = __name__

        configuration, data, sub_components = super(
            MixtureModel, self
        )._configure_derived_class(
            base_models=base_models,
            config=config,
            name=name,
        )

        return configuration, data, sub_components

    def pdf(self, x, latent_vars, *args, **kwargs):
        """Evaluate the decoded PDF at x.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor at which to evaluate the PDF.
            Broadcastable to the shape of the latent variables
            without the last dimension (num_parameters).
        latent_vars : tf.Tensor
            The latent variables.
            Shape: [..., num_parameters]
        *args
            Additional arguments.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tf.Tensor
            The PDF evaluated at x for the given latent variables.
        """
        self.assert_configured(True)
        raise NotImplementedError

    def cdf(self, x, latent_vars, *args, **kwargs):
        """Evaluate the decoded CDF at x.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor at which to evaluate the CDF.
            Broadcastable to the shape of the latent variables
            without the last dimension (num_parameters).
        latent_vars : tf.Tensor
            The latent variables.
            Shape: [..., num_parameters]
        *args
            Additional arguments.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tf.Tensor
            The CDF evaluated at x for the given latent variables.
        """
        self.assert_configured(True)
        raise NotImplementedError

    def ppf(self, q, latent_vars, *args, **kwargs):
        """Evaluate the decoded PPF at p.

        Parameters
        ----------
        q : tf.Tensor
            The input tensor at which to evaluate the PPF.
            Broadcastable to the shape of the latent variables
            without the last dimension (num_parameters).
        latent_vars : tf.Tensor
            The latent variables.
            Shape: [..., num_parameters]
        *args
            Additional arguments.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tf.Tensor
            The PPF evaluated at q for the given latent variables.
        """
        self.assert_configured(True)
        raise NotImplementedError
