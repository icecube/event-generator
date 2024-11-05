import logging

from egenerator import misc
from egenerator.model.base import Model
from egenerator.manager.component import Configuration


class LatentToPDFDecoder(Model):
    """Base class to decode latent variables into PDFs.

    This is an abstract class that defines the interface for all decoders
    that decode latent variables into PDFs. The decoder is a model that
    takes latent variables as input and produces a properly normalized
    probability density (PDF).

    Derived classes must implement the following methods:

        - _build_architecture:
            Set up and build architecture: create and save
            all model weights. Must return a list of latent variable names.
        - pdf:
            Evaluate the PDF at a given input tensor.
        - cdf:
            Evaluate the CDF at a given input tensor.

    Optionally, derived classes may implement the following methods:
        - ppf:
            Evaluate the percent point function at a given input tensor.
    """

    def __init__(self, logger=None):
        """Instantiate Decoder class

        Parameters
        ----------
        logger : logging.logger, optional
            The logger to use.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(LatentToPDFDecoder, self).__init__(logger=self._logger)

    def _configure_derived_class(self, config, name=None):
        """Setup and configure the Decoder's architecture.

        After this function call, the decoders's architecture (weights) must
        be fully defined and may not change again afterwards.

        Parameters
        ----------
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

        # check that all created variables are part of the module.variables

        """Build the architecture. This must return:

            tf.Module:
                - Must contain all tensorflow variables as class attributes,
                    so that they are found by tf.Module.variables
        """
        if name is None:
            name = __name__

        # build architecture: create and save model weights (if any)
        # and return the names of the latent variables
        latent_names = self._build_architecture(config, name=name)

        # get names of latent variables
        self._untracked_data["name"] = name
        self._set_parameter_names(latent_names)

        # create configuration object
        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings=dict(config=config),
            mutable_settings=dict(name=name),
        )

        return configuration, {}, {}

    def _build_architecture(self, config, name=None):
        """Set up and build architecture: create and save all model weights.

        This is a virtual method which must be implemented by the derived
        source class.

        Parameters
        ----------
        config : dict
            A dictionary of settings that fully defines the architecture.
        name : str, optional
            The name of the source.
            If None is provided, the class name __name__ will be used.

        Returns
        -------
        list of str
            A list of parameter names. These parameters describe the names
            of the latent variables used as input to thd decoder.
            These name must be in the same order as the latent variables
            in the last dimension of the input tensor
            passed to the pdf, cdf, and ppf methods.
        """
        self.assert_configured(False)
        raise NotImplementedError()

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
