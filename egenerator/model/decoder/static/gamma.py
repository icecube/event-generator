from egenerator.utils import basis_functions
from egenerator.model.decoder.base import LatentToPDFDecoder


class GammaFunctionDecoder(LatentToPDFDecoder):
    """A decoder for a gamma function.

    A constant offset can be added to the gamma function by specifying
    the "offset" key in the configuration. The offset is applied
    to the input x of the gamma function, i.e. the pdf/cdf will be
    evaluated at `x' = x - offset`.
    """

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
        return ["alpha", "beta"]

    def _expectation(self, latent_vars, **kwargs):
        """Calculate the expectation value of the PDF.

        Parameters
        ----------
        latent_vars : tf.Tensor
            The latent variables which have already been transformed
            by the value range mapping.
            Shape: [..., n_parameters]
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tf.Tensor
            The expectation value of the PDF.
            Shape: [...]
        """
        expectation = basis_functions.tf_gamma_expectation(
            alpha=latent_vars[..., self.get_index("alpha")],
            beta=latent_vars[..., self.get_index("beta")],
            dtype=self.configuration.config["config"]["float_precision"],
        )

        if "offset" in self.configuration.config["config"]:
            expectation += self.configuration.config["config"]["offset"]

        return expectation

    def _pdf(self, x, latent_vars, **kwargs):
        """Evaluate the decoded PDF at x.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor at which to evaluate the PDF.
            Broadcastable to the shape of the latent variables
            without the last dimension (n_parameters).
            Shape: [...]
        latent_vars : tf.Tensor
            The latent variables which have already been transformed
            by the value range mapping.
            Shape: [..., n_parameters]
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tf.Tensor
            The PDF evaluated at x for the given latent variables.
            Shape: [...]
        """
        if "offset" in self.configuration.config["config"]:
            x = x - self.configuration.config["config"]["offset"]

        return basis_functions.tf_gamma_pdf(
            x=x,
            alpha=latent_vars[..., self.get_index("alpha")],
            beta=latent_vars[..., self.get_index("beta")],
            dtype=self.configuration.config["config"]["float_precision"],
        )

    def _cdf(self, x, latent_vars, **kwargs):
        """Evaluate the decoded CDF at x.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor at which to evaluate the CDF.
            Broadcastable to the shape of the latent variables
            without the last dimension (n_parameters).
            Shape: [...]
        latent_vars : tf.Tensor
            The latent variables which have already been transformed
            by the value range mapping.
            Shape: [..., n_parameters]
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tf.Tensor
            The CDF evaluated at x for the given latent variables.
        """
        if "offset" in self.configuration.config["config"]:
            x = x - self.configuration.config["config"]["offset"]
        return basis_functions.tf_gamma_cdf(
            x=x,
            alpha=latent_vars[..., self.get_index("alpha")],
            beta=latent_vars[..., self.get_index("beta")],
            dtype=self.configuration.config["config"]["float_precision"],
        )

    def _ppf(self, q, latent_vars, **kwargs):
        """Evaluate the decoded PPF at q.

        Parameters
        ----------
        q : tf.Tensor
            The input tensor at which to evaluate the PPF.
            Broadcastable to the shape of the latent variables
            without the last dimension (n_parameters).
            Shape: [...]
        latent_vars : tf.Tensor
            The latent variables which have already been transformed
            by the value range mapping.
            Shape: [..., n_parameters]
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tf.Tensor
            The PPF evaluated at q for the given latent variables.
        """
        x_values = basis_functions.tf_gamma_ppf(
            q=q,
            alpha=latent_vars[..., self.get_index("alpha")],
            beta=latent_vars[..., self.get_index("beta")],
            dtype=self.configuration.config["config"]["float_precision"],
        )
        if "offset" in self.configuration.config["config"]:
            x_values += self.configuration.config["config"]["offset"]
        return x_values


class ShiftedGammaFunctionDecoder(GammaFunctionDecoder):
    """A decoder for a shifted gamma function.

    An additional latent variable "offset" is added to the gamma function.
    The offset is applied to the input x of the gamma function, i.e.
    the pdf/cdf will be evaluated at `x' = x - offset`.
    """

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
        return ["alpha", "beta", "offset"]

    def _pdf(self, x, latent_vars, **kwargs):
        """Evaluate the decoded PDF at x.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor at which to evaluate the PDF.
            Broadcastable to the shape of the latent variables
            without the last dimension (n_parameters).
            Shape: [...]
        latent_vars : tf.Tensor
            The latent variables which have already been transformed
            by the value range mapping.
            Shape: [..., n_parameters]
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tf.Tensor
            The PDF evaluated at x for the given latent variables.
            Shape: [...]
        """
        offset = latent_vars[..., self.get_index("offset")]
        return super()._pdf(x - offset, latent_vars, **kwargs)

    def _cdf(self, x, latent_vars, **kwargs):
        """Evaluate the decoded CDF at x.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor at which to evaluate the CDF.
            Broadcastable to the shape of the latent variables
            without the last dimension (n_parameters).
            Shape: [...]
        latent_vars : tf.Tensor
            The latent variables which have already been transformed
            by the value range mapping.
            Shape: [..., n_parameters]
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tf.Tensor
            The CDF evaluated at x for the given latent variables.
        """
        offset = latent_vars[..., self.get_index("offset")]
        return super()._cdf(x - offset, latent_vars, **kwargs)

    def _ppf(self, q, latent_vars, **kwargs):
        """Evaluate the decoded PPF at q.

        Parameters
        ----------
        q : tf.Tensor
            The input tensor at which to evaluate the PPF.
            Broadcastable to the shape of the latent variables
            without the last dimension (n_parameters).
            Shape: [...]
        latent_vars : tf.Tensor
            The latent variables which have already been transformed
            by the value range mapping.
            Shape: [..., n_parameters]
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tf.Tensor
            The PPF evaluated at q for the given latent variables.
        """
        offset = latent_vars[..., self.get_index("offset")]
        x_values = super()._ppf(q, latent_vars, **kwargs)
        x_values += offset
        return x_values
