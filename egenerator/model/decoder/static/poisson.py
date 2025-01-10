import tensorflow as tf

from egenerator.utils import basis_functions
from egenerator.model.decoder.base import LatentToPDFDecoder


class NegativeBinomialDecoder(LatentToPDFDecoder):
    """Continuous Negative Binomial distribution.

    The parameterization chosen here is defined by the mean mu and
    the over-dispersion factor alpha.

        Var(x) = mu + alpha*mu**2
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
            of the latent variables used as input to the decoder.
            These name must be in the same order as the latent variables
            in the last dimension of the input tensor
            passed to the pdf, cdf, and ppf methods.
        list of str
            A list of location parameters that directly
            shift the expectation value of the PDF.
            Note: this does not directly exist for all distributions.
            The charge scaling will be applied to these laten variables.
        """
        self.assert_configured(False)
        return ["mu", "alpha"], ["mu"]

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
        expectation = tf.cast(
            latent_vars[..., self.get_index("mu")],
            self.configuration.config["config"]["float_precision"],
        )
        if "offset" in self.configuration.config["config"]:
            expectation += self.configuration.config["config"]["offset"]
        return expectation

    def _variance(self, latent_vars, **kwargs):
        """Calculate the variance of the PDF.

        Parameters
        ----------
        latent_vars : tf.Tensor
            The latent variables.
            Shape: [..., n_parameters]
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tf.Tensor
            The variance of the PDF.
            Shape: [...]
        """
        mu = latent_vars[..., self.get_index("mu")]
        alpha = latent_vars[..., self.get_index("alpha")]
        variance = tf.cast(
            mu + alpha * mu**2,
            self.configuration.config["config"]["float_precision"],
        )
        return variance

    def _pdf(self, x, latent_vars, add_normalization_term=True, **kwargs):
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
        add_normalization_term : bool, optional
            If True, the normalization term is computed and added.
            Note: this term is not required for minimization and the
            negative binomial distribution only has a proper normalization
            for integer x. For real-valued x the negative binomial is
            not properly normalized and hence adding the normalization
            term does not ensure normalization.
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
        return tf.math.exp(
            basis_functions.tf_log_negative_binomial(
                x=x,
                mu=latent_vars[..., self.get_index("mu")],
                alpha=latent_vars[..., self.get_index("alpha")],
                add_normalization_term=add_normalization_term,
                dtype=self.configuration.config["config"]["float_precision"],
            )
        )

    def _log_pdf(self, x, latent_vars, add_normalization_term=True, **kwargs):
        """Evaluate the logarithm of the decoded PDF at x.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor at which to evaluate the PDF.
            Broadcastable to the shape of the latent variables
            without the last dimension (n_parameters).
            Shape: [...]
        latent_vars : tf.Tensor
            The latent variables.
            Shape: [..., n_parameters]
        add_normalization_term : bool, optional
            If True, the normalization term is computed and added.
            Note: this term is not required for minimization and the
            negative binomial distribution only has a proper normalization
            for integer x. For real-valued x the negative binomial is
            not properly normalized and hence adding the normalization
            term does not ensure normalization.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tf.Tensor
            The PDF evaluated at x for the given latent variables.
        """
        if "offset" in self.configuration.config["config"]:
            x = x - self.configuration.config["config"]["offset"]
        return basis_functions.tf_log_negative_binomial(
            x=x,
            mu=latent_vars[..., self.get_index("mu")],
            alpha=latent_vars[..., self.get_index("alpha")],
            add_normalization_term=add_normalization_term,
            dtype=self.configuration.config["config"]["float_precision"],
        )


class PoissonDecoder(LatentToPDFDecoder):

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
            of the latent variables used as input to the decoder.
            These name must be in the same order as the latent variables
            in the last dimension of the input tensor
            passed to the pdf, cdf, and ppf methods.
        list of str
            A list of location parameters that directly
            shift the expectation value of the PDF.
            Note: this does not directly exist for all distributions.
            The charge scaling will be applied to these laten variables.
        """
        self.assert_configured(False)
        return ["mu"], ["mu"]

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
        expectation = tf.cast(
            latent_vars[..., self.get_index("mu")],
            self.configuration.config["config"]["float_precision"],
        )
        if "offset" in self.configuration.config["config"]:
            expectation += self.configuration.config["config"]["offset"]
        return expectation

    def _variance(self, latent_vars, **kwargs):
        """Calculate the variance of the PDF.

        Parameters
        ----------
        latent_vars : tf.Tensor
            The latent variables.
            Shape: [..., n_parameters]
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tf.Tensor
            The variance of the PDF.
            Shape: [...]
        """
        variance = tf.cast(
            latent_vars[..., self.get_index("mu")],
            self.configuration.config["config"]["float_precision"],
        )
        return variance

    def _pdf(self, x, latent_vars, add_normalization_term=True, **kwargs):
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
        add_normalization_term : bool, optional
            If True, the normalization term is computed and added.
            Note: this term is not required for minimization and the
            negative binomial distribution only has a proper normalization
            for integer x. For real-valued x the negative binomial is
            not properly normalized and hence adding the normalization
            term does not ensure normalization.
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

        return basis_functions.tf_poisson_pdf(
            x=x,
            mu=latent_vars[..., self.get_index("mu")],
            add_normalization_term=add_normalization_term,
            dtype=self.configuration.config["config"]["float_precision"],
        )

    def _log_pdf(self, x, latent_vars, add_normalization_term=True, **kwargs):
        """Evaluate the logarithm of the decoded PDF at x.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor at which to evaluate the PDF.
            Broadcastable to the shape of the latent variables
            without the last dimension (n_parameters).
            Shape: [...]
        latent_vars : tf.Tensor
            The latent variables.
            Shape: [..., n_parameters]
        add_normalization_term : bool, optional
            If True, the normalization term is computed and added.
            Note: this term is not required for minimization and the
            negative binomial distribution only has a proper normalization
            for integer x. For real-valued x the negative binomial is
            not properly normalized and hence adding the normalization
            term does not ensure normalization.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tf.Tensor
            The PDF evaluated at x for the given latent variables.
        """
        if "offset" in self.configuration.config["config"]:
            x = x - self.configuration.config["config"]["offset"]

        return basis_functions.tf_poisson_log_pdf(
            x=x,
            mu=latent_vars[..., self.get_index("mu")],
            add_normalization_term=add_normalization_term,
            dtype=self.configuration.config["config"]["float_precision"],
        )
