from egenerator.utils import basis_functions
from egenerator.model.decoder.base import LatentToPDFDecoder


class AsymmetricGaussianDecoder(LatentToPDFDecoder):

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
        return ["mu", "sigma", "r"]

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
        return basis_functions.tf_asymmetric_gauss_expectation(
            mu=latent_vars[..., self.get_index("mu")],
            sigma=latent_vars[..., self.get_index("sigma")],
            r=latent_vars[..., self.get_index("r")],
            dtype=self.configuration.config["config"]["float_precision"],
        )

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
        return basis_functions.tf_asymmetric_gauss(
            x=x,
            mu=latent_vars[..., self.get_index("mu")],
            sigma=latent_vars[..., self.get_index("sigma")],
            r=latent_vars[..., self.get_index("r")],
            dtype=self.configuration.config["config"]["float_precision"],
        )

    def _log_pdf(self, x, latent_vars, **kwargs):
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
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tf.Tensor
            The PDF evaluated at x for the given latent variables.
        """
        return basis_functions.tf_log_asymmetric_gauss(
            x=x,
            mu=latent_vars[..., self.get_index("mu")],
            sigma=latent_vars[..., self.get_index("sigma")],
            r=latent_vars[..., self.get_index("r")],
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
        return basis_functions.tf_asymmetric_gauss_cdf(
            x=x,
            mu=latent_vars[..., self.get_index("mu")],
            sigma=latent_vars[..., self.get_index("sigma")],
            r=latent_vars[..., self.get_index("r")],
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
        return basis_functions.tf_asymmetric_gauss_ppf(
            q=q,
            mu=latent_vars[..., self.get_index("mu")],
            sigma=latent_vars[..., self.get_index("sigma")],
            r=latent_vars[..., self.get_index("r")],
            dtype=self.configuration.config["config"]["float_precision"],
        )
