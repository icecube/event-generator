import tensorflow as tf

from egenerator import misc
from egenerator.utils import tf_helpers
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
        - _pdf:
            Evaluate the PDF at a given input tensor.
        - _cdf:
            Evaluate the CDF at a given input tensor.

    Optionally, derived classes may implement the following methods:
        - _ppf:
            Evaluate the percent point function at a given input tensor.
    """

    def is_charge_decoder(self):
        """Check if the decoder is a charge decoder.

        Returns
        -------
        bool
            True if the decoder is a charge decoder, False otherwise.
        """
        raise NotImplementedError

    @property
    def value_range_mapping(self):
        """Get the value range mapping.

        Returns
        -------
        dict
            The value range mapping.
        """
        if self._untracked_data is not None:
            return self._untracked_data.get("value_range_mapping", None)
        else:
            return None

    @property
    def loc_parameters(self):
        """Get the location parameters.

        The location parameters are the names of the latent variables
        that affect the expectation value of the PDF.
        Not all decoders have location parameters.

        Returns
        -------
        list of str
            The location parameters.
        """
        if self._untracked_data is not None:
            return self._untracked_data.get("loc_parameters", None)
        else:
            return None

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
        latent_names, loc_parameters = self._build_architecture(
            config,
            name=name,
        )

        # get names of latent variables
        self._untracked_data["name"] = name
        self._set_parameter_names(latent_names)
        self._untracked_data["loc_parameters"] = loc_parameters

        # create value range object
        self._untracked_data["value_range_mapping"] = {}
        if "value_range_mapping" in config:
            for key, settings in config["value_range_mapping"].items():
                assert (
                    key in latent_names
                ), f"Key {key} not in latent names {latent_names}"
                ValueClass = misc.load_class(settings["value_range_class"])
                self._untracked_data["value_range_mapping"][key] = ValueClass(
                    **settings["config"]
                )

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

    def _apply_value_range(self, latent_vars):
        """Apply value range function to latent variables.

        Parameters
        ----------
        latent_vars : tf.Tensor
            The latent variables.
            Shape: [..., n_parameters]

        Returns
        -------
        tf.Tensor
            The latent variables after applying the value range mapping.
        """
        latent_vars_list = []
        for key in self.parameter_names:
            if key in self.value_range_mapping:
                latent_vars_list.append(
                    self.value_range_mapping[key](
                        latent_vars[..., self.get_index(key)]
                    )
                )
            else:
                latent_vars_list.append(latent_vars[..., self.get_index(key)])

        return tf.stack(latent_vars_list, axis=-1)

    def expectation(self, latent_vars, **kwargs):
        """Calculate the expectation value of the PDF.

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
            The expectation value of the PDF.
            Shape: [...]
        """
        self.assert_configured(True)
        latent_vars = self._apply_value_range(latent_vars)
        return self._expectation(latent_vars, **kwargs)

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
        raise NotImplementedError

    def variance(self, latent_vars, **kwargs):
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
        self.assert_configured(True)
        latent_vars = self._apply_value_range(latent_vars)
        return self._variance(latent_vars, **kwargs)

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
        raise NotImplementedError

    def pdf(self, x, latent_vars, **kwargs):
        """Evaluate the decoded PDF at x.

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
        self.assert_configured(True)
        latent_vars = self._apply_value_range(latent_vars)
        return self._pdf(x, latent_vars, **kwargs)

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
        """
        raise NotImplementedError

    def log_pdf(self, x, latent_vars, **kwargs):
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
        self.assert_configured(True)
        latent_vars = self._apply_value_range(latent_vars)
        return self._log_pdf(x, latent_vars, **kwargs)

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
        return tf.math.log(self._pdf(x, latent_vars, **kwargs))

    def cdf(self, x, latent_vars, **kwargs):
        """Evaluate the decoded CDF at x.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor at which to evaluate the CDF.
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
            The CDF evaluated at x for the given latent variables.
        """
        self.assert_configured(True)
        latent_vars = self._apply_value_range(latent_vars)
        return tf_helpers.safe_cdf_clip(self._cdf(x, latent_vars, **kwargs))

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
        raise NotImplementedError

    def ppf(self, q, latent_vars, **kwargs):
        """Evaluate the decoded PPF at q.

        Parameters
        ----------
        q : tf.Tensor
            The input tensor at which to evaluate the PPF.
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
            The PPF evaluated at q for the given latent variables.
        """
        self.assert_configured(True)
        latent_vars = self._apply_value_range(latent_vars)
        return self._ppf(q, latent_vars, **kwargs)

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
        raise NotImplementedError

    def sample(self, random_numbers, latent_vars, **kwargs):
        """Get samples for provided uniform random numbers.

        Parameters
        ----------
        random_numbers : tf.Tensor
            The random_numbers in (0, 1) for which to sample.
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
            The sampled values for the provided random_numbers
            and latent variables.
        """
        return self.ppf(random_numbers, latent_vars, **kwargs)
