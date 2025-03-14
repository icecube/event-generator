import tensorflow as tf

from egenerator import misc
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

    The config dictionary must contain a mapping of mixture model components
    (decoders) together with the number of these components to add.
    A full example indicating how this would be added to a yaml
    configuration file is shown below:

    '''
    decoder_settings: {

        # The decoder class to use
        decoder_class: "egenerator.model.decoder.mixture.MixtureModel",

        # configuration of the mixture model
        config: {
            # Note that mapping here will just repeat the names,
            # but in other cases it could be used to reuse the
            # same base model multiple times (as for MultiSource objects)
            # Structure is: [name, n_components, weight_factor]
            "decoder_mapping": {
                "AssymetricGaussian": ["AssymetricGaussian", 7, 1.0],
                "GammaFunction": ["GammaFunction", 1, 1.0],
                "ShiftedGammaFunction": ["ShiftedGammaFunction", 2, 1.0],
            },

            # Define a mapping of latent variable to value range functions
            "value_range_mapping": {
                "weight": {
                    "value_range_class": "egenerator.utils.value_range.EluValueRange",
                    "config": {
                        "scale": 1.0,
                        "offset": 1.0,
                        "min_value": 0.00001
                    },
                },
            },
        },

        # Define the decoder types that are used as components of the mixture
        # model
        base_decoders: {
            AssymetricGaussian: {
                'decoder_class': 'egenerator.model.decoder.static.asymmetric_gaussian.AsymmetricGaussianDecoder',
                'config': {
                    'float_precision': float64,

                    # Define a mapping of latent variable to value range functions
                    "value_range_mapping": {
                        "mu": {
                            "value_range_class": "egenerator.utils.value_range.BaseValueRange",
                            "config": {
                                "scale": 0.5,
                                "offset": 0.0,
                            },
                        },
                        "sigma": {
                            "value_range_class": "egenerator.utils.value_range.EluValueRange",
                            "config": {
                                "scale": 1.0,
                                "offset": 2.0,
                                "min_value": 0.0001,
                            },
                        },
                        "r": {
                            "value_range_class": "egenerator.utils.value_range.EluValueRange",
                            "config": {
                                "scale": 1.0,
                                "offset": 1.0,
                                "min_value": 0.0001,
                            },
                        },
                    },
                },

                # Define load_dir here if you want to load the model from a file
                # This does not make sense for a paremeter-less model, but it
                # could be used for a model that has been trained and saved
                # previously such as a normalizing flow model.
                # 'load_dir': 'path/to/model',
            },
            GammaFunction: {
                'decoder_class': 'egenerator.model.decoder.static.gamma.GammaFunctionDecoder',
                'config': {
                    'float_precision': float64,
                    'offset': -0.005, # note that offset is in model time units

                    # Define a mapping of latent variable to value range functions
                    "value_range_mapping": {
                        "alpha": {
                            "value_range_class": "egenerator.utils.value_range.EluValueRange",
                            "config": {
                                "scale": 0.3,
                                "offset": 2.5,
                                "min_value": 0.5,
                            },
                        },
                        "beta": {
                            "value_range_class": "egenerator.utils.value_range.EluValueRange",
                            "config": {
                                "scale": 1.0,
                                "offset": 10.0,
                                "min_value": 1.0,
                            },
                        },
                    },
                },
            },
            ShiftedGammaFunction: {
                'decoder_class': 'egenerator.model.decoder.static.gamma.ShiftedGammaFunctionDecoder',
                'config': {
                    'float_precision': float64,

                    # Define a mapping of latent variable to value range functions
                    "value_range_mapping": {
                        "offset": {
                            "value_range_class": "egenerator.utils.value_range.BaseValueRange",
                            "config": {
                                "scale": 0.001,
                                "offset": -0.005,
                            },
                        },
                        "alpha": {
                            "value_range_class": "egenerator.utils.value_range.EluValueRange",
                            "config": {
                                "scale": 0.3,
                                "offset": 2.5,
                                "min_value": 0.5,
                            },
                        },
                        "beta": {
                            "value_range_class": "egenerator.utils.value_range.EluValueRange",
                            "config": {
                                "scale": 1.0,
                                "offset": 10.0,
                                "min_value": 1.0,
                            },
                        },
                    },
                },
            },
        },
    }
    '''

    In the above example, a mixture model is created with 7 components of
    asymmetric Gaussian decoders and 1 component of a constant offset Gamma
    decoder and 2 components of a shifted Gamma decoder.
    Relevant value range mappings are also defined for the parameters of the
    mixture model.

    """

    @property
    def n_components_total(self):
        """Get the total number of components in the mixture model.

        Returns
        -------
        int
            The total number of components in the mixture model.
        """
        if self._untracked_data is not None:
            return self._untracked_data.get("n_components_total", None)
        else:
            return None

    def is_charge_decoder(self):
        """Check if the decoder is a charge decoder.

        Returns
        -------
        bool or None
            True if the decoder is a charge decoder, False otherwise.
        """
        for base_name in self._untracked_data["models_mapping"].values():
            if not self.sub_components[base_name].is_charge_decoder:
                return False
        return True

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
        decoder_names = sorted(config["decoder_mapping"].keys())
        parameter_names = []
        n_parameters_per_decoder = []
        parameter_slice_per_decoder = []
        n_components_per_decoder = []
        weight_factor_per_decoder = []
        n_components_total = 0
        models_mapping = {}
        current_index = 0
        for decoder_name in decoder_names:
            base_name, n_components, weight_factor = config["decoder_mapping"][
                decoder_name
            ]

            if weight_factor < 0:
                raise ValueError(
                    f"The weight factor '{weight_factor}' "
                    "must be greater than or equal to zero."
                )

            # some sanity checks
            if base_name not in base_models:
                raise ValueError(
                    f"The base model {base_name} is not in "
                    f"the base models: {base_models.keys()}"
                )

            if not isinstance(n_components, int):
                raise ValueError(
                    f"The number of components '{n_components}' "
                    "must be an integer."
                )

            if n_components < 1:
                raise ValueError(
                    f"The number of components '{n_components}' "
                    "must be greater than zero."
                )
            n_components_total += n_components

            # get the parameter names of the base model
            base_decoder = base_models[base_name]
            for i in range(n_components):
                parameter_names += [
                    f"{decoder_name}_{name}_{i:03d}"
                    for name in base_decoder.parameter_names + ["weight"]
                ]
            n_parameters_per_decoder.append(base_decoder.n_parameters + 1)
            n_components_per_decoder.append(n_components)
            weight_factor_per_decoder.append(weight_factor)
            parameter_slice_per_decoder.append(
                slice(
                    current_index,
                    current_index
                    + n_parameters_per_decoder[-1] * n_components,
                )
            )
            models_mapping[decoder_name] = base_name

            current_index += n_parameters_per_decoder[-1] * n_components

        self._untracked_data["n_parameters_per_decoder"] = (
            n_parameters_per_decoder
        )
        self._untracked_data["n_components_per_decoder"] = (
            n_components_per_decoder
        )
        self._untracked_data["weight_factor_per_decoder"] = (
            weight_factor_per_decoder
        )
        self._untracked_data["parameter_slice_per_decoder"] = (
            parameter_slice_per_decoder
        )
        self._untracked_data["decoder_names"] = decoder_names
        self._untracked_data["n_components_total"] = n_components_total

        return parameter_names, models_mapping

    def get_model_inputs_x(self, x, x_is_per_component):
        """Get the input tensor x for the individual components.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor at which to evaluate the MixtureModel.
            Must be same rank as parameters or have one less.
            If same rank, the last dimension must have the size
            of the number of components in the mixture model.
            Shape: [..., n_components] or [...]
        x_is_per_component : bool
            If True, the inputs x is interpreted to have different
            values for each component. In this case, the last dimension
            of x must have the size of the total number of components
            in the mixture model.

        Returns
        -------
        dict of tf.Tensor
            Returns a dictionary of {name: x} pairs, where
            name is the name of the nested Model and x is the input
            tensor at which to evaluate the Model.
            Shape: [..., n_components] or [..., 1].
        """
        x = tf.cast(x, self.float_precision)
        if x_is_per_component:
            if not x.shape[-1] == self.n_components_total:
                raise ValueError(
                    "If `x_is_per_component` is True, the last dimension "
                    "of x must have the size of the total number of "
                    "components in the mixture model."
                )
        else:
            x = x[..., tf.newaxis]

        model_inputs_dict = {}
        component_counter = 0
        for i, name in enumerate(self._untracked_data["decoder_names"]):
            n_components = self._untracked_data["n_components_per_decoder"][i]

            # get the tensor at which to evaluate the base model
            if x_is_per_component:
                x_i = x[
                    ..., component_counter : component_counter + n_components
                ]
            else:
                x_i = x

            model_inputs_dict[name] = x_i

            # update the component counter
            component_counter += n_components

        return model_inputs_dict

    def get_model_parameters(self, parameters):
        """Get the input parameters for the individual models.

        Parameters
        ----------
        parameters : tf.Tensor
            The input parameters for the NestedModel object.
            The input parameters of the individual Model objects are composed
            from these.
            Shape: [..., n_parameters]

        Returns
        -------
        dict of tf.Tensor
            Returns a dictionary of {name: input_parameters} pairs, where
            name is the name of the nested Model and input_parameters
            is a tf.Tensor for the input parameters of that Model.
            Each input_parameters tensor has shape [..., n_parameters_i].
        """
        parameters = tf.cast(parameters, self.float_precision)
        model_parameter_dict = {}
        for i, name in enumerate(self._untracked_data["decoder_names"]):
            n_parameters = self._untracked_data["n_parameters_per_decoder"][i]
            n_components = self._untracked_data["n_components_per_decoder"][i]
            slice_i = self._untracked_data["parameter_slice_per_decoder"][i]
            weight_factor = self._untracked_data["weight_factor_per_decoder"][
                i
            ]

            # get the parameters for the base model
            # shape: [..., n_components_per_base, n_parameters]
            parameters_i = tf.reshape(
                parameters[..., slice_i],
                tf.concat(
                    [
                        tf.shape(parameters)[:-1],
                        (n_components, n_parameters),
                    ],
                    axis=0,
                ),
            )

            # apply weight factor and ensure positive weights
            parameters_i = tf.unstack(parameters_i, axis=-1)
            if weight_factor != 1.0:
                parameters_i[-1] *= weight_factor

            # some safety checks to make sure we aren't clipping too much
            asserts = []
            asserts.append(
                tf.debugging.Assert(
                    tf.reduce_all(parameters_i[-1] > -self.epsilon),
                    ["Weights < 0!", tf.reduce_min(parameters_i[-1])],
                )
            )
            with tf.control_dependencies(asserts):
                parameters_i[-1] = tf.clip_by_value(
                    parameters_i[-1],
                    0.0,
                    float("inf"),
                )

            model_parameter_dict[name] = tf.stack(parameters_i, axis=-1)

        return model_parameter_dict

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

        # create value range object
        self._untracked_data["value_range_mapping"] = {}
        if "value_range_mapping" in config and config["value_range_mapping"]:
            if list(config["value_range_mapping"].keys()) != ["weight"]:
                raise ValueError(
                    "The MixtureModel only modifies the weights to "
                    "individual components. The value range mapping "
                    "therefore should only contain the 'weight' key. "
                    "For other parameters, the value range mapping "
                    "should be defined in the base decoders."
                    f"Provided keys: {config['value_range_mapping'].keys()}"
                )
            settings = config["value_range_mapping"]["weight"]
            ValueClass = misc.load_class(settings["value_range_class"])
            value_range_object = ValueClass(**settings["config"])
            for key in self.parameter_names:
                if "_weight_" in key:
                    self._untracked_data["value_range_mapping"][
                        key
                    ] = value_range_object

        # gather loc parameters
        loc_parameters = []
        for idx, decoder_name in enumerate(
            self._untracked_data["decoder_names"]
        ):

            base = base_models[
                self._untracked_data["models_mapping"][decoder_name]
            ]
            n_components = self._untracked_data["n_components_per_decoder"][
                idx
            ]

            for i in range(n_components):
                loc_parameters += [
                    f"{decoder_name}_{loc_parameter}_{i:03d}"
                    for loc_parameter in base.loc_parameters
                ]
        self._untracked_data["loc_parameters"] = loc_parameters

        return configuration, data, sub_components

    def _component_sum(
        self,
        func_name,
        latent_vars,
        reduce_components=True,
        **kwargs,
    ):
        """Calculate the variance of the PDF.

        Parameters
        ----------
        func_name : str
            The name of the function to call. Values of this function
            are reduced via "weighted sum" operation.
        latent_vars : tf.Tensor
            The latent variables.
            Shape: [..., n_parameters]
        reduce_components : bool, optional
            If True, the contributions of the individual components
            to the overall variance are summed up and returned.
            If False, the output shape will be [..., n_components] where
            n_components is the number of components in the mixture model
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tf.Tensor
            The variance of the PDF.
            Shape: [...]
        """
        parameter_dict = self.get_model_parameters(latent_vars)

        values = []
        total_weight = None
        for name in self._untracked_data["decoder_names"]:

            # shape: [..., n_components_per_base, n_parameters]
            latent_vars_i = parameter_dict[name]

            # shape: [..., n_components_per_base]
            weight = latent_vars_i[..., -1]

            # shape: [..., n_components_per_base]
            base_name = self._untracked_data["models_mapping"][name]
            func = getattr(self.sub_components[base_name], func_name)
            values_i = (
                func(latent_vars=latent_vars_i[..., :-1], **kwargs) * weight
            )

            # shape: [...]
            if total_weight is None:
                total_weight = tf.reduce_sum(weight, axis=-1)
            else:
                total_weight += tf.reduce_sum(weight, axis=-1)

            if reduce_components:
                # shape: [...]
                values_i = tf.reduce_sum(values_i, axis=-1)

            values.append(values_i)

        # shape: [...]
        total_weight += self.epsilon

        if reduce_components:
            values = tf.add_n(values)
            values /= total_weight
        else:
            # shape: [..., n_components_total]
            values = tf.concat(values, axis=-1)
            values /= total_weight[..., tf.newaxis]

        return values

    def _expectation(self, latent_vars, reduce_components=True, **kwargs):
        """Calculate the expectation value of the PDF.

        Parameters
        ----------
        latent_vars : tf.Tensor
            The latent variables which have already been transformed
            by the value range mapping.
            Shape: [..., n_parameters]
        reduce_components : bool, optional
            If True, the contributions of the individual components
            to the overall expectation value are summed up and returned.
            If False, the output shape will be [..., n_components] where
            n_components is the number of components in the mixture model.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tf.Tensor
            The expectation value of the PDF.
            Shape: [...] or [..., n_components]
        """
        return self._component_sum(
            func_name="expectation",
            latent_vars=latent_vars,
            reduce_components=reduce_components,
            **kwargs,
        )

    def _variance(self, latent_vars, reduce_components=True, **kwargs):
        """Calculate the variance of the PDF.

        Parameters
        ----------
        latent_vars : tf.Tensor
            The latent variables.
            Shape: [..., n_parameters]
        reduce_components : bool, optional
            If True, the contributions of the individual components
            to the overall variance are summed up and returned.
            If False, the output shape will be [..., n_components] where
            n_components is the number of components in the mixture model
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tf.Tensor
            The variance of the PDF.
            Shape: [...]
        """
        return self._component_sum(
            func_name="variance",
            latent_vars=latent_vars,
            reduce_components=reduce_components,
            **kwargs,
        )

    def _pdf_or_cdf(
        self, x, latent_vars, func_name, reduce_components=True, **kwargs
    ):
        """Evaluate the decoded PDF or CDF at x.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor at which to evaluate the MixtureModel.
            Must be same rank as parameters or have one less.
            If same rank, the last dimension must have the size
            of the number of components in the mixture model.
            Shape: [..., n_components] or [...]
        latent_vars : tf.Tensor
            The latent variables which have already been transformed
            by the value range mapping.
            Shape: [..., n_parameters]
        func_name : str
            The name of the function to call.
            Can be either 'pdf' or 'cdf'.
        reduce_components : bool, optional
            If True, the PDFs of the individual components are summed up
            and returned as the final PDF. If False, the output shape will
            be [..., n_components] where n_components is the number of
            components in the mixture model.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tf.Tensor
            The PDF evaluated at x for the given latent variables.
        """
        if func_name not in ["pdf", "cdf"]:
            raise ValueError(
                f"Function name '{func_name}' is not supported. "
                "Use either 'pdf' or 'cdf'."
            )

        x_is_per_component = len(x.shape) == len(latent_vars.shape)
        if x_is_per_component and not x.shape[-1] == self.n_components_total:
            raise ValueError(
                "If the input tensor is of the same rank as the "
                "parameters tensor, the last dimension must be the "
                "total number of components in the mixture model."
            )

        self.assert_configured(True)
        parameter_dict = self.get_model_parameters(latent_vars)
        inputs_dict = self.get_model_inputs_x(
            x,
            x_is_per_component=x_is_per_component,
        )

        values = []
        total_weight = None
        for name in self._untracked_data["decoder_names"]:

            # shape: [..., n_components_per_base]
            x_i = inputs_dict[name]

            # shape: [..., n_components_per_base, n_parameters]
            latent_vars_i = parameter_dict[name]

            # shape: [..., n_components_per_base]
            weight = latent_vars_i[..., -1]

            # shape: [..., n_components_per_base]
            base_name = self._untracked_data["models_mapping"][name]
            func = getattr(self.sub_components[base_name], func_name)
            values_i = (
                func(x=x_i, latent_vars=latent_vars_i[..., :-1], **kwargs)
                * weight
            )

            # shape: [...]
            if total_weight is None:
                total_weight = tf.reduce_sum(weight, axis=-1)
            else:
                total_weight += tf.reduce_sum(weight, axis=-1)

            if reduce_components:
                # shape: [...]
                values_i = tf.reduce_sum(values_i, axis=-1)

            values.append(values_i)

        # shape: [...]
        total_weight += self.epsilon

        if reduce_components:
            values = tf.add_n(values)
            values /= total_weight
        else:
            # shape: [..., n_components_total]
            values = tf.concat(values, axis=-1)
            values /= total_weight[..., tf.newaxis]

        return values

    def _pdf(self, x, latent_vars, reduce_components=True, **kwargs):
        """Evaluate the decoded PDF at x.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor at which to evaluate the MixtureModel.
            Must be same rank as parameters or have one less.
            If same rank, the last dimension must have the size
            of the number of components in the mixture model.
            Shape: [..., n_components] or [...]
        latent_vars : tf.Tensor
            The latent variables which have already been transformed
            by the value range mapping.
            Shape: [..., n_parameters]
        reduce_components : bool, optional
            If True, the PDFs of the individual components are summed up
            and returned as the final PDF. If False, the output shape will
            be [..., n_components] where n_components is the number of
            components in the mixture model.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tf.Tensor
            The PDF evaluated at x for the given latent variables.
        """
        return self._pdf_or_cdf(
            x=x,
            latent_vars=latent_vars,
            func_name="pdf",
            reduce_components=reduce_components,
            **kwargs
        )

    def _cdf(self, x, latent_vars, reduce_components=True, **kwargs):
        """Evaluate the decoded CDF at x.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor at which to evaluate the MixtureModel.
            Must be same rank as parameters or have one less.
            If same rank, the last dimension must have the size
            of the number of components in the mixture model.
            Shape: [..., n_components] or [...]
        latent_vars : tf.Tensor
            The latent variables which have already been transformed
            by the value range mapping.
            Shape: [..., n_parameters]
        reduce_components : bool, optional
            If True, the PDFs of the individual components are summed up
            and returned as the final PDF. If False, the output shape will
            be [..., n_components] where n_components is the number of
            components in the mixture model.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tf.Tensor
            The CDF evaluated at x for the given latent variables.
        """
        return self._pdf_or_cdf(
            x=x,
            latent_vars=latent_vars,
            func_name="cdf",
            reduce_components=reduce_components,
            **kwargs
        )

    def sample(self, random_numbers, latent_vars, offsets=None, **kwargs):
        """Get samples for provided uniform random numbers.

        Parameters
        ----------
        random_numbers : tf.Tensor
            The random_numbers in (0, 1) for which to sample.
            Two random numbers are required for each sample.
            Last dimension of random_numbers is therefore 2.
            The first random number is used to select the model and
            component index, the second random number is used to sample
            the value from the selected component.
            Dimensions before that must be broadcastable to the shape of
            the latent variables without the last dimension (n_parameters).
            Shape: [..., 2]
        latent_vars : tf.Tensor
            The latent variables.
            Shape: [..., n_parameters]
        offsets : tf.Tensor, optional
            Time offsets for each of the mixture components.
            Shape: [..., n_components_total]
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tf.Tensor
            The sampled values for the provided random_numbers
            and latent variables.
        """
        self.assert_configured(True)
        latent_vars = self._apply_value_range(latent_vars)
        parameter_dict = self.get_model_parameters(latent_vars)

        if random_numbers.shape[-1] != 2:
            raise ValueError("The last dimension of random_numbers must be 2.")

        # compute cumulative weights
        weights = []
        for name in self._untracked_data["decoder_names"]:

            # shape: [..., n_components_per_base]
            weights_i = parameter_dict[name][..., -1]
            weights.append(weights_i)

        # shape: [..., n_components_total]
        weights = tf.concat(weights, axis=-1)
        cum_weights = tf.math.cumsum(weights, axis=-1)
        cum_weights /= tf.reduce_sum(weights, axis=-1, keepdims=True)

        # set last element to 1.0 + epsilon to avoid numerical issues
        cum_weights = tf.concat(
            [
                cum_weights[..., :-1],
                tf.ones_like(cum_weights[..., -1:]) + self.epsilon,
            ],
            axis=-1,
        )

        # select component index based on q
        # shape: [...]
        component_idx = tf.argmax(
            tf.cast(random_numbers[..., :1] <= cum_weights, tf.int32), axis=-1
        )

        # create ppf tensor for each model and component
        ppf_values = []
        for name in self._untracked_data["decoder_names"]:
            latent_vars_i = parameter_dict[name]

            # shape: [..., n_components_per_base]
            ppf_value = self.sub_components[name].ppf(
                q=random_numbers[..., 1:],
                latent_vars=latent_vars_i[..., :-1],
                **kwargs
            )
            ppf_values.append(ppf_value)

        # shape: [..., n_components_total]
        ppf_values = tf.concat(ppf_values, axis=-1)

        # choose ppf value based on model and component index
        # shape: [...]
        ppf_values = tf.gather(
            ppf_values,
            component_idx,
            batch_dims=len(tf.shape(random_numbers)[:-1]),
            axis=-1,
        )

        # add offsets
        if offsets is not None:
            offsets_chosen = tf.gather(
                offsets,
                component_idx,
                batch_dims=len(tf.shape(random_numbers)[:-1]),
                axis=-1,
            )
            ppf_values += offsets_chosen

        return ppf_values
