import logging
import tensorflow as tf

from egenerator.model.nested import NestedModel, ConcreteFunctionCache
from egenerator.model.source.base import Source
from egenerator.utils import tf_helpers


class MultiSource(NestedModel, Source):
    """Defines base class for an unbinned MultiSource.

    This is an abstract class for a combination of unbinned sources.
    Unbinned in this context, means that an unbinned likelihood is used for
    the measured pulses.

    A derived class must implement
    ------------------------------
    get_parameters_and_mapping(self, config, sources):
        Get parameter names and their ordering as well as source mapping.

        This is a pure virtual method that must be implemented by
        derived class.

        Parameters
        ----------
        config : dict
            A dictionary of settings.
        base_models : dict of Source objects
            A dictionary of sources. These sources are used as a basis for
            the MultiSource object. The event hypothesis can be made up of
            multiple sources which may be created from one or more
            base source objects.

        Returns
        -------
        list of str
            A list of parameters of the MultiSource object.
        dict
            This describes the sources which compose the event hypothesis.
            The dictionary is a mapping from source_name (str) to
            base_source (str). This mapping allows the reuse of a single
            source component instance. For instance, a muon can be build up of
            multiple cascades. However, all cascades should use the same
            underlying model. Hence, in this case only one base_source is
            required: the cascade source. The mapping will then map all
            cascades in the hypothesis to this one base cascade source.

    get_model_parameters(self, parameters):
        Get the input parameters for the individual models.

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

    Attributes
    ----------
    name : str
        The name of the source.

    parameter_names: list of str
        The names of the n_params number of parameters.
    """

    def __init__(self, logger=None):
        """Instantiate Source class

        Parameters
        ----------
        logger : logging.logger, optional
            The logger to use.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(MultiSource, self).__init__(logger=self._logger)

    def _configure_derived_class(
        self, base_models, config, data_trafo=None, decoder=None, name=None
    ):
        """Setup and configure the Source's architecture.

        After this function call, the sources's architecture (weights) must
        be fully defined and may not change again afterwards.

        Parameters
        ----------
        base_models : dict of Source objects
            A dictionary of sources. These sources are used as a basis for
            the MultiSource object. The event hypothesis can be made up of
            multiple sources which may be created from one or more
            base source objects.
        config : dict
            A dictionary of settings which is used to set up the model
            architecture and weights.
        data_trafo : DataTrafo
            A data trafo object.
        decoder : LatentToPDFDecoder, optional
            The decoder object. This is an optional object that can
            be used to decode the latent variables into a PDF.
        name : str, optional
            The name of the source.

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
        NotImplementedError
            Description
        ValueError
            Description
        """
        if name is None:
            name = __name__

        configuration, data, sub_components = super(
            MultiSource, self
        )._configure_derived_class(
            base_models=base_models,
            config=config,
            name=name,
        )

        if data_trafo is not None:
            sub_components["data_trafo"] = data_trafo

        if decoder is not None:
            sub_components["decoder"] = decoder

        return configuration, data, sub_components

    @tf.function
    def get_tensors(
        self,
        data_batch_dict,
        is_training,
        parameter_tensor_name="x_parameters",
    ):
        """Get tensors computed from input parameters and pulses.

        Parameters are the hypothesis tensor of the source with
        shape [-1, n_params]. The get_tensors method must compute all tensors
        that are to be used in later steps. It returns these as a dictionary
        of output tensors.

        Parameters
        ----------
        data_batch_dict : dict of tf.Tensor
            parameters : tf.Tensor
                A tensor which describes the input parameters of the source.
                This fully defines the source hypothesis. The tensor is of
                shape [-1, n_params] and the last dimension must match the
                order of the parameter names (self.parameter_names),
            pulses : tf.Tensor
                The input pulses (charge, time) of all events in a batch.
                Shape: [-1, 2]
            pulses_ids : tf.Tensor
                The pulse indices (batch_index, string, dom, pulse_number)
                of all pulses in the batch of events.
                Shape: [-1, 4]
        is_training : bool, optional
            Indicates whether currently in training or inference mode.
            Must be provided if batch normalisation is used.
            True: in training mode
            False: inference mode.
        parameter_tensor_name : str, optional
            The name of the parameter tensor to use. Default: 'x_parameters'

        Raises
        ------
        ValueError
            Description

        Returns
        -------
        dict of tf.Tensor
            A dictionary of output tensors.
            This  dictionary must at least contain:

                'dom_charges': the predicted charge at each DOM
                               Shape: [-1, 86, 60, 1]
                'pulse_pdf': The likelihood evaluated for each pulse
                             Shape: [-1]

            Optional:

                'pulse_cdf': The cumulative likelihood evaluated
                             for each pulse
                             Shape: [-1]

        """
        self.assert_configured(True)

        parameters = data_batch_dict[parameter_tensor_name]
        pulses_ids = data_batch_dict["x_pulses_ids"][:, :3]

        parameters = self.add_parameter_indexing(parameters)
        source_parameters = self.get_model_parameters(parameters)

        # check if time exclusions exist
        tensors = self.data_trafo.data["tensors"]
        if (
            "x_time_exclusions" in tensors.names
            and tensors.list[tensors.get_index("x_time_exclusions")].exists
        ):
            time_exclusions_exist = True
        else:
            time_exclusions_exist = False

        # -----------------------------------------------
        # get concrete functions of base sources.
        # That way tracing only needs to be applied once.
        # -----------------------------------------------
        func_cache = ConcreteFunctionCache(
            model_parameters=source_parameters,
            sub_components=self.sub_components,
            data_batch_dict=data_batch_dict,
            is_training=is_training,
        )
        # -----------------------------------------------

        dom_charges = None
        dom_charges_variance = None
        dom_cdf_exclusion_sum = None
        all_models_have_cdf_values = True
        pulse_pdf = None
        pulse_cdf = None
        nested_results = {}
        for name, base in sorted(
            self._untracked_data["models_mapping"].items()
        ):

            # get the base source
            sub_component = self.sub_components[base]

            # get input parameters for Source i
            parameters_i = source_parameters[name]
            parameters_i = sub_component.add_parameter_indexing(parameters_i)

            # Get expected DOM charge and Likelihood evaluations for source i
            data_batch_dict_i = {"x_parameters": parameters_i}
            for key, values in data_batch_dict.items():
                if key != "x_parameters":
                    data_batch_dict_i[key] = values

            result_tensors_i = func_cache.get_or_add_tf_func(
                model_name=name,
                base_name=base,
                func_name="get_tensors",
            )(data_batch_dict_i)
            nested_results[name] = result_tensors_i

            dom_charges_i = result_tensors_i["dom_charges"]
            dom_charges_variance_i = result_tensors_i["dom_charges_variance"]
            pulse_pdf_i = result_tensors_i["pulse_pdf"]
            if "pulse_cdf" in result_tensors_i:
                pulse_cdf_i = result_tensors_i["pulse_cdf"]
            else:
                all_models_have_cdf_values = False

            if dom_charges_i.shape[1:] != [86, 60, 1]:
                msg = "DOM charges of source {!r} ({!r}) have an unexpected "
                msg += "shape {!r}."
                raise ValueError(msg.format(name, base, dom_charges_i.shape))

            if dom_charges_variance_i.shape[1:] != [86, 60, 1]:
                msg = "DOM charge variances of source {!r} ({!r}) have an "
                msg += "unexpected shape {!r}."
                raise ValueError(msg.format(name, base, dom_charges_i.shape))

            if time_exclusions_exist:
                dom_cdf_exclusion_sum_i = result_tensors_i[
                    "dom_cdf_exclusion_sum"
                ]
                if dom_cdf_exclusion_sum_i.shape[1:] != [86, 60, 1]:
                    msg = "DOM exclusions of source {!r} ({!r}) have an  "
                    msg += "unexpected shape {!r}."
                    raise ValueError(
                        msg.format(name, base, dom_cdf_exclusion_sum_i.shape)
                    )

                # undo re-normalization of PDF for the individual source at
                # a specific DOM. We will need to re-normalize, once everything
                # from all sources is there at a particular DOM.
                # Shape: [n_pulses]
                pulse_cdf_exclusion = tf.gather_nd(
                    tf.squeeze(dom_cdf_exclusion_sum_i, axis=-1), pulses_ids
                )
                pulse_pdf_i *= 1.0 - pulse_cdf_exclusion + self.epsilon
                if all_models_have_cdf_values:
                    pulse_cdf_i *= 1.0 - pulse_cdf_exclusion + self.epsilon

            # accumulate charge
            # (assumes that sources are linear and independent)
            if dom_charges is None:
                dom_charges = dom_charges_i
                dom_charges_variance = dom_charges_variance_i
                if time_exclusions_exist:
                    dom_cdf_exclusion_sum = (
                        dom_cdf_exclusion_sum_i * dom_charges_i
                    )
            else:
                dom_charges += dom_charges_i
                dom_charges_variance += dom_charges_variance_i
                if time_exclusions_exist:
                    dom_cdf_exclusion_sum += (
                        dom_cdf_exclusion_sum_i * dom_charges_i
                    )

            # accumulate likelihood values
            # (reweight by fraction of charge of source i vs total DOM charge)
            # Shape: [n_pulses]
            pulse_weight_i = tf.gather_nd(
                tf.squeeze(dom_charges_i, axis=3), pulses_ids
            )
            if pulse_pdf is None:
                pulse_pdf = pulse_pdf_i * pulse_weight_i
            else:
                pulse_pdf += pulse_pdf_i * pulse_weight_i

            if all_models_have_cdf_values:
                if pulse_cdf is None:
                    pulse_cdf = pulse_cdf_i * pulse_weight_i
                else:
                    pulse_cdf += pulse_cdf_i * pulse_weight_i

        # normalize pulse_pdf values: divide by total charge at DOM
        pulse_weight_total = tf.gather_nd(
            tf.squeeze(dom_charges, axis=3), pulses_ids
        )

        pulse_pdf /= pulse_weight_total + self.epsilon
        if all_models_have_cdf_values:
            pulse_cdf /= pulse_weight_total + self.epsilon

        result_tensors = {
            "dom_charges": dom_charges,
            "dom_charges_variance": dom_charges_variance,
            "pulse_pdf": pulse_pdf,
            "nested_results": nested_results,
        }
        if all_models_have_cdf_values:
            result_tensors["pulse_cdf"] = pulse_cdf

        # normalize time exclusion sum: divide by total charge at DOM
        if time_exclusions_exist:
            dom_cdf_exclusion_sum = tf_helpers.safe_cdf_clip(
                dom_cdf_exclusion_sum / (dom_charges + self.epsilon)
            )

            result_tensors["dom_cdf_exclusion_sum"] = dom_cdf_exclusion_sum

            # Also re-normalize PDF for exclusions if present
            pulse_cdf_exclusion = tf.gather_nd(
                tf.squeeze(dom_cdf_exclusion_sum, axis=-1), pulses_ids
            )
            result_tensors["pulse_pdf"] /= (
                1.0 - pulse_cdf_exclusion + self.epsilon
            )

            if all_models_have_cdf_values:
                result_tensors["pulse_cdf"] /= (
                    1.0 - pulse_cdf_exclusion + self.epsilon
                )

        # ensure proper CDF ranges
        if "pulse_cdf" in result_tensors:
            result_tensors["pulse_cdf"] = tf_helpers.safe_cdf_clip(
                result_tensors["pulse_cdf"]
            )

        return result_tensors

    def _flatten_nested_results(self, result_tensors, parent=None):
        """Gather nested result tensors and return as flattened dictionary.

        Parameters
        ----------
        result_tensors : dict of tf.tensor
            The dictionary of output tensors as obtained from `get_tensors`.
        parent : str, optional
            The name of the parent Multi-Source object.
            None, if no parent exists, i.e. this is the root Multi-Source
            object.

        Returns
        -------
        dict of tf.tensor
            A dictionary of the flattened results and models:
            {
                model_name: (base_source, result_tensors_i),
            }
        """
        flattened_results = {}
        for name, result_tensors_i in result_tensors["nested_results"].items():

            base_name = self._untracked_data["models_mapping"][name]
            base_source = self.sub_components[base_name]

            if "nested_results" in result_tensors_i:
                # found another Multi-Source, so keep going
                flattened_results_i = base_source._flatten_nested_results(
                    result_tensors_i, parent=name
                )
                flattened_results.update(flattened_results_i)
            else:
                # found a base source
                values = (base_source, result_tensors_i)
                if parent is None:
                    flattened_results[name] = values
                else:
                    flattened_results["{}/{}".format(parent, name)] = values

        return flattened_results

    def cdf(
        self,
        x,
        result_tensors,
        output_nested_pdfs=False,
        tw_exclusions=None,
        tw_exclusions_ids=None,
        strings=slice(None),
        doms=slice(None),
        **kwargs
    ):
        """Compute CDF values at x for given result_tensors

        This is a numpy, i.e. not tensorflow, method to compute the PDF based
        on a provided `result_tensors`. This can be used to investigate
        the generated PDFs.

        Note: this function only works for sources that use asymmetric
        Gaussians to parameterize the PDF. The latent values of the AG
        must be included in the `result_tensors`.

        Note: the PDF does not set values inside excluded time windows to zero,
        but it does adjust the normalization. It is assumed that pulses will
        already be masked before evaluated by Event-Generator. Therefore, an
        extra check for exclusions is not performed due to performance issues.

        Parameters
        ----------
        x : array_like
            The times in ns at which to evaluate the result tensors.
            Shape: () or [n_points]
        result_tensors : dict of tf.tensor
            The dictionary of output tensors as obtained from `get_tensors`.
        output_nested_pdfs : bool, optional
            If True, the PDFs of the nested sources will be returned as a
            dictionary:
            {
                # str: ([n_events, 86, 60, 1], [n_events, 86, 60, n_points])
                source_name: (multi_source_fraction, pdf_values),
            }
            Note: that the PDFs need to be multiplied by
            `multi_source_fraction` in order to obtain the mixture model for
            the Multi-Source.
        tw_exclusions : list of list, optional
            Optionally, time window exclusions may be applied. If these are
            provided, both `tw_exclusions` and `tw_exclusions_ids` must be set.
            Note: the event-generator does not internally modify the PDF
            and sets it to zero when in an exclusion. It is assumed that pulses
            are already masked. This reduces computation costs.
            The time window exclusions are defined as a list of
            [(t1_start, t1_stop), ..., (tn_start, tn_stop)]
            Shape: [n_exclusions, 2]
        tw_exclusions_ids : list of list, optional
            Optionally, time window exclusions may be applied. If these are
            provided, both `tw_exclusions` and `tw_exclusions_ids` must be set.
            Note: the event-generator does not internally modify the PDF
            and sets it to zero when in an exclusion. It is assumed that pulses
            are already masked. This reduces computation costs.
            The time window exclusion ids define to which event and DOM the
            time exclusions `tw_exclusions` belong to. They are defined
            as a list of:
            [(event1, string1, dom1), ..., (eventN, stringN, domN)]
            Shape: [n_exclusions, 3]
        strings : list of int, optional
            The strings to slice the PDF for.
            If None, all strings are used.
            Shape: [n_strings]
        doms : list of int, optional
            The doms to slice the PDF for.
            If None, all doms are used.
            Shape: [n_doms]
        **kwargs
            Keyword arguments.

        Returns
        -------
        array_like
            The CDF values at times x for the given event hypothesis and
            exclusions that were used to compute `result_tensors`.
            Shape: [n_events, 86, 60, n_points] if all DOMs returned
            Shape: [n_events, n_strings, n_doms] if DOMs specified
        dict [optional]
            A dictionary with the CDFs of the nested sources. See description
            of `output_nested_pdfs`.
        """
        # dict: {model_name: (base_source, result_tensors_i)}
        flattened_results = self._flatten_nested_results(result_tensors)

        nested_cdfs = {}
        cdf_values = None
        dom_charges = result_tensors["dom_charges"].numpy()[:, strings, doms]
        for name, (base_source, result_tensors_i) in sorted(
            flattened_results.items()
        ):

            # shape: [n_events, n_strings, n_doms, n_points]
            cdf_values_i = base_source.cdf(
                x,
                result_tensors_i,
                tw_exclusions=tw_exclusions,
                tw_exclusions_ids=tw_exclusions_ids,
                strings=strings,
                doms=doms,
            )

            # shape: [n_events, n_strings, n_doms, 1]
            dom_charges_i = result_tensors_i["dom_charges"].numpy()[
                :, strings, doms
            ]

            if cdf_values is None:
                cdf_values = cdf_values_i * dom_charges_i
            else:
                cdf_values += cdf_values_i * dom_charges_i

            if output_nested_pdfs:
                multi_source_fraction = dom_charges_i / dom_charges
                nested_cdfs[name] = (multi_source_fraction, cdf_values_i)

        cdf_values /= dom_charges

        if output_nested_pdfs:
            return cdf_values, nested_cdfs
        else:
            return cdf_values

    def pdf(
        self,
        x,
        result_tensors,
        output_nested_pdfs=False,
        tw_exclusions=None,
        tw_exclusions_ids=None,
        strings=slice(None),
        doms=slice(None),
        **kwargs
    ):
        """Compute PDF values at x for given result_tensors

        This is a numpy, i.e. not tensorflow, method to compute the PDF based
        on a provided `result_tensors`. This can be used to investigate
        the generated PDFs.

        Note: this function only works for sources that use asymmetric
        Gaussians to parameterize the PDF. The latent values of the AG
        must be included in the `result_tensors`.

        Note: the PDF does not set values inside excluded time windows to zero,
        but it does adjust the normalization. It is assumed that pulses will
        already be masked before evaluated by Event-Generator. Therefore, an
        extra check for exclusions is not performed due to performance issues.

        Parameters
        ----------
        x : array_like
            The times in ns at which to evaluate the result tensors.
            Shape: () or [n_points]
        result_tensors : dict of tf.tensor
            The dictionary of output tensors as obtained from `get_tensors`.
        output_nested_pdfs : bool, optional
            If True, the PDFs of the nested sources will be returned as a
            dictionary:
            {
                # str: ([n_events, 86, 60, 1], [n_events, 86, 60, n_points])
                source_name: (multi_source_fraction, pdf_values),
            }
            Note: that the PDFs need to be multiplied by
            `multi_source_fraction` in order to obtain the mixture model for
            the Multi-Source.
        tw_exclusions : list of list, optional
            Optionally, time window exclusions may be applied. If these are
            provided, both `tw_exclusions` and `tw_exclusions_ids` must be set.
            Note: the event-generator does not internally modify the PDF
            and sets it to zero when in an exclusion. It is assumed that pulses
            are already masked. This reduces computation costs.
            The time window exclusions are defined as a list of
            [(t1_start, t1_stop), ..., (tn_start, tn_stop)]
            Shape: [n_exclusions, 2]
        tw_exclusions_ids : list of list, optional
            Optionally, time window exclusions may be applied. If these are
            provided, both `tw_exclusions` and `tw_exclusions_ids` must be set.
            Note: the event-generator does not internally modify the PDF
            and sets it to zero when in an exclusion. It is assumed that pulses
            are already masked. This reduces computation costs.
            The time window exclusion ids define to which event and DOM the
            time exclusions `tw_exclusions` belong to. They are defined
            as a list of:
            [(event1, string1, dom1), ..., (eventN, stringN, domN)]
            Shape: [n_exclusions, 3]
        strings : list of int, optional
            The strings to slice the PDF for.
            If None, all strings are used.
            Shape: [n_strings]
        doms : list of int, optional
            The doms to slice the PDF for.
            If None, all doms are used.
            Shape: [n_doms]
        **kwargs
            Keyword arguments.

        Returns
        -------
        array_like
            The PDF values at times x for the given event hypothesis and
            exclusions that were used to compute `result_tensors`.
            Shape: [n_events, 86, 60, n_points] if all DOMs returned
            Shape: [n_events, n_strings, n_doms] if DOMs specified
        dict [optional]
            A dictionary with the PDFs of the nested sources. See description
            of `output_nested_pdfs`.
        """
        # dict: {model_name: (base_source, result_tensors_i)}
        flattened_results = self._flatten_nested_results(result_tensors)

        nested_pdfs = {}
        pdf_values = None
        dom_charges = result_tensors["dom_charges"].numpy()[:, strings, doms]
        for name, (base_source, result_tensors_i) in sorted(
            flattened_results.items()
        ):

            # shape: [n_events, n_strings, n_doms, n_points]
            pdf_values_i = base_source.pdf(
                x,
                result_tensors_i,
                tw_exclusions=tw_exclusions,
                tw_exclusions_ids=tw_exclusions_ids,
                strings=strings,
                doms=doms,
            )

            # shape: [n_events, n_strings, n_doms, 1]
            dom_charges_i = result_tensors_i["dom_charges"].numpy()[
                :, strings, doms
            ]

            if pdf_values is None:
                pdf_values = pdf_values_i * dom_charges_i
            else:
                pdf_values += pdf_values_i * dom_charges_i

            if output_nested_pdfs:
                multi_source_fraction = dom_charges_i / dom_charges
                nested_pdfs[name] = (multi_source_fraction, pdf_values_i)

        pdf_values /= dom_charges

        if output_nested_pdfs:
            return pdf_values, nested_pdfs
        else:
            return pdf_values
