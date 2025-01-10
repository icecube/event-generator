import numpy as np
import tensorflow as tf

from egenerator import misc
from egenerator.model.base import Model
from egenerator.manager.component import Configuration


class Source(Model):
    """Defines base class for an unbinned Source.

    This is an abstract class for an unbinned source. Unbinned in this context,
    means that an unbinned likelihood is used for the measured pulses.

    A derived class must implement
    ------------------------------
    _build_architecture(self, config, name=name):
        builds architecture and creates all model weights
        returns parameter_names

    get_tensors(self, data_batch_dict, is_training, parameter_tensor_name):
        The data_batch_dict is a dictionary of named inputs. This dictionary
        must contain at least the following keys:
            parameters, pulses, pulses_ids
        Parameters are the hypothesis tensor of the source with
        shape [-1, n_params]. The get_tensors method must compute all tensors
        that are to be used in later steps. It returns these as a dictionary
        of output tensors.

    Attributes
    ----------
    data_trafo : DataTrafo
        The data transformation object.

    name : str
        The name of the source.

    parameter_names : list of str
        The names of the n_params number of parameters.
    """

    @property
    def time_unit_in_ns(self):
        return 1000.0

    @property
    def data_trafo(self):
        if (
            self.sub_components is not None
            and "data_trafo" in self.sub_components
        ):
            return self.sub_components["data_trafo"]
        else:
            return None

    @property
    def decoder(self):
        if (
            self.sub_components is not None
            and "decoder" in self.sub_components
        ):
            return self.sub_components["decoder"]
        else:
            return None

    @property
    def decoder_charge(self):
        if (
            self.sub_components is not None
            and "decoder_charge" in self.sub_components
        ):
            return self.sub_components["decoder_charge"]
        else:
            return None

    def _configure_derived_class(
        self,
        config,
        data_trafo,
        decoder=None,
        decoder_charge=None,
        name=None,
    ):
        """Setup and configure the Source's architecture.

        After this function call, the sources's architecture (weights) must
        be fully defined and may not change again afterwards.

        Parameters
        ----------
        config : dict
            A dictionary of settings which is used to set up the model
            architecture and weights.
        data_trafo : DataTrafo
            A data trafo object.
        decoder : LatentToPDFDecoder, optional
            The decoder object. This is an optional object that is
            used to decode the latent variables into a PDF.
        decoder_charge : LatentToPDFDecoder, optional
            The decoder object for the charge. This is an optional
            object that is used to decode the latent variables
            into a PDF for the charge expectation.
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
            name = self.__class__.__name__

        # # collect all tensorflow variables before creation
        # variables_before = set([
        #     v.ref() for v in tf.compat.v1.global_variables()])

        # build architecture: create and save model weights
        # returns parameter_names
        parameter_names = self._build_architecture(config, name=name)

        # # collect all tensorflow variables after creation and match
        # variables_after = set([
        #     v.ref() for v in tf.compat.v1.global_variables()])

        # set_diff = variables_after - variables_before
        # model_variables = set([v.ref() for v in self.variables])
        # new_unaccounted_variables = set_diff - model_variables
        # if len(new_unaccounted_variables) > 0:
        #     msg = 'Found new variables that are not part of the tf.Module: {}'
        #     raise ValueError(msg.format(new_unaccounted_variables))

        # get names of parameters
        self._untracked_data["name"] = name
        self._set_parameter_names(parameter_names)

        # create sub components
        sub_components = {"data_trafo": data_trafo}
        if decoder is not None:
            sub_components["decoder"] = decoder
            settings = dict(config=config)
        else:
            # add empty decoder to config to keep track of it
            settings = dict(config=config, decoder=None)
        if decoder_charge is not None:
            sub_components["decoder_charge"] = decoder_charge
            settings["decoder_charge"] = decoder_charge
        else:
            # add empty decoder to config to keep track of it
            settings["decoder_charge"] = None

        # create configuration object
        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings=settings,
            mutable_settings=dict(name=name),
        )

        return configuration, {}, sub_components

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
            A list of parameter names. These parameters fully describe the
            source hypothesis. The model expects the hypothesis tensor input
            to be in the same order as this returned list.
        """
        self.assert_configured(False)
        raise NotImplementedError()

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
        data_batch_dict: dict of tf.Tensor
            parameters : tf.Tensor
                A tensor which describes the input parameters of the source.
                This fully defines the source hypothesis. The tensor is of
                shape [-1, n_params] and the last dimension must match the
                order of the parameter names (self.parameter_names),
            pulses : tf.Tensor
                The input pulses (charge, time) of all events in a batch.
                Shape: [-1, 2]
            pulses_ids : tf.Tensor
                The pulse indices (batch_index, string, dom) of all pulses in
                the batch of events.
                Shape: [-1, 3]
        is_training : bool, optional
            Indicates whether currently in training or inference mode.
            Must be provided if batch normalisation is used.
            True: in training mode
            False: inference mode.
        parameter_tensor_name : str, optional
            The name of the parameter tensor to use. Default: 'x_parameters'

        Raises
        ------
        NotImplementedError
            Description

        Returns
        -------
        dict of tf.Tensor
            A dictionary of output tensors.
            This  dictionary must at least contain:

                'dom_charges':
                    The predicted charge at each DOM
                    Shape: [n_events, 86, 60, 1]
                'pulse_pdf':
                    The likelihood evaluated for each pulse
                    Shape: [n_pulses]
                'time_offsets':
                    The global time offsets for each event.
                    Shape: [n_events]

            Other relevant optional tensors are:
                'latent_vars_time':
                    Shape: [n_events, 86, 60, n_latent]
                'latent_vars_charge':
                    Shape: [n_events, 86, 60, n_charge]
                'time_offsets_per_dom':
                    The time offsets per DOM (includes global offset).
                    Shape: [n_events, 86, 60, n_components]
                'dom_cdf_exclusion':
                    Shape: [n_events, 86, 60]
                'pulse_cdf':
                    Shape: [n_pulses]
        """
        self.assert_configured(True)
        raise NotImplementedError()

    def _select_slice(self, values, strings, doms):
        """Select subset of strings and doms from tensor

        Parameters
        ----------
        values : np.array
            The varlues to slice.
            Shape: [n_events, 86, 60, ...]
        strings : list of int
            The strings to slice the tensor for.
            If None, all strings are used.
            Shape: [n_strings]
        doms : list of int
            The doms to slice the tensor for.
            If None, all doms are used.
            Shape: [n_doms]

        Returns
        -------
        np.array
            The sliced values.
            Shape: [n_events, n_strings, n_doms, ...]
        """
        # check if strings in a slice
        if not isinstance(strings, slice):
            strings = np.atleast_1d(strings)

        values = values[:, strings, ...]

        if not isinstance(doms, slice):
            doms = np.atleast_1d(doms)

        values = values[:, :, doms, ...]
        return values

    def cdf(
        self,
        x,
        result_tensors,
        tw_exclusions=None,
        tw_exclusions_ids=None,
        strings=slice(None),
        doms=slice(None),
        **kwargs
    ):
        """Compute CDF values at x for given result_tensors

        This is a numpy, i.e. not tensorflow, method to compute the CDF based
        on a provided `result_tensors`. This can be used to investigate
        the generated PDFs.

        Note: the PDF does not set values inside excluded time windows to zero,
        but it does adjust the normalization. It is assumed that pulses will
        already be masked before evaluated by Event-Generator. Therefore, an
        extra check for exclusions is not performed due to performance issues.

        Important: if time window exclusions are provided, the exact same
        exclusions must have been used to compute the `result_tensors`.
        If this is not the case, the CDF values will be incorrect!

        Parameters
        ----------
        x : array_like
            The times in ns at which to evaluate the result tensors.
            Shape: () or [n_points]
        result_tensors : dict of tf.tensor
            The dictionary of output tensors as obtained from `get_tensors`.
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

        Raises
        ------
        NotImplementedError
            If asymmetric Gaussian latent variables are not present in
            `result_tensors` dictionary.
        """
        x_orig = np.atleast_1d(x)
        assert len(x_orig.shape) == 1, x_orig.shape
        n_points = len(x_orig)

        # shape: [1, 1, 1, n_points, 1]
        x = np.reshape(x_orig, (1, 1, 1, -1, 1))

        if result_tensors["time_offsets"] is not None:
            # shape: [n_events]
            t_offsets = result_tensors["time_offsets"].numpy()
            # shape: [n_events, 1, 1, 1, 1]
            t_offsets = np.reshape(t_offsets, [-1, 1, 1, 1, 1])
            # shape: [n_events, 86, 60, 1, n_components_total]
            #       = [n_events, 1, 1, 1, 1]
            #           + [n_events, 86, 60, 1, n_components_total]
            t_offsets = t_offsets + np.reshape(
                result_tensors["time_offsets_per_dom"].numpy(),
                [-1, 86, 60, 1, self.decoder.n_components_total],
            )
            # shape: [n_events, 86, 60, n_points, n_components_total]
            x = x - t_offsets
        else:
            t_offsets = 0.0

        # internally we are working with different time units
        x = x / self.time_unit_in_ns

        # select subset of strings and doms
        # shape: [n_events, n_strings, n_doms, n_points]
        x = self._select_slice(x, strings, doms)
        # shape: [n_events, 86, 60, n_latent]
        latent_vars_time = result_tensors["latent_vars_time"].numpy()
        # shape: [n_events, n_strings, n_doms, 1, n_latent]
        latent_vars_time = self._select_slice(latent_vars_time, strings, doms)[
            :, :, :, np.newaxis, :
        ]

        # evaluate the PDF
        # shape: [n_events, n_strings, n_doms, n_points]
        cdf_values = self.decoder.cdf(
            x=x,
            latent_vars=latent_vars_time,
            reduce_components=True,
        ).numpy()

        # uniformly scale up cdf values due to excluded regions
        if "dom_cdf_exclusion" in result_tensors:

            # shape: [n_events, n_strings, n_doms, 1]
            dom_cdf_exclusion = self._select_slice(
                result_tensors["dom_cdf_exclusion"].numpy(), strings, doms
            )[:, :, :, np.newaxis]

            # shape: [n_events, n_strings, n_doms, 1]
            cdf_values /= 1.0 - dom_cdf_exclusion + self.epsilon

        # apply time window exclusions:
        if tw_exclusions is not None:
            assert tw_exclusions_ids is not None, "Both tw and ids needed!"

            # # The helper function expects the pulses to have to
            # # values in second dimension: (charge, time)
            # # We will create dummy charge values for the pulses
            # x_pulses = np.tile(x_orig[:, np.newaxis], [1, 2])
            # assert x_pulses.shape == (n_points, 2)

            # cdf_excluded = tf_helpers.get_prior_pulse_cdf_exclusion(
            #     x_pulses=x_pulses,
            #     x_pulses_ids=pulses_ids,
            #     x_time_exclusions=x_time_exclusions,
            #     x_time_exclusions_ids=x_time_exclusions_ids,
            #     tw_cdf_exclusion=tw_cdf_exclusion,
            # )

            for tw, ids in zip(tw_exclusions, tw_exclusions_ids):

                # get indices in reduced subset of strings and doms
                if doms != slice(None):
                    if isinstance(doms, (list, tuple)):
                        dom_sub_id = np.where(ids[2] == doms)
                    else:
                        raise NotImplementedError(
                            "Only list or tuple supported for doms, "
                            f"not {type(doms)}"
                        )
                if strings != slice(None):
                    if isinstance(strings, (list, tuple)):
                        string_sub_id = np.where(ids[1] == strings)
                    else:
                        raise NotImplementedError(
                            "Only list or tuple supported for strings, "
                            f"not {type(strings)}"
                        )

                # get time points after exclusion window begin
                t_after_start = x_orig >= tw[0]
                t_before_stop = x_orig <= tw[1]
                t_eval = np.zeros([n_points, 2])
                t_eval[:, 0] = np.array(x_orig)
                t_eval[:, 1] = np.array(x_orig)
                t_eval[t_after_start, 0] = tw[0]
                t_eval[t_after_start, 1] = np.where(
                    t_before_stop[t_after_start],
                    x_orig[t_after_start],
                    tw[1],
                )

                # t_eval now defines the ranges of excluded region for each
                # time point. We now need to subtract the CDF in this region
                # shape: [1, 1, 1, n_points, 2, 1]
                t_eval = np.reshape(t_eval, (1, 1, 1, -1, 2, 1))

                # shape: [n_events, 86, 60, n_points, 2, n_components_total]
                #       = [1, 1, 1, n_points, 2, 1] -
                #           [n_events, 86, 60, 1, 1, n_components_total]
                t_eval_trafo = (
                    t_eval
                    - np.reshape(
                        t_offsets,
                        [-1, 86, 60, 1, 1, self.decoder.n_components_total],
                    )
                ) / self.time_unit_in_ns

                # select subset of strings and doms based on exclusion ids
                # shape: [n_points, 2, n_components_total]
                t_eval_trafo = t_eval_trafo[ids[0], ids[1], ids[2], ...]
                # shape: [n_events, 86, 60, n_latent]
                latent_vars_time = result_tensors["latent_vars_time"].numpy()
                # shape: [1, n_latent]
                latent_vars_time = latent_vars_time[
                    ids[0], ids[1], ids[2], np.newaxis, :
                ]

                # shape: [n_points, 2]
                cdf_exclusion_values = self.decoder.cdf(
                    x=t_eval_trafo,
                    latent_vars=latent_vars_time,
                    reduce_components=True,
                ).numpy()

                # shape: [n_points]
                cdf_excluded = (
                    cdf_exclusion_values[..., 1] - cdf_exclusion_values[..., 0]
                )

                # update cdf values
                cdf_values[ids[0], dom_sub_id, string_sub_id] -= cdf_excluded

            eps = 1e-6
            if (cdf_values < 0 - eps).any():
                self._logger.warning(
                    "CDF values below zero: {}".format(
                        cdf_values[cdf_values < 0 - eps]
                    )
                )
            if (cdf_values > 1 + eps).any():
                self._logger.warning(
                    "CDF values above one: {}".format(
                        cdf_values[cdf_values > 1 + eps]
                    )
                )

        return cdf_values

    def pdf(
        self,
        x,
        result_tensors,
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

        Note: the PDF does not set values inside excluded time windows to zero,
        but it does adjust the normalization. It is assumed that pulses will
        already be masked before evaluated by Event-Generator. Therefore, an
        extra check for exclusions is not performed due to performance issues.

        Important: if time window exclusions are provided, the exact same
        exclusions must have been used to compute the `result_tensors`.
        If this is not the case, the PDF values will be incorrect!

        Parameters
        ----------
        x : array_like
            The times in ns at which to evaluate the result tensors.
            Shape: () or [n_points]
        result_tensors : dict of tf.tensor
            The dictionary of output tensors as obtained from `get_tensors`.
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
            Shape: [n_events, n_strings, n_doms, n_points] if DOMs specified

        Raises
        ------
        NotImplementedError
            If asymmetric Gaussian latent variables are not present in
            `result_tensors` dictionary.
        """
        x_orig = np.atleast_1d(x)
        assert len(x_orig.shape) == 1, x_orig.shape

        # shape: [1, 1, 1, n_points, 1]
        x = np.reshape(x_orig, (1, 1, 1, -1, 1))

        if result_tensors["time_offsets"] is not None:
            # shape: [n_events]
            t_offsets = result_tensors["time_offsets"].numpy()
            # shape: [n_events, 1, 1, 1, 1]
            t_offsets = np.reshape(t_offsets, [-1, 1, 1, 1, 1])
            # shape: [n_events, 86, 60, 1, n_components_total]
            #       = [n_events, 1, 1, 1, 1]
            #           + [n_events, 86, 60, 1, n_components_total]
            t_offsets = t_offsets + np.reshape(
                result_tensors["time_offsets_per_dom"].numpy(),
                [-1, 86, 60, 1, self.decoder.n_components_total],
            )
            # shape: [n_events, 86, 60, n_points, n_components_total]
            x = x - t_offsets

        # internally we are working with different time units
        x = x / self.time_unit_in_ns

        # select subset of strings and doms
        # shape: [n_events, n_strings, n_doms, n_points, n_components_total]
        x = self._select_slice(x, strings, doms)
        # shape: [n_events, 86, 60, 1, n_latent]
        latent_vars_time = self._select_slice(
            result_tensors["latent_vars_time"].numpy(), strings, doms
        )[:, :, :, np.newaxis, :]
        # evaluate the PDF
        # shape: [n_events, n_strings, n_doms, n_points]
        pdf_values = self.decoder.pdf(
            x=x,
            latent_vars=latent_vars_time,
            reduce_components=True,
        ).numpy()

        # uniformly scale up pdf values due to excluded regions
        if "dom_cdf_exclusion" in result_tensors:

            # shape: [n_events, n_strings, n_doms, 1]
            dom_cdf_exclusion = self._select_slice(
                result_tensors["dom_cdf_exclusion"].numpy(), strings, doms
            )[:, :, :, np.newaxis]

            # shape: [n_events, n_strings, n_doms, 1]
            pdf_values /= 1.0 - dom_cdf_exclusion + self.epsilon

        # apply time window exclusions:
        pdf_values = self._apply_pdf_time_window_exclusions(
            times=x_orig,
            pdf_values=pdf_values,
            tw_exclusions=tw_exclusions,
            tw_exclusions_ids=tw_exclusions_ids,
        )

        # invert back to PDF in ns
        pdf_values = pdf_values / self.time_unit_in_ns

        return pdf_values

    def _apply_pdf_time_window_exclusions(
        self, times, pdf_values, tw_exclusions, tw_exclusions_ids
    ):
        """Apply time window exclusions

        PDF values that correspond to excluded time windows are set to zero.
        Note: internally, the event-generator does not apply these exclusions.
        It is assumed that the pulses are already masked. However, the PDF
        is renormalized to account for the excluded regions.

        Parameters
        ----------
        times : array_like
            The times in ns at which to evaluate the result tensors.
            Shape: [n_points]
        pdf_values : array_like
            The PDF values at the specified times.
            Shape: [n_events, 86, 60, n_points]
        tw_exclusions : list of list, optional
            The time window exclusions to apply.
            These are defined as a list of:
            [(t1_start, t1_stop), ..., (tn_start, tn_stop)]
            Shape: [n_exclusions, 2]
        tw_exclusions_ids : list of list, optional
            The time window exclusion ids define to which event and DOM the
            time exclusions `tw_exclusions` belong to. They are defined
            as a list of:
            [(event1, string1, dom1), ..., (eventN, stringN, domN)]
            Shape: [n_exclusions, 3]

        Returns
        -------
        array_like
            The corrected PDF values with the time exclusions applied.
            Shape: [n_events, 86, 60, n_points]
        """
        if tw_exclusions is not None:
            assert tw_exclusions_ids is not None, "Both tw and ids needed!"

            mask_excluded = np.zeros_like(pdf_values, dtype=bool)
            for tw, ids in zip(tw_exclusions, tw_exclusions_ids):
                t_excluded = np.logical_and(times >= tw[0], times <= tw[1])

                mask_excluded[ids[0], ids[1], ids[2], t_excluded] = True
            pdf_values[mask_excluded] = -float("inf")

        return pdf_values

    # def _get_top_node(self):
    #     """Helper function to get the top

    #     Returns
    #     -------
    #     TYPE
    #         Description
    #     """
    #     parent_node = tf.constant(np.ones(shape=[1, self.n_parameters]))
    #     output = self._untracked_data['module'](parent_node)
    #     return _find_top_nodes(output)

    # def _find_top_nodes(self, tensor):
    #     if len(tensor.op.inputs) == 0:
    #         return set([tensor])

    #     tensor_list = set()
    #     for in_tensor in tensor.op.inputs:
    #         tensor_list = tensor_list.union(find_top_nodes(in_tensor))

    #     return tensor_list
