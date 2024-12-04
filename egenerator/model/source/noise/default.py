import tensorflow as tf
import numpy as np

from tfscripts.weights import new_weights

from egenerator.model.source.base import Source
from egenerator.utils import detector, tf_helpers


class DefaultNoiseModel(Source):

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

        # ---------------------------------------
        # Define input parameters of noise source
        # ---------------------------------------
        parameter_names = []

        # get weights for general scaling of mu and over-dispersion
        self._untracked_data["local_vars"] = new_weights(
            shape=[2],
            stddev=1e-5,
            float_precision=config["float_precision"],
            name="noise_scaling",
        )

        return parameter_names

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
        print("Applying Noise Model...")

        tensor_dict = {}

        # get time exclusions
        tensors = self.data_trafo.data["tensors"]
        if (
            "x_time_exclusions" in tensors.names
            and tensors.list[tensors.get_index("x_time_exclusions")].exists
        ):
            time_exclusions_exist = True

            # shape: [n_tw, 2]
            x_time_exclusions = data_batch_dict["x_time_exclusions"]

            # shape: [n_tw, 3]
            x_time_exclusions_ids = data_batch_dict["x_time_exclusions_ids"]
            x_time_excl_batch_id = x_time_exclusions_ids[:, 0]
        else:
            time_exclusions_exist = False
        print("\t Applying time exclusions:", time_exclusions_exist)

        # get parameters tensor dtype
        param_dtype_np = tensors[parameter_tensor_name].dtype_np

        # shape: [n_pulses, 3]
        pulses_ids = data_batch_dict["x_pulses_ids"][:, :3]

        # Shape: [n_pulses, 2] [charge, time]
        pulses = data_batch_dict["x_pulses"]

        # shape: [n_batch, 2]
        time_window = data_batch_dict["x_time_window"]

        # shape: [n_batch]
        livetime = time_window[:, 1] - time_window[:, 0]

        # shape: [n_batch, 1, 1, 1]
        livetime_exp = tf.reshape(
            time_window[:, 1] - time_window[:, 0], [-1, 1, 1, 1]
        )

        # compute the expected charge at each DOM based off of noise rate
        # shape: [1, 86, 60, 1]
        dom_noise_rates = tf.reshape(
            detector.dom_noise_rates.astype(param_dtype_np),
            shape=[1, 86, 60, 1],
        )

        # shape: [n_batch, 86, 60, 1]
        dom_charges = dom_noise_rates * livetime_exp

        # shape: [n_batch]
        dom_pdf_constant = 1.0 / livetime

        # ----------------------------
        # Apply time window exclusions
        # ----------------------------
        if time_exclusions_exist:

            # limit exclusion windows to read out window
            # shape: [n_tw, 2]
            t_min = tf.gather(time_window[:, 0], indices=x_time_excl_batch_id)
            t_max = tf.gather(time_window[:, 1], indices=x_time_excl_batch_id)
            tw_livetime = tf.gather(livetime, indices=x_time_excl_batch_id)
            tw_reduced = tf.clip_by_value(
                x_time_exclusions,
                tf.expand_dims(t_min, axis=-1),
                tf.expand_dims(t_max, axis=-1),
            )

            # now calculate exclusions cdf
            # shape: [n_tw]
            tw_cdf_exclusion = (
                tw_reduced[:, 1] - tw_reduced[:, 0]
            ) / tw_livetime

            # some safety checks to make sure we aren't clipping too much
            tw_cdf_exclusion = tf_helpers.safe_cdf_clip(tw_cdf_exclusion)

            # accumulate time window exclusions for each event
            # shape: [n_batch, 86, 60]
            dom_cdf_exclusion = tf.tensor_scatter_nd_add(
                tf.zeros_like(
                    tf.squeeze(data_batch_dict["x_dom_charge"], axis=3)
                ),
                indices=x_time_exclusions_ids,
                updates=tw_cdf_exclusion,
            )

            # some safety checks to make sure we aren't clipping too much
            dom_cdf_exclusion = tf_helpers.safe_cdf_clip(dom_cdf_exclusion)
        # ----------------------------

        # local scaling vars are initialized around zero with small std dev
        local_vars = tf.nn.elu(self._untracked_data["local_vars"]) + 1.01

        # scaling of expected noise hits: ensure positive values.
        # shape: [1, 1, 1]
        mean_scaling = tf.reshape(tf.nn.elu(local_vars[0]) + 1.01, [1, 1, 1])

        # scaling of uncertainty. shape: [1, 1, 1]
        # The over-dispersion parameterized by alpha must be greater zero
        # Var(x) = mu + alpha*mu**2
        dom_charges_alpha = tf.reshape(
            tf.nn.elu(local_vars[1] - 5) + 1.000001, [1, 1, 1]
        )

        # scale dom charge and uncertainty by learned scaling
        dom_charges = dom_charges * mean_scaling

        # scale by time exclusions
        if time_exclusions_exist:
            dom_charges *= (
                1.0 - dom_cdf_exclusion[..., tf.newaxis] + self.epsilon
            )

        # add small constant to make sure dom charges are > 0:
        dom_charges += self.epsilon

        # compute standard deviation
        # std = sqrt(var) = sqrt(mu + alpha*mu**2)
        dom_charges_variance = dom_charges + dom_charges_alpha * dom_charges**2
        dom_charges_unc = tf.sqrt(dom_charges_variance)

        # Compute Log Likelihood for pulses
        # PDF is a uniform distribution in the specified time window.
        # The time window is constructed such that every pulse is part of it
        # That means that every pulse of an event has the same likelihood.
        # shape: [n_pulses]
        pulse_pdf = tf.gather(dom_pdf_constant, indices=pulses_ids[:, 0])

        # compute pulse cdf values
        # shape: [n_pulses, 2]
        pulse_tw = tf.gather(time_window, indices=pulses_ids[:, 0])
        # shape: [n_pulses]
        pulse_cdf = (pulses[:, 1] - pulse_tw[:, 0]) / (
            pulse_tw[:, 1] - pulse_tw[:, 0]
        )

        # scale up pulse pdf by time exclusions if needed
        if time_exclusions_exist:
            # shape: [n_pulses]
            pulse_cdf_exclusion_total = tf.gather_nd(
                dom_cdf_exclusion, pulses_ids
            )

            # subtract excluded regions from cdf values
            # shape: [n_pulses]
            pulse_cdf_exclusion = tf_helpers.get_prior_pulse_cdf_exclusion(
                x_pulses=pulses,
                x_pulses_ids=pulses_ids,
                x_time_exclusions=x_time_exclusions,
                x_time_exclusions_ids=x_time_exclusions_ids,
                tw_cdf_exclusion=tw_cdf_exclusion,
            )
            pulse_cdf -= pulse_cdf_exclusion

            pulse_pdf /= 1.0 - pulse_cdf_exclusion_total + self.epsilon
            pulse_cdf /= 1.0 - pulse_cdf_exclusion_total + self.epsilon

        # add tensors to tensor dictionary
        tensor_dict["time_offsets"] = None
        tensor_dict["dom_charges"] = dom_charges
        tensor_dict["dom_charges_alpha"] = dom_charges_alpha
        tensor_dict["dom_charges_unc"] = dom_charges_unc
        tensor_dict["dom_charges_variance"] = dom_charges_variance
        tensor_dict["pdf_constant"] = dom_pdf_constant
        tensor_dict["pdf_time_window"] = time_window
        tensor_dict["pulse_pdf"] = pulse_pdf
        tensor_dict["pulse_cdf"] = pulse_cdf

        if time_exclusions_exist:
            tensor_dict["dom_cdf_exclusion"] = dom_cdf_exclusion
        # -------------------------------------------

        return tensor_dict

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

        This is a numpy, i.e. not tensorflow, method to compute the PDF based
        on a provided `result_tensors`. This can be used to investigate
        the generated PDFs.

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
            Shape: [n_events, n_strings, n_doms, n_points] if DOMs specified

        No Longer Raises
        ----------------
        NotImplementedError
            If asymmetric Gaussian latent variables are not present in
            `result_tensors` dictionary.
        """
        # shape: [n_points]
        x_orig = np.atleast_1d(x)
        assert len(x_orig.shape) == 1, x_orig.shape

        n_points = len(x_orig)
        # shape: [1, 1, 1, n_points]
        x = np.reshape(x_orig, [1, 1, 1, -1])

        # extract values
        # Shape: [n_events, 2]
        time_window = result_tensors["pdf_time_window"].numpy()
        # shape: [n_events, 1, 1, 1, 2]
        time_window = np.reshape(time_window, [-1, 1, 1, 1, 2])

        # shape: [n_events, 1, 1, 1]
        livetime = time_window[..., 1] - time_window[..., 0]

        # Shape: [n_events, 86, 60, 1]
        if "dom_cdf_exclusion" in result_tensors:
            dom_cdf_exclusion = result_tensors["dom_cdf_exclusion"].numpy()
            dom_cdf_exclusion = dom_cdf_exclusion[..., np.newaxis]
        else:
            dom_cdf_exclusion = np.zeros((len(time_window), 86, 60, 1))

        dom_cdf_exclusion = dom_cdf_exclusion[:, strings, doms]

        # shape: [n_events, 1, 1, n_points]
        t_end = np.where(
            x >= time_window[..., 1],
            time_window[..., 1],
            x,
        )

        # Shape: [n_events, n_strings, n_doms, n_points]
        cdf_values = (
            (t_end - time_window[..., 0])
            / livetime
            / (1.0 - dom_cdf_exclusion + self.epsilon)
        )

        # apply time window exclusions:
        if tw_exclusions is not None:
            assert tw_exclusions_ids is not None, "Both tw and ids needed!"

            for tw, ids in zip(tw_exclusions, tw_exclusions_ids):

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
                # Shape: [n_points, 2]
                t_eval = np.where(
                    t_eval > time_window[ids[0], 0, 0, 0, 1],
                    time_window[ids[0], 0, 0, 0, 1],
                    t_eval,
                )
                t_eval = np.where(
                    t_eval < time_window[ids[0], 0, 0, 0, 0],
                    time_window[ids[0], 0, 0, 0, 0],
                    t_eval,
                )
                # Shape: [n_points]
                cdf_excluded = (
                    (t_eval[:, 1] - t_eval[:, 0])
                    / livetime[ids[0], 0, 0, 0]
                    / (
                        1.0
                        - dom_cdf_exclusion[ids[0], ids[1], ids[2]]
                        + self.epsilon
                    )
                )

                # subtract excluded region
                cdf_values[ids[0], ids[1], ids[2]] -= cdf_excluded

            eps = 1e-3
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
            Shape: [n_events, n_strings, n_doms] if DOMs specified

        No Longer Raises
        ----------------
        NotImplementedError
            If asymmetric Gaussian latent variables are not present in
            `result_tensors` dictionary.
        """
        # shape: [n_points]
        x_orig = np.atleast_1d(x)
        assert len(x_orig.shape) == 1, x_orig.shape

        n_points = len(x_orig)
        # shape: [1, 1, 1, n_points]
        x = np.reshape(x_orig, [1, 1, 1, -1])

        # extract values

        # Shape: (n_events)
        pdf_constant = result_tensors["pdf_constant"].numpy()
        # shape: [n_events, 1, 1, 1]
        pdf_constant = np.reshape(pdf_constant, [-1, 1, 1, 1])

        # Shape: [n_events, 2]
        time_window = result_tensors["pdf_time_window"].numpy()
        # shape: [n_events, 1, 1, 1, 2]
        time_window = np.reshape(time_window, [-1, 1, 1, 1, 2])

        # Shape: [n_events, 86, 60, 1]
        if "dom_cdf_exclusion" in result_tensors:
            dom_cdf_exclusion = result_tensors["dom_cdf_exclusion"].numpy()
            dom_cdf_exclusion = dom_cdf_exclusion[..., np.newaxis]
        else:
            dom_cdf_exclusion = np.zeros((len(time_window), 86, 60, 1))

        dom_cdf_exclusion = dom_cdf_exclusion[:, strings, doms]

        # Shape: [n_events, n_strings, n_doms, 1]
        pdf_values = pdf_constant / (1.0 - dom_cdf_exclusion + self.epsilon)

        # Shape: [n_events, n_strings, n_doms, n_points]
        pdf_values = np.tile(pdf_values, reps=(1, 1, 1, n_points))

        pdf_values = np.where(
            # mask shape: [n_events, 1, 1, n_points]
            np.logical_and(x >= time_window[..., 0], x <= time_window[..., 1]),
            pdf_values,
            np.zeros_like(pdf_values),
        )

        # apply time window exclusions:
        pdf_values = self._apply_pdf_time_window_exclusions(
            times=x_orig,
            pdf_values=pdf_values,
            tw_exclusions=tw_exclusions,
            tw_exclusions_ids=tw_exclusions_ids,
        )

        return pdf_values
