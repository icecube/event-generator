import logging
import tensorflow as tf

from egenerator import misc
from egenerator.utils import basis_functions
from egenerator.manager.component import BaseComponent, Configuration
from egenerator.utils import tf_helpers


class DefaultLossModule(BaseComponent):
    """Default loss module that implements some standard loss functions.

    A loss component that is used to compute the loss. The component
    must provide a
    loss_module.get_loss(data_batch_dict, result_tensors, tensors,
                         parameter_tensor_name='x_parameters', **kwargs)
    method.
    """

    @property
    def loss_function(self):
        if (
            self.untracked_data is not None
            and "loss_function" in self.untracked_data
        ):
            return self.untracked_data["loss_function"]
        else:
            return None

    @property
    def epsilon(self):
        if self.configuration.config["config"]["float_precision"] == "float32":
            return 1e-7
        elif (
            self.configuration.config["config"]["float_precision"] == "float64"
        ):
            return 1e-15
        else:
            raise ValueError(
                "Invalid float precision: {}".format(
                    self.configuration.config["config"]["float_precision"]
                )
            )

    def __init__(self, logger=None):
        """Initializes LossModule object.

        Parameters
        ----------
        logger : logging.logger, optional
            A logging instance.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(DefaultLossModule, self).__init__(logger=self._logger)

    def _configure(self, config):
        """Configure the LossModule component instance.

        Parameters
        ----------
        config : dict
            Configuration settings of the LossModule object.
            Must contain:
                'float_precision': str
                    The float precision to use
                'loss_function_name': str
                    The name of the loss function to use.

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
        dict
            A dictionary of dependent sub components. This is a dictionary
            of sub components that need to be saved and loaded recursively
            when the component is saved and loaded.
            Return None if no dependent sub components exist.
        """

        # choose loss function
        self.untracked_data["loss_function"] = getattr(
            self, config["loss_function_name"]
        )

        # create configuration object
        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings=dict(config=config),
        )

        return configuration, {}, {}

    def get_loss(
        self,
        data_batch_dict,
        result_tensors,
        tensors,
        model,
        parameter_tensor_name="x_parameters",
        reduce_to_scalar=True,
        normalize_by_total_charge=False,
        sort_loss_terms=False,
        **kwargs
    ):
        """Get the scalar loss for a given data batch and result tensors.

        Parameters
        ----------
        data_batch_dict : dict of tf.Tensor
            parameters : tf.Tensor
                A tensor which describes the input parameters of the source.
                This fully defines the source hypothesis. The tensor is of
                shape [-1, n_params] and the last dimension must match the
                order of the parameter names (model.parameter_names),
            pulses : tf.Tensor
                The input pulses (charge, time) of all events in a batch.
                Shape: [-1, 2]
            pulses_ids : tf.Tensor
                The pulse indices (batch_index, string, dom, pulse_number)
                of all pulses in the batch of events.
                Shape: [-1, 4]
        result_tensors : dict of tf.Tensor
            A dictionary of output tensors.
            This  dictionary must at least contain:

                'dom_charges': the predicted charge at each DOM
                               Shape: [-1, 86, 60]
                'pulse_pdf': The likelihood evaluated for each pulse
                             Shape: [-1]
        tensors : DataTensorList
            The data tensor list describing the input data
        model : Model
            The model object used to calculate the result tensors.
        parameter_tensor_name : str, optional
            The name of the parameter tensor to use. Default: 'x_parameters'.
            This option is ignored here and has no effect.
        reduce_to_scalar : bool, optional
            If True, the individual terms of the log likelihood loss will be
            reduced (aggregated) to a scalar loss.
            If False, a list of tensors will be returned that contain the terms
            of the log likelihood. Note that each of the returned tensors may
            have a different shape.
        normalize_by_total_charge : bool, optional
            If True, the loss will be normalized (divided) by the total charge.
            This will make the loss of events with vastly different amounts of
            detected photons be more comparable.
        sort_loss_terms : bool, optional
            If true, the loss terms will be sorted and aggregated in three
            types of loss terms (this requires `reduce_to_scalar` == False):
                scalar: shape []
                    scalar loss for the whole batch of events
                event: shape [n_batch]
                    vector loss with one value per event
                dom: shape [n_batch, 86, 60]
                    tensor loss with one value for each DOM and event
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        tf.Tensor or list of tf.Tensor
            if `reduce_to_scalar` is True:
                Scalar loss
                Shape: []
            else:
                List of tensors defining the terms of the log likelihood
        """
        # sanity check
        if reduce_to_scalar and sort_loss_terms:
            raise ValueError(
                "Both sort_loss_terms and reduce_to_scalar are set to True. "
                "Sorting of loss terms is unnecessary when reducing to scalar"
            )

        # cast to specified float precision
        precision = self.configuration.config["config"]["float_precision"]
        data_batch_dict_cast = {}
        for key, value in data_batch_dict.items():
            if tf.is_tensor(value) and value.dtype in (
                tf.float16,
                tf.float32,
                tf.float64,
            ):
                data_batch_dict_cast[key] = tf.cast(value, precision)
            else:
                data_batch_dict_cast[key] = value

        result_tensors_cast = {}
        for key, value in result_tensors.items():
            if tf.is_tensor(value) and value.dtype in (
                tf.float16,
                tf.float32,
                tf.float64,
            ):
                result_tensors_cast[key] = tf.cast(value, precision)
            else:
                result_tensors_cast[key] = value

        loss_terms = self.loss_function(
            data_batch_dict=data_batch_dict_cast,
            result_tensors=result_tensors_cast,
            tensors=tensors,
            sort_loss_terms=sort_loss_terms,
        )

        # fill Nones with zeros of appropriate shape if sort_loss_terms
        if sort_loss_terms:
            assert len(loss_terms) == 3, loss_terms

            dom_tensor = data_batch_dict["x_dom_charge"][..., 0]
            if loss_terms[0] is None:
                loss_terms[0] = tf.zeros_like(dom_tensor[0, 0, 0])
            if loss_terms[1] is None:
                loss_terms[1] = tf.zeros_like(dom_tensor[:, 0, 0])
            if loss_terms[2] is None:
                loss_terms[2] = tf.zeros_like(dom_tensor)

        if normalize_by_total_charge:
            total_charge = tf.cast(
                tf.clip_by_value(
                    tf.reduce_sum(data_batch_dict["x_pulses"][:, 0]),
                    1,
                    float("inf"),
                ),
                dtype=precision,
            )
            loss_terms = [loss / total_charge for loss in loss_terms]

        if reduce_to_scalar:
            return tf.math.add_n(
                [tf.reduce_sum(loss_term) for loss_term in loss_terms]
            )
        else:
            return loss_terms

    def unbinned_extended_pulse_llh(
        self, data_batch_dict, result_tensors, tensors, sort_loss_terms
    ):
        """Unbinned extended poisson likelhood for data pulses.

        Pulses must *not* contain any pulses in excluded DOMs or excluded time
        windows. It is assumed that these pulses are already removed, e.g.
        the time pdf is calculated for all pulses.

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
        result_tensors : dict of tf.Tensor
            A dictionary of output tensors.
            This  dictionary must at least contain:

                'dom_charges': the predicted charge at each DOM
                               Shape: [-1, 86, 60]
                'pulse_pdf': The likelihood evaluated for each pulse
                             Shape: [-1]
        tensors : DataTensorList
            The data tensor list describing the input data
        sort_loss_terms : bool, optional
            If true, the loss terms will be sorted and aggregated in three
            types of loss terms (this requires `reduce_to_scalar` == False):
                scalar: shape []
                    scalar loss for the whole batch of events
                event: shape [n_batch]
                    vector loss with one value per event
                dom: shape [n_batch, 86, 60]
                    tensor loss with one value for each DOM and event

        Returns
        -------
        List of tf.tensor
            Poisson Likelihood.
            List of tensors defining the terms of the log likelihood

        Raises
        ------
        NotImplementedError
            Description
        """
        dtype = getattr(
            tf, self.configuration.config["config"]["float_precision"]
        )

        # shape: [n_pulses]
        pulse_charges = data_batch_dict["x_pulses"][:, 0]
        pulse_pdf_values = result_tensors["pulse_pdf"]

        # shape: [n_batch, 86, 60, 1]
        hits_true = data_batch_dict["x_dom_charge"]

        # shape: [n_batch, 86, 60]
        dom_charges_true = tf.squeeze(hits_true, axis=-1)
        dom_charges_pred = result_tensors["dom_charges"]

        # throw error if this is being used with time window exclusions
        # one needs to calculate cumulative pdf from exclusion window and
        # reduce the predicted charge by this factor
        if (
            "x_time_exclusions" in tensors.names
            and tensors.list[tensors.get_index("x_time_exclusions")].exists
        ):
            assert (
                "dom_cdf_exclusion" in result_tensors
            ), "Model must deal with time exclusions!"

        # mask out dom exclusions
        if (
            "x_dom_exclusions" in tensors.names
            and tensors.list[tensors.get_index("x_dom_exclusions")].exists
        ):
            mask_valid = tf.cast(
                tf.squeeze(data_batch_dict["x_dom_exclusions"], axis=-1),
                dtype=dtype,
            )
            dom_charges_true = dom_charges_true * mask_valid
            dom_charges_pred = dom_charges_pred * mask_valid

        # prevent log(zeros) issues
        pulse_log_pdf_values = tf_helpers.safe_log(pulse_pdf_values)

        # compute unbinned negative likelihood over pulse times with given
        # time pdf: -sum( charge_i * log(pdf_d(t_i)) )
        time_log_likelihood = -pulse_charges * pulse_log_pdf_values

        # get poisson likelihood over total charge at a DOM for extendended LLH
        llh_poisson = (
            dom_charges_pred
            - dom_charges_true * tf_helpers.safe_log(dom_charges_pred)
        )

        if sort_loss_terms:
            loss_doms = tf.tensor_scatter_nd_add(
                llh_poisson,
                indices=data_batch_dict["x_pulses_ids"][:, :3],
                updates=time_log_likelihood,
            )
            loss_terms = [None, None, loss_doms]
        else:
            loss_terms = [llh_poisson, time_log_likelihood]

        # Add normalization terms if desired
        # Note: these are irrelevant for the minimization, but will make loss
        # curves more meaningful
        if self.configuration.config["config"]["add_normalization_term"]:
            norm_doms = basis_functions.log_faculty(dom_charges_true)
            if sort_loss_terms:
                loss_terms[2] += norm_doms
            else:
                loss_terms.append(norm_doms)

        return loss_terms

    def unbinned_pulse_time_llh(
        self, data_batch_dict, result_tensors, tensors, sort_loss_terms
    ):
        """Unbinned Pulse Time likelhood.

        Pulses must *not* contain any pulses in excluded DOMs or excluded time
        windows. It is assumed that these pulses are already removed, e.g.
        the time pdf is calculated for all pulses.

        This is similar to `unbinned_extended_pulse_llh`. Major differences:
            - No additional likelihood terms for DOM charge

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
        result_tensors : dict of tf.Tensor
            A dictionary of output tensors.
            This  dictionary must at least contain:
                'pulse_pdf': The likelihood evaluated for each pulse
                             Shape: [-1]
        tensors : DataTensorList
            The data tensor list describing the input data
        sort_loss_terms : bool, optional
            If true, the loss terms will be sorted and aggregated in three
            types of loss terms (this requires `reduce_to_scalar` == False):
                scalar: shape []
                    scalar loss for the whole batch of events
                event: shape [n_batch]
                    vector loss with one value per event
                dom: shape [n_batch, 86, 60]
                    tensor loss with one value for each DOM and event

        Returns
        -------
        List of tf.tensor
            Poisson Likelihood.
            List of tensors defining the terms of the log likelihood

        Raises
        ------
        NotImplementedError
            Description
        """

        # shape: [n_pulses]
        pulse_charges = data_batch_dict["x_pulses"][:, 0]
        pulse_pdf_values = result_tensors["pulse_pdf"]

        # throw error if this is being used with time window exclusions
        # one needs to calculate cumulative pdf from exclusion window and
        # reduce the predicted charge by this factor
        if (
            "x_time_exclusions" in tensors.names
            and tensors.list[tensors.get_index("x_time_exclusions")].exists
        ):
            assert (
                "dom_cdf_exclusion" in result_tensors
            ), "Model must deal with time exclusions!"

        # prevent log(zeros) issues
        pulse_log_pdf_values = tf_helpers.safe_log(pulse_pdf_values)

        # compute unbinned negative likelihood over pulse times with given
        # time pdf: -sum( charge_i * log(pdf_d(t_i)) )
        time_loss = -pulse_charges * pulse_log_pdf_values

        if sort_loss_terms:
            loss_terms = [
                None,
                None,
                tf.tensor_scatter_nd_add(
                    tf.zeros_like(data_batch_dict["x_dom_charge"][..., 0]),
                    indices=data_batch_dict["x_pulses_ids"][:, :3],
                    updates=time_loss,
                ),
            ]
        else:
            loss_terms = [time_loss]

        # Add normalization terms if desired
        # Note: these are irrelevant for the minimization, but will make loss
        # curves more meaningful
        if self.configuration.config["config"]["add_normalization_term"]:
            # pulse likelihood has everything included due to utilize
            # asymmetric Gaussian
            pass

        return loss_terms

    def unbinned_pulse_time_mpe_llh(
        self, data_batch_dict, result_tensors, tensors, sort_loss_terms
    ):
        """Unbinned Pulse Time MPE likelhood.

        Pulses must *not* contain any pulses in excluded DOMs or excluded time
        windows. It is assumed that these pulses are already removed, e.g.
        the time pdf is calculated for all pulses.

        This computes the MPE likelihood by taking in account the time
        of the first pulse at each DOM in addition to the total charge
        at the DOM.

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
        result_tensors : dict of tf.Tensor
            A dictionary of output tensors.
            This  dictionary must at least contain:
                'pulse_pdf': The likelihood evaluated for each pulse
                             Shape: [-1]
                'pulse_cdf': The cumulative distribution function evaluated
                             for each pulse
                             Shape: [-1]
        tensors : DataTensorList
            The data tensor list describing the input data
        sort_loss_terms : bool, optional
            If true, the loss terms will be sorted and aggregated in three
            types of loss terms (this requires `reduce_to_scalar` == False):
                scalar: shape []
                    scalar loss for the whole batch of events
                event: shape [n_batch]
                    vector loss with one value per event
                dom: shape [n_batch, 86, 60]
                    tensor loss with one value for each DOM and event

        Returns
        -------
        List of tf.tensor
            Poisson Likelihood.
            List of tensors defining the terms of the log likelihood

        Raises
        ------
        NotImplementedError
            Description
        """
        dtype = getattr(
            tf, self.configuration.config["config"]["float_precision"]
        )

        # throw error if this is being used with time window exclusions
        # one needs to calculate cumulative pdf from exclusion window and
        # reduce the predicted charge by this factor
        if (
            "x_time_exclusions" in tensors.names
            and tensors.list[tensors.get_index("x_time_exclusions")].exists
        ):
            assert (
                "dom_cdf_exclusion" in result_tensors
            ), "Model must deal with time exclusions!"

        # shape: [n_pulses]
        pulse_charges = data_batch_dict["x_pulses"][:, 0]
        pulse_pdf_values = result_tensors["pulse_pdf"]
        pulse_cdf_values = result_tensors["pulse_cdf"]

        # get the index of the first pulse at each DOM
        # Shape: [n_pulses, 4] # [batch, string, dom, pulse_number]
        mask_first = data_batch_dict["x_pulses_ids"][:, 3] == 0

        # add pulses up to the defined quantile
        if "mpe_quantile" in self.configuration.config["config"]:
            mpe_quantile = self.configuration.config["config"]["mpe_quantile"]
            pulse_quantiles = data_batch_dict["x_pulses"][:, 2]
            mask_first = tf.math.logical_or(
                mask_first, pulse_quantiles <= mpe_quantile
            )
            print(f"Using quantile {mpe_quantile} for MPE loss")

        # Shape: [n_pulses_first]
        pulses_ids_first = data_batch_dict["x_pulses_ids"][:, :3][mask_first]
        pulse_pdf_value_first = pulse_pdf_values[mask_first]
        pulse_cdf_value_first = pulse_cdf_values[mask_first]
        pulse_charge_first = pulse_charges[mask_first]

        # shape: [n_batch, 86, 60, 1]
        hits_true = data_batch_dict["x_dom_charge"]

        # shape: [n_batch, 86, 60]
        dom_charges_true = tf.squeeze(hits_true, axis=-1)

        # mask out dom exclusions
        if (
            "x_dom_exclusions" in tensors.names
            and tensors.list[tensors.get_index("x_dom_exclusions")].exists
        ):
            mask_valid = tf.cast(
                tf.squeeze(data_batch_dict["x_dom_exclusions"], axis=-1),
                dtype=dtype,
            )
            dom_charges_true = dom_charges_true * mask_valid

        # get the total DOM charge at the DOMs of the first pulses
        # Shape: [n_pulses_first]
        dom_charges_true_pulses = tf.gather_nd(
            dom_charges_true, pulses_ids_first
        )

        # compute MPE log-likelihood
        # Shape: [n_pulses_first]
        if "mpe_quantile" in self.configuration.config["config"]:
            # Contribution at DOM i:
            #   charge_i * pdf_i(t_i)^c_i  * (1 - cdf_i(t_i))^(charge_i * (1 - quantile_i))
            #   with t_0 and c_0 the first pulse time and charge at DOM i
            # Note: The cdf term should really only be applied for the
            #       last pulse of the specified quantile. However, finding
            #       the last pulse at the given quantile for each hit DOM
            #       is more expensive than simply applying the cdf to all
            #       pulses in the quantile. Let's see how this works out...
            pulse_quantiles_first = pulse_quantiles[mask_first]
            mpe_log_llh = (
                tf_helpers.safe_log(dom_charges_true_pulses)
                + pulse_charge_first
                * tf_helpers.safe_log(pulse_pdf_value_first)
                + (
                    dom_charges_true_pulses
                    * tf.clip_by_value(1 - pulse_quantiles_first, 0, 1)
                )
                * tf_helpers.safe_log(1 - pulse_cdf_value_first)
            )
        else:
            # Contribution at DOM i:
            #   charge_i * pdf_i(t_0)^c_0 * (1 - cdf_i(t_0))^(charge_i - c_0)
            #   with t_0 and c_0 the first pulse time and charge at DOM i
            mpe_log_llh = (
                tf_helpers.safe_log(dom_charges_true_pulses)
                + pulse_charge_first
                * tf_helpers.safe_log(pulse_pdf_value_first)
                + (dom_charges_true_pulses - pulse_charge_first)
                * tf_helpers.safe_log(1 - pulse_cdf_value_first)
            )
        time_loss = -mpe_log_llh

        if sort_loss_terms:
            loss_terms = [
                None,
                None,
                tf.tensor_scatter_nd_add(
                    tf.zeros_like(data_batch_dict["x_dom_charge"][..., 0]),
                    indices=pulses_ids_first,
                    updates=time_loss,
                ),
            ]
        else:
            loss_terms = [time_loss]

        # Add normalization terms if desired
        # Note: these are irrelevant for the minimization, but will make loss
        # curves more meaningful
        if self.configuration.config["config"]["add_normalization_term"]:
            # pulse likelihood has everything included due to utilize
            # asymmetric Gaussian
            pass

        return loss_terms

    def unbinned_pulse_and_dom_charge_pdf(
        self, data_batch_dict, result_tensors, tensors, sort_loss_terms
    ):
        """Unbinned extended poisson likelhood with DOM charge PDF.

        Pulses must *not* contain any pulses in excluded DOMs or excluded time
        windows. It is assumed that these pulses are already removed, e.g.
        the time pdf is calculated for all pulses.

        This is similar to `unbinned_extended_pulse_llh`. Major differences:
            - No Poisson Likelihood is assumed here. Instead, the model
              estimates the charge PDF for each DOM

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
        result_tensors : dict of tf.Tensor
            A dictionary of output tensors.
            This  dictionary must at least contain:

                'dom_charges': the predicted charge at each DOM
                               Shape: [-1, 86, 60]
                'dom_charges_pdf': charge likelihood evaluated for each DOM
                               Shape: [-1, 86, 60]
                'pulse_pdf': The likelihood evaluated for each pulse
                             Shape: [-1]
        tensors : DataTensorList
            The data tensor list describing the input data
        sort_loss_terms : bool, optional
            If true, the loss terms will be sorted and aggregated in three
            types of loss terms (this requires `reduce_to_scalar` == False):
                scalar: shape []
                    scalar loss for the whole batch of events
                event: shape [n_batch]
                    vector loss with one value per event
                dom: shape [n_batch, 86, 60]
                    tensor loss with one value for each DOM and event

        Returns
        -------
        List of tf.tensor
            Poisson Likelihood.
            List of tensors defining the terms of the log likelihood

        Raises
        ------
        NotImplementedError
            Description
        """
        dtype = getattr(
            tf, self.configuration.config["config"]["float_precision"]
        )

        # shape: [n_pulses]
        pulse_charges = data_batch_dict["x_pulses"][:, 0]
        pulse_pdf_values = result_tensors["pulse_pdf"]

        # get charge likelihood over total charge at a DOM for extendended LLH
        # shape: [n_batch, 86, 60]
        llh_charge = tf_helpers.safe_log(result_tensors["dom_charges_pdf"])

        # throw error if this is being used with time window exclusions
        # one needs to calculate cumulative pdf from exclusion window and
        # reduce the predicted charge by this factor
        if (
            "x_time_exclusions" in tensors.names
            and tensors.list[tensors.get_index("x_time_exclusions")].exists
        ):
            assert (
                "dom_cdf_exclusion" in result_tensors
            ), "Model must deal with time exclusions!"

        # mask out dom exclusions
        if (
            "x_dom_exclusions" in tensors.names
            and tensors.list[tensors.get_index("x_dom_exclusions")].exists
        ):
            mask_valid = tf.cast(
                tf.squeeze(data_batch_dict["x_dom_exclusions"], axis=-1),
                dtype=dtype,
            )
            llh_charge = llh_charge * mask_valid

        # prevent log(zeros) issues
        pulse_log_pdf_values = tf_helpers.safe_log(pulse_pdf_values)

        # compute unbinned negative likelihood over pulse times with given
        # time pdf: -sum( charge_i * log(pdf_d(t_i)) )
        time_loss = -pulse_charges * pulse_log_pdf_values

        if sort_loss_terms:
            loss_doms = tf.tensor_scatter_nd_add(
                -llh_charge,
                indices=data_batch_dict["x_pulses_ids"][:, :3],
                updates=time_loss,
            )
            loss_terms = [None, None, loss_doms]
        else:
            loss_terms = [-llh_charge, time_loss]

        # Add normalization terms if desired
        # Note: these are irrelevant for the minimization, but will make loss
        # curves more meaningful
        if self.configuration.config["config"]["add_normalization_term"]:
            # the carge PDF should already be normalized
            pass

        return loss_terms

    def unbinned_charge_quantile_pdf(
        self, data_batch_dict, result_tensors, tensors, sort_loss_terms
    ):
        """Unbinned Pulse Quantile PDF.

        Pulses must *not* contain any pulses in excluded DOMs or excluded time
        windows. It is assumed that these pulses are already removed, e.g.
        the time pdf is calculated for all pulses.

        This loss will compute the likelihood of
            p(t_i| q_i, D_i)
            t_i: Time of pulse i
            q_i: Cumulative fractional DOM charge (charge quantile) of pulse i
            D_i: Total charge at the DOM at which pulse i was measured

        Note this likelihood does not place any constraints on total measured
        charge. It is therefore not suited to reconstruct Energy.

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
        result_tensors : dict of tf.Tensor
            A dictionary of output tensors.
            This  dictionary must at least contain:
                'pulse_quantile_pdf':
                    The likelihood evaluated for each pulse
                    Shape: [-1]
        tensors : DataTensorList
            The data tensor list describing the input data
        sort_loss_terms : bool, optional
            If true, the loss terms will be sorted and aggregated in three
            types of loss terms (this requires `reduce_to_scalar` == False):
                scalar: shape []
                    scalar loss for the whole batch of events
                event: shape [n_batch]
                    vector loss with one value per event
                dom: shape [n_batch, 86, 60]
                    tensor loss with one value for each DOM and event

        Returns
        -------
        List of tf.tensor
            Quantile PDF Likelihood.
            List of tensors defining the terms of the log likelihood
        """

        # shape: [n_pulses]
        pulse_quantiles = data_batch_dict["x_pulses"][:, 2]
        pulse_pdf_values = result_tensors["pulse_quantile_pdf"]

        # throw error if this is being used with time window exclusions
        # one needs to calculate cumulative pdf from exclusion window and
        # reduce the predicted charge by this factor
        if (
            "x_time_exclusions" in tensors.names
            and tensors.list[tensors.get_index("x_time_exclusions")].exists
        ):
            self._logger.warning(
                "Pulses in excluded time windows must have "
                "already been removed!"
            )
            assert (
                "dom_cdf_exclusion" in result_tensors
            ), "Model must deal with time exclusions!"

        # mask out dom exclusions
        if (
            "x_dom_exclusions" in tensors.names
            and tensors.list[tensors.get_index("x_dom_exclusions")].exists
        ):
            self._logger.warning(
                "Pulses at excluded DOMs must have already " "been removed!"
            )

        # prevent log(zeros) issues
        pulse_log_pdf_values = tf_helpers.safe_log(pulse_pdf_values)

        # compute unbinned negative likelihood over pulse times with given
        # time pdf: -sum( log(pdf_d(t_i)) )
        # do not weight with pulse charges here: a very high pulse charge
        # means that multiple pulses were combinend and that the true quantile
        # is not very accurate. Contrary, a low pulse charge
        # (probable single pulse) defines a very accurate time and quantile.
        # Therefore: do not weight log_pdf by charge
        time_loss = -pulse_log_pdf_values

        # only train for certain quantiles if bounds are provided
        low_lim = self.configuration.config["config"]["min_charge_quantile"]
        max_lim = self.configuration.config["config"]["max_charge_quantile"]

        if low_lim is not None:
            time_loss = tf.where(
                pulse_quantiles < low_lim, tf.zeros_like(time_loss), time_loss
            )
        if max_lim is not None:
            time_loss = tf.where(
                pulse_quantiles > max_lim, tf.zeros_like(time_loss), time_loss
            )

        if sort_loss_terms:
            loss_terms = [
                None,
                None,
                tf.tensor_scatter_nd_add(
                    tf.zeros_like(data_batch_dict["x_dom_charge"][..., 0]),
                    indices=data_batch_dict["x_pulses_ids"][:, :3],
                    updates=time_loss,
                ),
            ]
        else:
            loss_terms = [time_loss]

        # Add normalization terms if desired
        # Note: these are irrelevant for the minimization, but will make loss
        # curves more meaningful
        if self.configuration.config["config"]["add_normalization_term"]:
            # the pulse_pdf is already correctly normalized due to the
            # mixture model
            pass

        return loss_terms

    def dom_charge_pdf(
        self, data_batch_dict, result_tensors, tensors, sort_loss_terms
    ):
        """Charge PDF (estimated by Model)

        This is a likelihood over charge measured at each DOM.
        The PDF and likelihood used is defined
        by the model which must provide the computed charge likelihood
        values for the DOMs: `dom_charges_pdf`.

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
        result_tensors : dict of tf.Tensor
            A dictionary of output tensors.
            This  dictionary must at least contain:
                'dom_charges_pdf':
                    The likelihood evaluated for each DOM.
                    Shape: [-1, 86, 60]
        tensors : DataTensorList
            The data tensor list describing the input data
        sort_loss_terms : bool, optional
            If true, the loss terms will be sorted and aggregated in three
            types of loss terms (this requires `reduce_to_scalar` == False):
                scalar: shape []
                    scalar loss for the whole batch of events
                event: shape [n_batch]
                    vector loss with one value per event
                dom: shape [n_batch, 86, 60]
                    tensor loss with one value for each DOM and event

        Returns
        -------
        List of tf.tensor
            Charge PDF Likelihood.
            List of tensors defining the terms of the log likelihood
        """
        dtype = getattr(
            tf, self.configuration.config["config"]["float_precision"]
        )

        # get charge likelihood over total charge at a DOM for extendended LLH
        # shape: [n_batch, 86, 60]
        llh_charge = tf_helpers.safe_log(result_tensors["dom_charges_pdf"])

        # throw error if this is being used with time window exclusions
        # one needs to calculate cumulative pdf from exclusion window and
        # reduce the predicted charge by this factor
        if (
            "x_time_exclusions" in tensors.names
            and tensors.list[tensors.get_index("x_time_exclusions")].exists
        ):
            assert (
                "dom_cdf_exclusion" in result_tensors
            ), "Model must deal with time exclusions!"

        # mask out dom exclusions
        if (
            "x_dom_exclusions" in tensors.names
            and tensors.list[tensors.get_index("x_dom_exclusions")].exists
        ):
            mask_valid = tf.cast(
                tf.squeeze(data_batch_dict["x_dom_exclusions"], axis=-1),
                dtype=dtype,
            )
            llh_charge = llh_charge * mask_valid

        if sort_loss_terms:
            loss_terms = [
                None,
                None,
                -llh_charge,
            ]
        else:
            loss_terms = [-llh_charge]

        # Add normalization terms if desired
        # Note: these are irrelevant for the minimization, but will make loss
        # curves more meaningful
        if self.configuration.config["config"]["add_normalization_term"]:
            # model should already normalize the charge PDF
            pass

        return loss_terms

    def negative_binomial_charge_pdf(
        self, data_batch_dict, result_tensors, tensors, sort_loss_terms
    ):
        """Negative Binomial Charge PDF

        This is a likelihood over the charge measured at each DOM. A negative
        binomial distribution is used to calculate the charge likelihood. This
        allows for the inclusion of over-dispersion in contrast to the simple
        Poisson Likelihood. The model must provide the dom charges and
        variances thereof: `dom_charges`, `dom_charges_variance`.
        This loss pairs well with a loss for the time PDF such
        as `unbinned_pulse_time_llh`.

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
        result_tensors : dict of tf.Tensor
            A dictionary of output tensors.
            This  dictionary must at least contain:

                'dom_charges': the predicted charge at each DOM
                               Shape: [-1, 86, 60]
                'dom_charges_variance':
                    the predicted variance on the charge at each DOM.
                    This assumes the underlying distribution is a negative
                    binomial distribution.
                    Shape: [-1, 86, 60]
        tensors : DataTensorList
            The data tensor list describing the input data
        sort_loss_terms : bool, optional
            If true, the loss terms will be sorted and aggregated in three
            types of loss terms (this requires `reduce_to_scalar` == False):
                scalar: shape []
                    scalar loss for the whole batch of events
                event: shape [n_batch]
                    vector loss with one value per event
                dom: shape [n_batch, 86, 60]
                    tensor loss with one value for each DOM and event

        Returns
        -------
        List of tf.tensor
            Charge PDF Likelihood.
            List of tensors defining the terms of the log likelihood
        """

        # underneath 5e-5 the log_negative_binomial function becomes unstable
        eps = 5e-5
        dtype = getattr(
            tf, self.configuration.config["config"]["float_precision"]
        )

        # shape: [n_batch, 86, 60]
        hits_true = tf.squeeze(data_batch_dict["x_dom_charge"], axis=-1)
        hits_pred = result_tensors["dom_charges"]
        dom_charges_variance = result_tensors["dom_charges_variance"]

        # compute over-dispersion factor alpha
        # var = mu + alpha*mu**2
        # alpha = (var - mu) / (mu**2)
        dom_charges_alpha = (dom_charges_variance - hits_pred) / (
            hits_pred**2 + self.epsilon
        )
        # Make sure alpha is positive
        dom_charges_alpha = tf.clip_by_value(
            dom_charges_alpha, eps, float("inf")
        )

        # compute negative binomial charge likelihood over DOMs
        # shape: [n_batch, 86, 60]
        llh_charge = basis_functions.tf_log_negative_binomial(
            x=hits_true,
            mu=hits_pred,
            alpha=dom_charges_alpha,
            add_normalization_term=self.configuration.config["config"][
                "add_normalization_term"
            ],
        )

        # throw error if this is being used with time window exclusions
        # one needs to calculate cumulative pdf from exclusion window and
        # reduce the predicted charge by this factor
        if (
            "x_time_exclusions" in tensors.names
            and tensors.list[tensors.get_index("x_time_exclusions")].exists
        ):
            assert (
                "dom_cdf_exclusion" in result_tensors
            ), "Model must deal with time exclusions!"

        # mask out dom exclusions
        if (
            "x_dom_exclusions" in tensors.names
            and tensors.list[tensors.get_index("x_dom_exclusions")].exists
        ):
            mask_valid = tf.cast(
                tf.squeeze(data_batch_dict["x_dom_exclusions"], axis=-1),
                dtype=dtype,
            )
            llh_charge = llh_charge * mask_valid

        if sort_loss_terms:
            loss_terms = [
                None,
                None,
                -llh_charge,
            ]
        else:
            loss_terms = [-llh_charge]

        # Add normalization terms if desired
        # Note: these are irrelevant for the minimization, but will make loss
        # curves more meaningful
        if self.configuration.config["config"]["add_normalization_term"]:
            # total event charge is properly normalized due to the used gauss
            pass

        return loss_terms

    def negative_binomial_event_charge_pdf(
        self, data_batch_dict, result_tensors, tensors, sort_loss_terms
    ):
        """Negative Binomial Event Charge PDF

        This is a likelihood over the total event charge. A negative binomial
        distribution is used to calculate the charge likelihood. This allows
        for the inclusion of over-dispersion in contrast to the simple Poisson
        Likelihood. The model must provide the dom charges and variances
        thereof: `dom_charges`, `dom_charges_variance`.
        This loss should not be used in connection with per-DOM charge
        likelihoods, as the charge information would be double counted.
        This loss pairs well with a loss for the normalized DOM charge PDF such
        as `normalized_dom_charge_pdf`.

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
        result_tensors : dict of tf.Tensor
            A dictionary of output tensors.
            This  dictionary must at least contain:

                'dom_charges': the predicted charge at each DOM
                               Shape: [-1, 86, 60]
                'dom_charges_variance':
                    the predicted variance on the charge at each DOM.
                    This assumes the underlying distribution is a negative
                    binomial distribution.
                    Shape: [-1, 86, 60]
        tensors : DataTensorList
            The data tensor list describing the input data
        sort_loss_terms : bool, optional
            If true, the loss terms will be sorted and aggregated in three
            types of loss terms (this requires `reduce_to_scalar` == False):
                scalar: shape []
                    scalar loss for the whole batch of events
                event: shape [n_batch]
                    vector loss with one value per event
                dom: shape [n_batch, 86, 60]
                    tensor loss with one value for each DOM and event

        Returns
        -------
        List of tf.tensor
            Charge PDF Likelihood.
            List of tensors defining the terms of the log likelihood
        """

        # underneath 5e-5 the log_negative_binomial function becomes unstable
        eps = 5e-5
        dtype = getattr(
            tf, self.configuration.config["config"]["float_precision"]
        )

        # shape: [n_batch, 86, 60]
        hits_true = tf.squeeze(data_batch_dict["x_dom_charge"], axis=-1)
        hits_pred = result_tensors["dom_charges"]
        dom_charges_variance = result_tensors["dom_charges_variance"]

        # throw error if this is being used with time window exclusions
        # one needs to calculate cumulative pdf from exclusion window and
        # reduce the predicted charge by this factor
        if (
            "x_time_exclusions" in tensors.names
            and tensors.list[tensors.get_index("x_time_exclusions")].exists
        ):
            assert (
                "dom_cdf_exclusion" in result_tensors
            ), "Model must deal with time exclusions!"

        # mask out dom exclusions
        if (
            "x_dom_exclusions" in tensors.names
            and tensors.list[tensors.get_index("x_dom_exclusions")].exists
        ):
            mask_valid = tf.cast(
                tf.squeeze(data_batch_dict["x_dom_exclusions"], axis=-1),
                dtype=dtype,
            )
            hits_true = hits_true * mask_valid
            hits_pred = hits_pred * mask_valid
            dom_charges_variance = dom_charges_variance * mask_valid
        else:
            mask_valid = tf.ones_like(hits_true)

        # compute negative binomial charge likelihood over total event charge
        event_charges_true = tf.reduce_sum(hits_true, axis=[1, 2])
        event_charges_pred = tf.reduce_sum(hits_pred, axis=[1, 2])
        event_charges_variance = tf.reduce_sum(
            dom_charges_variance, axis=[1, 2]
        )

        # compute over-dispersion factor alpha
        # var = mu + alpha*mu**2
        # alpha = (var - mu) / (mu**2)
        event_charges_alpha = (event_charges_variance - event_charges_pred) / (
            event_charges_pred**2
        )

        # Make sure alpha is positive
        event_charges_alpha = tf.clip_by_value(
            event_charges_alpha, eps, float("inf")
        )

        llh_event = basis_functions.tf_log_negative_binomial(
            x=event_charges_true,
            mu=event_charges_pred,
            alpha=event_charges_alpha,
            add_normalization_term=self.configuration.config["config"][
                "add_normalization_term"
            ],
        )

        if sort_loss_terms:
            loss_terms = [
                None,
                -llh_event,
                None,
            ]
        else:
            loss_terms = [-llh_event]

        # Add normalization terms if desired
        # Note: these are irrelevant for the minimization, but will make loss
        # curves more meaningful
        if self.configuration.config["config"]["add_normalization_term"]:
            # total event charge is properly normalized due to the used gauss
            pass

        return loss_terms

    def normalized_dom_charge_pdf(
        self, data_batch_dict, result_tensors, tensors, sort_loss_terms
    ):
        """Normalized DOM Charge PDF

        This is a likelihood over the normalized DOM charge PDF, e.g. this
        likelihood is insensitive to the overall event charge normalization.
        The DOM PDF is calculated based on the expected fractional event charge
        at each DOM. The model must provide the dom charges key:
        `dom_charges`. Note that the variance on the expected charge
        (`dom_charges_variance`) is not taken into account here. The likelihood
        basically assumes that each DOM has the same uncertainty on the charge.
        This loss pairs well with a loss for the time PDF such
        as `unbinned_pulse_time_llh`.
        In addition, it can be transformed to an 'extended' likelihood by
        adding a likelihood over the event charge normalization, such as
        `negative_binomial_event_charge_pdf`.

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
        result_tensors : dict of tf.Tensor
            A dictionary of output tensors.
            This  dictionary must at least contain:

                'dom_charges': the predicted charge at each DOM
                               Shape: [-1, 86, 60]
        tensors : DataTensorList
            The data tensor list describing the input data
        sort_loss_terms : bool, optional
            If true, the loss terms will be sorted and aggregated in three
            types of loss terms (this requires `reduce_to_scalar` == False):
                scalar: shape []
                    scalar loss for the whole batch of events
                event: shape [n_batch]
                    vector loss with one value per event
                dom: shape [n_batch, 86, 60]
                    tensor loss with one value for each DOM and event

        Returns
        -------
        List of tf.tensor
            Charge PDF Likelihood.
            List of tensors defining the terms of the log likelihood
        """

        # prevent log(zeros) issues
        dtype = getattr(
            tf, self.configuration.config["config"]["float_precision"]
        )

        # shape: [n_batch, 86, 60]
        hits_true = tf.squeeze(data_batch_dict["x_dom_charge"], axis=-1)
        hits_pred = result_tensors["dom_charges"]

        # throw error if this is being used with time window exclusions
        # one needs to calculate cumulative pdf from exclusion window and
        # scale up the pulse pdf by this factor
        if (
            "x_time_exclusions" in tensors.names
            and tensors.list[tensors.get_index("x_time_exclusions")].exists
        ):
            assert (
                "dom_cdf_exclusion" in result_tensors
            ), "Model must deal with time exclusions!"

        # mask out dom exclusions
        # Note that this needs to be done prior to computing `event_total`
        # such that the PDF is properly normalized over active DOMs
        if (
            "x_dom_exclusions" in tensors.names
            and tensors.list[tensors.get_index("x_dom_exclusions")].exists
        ):
            mask_valid = tf.cast(
                tf.squeeze(data_batch_dict["x_dom_exclusions"], axis=-1),
                dtype=dtype,
            )
            hits_true = hits_true * mask_valid
            hits_pred = hits_pred * mask_valid

        # shape: [n_batch, 1, 1]
        event_total = tf.reduce_sum(hits_pred, axis=[1, 2], keepdims=True)

        # shape: [n_batch, 86, 60]
        dom_pdf = hits_pred / (event_total + self.epsilon)
        llh_dom = hits_true * tf_helpers.safe_log(dom_pdf)

        if sort_loss_terms:
            loss_terms = [
                None,
                None,
                -llh_dom,
            ]
        else:
            loss_terms = [-llh_dom]

        # Add normalization terms if desired
        # Note: these are irrelevant for the minimization, but will make loss
        # curves more meaningful
        if self.configuration.config["config"]["add_normalization_term"]:
            pass

        return loss_terms
