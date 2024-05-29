import os
import numpy as np
import tensorflow as tf

from egenerator.utils import func_tools


class VisualizePulseLikelihood:

    def __init__(
        self,
        manager,
        loss_module,
        function_cache,
        reco_key,
        output_dir,
        pdf_file_template="dom_pdf_{event_counter:06d}.png",
        cdf_file_template="dom_cdf_{event_counter:06d}.png",
        n_doms_x=5,
        n_doms_y=5,
        parameter_tensor_name="x_parameters",
        dom_pdf_kwargs={},
        dom_cdf_kwargs={},
    ):
        """Initialize module and setup tensorflow functions.

        Parameters
        ----------
        manager : Manager object
            The SourceManager object.
        loss_module : LossModule object
            The LossModule object to use for the reconstruction steps.
        function_cache : FunctionCache object
            A cache to store and share created concrete tensorflow functions.
        reco_key : str
            The name of the frame key to which the reco result is written to.
        output_dir : str
            The name to the output directory for the plots.
        pdf_file_template : str, optional
            The file template for PDF plots. Available parameters that are
            filled via str.format(). This includes a counter `event_counter`
            as well as any information that is passed via the `event_header`
            to the tray.execute() method.
            If None, no PDF plot will be made.
        cdf_file_template : str, optional
            The file template for CDF plots. Available parameters that are
            filled via str.format(). This includes a counter `event_counter`
            as well as any information that is passed via the `event_header`
            to the tray.execute() method.
            If None, no CDF plot will be made.
        n_doms_x : int, optional
            The number of DOMs to plot along the x-axis.
            Total number of plotted DOMs is `n_doms_x` * `n_doms_y`.
        n_doms_y : int, optional
            The number of DOMs to plot along the y-axis.
            Total number of plotted DOMs is `n_doms_x` * `n_doms_y`.
        parameter_tensor_name : str, optional
            Description
        dom_pdf_kwargs : dict, optional
            Additional keyword arguments passed on to `plot_dom_pdf`.
        dom_cdf_kwargs : dict, optional
            Additional keyword arguments passed on to `plot_dom_cdf`.
        """

        # store settings
        self.event_counter = 0
        self.manager = manager
        self.reco_key = reco_key
        self.output_dir = output_dir
        self.pdf_file_template = pdf_file_template
        self.cdf_file_template = cdf_file_template
        self.n_doms_x = n_doms_x
        self.n_doms_y = n_doms_y
        self.parameter_tensor_name = parameter_tensor_name
        self.dom_pdf_kwargs = dom_pdf_kwargs
        self.dom_cdf_kwargs = dom_cdf_kwargs

        if not os.path.exists(output_dir):
            print("Creating output directory: {}".format(output_dir))
            os.makedirs(output_dir)

        self.reco_index = manager.data_handler.tensors.get_index(self.reco_key)

        param_tensor = manager.data_trafo.data["tensors"][
            parameter_tensor_name
        ]
        param_dtype = param_tensor.dtype_tf
        param_signature = tf.TensorSpec(
            shape=param_tensor.shape, dtype=param_dtype
        )

        # define data batch tensor specification
        data_batch_signature = manager.data_handler.get_data_set_signature()

        # --------------------------------------------------
        # get concrete functions for reconstruction and loss
        # --------------------------------------------------

        # -------------------------
        # get model tensor function
        # -------------------------
        model_tensor_settings = {"model_index": 0}
        self.model_tensor_function = function_cache.get(
            "model_tensors_function", model_tensor_settings
        )

        if self.model_tensor_function is None:
            self.model_tensor_function = manager.get_model_tensors_function(
                **model_tensor_settings
            )
            function_cache.add(
                self.model_tensor_function, model_tensor_settings
            )

        # ---------------------------
        # Get parameter loss function
        # ---------------------------
        # Get loss function
        loss_settings = dict(
            input_signature=(param_signature, data_batch_signature),
            loss_module=loss_module,
            fit_parameter_list=True,
            minimize_in_trafo_space=False,
            seed=None,
            reduce_to_scalar=False,
            sort_loss_terms=True,
            parameter_tensor_name=parameter_tensor_name,
        )
        self.parameter_loss_function = function_cache.get(
            "parameter_loss_function", loss_settings
        )

        if self.parameter_loss_function is None:
            self.parameter_loss_function = manager.get_parameter_loss_function(
                **loss_settings
            )
            function_cache.add(self.parameter_loss_function, loss_settings)

    def execute(self, data_batch, results, event_header=None, **kwargs):
        """Execute module for a given batch of data.

        Parameters
        ----------
        data_batch : tuple of array_like
            A batch of data consisting of a tuple of data arrays.
        results : dict
            A dictionary with the results of previous modules.
        event_header : dict, optional
            A dictionary with event meta information on:
            {
                run_id: Run ID
                sub_run_id: SubrunID
                event_id: EventID
                sub_event_id: SubEventID
                sub_event_stream: SubEventStream
                start_time: start time MJD
                end_time: end time MJD
            }
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        TYPE
            Description
        """
        reco_result = data_batch[self.reco_index]
        loss_scalar, loss_event, loss_dom = self.parameter_loss_function(
            reco_result, data_batch
        )

        data_batch_dict = {}
        for i, tensor in enumerate(self.manager.data_handler.tensors.list):
            data_batch_dict[tensor.name] = data_batch[i]

        # -------------------------
        # compute model expectation
        # -------------------------
        tensor_names = [
            "x_pulses",
            "x_pulses_ids",
            "x_dom_exclusions",
            "x_dom_charge",
            "x_time_window",
            "x_time_exclusions",
            "x_time_exclusions_ids",
        ]
        model_kwargs = {}
        for tensor_name in tensor_names:
            if tensor_name in data_batch_dict:
                model_kwargs[tensor_name] = data_batch_dict[tensor_name]

        result_tensors = self.model_tensor_function(
            parameters=reco_result, **model_kwargs
        )

        format_cfg = {"event_counter": self.event_counter}
        if event_header is not None:
            format_cfg.update(dict(event_header))

        if self.pdf_file_template is not None:
            pdf_file_name = os.path.join(
                self.output_dir, self.pdf_file_template.format(**format_cfg)
            )

            self.plot_dom_pdf(
                reco_result=reco_result,
                data_batch_dict=data_batch_dict,
                result_tensors=result_tensors,
                losses=(
                    loss_scalar.numpy(),
                    loss_event.numpy(),
                    loss_dom.numpy(),
                ),
                file_name=pdf_file_name,
                n_doms_x=self.n_doms_x,
                n_doms_y=self.n_doms_y,
                event_header=event_header,
                **self.dom_pdf_kwargs
            )

        if self.cdf_file_template is not None:
            cdf_file_name = os.path.join(
                self.output_dir, self.cdf_file_template.format(**format_cfg)
            )

            self.plot_dom_cdf(
                reco_result=reco_result,
                data_batch_dict=data_batch_dict,
                result_tensors=result_tensors,
                losses=(
                    loss_scalar.numpy(),
                    loss_event.numpy(),
                    loss_dom.numpy(),
                ),
                file_name=cdf_file_name,
                n_doms_x=self.n_doms_x,
                n_doms_y=self.n_doms_y,
                event_header=event_header,
                **self.dom_cdf_kwargs
            )

        self.event_counter += 1
        return {}

    def plot_meta_data(
        self,
        ax,
        event_header,
        reco_result,
        event_charge=None,
        max_params=9,
        fontsize=8,
        max_width=15,
    ):
        """Plot Event Meta Data

        Parameters
        ----------
        ax : matplotlib.axis
            The axis on which the meta data will be plotted.
        event_header : dict
            A dictionary with event meta information.
        reco_result : array_like
            The reconstruction result.
            Shape: [1, num_parameters]
        event_charge : None, optional
            If provided, the total event charge will be displayed.
        max_params : int, optional
            Maximum number of best fit parameters to display.
        fontsize : float, optional
            The font size for the meta data.
        max_width : int, optional
            Defines how wide the text box will be (2*max_width).
        """
        ax.axis("off")

        assert len(reco_result) == 1, reco_result

        textstr = "Run ID: {}\n".format(event_header["run_id"])
        textstr += "SubRun ID: {}\n".format(event_header["sub_run_id"])
        textstr += "Event ID: {}\n".format(event_header["event_id"])
        textstr += "Sub Event ID: {}\n".format(event_header["sub_event_id"])
        textstr += "Date: {}\n".format(event_header["date_string"])
        textstr += "Time: {}\n".format(event_header["time_string"])

        if event_charge is not None:
            textstr += "Event Charge: {:3.1f} PE\n".format(event_charge)

        textstr += "Best Fit Parameters:\n"
        for i, param in enumerate(
            self.manager.models[0].parameter_names[:max_params]
        ):

            if len(param) > max_width:
                param_red = param[:3] + "..." + param[-max_width + 3 :]
            else:
                param_red = param
            textstr += "    {}: {:3.3f}\n".format(param_red, reco_result[0, i])
        if len(self.manager.models[0].parameter_names) > max_params:
            textstr += "    ...\n"

        red_key = self.reco_key.replace("EventGenerator_", "")
        reco_len = len(red_key)
        index = 0
        step_size = 2 * max_width
        while index < reco_len:
            textstr += "{}\n".format(red_key[index : index + step_size])
            index += step_size

        props = dict(
            boxstyle="round", edgecolor="0.8", facecolor="white", alpha=0.5
        )
        ax.text(
            0.0,
            1.0,
            textstr,
            transform=ax.transAxes,
            fontsize=fontsize,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=props,
        )

    def plot_dom_pdf(
        self,
        reco_result,
        data_batch_dict,
        result_tensors,
        losses,
        file_name,
        n_doms_x=2,
        n_doms_y=2,
        n_components=5,
        limit_charge_fraction=0.9,
        limit_y_range=True,
        scale_by_charge=False,
        event_header=None,
        yscale="log",
        num_bins=30,
        figsize_scale_x=3,
        figsize_scale_y=3,
        meta_data_kwargs={},
    ):
        """Create DOM PDF plots

        Parameters
        ----------
        reco_result : array_like
            The reconstruction result.
            Shape: [1, num_parameters]
        data_batch_dict : dict or tf.Tensor
            The data batch dictionary.
        result_tensors : dict of tf.tensor
            The dictionary of output tensors as obtained from `get_tensors`.
        losses : tuple
            Tuple of losses:
            scalar loss: Shape: ()
            event loss: Shape: (1,)
            DOM loss: Shape: (1, 86, 60)
        file_name : str
            The name of the output file for the plot.
        n_doms_x : int, optional
            The number of DOMs to plot along the x-axis.
            Total number of plotted DOMs is `n_doms_x` * `n_doms_y`.
        n_doms_y : int, optional
            The number of DOMs to plot along the y-axis.
            Total number of plotted DOMs is `n_doms_x` * `n_doms_y`.
        n_components : int, optional
            The maximum number of subcomponents to plot.
        limit_charge_fraction : float, optional
            Limit the x-axis time range to only show pulses up to this fraction
            of total charge.
        limit_y_range : bool, optional
            If True, lower value y-scale is limited to start at lowest charge
            values. In this case subcomponents of the event-generator might
            be cut off and not visible.
        scale_by_charge : bool, optional
            If True, the charge scaled PDF, i.e. charge distribution
            is shown, rather than the normalized PDF.
        event_header : dict, optional
            If provided, the first axis will be used to display meta data.
        yscale : str, optional
            The scale of the y-axis.
        num_bins : int, optional
            The number of time bins to use for the pulse histogram.
        figsize_scale_x : float, optional
            The figsize along the x-axis for an individual sub-plot.
        figsize_scale_y : float, optional
            The figsize along the y-axis for an individual sub-plot.
        meta_data_kwargs : dict, optional
            Additional kwargs passed on to `plot_meta_data`.
        """
        from matplotlib import pyplot as plt
        import itertools

        # Number of DOMs and components to plot
        event = 0
        n_doms = n_doms_x * n_doms_y

        if event_header is not None:
            n_doms -= 1

        loss_scalar, loss_event, loss_dom = losses
        charge_values = result_tensors["dom_charges"]

        assert len(loss_event) == 1, "Expects one event at a time"
        assert len(reco_result) == 1, "Expects one event at a time"
        assert loss_dom.shape == (1, 86, 60), loss_dom.shape
        assert charge_values.shape == (1, 86, 60, 1), charge_values.shape

        loss_dom = loss_dom[event]
        loss_event = loss_event[event]
        charge_values = charge_values[..., 0]

        # --------------------------------------
        # Select DOMs with highest impact to LLh
        # --------------------------------------
        loss_dom_flat = np.reshape(loss_dom, [86 * 60])
        max_doms = np.argsort(np.abs(loss_dom_flat))[-n_doms:]

        # convert to list of (zero-based) strings and DOM numbers
        strings_zb = []
        doms_zb = []
        for flat_index in max_doms:
            strings_zb.append(flat_index // 60)
            doms_zb.append(flat_index % 60)

        fig, axes = plt.subplots(
            n_doms_y,
            n_doms_x,
            figsize=(figsize_scale_x * n_doms_x, figsize_scale_y * n_doms_y),
        )

        if event_header is None:
            dom_axes = axes.ravel()
        else:
            self.plot_meta_data(
                ax=axes.ravel()[0],
                event_header=event_header,
                reco_result=reco_result,
                event_charge=np.sum(data_batch_dict["x_pulses"][:, 0]),
                **meta_data_kwargs
            )
            dom_axes = axes.ravel()[1:]

        for ax, string, dom in zip(dom_axes, strings_zb, doms_zb):

            color_cycler = itertools.cycle(
                plt.rcParams["axes.prop_cycle"].by_key()["color"]
            )

            color = next(color_cycler)

            # get pulses for this DOM
            pulses = self.get_pulses(
                event=event,
                string=string,
                dom=dom,
                data_batch_dict=data_batch_dict,
            )

            # define time range for plot
            min_time, max_time = self.compute_time_range(
                pulses, limit_charge_fraction=limit_charge_fraction
            )

            pulse_bins = np.linspace(min_time, max_time, num_bins)
            _pulse_bin_width = np.unique(np.diff(pulse_bins))
            pulse_bin_width = _pulse_bin_width[0]
            assert np.allclose(
                pulse_bin_width, _pulse_bin_width
            ), pulse_bin_width
            x = np.linspace(min_time, max_time, 100)

            # plot pulses
            if len(pulses) > 0:
                pulse_hist, _, _ = ax.hist(
                    pulses[:, 1],
                    bins=pulse_bins,
                    weights=pulses[:, 0],
                    density=not scale_by_charge,
                    color="0.2",
                    label="Pulses: {:3.2f} PE".format(np.sum(pulses[:, 0])),
                    histtype="step",
                )

            # -----------
            # compute pdf
            # -----------
            pdf_results = self.manager.models[0].pdf(
                x,
                result_tensors=result_tensors,
                tw_exclusions=data_batch_dict["x_time_exclusions"],
                tw_exclusions_ids=data_batch_dict["x_time_exclusions_ids"],
                output_nested_pdfs=True,
            )
            if len(pdf_results) == 2 and isinstance(pdf_results[1], dict):
                pdf, nested_pdfs = pdf_results
            else:
                pdf = pdf_results
                nested_pdfs = None

            # ---------------
            # plot components
            # ---------------
            if nested_pdfs is not None:
                fractions = []
                names = []
                pdf_values_list = []
                for source_name, (fraction, pdf_values) in nested_pdfs.items():
                    fractions.append(fraction[event, string, dom, 0])
                    names.append(source_name)
                    pdf_values_list.append(pdf_values[event, string, dom])

                sorted_indices = np.argsort(fractions)[::-1]
                alpha = 1.0
                ls_cycler = itertools.cycle(["--", ":", "-."])
                for index in sorted_indices[-n_components:]:
                    pdf_i_comp = pdf_values_list[index] * fractions[index]
                    if scale_by_charge:
                        plot_values = (
                            pdf_i_comp
                            * charge_values[event, string, dom]
                            * pulse_bin_width
                        )
                    else:
                        plot_values = pdf_i_comp
                    ax.plot(
                        x,
                        plot_values,
                        label="{}".format(names[index]),
                        ls=next(ls_cycler),
                        color=color,
                        alpha=alpha,
                    )
                    alpha -= 0.8 / n_components

            # plot pdf
            pdf_i = pdf[event, string, dom]
            if scale_by_charge:
                plot_values = (
                    pdf_i * charge_values[event, string, dom] * pulse_bin_width
                )
            else:
                plot_values = pdf_i
            ax.plot(
                x,
                plot_values,
                label="Expectation: {:3.2f} PE".format(
                    charge_values[event, string, dom]
                ),
                color=color,
            )

            # ---------------------------
            # plot time window exclusions
            # ---------------------------
            tw_exclusions = self.get_tw_exclusions(
                event=event,
                string=string,
                dom=dom,
                data_batch_dict=data_batch_dict,
            )
            for tw_exclusion in tw_exclusions:
                ax.axvspan(
                    tw_exclusion[0], tw_exclusion[1], alpha=0.2, color="red"
                )

            if not data_batch_dict["x_dom_exclusions"][event, string, dom]:
                ax.axvspan(min_time, max_time, alpha=0.2, color="red")

            # -----------
            # Axis labels
            # -----------
            loss = loss_dom[string, dom]
            ax.set_title(
                "[{:02d}, {:02d}] LLH: {:3.3f}".format(
                    string + 1, dom + 1, -loss
                )
            )
            ax.set_yscale(yscale)
            ax.set_xlabel("Time [ns]")
            if scale_by_charge:
                ax.set_ylabel(
                    "Charge per bin [PE/{:3.0f}ns]".format(pulse_bin_width)
                )
            else:
                ax.set_ylabel("PDF")

            ax.set_xlim(min_time, max_time)
            if limit_y_range:
                ax.set_ylim(np.min(pulse_hist[pulse_hist > 0]) * 0.9)
            ax.legend(fontsize=7)

        fig.tight_layout()
        fig.savefig(file_name)
        plt.close(fig)

    def plot_dom_cdf(
        self,
        reco_result,
        data_batch_dict,
        result_tensors,
        losses,
        file_name,
        n_doms_x=2,
        n_doms_y=2,
        n_components=5,
        limit_charge_fraction=0.9,
        limit_y_range=True,
        scale_by_charge=False,
        event_header=None,
        yscale="log",
        meta_data_kwargs={},
    ):
        """Create DOM CDF plots

        Parameters
        ----------
        reco_result : array_like
            The reconstruction result.
            Shape: [1, num_parameters]
        data_batch_dict : dict or tf.Tensor
            The data batch dictionary.
        result_tensors : dict of tf.tensor
            The dictionary of output tensors as obtained from `get_tensors`.
        losses : tuple
            Tuple of losses:
            scalar loss: Shape: ()
            event loss: Shape: (1,)
            DOM loss: Shape: (1, 86, 60)
        file_name : str
            The name of the output file for the plot.
        n_doms_x : int, optional
            The number of DOMs to plot along the x-axis.
            Total number of plotted DOMs is `n_doms_x` * `n_doms_y`.
        n_doms_y : int, optional
            The number of DOMs to plot along the y-axis.
            Total number of plotted DOMs is `n_doms_x` * `n_doms_y`.
        n_components : int, optional
            The maximum number of subcomponents to plot.
        limit_charge_fraction : float, optional
            Limit the x-axis time range to only show pulses up to this fraction
            of total charge.
        limit_y_range : bool, optional
            If True, lower value y-scale is limited to start at lowest charge
            values. In this case subcomponents of the event-generator might
            be cut off and not visible.
        scale_by_charge : bool, optional
            If True, the charge scaled CDF, i.e. cumulative charge distribution
            is shown, rather than the normalized CDF.
        event_header : dict, optional
            If provided, the first axis will be used to display meta data.
        yscale : str, optional
            The scale of the y-axis.
        meta_data_kwargs : dict, optional
            Additional kwargs passed on to `plot_meta_data`.
        """
        from matplotlib import pyplot as plt
        import itertools

        # Number of DOMs and components to plot
        event = 0
        n_doms = n_doms_x * n_doms_y

        if event_header is not None:
            n_doms -= 1

        loss_scalar, loss_event, loss_dom = losses
        charge_values = result_tensors["dom_charges"]

        assert len(loss_event) == 1, "Expects one event at a time"
        assert len(reco_result) == 1, "Expects one event at a time"
        assert loss_dom.shape == (1, 86, 60), loss_dom.shape
        assert charge_values.shape == (1, 86, 60, 1), charge_values.shape

        loss_dom = loss_dom[event]
        loss_event = loss_event[event]
        charge_values = charge_values[..., 0]

        # --------------------------------------
        # Select DOMs with highest impact to LLh
        # --------------------------------------
        loss_dom_flat = np.reshape(loss_dom, [86 * 60])
        max_doms = np.argsort(np.abs(loss_dom_flat))[-n_doms:]

        # convert to list of (zero-based) strings and DOM numbers
        strings_zb = []
        doms_zb = []
        for flat_index in max_doms:
            strings_zb.append(flat_index // 60)
            doms_zb.append(flat_index % 60)

        fig, axes = plt.subplots(
            n_doms_y, n_doms_x, figsize=(3 * n_doms_x, 3 * n_doms_y)
        )

        if event_header is None:
            dom_axes = axes.ravel()
        else:
            self.plot_meta_data(
                ax=axes.ravel()[0],
                event_header=event_header,
                reco_result=reco_result,
                event_charge=np.sum(data_batch_dict["x_pulses"][:, 0]),
                **meta_data_kwargs
            )
            dom_axes = axes.ravel()[1:]

        for ax, string, dom in zip(dom_axes, strings_zb, doms_zb):

            color_cycler = itertools.cycle(
                plt.rcParams["axes.prop_cycle"].by_key()["color"]
            )

            color = next(color_cycler)

            # get pulses for this DOM
            pulses = self.get_pulses(
                event=event,
                string=string,
                dom=dom,
                data_batch_dict=data_batch_dict,
            )

            # define time range for plot
            min_time, max_time = self.compute_time_range(
                pulses, limit_charge_fraction=limit_charge_fraction
            )

            x = np.linspace(min_time, max_time, 100)

            # plot pulses
            if len(pulses) > 0:
                cum_sum = np.cumsum(pulses[:, 0])
                charge_sum = np.sum(pulses[:, 0])
                if scale_by_charge:
                    plot_values = cum_sum
                else:
                    plot_values = cum_sum / charge_sum
                ax.plot(
                    pulses[:, 1],
                    plot_values,
                    color="0.2",
                    label="Pulses: {:3.2f} PE".format(charge_sum),
                )

            # -----------
            # compute cdf
            # -----------
            cdf_results = self.manager.models[0].cdf(
                x,
                result_tensors=result_tensors,
                tw_exclusions=data_batch_dict["x_time_exclusions"],
                tw_exclusions_ids=data_batch_dict["x_time_exclusions_ids"],
                output_nested_pdfs=True,
            )
            if len(cdf_results) == 2 and isinstance(cdf_results[1], dict):
                cdf, nested_cdfs = cdf_results
            else:
                cdf = cdf_results
                nested_cdfs = None

            # ---------------
            # plot components
            # ---------------
            if nested_cdfs is not None:
                fractions = []
                names = []
                cdf_values_list = []
                for source_name, (fraction, cdf_values) in nested_cdfs.items():
                    fractions.append(fraction[event, string, dom, 0])
                    names.append(source_name)
                    cdf_values_list.append(cdf_values[event, string, dom])

                sorted_indices = np.argsort(fractions)[::-1]
                alpha = 1.0
                ls_cycler = itertools.cycle(["--", ":", "-."])
                for index in sorted_indices[-n_components:]:
                    cdf_i_comp = cdf_values_list[index] * fractions[index]
                    if scale_by_charge:
                        plot_values = (
                            cdf_i_comp * charge_values[event, string, dom]
                        )
                    else:
                        plot_values = cdf_i_comp
                    ax.plot(
                        x,
                        plot_values,
                        label="{}".format(names[index]),
                        ls=next(ls_cycler),
                        color=color,
                        alpha=alpha,
                    )
                    alpha -= 0.8 / n_components

            # plot cdf
            cdf_i = cdf[event, string, dom]
            charge_i = charge_values[event, string, dom]
            if scale_by_charge:
                plot_values = cdf_i * charge_i
            else:
                plot_values = cdf_i
            ax.plot(
                x,
                plot_values,
                label="Expectation: {:3.2f} PE".format(
                    charge_values[event, string, dom]
                ),
                color=color,
            )

            # ---------------------------
            # plot time window exclusions
            # ---------------------------
            tw_exclusions = self.get_tw_exclusions(
                event=event,
                string=string,
                dom=dom,
                data_batch_dict=data_batch_dict,
            )
            for tw_exclusion in tw_exclusions:
                ax.axvspan(
                    tw_exclusion[0], tw_exclusion[1], alpha=0.2, color="red"
                )

            if not data_batch_dict["x_dom_exclusions"][event, string, dom]:
                ax.axvspan(min_time, max_time, alpha=0.2, color="red")

            # -----------
            # Axis labels
            # -----------
            loss = loss_dom[string, dom]
            ax.set_title(
                "[{:02d}, {:02d}] LLH: {:3.3f}".format(
                    string + 1, dom + 1, -loss
                )
            )
            ax.set_yscale(yscale)
            ax.set_xlabel("Time [ns]")
            if scale_by_charge:
                ax.set_ylabel("Cumulative Charge")
            else:
                ax.set_ylabel("CDF")
            ax.set_xlim(min_time, max_time)

            if limit_y_range and len(pulses) > 0:
                if scale_by_charge:
                    ax.set_ylim(np.min(pulses[:, 0]) * 0.9)
                else:
                    ax.set_ylim(np.min(pulses[:, 0]) / charge_sum * 0.9)

            ax.legend(fontsize=7)

        fig.tight_layout()
        fig.savefig(file_name)
        plt.close(fig)

    def compute_time_range(
        self, pulses, limit_charge_fraction=None, min_length=1000.0
    ):
        """Compute the time range for the visible area

        Parameters
        ----------
        pulses : array_like
            The pulses as a list of [charge, time (, quantile)]
            Shape: [n_pulses, (2, 3)]
        limit_charge_fraction : float, optional
            Limit the x-axis time range to only show pulses up to this fraction
            of total charge.
        min_length : float, optional
            Minimum time length. If the time range based on the pulses
            is smaller than this, then it will get expanded to have at
            least this specified length.

        Returns
        -------
        float
            The minimum time.
        float
            The maximum time.
        """
        if len(pulses) == 0:
            min_time = 9500
            max_time = 12000
        else:
            min_time = np.min(pulses[:, 1]) - 50
            max_time = np.max(pulses[:, 1]) + 50

        if len(pulses) > 1 and limit_charge_fraction is not None:
            max_time = func_tools.weighted_quantile(
                x=pulses[:, 1],
                weights=pulses[:, 0],
                quantile=limit_charge_fraction,
            )

        length_diff = min_length - (max_time - min_time)
        if length_diff > 0.0:
            if limit_charge_fraction is not None:
                # cut off less at the end
                max_time += length_diff
            else:
                # expand equally in both directions
                min_time -= length_diff / 2.0
                max_time += length_diff / 2.0

        return min_time, max_time

    def get_pulses(self, event, string, dom, data_batch_dict):
        """Extract pulses from data batch dictionary

        Parameters
        ----------
        event : int
            The batch id number (should always be zero here).
        string : int
            The (zero-based) string number.
        dom : int
            The (zero-based) string number.
        data_batch_dict : dict of array_like
            The data batch dictionary which holds all the data tensors read in
            for this event.

        Returns
        -------
        array_like
            The pulses belonging to the specified DOM.
            Each pulse is defined by [charge, time (, quantile)]
            Shape: [n_pulses, 2(3)]
        """
        pulses = data_batch_dict["x_pulses"]
        pulses_ids = data_batch_dict["x_pulses_ids"]

        mask = np.logical_and(
            pulses_ids[:, 0] == event, pulses_ids[:, 1] == string
        )
        mask = np.logical_and(mask, pulses_ids[:, 2] == dom)

        return pulses[mask]

    def get_tw_exclusions(self, event, string, dom, data_batch_dict):
        """Extract time window exclusions from data batch dictionary

        Parameters
        ----------
        event : int
            The batch id number (should always be zero here).
        string : int
            The (zero-based) string number.
        dom : int
            The (zero-based) string number.
        data_batch_dict : dict of array_like
            The data batch dictionary which holds all the data tensors read in
            for this event.

        Returns
        -------
        array_like
            The time window exclusions belonging to the specified DOM.
            Each time window exclusions is defined by [t_start, t_stop]
            Shape: [n_pulses, 2(3)]
        """
        tw_exclusions = data_batch_dict["x_time_exclusions"]
        tw_ids = data_batch_dict["x_time_exclusions_ids"]

        mask = np.logical_and(tw_ids[:, 0] == event, tw_ids[:, 1] == string)
        mask = np.logical_and(mask, tw_ids[:, 2] == dom)

        return tw_exclusions[mask]
