import os
import numpy as np

from icecube import dataclasses, icetray
from icecube.icetray.i3logging import log_info

from egenerator.utils.configurator import ManagerConfigurator
from egenerator.manager.reconstruction.tray import ReconstructionTray
from egenerator.utils import angles


class EventGeneratorReconstruction(icetray.I3ConditionalModule):
    """Class to apply Event-Generator model."""

    def __init__(self, context):
        """Class to apply Event-Generator Model.

        Parameters
        ----------
        context : TYPE
            Description
        """
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddOutBox("OutBox")

        # Required settings
        self.AddParameter("seed_keys", "A seed or list of seeds to use.")
        self.AddParameter(
            "model_names",
            "A model name or list of model names "
            "defining which model or ensemble of models to "
            "apply. The full path is given by `model_base_dir` +"
            " model_name if `model_base_dir` is provided. "
            "Otherwise `model_names` must define the full "
            "model path.",
        )

        # Optional settings
        self.AddParameter(
            "model_base_dir",
            "The base directory in which the models are located."
            " The full path is given by `model_base_dir` + "
            " model_name if `model_base_dir` is provided. "
            "Otherwise `model_names` must define the full "
            "model path.",
            "/data/user/mhuennefeld/exported_models/egenerator",
        )
        self.AddParameter(
            "output_key",
            "The output base key to which results will be saved",
            "EventGenerator",
        )
        self.AddParameter(
            "pulse_key",
            "The pulses to use for the reconstruction. Note: "
            "pulses in exclusion windows must have already been "
            "excluded!",
            "InIceDSTPulses",
        )
        self.AddParameter(
            "dom_exclusions_key", "The DOM exclusions to use.", "BadDomsList"
        )
        self.AddParameter(
            "time_exclusions_key", "The time exclusions to use.", None
        )
        self.AddParameter(
            "add_circular_err",
            "Add circularized angular uncertainty estimate.",
            False,
        )
        self.AddParameter(
            "add_covariances",
            "Add calculation of covariance matrices via Hessian "
            "matrix evaluated at the best fit point.",
            False,
        )
        self.AddParameter(
            "add_goodness_of_fit",
            "Add calculation of goodness of fit. Points are "
            "sampled from the fitted posterior and reconstructed"
            ". The llh of these points is then used to construct"
            " a test-statistic. This test-statistic distribution"
            " is used to obtain a p-value for the goodness "
            "of the fit, e.g. it is tested if the llh of the "
            "best fit position of the data event matches. If it "
            "does not this can indicate that the fit found a "
            "local minimum, or that the provided data event "
            "is not well described by the chosen model. "
            "Note: if `add_covariances` is True, then the "
            "computed covariance matrix will be used to define "
            "the posterior for the sampling, otherwise the "
            "samples will simply be set to the best fit point.",
            False,
        )
        self.AddParameter(
            "add_mcmc_samples",
            "Add samples from a Markov-Chain-Monte-Carlo. "
            "Settings for MCMC are defined in key "
            "`mcmc_settings`.",
            False,
        )
        self.AddParameter(
            "add_skyscan",
            "Perform a skyscan with azimuth/zenith fixed. "
            "Settings for skyscan are defined in key "
            "`skyscan_settings`.",
            False,
        )
        self.AddParameter(
            "label_key",
            "Only relevant if labels are being loaded. "
            "The key from which to load labels.",
            "LabelsDeepLearning",
        )
        self.AddParameter(
            "snowstorm_key",
            "Only relevant if labels are being loaded. "
            "The key from which to load snowstorm parameters.",
            "SnowstormParameterDict",
        )
        self.AddParameter(
            "num_threads",
            "Number of threads to use for tensorflow. This will "
            "be passed on to tensorflow where it is used to set "
            'the "intra_op_parallelism_threads" and '
            '"inter_op_parallelism_threads" settings. If a '
            "value of zero (default) is provided, the system "
            "uses an appropriate number. Note: when running "
            "this as a job on a cluster you might want to limit "
            '"num_threads" to the amount of allocated CPUs.',
            0,
        )

        # Reconstruction specific optional settings
        self.AddParameter(
            "minimize_in_trafo_space",
            "Perform minimization in normalized coordinates if "
            "True (this is usually desired).",
            True,
        )
        self.AddParameter(
            "parameter_boundaries",
            "A dictionary which specifies the boundaries of "
            "parameter values. Internally a pseudo uniform "
            "prior is applied to penalize exceeding beyond "
            'these boundaries. This is a "pseudo" uniform prior '
            "because this will not affect the LLH values if the "
            "parameters are in bounds. The specified parameter "
            "boundaries must be a dictionary of the format: "
            '{"ParamName": [lower_bound, upper_bound]} '
            "and the boundaries must be finite",
            None,
        )
        self.AddParameter(
            "minimize_parameter_dict",
            "A dictionary with elements fit_parameter: boolean "
            "that indicates whether the parameter will be fit "
            "(if set to True) or if it will be held constant "
            "(if set to False). Values for unspecified "
            "parameters will default to "
            "`minimize_parameter_default_value`. ",
            {},
        )
        self.AddParameter(
            "minimize_parameter_default_value",
            "The default value for parameters not defined in "
            "the `minimize_parameter_dict`: "
            "a dictionary with elements fit_parameter: boolean "
            "that indicates whether the parameter will be fit "
            "(if set to True) or if it will be held constant "
            "(if set to False). Values for unspecified "
            "parameters will default to "
            "`minimize_parameter_default_value`.",
            True,
        )
        self.AddParameter(
            "missing_seed_value_dict",
            "If a certain model parameter does not exist in the "
            "frame key set via `seed_keys`, a default value may "
            "be specified with the `missing_seed_value_dict` "
            "dictionary. Entries have the format: "
            r"{parameter_name}: default_value.",
            {},
        )
        self.AddParameter(
            "missing_seed_value",
            "If a model parameter key is not given in the "
            "provided seed_key, it can be replaced with a "
            "default value defined in `missing_seed_value`. "
            "Note that this value will be set for any and all "
            "parameters that are not found. Keep in mind that "
            "this will also work if the wrong frame key is "
            "provided by accident to `seed_keys`.",
            None,
        )
        self.AddParameter(
            "reco_optimizer_interface",
            "The reconstruction interface to use. Options are: "
            '"scipy" or "tfp" (tensorflow_probability).',
            "scipy",
        )
        self.AddParameter(
            "scipy_optimizer_settings",
            "Only relevant if `reco_optimizer_interface` is set "
            ' to "scipy" and/or for `add_circular_err`. '
            "Defines settings for scipy optimizer",
            {"method": "BFGS"},
        )
        self.AddParameter(
            "tf_optimizer_settings",
            "Only relevant if `reco_optimizer_interface` is set "
            ' to "tfp". '
            "Defines settings for tensorflow optimizer",
            {"method": "bfgs_minimize", "x_tolerance": 0.001},
        )
        self.AddParameter(
            "goodness_of_fit_settings",
            "Only relevant if `add_goodness_of_fit` is set "
            ' to "True". '
            "Defines settings for goodness of fit calculation.",
            {
                "scipy_optimizer_settings": {
                    "method": "L-BFGS-B",
                    "options": {"ftol": 1e-6},
                },
                "num_samples": 50,
                "reconstruct_samples": True,
                "add_per_dom_calculation": True,
                "normalize_by_total_charge": False,
            },
        )

        # MCMC specific optional settings
        self.AddParameter(
            "mcmc_settings",
            "Only relevant if `add_mcmc_samples` is set "
            ' to "True". Defines settings for MCMC sampling.',
            {
                "mcmc_num_chains": 10,
                "mcmc_method": "HamiltonianMonteCarlo",
                "mcmc_num_results": 1000,
                "mcmc_num_burnin_steps": 100,
                "mcmc_num_steps_between_results": 0,
                "mcmc_num_parallel_iterations": 1,
                "distribution_settings": {},
            },
        )
        self.AddParameter(
            "mcmc_quantiles",
            "Only relevant if `add_mcmc_samples` is set "
            ' to "True". Defines quantiles of MCMC samples to '
            "add to result frame key of reconstruction.",
            [0.5, 0.68, 0.9],
        )

        # SkyScan specific optional settings
        self.AddParameter(
            "skyscan_seed",
            "Only relevant if `add_skyscan` is set to True. "
            "If provided this I3Frame key will be used to "
            "specify the seed that will be used for the skyscan "
            'reconstructions. If set to "reco", the result of '
            "the reconstruction module will be used.",
            "reco",
        )
        self.AddParameter(
            "skyscan_settings",
            "Only relevant if `add_skyscan` is set "
            ' to "True". These settings are passed on to the '
            " skyscan module `SkyScanner`.",
            {
                "skyscan_nside": 2,
                "skyscan_focus_bounds": [5, 15, 30],
                "skyscan_focus_nsides": [32, 16, 8],
                "skyscan_focus_seeds": [],
                "distribution_settings": {},
            },
        )

    def Configure(self):
        """Configures Module and loads model from file."""
        self.seed_keys = self.GetParameter("seed_keys")
        self.model_names = self.GetParameter("model_names")
        self.model_base_dir = self.GetParameter("model_base_dir")
        self.output_key = self.GetParameter("output_key")
        self.pulse_key = self.GetParameter("pulse_key")
        self.dom_exclusions_key = self.GetParameter("dom_exclusions_key")
        self.time_exclusions_key = self.GetParameter("time_exclusions_key")
        self.add_circular_err = self.GetParameter("add_circular_err")
        self.add_covariances = self.GetParameter("add_covariances")
        self.add_goodness_of_fit = self.GetParameter("add_goodness_of_fit")
        self.add_mcmc_samples = self.GetParameter("add_mcmc_samples")
        self.add_skyscan = self.GetParameter("add_skyscan")
        self.label_key = self.GetParameter("label_key")
        self.snowstorm_key = self.GetParameter("snowstorm_key")
        self.num_threads = self.GetParameter("num_threads")

        # Reconstruction specific settings
        self.minimize_in_trafo_space = self.GetParameter(
            "minimize_in_trafo_space"
        )
        self.parameter_boundaries = self.GetParameter("parameter_boundaries")
        self.minimize_parameter_default_value = self.GetParameter(
            "minimize_parameter_default_value"
        )
        self.minimize_parameter_dict = self.GetParameter(
            "minimize_parameter_dict"
        )
        self.reco_optimizer_interface = self.GetParameter(
            "reco_optimizer_interface"
        )
        self.scipy_optimizer_settings = self.GetParameter(
            "scipy_optimizer_settings"
        )
        self.tf_optimizer_settings = self.GetParameter("tf_optimizer_settings")
        self.goodness_of_fit_settings = self.GetParameter(
            "goodness_of_fit_settings"
        )
        self.mcmc_settings = self.GetParameter("mcmc_settings")
        self.mcmc_quantiles = self.GetParameter("mcmc_quantiles")
        self.skyscan_seed = self.GetParameter("skyscan_seed")
        self.skyscan_settings = self.GetParameter("skyscan_settings")

        self.missing_seed_value_dict = self.GetParameter(
            "missing_seed_value_dict"
        )
        self.missing_seed_value = self.GetParameter("missing_seed_value")

        if "reconstruct_samples" not in self.goodness_of_fit_settings:
            self.goodness_of_fit_settings["reconstruct_samples"] = True
        if "add_per_dom_calculation" not in self.goodness_of_fit_settings:
            self.goodness_of_fit_settings["add_per_dom_calculation"] = True

        if isinstance(self.seed_keys, str):
            self.seed_keys = [self.seed_keys]

        if isinstance(self.model_names, str):
            self.model_names = [self.model_names]

        manager_dirs = []
        for name in self.model_names:
            if self.model_base_dir is not None:
                manager_dirs.append(os.path.join(self.model_base_dir, name))
            else:
                manager_dirs.append(name)

        # Add boundaries (approximate uniform priors)
        if self.parameter_boundaries is not None:
            additional_loss_modules = [
                {
                    "loss_module": "egenerator.loss.snowstorm.SnowstormPriorLossModule",
                    "config": {
                        "sigmas": [],
                        "uniform_parameters": self.parameter_boundaries,
                        "float_precision": "float32",
                    },
                }
            ]
        else:
            additional_loss_modules = None

        # Build and configure SourceManager
        misc_seed_names = list(self.seed_keys)
        if self.skyscan_seed != "reco":
            if self.skyscan_seed not in self.seed_keys:
                misc_seed_names.append(self.skyscan_seed)

        self.manager_configurator = ManagerConfigurator(
            manager_dirs=manager_dirs,
            reco_config_dir=None,
            load_labels=False,
            replace_existing_loss_modules=False,
            additional_loss_modules=additional_loss_modules,
            misc_setting_updates={
                "seed_names": misc_seed_names,
                "missing_value_dict": self.missing_seed_value_dict,
                "missing_value": self.missing_seed_value,
            },
            label_setting_updates={
                "label_key": self.label_key,
                "snowstorm_key": self.snowstorm_key,
            },
            data_setting_updates={
                "pulse_key": self.pulse_key,
                "dom_exclusions_key": self.dom_exclusions_key,
                "time_exclusions_key": self.time_exclusions_key,
            },
            num_threads=self.num_threads,
        )
        self.manager = self.manager_configurator.manager
        self.loss_module = self.manager_configurator.loss_module

        if "I3ParticleMapping" in self.manager.configuration.config["config"]:
            self.i3_mapping = self.manager.configuration.config["config"][
                "I3ParticleMapping"
            ]
        else:
            self.i3_mapping = None

        for model in self.manager.models:
            num_vars, num_total_vars = model.num_variables
            msg = f"\nNumber of Model Variables for {model.name}:\n"
            msg += f"\tFree: {num_vars}\n"
            msg += f"\tTotal: {num_total_vars}"
            log_info(msg)

        # ------------------------------
        # Gather Reconstruction Settings
        # ------------------------------
        # get a list of parameters which are transformed in log-space
        self.log_names = []
        param_tensor = self.manager.data_trafo.data["tensors"]["x_parameters"]
        for i, name in enumerate(self.manager.models[0].parameter_names):
            if param_tensor.trafo_log[i]:
                self.log_names.append(name)

        # get a list of parameters to fit
        fit_parameter_list = [
            self.minimize_parameter_default_value
            for i in range(self.manager.models[0].n_parameters)
        ]
        for name, value in self.minimize_parameter_dict.items():
            fit_parameter_list[self.manager.models[0].get_index(name)] = value
        self.fit_parameter_list = fit_parameter_list

        self.fitted_parameters = []
        for i, n in enumerate(self.manager.models[0].parameter_names):
            if self.fit_parameter_list[i]:
                self.fitted_parameters.append(n)

        self.fitted_param_to_index = {}
        for i, n in enumerate(self.fitted_parameters):
            self.fitted_param_to_index[n] = i

        # parameter input signature
        parameter_tensor_name = "x_parameters"

        # -----------------------
        # Gather skyscan settings
        # -----------------------
        default_settings = {
            "fit_parameter_list": fit_parameter_list,
            "seed_tensor_name": self.skyscan_seed,
            "minimize_in_trafo_space": self.minimize_in_trafo_space,
            "parameter_tensor_name": parameter_tensor_name,
            "reco_optimizer_interface": self.reco_optimizer_interface,
            "scipy_optimizer_settings": self.scipy_optimizer_settings,
            "tf_optimizer_settings": self.tf_optimizer_settings,
        }
        if self.i3_mapping is not None:
            default_settings["zenith_key"] = self.i3_mapping["zenith"]
            default_settings["azimuth_key"] = self.i3_mapping["azimuth"]

        default_settings.update(self.skyscan_settings)
        self.skyscan_settings = default_settings

        # -------------------------
        # Build reconstruction tray
        # -------------------------

        # create reconstruction tray
        self.reco_tray = ReconstructionTray(
            manager=self.manager, loss_module=self.loss_module
        )

        # add reconstruction module
        reco_names = []
        for seed_tensor_name in self.seed_keys:
            reco_name = "reco_" + seed_tensor_name
            reco_names.append(reco_name)

            self.reco_tray.add_module(
                "Reconstruction",
                name=reco_name,
                fit_parameter_list=fit_parameter_list,
                seed_tensor_name=seed_tensor_name,
                minimize_in_trafo_space=self.minimize_in_trafo_space,
                parameter_tensor_name=parameter_tensor_name,
                reco_optimizer_interface=self.reco_optimizer_interface,
                scipy_optimizer_settings=self.scipy_optimizer_settings,
                tf_optimizer_settings=self.tf_optimizer_settings,
            )

        # choose best reconstruction
        self.reco_tray.add_module(
            "SelectBestReconstruction",
            name="reco",
            reco_names=reco_names,
        )

        # add covariance module
        if self.add_covariances:
            self.reco_tray.add_module(
                "CovarianceMatrix",
                name="covariance",
                fit_parameter_list=fit_parameter_list,
                reco_key="reco",
                minimize_in_trafo_space=self.minimize_in_trafo_space,
                parameter_tensor_name=parameter_tensor_name,
            )

        if self.add_goodness_of_fit:
            if self.add_covariances:
                covariance_key = "covariance"
            else:
                covariance_key = None
            self.reco_tray.add_module(
                "GoodnessOfFit",
                name="GoodnessOfFit",
                fit_parameter_list=fit_parameter_list,
                reco_key="reco",
                covariance_key=covariance_key,
                minimize_in_trafo_space=self.minimize_in_trafo_space,
                parameter_tensor_name=parameter_tensor_name,
                **self.goodness_of_fit_settings
            )

        # add circularized angular uncertainty estimation module
        if self.add_circular_err:
            if self.add_covariances:
                covariance_key = "covariance"
            else:
                covariance_key = None

            self.reco_tray.add_module(
                "CircularizedAngularUncertainty",
                name="CircularizedAngularUncertainty",
                fit_parameter_list=fit_parameter_list,
                reco_key="reco",
                covariance_key=covariance_key,
                minimize_in_trafo_space=self.minimize_in_trafo_space,
                parameter_tensor_name=parameter_tensor_name,
                scipy_optimizer_settings=self.scipy_optimizer_settings,
            )

        # add MCMC
        if self.add_mcmc_samples:
            dist_settings = self.mcmc_settings.pop("distribution_settings")
            self.reco_tray.add_module(
                "MarkovChainMonteCarlo",
                name="MarkovChainMonteCarlo",
                fit_parameter_list=fit_parameter_list,
                seed_tensor_name="reco",
                reco_key="reco",
                minimize_in_trafo_space=self.minimize_in_trafo_space,
                parameter_tensor_name=parameter_tensor_name,
                **self.mcmc_settings
            )

            if self.i3_mapping is not None:
                self.reco_tray.add_module(
                    "FitDistributionsOnSphere",
                    name="MCMCDistributions",
                    input_module="MarkovChainMonteCarlo",
                    zenith_key=self.i3_mapping["zenith"],
                    azimuth_key=self.i3_mapping["azimuth"],
                    reco_key="reco",
                    **dist_settings
                )

        if self.add_skyscan:
            dist_settings = self.skyscan_settings.pop("distribution_settings")
            self.reco_tray.add_module(
                "SkyScanner", name="SkyScanner", **self.skyscan_settings
            )

            if self.i3_mapping is not None:
                self.reco_tray.add_module(
                    "FitDistributionsOnSphere",
                    name="SkyScanDistributions",
                    input_module="SkyScanner",
                    zenith_key=self.i3_mapping["zenith"],
                    azimuth_key=self.i3_mapping["azimuth"],
                    reco_key="reco",
                    **dist_settings
                )

    def Physics(self, frame):
        """Apply Event-Generator model to physics frames.

        Parameters
        ----------
        frame : I3Frame
            The current P-Frame.
        """

        # get data batch
        n, data_batch = self.manager.data_handler.get_data_from_frame(frame)
        assert n == 1, "Currently only 1-event at a time is supported"

        # reconstruct data
        results = self.reco_tray.execute(data_batch)

        # --------------
        # write to frame
        # --------------

        # write best fit results to frame
        best_fit = results["reco"]["result"][0]
        best_fit_obj = results["reco"]["result_object"]

        result_dict = dataclasses.I3MapStringDouble()
        for i, name in enumerate(self.manager.models[0].parameter_names):
            result_dict[name] = float(best_fit[i])
        for name in self.reco_tray.module_names:
            result_dict["runtime_" + name] = results[name]["runtime"]
        result_dict["minimization_success"] = best_fit_obj.success
        result_dict["loss_reco"] = float(results["reco"]["loss_reco"])
        result_dict["loss_seed"] = float(results["reco"]["loss_seed"])

        # add an I3Particle
        if self.i3_mapping:
            particle = dataclasses.I3Particle()

            for key, param in self.i3_mapping.items():

                if key in ["x", "y", "z"]:
                    setattr(particle.pos, key, result_dict[param])

                elif key in ["zenith", "azimuth"]:
                    particle.dir = dataclasses.I3Direction(
                        result_dict[self.i3_mapping["zenith"]],
                        result_dict[self.i3_mapping["azimuth"]],
                    )
                else:
                    setattr(particle, key, result_dict[param])

            # transform zenith and azimuth to proper range:
            particle.dir = dataclasses.I3Direction(
                particle.dir.x, particle.dir.y, particle.dir.z
            )

            # set particle shape to infinite track even though this is not
            # necessarily true. This will allow for visualization
            # in steamshovel
            particle.shape = dataclasses.I3Particle.InfiniteTrack

        # write covariance Matrices to frame
        if self.add_covariances:
            for name, value in results["covariance"].items():
                if name == "runtime":
                    result_dict["runtime_covariance"] = float(value)
                else:
                    self.write_cov_matrix(frame, cov=value, cov_name=name)
        else:
            # write covariance matrix from minimizer to frame
            pass

        # write goodness of fit variables to frame
        if self.add_goodness_of_fit:
            result_dict["goodness_of_fit_1sided"] = float(
                results["GoodnessOfFit"]["event_p_value_1sided"]
            )
            result_dict["goodness_of_fit_2sided"] = float(
                results["GoodnessOfFit"]["event_p_value_2sided"]
            )
            result_dict["goodness_loss_sample_min"] = float(
                results["GoodnessOfFit"]["loss_sample_min"]
            )
            result_dict["goodness_loss_sample_max"] = float(
                results["GoodnessOfFit"]["loss_sample_max"]
            )
            result_dict["goodness_loss_sample_mean"] = float(
                results["GoodnessOfFit"]["loss_sample_mean"]
            )
            result_dict["goodness_loss_sample_median"] = float(
                results["GoodnessOfFit"]["loss_sample_median"]
            )
            result_dict["goodness_loss_sample_std"] = float(
                results["GoodnessOfFit"]["loss_sample_std"]
            )
            result_dict["goodness_loss_data_fit"] = float(
                results["GoodnessOfFit"]["loss_data_fit"]
            )
            if self.goodness_of_fit_settings["reconstruct_samples"]:
                self.write_cov_matrix(
                    frame,
                    cov=results["GoodnessOfFit"]["sample_reco_cov"],
                    cov_name="goodness_of_fit",
                )
                for i, n in enumerate(self.manager.models[0].parameter_names):
                    result_dict[n + "_sample_reco_bias"] = float(
                        results["GoodnessOfFit"]["sample_reco_bias"][i]
                    )

            # write per DOM p-values to frame
            if self.goodness_of_fit_settings["add_per_dom_calculation"]:

                # create containers
                map_pvalue1 = dataclasses.I3MapKeyDouble()
                map_pvalue2 = dataclasses.I3MapKeyDouble()

                # extract data from results dict
                dom_p_value1 = results["GoodnessOfFit"]["dom_p_value1"]
                dom_p_value2 = results["GoodnessOfFit"]["dom_p_value2"]

                # loop through DOMs and fill values
                for string in range(86):
                    for om in range(60):
                        om_key = icetray.OMKey(string + 1, om + 1)
                        map_pvalue1[om_key] = float(dom_p_value1[string, om])
                        map_pvalue2[om_key] = float(dom_p_value2[string, om])

                # write to frame
                frame[self.output_key + "_GoodnessOfFit_1sided"] = map_pvalue1
                frame[self.output_key + "_GoodnessOfFit_2sided"] = map_pvalue2

        if self.add_circular_err:
            result_dict["circular_unc"] = float(
                results["CircularizedAngularUncertainty"]["circular_unc"]
            )
            result_dict["runtime_circular_err"] = float(
                results["CircularizedAngularUncertainty"]["runtime"]
            )

        # write MCMC samples to frame
        if self.add_mcmc_samples:
            mcmc_res = results["MarkovChainMonteCarlo"]
            num_accepted = len(mcmc_res["log_prob_values"])
            result_dict["MCMC_acceptance_ratio"] = mcmc_res["acceptance_ratio"]

            # modify azimuth and zenith to be in range
            if self.i3_mapping is not None and num_accepted > 0:
                index_azimuth = self.fitted_param_to_index[
                    self.i3_mapping["azimuth"]
                ]
                index_zenith = self.fitted_param_to_index[
                    self.i3_mapping["zenith"]
                ]

                zenith, azimuth = angles.convert_to_range(
                    zenith=mcmc_res["samples"][:, index_zenith],
                    azimuth=mcmc_res["samples"][:, index_azimuth],
                )
                mcmc_res["samples"][:, index_zenith] = zenith
                mcmc_res["samples"][:, index_azimuth] = azimuth

            # fill in median and quantiles
            for i, n in enumerate(self.fitted_parameters):

                if num_accepted > 0:
                    values = mcmc_res["samples"][:, i]
                else:
                    # create some dummy values so that we fill in NaNs
                    values = np.ones(10) * float("nan")

                result_dict["MCMC_{}_median".format(n)] = float(
                    np.median(values)
                )
                for q in self.mcmc_quantiles:
                    result_dict["MCMC_{}_q{:3.3f}_lower".format(n, q)] = float(
                        np.quantile(values, 0.5 - 0.5 * q)
                    )
                    result_dict["MCMC_{}_q{:3.3f}_upper".format(n, q)] = float(
                        np.quantile(values, 0.5 + 0.5 * q)
                    )

            if num_accepted > 0:

                # create vectors for output quantities
                vectors = {}
                for i, n in enumerate(self.fitted_parameters):
                    vectors[n] = dataclasses.I3VectorFloat(
                        mcmc_res["samples"][:, i]
                    )
                vectors["log_prob_values"] = dataclasses.I3VectorFloat(
                    mcmc_res["log_prob_values"]
                )

                for n, vector in vectors.items():
                    frame[self.output_key + "_MCMC_" + n] = vector

                # get fitted distribution parameters
                base = "MCMC_{}__{}"
                for name, params in results["MCMCDistributions"].items():
                    if name != "runtime":
                        for param, val in params.items():
                            result_dict[base.format(name, param)] = float(val)

        # write SkyScan results to frame
        if self.add_skyscan:
            scan_res = results["SkyScanner"]

            for nside, llh_dict in scan_res["skyscan_llh"].items():

                llh_values = []
                indices = []
                for ipix, llh_val in llh_dict.items():
                    indices.append(int(ipix))
                    llh_values.append(llh_val)

                llh_values = dataclasses.I3VectorFloat(llh_values)
                indices = dataclasses.I3VectorUInt64(indices)

                out_base = self.output_key + "_SkyScan_{:03d}".format(nside)
                frame[out_base + "_loss"] = llh_values
                frame[out_base + "_ipix"] = indices

            # write out info on scan minimum
            result_dict["SkyScan_min_loss"] = float(scan_res["scan_min_val"])
            result_dict["SkyScan_min_nside"] = float(
                scan_res["scan_min_nside"]
            )
            result_dict["SkyScan_min_ipix"] = float(scan_res["scan_min_ipix"])
            assert len(scan_res["scan_min_fit"]) == 1
            scan_min_fit = scan_res["scan_min_fit"][0]

            for i, name in enumerate(self.manager.models[0].parameter_names):
                result_dict["SkyScan_min_" + name] = float(scan_min_fit[i])

            # get fitted distribution parameters
            base = "SkyScan_{}__{}"
            for name, params in results["SkyScanDistributions"].items():
                if name != "runtime":
                    for param, value in params.items():
                        result_dict[base.format(name, param)] = float(value)

        # save to frame
        frame[self.output_key] = result_dict
        if self.i3_mapping:
            frame[self.output_key + "_I3Particle"] = particle

        # push frame to next modules
        self.PushFrame(frame)

    def write_cov_matrix(self, frame, cov, cov_name):
        """Write covariance matrix to the frame.

        Parameters
        ----------
        frame : i3Frame
            The current I3Frame.
        cov : array_like
            The covariance matrix.
        cov_name : str
            The name of the covariance matrix.
            It will be saved to: self.output_key + '_cov_matrix_' + cov_name
        """
        # frame[self.output_key+'_cov_matrix_'+name] = \
        #     dataclasses.I3Matrix(value)
        cov_dict = dataclasses.I3MapStringDouble()
        for i, name_i in enumerate(self.manager.models[0].parameter_names):

            # adjust name to log_* if it unc. is in log-space
            if name_i in self.log_names:
                name_i = "log_" + name_i
            for j, name_j in enumerate(
                self.manager.models[0].parameter_names[i:]
            ):
                j = j + i

                # adjust name to log_* if it unc. is in log-space
                if name_j in self.log_names:
                    name_j = "log_" + name_j

                cov_dict[name_i + "_" + name_j] = float(cov[i, j])
        frame[self.output_key + "_cov_matrix_" + cov_name] = cov_dict
