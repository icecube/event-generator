import timeit
import numpy as np
import tensorflow as tf

from egenerator.utils import basis_functions
from egenerator.manager.reconstruction.modules.utils import trafo


class GoodnessOfFit:

    def __init__(self, manager, loss_module, function_cache,
                 fit_paramater_list,
                 reco_key,
                 covariance_key=None,
                 minimize_in_trafo_space=True,
                 parameter_tensor_name='x_parameters',
                 scipy_optimizer_settings={
                    'method': 'L-BFGS-B',
                    'options': {'ftol': 1e-6},
                 },
                 num_samples=50,
                 reconstruct_samples=True,
                 add_per_dom_calculation=True,
                 normalize_by_total_charge=True,
                 random_seed=42):
        """Initialize module and setup tensorflow functions.

        Parameters
        ----------
        manager : Manager object
            The SourceManager object.
        loss_module : LossModule object
            The LossModule object to use for the reconstruction steps.
        function_cache : FunctionCache object
            A cache to store and share created concrete tensorflow functions.
        fit_paramater_list : bool or list of bool, optional
            Indicates whether a parameter is to be minimized.
            The ith element in the list specifies if the ith parameter
            is minimized.
        reco_key : str
            The name of the reconstruction module to use. The covariance
            matrix will be calculated at the best fit point of the specified
            reconstruction module.
        covariance_key : str, optional
            The name of the covariance matrix module to use. The covariance
            is used to estimate the range of delta psi from which to sample.
            If None is provided, the range will be estimated by performing
            a bisection search to find the appropriate range for delta psi.
        minimize_in_trafo_space : bool, optional
            If True, covariance is calculated in transformed and normalized
            parameter space. This is usually desired, because the scales of
            the parameters will all be normalized which should facilitate
            calculation and inversion.
        parameter_tensor_name : str, optional
            The name of the parameter tensor to use. Default: 'x_parameters'.
        scipy_optimizer_settings : dict, optional
            Settings that will be passed on to the scipy.optmize.minimize
            function.
        num_samples : int, optional
            This defines the number of points to sample from the estimated
            posterior distribution as defined by the the best fit
            reconstruction and covariance matrix (if provided).
            These points are used to obtain the test-statistic distribution
            for the null hypothesis: the data is well described by the model.
            The sampled hypotheses are used to simulate new events which are
            then reconstructed if `reconstruct_samples` is True. For each
            simulated event, the reduced log likelihood is calculated which is
            then used as the test-statistic. Afterwards the reduced log
            likelihood is calculated for the actual event. The value is
            compared to the test-statistic distribution to obtain a p-value,
            which quantifies how significantly the data is not described by
            the event-generator model and/or given posterior.
        reconstruct_samples : bool, optional
            If True, each sampled and simulated event from the provided
            poserior is reconstructed. This is usually desired, although
            very slow.
            If False, the sampled events are not reconstructed and the best fit
            point is set to the truth. Note: this will result in a bias of
            obtained likelihood values.
        add_per_dom_calculation : bool, optional
            If True, a goodness of fit value is calculated for every DOM in
            addition to the total event goodness of fit value.
        normalize_by_total_charge : bool, optional
            If True, normalize likelihood by total event charge or by total DOM
            charge for the per DOM likelihood values.
        random_seed : int, optional
            A random seed for the numpy Random State which is used to sample
            the random opening angles (delta psi).
        """

        # store settings
        self.manager = manager
        self.fit_paramater_list = fit_paramater_list
        self.minimize_in_trafo_space = minimize_in_trafo_space
        self.parameter_tensor_name = parameter_tensor_name
        self.num_samples = num_samples
        self.reconstruct_samples = reconstruct_samples
        self.add_per_dom_calculation = add_per_dom_calculation
        self.reco_key = reco_key
        self.covariance_key = covariance_key
        self.normalize_by_total_charge = normalize_by_total_charge

        # get a list of parameters which are transformed in log-space
        param_tensor = self.manager.data_trafo.data['tensors']['x_parameters']
        self.log_params = np.array(param_tensor.trafo_log)

        # specify a random number generator for reproducibility
        self.rng = np.random.RandomState(random_seed)

        # get indices of data tensors
        self.x_pulses_index = self.manager.data_handler.tensors.get_index(
            'x_pulses')
        self.x_pulses_ids_index = self.manager.data_handler.tensors.get_index(
            'x_pulses_ids')

        # get indices of parameters
        self.param_time_index = self.manager.models[0].get_index('time')

        # parameter input signature
        self.param_dtype = getattr(tf, self.manager.data_trafo.data['tensors'][
            parameter_tensor_name].dtype)
        param_signature = tf.TensorSpec(
            shape=[None, np.sum(fit_paramater_list, dtype=int)],
            dtype=self.param_dtype)
        param_signature_full = tf.TensorSpec(
            shape=[None, len(fit_paramater_list)],
            dtype=self.param_dtype)

        data_batch_signature = []
        for tensor in manager.data_handler.tensors.list:
            if tensor.exists:
                shape = tf.TensorShape(tensor.shape)
            else:
                shape = tf.TensorShape(None)
            data_batch_signature.append(tf.TensorSpec(
                shape=shape,
                dtype=getattr(tf, tensor.dtype)))

        data_batch_signature = tuple(data_batch_signature)

        # --------------------------------------------------
        # get concrete functions for reconstruction and loss
        # --------------------------------------------------

        # -------------------------
        # get model tensor function
        # -------------------------
        model_tensor_settings = {'model_index': 0}
        self.model_tensor_function = function_cache.get(
            'model_tensors_function', model_tensor_settings)

        if self.model_tensor_function is None:
            self.model_tensor_function = manager.get_model_tensors_function(
                **model_tensor_settings)
            function_cache.add(
                self.model_tensor_function, model_tensor_settings)

        # ---------------------------
        # Get parameter loss function
        # ---------------------------
        loss_settings = dict(
            input_signature=(
                param_signature, data_batch_signature, param_signature_full),
            loss_module=loss_module,
            fit_paramater_list=fit_paramater_list,
            minimize_in_trafo_space=minimize_in_trafo_space,
            seed=None,
            parameter_tensor_name=parameter_tensor_name,
            reduce_to_scalar=not self.add_per_dom_calculation,
            sort_loss_terms=self.add_per_dom_calculation,
            normalize_by_total_charge=False,  # we want to normalize per DOM
        )

        self.loss_function = function_cache.get(
            'parameter_loss_function', loss_settings)

        if self.loss_function is None:
            self.loss_function = manager.get_parameter_loss_function(
                **loss_settings)
            function_cache.add(self.loss_function, loss_settings)

        # -------------------------------
        # Get loss and gradients function
        # -------------------------------
        function_settings = dict(
            input_signature=(
                param_signature, data_batch_signature, param_signature_full),
            loss_module=loss_module,
            fit_paramater_list=fit_paramater_list,
            minimize_in_trafo_space=minimize_in_trafo_space,
            seed=None,
            parameter_tensor_name=parameter_tensor_name,
        )

        loss_and_gradients_function = function_cache.get(
            'loss_and_gradients_function', function_settings)

        if loss_and_gradients_function is None:
            loss_and_gradients_function = \
                manager.get_loss_and_gradients_function(**function_settings)
            function_cache.add(loss_and_gradients_function, function_settings)
        # -------------------------------

        # choose reconstruction method depending on the optimizer interface
        def reconstruction_method(data_batch, seed_tensor):
            return manager.reconstruct_events(
                data_batch, loss_module,
                loss_and_gradients_function=loss_and_gradients_function,
                fit_paramater_list=fit_paramater_list,
                minimize_in_trafo_space=minimize_in_trafo_space,
                seed=seed_tensor,
                parameter_tensor_name=parameter_tensor_name,
                **scipy_optimizer_settings)

        self.reconstruction_method = reconstruction_method

    def execute(self, data_batch, results):
        """Execute module for a given batch of data.

        Parameters
        ----------
        data_batch : tuple of tf.Tensors
            A data batch which consists of a tuple of tf.Tensors.
        results : dict
            A dictrionary with the results of previous modules.

        Returns
        -------
        TYPE
            Description
        """

        # start time
        t_0 = timeit.default_timer()

        result_trafo = results[self.reco_key]['result_trafo']
        result_inv = results[self.reco_key]['result']

        # The following assumes that there is only one event at a time
        assert len(result_inv) == 1
        result_inv = result_inv[0]

        # --------------------------------------
        # sample event hypotheses from posterior
        # --------------------------------------
        if self.covariance_key is not None:
            cov = results[self.covariance_key]['cov_sand']

            # transform log-parameters to log space
            result_inv_log = np.array(result_inv)
            result_inv_log[self.log_params] = np.log(
                1.0 + result_inv_log[self.log_params])

            sampled_hypotheses = self.rng.multivariate_normal(
                mean=result_inv_log,
                cov=cov,
                size=self.num_samples,
            )

            # revert log-trafo
            sampled_hypotheses[:, self.log_params] = np.exp(
                sampled_hypotheses[:, self.log_params]) - 1.0
        else:
            sampled_hypotheses = np.tile(
                result_inv, reps=[self.num_samples, 1])

        # -----------------------------------------
        # compute expectation from egenerator model
        # -----------------------------------------
        result_tensors = self.model_tensor_function(sampled_hypotheses)

        # draw total charge per DOM and cascade
        dom_charges = self.sample_num_pe(result_tensors)
        # dom_charges = self.rng.poisson(result_tensors['dom_charges'].numpy())
        event_charges = np.sum(dom_charges, axis=(1, 2, 3))

        source_times = sampled_hypotheses[:, self.param_time_index]

        # get cumuluative sum of mixture model contributions
        cum_scale = np.cumsum(
            result_tensors['latent_var_scale'].numpy(), axis=-1)

        latent_var_mu = result_tensors['latent_var_mu'].numpy()
        latent_var_sigma = result_tensors['latent_var_sigma'].numpy()
        latent_var_r = result_tensors['latent_var_r'].numpy()

        # for numerical stability:
        cum_scale[..., -1] = 1.00000001

        # ---------------------------------
        # Iterate through individual events
        # ---------------------------------

        # create empty arrays to store results
        sample_recos_trafo = np.empty_like(sampled_hypotheses)
        sample_event_llh = np.empty(self.num_samples)
        if self.add_per_dom_calculation:
            sample_dom_llh = np.empty([self.num_samples, 86, 60])

        # calculate time needed for each step
        t_sampling = 0.
        t_reconstruction = 0.
        t_loss = 0.

        # Now walk through events: simulate + reconstruct + compute loss
        for event_id in range(self.num_samples):

            # -----------------------------
            # simulate event: sample pulses
            # -----------------------------
            t_1 = timeit.default_timer()

            # figure out if charge quantile also needs to be added
            data_module = self.manager.data_handler.data_module
            add_charge_quantiles = data_module.configuration.config[
                'add_charge_quantiles']

            x_pulses, x_pulses_ids = self.sample_event_pulses(
                rng=self.rng,
                dom_charges=dom_charges[event_id],
                cum_scale=cum_scale[event_id],
                source_time=source_times[event_id],
                latent_mu=latent_var_mu[event_id],
                latent_sigma=latent_var_sigma[event_id],
                latent_r=latent_var_r[event_id],
                add_charge_quantiles=add_charge_quantiles)

            # create data_batch_dict based on these new pulses
            data_batch_new = [t for t in data_batch]
            data_batch_new[self.x_pulses_index] = x_pulses
            data_batch_new[self.x_pulses_ids_index] = x_pulses_ids
            data_batch_new = tuple(data_batch_new)
            tf_seed_tensor = tf.convert_to_tensor(
                np.expand_dims(sampled_hypotheses[event_id], axis=0),
                self.param_dtype)

            t_2 = timeit.default_timer()

            # -----------------------------------------------
            # reconstruct event based on the simulated pulses
            # -----------------------------------------------
            if self.reconstruct_samples:
                sample_result_trafo, res_obj = self.reconstruction_method(
                    data_batch_new, tf_seed_tensor)
            else:
                if self.minimize_in_trafo_space:
                    sample_result_trafo = self.manager.data_trafo.transform(
                        data=tf_seed_tensor.numpy(),
                        tensor_name=self.parameter_tensor_name)
                else:
                    sample_result_trafo = tf_seed_tensor.numpy()

            assert len(sample_result_trafo) == 1
            sample_recos_trafo[event_id] = sample_result_trafo[0]
            t_3 = timeit.default_timer()

            # ---------------------------------------------------
            # compute log likelihood for each DOM and total event
            # ---------------------------------------------------
            sample_loss = self.loss_function(
                parameters_trafo=sample_result_trafo,
                data_batch=data_batch_new,
                seed=tf_seed_tensor)

            if self.add_per_dom_calculation:
                # we need to sort through and compute loss for each DOM
                sample_dom_llh[event_id] = sample_loss[2].numpy()[0]
                sample_event_llh[event_id] = (
                    np.sum(sample_loss[2].numpy()[0])
                    + sample_loss[1].numpy()[0] + sample_loss[0].numpy())
            else:
                # we just have the scalar loss for the whole event
                sample_event_llh[event_id] = sample_loss.numpy()

            t_4 = timeit.default_timer()

            # accumulate times
            t_sampling += t_2 - t_1
            t_reconstruction += t_3 - t_2
            t_loss += t_4 - t_3
            # print('Sample loop took: {:3.3f}s'.format(t_4 - t_1))
            # print('\t Pulse Sampling: {:3.3f}s'.format(t_2 - t_1))
            # print('\t Reconstruction: {:3.3f}s'.format(t_3 - t_2))
            # print('\t Loss Calculation: {:3.3f}s'.format(t_4 - t_3))

        # invert possible transformation and put full hypothesis together
        sample_recos = trafo.get_reco_result_batch(
            result_trafo=sample_recos_trafo,
            seed_tensor=sampled_hypotheses,
            fit_paramater_list=self.fit_paramater_list,
            minimize_in_trafo_space=self.minimize_in_trafo_space,
            data_trafo=self.manager.data_trafo,
            parameter_tensor_name=self.parameter_tensor_name)
        sample_diff = sampled_hypotheses - sample_recos
        sample_reco_bias = np.mean(sample_diff, axis=0)
        sample_reco_cov = np.cov(sample_diff.T)

        # compute loss for actual data
        data_loss = self.loss_function(
            parameters_trafo=result_trafo,
            data_batch=data_batch,
            seed=tf_seed_tensor)

        if self.add_per_dom_calculation:
            data_dom_llh = data_loss[2].numpy()[0]
            data_event_llh = (
                np.sum(data_loss[2].numpy()[0])
                + data_loss[1].numpy()[0] + data_loss[0].numpy())
        else:
            data_event_llh = data_loss

        # Normalize likelihood values by total event and DOM charge
        if self.normalize_by_total_charge:
            eps = 1.
            sample_event_llh /= event_charges + eps
            data_event_llh /= np.sum(
                data_batch['x_dom_charge'], axis=(1, 2, 3)) + eps

            if self.add_per_dom_calculation:
                data_dom_llh /= data_batch['x_dom_charge'][..., 0] + eps
                sample_dom_llh /= dom_charges[..., 0] + eps

        # ---------------------------------------------------------
        # compare to test-statistic distribution to compute p-value
        # ---------------------------------------------------------
        event_p_value1, event_p_value2 = self.compute_p_value(
            sample_event_llh, data_event_llh)

        if self.add_per_dom_calculation:
            dom_p_value1 = np.empty_like(data_dom_llh)
            dom_p_value2 = np.empty_like(data_dom_llh)

            # walk through DOMs
            for string in range(86):
                for om in range(60):
                    p_value1, p_value2 = self.compute_p_value(
                        sample_dom_llh[:, string, om],
                        data_dom_llh[string, om],
                    )
                    dom_p_value1[string, om] = p_value1
                    dom_p_value2[string, om] = p_value2

        # Calculate elapsed time
        t_5 = timeit.default_timer()
        t_p_value = t_5 - t_4
        print('GoodnessOfFit elapsed time: {:3.3f}s'.format(t_5 - t_0))
        print('\t Pulse Sampling: {:3.3f}s'.format(t_sampling))
        print('\t Reconstruction: {:3.3f}s'.format(t_reconstruction))
        print('\t Loss Calculation: {:3.3f}s'.format(t_loss))
        print('\t P-Value Calculation: {:3.3f}s'.format(t_p_value))

        print('data_event_llh', data_event_llh)
        print(
            'sample_event_llh',
            np.min(sample_event_llh),
            np.mean(sample_event_llh),
            np.max(sample_event_llh),
        )
        print('event_p_value one-sided', event_p_value1)
        print('event_p_value two-sided', event_p_value2)

        # -----------------------------------------
        # write everything to the result dictionary
        # -----------------------------------------
        results = {
            'event_p_value_1sided': event_p_value1,
            'event_p_value_2sided': event_p_value2,
            'num_samples': self.num_samples,
            'sampled_hypotheses': sampled_hypotheses,
        }
        if self.add_per_dom_calculation:
            results['dom_p_value1'] = dom_p_value1
            results['dom_p_value2'] = dom_p_value2
        if self.reconstruct_samples:
            results.update({
                'sample_recos': sample_recos,
                'sample_reco_bias': sample_reco_bias,
                'sample_reco_cov': sample_reco_cov,
            })

        return results

    def compute_p_value(self, sample_llh, data_llh):
        """Compute the p-value for a given ts-distribution `sample_llh`.

        Parameters
        ----------
        sample_llh : array_like
            The test-statistic distribution.
        data_llh : float
            The test-statistic value for which to compute the one and two
            sided p-values

        Returns
        -------
        float
            1-sided p-value
        float
            2-sided p-value
        """
        num_samples = len(sample_llh)
        half_num = num_samples/2.
        sample_llh_sorted = np.sort(sample_llh)
        idx = np.searchsorted(sample_llh_sorted, data_llh)
        p_value1 = float(num_samples - idx) / num_samples
        p_value2 = (half_num - abs(half_num - idx)) / (half_num)

        return p_value1, p_value2

    def sample_num_pe(self, result_tensors):
        """Sample number of PE for each DOM.

        Parameters
        ----------
        result_tensors : dict of tf.Tensor
            The dictionary of result tensors from the event-generator model.

        Returns
        -------
        array_like
            The number of PE for each DOM
        """
        dom_charges = basis_functions.sample_from_negative_binomial(
            rng=self.rng,
            mu=result_tensors['dom_charges'].numpy(),
            alpha_or_var=result_tensors['dom_charges_variance'].numpy(),
            param_is_alpha=False,
        )
        return dom_charges

    def sample_event_pulses(self, rng, dom_charges, cum_scale, source_time,
                            latent_mu, latent_sigma, latent_r,
                            add_charge_quantiles=False):
        """Sample pulses from PDF and create a I3RecoPulseSeriesMap

        Parameters
        ----------
        rng : RandomService
            The random number service to use.
        dom_charges : array_like
            The sampled charges at each DOM.
            Shape: [86, 60, 1]
        cum_scale : array_like
            The cumulative sum of latent scales for each mixture model comp.
            Shape: [86, 60]
        source_time : float
            The time of the source.
        latent_mu : array_like
            The latent mus of the AG mixture model.
            Shape: [86, 60]
        latent_sigma : array_like
            The latent sigmas of the AG mixture model.
            Shape: [86, 60]
        latent_r : array_like
            The latent rs of the AG mixture model.
            Shape: [86, 60]
        add_charge_quantiles : bool, optional
            If True, charge quantiles are added to the pulses.

        Returns
        -------
        array_like
            The sampled pulses.
            Shape: [n_pulses, 2 or 3]
        array_like
            The ids of the sampled pulses.
            Shape: [n_pulses, 3]
        """

        # this is meant to run for a single event, make sure this is true
        assert len(dom_charges.shape) == 3

        x_pulses = []
        x_pulses_ids = []

        # walk through DOMs
        for string in range(86):
            for om in range(60):

                num_pe = dom_charges[string, om, 0]
                if num_pe <= 0:
                    continue

                # we will uniformly choose the charge and then correct
                # again to obtain correct total charge
                # ToDo: figure out actual chage distribution of pulses!
                pulse_charges = rng.uniform(0.25, 1.75, size=num_pe)
                pulse_charges *= num_pe / np.sum(pulse_charges)

                # for each pulse, draw 2 random numbers which we will need
                # to figure out which mixtue model component to choose
                # and at what time the pulse gets injected
                rngs = rng.uniform(size=(num_pe, 2))

                idx = np.searchsorted(cum_scale[string, om], rngs[:, 0])

                # get parameters for chosen asymmetric gaussian
                pulse_mu = latent_mu[string, om, idx]
                pulse_sigma = latent_sigma[string, om, idx]
                pulse_r = latent_r[string, om, idx]

                # caclulate time of pulse
                pulse_times = basis_functions.asymmetric_gauss_ppf(
                    rngs[:, 1], mu=pulse_mu, sigma=pulse_sigma, r=pulse_r)

                # fix scale
                pulse_times *= 1000.

                # fix offset
                pulse_times += source_time

                # sort pulses in time
                sorted_indices = np.argsort(pulse_times)
                pulse_times = pulse_times[sorted_indices]
                pulse_charges = pulse_charges[sorted_indices]

                # append pulses
                pulse_elements = [pulse_charges, pulse_times]
                if add_charge_quantiles:
                    pulse_elements.append(
                        np.cumsum(pulse_charges) / np.sum(pulse_charges))
                pulses = np.stack(pulse_elements, axis=1)
                pulses_ids = np.tile((0, string, om), reps=[num_pe, 1])

                x_pulses.append(pulses)
                x_pulses_ids.append(pulses_ids)

        x_pulses = np.concatenate(x_pulses)
        x_pulses_ids = np.concatenate(x_pulses_ids)

        return x_pulses, x_pulses_ids
