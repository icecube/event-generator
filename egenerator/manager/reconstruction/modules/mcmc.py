import timeit
import numpy as np
import tensorflow as tf

from egenerator.manager.reconstruction.modules.utils import trafo


class MarkovChainMonteCarlo:

    def __init__(self, manager, loss_module, function_cache,
                 fit_parameter_list,
                 seed_tensor_name,
                 reco_key,
                 minimize_in_trafo_space=True,
                 parameter_tensor_name='x_parameters',
                 mcmc_num_chains=10,
                 mcmc_num_results=100,
                 mcmc_num_burnin_steps=30,
                 mcmc_num_steps_between_results=0,
                 mcmc_num_parallel_iterations=1,
                 mcmc_method='HamiltonianMonteCarlo',
                 mcmc_step_size=0.01,
                 mcmc_seed_randomization=0.01,
                 random_seed=42,
                 verbose=True,
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
        fit_parameter_list : TYPE
            Description
        seed_tensor_name : TYPE
            Description
        reco_key : TYPE
            Description
        minimize_in_trafo_space : bool, optional
            If True, the MCMC is performed in transformed and normalized
            parameter space. This is usually desired, because the scales of
            the parameters will all be normalized which should facilitate
            finding proper samples.
        parameter_tensor_name : str, optional
            Description
        mcmc_num_chains : int, optional
            Description
        mcmc_num_results : int, optional
            Description
        mcmc_num_burnin_steps : int, optional
            Description
        mcmc_num_steps_between_results : int, optional
            Description
        mcmc_num_parallel_iterations : int, optional
            Description
        mcmc_method : str, optional
            HamiltonianMonteCarlo
            RandomWalkMetropolis
            NoUTurnSampler
            EMCEE
        mcmc_step_size : float or array_like or str, optional
            The step size for the parameters may be provided as a list of
            float for each parameter (in original physics parameter space,
            but in log10 for variables fit in log10 if
            `minimize_in_trafo_space` is set to true),
            a single float for all parameters (in transformed parameter space),
            or as a string for one of the implemented methods consisting
            of: [].
        mcmc_seed_randomization : float or array_like or str, optional
            The randomization to apply to the initial MCMC seed position.
            The randomization values may be provided as a list of
            float for each parameter (in original physics parameter space,
            but in log10 for variables fit in log10 if
            `minimize_in_trafo_space` is set to true),
            a single float for all parameters (in transformed parameter space),
            or as a string for one of the implemented methods consisting
            of: [].
        random_seed : int, optional
            Description
        verbose : bool, optional
            If True, additional information will be printed to the console.
        """

        # store settings
        self.manager = manager
        self.fit_parameter_list = fit_parameter_list
        self.minimize_in_trafo_space = minimize_in_trafo_space
        self.parameter_tensor_name = parameter_tensor_name
        self.mcmc_num_chains = mcmc_num_chains
        self.mcmc_method = mcmc_method
        self.mcmc_seed_randomization = mcmc_seed_randomization
        self.reco_key = reco_key
        self.seed_tensor_name = seed_tensor_name
        self.verbose = verbose

        # specify a random number generator for reproducibility
        self.rng = np.random.RandomState(random_seed)

        # parameter input signature
        self.param_dtype = getattr(tf, manager.data_trafo.data['tensors'][
            parameter_tensor_name].dtype)
        param_signature = tf.TensorSpec(
            shape=[None, np.sum(fit_parameter_list, dtype=int)],
            dtype=self.param_dtype)
        param_signature_full = tf.TensorSpec(
            shape=[None, len(fit_parameter_list)],
            dtype=self.param_dtype)

        data_batch_signature = manager.data_handler.get_data_set_signature()

        # get normal parameter loss function
        func_settings = dict(
            input_signature=(
                param_signature, data_batch_signature, param_signature_full),
            loss_module=loss_module,
            fit_parameter_list=fit_parameter_list,
            minimize_in_trafo_space=minimize_in_trafo_space,
            seed=None,
            parameter_tensor_name=parameter_tensor_name,
        )

        # Get parameter loss function
        self.parameter_loss_function = function_cache.get(
            'parameter_loss_function', func_settings)

        if self.parameter_loss_function is None:
            self.parameter_loss_function = manager.get_parameter_loss_function(
                **func_settings)
            function_cache.add(self.parameter_loss_function, func_settings)

        # ------------------------------
        # MCMC with python package emcee
        # ------------------------------
        if self.mcmc_method.lower() == 'emcee':
            import emcee

            def log_prob(parameters_trafo, data_batch, seed):
                llh = -self.parameter_loss_function(
                    parameters_trafo=[parameters_trafo],
                    data_batch=data_batch,
                    seed=seed,
                )
                return llh.numpy()

            def run_mcmc_on_events(initial_position, data_batch, seed):
                sampler = emcee.EnsembleSampler(
                    mcmc_num_chains,
                    ndim=np.sum(fit_parameter_list, dtype=int),
                    log_prob_fn=log_prob,
                    args=[data_batch, seed],
                )
                sampler.run_mcmc(
                    initial_position,
                    mcmc_num_results,
                    progress=False,
                )

                if mcmc_num_burnin_steps in [None, -1]:
                    tau = sampler.get_autocorr_time(quiet=True)
                    discard = 2*int(max(tau))
                else:
                    discard = mcmc_num_burnin_steps

                flat_samples = sampler.get_chain(
                    discard=discard,
                    thin=mcmc_num_steps_between_results,
                    flat=True,
                )
                flat_llh = sampler.get_log_prob(
                    discard=discard,
                    thin=mcmc_num_steps_between_results,
                    flat=True,
                )
                return flat_samples, (sampler, flat_llh)

        # --------------------------------
        # MCMC with tensorflow probability
        # --------------------------------
        else:
            @tf.function(input_signature=(
                param_signature, data_batch_signature, param_signature_full))
            def run_mcmc_on_events(initial_position, data_batch, seed):
                return manager.run_mcmc_on_events(
                    initial_position=initial_position,
                    data_batch=data_batch,
                    loss_module=loss_module,
                    parameter_loss_function=self.parameter_loss_function,
                    fit_parameter_list=fit_parameter_list,
                    minimize_in_trafo_space=minimize_in_trafo_space,
                    num_chains=mcmc_num_chains,
                    method=mcmc_method,
                    num_results=mcmc_num_results,
                    num_burnin_steps=mcmc_num_burnin_steps,
                    num_steps_between_results=mcmc_num_steps_between_results,
                    step_size=mcmc_step_size,
                    num_parallel_iterations=mcmc_num_parallel_iterations,
                    parameter_tensor_name=parameter_tensor_name,
                    seed=seed)

        self.run_mcmc_on_events = run_mcmc_on_events

    def execute(self, data_batch, results, **kwargs):
        """Execute module for a given batch of data.

        Parameters
        ----------
        data_batch : tuple of array_like
            A batch of data consisting of a tuple of data arrays.
        results : dict
            A dictrionary with the results of previous modules.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        TYPE
            Description
        """
        # get seed: either from seed tensor or from previous results
        if 'result' in results[self.reco_key]:
            # this is a previous reconstruction result
            result_inv = np.array(results[self.reco_key]['result'])
        else:
            # this could be a seed tensor
            result_inv = np.array(results[self.reco_key])

        assert len(result_inv) == 1

        initial_position = np.reshape(
            np.tile(result_inv[0], [self.mcmc_num_chains, 1]),
            [self.mcmc_num_chains, len(self.fit_parameter_list)])

        # get randomization values for seed
        n_params = len(self.fit_parameter_list)
        if isinstance(self.mcmc_seed_randomization, float):
            # randomization value given in transformed coordinates
            seed_rand_trafo = np.array([
                self.mcmc_seed_randomization for p in range(n_params)])
            seed_rand = seed_rand_trafo * self.manager.data_trafo.data[
                self.parameter_tensor_name+'_std']

        elif isinstance(self.mcmc_seed_randomization, str):
            raise NotImplementedError(self.mcmc_seed_randomization)

        else:
            assert len(self.mcmc_seed_randomization) == n_params
            seed_rand = self.mcmc_seed_randomization
            seed_rand_trafo = seed_rand / self.manager.data_trafo.data[
                self.parameter_tensor_name+'_std']

        if self.minimize_in_trafo_space:
            initial_position = self.manager.data_trafo.transform(
                                    data=initial_position,
                                    tensor_name=self.parameter_tensor_name)

            # randomize seed
            initial_position += self.rng.normal(
                loc=0.0, scale=seed_rand_trafo, size=initial_position.shape)
        else:
            # randomize seed
            initial_position += self.rng.normal(
                loc=0.0, scale=seed_rand, size=initial_position.shape)

        # get seed parameters
        if np.all(self.fit_parameter_list):
            initial_position = initial_position
        else:
            # get seed parameters
            initial_position = initial_position[..., self.fit_parameter_list]

        initial_position = tf.convert_to_tensor(
            initial_position, dtype=self.param_dtype)

        mcmc_start_t = timeit.default_timer()
        samples, info = self.run_mcmc_on_events(
            initial_position, data_batch, result_inv)
        mcmc_end_t = timeit.default_timer()

        if self.mcmc_method.lower() == 'emcee':
            (sampler, log_prob_values) = info
            acceptance_ratio = np.mean(sampler.acceptance_fraction)

        else:
            trace = info
            samples = samples.numpy()
            accepted = trace[0].numpy()
            log_prob_values = trace[1].numpy()
            if len(trace) > 2:
                steps = trace[2].numpy()
                step_size_trafo = steps[0][0]
                if self.minimize_in_trafo_space:
                    step_size = step_size_trafo * self.manager.data_trafo.data[
                        self.parameter_tensor_name+'_std'][
                            self.fit_parameter_list]
                else:
                    step_size = step_size_trafo

            num_accepted = np.sum(accepted)
            num_samples = samples.shape[0] * samples.shape[1]
            acceptance_ratio = float(num_accepted) / num_samples

            samples = samples[accepted]
            log_prob_values = log_prob_values[accepted]

        # invert possible transformation and put full hypothesis together
        samples = trafo.get_reco_result_batch(
            result_trafo=samples,
            seed_tensor=np.tile(result_inv[0], [len(samples), 1]),
            fit_parameter_list=self.fit_parameter_list,
            minimize_in_trafo_space=self.minimize_in_trafo_space,
            data_trafo=self.manager.data_trafo,
            parameter_tensor_name=self.parameter_tensor_name)

        if self.verbose:
            print('MCMC Results took {:3.3f}s:'.format(
                mcmc_end_t - mcmc_start_t))
            print('\tAcceptance Ratio: {:2.1f}%'.format(100.*acceptance_ratio))
            msg = ''
            for s in step_size:
                msg += '{:1.4f} '
            if len(info) > 2 and mcmc_method.lower() != 'emcee':
                print('\tStepsize [sampled space]: ' + msg.format(
                    *step_size_trafo))
                print('\tStepsize [true space]: ' + msg.format(*step_size))

        # gather results
        results = {
            'samples': samples,
            'log_prob_values': log_prob_values,
            'acceptance_ratio': acceptance_ratio,
        }

        return results
