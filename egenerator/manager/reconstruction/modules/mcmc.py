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

        num_params = len(self.fit_parameter_list)

        # get seed: either from seed tensor or from previous results
        if 'result' in results[self.reco_key]:
            # this is a previous reconstruction result
            result_inv = results[self.reco_key]['result']
        else:
            # this could be a seed tensor
            result_inv = results[self.reco_key]

        assert len(result_inv) == 1

        initial_position = np.reshape(
            np.tile(result_inv[0], [self.mcmc_num_chains, 1]),
            [self.mcmc_num_chains, len(self.fit_parameter_list)])
        print('initial_position', initial_position)
        print('initial_position.shape', initial_position.shape)

        if self.minimize_in_trafo_space:
            initial_position = self.manager.data_trafo.transform(
                                    data=initial_position,
                                    tensor_name=self.parameter_tensor_name)
            print('initial_position [norm]', initial_position)

        # get seed parameters
        if np.all(self.fit_parameter_list):
            initial_position = initial_position
        else:
            # get seed parameters
            initial_position = initial_position[..., self.fit_parameter_list]

        print('initial_position.shape [fit_parameter_list]', initial_position.shape)

        initial_position = tf.convert_to_tensor(
            initial_position, dtype=self.param_dtype)

        mcmc_start_t = timeit.default_timer()
        samples, trace = self.run_mcmc_on_events(
            initial_position, data_batch, result_inv)
        mcmc_end_t = timeit.default_timer()

        samples = samples.numpy()
        accepted = trace[0].numpy()
        log_prob_values = trace[1].numpy()
        if len(trace) > 2:
            steps = trace[2].numpy()
            step_size = steps[0][0]
            if self.minimize_in_trafo_space:
                step_size *= self.manager.data_trafo.data[
                                        self.parameter_tensor_name+'_std']

        num_accepted = np.sum(accepted)
        num_samples = samples.shape[0] * samples.shape[1]

        print('samples pre sel:', samples.shape)
        print('log_prob_values pre sel:', log_prob_values.shape)
        samples = samples[accepted]
        log_prob_values = log_prob_values[accepted]
        print('samples post sel:', samples.shape)
        print('log_prob_values post sel:', log_prob_values.shape)

        # invert possible transformation and put full hypothesis together
        samples = trafo.get_reco_result_batch(
            result_trafo=samples,
            seed_tensor=result_inv,
            fit_parameter_list=self.fit_parameter_list,
            minimize_in_trafo_space=self.minimize_in_trafo_space,
            data_trafo=self.manager.data_trafo,
            parameter_tensor_name=self.parameter_tensor_name)

        if self.verbose:
            print('MCMC Results took {:3.3f}s:'.format(
                mcmc_end_t - mcmc_start_t))
            print('\tAcceptance Ratio: {:2.1f}%'.format(
                (100. * num_accepted) / num_samples))
            msg = '{:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f}'
            if len(trace) > 2:
                print('\tStepsize: ' + msg.format(*step_size))

        # gather results
        results = {
            'samples': samples,
            'log_prob_values': log_prob_values,
        }

        return results
