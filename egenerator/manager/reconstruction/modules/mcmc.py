import timeit
import numpy as np
import tensorflow as tf


class MarkovChainMonteCarlo:

    def __init__(self, manager, loss_module, function_cache,
                 fit_paramater_list,
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
        fit_paramater_list : TYPE
            Description
        seed_tensor_name : TYPE
            Description
        minimize_in_trafo_space : bool, optional
            Description
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

        Raises
        ------
        NotImplementedError
            Description

        Deleted Parameters
        ------------------
        **settings
            Description
        """

        # store settings
        self.manager = manager
        self.fit_paramater_list = fit_paramater_list
        self.minimize_in_trafo_space = minimize_in_trafo_space
        self.parameter_tensor_name = parameter_tensor_name
        self.mcmc_num_chains = mcmc_num_chains
        self.reco_key = reco_key

        # specify a random number generator for reproducibility
        self.rng = np.random.RandomState(random_seed)

        # parameter input signature
        self.param_dtype = getattr(tf, manager.data_trafo['tensors'][
            parameter_tensor_name].dtype)
        param_signature = tf.TensorSpec(
            shape=[None, np.sum(fit_paramater_list, dtype=int)],
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

        # get normal parameter loss function
        func_settings = dict(
            input_signature=(param_signature, data_batch_signature),
            loss_module=loss_module,
            fit_paramater_list=fit_paramater_list,
            minimize_in_trafo_space=minimize_in_trafo_space,
            seed=seed_tensor_name,
            parameter_tensor_name=parameter_tensor_name,
        )

        # Get parameter loss function
        self.parameter_loss_function = function_cache.get(
            'parameter_loss_function', func_settings)

        if self.parameter_loss_function is None:
            self.parameter_loss_function = manager.get_parameter_loss_function(
                **func_settings)
            function_cache.add(self.parameter_loss_function, func_settings)

        @tf.function(input_signature=(param_signature, data_batch_signature))
        def run_mcmc_on_events(initial_position, data_batch):
            return manager.run_mcmc_on_events(
                initial_position=initial_position,
                data_batch=data_batch,
                loss_module=loss_module,
                parameter_loss_function=self.parameter_loss_function,
                fit_paramater_list=fit_paramater_list,
                minimize_in_trafo_space=minimize_in_trafo_space,
                num_chains=mcmc_num_chains,
                seed=seed_tensor_name,
                method=mcmc_method,
                num_results=mcmc_num_results,
                num_burnin_steps=mcmc_num_burnin_steps,
                num_steps_between_results=mcmc_num_steps_between_results,
                num_parallel_iterations=mcmc_num_parallel_iterations,
                parameter_tensor_name=parameter_tensor_name)

        self.run_mcmc_on_events = run_mcmc_on_events

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

        num_params = len(self.fit_paramater_list)
        result_inv = results[self.reco_key]['result']

        assert len(result_inv) == 1
        result_inv = result_inv[0]

        # # 0, 1, 2,      3,       4,      5,    6
        # # x, y, z, zenith, azimuth, energy, time
        # scale = np.array([10., 10., 10., 0.2, 0.2, 0., 20.])
        # scale = 0
        # low = result_inv - scale
        # high = result_inv + scale
        # # low[3] = 0.0
        # # low[4] = 0.0
        # # high[3] = np.pi
        # # high[4] = 2*np.pi
        # low[5] *= 0.9
        # high[5] *= 1.1
        # initial_position = self.rng.uniform(
        #     low=low, high=high, size=[self.mcmc_num_chains, num_params])
        # initial_position = tf.convert_to_tensor(initial_position,
        #                                         dtype=self.param_dtype)

        # if self.minimize_in_trafo_space:
        #     initial_position = self.manager.data_trafo.transform(
        #                             data=initial_position,
        #                             tensor_name=self.parameter_tensor_name)

        initial_position = tf.reshape(
            tf.convert_to_tensor(
                np.tile(result_inv, [self.mcmc_num_chains, 1]),
                dtype=self.param_dtype),
            [self.mcmc_num_chains, len(self.fit_paramater_list)])
        print('initial_position', initial_position)
        print('initial_position.shape', initial_position.shape)
        mcmc_start_t = timeit.default_timer()
        samples, trace = self.run_mcmc_on_events(initial_position, data_batch)
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
        samples = samples[accepted]
        log_prob_values = log_prob_values[accepted]
        print('MCMC Results took {:3.3f}s:'.format(
            mcmc_end_t - mcmc_start_t))
        print('\tAcceptance Ratio: {:2.1f}%'.format(
            (100. * num_accepted) / num_samples))
        msg = '{:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f}'
        if len(trace) > 2:
            print('\tStepsize: ' + msg.format(*step_size))
        msg = '{:1.2f} {:1.2f} {:1.2f} {:1.2f} {:1.2f} {:1.2f} {:1.2f}'

        # gather results
        results = {
            'samples': samples,
            'log_prob_values': log_prob_values,
        }

        return results
