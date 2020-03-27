from __future__ import division, print_function
import os
import logging
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import timeit
from scipy import optimize

from egenerator import misc
from egenerator.manager.component import Configuration
from egenerator.manager.base import BaseModelManager


class SourceManager(BaseModelManager):

    def __init__(self, logger=None):
        """Initializes ModelManager object.

        Parameters
        ----------
        logger : logging.logger, optional
            A logging instance.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(SourceManager, self).__init__(logger=self._logger)

    def parameter_loss_function(self, parameters_trafo, data_batch,
                                loss_module, fit_paramater_list,
                                minimize_in_trafo_space=True,
                                seed=None,
                                parameter_tensor_name='x_parameters'):
        """Compute loss for a chosen set of parameters.

        Parameters
        ----------
        parameters_trafo : tf.Tensor
            The tensor describing the parameters.
            Shape: [-1, num_params]
        data_batch : tuple of tf.Tensor
            The tf.data.Dataset batch.
        loss_module : LossComponent
            A loss component that is used to compute the loss. The component
            must provide a
            loss_module.get_loss(data_batch_dict, result_tensors)
            method.
        fit_paramater_list : bool or list of bool, optional
            Indicates whether a parameter is to be minimized.
            The ith element in the list specifies if the ith parameter
            is minimized.
        minimize_in_trafo_space : bool, optional
            If True, minimization is performed in transformed and normalized
            parameter space. This is usually desired, because the scales of
            the parameters will all be normalized which should facilitate
            minimization.
        seed : str, optional
            If a fit_paramater_list is provided with at least one 'False'
            entry, the seed name must also be provided.
            The seed is the name of the data tensor by which the reconstruction
            is seeded.
        parameter_tensor_name : str, optional
            The name of the parameter tensor to use. Default: 'x_parameters'

        Returns
        -------
        tf.scalar
            The loss for the given data_batch and chosen set of parameters.
        """
        data_batch_dict = {}
        for i, name in enumerate(self.data_handler.tensors.names):
            data_batch_dict[name] = data_batch[i]

        seed_index = self.data_handler.tensors.get_index(seed)

        # gather a list of parameters that are to be fitted
        if not np.all(fit_paramater_list):
            unstacked_params = tf.unstack(parameters_trafo, axis=1)
            unstacked_seed = tf.unstack(data_batch[seed_index], axis=1)
            all_params = []
            counter = 0
            for i, fit in enumerate(fit_paramater_list):
                if fit:
                    all_params.append(unstacked_params[counter])
                    counter += 1
                else:
                    all_params.append(unstacked_seed[i])

            parameters_trafo = tf.stack(all_params, axis=1)

        # unnormalize if minimization is perfomed in trafo space
        if minimize_in_trafo_space:
            parameters = self.model.data_trafo.inverse_transform(
                data=parameters_trafo, tensor_name=parameter_tensor_name)
        else:
            parameters = parameters_trafo

        data_batch_dict[parameter_tensor_name] = parameters

        result_tensors = self.model.get_tensors(
                                data_batch_dict,
                                is_training=False,
                                parameter_tensor_name=parameter_tensor_name)

        loss = loss_module.get_loss(data_batch_dict,
                                    result_tensors,
                                    self.data_handler.tensors)
        return loss

    def get_parameter_loss_function(self, loss_module, input_signature,
                                    fit_paramater_list,
                                    minimize_in_trafo_space=True,
                                    seed=None,
                                    parameter_tensor_name='x_parameters'):
        """Get a function that returns the loss for a chosen set of parameters.

        Parameters
        ----------
        loss_module : LossComponent
            A loss component that is used to compute the loss. The component
            must provide a
            loss_module.get_loss(data_batch_dict, result_tensors)
            method.
        input_signature : tf.TensorSpec or nested tf.TensorSpec
            The input signature of the parameters and data_batch arguments
        fit_paramater_list : bool or list of bool, optional
            Indicates whether a parameter is to be minimized.
            The ith element in the list specifies if the ith parameter
            is minimized.
        minimize_in_trafo_space : bool, optional
            If True, minimization is performed in transformed and normalized
            parameter space. This is usually desired, because the scales of
            the parameters will all be normalized which should facilitate
            minimization.
        seed : str, optional
            If a fit_paramater_list is provided with at least one 'False'
            entry, the seed name must also be provided.
            The seed is the name of the data tensor by which the reconstruction
            is seeded.
        parameter_tensor_name : str, optional
            The name of the parameter tensor to use. Default: 'x_parameters'

        Returns
        -------
        tf.function
            A tensorflow function: f(parameters, data_batch) -> loss
            that returns the loss for the given data_batch and the chosen
            set of parameters.
        """

        @tf.function(input_signature=input_signature)
        def parameter_loss_function(parameters_trafo, data_batch):

            loss = self.parameter_loss_function(
                    parameters_trafo=parameters_trafo,
                    data_batch=data_batch,
                    loss_module=loss_module,
                    fit_paramater_list=fit_paramater_list,
                    minimize_in_trafo_space=minimize_in_trafo_space,
                    seed=seed,
                    parameter_tensor_name=parameter_tensor_name)
            return loss

        return parameter_loss_function

    def get_loss_and_gradients_function(self, loss_module, input_signature,
                                        fit_paramater_list,
                                        minimize_in_trafo_space=True,
                                        seed=None,
                                        parameter_tensor_name='x_parameters'):
        """Get a function that returns the loss and gradients wrt parameters.

        Parameters
        ----------
        loss_module : LossComponent
            A loss component that is used to compute the loss. The component
            must provide a
            loss_module.get_loss(data_batch_dict, result_tensors)
            method.
        input_signature : tf.TensorSpec or nested tf.TensorSpec
            The input signature of the parameters and data_batch arguments
        fit_paramater_list : bool or list of bool, optional
            Indicates whether a parameter is to be minimized.
            The ith element in the list specifies if the ith parameter
            is minimized.
        minimize_in_trafo_space : bool, optional
            If True, minimization is performed in transformed and normalized
            parameter space. This is usually desired, because the scales of
            the parameters will all be normalized which should facilitate
            minimization.
        seed : str, optional
            If a fit_paramater_list is provided with at least one 'False'
            entry, the seed name must also be provided.
            The seed is the name of the data tensor by which the reconstruction
            is seeded.
        parameter_tensor_name : str, optional
            The name of the parameter tensor to use. Default: 'x_parameters'

        Returns
        -------
        tf.function
            A tensorflow function: f(parameters, data_batch) -> loss, gradient
            that returns the loss and the gradients of the loss with
            respect to the model parameters.
        """

        @tf.function(input_signature=input_signature)
        def loss_and_gradients_function(parameters_trafo, data_batch):

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(parameters_trafo)

                loss = self.parameter_loss_function(
                    parameters_trafo=parameters_trafo,
                    data_batch=data_batch,
                    loss_module=loss_module,
                    fit_paramater_list=fit_paramater_list,
                    minimize_in_trafo_space=minimize_in_trafo_space,
                    seed=seed,
                    parameter_tensor_name=parameter_tensor_name)

            grad = tape.gradient(loss, parameters_trafo)
            return loss, grad

        return loss_and_gradients_function

    def reconstruct_events(self, data_batch, loss_module,
                           loss_and_gradients_function,
                           fit_paramater_list,
                           minimize_in_trafo_space=True,
                           seed='x_parameters',
                           parameter_tensor_name='x_parameters',
                           jac=True,
                           method='L-BFGS-B',
                           **kwargs):
        """Reconstruct events with scipy.optimize.minimize interface.

        Parameters
        ----------
        data_batch : tuple of tf.Tensor
            A tuple of tensors. This is the batch received from the tf.Dataset.
        loss_module : LossComponent
            A loss component that is used to compute the loss. The component
            must provide a
            loss_module.get_loss(data_batch_dict, result_tensors)
            method.
        loss_and_gradients_function : tf.function
            The tensorflow function:
                f(parameters, data_batch) -> loss, gradients
        fit_paramater_list : bool or list of bool, optional
            Indicates whether a parameter is to be minimized.
            The ith element in the list specifies if the ith parameter
            is minimized.
        minimize_in_trafo_space : bool, optional
            If True, minimization is performed in transformed and normalized
            parameter space. This is usually desired, because the scales of
            the parameters will all be normalized which should facilitate
            minimization.
        seed : str, optional
            Name of seed tensor
        parameter_tensor_name : str, optional
            The name of the parameter tensor to use. Default: 'x_parameters'
        jac : bool, optional
            Passed on to scipy.optimize.minimize
        method : str, optional
            Passed on to scipy.optimize.minimize
        **kwargs
            Keyword arguments that will be passed on to scipy.optimize.minimize

        Returns
        -------
        scipy.optimize.minimize results
            The results of the minimization

        Raises
        ------
        ValueError
            Description
        """
        param_tensor = self.data_handler.tensors[parameter_tensor_name]
        parameter_dtype = getattr(tf, param_tensor.dtype)
        param_shape = [-1, param_tensor.shape[1]]
        param_shape = [-1, np.sum(fit_paramater_list, dtype=int)]

        if (len(fit_paramater_list) != param_tensor.shape[1]):
            msg = 'Wrong length of fit_paramater_list: {!r} != {!r}'
            raise ValueError(msg.format(param_tensor.shape[1],
                                        len(fit_paramater_list)))

        # define helper function
        def func(x, data_batch):
            # reshape and convert to tensor
            x = tf.reshape(tf.convert_to_tensor(x, dtype=parameter_dtype),
                           param_shape)
            loss, grad = loss_and_gradients_function(x, data_batch)
            loss = loss.numpy().astype('float64')
            grad = grad.numpy().astype('float64')
            assert len(grad) == 1
            return loss, grad[0]

        # get seed parameters
        seed_index = self.data_handler.tensors.get_index(seed)
        if np.all(fit_paramater_list):
            x0 = data_batch[seed_index]
        else:
            # get seed parameters
            unstacked_seed = tf.unstack(data_batch[seed_index], axis=1)
            tracked_params = [p for p, fit in
                              zip(unstacked_seed, fit_paramater_list) if fit]
            x0 = tf.stack(tracked_params, axis=1)

        # transform seed if minimization is performed in trafo space
        if minimize_in_trafo_space:
            x0 = self.model.data_trafo.transform(
                data=x0, tensor_name=parameter_tensor_name)[0]

        result = optimize.minimize(fun=func, x0=x0, jac=jac, method=method,
                                   args=(data_batch,), **kwargs)
        return result.x, result

    def tf_reconstruct_events(self, data_batch, loss_module,
                              loss_and_gradients_function,
                              fit_paramater_list,
                              minimize_in_trafo_space=True,
                              seed='x_parameters',
                              parameter_tensor_name='x_parameters',
                              method='bfgs_minimize',
                              **kwargs):
        """Reconstruct events with tensorflow probability interface.

        Parameters
        ----------
        data_batch : tuple of tf.Tensor
            A tuple of tensors. This is the batch received from the tf.Dataset.
        loss_module : LossComponent
            A loss component that is used to compute the loss. The component
            must provide a
            loss_module.get_loss(data_batch_dict, result_tensors)
            method.
        loss_and_gradients_function : tf.function
            The tensorflow function:
                f(parameters, data_batch) -> loss, gradients
        fit_paramater_list : bool or list of bool, optional
            Indicates whether a parameter is to be minimized.
            The ith element in the list specifies if the ith parameter
            is minimized.
        minimize_in_trafo_space : bool, optional
            If True, minimization is performed in transformed and normalized
            parameter space. This is usually desired, because the scales of
            the parameters will all be normalized which should facilitate
            minimization.
        seed : str, optional
            Name of seed tensor
        parameter_tensor_name : str, optional
            The name of the parameter tensor to use. Default: 'x_parameters'
        method : str, optional
            The tensorflow probability optimizer. Must be part of tfp.optimizer
        **kwargs
            Keyword arguments that will be passed on to the tensorflow
            probability optimizer.

        Returns
        -------
        tfp optimizer_results
            The results of the minimization

        Raises
        ------
        ValueError
            Description
        """
        param_tensor = self.data_handler.tensors[parameter_tensor_name]
        parameter_dtype = getattr(tf, param_tensor.dtype)
        param_shape = [-1, param_tensor.shape[1]]
        param_shape = [-1, np.sum(fit_paramater_list, dtype=int)]

        if (len(fit_paramater_list) != param_tensor.shape[1]):
            raise ValueError('Wrong length of fit_paramater_list: {!r}'.format(
                len(fit_paramater_list)))

        # get seed parameters
        seed_index = self.data_handler.tensors.get_index(seed)
        if np.all(fit_paramater_list):
            x0 = data_batch[seed_index]
        else:
            # get seed parameters
            unstacked_seed = tf.unstack(data_batch[seed_index], axis=1)
            tracked_params = [p for p, fit in
                              zip(unstacked_seed, fit_paramater_list) if fit]
            x0 = tf.stack(tracked_params, axis=1)

        # transform seed if minimization is performed in trafo space
        if minimize_in_trafo_space:
            x0 = self.model.data_trafo.transform(
                data=x0, tensor_name=parameter_tensor_name)

        def const_loss_and_gradients_function(x):
            loss, grad = loss_and_gradients_function(x, data_batch)
            loss = tf.reshape(loss, [1])
            return loss, grad

        optimizer = getattr(tfp.optimizer, method)
        otpim_results = optimizer(
            value_and_gradients_function=const_loss_and_gradients_function,
            initial_position=x0)
        return otpim_results.position[0], otpim_results

    def run_mcmc_on_events(self, initial_position, data_batch, loss_module,
                           parameter_loss_function,
                           fit_paramater_list,
                           minimize_in_trafo_space=True,
                           num_chains=1,
                           seed=None,
                           parameter_tensor_name='x_parameters'):
        """Reconstruct events with tensorflow probability interface.

        Parameters
        ----------
        initial_position : tf.Tensor
            The tensor describing the parameters.
        data_batch : tuple of tf.Tensor
            A tuple of tensors. This is the batch received from the tf.Dataset.
        loss_module : LossComponent
            A loss component that is used to compute the loss. The component
            must provide a
            loss_module.get_loss(data_batch_dict, result_tensors)
            method.
        parameter_loss_function : tf.function
            The tensorflow function:
                f(parameters, data_batch) -> loss
        fit_paramater_list : bool or list of bool, optional
            Indicates whether a parameter is to be minimized.
            The ith element in the list specifies if the ith parameter
            is minimized.
        minimize_in_trafo_space : bool, optional
            If True, minimization is performed in transformed and normalized
            parameter space. This is usually desired, because the scales of
            the parameters will all be normalized which should facilitate
            minimization.
        num_chains : int, optional
            Number of chains to run
        seed : str, optional
            Name of seed tensor
        parameter_tensor_name : str, optional
            The name of the parameter tensor to use. Default: 'x_parameters'
        Shape: [-1, num_params]

        Returns
        -------
        tfp optimizer_results
            The results of the minimization

        Raises
        ------
        NotImplementedError
            Description
        """

        num_params = initial_position.shape[1]
        initial_position = tf.ensure_shape(initial_position,
                                           [num_chains, num_params])

        def unnormalized_log_prob(x):
            """This needs to be the *positive* log(prob).
            Since our loss is negative llh, we need to subtract this
            """
            # unstack chains
            x = tf.reshape(x, [num_chains, 1, num_params])
            log_prob_list = []
            for i in range(num_chains):
                log_prob_list.append(-parameter_loss_function(
                    x[i], data_batch))
            return tf.stack(log_prob_list, axis=0)

        # Initialize the HMC transition kernel.
        num_results = int(1e2)
        num_burnin_steps = int(1e2)
        # step sizes for x, y, z, zenith, azimuth, energy, time
        step_size = np.array([[.1, .1, .1, 0.001, 0.001, 10., 1.]])

        param_tensor = self.data_handler.tensors[parameter_tensor_name]
        parameter_dtype = getattr(tf, param_tensor.dtype)

        if minimize_in_trafo_space:
            for i, trafo in enumerate(param_tensor.trafo_log):
                if trafo:
                    if i != 5:
                        raise NotImplementedError()
                    step_size[0][i] = 0.01
            step_size /= self.model.data_trafo.data[
                                        parameter_tensor_name+'_std']

        step_size = tf.convert_to_tensor(step_size, dtype=parameter_dtype)
        step_size = tf.reshape(step_size, [1, len(fit_paramater_list)])

        # # get seed parameters
        # seed_index = self.data_handler.tensors.get_index(seed)
        # if np.all(fit_paramater_list):
        #     x0 = data_batch[seed_index]
        # else:
        #     # get seed parameters
        #     unstacked_seed = tf.unstack(data_batch[seed_index], axis=1)
        #     tracked_params = [p for p, fit in
        #                       zip(unstacked_seed, fit_paramater_list)
        #                       if fit]
        #     x0 = tf.stack(tracked_params, axis=1)

        # # transform seed if minimization is performed in trafo space
        # if minimize_in_trafo_space:
        #     x0 = self.model.data_trafo.transform(
        #         data=x0, tensor_name=parameter_tensor_name)

        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=unnormalized_log_prob,
                num_leapfrog_steps=3,
                step_size=step_size),
            # tfp.mcmc.NoUTurnSampler(
            #     target_log_prob_fn=unnormalized_log_prob,
            #     step_size=step_size,
            # ),
            num_adaptation_steps=int(num_burnin_steps * 0.8))

        def trace_fn(states, previous_kernel_results):
            pkr = previous_kernel_results
            return (pkr.inner_results.is_accepted,
                    pkr.inner_results.accepted_results.target_log_prob,
                    pkr.inner_results.accepted_results.step_size)

        # Run the chain (with burn-in).
        @tf.function
        def run_chain():
            # Run the chain (with burn-in).
            samples, trace = tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                current_state=initial_position,
                kernel=adaptive_hmc,
                trace_fn=trace_fn)
            samples = tf.reshape(samples,
                                 [num_chains*num_results, num_params])
            if minimize_in_trafo_space:
                samples = self.model.data_trafo.inverse_transform(
                    data=samples, tensor_name=parameter_tensor_name)

            samples = tf.reshape(samples,
                                 [num_results, num_chains, num_params])

            return samples, trace
        return run_chain()
        # ------------------

    def reconstruct_testdata(self, config, loss_module):
        """Reconstruct test data events.

        Parameters
        ----------
        config: dict
            A config describing all of the settings for the training script.
            Amongst others, this config must contain:

            train_iterator_settings : dict
                The settings for the training data iterator that will be
                created from the data handler.
            validation_iterator_settings : dict
                The settings for the validation data iterator that will be
                created from the data handler.
            training_settings : dict
                Optimization configuration with settings for the optimizer
                and regularization.

        loss_module : LossComponent
            A loss component that defines the loss function. The loss component
            must provide the method
                loss_module.get_loss(data_batch_dict, result_tensors)
        """

        self.assert_configured(True)

        if config['data_iterator_settings']['test']['batch_size'] != 1:
            raise NotImplementedError('Only supports batch size of 1.')

        # print out number of model variables
        num_vars, num_total_vars = self.model.num_variables
        msg = '\nNumber of Model Variables:\n'
        msg += '\tFree: {}\n'
        msg += '\tTotal: {}'
        print(msg.format(num_vars, num_total_vars))

        # get reconstruction config
        reco_config = config['reconstruction_settings']
        minimize_in_trafo_space = reco_config['minimize_in_trafo_space']

        # get a list of parameters to fit
        fit_paramater_list = [reco_config['minimize_parameter_default_value']
                              for i in range(self.model.num_parameters)]
        for name, value in reco_config['minimize_parameter_dict'].items():
            fit_paramater_list[self.model.get_index(name)] = value

        # create directory if needed
        directory = os.path.dirname(reco_config['reco_output_file'])
        if not os.path.exists(directory):
            os.makedirs(directory)
            self._logger.info('Creating directory: {!r}'.format(directory))

        test_dataset = iter(self.data_handler.get_tf_dataset(
            **config['data_iterator_settings']['test']))

        # parameter input signature
        parameter_tensor_name = reco_config['parameter_tensor_name']
        param_index = self.data_handler.tensors.get_index(
                                                        parameter_tensor_name)
        seed_index = self.data_handler.tensors.get_index(reco_config['seed'])
        param_dtype = test_dataset.element_spec[param_index].dtype
        param_signature = tf.TensorSpec(
            shape=[None, np.sum(fit_paramater_list, dtype=int)],
            dtype=param_dtype)

        # --------------------------------------------------
        # get concrete functions for reconstruction and loss
        # --------------------------------------------------
        get_loss = self.get_concrete_function(
            function=self.get_loss,
            input_signature=(test_dataset.element_spec,),
            loss_module=loss_module,
            opt_config={'l1_regularization': 0., 'l2_regularization': 0},
            is_training=False,
            parameter_tensor_name=parameter_tensor_name)
        loss_and_gradients_function = self.get_loss_and_gradients_function(
            input_signature=(param_signature, test_dataset.element_spec),
            loss_module=loss_module,
            fit_paramater_list=fit_paramater_list,
            minimize_in_trafo_space=minimize_in_trafo_space,
            seed=reco_config['seed'],
            parameter_tensor_name=parameter_tensor_name)

        # choose reconstruction method depending on the optimizer interface
        if reco_config['reco_optimizer_interface'].lower() == 'scipy':
            def reconstruction_method(data_batch):
                return self.reconstruct_events(
                    data_batch, loss_module,
                    loss_and_gradients_function=loss_and_gradients_function,
                    fit_paramater_list=fit_paramater_list,
                    minimize_in_trafo_space=minimize_in_trafo_space,
                    seed=reco_config['seed'],
                    parameter_tensor_name=parameter_tensor_name,
                    **reco_config['scipy_optimizer_settings'])

        elif reco_config['reco_optimizer_interface'].lower() == 'tfp':
            # @tf.function(input_signature=(test_dataset.element_spec,))
            def reconstruction_method(data_batch):
                return self.tf_reconstruct_events(
                    data_batch, loss_module,
                    loss_and_gradients_function=loss_and_gradients_function,
                    fit_paramater_list=fit_paramater_list,
                    minimize_in_trafo_space=minimize_in_trafo_space,
                    seed=reco_config['seed'],
                    parameter_tensor_name=parameter_tensor_name,
                    **reco_config['tf_optimizer_settings'])
        else:
            msg = 'Unknown interface {!r}. Options are {!r}'
            raise ValueError(msg.format(
                reco_config['reco_optimizer_interface'], ['scipy', 'tfp']))

        # ------------------
        # Build MCMC Sampler
        # ------------------
        run_mcmc = False
        if run_mcmc:
            reco_config['mcmc_num_chains'] = 5

            parameter_loss_function = self.get_parameter_loss_function(
                    input_signature=(param_signature,
                                     test_dataset.element_spec),
                    loss_module=loss_module,
                    fit_paramater_list=fit_paramater_list,
                    minimize_in_trafo_space=minimize_in_trafo_space,
                    seed=reco_config['seed'],
                    parameter_tensor_name=parameter_tensor_name)

            @tf.function(input_signature=(param_signature,
                                          test_dataset.element_spec))
            def run_mcmc_on_events(initial_position, data_batch):
                return self.run_mcmc_on_events(
                    initial_position=initial_position,
                    data_batch=data_batch,
                    loss_module=loss_module,
                    parameter_loss_function=parameter_loss_function,
                    fit_paramater_list=fit_paramater_list,
                    minimize_in_trafo_space=minimize_in_trafo_space,
                    num_chains=reco_config['mcmc_num_chains'],
                    seed=reco_config['seed'],
                    parameter_tensor_name=parameter_tensor_name)
        # ------------------

        # create empty lists
        cascade_parameters_true = []
        cascade_parameters_reco = []
        cascade_parameters_seed = []
        loss_true_list = []
        loss_reco_list = []
        loss_seed_list = []

        for event_counter, data_batch in enumerate(test_dataset):

            # reconstruct event
            result, result_obj = reconstruction_method(data_batch)

            cascade_true = data_batch[param_index].numpy()[0]
            cascade_seed = data_batch[seed_index].numpy()[0]

            # -------------
            # run mcmc test
            # -------------
            if run_mcmc:
                num_params = len(fit_paramater_list)

                if minimize_in_trafo_space:
                    result_inv = self.model.data_trafo.inverse_transform(
                            data=np.expand_dims(result, axis=0),
                            tensor_name=parameter_tensor_name)[0]
                else:
                    result_inv = result

                # 0, 1, 2,      3,       4,      5,    6
                # x, y, z, zenith, azimuth, energy, time
                scale = np.array([5., 5., 2., 0.1, 0.1, 0., 10.])
                low = result_inv - scale
                high = result_inv + scale
                # low[3] = 0.0
                # low[4] = 0.0
                # high[3] = np.pi
                # high[4] = 2*np.pi
                low[5] *= 0.5
                high[5] *= 2.0
                initial_position = np.random.uniform(
                    low=low, high=high,
                    size=[reco_config['mcmc_num_chains'], num_params])
                initial_position = tf.convert_to_tensor(initial_position,
                                                        dtype=param_dtype)

                if minimize_in_trafo_space:
                    initial_position = self.model.data_trafo.transform(
                                            data=initial_position,
                                            tensor_name=parameter_tensor_name)

                # initial_position = tf.reshape(
                #     tf.convert_to_tensor(result, dtype=param_dtype),
                #     [1, len(fit_paramater_list)])
                # print('initial_position', initial_position)
                # print('initial_position.shape', initial_position.shape)
                samples, trace = run_mcmc_on_events(initial_position,
                                                    data_batch)
                samples = samples.numpy()
                accepted = trace[0].numpy()
                log_prob_values = trace[1].numpy()
                steps = trace[2].numpy()

                num_accepted = np.sum(accepted)
                num_samples = np.prod(samples.shape)
                samples = samples[accepted]
                log_prob_values = log_prob_values[accepted]
                print('Acceptance Ratio', float(num_accepted) / num_samples)

                if num_accepted > 0:
                    index_max = np.argmax(log_prob_values)
                    print('Best Sample:', samples[index_max])
                    print('min loss_values', -max(log_prob_values))
                    sorted_indices = np.argsort(log_prob_values)

                    central_50p = samples[sorted_indices[
                                                    int(num_accepted*0.5):]]
                    central_90p = samples[sorted_indices[
                                                    int(num_accepted*0.1):]]

                    for i, name in enumerate(self.model.parameter_names):
                        min_val_90 = min(central_90p[:, i])
                        max_val_90 = max(central_90p[:, i])
                        min_val_50 = min(central_50p[:, i])
                        max_val_50 = max(central_50p[:, i])

                        msg = '\t{:8.2f}: {:8.2f} {:8.2f} {:8.2f} {:8.2f} [{}]'
                        print(msg.format(cascade_true[i], min_val_90,
                                         min_val_50, max_val_50, max_val_90,
                                         name))
            # -------------

            # get reco cascade
            if np.all(fit_paramater_list):
                cascade_reco = result
            else:
                # get seed parameters
                cascade_reco = []
                result_counter = 0
                for i, fit in enumerate(fit_paramater_list):
                    if fit:
                        cascade_reco.append(result[result_counter])
                        result_counter += 1
                    else:
                        cascade_reco.append(cascade_seed[i])

            # transform back if minimization was performed in trafo space
            if reco_config['minimize_in_trafo_space']:
                cascade_reco = self.model.data_trafo.inverse_transform(
                    data=np.expand_dims(cascade_reco, axis=0),
                    tensor_name=parameter_tensor_name)[0]

            data_batch_seed = list(data_batch)
            data_batch_seed[param_index] = tf.reshape(
                                cascade_seed, [-1, self.model.num_parameters])
            data_batch_seed = tuple(data_batch_seed)

            data_batch_reco = list(data_batch)
            data_batch_reco[param_index] = tf.reshape(tf.convert_to_tensor(
                                cascade_reco, dtype=param_signature.dtype),
                            [-1, self.model.num_parameters])
            data_batch_reco = tuple(data_batch_reco)

            loss_true = get_loss(data_batch).numpy()
            loss_seed = get_loss(data_batch_seed).numpy()
            loss_reco = get_loss(data_batch_reco).numpy()

            # Print result to console
            msg = '\t{:6s} {:>10s} {:>10s} {:>10s} {:>10s}'.format(
                'Fitted', 'True', 'Seed', 'Reco', 'Diff')
            pattern = '\n\t{:6s} {:10.2f} {:10.2f} {:10.2f} {:10.2f} [{}]'
            msg += pattern.format('', loss_true, loss_seed, loss_reco,
                                  loss_true - loss_reco, 'Loss')
            for index, (name, fit) in enumerate(zip(self.model.parameter_names,
                                                    fit_paramater_list)):
                msg += pattern.format(str(fit),
                                      cascade_true[index],
                                      cascade_seed[index],
                                      cascade_reco[index],
                                      cascade_true[index]-cascade_reco[index],
                                      name)
            print('At event {}'.format(event_counter))
            print(msg)

            # keep track of results
            cascade_parameters_true.append(cascade_true)
            cascade_parameters_seed.append(cascade_seed)
            cascade_parameters_reco.append(cascade_reco)

            loss_true_list.append(loss_true)
            loss_seed_list.append(loss_seed)
            loss_reco_list.append(loss_reco)

        cascade_parameters_true = np.stack(cascade_parameters_true, axis=0)
        cascade_parameters_seed = np.stack(cascade_parameters_seed, axis=0)
        cascade_parameters_reco = np.stack(cascade_parameters_reco, axis=0)

        # ----------------
        # create dataframe
        # ----------------
        df_reco = pd.DataFrame()
        for index, param_name in enumerate(self.model.parameter_names):
            for name, params in (['', cascade_parameters_true],
                                 ['_reco', cascade_parameters_reco],
                                 ['_seed', cascade_parameters_seed]):
                df_reco[param_name + name] = params[:, index]

        df_reco['loss_true'] = loss_true_list
        df_reco['loss_reco'] = loss_reco_list
        df_reco['loss_seed'] = loss_seed_list

        df_reco.to_hdf(reco_config['reco_output_file'],
                       key='Variables', mode='w', format='t',
                       data_columns=True)
