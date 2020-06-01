from __future__ import division, print_function
import os
import logging
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import timeit
from scipy import optimize
from scipy.stats import chi2

from egenerator import misc
from egenerator.utils import angles, basis_functions
from egenerator.manager.component import Configuration
from egenerator.manager.base import BaseModelManager
from egenerator.manager.reconstruction.tray import ReconstructionTray


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
                                parameter_tensor_name='x_parameters',
                                reduce_to_scalar=True):
        """Compute loss for a chosen set of parameters.

        Parameters
        ----------
        parameters_trafo : tf.Tensor
            The tensor describing the parameters.
            If minimize_in_trafo_space is True, it is also expected that
            parameters_trafo are given in transformed data space.
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
        reduce_to_scalar : bool, optional
            If True, the individual terms of the log likelihood loss will be
            reduced (aggregated) to a scalar loss.
            If False, a list of tensors will be returned that contain the terms
            of the log likelihood. Note that each of the returend tensors may
            have a different shape.

        Returns
        -------
        tf.Tensor or list of tf.Tensor
            The loss for the given data_batch and chosen set of parameters.
            if `reduce_to_scalar` is True:
                Scalar loss
                Shape: []
            else:
                List of tensors defining the terms of the log likelihood
        """
        data_batch_dict = {}
        for i, name in enumerate(self.data_handler.tensors.names):
            data_batch_dict[name] = data_batch[i]

        seed_index = self.data_handler.tensors.get_index(seed)

        # transform seed data if necessary
        if minimize_in_trafo_space:
            seed_trafo = self.data_trafo.transform(
                data=data_batch[seed_index], tensor_name=parameter_tensor_name)
        else:
            seed_trafo = data_batch[seed_index]

        # gather a list of parameters that are to be fitted
        if not np.all(fit_paramater_list):
            unstacked_params_trafo = tf.unstack(parameters_trafo, axis=1)
            unstacked_seed_trafo = tf.unstack(seed_trafo, axis=1)
            all_params = []
            counter = 0
            for i, fit in enumerate(fit_paramater_list):
                if fit:
                    all_params.append(unstacked_params_trafo[counter])
                    counter += 1
                else:
                    all_params.append(unstacked_seed_trafo[i])

            parameters_trafo = tf.stack(all_params, axis=1)

        # unnormalize if minimization is perfomed in trafo space
        if minimize_in_trafo_space:
            parameters = self.data_trafo.inverse_transform(
                data=parameters_trafo, tensor_name=parameter_tensor_name)
        else:
            parameters = parameters_trafo

        data_batch_dict[parameter_tensor_name] = parameters

        loss = None
        for model in self.models:
            result_tensors = model.get_tensors(
                                data_batch_dict,
                                is_training=False,
                                parameter_tensor_name=parameter_tensor_name)

            loss_i = loss_module.get_loss(
                    data_batch_dict,
                    result_tensors,
                    self.data_handler.tensors,
                    model=model,
                    parameter_tensor_name=parameter_tensor_name,
                    reduce_to_scalar=reduce_to_scalar)
            if loss is None:
                loss = loss_i
            else:
                if reduce_to_scalar:
                    loss += loss_i
                else:
                    for index in range(len(loss_i)):
                        loss[index] += loss_i[index]
        return loss

    def get_parameter_loss_function(self, loss_module, input_signature,
                                    fit_paramater_list,
                                    minimize_in_trafo_space=True,
                                    seed=None,
                                    parameter_tensor_name='x_parameters',
                                    reduce_to_scalar=True):
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
        reduce_to_scalar : bool, optional
            If True, the individual terms of the log likelihood loss will be
            reduced (aggregated) to a scalar loss.
            If False, a list of tensors will be returned that contain the terms
            of the log likelihood. Note that each of the returend tensors may
            have a different shape.

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
                    parameter_tensor_name=parameter_tensor_name,
                    reduce_to_scalar=reduce_to_scalar)
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

    def get_opg_estimate_function(self, loss_module, input_signature,
                                  fit_paramater_list,
                                  minimize_in_trafo_space=True,
                                  seed=None,
                                  parameter_tensor_name='x_parameters'):
        """Get a fucntion that returns the outer products of gradients (OPG).

        The outer product of gradients (OPG) estimate can be used in connection
        with the inverse Hessian matrix in order to obtain a robust estimate
        for the covariance matrix.

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

        # get the loss function
        loss_function = self.get_parameter_loss_function(
            loss_module=loss_module,
            input_signature=input_signature,
            fit_paramater_list=fit_paramater_list,
            minimize_in_trafo_space=minimize_in_trafo_space,
            seed=seed,
            parameter_tensor_name=parameter_tensor_name,
            reduce_to_scalar=False)

        @tf.function(input_signature=input_signature)
        def opg_estimate_function(parameters_trafo, data_batch):

            """
            We need to accumulate the Jacobian over colums (xs) in
            the forward accumulator.
            If we did this via back propagation we would need to compute
            the Jacobian over rows (ys) and therefore perform a loop over
            each loss term.

            See:
            https://www.tensorflow.org/api_docs/python/tf/
            autodiff/ForwardAccumulator
            """

            kernel_fprop = []
            for i in range(parameters_trafo.shape[1]):
                tangent = np.zeros([1, parameters_trafo.shape[1]])
                tangent[:, i] = 1
                tangent = tf.convert_to_tensor(
                    tangent, dtype=parameters_trafo.dtype)

                with tf.autodiff.ForwardAccumulator(
                        # parameters for which we want to compute gradients
                        primals=parameters_trafo,
                        # tangent vector which defines the direction, e.g.
                        # parameter (xs) we want to compute the gradients for
                        tangents=tangent) as acc:

                    loss_terms = loss_function(
                        parameters_trafo=parameters_trafo,
                        data_batch=data_batch)

                    loss_terms_concat = tf.concat(
                        values=[tf.reshape(term, [-1]) for term in loss_terms],
                        axis=0)

                    kernel_fprop.append(acc.jvp(loss_terms_concat))

            # shape: [n_terms, n_params, 1]
            kernel_fprop = tf.stack(kernel_fprop, axis=1)[..., tf.newaxis]
            print('kernel_fprop', kernel_fprop)

            # shape: [n_terms, n_params, n_params]
            opg_estimate = tf.linalg.matmul(kernel_fprop, kernel_fprop,
                                            transpose_b=True)
            print('opg_estimate', opg_estimate)
            tf.print('opg_estimate shape', tf.shape(opg_estimate))

            return tf.reduce_sum(opg_estimate, axis=0)

        return opg_estimate_function

    def get_hessian_function(self, loss_module, input_signature,
                             fit_paramater_list,
                             minimize_in_trafo_space=True,
                             seed=None,
                             parameter_tensor_name='x_parameters'):
        """Get a function that returns the  Hessian wrt parameters.

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
            A tensorflow function: f(parameters, data_batch) -> hessian
            that returns the Hessian of the loss with respect to the
            model parameters.
        """

        @tf.function(input_signature=input_signature)
        def hessian_function(parameters_trafo, data_batch):
            loss = self.parameter_loss_function(
                    parameters_trafo=parameters_trafo,
                    data_batch=data_batch,
                    loss_module=loss_module,
                    fit_paramater_list=fit_paramater_list,
                    minimize_in_trafo_space=minimize_in_trafo_space,
                    seed=seed,
                    parameter_tensor_name=parameter_tensor_name)

            hessian = tf.hessians(loss, parameters_trafo)[0]

            # we will limit this to a batch dimension of 1 for now
            # Note: this runs through and works for a batch dimension
            # but it requires some thinking of what the result actually means
            hessian = tf.squeeze(tf.ensure_shape(
                hessian, [1, parameters_trafo.shape[1]]*2))

            return hessian

        return hessian_function

    def reconstruct_events(self, data_batch, loss_module,
                           loss_and_gradients_function,
                           fit_paramater_list,
                           minimize_in_trafo_space=True,
                           seed='x_parameters',
                           parameter_tensor_name='x_parameters',
                           jac=True,
                           method='L-BFGS-B',
                           hessian_function=None,
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
        hessian_function : tf.function, optional
            The tensorflow function:
                f(parameters, data_batch) -> hessian
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
        num_fit_params = np.sum(fit_paramater_list, dtype=int)
        param_tensor = self.data_trafo['tensors'][parameter_tensor_name]
        parameter_dtype = getattr(tf, param_tensor.dtype)
        param_shape = [-1, num_fit_params]

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

            grad_flat = np.reshape(grad, [-1])
            return loss, grad_flat

        if hessian_function is not None:
            def get_hessian(x, data_batch):
                # reshape and convert to tensor
                x = tf.reshape(tf.convert_to_tensor(x, dtype=parameter_dtype),
                               param_shape)
                hessian = hessian_function(x, data_batch)
                hessian = hessian.numpy().astype('float64')
                return hessian

            kwargs['hess'] = get_hessian

        # transform seed if minimization is performed in trafo space
        seed_index = self.data_handler.tensors.get_index(seed)
        seed_tensor = data_batch[seed_index]
        if minimize_in_trafo_space:
            seed_tensor = self.data_trafo.transform(
                data=seed_tensor, tensor_name=parameter_tensor_name)

        # get seed parameters
        if np.all(fit_paramater_list):
            x0 = seed_tensor
        else:
            # get seed parameters
            unstacked_seed = tf.unstack(seed_tensor, axis=1)
            tracked_params = [p for p, fit in
                              zip(unstacked_seed, fit_paramater_list) if fit]
            x0 = tf.stack(tracked_params, axis=1)

        x0_flat = tf.reshape(x0, [-1])
        result = optimize.minimize(fun=func, x0=x0_flat, jac=jac,
                                   method=method,
                                   args=(data_batch,), **kwargs)

        best_fit = np.reshape(result.x, param_shape)
        return best_fit, result

    def tf_reconstruct_events(self, data_batch, loss_module,
                              loss_and_gradients_function,
                              fit_paramater_list,
                              minimize_in_trafo_space=True,
                              seed='x_parameters',
                              parameter_tensor_name='x_parameters',
                              method='bfgs_minimize',
                              hessian_function=None,
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
        hessian_function : tf.function, optional
            The tensorflow function:
                f(parameters, data_batch) -> hessian
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
        num_fit_params = np.sum(fit_paramater_list, dtype=int)
        param_tensor = self.data_trafo['tensors'][parameter_tensor_name]
        parameter_dtype = getattr(tf, param_tensor.dtype)
        param_shape = [-1, num_fit_params]

        if (len(fit_paramater_list) != param_tensor.shape[1]):
            raise ValueError('Wrong length of fit_paramater_list: {!r}'.format(
                len(fit_paramater_list)))

        # transform seed if minimization is performed in trafo space
        seed_index = self.data_handler.tensors.get_index(seed)
        seed_tensor = data_batch[seed_index]
        if minimize_in_trafo_space:
            seed_tensor = self.data_trafo.transform(
                data=seed_tensor, tensor_name=parameter_tensor_name)

        # get seed parameters
        if np.all(fit_paramater_list):
            x0 = seed_tensor
        else:
            # get seed parameters
            unstacked_seed = tf.unstack(seed_tensor, axis=1)
            tracked_params = [p for p, fit in
                              zip(unstacked_seed, fit_paramater_list) if fit]
            x0 = tf.stack(tracked_params, axis=1)

        def const_loss_and_gradients_function(x):
            loss, grad = loss_and_gradients_function(x, data_batch)
            loss = tf.reshape(loss, [1])
            return loss, grad

        if hessian_function is not None:
            raise NotImplementedError(
                'Use of Hessian currently not implemented')

        optimizer = getattr(tfp.optimizer, method)
        otpim_results = optimizer(
            value_and_gradients_function=const_loss_and_gradients_function,
            initial_position=x0)
        return otpim_results.position, otpim_results

    def run_mcmc_on_events(self, initial_position, data_batch, loss_module,
                           parameter_loss_function,
                           fit_paramater_list,
                           minimize_in_trafo_space=True,
                           num_chains=1,
                           seed=None,
                           num_results=100,
                           num_burnin_steps=100,
                           num_parallel_iterations=1,
                           num_steps_between_results=0,
                           method='HamiltonianMonteCarlo',
                           mcmc_seed=42,
                           parameter_tensor_name='x_parameters'):
        """Reconstruct events with tensorflow probability interface.

        Parameters
        ----------
        initial_position : tf.Tensor
            The tensor describing the parameters.
            Shape: [-1, num_params]
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
        num_results : int, optional
            The number of chain steps to perform after burnin phase.
        num_burnin_steps : int, optional
            The number of chain steps to perform for burnin phase.
        num_parallel_iterations : int, optional
            The number of parallel iterations to perform during MCMC chain.
            If reproducible results are required, this must be set to 1!
        num_steps_between_results : int, optional
            The number of steps between accepted results. This applies
            thinning to the sampled point and can reduce correlation.
        method : str, optional
            The MCMC method to use:
            'HamiltonianMonteCarlo', 'RandomWalkMetropolis', ...
        mcmc_seed : int, optional
            The seed value for the MCMC chain.
        parameter_tensor_name : str, optional
            The name of the parameter tensor to use. Default: 'x_parameters'

        Returns
        -------
        tfp optimizer_results
            The results of the minimization

        Raises
        ------
        NotImplementedError
            Description
        ValueError
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
        # step sizes for x, y, z, zenith, azimuth, energy, time
        step_size = [[.5, .5, .5, 0.02, 0.02, 10., 1.]]
        if method == 'HamiltonianMonteCarlo':
            step_size = [[.1, .1, .1, 0.01, 0.02, 10., 1.]]

        if num_params != len(step_size):
            step_size = [[0.1 for p in range(num_params)]]

        step_size = np.array(step_size)

        param_tensor = self.data_trafo['tensors'][parameter_tensor_name]
        parameter_dtype = getattr(tf, param_tensor.dtype)

        if minimize_in_trafo_space:
            for i, trafo in enumerate(param_tensor.trafo_log):
                if trafo:
                    if i != 5:
                        raise NotImplementedError()
                    step_size[0][i] = 0.01
            step_size /= self.data_trafo.data[parameter_tensor_name+'_std']

        step_size = tf.convert_to_tensor(step_size, dtype=parameter_dtype)
        step_size = tf.reshape(step_size, [1, len(fit_paramater_list)])

        # Define transition kernel
        if method == 'HamiltonianMonteCarlo':
            adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
                tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=unnormalized_log_prob,
                    num_leapfrog_steps=3,
                    # num_leapfrog_steps=tf.random.uniform(
                    #     shape=(), minval=1, maxval=30,
                    #     dtype=tf.int32, seed=mcmc_seed),
                    step_size=step_size),
                num_adaptation_steps=int(num_burnin_steps * 0.8))

        elif method == 'NoUTurnSampler':
            adaptive_hmc = tfp.mcmc.NoUTurnSampler(
                    target_log_prob_fn=unnormalized_log_prob,
                    step_size=step_size)

        elif method == 'RandomWalkMetropolis':
            adaptive_hmc = tfp.mcmc.RandomWalkMetropolis(
                target_log_prob_fn=unnormalized_log_prob,
                new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=step_size),
                seed=mcmc_seed)
        else:
            raise ValueError('Unknown method: {!r}'.format(method))

        # define trace function, e.g. which kernel results to keep
        def trace_fn(states, previous_kernel_results):
            pkr = previous_kernel_results
            if method == 'HamiltonianMonteCarlo':
                return (pkr.inner_results.is_accepted,
                        pkr.inner_results.accepted_results.target_log_prob,
                        pkr.inner_results.accepted_results.step_size)

            elif method == 'NoUTurnSampler':
                return (pkr.is_accepted,
                        pkr.target_log_prob)

            elif method == 'RandomWalkMetropolis':
                return (pkr.is_accepted,
                        pkr.accepted_results.target_log_prob)

        @tf.function
        def run_chain():
            # Run the chain (with burn-in).
            samples, trace = tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                num_steps_between_results=num_steps_between_results,
                current_state=initial_position,
                kernel=adaptive_hmc,
                trace_fn=trace_fn,
                parallel_iterations=num_parallel_iterations)
            samples = tf.reshape(samples,
                                 [num_chains*num_results, num_params])
            if minimize_in_trafo_space:
                samples = self.data_trafo.inverse_transform(
                    data=samples, tensor_name=parameter_tensor_name)

            samples = tf.reshape(samples,
                                 [num_results, num_chains, num_params])

            return samples, trace
        return run_chain()

    def reconstruct_testdata(self, config, loss_module):
        """Reconstruct test data events from hdf5 files.

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

        # print out number of model variables
        for model in self.models:
            num_vars, num_total_vars = model.num_variables
            msg = '\nNumber of Model Variables:\n'
            msg += '\tFree: {}\n'
            msg += '\tTotal: {}'
            print(msg.format(num_vars, num_total_vars))

        # get reconstruction config
        reco_config = config['reconstruction_settings']
        minimize_in_trafo_space = reco_config['minimize_in_trafo_space']

        # get a list of parameters to fit
        fit_paramater_list = [reco_config['minimize_parameter_default_value']
                              for i in range(self.models[0].num_parameters)]
        for name, value in reco_config['minimize_parameter_dict'].items():
            fit_paramater_list[self.models[0].get_index(name)] = value

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

        # get concrete function to compute loss
        get_loss = self.get_concrete_function(
            function=self.get_loss,
            input_signature=(test_dataset.element_spec,),
            loss_module=loss_module,
            opt_config={'l1_regularization': 0., 'l2_regularization': 0},
            is_training=False,
            parameter_tensor_name=parameter_tensor_name)

        # ---------------
        # Define Settings
        # ---------------
        calculate_covariance_matrix = True
        estimate_angular_uncertainty = True
        run_mcmc = False
        make_1d_llh_scans = False

        reco_config['mcmc_num_chains'] = 10
        reco_config['mcmc_num_results'] = 100  # 10000
        reco_config['mcmc_num_burnin_steps'] = 30  # 100
        reco_config['mcmc_num_steps_between_results'] = 0
        reco_config['mcmc_num_parallel_iterations'] = 1
        reco_config['mcmc_method'] = 'HamiltonianMonteCarlo'
        # HamiltonianMonteCarlo
        # RandomWalkMetropolis
        # NoUTurnSampler

        plot_file = os.path.splitext(reco_config['reco_output_file'])[0]
        plot_file += '_llh_scan_{event_counter:08d}_{parameter}'

        # -------------------------
        # Build reconstruction tray
        # -------------------------

        # create reconstruction tray
        reco_tray = ReconstructionTray(manager=self, loss_module=loss_module)

        # add reconstruction module
        reco_tray.add_module(
            'Reconstruction',
            name='reco',
            fit_paramater_list=fit_paramater_list,
            seed_tensor_name=reco_config['seed'],
            minimize_in_trafo_space=minimize_in_trafo_space,
            parameter_tensor_name=parameter_tensor_name,
            reco_optimizer_interface=reco_config['reco_optimizer_interface'],
            scipy_optimizer_settings=reco_config['scipy_optimizer_settings'],
            tf_optimizer_settings=reco_config['tf_optimizer_settings'],
        )

        # add covariance module
        if calculate_covariance_matrix:
            reco_tray.add_module(
                'CovarianceMatrix',
                name='covariance',
                fit_paramater_list=fit_paramater_list,
                seed_tensor_name=reco_config['seed'],
                reco_key='reco',
                minimize_in_trafo_space=minimize_in_trafo_space,
                parameter_tensor_name=parameter_tensor_name,
            )

        # add circularized angular uncertainty estimation module
        if estimate_angular_uncertainty:
            reco_tray.add_module(
                'CircularizedAngularUncertainty',
                name='CircularizedAngularUncertainty',
                fit_paramater_list=fit_paramater_list,
                seed_tensor_name=reco_config['seed'],
                reco_key='reco',
                # covariance_key='covariance',
                minimize_in_trafo_space=minimize_in_trafo_space,
                parameter_tensor_name=parameter_tensor_name,
            )

        # add MCMC module
        if run_mcmc:
            reco_tray.add_module(
                'MarkovChainMonteCarlo',
                name='mcmc',
                fit_paramater_list=fit_paramater_list,
                seed_tensor_name=reco_config['seed'],
                reco_key='reco',
                minimize_in_trafo_space=minimize_in_trafo_space,
                parameter_tensor_name=parameter_tensor_name,
                mcmc_num_chains=reco_config['mcmc_num_chains'],
                mcmc_method=reco_config['mcmc_method'],
                mcmc_num_results=reco_config['mcmc_num_results'],
                mcmc_num_burnin_steps=reco_config['mcmc_num_burnin_steps'],
                mcmc_num_steps_between_results=reco_config[
                    'mcmc_num_steps_between_results'],
                mcmc_num_parallel_iterations=reco_config[
                    'mcmc_num_parallel_iterations'],
            )

        # add plotting module
        if make_1d_llh_scans:
            reco_tray.add_module(
                'Visualize1DLikelihoodScan',
                name='visualization',
                fit_paramater_list=fit_paramater_list,
                seed_tensor_name=reco_config['seed'],
                plot_file_template=plot_file,
                reco_key='reco',
                # covariance_key='covariance',
                minimize_in_trafo_space=minimize_in_trafo_space,
                parameter_tensor_name=parameter_tensor_name,
            )
        # -------------------------

        # create empty lists
        cascade_parameters_true = []
        cascade_parameters_reco = []
        cascade_parameters_seed = []
        loss_true_list = []
        loss_reco_list = []
        loss_seed_list = []
        std_devs = []
        std_devs_fit = []
        std_devs_sandwich = []
        std_devs_sandwich_fit = []
        cov_zen_azi_list = []
        cov_fit_zen_azi_list = []
        circular_unc_list = []

        event_counter = 0
        for data_batch in test_dataset:

            # # -------------------
            # # Hack to modify seed
            # # -------------------
            # x0 = data_batch[seed_index]
            # seed_shape = x0.numpy().shape
            # # x0 = np.random.normal(loc=x0.numpy()[0],
            # #                       scale=[300, 300, 300, 0.5, 0.5, 100, 1000],
            # #                       size=[1, 7])
            # # x0[0, 5] = 100

            # # set snowstorm params to expectation
            # x0 = x0.numpy()

            # # # all snowstorm variations
            # # x0[:, 7:10] = 1.0
            # # x0[:, 10] = -0.5
            # # x0[:, 11:36] = 0.
            # # x0[:, 36] = 1.0

            # # wihtout ft
            # x0[:, 7:10] = 1.0
            # x0[:, 10] = 0.
            # x0[:, 11] = 0.

            # x0 = tf.reshape(tf.convert_to_tensor(x0, param_dtype), seed_shape)
            # new_batch = [b for b in data_batch]
            # new_batch[seed_index] = x0
            # data_batch = tuple(new_batch)
            # # -------------------

            # ---------------------------
            # Execute reconstruction tray
            # ---------------------------
            reco_start_t = timeit.default_timer()
            results = reco_tray.execute(data_batch)
            reco_end_t = timeit.default_timer()

            cascade_reco_batch = results['reco']['result']
            cascade_true_batch = data_batch[param_index].numpy()
            cascade_seed_batch = data_batch[seed_index].numpy()

            # -----------------
            # Covariance-Matrix
            # -----------------
            if calculate_covariance_matrix:

                # extract results
                cov = results['covariance']['cov']
                cov_fit = results['covariance']['cov_fit']
                cov_sand = results['covariance']['cov_sand']
                cov_sand_fit = results['covariance']['cov_sand_fit']

                # save correlation between zenith and azimuth
                zen_index = self.models[0].get_index('zenith')
                azi_index = self.models[0].get_index('azimuth')

                cov_zen_azi_list.append(cov[zen_index, azi_index])
                cov_fit_zen_azi_list.append(cov_fit[zen_index, azi_index])

                std_devs_sandwich.append(np.sqrt(np.diag(cov_sand)))
                std_devs_sandwich_fit.append(np.sqrt(np.diag(cov_sand_fit)))
                std_devs.append(np.sqrt(np.diag(cov)))
                std_devs_fit.append(np.sqrt(np.diag(cov_fit)))

                # Write to file
                cov_file = '{}_cov_{:08d}.npy'.format(
                    os.path.splitext(reco_config['reco_output_file'])[0],
                    event_counter)
                # np.save(cov_file, np.stack([cov, cov_fit]))

            # -------------------
            # Angular Uncertainty
            # -------------------
            if estimate_angular_uncertainty:

                circular_unc_list.append(
                    results['CircularizedAngularUncertainty']['circular_unc'])

                # ---------------------
                # write samples to file
                # ---------------------
                if False:
                    df = pd.DataFrame()
                    # df['loss'] = unc_losses
                    df['delta_loss'] = delta_loss
                    df['delta_psi'] = delta_psi

                    param_counter = 0
                    for name in self.models[0].parameter_names:
                        if name == 'azimuth':
                            values = azi
                        elif name == 'zenith':
                            values = zen
                        else:
                            values = unc_results[:, param_counter]
                            param_counter += 1
                        df[name] = values

                    unc_file = '{}_unc_{:08d}.hdf5'.format(
                        os.path.splitext(reco_config['reco_output_file'])[0],
                        event_counter)
                    df.to_hdf(unc_file, key='Variables', mode='w', format='t',
                              data_columns=True)

            # -------------
            # run mcmc test
            # -------------
            if run_mcmc:

                # extract results
                samples = results['mcmc']['samples']
                log_prob_values = results['mcmc']['log_prob_values']
                num_accepted = len(log_prob_values)

                if num_accepted > 0:
                    index_max = np.argmax(log_prob_values)
                    print('\tBest Sample: ' + msg.format(*samples[index_max]))
                    print('\tmin loss_values: {:3.2f}'.format(
                                                        -max(log_prob_values)))

                    sorted_indices = np.argsort(log_prob_values)
                    sorted_log_prob_values = log_prob_values[sorted_indices]
                    sorted_samples = samples[sorted_indices]

                    # ---------------------
                    # write samples to file
                    # ---------------------
                    df = pd.DataFrame()
                    df['log_prob'] = sorted_log_prob_values
                    for i, name in enumerate(self.models[0].parameter_names):
                        df['samples_{}'.format(name)] = sorted_samples[:, i]

                    mcmc_file = '{}_mcmc_{:08d}.hdf5'.format(
                        os.path.splitext(reco_config['reco_output_file'])[0],
                        event_counter)
                    df.to_hdf(mcmc_file, key='Variables', mode='w', format='t',
                              data_columns=True)

                    # -------------
                    # Print Results
                    # -------------
                    central_50p = sorted_samples[int(num_accepted*0.5):]
                    central_90p = sorted_samples[int(num_accepted*0.1):]

                    for i, name in enumerate(self.models[0].parameter_names):
                        min_val_90 = min(central_90p[:, i])
                        max_val_90 = max(central_90p[:, i])
                        min_val_50 = min(central_50p[:, i])
                        max_val_50 = max(central_50p[:, i])

                        msg = '\t{:8.2f}: {:8.2f} {:8.2f} {:8.2f} {:8.2f} [{}]'
                        print(msg.format(cascade_true[i], min_val_90,
                                         min_val_50, max_val_50, max_val_90,
                                         name))
            # -------------

            # Now loop through events in this batch
            for cascade_reco, cascade_seed, cascade_true in zip(
                    cascade_reco_batch,
                    cascade_seed_batch,
                    cascade_true_batch):

                data_batch_seed = list(data_batch)
                data_batch_seed[param_index] = tf.reshape(
                            cascade_seed, [-1, self.models[0].num_parameters])
                data_batch_seed = tuple(data_batch_seed)

                data_batch_reco = list(data_batch)
                data_batch_reco[param_index] = tf.reshape(tf.convert_to_tensor(
                                    cascade_reco, dtype=param_signature.dtype),
                                [-1, self.models[0].num_parameters])
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
                for index, (name, fit) in enumerate(zip(
                        self.models[0].parameter_names, fit_paramater_list)):
                    msg += pattern.format(
                        str(fit),
                        cascade_true[index],
                        cascade_seed[index],
                        cascade_reco[index],
                        cascade_true[index]-cascade_reco[index],
                        name)
                print('At event {} [Reconstruction took {:3.3f}s]'.format(
                    event_counter, reco_end_t - reco_start_t))
                print(msg)

                # keep track of results
                cascade_parameters_true.append(cascade_true)
                cascade_parameters_seed.append(cascade_seed)
                cascade_parameters_reco.append(cascade_reco)

                loss_true_list.append(loss_true)
                loss_seed_list.append(loss_seed)
                loss_reco_list.append(loss_reco)

                # update event counter
                event_counter += 1

        cascade_parameters_true = np.stack(cascade_parameters_true, axis=0)
        cascade_parameters_seed = np.stack(cascade_parameters_seed, axis=0)
        cascade_parameters_reco = np.stack(cascade_parameters_reco, axis=0)

        if calculate_covariance_matrix:
            std_devs_fit = np.stack(std_devs_fit, axis=0)
            std_devs = np.stack(std_devs, axis=0)
            std_devs_sandwich = np.stack(std_devs_sandwich, axis=0)
            std_devs_sandwich_fit = np.stack(std_devs_sandwich_fit, axis=0)

        if estimate_angular_uncertainty:
            circular_unc_list = np.stack(circular_unc_list, axis=0)

        # ----------------
        # create dataframe
        # ----------------
        df_reco = pd.DataFrame()
        for index, param_name in enumerate(self.models[0].parameter_names):
            for name, params in (['', cascade_parameters_true],
                                 ['_reco', cascade_parameters_reco],
                                 ['_seed', cascade_parameters_seed]):
                df_reco[param_name + name] = params[:, index]

        if calculate_covariance_matrix:
            for index, param_name in enumerate(self.models[0].parameter_names):
                for name, unc in (['_unc', std_devs],
                                  ['_unc_fit', std_devs_fit],
                                  ['_unc_sandwhich', std_devs_sandwich],
                                  ['_unc_sandwhich_fit',
                                   std_devs_sandwich_fit],
                                  ):
                    df_reco[param_name + name] = unc[:, index]

            # save correlation between zenith and azimuth
            df_reco['cov_zenith_azimuth'] = cov_zen_azi_list
            df_reco['cov_fit_zenith_azimuth'] = cov_fit_zen_azi_list

        if estimate_angular_uncertainty:
            df_reco['circular_unc'] = circular_unc_list

        df_reco['loss_true'] = loss_true_list
        df_reco['loss_reco'] = loss_reco_list
        df_reco['loss_seed'] = loss_seed_list

        df_reco.to_hdf(reco_config['reco_output_file'],
                       key='Variables', mode='w', format='t',
                       data_columns=True)
