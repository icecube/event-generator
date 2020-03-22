from __future__ import division, print_function
import os
import logging
import tensorflow as tf
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

    def get_loss_and_gradients_function(self, loss_module, input_signature,
                                        fit_paramater_list,
                                        minimize_in_trafo_space=True,
                                        seed=None):
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

        Returns
        -------
        tf.function
            A tensorflow function: f(parameters) -> loss, gradient
            that returns the loss and the gradients of the loss with
            respect to the model parameters.
        """

        @tf.function(input_signature=input_signature)
        def loss_and_gradients_function(parameters_trafo, data_batch):

            data_batch_dict = {}
            for i, name in enumerate(self.data_handler.tensors.names):
                data_batch_dict[name] = data_batch[i]

            seed_index = self.data_handler.tensors.get_index(seed)

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(parameters_trafo)

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
                        data=parameters_trafo, tensor_name='x_parameters')
                else:
                    parameters = parameters_trafo

                data_batch_dict['x_parameters'] = parameters

                result_tensors = self.model.get_tensors(data_batch_dict,
                                                        is_training=False)

                loss = loss_module.get_loss(data_batch_dict,
                                            result_tensors,
                                            self.data_handler.tensors)
            grad = tape.gradient(loss, parameters)
            return loss, grad

        return loss_and_gradients_function

    def reconstruct_events(self, data_batch, loss_module,
                           loss_and_gradients_function,
                           fit_paramater_list,
                           minimize_in_trafo_space=True,
                           seed='x_parameters',
                           jac=True,
                           method='L-BFGS-B',
                           **kwargs):
        """Reconstruct events.

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
        parameter_dtype = getattr(
            tf, self.data_handler.tensors['x_parameters'].dtype)
        param_shape = [-1, self.data_handler.tensors['x_parameters'].shape[1]]
        param_shape = [-1, np.sum(fit_paramater_list, dtype=int)]

        if (len(fit_paramater_list) !=
                self.data_handler.tensors['x_parameters'].shape[1]):
            raise ValueError('Wrong length of fit_paramater_list: {!r}'.format(
                len(fit_paramater_list)))

        # define helper function
        def func(x, data_batch):
            # reshape and convert to tensor
            x = tf.reshape(tf.convert_to_tensor(x, dtype=parameter_dtype),
                           param_shape)
            return [vv.numpy().astype(np.float64) for vv in
                    loss_and_gradients_function(x, data_batch)]

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
            x0 = self.model.data_trafo.transform(data=x0,
                                                 tensor_name='x_parameters')[0]

        result = optimize.minimize(fun=func, x0=x0, jac=jac, method=method,
                                   args=(data_batch,), **kwargs)
        return result

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
        param_index = self.data_handler.tensors.get_index('x_parameters')
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
            is_training=False)
        loss_and_gradients_function = self.get_loss_and_gradients_function(
            input_signature=(param_signature, test_dataset.element_spec),
            loss_module=loss_module,
            fit_paramater_list=fit_paramater_list,
            minimize_in_trafo_space=reco_config['minimize_in_trafo_space'],
            seed=reco_config['seed'])

        # create empty lists
        cascade_parameters_true = []
        cascade_parameters_reco = []
        cascade_parameters_seed = []
        loss_true_list = []
        loss_reco_list = []
        loss_seed_list = []

        for event_counter, data_batch in enumerate(test_dataset):

            result = self.reconstruct_events(
                data_batch, loss_module,
                loss_and_gradients_function=loss_and_gradients_function,
                fit_paramater_list=fit_paramater_list,
                minimize_in_trafo_space=reco_config['minimize_in_trafo_space'],
                seed=reco_config['seed'],
                **reco_config['scipy_optimizer_settings'])

            cascade_true = data_batch[param_index].numpy()[0]
            cascade_seed = data_batch[seed_index].numpy()[0]

            # get reco cascade
            if np.all(fit_paramater_list):
                cascade_reco = result.x
            else:
                # get seed parameters
                cascade_reco = []
                result_counter = 0
                for i, fit in enumerate(fit_paramater_list):
                    if fit:
                        cascade_reco.append(result.x[result_counter])
                        result_counter += 1
                    else:
                        cascade_reco.append(cascade_seed[i])

            # transform back if minimization was performed in trafo space
            if reco_config['minimize_in_trafo_space']:
                cascade_reco = self.model.data_trafo.inverse_transform(
                    data=np.expand_dims(cascade_reco, axis=0),
                    tensor_name='x_parameters')[0]

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
