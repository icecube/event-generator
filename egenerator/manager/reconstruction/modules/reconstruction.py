import numpy as np
import tensorflow as tf


class Reconstruction:

    def __init__(self, manager, loss_module, function_cache,
                 fit_paramater_list,
                 seed_tensor_name,
                 minimize_in_trafo_space=True,
                 parameter_tensor_name='x_parameters',
                 reco_optimizer_interface='scipy',
                 scipy_optimizer_settings={'method': 'BFGS'},
                 tf_optimizer_settings={'method': 'bfgs_minimize',
                                        'x_tolerance': 0.001},
                 ):
        """Initialize reconstruction module and setup tensorflow functions.

        Parameters
        ----------
        manager : Manager object
            The SourceManager object.
        loss_module : LossModule object
            The LossModule object to use for the reconstruction steps.
        function_cache : FunctionCache object
            A cache to store and share created concrete tensorflow functions.
        **settings
            Description

        Raises
        ------
        NotImplementedError
            Description
        """

        # store settings
        self.manager = manager
        self.fit_paramater_list = fit_paramater_list
        self.minimize_in_trafo_space = minimize_in_trafo_space
        self.parameter_tensor_name = parameter_tensor_name

        # parameter input signature
        # param_index = manager.data_handler.tensors.get_index(
        #     parameter_tensor_name)
        self.seed_index = manager.data_handler.tensors.get_index(
            seed_tensor_name)
        param_dtype = getattr(tf, manager.data_trafo['tensors'][
            parameter_tensor_name].dtype)
        param_signature = tf.TensorSpec(
            shape=[None, np.sum(fit_paramater_list, dtype=int)],
            dtype=param_dtype)

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
        function_settings = dict(
            input_signature=(param_signature, data_batch_signature),
            loss_module=loss_module,
            fit_paramater_list=fit_paramater_list,
            minimize_in_trafo_space=minimize_in_trafo_space,
            seed=seed_tensor_name,
            parameter_tensor_name=parameter_tensor_name,
        )

        loss_and_gradients_function = function_cache.get(
            'loss_and_gradients_function', function_settings)

        if loss_and_gradients_function is None:
            loss_and_gradients_function = \
                manager.get_loss_and_gradients_function(**function_settings)
            function_cache.add(loss_and_gradients_function, function_settings)

        # choose reconstruction method depending on the optimizer interface
        if reco_optimizer_interface.lower() == 'scipy':
            def reconstruction_method(data_batch):
                return manager.reconstruct_events(
                    data_batch, loss_module,
                    loss_and_gradients_function=loss_and_gradients_function,
                    fit_paramater_list=fit_paramater_list,
                    minimize_in_trafo_space=minimize_in_trafo_space,
                    seed=seed_tensor_name,
                    parameter_tensor_name=parameter_tensor_name,
                    **scipy_optimizer_settings)

        elif reco_optimizer_interface.lower() == 'tfp':
            def reconstruction_method(data_batch):
                return manager.tf_reconstruct_events(
                    data_batch, loss_module,
                    loss_and_gradients_function=loss_and_gradients_function,
                    fit_paramater_list=fit_paramater_list,
                    minimize_in_trafo_space=minimize_in_trafo_space,
                    seed=seed_tensor_name,
                    parameter_tensor_name=parameter_tensor_name,
                    **tf_optimizer_settings)
        else:
            raise ValueError('Unknown interface {!r}. Options are {!r}'.format(
                reco_optimizer_interface, ['scipy', 'tfp']))

        self.reconstruction_method = reconstruction_method

    def execute(self, data_batch, results):
        """Execute reconstruction for a given batch of data.

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
        result_trafo, result_object = self.reconstruction_method(data_batch)

        # invert possible transformation and put full hypothesis together
        result = self.get_reco_result_batch(
            result_trafo=result_trafo,
            data_batch=data_batch,
            fit_paramater_list=self.fit_paramater_list,
            minimize_in_trafo_space=self.minimize_in_trafo_space,
            parameter_tensor_name=self.parameter_tensor_name)

        result_dict = {
            'result': result,
            'result_trafo': result_trafo,
            'result_object': result_object,
        }
        return result_dict

    def get_reco_result_batch(self, result_trafo,
                              data_batch,
                              fit_paramater_list,
                              minimize_in_trafo_space,
                              parameter_tensor_name='x_parameters'):
        """Get the reco result batch.

        This inverts a possible transformation if minimize_in_trafo_space is
        True and also puts the full hypothesis back together if only parts
        of it were fitted

        Parameters
        ----------
        result_trafo : TYPE
            Description
        data_batch : TYPE
            Description
        fit_paramater_list : TYPE
            Description
        minimize_in_trafo_space : TYPE
            Description
        parameter_tensor_name : str, optional
            Description

        Returns
        -------
        tf.Tensor
            The full result batch.

        """
        if minimize_in_trafo_space:
            cascade_seed_batch_trafo = self.manager.data_trafo.transform(
                        data=data_batch[self.seed_index],
                        tensor_name=parameter_tensor_name).numpy()
        else:
            cascade_seed_batch_trafo = cascade_seed_batch

        if np.all(fit_paramater_list):
            cascade_reco_batch = result_trafo
        else:
            # get seed parameters
            cascade_reco_batch = []
            result_counter = 0
            for i, fit in enumerate(fit_paramater_list):
                if fit:
                    cascade_reco_batch.append(result_trafo[:, result_counter])
                    result_counter += 1
                else:
                    cascade_reco_batch.append(cascade_seed_batch_trafo[:, i])
            cascade_reco_batch = np.array(cascade_reco_batch).T

        # transform back if minimization was performed in trafo space
        if minimize_in_trafo_space:
            cascade_reco_batch = self.manager.data_trafo.inverse_transform(
                data=cascade_reco_batch,
                tensor_name=parameter_tensor_name)
        return cascade_reco_batch
