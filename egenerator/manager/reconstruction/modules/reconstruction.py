import tensorflow as tf


class Reconstruction:

    def __init__(self, manager, loss_module, tf_functions,
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
        tf_functions : dict
            Dictionary of created tf functions. These are saved such that they
            may be reused without having to create new ones.
            Structure is as follows:

            tf_functions = {
                func_name1: [(settings1, func1), (settings2, func2)],
                func_name2: [(settings3, func3), (settings4, func4)],
            }

            where func1 and func2 are based of the same function, but use
            different settings. Sampe applies to func3 and func4.
        **settings
            Description

        Raises
        ------
        NotImplementedError
            Description
        """

        # parameter input signature
        param_index = manager.data_handler.tensors.get_index(
                                                        parameter_tensor_name)
        seed_index = manager.data_handler.tensors.get_index(seed_tensor_name)
        param_dtype = getattr(
            tf, manager.data_handler.tensors[parameter_tensor_name].dtype)
        param_signature = tf.TensorSpec(
            shape=[None, np.sum(fit_paramater_list, dtype=int)],
            dtype=param_dtype)

        data_batch_signature = []
        for tensor in manager.data_handler.tensors.list:
            data_batch_signature.append(tf.TensorSpec(
                shape=tensor.shape,
                dtype=getattr(tf, tensor.dtype)))

        # --------------------------------------------------
        # get concrete functions for reconstruction and loss
        # --------------------------------------------------
        loss_and_gradients_function = manager.get_loss_and_gradients_function(
            input_signature=(param_signature, data_batch_signature),
            loss_module=loss_module,
            fit_paramater_list=fit_paramater_list,
            minimize_in_trafo_space=minimize_in_trafo_space,
            seed=seed_tensor_name,
            parameter_tensor_name=parameter_tensor_name)

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
        result, result_obj = self.reconstruction_method(data_batch)
        return result
