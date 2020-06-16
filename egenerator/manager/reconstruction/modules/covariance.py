import numpy as np
import tensorflow as tf


class CovarianceMatrix:

    def __init__(self, manager, loss_module, function_cache,
                 fit_paramater_list,
                 seed_tensor_name,
                 reco_key,
                 minimize_in_trafo_space=True,
                 parameter_tensor_name='x_parameters',):
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
        reco_key : TYPE
            Description
        minimize_in_trafo_space : bool, optional
            Description
        parameter_tensor_name : str, optional
            Description

        No Longer Raises
        ----------------
        NotImplementedError
            Description
        """

        # store settings
        self.manager = manager
        self.fit_paramater_list = fit_paramater_list
        self.reco_key = reco_key
        self.minimize_in_trafo_space = minimize_in_trafo_space
        self.parameter_tensor_name = parameter_tensor_name

        param_dtype = getattr(tf, manager.data_trafo.data['tensors'][
            parameter_tensor_name].dtype)
        param_signature = tf.TensorSpec(
            shape=[None, np.sum(fit_paramater_list, dtype=int)],
            dtype=param_dtype)

        # define data batch tensor specification
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

        # define function settings
        func_settings = dict(
            input_signature=(param_signature, data_batch_signature),
            loss_module=loss_module,
            fit_paramater_list=fit_paramater_list,
            minimize_in_trafo_space=minimize_in_trafo_space,
            seed=seed_tensor_name,
            parameter_tensor_name=parameter_tensor_name,
        )

        # Get Hessian Function
        self.hessian_function = function_cache.get(
            'hessian_function', func_settings)

        if self.hessian_function is None:
            self.hessian_function = manager.get_hessian_function(
                **func_settings)
            function_cache.add(self.hessian_function, func_settings)

        # Get Outer-Product-Estimate Function
        self.opg_estimate_function = function_cache.get(
            'opg_estimate_function', func_settings)

        if self.opg_estimate_function is None:
            self.opg_estimate_function = manager.get_opg_estimate_function(
                **func_settings)
            function_cache.add(self.opg_estimate_function, func_settings)

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

        result_trafo = results[self.reco_key]['result_trafo']
        result_obj = results[self.reco_key]['result_object']

        # get Hessian at reco best fit
        hessian = self.hessian_function(
            parameters_trafo=result_trafo,
            data_batch=data_batch).numpy().astype('float64')

        opg_estimate = self.opg_estimate_function(
            parameters_trafo=result_trafo,
            data_batch=data_batch).numpy().astype('float64')

        cov_trafo = np.linalg.inv(hessian)
        cov_sand_trafo = np.matmul(np.matmul(
            cov_trafo, opg_estimate), cov_trafo)
        if hasattr(result_obj, 'hess_inv'):
            cov_sand_fit_trafo = np.matmul(
                np.matmul(result_obj.hess_inv, opg_estimate),
                result_obj.hess_inv)

        if self.minimize_in_trafo_space:
            cov = self.manager.data_trafo.inverse_transform_cov(
                cov_trafo=cov_trafo, tensor_name=self.parameter_tensor_name)

            cov_sand = self.manager.data_trafo.inverse_transform_cov(
                cov_trafo=cov_sand_trafo,
                tensor_name=self.parameter_tensor_name
            )

            if hasattr(result_obj, 'hess_inv'):
                cov_sand_fit = self.manager.data_trafo.inverse_transform_cov(
                    cov_trafo=cov_sand_fit_trafo,
                    tensor_name=self.parameter_tensor_name
                )
                cov_fit = self.manager.data_trafo.inverse_transform_cov(
                    cov_trafo=result_obj.hess_inv,
                    tensor_name=self.parameter_tensor_name,
                )

        results = {
            'cov': cov,
            'cov_sand': cov_sand,
            'cov_trafo': cov_trafo,
            'cov_sand_trafo': cov_sand_trafo,
        }
        if hasattr(result_obj, 'hess_inv'):
            results.update({
                'cov_fit': cov_fit,
                'cov_sand_fit': cov_sand_fit,
                'cov_fit_trafo': result_obj.hess_inv,
                'cov_sand_fit_trafo': cov_sand_fit_trafo,
            })

        return results
