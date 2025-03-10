import numpy as np
import tensorflow as tf

from scipy.optimize import LbfgsInvHessProduct


class CovarianceMatrix:

    def __init__(
        self,
        manager,
        loss_module,
        function_cache,
        fit_parameter_list,
        reco_key,
        minimize_in_trafo_space=True,
        parameter_tensor_name="x_parameters",
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
        fit_parameter_list : bool or list of bool, optional
            Indicates whether a parameter is to be minimized.
            The ith element in the list specifies if the ith parameter
            is minimized.
        reco_key : str
            The name of the reconstruction module to use. The covariance
            matrix will be calculated at the best fit point of the specified
            reconstruction module.
        minimize_in_trafo_space : bool, optional
            If True, covariance is calculated in transformed and normalized
            parameter space. This is usually desired, because the scales of
            the parameters will all be normalized which should facilitate
            calculation and inversion.
        parameter_tensor_name : str, optional
            The name of the parameter tensor to use. Default: 'x_parameters'.
        """

        if not np.all(fit_parameter_list):
            raise NotImplementedError(
                "Covariance currently only supports "
                "calculation in all parameters"
            )

        # store settings
        self.manager = manager
        self.fit_parameter_list = fit_parameter_list
        self.reco_key = reco_key
        self.minimize_in_trafo_space = minimize_in_trafo_space
        self.parameter_tensor_name = parameter_tensor_name

        param_dtype = manager.data_handler.tensors[
            parameter_tensor_name
        ].dtype_tf
        param_signature = tf.TensorSpec(
            shape=[None, np.sum(fit_parameter_list, dtype=int)],
            dtype=param_dtype,
        )
        param_signature_full = tf.TensorSpec(
            shape=[None, len(fit_parameter_list)], dtype=param_dtype
        )

        # define data batch tensor specification
        data_batch_signature = manager.data_handler.get_data_set_signature()

        # define function settings
        func_settings = dict(
            input_signature=(
                param_signature,
                data_batch_signature,
                param_signature_full,
            ),
            loss_module=loss_module,
            fit_parameter_list=fit_parameter_list,
            minimize_in_trafo_space=minimize_in_trafo_space,
            seed=None,
            parameter_tensor_name=parameter_tensor_name,
        )

        # Get Hessian Function
        self.hessian_function = function_cache.get(
            "hessian_function", func_settings
        )

        if self.hessian_function is None:
            self.hessian_function = manager.get_hessian_function(
                **func_settings
            )
            function_cache.add(self.hessian_function, func_settings)

        # Get Outer-Product-Estimate Function
        self.opg_estimate_function = function_cache.get(
            "opg_estimate_function", func_settings
        )

        if self.opg_estimate_function is None:
            self.opg_estimate_function = manager.get_opg_estimate_function(
                **func_settings
            )
            function_cache.add(self.opg_estimate_function, func_settings)

    def execute(self, data_batch, results, **kwargs):
        """Execute module for a given batch of data.

        Parameters
        ----------
        data_batch : tuple of array_like
            A batch of data consisting of a tuple of data arrays.
        results : dict
            A dictionary with the results of previous modules.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        TYPE
            Description
        """

        result_inv = results[self.reco_key]["result"]
        result_trafo = results[self.reco_key]["result_trafo"]
        result_obj = results[self.reco_key]["result_object"]

        # get Hessian at reco best fit
        hessian = (
            self.hessian_function(
                parameters_trafo=result_trafo,
                data_batch=data_batch,
                seed=result_inv,
            )
            .numpy()
            .astype("float64")
        )

        opg_estimate = (
            self.opg_estimate_function(
                parameters_trafo=result_trafo,
                data_batch=data_batch,
                seed=result_inv,
            )
            .numpy()
            .astype("float64")
        )

        try:
            cov_trafo = np.linalg.inv(hessian)
        except np.linalg.LinAlgError as e:
            print(e)
            print("Hessian is a singular matrix and cannot be inverted!")
            try:
                cov_trafo = np.linalg.pinv(hessian, rcond=1e-8, hermitian=True)
                print("Using (Moore-Penrose) pseudo-inverse of Hessian.")
            except np.linalg.LinAlgError as e2:
                print(e2)
                print("Setting covariance matrix to NaNs!")
                cov_trafo = np.zeros_like(hessian) * float("nan")

        cov_sand_trafo = np.matmul(
            np.matmul(cov_trafo, opg_estimate), cov_trafo
        )
        if hasattr(result_obj, "hess_inv"):
            if isinstance(result_obj.hess_inv, LbfgsInvHessProduct):
                result_obj.hess_inv = result_obj.hess_inv.todense()

            cov_sand_fit_trafo = np.matmul(
                np.matmul(result_obj.hess_inv, opg_estimate),
                result_obj.hess_inv,
            )

        if self.minimize_in_trafo_space:
            cov = self.manager.data_trafo.inverse_transform_cov(
                cov_trafo=cov_trafo, tensor_name=self.parameter_tensor_name
            )

            cov_sand = self.manager.data_trafo.inverse_transform_cov(
                cov_trafo=cov_sand_trafo,
                tensor_name=self.parameter_tensor_name,
            )

            if hasattr(result_obj, "hess_inv"):
                cov_sand_fit = self.manager.data_trafo.inverse_transform_cov(
                    cov_trafo=cov_sand_fit_trafo,
                    tensor_name=self.parameter_tensor_name,
                )
                cov_fit = self.manager.data_trafo.inverse_transform_cov(
                    cov_trafo=result_obj.hess_inv,
                    tensor_name=self.parameter_tensor_name,
                )

        results = {
            "cov": cov,
            "cov_sand": cov_sand,
            "cov_trafo": cov_trafo,
            "cov_sand_trafo": cov_sand_trafo,
        }
        if hasattr(result_obj, "hess_inv"):
            results.update(
                {
                    "cov_fit": cov_fit,
                    "cov_sand_fit": cov_sand_fit,
                    "cov_fit_trafo": result_obj.hess_inv,
                    "cov_sand_fit_trafo": cov_sand_fit_trafo,
                }
            )

        return results
