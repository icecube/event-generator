import numpy as np
import logging
from tqdm import tqdm
from datetime import datetime
import tensorflow as tf

from egenerator import misc
from egenerator.manager.component import BaseComponent, Configuration


class DataTransformer(BaseComponent):
    """Data Transformer class

    Attributes
    ----------
    trafo_model : dict
        The data transformation model.
    """

    def __init__(self, logger=None):
        """Instantiate DataTransformer class

        Parameters
        ----------
        logger : logging.logger, optional
            The logger to use.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(DataTransformer, self).__init__(logger=self._logger)

    def _configure(
        self,
        data_handler,
        data_iterator_settings,
        num_batches,
        float_precision="float64",
        norm_constant=1e-6,
    ):
        """Configure DataTransformer object.

        Iteratively create a transformation model for a given data_handler and
        data iterator settings.

        Parameters
        ----------
        data_handler : DataHandler
            A data handler object.
        data_iterator_settings : dict
            The settings for the data iterator that will be created from the
            data handler.
        num_batches : int
            How many batches to use to create the transformation model.
        float_precision : str, optional
            Float precision to use for trafo methods.
            Examples: 'float32', 'float64'
        norm_constant : float
            A small constant that is added to the denominator during
            normalization to ensure finite values.

        Returns
        -------
        Configuration object
            The configuration object of the newly configured component.
            This does not need to include configurations of sub components
            which are passed directly as parameters into the configure method,
            as these are automatically gathered. Components passed as lists,
            tuples, and dicts are also collected, unless they are nested
            deeper (list of list of components will not be detected).
            The dependent_sub_components may also be left empty for these
            passed and detected sub components.
            Deeply nested sub components or sub components created within
            (and not directly passed as an argument to) this component
            must be added manually.
            Settings that need to be defined are:
                class_string:
                    misc.get_full_class_string_of_object(self)
                settings: dict
                    The settings of the component.
                mutable_settings: dict, default={}
                    The mutable settings of the component.
                check_values: dict, default={}
                    Additional check values.
        dict
            The data of the component.
            Return None if the component has no data.
        dict
            A dictionary of dependent sub components. This is a dictionary
            of sub components that need to be saved and loaded recursively
            when the component is saved and loaded.
            Return None if no dependent sub components exist.

        Raises
        ------
        ValueError
            Description
        """
        if self.is_configured:
            raise ValueError("Trafo model is already setup!")

        # create data iterator
        data_iterator = data_handler.get_batch_generator(
            **data_iterator_settings
        )

        check_values = {}
        data = {}
        data["tensors"] = data_handler.tensors

        # set precision and norm constant
        data["norm_constant"] = norm_constant
        data["np_float_dtype"] = getattr(np, float_precision)
        data["tf_float_dtype"] = getattr(tf, float_precision)

        # create empty onlince variance variables
        var_dict = {}
        for tensor in data_handler.tensors.list:

            # check if tensor exists and whether a transformation is defined
            if tensor.exists and tensor.trafo:

                trafo_shape = list(tensor.shape)
                # remove batch axis
                trafo_shape.pop(tensor.trafo_batch_axis)

                var_dict[tensor.name] = {
                    "n": 0.0,
                    "mean": np.zeros(trafo_shape),
                    "M2": np.zeros(trafo_shape),
                }

        for i in tqdm(range(num_batches)):

            data_batch = next(data_iterator)

            # loop through tensors and update online variance calculation
            for tensor in data_handler.tensors.list:
                if tensor.exists and tensor.trafo:
                    index = data_handler.tensors.get_index(tensor.name)
                    n, mean, m2 = self._perform_update_step(
                        trafo_log=tensor.trafo_log,
                        data_batch=data_batch[index],
                        n=var_dict[tensor.name]["n"],
                        mean=var_dict[tensor.name]["mean"],
                        M2=var_dict[tensor.name]["M2"],
                        dtype=data["np_float_dtype"],
                    )
                    var_dict[tensor.name]["n"] = n
                    var_dict[tensor.name]["mean"] = mean
                    var_dict[tensor.name]["M2"] = m2

        # Calculate standard deviation
        for tensor in data_handler.tensors.list:
            if tensor.exists and tensor.trafo:
                std_dev = np.sqrt(
                    var_dict[tensor.name]["M2"] / var_dict[tensor.name]["n"]
                )

                # combine mean and std. dev. values over specified
                # reduction axes
                data[tensor.name + "_std"] = np.mean(
                    std_dev, axis=tensor.trafo_reduce_axes, keepdims=True
                )

                data[tensor.name + "_mean"] = np.mean(
                    var_dict[tensor.name]["mean"],
                    axis=tensor.trafo_reduce_axes,
                    keepdims=True,
                )

                # set constant parameters to have a std dev of 1
                # instead of zero
                mask = data[tensor.name + "_std"] == 0
                data[tensor.name + "_std"][mask] = 1.0

                # create check values for a simplified test to see if the data
                # matches. This is only a simple hash (mean of tensor values)
                # and does not guarantee that two models are identical
                for suffix in ["_std", "_mean"]:
                    check_values[tensor.name + suffix] = float(
                        np.mean(data[tensor.name + suffix])
                    )

        # create an identifer for the trafo model
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
        data["creation_time"] = dt_string

        # create configuration object
        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings=dict(
                data_iterator_settings=data_iterator_settings,
                num_batches=num_batches,
                float_precision=float_precision,
                norm_constant=norm_constant,
            ),
            check_values=check_values,
        )

        return configuration, data, {}

    def _update_online_variance_vars(self, data_batch, n, mean, M2):
        """Update online variance variables.

        This can be used to iteratively calculate the mean and variance of
        a dataset.

        Parameters
        ----------
        data_batch : numpy ndarray
            A batch of data for which to update the variance variables of the
            dataset.
        n : int
            Counter for number of data elements.
        mean : numpy ndarray
            Mean of dataset.
        M2 : numpy ndarray
            Variance * size of dataset

        Returns
        -------
        int, np.ndarray, np.ndarray
            n, mean, M2
            Returns the updated online variance variables
        """
        for x in data_batch:
            n += 1
            delta = x - mean
            mean += delta / n
            delta2 = x - mean
            M2 += delta * delta2
        return n, mean, M2

    def _perform_update_step(self, trafo_log, data_batch, n, mean, M2, dtype):
        """Update online variance variables.

        This can be used to iteratively calculate the mean and variance of
        a dataset.

        Parameters
        ----------
        trafo_log : tuple of bool
            Defines whether the natural logarithm is appllied to bins along
            last axis. Must have same length as data_batch.shape[-1].
        data_batch : numpy ndarray
            A batch of data for which to update the variance variables of the
            dataset.
        n : int
            Counter for number of data elements.
        mean : numpy ndarray
            Mean of dataset.
        M2 : numpy ndarray
            Variance * size of dataset
        dtype : numpy.dtype
            The data type to use.

        Returns
        -------
        int, np.ndarray, np.ndarray
            n, mean, M2
            Returns the updated online variance variables
        """
        data_batch = np.array(data_batch, dtype=dtype)

        # perform logarithm on bins
        if trafo_log is not None:
            if np.alltrue(trafo_log):
                data_batch = np.log(1.0 + data_batch)
            else:
                for bin_i, log_bin in enumerate(trafo_log):
                    if log_bin:
                        data_batch[..., bin_i] = np.log(
                            1.0 + data_batch[..., bin_i]
                        )

        # calculate onlince variance and mean for DOM responses
        return self._update_online_variance_vars(
            data_batch=data_batch, n=n, mean=mean, M2=M2
        )

    def _check_settings(self, data, tensor_name, check_shape=True):
        """Check settings and return necessary parameters for trafo and inverse
        trafo method.

        Parameters
        ----------
        data : numpy.ndarray or tf.Tensor
            The data that will be transformed.
        tensor_name : str
            The name of the tensor which will be transformed.
        check_shape : bool, optional
            If True, check shape of provided data tensor.

        Returns
        -------
        type(data)
            The transformed data

        Raises
        ------
        ValueError
            If DataTransformer object has not created or loaded a trafo model.
            If provided data_type is unkown.
        """
        dtype = data.dtype

        if not self.is_configured:
            raise ValueError(
                "DataTransformer needs to create or load a trafo "
                "model prior to transform call."
            )

        if tensor_name not in self.data["tensors"].names:
            raise ValueError("Tensor {!r} is unknown!".format(tensor_name))

        # get tensor
        tensor = self.data["tensors"][tensor_name]

        # check if shape of data matches expected shape
        if check_shape:
            trafo_shape = list(tensor.shape)
            trafo_shape.pop(tensor.trafo_batch_axis)
            data_shape = list(data.shape)
            data_shape.pop(tensor.trafo_batch_axis)

            if list(data_shape) != trafo_shape:
                msg = (
                    "Shape of data {} for tensor {} does not match "
                    "expected shape {}"
                )
                raise ValueError(
                    msg.format(data_shape, tensor_name, trafo_shape)
                )

        is_tf = tf.is_tensor(data)

        if is_tf:
            if dtype != self.data["tf_float_dtype"]:
                data = tf.cast(data, dtype=self.data["tf_float_dtype"])
        else:
            # we need to create a copy of the array, so that we do not alter
            # the original one during the transformation steps
            data = np.array(data, dtype=self.data["np_float_dtype"])

        # choose numpy or tensorflow log function
        if is_tf:
            log_func = tf.math.log
            exp_func = tf.math.exp
        else:
            log_func = np.log
            exp_func = np.exp

        return data, log_func, exp_func, is_tf, dtype, tensor

    def transform(self, data, tensor_name, bias_correction=True):
        """Applies transformation to the specified data.

        Parameters
        ----------
        data : numpy.ndarray or tf.Tensor
            The data that will be transformed.
        tensor_name : str
            The name of the tensor which will be transformed.
        bias_correction : bool, optional
            If true, the transformation will correct the bias, e.g. subtract
            of the data mean to make sure that the transformed data is centered
            around zero. Usually this behaviour is desired. However, when
            transforming uncertainties, this might not be useful.
            If false, it is assumed that uncertaintes are being transformed,
            hence, the logarithm will not be applied.

        Returns
        -------
        type(data)
            The transformed data.
        """
        data, log_func, exp_func, is_tf, dtype, tensor = self._check_settings(
            data, tensor_name
        )

        # perform logarithm on bins
        if bias_correction and tensor.trafo_log is not None:

            # trafo log axis other than last axis of tensor is not yet
            # supported
            if tensor.trafo_log_axis != -1:
                raise NotImplementedError()

            if np.all(tensor.trafo_log):
                # logarithm is applied to all bins: one operation
                data = log_func(1.0 + data)

            else:
                # logarithm is only applied to some bins
                if is_tf:
                    data_list = tf.unstack(data, axis=-1)
                    for bin_i, do_log in enumerate(tensor.trafo_log):
                        if do_log:
                            data_list[bin_i] = log_func(1.0 + data_list[bin_i])
                    data = tf.stack(data_list, axis=-1)
                else:
                    for bin_i, do_log in enumerate(tensor.trafo_log):
                        if do_log:
                            data[..., bin_i] = log_func(1.0 + data[..., bin_i])

        # normalize data
        if bias_correction:
            data -= self.data["{}_mean".format(tensor_name)]
        data /= (
            self.data["norm_constant"]
            + self.data["{}_std".format(tensor_name)]
        )

        # cast back to original dtype
        if is_tf:
            if dtype != self.data["tf_float_dtype"]:
                data = tf.cast(data, dtype=dtype)
        else:
            data = data.astype(dtype)
        return data

    def inverse_transform(self, data, tensor_name, bias_correction=True):
        """Applies inverse transformation to the specified data.

        Parameters
        ----------
        data : numpy.ndarray or tf.Tensor
            The data that will be transformed.
        tensor_name : str
            The name of the tensor which will be transformed.
        bias_correction : bool, optional
            If true, the transformation will correct the bias, e.g. subtract
            of the data mean to make sure that the transformed data is centered
            around zero. Usually this behaviour is desired. However, when
            transforming uncertainties, this might not be useful.
            If false, it is assumed that uncertaintes are being transformed,
            hence, the exponential will not be applied.

        Returns
        -------
        type(data)
            Returns the inverse transformed DOM respones and
            cascade_parameters.
        """
        data, log_func, exp_func, is_tf, dtype, tensor = self._check_settings(
            data, tensor_name
        )

        # de-normalize data
        data *= (
            self.data["norm_constant"]
            + self.data["{}_std".format(tensor_name)]
        )
        if bias_correction:
            data += self.data["{}_mean".format(tensor_name)]

        # undo logarithm on bins
        if bias_correction and tensor.trafo_log is not None:

            # trafo log axis other than last axis of tensor is not yet
            # supported
            if tensor.trafo_log_axis != -1:
                raise NotImplementedError()

            if np.all(tensor.trafo_log):
                # logarithm is applied to all bins: one operation
                data = exp_func(data) - 1.0

            else:
                # logarithm is only applied to some bins
                if is_tf:
                    data_list = tf.unstack(data, axis=-1)
                    for bin_i, do_log in enumerate(tensor.trafo_log):
                        if do_log:
                            data_list[bin_i] = tf.clip_by_value(
                                data_list[bin_i], -60.0, 60.0
                            )
                            data_list[bin_i] = exp_func(data_list[bin_i]) - 1.0
                    data = tf.stack(data_list, axis=-1)
                else:
                    for bin_i, do_log in enumerate(tensor.trafo_log):
                        if do_log:
                            data[..., bin_i] = exp_func(data[..., bin_i]) - 1.0

        # cast back to original dtype
        if is_tf:
            if dtype != self.data["tf_float_dtype"]:
                data = tf.cast(data, dtype=dtype)
        else:
            data = data.astype(dtype)
        return data

    def inverse_transform_cov(self, cov_trafo, tensor_name):
        """Applies inverse transformation of covariance matrix.

        Note: this only corrects for scaling vie the standard deviation.
        If a logarithm was applied, this does not undo that transformation.
        E.g. in that case the return covariance matrix is provided in log-space
        for that parameter.

        Parameters
        ----------
        cov_trafo : array_like
            The covariance matrix of the transformed parameter tensor.
        tensor_name : str
            The name of the tensor which the provided covariance matrix
            describes.

        Returns
        -------
        array_like
            The covariance matrix of the inverse transformed parameter tensor
            excluding possible log-transformations.
        """

        cov_trafo, _, _, is_tf, dtype, tensor = self._check_settings(
            cov_trafo, tensor_name
        )

        # V_trafo = B V B^T
        # V = B^-1 V_trafo (B^T)^-1
        # Define B^-1 which is a diagonal matrix
        B_inv = np.diag(
            self.data["norm_constant"] + self.data[tensor_name + "_std"]
        )

        # Since B is diagonal: B = B^T
        if is_tf:
            cov = tf.matmul(tf.matmul(B_inv, cov_trafo), B_inv)
        else:
            cov = np.matmul(np.matmul(B_inv, cov_trafo), B_inv)

        # cast back to original dtype
        if is_tf:
            if dtype != self.data["tf_float_dtype"]:
                cov = tf.cast(cov, dtype=dtype)
        else:
            cov = cov.astype(dtype)

        return cov
