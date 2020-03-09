from __future__ import division, print_function
import os
import numpy as np
import logging
import pickle
from copy import deepcopy
from tqdm import tqdm
from datetime import datetime
import tensorflow as tf

from egenerator.manager.component import BaseComponent
from egenerator.data.tensor import DataTensorList, DataTensor


class DataTransformer(BaseComponent):

    """Data Transformer class

    Attributes
    ----------
    trafo_model : dict
        The data transformation model.
    """
    def __init__(self, logger=None):
        """Instanciate DataTransformer class

        Parameters
        ----------
        logger : logging.logger, optional
            The logger to use.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(DataTransformer, self).__init__(logger=self._logger)

    def _configure(self, data_handler, data_iterator, num_batches,
                   float_precision='float64',
                   norm_constant=1e-6):
        """Configure DataTransformer object.

        Iteratively create a transformation model for a given data_handler and
        data_iterator.

        Parameters
        ----------
        data_handler : DataHandler
            A data handler object.
        data_iterator : generator object
            A python generator object which generates batches of
            dom_responses and cascade_parameters.
        num_batches : int
            How many batches to use to create the transformation model.
        float_precision : str, optional
            Float precision to use for trafo methods.
            Examples: 'float32', 'float64'
        norm_constant : float
            A small constant that is added to the denominator during
            normalization to ensure finite values.

        Raises
        ------
        ValueError
            Description
        """
        if self.is_configured:
            raise ValueError('Trafo model is already setup!')

        self._data['tensors'] = data_handler.tensors

        # set precision and norm constant
        self._data['float_precision'] = float_precision
        self._data['norm_constant'] = norm_constant
        self._data['np_float_dtype'] = \
            getattr(np, self.data['float_precision'])
        self._data['tf_float_dtype'] = \
            getattr(tf, self.data['float_precision'])

        # create empty onlince variance variables
        var_dict = {}
        for tensor in data_handler.tensors.list:

            # check if tensor exists and whether a transformation is defined
            if tensor.exists and tensor.trafo:

                trafo_shape = list(tensor.shape)
                # remove batch axis
                trafo_shape.pop(tensor.trafo_batch_axis)

                var_dict[tensor.name] = {
                    'n': 0.,
                    'mean': np.zeros(trafo_shape),
                    'M2': np.zeros(trafo_shape),
                }

        for i in tqdm(range(num_batches)):

            data = next(data_iterator)

            # loop through tensors and update online variance calculation
            for tensor in data_handler.tensors.list:
                if tensor.exists and tensor.trafo:
                    index = data_handler.tensors.get_index(tensor.name)
                    n, mean, m2 = self._perform_update_step(
                                    trafo_log=tensor.trafo_log,
                                    data_batch=data[index],
                                    n=var_dict[tensor.name]['n'],
                                    mean=var_dict[tensor.name]['mean'],
                                    M2=var_dict[tensor.name]['M2'])
                    var_dict[tensor.name]['n'] = n
                    var_dict[tensor.name]['mean'] = mean
                    var_dict[tensor.name]['M2'] = m2

        # Calculate standard deviation
        for tensor in data_handler.tensors.list:
            if tensor.exists and tensor.trafo:
                std_dev = np.sqrt(var_dict[tensor.name]['M2'] /
                                  var_dict[tensor.name]['n'])

                # combine mean and std. dev. values over specified
                # reduction axes
                self._data[tensor.name+'_std'] = np.mean(
                    std_dev, axis=tensor.trafo_reduce_axes, keepdims=True)

                self._data[tensor.name+'_mean'] = np.mean(
                    var_dict[tensor.name]['mean'],
                    axis=tensor.trafo_reduce_axes, keepdims=True)

                # set constant parameters to have a std dev of 1
                # instead of zero
                mask = self._data[tensor.name+'_std'] == 0
                self._data[tensor.name+'_std'][mask] = 1.

        # create an identifer for the trafo model
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d %H:%M:%S")

        self._data['creation_time'] = dt_string

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
            mean += delta/n
            delta2 = x - mean
            M2 += delta*delta2
        return n, mean, M2

    def _perform_update_step(self, trafo_log, data_batch, n, mean, M2):
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

        Returns
        -------
        int, np.ndarray, np.ndarray
            n, mean, M2
            Returns the updated online variance variables
        """
        data_batch = np.array(data_batch, dtype=self.data['np_float_dtype'])

        # perform logarithm on bins
        if trafo_log is not None:
            if np.alltrue(trafo_log):
                data_batch = np.log(1.0 + data_batch)
            else:
                for bin_i, log_bin in enumerate(trafo_log):
                    if log_bin:
                        data_batch[..., bin_i] = \
                            np.log(1.0 + data_batch[..., bin_i])

        # calculate onlince variance and mean for DOM responses
        return self._update_online_variance_vars(data_batch=data_batch, n=n,
                                                 mean=mean, M2=M2)

    # def load_trafo_model(self, model_path):
    #     """Load a transformation model from file.

    #     Parameters
    #     ----------
    #     model_path : str
    #         Path to trafo model file.

    #     Raises
    #     ------
    #     ValueError
    #         If settings in loaded transformation model do not match specified
    #         settings.
    #         If not all specified settings are defined in the loaded
    #         transformation model.
    #     """
    #     if self.is_configured:
    #         raise ValueError('Trafo model is already setup!')

    #     # load trafo model from file
    #     with open(model_path, 'rb') as handle:
    #         trafo_model = pickle.load(handle)

    #     # make sure that settings match
    #     for key in self.data:
    #         if key not in trafo_model:
    #             raise KeyError('Key {!r} does not exist in {!r}'.format(
    #                 key, model_path))

    #         mismatch = self.data[key] != trafo_model[key]
    #         error_msg = 'Setting {!r} does not match!'.format(key)
    #         if isinstance(mismatch, bool):
    #             if mismatch:
    #                 raise ValueError(error_msg)
    #         elif mismatch.any():
    #             raise ValueError(error_msg)

    #     # update trafo model
    #     self._data = trafo_model
    #     self.data['np_float_dtype'] = getattr(np, self.data['float_precision'])
    #     self.data['tf_float_dtype'] = getattr(tf, self.data['float_precision'])

    #     self.is_configured = True

    # def save_trafo_model(self, model_path, overwrite=False):
    #     """Saves transformation model to file.

    #     Parameters
    #     ----------
    #     model_path : str
    #         Path to trafo model file.
    #     overwrite : bool, optional
    #         If True, potential existing files will be overwritten.
    #         If False, an error will be raised.
    #     """
    #     if os.path.exists(model_path):
    #         if overwrite:
    #             self._logger.info('Overwriting existing file at: {}'.format(
    #                                                             model_path))
    #         else:
    #             raise IOError('File already exists!')

    #     directory = os.path.dirname(model_path)
    #     if not os.path.isdir(directory):
    #         os.makedirs(directory)
    #         self._logger.info('Creating directory: {}'.format(directory))

    #     with open(model_path, 'wb') as handle:
    #         pickle.dump(self.data, handle,
    #                     protocol=pickle.HIGHEST_PROTOCOL)

    def _check_settings(self, data, tensor_name):
        """Check settings and return necessary parameters for trafo and inverse
        trafo method.

        Parameters
        ----------
        data :  numpy.ndarray or tf.Tensor
            The data that will be transformed.
        tensor_name : str
            The name of the tensor which will be transformed.

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
            raise ValueError('DataTransformer needs to create or load a trafo '
                             'model prior to transform call.')

        if tensor_name not in self.data['tensors'].names:
            raise ValueError('Tensor {!r} is unknown!'.format(tensor_name))

        # get tensor
        tensor = self.data['tensors'][tensor_name]

        # check if shape of data matches expected shape
        trafo_shape = list(tensor.shape)
        trafo_shape.pop(tensor.trafo_batch_axis)
        data_shape = list(data.shape)
        data_shape.pop(tensor.trafo_batch_axis)

        if list(data_shape) != trafo_shape:
            msg = 'Shape of data {!r} does not match expected shape {!r}'
            raise ValueError(msg.format(data_shape, trafo_shape))

        is_tf = tf.is_tensor(data)

        if is_tf:
            if dtype != self.data['tf_float_dtype']:
                data = tf.cast(data, dtype=self.data['tf_float_dtype'])
        else:
            # we need to create a copy of the array, so that we do not alter
            # the original one during the transformation steps
            data = np.array(data, dtype=self.data['np_float_dtype'])

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
        data, log_func, exp_func, is_tf, dtype, tensor = \
            self._check_settings(data, tensor_name)

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
            data -= self.data['{}_mean'.format(tensor_name)]
        data /= (self.data['norm_constant'] +
                 self.data['{}_std'.format(tensor_name)])

        # cast back to original dtype
        if is_tf:
            if dtype != self.data['tf_float_dtype']:
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
        data, log_func, exp_func, is_tf, dtype, tensor = \
            self._check_settings(data, tensor_name)

        # de-normalize data
        data *= (self.data['norm_constant'] +
                 self.data['{}_std'.format(tensor_name)])
        if bias_correction:
            data += self.data['{}_mean'.format(tensor_name)]

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
                            data_list[bin_i] = \
                                tf.clip_by_value(data_list[bin_i], -60., 60.)
                            data_list[bin_i] = exp_func(data_list[bin_i]) - 1.0
                    data = tf.stack(data_list, axis=-1)
                else:
                    for bin_i, do_log in enumerate(tensor.trafo_log):
                        if do_log:
                            data[..., bin_i] = exp_func(data[..., bin_i]) - 1.0

        # cast back to original dtype
        if is_tf:
            if dtype != self.data['tf_float_dtype']:
                data = tf.cast(data, dtype=dtype)
        else:
            data = data.astype(dtype)
        return data
