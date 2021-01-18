import numpy as np


def get_reco_result_batch(result_trafo,
                          seed_tensor,
                          fit_parameter_list,
                          minimize_in_trafo_space,
                          data_trafo,
                          parameter_tensor_name='x_parameters'):
    """Get the reco result batch.

    This inverts a possible transformation if minimize_in_trafo_space is
    True and also puts the full hypothesis back together if only parts
    of it were fitted

    Parameters
    ----------
    result_trafo : array_like or tf.Tensor
        The result of the reconstruction. This should be the normalized
        and transformed result tensor (if reconstruction was performed
        in normalized space).
    seed_tensor : array_like or tf.Tensor
        The reconstruction seeds in *unnormalized* and original
        parameter space.
    fit_parameter_list : bool or list of bool, optional
        Indicates whether a parameter has been minimized.
        The ith element in the list specifies if the ith parameter
        is minimized.
    minimize_in_trafo_space : bool, optional
        If True, minimization is assumed to have been performed in transformed
        and normalized parameter space, e.g. the result_trafo tensor is also
        in that space.
    parameter_tensor_name : str, optional
        The name of the parameter tensor to use. Default: 'x_parameters'.

    Returns
    -------
    tf.Tensor
        The full result batch.

    """
    if minimize_in_trafo_space:
        cascade_seed_batch_trafo = data_trafo.transform(
            data=seed_tensor,
            tensor_name=parameter_tensor_name)
        try:
            cascade_seed_batch_trafo = cascade_seed_batch_trafo.numpy()
        except AttributeError:
            pass
    else:
        cascade_seed_batch_trafo = seed_tensor

    if np.all(fit_parameter_list):
        cascade_reco_batch = result_trafo
    else:
        # get seed parameters
        cascade_reco_batch = []
        result_counter = 0
        for i, fit in enumerate(fit_parameter_list):
            if fit:
                cascade_reco_batch.append(result_trafo[:, result_counter])
                result_counter += 1
            else:
                cascade_reco_batch.append(cascade_seed_batch_trafo[:, i])
        cascade_reco_batch = np.array(cascade_reco_batch).T

    # transform back if minimization was performed in trafo space
    if minimize_in_trafo_space:
        cascade_reco_batch = data_trafo.inverse_transform(
            data=cascade_reco_batch,
            tensor_name=parameter_tensor_name)
    return cascade_reco_batch
