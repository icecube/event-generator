import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


def get_angle(vec1, vec2):
    """Calculate opening angle between two direction vectors.

    vec1/2 : shape: [?,3] or [3]
    https://www.cs.berkeley.edu/~wkahan/Mindless.pdf p.47/56

    Parameters
    ----------
    vec1 : tf.Tensor
        Direction vector 1.
        Shape: [?,3] or [3]
    vec2 : tf.Tensor
        Direction vector 2.
        Shape: [?,3] or [3]

    Returns
    -------
    tf.Tensor
        The opening angle in rad between the direction vector 1 and 2.
        Same shape as input vectors.
    """
    # transform into numpy array with dtype
    assert vec1.get_shape().as_list()[-1] == 3, \
        "Expect shape [?,3] or [3], but got {}".format(vec1.get_shape())
    assert vec2.get_shape().as_list()[-1] == 3, \
        "Expect shape [?,3] or [3], but got {}".format(vec2.get_shape())

    norm1 = tf.linalg.norm(vec1, axis=-1, keepdims=True)
    norm2 = tf.linalg.norm(vec2, axis=-1, keepdims=True)
    tmp1 = vec1 * norm2
    tmp2 = vec2 * norm1

    tmp3 = tf.linalg.norm(tmp1 - tmp2, axis=-1)
    tmp4 = tf.linalg.norm(tmp1 + tmp2, axis=-1)

    theta = 2*tf.atan2(tmp3, tmp4)

    return theta


def get_angle_deviation(azimuth1, zenith1, azimuth2, zenith2):
    """Get opening angle of two vectors defined by (azimuth, zenith)

    Parameters
    ----------
    azimuth1 : tf.Tensor
        Azimuth of vector 1 in rad.
    zenith1 : tf.Tensor
        Zenith of vector 1 in rad.
    azimuth2 : tf.Tensor
        Azimuth of vector 2 in rad.
    zenith2 : tf.Tensor
        Zenith of vector 2 in rad.

    Returns
    -------
    tf.Tensor
        The opening angle in rad between the vector 1 and 2.
        Same shape as input vectors.
    """
    cos_dist = (tf.cos(azimuth1 - azimuth2) *
                tf.sin(zenith1) * tf.sin(zenith2) +
                tf.cos(zenith1) * tf.cos(zenith2))
    cos_dist = tfp.math.clip_by_value_preserve_gradient(cos_dist, -1., 1.)
    return tf.acos(cos_dist)


def get_delta_psi_vector(zenith, azimuth, delta_psi,
                         random_service=None,
                         randomize_for_each_delta_psi=True,
                         is_degree=True,
                         return_angles=True):
    """Get new angles with an opening angle of delta_psi.

    Parameters
    ----------
    zenith : array_like
        The zenith angle of the input vector for which to compute a random
        new vector with an opening angle of delta_psi.
    azimuth : TYPE
        The azimuth angle of the input vector for which to compute a random
        new vector with an opening angle of delta_psi.
    delta_psi : float or array_like
        The opening angle. If 'is_degree' is True, then the unit is in degree,
        otherwise it is in radians.
        If an array is provided, broadcasting will be applied.
    random_service : None, optional
        An optional random number service to use for reproducibility.
    randomize_for_each_delta_psi : bool, optional
        If True, a random orthogonal vector is sampled for each specified
        delta_psi.
        If False, the direction vectors for the delta_psi opening angles
        are computed along along the same (random) geodesic.
    is_degree : bool, optional
        This specifies the input unit of 'delta_psi'.
        If True, the input unit of 'delta_psi' is degree.
        If False, it is radians.
    return_angles : bool, optional
        If True, the new random vector will be returned as zenith and azimuth
        angles: shape:  tuple([..., 1], [..., 1]).
        If False, it will be returned as a direction vector: shape: [..., 3].

    Returns
    -------
    array_like or tuple of array_like
        If return_angles is True:
            Return values are (zenith, azimuth)
            Shape:  tuple([..., 1], [..., 1])
        If return_angles is False:
            Return values are the new direction vectors in cartesian
            coordinates.
            Shape: [..., 3]
    """
    vec = np.array([np.sin(zenith) * np.cos(azimuth),
                    np.sin(zenith) * np.sin(azimuth),
                    np.cos(zenith)]).T
    vec = np.atleast_2d(vec)
    delta_vec = get_delta_psi_vector_dir(
        vec,
        delta_psi=delta_psi,
        random_service=random_service,
        randomize_for_each_delta_psi=randomize_for_each_delta_psi,
        is_degree=is_degree)
    if return_angles:
        # calculate zenith
        d_zenith = np.arccos(np.clip(delta_vec[..., 2], -1, 1))

        # calculate azimuth
        d_azimuth = (np.arctan2(delta_vec[..., 1], delta_vec[..., 0])
                     + 2 * np.pi) % (2 * np.pi)
        return d_zenith, d_azimuth
    else:
        return delta_vec


def get_delta_psi_vector_dir(vec, delta_psi,
                             randomize_for_each_delta_psi=True,
                             random_service=None,
                             is_degree=True):
    """Get a new direction vector with an opening angle of delta_psi to vec.

    Parameters
    ----------
    vec : array_like
        The vector for which to calculate a new random vector with an opening
        angle of delta_psi.
        Shape: [..., 3]
    delta_psi : float or array_like
        The opening angle. If 'is_degree' is True, then the unit is in degree,
        otherwise it is in radians.
        If an array is provided, broadcasting will be applied.
    randomize_for_each_delta_psi : bool, optional
        If True, a random orthogonal vector is sampled for each specified
        delta_psi.
        If False, the direction vectors for the delta_psi opening angles
        are computed along along the same (random) geodesic.
    random_service : None, optional
        An optional random number service to use for reproducibility.
    is_degree : bool, optional
        This specifies the input unit of 'delta_psi'.
        If True, the input unit of 'delta_psi' is degree.
        If False, it is radians.

    Returns
    -------
    array_like
        The new random vectors with an opening angle of 'delta_psi'.
        Shape: [..., 3]

    Raises
    ------
    ValueError
        If the specified opening angle is larger or equal to 90 degree.
        This calculation only supports angles up to 90 degree.
    """
    if random_service is None:
        random_service = np.random

    if is_degree:
        delta_psi = np.deg2rad(delta_psi)

    # allow broadcasting
    delta_psi = np.expand_dims(delta_psi, axis=-1)

    # This calculation is only valid if delta_psi < 90 degree
    if np.any(delta_psi >= np.deg2rad(90)):
        msg = 'Delta Psi angle must be smaller than 90 degrees, but it is {!r}'
        raise ValueError(msg.format(np.rad2deg(delta_psi)))

    # get a random orthogonal vector
    if randomize_for_each_delta_psi:
        num_temp_vecs = max(len(vec), len(delta_psi))
    else:
        num_temp_vecs = len(vec)

    temp_vec = random_service.uniform(low=-1, high=1, size=(num_temp_vecs, 3))

    vec_orthogonal = np.cross(vec, temp_vec)
    vec_orthogonal /= np.linalg.norm(vec_orthogonal, axis=-1, keepdims=True)

    # calculate new vector with specified opening angle
    new_vec = vec + np.tan(delta_psi) * vec_orthogonal
    new_vec /= np.linalg.norm(new_vec, axis=-1, keepdims=True)
    return new_vec
