import tensorflow as tf


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
    cos_dist = tf.clip(cos_dist, -1., 1.)
    return tf.acos(cos_dist)
