import tensorflow as tf


def safe_cdf_clip(cdf_values):
    """Perform clipping of CDF values

    Clips provided CDF values to be between 0 and 1.
    Performs some debugging safety checks to make sure
    to not clip too much.

    Parameters
    ----------
    cdf_values : tf.Tensor
        The CDF values to clip.

    Returns
    -------
    tf.Tensor
        The clipped CDF values.
    """
    # some safety checks to make sure we aren't clipping too much
    asserts = []
    asserts.append(
        tf.debugging.Assert(
            tf.reduce_all(
                tf.math.logical_or(
                    cdf_values > -1e-4,
                    ~tf.math.is_finite(cdf_values),
                )
            ),
            ["CDF values < 0!", tf.reduce_min(cdf_values)],
        )
    )
    asserts.append(
        tf.debugging.Assert(
            tf.reduce_all(
                tf.math.logical_or(
                    cdf_values < 1.0001,
                    ~tf.math.is_finite(cdf_values),
                )
            ),
            ["CDF values > 1!", tf.reduce_max(cdf_values)],
        )
    )
    with tf.control_dependencies(asserts):
        cdf_values = tf.clip_by_value(cdf_values, 0.0, 1.0)
    return cdf_values
