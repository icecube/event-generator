import tensorflow as tf


def is_in(variable, variable_list):
    """Check if a tensorflow variable is in a list

    Checks if a variable references the same memory location
    as another variable in a list.
    """
    for variable_i in variable_list:
        if variable_i is variable:
            return True
    return False


def clip_logits(logits, eps=None):
    """Clip logits to avoid numerical instabilities

    Parameters
    ----------
    logits : tf.Tensor
        The logits to clip.
    eps : float, optional
        The epsilon value to clip to, by default None.
        If None the value will be clipped to 1e-37 for float32
        and 1e-307 for float64.

    Returns
    -------
    tf.Tensor
        The clipped logits.
    """
    if eps is None:
        # Up to these values, the log returns finite values
        if logits.dtype == tf.float32:
            eps = 1e-37
        elif logits.dtype == tf.float64:
            eps = 1e-307
        else:
            raise ValueError(f"Unknown dtype for logits: {logits.dtype}")

    # some safety checks to make sure we aren't clipping too much
    asserts = []
    asserts.append(
        tf.debugging.Assert(
            tf.reduce_all(logits > -1e-7),
            ["Values < 0!", tf.reduce_min(logits)],
        )
    )
    with tf.control_dependencies(asserts):
        clipped_logits = tf.clip_by_value(logits, eps, float("inf"))
    return clipped_logits


def safe_log(logits, eps=None):
    """Safe log operation

    Parameters
    ----------
    logits : tf.Tensor
        The logits to log.
    eps : float, optional
        The epsilon value to clip to, by default None.
        If None the value will be clipped to 1e-37 for float32
        and 1e-307 for float64.

    Returns
    -------
    tf.Tensor
        The safe log of the logits.
    """
    return tf.math.log(clip_logits(logits, eps=eps))


def safe_cdf_clip(cdf_values, tol=1e-5):
    """Perform clipping of CDF values

    Clips provided CDF values to be between 0 and 1.
    Performs some debugging safety checks to make sure
    to not clip too much.

    Parameters
    ----------
    cdf_values : tf.Tensor
        The CDF values to clip.
    tol : float, optional
        The tolerance for clipping, by default 1e-5.

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
                    cdf_values > -tol,
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
                    cdf_values < 1.0 + tol,
                    ~tf.math.is_finite(cdf_values),
                )
            ),
            ["CDF values > 1!", tf.reduce_max(cdf_values)],
        )
    )
    with tf.control_dependencies(asserts):
        cdf_values = tf.clip_by_value(cdf_values, 0.0, 1.0)
    return cdf_values


def get_prior_pulse_cdf_exclusion(
    x_pulses,
    x_pulses_ids,
    x_time_exclusions,
    x_time_exclusions_ids,
    tw_cdf_exclusion,
):
    """Get prior CDF exclusion for each pulse

    This function calculates the excluded CDF range
    up to the pulse time of an individual pulse.
    This calculatation is necessary when correcting the CDF
    values to account for the time exclusions.

    Parameters
    ----------
    x_pulses : tf.Tensor
        The pulses.
        Shape: [n_pulses, 2]
    x_pulses_ids : tf.Tensor
        The pulse ids.
        Shape: [n_pulses, 3]
    x_time_exclusions : tf.Tensor
        The time exclusions.
        Shape: [n_tw, 2]
    x_time_exclusions_ids : tf.Tensor
        The time exclusion ids.
        Shape: [n_tw, 3]
    tw_cdf_exclusion : tf.Tensor
        The (reduced) CDF exclusion for the entire model.
        Shape: [n_tw]

    Returns
    -------
    tf.Tensor
        The CDF exclusion for each pulse.
        Shape: [n_pulses]
    """

    def get_pulse_cdf_exclusion_per_tw(elem_i):
        """Get pulse cdf exclusion per time window

        Parameters
        ----------
        elem_i : tuple
            Tuple of x_time_exclusions_ids, x_time_exclusions, tw_cdf_exclusion
            for a single time window.

        Returns
        -------
        tf.Tensor
            The CDF exclusion prior to the pulse time for each pulse.
            Shape: [n_pulses]
        """
        (
            x_time_exclusions_ids_i,  # shape: [3]
            x_time_exclusions_i,  # shape: [2]
            tw_cdf_exclusion_i,  # shape: []
        ) = elem_i

        # select pulses in the same event, string, and dom
        mask = tf.reduce_all(x_pulses_ids == x_time_exclusions_ids_i, axis=1)

        # of those pulses, select those that are after the time exclusion
        # Only the CDF values of the pulses afterwards needs correction
        mask &= x_pulses[:, 1] > x_time_exclusions_i[0]

        # return the excluded CDF time range up to the pulse time
        # for each individual pulse
        return tf.where(
            mask,
            tw_cdf_exclusion_i,
            tf.zeros_like(tw_cdf_exclusion_i),
        )

    # -------------------------
    # Implementation via map_fn
    # -------------------------
    # pulse_cdf_exclusion = tf.map_fn(
    #     get_pulse_cdf_exclusion_per_tw,
    #     elems=(x_time_exclusions_ids, x_time_exclusions, tw_cdf_exclusion),
    #     fn_output_signature=tf.TensorSpec(shape=x_pulses.shape[0], dtype=x_pulses.dtype),
    # )
    # pulse_cdf_exclusion = tf.reduce_sum(pulse_cdf_exclusion, axis=0)

    # ---------------------------------
    # Implementation via vectorized_map
    # ---------------------------------
    # This implementation is faster than map_fn, but may require more memory
    pulse_cdf_exclusion = tf.vectorized_map(
        get_pulse_cdf_exclusion_per_tw,
        (x_time_exclusions_ids, x_time_exclusions, tw_cdf_exclusion),
    )
    pulse_cdf_exclusion = tf.reduce_sum(pulse_cdf_exclusion, axis=0)

    # -------------------------------------------------
    # Implementation via unstack (requires known shape)
    # -------------------------------------------------
    # pulse_cdf_exclusion = tf.zeros_like(x_pulses[:, 0])
    # for i, tw_exclusion_id in enumerate(tf.unstack(x_time_exclusions_ids, axis=0)):
    #     mask = tf.reduce_all(x_pulses_ids == tw_exclusion_id, axis=1)
    #     mask &= x_pulses[:, 1] > x_time_exclusions[i, 0]
    #     pulse_cdf_exclusion = tf.where(
    #         mask, pulse_cdf_exclusion + tw_cdf_exclusion[i], pulse_cdf_exclusion
    #     )
    # -------------------------------------------------

    return pulse_cdf_exclusion
