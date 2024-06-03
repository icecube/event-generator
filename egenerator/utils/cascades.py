import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def get_dist_to_shower_max(ref_energy, eps=1e-6):
    """Get cascade extension for an EM cascade

    Parameters
    ----------
    ref_energy : float or array_like
        Energy of cascade in GeV.
    eps : float, optional
        Small constant float.

    Returns
    -------
    float or array_like
        The distance from the cascade vertex to the average shower maximum.
    """

    # Radiation length in meters, assuming an ice density of 0.9216 g/cm^3
    l_rad = 0.358 / 0.9216  # in meter

    """
    Parameters taken from I3SimConstants (for particle e-):
    https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/
    meta-projects/combo/trunk/sim-services/private/
    sim-services/I3SimConstants.cxx
    """
    if tf.is_tensor(ref_energy):
        log_func = tf.math.log
        clip_func = tfp.math.clip_by_value_preserve_gradient
    else:
        log_func = np.log
        clip_func = np.clip

    a = 2.01849 + 0.63176 * log_func(ref_energy + eps)
    b = l_rad / 0.63207

    # Mode of the gamma distribution gamma_dist(a, b) is: (a-1.)/b
    length_to_maximum = clip_func(((a - 1.0) / b) * l_rad, 0.0, float("inf"))
    return length_to_maximum


def shift_to_maximum(
    x, y, z, zenith, azimuth, ref_energy, t, eps=1e-6, reverse=False
):
    """Shift cascade to/from shower maximum

    PPC does its own cascade extension, leaving the showers at the
    production vertex. Reapply the parametrization to find the
    position of the shower maximum, which is also the best approximate
    position for a point cascade.

    Parameters
    ----------
    x : float or array_like
        Cascade interaction vertex x (unshifted) in meters.
    y : float or array_like
        Cascade interaction vertex y (unshifted) in meters.
    z : float or array_like
        Cascade interaction vertex z (unshifted) in meters.
    zenith : float or array_like
        Cascade zenith direction in rad.
    azimuth : float or array_like
        Cascade azimuth direction in rad.
    ref_energy : float or array_like
        Energy of cascade in GeV.
    t : float or array_like
        Cascade interaction vertex time (unshifted) in ns.
    eps : float, optional
        Small constant float.
    reverse : bool, optional
        If True, the reverse shift will be applied. This can be used
        to shift the reconstructed shower maximum back to the (average)
        vertex position and time.

    Returns
    -------
    Tuple of float or tuple of array_like
        Shifted vertex position (position of shower maximum) in meter and
        shifted vertex time in nano seconds.
    """
    if tf.is_tensor(ref_energy):
        sin_func = tf.math.sin
        cos_func = tf.math.cos
    else:
        sin_func = np.sin
        cos_func = np.cos

    length_to_maximum = get_dist_to_shower_max(ref_energy=ref_energy, eps=eps)
    if reverse:
        length_to_maximum *= -1

    c = 0.299792458  # meter / ns
    dir_x = -sin_func(zenith) * cos_func(azimuth)
    dir_y = -sin_func(zenith) * sin_func(azimuth)
    dir_z = -cos_func(zenith)

    x_shifted = x + dir_x * length_to_maximum
    y_shifted = y + dir_y * length_to_maximum
    z_shifted = z + dir_z * length_to_maximum
    t_shifted = t + length_to_maximum / c
    return x_shifted, y_shifted, z_shifted, t_shifted
