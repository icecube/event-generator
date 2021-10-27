import numpy as np


def get_dist_to_shower_max(ref_energy, eps=1e-6):
    """Get cascade extension for an EM cascade

    Parameters
    ----------
    ref_energy : float or np.ndarray of floats
        Energy of cascade in GeV.
    eps : float, optional
        Small constant float.

    Returns
    -------
    float or array_like
        The distance from the cascade vertex to the average shower maximum.
    """

    # Radiation length in meters, assuming an ice density of 0.9216 g/cm^3
    l_rad = (0.358/0.9216)  # in meter

    """
    Parameters taken from I3SimConstants (for particle e-):
    https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/
    meta-projects/combo/trunk/sim-services/private/
    sim-services/I3SimConstants.cxx
    """
    a = 2.01849 + 0.63176 * np.log(ref_energy + eps)
    b = l_rad/0.63207

    # Mode of the gamma distribution gamma_dist(a, b) is: (a-1.)/b
    length_to_maximum = np.clip(((a-1.)/b)*l_rad, 0., float('inf'))
    return length_to_maximum


def shift_to_maximum(
        self, x, y, z, zenith, azimuth, ref_energy, t,
        eps=1e-6, reverse=False):
    """Shift cascade to/from shower maximum

    PPC does its own cascade extension, leaving the showers at the
    production vertex. Reapply the parametrization to find the
    position of the shower maximum, which is also the best approximate
    position for a point cascade.

    Parameters
    ----------
    x : float or np.ndarray of floats
        Cascade interaction vertex x (unshifted) in meters.
    y : float or np.ndarray of floats
        Cascade interaction vertex y (unshifted) in meters.
    z : float or np.ndarray of floats
        Cascade interaction vertex z (unshifted) in meters.
    zenith : float or np.ndarray of floats
        Cascade zenith direction in rad.
    azimuth : float or np.ndarray of floats
        Cascade azimuth direction in rad.
    ref_energy : float or np.ndarray of floats
        Energy of cascade in GeV.
    t : float or np.ndarray of floats
        Cascade interaction vertex time (unshifted) in ns.
    eps : float, optional
        Small constant float.
    reverse : bool, optional
        If True, the reverse shift will be applied. This can be used
        to shift the reconstructed shower maximum back to the (average)
        vertex position and time.

    Returns
    -------
    Tuple of float or tuple of np.ndarray
        Shifted vertex position (position of shower maximum) in meter and
        shifted vertex time in nano seconds.
    """
    length_to_maximum = get_dist_to_shower_max(
        ref_energy=ref_energy, eps=eps)
    if reverse:
        length_to_maximum *= -1

    c = 0.299792458  # meter / ns
    dir_x = -np.sin(zenith) * np.cos(azimuth)
    dir_y = -np.sin(zenith) * np.sin(azimuth)
    dir_z = -np.cos(zenith)

    x_shifted = x + dir_x * length_to_maximum
    y_shifted = y + dir_y * length_to_maximum
    z_shifted = z + dir_z * length_to_maximum
    t_shifted = t + length_to_maximum / c
    return x_shifted, y_shifted, z_shifted, t_shifted
