import tensorflow as tf
import numpy as np
from scipy import stats
from scipy.optimize import minimize

from egenerator.utils import basis_functions


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
    cos_dist = tf.clip_by_value(cos_dist, -1., 1.)
    return tf.acos(cos_dist)


def get_angle_deviation_np(azimuth1, zenith1, azimuth2, zenith2):
    """Get opening angle of two vectors defined by (azimuth, zenith)

    Parameters
    ----------
    azimuth1 : array_like
        Azimuth of vector 1 in rad.
    zenith1 : array_like
        Zenith of vector 1 in rad.
    azimuth2 : array_like
        Azimuth of vector 2 in rad.
    zenith2 : array_like
        Zenith of vector 2 in rad.

    Returns
    -------
    array_like
        The opening angle in rad between the vector 1 and 2.
        Same shape as input vectors.
    """
    cos_dist = (np.cos(azimuth1 - azimuth2) *
                np.sin(zenith1) * np.sin(zenith2) +
                np.cos(zenith1) * np.cos(zenith2))
    cos_dist = np.clip(cos_dist, -1., 1.)
    return np.arccos(cos_dist)


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


def convert_to_range(zenith, azimuth):
    """Ensures that zenith and azimuth are in proper range

    Converts zenith and azimuth such that they are in
    [0, pi) and [0, 2pi).
    Conversion is done by calculating the unit vector and then
    re-computing the zenith and azimuth angles. Note that this
    is susceptible to numerical issues and therefore this
    function should not be used if precise sub-degree results
    are required.

    Parameters
    ----------
    zenith : array_like
        The zenith angles to convert.
    azimuth : array_like
        The azimuth angles to convert.

    Returns
    -------
    array_like
        The converted zenith angles.
    array_like
        The converted azimuth angles.
    """

    # create copies to not edit in place
    zenith = np.array(zenith)
    azimuth = np.array(azimuth)

    x = np.sin(zenith) * np.cos(azimuth)
    y = np.sin(zenith) * np.sin(azimuth)
    z = np.cos(zenith)

    zenith = np.arccos(z)
    azimuth = np.arctan2(y, x)

    azimuth = np.mod(azimuth, 2*np.pi)

    return zenith, azimuth


def normalize(dir_x, dir_y, dir_z):
    """Normalizes a vector on the sphere to have length 1.

    Parameters
    ----------
    dir_x : array_like
        The x-coordinate of the vector.
    dir_y : array_like
        The y-coordinate of the vector.
    dir_z : array_like
        The z-coordinate of the vector.

    Returns
    -------
    array_like
        The x-coordinate of the normalized unit vector.
    array_like
        The y-coordinate of the normalized unit vector.
    array_like
        The z-coordinate of the normalized unit vector.
    """
    norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
    return dir_x/norm, dir_y/norm, dir_z/norm


def angle2vec(zenith, azimuth, with_flip=False):
    """Convert angle to unit direction vector

    Converts a position on the sphere provided via two angles,
    zenith and azimuth, to the components of the unit vector.

    Parameters
    ----------
    zenith : array_like
        The zenith angle.
    azimuth : array_like
        The azimuth angle
    with_flip : bool, optional
        If True, the direction vector is flipped 180° degrees.
        This may be useful when utilizing these functions in the context
        of IceCube software, which has the convention that the direction
        vector points in the direction of flight while the angles are used
        to describe the origin of the particle (opposite direction).

    Returns
    -------
    array_like
        The x-coordinate of the unit vector on the sphere.
    array_like
        The y-coordinate of the unit vector on the sphere.
    array_like
        The z-coordinate of the unit vector on the sphere.
    """
    sin_zenith = np.sin(zenith)
    dir_x = sin_zenith * np.cos(azimuth)
    dir_y = sin_zenith * np.sin(azimuth)
    dir_z = np.cos(zenith)
    if with_flip:
        dir_x = -dir_x
        dir_y = -dir_y
        dir_z = -dir_z

    return dir_x, dir_y, dir_z


def dir2angle(dir_x, dir_y, dir_z, with_flip=False):
    """Convert unit direction vector to angle

    Converts a position on the sphere provided via the components
    of the unit direction vector into two angles, zenith and azimuth.

    Parameters
    ----------
    dir_x : array_like
        The x-coordinate of the unit vector on the sphere.
    dir_y : array_like
        The y-coordinate of the unit vector on the sphere.
    dir_z : array_like
        The y-coordinate of the unit vector on the sphere.
    with_flip : bool, optional
        If True, the direction vector is flipped 180° degrees.
        This may be useful when utilizing these functions in the context
        of IceCube software, which has the convention that the direction
        vector points in the direction of flight while the angles are used
        to describe the origin of the particle (opposite direction).

    Returns
    -------
    array_like
        The zenith angle.
    array_like
        The azimuth angle.
    """
    # normalize
    dir_x_normed, dir_y_normed, dir_z_normed = normalize(dir_x, dir_y, dir_z)

    if with_flip:
        dir_x_normed = -dir_x_normed
        dir_y_normed = -dir_y_normed
        dir_z_normed = -dir_z_normed

    zenith = np.arccos(np.clip(-dir_z_normed, -1, 1))
    azimuth = np.mod(np.arctan2(dir_y_normed, dir_x_normed), 2 * np.pi)

    return zenith, azimuth


def compute_coverage(
        cdf_values,
        weights=None,
        quantiles=np.linspace(0.001, 1, 100),
        verbose=True,
        ):
    """Compute Coverage

    Parameters
    ----------
    cdf_values : array_like
        The cumulative distribution function evaluated
        at each (dir, unc) pair.
    weights : array_like, optional
        The event weights
    quantiles : array_like, optional
        The quantile values for which to compute the coverage
    verbose : bool, optional
        If True, additional putput will be printed.

    Returns
    -------
    array_like
        The quantile values at which the coverage is computed.
    array_like
        The coverage values for each of the quantiles
    """
    if weights is None:
        weights = np.ones_like(cdf_values)

    num_events = np.sum(weights)

    # sort values
    sorted_indices = np.argsort(cdf_values)
    sorted_cdf_values = cdf_values[sorted_indices]
    sorted_weights = weights[sorted_indices]

    cum_sum = np.cumsum(sorted_weights) / num_events
    indices = np.searchsorted(sorted_cdf_values, quantiles)
    mask_over = indices >= len(cum_sum)
    if verbose and np.sum(mask_over) > 0:
        print('Clipping {} events to max.'.format(np.sum(mask_over)))
        indices = np.clip(indices, 0, len(cum_sum) - 1)
    coverage = cum_sum[indices]

    return quantiles, coverage


class DistributionOnSphere:

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        self._params = dict(params)

    def fit(self, samples, x0, *args, **kwargs):
        """Fits the distribution parameters to the provided samples

        Parameters
        ----------
        samples : array_like
            The sample points on the sphere.
            Can either be zenith, azimuth or x, y, z of unit direction
            vector. The size of the last dimension of `samples` defines
            which one will be used.
        x0 : array_like
            The initial seed for the parameters.
        *args
            Additional arguments.
        **kwargs
            Additional keyword arguments

        Raises
        ------
        NotImplementedError
            Description
        """
        raise NotImplementedError

    def log_pdf(self, zenith, azimuth, *args, **kwargs):
        """Computes the logarithm of the PDF.

        Parameters
        ----------
        zenith : array_like
            The zenith angle.
        azimuth : array_like
            The azimuth angle.
        *args
            Additional arguments.
        **kwargs
            Additional keyword arguments

        Returns
        -------
        array_like
            The log PDF evaluated at the provided positions on the sphere.
        """
        raise NotImplementedError

    def cdf(self, zenith, azimuth, *args, **kwargs):
        """Computes the CDF.

        Parameters
        ----------
        zenith : array_like
            The zenith angle.
        azimuth : array_like
            The azimuth angle.
        *args
            Additional arguments.
        **kwargs
            Additional keyword arguments

        Returns
        -------
        array_like
            The CDF evaluated at the provided positions on the sphere.
        """
        raise NotImplementedError

    def log_pdf_dir(self, dir_x, dir_y, dir_z, *args, **kwargs):
        """Computes the log PDF for the provided unit direction vector.

        Parameters
        ----------
        dir_x : array_like
            The x-component of the unit-direction vector.
        dir_y : array_like
            The y-component of the unit-direction vector.
        dir_z : array_like
            The z-component of the unit-direction vector.
        *args
            Additional arguments.
        **kwargs
            Additional keyword arguments

        Returns
        -------
        array_like
            The log PDF evaluated at the provided positions on the sphere.
        """
        zenith, azimuth = dir2angle(dir_x, dir_y, dir_z)
        return self.log_pdf(zenith, azimuth, *args, **kwargs)

    def cdf_dir(self, dir_x, dir_y, dir_z, *args, **kwargs):
        """Compute the CDF for the provided unit direction vector.

        Parameters
        ----------
        dir_x : array_like
            The x-component of the unit-direction vector.
        dir_y : array_like
            The y-component of the unit-direction vector.
        dir_z : array_like
            The z-component of the unit-direction vector.
        *args
            Additional arguments.
        **kwargs
            Additional keyword arguments

        Returns
        -------
        array_like
            The CDF evaluated at the provided positions on the sphere.
        """
        zenith, azimuth = dir2angle(dir_x, dir_y, dir_z)
        return self.cdf(zenith, azimuth, *args, **kwargs)

    def _convert_samples(self, samples):
        """Converts provided sample points to angles and unit vectors.

        Parameters
        ----------
        samples : array_like
            The sample points on the sphere.
            Can either be zenith, azimuth or x, y, z of unit direction
            vector. The size of the last dimension of `samples` defines
            which one will be used.

        Returns
        -------
        array_like
            A tuple of the zenith and azimuth angle.
        array_like
            A tuple of the x, y, and z-components of the unit direction vector.
        """
        samples = np.asarray(samples)

        if samples.shape[-1] == 3:
            dir_x, dir_y, dir_z = (samples[..., i] for i in range(3))
            zenith, azimuth = dir2angle(dir_x, dir_y, dir_z)

        elif samples.shape[-1] == 2:
            zenith, azimuth = (samples[..., i] for i in range(2))
            dir_x, dir_y, dir_z = angle2vec(zenith, azimuth)
        else:
            raise ValueError('Shape not understood:', samples.shape)

        zenith, azimuth = convert_to_range(zenith, azimuth)

        return (zenith, azimuth), (dir_x, dir_y, dir_z)

    def goodness_of_fit(self, samples, *args, **kwargs):
        """Compute a goodness of fit by checking uniformity of CDF values

        Parameters
        ----------
        samples : array_like
            The sample points on the sphere. These must be drawn from the
            underlying distribution.
            Can either be zenith, azimuth or x, y, z of unit direction
            vector. The size of the last dimension of `samples` defines
            which one will be used.
        *args
            Additional arguments passed on to the CDF function.
        **kwargs
            Additional keyword arguments passed on to the CDF function.

        Returns
        -------
        float
            The p-value of the KS-test that tests the null-hypothesis that
            the CDF values are distributed uniformly.
        """
        # get zenith and azimuth angles
        zenith, azimuth = self._convert_samples(samples)[0]

        # compute CDF values of provided samples
        cdf_values = self.cdf(zenith, azimuth, *args, **kwargs)

        # the cdf values should be distributed uniformly
        # We will check for uniformity via a  KS-test
        res, pval = stats.kstest(cdf_values, stats.uniform.cdf)
        return pval


class FB8Distribution(DistributionOnSphere):

    def fit(self, samples, cdf_levels=np.linspace(0., .999, 1000),
            fb5_only=True, warning=None, seed=42, *args, **kwargs):
        """Fits the distribution parameters to the provided samples

        Parameters
        ----------
        samples : array_like
            The sample points on the sphere.
            Can either be zenith, azimuth or x, y, z of unit direction
            vector. The size of the last dimension of `samples` defines
            which one will be used.
        cdf_levels : array_like, optional
            The quantile
        fb5_only : bool, optional
            If True, only fit for the FB5 (Kent) distribution.
        warning : None, optional
            Define logging level for warnings.
        *args
            Additional arguments.
        **kwargs
            Additional keyword arguments
        """

        # python package fb8
        from sphere import distribution

        dir_x, dir_y, dir_z = self._convert_samples(samples)[1]

        # xs has format [z, x, y] and NOT [x, y, z]!!
        xs = np.array([dir_z, dir_x, dir_y]).T
        self.fb8 = distribution.fb8_mle(xs, fb5_only=fb5_only, warning=None)

        # compute levels needed for cdf calculation
        self.seed = seed
        self.cdf_levels = np.asarray(cdf_levels)
        self.neg_log_p_levels = np.empty_like(self.cdf_levels)
        for i, level in enumerate(self.cdf_levels):
            # older fb8 package version did not have seed parameter
            try:
                self.neg_log_p_levels[i] = self.fb8.level(
                    percentile=level*100, seed=self.seed)
            except TypeError as e:
                self.neg_log_p_levels[i] = self.fb8.level(
                    percentile=level*100)

        # set parameters
        self._params = {
            'theta': self.fb8.theta,
            'phi': self.fb8.phi,
            'psi': self.fb8.psi,
            'kappa': self.fb8.kappa,
            'beta': self.fb8.beta,
            'eta': self.fb8.eta,
            'alpha': self.fb8.alpha,
            'rho': self.fb8.rho,
        }

    def log_pdf(self, zenith, azimuth, *args, **kwargs):
        """Computes the logarithm of the PDF.

        Parameters
        ----------
        zenith : array_like
            The zenith angle.
        azimuth : array_like
            The azimuth angle.
        *args
            Additional arguments.
        **kwargs
            Additional keyword arguments

        Returns
        -------
        array_like
            The log PDF evaluated at the provided positions on the sphere.
        """
        dir_x, dir_y, dir_z = angle2vec(zenith, azimuth)

        # xs has format [z, x, y] and NOT [x, y, z]!!
        xs = np.array([dir_z, dir_x, dir_y]).T

        return self.fb8.log_pdf(xs)

    def cdf(self, zenith, azimuth, *args, **kwargs):
        """Computes the CDF.

        Parameters
        ----------
        zenith : array_like
            The zenith angle.
        azimuth : array_like
            The azimuth angle.
        *args
            Additional arguments.
        **kwargs
            Additional keyword arguments

        Returns
        -------
        array_like
            The CDF evaluated at the provided positions on the sphere.
        """
        neg_log = -self.log_pdf(zenith, azimuth, *args, **kwargs)

        # find index of containment level
        indices = np.searchsorted(self.neg_log_p_levels, neg_log)
        indices = np.clip(indices, 0, len(self.cdf_levels) - 1)

        return self.cdf_levels[indices]


class VonMisesFisherDistribution(DistributionOnSphere):

    def fit(self, samples, x0, fit_position=True, *args, **kwargs):
        """Fits the distribution parameters to the provided samples

        Parameters
        ----------
        samples : array_like
            The sample points on the sphere.
            Can either be zenith, azimuth or x, y, z of unit direction
            vector. The size of the last dimension of `samples` defines
            which one will be used.
        x0 : array_like
            The initial best-fit/central position of the vMF distribution.
            Must be given as a tuple of (zenith, azimuth, sigma).
        fit_position : bool, optional
            If True, the position will be fit.
            If False, the position will be kept constant at the provided x0.
        *args
            Additional arguments.
        **kwargs
            Additional keyword arguments
        """

        zenith, azimuth = self._convert_samples(samples)[0]
        self.fit_position = fit_position

        eps = 1e-4

        if fit_position:
            def loss(params):
                zen, azi, sigma = params

                if sigma < eps:
                    return np.inf

                # get opening angle to center of distribution
                delta_psi = get_angle_deviation_np(
                    azimuth1=azi,
                    zenith1=zen,
                    azimuth2=azimuth,
                    zenith2=zenith,
                )

                return -np.sum(np.log(basis_functions.von_mises_in_dPsi_pdf(
                    x=delta_psi, sigma=sigma,
                )))

            res = minimize(loss, x0=x0)
            self._params = {
                'zenith': res.x[0],
                'azimuth': res.x[1],
                'sigma': res.x[2],
            }
        else:
            def loss(sigma):

                if sigma < eps:
                    return np.inf

                # get opening angle to center of distribution
                delta_psi = get_angle_deviation_np(
                    azimuth1=x0[1],
                    zenith1=x0[0],
                    azimuth2=azimuth,
                    zenith2=zenith,
                )

                return -np.sum(np.log(basis_functions.von_mises_in_dPsi_pdf(
                    x=delta_psi, sigma=sigma,
                )))

            res = minimize(loss, x0=x0[2])

            self._params = {
                'zenith': x0[0],
                'azimuth': x0[1],
                'sigma': res.x[0],
            }

    def log_pdf(self, zenith, azimuth, *args, **kwargs):
        """Computes the logarithm of the PDF.

        Parameters
        ----------
        zenith : array_like
            The zenith angle.
        azimuth : array_like
            The azimuth angle.
        *args
            Additional arguments.
        **kwargs
            Additional keyword arguments

        Returns
        -------
        array_like
            The log PDF evaluated at the provided positions on the sphere.
        """

        # get opening angle to center of distribution
        delta_psi = get_angle_deviation_np(
            azimuth1=self.params['azimuth'],
            zenith1=self.params['zenith'],
            azimuth2=azimuth,
            zenith2=zenith,
        )

        return np.log(basis_functions.von_mises_in_dPsi_pdf(
            x=delta_psi, sigma=self.params['sigma'],
        ))

    def cdf(self, zenith, azimuth, *args, **kwargs):
        """Computes the CDF.

        Parameters
        ----------
        zenith : array_like
            The zenith angle.
        azimuth : array_like
            The azimuth angle.
        *args
            Additional arguments.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        array_like
            The CDF evaluated at the provided positions on the sphere.
        """

        # get opening angle to center of distribution
        delta_psi = get_angle_deviation_np(
            azimuth1=self.params['azimuth'],
            zenith1=self.params['zenith'],
            azimuth2=azimuth,
            zenith2=zenith,
        )
        sigma = np.zeros_like(delta_psi) + self.params['sigma']

        return basis_functions.von_mises_in_dPsi_cdf(
            x=delta_psi, sigma=sigma, *args, **kwargs
        )


class Gauss2D(DistributionOnSphere):

    """Note this does not define a proper PDF on the sphere!

    Only approximately valid in cartesian approximation of small angles
    """

    def get_azimuth_residuals(self, azimuth1, azimuth2):
        """Get azimuth residuals while taking 2pi periodicity into account

        Parameters
        ----------
        azimuth1 : array_like
            The first azimuth values.
        azimuth2 : array_like
            The second azimuth values.

        Returns
        -------
        array_like
            The residuals between the two provided azimuth values.
        """
        azimuth1 = np.mod(azimuth1, 2*np.pi)
        azimuth2 = np.mod(azimuth2, 2*np.pi)

        residuals = azimuth1 - azimuth2
        residuals[residuals < -np.pi] = residuals[residuals < -np.pi] + 2*np.pi
        residuals[residuals > +np.pi] = residuals[residuals > +np.pi] - 2*np.pi
        return residuals

    def nearestPD(self, A):
        """Find the nearest positive-definite matrix to input

        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1],
        which credits [2].

        [1] https://www.mathworks.com/matlabcentral/fileexchange/
            42885-nearestspd

        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6

        Credits to solution from Ahmed Fasih:
            https://stackoverflow.com/questions/43238173/
            python-convert-matrix-to-positive-semi-definite

        Parameters
        ----------
        A : array_like
            The input matrix.

        Returns
        -------
        array_like
            The output matrix
        """

        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)

        H = np.dot(V.T, np.dot(np.diag(s), V))

        A2 = (B + H) / 2

        A3 = (A2 + A2.T) / 2

        if self.isPD(A3):
            return A3

        spacing = np.spacing(np.linalg.norm(A))
        # The above is different from [1]. It appears that MATLAB's
        # `chol` Cholesky decomposition will accept matrixes with exactly
        # 0-eigenvalue, whereas Numpy's will not. So where [1] uses
        # `eps(mineig)` (where `eps` is Matlab for `np.spacing`), we use the
        # above definition. CAVEAT: our `spacing` will be much larger than
        # [1]'s `eps(mineig)`, since `mineig` is usually on the order of
        # 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas `spacing`
        # will, for Gaussian random matrixes of small dimension, be on
        # othe order of 1e-16. In practice, both ways converge,
        # as the unit test below suggests.
        identity = np.eye(A.shape[0])
        k = 1
        while not self.isPD(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += identity * (-mineig * k**2 + spacing)
            k += 1

        return A3

    def isPD(self, B):
        """Returns true when input is positive-definite, via Cholesky"""
        try:
            _ = np.linalg.cholesky(B)
            return True
        except np.linalg.LinAlgError:
            return False

    def fit(self, samples, x0=None, fit_position=True, allow_singular=True,
            seed=42, *args, **kwargs):
        """Fits the distribution parameters to the provided samples

        Parameters
        ----------
        samples : array_like
            The sample points on the sphere.
            Can either be zenith, azimuth or x, y, z of unit direction
            vector. The size of the last dimension of `samples` defines
            which one will be used.
        x0 : array_like
            The initial best-fit/central position of the vMF distribution.
            Must be given as a tuple of (zenith, azimuth, sigma).
        fit_position : bool, optional
            If True, the position will be fit.
            If False, the position will be kept constant at the provided x0.
        allow_singular : bool, optional
            Allow singular covariance matrices.
        seed : int, optional
            Description
        *args
            Additional arguments.
        **kwargs
            Additional keyword arguments
        """

        zenith, azimuth = self._convert_samples(samples)[0]
        self.fit_position = fit_position
        self.seed = seed

        eps = 1e-4

        if x0 is None:
            cov = np.cov(zenith, azimuth)
            x0 = [
                np.median(zenith), np.median(azimuth),
                cov[0, 0], cov[0, 1], cov[1, 1],
            ]

        if fit_position:

            def loss(params):
                zen, azi, cov_00, cov_01, cov_11 = params

                if cov_00 < eps or cov_11 < eps:
                    return np.inf

                if cov_01 >= 1 or cov_01 <= -1:
                    return np.inf

                # get delta values
                d_zen = zen - zenith
                d_azi = self.get_azimuth_residuals(azi, azimuth)

                cov = np.array([[cov_00, cov_01], [cov_01, cov_11]])
                cov = self.nearestPD(cov)

                dist = stats.multivariate_normal(
                    mean=[0, 0],
                    cov=cov,
                    seed=seed,
                    allow_singular=allow_singular,
                )

                values = np.stack([d_zen, d_azi], axis=1)
                # print('values', values)
                # print('loss', -np.sum(dist.logpdf(values)))
                return -np.sum(dist.logpdf(values))

            res = minimize(loss, x0=x0)
            self._params = {
                'zenith': res.x[0],
                'azimuth': res.x[1],
                'cov_00': res.x[2],
                'cov_01': res.x[3],
                'cov_11': res.x[4],
            }
        else:
            def loss(params):

                cov_00, cov_01, cov_11 = params

                if cov_00 < eps or cov_11 < eps:
                    return np.inf

                if cov_01 >= 1 or cov_01 <= -1:
                    return np.inf

                # get delta values
                d_zen = x0[0] - zenith
                d_azi = self.get_azimuth_residuals(x0[1], azimuth)
                # d_azi = x0[1] - azimuth

                cov = np.array([[cov_00, cov_01], [cov_01, cov_11]])
                cov = self.nearestPD(cov)
                # print('cov', cov, self.isPD(cov))

                dist = stats.multivariate_normal(
                    mean=[0, 0],
                    cov=cov,
                    seed=seed,
                    allow_singular=allow_singular,
                )

                values = np.stack([d_zen, d_azi], axis=1)
                return -np.sum(dist.logpdf(values))

            res = minimize(loss, x0=x0[2:])

            self._params = {
                'zenith': x0[0],
                'azimuth': x0[1],
                'cov_00': res.x[0],
                'cov_01': res.x[1],
                'cov_11': res.x[2],
            }

        self.cov = np.array([
            [self.params['cov_00'], self.params['cov_01']],
            [self.params['cov_01'], self.params['cov_11']],
        ])

        # update cov matrix
        self.cov = self.nearestPD(self.cov)
        self.params['cov_00'] = self.cov[0, 0]
        self.params['cov_01'] = self.cov[0, 1]
        self.params['cov_11'] = self.cov[1, 1]

        self.dist = stats.multivariate_normal(
            mean=[0, 0],
            cov=self.cov,
            seed=self.seed,
            allow_singular=allow_singular,
        )

    def log_pdf(self, zenith, azimuth, *args, **kwargs):
        """Computes the logarithm of the PDF.

        Parameters
        ----------
        zenith : array_like
            The zenith angle.
        azimuth : array_like
            The azimuth angle.
        *args
            Additional arguments.
        **kwargs
            Additional keyword arguments

        Returns
        -------
        array_like
            The log PDF evaluated at the provided positions on the sphere.
        """

        # get delta values
        d_zen = self.params['zenith'] - zenith
        d_azi = self.get_azimuth_residuals(self.params['azimuth'], azimuth)

        values = np.stack([d_zen, d_azi], axis=1)
        return self.dist.logpdf(values)

    def cdf(self, zenith, azimuth, *args, **kwargs):
        """Computes the CDF.

        Parameters
        ----------
        zenith : array_like
            The zenith angle.
        azimuth : array_like
            The azimuth angle.
        *args
            Additional arguments.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        array_like
            The CDF evaluated at the provided positions on the sphere.
        """

        # get delta values
        d_zen = self.params['zenith'] - zenith
        d_azi = self.get_azimuth_residuals(self.params['azimuth'], azimuth)

        values = np.stack([d_zen, d_azi], axis=1)
        return self.dist.cdf(values)
