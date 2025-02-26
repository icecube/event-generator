import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import special
from scipy import stats
from scipy.integrate import quad

from egenerator.utils import tf_helpers


def cast(dtype, *args):
    """Cast all input arguments to the provided data type.

    Parameters
    ----------
    dtype : str
        The data type to cast the inputs to.
        If None, no casting is performed.
    *args : array_like
        The input arguments to cast.

    Returns
    -------
    array_like
        The casted input arguments.
    """
    if dtype is None:
        return args
    return tuple(np.array(arg, dtype=dtype) for arg in args)


def tf_cast(dtype, *args):
    """Cast all input arguments to the provided data type.

    Parameters
    ----------
    dtype : str
        The data type to cast the inputs to.
        If None, no casting is performed.
    *args : tf.Tensor
        The input arguments to cast.

    Returns
    -------
    tf.Tensor
        The casted input arguments.
    """
    if dtype is None:
        return args
    return tuple(tf.cast(arg, dtype) for arg in args)


def tf_gauss(x, mu, sigma, dtype=None):
    """Gaussian PDF

    Parameters
    ----------
    x : tf.Tensor
        The input tensor.
    mu : tf.Tensor
        Mu parameter of Gaussian.
    sigma : tf.Tensor
        Sigma parameter of Gaussian.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    tf.Tensor
        The Gaussian PDF evaluated at x
    """
    x, mu, sigma = tf_cast(dtype, x, mu, sigma)
    return (
        tf.exp(-0.5 * ((x - mu) / sigma) ** 2) / (2 * np.pi * sigma**2) ** 0.5
    )


def gauss(x, mu, sigma, dtype=None):
    """Gaussian PDF

    Parameters
    ----------
    x : array_like
        The input tensor.
    mu : array_like
        Mu parameter of Gaussian.
    sigma : array_like
        Sigma parameter of Gaussian.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    array_like
        The Gaussian PDF evaluated at x
    """
    x, mu, sigma = cast(dtype, x, mu, sigma)
    return (
        np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (2 * np.pi * sigma**2) ** 0.5
    )


def tf_log_gauss(x, mu, sigma, dtype=None):
    """Log Gaussian PDF

    Parameters
    ----------
    x : tf.Tensor
        The input tensor.
    mu : tf.Tensor
        Mu parameter of Gaussian.
    sigma : tf.Tensor
        Sigma parameter of Gaussian.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    tf.Tensor
        The Gaussian PDF evaluated at x
    """
    x, mu, sigma = tf_cast(dtype, x, mu, sigma)
    norm = np.log(np.sqrt(2 * np.pi))
    return -0.5 * ((x - mu) / sigma) ** 2 - tf.math.log(sigma) - norm


def log_gauss(x, mu, sigma, dtype=None):
    """Log Gaussian PDF

    Parameters
    ----------
    x : array_like
        The input tensor.
    mu : array_like
        Mu parameter of Gaussian.
    sigma : array_like
        Sigma parameter of Gaussian.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    array_like
        The Gaussian PDF evaluated at x
    """
    x, mu, sigma = cast(dtype, x, mu, sigma)
    norm = np.log(np.sqrt(2 * np.pi))
    return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma) - norm


def tf_log_asymmetric_gauss(x, mu, sigma, r, dtype=None):
    """Asymmetric Log Gaussian PDF

    Parameters
    ----------
    x : tf.Tensor
        The input tensor.
    mu : tf.Tensor
        Mu parameter of Gaussian.
    sigma : tf.Tensor
        Sigma parameter of Gaussian.
    r : tf.Tensor
        The asymmetry of the Gaussian.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    tf.Tensor
        The asymmetric Gaussian PDF evaluated at x
    """
    x, mu, sigma, r = tf_cast(dtype, x, mu, sigma, r)
    norm = tf.math.log(2.0 / (tf.sqrt(2 * np.pi * sigma**2) * (r + 1)))
    exp = tf.where(
        x < mu,
        -0.5 * ((x - mu) / sigma) ** 2,
        -0.5 * ((x - mu) / (sigma * r)) ** 2,
    )
    return norm + exp


def log_asymmetric_gauss(x, mu, sigma, r, dtype=None):
    """Asymmetric Log Gaussian PDF

    Parameters
    ----------
    x : array_like
        The input tensor.
    mu : array_like
        Mu parameter of Gaussian.
    sigma : array_like
        Sigma parameter of Gaussian.
    r : array_like
        The asymmetry of the Gaussian.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    array_like
        The asymmetric Gaussian PDF evaluated at x
    """
    x, mu, sigma, r = cast(dtype, x, mu, sigma, r)
    norm = np.log(2.0 / (np.sqrt(2 * np.pi * sigma**2) * (r + 1)))
    exp = np.where(
        x < mu,
        -0.5 * ((x - mu) / sigma) ** 2,
        -0.5 * ((x - mu) / (sigma * r)) ** 2,
    )
    return norm + exp


def tf_asymmetric_gauss(x, mu, sigma, r, dtype=None):
    """Asymmetric Gaussian PDF

    Parameters
    ----------
    x : tf.Tensor
        The input tensor.
    mu : tf.Tensor
        Mu parameter of Gaussian.
    sigma : tf.Tensor
        Sigma parameter of Gaussian.
    r : tf.Tensor
        The asymmetry of the Gaussian.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    tf.Tensor
        The asymmetric Gaussian PDF evaluated at x
    """
    x, mu, sigma, r = tf_cast(dtype, x, mu, sigma, r)
    norm = 2.0 / (tf.sqrt(2 * np.pi * sigma**2) * (r + 1))
    exp = tf.where(
        x < mu,
        tf.exp(-0.5 * ((x - mu) / sigma) ** 2),
        tf.exp(-0.5 * ((x - mu) / (sigma * r)) ** 2),
    )
    return norm * exp


def asymmetric_gauss(x, mu, sigma, r, dtype=None):
    """Asymmetric Gaussian PDF

    Parameters
    ----------
    x : array_like
        The input tensor.
    mu : array_like
        Mu parameter of Gaussian.
    sigma : array_like
        Sigma parameter of Gaussian.
    r : array_like
        The asymmetry of the Gaussian.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    array_like
        The asymmetric Gaussian PDF evaluated at x
    """
    x, mu, sigma, r = cast(dtype, x, mu, sigma, r)
    norm = 2.0 / (np.sqrt(2 * np.pi * sigma**2) * (r + 1))
    exp = np.where(
        x < mu,
        np.exp(-0.5 * ((x - mu) / sigma) ** 2),
        np.exp(-0.5 * ((x - mu) / (sigma * r)) ** 2),
    )
    return norm * exp


def tf_asymmetric_gauss_cdf(x, mu, sigma, r, dtype=None):
    """Asymmetric Gaussian CDF

    Parameters
    ----------
    x : tf.Tensor
        The input tensor.
    mu : tf.Tensor
        Mu parameter of Gaussian.
    sigma : tf.Tensor
        Sigma parameter of Gaussian.
    r : tf.Tensor
        The asymmetry of the Gaussian.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    tf.Tensor
        The asymmetric Gaussian CDF evaluated at x
    """
    x, mu, sigma, r = tf_cast(dtype, x, mu, sigma, r)
    norm = 1.0 / (r + 1)
    exp = tf.where(
        x < mu,
        1.0 + tf.math.erf((x - mu) / (np.sqrt(2) * sigma)),
        1.0 + r * tf.math.erf((x - mu) / (np.sqrt(2) * r * sigma)),
    )
    return norm * exp


def asymmetric_gauss_cdf(x, mu, sigma, r, dtype=None):
    """Asymmetric Gaussian CDF

    Parameters
    ----------
    x : array_like
        The input tensor.
    mu : array_like
        Mu parameter of Gaussian.
    sigma : array_like
        Sigma parameter of Gaussian.
    r : array_like
        The asymmetry of the Gaussian.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    array_like
        The asymmetric Gaussian CDF evaluated at x
    """
    x, mu, sigma, r = cast(dtype, x, mu, sigma, r)
    norm = 1.0 / (r + 1)
    exp = np.where(
        x < mu,
        1.0 + special.erf((x - mu) / (np.sqrt(2) * sigma)),
        1.0 + r * special.erf((x - mu) / (np.sqrt(2) * r * sigma)),
    )
    return norm * exp


def tf_asymmetric_gauss_ppf(q, mu, sigma, r, dtype=None):
    """Asymmetric Gaussian PPF

    Parameters
    ----------
    q : tf.Tensor
        The input tensor.
    mu : tf.Tensor
        Mu parameter of Gaussian.
    sigma : tf.Tensor
        Sigma parameter of Gaussian.
    r : tf.Tensor
        The asymmetry of the Gaussian.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    tf.Tensor
        The asymmetric Gaussian PPF evaluated at q
    """
    q, mu, sigma, r = tf_cast(dtype, q, mu, sigma, r)
    return tf.where(
        q < 1.0 / (r + 1),
        mu + np.sqrt(2) * sigma * tf.math.erfinv(q * (r + 1) - 1),
        mu + r * np.sqrt(2) * sigma * tf.math.erfinv((q * (r + 1) - 1) / r),
    )


def asymmetric_gauss_ppf(q, mu, sigma, r, dtype=None):
    """Asymmetric Gaussian PPF

    Parameters
    ----------
    q : array_like
        The input tensor.
    mu : array_like
        Mu parameter of Gaussian.
    sigma : array_like
        Sigma parameter of Gaussian.
    r : array_like
        The asymmetry of the Gaussian.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    array_like
        The asymmetric Gaussian PPF evaluated at q
    """
    q, mu, sigma, r = cast(dtype, q, mu, sigma, r)
    return np.where(
        q < 1.0 / (r + 1),
        mu + np.sqrt(2) * sigma * special.erfinv(q * (r + 1) - 1),
        mu + r * np.sqrt(2) * sigma * special.erfinv((q * (r + 1) - 1) / r),
    )


def tf_asymmetric_gauss_expectation(mu, sigma, r, dtype=None):
    """Asymmetric Gaussian: Expectation value

    Parameters
    ----------
    mu : tf.Tensor
        Mu parameter of Gaussian.
    sigma : tf.Tensor
        Sigma parameter of Gaussian.
    r : tf.Tensor
        The asymmetry of the Gaussian.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    tf.Tensor
        The expectation value of the asymmetric Gaussian
    """
    mu, sigma, r = tf_cast(dtype, mu, sigma, r)
    expectation = mu + (2 * sigma * (r - 1)) / np.sqrt(2 * np.pi)

    return expectation


def asymmetric_gauss_expectation(mu, sigma, r, dtype=None):
    """Asymmetric Gaussian: Expectation value

    Parameters
    ----------
    mu : array_like
        Mu parameter of Gaussian.
    sigma : array_like
        Sigma parameter of Gaussian.
    r : array_like
        The asymmetry of the Gaussian.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    array_like
        The expectation value of the asymmetric Gaussian
    """
    mu, sigma, r = cast(dtype, mu, sigma, r)
    expectation = mu + (2 * sigma * (r - 1)) / np.sqrt(2 * np.pi)

    return expectation


def _tf_evaluate_gamma(func_name, x, alpha, beta, replacement_value=0.0):
    """Evaluate the Gamma distribution function

    Parameters
    ----------
    func_name : str
        The name of the function to evaluate.
    x : tf.tensor
        The input tensor.
    alpha : tf.tensor
        Alpha tf.tensor of Gamma distribution.
    beta : tf.tensor
        Beta tf.tensor of Gamma distribution.
    replacement_value : float, optional
        The value to replace NaNs with, by default 0.0.

    Returns
    -------
    tf.tensor
        The evaluated function.
    """
    if alpha.dtype == tf.float32:
        eps = 1e-37
    elif alpha.dtype == tf.float64:
        eps = 1e-307
    else:
        raise ValueError(f"Unknown dtype for alpha: {alpha.dtype}")

    distribution = tfp.distributions.Gamma(concentration=alpha, rate=beta)
    return tf_helpers.double_where_trick_greater_zero(
        function=getattr(distribution, func_name),
        x=x,
        cut_x_min=tf.cast(eps, dtype=x.dtype),
        replacement_value=replacement_value,
    )


def tf_gamma_log_pdf(x, alpha, beta, dtype=None):
    """Gamma log PDF

    Parameters
    ----------
    x : tf.Tensor
        The input tensor.
    alpha : tf.Tensor
        Alpha parameter of Gamma distribution.
    beta : tf.Tensor
        Beta parameter of Gamma distribution.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    tf.Tensor
        The Gamma log PDF evaluated at x
    """
    x, alpha, beta, inf = tf_cast(dtype, x, alpha, beta, np.inf)
    return _tf_evaluate_gamma(
        "log_prob", x, alpha, beta, replacement_value=-inf
    )


def tf_gamma_pdf(x, alpha, beta, dtype=None):
    """Gamma PDF

    Parameters
    ----------
    x : tf.Tensor
        The input tensor.
    alpha : tf.Tensor
        Alpha parameter of Gamma distribution.
    beta : tf.Tensor
        Beta parameter of Gamma distribution.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    tf.Tensor
        The Gamma PDF evaluated at x
    """
    x, alpha, beta, zero = tf_cast(dtype, x, alpha, beta, 0.0)
    return _tf_evaluate_gamma("prob", x, alpha, beta, replacement_value=zero)


def gamma_pdf(x, alpha, beta, dtype=None):
    """Gamma PDF

    Parameters
    ----------
    x : array_like
        The input tensor.
    alpha : array_like
        Alpha parameter of Gamma distribution.
    beta : array_like
        Beta parameter of Gamma distribution.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    array_like
        The Gamma PDF evaluated at x
    """
    x, alpha, beta = cast(dtype, x, alpha, beta)
    return np.where(
        x < 0,
        0.0,
        beta**alpha
        / special.gamma(alpha)
        * x ** (alpha - 1)
        * np.exp(-beta * x),
    )


def gamma_log_pdf(x, alpha, beta, dtype=None):
    """Gamma log PDF

    Parameters
    ----------
    x : array_like
        The input tensor.
    alpha : array_like
        Alpha parameter of Gamma distribution.
    beta : array_like
        Beta parameter of Gamma distribution.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    array_like
        The Gamma log PDF evaluated at x
    """
    x, alpha, beta = cast(dtype, x, alpha, beta)
    return np.where(
        x < 0,
        -np.inf,
        alpha * np.log(beta)
        - special.gammaln(alpha)
        + (alpha - 1) * np.log(x)
        - beta * x,
    )


def tf_gamma_cdf(x, alpha, beta, dtype=None):
    """Gamma CDF

    Parameters
    ----------
    x : tf.Tensor
        The input tensor.
    alpha : tf.Tensor
        Alpha parameter of Gamma distribution.
    beta : tf.Tensor
        Beta parameter of Gamma distribution.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    tf.Tensor
        The Gamma CDF evaluated at x
    """
    x, alpha, beta, zero = tf_cast(dtype, x, alpha, beta, 0.0)
    return _tf_evaluate_gamma("cdf", x, alpha, beta, replacement_value=zero)


def gamma_cdf(x, alpha, beta, dtype=None):
    """Gamma CDF

    Parameters
    ----------
    x : array_like
        The input tensor.
    alpha : array_like
        Alpha parameter of Gamma distribution.
    beta : array_like
        Beta parameter of Gamma distribution.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    array_like
        The Gamma CDF evaluated at x
    """
    x, alpha, beta = cast(dtype, x, alpha, beta)
    return np.where(x < 0, 0.0, special.gammainc(alpha, beta * x))


def tf_gamma_ppf(q, alpha, beta, dtype=None):
    """Gamma PPF

    Parameters
    ----------
    q : tf.Tensor
        The input tensor.
    alpha : tf.Tensor
        Alpha parameter of Gamma distribution.
    beta : tf.Tensor
        Beta parameter of Gamma distribution.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    tf.Tensor
        The Gamma PPF evaluated at q
    """
    q, alpha, beta = tf_cast(dtype, q, alpha, beta)
    return tfp.math.igammainv(alpha, q) / beta


def gamma_ppf(q, alpha, beta, dtype=None):
    """Gamma PPF

    Parameters
    ----------
    q : array_like
        The input tensor.
    alpha : array_like
        Alpha parameter of Gamma distribution.
    beta : array_like
        Beta parameter of Gamma distribution.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    array_like
        The Gamma PPF evaluated at q
    """
    q, alpha, beta = cast(dtype, q, alpha, beta)
    return special.gammaincinv(alpha, q) / beta


def tf_gamma_expectation(alpha, beta, dtype=None):
    """Gamma: Expectation value

    Parameters
    ----------
    alpha : tf.Tensor
        Alpha parameter of Gamma distribution.
    beta : tf.Tensor
        Beta parameter of Gamma distribution.

    Returns
    -------
    tf.Tensor
        The expectation value of the Gamma distribution
    """
    alpha, beta = tf_cast(dtype, alpha, beta)
    return alpha / beta


def gamma_expectation(alpha, beta, dtype=None):
    """Gamma: Expectation value

    Parameters
    ----------
    alpha : array_like
        Alpha parameter of Gamma distribution.
    beta : array_like
        Beta parameter of Gamma distribution.

    Returns
    -------
    array_like
        The expectation value of the Gamma distribution
    """
    alpha, beta = cast(dtype, alpha, beta)
    return alpha / beta


def tf_gamma_variance(alpha, beta, dtype=None):
    """Gamma: Variance

    Parameters
    ----------
    alpha : tf.Tensor
        Alpha parameter of Gamma distribution.
    beta : tf.Tensor
        Beta parameter of Gamma distribution.

    Returns
    -------
    tf.Tensor
        The variance of the Gamma distribution
    """
    alpha, beta = tf_cast(dtype, alpha, beta)
    return alpha / beta**2


def gamma_variance(alpha, beta, dtype=None):
    """Gamma: Variance

    Parameters
    ----------
    alpha : array_like
        Alpha parameter of Gamma distribution.
    beta : array_like
        Beta parameter of Gamma distribution.

    Returns
    -------
    array_like
        The variance of the Gamma distribution
    """
    alpha, beta = cast(dtype, alpha, beta)
    return alpha / beta**2


def tf_log_faculty(x):
    """Continuous log faculty approximation via gamma distribution

    Parameters
    ----------
    x : tf.Tensor
        The tensor for which to compute the log faculty approximation.

    Returns
    -------
    tf.Tensor
        The log faculty approximation assuming x is a continuous variable.
    """
    return tf.math.lgamma(tf.clip_by_value(x + 1, 1, float("inf")))


def log_faculty(x):
    """Continuous log faculty approximation via gamma distribution

    Parameters
    ----------
    x : array_like
        The tensor for which to compute the log faculty approximation.

    Returns
    -------
    array_like
        The log faculty approximation assuming x is a continuous variable.
    """
    return special.gammaln(np.clip(x + 1, 1, np.inf))


def tf_poisson_pdf(x, mu, add_normalization_term=False, dtype=None):
    """Poisson PDF

    Note that this is an approximation for continuous values of x.

    Parameters
    ----------
    x : tf.Tensor
        The input tensor.
    mu : tf.Tensor
        Mu parameter of Poisson distribution.
    add_normalization_term : bool, optional
        If True, the normalization term is computed and added.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    tf.Tensor
        The Poisson PDF evaluated at x
    """
    x, mu = tf_cast(dtype, x, mu)

    def function(x):
        result = tf.exp(-mu) * mu**x
        if add_normalization_term:
            result /= tf.exp(tf_log_faculty(x))
        return result

    # avoid non-finite values in the gradient
    result = tf_helpers.double_where_trick_greater_zero(
        function=function,
        x=x,
        cut_x_min=tf.cast(0, dtype=x.dtype),
        replacement_value=0.0,
    )
    return result


def poisson_pdf(x, mu, add_normalization_term=False, dtype=None):
    """Poisson PDF

    Note that this is an approximation for continuous values of x.

    Parameters
    ----------
    x : array_like
        The input tensor.
    mu : array_like
        Mu parameter of Poisson distribution.
    add_normalization_term : bool, optional
        If True, the normalization term is computed and added.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    array_like
        The Poisson PDF evaluated at x
    """
    x, mu = cast(dtype, x, mu)
    result = np.exp(-mu) * mu**x
    if add_normalization_term:
        result /= np.exp(log_faculty(x))

    result = np.where(x < 0, 0.0, result)
    return result


def tf_poisson_log_pdf(x, mu, add_normalization_term=False, dtype=None):
    """Poisson log PDF

    Note that this is an approximation for continuous values of x.

    Parameters
    ----------
    x : tf.Tensor
        The input tensor.
    mu : tf.Tensor
        Mu parameter of Poisson distribution.
    add_normalization_term : bool, optional
        If True, the normalization term is computed and added.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    tf.Tensor
        The Poisson log PDF evaluated at x
    """
    x, mu = tf_cast(dtype, x, mu)

    def function(x):
        result = -mu + x * tf.math.log(mu)
        if add_normalization_term:
            result -= tf_log_faculty(x)
        return result

    result = tf_helpers.double_where_trick_greater_zero(
        function=function,
        x=x,
        cut_x_min=tf.cast(0, dtype=x.dtype),
        replacement_value=-np.inf,
    )
    return result


def poisson_log_pdf(x, mu, add_normalization_term=False, dtype=None):
    """Poisson log PDF

    Note that this is an approximation for continuous values of x.

    Parameters
    ----------
    x : array_like
        The input tensor.
    mu : array_like
        Mu parameter of Poisson distribution.
    add_normalization_term : bool, optional
        If True, the normalization term is computed and added.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    array_like
        The Poisson log PDF evaluated at x
    """
    x, mu = cast(dtype, x, mu)
    result = -mu + x * np.log(mu)
    if add_normalization_term:
        result -= log_faculty(x)

    result = np.where(x < 0, -np.inf, result)
    return result


def tf_log_negative_binomial(
    x, mu, alpha, add_normalization_term=False, dtype=None
):
    """Computes the logarithm of the negative binomial PDF

    The parameterization chosen here is defined by the mean mu and
    the over-dispersion factor alpha.

        Var(x) = mu + alpha*mu**2

    [https://arxiv.org/pdf/1704.04110.pdf page 4]

    Alpha must be greater than zero (best if >5e-5 due to numerical issues).

    Parameters
    ----------
    x : tf.Tensor
        The input tensor.
    mu : tf.Tensor
        Mean of the distribution.
        Must be greater equal zero.
    alpha : tf.Tensor
        The over-dispersion parameter.
        Must be greater zero.
    add_normalization_term : bool, optional
        If True, the normalization term is computed and added.
        Note: this term is not required for minimization and the negative
        binomial distribution only has a proper normalization for integer x.
        For real-valued x the negative binomial is not properly normalized and
        hence adding the normalization term does not help.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    tf.Tensor
        Logarithm of the negative binomal PDF evaluated at x.
    """
    x, mu, alpha = tf_cast(dtype, x, mu, alpha)

    inv_alpha = 1.0 / alpha
    alpha_mu = alpha * mu

    # compute gamma terms
    gamma_terms = tf.math.lgamma(x + inv_alpha) - tf.math.lgamma(inv_alpha)

    if add_normalization_term:
        gamma_terms -= tf_log_faculty(x)

    term1 = -inv_alpha * tf.math.log(1.0 + alpha_mu)
    term2 = x * tf.math.log(alpha_mu / (1.0 + alpha_mu))

    result = tf.where(x < 0, -np.inf, gamma_terms + term1 + term2)

    return result


def log_negative_binomial(
    x, mu, alpha, add_normalization_term=False, dtype=None
):
    """Computes the logarithm of the negative binomial PDF

    The parameterization chosen here is defined by the mean mu and
    the over-dispersion factor alpha.

        Var(x) = mu + alpha*mu**2

    [https://arxiv.org/pdf/1704.04110.pdf page 4]

    Alpha must be greater than zero (best if >5e-5 due to numerical issues).

    Note: alternatively scipy.stats.nbinom can be used
        def negative_binomial(x, mu, alpha):
            sigma2 = mu + alpha*mu**2
            p = 1 - (sigma2 - mu) / sigma2  # need 1-p here
            r = (mu**2 ) / (sigma2 - mu)
            return nbinom(r, p).pmf(x)

    Parameters
    ----------
    x : array_like
        The input values.
    mu : array_like
        Mean of the distribution.
        Must be greater equal zero.
    alpha : array_like
        The over-dispersion parameter.
        Must be greater zero.
    add_normalization_term : bool, optional
        If True, the normalization term is computed and added.
        Note: this term is not required for minimization and the negative
        binomial distribution only has a proper normalization for integer x.
        For real-valued x the negative binomial is not properly normalized and
        hence adding the normalization term does not help.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    array_like
        Logarithm of the negative binomal PDF evaluated at x.
    """
    x, mu, alpha = cast(dtype, x, mu, alpha)

    inv_alpha = 1.0 / alpha
    alpha_mu = alpha * mu

    # compute gamma terms
    gamma_terms = special.gammaln(x + inv_alpha) - special.gammaln(inv_alpha)

    if add_normalization_term:
        gamma_terms -= log_faculty(x)

    term1 = -inv_alpha * np.log(1.0 + alpha_mu)
    term2 = x * np.log(alpha_mu / (1.0 + alpha_mu))

    result = np.where(x < 0, -np.inf, gamma_terms + term1 + term2)

    return result


def convert_neg_binomial_params(mu, alpha_or_var, param_is_alpha):
    """Converts parameterization of negative binomial PDF

    The parameterization chosen here is defined by the mean mu and
    the over-dispersion factor alpha (or directly by variance).

        Var(x) = mu + alpha*mu**2

    [https://arxiv.org/pdf/1704.04110.pdf page 4]

    Alpha must be greater than zero (best if >5e-5 due to numerical issues).

    Parameters
    ----------
    mu : array_like
        Mean of the distribution.
        Must be greater equal zero.
    alpha_or_var : array_like
        if param_is_alpha == True:
            The over-dispersion parameter.
            Must be greater zero.
        if param_is_alpha == False:
            The expected variance.
            Must be greater zero.
    param_is_alpha : bool
        If True, the parameter passed as `alpha_or_var` is alpha.
        If False, the parameter passed as `alpha_or_var` is the variance.

    Returns
    -------
    array_like
        Parameter `p` for parameterization as used by scipy.stats.nbinom
        and numpy.random.negative_binomial.
    array_like
        Parameter `r` for parameterization as used by scipy.stats.nbinom
        and numpy.random.negative_binomial.
    """
    eps = 1e-6
    if param_is_alpha:
        var = mu + alpha_or_var * mu**2
    else:
        var = alpha_or_var
    p = 1 - (var - mu) / var
    r = (mu**2) / (var - mu + eps)
    return p, r


def sample_from_negative_binomial(
    rng, mu, alpha_or_var, param_is_alpha, size=None
):
    """Sample points from the negative binomial distribution.

    The parameterization chosen here is defined by the mean mu and
    the over-dispersion factor alpha (or via the variance).

        Var(x) = mu + alpha*mu**2

    [https://arxiv.org/pdf/1704.04110.pdf page 4]

    Alpha must be greater than zero.

    Parameters
    ----------
    rng : RandomState
        The random number generator to use.
    mu : array_like
        Mean of the distribution.
        Must be greater equal zero.
    alpha_or_var : array_like
        if param_is_alpha == True:
            The over-dispersion parameter.
            Must be greater zero.
        if param_is_alpha == False:
            The expected variance.
            Must be greater zero.
    param_is_alpha : bool
        If True, the parameter passed as `alpha_or_var` is alpha.
        If False, the parameter passed as `alpha_or_var` is the variance.
    size : int or array_like, optional
        The shape of random numbers to sample.

    Returns
    -------
    array_like
        The sampled points from the negative binomial distribution.
    """
    p, r = convert_neg_binomial_params(
        mu=mu, alpha_or_var=alpha_or_var, param_is_alpha=param_is_alpha
    )
    return rng.negative_binomial(r, p, size=size)


def negative_binomial_cdf(x, mu, alpha_or_var, param_is_alpha, dtype=None):
    """Computes the CDF of the negative binomial PDF

    The parameterization chosen here is defined by the mean mu and
    the over-dispersion factor alpha.

        Var(x) = mu + alpha*mu**2

    [https://arxiv.org/pdf/1704.04110.pdf page 4]

    Alpha must be greater than zero (best if >5e-5 due to numerical issues).

    Parameters
    ----------
    x : array_like
        The input values.
    mu : array_like
        Mean of the distribution.
        Must be greater equal zero.
    alpha_or_var : array_like
        if param_is_alpha == True:
            The over-dispersion parameter.
            Must be greater zero.
        if param_is_alpha == False:
            The expected variance.
            Must be greater zero.
    param_is_alpha : bool
        If True, the parameter passed as `alpha_or_var` is alpha.
        If False, the parameter passed as `alpha_or_var` is the variance.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    array_like
        CDF of the negative binomal PDF evaluated at x.
    """
    x, mu, alpha_or_var = cast(dtype, x, mu, alpha_or_var)
    p, r = convert_neg_binomial_params(
        mu=mu, alpha_or_var=alpha_or_var, param_is_alpha=param_is_alpha
    )
    return stats.nbinom(r, p).cdf(x)


def negative_binomial_ppf(q, mu, alpha_or_var, param_is_alpha, dtype=None):
    """Computes the PPF of the negative binomial PDF

    The parameterization chosen here is defined by the mean mu and
    the over-dispersion factor alpha.

        Var(x) = mu + alpha*mu**2

    [https://arxiv.org/pdf/1704.04110.pdf page 4]

    Alpha must be greater than zero (best if >5e-5 due to numerical issues).

    Parameters
    ----------
    q : array_like
        The quantiles at which to evaluate the PPF.
    mu : array_like
        Mean of the distribution.
        Must be greater equal zero.
    alpha_or_var : array_like
        if param_is_alpha == True:
            The over-dispersion parameter.
            Must be greater zero.
        if param_is_alpha == False:
            The expected variance.
            Must be greater zero.
    param_is_alpha : bool
        If True, the parameter passed as `alpha_or_var` is alpha.
        If False, the parameter passed as `alpha_or_var` is the variance.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    array_like
        PPF of the negative binomal PDF evaluated at x.
    """
    mu, alpha_or_var = cast(dtype, mu, alpha_or_var)
    p, r = convert_neg_binomial_params(
        mu=mu, alpha_or_var=alpha_or_var, param_is_alpha=param_is_alpha
    )
    return stats.nbinom(r, p).ppf(q)


def tf_rayleigh(x, sigma, dtype=None):
    """Computes Rayleigh PDF

    Parameters
    ----------
    x : tf.Tensor
        The input tensor.
    sigma : tf.Tensor
        The sigma parameter of the Rayleigh distribution.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    tf.Tensor
        The PDF of the Rayleigh distribution evaluated at x.
    """
    x, sigma = tf_cast(dtype, x, sigma)
    return x / (sigma**2) * tf.exp(-0.5 * (x / sigma) ** 2)


def rayleigh(x, sigma, dtype=None):
    """Computes Rayleigh PDF

    Parameters
    ----------
    x : array_like
        The input tensor.
    sigma : array_like
        The sigma parameter of the Rayleigh distribution.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    array_like
        The PDF of the Rayleigh distribution evaluated at x.
    """
    x, sigma = cast(dtype, x, sigma)
    return x / (sigma**2) * np.exp(-0.5 * (x / sigma) ** 2)


def tf_rayleigh_cdf(x, sigma, dtype=None):
    """Computes CDF of Rayleigh distribution.

    Parameters
    ----------
    x : tf.Tensor
        The input tensor.
    sigma : tf.Tensor
        The sigma parameter of the Rayleigh distribution.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    tf.Tensor
        The CDF of the Rayleigh distribution evaluated at x.
    """
    x, sigma = tf_cast(dtype, x, sigma)
    return 1 - tf.exp(-0.5 * (x / sigma) ** 2)


def rayleigh_cdf(x, sigma, dtype=None):
    """Computes CDF of Rayleigh distribution.

    Parameters
    ----------
    x : array_like
        The input tensor.
    sigma : array_like
        The sigma parameter of the Rayleigh distribution.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    array_like
        The CDF of the Rayleigh distribution evaluated at x.
    """
    x, sigma = cast(dtype, x, sigma)
    return 1 - np.exp(-0.5 * (x / sigma) ** 2)


def von_mises_pdf(x, sigma, kent_min=np.deg2rad(7), dtype=None):
    """Computes the von Mises-Fisher PDF on the sphere

    The PDF is defined and normalized in cartesian
    coordinates (dx1, dx2), i.e x = sqrt(dx1^2 + dx2^2).

    Parameters
    ----------
    x : array_like
        The opening angle (approximation in cartesian coordinates
        around the best-fit position).
    sigma : array_like
        The estimated uncertainty.
    kent_min : float, optional
        The value over which to use the von Mises-Fisher distribution.
        Underneath, a 2D Gaussian approximation is used for more numerical
        stability.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    array_like
        The PDF evaluated at the provided opening angles x.
    """
    x, sigma = cast(dtype, x, sigma)
    x = np.atleast_1d(x)
    sigma = np.atleast_1d(sigma)

    cos_dpsi = np.cos(x)
    kappa = 1.0 / sigma**2
    result = np.where(
        kent_min < sigma,
        (
            # kappa / (4 * np.pi * np.sinh(kappa)) *
            # np.exp(kappa * cos_dpsi)
            # stabilized version:
            kappa
            / (4 * np.pi)
            * 2
            * np.exp(kappa * (cos_dpsi - 1.0))
            / (1.0 - np.exp(-2.0 * kappa))
        ),
        # 1./(2*np.pi*sigma**2) * np.exp(-x**2 / (2*sigma**2)),
        1.0 / (2 * np.pi * sigma**2) * np.exp(-0.5 * (x / sigma) ** 2),
    )
    return result


def von_mises_in_dPsi_pdf(x, sigma, kent_min=np.deg2rad(7), dtype=None):
    """Computes the von Mises-Fisher PDF on the sphere

    The PDF is defined and normalized in the opening angle dPsi.

    Parameters
    ----------
    x : array_like
        The opening angle.
    sigma : array_like
        The estimated uncertainty.
    kent_min : float, optional
        The value over which to use the von Mises-Fisher distribution.
        Underneath, a 2D Gaussian approximation is used for more numerical
        stability.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    array_like
        The PDF evaluated at the provided opening angles x.
    """
    x, sigma = cast(dtype, x, sigma)

    # switching coordinates from (dx1, dx2) to spherical
    # coordinates (dPsi, phi) means that we have to include
    # the jakobi determinant sin dPsi
    jakobi_det = np.sin(x)  # sin(dPsi)
    phi_integration = 2 * np.pi
    return (
        phi_integration
        * jakobi_det
        * von_mises_pdf(x, sigma, kent_min=kent_min)
    )


def von_mises_in_dPsi_cdf(x, sigma, kent_min=np.deg2rad(7), dtype=None):
    """Computes the von Mises-Fisher CDF on the sphere

    The underlying PDF is defined and normalized in the opening angle dPsi.
    This function numerically integrates the underlying PDF and may therefore
    be rather slow. ToDo: analytic solution?

    Parameters
    ----------
    x : array_like
        The opening angle.
    sigma : array_like
        The estimated uncertainty.
    kent_min : float, optional
        The value over which to use the von Mises-Fisher distribution.
        Underneath, a 2D Gaussian approximation is used for more numerical
        stability.
    dtype : str, optional
        The data type of the output tensor, by default None.
        If provided, the inputs are cast to this data type.

    Returns
    -------
    array_like
        The CDF evaluated at the provided opening angles x.
    """
    assert len(x) == len(sigma), ("Unequal lengths:", len(x), len(sigma))
    x, sigma = cast(dtype, x, sigma)
    x = np.atleast_1d(x)
    sigma = np.atleast_1d(sigma)
    result = []
    for x_i, sigma_i in zip(x, sigma):
        integration = quad(
            von_mises_in_dPsi_pdf, a=0, b=x_i, args=(sigma_i, kent_min)
        )
        result.append(integration[0])
    return np.array(result)
