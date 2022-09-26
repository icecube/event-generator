import numpy as np
import tensorflow as tf
from scipy import special
from scipy import stats


def tf_gauss(x, mu, sigma):
    """Gaussian PDF

    Parameters
    ----------
    x : tf.Tensor
        The input tensor.
    mu : tf.Tensor
        Mu parameter of Gaussian.
    sigma : tf.Tensor
        Sigma parameter of Gaussian.

    Returns
    -------
    tf.Tensor
        The Gaussian PDF evaluated at x
    """
    return tf.exp(-0.5*((x - mu) / sigma)**2) / (2*np.pi*sigma**2)**0.5


def gauss(x, mu, sigma):
    """Gaussian PDF

    Parameters
    ----------
    x : array_like
        The input tensor.
    mu : array_like
        Mu parameter of Gaussian.
    sigma : array_like
        Sigma parameter of Gaussian.

    Returns
    -------
    array_like
        The Gaussian PDF evaluated at x
    """
    return np.exp(-0.5*((x - mu) / sigma)**2) / (2*np.pi*sigma**2)**0.5


def tf_log_gauss(x, mu, sigma):
    """Log Gaussian PDF

    Parameters
    ----------
    x : tf.Tensor
        The input tensor.
    mu : tf.Tensor
        Mu parameter of Gaussian.
    sigma : tf.Tensor
        Sigma parameter of Gaussian.

    Returns
    -------
    tf.Tensor
        The Gaussian PDF evaluated at x
    """
    norm = np.log(np.sqrt(2*np.pi))
    return -0.5*((x - mu) / sigma)**2 - tf.math.log(sigma) - norm


def log_gauss(x, mu, sigma):
    """Log Gaussian PDF

    Parameters
    ----------
    x : array_like
        The input tensor.
    mu : array_like
        Mu parameter of Gaussian.
    sigma : array_like
        Sigma parameter of Gaussian.

    Returns
    -------
    array_like
        The Gaussian PDF evaluated at x
    """
    norm = np.log(np.sqrt(2*np.pi))
    return -0.5*((x - mu) / sigma)**2 - np.log(sigma) - norm


def tf_log_asymmetric_gauss(x, mu, sigma, r):
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

    Returns
    -------
    tf.Tensor
        The asymmetric Gaussian PDF evaluated at x
    """
    norm = tf.math.log(2. / (tf.sqrt(2*np.pi*sigma**2) * (r+1)))
    exp = tf.where(
        x < mu,
        -0.5*((x - mu) / sigma)**2,
        -0.5*((x - mu) / (sigma*r))**2,
    )
    return norm + exp


def log_asymmetric_gauss(x, mu, sigma, r):
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

    Returns
    -------
    array_like
        The asymmetric Gaussian PDF evaluated at x
    """
    norm = np.log(2. / (np.sqrt(2*np.pi*sigma**2) * (r+1)))
    exp = np.where(
        x < mu,
        -0.5*((x - mu) / sigma)**2,
        -0.5*((x - mu) / (sigma*r))**2,
    )
    return norm + exp


def tf_asymmetric_gauss(x, mu, sigma, r):
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

    Returns
    -------
    tf.Tensor
        The asymmetric Gaussian PDF evaluated at x
    """
    norm = 2. / (tf.sqrt(2*np.pi*sigma**2) * (r+1))
    exp = tf.where(x < mu,
                   tf.exp(-0.5*((x - mu) / sigma)**2),
                   tf.exp(-0.5*((x - mu) / (sigma*r))**2),
                   )
    return norm * exp


def asymmetric_gauss(x, mu, sigma, r):
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

    Returns
    -------
    array_like
        The asymmetric Gaussian PDF evaluated at x
    """
    norm = 2. / (np.sqrt(2*np.pi*sigma**2) * (r+1))
    exp = np.where(x < mu,
                   np.exp(-0.5*((x - mu) / sigma)**2),
                   np.exp(-0.5*((x - mu) / (sigma*r))**2),
                   )
    return norm * exp


def tf_asymmetric_gauss_cdf(x, mu, sigma, r):
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

    Returns
    -------
    tf.Tensor
        The asymmetric Gaussian CDF evaluated at x
    """
    norm = 1. / (r + 1)
    exp = tf.where(x < mu,
                   1. + tf.math.erf((x - mu) / (np.sqrt(2) * sigma)),
                   1. + r * tf.math.erf((x - mu) / (np.sqrt(2) * r * sigma)),
                   )
    return norm * exp


def asymmetric_gauss_cdf(x, mu, sigma, r):
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

    Returns
    -------
    array_like
        The asymmetric Gaussian CDF evaluated at x
    """
    norm = 1. / (r + 1)
    exp = np.where(x < mu,
                   1. + special.erf((x - mu) / (np.sqrt(2) * sigma)),
                   1. + r * special.erf((x - mu) / (np.sqrt(2) * r * sigma)),
                   )
    return norm * exp


def tf_asymmetric_gauss_ppf(q, mu, sigma, r):
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

    Returns
    -------
    tf.Tensor
        The asymmetric Gaussian PPF evaluated at q
    """
    return tf.where(q < 1. / (r + 1),
                    mu + np.sqrt(2) * sigma * tf.math.erfinv(q*(r+1) - 1),
                    mu + r*np.sqrt(2)*sigma * tf.math.erfinv((q*(r+1) - 1)/r),
                    )


def asymmetric_gauss_ppf(q, mu, sigma, r):
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

    Returns
    -------
    array_like
        The asymmetric Gaussian PPF evaluated at q
    """
    return np.where(q < 1. / (r + 1),
                    mu + np.sqrt(2) * sigma * special.erfinv(q*(r+1) - 1),
                    mu + r*np.sqrt(2)*sigma * special.erfinv((q*(r+1) - 1)/r),
                    )


def tf_log_negative_binomial(x, mu, alpha, add_normalization_term=False):
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

    Returns
    -------
    tf.Tensor
        Logarithm of the negative binomal PDF evaluated at x.
    """
    inv_alpha = 1./alpha
    alpha_mu = alpha*mu

    # compute gamma terms
    gamma_terms = tf.math.lgamma(x + inv_alpha) - tf.math.lgamma(inv_alpha)

    if add_normalization_term:
        gamma_terms -= tf.math.lgamma(x + 1.)

    term1 = -inv_alpha * tf.math.log(1. + alpha_mu)
    term2 = x * tf.math.log(alpha_mu / (1. + alpha_mu))

    return gamma_terms + term1 + term2


def log_negative_binomial(x, mu, alpha, add_normalization_term=False):
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

    Returns
    -------
    array_like
        Logarithm of the negative binomal PDF evaluated at x.
    """
    inv_alpha = 1./alpha
    alpha_mu = alpha*mu

    # compute gamma terms
    gamma_terms = special.gammaln(x + inv_alpha) - special.gammaln(inv_alpha)

    if add_normalization_term:
        gamma_terms -= special.gammaln(x + 1.)

    term1 = -inv_alpha * np.log(1. + alpha_mu)
    term2 = x * np.log(alpha_mu / (1. + alpha_mu))

    return gamma_terms + term1 + term2


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
        var = mu + alpha_or_var*mu**2
    else:
        var = alpha_or_var
    p = 1 - (var - mu) / var
    r = (mu**2) / (var - mu + eps)
    return p, r


def sample_from_negative_binomial(rng, mu, alpha_or_var, param_is_alpha,
                                  size=None):
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
        mu=mu, alpha_or_var=alpha_or_var, param_is_alpha=param_is_alpha)
    return rng.negative_binomial(r, p, size=size)


def negative_binomial_cdf(x, mu, alpha_or_var, param_is_alpha):
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

    Returns
    -------
    array_like
        CDF of the negative binomal PDF evaluated at x.
    """
    p, r = convert_neg_binomial_params(
        mu=mu, alpha_or_var=alpha_or_var, param_is_alpha=param_is_alpha)
    return stats.nbinom(r, p).cdf(x)


def negative_binomial_ppf(q, mu, alpha_or_var, param_is_alpha):
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

    Returns
    -------
    array_like
        PPF of the negative binomal PDF evaluated at x.
    """
    p, r = convert_neg_binomial_params(
        mu=mu, alpha_or_var=alpha_or_var, param_is_alpha=param_is_alpha)
    return stats.nbinom(r, p).ppf(q)


def tf_rayleigh(x, sigma):
    """Computes Rayleigh PDF

    Parameters
    ----------
    x : tf.Tensor
        The input tensor.
    sigma : tf.Tensor
        The sigma parameter of the Rayleigh distribution.

    Returns
    -------
    tf.Tensor
        The PDF of the Rayleigh distribution evaluated at x.
    """
    return x/(sigma**2) * tf.exp(-0.5*(x/sigma)**2)


def rayleigh(x, sigma):
    """Computes Rayleigh PDF

    Parameters
    ----------
    x : array_like
        The input tensor.
    sigma : array_like
        The sigma parameter of the Rayleigh distribution.

    Returns
    -------
    array_like
        The PDF of the Rayleigh distribution evaluated at x.
    """
    return x/(sigma**2) * np.exp(-0.5*(x/sigma)**2)


def tf_rayleigh_cdf(x, sigma):
    """Computes CDF of Rayleigh distribution.

    Parameters
    ----------
    x : tf.Tensor
        The input tensor.
    sigma : tf.Tensor
        The sigma parameter of the Rayleigh distribution.

    Returns
    -------
    tf.Tensor
        The CDF of the Rayleigh distribution evaluated at x.
    """
    return 1 - tf.exp(-0.5*(x/sigma)**2)


def rayleigh_cdf(x, sigma):
    """Computes CDF of Rayleigh distribution.

    Parameters
    ----------
    x : array_like
        The input tensor.
    sigma : array_like
        The sigma parameter of the Rayleigh distribution.

    Returns
    -------
    array_like
        The CDF of the Rayleigh distribution evaluated at x.
    """
    return 1 - np.exp(-0.5*(x/sigma)**2)
