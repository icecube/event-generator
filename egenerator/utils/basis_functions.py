import numpy as np
import tensorflow as tf


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


def tf_log_negative_binomial(x, mu, alpha, add_normalization_term=False):
    """Computes the logarithm of the negative binomial PDF

    The parameterization chosen here is defined by the mean mu and
    the over-dispersion factor alpha.

        Var(x) = mu + alpha*mu**2

    Alpha must be greater than zero.

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
        gamma_terms -= tf.math.gamma(x + 1.)

    term1 = -inv_alpha * tf.log(1. + alpha_mu)
    term2 = x * tf.log(alpha_mu / (1. + alpha_mu))

    return gamma_terms + term1 + term2
