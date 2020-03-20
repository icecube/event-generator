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
    return tf.exp(-0.5*(x - mu)**2 / sigma**2) / (2*np.pi*sigma**2)**0.5


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
