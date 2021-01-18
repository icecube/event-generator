import numpy as np


def weighted_quantile(x, weights, quantile=0.68):
    """Compute the weighted quantile

    Parameters
    ----------
    x : array_like
        The data points for which to compute the quantile.
    weights : array_like
        The weights for each data point.
    quantile : float, optional
        The quantile to compute

    Returns
    -------
    float
        The weighted quantile.
    """
    if weights is None:
        weights = np.ones_like(x)

    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    weights_sorted = weights[sorted_indices]
    cum_weights = np.cumsum(weights_sorted) / np.sum(weights)
    mask = cum_weights >= quantile

    return x_sorted[mask][0]


def weighted_std(x, weights=None):
    """"
    Weighted std deviation.
    Source:
    http://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf

    returns 0 if len(x)==1

    Parameters
    ----------
    x : array_like
        The data points for which to compute the weighted std. deviation.
    weights : array_like, optional
        The weights for each data point.

    Returns
    -------
    float
        The weighted std. deviation.
    """
    if len(x) == 1:
        return 0

    if weights is None:
        return np.std(x, ddof=1)

    x = np.asarray(x)
    weights = np.asarray(weights)

    w_mean_x = np.average(x, weights=weights)
    n = len(weights[weights != 0])

    s = n * np.sum(
        weights*(x - w_mean_x)*(x - w_mean_x)) / ((n - 1) * np.sum(weights))
    return np.sqrt(s)


def weighted_cov(x, y, w):
    """Weighted Covariance between two features.

    Parameters
    ----------
    x : array_like
        First component of data points.
    y : array_like
        Second component of data points.
    w : array_like
        The weights for each data point.

    Returns
    -------
    float
        The weighted covariance betwee x and y.
    """
    w_mean_x = np.average(x, weights=w)
    w_mean_y = np.average(y, weights=w)
    return np.sum(w * (x - w_mean_x) * (y - w_mean_y)) / np.sum(w)


def weighted_pearson_corr(x, y, w=None):
    """Weighted Pearson Correlation between two features.

    Parameters
    ----------
    x : array_like
        First component of data points.
    y : array_like
        Second component of data points.
    w : array_like, optional
        The weights for each data point.

    Returns
    -------
    float
        The weighted pearson correlatoin coefficient between x and y.
    """

    if w is None:
        return np.corrcoef(x, y)[0][1]
        # w = np.ones_like(x)

    return weighted_cov(x, y, w) / np.sqrt(
        weighted_cov(x, x, w) * weighted_cov(y, y, w))


def weighted_spearman_corr(x, y, w=None):
    """
    Weighted Spearman Correlation between two features.
    source:
    http://onlinelibrary.wiley.com/doi/10.1111/j.1467-842X.2005.00413.x/pdf

    Parameters
    ----------
    x : array_like
        First component of data points.
    y : array_like
        Second component of data points.
    w : array_like, optional
        The weights for each data point.

    Returns
    -------
    float
        The weighted spearman correlatoin coefficient between x and y.
    """

    if w is None:
        w = np.ones_like(x)

    ranks_x = stats.rankdata(x)
    ranks_y = stats.rankdata(y)
    return weighted_pearson_corr(ranks_x, ranks_y, w)
