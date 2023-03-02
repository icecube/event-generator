#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
import os
import sys
misc_dir = os.path.dirname(__file__)
import numpy as np
import matplotlib
# matplotlib.use("AGG")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm 
from matplotlib.offsetbox import OffsetImage,AnnotationBbox
from cycler import cycler
import timeit
#import dashi as d
from itertools import cycle
import bottleneck as bn
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.utils import shuffle
from scipy import stats
from scipy.interpolate import LSQUnivariateSpline,UnivariateSpline,InterpolatedUnivariateSpline
from scipy.optimize import curve_fit

import math

DPI = 24

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def hasNans(List1D, quiet=False):
    counter = 0
    index = 0
    lastNan = -1
    for i in List1D:
        if i ==-float('Inf') or i==float('Inf') or math.isnan(i):
            # print i
            counter += 1
            lastNan = index
        index += 1
    if not quiet:
        print 'hasNans found',counter,'nans','last index:',lastNan
    if counter > 0:
        return True
    return False

def get_angle_deviation(azimuth1, zenith1, azimuth2, zenith2):
    cos_dist = ( np.cos(azimuth1 - azimuth2) *
                 np.sin(zenith1) * np.sin(zenith2) +
                 np.cos(zenith1) * np.cos(zenith2)
                )
    cos_dist = np.clip(cos_dist, -1. , 1.)
    return np.arccos(cos_dist)

def get_angle(vec1, vec2, dtype=np.float64):
    """
    vec1/2 : shape: [?,3] or [3]
    https://www.cs.berkeley.edu/~wkahan/Mindless.pdf p.47/56
    """
    # transform into numpy array with dtype
    vec1 = np.array(vec1, dtype=dtype)
    vec2 = np.array(vec2, dtype=dtype)

    assert vec1.shape[-1] == 3, "Expect shape [?,3] or [3], but got {}".format(vec1.shape)
    assert vec2.shape[-1] == 3, "Expect shape [?,3] or [3], but got {}".format(vec2.shape)

    norm1 = np.linalg.norm(vec1, axis=-1, keepdims = True)
    norm2 = np.linalg.norm(vec2, axis=-1, keepdims = True)
    tmp1 = vec1 * norm2
    tmp2 = vec2 * norm1

    tmp3 = np.linalg.norm( tmp1 - tmp2, axis=-1)
    tmp4 = np.linalg.norm( tmp1 + tmp2, axis=-1)

    theta = 2*np.arctan2(tmp3, tmp4)

    return theta

def rayleigh_distribution(x, sigma):
    exponent = (-x*x) / (2*sigma*sigma)
    return ( x/ ( sigma * sigma ) ) * np.exp( exponent )

def log_normal(x, sigma):
    mu = 0
    result = 1/( np.sqrt(2*np.pi)*sigma*x) * np.exp( - (np.log(x) - mu)**2/(2*sigma*sigma))
    result[x<0] = 0.0
    return result



#-----------------
# Weights
#-----------------
# https://de.mathworks.com/matlabcentral/answers/73307-can-matlab-do-weighted-spearman-rank-correlation
# https://stackoverflow.com/questions/38641691/weighted-correlation-coefficient-with-pandas
def m(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)

def weighted_quantile(x, weights, quantile=0.68):

    if weights is None:
        weights = np.ones_like(x)

    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    weights_sorted = weights[sorted_indices]
    cum_weights = np.cumsum(weights_sorted) / np.sum(weights)
    mask = cum_weights >= quantile

    return x_sorted[mask][0]

def weighted_median(x, weights):
    if weights is None:
        return np.median(x)
    return weighted_quantile(x,weights=weights, quantile=0.5)

def weighted_std(x,weights=None):
    """"
        Weighted std deviation. 
        Source: http://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf

        returns 0 if len(x)==1
    """
    if len(x) == 1:
        return 0
        
    if weights is None:
        return np.std(x, ddof=1)

    x = np.asarray(x)
    weights = np.asarray(weights)

    w_mean_x = np.average(x, weights=weights)
    n = len(weights[weights!=0])

    s = n * np.sum( weights*(x - w_mean_x)*(x - w_mean_x) ) / ( (n - 1) * np.sum(weights) )
    return np.sqrt(s)

def weighted_cov(x, y, w):
    """Weighted Covariance"""
    w_mean_x = np.average(x, weights=w)
    w_mean_y = np.average(y, weights=w)
    return np.sum(w * (x - w_mean_x) * (y - w_mean_y)) / np.sum(w)

def weighted_pearson_corr(x, y, w=None):
    """Weighted Pearson Correlation"""

    if w is None:
        return np.corrcoef(x,y)[0][1]
        # w = np.ones_like(x)

    return weighted_cov(x, y, w) / np.sqrt(weighted_cov(x, x, w) * weighted_cov(y, y, w))

def weighted_spearman_corr(x, y, w=None):
    """
        Weighted Spearman Correlation
        source: http://onlinelibrary.wiley.com/doi/10.1111/j.1467-842X.2005.00413.x/pdf
    """

    if w is None:
        return stats.spearmanr(y_test,y_pred)[0]
        # w = np.ones_like(x)

    ranks_x = stats.rankdata(x)
    ranks_y = stats.rankdata(y)
    return weighted_pearson_corr(ranks_x, ranks_y, w)

def get_powerlaw_weight(energy,
                        num_events_per_run,
                        num_runs,
                        one_weight,
                        gamma = 2.0,
                        norm = 0.9e-18,
                        e_pivot = 1e5,
                        ):
    '''
    Calculates weight for a powerlaw
    spectrum with index gamma.

    Parameters:
    ---

    energy: array-like
      Energy of primary particle in GeV

    num_events_per_run: int.
      Number of events per run/file

    num_runs: int.
      Number of runs or simulated files

    one_weight: array-like
      OneWeight saved in I3MCWeightDict

    gamma: float.
      Powerlaw spectrum index.
    '''

    assert gamma > 0, "gamma must be greater 0"
    n_gen = num_events_per_run*num_runs
    weight = norm*np.power(energy / e_pivot, - gamma)*one_weight/n_gen
    return weight


def get_weight( energy,
                ptype,
                cos_theta,
                p_int,
                dataset, # eg: 11069
                n_files,
                add_conv_flux = False,
                add_astro_flux = False,
                astro_gamma = -2.13,
            ):
    
    # make sure input are numpy arrays
    energy = np.asarray(energy, dtype=np.float64)
    ptype = np.asarray(ptype, dtype=np.int32)
    cos_theta = np.asarray(cos_theta, dtype=np.float64)
    p_int = np.asarray(p_int, dtype=np.float64)

    # print energy.shape, ptype.shape, cos_theta.shape, p_int.shape
    from icecube.weighting.weighting import from_simprod
    from icecube.icetray import I3Units

    flux = 0.

    if add_conv_flux:
        from icecube import NewNuFlux as nnflux

        conv = nnflux.makeFlux('honda2006')
        conv.knee_reweighting_model = 'gaisserH3a_elbert'
        conv_factor = 1.1
        flux += conv.getFlux(ptype, energy, cos_theta) * conv_factor

    # prompt = nnflux.makeFlux('sarcevic_std')
    # prompt.knee_reweighting_model = 'gaisserH3a_elbert'

    if add_astro_flux:
        norm = 0.9e-18
        e_pivot = 1e5
        # gamma = -2.13
        astro_flux = lambda energy: norm * np.power(energy / e_pivot, astro_gamma)
        flux += astro_flux(energy)


    generator = from_simprod(dataset) * n_files
    unit = I3Units.cm2 / I3Units.m2
    normalization = generator(energy, ptype, cos_theta)
    weight = p_int * (flux / unit) / normalization
    return weight

#-----------------
# Plotting
#-----------------
def splitTrainEvaluate(regressor,X,y,relTrainSize=0.5,calibrateRegressor=True,calibrateFeatures=False,ylabel=r'$\log_{10} \left( E_{MC} / {GeV} \right) $', xlabel=r'$\log_{10} \left( E_{rec} / {GeV} \right) $'):
    X,y = shuffle(X,y)
    size = len(y)
    split = int(size*relTrainSize)
    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]

    # if calibrateFeatures:
    #     zippedX_train = zip(*X_train)
    #     print 'zippedX_train',len(zippedX_train)
    #     # Train Calibrator on train data
    #     for attr in zippedX_train:
    #         print abs(stats.trim_mean(attr,0.2) - np.mean(attr)) / stats.trim_mean(attr,0.2) > 0.5
    #         plt.scatter(attr,y_train)
    #         plt.show()
    #     calibratorList = [ getCalibrator(attr,y_train) for attr in zippedX_train ]

    regressor.fit(X_train,y_train)
    start_time = timeit.default_timer()
    y_pred = regressor.predict(X_test)
    elapsed = timeit.default_timer() - start_time
    print '---------------------------------------------------------'
    print 'Time needed for Prediction {:3.3f}s. {:3.3f}ms / event'.format(elapsed,1000*elapsed/(size-split))
    print '---------------------------------------------------------'

    if calibrateRegressor:
        # Train Calibrator on train data
        y_pred_train = regressor.predict(X_train)
        calibrator = getCalibrator(y_pred_train,y_train, minBinContent=30)
        EvaluateRegPrediction(y_test,calibrator(y_pred),fileSuffix='Calibrated',ylabel=ylabel,xlabel=xlabel)
    EvaluateRegPrediction(y_test,y_pred,ylabel=ylabel,xlabel=xlabel)

def create_empty_array_of_shape(shape,dtype=np.float32):
        if shape: 
            return np.array([create_empty_array_of_shape(shape[1:]) for i in xrange(shape[0])],dtype=dtype)
        # else:
        #     return []

def uFloatOfList(listOfValues):
    return ufloat(np.mean(listOfValues),np.std(listOfValues))

def EvaluateRegEstimator(estimator,X,y,n_folds=10,dnnParams={},logdir='',
                            drawPlots=False,
                            ylabel=r'$\log_{10} \left( E_{MC} / {GeV} \right) $', 
                            xlabel=r'$\log_{10} \left( E_{rec} / {GeV} \right) $',
                            weights=None,
                            verbose=False,
                            **kwargs):
    X = np.array(X)
    y = np.array(y)
    X,y = shuffle(X,y)
    # KFold
    kf = KFold(len(X), n_folds=n_folds, shuffle=True)
    # kf = StratifiedKFold(Y, n_folds=n_folds, shuffle=True)

    mean_squared_error = []
    mean_absolute_error = []
    explained_variance_score = []
    r2_score = []
    residualMean = []
    residualStd = []
    corrPearson = []
    corrSpearman = []
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # fit and predict
        if dnnParams:
            estimator = skflow.TensorFlowDNNClassifier(**dnnParams)
        if logdir:
            estimator.fit(X_train,y_train,logdir=logdir)
        else:
            estimator.fit(X_train,y_train)
        y_pred = estimator.predict(X_test)

        # calculate scores
        residualMean.append(np.mean(y_pred - y_test))
        residualStd.append(np.std(y_pred - y_test))
        corrPearson.append(np.corrcoef(y_test,y_pred)[0][1])
        corrSpearman.append(stats.spearmanr(y_test,y_pred)[0])
        mean_squared_error.append(metrics.mean_squared_error(y_test,y_pred))
        mean_absolute_error.append(metrics.mean_absolute_error(y_test,y_pred))
        explained_variance_score.append(metrics.explained_variance_score(y_test,y_pred))
        r2_score.append(metrics.r2_score(y_test,y_pred))
    results = [ uFloatOfList(mean_squared_error),       uFloatOfList(mean_absolute_error),
                uFloatOfList(explained_variance_score), uFloatOfList(r2_score),
                uFloatOfList(residualMean),             uFloatOfList(residualStd),
                uFloatOfList(corrPearson),              uFloatOfList(corrSpearman)]
    if verbose:
        print '---------------------------------------------'
        print 'Results'
        print '---------------------------------------------'
        print 'Mean Squared Error: ', results[0]
        print 'Mean Absolute Error:', results[1]
        print 'Explained Variance: ', results[2]
        print 'R^2:                ', results[3]
        print 'Residual Mean:      ', results[4]
        print 'Residual Std:       ', results[5]
        print 'Pearson Corr:       ', results[6]
        print 'Spearman Corr:      ', results[7]

    if drawPlots:
        X,y = shuffle(X,y)
        size = len(y)
        cut = int(size/2)
        X_train = X[:cut]
        y_train = y[:cut]
        X_test = X[cut:]
        y_test = y[cut:]
        # fit and predict
        if dnnParams:
            estimator = skflow.TensorFlowDNNClassifier(**dnnParams)
        if logdir:
            estimator.fit(X_train,y_train,logdir=logdir)
        else:
            estimator.fit(X_train,y_train)
        y_pred = estimator.predict(X_test)
        dy = y_pred - y_test
        # results = (len(y_test),np.mean(dy),np.std(dy),np.mean(abs(dy)),np.corrcoef(y_test,y_pred)[0][1],stats.spearmanr(y_test,y_pred)[0])
        results = ( len(y_test),
                    np.average(dy,weights=weights),
                    weighted_std(dy, weights=weights),
                    np.average(abs(dy),weights=weights),
                    weighted_pearson_corr(y_test,y_pred,w=weights),
                    weighted_spearman_corr(y_test,y_pred,w=weights),
                  )
        plotAttributes(y_pred,y_test,ylabel=ylabel,xlabel=xlabel,xyrange=[[-1,8],[-1,8]],results=results,weights=weights,**kwargs)
    return results


def EvaluateRegPrediction(y_true,y_pred,verbose=False,drawSplineFit=False,pathToPlots='',xyrange=[[-1,8],[-1,8]],filePrefix='',file='',plotTitle='',
                            fileSuffix='',ylabel=r'$\log_{10} \left( E_{MC} / {GeV} \right) $', xlabel=r'$\log_{10} \left( E_{rec} / {GeV} \right) $',
                            alpha_y_true=None, alpha_y_pred=None, alpha_weights=None,
                            watermark=None, weights=None,use_energy_cut=False,
                            correct_2pi_periodicity=False,
                            **kwargs):
    if verbose:
        print '---------------------------------------------'
        print xlabel
        print 'Results'
        print '---------------------------------------------'
        print 'Mean Squared Error: ', metrics.mean_squared_error(y_true,y_pred)
        print 'Mean Absolute Error:', metrics.mean_absolute_error(y_true,y_pred)
        print 'Explained Variance: ', metrics.explained_variance_score(y_true,y_pred)
        print 'R^2:                ', metrics.r2_score(y_true,y_pred)
        print 'Residual Mean:      ', np.mean(y_pred - y_true)
        print 'Residual Std:       ', np.std(y_pred - y_true)
        print 'Pearson Corr:       ', np.corrcoef(y_true,y_pred)[0][1]
        print 'Spearman Corr:      ', stats.spearmanr(y_true,y_pred)[0]
    dy = y_pred - y_true

    if correct_2pi_periodicity:

        # check if given in radians
        assert np.max(y_true) < 2*np.pi + 1e-4, 'Expecting radians as unit'
        assert np.min(y_true) > -1e-4, 'Expecting radians as unit'

        mask_greater_pi = dy > np.pi
        mask_less_n_pi = dy < -np.pi
        dy[mask_greater_pi] = dy[mask_greater_pi] - 2*np.pi
        dy[mask_less_n_pi] = dy[mask_less_n_pi] + 2*np.pi

    # results = (len(y_true),np.mean(dy),np.std(dy),np.mean(abs(dy)),np.corrcoef(y_true,y_pred)[0][1],stats.spearmanr(y_true,y_pred)[0])
    # if weights is None or (weights == 1).all():
    #     size = len(y_true)
    # else:
    #     size = int(np.sum(weights)*3600*24*365) # events per year
    size = len(y_true)
    
    results = ( size,
                    np.average(dy,weights=weights),
                    weighted_std(dy, weights=weights),
                    np.average(abs(dy),weights=weights),
                    weighted_pearson_corr(y_true,y_pred,w=weights),
                    weighted_spearman_corr(y_true,y_pred,w=weights),
                    get_proxy_resolution(proxy=y_pred, label=y_true, 
                                        proxy_bins=50, 
                                        label_bins=50, 
                                        weights=weights)[0]
                  )
    #-------- 
    # HACK ENERGY CUT
    #--------
    # HACK to insert a boundary at 1TeV; ENERGY CUT
    if use_energy_cut:
        mask = y_true > 3 # HACK

        results = ( size,# HACK ENERGY CUT
                        np.average(dy[mask],weights=weights[mask]),# HACK ENERGY CUT
                        weighted_std(dy[mask], weights=weights[mask]),# HACK ENERGY CUT
                        np.average(abs(dy[mask]),weights=weights[mask]),# HACK ENERGY CUT
                        weighted_pearson_corr(y_true[mask],y_pred[mask],w=weights[mask]),# HACK ENERGY CUT
                        weighted_spearman_corr(y_true[mask],y_pred[mask],w=weights[mask]),# HACK ENERGY CUT
                        get_proxy_resolution(proxy=y_pred[mask], label=y_true[mask], 
                                        proxy_bins=50, 
                                        label_bins=50, 
                                        weights=weights[mask])[0]
                      )# HACK ENERGY CUT
    #--------
    plotAttributes(y_pred,y_true,ylabel=ylabel,xlabel=xlabel,xyrange=xyrange,pathToPlots=pathToPlots,
                filePrefix=filePrefix,fileSuffix=fileSuffix,results=results,drawSplineFit=drawSplineFit,
                file=file,plotTitle=plotTitle,alpha_x=alpha_y_pred, alpha_y=alpha_y_true, alpha_weights=alpha_weights,
                watermark=watermark,weights=weights,use_energy_cut=use_energy_cut,
                correct_2pi_periodicity=correct_2pi_periodicity,**kwargs)

#--------------------
# Watermark
#--------------------
def draw_watermark(watermark,ax, 
                            xcoord=None, 
                            ycoord=None, 
                            zoom=0.5,
                            Lfontsize=200):

    if not watermark is None:
        if xcoord is None:
            xlim = ax.get_xlim()
            if ax.get_xaxis().get_scale() == 'log':
                xcoord = xlim[0]
            else:
                xcoord = xlim[0] + 0.01*(xlim[1] - xlim[0])
        if ycoord is None:
            ylim = ax.get_ylim()
            if ax.get_yaxis().get_scale() == 'log':
                ycoord = ylim[1]
            else:
                ycoord = ylim[1] - 0.01*(ylim[1] - ylim[0])

        if watermark == 'DNN':
            filename = os.path.join(misc_dir, 'figures/DNN_Icon2.png')
            arr_img = plt.imread(filename, format='png')

            
            imagebox = OffsetImage(arr_img, zoom=zoom, alpha=.2, zorder=-1)
            imagebox.image.axes = ax

            ab = AnnotationBbox(imagebox, [xcoord,ycoord],
                                xybox=(0,0),#(300*zoom, 300*zoom),
                                xycoords='data',
                                boxcoords="offset points",
                                box_alignment=(0,1),
                                pad=0.0,
                                frameon=False,
                                )
            ab.zorder=-1

            ax.add_artist(ab)
        elif watermark == 'L':
             # plt.text(xcoord, ycoord, r'$\mathcal{L}$', ha='left', va='top',fontsize=Lfontsize, alpha=0.2, zorder=-1)
             plt.text(xcoord, ycoord, r'$\mathcal{L}$', ha='left', va='top',fontsize=Lfontsize, color='0.8', zorder=-1)
        else:
            raise ValueError('Unknown Watermark: {}'.format(watermark))


def plotAttributes(x, y, file='',xyrange=[[-4,4],[-4,4]],results='',k='',plotTitle='',filePrefix='',
                    fileSuffix='',drawSplineFit=False,pathToPlots='',
                    ylabel=r'$\log_{10} \left( E_{MC} / {GeV} \right) $', 
                    xlabel=r'$\log_{10} \left( E_{rec} / {GeV} \right) $',
                    vmin=None, vmax=None,
                    alpha_x=None, alpha_y=None,
                    watermark=None,
                    weights=None,
                    alpha_weights=None,
                    use_energy_cut=False,
                    which_error = 'x',
                    print_alpha_results=True,
                    correct_2pi_periodicity=False,
                    auto_adjust_angles=True,
                    ):
    
    if weights is None:
        weights = np.ones_like(x)

    # print('Total Rate: {}Hz'.format(np.sum(weights))) # DEBUG
    # print('Neutrinos per Year: {}'.format(np.sum(weights) * 3600 * 24 * 365))  # DEBUG
    # if not alpha_x is None:
    #     raw_input() # DEBUG
    
    plt.figure(figsize=(20, 16), dpi=80)
    # plt.figure(figsize=(20, 20), dpi=300)# QUALITY
    matplotlib.rcParams.update({'font.size': 43})
    plt.clf()
    # Adjust Axes for Zenith and Azimuth
    ax = plt.gca()

    if auto_adjust_angles:
        if 'Zenith' in xlabel:
            xyrange[0][0] = -0.5
            xyrange[0][1] = np.pi +0.5
            ax.set_xticks([0., .25*np.pi, .5*np.pi, .75*np.pi, 1*np.pi])
            ax.set_xticklabels(["$0$", r"$\frac{1}{4}\pi$", r"$\frac{1}{2}\pi$", r"$\frac{3}{4}\pi$", r"$\pi$"])
        elif 'Azimuth' in xlabel:
            xyrange[0][0] = -1.5
            xyrange[0][1] = 2*np.pi +0.5
            ax.set_xticks([0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])
            ax.set_xticklabels(["$0$", r"$\frac{1}{2}\pi$",r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"])
        if 'Zenith' in ylabel:
            xyrange[1][0] = -0.5
            xyrange[1][1] = np.pi +0.5
            ax.set_yticks([0., .25*np.pi, .5*np.pi, .75*np.pi, 1*np.pi])
            ax.set_yticklabels(["$0$", r"$\frac{1}{4}\pi$", r"$\frac{1}{2}\pi$", r"$\frac{3}{4}\pi$", r"$\pi$"])
        elif 'Azimuth' in ylabel:
            xyrange[1][0] = -0.5
            xyrange[1][1] = 2*np.pi + 0.75
            ax.set_yticks([0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])
            ax.set_yticklabels(["$0$", r"$\frac{1}{2}\pi$",r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"])

    xdata = np.linspace(xyrange[0][0], xyrange[0][1], 100)
    heatmap, xedges, yedges = np.histogram2d(y,x,range=xyrange, bins=100, weights=weights)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # # if vmax == None:
    # #     vmax = np.max(heatmap) # 1e4
    # #     vmin = np.min(heatmap)
    # #     print heatmap
    # #     print 'vmax:',vmax
    # #     print 'vmin:',vmin
    # if vmin == None:
    #     weights_sorted = np.sort(weights)
    #     # tmp = np.sum(weights_sorted[:int(weights_sorted.shape[0]*0.03)])
    #     tmp = weights_sorted[int(weights_sorted.shape[0]*0.4)]
    #     if tmp < 1.0:
    #         vmin = tmp
    #         vmin = 1e-9
    #         print('Setting vmin to {}'.format(vmin))

    #--------------------
    # Watermark
    #--------------------
    xcoord = xyrange[0][0] + 0.01*(xyrange[0][1] - xyrange[0][0])
    ycoord = xyrange[1][1] - 0.1*(xyrange[1][1] - xyrange[1][0])
    draw_watermark(watermark,ax,
                    xcoord, ycoord)

    #--------------------
    # alpha
    #--------------------
    # if isinstance(alpha_x, np.ndarray) and isinstance(alpha_y, np.ndarray) :
    if (not alpha_x is None) and (not alpha_y is None) :
        # if weights is None or (weights == 1).all():
        #     alpha_size = len(alpha_x)
        # else:
        #     alpha_size = int(np.sum(alpha_weights)*3600*24*365) # events per year
        alpha_size = len(alpha_x)

        dy = alpha_x - alpha_y

        if correct_2pi_periodicity:

            # check if given in radians
            assert np.max(alpha_y) < 2*np.pi + 1e-4, 'Expecting radians as unit'
            assert np.min(alpha_y) > -1e-4, 'Expecting radians as unit'

            mask_greater_pi = dy > np.pi
            mask_less_n_pi = dy < -np.pi
            dy[mask_greater_pi] = dy[mask_greater_pi] - 2*np.pi
            dy[mask_less_n_pi] = dy[mask_less_n_pi] + 2*np.pi

        alpha_results = ( alpha_size,
                    np.average(dy,weights=alpha_weights),
                    weighted_std(dy, weights=alpha_weights),
                    np.average(abs(dy),weights=alpha_weights),
                    weighted_pearson_corr(alpha_y,alpha_x,w=alpha_weights),
                    weighted_spearman_corr(alpha_y,alpha_x,w=alpha_weights),
                    get_proxy_resolution(proxy=alpha_x, label=alpha_y, 
                                        proxy_bins=50, 
                                        label_bins=50, 
                                        weights=alpha_weights)[0]
                  )

        # plotTitle = False
        plt.hexbin(alpha_x, alpha_y, 
                        gridsize=100, 
                        # bins='log', 
                        # vmin=None, 
                        # mincnt=0, 
                        extent=extent, 
                        cmap='viridis', 
                        alpha=0.2, 
                        C=alpha_weights, 
                        norm=LogNorm(vmin=vmin, vmax=vmax),
                        reduce_C_function=np.sum,
                        rasterized=True,
                        )

    #--------------------
    # Plot Hexbin
    #--------------------
    # plt.imshow(heatmap,extent=extent,origin='lower',cmap='jet', interpolation='nearest',norm=LogNorm(vmin=vmin))#vaxm=vmax
    # plt.colorbar()
    # plt.hexbin(x, y, gridsize=100, bins='log', extent=extent, cmap='viridis', vmin=None, mincnt=0, C=weights, reduce_C_function=np.sum,)
    plt.hexbin(x, y, gridsize=100, 
                     extent=extent, 
                     cmap='viridis', 
                     norm=LogNorm(vmin=vmin, vmax=vmax), 
                     C=weights, 
                     reduce_C_function=np.sum,
                     rasterized=True,
                     )
    
    cb = plt.colorbar()
    # cb.set_label('log10(counts)')
    cb.set_label('Event Rate / arb. unit')
    # cb.set_label('Event Rate / Hz')
    # cb.set_label('Number of Events')
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot((xyrange[0][0],xyrange[0][1]),(xyrange[0][0],xyrange[0][1]),'-',color='red',linewidth=1)
    plt.xlim(xyrange[0])
    plt.ylim(xyrange[1])
    

    #--------------------
    # Plot Error
    #    Todo: clean this up and make it more clear
    #--------------------
    if which_error == 'x': #Fast HACK to change error bar axis (very ugly)
        axis_x = y
        axis_y = x
    else:
        axis_x = x
        axis_y = y
    hist, bin_edges = np.histogram(axis_x,bins='doane')
    indices = np.digitize(axis_x,bin_edges)
    noOfBins = len(hist)
    binDataX = [[] for i in xrange(max(indices))]
    binDataY = [[] for i in xrange(max(indices))]
    binWeights = [[] for i in xrange(max(indices))]
    for index,yValue,xValue,weightValue in zip(indices,axis_y,axis_x,weights):
        binDataX[index-1].append(xValue)
        binDataY[index-1].append(yValue)
        binWeights[index-1].append(weightValue)

    minBinContent = 5
    binMids = np.array([(bin_edges[i]+bin_edges[i+1])/2 for i in xrange(noOfBins) if len(binDataY[i]) >= minBinContent ])
    # binMeans = np.array([bn.nanmean(binDataY[i]) for i in xrange(noOfBins) if len(binDataY[i]) >= minBinContent ])
    # binStdsY = np.array([bn.nanstd(binDataY[i]) for i in xrange(noOfBins) if len(binDataY[i]) >= minBinContent ])
    # binStdsX = np.array([bn.nanstd(binDataX[i]) for i in xrange(noOfBins) if len(binDataY[i]) >= minBinContent ])
    
    binMeans = []
    binStdsY = []
    binStdsX = []
    for i in xrange(noOfBins):
        if len(binDataY[i]) >= minBinContent:
            nonNanY = np.isfinite(binDataY[i])
            binDataY_valid = np.asarray( binDataY[i] ) [nonNanY]
            binWeights_valid = np.asarray( binWeights[i] ) [nonNanY]
            binMeans.append( np.average(binDataY_valid,weights=binWeights_valid) )
            binStdsY.append( weighted_std(binDataY_valid,weights=binWeights_valid) )
        if len(binDataX[i]) >= minBinContent:
            nonNanX = np.isfinite(binDataX[i])
            binDataX_valid = np.asarray( binDataX[i] ) [nonNanX]
            binWeights_valid = np.asarray( binWeights[i] ) [nonNanX]
            binStdsX.append( weighted_std(binDataX_valid,weights=binWeights_valid) )

    binMeans = np.asarray(binMeans)
    binStdsY = np.asarray(binStdsY)
    binStdsX = np.asarray(binStdsX)

    mask = np.logical_not(np.isnan(binMeans))
    binMids = binMids[mask]
    binMeans = binMeans[mask]
    binStdsY = binStdsY[mask]
    binStdsX = binStdsX[mask]
    # plt.errorbar(binMids, binMeans, yerr=binStdsY,xerr=binStdsX, fmt='o',markersize=15,color='black',elinewidth=5,capthick=5,capsize=10)
    if which_error == 'y':
        plt.errorbar(binMids, binMeans, yerr=binStdsY, fmt='o',markersize=15,color='black',elinewidth=5,capthick=5,capsize=10)
    else:
        plt.errorbar(binMeans,binMids, xerr=binStdsY, fmt='o',markersize=15,color='black',elinewidth=5,capthick=5,capsize=10)
    #--------------------------------
    if results!='':
        xcoord = xyrange[0][0] + 0.04545*(xyrange[0][1] - xyrange[0][0])
        ycoord = xyrange[1][1] - 0.05454*(xyrange[1][1] - xyrange[1][0])
        plt.text(xcoord,ycoord, 'Size: {}'.format(results[0]), ha='left', va='bottom',fontsize=32)

        #-------
        # alpha
        #-------
        if print_alpha_results and isinstance(alpha_x, np.ndarray) and isinstance(alpha_y, np.ndarray):
            ycoord = xyrange[1][1] - 0.09*(xyrange[1][1] - xyrange[1][0])
            plt.text(xcoord, ycoord,  'Size: {}'.format(alpha_size), ha='left', va='bottom',fontsize=30, alpha=0.3)
            if correct_2pi_periodicity:
                s = ("Residuals: \n  Mean: %2.4f \n  Stddev: %2.3f \nMAE: %2.4f " % (alpha_results[1:4]))
                xcoord = xyrange[0][1] - 0.27*(xyrange[0][1] - xyrange[0][0])
                # ycoord = xyrange[1][1] - 0.095*(xyrange[1][1] - xyrange[1][0]) # plot on top
                ycoord = xyrange[1][0] + 0.27*(xyrange[1][1] - xyrange[1][0]) # plot over other text box
            else:
                s = ("Residuals: \n  Mean: %2.4f \n  Stddev: %2.3f \nMAE: %2.4f \nPearson: %1.3f \nSpearMr: %1.3f \nRes.: %1.3f " % (alpha_results[1:]))
                xcoord = xyrange[0][1] - 0.27*(xyrange[0][1] - xyrange[0][0])
                ycoord = xyrange[1][0] + 0.44*(xyrange[1][1] - xyrange[1][0])
            props = dict(boxstyle='round', facecolor='0.9',edgecolor=u'#1f77b4', alpha=0.3)
            plt.text(xcoord,ycoord,s, ha='left', va='center',fontsize=30, alpha=0.3, bbox=props)

        if correct_2pi_periodicity:
            s = ("Residuals: \n  Mean: %2.4f \n  Stddev: %2.3f \nMAE: %2.4f " % (results[1:4]))
            xcoord = xyrange[0][1] - 0.27*(xyrange[0][1] - xyrange[0][0])
            ycoord = xyrange[1][0] + 0.095*(xyrange[1][1] - xyrange[1][0])
        else:
            s = ("Residuals: \n  Mean: %2.4f \n  Stddev: %2.3f \nMAE: %2.4f \nPearson: %1.3f \nSpearMr: %1.3f \nRes.: %1.3f " % (results[1:]))
            xcoord = xyrange[0][1] - 0.27*(xyrange[0][1] - xyrange[0][0])
            ycoord = xyrange[1][0] + 0.15*(xyrange[1][1] - xyrange[1][0])

        # xcoord = xyrange[0][1] - 0.05*(xyrange[0][1] - xyrange[0][0])
        # ycoord = xyrange[1][0] + 0.05*(xyrange[1][1] - xyrange[1][0])
        props = dict(boxstyle='round', facecolor='0.9',edgecolor=u'#1f77b4', alpha=0.8)
        plt.text(xcoord,ycoord,s, ha='left', va='center',fontsize=30, bbox=props,
                # horizontalalignment='right', verticalalignment='bottom'
                )

    if plotTitle!='' and not plotTitle is None:
        if k == '':
            plt.title(plotTitle,color='black',y=1.01,x=0.47,fontsize=64)
            # plt.title(plotTitle,color='black',y=1.08,fontsize=80)
        else:
            plt.title(plotTitle+' ['+str(k)+']',y=1.08)
    #--------------------------------
    if drawSplineFit:
        print 'Drawing Spline Fit'
        f = getCalibrator(x,y,minBinContent=minBinContent, which_error='y', weights=weights)
        # if which_error == 'y':
        #     f = InterpolatedUnivariateSpline(binMids,binMeans,k=1)
        # else:
        #     f = InterpolatedUnivariateSpline(binMeans,binMids,k=1)
        xValues = np.linspace(-1,8,100)
        plt.plot(xValues,f(xValues),'-',color='pink',linewidth='4')
    #--------------------------------
    
    #-------- 
    # HACK ENERGY CUT
    #--------
    if use_energy_cut:
        plt.plot((xyrange[0][0],xyrange[0][1]),(3.0,3.0),'--',color='red',linewidth=5) # HACK ENERGY CUT
        plt.arrow(0.5, 3.0, 0.0, 0.2, head_width=0.15, head_length=0.1, fc='r', ec='r')# HACK ENERGY CUT
        plt.arrow(7.5, 3.0, 0.0, 0.2, head_width=0.15, head_length=0.1, fc='r', ec='r')# HACK ENERGY CUT
    #--------

    plt.tight_layout()
    # plt.show()
    if file=='':
        file = ylabel+'_'+xlabel
    plt.savefig(pathToPlots+'plots/'+filePrefix+'scatter_' +str(file) +fileSuffix+'.png', dpi=DPI)
    # plt.savefig(pathToPlots+'plots/'+filePrefix+'scatter_' +str(file) +fileSuffix+'.svg', dpi=DPI)
    # plt.savefig(pathToPlots+'plots/'+filePrefix+'scatter_' +str(file) +fileSuffix+'.svg')# QUALITY
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plt.close()

def getCalibrator(y_pred, y_true, minBinContent=30,verbose=False,which_error='y', weights=None):

    if weights is None:
        weights = np.ones_like(y_pred)

    if which_error == 'x':
        axis_x = y_true
        axis_y = y_pred
    else:
        axis_x = y_pred
        axis_y = y_true
    hist, bin_edges = np.histogram(axis_x,bins='doane')
    indices = np.digitize(axis_x,bin_edges)
    noOfBins = len(hist)
    binDataX = [[] for i in xrange(max(indices))]
    binDataY = [[] for i in xrange(max(indices))]
    binWeights = [[] for i in xrange(max(indices))]
    for index,yValue,xValue,weightValue in zip(indices,axis_y,axis_x,weights):
        binDataX[index-1].append(xValue)
        binDataY[index-1].append(yValue)
        binWeights[index-1].append(weightValue)
    minBinContent = 30
    binMids = np.array([(bin_edges[i]+bin_edges[i+1])/2 for i in xrange(noOfBins) if len(binDataY[i]) >= minBinContent ])
    # binMeans = np.array([bn.nanmean(binDataY[i]) for i in xrange(noOfBins) if len(binDataY[i]) >= minBinContent ])
    binMeans = []
    for i in xrange(noOfBins):
        if len(binDataY[i]) >= minBinContent:
            nonNanY = np.isfinite(binDataY[i])
            binDataY_valid = np.asarray( binDataY[i] ) [nonNanY]
            binWeights_valid = np.asarray( binWeights[i] ) [nonNanY]
            binMeans.append( np.average(binDataY_valid,weights=binWeights_valid) )
    binMeans = np.asarray(binMeans)
    
    mask = np.logical_not(np.isnan(binMeans))
    binMids = binMids[mask]
    binMeans = binMeans[mask]

    if which_error == 'y':
        f = InterpolatedUnivariateSpline(binMids,binMeans,k=1)
    else:
        f = InterpolatedUnivariateSpline(binMeans,binMids,k=1)

    # hist, bin_edges = np.histogram(y_pred,bins='doane')
    # indices = np.digitize(y_pred,bin_edges)
    # noOfBins = len(hist)
    # binDataX = [[] for i in xrange(max(indices))]
    # binDataY = [[] for i in xrange(max(indices))]
    # for index,yValue,xValue in zip(indices,y_true,y_pred):
    #     binDataX[index-1].append(xValue)
    #     binDataY[index-1].append(yValue)
    # binMids = np.array([(bin_edges[i]+bin_edges[i+1])/2 for i in xrange(noOfBins) if len(binDataY[i]) >= minBinContent ])
    # binMeans = np.array([bn.nanmean(binDataY[i]) for i in xrange(noOfBins) if len(binDataY[i]) >= minBinContent ])
    # mask = np.logical_not(np.isnan(binMeans))
    # binMids = binMids[mask]
    # binMeans = binMeans[mask]
    # f = InterpolatedUnivariateSpline(binMids,binMeans,k=1)

    if len(binMeans) <3:
        if verbose:
            print 'getCalibrator: Too few non-empty bins, returning lambda x: x'
        return lambda x: x 

    return f

def PlotScatter(x,y,file='',bins=50,xLabel='x',yLabel='y'):
    d.visual()
    h = d.factory.hist2d( (x,y), bins, labels=(xLabel, yLabel))
    figure = plt.figure(figsize=(8,8))
    h.imshow(log=1)
    cb = plt.colorbar()
    plt.grid()
    if file!='':
        plt.savefig('plots/'+file+'.png',dpi=DPI)
    else:
        plt.show()
    plt.clf()
    plt.close(figure)

def plotResolution(pred, true, binning_variable):
    '''
    Plots resolution in bins along binning_variable

    Parameters
    ----------
    pred : list/array of floats 
            predicted value


    true : list/array of floats 
            true value

    binning_variable : list/array of floats 
            value of the variable along which the 
            binning should be performed
    '''
    raise NotImplemented()


def plot_pull_distribution(true_residuals,
                            pred_residuals, 
                            binning_variable, 
                            bin_width = None,
                            weights = None,
                            xlabel = 'Binning Variable',
                            ylabel = r'Pull Value $\Delta\Psi / \sigma_{\text{pred}}$',
                            vmin = None,
                            vmax = None,
                            xyrange = None,
                            file = None,
                            pathToPlots='',
                            title = None,
                            watermark = None,
                            ):
    '''
    Plots pull distribution in bins along binning_variable

    Parameters
    ----------
    true_residuals : list/array of floats 
            true residual values


    pred_residuals : list/array of floats 
            predicted residual values

    binning_variable : list/array of floats 
            value of the variable along which the 
            binning should be performed

    bin_width : float
            Determine size of bin: bin mid +- bin_width.
            If bin_width is None, this will automatically be chosen
    '''
    if weights is None:
        weights= np.ones_like(true_residuals)
    # pred_residuals *= 0.707106
    pull_values = true_residuals / pred_residuals

    #---------------------------------------------------
    # calculate extend and binning
    #---------------------------------------------------
    heatmap, xedges, yedges = np.histogram2d(pull_values, binning_variable, range=xyrange, bins=100, weights=weights)# range=xyrange
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    #---------------------------------------------------
    # calculate median and mean along binning_variable
    #---------------------------------------------------
    min_bin = xedges[0]#min(binning_variable)
    max_bin = xedges[-1]#max(binning_variable)

    x_bin_centers = np.linspace(min_bin, max_bin, 100)
    if bin_width is None:
        bin_width = (max_bin - min_bin) / binning_variable.shape[0] * 50
        # print 'Using a bin_width of', bin_width

    mean_pulls = []
    median_pulls = []
    x_centers = []
    for x_center in x_bin_centers:
        # get a mask of all events within +- bin_width of x_center
        mask = (x_center - bin_width < binning_variable) &  (binning_variable < x_center + bin_width)
        # print len(angle_deviation[mask]), x_center, np.median(binning_variable[mask]), np.mean(binning_variable[mask])
        if binning_variable[mask].shape[0] > 10:
            mean_pulls.append( np.average(pull_values[mask], weights=weights[mask]) )
            median_pulls.append( weighted_median(pull_values[mask], weights=weights[mask]) )
            x_centers.append(x_center)

    # print median_pulls, np.mean(median_pulls), np.median(median_pulls)
    #---------------------------------------------------
    # Plot scatter plot: x: binning_variable, y: pull
    #---------------------------------------------------
    plt.figure(figsize=(18, 15), dpi=80)
    # plt.figure(figsize=(20, 20), dpi=300)# QUALITY
    matplotlib.rcParams.update({'font.size': 48})
    plt.clf()

    plt.hexbin(binning_variable, pull_values, gridsize=100, 
                     extent=extent, 
                     cmap='viridis', 
                     norm=LogNorm(vmin=vmin, vmax=vmax), 
                     C=weights, 
                     reduce_C_function=np.sum,
                     rasterized=True,
                     )
    plt.plot(x_centers, median_pulls,'-',color='red', linewidth=5, label='Median')
    plt.plot(x_centers, mean_pulls,'--',color='red', linewidth=5, label='Mean')

    #--------------------
    # Watermark
    #--------------------
    ax = plt.gca()
    xcoord = xyrange[0][0] + 0.01*(xyrange[0][1] - xyrange[0][0])
    ycoord = xyrange[1][1] - 0.01*(xyrange[1][1] - xyrange[1][0])
    draw_watermark(watermark,ax,
                    xcoord, ycoord,
                    zoom=0.48)
    #------------------
    
    cb = plt.colorbar()
    cb.set_label('Event Rate / arb. unit')
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=1, fancybox=True, framealpha=0.7)

    if not title is None:
        plt.title(title)

    if not xyrange is None:
        plt.xlim(xyrange[0])
        plt.ylim(xyrange[1])

    plt.tight_layout()
    if file is None:
        plt.show()
    else:
        plt.savefig(pathToPlots+'plots/'+str(file)+'.png',dpi=DPI)
        # plt.savefig(pathToPlots+'plots/' +str(file) +'.svg')# QUALITY
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plt.close()


    #---------------------------------------------------
    # Plot Pull Distribution for Intervall
    #---------------------------------------------------
    x_min = 0
    x_max = 10
    y_min = 0
    y_max = 1.0
    use_log_x_axis = True


    for x_start in np.linspace(0,6,7):

        # get a mask of all events within x_start, x_start + 1.0
        mask = (x_start <= binning_variable) &  (binning_variable <= x_start + 1.0)

        if binning_variable[mask].shape[0] > 10:

            plt.figure(figsize=(9, 4.3), dpi=80)
            # plt.figure(figsize=(20, 20), dpi=300)# QUALITY
            matplotlib.rcParams.update({'font.size': 22,'legend.fontsize': 18})
            plt.clf()

            
            if use_log_x_axis:
                x = np.logspace(-2, 1, 300)
                bins = np.logspace(-2, 1, 30)
                x_min = min(x)
                x_max = max(x)
            else:
                x = np.linspace(x_min, x_max, 200)
                bins = x[::3]

            # histogramm
            hist, bins, patches = plt.hist(pull_values[mask], weights=weights[mask], 
                                            bins=bins, normed=True,
                                            facecolor='0.9',edgecolor=u'#1f77b4',
                                            histtype='step',fill=True,lw=2,
                                            )

            # perfect rayleigh distribution:
            plt.plot(x, rayleigh_distribution(x, 1.0),'-', color=u'#2ca02c', label=r'$\sigma_{\mathrm{r}} = 1.00$', linewidth=2)

            # best fit rayleigh distribution
            bin_centers = [ ( bins[i+1] + bins[i]) /2. for i in range(len(bins) - 1)]


            popt, pcov = curve_fit(rayleigh_distribution, bin_centers, hist )
            # print('Best fit sigma: {:1.3f}+-{:1.3f}'.format(popt[0], np.sqrt(pcov[0,0]) ) )
            sigma = popt[0]
            plt.plot(x, rayleigh_distribution(x, sigma),'-', color=u'#ff7f0e', label=r'$\sigma_{\mathrm{r}} = $'+'{:1.2f}'.format(sigma), linewidth=2)
            
            # def gauss(x, mu, sigma):
            #     return 1/( np.sqrt(2*np.pi)*sigma) * np.exp( - (x - mu)**2/(2*sigma*sigma))
            # plt.plot(x, gauss(x,0.,1.),'-',label='Gauss',linewidth=2)

            # # best fit log normal distribution
            # popt, pcov = curve_fit(log_normal, bin_centers, hist )
            # # print('Best fit sigma [log_normal]: {:1.3f}+-{:1.3f}'.format(popt[0], np.sqrt(pcov[0,0]) ) )
            # sigma = popt[0]
            # plt.plot(x, log_normal(x, sigma),'-', color='green', label=r'Log Normal: $\sigma = ${}'.format(sigma), linewidth=4)

            #--------------------
            # Watermark
            #--------------------
            ax = plt.gca()
            xcoord = x_min
            ycoord = y_max - 0.01*(y_max - y_min)
            draw_watermark(watermark,ax,
                            xcoord, ycoord,
                            zoom=0.2,
                            Lfontsize=100,)
            #------------------

            if use_log_x_axis:
                ax.set_xscale("log", nonposx='clip')
            plt.legend(loc=1, fancybox=True, framealpha=0.7)
            plt.grid()
            plt.xlabel(ylabel)
            plt.ylabel('Probability Density')
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)

            plt.title(r'{} $\in$ [{:1.2f},{:1.2f}]'.format(xlabel, 
                                x_start, 
                                x_start + 1.0))
            plt.tight_layout()

            # plt.show()
            if file is None:
                plt.show()
            else:
                plt.savefig(pathToPlots+'plots/'+str(file)+'_{:02d}.png'.format(int(x_start)),dpi=DPI*2)
                # plt.savefig(pathToPlots+'plots/' +str(file) +'.svg')# QUALITY
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)
            plt.close()
    #---------------------------------------------------

def plot_angular_resolution(azimuth_true_list, zenith_true_list,
                            azimuth_pred_list, zenith_pred_list, 
                            proxy_names,
                            binning_variable_list, 
                            bins, 
                            min_bin_count=100,
                            weights_list=None,
                            title=None,
                            xlabel=r'Binning Variable', 
                            ylabel=ur'Median Angular Resolution [°]',
                            file=None,
                            xlim=None,
                            ylim=None,
                            colors=None,
                            linestyles=None,
                            res_method='median',
                            watermark=None,
                            ):
    '''
    Plots resolution in bins along binning_variable_list

    Parameters
    ----------
    azimuth_true_list, zenith_true_list : 
                list/array of floats 
                true label value

    azimuth_pred_list, zenith_pred_list : 
                list of proxy variables (list of list of floats)
                All proxy variables given will be plotted 
                in same resolution plot

    proxy_names: List of str
                Name for each proxy in legend.

    binning_variable_list: list of float.
                variable in which binning
                is performed

    bins: bins keyword for np.histogram
                defines bins

    min_bin_count: int
             Number of minimum events required
             in a bin

    weights_list: list of float
             weight for each event
             If None, each event gets weight 1

    res_method: str
            'median' or 'mean'
            Defines if median or mean in each bin
            is calculated and plotted
    '''

    if weights_list is None:
        weights_list = np.ones_like(azimuth_true_list)
    
    plt.figure(figsize=(20, 15), dpi=80)
    matplotlib.rcParams.update({'font.size': 48})

    #--------------------------
    # set linestyles and colors
    #--------------------------
    if colors is None:
        color_cycler = cycle([ u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', 
                                u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf' ])
    else:
        color_cycler = cycle(colors)

    if linestyles is None:
        linestyle_cycler = cycle([":","--","-.","-"])
    else:
        linestyle_cycler = cycle(linestyles)
    #--------------------------

    if not xlim is None:
        plt.xlim(xlim)
    if not ylim is None:
        plt.ylim(ylim)

    # plot resolution curves for all proxy variables
    for proxy_name, azimuth_true,zenith_true, azimuth_pred, \
        zenith_pred,weights,binning_variable in zip(proxy_names,
                                                     azimuth_true_list,
                                                     zenith_true_list,
                                                     azimuth_pred_list, 
                                                     zenith_pred_list,
                                                     weights_list,
                                                     binning_variable_list):

        # calculate angular error
        angular_error = get_angle_deviation(azimuth_true, zenith_true, 
                                            azimuth_pred, zenith_pred)
        angular_error *= 180/np.pi

        #---
        # create bins
        #---
        _, bin_edges = np.histogram(binning_variable, bins=bins)
        bins_indices = np.digitize(binning_variable, bins=bin_edges) - 1 # -1 because digitize starts at one
        bin_mids = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1])/2.
        bin_widths = bin_edges[1:] - bin_edges[:-1] 
        num_bins = len(bin_edges) - 1
        #---

        #-----
        # get median/mean angular error in each bin
        #-----
        bin_resolution_list = []
        used_bin_mids = []
        for bin_number in range(num_bins):

            # get a mask of all events in this label bin
            mask_events_in_bin = bins_indices == bin_number

            if len(angular_error[mask_events_in_bin]) > min_bin_count:
                if res_method == 'median':
                    bin_res = weighted_median(angular_error[mask_events_in_bin], 
                                            weights=weights[mask_events_in_bin])
                elif res_method == 'mean':
                    bin_res = np.average(angular_error[mask_events_in_bin], 
                                            weights=weights[mask_events_in_bin])
                else:
                    raise ValueError('res_method must be median or mean')
                bin_resolution_list.append(bin_res)
                used_bin_mids.append(bin_mids[bin_number])
        #-----
        plt.plot(used_bin_mids, bin_resolution_list,
                        linestyle=next(linestyle_cycler),
                        color=next(color_cycler),
                        linewidth=5,label=proxy_name)
        #-----

    plt.legend(loc='best', fancybox=True, framealpha=0.7)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    #--------------------
    # Watermark
    #--------------------
    ax = plt.gca()
    draw_watermark(watermark,ax)
    #------------------

    if not title is None:
        plt.title(title,color='black',y=1.01,x=0.48,fontsize=64)

    if file is None:
        plt.show()
    else:
        plt.savefig(file+'.svg', dpi=DPI)
        plt.savefig(file, dpi=DPI)
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plt.close()

def plotSingleAngularResolution(azimuth_pred, zenith_pred, azimuth_true, 
                            zenith_true, binning_variable,delta_x = None,
                            title = 'Angular Resolution',
                            xlabel='Binning Variable',
                            ylabel='Angular Resolution',
                            pathToPlots='',
                            file='AngularResolution',
                            convertToDegree = True,
                            xlim = None,
                            ylim = None,
                            yscale = 'linear',
                            weights=None,
                            verbose=False,
                            ):
    '''
    Plots angular resolution in bins along binning_variable

    Parameters
    ----------
    pred : list/array of floats 
            predicted value


    true : list/array of floats 
            true value

    binning_variable : list/array of floats 
            value of the variable along which the 
            binning should be performed
    '''

    if weights is None:
        weights  = np.ones_like(azimuth_pred)

    # calculate angular deviation 
    angle_deviation = get_angle_deviation(azimuth_true, zenith_true, azimuth_pred, zenith_pred)

    # convert to degree:
    if convertToDegree:
        angle_deviation = angle_deviation / np.pi * 180.0

    binning_variable = np.asarray(list(binning_variable))

    # remove nans:
    if verbose:
        print 'length before removing nans:',binning_variable.shape[0]
    invalidEntries = np.isnan(angle_deviation)
    angle_deviation = angle_deviation[~invalidEntries]
    binning_variable = binning_variable[~invalidEntries]
    weights = weights[~invalidEntries]
    if verbose:
        print 'length after removing nans:',binning_variable.shape[0]

    if len(binning_variable) == 0:
        print('\033[93mWarning: no values left after removing nans. Not plotting {} or saving {}\033[0m'.format(title, file))
        return

    min_x = min(binning_variable)
    max_x = max(binning_variable)

    x_bin_centers = np.linspace(min_x, max_x, 100)#---------------------------- DEBUG 100
    if delta_x == None:
        delta_x = (max_x - min_x) / binning_variable.shape[0] * 50
        if verbose:
            print 'Using a delta_x of', delta_x

    median_angular_resolution = []
    mean_angular_resolution = []
    quantile_68 = []
    quantile_80 = []
    quantile_90 = []
    median_x_points = []
    for x_center in x_bin_centers:
        # get a mask of all events within +- delta_x of x_center
        mask = (x_center - delta_x < binning_variable) &  (binning_variable < x_center + delta_x)
        # print len(angle_deviation[mask]), x_center, np.median(binning_variable[mask]), np.mean(binning_variable[mask])
        if binning_variable[mask].shape[0] > 0:
            median_x_points.append(np.median(binning_variable[mask]))
            angle_deviation_masked = angle_deviation[mask]
            weights_masked = weights[mask]
            sorted_indices = angle_deviation_masked.argsort()
            angle_deviation_masked = angle_deviation_masked[sorted_indices]
            weights_masked = weights_masked[sorted_indices]

            median_angular_resolution.append(weighted_median(angle_deviation_masked, weights=weights_masked))
            mean_angular_resolution.append(np.average(angle_deviation_masked, weights=weights_masked))
            quantile_68.append(weighted_quantile(angle_deviation_masked, weights=weights_masked, quantile=0.68))
            quantile_80.append(weighted_quantile(angle_deviation_masked, weights=weights_masked, quantile=0.80))
            quantile_90.append(weighted_quantile(angle_deviation_masked, weights=weights_masked, quantile=0.90))

        # plt.hist(angle_deviation[mask],bins=1000) #---------------------------- DEBUG
        # plt.show()#---------------------------- DEBUG
        # plt.clf()#---------------------------- DEBUG
        # # PlotScatter(binning_variable[mask],angle_deviation[mask])#---------------------------- DEBUG

    # plot resolution
    plt.figure(figsize=(20, 20), dpi=80)
    matplotlib.rcParams.update({'font.size': 48})
    plt.clf()

    plt.plot(median_x_points, median_angular_resolution, label='Median Resolution', linewidth=7)
    plt.plot(median_x_points, mean_angular_resolution, label='Mean Resolution', linewidth=7)
    plt.plot(median_x_points, quantile_68, label='68% Quantile Resolution', linewidth=7)
    # plt.plot(median_x_points, quantile_80, label='80% Quantile Resolution', linewidth=7)
    # plt.plot(median_x_points, quantile_90, label='90% Quantile Resolution', linewidth=7)
    # plt.plot(x_bin_centers, median_angular_resolution, label='Median Resolution')
    plt.grid()
    plt.legend(loc='best', fancybox=True, framealpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale(yscale)
    if xlim != None:
        plt.xlim(xlim)
    if ylim != None:
        plt.ylim(ylim)


    # plt.show()
    plt.savefig(pathToPlots+'plots/'+str(file)+'.png', dpi=DPI)
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plt.close()


def plot_proxy_resolution(label, proxy_list, 
                            proxy_names,
                            proxy_bins, 
                            label_bins, 
                            weights=None,
                            label_and_weights_as_list=False,
                            title=None,
                            xlabel=r'true label: $y_\mathrm{true}$', 
                            ylabel=r'Resolution',# $\sigma_{\log_{10} E_\mu}$
                            file=None,
                            xlim=None,
                            ylim=None,
                            which_resolution='res',
                            colors=None,
                            linestyles=None,
                            figsize=(20,15),
                            ):
    '''
    Plots resolution in bins along binning_variable

    Parameters
    ----------
    label : list/array of floats 
            true label value

    proxy_list : list of proxy variables (list of list of floats)
                All proxy variables given will be plotted 
                in same resolution plot

    proxy_names: List of str
                Name for each proxy in legend.

    proxy_bins: bins keyword for np.histogram
                defines proxy bins

    label_bins: bins keyword for np.histogram
                defines label bins

    weights: list of float
             weight for each event
             If None, each event gets weight 1
    '''

    if weights is None:
        weights = np.ones_like(label)

    if label_and_weights_as_list:
        label_list = label
        weights_list = weights
    else:
        label_list = [label] * len(proxy_list)
        weights_list = [weights] * len(proxy_list)
    
    plt.figure(figsize=figsize, dpi=80)
    matplotlib.rcParams.update({'font.size': 42})

    #--------------------------
    # set linestyles and colors
    #--------------------------
    if colors is None:
        color_cycler = cycle([ u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', 
                                u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf' ])
    else:
        color_cycler = cycle(colors)

    if linestyles is None:
        linestyle_cycler = cycle([":","--","-.","-"])
    else:
        linestyle_cycler = cycle(linestyles)
    #--------------------------

    if not xlim is None:
        plt.xlim(xlim)
    if not ylim is None:
        plt.ylim(ylim)

    # plot resolution curves for all proxy variables
    for label,weights, proxy,proxy_name in zip(label_list,
                                        weights_list,
                                        proxy_list,
                                        proxy_names):
        res = get_proxy_resolution(proxy, label, 
                                        proxy_bins = proxy_bins,
                                        label_bins = label_bins, 
                                        weights=weights)

        if which_resolution == 'res':
            plt.plot(res[1], res[2],
                    linestyle=next(linestyle_cycler), 
                    color=next(color_cycler),
                    linewidth=5,label=proxy_name)

        elif which_resolution == 'std':
            plt.plot(res[1], res[3],
                    linestyle=next(linestyle_cycler), 
                    color=next(color_cycler),
                    linewidth=5,label=proxy_name)

        elif which_resolution == 'rmse':
            plt.plot(res[1], res[4],
                    linestyle=next(linestyle_cycler), 
                    color=next(color_cycler),
                    linewidth=5,label=proxy_name)
        else:
            raise ValueError('which_resolution must be res,std, or rmse')

        # plt.plot(res[1], res[3],linestyle='dashed',label=proxy_name+' - std. dev.')
        # plt.plot(res[1], res[4],linestyle='dotted',label=proxy_name+' - RMSE')

    plt.legend(loc='best', fancybox=True, framealpha=0.7)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    if not title is None:
        plt.title(title,color='black',y=1.01,x=0.48,fontsize=64)

    if file is None:
        plt.show()
    else:
        plt.savefig(file+'.svg', dpi=DPI)
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plt.close()

def get_proxy_resolution(proxy, label, proxy_bins, label_bins, weights=None, verbose=False):
    '''
    Calculates resolution for an observable that is being used
    as a proxy for the desired label.

    Parameters:
    ---------

    proxy: list of float.
            observable that is meant to be used as a proxy
            for the label

    label: list of float.
            label

    proxy_bins: bins keyword for np.histogram
                defines proxy bins

    label_bins: bins keyword for np.histogram
                defines label bins

    weights: list of float
             weight for each event
             If None, each event gets weight 1

    Returns:
    --------

    resolution: tuple
            overall resolution, resolution bins, resolution, std_dev, rmse
            A list containing the resolution for each label_bin
    '''

    if weights is None:
        weights = np.ones(len(label))

    #---------------------
    # get proxy and label bin_edges
    #---------------------
    _, proxy_bin_edges = np.histogram(proxy, bins=proxy_bins)
    _, label_bin_edges = np.histogram(label, bins=label_bins)

    num_proxy_bins = len(proxy_bin_edges) - 1
    num_label_bins = len(label_bin_edges) - 1

    # get proxy and label bin indices for all events
    proxy_bins_indices = np.digitize(proxy, bins=proxy_bin_edges) - 1 # -1 because digitize starts at one
    label_bins_indices = np.digitize(label, bins=label_bin_edges) - 1 # -1 because digitize starts at one

    label_bin_mids = label_bin_edges[:-1] + (label_bin_edges[1:] - label_bin_edges[:-1])/2.

    proxy_bin_widths = proxy_bin_edges[1:] - proxy_bin_edges[:-1] 
    label_bin_widths = label_bin_edges[1:] - label_bin_edges[:-1] 

    #---------------------
    # get distribution in proxy observable for a given label bin P(O|E=E'+-dE')
    ## for all label bins
    #---------------------
    P_of_O_given_E_bins = []
    for label_bin in range(num_label_bins):

        # get a mask of all events in this label bin
        mask_events_in_label_bin = label_bins_indices == label_bin

        # get distribution in proxy observable for a given label bin P(O|E=E'+-dE')
        if len(proxy[mask_events_in_label_bin]) > 0:
            P_of_O_given_E,_ = np.histogram(proxy[mask_events_in_label_bin],
                                bins=proxy_bin_edges,
                                weights=weights[mask_events_in_label_bin],
                                density=True)
        else:
            if verbose:
                print('No events in label bin number {}'.format(label_bin))
            P_of_O_given_E = np.zeros(num_proxy_bins)

        P_of_O_given_E_bins.append(P_of_O_given_E)

    P_of_O_given_E_bins = np.asarray(P_of_O_given_E_bins)


    #---------------------
    # get distribution in label for a given proxy value bin P(E|O=O'+-dO')
    ## for all proxy bins
    #---------------------
    P_of_E_given_O_bins = []
    for proxy_bin in range(num_proxy_bins):

        # get a mask of all events in this proxy bin
        mask_events_in_proxy_bin = proxy_bins_indices == proxy_bin

        # get distribution in label observable for a given proxy value bin P(E|O=O'+-dO')
        if len(label[mask_events_in_proxy_bin]) > 0:
            P_of_E_given_O,_ = np.histogram(label[mask_events_in_proxy_bin],
                                bins=label_bin_edges,
                                weights=weights[mask_events_in_proxy_bin],
                                density=True)
        else:
            if verbose:
                print('No events in proxy bin number {}'.format(proxy_bin))
            P_of_E_given_O = np.zeros(num_label_bins)

        P_of_E_given_O_bins.append(P_of_E_given_O)

    P_of_E_given_O_bins = np.asarray(P_of_E_given_O_bins)


    #---------------------
    # Calculate resolution for each label bin
    ## Todo: possible speed up through matrix operations?
    #---------------------
    resolution = []
    resolution_label_bins = []
    overall_resolution = 0
    overall_resolution_norm = 0

    # go through all label bins
    for label_bin in range(num_label_bins):

        # perform discretized integral over all proxy bins
        ## sum of all distributions P(E|O=O')*P(O=O'|E=E') 
        ## for all given proxy values O' in O
        integral_distribution = np.zeros(num_label_bins)
        for proxy_bin in range(num_proxy_bins):

            # calculate P(E|O=O')*P(O=O'|E=E')
            integral_distribution += P_of_E_given_O_bins[proxy_bin] * proxy_bin_widths[proxy_bin] * \
                                    P_of_O_given_E_bins[label_bin,proxy_bin]

        # only add resolution and label bin, if any events were in it
        inegral_sum = np.sum(integral_distribution)
        if inegral_sum > 0.0:
            res = weighted_std(label_bin_mids,weights=integral_distribution)
            overall_resolution += res*inegral_sum
            overall_resolution_norm += inegral_sum
            resolution.append(res)
            resolution_label_bins.append(label_bin_mids[label_bin])
        else:
            if verbose:
                print('Skipping empty label bin {}'.format(label_bin))

        # # print integral_distribution
        # # print np.sum(integral_distribution)
        # if np.sum(integral_distribution) > 0.0:
        #     print('At Energy {}'.format(label_bin_mids[label_bin]))
        #     print('\tResolution: {}'.format(weighted_std(label_bin_mids,weights=integral_distribution)))
        #     print np.average(label_bin_mids, weights=integral_distribution)
        #     print weighted_median(label_bin_mids, weights=integral_distribution)
        #     plt.bar(label_bin_mids,integral_distribution, width=label_bin_widths)
        #     plt.show()
        # #     # break

    overall_resolution /= overall_resolution_norm

    std_dev_in_proxy = []
    std_dev_residuals = []
    rmse_residuals = []
    for label_bin in range(num_label_bins):
        mask_events_in_label_bin = label_bins_indices == label_bin
        if label_bin_mids[label_bin] in resolution_label_bins:
            std_dev_in_proxy.append(weighted_std(proxy[mask_events_in_label_bin],weights=weights[mask_events_in_label_bin]))
            residuals = proxy[mask_events_in_label_bin] - label[mask_events_in_label_bin]
            std_dev_residuals.append(weighted_std(residuals,weights=weights[mask_events_in_label_bin]))
            mse = np.sum( (residuals*weights[mask_events_in_label_bin])**2) / np.sum(weights[mask_events_in_label_bin]**2)
            rmse_residuals.append(np.sqrt(mse))
    
    # plt.figure(figsize=(20, 20), dpi=80)
    # matplotlib.rcParams.update({'font.size': 48})
    # plt.plot(resolution_label_bins, std_dev_in_proxy, label=r'$\sigma$ in proxy', linewidth=5)
    # plt.plot(resolution_label_bins, std_dev_residuals, label=r'$\sigma$ in residuals', linewidth=5)
    # plt.plot(resolution_label_bins, rmse_residuals, label=r'RMSE', linewidth=5)
    # plt.plot(resolution_label_bins, resolution, label='Resolution (RecoPaper)', linewidth=5)
    # plt.legend(loc='best')
    # plt.title('Truncated Energy - E^-1',color='black',y=1.08,x=0.50,fontsize=80)
    # # plt.tight_layout()
    # plt.grid()
    # plt.show()
    # # plt.savefig(pathToPlots+'plots/'+file+'_alpha.png') 
    # plt.close()
    # matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    return overall_resolution, np.asarray(resolution_label_bins), np.asarray(resolution), np.asarray(std_dev_residuals), np.asarray(rmse_residuals)



def evaluateUncertainty(true, pred, uncertainty,label,file,pathToPlots='',
                        limits=None,plot=False,verbose=False,watermark=None,
                        weights=None, title=None,
                        resolution_unit_str='',
                        plot_scaled_line=None,
                        scaled_line_label=None,
                        dpi = 72,
                        ):
    '''
    Makes plots to evaluate uncertainty estimations

    Parameters
    ----------
    true : list/array of floats 
            true value

    pred : list/array of floats 
            predicted value

    uncertainty : list/array of floats 
            estimated uncertainty

    label : string
            name of label

    title : string
            human readable name of label, e.g. for plot title 

    '''
    # uncertainty *= np.sqrt(2) # 1.482578

    if weights is None:
        weights  = np.ones_like(true)

    residuals = true - pred
    if limits == None:
        limits = [0,max(uncertainty)*1.05]

    steps = np.linspace(limits[0],limits[1],100)

    if 'Azimuth' in label:
        # correct for 359° - 2° = 3° and not 357°
        residuals_2pi = np.abs(residuals-2*np.pi) 
        mask = residuals_2pi < residuals
        residuals[mask] = residuals_2pi[mask]
    
    estimated_err = []
    stds = []
    median_err = []
    mean_err = []
    maxx = 0
    for i in range(len(steps)-1):
        # find recos with ucnertainty within (step[i],step[i+1])
        mask = (uncertainty > steps[i]) & (uncertainty < steps[i+1])
        if len(residuals[mask]) > 40: # In MasterThesis this was 50
            maxx = steps[i+1]
            estimated_err.append((steps[i]+steps[i+1])/2.)
            # stds.append(np.std(residuals[mask]))
            # median_err.append(np.median(np.abs(residuals[mask])))
            # mean_err.append(np.mean(np.abs(residuals[mask])))
            stds.append( weighted_std(residuals[mask], weights[mask]))
            median_err.append(weighted_median(np.abs(residuals[mask]), weights[mask]))
            mean_err.append(np.average(np.abs(residuals[mask]), weights=weights[mask]))

    matplotlib.rcParams.update({'font.size': 22,'legend.fontsize': 18})
    fig, ax1 = plt.subplots(figsize=(9, 6), dpi=80)
    xcoord = 0.01*maxx
    ycoord = 0.99*max(stds) 
    draw_watermark(watermark,ax1,
                    xcoord, ycoord,
                    zoom=0.3)
    ax1.plot((0,maxx),(0,maxx),'-',color=u'#d62728',linewidth=1)
    if not plot_scaled_line is None:
        if scaled_line_label is None:
            scaled_line_label = r'$f(x) = $'+'{:1.2f}'.format(plot_scaled_line)+r'$\cdot x$'
        ax1.plot((0,maxx),(0,maxx * plot_scaled_line),'--',color='0.3',linewidth=1.5, label=scaled_line_label)

    ax1.plot(estimated_err,stds,'+',label='Std of Residuals', markersize=10)
    ax1.plot(estimated_err,mean_err,'x',label='Mean of abs Residuals', markersize=10)
    ax1.plot(estimated_err,median_err,'2',label='Median of abs Residuals', markersize=10)
    ax1.set_xlabel(r'Estimated Uncertainty $\sigma_\mathrm{pred}$' + resolution_unit_str)
    ax1.set_ylabel('Resolution' + resolution_unit_str)
    ax1.set_xlim(0,maxx)
    # plt.title('Resolution of '+title)
    if not title is None:
        plt.title(title)
    legend = plt.legend(loc=4, fancybox=True, framealpha=0.7)
    plt.tight_layout()
    plt.grid()
    plt.savefig(pathToPlots+'plots/'+file+'_resolution.png',dpi=dpi)
    plt.close()
    
    
    plt.figure()
    plt.hexbin(uncertainty,np.abs(residuals),bins='log',C=weights, reduce_C_function=np.sum)
    if not title is None:
        plt.title(title)
    plt.xlabel('Estimated Uncertainty' + resolution_unit_str)
    plt.ylabel('Absolute Residuals' + resolution_unit_str)
    cb = plt.colorbar()
    cb.set_label(ur'$\log10$(counts)')
    plt.grid()
    plt.tight_layout()
    plt.savefig(pathToPlots+'plots/'+file+'_correlation.png',dpi=dpi)
    plt.close()
    
    plt.figure()
    plt.xlim(0,maxx)
    plt.hexbin(uncertainty,np.abs(residuals),bins='log', C=weights, reduce_C_function=np.sum)
    if not title is None:
        plt.title(title)
    plt.xlabel('Estimated Uncertainty' + resolution_unit_str)
    plt.ylabel('Absolute Residuals' + resolution_unit_str)
    cb = plt.colorbar()
    cb.set_label(ur'$\log10$(counts)')
    plt.grid()
    plt.tight_layout()
    plt.savefig(pathToPlots+'plots/'+file+'_correlation_zoomed.png',dpi=dpi)
    plt.close()

    # Pull distribution
    def gauss(x, mu, sigma):
        return 1/( np.sqrt(2*np.pi)*sigma) * np.exp( - (x - mu)**2/(2*sigma*sigma))


    values = residuals/(uncertainty)
    mu = np.mean(values)
    sigma = np.std(values)

    mask_less10 = np.abs(values) < 10
    mu_trunc = np.mean(values[mask_less10])
    sigma_trunc = np.std(values[mask_less10])
    
    xlim = mu - 5*sigma,mu + 5*sigma
    xlim = np.clip(xlim, -10,10)
    x = np.linspace(xlim[0], xlim[1], 100)

    y = gauss(x, mu, sigma)
    y_trunc = gauss(x, mu_trunc, sigma_trunc)

    

    fig, ax1 = plt.subplots(figsize=(9, 6), dpi=dpi)
    n,bins,patches = plt.hist(values,bins=x,log=False, 
                                normed=True, weights=weights,
                                facecolor='0.9',edgecolor=u'#1f77b4',
                                histtype='step',fill=True,lw=2,
                                label='Pull Distribution'
                                )
    # def gauss_centered(x, sigma):
    #     return 1/( np.sqrt(2*np.pi)*sigma) * np.exp( - (x - 0.0)**2/(2*sigma*sigma))
    # popt, pcov = curve_fit(gauss_centered, np.diff(bins), n, p0 = (1.3) )
    # ax1.plot(x,  gauss_centered(x,sigma=popt[0]),'-',lw=2,label='$\mu$ : {:1.3f}\n$\sigma$ : {:1.3f}'.format(0.0, popt[0]))


    xcoord = xlim[0] + 0.01*(xlim[1] - xlim[0])
    ycoord = 0.99*max(n)
    draw_watermark(watermark,ax1,
                    xcoord, ycoord,
                    zoom=0.3)
    ax1.plot(x, y,'-',lw=2,label='$\mu$ : {:1.3f}\n$\sigma$ : {:1.3f}'.format(mu,sigma))
    ax1.plot(x, y_trunc,'-',lw=2,label='$\mu$ : {:1.3f}\n$\sigma$ : {:1.3f}'.format(mu_trunc,sigma_trunc))
    ax1.set_xlabel(r'$(y_\mathrm{true} - y_\mathrm{pred})$ / $\sigma_\mathrm{pred}$')
    ax1.set_ylabel(r'Relative Frequency')
    # x_coord = 3*sigma
    # plt.text(x_coord, 0.1, r'$\mu$: {:1.3f}'.format(mu), ha='left', va='center',fontsize=18)
    # plt.text(x_coord, 0.05, r'$\sigma$: {:1.3f}'.format(sigma), ha='left', va='center',fontsize=18)
    if not title is None:
        plt.title(title)
    ax1.set_xlim(xlim)
    plt.legend(loc='best', fancybox=True, framealpha=0.7)
    plt.grid()
    plt.tight_layout()
    plt.savefig(pathToPlots+'plots/'+file+'_factor.png',dpi=dpi)
    plt.close()



    #----------------------------------------------------
    # Residual Plot
    #----------------------------------------------------
    # if plot_residual_plot:

    # create figure
    plt.figure(figsize=(20, 20), dpi=dpi)
    # plt.figure(figsize=(20, 20), dpi=300) # QUALITY
    matplotlib.rcParams.update({'font.size': 48})
    ax = plt.gca()

    # sort along uncertainty prediction
    sorted_indices = np.argsort(uncertainty)

    # define x range of plot and bins
    std = np.std(residuals)
    # std = weighted_std(residuals,weights=weights)
    range_factor = 1

    xLabel = 'Residuals'

    # azimuth specific modifications
    if 'Azimuth' in label:
        if watermark=='L':
            range_factor = 0.07
        xLabel = '( True Azimuth - Reconstructed Azimuth ) / rad'
        xLabel = 'True - Reconstructed Azimuth / rad'
        xLabel = '( True Azimuth - Rec. Azimuth ) / rad'
        if 'Primary' in label:
            title = 'Neutrino Azimuth'

    # zenith specific modifications
    if 'Zenith' in label:
        if watermark=='L':
            range_factor = 0.1
        xLabel = '( True Zenith - Reconstructed Zenith ) / rad'
        xLabel = 'True - Reconstructed Zenith / rad'
        xLabel = '( True Zenith - Rec. Zenith ) / rad'
        if 'Primary' in label:
            title = 'Neutrino Zenith'


    # get bins
    bin_heights, bins = np.histogram(residuals, bins=70, range=(-range_factor*std,range_factor*std), weights=weights) #get the bin edges
    # bin_heights, bins = np.histogram(residuals, bins=70, weights=weights) #get the bin edges


    # define percentages to plot additionally to all events
    percentages = [0.05, 0.7]

    # get values, labels and colors
    percentages += [1.0]
    values = [  residuals[sorted_indices[:int(len(pred) * p)]] for p in percentages]
    weight_values = [  weights[sorted_indices[:int(len(pred) * p)]] for p in percentages]
    labels = [ 'Top {:d}%: std. {:1.3f}'.format( int(percentages[i]*100), np.std(values[i]) ) 
                                for i in range(len(percentages) -1)]
    labels += ['All Events: std. {:1.3f}'.format(np.std(values[-1])) ]
    colors = [ cm.viridis(i) for i in np.linspace(0.95,0.0,len(labels)) ]
    colors = [ cm.viridis(i) for i in [0.90,0.4,0.0] ]

    for v,label, color,w in zip(values,labels, colors, weight_values):
        # bin_height,bin_boundary = np.histogram(v,bins=bins)
        bin_height,bin_boundary = np.histogram(v,bins=bins,weights=w)
        hist_weights = np.ones(shape=v.shape) / max(bin_height)
        plt.hist(v, bins=bins, weights=hist_weights, label=label, color=color, histtype = 'step', linewidth=8, log=False)

    # #-------------------
    # # Box Plots
    # #-------------------
    # for value in values:
    #     value = np.sort(value)
    #     print '10%: {:3.3f}  90% {:3.3f}'.format( value[int(len(value)*0.1)], value[int(len(value)*0.9)] )
    # raw_input()
    # import seaborn as sns
    # ax = sns.boxplot(data=values, orient="h", palette=colors, whis=10)
    # # ax.set_yticks(range(len(values)))
    # # ax.set_yticklabels(labels)
    # box_labels = [ 'Top {:d}%'.format( int(percentages[i]*100) )for i in range(len(percentages) -1)]
    # box_labels += ['All Events'] 
    # plt.yticks(plt.yticks()[0], box_labels)
    # #-------------------

    # get legend correct
    handles,labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1],labels[::-1], fontsize=32, fancybox=True, framealpha=0.7)

    plt.xlabel(xLabel)
    plt.ylabel('Relative Bin Counts')
    plt.xlim(-range_factor*std,range_factor*std)

    #--------------------
    # Watermark
    #--------------------
    if watermark != None:
        xcoord = -range_factor*std + 0.02*range_factor*std
        ycoord = max(bin_heights)*weights[0] #- 0.01*(max(bin_heights) -min(bin_heights))

        if watermark == 'DNN':
            filename = os.path.join(misc_dir, 'figures/DNN_Icon2.png')
            arr_img = plt.imread(filename, format='png')

            zoom = 0.5
            imagebox = OffsetImage(arr_img, zoom=zoom, alpha=.2)
            imagebox.image.axes = ax

            ab = AnnotationBbox(imagebox, [xcoord,ycoord],
                                xybox=(0,0),#(300*zoom, 300*zoom),
                                xycoords='data',
                                boxcoords="offset points",
                                box_alignment=(0,1),
                                pad=0.0,
                                frameon=False,
                                )

            ax.add_artist(ab)
        elif watermark == 'L':
             plt.text(xcoord, ycoord, r'$\mathcal{L}$', ha='left', va='top',fontsize=200, alpha=0.2)
        else:
            raise ValueError('Unknown Watermark: {}'.format(watermark))
    #-------------------

    plt.title(title,color='black',y=1.08,x=0.45,fontsize=80)
    # plt.tight_layout()
    plt.grid()
    plt.savefig(pathToPlots+'plots/'+file+'_alpha.png',dpi=dpi) 
    # plt.savefig(pathToPlots+'plots/'+file+'_alpha.svg')# QUALITY
    plt.close()

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    #----------------------------------------------------


def plot_angular_distribution(
                            zenith_list,
                            weights_list,
                            label_list,
                            xlabel = r'Zenith Angle $\theta$ /$\,^{\circ}$',
                            ylabel = r'Event Rate  / arb. unit',
                            ylim = None,
                            linestyles = None,
                            colors = None,
                            zenith_unit_is_radians = True,
                            y_pos_arrows = 1e-5,
                            file = None,
                            ):
    '''
    Plots histogram of event rates binned
    in zenith or azimuth

    Parameters:
    -----------

    zenith_list:
                list of zenith values to plot
                [zenith_values_distribution1, zenith_values_distribution2,...]

    weights_list:
                list of weights for the events
                [weights_distribution1, weights_distribution2,...]

    label_list: list of str
                labels for each distribution
                ['Distribution1','Distribution2',...]

    '''
    matplotlib.rcParams.update({'font.size': 14})

    # set linestyles and colors
    if colors is None:
        color_cycler = cycle([ u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', 
                                u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf' ])
    else:
        color_cycler = cycle(colors)

    if linestyles is None:
        linestyle_cycler = cycle(['-'])
    else:
        linestyle_cycler = cycle(linestyles)

    binning = np.linspace(0, 180, 26)

    fig, ax1 = plt.subplots(figsize=(9, 6))

    ax1.axvline(90.,
                color='C0',  # 'C0',
                ls='-',
                lw=.7,
                zorder=0)
    ax1.text(91.,
             1e-6,
             'Horizon',
             color='C0',  # 'C0',
             rotation='vertical')

    ax1.fill_between([0, 86],
                     [0, 0], [1, 1],
                     color='0.9')

    for zenith, weights, label in zip(zenith_list,
                                    weights_list,
                                    label_list,
                                    ):

        if zenith_unit_is_radians:
            zenith = np.rad2deg(zenith)

        ax1.hist(zenith,
                 bins=binning,
                 weights=weights,
                 color=next(color_cycler),
                 ls=next(linestyle_cycler),
                 histtype='step',
                 lw=2.,
                 label=label,
                 zorder=2)

    ax1.text(43., y_pos_arrows, r'downgoing',
             size=10,
             ha='center',
             va='center',
             color='0.3')

    ax1.text(133., y_pos_arrows, r'upgoing',
             size=10,
             ha='center',
             va='center',
             color='0.3')

    x = np.array([25, 0])
    y = np.array([y_pos_arrows, y_pos_arrows])
    ax1.quiver(x[:-1], y[:-1],
               x[1:] - x[:-1], y[1:] - y[:-1],
               scale_units='xy',
               angles='xy',
               scale=1,
               color='0.3')

    x = np.array([61, 86])
    y = np.array([y_pos_arrows, y_pos_arrows])
    ax1.quiver(x[:-1], y[:-1],
               x[1:] - x[:-1], y[1:] - y[:-1],
               scale_units='xy',
               angles='xy',
               scale=1,
               color='0.3')

    x = np.array([111, 86])
    y = np.array([y_pos_arrows, y_pos_arrows])
    ax1.quiver(x[:-1], y[:-1],
               x[1:] - x[:-1], y[1:] - y[:-1],
               scale_units='xy',
               angles='xy',
               scale=1,
               color='0.3')

    x = np.array([155, 180])
    y = np.array([y_pos_arrows, y_pos_arrows])
    ax1.quiver(x[:-1], y[:-1],
               x[1:] - x[:-1], y[1:] - y[:-1],
               scale_units='xy',
               angles='xy',
               scale=1,
               color='0.3')

    ax1.set_xlim(0, 180)

    if ylim is not None:
        ax1.set_ylim(ylim)
    ax1.set_yscale("log", nonposy='clip')
    ax1.legend(loc=3, fancybox=True, framealpha=0.7)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    fig.tight_layout()

    if file is None:
        plt.show()
    else:
        fig.savefig(file, dpi=300)

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)