import os, glob, datetime

import pandas as pd
import numpy as np
import pickle as pkl
import scipy.stats as stats

from scipy.stats import multivariate_normal, norm
from functional_utils import _fDepth, _fQuantile
from sklearn.neighbors import KernelDensity
from statsmodels.distributions.empirical_distribution import ECDF

# Calculate weighted (w_) distance between X_ and x_
def _euclidian_dist(X_, x_, w_=[]):
    if len(w_) == 0:
        w_ = np.ones(x_.shape)
    w_ = w_ / w_.sum()
    d_ = np.zeros((X_.shape[0],))
    for i in range(X_.shape[0]):
        d_[i] = w_.T @ (X_[i, :] - x_) ** 2
    return d_

# Radial Basis function kernel based on distance (d_)
def _kernel(d_, length_scale):
    w_ = np.exp(-d_ / length_scale)
    return w_  # /w_.sum()

# Define exponential growth function
def _exponential_growth(t, dacay_rate, innit = 0):
    tau_ = np.linspace(t - 1, 0, t)
    return np.exp(-dacay_rate*tau_)

# Define exponential dacay function
def _exponential_decay(S, dacay_rate):
    s_ = np.linspace(0, S - 1, S)
    return np.exp(-dacay_rate*s_)

def _haversine_dist(x_1_, x_2_):
    """
    Calculate the distance between two points on Earth using the Haversine formula.

    Args:
        x_1_ (float): Longitude and latitude of the first point in degrees.
        x_2_ (float): Longitude and latitude of the second point in degrees.

    Returns:
        float: Distance between the two points in kilometers.
    """
    R = 6371  # Radius of Earth in kilometers
    
    dlat_ = np.deg2rad(x_2_[:, 1]) - np.deg2rad(x_1_[1])
    dlon_ = np.deg2rad(x_2_[:, 0]) - np.deg2rad(x_1_[0])
    
    theta = np.sin(dlat_/2)**2 + np.cos(np.deg2rad(x_1_[1]))*np.cos(np.deg2rad(x_2_[:, 1]))*np.sin(dlon_/2)**2
    
    return 2.*R*np.arcsin(np.sqrt(theta))

def _logistic(x_, k = 1.):
    return 1. / (1. + np.exp( - k*x_))

# Define a function to calculate quantiles
def _KDE_quantile(_KDE, q_, x_min     = 0., 
                            x_max     = 1., 
                            n_samples = 1000):
    
    """
    Calculates the quantile for a given probability using KDE.

    Parameters:
    _KDE: Kernel density estimate object (e.g., from scipy.stats.gaussian_kde).
    q:    Probability value (between 0 and 1) for which to calculate the quantile.

    Returns:
    The quantile value.
    """

    # Calculate CDF
    x_ = np.linspace(x_min, x_max, n_samples)
    #z_ = np.exp(_KDE.score_samples(x_[:, np.newaxis]))
    w_ = np.cumsum(np.exp(_KDE.score_samples(x_[:, np.newaxis])))
    # Normalize CDF
    w_ /= w_[-1] 
    
    return np.interp(np.array(q_), w_, x_), np.interp(1. - np.array(q_), w_, x_)

# Silverman's Rule
def _silverman_rule(x_):
    IQR = np.percentile(x_, 75) - np.percentile(x_, 25)
    return 0.9 * min(np.std(x_), IQR / 1.34) * x_.shape[0] ** (-1 / 5)

# Periodic distance to rank samples by day of the year
def _periodic_dist(x_1_, x_2_, day_to_degree=360/365, degree_to_rad=np.pi / 180):
    return np.sin(0.5 * (day_to_degree * (x_2_ - x_1_) * degree_to_rad) ) ** 2

# Filtering scenarios when they are above the upper threshold or below the lower threshold
def _scenario_filtering(W_, d_h_, d_p_, xi, gamma, kappa_min, kappa_max):

    status = 0
    sigma  = 0

    # Similarity ranking
    idx_rank_ = np.argmin(W_, axis=0)

    # Similarity filter
    w_ = np.min(W_, axis=0)
    idx_bool_ = w_ >= xi
    print(kappa_min, idx_bool_.sum(), kappa_max)

    # Index from selected scenarios
    idx_1_ = np.arange(w_.shape[0])[idx_bool_]
    # Filter by temporal distance
    if idx_bool_.sum() > kappa_max:
        print("(1) Filtering by date: ")
        idx_bool_p_ = idx_bool_ & (d_p_ <= gamma)
        print(idx_bool_p_.sum())

        if idx_bool_p_.sum() > kappa_min:
            status    = 1
            idx_bool_ = idx_bool_p_.copy()
        else:
            print(" Bypass filtering by date: ")
            gamma = 0
            print(idx_bool_.sum())

    idx_2_ = np.arange(w_.shape[0])[idx_bool_]

    # Filter by spatial distance
    if idx_bool_.sum() > kappa_max:
        print("(2) Filtering by distance: ")
        status    = 2
        sigma     = np.sort(d_h_[idx_bool_])[kappa_max]
        idx_bool_ = idx_bool_ & (d_h_ <= sigma)
        print(idx_bool_.sum())

    if idx_bool_.sum() < kappa_min:
        print("Increasing similarity threshold: ")
        status    = 2
        gamma     = 0
        xi        = np.sort(w_)[::-1][kappa_min]
        idx_bool_ = w_ >= xi
        print(idx_bool_.sum())

    idx_3_ = np.arange(w_.shape[0])[idx_bool_]

    return w_, idx_rank_, idx_bool_, idx_1_, idx_2_, idx_3_, xi, gamma, sigma, status

def _empirical_interval_score(y_true, y_pred_upper, y_pred_lower, alpha):

    """
    Calculate the interval score for probabilistic forecasts with an interval [lower, upper].
    
    Parameters:
    - y_true: Observed (true) values
    - y_pred_upper: upper confidence interval for significance level alpha
    - y_pred_lower: low confidence interval for significance level alpha
    - alpha: Significance level (default 0.05 for 90% confidence interval)
    
    Returns:
    - interval_score: The calculated interval score
    """
        
    # Penalty for observation outside the lower bound
    penalty_lower = 2.*np.maximum(0, y_pred_lower - y_true)/alpha
    
    # Penalty for observation outside the upper bound
    penalty_upper = 2.*np.maximum(0, y_true - y_pred_upper)/alpha
    
    # Interval width penalty
    penalty_width = y_pred_upper - y_pred_lower
    
    # Total interval score
    return penalty_lower + penalty_upper + penalty_width

def _eQuantile(_eCDF, q_):
    """
    Calculates quantiles from an ECDF.

    Args:
    _eCDF: function from statsmodels api
    q_: A list or numpy array of quantiles to calculate (values between 0 and 1).

    Returns:
    _Q: A dictionary where keys are the input quantiles and values are the corresponding
    Quantile values from the ECDF.
    """

    return np.array([_eCDF.x[np.searchsorted(_eCDF.y, q)] for q in q_])

# Derive confidence intervals from Directional Quantiles
def _confidence_intervals_from_DQ(M_, alpha_, path):

    # Calculate functional Directional Quantiles (DQ)
    DQ_ = _fQuantile(M_, path).to_numpy()[:, ::-1]


    _y_pred_upper = {}
    _y_pred_lower = {}
    for i in range(len(alpha_)):
        scen_                         = 100.0 * M_[DQ_[:, i] <= 1.0]
        _y_pred_upper[f'{alpha_[i]}'] = np.max(scen_, axis = 0)
        _y_pred_lower[f'{alpha_[i]}'] = np.min(scen_, axis = 0)

    m_ = 100.0 * np.mean(M_, axis = 0)

    return m_, _y_pred_upper, _y_pred_lower

# Derive confidence intervals from a functional depth metric
def _confidence_intervals_from_fDepth(M_, alpha_, depth, path):

    # Calculate functional depth ranking
    D_ = _fDepth(M_, depth, path).to_numpy()[:, 0]
    I_ = np.argsort(D_)

    _y_pred_upper = {}
    _y_pred_lower = {}
    for i in range(len(alpha_)):
        scen_                         = 100.0 * M_[I_[int(M_.shape[0] * alpha_[i]) :],]
        _y_pred_upper[f'{alpha_[i]}'] = np.max(scen_, axis = 0)
        _y_pred_lower[f'{alpha_[i]}'] = np.min(scen_, axis = 0)

    m_ = 100.0 * np.median(M_, axis = 0)

    return m_, _y_pred_upper, _y_pred_lower

# Derive confidence intervals from a functional depth metric
def _confidence_intervals_from_eCDF(M_, alpha_):    

    _y_pred_upper = {}
    _y_pred_lower = {}
    for i in range(len(alpha_)):

        _y_pred_lower[f'{alpha_[i]}'] = 100.0 * np.stack([_eQuantile(ECDF(M_[:, j]), [alpha_[i]/2.])
                                                          for j in range(M_.shape[1])])[:, 0]
        _y_pred_upper[f'{alpha_[i]}'] = 100.0 * np.stack([_eQuantile(ECDF(M_[:, j]), [1. - alpha_[i]/2.])
                                                          for j in range(M_.shape[1])])[:, 0]

    m_ = 100.0 * np.median(M_, axis = 0)

    return m_, _y_pred_upper, _y_pred_lower

def _weighted_interval_score(y_true, y_pred, _y_pred_upper, _y_pred_lower, alpha_):
    """
    Calculate the interval score for probabilistic forecasts with an interval [lower, upper].
    
    Parameters:
    - y_true: Observed (true) values
    - _y_pred_lower: dictionary with upper confidence interval for all significance levels alpha
    - _y_pred_lower: dictionary with lower confidence interval for all significance levels alpha
    - alpha: all significance level alpha (default 0.05 for 90% confidence interval)

    Returns:
    - WIS: float, the Weighted Interval Score.
    """

    # Calculate the interval score
    w0  = 1/2.
    w_  = np.array(alpha_)/2.
    is_ = np.zeros((y_true.shape[0], w_.shape[0]))
    for i in range(len(alpha_)):
        is_[:, i] = _empirical_interval_score(y_true, 
                                              _y_pred_upper[f'{alpha_[i]}'],
                                              _y_pred_lower[f'{alpha_[i]}'], 
                                               alpha_[i])
    
    term0 = 1./(len(alpha_) + 1/2.)
    term1 = w0 * np.absolute(y_true - y_pred)

    for i in range(w_.shape[0]):
        is_[:, i] = w_[i] * is_[:, i]
    term2 = np.sum(is_, axis = 1)
        
    return term1 * (term1 + term2)

