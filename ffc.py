import os, glob, datetime

import pandas as pd
import numpy as np
import pickle as pkl
import scipy.stats as stats
import properscoring as ps
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import geopandas as gpd
import seaborn as sns

from scores_utils import *
from functional_utils import _fDepth, _fQuantile

from loading_utils import (_process_metadata, 
                           _process_training_curves, 
                           _process_testing_curves, 
                           _process_traning_forecasts, 
                           _process_testing_forecasts)

from sklearn.neighbors import KernelDensity
from scipy.integrate import quad
from ipywidgets import *
from scipy.stats import multivariate_normal, norm
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib import cm
from matplotlib_scalebar.scalebar import ScaleBar

path_to_fDepth = '/Users/Guille/Desktop/dynamic_update/functional_forecast_dynamic_update/fDepth'
path_to_data   = '/Users/Guille/Desktop/dynamic_update/data'

# Calculate weighted (w_) distance between X_ and x_
def _dist(X_, x_, w_ = []):
    if len(w_) == 0:
        w_ = np.ones(x_.shape)
    w_ = w_/w_.sum()
    d_ = np.zeros((X_.shape[0], ))
    for i in range(X_.shape[0]):
        d_[i] = w_.T @ (X_[i, :] - x_)**2
    return d_

# Radial Basis function kernel based on distance (d_)
def _kernel(d_, length_scale):
    w_ = np.exp(-d_/length_scale)
    return w_#/w_.sum()

def _inv_dist(d_, length_scale):
    w_ = 1./(d_ + length_scale)
    return w_#/w_.sum()

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

# # Fuse day-ahead forecast with real-time forecast
# def _update_forecast(F_ac_, f_hat_, fc_, update_rate):

#     w_update_ = 1. - _exponential_decay_plus(F_ac_.shape[1] + 1, update_rate)[1:]
#     #w_update_ = eta_/eta_.max()
#     f_update_ = f_hat_*(1. - w_update_) + fc_*w_update_

#     plt.figure(figsize = (10, 2))
#     plt.title('Trust Rate')
#     plt.plot(w_update_)
#     plt.show()

#     return f_update_

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
    return 0.9 * min(np.std(x_), IQR / 1.34) * x_.shape[0] ** (-1/5)

# Circular distance to rank samples by day of the year
def _circular_dist(x_1_, x_2_, 
                   day_to_degree = 360/365, 
                   degree_to_rad = np.pi/180):
    return np.sin(.5*(day_to_degree * (x_2_ - x_1_) * degree_to_rad))**2


# Filtering scenarios when are above the upper threshold or below lower threshold
def _scenario_filtering(W_, d_s_, d_t_, xi, gamma, kappa_min, kappa_max):
    
    status = 0
    s      = 0

    # Similarity ranking
    idx_rank_ = np.argmin(W_, axis = 0)
    
    # Similarity filter
    w_        = np.min(W_, axis = 0)
    idx_bool_ = w_ >= xi
    #print(kappa_min, idx_bool_.sum(), kappa_max)
    
    # Index from selected scenarios
    idx_1_ = np.arange(w_.shape[0])[idx_bool_]
    
    # Filter by temporal distance
    if idx_bool_.sum() > kappa_max:
        idx_bool_p_ = idx_bool_ & (d_t_ <= gamma)
        #print('(1) Filtering by date: ')
        #print(idx_bool_p_.sum())

        if idx_bool_p_.sum() > kappa_min:
            status    = 1
            idx_bool_ = idx_bool_p_.copy()
        else:
            gamma = 0
            #print(' Bypass filtering by date: ')
            #print(idx_bool_.sum())

    # Filter by spatial distance
    if idx_bool_.sum() > kappa_max:
        status    = 2
        s         = np.sort(d_s_[idx_bool_])[kappa_max]
        idx_bool_ = idx_bool_ & (d_s_ <= s)
        #print('(2) Filtering by distance: ')
        #print(idx_bool_.sum())

    if idx_bool_.sum() < kappa_min:
        status    = 2
        iota      = 0
        xi        = np.sort(w_)[::-1][kappa_min]
        idx_bool_ = w_ >= xi
        #print('Increasing similarity threshold: ')
        #print(idx_bool_.sum())
    
    idx_2_ = np.arange(w_.shape[0])[idx_bool_]

    return w_, idx_rank_, idx_bool_, idx_1_, idx_2_, xi, gamma, s, status

def _fknn_forecast_dynamic_update(F_tr_, E_tr_, x_tr_, dt_, f_, e_, x_, f_hat_,
                                  forget_rate_f  = 1.,
                                  forget_rate_e  = .5,
                                  length_scale_f = .1,
                                  length_scale_e = .75,
                                  lookup_rate    = .05,
                                  trust_rate     = .005,
                                  gamma          = .2,
                                  xi             = 0.99,
                                  kappa_min      = 100,
                                  kappa_max      = 250):


    # Get constants
    t    = f_.shape[0]
    tau_ = dt_[:t]
    s_   = dt_[t:]

    # phi: importance weights based on past time distance
    phi_ = _exponential_growth(t, forget_rate_f)

    # psi: importance weights based on past and future time distance
    psi_1_ = _exponential_growth(t, forget_rate_e)
    psi_2_ = _exponential_decay(t, lookup_rate)
    psi_   = np.concatenate([psi_1_, psi_2_], axis = 0)
    
    # d: euclidian distance between samples weighted by importance weights
    d_f_ = _dist(F_tr_[:, :t], f_, w_ = phi_)
    d_e_ = _dist(E_tr_, e_, w_ = psi_)
    d_s_ = _haversine_dist(x_ts_[a, :], x_tr_)
    d_t_ = _circular_dist(t_tr_[d], t_tr_)
    # print(x_tr_.shape, x_ts_.shape, d_s_.shape)
    # print(t_tr_.shape, t_ts_.shape, d_t_.shape)

    # w: normalized weights distance across observations based exponential link function
    w_f_ = _kernel(d_f_, length_scale_f)
    w_e_ = _kernel(d_e_, length_scale_e)
    W_   = np.stack([w_f_, w_e_])

    w_, idx_rank_, idx_bool_, idx_1_, idx_2_, xi, gamma, s, status = _scenario_filtering(W_, 
                                                                                         d_s_, 
                                                                                         d_t_, 
                                                                                         xi, 
                                                                                         gamma, 
                                                                                         kappa_min, 
                                                                                         kappa_max)

    # eta: importance weights based on future time distance
    rmse  = np.sqrt(np.mean((e_[:t] - f_)**2))
    wrmse = np.sqrt(np.sum(psi_1_*(e_[:t] - f_)**2)/psi_1_.sum())
    nu    = np.sqrt(rmse)*2750
    if nu < 875: nu = 875
    #print(rmse, wrmse)

    eta_ = _logistic(s_ - nu, k = trust_rate)

    # Fuse scenarios with day-ahead forecasts
    M_ = np.zeros((idx_2_.shape[0], eta_.shape[0]))
    for i, j in zip(idx_2_, range(idx_2_.shape[0])):
        M_[j, :] = F_tr_[i, t:]*(1. - eta_) + E_tr_[i, t:]*eta_

    return M_, phi_, psi_, eta_

# Timestamps in interval
T = 288

# Loading color palette
palette_ = pd.read_csv(path_to_data + '/palette.csv')
print(palette_)

# Loading Texas map
TX_ = gpd.read_file(path_to_data + '/maps/TX/State.shp')

# Loading and processing of sites metadata
meta_ = _process_metadata(file_name = '/wind_meta.xlsx', 
                          path      = path_to_data)
assets_ = meta_.index
X_tr_   = meta_[['lon', 'lat']].to_numpy()
print(X_tr_.shape)

meta_      = meta_.reset_index(drop = False)
vals, idx_ = np.unique(X_tr_, return_index = True, axis = 0)
assets_    = assets_[idx_]
X_tr_      = X_tr_[idx_, :]
print(assets_.shape)

idx_       = np.argsort(assets_)
assets_    = assets_[idx_]
X_tr_      = X_tr_[idx_, :]
print(assets_.shape)

# Loading and processing of historical curves for the training dataset
F_tr_, T_tr_, x_tr_, p_ = _process_training_curves(X_tr_, assets_, T,
                                                   file_name = '/actuals/wind_actual_5min_site_2017.csv',
                                                   path      = path_to_data)

print(F_tr_.shape, T_tr_.shape, x_tr_.shape, p_.shape)

# Loading and processing of historical curves for the testing dataset
F_ts_, T_ts_, x_ts_ = _process_testing_curves(X_tr_, assets_, p_, T,
                                              file_name = '/actuals/wind_actual_5min_site_2018.csv',
                                              path      = path_to_data)
print(F_ts_.shape, T_ts_.shape, x_ts_.shape)

# Loading and processing of historical day-ahead forecast for the training dataset
E_tr_ = _process_traning_forecasts(assets_, p_, T, 
                                   file_name = '/actuals/wind_day_ahead_forecast_2017.csv',
                                   path      = path_to_data)
print(E_tr_.shape)

# Loading and processing of historical day-ahead forecast for the testing dataset
E_ts_ = _process_testing_forecasts(assets_, p_, T,
                                   file_name = '/actuals/wind_day_ahead_forecast_2018.csv', 
                                   path      = path_to_data)
print(E_ts_.shape)

dt_ = np.array([t*5 for t in range(T)])
dx_ = pd.to_datetime(pd.DataFrame({'time': dt_}).time, unit = 'm').dt.strftime('%H:%M').to_numpy()
print(dt_.shape, dx_.shape)

t_ts_ = np.array([datetime.datetime.strptime(t_ts, '%Y-%m-%d').timetuple().tm_yday for t_ts in T_ts_]) - 1
t_tr_ = np.array([datetime.datetime.strptime(t_tr, '%Y-%m-%d').timetuple().tm_yday for t_tr in T_tr_]) - 1
print(t_tr_.shape, t_ts_.shape)



a = 2
d = 7
t = 12*12

t1 = datetime.datetime.now()
for d in range(363):

    file_name = f'{a}-{d}-{t}'
    #print(file_name)

    f_     = F_ts_[d, :t, a]
    e_     = E_ts_[d, :, a]
    x_     = x_ts_[a, :]
    f_hat_ = F_ts_[d, t:, a]

    # Get constants
    tau_ = dt_[:t]
    s_   = dt_[t:]

    M_, phi_, psi_, eta_ = _fknn_forecast_dynamic_update(F_tr_, E_tr_, x_tr_, dt_, f_, e_, x_, f_hat_, 
                                                         forget_rate_f  = 1.,
                                                         forget_rate_e  = .5,
                                                         length_scale_f = .1,
                                                         length_scale_e = .75,
                                                         lookup_rate    = .05,
                                                         trust_rate     = .005,
                                                         xi             = 0.99,
                                                         gamma          = .2)

t2 = datetime.datetime.now()
print(t2 - t1)

# forget_rate_f_ = []
# forget_rate_e_ = []
# lookup_rate_   = []
# trust_rate_    = []

# length_scale_f_ = []      
# length_scale_e_ = []

# xi_    = []
# gamma_ = []