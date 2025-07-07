import os, glob, datetime

import pandas as pd
import numpy as np
import pickle as pkl
import scipy.stats as stats

from scipy.integrate import quad
from scipy.stats import multivariate_normal, norm
from functional_utils import _fDepth, _fDepth4POD
from sklearn.neighbors import KernelDensity
from statsmodels.distributions.empirical_distribution import ECDF
from mpi4py import MPI

from loading_utils import (_process_metadata, 
                           _process_training_curves, 
                           _process_testing_curves, 
                           _process_traning_forecasts, 
                           _process_testing_forecasts)

from ffc_utils import (_euclidian_dist, 
                           _periodic_dist, 
                           _haversine_dist,
                           _kernel,
                           _logistic, 
                           _scenario_filtering, 
                           _exponential_growth, 
                           _exponential_decay, 
                           _silverman_rule, 
                           _weighted_interval_score, 
                           _confidence_intervals_from_eCDF, 
                           _confidence_intervals_from_fDepth, 
                           _confidence_intervals_from_DQ)


path_to_fDepth = '/home/gterren/dynamic_update/functional_forecast_dynamic_update/fDepth'
path_to_data   = '/home/gterren/dynamic_update/data'

def _fknn_forecast_dynamic_update(F_tr_, E_tr_, x_tr_, dt_, f_, e_, x_, f_hat_,
                                  forget_rate_f  = 1.,
                                  forget_rate_e  = .5,
                                  length_scale_f = .1,
                                  length_scale_e = .75,
                                  lookup_rate    = .05,
                                  trust_rate     = 0.0175,
                                  gamma          = .2,
                                  xi             = 0.99,
                                  kappa_min      = 100,
                                  kappa_max      = 1000):


    # Get constants
    t    = f_.shape[0]
    tau_ = dt_[:t]
    s_   = dt_[t:]

    # phi: importance weights based on past time distance
    phi_ = _exponential_growth(t, forget_rate_f)

    # psi: importance weights based on past and future time distance
    psi_1_ = _exponential_growth(t, forget_rate_e)
    psi_2_ = _exponential_decay(T - t, lookup_rate)
    psi_   = np.concatenate([psi_1_, psi_2_], axis = 0)

    # d: Euclidean distance between samples weighted by importance weights
    d_f_ = _euclidian_dist(F_tr_[:, :t], f_, w_ = phi_)
    d_e_ = _euclidian_dist(E_tr_, e_, w_ = psi_)
    d_h_ = _haversine_dist(x_ts_[a, :], x_tr_)
    d_p_ = _periodic_dist(t_tr_[d], t_tr_)
    # print(x_tr_.shape, x_ts_.shape, d_s_.shape)
    # print(t_tr_.shape, t_ts_.shape, d_t_.shape)

    # w: normalized weights distance across observations based exponential link function
    w_f_ = _kernel(d_f_, length_scale_f)
    w_e_ = _kernel(d_e_, length_scale_e)
    W_   = np.stack([w_f_, w_e_])

    (w_, 
    idx_rank_, 
    idx_bool_, 
    idx_1_, 
    idx_2_, 
    idx_3_, 
    xi, 
    gamma, 
    sigma, 
    status) = _scenario_filtering(W_, 
                                  d_h_, 
                                  d_p_, 
                                  xi, 
                                  gamma, 
                                  kappa_min, 
                                  kappa_max)

    nu   = t*5 + 340
    eta_ = _logistic(s_ - nu, k = trust_rate)

    # Fuse scenarios with day-ahead forecasts
    M_ = np.zeros((idx_2_.shape[0], eta_.shape[0]))
    for i, j in zip(idx_2_, range(idx_2_.shape[0])):
        M_[j, :] = F_tr_[i, t:]*(1. - eta_) + E_tr_[i, t:]*eta_

    return M_, phi_, psi_, eta_

# Get MPI node information
def _get_node_info(verbose = False):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    if verbose:
        print('>> MPI: Name: {} Rank: {} Size: {}'.format(name, rank, size) )
    return int(rank), int(size), comm


# MPI job variables
#i_job, N_jobs, _comm = _get_node_info()

# Timestamps in interval
T = 288

a = 2
d = 7
t = 12*12

# forget_rate_f_ 
# forget_rate_e_
# lookup_rate_ 
# trust_rate_ 
# length_scale_f_     
# length_scale_e_ 
# xi_   
# gamma_

alpha_ = [0.1, 0.2, 0.4]


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

t1 = datetime.datetime.now()
for d in range(365):

    file_name = f'{a}-{d}-{t}'
    print(file_name)

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
                                                         gamma          = _periodic_dist(t_tr_[d], 90),
                                                         kappa_min      = 100,
                                                         kappa_max      = 1000)
    
    # Calculate confidence intervals from Directional Quantiles
    m_, _upper, _lower = _confidence_intervals_from_DQ(M_, alpha_, path_to_fDepth)
    WIS_ = _weighted_interval_score(f_hat_, m_, _upper, _lower, alpha_)
    print(WIS_.mean())

    x

    # Calculate confidence intervals from functional depth metrics
    # m_, _upper, _lower = _confidence_intervals_from_fDepth(M_, alpha_, 'MBD', path_to_fDepth)
    # WIS_ = _weighted_interval_score(f_hat_, m_, _upper, _lower, alpha_)
    # print(WIS_.mean())

    # Calculate confidence intervals from functional depth metrics
    m_, _upper, _lower = _confidence_intervals_from_eCDF(M_, alpha_)
    WIS_ = _weighted_interval_score(f_hat_, m_, _upper, _lower, alpha_)
    print(WIS_.mean())

t2 = datetime.datetime.now()
print(t2 - t1)
