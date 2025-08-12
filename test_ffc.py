import os, glob, datetime, sys, time

import pandas as pd
import numpy as np
import pickle as pkl

from itertools import product
from mpi4py import MPI

from loading_utils import (_process_metadata, 
                           _process_training_curves, 
                           _process_testing_curves, 
                           _process_traning_forecasts, 
                           _process_testing_forecasts)

from ffc_utils import (_euclidian_dist, 
                       _periodic_dist, 
                       _haversine_dist,
                       _rbf_kernel,
                       _logistic, 
                       _scenario_filtering, 
                       _exponential_growth, 
                       _exponential_decay,
                       _silverman_rule)

from functional_utils import (_confidence_intervals_from_eCDF,
                              _confidence_intervals_from_MBD, 
                              _confidence_intervals_from_DQ)

from scores_utils import (_empirical_interval_score,
                          _empirical_coverage_score,
                          _weighted_empirical_interval_score)

path_to_fDepth     = '/home/gterren/dynamic_update/functional_forecast_dynamic_update/fDepth'
path_to_data       = '/home/gterren/dynamic_update/data'
path_to_validation = '/home/gterren/dynamic_update/validation'

def _fknn_forecast_dynamic_update(F_tr_, E_tr_, x_tr_, dt_, f_, e_, x_, f_hat_,
                                  forget_rate_f  = 1.,
                                  forget_rate_e  = .5,
                                  length_scale_f = .1,
                                  length_scale_e = .75,
                                  lookup_rate    = .05,
                                  trust_rate     = 0.0175,
                                  nu             = 340,
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
    w_f_ = _rbf_kernel(d_f_, length_scale_f)
    w_e_ = _rbf_kernel(d_e_, length_scale_e)
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

    nu   = t*5 + nu
    eta_ = _logistic(s_ - nu, k = trust_rate)

    # Fuse scenarios with day-ahead forecasts
    M_ = np.zeros((idx_3_.shape[0], eta_.shape[0]))
    for i, j in zip(idx_3_, range(idx_3_.shape[0])):
        M_[j, :] = F_tr_[i, t:]*(1. - eta_) + E_tr_[i, t:]*eta_

    return M_, phi_, psi_, eta_, status

def _save_validaiton_csv(df_new_, path_to_file):

    # Check if the CSV exists
    if os.path.exists(path_to_file):
        for i in range(5):
            try:
                # Read existing data
                df_existing_ = pd.read_csv(path_to_file)
                break
            except pd.errors.EmptyDataError:
                print('File is being written by another process. Retrying...')
                time.sleep(5)

        df_new_ = pd.concat([df_existing_, df_new_], ignore_index = True).reset_index(drop = True)

    # Overwrite the CSV with the updated data
    df_new_.to_csv(path_to_file, index = False)

# Timestamps in interval
T   = 288
dt_ = np.array([t*5 for t in range(T)])

# Loading and processing of sites metadata
meta_ = _process_metadata(file_name = '/wind_meta.xlsx', 
                          path      = path_to_data)
assets_ = meta_.index
X_tr_   = meta_[['lon', 'lat']].to_numpy()
#print(X_tr_.shape)

meta_      = meta_.reset_index(drop = False)
vals, idx_ = np.unique(X_tr_, return_index = True, axis = 0)
assets_    = assets_[idx_]
X_tr_      = X_tr_[idx_, :]
#print(assets_.shape)

idx_       = np.argsort(assets_)
assets_    = assets_[idx_]
X_tr_      = X_tr_[idx_, :]
#print(assets_.shape)

# Loading and processing of historical curves for the training dataset
F_tr_, T_tr_, x_tr_, p_ = _process_training_curves(X_tr_, assets_, T,
                                                   file_name = '/actuals/wind_actual_5min_site_2017.csv',
                                                   path      = path_to_data)
print(F_tr_.shape, x_tr_.shape)

# Loading and processing of historical curves for the testing dataset
F_ts_, T_ts_, x_ts_ = _process_testing_curves(X_tr_, assets_, p_, T,
                                              file_name = '/actuals/wind_actual_5min_site_2018.csv',
                                              path      = path_to_data)
print(F_ts_.shape, x_ts_.shape)

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

t_ts_ = np.array([datetime.datetime.strptime(t_ts, '%Y-%m-%d').timetuple().tm_yday for t_ts in T_ts_]) - 1
t_tr_ = np.array([datetime.datetime.strptime(t_tr, '%Y-%m-%d').timetuple().tm_yday for t_tr in T_tr_]) - 1
#print(t_tr_.shape, t_ts_.shape)

# Test setup
a = 0
d = 0
t = 144
print(f'{a}-{d}-{t}')

# Get functional predictors for a given test
f_     = F_ts_[d, :t, a]
e_     = E_ts_[d, :, a]
x_     = x_ts_[a, :]
f_hat_ = F_ts_[d, t:, a]
print(f_.shape, e_.shape, x_.shape, f_hat_.shape)

# Significance levels for confidence intervals
alpha_ = [0.1, 0.2, 0.3, 0.4]
zeta_  = [0.1, 0.2, 0.3, 0.4]

t1 = datetime.datetime.now()

M_, phi_, psi_, eta_, status = _fknn_forecast_dynamic_update(F_tr_, E_tr_, x_tr_, dt_, f_, e_, x_, f_hat_, 
                                                             forget_rate_f  = 1.,
                                                             forget_rate_e  = 0.7,
                                                             length_scale_f = 0.025,
                                                             length_scale_e = 0.25,
                                                             lookup_rate    = 0.005,
                                                             trust_rate     = 0.025,
                                                             nu             = 600,
                                                             gamma          = _periodic_dist(t_tr_[d], 250),
                                                             xi             = 0.7,
                                                             kappa_min      = 100,
                                                             kappa_max      = 1000)

t2 = datetime.datetime.now()
print(t2 - t1)

# Calculate confidence intervals from Directional Quantiles
m_, _upper, _lower = _confidence_intervals_from_DQ(M_, alpha_, zeta_)
WIS_               = _weighted_empirical_interval_score(f_hat_, m_, _lower, _upper, alpha_)
CS_                = _empirical_coverage_score(f_hat_, _lower, _upper, alpha_)

t3 = datetime.datetime.now()
print(t3 - t2)

# Calculate confidence intervals from Modified Band Depth
m_, _upper, _lower = _confidence_intervals_from_MBD(M_, alpha_, zeta_)
WIS_               = _weighted_empirical_interval_score(f_hat_, m_, _lower, _upper, alpha_)
CS_                = _empirical_coverage_score(f_hat_, _lower, _upper, alpha_)

t4 = datetime.datetime.now()
print(t4 - t3)

# Calculate marginal empirical confidence intervals
m_, _upper, _lower = _confidence_intervals_from_eCDF(M_, alpha_, zeta_)
WIS_               = np.mean(_weighted_empirical_interval_score(f_hat_, m_, _lower, _upper, alpha_))
CS_                = _empirical_coverage_score(f_hat_, _lower, _upper, alpha_)

t5 = datetime.datetime.now()
print(t5 - t3)