import os, glob, datetime, sys

import pandas as pd
import numpy as np
import pickle as pkl
import scipy.stats as stats

from scipy.integrate import quad
from scipy.stats import multivariate_normal, norm
from itertools import product

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
        # Read existing data
        df_existing_ = pd.read_csv(path_to_file)
        df_new_      = pd.concat([df_existing_, df_new_], ignore_index = True).reset_index(drop = True)

    # Overwrite the CSV with the updated data
    df_new_.to_csv(path_to_file, index = False)
    print(df_new_.shape)

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
i_job, N_jobs, _comm = _get_node_info()

# Timestamps in interval
T = 288

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

# Calibration experiments setup
#times_  = [12*12]
times_  = [int(sys.argv[2])]
assets_ = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
alpha_  = [0.1, 0.2, 0.4]

# Hyperparameters for the functional forecast dynamic update
forget_rate_f_  = [.1]
forget_rate_e_  = [.1]
lookup_rate_    = [.1]
trust_rate_     = [.1]
length_scale_f_ = [.1]   
length_scale_e_ = [.1]
nu_             = [375]
gamma_          = [180]
xi_             = [0.9]
zeta_           = [1.]
kappa_min_      = [200]
kappa_max_      = [1000]

if sys.argv[1] == 'forget_rate_f':
    forget_rate_f_ = np.geomspace(0.001, 2., 13).tolist()

if sys.argv[1] == 'forget_rate_e':
    forget_rate_e_ = np.geomspace(0.001, 2., 13).tolist()

if sys.argv[1] == 'length_scale_f':
    length_scale_f_ = np.geomspace(0.01, 2., 13).tolist()

if sys.argv[1] == 'length_scale_e':
    length_scale_e_ = np.geomspace(0.01, 2., 13).tolist()

if sys.argv[1] == 'lookup_rate':
    lookup_rate_ = np.geomspace(0.001, 2., 13).tolist()

if sys.argv[1] == 'trust_rate':
    trust_rate_ = np.geomspace(0.001, 2., 13).tolist()

if sys.argv[1] == 'nu':
    nu_ = np.linspace(0, 750., 13).tolist()

if sys.argv[1] == 'gamma':
    gamma_ = np.linspace(15, 365, 13, dtype = int).tolist()

if sys.argv[1] == 'xi':
    xi_ = np.logspace(-.1, -0.0001, 13).tolist()

if sys.argv[1] == 'kappa_min':
    kappa_min_ = np.linspace(50, 500, 13, dtype = int).tolist()

if sys.argv[1] == 'kappa_max':
    kappa_max_ = np.linspace(500, 2000, 13, dtype = int).tolist()

params_ = tuple(product(forget_rate_f_, 
                        forget_rate_e_, 
                        length_scale_f_,
                        length_scale_e_,
                        lookup_rate_, 
                        trust_rate_, 
                        nu_,
                        gamma_,
                        xi_, 
                        kappa_min_, 
                        kappa_max_))[i_job]

dfs_1_ = []
dfs_2_ = []
for t in times_:
    for a in assets_:
        for d in range(365):
            t1 = datetime.datetime.now()

            file_name = f'{a}-{d}-{t}'
            print(file_name)

            f_     = F_ts_[d, :t, a]
            e_     = E_ts_[d, :, a]
            x_     = x_ts_[a, :]
            f_hat_ = F_ts_[d, t:, a]

            # Get time constants
            tau_ = dt_[:t]
            s_   = dt_[t:]

            M_, phi_, psi_, eta_, status = _fknn_forecast_dynamic_update(F_tr_, E_tr_, x_tr_, dt_, f_, e_, x_, f_hat_, 
                                                                         forget_rate_f  = params_[0],
                                                                         forget_rate_e  = params_[1],
                                                                         length_scale_f = params_[2],
                                                                         length_scale_e = params_[3],
                                                                         lookup_rate    = params_[4],
                                                                         trust_rate     = params_[5],
                                                                         nu             = params_[6],
                                                                         gamma          = _periodic_dist(t_tr_[d], params_[7]),
                                                                         xi             = params_[8],
                                                                         kappa_min      = params_[9],
                                                                         kappa_max      = params_[10])

            # # Calculate confidence intervals from Directional Quantiles
            # m_, _upper, _lower = _confidence_intervals_from_DQ(M_, alpha_, params_[8])
            # WIS_DQ_            = _weighted_empirical_interval_score(f_hat_, m_, _lower, _upper, alpha_)
            # CS_DQ_             = _empirical_coverage_score(f_hat_, _lower, _upper, alpha_)

            # # Calculate confidence intervals from Modified Band Depth
            # m_, _upper, _lower = _confidence_intervals_from_MBD(M_, alpha_, params_[8])
            # WIS_MBD_           = _weighted_empirical_interval_score(f_hat_, m_, _lower, _upper, alpha_)
            # CS_MBD_            = _empirical_coverage_score(f_hat_, _lower, _upper, alpha_)

            # Calculate marginal empirical confidence intervals
            m_, _upper, _lower = _confidence_intervals_from_eCDF(M_, alpha_, 1)
            WIS_eCDF_          = _weighted_empirical_interval_score(f_hat_, m_, _lower, _upper, alpha_)
            CS_eCDF_           = _empirical_coverage_score(f_hat_, _lower, _upper, alpha_)

            # Save results
            x_ = list(params_ + tuple([t, a, d, M_.shape[0], status, float(WIS_eCDF_.mean())]))
            dfs_1_.append(x_)

            y_ = list(params_ + tuple([t, a, d, M_.shape[0], status]) + tuple(CS_eCDF_))
            dfs_2_.append(y_)

            t2 = datetime.datetime.now()
            print(t2 - t1)

dfs_1_ = pd.DataFrame(dfs_1_, columns = ['forget_rate_f', 
                                         'forget_rate_e', 
                                         'length_scale_f',
                                         'length_scale_e', 
                                         'lookup_rate', 
                                         'trust_rate',
                                         'nu', 
                                         'gamma', 
                                         'xi',
                                         'kappa_min', 
                                         'kappa_max',
                                         'time', 
                                         'asset', 
                                         'day', 
                                         'n_scenarios', 
                                         'status',
                                         'WIS_eCDF'])
print(dfs_1_.shape)

_save_validaiton_csv(dfs_1_, path_to_file = f'{path_to_validation}/ffc_validation-WIS-{sys.argv[1]}.csv')

dfs_2_ = pd.DataFrame(dfs_2_, columns = ['forget_rate_f', 
                                         'forget_rate_e', 
                                         'length_scale_f',
                                         'length_scale_e', 
                                         'lookup_rate', 
                                         'trust_rate',
                                         'nu', 
                                         'gamma', 
                                         'xi',
                                         'kappa_min', 
                                         'kappa_max',
                                         'time', 
                                         'asset', 
                                         'day', 
                                         'n_scenarios', 
                                         'status',
                                         'CS_eCDF_90', 
                                         'CS_eCDF_80', 
                                         'CS_eCDF_60'])
print(dfs_2_.shape)
_save_validaiton_csv(dfs_2_, path_to_file = f'{path_to_validation}/ffc_validation-CS-{sys.argv[1]}.csv')
