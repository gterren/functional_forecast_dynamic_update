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

from ffc_utils import _fknn_forecast_dynamic_update


from functional_utils import (_confidence_intervals_from_eCDF,
                              _confidence_intervals_from_MBD, 
                              _confidence_intervals_from_DQ)

from scores_utils import (_empirical_interval_score,
                          _empirical_coverage_score,
                          _weighted_empirical_interval_score)

path_to_fDepth     = '/home/gterren/dynamic_update/functional_forecast_dynamic_update/fDepth'
path_to_data       = '/home/gterren/dynamic_update/data'
path_to_validation = '/home/gterren/dynamic_update/validation'

def _save_validation_csv(df_new_, path_to_file):

    if isinstance(df_new_, pd.DataFrame):

        # Check if the CSV exists
        if os.path.exists(path_to_file):

            df_existing_ = pd.read_csv(path_to_file)
            df_new_      = pd.concat([df_existing_, df_new_], ignore_index = True).reset_index(drop = True)

        # Overwrite the CSV with the updated data
        df_new_.to_csv(path_to_file, index = False)
        print(path_to_file)

# Gather data from all MPI nodes
def _gather_node_data(_comm, df_):

    # Gather all dictionaries at root (rank 0)
    _gathered = _comm.gather(df_.to_dict(), root = 0)

    if _comm.Get_rank() == 0:
        # Convert back to DataFrames and concatenate
        return pd.concat([pd.DataFrame.from_dict(d) for d in _gathered], ignore_index = True)
    else:
        return None
    
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
#print(F_tr_.shape, T_tr_.shape, x_tr_.shape, p_.shape)

# Loading and processing of historical curves for the testing dataset
F_ts_, T_ts_, x_ts_ = _process_testing_curves(X_tr_, assets_, p_, T,
                                              file_name = '/actuals/wind_actual_5min_site_2018.csv',
                                              path      = path_to_data)
#print(F_ts_.shape, T_ts_.shape, x_ts_.shape)

# Loading and processing of historical day-ahead forecast for the training dataset
E_tr_ = _process_traning_forecasts(assets_, p_, T, 
                                   file_name = '/actuals/wind_day_ahead_forecast_2017.csv',
                                   path      = path_to_data)
#print(E_tr_.shape)

# Loading and processing of historical day-ahead forecast for the testing dataset
E_ts_ = _process_testing_forecasts(assets_, p_, T,
                                   file_name = '/actuals/wind_day_ahead_forecast_2018.csv', 
                                   path      = path_to_data)
#print(E_ts_.shape)

dt_ = np.array([t*5 for t in range(T)])
dx_ = pd.to_datetime(pd.DataFrame({'time': dt_}).time, unit = 'm').dt.strftime('%H:%M').to_numpy()
#print(dt_.shape, dx_.shape)

t_ts_ = np.array([datetime.datetime.strptime(t_ts, '%Y-%m-%d').timetuple().tm_yday for t_ts in T_ts_])
t_tr_ = np.array([datetime.datetime.strptime(t_tr, '%Y-%m-%d').timetuple().tm_yday for t_tr in T_tr_]) - 1
#print(t_tr_.shape, t_ts_.shape)

# Calibration experiments setup
#times_  = [12*12]
times_  = [int(sys.argv[2])]

# Assets in the calibration experiments
assets_ = [0]

# Significance levels for the confidence intervals
alpha_ = [0.1, 0.2, 0.3, 0.4]

# Hyperparameters for the functional forecast dynamic update
forget_rate_f_  = [3.]
forget_rate_e_  = [.25]
length_scale_f_ = [0.005]   
length_scale_e_ = [0.5]
lookup_rate_    = [64]
trust_rate_     = [3]
nu_             = [8]
gamma_          = [90]
xi_             = [0.6]
kappa_min_      = [250]
kappa_max_      = [2000]

if sys.argv[1] == 'forget_rate_f':
    forget_rate_f_ = [0.25, 0.5, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]

if sys.argv[1] == 'forget_rate_e':
    forget_rate_e_ =  [0.25, 0.5, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]

if sys.argv[1] == 'length_scale_f':
    length_scale_f_ = [0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.]

if sys.argv[1] == 'length_scale_e':
    length_scale_e_ = [0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.]

if sys.argv[1] == 'lookup_rate':
    lookup_rate_ = [0.5, 1., 2., 4., 8., 16., 32., 64., 128., 256., 512., 1024.]

if sys.argv[1] == 'trust_rate':
    trust_rate_ = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]

if sys.argv[1] == 'nu':
    nu_ = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]

if sys.argv[1] == 'gamma':
    gamma_ = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]

if sys.argv[1] == 'xi':
    xi_ = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

if sys.argv[1] == 'kappa_min':
    kappa_min_ = [50, 100, 150, 200, 250, 300, 350, 400, 500, 750, 1000, 1500]

if sys.argv[1] == 'kappa_max':
    kappa_max_ = [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250]

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
print(params_)

dfs_ = []
for time in times_:
    for asset in assets_:
        for day in range(365):
            #t1 = datetime.datetime.now()

            file_name = f'{asset}-{day}-{time}'
            #print(i_job, file_name)

            f_     = F_ts_[day, :time, asset]
            e_     = E_ts_[day, :, asset]
            x_     = x_ts_[asset, :]
            t      = t_ts_[day]
            f_hat_ = F_ts_[day, time:, asset]

            # Get time constants
            tau_ = dt_[:time]
            s_   = dt_[time:]

            _meta, M_, status = _fknn_forecast_dynamic_update(F_tr_, E_tr_, x_tr_, t_tr_, dt_, f_, e_, x_, t,
                                                              forget_rate_f  = params_[0],
                                                              forget_rate_e  = params_[1],
                                                              length_scale_f = params_[2],
                                                              length_scale_e = params_[3],
                                                              lookup_rate    = params_[4],
                                                              trust_rate     = params_[5],
                                                              nu             = params_[6],
                                                              gamma          = params_[7],
                                                              xi             = params_[8],
                                                              kappa_min      = params_[9],
                                                              kappa_max      = params_[10])

            f_tau_rmse = np.sqrt(np.mean((f_ - e_[:time])**2))
            f_s_rmse   = np.sqrt(np.mean((np.median(M_, axis = 0) - e_[time:])**2))
       
            # Calculate marginal empirical confidence intervals
            m_, _upper, _lower = _confidence_intervals_from_eCDF(M_, alpha_)
            WIS                = np.mean(_weighted_empirical_interval_score(f_hat_, m_, _lower, _upper, alpha_))
            CS_                = _empirical_coverage_score(f_hat_, _lower, _upper, alpha_)

            # Save results
            x_ = list(params_ + tuple([time, asset, day, M_.shape[0], status, float(WIS), float(f_tau_rmse), float(f_s_rmse)]))
            dfs_.append(x_)
            
            #t2 = datetime.datetime.now()
            #print(t2 - t1)

print(i_job, sys.argv[1], sys.argv[2], datetime.datetime.now())

dfs_ = pd.DataFrame(dfs_, columns = ['forget_rate_f', 
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
                                     'WIS', 
                                     'RMSE_tau', 
                                     'RMSE_s'])

dfs_ = _gather_node_data(_comm, dfs_)

_save_validation_csv(dfs_, path_to_file = f'{path_to_validation}/validation_ffc-WIS-{sys.argv[1]}-{sys.argv[2]}.csv')


