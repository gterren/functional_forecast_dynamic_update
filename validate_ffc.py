import os, datetime, sys, time

import pandas as pd
import numpy as np
import pickle as pkl

from itertools import product
from mpi4py import MPI

from ffc_utils import _fknn_forecast_dynamic_update

from functional_utils import _confidence_intervals_from_eCDF

from scores_utils import (_empirical_coverage_score,
                          _weighted_empirical_interval_score)

path_to_fDepth     = '/home/gterren/dynamic_update/functional_forecast_dynamic_update/fDepth'
path_to_data       = '/home/gterren/dynamic_update/data'
path_to_validation = '/home/gterren/dynamic_update/validation'

def _save_validation_csv(df_new_, path_to_file):

    if isinstance(df_new_, pd.DataFrame):

        # Check if the CSV exists
        if os.path.exists(path_to_file):

            df_existing_ = pd.read_csv(path_to_file)
            df_new_      = pd.concat([df_existing_, 
                                      df_new_], 
                                      ignore_index = True).reset_index(drop = True)

        # Overwrite the CSV with the updated data
        df_new_.to_csv(path_to_file, index = False)
        print(path_to_file)

# Gather data from all MPI nodes
def _gather_node_data(_comm, df_):

    # Gather all dictionaries at root (rank 0)
    _gathered = _comm.gather(df_.to_dict(), root = 0)

    if _comm.Get_rank() == 0:
        # Convert back to DataFrames and concatenate
        return pd.concat([pd.DataFrame.from_dict(d) for d in _gathered], 
                         ignore_index = True)
    else:
        return None
    
# Get MPI node information
def _get_node_info(verbose = False):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    if verbose:
        print(f'>> MPI: Name: {name} Rank: {rank} Size: {size}')
    return int(rank), int(size), comm

# MPI job variables
i_job, N_jobs, _comm = _get_node_info()

T        = 288
resource = 'wind'

# Load 2017 data as training set
with open(path_to_data + f"/preprocessed_{resource}_2017.pkl", 'rb') as f:
    _data = pkl.load(f)

assets_tr_ = _data["assets"]
x_tr_      = _data["locations"]
dates_tr_  = _data["dates"]
F_tr_      = _data["observations"]
E_tr_      = _data["forecasts"]
#print(assets_tr_.shape, x_tr_.shape, dates_tr_.shape, F_tr_.shape, E_tr_.shape)

# Reshape to day x interval x asset format
F_tr_ = F_tr_.reshape(int(F_tr_.shape[0]/T), T, F_tr_.shape[1])
E_tr_ = E_tr_.reshape(int(E_tr_.shape[0]/T), T, E_tr_.shape[1])
T_tr_ = dates_tr_.reshape(int(dates_tr_.shape[0]/T), T)
#print(F_tr_.shape, E_tr_.shape, T_tr_.shape)

# Load 2018 data as testing set
with open(path_to_data + f"/preprocessed_{resource}_2018.pkl", 'rb') as f:
    _data = pkl.load(f)

assets_ts_ = _data["assets"]
x_ts_      = _data["locations"]
dates_ts_  = _data["dates"]
F_ts_      = _data["observations"]
E_ts_      = _data["forecasts"]
#print(assets_ts_.shape, x_ts_.shape, dates_ts_.shape, F_ts_.shape, E_ts_.shape)

# Reshape to day x interval x asset format
F_ts_ = F_ts_.reshape(int(F_ts_.shape[0]/T), T, F_ts_.shape[1])
E_ts_ = E_ts_.reshape(int(E_ts_.shape[0]/T), T, E_ts_.shape[1])
T_ts_ = dates_ts_.reshape(int(dates_ts_.shape[0]/T), T)
#print(F_ts_.shape, E_ts_.shape, T_ts_.shape)

dt_ = np.array([t * 5 for t in range(T)])
dx_ = pd.to_datetime(pd.DataFrame({"time": dt_}).time, unit = "m").dt.strftime("%H:%M").to_numpy()
#print(dt_.shape, dx_.shape)

# Filter solar hours with loading solar set
idx_hours_ = (np.sum(np.sum(F_tr_, axis = 0), axis = 1) 
              + np.sum(np.sum(F_ts_, axis = 0), axis = 1)
              + np.sum(np.sum(E_tr_, axis = 0), axis = 1)
              + np.sum(np.sum(E_ts_, axis = 0), axis = 1)) > 0.

# F_tr_ = F_tr_[:, idx_, :]
# E_tr_ = E_tr_[:, idx_, :]
# T_tr_ = T_tr_[:, idx_]
# print(F_tr_.shape, E_tr_.shape, T_tr_.shape)

# F_ts_ = F_ts_[:, idx_, :]
# E_ts_ = E_ts_[:, idx_, :]
# T_ts_ = T_ts_[:, idx_]
# print(F_ts_.shape, E_ts_.shape, T_ts_.shape)

# Short testing set with training set order
order      = {v: i for i, v in enumerate(assets_tr_)}
idx_       = np.argsort([order[x] for x in assets_ts_])
assets_ts_ = assets_ts_[idx_]
x_ts_      = x_ts_[idx_]
F_ts_      = F_ts_[:, :, idx_]
E_ts_      = E_ts_[:, :, idx_]
#print(F_ts_.shape, E_ts_.shape, T_ts_.shape)

# From generation to capacity factor
p_tr_ = np.max(np.max(F_tr_, axis = 0), axis = 0)
p_ts_ = np.max(np.max(F_ts_, axis = 0), axis = 0)
#print(p_tr_.shape, p_ts_.shape)

F_tr_ /= np.tile(p_tr_, (F_tr_.shape[0], F_tr_.shape[1], 1))
E_tr_ /= np.tile(p_tr_, (E_tr_.shape[0], E_tr_.shape[1], 1))
F_ts_ /= np.tile(p_ts_, (F_ts_.shape[0], F_ts_.shape[1], 1))
E_ts_ /= np.tile(p_ts_, (E_ts_.shape[0], E_ts_.shape[1], 1))
# print(F_tr_.min(), F_tr_.max())
# print(E_tr_.min(), E_tr_.max())
# print(F_ts_.min(), F_ts_.max())
# print(E_ts_.min(), E_ts_.max())

# No possible a capacity factor is larger than 1 or smaller than 0
F_tr_[F_tr_ > 1.] = 1.
F_tr_[F_tr_ < 0.] = 0.
F_ts_[F_ts_ > 1.] = 1.
F_ts_[F_ts_ < 0.] = 0.
E_tr_[E_tr_ > 1.] = 1.
E_tr_[E_tr_ < 0.] = 0.
E_ts_[E_ts_ > 1.] = 1.
E_ts_[E_ts_ < 0.] = 0.

#print(assets_tr_.shape, T_tr_.shape)
# Format training set from day x interval x asset to [day * asset] x interval
T_tr_      = np.concatenate([T_tr_ for k in range(assets_tr_.shape[0])], axis = 0)
assets_tr_ = np.concatenate([np.tile(assets_tr_[k], (F_tr_.shape[0], 1)) for k in range(assets_tr_.shape[0])], axis = 0)
x_tr_      = np.concatenate([np.tile(x_tr_[k, :], (F_tr_.shape[0], 1)) for k in range(x_tr_.shape[0])], axis = 0)
F_tr_      = np.concatenate([F_tr_[..., k] for k in range(F_tr_.shape[2])], axis = 0)
E_tr_      = np.concatenate([E_tr_[..., k] for k in range(E_tr_.shape[2])], axis = 0)
#print(x_tr_.shape, assets_tr_.shape, F_tr_.shape, E_tr_.shape, T_tr_.shape)
#print(x_ts_.shape, assets_ts_.shape, F_ts_.shape, E_ts_.shape, T_ts_.shape)

t_tr_ = np.array([datetime.datetime.strptime(t_tr, "%Y-%m-%d %H:%M:%S").timetuple().tm_yday for t_tr in T_tr_[:, 0]]) - 1
t_ts_ = np.array([datetime.datetime.strptime(t_ts, "%Y-%m-%d %H:%M:%S").timetuple().tm_yday for t_ts in T_ts_[:, 0]]) - 1
#print(t_tr_.shape, t_ts_.shape)

# Calibration experiments setup
#times_  = [12*12]
times_  = [int(sys.argv[2])]

# Assets in the calibration experiments
assets_ = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

# Significance levels for the confidence intervals
alpha_ = [0.1, 0.2, 0.3, 0.4]

# Hyperparameters for the functional forecast dynamic update:
# Day-Ahead + Observations: Wind
# forget_rate_f_  = [0.0625]
# forget_rate_e_  = [1.]
# lookup_rate_    = [64.]
# length_scale_f_ = [0.05]   
# length_scale_e_ = [0.1]
# trust_rate_     = [0.1]
# nu_             = [4]
# xi_             = [0.9]
# gamma_          = [90]
# kappa_min_      = [100]
# kappa_max_      = [125]

# Day-Ahead + Observations: Solar
forget_rate_f_  = [0.0625]
forget_rate_e_  = [.5]
lookup_rate_    = [512.]
length_scale_f_ = [0.025]   
length_scale_e_ = [0.05]
trust_rate_     = [0.2]
nu_             = [10]
xi_             = [0.85]
gamma_          = [30]
kappa_min_      = [50]
kappa_max_      = [250]

# Observations only
# forget_rate_f_  = [2.]
# forget_rate_e_  = [100.]
# lookup_rate_    = [100.]
# length_scale_f_ = [0.001]   
# length_scale_e_ = [100]
# xi_             = [0.95]
# trust_rate_     = [0.]
# nu_             = [0.1]
# gamma_          = [105]
# kappa_min_      = [50]
# kappa_max_      = [1500]

# Day-Ahead only
# forget_rate_f_  = [100.]
# forget_rate_e_  = [0.25]
# lookup_rate_    = [2.]
# length_scale_f_ = [100]   
# length_scale_e_ = [.5]
# xi_             = [0.95]
# trust_rate_     = [1]
# nu_             = [0.1]
# gamma_          = [75]
# kappa_min_      = [50]
# kappa_max_      = [1250]

if sys.argv[1] == 'forget_rate_f':
    forget_rate_f_ = [0.0625, 0.125, 0.25, 0.5, 1., 2., 3., 4., 5., 6., 7., 8.]

if sys.argv[1] == 'forget_rate_e':
    forget_rate_e_ =  [0.25, 0.5, 1., 2., 4., 8., 16., 32., 64., 128., 256., 512.]

if sys.argv[1] == 'length_scale_f':
    length_scale_f_ = [0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5]

if sys.argv[1] == 'length_scale_e':
    length_scale_e_ = [0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5]

if sys.argv[1] == 'lookup_rate':
    lookup_rate_ = [0.5, 1., 2., 4., 8., 16., 32., 64., 128., 256., 512., 1028]
    #lookup_rate_ = [0.5, 1., 2., 3., 4., 5., 6., 7., 8., 10., 16., 20.]

if sys.argv[1] == 'trust_rate':
    trust_rate_ = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.]

if sys.argv[1] == 'nu':
    nu_ = [1., 2., 3, 4., 5, 6., 8., 10., 12., 14., 16., 18]

if sys.argv[1] == 'gamma':
    gamma_ = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

if sys.argv[1] == 'xi':
    xi_ = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]

if sys.argv[1] == 'kappa_min':
    kappa_min_ = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 375, 400]

if sys.argv[1] == 'kappa_max':
    kappa_max_ = [100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 750, 1000]
print(i_job, sys.argv[1], sys.argv[2], resource)

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
try:
    for time in times_:
        for asset in assets_:
            for day in range(363):
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

                _meta, M_ = _fknn_forecast_dynamic_update(F_tr_, E_tr_, x_tr_, t_tr_, dt_, f_, e_, x_, t,
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
                                                          kappa_max      = params_[10], 
                                                          idx_hours_     = idx_hours_)

                f_tau_rmse = np.sqrt(np.mean((f_ - e_[:time])**2))
                f_s_rmse   = np.sqrt(np.mean((np.median(M_, axis = 0) - e_[time:])**2))
        
                # Calculate functional confidence bands
                m_, _upper, _lower = _confidence_intervals_from_eCDF(M_, alpha_)

                WIS_f = np.mean(_weighted_empirical_interval_score(f_hat_, 
                                                                   m_, 
                                                                   _lower, 
                                                                   _upper, 
                                                                   alpha_))

                WIS_e = np.mean(_weighted_empirical_interval_score(e_[time:], 
                                                                   m_, 
                                                                   _lower, 
                                                                   _upper, 
                                                                   alpha_))

                # Save results
                dfs_.append(list(params_ + tuple([time, 
                                                asset, 
                                                day, 
                                                x_[0], 
                                                x_[1], 
                                                M_.shape[0], 
                                                float(WIS_e), 
                                                float(WIS_f), 
                                                float(f_tau_rmse), 
                                                float(f_s_rmse)])))
                
                #t2 = datetime.datetime.now()
                #print(t2 - t1)
except:
    print(params_, file_name)

print(i_job, sys.argv[1], sys.argv[2], resource, datetime.datetime.now())

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
                                     'lon',
                                     'lat',
                                     'n_scenarios', 
                                     'WIS_e', 
                                     'WIS_f', 
                                     'RMSE_tau', 
                                     'RMSE_s'])

dfs_ = _gather_node_data(_comm, dfs_)

_save_validation_csv(dfs_, 
                     path_to_file = path_to_validation + f'/validation_ffc-{resource}-WIS-{sys.argv[1]}.csv')
