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

path_to_fDepth = '/home/gterren/dynamic_update/functional_forecast_dynamic_update/fDepth'
path_to_data   = '/home/gterren/dynamic_update/data'
path_to_test   = '/home/gterren/dynamic_update/test'

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
print(assets_tr_.shape, x_tr_.shape, dates_tr_.shape, F_tr_.shape, E_tr_.shape)

# Reshape to day x interval x asset format
F_tr_ = F_tr_.reshape(int(F_tr_.shape[0]/T), T, F_tr_.shape[1])
E_tr_ = E_tr_.reshape(int(E_tr_.shape[0]/T), T, E_tr_.shape[1])
T_tr_ = dates_tr_.reshape(int(dates_tr_.shape[0]/T), T)
print(F_tr_.shape, E_tr_.shape, T_tr_.shape)

# Load 2018 data as testing set
with open(path_to_data + f"/preprocessed_{resource}_2018.pkl", 'rb') as f:
    _data = pkl.load(f)

assets_ts_ = _data["assets"]
x_ts_      = _data["locations"]
dates_ts_  = _data["dates"]
F_ts_      = _data["observations"]
E_ts_      = _data["forecasts"]
print(assets_ts_.shape, x_ts_.shape, dates_ts_.shape, F_ts_.shape, E_ts_.shape)

# Reshape to day x interval x asset format
F_ts_ = F_ts_.reshape(int(F_ts_.shape[0]/T), T, F_ts_.shape[1])
E_ts_ = E_ts_.reshape(int(E_ts_.shape[0]/T), T, E_ts_.shape[1])
T_ts_ = dates_ts_.reshape(int(dates_ts_.shape[0]/T), T)
print(F_ts_.shape, E_ts_.shape, T_ts_.shape)

dt_ = np.array([t * 5 for t in range(T)])
dx_ = pd.to_datetime(pd.DataFrame({"time": dt_}).time, unit = "m").dt.strftime("%H:%M").to_numpy()
print(dt_.shape, dx_.shape)

# Filter solar hours with loading solar set
idx_ = (np.sum(np.sum(F_tr_, axis = 0), axis = 1) 
        + np.sum(np.sum(F_ts_, axis = 0), axis = 1)
        + np.sum(np.sum(E_tr_, axis = 0), axis = 1)
        + np.sum(np.sum(E_ts_, axis = 0), axis = 1)) > 0.

F_tr_ = F_tr_[:, idx_, :]
E_tr_ = E_tr_[:, idx_, :]
T_tr_ = T_tr_[:, idx_]
print(F_tr_.shape, E_tr_.shape, T_tr_.shape)

F_ts_ = F_ts_[:, idx_, :]
E_ts_ = E_ts_[:, idx_, :]
T_ts_ = T_ts_[:, idx_]
print(F_ts_.shape, E_ts_.shape, T_ts_.shape)

# Short testing set with training set order
order      = {v: i for i, v in enumerate(assets_tr_)}
idx_       = np.argsort([order[x] for x in assets_ts_])
assets_ts_ = assets_ts_[idx_]
x_ts_      = x_ts_[idx_]
F_ts_      = F_ts_[:, :, idx_]
E_ts_      = E_ts_[:, :, idx_]
print(F_ts_.shape, E_ts_.shape, T_ts_.shape)

# From generation to capacity factor
p_tr_ = np.max(np.max(F_tr_, axis = 0), axis = 0)
p_ts_ = np.max(np.max(F_ts_, axis = 0), axis = 0)
print(p_tr_.shape, p_ts_.shape)

F_tr_ /= np.tile(p_tr_, (F_tr_.shape[0], F_tr_.shape[1], 1))
E_tr_ /= np.tile(p_tr_, (E_tr_.shape[0], E_tr_.shape[1], 1))
F_ts_ /= np.tile(p_ts_, (F_ts_.shape[0], F_ts_.shape[1], 1))
E_ts_ /= np.tile(p_ts_, (E_ts_.shape[0], E_ts_.shape[1], 1))
print(F_tr_.min(), F_tr_.max())
print(E_tr_.min(), E_tr_.max())
print(F_ts_.min(), F_ts_.max())
print(E_ts_.min(), E_ts_.max())

# No possible a capacity factor is larger than 1 or smaller than 0
F_tr_[F_tr_ > 1.] = 1.
F_tr_[F_tr_ < 0.] = 0.
F_ts_[F_ts_ > 1.] = 1.
F_ts_[F_ts_ < 0.] = 0.
E_tr_[E_tr_ > 1.] = 1.
E_tr_[E_tr_ < 0.] = 0.
E_ts_[E_ts_ > 1.] = 1.
E_ts_[E_ts_ < 0.] = 0.

print(assets_tr_.shape, T_tr_.shape)
# Format training set from day x interval x asset to [day * asset] x interval
T_tr_      = np.concatenate([T_tr_ for k in range(assets_tr_.shape[0])], axis = 0)
assets_tr_ = np.concatenate([np.tile(assets_tr_[k], (F_tr_.shape[0], 1)) for k in range(assets_tr_.shape[0])], axis = 0)
x_tr_      = np.concatenate([np.tile(x_tr_[k, :], (F_tr_.shape[0], 1)) for k in range(x_tr_.shape[0])], axis = 0)
F_tr_      = np.concatenate([F_tr_[..., k] for k in range(F_tr_.shape[2])], axis = 0)
E_tr_      = np.concatenate([E_tr_[..., k] for k in range(E_tr_.shape[2])], axis = 0)
print(x_tr_.shape, assets_tr_.shape, F_tr_.shape, E_tr_.shape, T_tr_.shape)
print(x_ts_.shape, assets_ts_.shape, F_ts_.shape, E_ts_.shape, T_ts_.shape)

t_tr_ = np.array([datetime.datetime.strptime(t_tr, "%Y-%m-%d %H:%M:%S").timetuple().tm_yday for t_tr in T_tr_[:, 0]]) - 1
t_ts_ = np.array([datetime.datetime.strptime(t_ts, "%Y-%m-%d %H:%M:%S").timetuple().tm_yday for t_ts in T_ts_[:, 0]]) - 1
print(t_tr_.shape, t_ts_.shape)

# Significance levels for confidence intervals
alpha_ = [0.1, 0.2, 0.3, 0.4]

# Hyperparameters for the functional forecast dynamic update:
forget_rate_f_  = [0.125]
forget_rate_e_  = [9.]
lookup_rate_    = [.5]
length_scale_f_ = [0.00025]   
length_scale_e_ = [0.05]
xi_             = [0.85]
trust_rate_     = [12]
nu_             = [12]
gamma_          = [45]
kappa_min_      = [500]
kappa_max_      = [1500]

# Observations only
# forget_rate_f_  = [0.3]
# forget_rate_e_  = [100.]
# lookup_rate_    = [100.]
# length_scale_f_ = [0.00075]   
# length_scale_e_ = [100]
# xi_             = [0.75]
# trust_rate_     = [1]
# nu_             = [14]
# gamma_          = [45]
# kappa_min_      = [300]
# kappa_max_      = [2250]

# Day-Ahead only
# forget_rate_f_  = [100.]
# forget_rate_e_  = [9.]
# lookup_rate_    = [0.5]
# length_scale_f_ = [100]   
# length_scale_e_ = [1.]
# xi_             = [0.5]
# trust_rate_     = [1]
# nu_             = [-1]
# gamma_          = [120]
# kappa_min_      = [500]
# kappa_max_      = [1500]

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
                        kappa_max_))[0]

assets_ = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
times_  = [72, 144, 216]

# Test setup
dfs_ = []
for day in range(363):
    for time in times_:
        asset = assets_[i_job]
        print(f'{asset}-{day}-{time}')

        # Get functional predictors for a given test
        f_     = F_ts_[day, :time, asset]
        e_     = E_ts_[day, :, asset]
        x_     = x_ts_[asset, :]
        t      = t_ts_[day]
        f_hat_ = F_ts_[day, time:, asset]

        # Get time constants
        tau_ = dt_[:time]
        s_   = dt_[time:]

        t1 = datetime.datetime.now()

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
                                                  kappa_max      = params_[10])

        t2 = datetime.datetime.now()
        print(t2 - t1)

        f_tau_rmse = np.sqrt(np.mean((f_ - e_[:time])**2))
        f_s_rmse   = np.sqrt(np.mean((np.median(M_, axis = 0) - e_[time:])**2))

        # Calculate marginal empirical confidence intervals
        m_, _upper, _lower = _confidence_intervals_from_eCDF(M_, alpha_)

        WIS_e = np.mean(_weighted_empirical_interval_score(e_[time:], 
                                                           m_, 
                                                           _lower, 
                                                           _upper, 
                                                           alpha_))
        
        # Testing WIS
        WIS_f = np.mean(_weighted_empirical_interval_score(f_hat_, 
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

_save_validation_csv(dfs_, path_to_file = path_to_test + f'/test-WIS.csv')