import os, datetime, sys, time, traceback

import pandas as pd
import numpy as np
import pickle as pkl

from itertools import product
from mpi4py import MPI

from ffc_utils import _fknn_forecast_dynamic_update


from functional_utils import _confidence_bands_from_eCDF

from scores_utils import (_empirical_coverage_score,
                          _weighted_empirical_interval_score)

path_to_fDepth = '/home/gterren/dynamic_update/functional_forecast_dynamic_update/fDepth'
path_to_data   = '/home/gterren/dynamic_update/data'
path_to_test   = '/home/gterren/dynamic_update/test'
path_to_param  = '/home/gterren/dynamic_update/params'

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

# Calibration experiments setup
resource = sys.argv[1]
method   = sys.argv[2] 
time     = int(sys.argv[3])

# Assets in the calibration experiments
assets_ = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
# Significance levels for the confidence intervals
alpha_ = [0.1, 0.2, 0.3, 0.4]

T = 288
# Load 2017 data as training set
with open(path_to_data + f"/linear_preprocessed_{resource}_2017.pkl", 'rb') as f:
    _data = pkl.load(f)

assets_tr_ = _data["assets"]
F_tr_      = _data["observations"]
E_tr_      = _data["forecasts"]
#print(assets_tr_.shape, F_tr_.shape, E_tr_.shape)

# Reshape to day x interval x asset format
F_tr_ = F_tr_.reshape(int(F_tr_.shape[0]/T), T, F_tr_.shape[1])
E_tr_ = E_tr_.reshape(int(E_tr_.shape[0]/T), T, E_tr_.shape[1])
#print(F_tr_.shape, E_tr_.shape)

# Load 2018 data as testing set
with open(path_to_data + f"/linear_preprocessed_{resource}_2018.pkl", 'rb') as f:
    _data = pkl.load(f)

assets_ts_ = _data["assets"]
F_ts_      = _data["observations"]
E_ts_      = _data["forecasts"]
#print(assets_ts_.shape, F_ts_.shape, E_ts_.shape)

# Reshape to day x interval x asset format
F_ts_ = F_ts_.reshape(int(F_ts_.shape[0]/T), T, F_ts_.shape[1])
E_ts_ = E_ts_.reshape(int(E_ts_.shape[0]/T), T, E_ts_.shape[1])
#print(F_ts_.shape, E_ts_.shape)

# Short testing set with training set order
order  = {v: i for i, v in enumerate(assets_tr_)}
idx_   = np.argsort([order[x] for x in assets_ts_])
F_ts_  = F_ts_[:, :, idx_]
E_ts_  = E_ts_[:, :, idx_]
#print(F_ts_.shape, E_ts_.shape)

# From generation to capacity factor
p_tr_ = np.max(np.max(F_tr_, axis = 0), axis = 0)
p_ts_ = np.max(np.max(F_ts_, axis = 0), axis = 0)
# print(p_tr_.shape, p_ts_.shape)

E_tr_ /= np.tile(p_tr_, (E_tr_.shape[0], E_tr_.shape[1], 1))
E_ts_ /= np.tile(p_ts_, (E_ts_.shape[0], E_ts_.shape[1], 1))
# print(E_tr_.min(), E_tr_.max())
# print(E_ts_.min(), E_ts_.max())

# No possible a capacity factor is larger than 1 or smaller than 0
E_tr_[E_tr_ > 1.] = 1.
E_tr_[E_tr_ < 0.] = 0.
E_ts_[E_ts_ > 1.] = 1.
E_ts_[E_ts_ < 0.] = 0.

# Format training set from day x interval x asset to [day * asset] x interval
E_ts_lin_ = E_ts_.copy()
E_tr_lin_ = np.concatenate([E_tr_[..., k] 
                            for k in range(E_tr_.shape[2])], axis = 0)
#print(E_tr_lin_.shape, E_ts_lin_.shape)

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

# Format training set from day x interval x asset to [day * asset] x interval
T_tr_ = np.concatenate([T_tr_ for k in range(assets_tr_.shape[0])], axis = 0)
assets_tr_ = np.concatenate([np.tile(assets_tr_[k], (F_tr_.shape[0], 1)) for k in range(assets_tr_.shape[0])], axis = 0)
x_tr_ = np.concatenate([np.tile(x_tr_[k, :], (F_tr_.shape[0], 1)) for k in range(x_tr_.shape[0])], axis = 0)
F_tr_ = np.concatenate([F_tr_[..., k] for k in range(F_tr_.shape[2])], axis = 0)
E_tr_ = np.concatenate([E_tr_[..., k] for k in range(E_tr_.shape[2])], axis = 0)
#print(x_tr_.shape, assets_tr_.shape, F_tr_.shape, E_tr_.shape, T_tr_.shape)
#print(x_ts_.shape, assets_ts_.shape, F_ts_.shape, E_ts_.shape, T_ts_.shape)

t_tr_ = np.array([datetime.datetime.strptime(t_tr, "%Y-%m-%d %H:%M:%S").timetuple().tm_yday for t_tr in T_tr_[:, 0]]) - 1
t_ts_ = np.array([datetime.datetime.strptime(t_ts, "%Y-%m-%d %H:%M:%S").timetuple().tm_yday for t_ts in T_ts_[:, 0]]) - 1
#print(t_tr_.shape, t_ts_.shape)

hyper_         = pd.read_csv(path_to_param + f'/{resource}-{method}-params_init.csv')
hyper_         = hyper_.set_index("parameter")
hyper_.columns = hyper_.columns.astype(int)
print(hyper_)

# Test setup
dfs_ = []
asset = assets_[i_job]
for day in range(363):
    file_name = f'{asset}-{day}-{time}'
    print(i_job, file_name)

    try:

        # Get functional predictors for a given test
        f_     = F_ts_[day, :time, asset]
        e_lin_ = E_ts_lin_[day, :, asset]
        e_     = E_ts_[day, :, asset]
        x_     = x_ts_[asset, :]
        t      = t_ts_[day]
        f_hat_ = F_ts_[day, time:, asset]

        # Get time constants
        tau_ = dt_[:time]
        s_   = dt_[time:]

        t1 = datetime.datetime.now()

        # Filter solar hours with loading solar set
        idx_days_  = np.absolute(t_tr_ - day) < 7
        idx_hours_ = (np.sum(F_tr_[idx_days_, :], axis = 0) 
                    + np.sum(E_tr_[idx_days_, :], axis = 0)) > 1.
        
        _meta, M_ = _fknn_forecast_dynamic_update(F_tr_, E_tr_lin_, x_tr_, t_tr_, dt_, f_, e_lin_, x_, t,
                                                forget_rate_f  = hyper_.loc['forget_rate_f'][time],
                                                forget_rate_e  = hyper_.loc['forget_rate_e'][time],
                                                length_scale_f = hyper_.loc['length_scale_f'][time],
                                                length_scale_e = hyper_.loc['length_scale_e'][time],
                                                lookup_rate    = hyper_.loc['lookup_rate'][time],
                                                trust_rate     = hyper_.loc['trust_rate'][time],
                                                nu             = hyper_.loc['nu'][time],
                                                gamma          = hyper_.loc['gamma'][time],
                                                xi             = hyper_.loc['xi'][time],
                                                kappa_min      = hyper_.loc['kappa_min'][time],
                                                kappa_max      = hyper_.loc['kappa_max'][time], 
                                                idx_hours_     = idx_hours_)

        f_tau_rmse = np.sqrt(np.mean((f_ - e_[:time])**2))
        f_s_rmse   = np.sqrt(np.mean((np.median(M_, axis = 0) - e_[time:])**2))

        # Confidence bands from marginal empirical density function
        m_, _upper, _lower = _confidence_bands_from_eCDF(M_, alpha_)

        # Testing WIS
        WIS_e = np.mean(_weighted_empirical_interval_score(e_[time:], m_, _lower, _upper, alpha_))
        WIS_f = np.mean(_weighted_empirical_interval_score(f_hat_, m_, _lower, _upper, alpha_))

        # Save results
        dfs_.append([time, 
                    asset, 
                    day, 
                    x_[0], 
                    x_[1], 
                    M_.shape[0], 
                    float(WIS_e), 
                    float(WIS_f), 
                    float(f_tau_rmse), 
                    float(f_s_rmse)])
        
    except Exception as e:
        print(f"Error for asset={asset}, day={day}, file={file_name}")
        print(f"Exception: {e}")
        traceback.print_exc()
        # loop continues automatically

dfs_ = pd.DataFrame(dfs_, columns = ['time', 
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

_save_validation_csv(dfs_, path_to_file = path_to_test + f'/{resource}-{method}-test_ffc.csv')