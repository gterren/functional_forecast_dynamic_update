import os, datetime, sys, time, pickle

sys.path.append('/home/gterren/dynamic_update/functional_forecast_dynamic_update/')

import pandas as pd
import numpy as np
import pickle as pkl
import multiprocessing as mp

from mpi4py import MPI
from functools import partial
from skfda.exploratory.depth import ModifiedBandDepth
from datetime import datetime, timedelta

from src.fdu import functional_dynamic_update

from src.utils import (_KS, 
                       _weighted_empirical_interval_score, 
                       _empirical_coverage_score,
                       _empirical_interval_score,
                       _empirical_PIT, 
                       _energy_score)

np.seterr(all='raise')

path_to_fDepth = '/home/gterren/dynamic_update/functional_forecast_dynamic_update/fDepth'
path_to_data = '/home/gterren/dynamic_update/data'
path_to_validation = '/home/gterren/dynamic_update/validation'
path_to_param = '/home/gterren/dynamic_update/params'

def _run_ffc(process_, _DATA, hyper_, time, 
             parameter = None, 
             value = None):

    asset, day = process_
    hyper_p_ = hyper_.copy()

    if (parameter is not None) and (value is not None):
        hyper_p_.loc[parameter, time] = value

    file_name = f'{asset}-{day}-{time}'
    #print(file_name)

    # Get data for this region
    F_tr_, F_ts_ = _DATA['ac']
    E_tr_bias_, E_ts_bias_ = _DATA['day-ahead_bias_fc']
    E_tr_lin_, E_ts_lin_ = _DATA['day-ahead_linear_fc']
    E_tr_, E_ts_ = _DATA['day-ahead_fc']
    t_tr_, t_ts_ = _DATA['dates']
    X_tr_, X_ts_ = _DATA['locations'] 
    dt_, dx_ = _DATA['grid']

    # Filter solar hours with loading solar set
    idx_days_ = np.absolute(t_tr_ - day) < 7
    idx_hours_ = (np.sum(F_tr_[idx_days_, :], axis = 0) + np.sum(E_tr_bias_[idx_days_, :], axis = 0)) > 1.

    _fdu = functional_dynamic_update({'temporal': 'seasonal_equinox', 
                                      'spatial': 'haversine'}, assets_ts_[asset], T_ts_[day, time])
    #print(_fdu.name, _fdu.date)

    _fdu.fit(F_tr_, E_tr_lin_, dt_, 
             X_ = X_tr_, 
             t_ = t_tr_, 
             interval_mask = idx_hours_)

    # Get functional predictors for a given test
    f_ = F_ts_[day, :time, asset]
    e_lin_ = E_ts_lin_[day, :, asset]
    x_ts_ = X_ts_[asset, :]
    t_ts = t_ts_[day]

    try:
        M_ = _fdu.predict(f_, e_lin_, x_ts_, t_ts,
                          forget_rate_f=hyper_p_.loc['forget_rate_f'][time],
                          forget_rate_e=hyper_p_.loc['forget_rate_e'][time],
                          length_scale_f=hyper_p_.loc['length_scale_f'][time],
                          length_scale_e=hyper_p_.loc['length_scale_e'][time],
                          lookup_rate=hyper_p_.loc['lookup_rate'][time],
                          trust_rate=hyper_p_.loc['trust_rate'][time],
                          nu=hyper_p_.loc['nu'][time],
                          gamma=hyper_p_.loc['gamma'][time],
                          xi=hyper_p_.loc['xi'][time],
                          kappa=hyper_p_.loc['kappa'][time],
                          p_fusion=hyper_p_.loc['p_fusion'][time])
        #print(_fdu.idx_neighbors_.shape, _fdu.idx_temporal_.shape, _fdu.idx_spatial_.shape)

        e_ = E_ts_[day, time:, asset]
        f_hat_ = F_ts_[day, time:, asset]

        # Confidence bands from marginal empirical density function
        f_median_, _upper, _lower = _fdu.ecdf_confidence_bands(M_, alpha_)

        # Scoring rules
        ES = _energy_score(M_, f_hat_)
        WIS = _weighted_empirical_interval_score(f_hat_, f_median_, _lower, _upper, alpha_).sum()
        PIT_ = _empirical_PIT(f_hat_, M_.T, seed = 1234)

        # Error metrics
        RMSE = np.sqrt(np.mean((_fdu.f_focal_ - e_)**2))
        MBE  = np.mean(f_hat_ - _fdu.f_focal_)

        STD_ = pd.DataFrame(np.std(M_, axis = 0)).T
        STD_.columns = [f'H{str(i+1).zfill(2)}' for i in range(len(STD_.columns))]
        STD_['time'] = time
        STD_['region'] = asset
        STD_['day'] = day

    except Exception as e:
        print(rank, file_name, parameter, value, e)
        return None, None, None

    # Collect scoring rules
    PSR_ = np.array([time, asset, day, parameter, value, WIS, RMSE, MBE, ES])

    return PSR_, PIT_, STD_

# Run FFC experiments in parallel with MPI
def _run_parallel_mpi(_DATA, hyper_, processes_, parameter, value, time):

    _func = partial(_run_ffc,
                    _DATA=_DATA,
                    hyper_=hyper_,
                    time=time,
                    parameter=parameter,
                    value=value)
    
    local_psr = []
    local_pit = []
    local_std = []

    local_processes_ = np.array_split(processes_, size)[rank] 
    for process_ in local_processes_:
        psr_list, pit_list, std_list = _func(process_) 
        
        if (psr_list is not None) and (pit_list is not None) and (std_list is not None):
            local_psr.append(psr_list)
            local_pit.append(pit_list)
            local_std.append(std_list)

    if (len(local_psr) / len(local_processes_)) > 0.5:
        local_psr = np.stack(local_psr)
        local_pit = np.stack(local_pit)
        local_std = pd.concat(local_std, axis = 0)

        return local_psr, local_pit, local_std
    else:
        return None, None, None

def _run_ffc_envelop(process_, _DATA, hyper_, time, distances_, fractions_, alpha_, k_):
    
    asset, day = process_
    hyper_p_ = hyper_.copy()

    file_name = f'{asset}-{day}-{time}'
    #print(file_name)

    # Get data for this region
    F_tr_, F_ts_ = _DATA['ac']
    E_tr_bias_, E_ts_bias_ = _DATA['day-ahead_bias_fc']
    E_tr_lin_, E_ts_lin_ = _DATA['day-ahead_linear_fc']
    E_tr_, E_ts_ = _DATA['day-ahead_fc']
    t_tr_, t_ts_ = _DATA['dates']
    X_tr_, X_ts_ = _DATA['locations'] 
    dt_, dx_ = _DATA['grid']

    # Filter solar hours with loading solar set
    idx_days_ = np.absolute(t_tr_ - day) < 7
    idx_hours_ = (np.sum(F_tr_[idx_days_, :], axis = 0) + np.sum(E_tr_bias_[idx_days_, :], axis = 0)) > 1.

    _fdu = functional_dynamic_update({'temporal': 'seasonal_equinox', 
                                      'spatial': 'haversine'}, assets_ts_[asset], T_ts_[day, time])
    #print(_fdu.name, _fdu.date)

    _fdu.fit(F_tr_, E_tr_lin_, dt_, 
             X_ = X_tr_, 
             t_ = t_tr_, 
             interval_mask = idx_hours_)

    # Get functional predictors for a given test
    f_ = F_ts_[day, :time, asset]
    e_lin_ = E_ts_lin_[day, :, asset]
    x_ts_ = X_ts_[asset, :]
    t_ts = t_ts_[day]

    try:    
        M_ = _fdu.predict(f_, e_lin_, x_ts_, t_ts,
                        forget_rate_f=hyper_p_.loc['forget_rate_f'][time],
                        forget_rate_e=hyper_p_.loc['forget_rate_e'][time],
                        length_scale_f=hyper_p_.loc['length_scale_f'][time],
                        length_scale_e=hyper_p_.loc['length_scale_e'][time],
                        lookup_rate=hyper_p_.loc['lookup_rate'][time],
                        trust_rate=hyper_p_.loc['trust_rate'][time],
                        nu=hyper_p_.loc['nu'][time],
                        gamma=hyper_p_.loc['gamma'][time],
                        xi=hyper_p_.loc['xi'][time],
                        kappa=hyper_p_.loc['kappa'][time],
                        p_fusion=hyper_p_.loc['p_fusion'][time])

        e_ = E_ts_[day, time:, asset]
        f_hat_ = F_ts_[day, time:, asset]

        M_int_, M_int_ds_ = _fdu.functional_downsampling(subsample = 12, 
                                                        n_basis = int(f_.shape[0]/8)) 

        _depth = ModifiedBandDepth()

        results_ = []
        for distance in distances_:
            
            if (distance == 'MBD'):

                for fraction in fractions_:
                    if fraction is not None: 
                        k_ = [fraction, fraction, fraction, fraction]

                    f_deepest_, _upper_depth, _lower_depth = _fdu.depth_confidence_bands(_depth, M_int_, alpha_, k_)

                    for alpha in alpha_:
                        FIS_depth = _empirical_interval_score(f_hat_, _lower_depth[f'{alpha}'][1:], _upper_depth[f'{alpha}'][1:], alpha).sum()
                        FCS_depth = _empirical_coverage_score(f_hat_, _lower_depth[f'{alpha}'][1:], _upper_depth[f'{alpha}'][1:])

                        # Save results
                        results_.append([time, asset, day, alpha, fraction, distance, M_.shape[0], M_.shape[0], FIS_depth, FCS_depth])

        
            if (distance == 'l2') or (distance == 'sup') or (distance == 'fknn'):

                J_ = _fdu.focal_curve_envelope(_depth, M_int_, distance, max_iter = 100)

                for fraction in fractions_:
                    if fraction is not None: 
                        k_ = [fraction, fraction, fraction, fraction]
                    
                    f_focal_, _upper_focal, _lower_focal = _fdu.focal_envelop_confidence_bands(alpha_, k_)

                    for alpha in alpha_:
                        FIS_focal = _empirical_interval_score(f_hat_, _lower_focal[f'{alpha}'][1:], _upper_focal[f'{alpha}'][1:], alpha).sum()
                        FCS_focal = _empirical_coverage_score(f_hat_, _lower_focal[f'{alpha}'][1:], _upper_focal[f'{alpha}'][1:])

                        results_.append([time, asset, day, alpha, fraction, distance, M_.shape[0], J_.shape[0], FIS_focal, FCS_focal])
        
        results_ = np.stack(results_)

    except Exception as e:
        print(rank, file_name, e)

        return None

    return results_

# Run FFC experiments in parallel with MPI
def _run_ffc_envelop_parallel_mpi(_DATA, hyper_, processes_, distances_, alpha_, time,
                                  fractions_ = [None],
                                  k_ = None):

    _func = partial(_run_ffc_envelop,
                    _DATA=_DATA,
                    hyper_=hyper_,
                    distances_=distances_,
                    fractions_=fractions_, 
                    alpha_=alpha_, 
                    k_=k_,
                    time=time)

    local_results = []

    local_processes_ = np.array_split(processes_, size)[rank] 
    for process_ in local_processes_:
        results_list = _func(process_) 
        
        if (results_list is not None) and (results_list is not None):
            local_results.append(results_list)
    
    if (len(local_results) / len(local_processes_)) > 0.5:

        local_results = np.concatenate(local_results, axis = 0)

        return local_results
    else:
        return None
    
# Find optimal value
def _truncated_quadratic_min(x_, y_):
    
    # Fit quadratic
    a, b, c = np.polyfit(x_, y_, deg=2)
        
    # Check vertex
    if a != 0:
        x_star = -b / (2 * a)
    else:
        x_star = None

    # Use quadratic minimum if valid
    if (a > 0) and (x_star is not None) and (x_.min() <= x_star <= x_.max()):
        return x_star, (a * x_star**2) + (b * x_star) + c
    
    # Otherwise fallback to numerical minimum
    return x_[y_.argmin()], y_.min()

# Significance levels for the confidence intervals
alpha_ = [0.1, 0.2, 0.3, 0.4]
# Number of interation
N_iter = 36

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(f'[Process {rank}-{size}]')

# Calibration experiments setup
resource = sys.argv[1]
method = sys.argv[2] 
time = int(sys.argv[3])
init = int(sys.argv[4])
description = 'bias'

comm.Barrier()

# Hyperparameters for the functional forecast dynamic update:
_params = {
    'forget_rate_f': [0.0625, 0.125, 0.25, 0.5, 1., 2.],
    'forget_rate_e': [0.25, 0.5, 1., 2., 4., 8., 16.],
    'length_scale_f':[0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25],
    'length_scale_e': [0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.],
    'lookup_rate': [0.5, 1., 2., 4., 6., 8., 10., 12., 24.],
    'trust_rate': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
    'nu': [2., 3., 4., 5., 6., 7., 8., 9., 10.],
    'gamma': [30, 35, 40, 45, 50, 55, 60],
    'xi':[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
    'kappa': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
    'p_fusion': [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
}

if method == 'fusion':
    parameters_val_ = [
        'forget_rate_f', 
        'lookup_rate', 
        'length_scale_f', 
        'length_scale_e', 
        'trust_rate',
        'nu', 
        'gamma',
        'kappa', 
        'xi',  
        'p_fusion'
    ]

## -------------------------- LOAD DATA ---------------------------------
T = 288
# Load 2017 data as training set
with open(path_to_data + f"/preprocessed_{resource}_2017.pkl", 'rb') as f:
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
with open(path_to_data + f"/preprocessed_{resource}_2018.pkl", 'rb') as f:
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
#print(p_tr_.shape, p_ts_.shape)

F_tr_ /= np.tile(p_tr_, (F_tr_.shape[0], F_tr_.shape[1], 1))
F_ts_ /= np.tile(p_ts_, (F_ts_.shape[0], F_ts_.shape[1], 1))
E_tr_ /= np.tile(p_tr_, (E_tr_.shape[0], E_tr_.shape[1], 1))
E_ts_ /= np.tile(p_ts_, (E_ts_.shape[0], E_ts_.shape[1], 1))

# No possble a capacity factor is larger than 1 or smaller than 0
F_tr_[F_tr_ > 1.] = 1.
F_ts_[F_ts_ > 1.] = 1.
E_tr_[E_tr_ > 1.] = 1.
E_ts_[E_ts_ > 1.] = 1.
#print(F_tr_.max(), E_tr_.max())
#print(F_ts_.max(), F_ts_.max())

F_tr_[F_tr_ < 0.] = 0.
F_ts_[F_ts_ < 0.] = 0.
E_tr_[E_tr_ < 0.] = 0.
E_ts_[E_ts_ < 0.] = 0.
#print(F_tr_.min(), E_tr_.min())
#print(F_ts_.min(), E_ts_.min())

F_tr_ /= F_tr_.max()
F_ts_ /= F_ts_.max()
E_tr_ /= E_tr_.max()
E_ts_ /= E_ts_.max()

# Format training set from day x interval x asset to [day * asset] x interval
E_ts_bias_ = E_ts_.copy()
E_tr_bias_ = np.concatenate([E_tr_[..., k] for k in range(E_tr_.shape[2])], axis = 0)
#print(E_tr_bias_.shape, E_ts_bias_.shape)

## LOAD UNBIASED DATA WITH LINEAR INTERPOLATION
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

# # Unbias training set
# B_tr_ = np.mean(F_tr_ - E_tr_, axis = 0)
# for i in range(E_tr_.shape[-1]):
#     E_tr_[:, :, i] += B_tr_[:, i]

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

# # Unbias testing set
# B_ts_ = np.mean(F_ts_ - E_ts_, axis = 0)
# for i in range(E_ts_.shape[-1]):
#     E_ts_[:, :, i] += B_ts_[:, i]

# From generation to capacity factor
p_tr_ = np.max(np.max(F_tr_, axis = 0), axis = 0)
p_ts_ = np.max(np.max(F_ts_, axis = 0), axis = 0)
#print(p_tr_.shape, p_ts_.shape)

F_tr_ /= np.tile(p_tr_, (F_tr_.shape[0], F_tr_.shape[1], 1))
F_ts_ /= np.tile(p_ts_, (F_ts_.shape[0], F_ts_.shape[1], 1))
E_tr_ /= np.tile(p_tr_, (E_tr_.shape[0], E_tr_.shape[1], 1))
E_ts_ /= np.tile(p_ts_, (E_ts_.shape[0], E_ts_.shape[1], 1))

# No possble a capacity factor is larger than 1 or smaller than 0
F_tr_[F_tr_ > 1.] = 1.
F_ts_[F_ts_ > 1.] = 1.
E_tr_[E_tr_ > 1.] = 1.
E_ts_[E_ts_ > 1.] = 1.
#print(F_tr_.max(), E_tr_.max())
#print(F_ts_.max(), F_ts_.max())

F_tr_[F_tr_ < 0.] = 0.
F_ts_[F_ts_ < 0.] = 0.
E_tr_[E_tr_ < 0.] = 0.
E_ts_[E_ts_ < 0.] = 0.
#print(F_tr_.min(), E_tr_.min())
#print(F_ts_.min(), E_ts_.min())

F_tr_ /= F_tr_.max()
F_ts_ /= F_ts_.max()
E_tr_ /= E_tr_.max()
E_ts_ /= E_ts_.max()

# Format training set from day x interval x asset to [day * asset] x interval
E_ts_lin_ = E_ts_.copy()
E_tr_lin_ = np.concatenate([E_tr_[..., k] for k in range(E_tr_.shape[2])], axis = 0)
#print(E_ts_lin_.shape, E_tr_lin_.shape)

## LOAD UNBISED DATA
# Load 2017 data as training set
with open(path_to_data + f"/preprocessed_{resource}_2017.pkl", 'rb') as f:
    _data = pkl.load(f)

assets_tr_ = _data["assets"]
X_tr_      = _data["locations"]
dates_tr_  = _data["dates"]
F_tr_      = _data["observations"]
E_tr_      = _data["forecasts"]
#print(assets_tr_.shape, X_tr_.shape, dates_tr_.shape, F_tr_.shape, E_tr_.shape)

# Reshape to day x interval x asset format
F_tr_ = F_tr_.reshape(int(F_tr_.shape[0]/T), T, F_tr_.shape[1])
E_tr_ = E_tr_.reshape(int(E_tr_.shape[0]/T), T, E_tr_.shape[1])
T_tr_ = dates_tr_.reshape(int(dates_tr_.shape[0]/T), T)
#print(F_tr_.shape, E_tr_.shape, T_tr_.shape)

# # Unbias training set
# B_tr_ = np.mean(F_tr_ - E_tr_, axis = 0)
# for i in range(E_tr_.shape[-1]):
#     E_tr_[:, :, i] += B_tr_[:, i]

# Load 2018 data as testing set
with open(path_to_data + f"/preprocessed_{resource}_2018.pkl", 'rb') as f:
    _data = pkl.load(f)

assets_ts_ = _data["assets"]
X_ts_      = _data["locations"]
dates_ts_  = _data["dates"]
F_ts_      = _data["observations"]
E_ts_      = _data["forecasts"]
#print(assets_ts_.shape, X_ts_.shape, dates_ts_.shape, F_ts_.shape, E_ts_.shape)

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
X_ts_      = X_ts_[idx_]
F_ts_      = F_ts_[:, :, idx_]
E_ts_      = E_ts_[:, :, idx_]
#print(F_ts_.shape, E_ts_.shape, T_ts_.shape)

# # Unbias testing set
# B_ts_ = np.mean(F_ts_ - E_ts_, axis = 0)
# for i in range(E_ts_.shape[-1]):
#     E_ts_[:, :, i] += B_ts_[:, i]

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
F_ts_[F_ts_ > 1.] = 1.
E_tr_[E_tr_ > 1.] = 1.
E_ts_[E_ts_ > 1.] = 1.
#print(F_tr_.max(), E_tr_.max())
#print(F_ts_.max(), F_ts_.max())

F_tr_[F_tr_ < 0.] = 0.
F_ts_[F_ts_ < 0.] = 0.
E_tr_[E_tr_ < 0.] = 0.
E_ts_[E_ts_ < 0.] = 0.
#print(F_tr_.min(), E_tr_.min())
#print(F_ts_.min(), E_ts_.min())

F_tr_ /= F_tr_.max()
F_ts_ /= F_ts_.max()
E_tr_ /= E_tr_.max()
E_ts_ /= E_ts_.max()

# Format training set from day x interval x asset to [day * asset] x interval
T_tr_ = np.concatenate([T_tr_ for k in range(assets_tr_.shape[0])], axis = 0)
assets_tr_ = np.concatenate([np.tile(assets_tr_[k], (F_tr_.shape[0], 1)) for k in range(assets_tr_.shape[0])], axis = 0)
X_tr_ = np.concatenate([np.tile(X_tr_[k, :], (F_tr_.shape[0], 1)) for k in range(X_tr_.shape[0])], axis = 0)
F_tr_ = np.concatenate([F_tr_[..., k] for k in range(F_tr_.shape[2])], axis = 0)
E_tr_ = np.concatenate([E_tr_[..., k] for k in range(E_tr_.shape[2])], axis = 0)
#print(X_tr_.shape, assets_tr_.shape, F_tr_.shape, E_tr_.shape, T_tr_.shape)
#print(X_ts_.shape, assets_ts_.shape, F_ts_.shape, E_ts_.shape, T_ts_.shape)

t_tr_ = np.array([datetime.strptime(t_tr, "%Y-%m-%d %H:%M:%S").timetuple().tm_yday for t_tr in T_tr_[:, 0]]) - 1
t_ts_ = np.array([datetime.strptime(t_ts, "%Y-%m-%d %H:%M:%S").timetuple().tm_yday for t_ts in T_ts_[:, 0]]) - 1
#print(t_tr_.shape, t_ts_.shape)

if rank == 0: 
    print(f'[Resource {resource}][Method {method}][Horizon {time}h][Initialization {init}]')
    print('----- HYPERPARAMETERS VALIDATION ----')

# Load dataset
_DATA = {
    'ac': [F_tr_, F_ts_],
    'day-ahead_bias_fc': [E_tr_bias_, E_ts_bias_],
    'day-ahead_linear_fc': [E_tr_lin_, E_ts_lin_],
    'day-ahead_fc': [E_tr_, E_ts_],
    'dates': [t_tr_, t_ts_],
    'locations': [X_tr_, X_ts_],
    'grid': [dt_, dx_],
}
# Load initial hyperparameters
hyper_ = pd.read_csv(path_to_param + f'/{resource}/{resource}-{method}-hyper-sa.csv')
hyper_ = hyper_.set_index("parameter")
hyper_.columns = hyper_.columns.astype(int)

processes_val_ = [(asset, j) for asset in range(0, 10) for j in range(0, 360)]
processes_test_ = [(asset, j) for asset in range(10, 20) for j in range(0, 360)]

all_PSR_ = []
all_KS_ = []
mins_ = []

for iter in range(N_iter):

    KS_ = []
    df_PSR_ = []
    df_KS_ = []
    values_ = []
    
    # Broadcast inputs (only needed if not already shared)
    if rank == 0:
        print(hyper_[time])
        parameter = np.random.choice(parameters_val_)
        hyper_shared = hyper_.copy()

    else:
        parameter = None
        hyper_shared = None

    # Broadcast parameter and values
    parameter = comm.bcast(parameter, root=0)
    hyper_shared = comm.bcast(hyper_shared, root=0)

    for i, value in enumerate(_params[parameter]):
        #print(rank, parameter, value)
        if rank == 0:
            print(f'[Iteration {iter + 1}-{N_iter}][Parameter {i + 1}-{len(_params[parameter])}][{parameter}={value}]')

        # Run FFC experiments in parallel
        local_psr, local_pit, local_std = _run_parallel_mpi(_DATA, hyper_shared, 
                                                 processes_=processes_val_, 
                                                 parameter=parameter,
                                                 value=value,
                                                 time=time)

        # if any rank failed → skip this value everywhere
        local_failed = (local_psr is None) or (local_pit is None) or (local_std is None)

        any_failed = comm.allreduce(local_failed, op=MPI.LOR)

        if any_failed:
            if rank == 0:
                print(f"Skipping {parameter}={value}")
            continue

        # Gather all local results
        all_psr = comm.gather(local_psr, root=0)
        all_pit = comm.gather(local_pit, root=0)

        if rank == 0:
            try:
                # Only root processes all gathered result
                PSR_ = np.concatenate(all_psr, axis=0)
                PIT_ = np.concatenate(all_pit, axis=0)
            
                # Calculate aggregated PIT
                ks_ = [_KS(PIT_[:, j:(j + 12)].flatten()) for j in [0, 12, 24, 36, 48]]
                KS_.append(np.array(ks_).mean())
                #KS_.append(np.array(ks_).max())
                values_.append(value)

                val_metrics = {'ES': np.median(PSR_[:, -1].astype(float)),
                               'WIS': np.median(PSR_[:, -4].astype(float)),
                               'RMSE': np.median(PSR_[:, -3].astype(float)),
                               'MBE': np.median(PSR_[:, -2].astype(float)),
                               'KS1': ks_[0],
                               'KS2': ks_[1],
                               'KS3': ks_[2],
                               'KS4': ks_[3],
                               'KS5': ks_[4]}
                
                # Collect proper scoring rules
                df_PSR_.append(PSR_)

                # Collect KS scores
                df_KS_.append([time, parameter, value] + ks_)

            except FloatingPointError as e:
                print(f"    [Rank {rank}] KS ERROR at value={value}: {e}")
                print("     Skipping parameter due to KS failure")
                continue

    if rank == 0:

        if len(KS_) == 0:
            print(f"No successful runs for parameter {parameter}")
        else:
            # Find optimal parameter
            y_ = np.array(KS_)
            x_ = np.array(values_)
            print(x_.shape, y_.shape)

            hyper_.loc[parameter, time] = _truncated_quadratic_min(x_, y_)[0]
            #hyper_.loc[parameter, time] = x_[y_.argmin()]
            mins_.append(y_.min())
            print(y_.min(), x_[y_.argmin()])

            # Proprer scores and metrics to dataframe
            df_PSR_ = pd.DataFrame(np.concatenate(df_PSR_, axis = 0), columns = ['time', 
                                                                                'region', 
                                                                                'day', 
                                                                                'parameter',
                                                                                'value',
                                                                                'WIS', 
                                                                                'RMSE', 
                                                                                'MBE', 
                                                                                'ES'])

            # KS scores to dataframe
            df_KS_ = pd.DataFrame(df_KS_, columns = ['time', 
                                                    'parameter',
                                                    'value',
                                                    'KS1',
                                                    'KS2',
                                                    'KS3',
                                                    'KS4',
                                                    'KS5'])
            
            df_PSR_['iteration'] = iter
            df_KS_['iteration'] = iter

            all_PSR_.append(df_PSR_)
            all_KS_.append(df_KS_)

    comm.Barrier()

if rank == 0:
    mins_ = np.array(mins_)
    print(mins_)

    # Collect results
    all_PSR_ = pd.concat(all_PSR_, axis = 0)
    all_KS_ = pd.concat(all_KS_, axis = 0)

    # Make sure results are robust
    print(all_PSR_['WIS'].isna().sum())
    print(all_PSR_['ES'].isna().sum())
    print(all_PSR_['RMSE'].isna().sum())
    print(all_PSR_['MBE'].isna().sum())
    all_PSR_['WIS'] = pd.to_numeric(all_PSR_['WIS'], errors='coerce')
    all_PSR_['ES'] = pd.to_numeric(all_PSR_['ES'], errors='coerce')
    all_PSR_['RMSE'] = pd.to_numeric(all_PSR_['RMSE'], errors='coerce')
    all_PSR_['MBE'] = pd.to_numeric(all_PSR_['MBE'], errors='coerce')
    all_PSR_['time'] = pd.to_numeric(all_PSR_['time'], errors='coerce')
    all_PSR_['value'] = pd.to_numeric(all_PSR_['value'], errors='coerce')
    all_KS_['time'] = pd.to_numeric(all_KS_['time'], errors='coerce')
    all_KS_['value'] = pd.to_numeric(all_KS_['value'], errors='coerce')

    # Aggregate score across samples
    all_PSR_ = all_PSR_.groupby(['time',
                                 'iteration',
                                 'parameter', 
                                 'value']).agg({'WIS': 'median',
                                                'ES': 'median',
                                                'RMSE': 'median',
                                                'MBE': 'median'}).reset_index(drop = False)

    # Caculate average KS score across intervals
    all_KS_['KS'] = (all_KS_['KS1'] + all_KS_['KS2'] + all_KS_['KS3'] + all_KS_['KS4'] + all_KS_['KS5'])/5.

    # Merge dataframes to have a single scoring rules dataframe
    scores_ = all_PSR_.merge(all_KS_,
                             on=['iteration', "parameter", "value", "time"],
                             how="left")

    scores_['resource'] = resource
    scores_['method'] = method

    # Overwrite the CSV with the updated data
    # scores_.to_csv(path_to_validation + f'/{resource}/{resource}-{method}-zone-validation-{time}-0.csv', index = False)
    # hyper_.to_csv(path_to_param + f'/{resource}/{resource}-{method}-zone-hyper-{time}-0.csv', index = False)

# Broadcast inputs (only needed if not already shared)
if rank == 0:
    parameters_final = hyper_[time].to_dict()
    print('----- HYPERPARAMETERS TEST ----')
    print(hyper_)
    hyper_shared = hyper_
else:
    hyper_shared = None

# Broadcast parameter and values
hyper_shared = comm.bcast(hyper_shared, root=0)

# Run FFC experiments in parallel
local_psr, local_pit, local_std = _run_parallel_mpi(_DATA, hyper_shared, 
                                         processes_ = processes_test_, 
                                         parameter = parameter,
                                         value = value,
                                         time = time)

# Gather all local results
all_psr = comm.gather(local_psr, root=0)
all_pit = comm.gather(local_pit, root=0)
all_std = comm.gather(local_std, root=0)

if rank == 0:
    # Only root processes all gathered result
    PSR_ = np.concatenate(all_psr, axis=0)
    PIT_ = np.concatenate(all_pit, axis=0)
    STD_ = pd.concat(all_std, axis=0)
    print(PSR_.shape, PIT_.shape, STD_.shape)

    # Calculate aggregated PIT
    ks_ = np.array([_KS(PIT_[:, j:(j + 12)].flatten()) for j in [0, 12, 24, 36, 48]])

    # Collect KS scores
    print(ks_, ks_.max(), ks_.mean())

    # Proper scores and metrics 
    test_metrics = {
        'ES': np.median(PSR_[:, -1].astype(float)),
        'WIS': np.median(PSR_[:, -4].astype(float)),
        'RMSE': np.median(PSR_[:, -3].astype(float)),
        'MBE': np.median(PSR_[:, -2].astype(float)),
        'KS1': ks_[0],
        'KS2': ks_[1],
        'KS3': ks_[2],
        'KS4': ks_[3],
        'KS5': ks_[4],
    }

    # validation metrics already computed earlier → make sure you keep them
    # (you already defined val_metrics inside loop; keep last one or compute properly)
    row = {
        'initialization': init,
        'time': time,

        # validation
        'ES_val': val_metrics['ES'],
        'WIS_val': val_metrics['WIS'],
        'RMSE_val': val_metrics['RMSE'],
        'MBE_val': val_metrics['MBE'],
        'KS1_val': val_metrics['KS1'],
        'KS2_val': val_metrics['KS2'],
        'KS3_val': val_metrics['KS3'],
        'KS4_val': val_metrics['KS4'],
        'KS5_val': val_metrics['KS5'],

        # test
        'ES_test': test_metrics['ES'],
        'WIS_test': test_metrics['WIS'],
        'RMSE_test': test_metrics['RMSE'],
        'MBE_test': test_metrics['MBE'],
        'KS1_test': test_metrics['KS1'],
        'KS2_test': test_metrics['KS2'],
        'KS3_test': test_metrics['KS3'],
        'KS4_test': test_metrics['KS4'],
        'KS5_test': test_metrics['KS5'],

        # parameters
        **parameters_final
    }

    # define file path
    results_path = path_to_validation + f'/{resource}/{resource}_{method}_asset-hyper-{description}.csv'

    # load or create DataFrame
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
    else:
        results_df = pd.DataFrame()

    # append safely
    results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    print(results_df)

    # save
    results_df.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")

    # define PIT file path
    pit_path = path_to_validation + f'/{resource}/{resource}_{method}_asset-PIT_{time}-{description}.csv'

    # load or create DataFrame
    if os.path.exists(pit_path):
        pit_df = pd.read_csv(pit_path)
    else:
        pit_df = pd.DataFrame()

    # save PIT
    PIT_ = pd.DataFrame(PIT_)
    PIT_.columns = [f'H{str(i+1).zfill(2)}' for i in range(len(PIT_.columns))]
    PIT_['initialization'] = init

    pit_df = pd.concat([pit_df, PIT_], axis = 0, ignore_index=True)
    print(pit_df)

    pit_df.to_csv(pit_path, index = False)
    print(f"Saved PIT to {pit_path}")
    
    # define PIT file path
    std_path = path_to_validation + f'/{resource}/{resource}_{method}_asset-STD_{time}-{description}.csv'

    # load or create DataFrame
    if os.path.exists(std_path):
        std_df = pd.read_csv(std_path)
    else:
        std_df = pd.DataFrame()

    # save STD
    STD_['initialization'] = init

    std_df = pd.concat([std_df, STD_], axis = 0, ignore_index=True)
    print(std_df)

    pit_df.to_csv(std_path, index = False)
    print(f"Saved STD to {std_path}")

if rank == 0:
    print('----- ENVELOPE VALIDATION -----')

local_results = _run_ffc_envelop_parallel_mpi(_DATA, hyper_shared, 
                                              processes_ = processes_val_, 
                                              fractions_ = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
                                              alpha_ = [0.1, 0.2, 0.3, 0.4],
                                              distances_ = ['MBD', 'l2', 'sup', 'fknn'],
                                              time = time)

# Gather all local results
all_results = comm.gather(local_results, root=0)

if rank == 0:

    # Only root processes all gathered result
    all_results = np.concatenate(all_results, axis=0)
    all_results = pd.DataFrame(all_results, columns = ['time',
                                                       'asset',
                                                       'day',
                                                       'alpha',
                                                       'fraction',
                                                       'distance',
                                                       'n_scen',
                                                       'n_scen_evenlop',
                                                       'FIS',
                                                       'FCS'])

    all_results['FIS'] = all_results['FIS'].astype(float)
    all_results['FCS'] = all_results['FCS'].astype(float)
    all_results['alpha'] = all_results['alpha'].astype(float)

    agg_results = all_results.groupby(['time', 
                                       'alpha', 
                                       'distance', 
                                       'fraction']).agg({'FIS': 'median',
                                                         'FCS': 'median'}).reset_index(drop = False)

    agg_results['FCS'] = (agg_results['FCS'] - (1 - agg_results['alpha']))**2            

    best_results_fis_ = agg_results.loc[agg_results.groupby(['time', 'alpha', 'distance'])['FIS'].idxmin()].reset_index(drop=True)
    best_results_fis_ = best_results_fis_[['time', 'alpha', 'fraction', 'distance']]
    best_results_fis_['iteration'] = init
    best_results_fis_['score'] = 'FIS'

    best_results_fcs_ = agg_results.loc[agg_results.groupby(['time', 'alpha', 'distance'])['FCS'].idxmin()].reset_index(drop=True)
    best_results_fcs_ = best_results_fcs_[['time', 'alpha', 'fraction', 'distance']]
    best_results_fcs_['iteration'] = init
    best_results_fcs_['score'] = 'FCS'

    best_results = pd.concat([best_results_fis_, best_results_fcs_], axis = 0).reset_index(drop=True)
    print(best_results)

    # Overwrite the CSV with the updated data
    # scores_.to_csv(path_to_validation + f'/{resource}/{resource}_{method}_asset-envelop-validation_{time}-{description}.csv', index = False)

comm.Barrier()

# Broadcast inputs (only needed if not already shared)
if rank == 0:
    print('----- ENVELOPE TEST -----')
    best_results_shared = best_results
else:
    best_results_shared = None

# Broadcast envelop parameters
best_results_shared = comm.bcast(best_results_shared, root=0)

for score in best_results_shared['score'].unique():
    for distance in best_results_shared['distance'].unique():

        k_ = best_results_shared.loc[
            (best_results_shared['score'] == score) & (best_results_shared['distance'] == distance), 
            'fraction'].astype(float).tolist()

        alpha_ = best_results_shared.loc[
            (best_results_shared['score'] == score) & (best_results_shared['distance'] == distance), 
            'alpha'].astype(float).tolist()

        local_results = _run_ffc_envelop_parallel_mpi(_DATA, hyper_shared, 
                                                      processes_ = processes_test_, 
                                                      distances_ = [distance],
                                                      alpha_ = alpha_,
                                                      time = time, 
                                                      k_ = k_)

        # Gather all local results
        all_results = comm.gather(local_results, root=0)

        if rank == 0:
            # Only root processes all gathered result
            all_results = np.concatenate(all_results, axis=0)
            all_results = pd.DataFrame(all_results, columns = ['time',
                                                               'asset',
                                                               'day',
                                                               'alpha',
                                                               'fraction',
                                                               'distance',
                                                               'n_scen',
                                                               'n_scen_evenlop',
                                                               'FIS',
                                                               'FCS'])

            all_results['FIS'] = all_results['FIS'].astype(float)
            all_results['FCS'] = all_results['FCS'].astype(float)
            all_results['alpha'] = all_results['alpha'].astype(float)

            all_results = all_results.groupby(['time', 
                                               'alpha']).agg({'FIS': 'median',
                                                              'FCS': 'median'}).reset_index(drop = False)
            
            all_results['score'] = score
            all_results['fraction'] = k_
            all_results['distance'] = distance
            all_results['iteration'] = init

            # define envelop file path
            results_path = path_to_validation + f'/{resource}/{resource}_{method}_asset-envelop-{description}.csv'

            # load or create DataFrame
            if os.path.exists(results_path):
                results_df = pd.read_csv(results_path)
            else:
                results_df = pd.DataFrame()

            # append safely
            results_df = pd.concat([results_df, all_results], axis = 0, ignore_index=True)
            print(results_df)

            # save enevelop
            #results_df.to_csv(results_path, index=False)
            print(f"Saved results to {results_path}")