import os, datetime, sys, optuna

sys.path.append('/home/gterren/dynamic_update/functional_forecast_dynamic_update/')

from pathlib import Path

import pandas as pd
import numpy as np
import pickle as pkl
import multiprocessing as mp

from mpi4py import MPI
from functools import partial
from skfda.exploratory.depth import ModifiedBandDepth
from datetime import datetime
from scipy.stats import qmc
from time import sleep

from optuna.samplers import GPSampler, TPESampler
from optuna.importance import (get_param_importances,
                               FanovaImportanceEvaluator,
                               MeanDecreaseImpurityImportanceEvaluator,
                               PedAnovaImportanceEvaluator)

from src import loader
from src.fdu import functional_dynamic_update
from src.utils import (_KS,
                       _bias_corrected_forecast,
                       _weighted_interval_score,
                       _simultaneous_coverage,
                       _coverage_score,
                       _interval_score,
                       _empirical_PIT,
                       _energy_score)

VALIDATION = Path('/home/gterren/dynamic_update/validation')
PARAM = Path('/home/gterren/dynamic_update/params')
DATA = Path('/home/gterren/dynamic_update/data')

# load or create DataFrame
def _read_csv_safe(path):

    if os.path.exists(path):

        while os.path.getsize(path) == 0:
            sleep(1)

        return pd.read_csv(path)
    else:
        return pd.DataFrame()

# Run FFC experiments for a given process and set of parameters, and compute PIT values and scoring rules.
def _run_ffc(
        process_,
        _data,
        _bo_params,
        time
):

    asset, day = process_

    file_name = f'{asset}-{day}-{time}'
    #print(file_name)

    # Get data for this region
    F_tr_, F_ts_ = _data['ac']
    E_tr_bias_, E_ts_bias_ = _data['day-ahead_bias_fc']
    E_tr_lin_, E_ts_lin_ = _data['day-ahead_linear_fc']
    E_tr_, E_ts_ = _data['day-ahead_fc']
    t_tr_, t_ts_ = _data['dates']
    X_tr_, X_ts_ = _data['locations']
    dt_, dx_ = _data['grid']

    # Filter solar hours with loading solar set
    idx_days_ = np.absolute(t_tr_ - day) < 7
    idx_hours_ = (np.sum(F_tr_[idx_days_, :], axis = 0) + np.sum(E_tr_bias_[idx_days_, :], axis = 0)) > 1.

    # Initialize functional dynamic update model for this region and day
    _fdu = functional_dynamic_update(
        {'temporal': 'seasonal_equinox', 'spatial': 'haversine', 'fusion': 'dynamic'},
        assets_ts_[asset],
        T_ts_[day, time]
    )
    #_fdu = functional_dynamic_update({'temporal': 'seasonal_equinox', 'spatial': 'graph', 'fusion': 'None'}, region)

    # Fit functional dynamic update model on training data
    _fdu.fit(
        F_tr_, E_tr_lin_, dt_,
        X_ = X_tr_,
        t_ = t_tr_,
        interval_mask = idx_hours_,
        n_samples_per_hour = 12
    )

    # Get functional predictors for a given test
    f_ = F_ts_[day, :time, asset]
    e_lin_ = E_ts_lin_[day, :, asset]
    x_ts_ = X_ts_[asset, :]
    t_ts = t_ts_[day]

    _opt_params = _bo_params.copy()

    _hyperparams = {**_fixed_hyper[time], **_opt_params,}

    _hyperparams['length_scale_f'] = 1.0 / _hyperparams.pop('tau')
    _hyperparams['length_scale_e'] = _hyperparams.pop('rho_e') * _hyperparams['length_scale_f']
    _hyperparams['length_scale_d'] = _hyperparams.pop('rho_d') * _hyperparams['length_scale_f']

    try:
        # Forecasting update
        M_hat_ = _fdu.predict(f_, e_lin_, x_ts_, t_ts, **_hyperparams)

        e_ = E_ts_[day, time:, asset]
        f_hat_ = F_ts_[day, time:, asset]

        # PIT values from marginal empirical density function
        pit_ = _empirical_PIT(f_hat_, M_hat_.T, seed = 1234)

        # Confidence bands from depth function
        f_deepest_ = _fdu.depth_confidence_bands(ModifiedBandDepth(), M_hat_, ALPHAS, ALPHAS)[0]

        # Confidence bands from marginal empirical density function
        f_median_, _upper, _lower = _fdu.weighted_ecdf_confidence_bands(M_hat_, _fdu.w_prime_, ALPHAS)
        f_median_ = f_median_[:, 0]
        # Scoring rules
        es = _energy_score(M_hat_, f_hat_)
        wis = _weighted_interval_score(f_hat_, _fdu.f_focal_, _lower, _upper, ALPHAS).mean()
        rmse = np.sqrt(np.mean((f_hat_ - _fdu.f_focal_)**2))
        mae = np.mean(np.absolute(f_hat_ - _fdu.f_focal_))

        # Collect scoring rules
        psr_ = np.array([time, asset, day, es, wis, rmse, mae])

        # Collect statistics
        stat_ = pd.DataFrame(np.std(M_hat_, axis = 0)).T
        stat_.columns = [f'H{str(i+1).zfill(2)}' for i in range(len(stat_.columns))]
        stat_['time'] = time
        stat_['region'] = asset
        stat_['day'] = day

        # Collect functional forecasts and actuals
        f_median_ = pd.DataFrame(
            [f_median_],
            columns = [f'H{i:02d}' for i in range(len(f_median_))]
        )

        f_median_['type'] = 'median'

        f_deepest_ = pd.DataFrame(
            [f_deepest_],
            columns = [f'H{i:02d}' for i in range(len(f_deepest_))]
        )

        f_deepest_['type'] = 'deepest'

        f_focal_ = pd.DataFrame(
            [_fdu.f_focal_],
            columns = [f'H{i:02d}' for i in range(len(_fdu.f_focal_))]
        )

        f_focal_['type'] = 'focal'

        f_ac_ = pd.DataFrame(
            [f_hat_],
            columns = [f'H{i:02d}' for i in range(len(f_hat_))]
        )

        f_ac_['type'] = 'actual'

        f_fc_ = pd.DataFrame(
            [e_],
            columns = [f'H{i:02d}' for i in range(len(e_))]
        )

        f_fc_['type'] = 'forecast'

        f_ = pd.concat([f_median_, f_deepest_, f_focal_, f_ac_, f_fc_], axis = 0)
        f_['time'] = time
        f_['asset'] = asset
        f_['day'] = day

    except Exception as e:
        print(RANK, file_name, e)
        return None, None, None, None

    return pit_, psr_, stat_, f_

# Run FFC experiments for a given process and set of parameters, and compute confidence bands and scoring rules.
def _run_ffc_envelope(
        process_,
        _data,
        _bo_params,
        distances_,
        fractions_,
        alpha_,
        k_,
        time
):

    asset, day = process_

    file_name = f'{asset}-{day}-{time}'
    #print(file_name)

    # Get data for this region
    F_tr_, F_ts_ = _data['ac']
    E_tr_bias_, E_ts_bias_ = _data['day-ahead_bias_fc']
    E_tr_lin_, E_ts_lin_ = _data['day-ahead_linear_fc']
    E_tr_, E_ts_ = _data['day-ahead_fc']
    t_tr_, t_ts_ = _data['dates']
    X_tr_, X_ts_ = _data['locations']
    dt_, dx_ = _data['grid']

    # Filter solar hours with loading solar set
    idx_days_ = np.absolute(t_tr_ - day) < 7
    idx_hours_ = (np.sum(F_tr_[idx_days_, :], axis = 0) + np.sum(E_tr_bias_[idx_days_, :], axis = 0)) > 1.

    # Initialize functional dynamic update model
    _fdu = functional_dynamic_update(
        {'temporal': 'seasonal_equinox','spatial': 'haversine', 'fusion': 'dynamic'},
        assets_ts_[asset],
        T_ts_[day, time]
    )

    # Fit functional dynamic update model on training data
    _fdu.fit(
        F_tr_, E_tr_lin_, dt_,
        X_ = X_tr_,
        t_ = t_tr_,
        interval_mask = idx_hours_,
        n_samples_per_hour = 12
    )

    # Get functional predictors for a given test
    f_ = F_ts_[day, :time, asset]
    e_lin_ = E_ts_lin_[day, :, asset]
    x_ts_ = X_ts_[asset, :]
    t_ts = t_ts_[day]
    e_ = E_ts_[day, time:, asset]
    f_hat_ = F_ts_[day, time:, asset]

    _opt_params = _bo_params.copy()

    _hyperparams = {**_fixed_hyper[time], **_opt_params,}

    _hyperparams['length_scale_f'] = 1.0 / _hyperparams.pop('tau')
    _hyperparams['length_scale_e'] = _hyperparams.pop('rho_e') * _hyperparams['length_scale_f']
    _hyperparams['length_scale_d'] = _hyperparams.pop('rho_d') * _hyperparams['length_scale_f']

    try:
        M_hat_ = _fdu.predict(f_, e_lin_, x_ts_, t_ts, **_hyperparams)

        # Confidence bands from depth function
        M_hat_int_, M_hat_int_ds_ = _fdu.functional_downsampling(
            subsample = 12,
            n_basis = 20,
        )

        _depth = ModifiedBandDepth()

        results_ = []
        for distance in distances_:

            # Empirical confidence bands
            if (distance == 'ECDF'):

                # Confidence bands from marginal empirical density function
                f_median_, _upper_ecdf, _lower_ecdf = _fdu.weighted_ecdf_confidence_bands(M_hat_, _fdu.w_prime_, alpha_)
                f_median_ = f_median_[:, 0]

                for alpha in alpha_:
                    FIS_ecdf = _interval_score(f_hat_, _lower_ecdf[f'{alpha}'], _upper_ecdf[f'{alpha}'], alpha).mean()
                    FCS_ecdf = _coverage_score(f_hat_, _lower_ecdf[f'{alpha}'], _upper_ecdf[f'{alpha}'])
                    SCP_ecdf = _simultaneous_coverage(f_hat_, _lower_ecdf[f'{alpha}'], _upper_ecdf[f'{alpha}'])

                    # Save results
                    results_.append([time, asset, day, alpha, 1, 'ECDF', FIS_ecdf, FCS_ecdf, SCP_ecdf])

                f_ = f_median_

            # depth-based envelope
            if (distance == 'MBD'):

                for fraction in fractions_:
                    if fraction is not None:
                        k_ = [fraction, fraction, fraction, fraction]

                    f_deepest_, _upper_depth, _lower_depth = _fdu.depth_confidence_bands(_depth, M_hat_int_, alpha_, k_)

                    for alpha in alpha_:
                        FIS_depth = _interval_score(f_hat_, _lower_depth[f'{alpha}'][1:], _upper_depth[f'{alpha}'][1:], alpha).mean()
                        FCS_depth = _coverage_score(f_hat_, _lower_depth[f'{alpha}'][1:], _upper_depth[f'{alpha}'][1:])
                        SCP_depth = _simultaneous_coverage(f_hat_, _lower_depth[f'{alpha}'][1:], _upper_depth[f'{alpha}'][1:])

                        # Save results
                        results_.append([time, asset, day, alpha, fraction, distance, FIS_depth, FCS_depth, SCP_depth])

                f_ = f_deepest_[1:]

            # Focal curve envelope
            if (distance == 'fknn'):

                J_hat_ = _fdu.focal_curve_envelope(_depth, M_hat_int_, distance, max_iter = 100)

                for fraction in fractions_:
                    if fraction is not None:
                        k_ = [fraction, fraction, fraction, fraction]

                    f_focal_, _upper_focal, _lower_focal = _fdu.focal_envelope_confidence_bands(alpha_, k_)

                    for alpha in alpha_:
                        FIS_focal = _interval_score(f_hat_, _lower_focal[f'{alpha}'][1:], _upper_focal[f'{alpha}'][1:], alpha).mean()
                        FCS_focal = _coverage_score(f_hat_, _lower_focal[f'{alpha}'][1:], _upper_focal[f'{alpha}'][1:])
                        SCP_focal = _simultaneous_coverage(f_hat_, _lower_focal[f'{alpha}'][1:], _upper_focal[f'{alpha}'][1:])

                        results_.append([time, asset, day, alpha, fraction, distance, FIS_focal, FCS_focal, SCP_focal])

                f_ = f_focal_[1:]

        #print(distance, f_.shape, f_hat_.shape)
        rmse = np.sqrt(np.mean((f_hat_ - f_)**2))
        mae = np.mean(np.absolute(f_hat_ - f_))
        mbe = np.mean(f_hat_ - f_)

        prob_results_ = np.stack(results_)
        det_results_  = np.array([time, asset, day, distance, rmse, mae, mbe])[np.newaxis, :]

    except Exception as e:
        print(RANK, file_name, e)
        return None, None

    return prob_results_, det_results_

# Run FFC experiments in parallel with MPI
def _run_ffc_envelope_parallel_mpi(
        _data,
        _params,
        processes_,
        distances_,
        alpha_,
        time,
        fractions_ = [None],
        k_ = None
):

    _func = partial(
        _run_ffc_envelope,
        _data=_data,
        _bo_params=_params,
        distances_=distances_,
        fractions_=fractions_,
        alpha_=alpha_,
        k_=k_,
        time=time
    )

    prob_local_results = []
    det_local_results = []

    local_processes_ = np.array_split(processes_, SIZE)[RANK]
    for process_ in local_processes_:
        prob_results_list, det_results_list = _func(process_)

        if (prob_results_list is not None) and (det_results_list is not None):
            prob_local_results.append(prob_results_list)
            det_local_results.append(det_results_list)

    n_local = len(local_processes_)
    if n_local > 0 and (len(prob_local_results) / n_local) > 0.5:

        prob_local_results = np.concatenate(prob_local_results, axis = 0)
        det_local_results = np.concatenate(det_local_results, axis = 0)

        return prob_local_results, det_local_results
    else:
        return None, None

# Run FFC experiments in parallel with MPI
def _run_ffc_parallel_mpi(
        _data,
        _params,
        processes_,
        time,
        lambda_0
):

    _func = partial(
        _run_ffc,
        _data=_data,
        _bo_params=_params,
        time=time
    )

    # Split processes among ranks
    local_processes_ = np.array_split(processes_, SIZE)[RANK]

    # Each rank processes its assigned subset of processes and collects local results
    local_pit_ = []
    local_psr_ = []
    local_stat_ = []
    local_func_ = []

    for process_ in local_processes_:
        pit_, psr_, stat_, func_ = _func(process_)
        # Only keep results if not None (i.e., if the process succeeded)
        if (pit_ is not None) and (psr_ is not None):
            local_pit_.append(pit_)
            local_psr_.append(psr_)
            local_stat_.append(stat_)
            local_func_.append(func_)

    # Only keep results if more than 90% of local processes succeeded
    # (to avoid skewed results from too few samples)
    n_local = len(local_processes_)
    if n_local > 0 and (len(local_pit_) / n_local) > 0.9:
        local_pit_ = np.stack(local_pit_, axis = 0)
        local_psr_ = np.stack(local_psr_, axis = 0)
        local_stat_ = pd.concat(local_stat_, axis = 0)
        local_func_ = pd.concat(local_func_, axis = 0)

    else:
        local_pit_ = local_psr_ = local_stat_ = local_func_ = None

    # Gather all local results
    pit_ = COMM.gather(local_pit_, root=0)
    psr_ = COMM.gather(local_psr_, root=0)
    stat_ = COMM.gather(local_stat_, root=0)
    func_ = COMM.gather(local_func_, root=0)

    # Master computes KS
    if RANK == 0:
        # Make sure to filter out any None results from failed processes
        pit_ = [x for x in pit_ if x is not None]
        
        # If not all processes returned results, skip this parameter combination
        if len(pit_) != SIZE:
            score = 1e10

        else:
            # Concatenate results from all processes
            pit_ = np.concatenate(pit_, axis=0)
            psr_ = np.concatenate(psr_, axis=0)
            stat_ = pd.concat(stat_, axis=0)
            func_ = pd.concat(func_, axis=0)

            try:
                # Calculate aggregated PIT
                ks = np.array([_KS(pit_[:, j:(j + LEAD)].flatten()) for j in INTERVALS]).mean()
                rmse = np.mean(psr_[:, -2].astype(float))
                mae = np.mean(psr_[:, -1].astype(float))
                score = ks + lambda_0*rmse

                # Penalize score if produces numberical errors
                if np.isnan(score):
                    score = 1e10

                # # Penalize score if produces sparsity
                # if np.mean(psr_[:, 3].astype(float)) > 0.1:
                #     ks = 1e10

            # Penalize score if produces floating point error
            except FloatingPointError as e:
                print("     Skipping parameter due to KS failure")
                score = 1e10

    else:
        score = None
        pit_ = None
        psr_ = None
        stat_ = None
        func_ = None

    # Broadcast KS and PIT values
    score = COMM.bcast(score, root=0)
    pit_ = COMM.bcast(pit_, root=0)
    psr_ = COMM.bcast(psr_, root=0)
    stat_ = COMM.bcast(stat_, root=0)
    func_ = COMM.bcast(func_, root=0)

    return score, pit_, psr_, stat_, func_

# Objective function used by Optuna for Bayesian hyperparameter optimization.
def _objective(
        trial,
        _data,
        _hyper,
        processes_val_,
        time,
        lambda_0
):

    if RANK == 0:
        # Dictionary storing sampled hyperparameters
        _params = {}

        # Iterate over all hyperparameters
        for name in _hyper.keys():

            # Unpack lower bound, upper bound, and scaling type
            l, u, scale = _hyper[name]

            if type(u) == int:
                # Integer hyperparameters
                _params[name] = trial.suggest_int(name, l, u, log=(scale == 'log'))
            else:
                # Continuous hyperparameters
                _params[name] = trial.suggest_float(name, l, u, log=(scale == 'log'))

    else:
        _params = None

    # Broadcast parameter and values
    _params = COMM.bcast(_params, root=0)

    # Synchronize all processes before running FFC experiments
    COMM.Barrier()

    # Run function evaluation in parallel with MPI and return the KS score
    return _run_ffc_parallel_mpi(_data, _params, processes_val_, time, lambda_0)[0]

# Generate and enqueue Latin Hypercube Sampling (LHS) trials for Optuna Bayesian optimization.
def _latin_hypercube_initialization(
        _bo,
        _params,
        n_samples,
        seed,
):

    # Latin Hypercube sampler
    _lhs = qmc.LatinHypercube(d=len(_params), seed=seed)

    # Generate samples in [0,1]
    for sample in _lhs.random(n=n_samples):

        # Dictionary storing sampled hyperparameters
        x_ = {}

        # Iterate through hyperparameters
        for i, name in enumerate(list(_params.keys())):
            low, high, scale = _params[name]

            if scale == 'log':
                # Logarithmic scaling
                x_[name] = low * (high / low) ** sample[i]
            elif scale == 'linear':
                # Linear scaling
                x_[name] = low + (high - low) * sample[i]
            else:
                # Unknown scaling type
                raise ValueError(f"Unknown scale type: {scale}")

            if type(high) == int:
                # Interger parameter
                x_[name] = int(round(x_[name]))
            else:
                # Floating-point parameter
                x_[name] = float(x_[name])

        # Queue trial into Optuna study
        _bo.enqueue_trial(x_)

    return _bo

# Number of interation
N_bo_iter = 200
N_lhs_init = 100

# Calibration experiments setup
resource = sys.argv[1]
method = sys.argv[2]
time = int(sys.argv[3])
init = int(sys.argv[4])
lambda_0 = float(sys.argv[5])
unbiased = bool(int(sys.argv[6]))
description = sys.argv[7]
print(resource, method, time, init, lambda_0, unbiased, description)

# Significance levels for the confidence intervals
ALPHAS = [0.1, 0.2, 0.3, 0.4]
HYPERPARAMETERS = True
ENVELOPE = True
# KS evaluation intervals and lead time for the FFC experiments
if resource == 'wind':
    if time == 72:
        LEAD = 36
        INTERVALS = [0, 36, 72, 108, 144, 180]
    elif time == 144:
        LEAD = 24
        INTERVALS = [0, 24, 48, 72, 96, 120]
    elif time == 216:
        LEAD = 12
        INTERVALS = [0, 12, 24, 36, 48, 60]
    else:
        exit("Invalid time horizon. Must be one of [72, 144, 216].")
elif resource == 'solar':
    if time == 120:
        LEAD = 12
        INTERVALS = [0, 12, 24, 36, 48, 60]
    elif time == 132:
        LEAD = 9
        INTERVALS = [0, 9, 18, 27, 36, 45]
    elif time == 144:
        LEAD = 9
        INTERVALS = [0, 9, 18, 27, 36, 45]
    elif time == 168:
        LEAD = 6
        INTERVALS = [0, 6, 12, 18, 24, 30]
    else:
        exit("Invalid time horizon. Must be one of [72, 144, 216].")
else:
    exit("Invalid resource. Must be one of ['wind', 'solar'].")

# MPI setup
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
print(f'[Process {RANK}-{SIZE}]')

if resource == 'wind':

    # Fixed parameters depending on interval for wind
    _fixed_hyper = {
        72:  {'p_fusion': 0.75, 'forget_rate_f': 0.125, 'forget_rate_e': 1, 'lookahead_rate': 4, 'tau': 0.001, 'kappa': 150},
        144: {'p_fusion': 0.75, 'forget_rate_f': 0.125, 'forget_rate_e': 1, 'lookahead_rate': 4, 'tau': 0.001, 'kappa': 150},
        216: {'p_fusion': 0.75, 'forget_rate_f': 0.125, 'forget_rate_e': 1, 'lookahead_rate': 4, 'tau': 0.001, 'kappa': 150},
    }

    # Hyperparameter combinations
    _hyper = {
        'rho_e':   [1e-5,  1000.,  'log'],   # λ_e / λ_f
        'rho_d':   [1e-5,  1000.,  'log'],   # λ_d / λ_f
        #'tau':     [1e-5,  1000.,  'log'],   # temperature = 1 / λ_f
        'kappa_0': [150,   5000,   'log'],
        'nu':      [1,     12,     'linear'],
        'sigma':   [0.,    1.,     'linear'],
    }

elif resource == 'solar':

    # Fixed parameters depending on interval for solar
    _fixed_hyper = {
        120: {'p_fusion': 0.75, 'forget_rate_f': 0.125, 'forget_rate_e': 1, 'lookahead_rate': 4, 'tau': 0.001, 'kappa': 150},
        144: {'p_fusion': 0.75, 'forget_rate_f': 0.125, 'forget_rate_e': 1, 'lookahead_rate': 4, 'tau': 0.001, 'kappa': 150},
        168: {'p_fusion': 0.75, 'forget_rate_f': 0.125, 'forget_rate_e': 1, 'lookahead_rate': 4, 'tau': 0.001, 'kappa': 150},
    }

    # Hyperparameter combinations
    _hyper = {
        'rho_e':   [1e-5,  1000.,  'log'],   # λ_e / λ_f
        'rho_d':   [1e-5,  1000.,  'log'],   # λ_d / λ_f
        #'tau':     [1e-5,  1000.,  'log'],   # temperature = 1 / λ_f
        'kappa_0': [150,   5000,   'log'],
        'nu':      [1,     12,     'linear'],
        'sigma':   [0.,    1.,     'linear'],
    }

else:
    exit("Invalid resource. Must be one of ['wind', 'solar'].")


## -------------------------- LOAD DATA ---------------------------------
(
    F_tr_,
    F_ts_,
    E_tr_,
    E_ts_,
    X_tr_,
    X_ts_,
    T_tr_,
    T_ts_,
    assets_tr_,
    assets_ts_,
    t_tr_,
    t_ts_,
    dt_,
    dx_,
) = loader.preprocessed_dataset(
    unbiased = unbiased,
    path_to_training_data = DATA / "preprocessed_wind_2017.pkl",
    path_to_testing_data = DATA / "preprocessed_wind_2018.pkl",
    T = 288,
)

# print(F_tr_.shape, F_ts_.shape)
# print(E_tr_.shape, E_ts_.shape)
# print(X_tr_.shape, X_ts_.shape)
# print(T_tr_.shape, T_ts_.shape)
# print(assets_tr_.shape, assets_ts_.shape)
# print(t_tr_.shape, t_ts_.shape)
# print(dt_.shape, dx_.shape)

E_tr_lin_, E_ts_lin_ = loader.processed_dataset(
    unbiased = unbiased,
    path_to_training_data = DATA / "linear_preprocessed_wind_2017.pkl",
    path_to_testing_data = DATA / "linear_preprocessed_wind_2018.pkl",
    T = 288,
)

# print(E_tr_lin_.shape, E_ts_lin_.shape)

E_tr_biased_, E_ts_biased_ = loader.processed_dataset(
    unbiased = False,
    path_to_training_data = DATA / "preprocessed_wind_2017.pkl",
    path_to_testing_data = DATA / "preprocessed_wind_2018.pkl",
    T = 288,
)

# print(E_tr_biased_.shape, E_ts_biased_.shape)

# Load dataset
_data = {
    'ac': [F_tr_, F_ts_],
    'day-ahead_bias_fc': [E_tr_biased_, E_ts_biased_],
    'day-ahead_linear_fc': [E_tr_lin_, E_ts_lin_],
    'day-ahead_fc': [E_tr_, E_ts_],
    'dates': [t_tr_, t_ts_],
    'locations': [X_tr_, X_ts_],
    'grid': [dt_, dx_]
}

processes_val_ = [(asset, j) for asset in range(0, 10) for j in range(0, 360)]
processes_test_ = [(asset, j) for asset in range(10, 20) for j in range(0, 360)]

if (RANK == 0) and (HYPERPARAMETERS == True):
    print(f'[Resource {resource}][Method {method}][Horizon {time}h][Initialization {init}]')
    print('----- HYPERPARAMETERS VALIDATION ----')

if (RANK != 0) and (HYPERPARAMETERS == True):
    # Disable Optuna logging for non-master ranks to avoid cluttering output
    optuna.logging.disable_default_handler()

# Master execution starts here
if (RANK == 0) and (HYPERPARAMETERS == True):

    SEED = 1234 + init 
    # Bayesian optimization with Gaussian Process surrogate model
    _bo = optuna.create_study(
        direction="minimize",
        sampler=GPSampler(seed=SEED, n_startup_trials=0)
    )

    # Latin Hypercube initialization
    _bo = _latin_hypercube_initialization(_bo, _hyper, n_samples=N_lhs_init, seed=SEED)

else:
    _bo = None

# Define the objective function with fixed arguments using functools.partial
_func = partial(
    _objective,
    _data=_data,
    _hyper=_hyper,
    processes_val_=processes_val_,
    time=time,
    lambda_0=lambda_0
)

# Optimize hyperparameters: only rank 0 runs the Optuna loop; the other
# ranks call the objective the same number of times so they participate
# in the bcast/Barrier/gather collectives inside it.
if RANK == 0:
    _bo.optimize(_func, n_trials=N_bo_iter, n_jobs=1)
else:
    for _ in range(N_bo_iter):
        _func(None)

if (RANK == 0) and (HYPERPARAMETERS == True):

    print('----- HYPERPARAMETERS TEST -----')
    # Retrieve best hyperparameters
    best_ks = _bo.best_value
    _best_hyper = _bo.best_params
    print(best_ks)
    print(_best_hyper)
    print(_fixed_hyper[time])
else:
    best_ks = None
    _best_hyper = None

if (HYPERPARAMETERS == True):
    # Broadcast best hyperparameters to all ranks
    _best_hyper = COMM.bcast(_best_hyper, root=0)

    # Run FFC experiments in parallel
    ks, pit_, psr_, stat_, func_ = _run_ffc_parallel_mpi(
        _data,
        _best_hyper,
        processes_test_,
        time,
        lambda_0
    )

if (RANK == 0) and (HYPERPARAMETERS == True):

    # Calculate aggregated PIT
    ks_ = np.array([_KS(pit_[:, j:(j + LEAD)].flatten()) for j in INTERVALS])
    ks_labels_ = [f'S{i}' for i, j in enumerate(INTERVALS)]

    # Proper scores and metrics
    test_metrics = {
        'ES': np.mean(psr_[:, -4].astype(float)),
        'WIS': np.mean(psr_[:, -3].astype(float)),
        'RMSE': np.mean(psr_[:, -2].astype(float)),
        'MAE': np.mean(psr_[:, -1].astype(float)),
        'KS': np.mean(ks_.astype(float)),
        ks_labels_[0]: ks_[0],
        ks_labels_[1]: ks_[1],
        ks_labels_[2]: ks_[2],
        ks_labels_[3]: ks_[3],
        ks_labels_[4]: ks_[4],
        ks_labels_[5]: ks_[5]
    }

    row_ = {
        'initialization': init,
        'time': time,

        # Averege validation KS distance
        'KS_val': best_ks,

        # test proper scores
        'ES_test': float(test_metrics['ES']),
        'WIS_test': float(test_metrics['WIS']),

        # Averege test KS distances
        'RMSE_test': float(test_metrics['RMSE']),
        'MAE_test': float(test_metrics['MAE']),
        'KS_test': float(test_metrics['KS']),

        # Test KS distance
        ks_labels_[0] + '_test': float(test_metrics[ks_labels_[0]]),
        ks_labels_[1] + '_test': float(test_metrics[ks_labels_[1]]),
        ks_labels_[2] + '_test': float(test_metrics[ks_labels_[2]]),
        ks_labels_[3] + '_test': float(test_metrics[ks_labels_[3]]),
        ks_labels_[4] + '_test': float(test_metrics[ks_labels_[4]]),
        ks_labels_[5] + '_test': float(test_metrics[ks_labels_[5]]),

        # parameters
        **{**_fixed_hyper[time], **_best_hyper}
    }

    # define results file path
    results_path = VALIDATION / f'{resource}/{resource}_{method}_asset-hyper-{description}.csv'

    # load or create DataFrame
    results_df =  _read_csv_safe(results_path)

    # save results
    results_df = pd.concat([results_df, pd.DataFrame([row_])], ignore_index=True)
    #print(results_df)

    results_df.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")

    # define PIT file path
    pit_path = VALIDATION / f'{resource}/{resource}_{method}_asset-PIT_{time}-{description}.csv'

    # load or create DataFrame
    pit_df =  _read_csv_safe(pit_path)

    # save PIT
    pit_ = pd.DataFrame(pit_)
    pit_.columns = [f'H{str(i+1).zfill(2)}' for i in range(len(pit_.columns))]
    pit_['initialization'] = init
    pit_['time'] = time

    pit_df = pd.concat([pit_df, pit_], axis = 0, ignore_index=True)
    #print(pit_df)

    pit_df.to_csv(pit_path, index = False)
    print(f"Saved PIT to {pit_path}")

    # define stats file path
    stat_path = VALIDATION / f'{resource}/{resource}_{method}_asset-STATS_{time}-{description}.csv'

    # load or create DataFrame
    stat_df =  _read_csv_safe(stat_path)

    # save stats
    stat_['initialization'] = init

    stat_df = pd.concat([stat_df, stat_], axis = 0, ignore_index=True)
    #print(stat_df)

    stat_df.to_csv(stat_path, index = False)
    print(f"Saved STAT to {stat_path}")

    # define stats file path
    func_path = VALIDATION / f'{resource}/{resource}_{method}_asset-funcions_{time}-{description}.csv'

    # load or create DataFrame
    func_df =  _read_csv_safe(func_path)

    # save stats
    func_['initialization'] = init

    func_df = pd.concat([func_df, func_], axis = 0, ignore_index=True)
    #print(stat_df)

    func_df.to_csv(func_path, index = False)
    print(f"Saved funcions to {func_path}")

if (RANK == 0) and (ENVELOPE == True):
    print('----- ENVELOPE VALIDATION -----')

prob_local_results, det_local_results = _run_ffc_envelope_parallel_mpi(
    _data,
    _best_hyper,
    processes_val_,
    fractions_ = np.linspace(0.1, 0.9, 17),
    alpha_ = [0.1, 0.2, 0.3, 0.4],
    distances_ = ['MBD', 'fknn'],
    time = time
)

# Gather all local results
prob_results = COMM.gather(prob_local_results, root=0)

if (RANK == 0) and (ENVELOPE == True):

    # Only root processes all gathered result
    prob_results = pd.DataFrame(
        np.concatenate(prob_results, axis=0), columns = [
            'time',
            'asset',
            'day',
            'alpha',
            'fraction',
            'distance',
            'FIS',
            'FCS',
            'SCP'
        ]
    )

    prob_results['FIS'] = prob_results['FIS'].astype(float)
    prob_results['FCS'] = prob_results['FCS'].astype(float)
    prob_results['SCP'] = prob_results['SCP'].astype(float)

    prob_results = prob_results.groupby(
        ['time', 'alpha', 'distance', 'fraction']).agg({'FIS': 'mean', 'FCS': 'mean', 'SCP': 'mean'}
    ).reset_index(drop = False)

    prob_results['alpha'] = prob_results['alpha'].astype(float)

    prob_results['FCS'] = (prob_results['FCS'] - (1 - prob_results['alpha']))**2
    prob_results['SCP'] = (prob_results['SCP'] - (1 - prob_results['alpha']))**2

    best_results_fis_ = prob_results.loc[prob_results.groupby(
        ['time', 'alpha', 'distance']
    )['FIS'].idxmin()].reset_index(drop=True)

    best_results_fis_ = best_results_fis_[['time', 'alpha', 'fraction', 'distance']]
    best_results_fis_['iteration'] = init
    best_results_fis_['score'] = 'FIS'

    best_results_fcs_ = prob_results.loc[prob_results.groupby(
        ['time', 'alpha', 'distance']
    )['FCS'].idxmin()].reset_index(drop=True)

    best_results_fcs_ = best_results_fcs_[['time', 'alpha', 'fraction', 'distance']]
    best_results_fcs_['iteration'] = init
    best_results_fcs_['score'] = 'FCS'

    best_results_scp_ = prob_results.loc[prob_results.groupby(
        ['time', 'alpha', 'distance']
    )['SCP'].idxmin()].reset_index(drop=True)

    best_results_scp_ = best_results_scp_[['time', 'alpha', 'fraction', 'distance']]
    best_results_scp_['iteration'] = init
    best_results_scp_['score'] = 'SCP'

    best_results = pd.concat([best_results_fis_, best_results_fcs_, best_results_scp_], axis = 0).reset_index(drop=True)
    #print(best_results)

    # Overwrite the CSV with the updated data
    # best_results.to_csv(VALIDATION / f'{resource}/{resource}_{method}_asset-envelop-validation_{time}-{description}.csv', index = False)

COMM.Barrier()

# Broadcast inputs (only needed if not already shared)
if (RANK == 0) and (ENVELOPE == True):
    print('----- ENVELOPE TEST -----')
    best_results_shared = best_results
else:
    best_results_shared = None

if (ENVELOPE == True):
    # Broadcast envelop parameters
    best_results_shared = COMM.bcast(best_results_shared, root=0)

    for score in best_results_shared['score'].unique():
        for distance in best_results_shared['distance'].unique():

            k_ = best_results_shared.loc[
                (best_results_shared['score'] == score) & (best_results_shared['distance'] == distance), 'fraction'
            ].astype(float).tolist()

            alpha_ = best_results_shared.loc[
                (best_results_shared['score'] == score) & (best_results_shared['distance'] == distance), 'alpha'
            ].astype(float).tolist()

            prob_local_results, det_local_results = _run_ffc_envelope_parallel_mpi(
                _data, _best_hyper, processes_test_,
                distances_ = [distance],
                alpha_ = alpha_,
                k_ = k_,
                time = time
            )

            # Gather all local results
            prob_results = COMM.gather(prob_local_results, root=0)
            det_results = COMM.gather(det_local_results, root=0)

            if RANK == 0:
                # Only root processes all gathered result
                prob_results = pd.DataFrame(
                    np.concatenate(prob_results, axis=0), columns = [
                        'time',
                        'asset',
                        'day',
                        'alpha',
                        'fraction',
                        'distance',
                        'FIS',
                        'FCS',
                        'SCP'
                    ]
                )

                prob_results['FIS'] = prob_results['FIS'].astype(float)
                prob_results['FCS'] = prob_results['FCS'].astype(float)
                prob_results['SCP'] = prob_results['SCP'].astype(float)
                prob_results['alpha'] = prob_results['alpha'].astype(float)

                prob_results = prob_results.groupby(
                    ['time', 'alpha']).agg({'FIS': 'mean', 'FCS': 'mean', 'SCP': 'mean'}
                ).reset_index(drop = False)

                prob_results['score'] = score
                prob_results['fraction'] = k_
                prob_results['distance'] = distance
                prob_results['iteration'] = init

                # define envelop file path
                results_path = VALIDATION / f'{resource}/{resource}_{method}_asset-envelope-{description}.csv'

                # load or create DataFrame
                results_df = _read_csv_safe(results_path)

                # append safely
                results_df = pd.concat([results_df, prob_results], axis = 0, ignore_index=True)
                #print(results_df)

                # save enevelop
                results_df.to_csv(results_path, index=False)
                print(f"Saved results to {results_path}")

                det_results = pd.DataFrame(
                    np.concatenate(det_results, axis=0), columns = [
                        'time',
                        'asset',
                        'day',
                        'distance',
                        'MSE',
                        'MAE',
                        'MBE'
                    ]
                )

                det_results['MSE'] = det_results['MSE'].astype(float)
                det_results['MAE'] = det_results['MAE'].astype(float)
                det_results['MBE'] = det_results['MBE'].astype(float)

                det_results = det_results.groupby(
                    ['time']).agg({'MSE': 'mean', 'MAE': 'mean', 'MBE': 'mean'}
                ).reset_index(drop = False)

                det_results['score'] = score
                det_results['distance'] = distance
                det_results['iteration'] = init

                # define envelop file path
                results_path = VALIDATION / f'{resource}/{resource}_{method}_asset-error-{description}.csv'

                # load or create DataFrame
                results_df = _read_csv_safe(results_path)

                # append safely
                results_df = pd.concat([results_df, det_results], axis = 0, ignore_index=True)
                #print(results_df)

                # save enevelop
                results_df.to_csv(results_path, index=False)
                print(f"Saved results to {results_path}")

    k_ = [None, None, None, None]
    distance = 'ECDF'
    score = None

    prob_local_results, det_local_results = _run_ffc_envelope_parallel_mpi(
        _data,
        _best_hyper,
        processes_test_,
        distances_ = [distance],
        alpha_ = ALPHAS,
        k_ = k_,
        time = time
    )

    # Gather all local results
    prob_results = COMM.gather(prob_local_results, root=0)
    det_results = COMM.gather(det_local_results, root=0)

if (RANK == 0) and (ENVELOPE == True):

    # Only root processes all gathered result
    prob_results = pd.DataFrame(
        np.concatenate(prob_results, axis=0), columns = [
            'time',
            'asset',
            'day',
            'alpha',
            'fraction',
            'distance',
            'FIS',
            'FCS',
            'SCP'
        ]
    )

    prob_results['FIS'] = prob_results['FIS'].astype(float)
    prob_results['FCS'] = prob_results['FCS'].astype(float)
    prob_results['SCP'] = prob_results['SCP'].astype(float)
    prob_results['alpha'] = prob_results['alpha'].astype(float)

    prob_results = prob_results.groupby(
        ['time', 'alpha']).agg({'FIS': 'mean', 'FCS': 'mean', 'SCP': 'mean'}
    ).reset_index(drop = False)

    prob_results['score'] = score
    prob_results['fraction'] = k_
    prob_results['distance'] = distance
    prob_results['iteration'] = init

    # define envelop file path
    results_path = VALIDATION / f'{resource}/{resource}_{method}_asset-envelope-{description}.csv'

    # load or create DataFrame
    results_df =  _read_csv_safe(results_path)

    # append safely
    results_df = pd.concat([results_df, prob_results], axis = 0, ignore_index=True)
    #print(results_df)

    # save enevelop
    results_df.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")

    det_results = pd.DataFrame(
        np.concatenate(det_results, axis=0), columns = [
            'time',
            'asset',
            'day',
            'distance',
            'MSE',
            'MAE',
            'MBE'
        ]
    )

    det_results['MSE'] = det_results['MSE'].astype(float)
    det_results['MAE'] = det_results['MAE'].astype(float)
    det_results['MBE'] = det_results['MBE'].astype(float)

    det_results = det_results.groupby(
        ['time']).agg({'MSE': 'mean', 'MAE': 'mean', 'MBE': 'mean'}
    ).reset_index(drop = False)

    det_results['score'] = score
    det_results['distance'] = distance
    det_results['iteration'] = init

    # define envelop file path
    results_path = VALIDATION / f'{resource}/{resource}_{method}_asset-error-{description}.csv'

    # load or create DataFrame
    results_df =  _read_csv_safe(results_path)

    # append safely
    results_df = pd.concat([results_df, det_results], axis = 0, ignore_index=True)
    #print(results_df)

    # save enevelop
    results_df.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")
