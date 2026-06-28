import os, optuna, sys, pickle

sys.path.append('/home/gterren/dynamic_update/functional_forecast_dynamic_update/')

import pandas as pd
import numpy as np
import multiprocessing as mp

from mpi4py import MPI
from functools import partial
from skfda.exploratory.depth import ModifiedBandDepth
from scipy.stats import qmc
from time import sleep

from optuna.samplers import GPSampler, TPESampler
from optuna.importance import (get_param_importances,
                               FanovaImportanceEvaluator,
                               MeanDecreaseImpurityImportanceEvaluator,
                               PedAnovaImportanceEvaluator)

from src.fdu import functional_dynamic_update

from src.utils import (_KS, 
                       _weighted_interval_score, 
                       _simultaneous_coverage,
                       _coverage_score,
                       _interval_score,
                       _empirical_PIT, 
                       _energy_score)

path_to_validation = '/home/gterren/dynamic_update/validation'
path_to_param = '/home/gterren/dynamic_update/params'
path_to_data = '/home/gterren/dynamic_update/data'

# load or create DataFrame
def _read_csv_safe(path):

    if os.path.exists(path):

        while os.path.getsize(path) == 0:
            sleep(1)

        return pd.read_csv(path)
    else:
        return pd.DataFrame()
     
# load dataset
def _load_dataset(_dataset, region, N_test_samples):
    #print(region, N_test_samples)

    dt_tr_ = _dataset[region]['datetime_actuals'][:-N_test_samples, ...]
    dt_ts_ = _dataset[region]['datetime_actuals'][-N_test_samples:, ...]
    #print(dt_tr_.shape, dt_ts_.shape)
    
    d_tr_ = _dataset[region]['datetime_actuals'][:-N_test_samples, 0, 0, 24]
    d_ts_ = _dataset[region]['datetime_actuals'][-N_test_samples:, 0, 0, 24]
    d_tr_ = pd.to_datetime(d_tr_).dayofyear.to_numpy()
    d_ts_ = pd.to_datetime(d_ts_).dayofyear.to_numpy()
    #print(d_tr_.shape, d_ts_.shape)

    datetime_tr_ = _dataset[region]['datetime_actuals'][:-N_test_samples, ...]
    datetime_ts_ = _dataset[region]['datetime_actuals'][-N_test_samples:, ...]
    #print(datetime_tr_.shape, datetime_ts_.shape)

    F_tr_ = _dataset[region]['actuals'][:-N_test_samples, ...] 
    F_ts_ = _dataset[region]['actuals'][-N_test_samples:, ...]
    #print(F_tr_.shape, F_ts_.shape)

    if unbiased:
        E_tr_ = _dataset[region]['unbiased_forecasts'][:-N_test_samples, ...] 
        E_ts_ = _dataset[region]['unbiased_forecasts'][-N_test_samples:, ...]
    else:
        E_tr_ = _dataset[region]['forecasts'][:-N_test_samples, ...] 
        E_ts_ = _dataset[region]['forecasts'][-N_test_samples:, ...]
    #print(E_tr_.shape, E_ts_.shape)

    X_tr_ = _dataset[region]['neighbors'][:-N_test_samples, ...]
    X_ts_ = _dataset[region]['neighbors'][-N_test_samples:, ...]
    #print(X_tr_.shape, X_ts_.shape)

    # Formating 
    idx_  = X_ts_[..., 1] == 0.
    F_ts_ = F_ts_[idx_, ...]
    E_ts_ = E_ts_[idx_, ...]
    X_ts_ = X_ts_[idx_, ...]
    datetime_ts_ = datetime_ts_[idx_, ...]
    #print(F_ts_.shape, E_ts_.shape, d_ts_.shape, X_ts_.shape, datetime_ts_.shape)

    F_tr_ = np.concatenate([F_tr_[:, i, ...] for i in range(F_tr_.shape[1])], axis = 0)
    E_tr_ = np.concatenate([E_tr_[:, i, ...] for i in range(E_tr_.shape[1])], axis = 0)
    d_tr_ = np.concatenate([d_tr_ for i in range(X_tr_.shape[1])], axis = 0)
    X_tr_ = np.concatenate([X_tr_[:, i, ...] for i in range(X_tr_.shape[1])], axis = 0)
    datetime_tr_ = np.concatenate([datetime_tr_[:, i, ...] for i in range(datetime_tr_.shape[1])], axis = 0)
    #print(E_tr_.shape, E_tr_.shape, d_tr_.shape, X_tr_.shape, datetime_tr_.shape)

    dt_ = np.linspace(0, dt_tr_.shape[-1] - 1, dt_ts_.shape[-1])*12*5
    #print(dt_.shape)

    return F_tr_, F_ts_, E_tr_, E_ts_, d_tr_, d_ts_, datetime_tr_, datetime_ts_, X_tr_, X_ts_, dt_

# Run FFC experiments for a given process and set of parameters, and compute confidence bands and scoring rules.
def _run_ffc_envelop(process_, _data, _params, distances_, fractions_, alpha_, k_, time, 
                     N_test_days = 360):

    region, day = process_
    day = int(day)

    file_name = f'{region}-{day}-{time}'
    #print(file_name)

    # Get data for this region
    (F_tr_, F_ts_, 
    E_tr_, E_ts_, 
    d_tr_, d_ts_, 
    t_tr_, t_ts_, 
    X_tr_, X_ts_, dt_) = _load_dataset(_data['Dataset'], region, N_test_days)

    # Initialize functional dynamic update model
    _fdu = functional_dynamic_update({'temporal': 'seasonal_equinox', 'spatial': 'graph', 'fusion': 'None'}, region)

    # Get functional training set
    F_tr_p_ = F_tr_[:, time, :]
    E_tr_p_ = np.concatenate([np.zeros((E_tr_.shape[0], 24)), E_tr_[:, time, :]], axis = 1)

    # Fit functional dynamic update model on training data
    _fdu.fit(F_tr_p_, E_tr_p_, dt_, 
             X_ = X_tr_, 
             t_ = d_tr_,
             n_samples_per_hour = 1)

    # Get functional predictors for a given test
    f_ = F_ts_[day, time, :24]
    f_hat_ = F_ts_[day, time, 24:]
    e_ = E_ts_[day, time, :]

    f_ = np.asarray(f_, dtype=np.float64)
    f_hat_ = np.asarray(f_hat_, dtype=np.float64)
    e_ = np.asarray(e_, dtype=np.float64)

    e_p_ = np.concatenate([np.zeros(24,), e_], axis = 0)

    _hyperparams = {**_fixed_hyper[time], **_params,}

    try:    
        # Forecasting update    
        M_hat_ = _fdu.predict(f_, e_p_, X_ts_[day], d_ts_[day], **_hyperparams) 
    
        # Confidence bands from depth function
        _depth = ModifiedBandDepth()

        results_ = []
        for distance in distances_:

            if (distance == 'ECDF'):

                # Confidence bands from marginal empirical density function
                f_median_ext_, _upper_ecdf, _lower_ecdf = _fdu.ecdf_confidence_bands(M_hat_, alpha_)

                for alpha in alpha_:
                    FIS_ecdf = _interval_score(f_hat_, _lower_ecdf[f'{alpha}'], _upper_ecdf[f'{alpha}'], alpha).mean()
                    FCS_ecdf = _coverage_score(f_hat_, _lower_ecdf[f'{alpha}'], _upper_ecdf[f'{alpha}'])
                    SCP_ecdf = _simultaneous_coverage(f_hat_, _lower_ecdf[f'{alpha}'], _upper_ecdf[f'{alpha}'])

                    # Save results
                    results_.append([time, region, day, alpha, M_hat_.shape[0], 'ECDF', M_hat_.shape[0], M_hat_.shape[0], FIS_ecdf, FCS_ecdf, SCP_ecdf])

            # depth-based envelope
            elif (distance == 'MBD'):

                for fraction in fractions_:
                    if fraction is not None: 
                        k_ = [fraction, fraction, fraction, fraction]

                    f_deepest_, _upper_depth, _lower_depth = _fdu.depth_confidence_bands(_depth, M_hat_, alpha_, k_)

                    for alpha in alpha_:
                        FIS_depth = _interval_score(f_hat_, _lower_depth[f'{alpha}'], _upper_depth[f'{alpha}'], alpha).mean()
                        FCS_depth = _coverage_score(f_hat_, _lower_depth[f'{alpha}'], _upper_depth[f'{alpha}'])
                        SCP_depth = _simultaneous_coverage(f_hat_, _lower_depth[f'{alpha}'], _upper_depth[f'{alpha}'])

                        # Save results
                        results_.append([time, region, day, alpha, fraction, distance, M_hat_.shape[0], M_hat_.shape[0], FIS_depth, FCS_depth, SCP_depth])

            # Focal curve envelope
            elif (distance == 'l2') or (distance == 'sup') or (distance == 'fknn'):

                J_hat_ = _fdu.focal_curve_envelope(_depth, _fdu.M_ext_, distance, max_iter = 100)

                for fraction in fractions_:
                    if fraction is not None: 
                        k_ = [fraction, fraction, fraction, fraction]
                    
                    f_focal_, _upper_focal, _lower_focal = _fdu.focal_envelop_confidence_bands(alpha_, k_)

                    for alpha in alpha_:
                        FIS_focal = _interval_score(f_hat_, _lower_focal[f'{alpha}'][1:], _upper_focal[f'{alpha}'][1:], alpha).mean()
                        FCS_focal = _coverage_score(f_hat_, _lower_focal[f'{alpha}'][1:], _upper_focal[f'{alpha}'][1:])
                        SCP_focal = _simultaneous_coverage(f_hat_, _lower_focal[f'{alpha}'][1:], _upper_focal[f'{alpha}'][1:])

                        results_.append([time, region, day, alpha, fraction, distance, M_hat_.shape[0], J_hat_.shape[0], FIS_focal, FCS_focal, SCP_focal])

            else:
                raise ValueError(f"Unknown distance: {distance}")
        results_ = np.stack(results_)

    except Exception as e:
        print(RANK, file_name, e)
        return None

    return results_

# Run FFC experiments in parallel with MPI
def _run_ffc_envelop_parallel_mpi(_data, _params, processes_, distances_, alpha_, time,
                                  fractions_ = [None],
                                  k_ = None):

    _func = partial(_run_ffc_envelop,
                    _data=_data,
                    _params=_params,
                    distances_=distances_,
                    fractions_=fractions_, 
                    alpha_=alpha_, 
                    k_=k_,
                    time=time)

    local_results = []

    local_processes_ = np.array_split(processes_, SIZE)[RANK] 
    for process_ in local_processes_:
        results_list = _func(process_) 
        
        if (results_list is not None) and (results_list is not None):
            local_results.append(results_list)
    
    if (len(local_results) / len(local_processes_)) > 0.5:

        local_results = np.concatenate(local_results, axis = 0)

        return local_results
    else:
        return None
    
# Run FFC experiments for a given process and set of parameters, and compute PIT values and scoring rules.
def _run_ffc(process_, _data, _params, time, N_test_days = 360):

    region, day = process_
    day = int(day)

    file_name = f'{region}-{day}-{time}'
    #print(file_name)

    # Get data for this region
    (F_tr_, F_ts_, 
    E_tr_, E_ts_, 
    d_tr_, d_ts_, 
    t_tr_, t_ts_, 
    X_tr_, X_ts_, dt_) = _load_dataset(_data['Dataset'], region, N_test_days)

    # Initialize functional dynamic update model
    _fdu = functional_dynamic_update({'temporal': 'seasonal_equinox', 'spatial': 'graph', 'fusion': 'None'}, region)

    # Get functional training set
    F_tr_p_ = F_tr_[:, time, :]
    E_tr_p_ = np.concatenate([np.zeros((E_tr_.shape[0], 24)), E_tr_[:, time, :]], axis = 1)

    # Fit functional dynamic update model on training data
    _fdu.fit(F_tr_p_, E_tr_p_, dt_, 
             X_ = X_tr_, 
             t_ = d_tr_,
             n_samples_per_hour = 1)

    # Get functional predictors for a given test
    f_ = F_ts_[day, time, :24]
    f_hat_ = F_ts_[day, time, 24:]
    e_ = E_ts_[day, time, :]

    f_ = np.asarray(f_, dtype=np.float64)
    f_hat_ = np.asarray(f_hat_, dtype=np.float64)
    e_ = np.asarray(e_, dtype=np.float64)

    e_p_ = np.concatenate([np.zeros(24,), e_, ], axis = 0)

    _hyperparams = {**_fixed_hyper[time], **_params,}

    try:    
        # Forecasting update    
        M_hat_ = _fdu.predict(f_, e_p_, X_ts_[day, :], d_ts_[day], **_hyperparams)

        # PIT values from marginal empirical density function
        pit_ = _empirical_PIT(f_hat_, M_hat_.T, seed = 1234)

        # Confidence bands from depth function
        f_deepest_ = _fdu.depth_confidence_bands(ModifiedBandDepth(), M_hat_, ALPHAS, ALPHAS)[0]

        # Confidence bands from marginal empirical density function
        f_median_, _upper, _lower = _fdu.ecdf_confidence_bands(M_hat_, ALPHAS)

        # Scoring rules
        es = _energy_score(M_hat_, f_hat_)
        wis = _weighted_interval_score(f_hat_, f_median_, _lower, _upper, ALPHAS).mean()
        mse = np.mean((f_hat_ - _fdu.f_focal_)**2)
        mae = np.mean(np.absolute(f_hat_ - _fdu.f_focal_))

        # Collect scoring rules
        psr_ = np.array([time, region, day, 1.*_fdu.sparse_neighborhood, 1.*_fdu.sparse_temporal, es, wis, mse, mae])

        # Collect statistics
        stat_ = pd.DataFrame(np.std(M_hat_, axis = 0)).T
        stat_.columns = [f'H{str(i+1).zfill(2)}' for i in range(len(stat_.columns))]
        stat_['time'] = time
        stat_['region'] = region
        stat_['day'] = day

        # Collect functional forecasts and actuals
        f_median_ = pd.DataFrame([f_median_], columns = [f'H{i:02d}' for i in range(len(f_median_))])
        f_median_['type'] = 'median'
        f_deepest_ = pd.DataFrame([f_deepest_], columns = [f'H{i:02d}' for i in range(len(f_deepest_))])
        f_deepest_['type'] = 'deepest'
        f_focal_ = pd.DataFrame([_fdu.f_focal_], columns = [f'H{i:02d}' for i in range(len(_fdu.f_focal_))])
        f_focal_['type'] = 'focal'
        f_ac_ = pd.DataFrame([f_hat_], columns = [f'H{i:02d}' for i in range(len(f_hat_))])
        f_ac_['type'] = 'actual'
        f_fc_ = pd.DataFrame([e_], columns = [f'H{i:02d}' for i in range(len(e_))])
        f_fc_['type'] = 'forecast'

        func_ = pd.concat([f_median_, f_deepest_, f_focal_, f_ac_, f_fc_], axis = 0)  
        func_['time'] = time
        func_['region'] = region
        func_['day'] = day

    except Exception as e:
        print(RANK, file_name,  e)
        return None, None, None, None
    
    return pit_, psr_, stat_, func_

# Run FFC experiments in parallel with MPI
def _run_ffc_parallel_mpi(_data, _params, processes_, time, lambda_0):

    _func = partial(_run_ffc,
                    _data=_data,
                    _params=_params,
                    time=time)

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
    if (len(local_pit_) / len(local_processes_)) > 0.9:
        local_pit_ = np.stack(local_pit_, axis = 0)
        local_psr_ = np.stack(local_psr_, axis = 0)
        local_stat_ = pd.concat(local_stat_, axis = 0)
        local_func_ = pd.concat(local_func_, axis = 0)

    else:
        local_pit_ = None
        local_psr_ = None
        local_stat_ = None
        local_func_ = None
        
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
            ks = 1e10
        
        else:
            # Concatenate results from all processes
            pit_ = np.concatenate(pit_, axis=0)
            psr_ = np.concatenate(psr_, axis=0)
            stat_ = pd.concat(stat_, axis=0)
            func_ = pd.concat(func_, axis=0)

            try:
                # Calculate aggregated PIT
                ks    = np.array([_KS(pit_[:, j:(j + LEAD)].flatten()) for j in INTERVALS]).mean()
                rmse  = np.sqrt(np.mean(psr_[:, -2].astype(float)))
                mae   = np.mean(psr_[:, -1].astype(float))
                score = ks + lambda_0*rmse

                # Penalize score if produces numberical errors
                if np.isnan(score):
                    score = 1e10
                    
                # # Penalize score if produces sparsity
                # if np.mean(psr_[:, 4].astype(float)) > 0.25:
                #     ks = 1e10

            # Penalize score if produces floating point error
            except FloatingPointError as e:
                print("     Skipping parameter due to KS failure")
                return 1e10
        
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
def _objective(trial, _data, _hyper, processes_val_, time, lambda_0):

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
def _latin_hypercube_initialization(_bo, _params, n_samples):

    # Latin Hypercube sampler
    _lhs = qmc.LatinHypercube(d=len(_params))

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
N_bo_iter = 250
N_lhs_init = 125

# Calibration experiments setup
resource = sys.argv[1]
method = sys.argv[2] 
time = int(sys.argv[3])
init = int(sys.argv[4])
lambda_0 = float(sys.argv[5])
unbiased = bool(int(sys.argv[6]))
description = sys.argv[7]
print(resource, method, time, lambda_0, init, unbiased, description)

# Significance levels for the confidence intervals
ALPHAS = [0.1, 0.2, 0.3, 0.4]

# KS evaluation intervals and lead time for the FFC experiments
LEAD = 6
INTERVALS = [0, 6, 12, 18, 24, 30]

# MPI setup
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
print(f'[Process {RANK}-{SIZE}]')

# Fixed parameters depending on interval
# _fixed_hyper = {
#     6:  {'forget_rate_e': 1., 'p_fusion': 1.},
#     12: {'forget_rate_e': 1., 'p_fusion': 1.},
#     18: {'forget_rate_e': 1., 'p_fusion': 1.},
# }

_fixed_hyper = {
    6:  {'forget_rate_e': 1., 'length_scale_f': 0.0025, 'length_scale_e': 0.1, 'p_fusion': 1.},
    12: {'forget_rate_e': 1., 'length_scale_f': 0.0025, 'length_scale_e': 0.1, 'p_fusion': 1.},
    18: {'forget_rate_e': 1., 'length_scale_f': 0.0025, 'length_scale_e': 0.1, 'p_fusion': 1.},
}

# Hyperparameter combinations
_hyper = {
    'clique_order': [0, 2, 'linear'],
    'forget_rate_f': [0.0625, 6., 'log'],
    #'length_scale_f':[0.0025, 1., 'log'],
    #'length_scale_e': [0.0025, 1., 'log'],
    'lookup_rate': [0.0625, 96., 'log'],
    'trust_rate': [0.05, .95, 'linear'],
    'nu': [2., 24., 'linear'],
    'xi': [0.05, 0.95, 'linear'],
    'gamma': [7, 90, 'linear'],
    'kappa': [50, 200, 'linear'],
}

if RANK == 0: 
    print(f'[Resource {resource}][Method {method}][Horizon {time}h][Initialization {init}]')
    print('----- HYPERPARAMETERS VALIDATION -----')

# Load dataset
with open(path_to_data + "/processed_ERCOT_wind_data_v4.pkl", "rb") as f:
    _data = pickle.load(f)

dates_ = np.random.default_rng(1234).permutation(np.arange(360))
processes_val_ = [(region, j) for region in _data['Graph'].keys() for j in dates_[:180]]
processes_test_ = [(region, j) for region in _data['Graph'].keys() for j in dates_[180:]]

if RANK != 0:
    
    # Disable Optuna logging for non-master ranks to avoid cluttering output
    optuna.logging.disable_default_handler()

# Master execution starts here
if RANK == 0:

    # Bayesian optimization with Gaussian Process surrogate model
    _bo = optuna.create_study(direction="minimize",
                              #study_name=f"{time}_{init}-{description}",
                              #storage=db_path,
                              #sampler=TPESampler(seed=1234, multivariate=True, group=True, n_startup_trials=0),)
                              sampler=GPSampler(seed=1234, n_startup_trials=0))

    # Latin Hypercube initialization
    _bo = _latin_hypercube_initialization(_bo, _hyper, n_samples=N_lhs_init)

else:
    _bo = None

# Broadcast the Optuna study object to all ranks
_bo = COMM.bcast(_bo, root=0)

# Define the objective function with fixed arguments using functools.partial
_func = partial(_objective,
                _data=_data,
                _hyper=_hyper,
                processes_val_=processes_val_,
                time=time,
                lambda_0=lambda_0)

# Optimize hyperparameters
_bo.optimize(_func, n_trials=N_bo_iter, n_jobs=1)

if RANK == 0:

    # fANOVA
    importance_fanova = get_param_importances(_bo, evaluator=FanovaImportanceEvaluator())
    # Random forest impurity
    importance_mdi = get_param_importances(_bo, evaluator=MeanDecreaseImpurityImportanceEvaluator())
    # PED-ANOVA
    importance_pedanova = get_param_importances(_bo, evaluator=PedAnovaImportanceEvaluator())

    dfs = []
    for importance, results in {'fANOVA': importance_fanova,
                                'MDI': importance_mdi,
                                'PED-ANOVA': importance_pedanova}.items():

        df_ = pd.DataFrame({'method': importance, 
                            'parameter': list(results.keys()), 
                            'value': list(results.values())})

        dfs.append(df_)

    importance_ = pd.concat(dfs, ignore_index=True)

    importance_['initialization'] = init
    importance_['time'] = time

    # define results file path
    importance_path = path_to_validation + f'/{resource}/{resource}_{method}_zone-importance-{description}.csv'

    # load or create DataFrame
    importance_df = _read_csv_safe(importance_path)

    # save results
    importance_df = pd.concat([importance_df, importance_], ignore_index=True)
    print(importance_df)

    importance_df.to_csv(importance_path, index=False)
    print(f"Saved importance to {importance_path}")

    print('----- HYPERPARAMETERS TEST -----')
    # Retrieve best hyperparameters
    best_score = _bo.best_value
    _best_params = _bo.best_params
    print(best_score)
    print(_best_params)
else:
    best_score = None
    _best_params = None

# Broadcast best hyperparameters to all ranks
_best_params = COMM.bcast(_best_params, root=0)

# Run FFC experiments in parallel
ks, pit_, psr_, stat_, func_ = _run_ffc_parallel_mpi(_data, _best_params, processes_test_, time, lambda_0)

# Gather all local results
pit_ = COMM.gather(pit_, root=0)
psr_ = COMM.gather(psr_, root=0)
stat_ = COMM.gather(stat_, root=0)
func_ = COMM.gather(func_, root=0)

if RANK == 0:

    # Only root processes all gathered result
    pit_ = np.concatenate(pit_, axis=0)
    psr_ = np.concatenate(psr_, axis=0)
    stat_ = pd.concat(stat_, axis=0)
    func_ = pd.concat(func_, axis=0)
    #print(pit_.shape, psr_.shape, stat_.shape, func_.shape)

    # Calculate aggregated PIT
    ks_ = np.array([_KS(pit_[:, j:(j + LEAD)].flatten()) for j in INTERVALS])
    ks_labels_ = [f'S{i}' for i, j in enumerate(INTERVALS)]

    # Proper scores and metrics 
    test_metrics = {'n_neighbors': np.mean(psr_[:, -6].astype(float)),
                    'n_temporal': np.mean(psr_[:, -5].astype(float)),
                    'ES': np.mean(psr_[:, -4].astype(float)),
                    'WIS': np.mean(psr_[:, -3].astype(float)),
                    'RMSE': np.sqrt(np.mean(psr_[:, -2].astype(float))),
                    'MAE': np.mean(psr_[:, -1].astype(float)),
                    'KS': np.mean(ks_.astype(float)),
                    ks_labels_[0]: ks_[0],
                    ks_labels_[1]: ks_[1],
                    ks_labels_[2]: ks_[2],
                    ks_labels_[3]: ks_[3],
                    ks_labels_[4]: ks_[4],
                    ks_labels_[5]: ks_[5]}

    # validation metrics already computed earlier → make sure you keep them
    # (you already defined val_metrics inside loop; keep last one or compute properly)
    row_ = {'initialization': init,
            'time': time,

            # validation KS distance
            'KS_val': best_score,

            # Neighbors and temporal used in the FFC update (these are averaged over all test samples)
            'n_neighbors_test': float(test_metrics['n_neighbors']),
            'n_temporal_test': float(test_metrics['n_temporal']),

            # test proper scores
            'ES_test': float(test_metrics['ES']),
            'WIS_test': float(test_metrics['WIS']),

            # Averege test KS distances
            'RMSE_test': float(test_metrics['RMSE']),
            'MAE_test': float(test_metrics['MAE']),
            'KS_test': float(test_metrics['KS']),

            # Test KS distances
            ks_labels_[0] + '_test': float(test_metrics[ks_labels_[0]]),
            ks_labels_[1] + '_test': float(test_metrics[ks_labels_[1]]),
            ks_labels_[2] + '_test': float(test_metrics[ks_labels_[2]]),
            ks_labels_[3] + '_test': float(test_metrics[ks_labels_[3]]),
            ks_labels_[4] + '_test': float(test_metrics[ks_labels_[4]]),
            ks_labels_[5] + '_test': float(test_metrics[ks_labels_[5]]),

            # parameters
            **_best_params}
    
    # define results file path
    results_path = path_to_validation + f'/{resource}/{resource}_{method}_zone-hyper-{description}.csv'

    # load or create DataFrame
    results_df = _read_csv_safe(results_path)

    # save results
    results_df = pd.concat([results_df, pd.DataFrame([row_])], ignore_index=True)
    print(results_df)

    results_df.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")
    
    # define PIT file path
    pit_path = path_to_validation + f'/{resource}/{resource}_{method}_zone-PIT_{time}-{description}.csv'

    # load or create DataFrame
    pit_df = _read_csv_safe(pit_path)

    # save PIT
    pit_ = pd.DataFrame(pit_)
    pit_.columns = [f'H{str(i+1).zfill(2)}' for i in range(len(pit_.columns))]
    pit_['initialization'] = init
    pit_['time'] = time

    pit_df = pd.concat([pit_df, pit_], axis = 0, ignore_index=True)
    print(pit_df)

    pit_df.to_csv(pit_path, index = False)
    print(f"Saved PIT to {pit_path}")

    # define STATS file path
    stat_path = path_to_validation + f'/{resource}/{resource}_{method}_zone-STATS_{time}-{description}.csv'

    # load or create DataFrame
    stat_df = _read_csv_safe(stat_path)

    # save STATS
    stat_['initialization'] = init

    stat_df = pd.concat([stat_df, stat_], axis = 0, ignore_index=True)
    print(stat_df)

    stat_df.to_csv(stat_path, index = False)
    print(f"Saved STAT to {stat_path}")

    # define functions file path
    func_path = path_to_validation + f'/{resource}/{resource}_{method}_zone-functions_{time}-{description}.csv'

    # load or create DataFrame
    func_df = _read_csv_safe(func_path)

    # save functions
    func_['initialization'] = init

    func_df = pd.concat([func_df, func_], axis = 0, ignore_index=True)
    print(func_df)

    func_df.to_csv(func_path, index = False)
    print(f"Saved FUNCTIONS to {func_path}")

if RANK == 0:
    print('----- ENVELOPE VALIDATION -----')

local_results = _run_ffc_envelop_parallel_mpi(_data, _best_params, processes_val_, 
                                              fractions_ = np.linspace(0.1, 0.9, 17),
                                              alpha_ = [0.1, 0.2, 0.3, 0.4],
                                              distances_ = ['MBD', 'l2', 'sup', 'fknn'],
                                              time = time)

# Gather all local results
results_ = COMM.gather(local_results, root=0)

if RANK == 0:

    # Only root processes all gathered result
    results_ = np.concatenate(results_, axis=0)
    results_df = pd.DataFrame(results_, columns = ['time',
                                                   'asset',
                                                   'day',
                                                   'alpha',
                                                   'fraction',
                                                   'distance',
                                                   'n_scen',
                                                   'n_scen_evenlop',
                                                   'FIS',
                                                   'FCS',
                                                   'SCP'])

    results_df['FIS'] = results_df['FIS'].astype(float)
    results_df['FCS'] = results_df['FCS'].astype(float)
    results_df['SCP'] = results_df['SCP'].astype(float)
    results_df['alpha'] = results_df['alpha'].astype(float)

    agg_results = results_df.groupby(['time', 
                                      'alpha', 
                                      'distance', 
                                      'fraction']).agg({'FIS': 'mean',
                                                        'FCS': 'mean',
                                                        'SCP': 'mean'}).reset_index(drop = False)

    agg_results['FCS'] = (agg_results['FCS'] - (1 - agg_results['alpha']))**2            
    agg_results['SCP'] = (agg_results['SCP'] - (1 - agg_results['alpha']))**2            

    best_results_fis_ = agg_results.loc[agg_results.groupby(['time', 'alpha', 'distance'])['FIS'].idxmin()].reset_index(drop=True)
    best_results_fis_ = best_results_fis_[['time', 'alpha', 'fraction', 'distance']]
    best_results_fis_['iteration'] = init
    best_results_fis_['score'] = 'FIS'

    best_results_fcs_ = agg_results.loc[agg_results.groupby(['time', 'alpha', 'distance'])['FCS'].idxmin()].reset_index(drop=True)
    best_results_fcs_ = best_results_fcs_[['time', 'alpha', 'fraction', 'distance']]
    best_results_fcs_['iteration'] = init
    best_results_fcs_['score'] = 'FCS'

    best_results_scp_ = agg_results.loc[agg_results.groupby(['time', 'alpha', 'distance'])['SCP'].idxmin()].reset_index(drop=True)
    best_results_scp_ = best_results_scp_[['time', 'alpha', 'fraction', 'distance']]
    best_results_scp_['iteration'] = init
    best_results_scp_['score'] = 'SCP'

    best_results = pd.concat([best_results_fis_, best_results_fcs_, best_results_scp_], axis = 0).reset_index(drop=True)
    print(best_results)

    # Overwrite the CSV with the updated data
    # scores_.to_csv(path_to_validation + f'/{resource}/{resource}_{method}_zone-envelope-validation_{time}-{description}.csv', index = False)

COMM.Barrier()

# Broadcast inputs (only needed if not already shared)
if RANK == 0:
    print('----- ENVELOPE TEST -----')
    best_results_shared = best_results
else:
    best_results_shared = None

# Broadcast envelop parameters
best_results_shared = COMM.bcast(best_results_shared, root=0)

for score in best_results_shared['score'].unique():
    for distance in best_results_shared['distance'].unique():

        k_ = best_results_shared.loc[
            (best_results_shared['score'] == score) & (best_results_shared['distance'] == distance), 
            'fraction'].astype(float).tolist()

        alpha_ = best_results_shared.loc[
            (best_results_shared['score'] == score) & (best_results_shared['distance'] == distance), 
            'alpha'].astype(float).tolist()

        local_results = _run_ffc_envelop_parallel_mpi(_data, _best_params, processes_test_, 
                                                      distances_ = [distance],
                                                      alpha_ = alpha_,
                                                      k_ = k_,
                                                      time = time)

        # Gather all local results
        results_ = COMM.gather(local_results, root=0)

        if RANK == 0:
            # Only root processes all gathered result
            results_ = np.concatenate(results_, axis=0)
            results_ = pd.DataFrame(results_, columns = ['time',
                                                         'asset',
                                                         'day',
                                                         'alpha',
                                                         'fraction',
                                                         'distance',
                                                         'n_scen',
                                                         'n_scen_evenlop',
                                                         'FIS',
                                                         'FCS',
                                                         'SCP'])

            results_['FIS'] = results_['FIS'].astype(float)
            results_['FCS'] = results_['FCS'].astype(float)
            results_['SCP'] = results_['SCP'].astype(float)
            results_['alpha'] = results_['alpha'].astype(float)

            results_ = results_.groupby(['time', 
                                         'alpha']).agg({'FIS': 'mean',
                                                        'FCS': 'mean',
                                                        'SCP': 'mean'}).reset_index(drop = False)
            
            results_['score'] = score
            results_['fraction'] = k_
            results_['distance'] = distance
            results_['iteration'] = init

            # define envelop file path
            results_path = path_to_validation + f'/{resource}/{resource}_{method}_zone-envelope-{description}.csv'

            # load or create DataFrame
            results_df = _read_csv_safe(results_path)

            # append safely
            results_df = pd.concat([results_df, results_], axis = 0, ignore_index=True)
            print(results_df)

            # save enevelop
            results_df.to_csv(results_path, index=False)
            print(f"Saved results to {results_path}")

k_ = [None, None, None, None]
distance = 'ECDF'
score = None
local_results = _run_ffc_envelop_parallel_mpi(_data, _best_params, processes_test_, 
                                              distances_ = [distance],
                                              alpha_ = ALPHAS,
                                              k_ = k_,
                                              time = time)

# Gather all local results
results_ = COMM.gather(local_results, root=0)

if RANK == 0:
    # Only root processes all gathered result
    results_ = np.concatenate(results_, axis=0)
    results_ = pd.DataFrame(results_, columns = ['time',
                                                 'asset',
                                                 'day',
                                                 'alpha',
                                                 'fraction',
                                                 'distance',
                                                 'n_scen',
                                                 'n_scen_evenlop',
                                                 'FIS',
                                                 'FCS',
                                                 'SCP'])

    results_['FIS'] = results_['FIS'].astype(float)
    results_['FCS'] = results_['FCS'].astype(float)
    results_['SCP'] = results_['SCP'].astype(float)
    results_['alpha'] = results_['alpha'].astype(float)

    results_ = results_.groupby(['time', 
                                 'alpha']).agg({'FIS': 'mean',
                                                'FCS': 'mean',
                                                'SCP': 'mean'}).reset_index(drop = False)
    
    results_['score'] = score
    results_['fraction'] = k_
    results_['distance'] = distance
    results_['iteration'] = init

    # define envelop file path
    results_path = path_to_validation + f'/{resource}/{resource}_{method}_zone-envelope-{description}.csv'

    # load or create DataFrame
    results_df = _read_csv_safe(results_path)
        
    # append safely
    results_df = pd.concat([results_df, results_], axis = 0, ignore_index=True)
    print(results_df)

    # save enevelop
    results_df.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")
