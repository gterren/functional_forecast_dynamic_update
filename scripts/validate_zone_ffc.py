import os, datetime, sys, time, pickle

sys.path.append('/home/gterren/dynamic_update/functional_forecast_dynamic_update/')

import pandas as pd
import numpy as np
import pickle as pkl
import multiprocessing as mp

from mpi4py import MPI

from functools import partial

from datetime import datetime, timedelta

from src.fdu import functional_dynamic_update

from src.utils import (_KS, 
                       _weighted_empirical_interval_score, 
                       _empirical_PIT, 
                       _energy_score)

np.seterr(all='raise')

path_to_fDepth = '/home/gterren/dynamic_update/functional_forecast_dynamic_update/fDepth'
path_to_data = '/home/gterren/dynamic_update/data'
path_to_validation = '/home/gterren/dynamic_update/validation'
path_to_param = '/home/gterren/dynamic_update/params'

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

def _save_validation_csv(df_new_, path_to_file):

    if isinstance(df_new_, pd.DataFrame):

        # Check if the CSV exists
        if os.path.exists(path_to_file):
            
            # Read the data and append the new data
            df_existing_ = pd.read_csv(path_to_file,
                                       engine="python",
                                       on_bad_lines="warn")
            
            df_new_ = pd.concat([df_existing_,
                                 df_new_],
                                 ignore_index = True).reset_index(drop = True)

        # Overwrite the CSV with the updated data
        df_new_.to_csv(path_to_file, index = False)
        print(path_to_file)

# load dataset
def _load_dataset(_dataset, region, N_test_sampels):
    #print(region, N_test_sampels)
    
    zone = region
    N = N_test_sampels

    dt_tr_ = _dataset[zone]['datetime_actuals'][:-N, ...]
    dt_ts_ = _dataset[zone]['datetime_actuals'][-N:, ...]
    #print(dt_tr_.shape, dt_ts_.shape)
    
    d_tr_ = _dataset[zone]['datetime_actuals'][:-N, 0, 24]
    d_ts_ = _dataset[zone]['datetime_actuals'][-N:, 0, 24]
    d_tr_ = pd.to_datetime(d_tr_).dayofyear.to_numpy()
    d_ts_ = pd.to_datetime(d_ts_).dayofyear.to_numpy()
    #print(d_tr_.shape, d_ts_.shape)

    datetime_tr_ = _dataset[zone]['datetime_actuals'][:-N, ...]
    datetime_ts_ = _dataset[zone]['datetime_actuals'][-N:, ...]
    #print(datetime_tr_.shape, datetime_ts_.shape)

    F_tr_ = _dataset[zone]['actuals'][:-N, ...] 
    F_ts_ = _dataset[zone]['actuals'][-N:, ...]
    #print(F_tr_.min(), F_tr_.max())
    #print(F_ts_.min(), F_ts_.max())
    #print(F_tr_.shape, F_ts_.shape)

    E_tr_ = _dataset[zone]['forecasts'][:-N, ...] 
    E_ts_ = _dataset[zone]['forecasts'][-N:, ...]
    #print(E_tr_.min(), E_tr_.max())
    #print(E_ts_.min(), E_ts_.max())
    #print(E_tr_.shape, E_ts_.shape)

    x_tr_ = _dataset[zone]['neighbors'][:-N, ...]
    x_ts_ = _dataset[zone]['neighbors'][-N:, ...]
    #print(x_tr_.shape, x_ts_.shape)

    dt_ = np.linspace(0, dt_tr_.shape[-1] - 1, dt_ts_.shape[-1])*12*5
    #print(dt_.shape)

    return F_tr_, F_ts_, E_tr_, E_ts_, d_tr_, d_ts_, datetime_tr_, datetime_ts_, x_tr_, x_ts_, dt_

def _run_ffc(process_, _DATA, hyper_, time, parameter, value, N_test_days = 360):

    region, day = process_
    hyper_p_ = hyper_.copy()
    hyper_p_.loc[parameter, time] = value

    file_name = f'{region}-{day}-{time}'

    # Get data for this region
    (F_tr_, F_ts_, 
    E_tr_, E_ts_, 
    d_tr_, d_ts_, 
    t_tr_, t_ts_, 
    X_tr_, X_ts_, dt_) = _load_dataset(_DATA['Dataset'], region, N_test_days)

    _fdu = functional_dynamic_update({'temporal': 'seasonal', 'spatial': 'graph'}, region)

    # Get functional training set
    F_tr_p_ = F_tr_[:, time, :]
    E_tr_p_ = np.concatenate([np.zeros((E_tr_.shape[0], 24)), E_tr_[:, time, :]], axis = 1)

    _fdu.fit(F_tr_p_, E_tr_p_, dt_, 
             X_ = X_tr_, 
             t_ = d_tr_)

    # Get functional predictors for a given test
    f_ = F_ts_[day, time, :24]
    f_hat_ = F_ts_[day, time, 24:]
    e_ = E_ts_[day, time, :]
    e_p_ = np.concatenate([np.zeros(24,), e_, ], axis = 0)

    # Forecasting update    
    M_ = _fdu.predict(f_, e_p_, X_ts_[day], d_ts_[day],
                      clique_order = hyper_p_.loc['clique_order'][time],
                      forget_rate_f = hyper_p_.loc['forget_rate_f'][time],
                      forget_rate_e = hyper_p_.loc['forget_rate_e'][time],
                      length_scale_f = hyper_p_.loc['length_scale_f'][time],
                      length_scale_e = hyper_p_.loc['length_scale_e'][time],
                      lookup_rate = hyper_p_.loc['lookup_rate'][time],
                      trust_rate = hyper_p_.loc['trust_rate'][time],
                      nu = hyper_p_.loc['nu'][time],
                      gamma = hyper_p_.loc['gamma'][time],
                      xi = hyper_p_.loc['xi'][time],
                      kappa = hyper_p_.loc['kappa'][time],
                      p_fusion = hyper_p_.loc['p_fusion'][time])

    # Confidence bands from marginal empirical density function
    f_median_, _upper, _lower = _fdu.ecdf_confidence_bands(M_, alpha_)

    # Scoring rules
    ES = _energy_score(M_, f_hat_)
    WIS = _weighted_empirical_interval_score(f_hat_, f_median_, _lower, _upper, alpha_).sum()
    PIT = _empirical_PIT(f_hat_, M_.T, seed = 1234)

    # Error metrics
    RMSE = np.sqrt(np.mean((_fdu.f_focal_ - e_)**2))
    MBE  = np.mean(f_hat_ - _fdu.f_focal_)

    if not np.all(np.isfinite(np.array([WIS, RMSE, MBE, ES], dtype=float))):
        print(file_name)

    # Collect scoring rules
    PSR_ = np.array([time, region, day, parameter, value, WIS, RMSE, MBE, ES])

    return PSR_, PIT

# Run FFC experiments in parallel with MPI
def _run_parallel_mpi(_DATA, hyper_, processes_, parameter, value, time):

    # Broadcast inputs (only needed if not already shared)
    _DATA = comm.bcast(_DATA, root=0)
    hyper_ = comm.bcast(hyper_, root=0)

    _func = partial(_run_ffc,
                    _DATA=_DATA,
                    hyper_=hyper_,
                    time=time,
                    parameter=parameter,
                    value=value)

    # Split work and compute local processes
    _local_results = [_func(process) for process in np.array_split(processes_, size)[rank]]

    if _local_results:
        psr_list, pit_list = zip(*_local_results)
        local_psr = np.stack(psr_list)
        local_pit = np.stack(pit_list)
    else:
        local_psr = np.empty((0,))
        local_pit = np.empty((0,))

    # Gather all local results
    all_psr = comm.gather(local_psr, root=0)
    all_pit = comm.gather(local_pit, root=0)

    # Only root processes all gathered result
    if rank == 0:
        PSR_ = np.concatenate(all_psr, axis=0)
        PIT_ = np.concatenate(all_pit, axis=0)
    else:
        PSR_, PIT_ = None, None

    return PSR_, PIT_

# Run FFC experiments in parallel with MP
def _run_parallel_mp(_DATA, hyper_, processes_, time, parameter, value):

    PSR_ = []
    PIT_ = []

    with mp.Pool(processes = 32) as pool:

        # Launch experiments across pool of workers
        results_ = pool.map(partial(_run_ffc, 
                                    _DATA = _DATA, 
                                    hyper_ = hyper_, 
                                    time = time, 
                                    parameter = parameter, 
                                    value = value), processes_)

        psr_list, pit_list = zip(*results_)

        PSR_.append(np.stack(psr_list))
        PIT_.append(np.stack(pit_list))
            
    return np.concatenate(PSR_, axis = 0), np.concatenate(PIT_, axis = 0)

# Run FFC experiments in serie
def _run_serial(_DATA, hyper_, processes_, time, parameter, value):

    psr_list = []
    pit_list = []

    for process in processes_:
        #print(process)
        psr_, pit_ = _run_ffc(process,
                              _DATA=_DATA,
                              hyper_=hyper_,
                              time=time,
                              parameter=parameter,
                              value=value)
        
        psr_list.append(psr_)
        pit_list.append(pit_)

    PSR_ = np.stack(psr_list)
    PIT_ = np.stack(pit_list)

    return PSR_, PIT_

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


# Calibration experiments setup
resource = sys.argv[1]
method = sys.argv[2] 
time = int(sys.argv[3])
print(f'Resource: {resource} Method: {method} Horizon: {time}')

print(mp.cpu_count())
# Zones in the calibration experiments
zones_ = [0, 1, 2, 3, 4]
# Significance levels for the confidence intervals
alpha_ = [0.1, 0.2, 0.3, 0.4]

# Load dataset
with open(path_to_data + "/processed_ERCOT_wind_data.pkl", "rb") as f:
    _DATA = pickle.load(f)
regions_ = list(_DATA['Graph'].keys())

## LOAD HYPERPARAMETERS
hyper_ = pd.read_csv(path_to_param + f'/{resource}/{resource}-{method}-hyper-agg.csv')
hyper_ = hyper_.set_index("parameter")
hyper_.columns = hyper_.columns.astype(int)

# Hyperparameters for the functional forecast dynamic update:
_params = {
    'clique_order': [0, 1, 2],
    'forget_rate_f': [0.0625, 0.125, 0.25, 0.5, 1., 2., 3., 4., 5., 6., 7., 8.],
    'forget_rate_e': [0.25, 0.5, 1., 2., 4., 8., 16., 32., 64., 128., 256., 512.],
    'length_scale_f':[0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5],
    'length_scale_e': [0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5],
    'lookup_rate': [0.5, 1., 2., 4., 8., 16., 32., 64., 128., 256., 512., 1028.],
    'trust_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9],
    'nu': [1., 2., 3., 4., 5., 6., 8., 10., 12., 14., 16., 18],
    'gamma': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
    'xi':[0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975],
    'kappa': [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 400],
    'p_fusion': [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9],
}

hyper_.loc['clique_order', time] = 0

parameters_ = ['forget_rate_f', 
               'forget_rate_e', 
               'lookup_rate', 
               'length_scale_f', 
               'length_scale_e', 
               'trust_rate', 
               'nu', 
               'xi', 
               'gamma', 
               'kappa', 
               'p_fusion']

all_PSR_ = []
all_KS_ = []
N_iter = 2
mins_ = []
for iter in range(N_iter):
    print(hyper_)

    KS_ = []
    df_PSR_ = []
    df_KS_ = []
    values_ = []
    parameter = np.random.choice(parameters_)
    for i, value in enumerate(_params[parameter]):
        print(f'Iter.: {iter + 1}-{N_iter} Val.: {i + 1}-{len(_params[parameter])} Param.: {parameter} {value}')
        
        try:
            # # Run FFC experiments in parallel
            # PSR_, PIT_ = _run_parallel_mp(_DATA, hyper_, 
            #                               processes_ = [(region, j) for region in regions_ for j in range(18)], 
            #                               parameter = parameter,
            #                               value = value,
            #                               time = time)
            
            # Run FFC experiments in parallel
            PSR_, PIT_ = _run_serial(_DATA, hyper_, 
                                     processes_ = [(region, j) for region in regions_ for j in range(180)], 
                                     parameter = parameter,
                                     value = value,
                                     time = time)

            # Calculate aggregated PIT
            KS_.append(np.array([_KS(PIT_[:, j:(j + 1)].flatten()) for j in [0, 4, 8, 12, 16]]).mean())

            # Collect proper scoring rules
            df_PSR_.append(PSR_)
            values_.append(value)

            # Collect KS scores
            df_KS_.append([time, parameter, value] + [_KS(PIT_[:, j:(j + 1)].flatten()) for j in [0, 4, 8, 12, 16]])

        except Exception as e:
            #print(f"Error for asset={asset}, day={day}, file={file_name}")
            print(f"Exception: {e}")
            print(' Numerical Error (skipping parameter...)')

    # Find optimal parameter
    y_ = np.array(KS_)
    x_ = np.array(values_)
    # print(y_)
    # print(x_)
    # print(_truncated_quadratic_min(x_, y_))
    #hyper_.loc[parameter, time] = _truncated_quadratic_min(x_, y_)[0]
    hyper_.loc[parameter, time] = x_[y_.argmin()]

    mins_.append(y_.min())
    print(y_.min(), x_[y_.argmin()])

    # Scores to dataframe
    df_PSR_ = pd.DataFrame(np.concatenate(df_PSR_, axis = 0), columns = ['time', 
                                                                         'region', 
                                                                         'day', 
                                                                         'parameter',
                                                                         'value',
                                                                         'WIS', 
                                                                         'RMSE', 
                                                                         'MBE', 
                                                                         'ES'])

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

all_PSR_ = pd.concat(all_PSR_, axis = 0)
all_KS_ = pd.concat(all_KS_, axis = 0)

print(mins_)
# Caculate average KS score across intervals
all_KS_['KS'] = (all_KS_['KS1'] + all_KS_['KS2'] + all_KS_['KS3'] + all_KS_['KS4'] + all_KS_['KS5'])/5.
print(all_PSR_.shape, all_KS_.shape)

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

all_KS_['time']  = pd.to_numeric(all_KS_['time'], errors='coerce')
all_KS_['value']  = pd.to_numeric(all_KS_['value'], errors='coerce')

# Aggregate score across samples
all_PSR_ = all_PSR_.groupby(['time',
                             'iteration',
                             'parameter', 
                             'value']).agg({'WIS': 'median',
                                            'ES': 'median',
                                            'RMSE': 'median',
                                            'MBE': 'median'}).reset_index(drop = False)

# Merge dataframes to have a single scoring rules dataframe
scores_ = all_PSR_.merge(all_KS_,
                         on=['iteration', "parameter", "value", "time"],
                         how="left")

scores_['resource'] = resource
scores_['method'] = method

# Overwrite the CSV with the updated data
# scores_.to_csv(path_to_validation + f'/{resource}/{resource}-{method}-zone-validation-{time}-0.csv', index = False)
# hyper_.to_csv(path_to_param + f'/{resource}/{resource}-{method}-zone-hyper-{time}-0.csv', index = False)