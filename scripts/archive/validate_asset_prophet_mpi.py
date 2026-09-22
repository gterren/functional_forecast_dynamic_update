"""
Naive industry baseline for the intra-day functional dynamic update (FDU)
experiments: pooled per-interval Prophet + persistence completion.
================================================================================

Companion to ``validate_asset_ffc_mpi.py``. Same CLI, same MPI layout, same
output CSV schema, so the two result sets drop straight into the same tables.

This is deliberately the forecast an operations team would actually stand up in
an afternoon with an off-the-shelf library. There is no spatial model, no
analog pool, no hyperparameter search and no calibrated confidence region: every
number that is not estimated by Prophet itself is a fixed constant declared in
the SETTINGS block below.

The baseline
------------
  Model      One Prophet model per intraday interval, pooled across assets.
             For interval s the training series is daily-frequency - one point
             per (training day, asset) at that clock time - so Prophet sees a
             288-way decomposition of the fleet: interval 0 has its own trend
             and yearly seasonality, interval 1 has its own, and so on. This is
             the standard "one model per delivery hour" layout used across the
             industry for day-ahead energy forecasting, and it sidesteps the
             thing a single 5-minute Prophet is worst at: a diurnal profile
             whose shape changes across the year. Yearly seasonality only,
             flat trend, everything else left at library defaults.

  Inputs     The pooled history (training year) and, at test time, the observed
             part of the day, f(0:tau). The day-ahead forecast is never used -
             the day-ahead arrays are not even placed in the `_data` dict, so no
             experiment function can reach them by accident.

  Completion The observed part of the day is used exactly the way a practitioner
             uses it: persistence of the most recent error. The bias measured
             over the last PERSIST_WINDOW observed intervals is carried forward
             unchanged over the whole remaining horizon - multiplicative for
             solar (a clear-sky-index persistence), additive for wind. No decay,
             no spread rescaling, no tuning. One line, one fixed constant.

  Bands      Prophet's own predictive quantiles at each nominal alpha, i.e.
             exactly what `yhat_lower` / `yhat_upper` give the user. The
             posterior predictive draws are kept as an ensemble so the same
             ensemble scores as the FDU model (ES, PIT/KS, WIS) can be computed,
             but no depth envelope, no projection envelope and no calibration
             stage: alpha means what alpha usually means and nothing is tuned
             against the validation set.

Because nothing is fitted per test case, the whole experiment is one pass over
the test assets. The 288 Prophet fits are cached once per resource.

Reporting conventions follow ``region_comparison_protocol.md`` and
``finding_zone_rmse_pointforecast_mixup.md``:
  * every deterministic-error row carries a ``point_forecast`` label and both
    point summaries are scored on every row;
  * band width (mean and median) accompanies every coverage number, and median
    FIS accompanies mean FIS;
  * shapes are asserted before any error is computed;
  * failures are counted and reported per rank, never swallowed;
  * an un-aggregated per-(asset, day) file is written so the baseline can be
    paired against the FDU results for Diebold-Mariano / block-bootstrap tests;
  * for solar the daylight-restricted coverage scores are reported next to the
    unrestricted ones; for wind the fraction of the horizon at the {0, 1}
    boundary is reported.

Usage
-----
    mpirun -n <N> python validate_asset_prophet_mpi.py \
        <resource> <method> <time> <init> <lambda_0> <unbiased> <description>

    resource     'wind' | 'solar'
    method       tag used in output file names, e.g. 'prophet'
    time         cutoff tau in 5-minute intervals (wind 72/144/216,
                 solar 120/132/144/168)
    init         run index, written to the 'initialization' / 'iteration'
                 columns. The baseline is deterministic apart from the Prophet
                 sampling seed, so repeated inits measure Monte-Carlo noise in
                 the predictive draws only.
    lambda_0     accepted and ignored - there is no objective to weight. Kept so
                 the launch scripts are identical to the FDU ones.
    unbiased     0 | 1, passed to the loader for parity. It only affects the
                 day-ahead arrays, which this baseline does not use.
    description  free-form tag used in output file names

Environment
-----------
    pip install prophet
"""

import os, sys, logging, datetime

sys.path.append('/home/gterren/dynamic_update/functional_forecast_dynamic_update/')

from pathlib import Path

import pandas as pd
import numpy as np
import pickle as pkl

from mpi4py import MPI
from functools import partial
from datetime import datetime
from time import sleep

from prophet import Prophet

from src import loader
from src.utils import (_KS,
                       _weighted_interval_score,
                       _simultaneous_coverage,
                       _coverage_score,
                       _interval_score,
                       _empirical_PIT,
                       _energy_score)

# Silence the Stan backend; 288 fits would otherwise be 288 pages of log.
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)

VALIDATION = Path('/home/gterren/dynamic_update/validation')
PARAM = Path('/home/gterren/dynamic_update/params')
DATA = Path('/home/gterren/dynamic_update/data')
CACHE = PARAM / 'prophet'

# -------------------------------- SETTINGS ------------------------------------
#
# Every tunable of this baseline is on this page, and none of it is fitted to
# the validation set. That is the point: the comparison is against something a
# practitioner would deploy without a study behind it.

# Number of intervals per day (5-minute resolution).
T_DAY = 288

# Size of the Prophet posterior predictive ensemble. Only affects Monte-Carlo
# noise in the reported scores, not the method.
N_SAMPLES = 200

# Yearly Fourier order for the per-interval models. Prophet's own default is 10.
#
# Prophet logs "yearly seasonality is enabled with less than 730 days of
# history" once per fit, because the training year is a single cycle. The
# warning is about trend/seasonality confounding, and growth='flat' removes the
# trend entirely - there is one constant level and one yearly cycle to identify
# from 365 days, which is well posed. The warning is expected and benign here;
# it is not a reason to disable yearly seasonality.
FOURIER_YEARLY = 10

# Pool the training series across assets into one model per interval. This is
# what makes the baseline naive in the intended way: no spatial model, no
# per-site tuning, one fleet-level shape that the persistence step then adapts
# to the site. Set False to fit the per-date fleet mean instead (still 288
# fits, but one averaged series per interval rather than the full pool).
POOL_ASSETS = True

# Persistence completion: the bias measured over the last PERSIST_WINDOW
# observed intervals (12 x 5 min = the last hour) is carried forward unchanged.
# Multiplicative for solar - the ratio of observed to expected output is the
# practitioner's clear-sky index, and it is the quantity that persists - and
# additive for wind, where output is not bounded by a deterministic envelope.
PERSIST_WINDOW = 12
CORRECTION = {'solar': 'ratio', 'wind': 'offset'}

# Guard rails on the multiplicative correction, so an almost-zero denominator at
# dawn or dusk cannot produce an absurd factor. Not tuned; just sane bounds.
RATIO_CLIP = (0.2, 5.0)

# An interval whose training series is identically zero (solar, overnight) or too
# short to fit gets a degenerate zero ensemble instead of a Prophet fit.
ZERO_TOL = 1e-6
MIN_FIT_ROWS = 30

# ------------------------------------------------------------------------------


# load or create DataFrame
def _read_csv_safe(path):

    if os.path.exists(path):

        while os.path.getsize(path) == 0:
            sleep(1)

        return pd.read_csv(path)
    else:
        return pd.DataFrame()


# Map the loader's day coordinate onto wall-clock day timestamps.
#
# The loader hands back `t_tr_` / `t_ts_` as a numeric day coordinate (the FDU
# script uses it as `np.absolute(t_tr_ - day) < 7`). The per-interval models are
# daily-frequency, so a day index is all they need: day d becomes Jan 1 of
# `year` + d days. If your loader is later changed to return datetime64, this is
# the single place to adapt.
def _day_index(t_, year):

    t_ = np.asarray(t_)

    if np.issubdtype(t_.dtype, np.datetime64):
        return pd.DatetimeIndex(pd.to_datetime(t_)).normalize()

    return pd.DatetimeIndex(
        pd.Timestamp(f'{year}-01-01') + pd.to_timedelta(t_.astype(float), unit='D')
    )


# ---------------------- STAGE 0: FIT ONE PROPHET PER INTERVAL ------------------

# Fit the model for one intraday interval and return its predictive ensemble
# over the test year.
#
# Training rows for interval s are the (day, asset) curves of the training pool
# read at that interval: a daily-frequency series with n_assets observations per
# date when pooled. Prophet is perfectly happy
# with repeated timestamps - it is a regression, and the repeats simply give the
# yearly seasonality more to average over at each date.
def _fit_interval(s):

    part_path = CACHE / f'{resource}_{YEAR_TS}_int{s:03d}_S{N_SAMPLES}.npy'

    if part_path.exists():
        return part_path

    y_ = F_tr_[:, s]                                            # (n_curves_tr,)

    if POOL_ASSETS:
        # The pool already holds one row per (day, asset), so pooling across
        # assets is just taking it as it comes: repeated dates, one y per
        # (date, asset). No reshape, and nothing assumed about whether the
        # pool is stored day-major or asset-major.
        ds_ = ds_tr_
    else:
        # Per-date fleet mean. Grouping by the day coordinate is order-agnostic
        # too, so it survives any change in how the loader stacks the pool.
        _by_day = (pd.DataFrame({'ds': ds_tr_, 'y': y_.astype(float)})
                     .groupby('ds', sort = True)['y'].mean())
        ds_ = pd.DatetimeIndex(_by_day.index)
        y_ = _by_day.to_numpy()

    _frame = pd.DataFrame({'ds': ds_, 'y': y_.astype(float)}).dropna()

    # Solar overnight intervals are identically zero all year. Prophet scales y
    # by its absolute max and older versions do not guard the zero case, so skip
    # the fit and emit a degenerate zero ensemble - which is also the physically
    # right answer, and halves the number of fits for solar.
    if len(_frame) < MIN_FIT_ROWS or float(np.nanmax(np.absolute(_frame['y']))) < ZERO_TOL:
        np.save(part_path, np.zeros((len(ds_ts_), N_SAMPLES), dtype=np.float32))
        return part_path

    _model = Prophet(
        growth = 'flat',
        yearly_seasonality = FOURIER_YEARLY,
        weekly_seasonality = False,
        daily_seasonality = False,
        uncertainty_samples = N_SAMPLES,
    )

    _model.fit(_frame)

    Y_ = _model.predictive_samples(pd.DataFrame({'ds': ds_ts_}))['yhat']

    np.save(part_path, Y_.astype(np.float32))                   # (n_days_ts, S)

    return part_path


# Assemble the per-interval files into one (n_days_ts, T_DAY, N_SAMPLES) cube.
def _assemble_cube():

    cube_path = CACHE / f'{resource}_{YEAR_TS}_cube_S{N_SAMPLES}.npy'

    if cube_path.exists():
        return cube_path

    cube_ = np.lib.format.open_memmap(
        cube_path,
        mode = 'w+',
        dtype = np.float32,
        shape = (len(ds_ts_), T_DAY, N_SAMPLES),
    )

    for s in range(T_DAY):
        cube_[:, s, :] = np.load(CACHE / f'{resource}_{YEAR_TS}_int{s:03d}_S{N_SAMPLES}.npy')

    cube_.flush()
    del cube_

    return cube_path


_CUBE = [None]

def _load_cube():

    if _CUBE[0] is None:
        _CUBE[0] = np.load(
            CACHE / f'{resource}_{YEAR_TS}_cube_S{N_SAMPLES}.npy', mmap_mode = 'r'
        )

    return _CUBE[0]


# ------------------------- PERSISTENCE COMPLETION ------------------------------

# Complete the day from the partial observed curve.
#
#   Y_       (T_DAY, S)  Prophet's predictive draws for the whole day
#   f_obs_   (tau,)      observed curve up to the cutoff
#
# The bias over the last PERSIST_WINDOW observed intervals is carried forward
# unchanged: no decay, no spread rescaling, no fitted parameter. Returns the
# completed ensemble (S, H) over the forecast horizon.
def _complete(Y_, f_obs_, time, cap, mode, eps = 1e-6):

    Y_ = np.asarray(Y_, dtype=float)

    H = Y_.shape[0] - time
    if H <= 0:
        raise ValueError(f'cutoff {time} leaves no horizon')

    w = int(max(1, min(PERSIST_WINDOW, time)))

    m_in_ = Y_[time - w:time, :].mean(axis = 1)                 # expected, observed window
    f_in_ = np.asarray(f_obs_, dtype=float)[time - w:time]      # actual, observed window

    M_ = Y_[time:, :].T                                         # (S, H)

    if mode == 'ratio':
        den = float(np.sum(m_in_))
        r = float(np.sum(f_in_)) / den if den > eps * w else 1.
        M_ = M_ * float(np.clip(r, *RATIO_CLIP))

    elif mode == 'offset':
        M_ = M_ + float(np.mean(f_in_ - m_in_))

    else:
        raise ValueError(f"unknown correction mode '{mode}'")

    return np.clip(M_, 0., cap)


# ------------------------------ BANDS AND SCORING ------------------------------

# Prophet's own predictive interval: the marginal quantiles of the draws at each
# horizon. This is what `yhat_lower` / `yhat_upper` report, and it is the only
# region the baseline offers - alpha here is a probability, not a tuned knob.
def _prophet_interval(M_, alpha_):

    M_ = np.asarray(M_, dtype=float)

    f_median_ = np.median(M_, axis = 0)

    _upper, _lower = {}, {}
    for alpha in alpha_:
        _lower[f'{alpha}'] = np.quantile(M_, alpha / 2., axis = 0)
        _upper[f'{alpha}'] = np.quantile(M_, 1. - alpha / 2., axis = 0)

    return f_median_, _upper, _lower


# Number of excursions and mean excursion length outside a band - the diagnostic
# that explains the FCS/SCP gap (region_comparison_protocol.md Sec. 4).
def _excursion_stats(f_, lower_, upper_):

    out_ = (f_ < lower_) | (f_ > upper_)

    if not out_.any():
        return 0, 0.

    d_ = np.diff(out_.astype(int))
    n_runs = int(np.sum(d_ == 1)) + int(out_[0])

    return n_runs, float(np.sum(out_) / max(n_runs, 1))


# Deterministic errors for one point summary against the realisation.
def _point_errors(f_hat_, f_):

    f_ = np.asarray(f_, dtype=float)
    f_hat_ = np.asarray(f_hat_, dtype=float)

    # finding_zone_rmse_pointforecast_mixup.md Cause 3: a (H,) minus (H, 1)
    # silently broadcasts to (H, H) and reports the curve's dispersion as an
    # error. The prescribed fix is to ravel a column vector - but only a column
    # vector: anything with a genuine second dimension is a real mix-up.
    assert f_.ndim == 1 or (f_.ndim == 2 and 1 in f_.shape), (
        f'point forecast has shape {f_.shape}; expected a single curve'
    )

    f_ = f_.ravel()
    f_hat_ = f_hat_.ravel()

    assert f_.shape == f_hat_.shape, f'point forecast {f_.shape} vs actual {f_hat_.shape}'

    return (float(np.sqrt(np.mean((f_hat_ - f_)**2))),
            float(np.mean(np.absolute(f_hat_ - f_))),
            float(np.mean(f_hat_ - f_)))


# One row of the probabilistic results table.
def _score_row(time, asset, day, alpha, f_hat_, lo_, up_, mask_):

    assert lo_.shape == f_hat_.shape and up_.shape == f_hat_.shape, (
        f'band {lo_.shape}/{up_.shape} vs actual {f_hat_.shape}'
    )

    FIS = _interval_score(f_hat_, lo_, up_, alpha).mean()
    FCS = _coverage_score(f_hat_, lo_, up_)
    SCP = _simultaneous_coverage(f_hat_, lo_, up_)

    width_ = up_ - lo_

    if mask_.any():
        FCS_m = _coverage_score(f_hat_[mask_], lo_[mask_], up_[mask_])
        SCP_m = _simultaneous_coverage(f_hat_[mask_], lo_[mask_], up_[mask_])
        width_mean_m = float(np.mean(width_[mask_]))
    else:
        FCS_m = SCP_m = width_mean_m = np.nan

    n_exc, len_exc = _excursion_stats(f_hat_, lo_, up_)

    return [time, asset, day, alpha, 'ECDF',
            FIS, FCS, SCP,
            float(np.mean(width_)), float(np.median(width_)),
            FCS_m, SCP_m, width_mean_m,
            n_exc, len_exc]


PROB_COLUMNS = ['time', 'asset', 'day', 'alpha', 'distance',
                'FIS', 'FCS', 'SCP',
                'width_mean', 'width_median',
                'FCS_day', 'SCP_day', 'width_mean_day',
                'n_excursions', 'len_excursion']

DET_COLUMNS = ['time', 'asset', 'day', 'distance', 'point_forecast',
               'RMSE', 'MAE', 'MBE', 'frac_boundary']


# ------------------------------- EXPERIMENTS ----------------------------------

# One test case: complete the day from the partial curve and score everything.
def _run_prophet(process_, _data, time):

    asset, day = process_

    file_name = f'{asset}-{day}-{time}'

    F_tr_, F_ts_ = _data['ac']

    try:
        cube_ = _load_cube()

        Y_ = np.asarray(cube_[day, :, :], dtype=float)          # (T_DAY, S)

        f_obs_ = F_ts_[day, :time, asset]
        f_hat_ = F_ts_[day, time:, asset]

        M_ = _complete(Y_, f_obs_, time, cap = CAP, mode = CORRECTION[resource])

        assert M_.shape[1] == f_hat_.shape[0], (M_.shape, f_hat_.shape)

        # Prophet's own predictive interval.
        f_median_, _upper, _lower = _prophet_interval(M_, ALPHAS)
        f_mean_ = M_.mean(axis = 0)

        # Ensemble scores, identical to the FDU script.
        pit_ = _empirical_PIT(f_hat_, M_.T, seed = 1234)

        es = _energy_score(M_, f_hat_)
        wis = _weighted_interval_score(f_hat_, f_median_, _lower, _upper, ALPHAS).mean()
        rmse, mae, _ = _point_errors(f_hat_, f_median_)

        psr_ = np.array([time, asset, day, es, wis, rmse, mae])

        # Band rows, one per alpha, at the nominal level.
        mask_ = EVAL_MASK_(f_hat_)
        prob_ = [
            _score_row(time, asset, day, alpha,
                       f_hat_, _lower[f'{alpha}'], _upper[f'{alpha}'], mask_)
            for alpha in ALPHAS
        ]

        # finding_zone_rmse_pointforecast_mixup.md Cause 1: label the point
        # summary each error row belongs to, and score both of them.
        frac_boundary = float(np.mean((f_hat_ <= 1e-6) | (f_hat_ >= CAP - 1e-6)))

        det_ = []
        for curve_, label in [(f_median_, 'median'), (f_mean_, 'mean')]:
            r, m, b = _point_errors(f_hat_, curve_)
            det_.append([time, asset, day, 'ECDF', label, r, m, b, frac_boundary])

        # Per-horizon ensemble spread.
        stat_ = pd.DataFrame(np.std(M_, axis = 0)).T
        stat_.columns = [f'H{str(i+1).zfill(2)}' for i in range(len(stat_.columns))]
        stat_['time'] = time
        stat_['region'] = asset
        stat_['day'] = day

        func_ = []
        for curve_, label in [(f_median_, 'median'), (f_mean_, 'mean'), (f_hat_, 'actual')]:
            _df = pd.DataFrame([curve_], columns = [f'H{i:02d}' for i in range(len(curve_))])
            _df['type'] = label
            func_.append(_df)

        func_ = pd.concat(func_, axis = 0)
        func_['time'] = time
        func_['asset'] = asset
        func_['day'] = day

    except Exception as e:
        FAILURES.append((file_name, repr(e)))
        return None

    return pit_, psr_, np.array(prob_, dtype=object), np.array(det_, dtype=object), stat_, func_


# Failures are counted and reported, never swallowed
# (finding_zone_rmse_pointforecast_mixup.md, action 3).
FAILURES = []

def _report_failures(tag, n_local, n_ok):

    if not FAILURES:
        return

    from collections import Counter
    kinds_ = Counter(msg for _, msg in FAILURES)

    print(f'[Rank {RANK}] {tag}: {n_ok}/{n_local} succeeded. Failure modes:', flush = True)
    for msg, n in kinds_.most_common(5):
        print(f'[Rank {RANK}]   {n:5d} x {msg}', flush = True)

    FAILURES.clear()


# Run every test case in parallel and gather on rank 0.
def _run_parallel_mpi(_data, processes_, time):

    _func = partial(_run_prophet, _data = _data, time = time)

    local_ = [[] for _ in range(6)]

    local_processes_ = np.array_split(processes_, SIZE)[RANK]

    n_ok = 0
    for process_ in local_processes_:
        out_ = _func(process_)

        if out_ is not None:
            n_ok += 1
            for k, v in enumerate(out_):
                local_[k].append(v)

    _report_failures(f't={time}', len(local_processes_), n_ok)

    if n_ok == 0:
        local_ = [None] * 6
    else:
        local_ = [np.stack(local_[0], axis = 0),
                  np.stack(local_[1], axis = 0),
                  np.concatenate(local_[2], axis = 0),
                  np.concatenate(local_[3], axis = 0),
                  pd.concat(local_[4], axis = 0),
                  pd.concat(local_[5], axis = 0)]

    gathered_ = [COMM.gather(x, root = 0) for x in local_]

    if RANK != 0:
        return [None] * 6

    out_ = []
    for k, g_ in enumerate(gathered_):
        keep_ = [x for x in g_ if x is not None]
        if not keep_:
            return [None] * 6
        out_.append(pd.concat(keep_, axis = 0) if k >= 4
                    else np.concatenate(keep_, axis = 0))

    return out_


# ================================== MAIN ======================================

resource = sys.argv[1]
method = sys.argv[2]
time = int(sys.argv[3])
init = int(sys.argv[4])
unbiased = bool(int(sys.argv[5]))
description = sys.argv[6]
print(resource, method, time, init, unbiased, description)

# Years the loader's day coordinate refers to.
YEAR_TR = 2017
YEAR_TS = 2018

ALPHAS = [0.1, 0.2, 0.3, 0.4]
FIT = True                 # Stage 0: fit the 288 per-interval models and cache

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
        exit('Invalid time horizon. Must be one of [72, 144, 216].')
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
        exit('Invalid time horizon. Must be one of [120, 132, 144, 168].')
else:
    exit("Invalid resource. Must be one of ['wind', 'solar'].")

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
print(f'[Process {RANK}-{SIZE}]')


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
    path_to_training_data = DATA / f'preprocessed_{resource}_2017.pkl',
    path_to_testing_data = DATA / f'preprocessed_{resource}_2018.pkl',
    T = T_DAY,
)

# The loader returns the TRAINING set as a stacked pool of curves --
# (n_curves, T), one row per (day, asset) pair -- with `t_tr_` carrying a
# matching per-curve day coordinate. Only the TEST set keeps the
# (n_days, T, n_assets) layout, which is what `F_ts_[day, :time, asset]`
# indexes below. This is the same convention `validate_asset_ffc_mpi.py`
# relies on (`F_tr_[idx_days_, :]` with `idx_days_ = |t_tr_ - day| < 7`).
assert F_tr_.ndim == 2, (
    f'F_tr_ is expected to be a stacked curve pool (n_curves, T); '
    f'got {F_tr_.shape}.'
)
assert F_ts_.ndim == 3, (
    f'F_ts_ is expected to be (n_days, T, n_assets); got {F_ts_.shape}.'
)
assert len(t_tr_) == F_tr_.shape[0], (
    f'`t_tr_` must carry one day coordinate per training curve; got '
    f'{len(t_tr_)} coordinates for {F_tr_.shape[0]} curves.'
)

# The day-ahead forecast arrays are deliberately NOT placed in `_data`: the
# baseline is defined by not having access to them, and keeping them out makes
# that impossible to violate by accident.
_data = {
    'ac': [F_tr_, F_ts_],
    'dates': [t_tr_, t_ts_],
    'grid': [dt_, dx_]
}

ds_tr_ = _day_index(t_tr_, YEAR_TR)
ds_ts_ = _day_index(t_ts_, YEAR_TS)

# Physical upper bound used to clip the completed ensemble. Taken from the
# training data, not the test day.
CAP = float(max(np.nanmax(F_tr_), 1e-6))

# Evaluation mask: daylight for solar (from the climatological training profile,
# so it uses no test-day information), everything for wind.
if resource == 'solar':
    _profile = np.nanmean(F_tr_, axis = 0)                              # (T,)
    _daylight = _profile > (0.01 * np.nanmax(_profile))

    def EVAL_MASK_(f_hat_):
        return _daylight[T_DAY - len(f_hat_):]
else:
    def EVAL_MASK_(f_hat_):
        return np.ones(len(f_hat_), dtype=bool)

# The baseline has nothing to tune, so there is no validation split to protect:
# it is evaluated on the same held-out assets as the FDU model's test stage.
processes_test_ = [(asset, j) for asset in range(10, 20) for j in range(0, 360)]


## ------------------- STAGE 0: 288 PER-INTERVAL PROPHET FITS -------------------

if FIT:

    if RANK == 0:
        print('----- PROPHET FIT: one model per interval, pooled across assets -----')
        CACHE.mkdir(parents = True, exist_ok = True)

    COMM.Barrier()

    t_0 = datetime.now()
    for s in np.array_split(np.arange(T_DAY), SIZE)[RANK]:
        _fit_interval(int(s))

    print(f'[Rank {RANK}] fitted its intervals in '
          f'{(datetime.now() - t_0).total_seconds():.0f}s', flush = True)

    COMM.Barrier()

    if RANK == 0:
        _assemble_cube()
        print('----- PROPHET FIT DONE -----', flush = True)

    COMM.Barrier()


## ------------------------------ TEST ------------------------------------------

if RANK == 0:
    print(f'[Resource {resource}][Method {method}][Horizon {time}][Initialization {init}]')
    print(f'  pooled = {POOL_ASSETS}, correction = {CORRECTION[resource]}, '
          f'persistence window = {PERSIST_WINDOW} intervals, no tuning')
    print('----- TEST -----')

pit_, psr_, prob_, det_, stat_, func_ = _run_parallel_mpi(_data, processes_test_, time)

if RANK == 0:

    # ---- aggregated scoring rules and PIT ------------------------------------
    ks_ = np.array([_KS(pit_[:, j:(j + LEAD)].flatten()) for j in INTERVALS])
    ks_labels_ = [f'S{i}' for i in range(len(INTERVALS))]

    row_ = {
        'initialization': init,
        'time': time,

        'ES_test': float(np.mean(psr_[:, -4].astype(float))),
        'WIS_test': float(np.mean(psr_[:, -3].astype(float))),
        'RMSE_test': float(np.mean(psr_[:, -2].astype(float))),
        'MAE_test': float(np.mean(psr_[:, -1].astype(float))),
        'KS_test': float(np.mean(ks_)),

        **{ks_labels_[i] + '_test': float(ks_[i]) for i in range(len(INTERVALS))},

        # Fixed settings, recorded so a results file is self-describing.
        'model': 'prophet-per-interval',
        'pooled_assets': POOL_ASSETS,
        'correction': CORRECTION[resource],
        'persist_window': PERSIST_WINDOW,
        'fourier_yearly': FOURIER_YEARLY,
        'n_samples': N_SAMPLES,
    }

    results_path = VALIDATION / f'{resource}/{resource}_{method}_asset-hyper-{description}.csv'
    results_df = _read_csv_safe(results_path)
    results_df = pd.concat([results_df, pd.DataFrame([row_])], ignore_index = True)
    results_df.to_csv(results_path, index = False)
    print(f'Saved results to {results_path}')

    pit_path = VALIDATION / f'{resource}/{resource}_{method}_asset-PIT_{time}-{description}.csv'
    pit_df = _read_csv_safe(pit_path)

    pit_ = pd.DataFrame(pit_)
    pit_.columns = [f'H{str(i+1).zfill(2)}' for i in range(len(pit_.columns))]
    pit_['initialization'] = init
    pit_['time'] = time

    pit_df = pd.concat([pit_df, pit_], axis = 0, ignore_index = True)
    pit_df.to_csv(pit_path, index = False)
    print(f'Saved PIT to {pit_path}')

    stat_path = VALIDATION / f'{resource}/{resource}_{method}_asset-STATS_{time}-{description}.csv'
    stat_df = _read_csv_safe(stat_path)

    stat_['initialization'] = init
    stat_df = pd.concat([stat_df, stat_], axis = 0, ignore_index = True)
    stat_df.to_csv(stat_path, index = False)
    print(f'Saved STAT to {stat_path}')

    func_path = VALIDATION / f'{resource}/{resource}_{method}_asset-functions_{time}-{description}.csv'
    func_df = _read_csv_safe(func_path)

    func_['initialization'] = init
    func_df = pd.concat([func_df, func_], axis = 0, ignore_index = True)
    func_df.to_csv(func_path, index = False)
    print(f'Saved functions to {func_path}')

    # ---- per-case rows, for paired tests against the FDU results -------------
    percase_path = VALIDATION / f'{resource}/{resource}_{method}_asset-percase_{time}-{description}.csv'

    prob_df = pd.DataFrame(prob_, columns = PROB_COLUMNS)
    for c in ['alpha', 'FIS', 'FCS', 'SCP', 'width_mean', 'width_median',
              'FCS_day', 'SCP_day', 'width_mean_day', 'n_excursions', 'len_excursion']:
        prob_df[c] = pd.to_numeric(prob_df[c], errors = 'coerce')

    percase_df = prob_df.copy()
    percase_df['iteration'] = init
    percase_df['stage'] = 'nominal-ECDF'

    out_df = _read_csv_safe(percase_path)
    out_df = pd.concat([out_df, percase_df], axis = 0, ignore_index = True)
    out_df.to_csv(percase_path, index = False)
    print(f'Saved per-case rows to {percase_path}')

    # ---- aggregated band scores ---------------------------------------------
    env_ = prob_df.groupby(['time', 'alpha']).agg(
        FIS = ('FIS', 'mean'),
        FIS_median = ('FIS', 'median'),
        FCS = ('FCS', 'mean'),
        SCP = ('SCP', 'mean'),
        width_mean = ('width_mean', 'mean'),
        width_median = ('width_median', 'median'),
        FCS_day = ('FCS_day', 'mean'),
        SCP_day = ('SCP_day', 'mean'),
        width_mean_day = ('width_mean_day', 'mean'),
        n_excursions = ('n_excursions', 'mean'),
        len_excursion = ('len_excursion', 'mean'),
    ).reset_index(drop = False)

    env_['score'] = 'nominal'
    env_['distance'] = 'ECDF'
    env_['fraction'] = np.nan          # nothing is calibrated
    env_['iteration'] = init

    env_path = VALIDATION / f'{resource}/{resource}_{method}_asset-envelope-{description}.csv'
    env_df = _read_csv_safe(env_path)
    env_df = pd.concat([env_df, env_], axis = 0, ignore_index = True)
    env_df.to_csv(env_path, index = False)
    print(f'Saved results to {env_path}')

    # ---- aggregated deterministic errors, by point summary -------------------
    det_df = pd.DataFrame(det_, columns = DET_COLUMNS)
    for c in ['RMSE', 'MAE', 'MBE', 'frac_boundary']:
        det_df[c] = pd.to_numeric(det_df[c], errors = 'coerce')

    det_agg_ = det_df.groupby(['time', 'point_forecast']).agg(
        {'RMSE': 'mean', 'MAE': 'mean', 'MBE': 'mean', 'frac_boundary': 'mean'}
    ).reset_index(drop = False)

    det_agg_['score'] = 'nominal'
    det_agg_['distance'] = 'ECDF'
    det_agg_['iteration'] = init

    err_path = VALIDATION / f'{resource}/{resource}_{method}_asset-error-{description}.csv'
    err_df = _read_csv_safe(err_path)
    err_df = pd.concat([err_df, det_agg_], axis = 0, ignore_index = True)
    err_df.to_csv(err_path, index = False)
    print(f'Saved results to {err_path}')

    print('----- DONE -----')
