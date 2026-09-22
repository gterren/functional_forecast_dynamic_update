"""
Day-ahead reference for the intra-day functional dynamic update (FDU)
experiments: the day-ahead forecast re-anchored on the last observation and
damped back toward itself with lead time.
================================================================================

Companion to ``validate_asset_ffc_mpi.py`` and ``validate_asset_prophet_mpi.py``.
Same CLI, same MPI layout, same output CSV schema, so all three result sets drop
into the same tables.

What this is
------------
The forecast that was issued the day before, re-anchored on the last observation
and then damped back toward itself as lead time grows. It is the reference that
isolates the quantity the paper is about: whatever the FDU method gains, it
gains *over this*. It is also the only reference that sits on the day-ahead
information set - the Prophet baseline never sees the day-ahead forecast at all,
so it cannot play this role.

  Point forecast   The day-ahead curve over the remaining horizon, corrected by
                   a persistence anchor that decays geometrically in LEAD TIME:

                       b_tau  = f(tau-1) - e(tau-1)            [wind, additive]
                                f(tau-1) / e(tau-1)            [solar, ratio]
                       w(s)   = exp(beta (tau - s)),  beta = BETA per hour
                       e^(tau + k) = clip(w(k) (e(tau+k) + b_tau)
                                          + (1 - w(k)) e(tau+k), 0, CAP)
                                   = clip(e(tau+k) + b_tau w(k), 0, CAP)
                                     clip(e(tau+k) exp(log(b_tau) w(k)), 0, CAP)

                   The first line is the form the manuscript states: a weighted
                   average of the fully re-anchored curve and the uncorrected
                   day-ahead forecast. The second is the algebraically identical
                   offset form this script implements. The third is the solar
                   analogue, which damps in log space and is therefore NOT a
                   weighted average -- see daref_baseline_design.md.

                   with k = 0 the first forecast interval. At k = 0 the
                   correction is applied in full, so the curve starts on the
                   last observation; as k grows it decays and e^ -> e, so the
                   curve returns to the day-ahead forecast. DECAY_HALFLIFE is
                   the only free quantity in this script; it is fixed a priori
                   at 36 intervals (3 hours) and not tuned.

                   The two ingredients are deliberately kept apart. The ANCHOR
                   says where the corrected curve starts and is pure
                   persistence of the most recent usable observation - not a
                   weighted average over the prefix, which would start the
                   curve somewhere the plant has not been. The DECAY says how
                   fast that anchor is given up, and is the only thing that is
                   a function of lead time. A correction that is constant in
                   lead time is not damped persistence; it is a bias shift, and
                   it never rejoins the forecast it is supposed to reference.

                   The raw day-ahead curve is still scored alongside it under
                   the label 'day-ahead-no-update', so the horizon-composition
                   effect remains readable from the same results file.

  Band             The training-year error climatology OF THE CORRECTED CURVE,
                   dressed onto that curve. For every training curve i the same
                   anchor is taken at its own interval tau-1 and damped by the
                   same phi^k, giving e^_i(s); the pool is
                   d_i(s) = f_i(s) - e^_i(s) for s >= tau. The ensemble is
                   M_k(s) = clip(e^(s) + d_{i_k}(s), 0, CAP) for N_SAMPLES whole
                   curves i_k drawn with replacement from the pool, and the band
                   is its marginal quantiles at each nominal alpha. Dressing the
                   corrected curve with corrected-curve errors is what keeps the
                   band calibrated to the estimator it wraps. Because the
                   correction decays, the pool is tight at short lead and widens
                   into the raw day-ahead climatology at long lead, which is the
                   band shape a re-anchored forecast should have and the one the
                   earlier constant-offset version could not produce.

                   This is still a lookup table, not a model: nothing is fitted,
                   and the constants are DECAY_HALFLIFE, N_SAMPLES and the seed.

  Why whole curves WHOLE-CURVE resampling is the load-bearing choice. Drawing an
                   independent error at each interval would give the same
                   marginal band and destroy the temporal correlation of
                   forecast errors, so SCP would collapse for a reason that has
                   nothing to do with the forecast. Resampling whole error
                   trajectories keeps the correlation structure that the
                   training year actually exhibits. Set DRESS = 'interval' to
                   reproduce the degenerate version if a referee asks.

  Tau dependence   Both the anchor and the error pool depend on tau, so unlike
                   the earlier no-update version of this script the band does
                   sharpen through the day. The pool is rebuilt once per run.


Reporting conventions follow ``region_comparison_protocol.md`` and
``finding_zone_rmse_pointforecast_mixup.md``, plus one addition:

  * ``_point_errors`` is evaluated BOTH unrestricted and on the evaluation mask
    (daylight for solar), and the masked values are written as RMSE_day / MAE_day
    / MBE_day. ``validate_asset_prophet_mpi.py`` applies the mask to the band
    rows only, so its solar RMSE includes the overnight zeros and is not
    comparable to its wind RMSE; backfill that run before putting the two
    resources in one table.
  * four point summaries are scored on every case: the corrected day-ahead
    curve ('day-ahead' - this is the reference), the raw day-ahead curve
    ('day-ahead-no-update' - the null that isolates horizon composition), and
    the dressed ensemble's median and mean, for parity with the other two
    scripts.

Usage
-----
    mpirun -n <N> python validate_asset_daref_mpi.py \
        <resource> <method> <time> <init> <unbiased> <description>

    resource     'wind' | 'solar'
    method       tag used in output file names, e.g. 'daref'
    time         cutoff tau in 5-minute intervals (wind 72/144/216,
                 solar 120/132/144/168)
    init         run index. The reference is deterministic apart from the
                 dressing draw, so repeated inits measure Monte-Carlo noise in
                 the error resample only.
    unbiased     0 | 1, passed to the loader. UNLIKE the Prophet baseline this
                 flag matters here: it selects which day-ahead array is
                 corrected and scored, and which one the error pool is built
                 from. Use the same value as the FDU run being compared
                 against.
    description  free-form tag used in output file names

There is no fitting stage and no cache: one pass over the test assets, after
a single vectorised build of the tau-specific error pool.
"""

import os, sys

sys.path.append('/home/gterren/dynamic_update/functional_forecast_dynamic_update/')

from pathlib import Path

import pandas as pd
import numpy as np

from mpi4py import MPI
from functools import partial
from time import sleep

from src import loader
from src.utils import (_KS,
                       _weighted_interval_score,
                       _simultaneous_coverage,
                       _coverage_score,
                       _interval_score,
                       _empirical_PIT,
                       _energy_score)

VALIDATION = Path('/home/gterren/dynamic_update/validation')
PARAM = Path('/home/gterren/dynamic_update/params')
DATA = Path('/home/gterren/dynamic_update/data')

# -------------------------------- SETTINGS ------------------------------------
#
# Every constant of this reference is on this page and none of it is fitted.

# Number of intervals per day (5-minute resolution).
T_DAY = 288

# THE ONE FREE QUANTITY. Half-life, in 5-minute intervals, of the geometric
# decay applied to the persistence correction AS LEAD TIME GROWS. 36 intervals
# = 3 hours: the correction is applied in full at the first forecast interval,
# is half gone three hours later and has faded to ~1% after twelve. This is what
# makes the reference leave the last observation and rejoin the day-ahead curve
# rather than ride above it for the rest of the day. Fixed a priori, not tuned.
DECAY_HALFLIFE = 48.

# How the correction is anchored at tau. 'last' is persistence of the most
# recent usable observation. Deliberately NOT a weighted average over the
# prefix: the reference has to start ON the last observation, and any smoothing
# over past residuals lags it and starts the curve somewhere the asset has not
# been. Smoothing belongs in lead time, above, not in observation time.
ANCHOR = 'last'

# Offset form by resource. Level persistence for wind, clear-sky-index
# persistence for solar - the same convention as the Prophet benchmark, and a
# resource convention rather than a tuned choice.
OFFSET_BY_RESOURCE = {'wind': 'add', 'solar': 'mul'}

# Guard on the solar ratio, as in the Prophet benchmark. Inactive for wind.
RATIO_CLIP = (0.2, 5.0)

# Size of the dressed ensemble. Only affects Monte-Carlo noise in the reported
# scores, not the reference. Raised from 200: at alpha = 0.1 the 5% and 95%
# quantiles rest on about ten order statistics with 200 draws, which shows up as
# ragged band edges and as Monte-Carlo noise in SCP. Resampling from the error
# pool is cheap, so the extra draws cost almost nothing here. Keep this equal to
# N_SAMPLES in validate_asset_arima_mpi.py.
N_SAMPLES = 1000

# 'curve'    resample whole training error trajectories (default; keeps the
#            temporal correlation of day-ahead errors)
# 'interval' resample independently at each interval (same marginals, no
#            correlation - included only to show what it costs on SCP)
DRESS = 'curve'

# Seed for the dressing draw. Mixed with (asset, day, time) so every case is
# reproducible and no two cases share a draw.
SEED = 20260913

# Drop training curves with any missing value in the day. Kept explicit rather
# than nan-handled downstream so the pool size is reported.
DROP_NAN_CURVES = True

# ------------------------------------------------------------------------------


# --------------------------- DAMPED PERSISTENCE -------------------------------
#
# Two separate things, kept separate on purpose, because conflating them is
# exactly what the previous version of this script got wrong:
#
#   the ANCHOR   one scalar, the day-ahead residual at the last usable
#                observation. It says WHERE the corrected curve starts.
#
#   the DECAY    an exponential weight in LEAD TIME, w(k) =
#                exp(-k ln2 / DECAY_HALFLIFE). It says HOW FAST the corrected
#                curve gives the anchor up and returns to the day-ahead curve.
#
# Exponentially weighting the observed residuals and then carrying one number
# forward unchanged smooths the wrong axis: it produces a constant shift of the
# day-ahead curve that never converges back to it, and whose starting point lags
# the last observation by the smoother's own memory.

# Written as an exponential in lead time, with the half-life in the exponent:
#
#     w(s - tau) = exp( -(s - tau) * ln 2 / h ),      h = DECAY_HALFLIFE
#
# This is the SAME function as the geometric form phi^(s-tau) with
# phi = 2^(-1/h) -- exp and 2** of the same argument, equal to machine
# precision, one parameter either way. It is written exponentially because that
# is how the manuscript states it, and with ln2/h rather than a free rate so
# the reported constant stays the half-life, which is the interpretable one:
# half the correction is gone after h intervals. RATE is the reciprocal
# e-folding time; PHI is kept because the results files record it.
RATE = float(np.log(2.) / DECAY_HALFLIFE)
PHI = float(np.exp(-RATE))

# BETA is the constant the manuscript reports: the same decay expressed per
# hour, so that w(s) = exp(BETA (tau - s)) with the lead in hours. RATE is the
# same number per 5-minute interval, BETA / 12. The half-life is the input here
# because it is the quantity that was chosen; beta is what is quoted.
BETA = float(RATE * 12.)


def _decay(n):
    """Exponential decay weights over n lead intervals.

    Element k is w(k) at lead s - tau = k, so w(0) = 1: the correction is
    applied in full at the first forecast interval, which is what makes the
    reference depart from the last observation. w(DECAY_HALFLIFE) = 1/2 exactly.
    """

    if n <= 0:
        return np.zeros(0, dtype=float)

    return np.exp(-RATE * np.arange(n, dtype=float))


def _anchor(y_, e_):
    """Persistence anchor: the day-ahead residual at the last usable interval.

    y_, e_ : (tau,) realisation and day-ahead forecast over 0:tau. Returns a
    scalar: an additive level offset for wind, a multiplicative ratio for solar,
    both read off the most recent interval at which BOTH are finite. The neutral
    value is returned when nothing usable has been observed, so tau = 0 reduces
    exactly to the un-updated day-ahead curve.
    """

    y_ = np.asarray(y_, dtype=float).ravel()
    e_ = np.asarray(e_, dtype=float).ravel()

    assert y_.shape == e_.shape, (y_.shape, e_.shape)
    assert ANCHOR == 'last', f"unknown ANCHOR mode '{ANCHOR}'"

    neutral = 0. if OFFSET_MODE == 'add' else 1.

    if y_.size == 0:
        return neutral

    ok_ = np.isfinite(y_) & np.isfinite(e_)

    if not ok_.any():
        return neutral

    # The most recent USABLE interval, not simply the last one: a gap at the end
    # of the prefix must fall back to the newest real observation, never to the
    # neutral value, which would silently turn the reference into the null.
    i = int(np.flatnonzero(ok_)[-1])

    if OFFSET_MODE == 'add':
        return float(y_[i] - e_[i])

    if e_[i] <= 1e-6:
        return neutral

    return float(np.clip(y_[i] / e_[i], *RATIO_CLIP))


def _apply_offset(e_, b):
    """Damp the anchor over the remaining horizon and apply it to the curve.

    Additive for wind, and for solar the same thing in log space so that the
    correction decays to a ratio of one rather than to zero. Element k of the
    output is lead k, so k = 0 carries the full anchor: with a locally flat
    day-ahead curve, e^(tau) = e(tau) + f(tau-1) - e(tau-1) = f(tau-1), i.e. the
    reference starts on the last observation.
    """

    e_ = np.asarray(e_, dtype=float)
    w_ = _decay(e_.size)

    if OFFSET_MODE == 'add':
        return np.clip(e_ + b * w_, 0., CAP)

    return np.clip(e_ * np.exp(np.log(max(float(b), 1e-12)) * w_), 0., CAP)



# load or create DataFrame
def _read_csv_safe(path):

    if os.path.exists(path):

        while os.path.getsize(path) == 0:
            sleep(1)

        return pd.read_csv(path)

    return pd.DataFrame()


# Number of excursions and mean excursion length outside a band - the diagnostic
# that explains the FCS/SCP gap (region_comparison_protocol.md Sec. 4).
#
# NOTE: the aggregation of these two columns is a mean of per-forecast ratios,
# so n_excursions * len_excursion does NOT recover the total out-of-band count.
# `frac_out` below is the column to use for that.
def _excursion_stats(f_, lower_, upper_):

    out_ = (f_ < lower_) | (f_ > upper_)

    if not out_.any():
        return 0., 0.

    d_ = np.diff(out_.astype(int))
    n_runs = int(np.sum(d_ == 1)) + int(out_[0])

    return n_runs, float(np.sum(out_) / max(n_runs, 1))


# Deterministic errors for one point summary against the realisation.
def _point_errors(f_hat_, f_):

    f_ = np.asarray(f_, dtype=float)
    f_hat_ = np.asarray(f_hat_, dtype=float)

    # finding_zone_rmse_pointforecast_mixup.md Cause 3: a (H,) minus (H, 1)
    # silently broadcasts to (H, H). Ravel a column vector, reject anything with
    # a genuine second dimension.
    assert f_.ndim == 1 or (f_.ndim == 2 and 1 in f_.shape), (
        f'point forecast has shape {f_.shape}; expected a single curve'
    )

    f_ = f_.ravel()
    f_hat_ = f_hat_.ravel()

    assert f_.shape == f_hat_.shape, f'point forecast {f_.shape} vs actual {f_hat_.shape}'

    return (float(np.sqrt(np.mean((f_hat_ - f_)**2))),
            float(np.mean(np.absolute(f_hat_ - f_))),
            float(np.mean(f_hat_ - f_)))


# Same, restricted to the evaluation mask (daylight for solar).
def _point_errors_masked(f_hat_, f_, mask_):

    if mask_ is None or not mask_.any():
        return np.nan, np.nan, np.nan

    return _point_errors(np.asarray(f_hat_).ravel()[mask_],
                         np.asarray(f_).ravel()[mask_])


# One row of the probabilistic results table.
def _score_row(time, asset, day, alpha, f_hat_, lo_, up_, mask_):

    assert lo_.shape == f_hat_.shape and up_.shape == f_hat_.shape, (
        f'band {lo_.shape}/{up_.shape} vs actual {f_hat_.shape}'
    )

    FIS = _interval_score(f_hat_, lo_, up_, alpha).mean()
    FCS = _coverage_score(f_hat_, lo_, up_)
    SCP = _simultaneous_coverage(f_hat_, lo_, up_)

    width_ = up_ - lo_

    out_ = (f_hat_ < lo_) | (f_hat_ > up_)
    frac_out = float(np.mean(out_))

    if mask_.any():
        FCS_m = _coverage_score(f_hat_[mask_], lo_[mask_], up_[mask_])
        SCP_m = _simultaneous_coverage(f_hat_[mask_], lo_[mask_], up_[mask_])
        width_mean_m = float(np.mean(width_[mask_]))
    else:
        FCS_m = SCP_m = width_mean_m = np.nan

    n_exc, len_exc = _excursion_stats(f_hat_, lo_, up_)

    # Band saturation, measured on the BAND (frac_boundary in the other scripts
    # is measured on the realisation and is a different quantity).
    frac_degenerate = float(np.mean(width_ <= 1e-6))

    return [time, asset, day, alpha, 'ECDF',
            FIS, FCS, SCP,
            float(np.mean(width_)), float(np.median(width_)),
            FCS_m, SCP_m, width_mean_m,
            n_exc, len_exc, frac_out, frac_degenerate]


PROB_COLUMNS = ['time', 'asset', 'day', 'alpha', 'distance',
                'FIS', 'FCS', 'SCP',
                'width_mean', 'width_median',
                'FCS_day', 'SCP_day', 'width_mean_day',
                'n_excursions', 'len_excursion',
                'frac_out', 'frac_degenerate']

DET_COLUMNS = ['time', 'asset', 'day', 'distance', 'point_forecast',
               'RMSE', 'MAE', 'MBE',
               'RMSE_day', 'MAE_day', 'MBE_day',
               'frac_boundary']


# ------------------------------- EXPERIMENTS ----------------------------------

# One test case: score the offset-corrected day-ahead curve, with the band
# dressed from the corrected-curve error pool.
def _run_reference(process_, _data, time):

    asset, day = process_

    file_name = f'{asset}-{day}-{time}'

    F_tr_, F_ts_ = _data['ac']
    E_tr_, E_ts_ = _data['day-ahead_fc']

    try:
        # The realisation and the day-ahead forecast over the remaining horizon.
        f_hat_ = np.asarray(F_ts_[day, time:, asset], dtype=float)
        e_ = np.asarray(E_ts_[day, time:, asset], dtype=float)

        assert e_.shape == f_hat_.shape, (e_.shape, f_hat_.shape)
        assert np.isfinite(e_).all(), 'day-ahead curve contains non-finite values'

        # ---- re-anchor the curve on the observed part of the day -------------
        # The anchor uses only f(0:tau) and e(0:tau); nothing after tau enters.
        b = _anchor(np.asarray(F_ts_[day, :time, asset], dtype=float),
                    np.asarray(E_ts_[day, :time, asset], dtype=float))

        e_corr_ = _apply_offset(e_, b)

        # ---- dress the CORRECTED curve with its own error climatology -------
        rng = np.random.default_rng(
            (SEED * 1_000_003) ^ (int(asset) * 1_000_033 + int(day) * 1_009 + int(time))
        )

        if DRESS == 'curve':
            idx_ = rng.integers(0, D_POOL_.shape[0], N_SAMPLES)
            D_ = D_POOL_[idx_, :]                                # (S, H)

        elif DRESS == 'interval':
            idx_ = rng.integers(0, D_POOL_.shape[0], (N_SAMPLES, len(e_)))
            D_ = D_POOL_[idx_, np.arange(len(e_))[None, :]]

        else:
            raise ValueError(f"unknown DRESS mode '{DRESS}'")

        M_ = np.clip(e_corr_[None, :] + D_.astype(float), 0., CAP)    # (S, H)

        assert M_.shape == (N_SAMPLES, len(e_)), M_.shape

        # ---- bands and ensemble scores, identical to the other scripts -------
        f_median_ = np.median(M_, axis = 0)
        f_mean_ = M_.mean(axis = 0)

        _upper, _lower = {}, {}
        for alpha in ALPHAS:
            _lower[f'{alpha}'] = np.quantile(M_, alpha / 2., axis = 0)
            _upper[f'{alpha}'] = np.quantile(M_, 1. - alpha / 2., axis = 0)

        pit_ = _empirical_PIT(f_hat_, M_.T, seed = 1234)

        es = _energy_score(M_, f_hat_)
        wis = _weighted_interval_score(f_hat_, f_median_, _lower, _upper, ALPHAS).mean()

        mask_ = EVAL_MASK_(f_hat_, day)

        # The headline error is the corrected curve. The raw day-ahead curve is
        # carried alongside as the no-update null, and the ensemble median for
        # column parity with the other two scripts.
        rmse_da, mae_da, _ = _point_errors(f_hat_, e_corr_)
        rmse_md, mae_md, _ = _point_errors(f_hat_, f_median_)
        rmse_nu, mae_nu, _ = _point_errors(f_hat_, e_)

        psr_ = np.array([time, asset, day, es, wis,
                         rmse_da, mae_da, rmse_md, mae_md,
                         rmse_nu, mae_nu, b])

        prob_ = [
            _score_row(time, asset, day, alpha,
                       f_hat_, _lower[f'{alpha}'], _upper[f'{alpha}'], mask_)
            for alpha in ALPHAS
        ]

        # finding_zone_rmse_pointforecast_mixup.md Cause 1: label the point
        # summary each error row belongs to, and score all of them.
        frac_boundary = float(np.mean((f_hat_ <= 1e-6) | (f_hat_ >= CAP - 1e-6)))

        det_ = []
        for curve_, label in [(e_corr_, 'day-ahead'),
                              (e_, 'day-ahead-no-update'),
                              (f_median_, 'median'),
                              (f_mean_, 'mean')]:
            # NOT 'b': that name holds the persistence anchor, which is
            # written into psr_ above and must not be shadowed by an MBE here.
            r, m, mbe = _point_errors(f_hat_, curve_)
            rd, md, bd = _point_errors_masked(f_hat_, curve_, mask_)
            det_.append([time, asset, day, 'ECDF', label,
                         r, m, mbe, rd, md, bd, frac_boundary])

        # Per-horizon ensemble spread.
        stat_ = pd.DataFrame(np.std(M_, axis = 0)).T
        stat_.columns = [f'H{str(i+1).zfill(2)}' for i in range(len(stat_.columns))]
        stat_['time'] = time
        stat_['region'] = asset
        stat_['day'] = day

        func_ = []
        for curve_, label in [(e_corr_, 'day-ahead'),
                              (e_, 'day-ahead-no-update'),
                              (f_median_, 'median'),
                              (f_mean_, 'mean'), (f_hat_, 'actual')]:
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


# Failures are counted and reported, never swallowed.
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

    _func = partial(_run_reference, _data = _data, time = time)

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

# Resolved from the resource; see OFFSET_BY_RESOURCE in SETTINGS.
assert resource in OFFSET_BY_RESOURCE, f'unknown resource {resource!r}'
OFFSET_MODE = OFFSET_BY_RESOURCE[resource]
print(resource, method, time, init, unbiased, description)

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
print(f'[Process {RANK}-{SIZE}]')

ALPHAS = [0.1, 0.2, 0.3, 0.4]

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

# The loader returns the TRAINING set as a stacked pool of curves -- (n_curves,
# T), one row per (day, asset) pair -- and keeps the (n_days, T, n_assets)
# layout only for the TEST set. Both F_tr_ and E_tr_ come from the same call, so
# their rows are aligned; assert it rather than trust it.
assert F_tr_.ndim == 2, (
    f'F_tr_ is expected to be a stacked curve pool (n_curves, T); got {F_tr_.shape}.'
)
assert E_tr_.shape == F_tr_.shape, (
    f'E_tr_ {E_tr_.shape} must align row-for-row with F_tr_ {F_tr_.shape}; '
    f'the error pool is meaningless otherwise.'
)
assert F_ts_.ndim == 3 and E_ts_.shape == F_ts_.shape, (
    f'F_ts_ {F_ts_.shape} / E_ts_ {E_ts_.shape} expected (n_days, T, n_assets).'
)

# The BIASED day-ahead array. Loaded for one purpose only: the evaluation mask
# below is fusion's, and fusion builds it from this array rather than from the
# `unbiased`-selected one. Never scored, never corrected, never in `_data`.
E_tr_biased_, E_ts_biased_ = loader.processed_dataset(
    unbiased = False,
    path_to_training_data = DATA / f'preprocessed_{resource}_2017.pkl',
    path_to_testing_data = DATA / f'preprocessed_{resource}_2018.pkl',
    T = T_DAY,
)

assert E_tr_biased_.shape == F_tr_.shape, (
    f'biased day-ahead pool {E_tr_biased_.shape} must align row-for-row with '
    f'F_tr_ {F_tr_.shape}; the mask is meaningless otherwise.'
)

_data = {
    'ac': [F_tr_, F_ts_],
    'day-ahead_fc': [E_tr_, E_ts_],
    'dates': [t_tr_, t_ts_],
    'grid': [dt_, dx_],
}

# Physical upper bound used to clip the dressed ensemble. Training data only.
CAP = float(max(np.nanmax(F_tr_), 1e-6))

# ---- the evaluation mask, taken from validate_asset_ffc_mpi.py ---------------
#
# Fusion is the reference every benchmark has to be comparable to, so the mask
# is ITS mask, copied verbatim from `validate_asset_ffc_mpi.py` lines 75-76:
#
#     idx_days_  = np.absolute(t_tr_ - day) < 7
#     idx_hours_ = (np.sum(F_tr_[idx_days_, :], axis = 0)
#                   + np.sum(E_tr_bias_[idx_days_, :], axis = 0)) > 1.
#
# Three things about it are deliberate and must not be "improved":
#
#   * it is SEASONAL and depends on the test day, unlike the year-round
#     climatological daylight profile this script used before;
#   * it has NO resource branch - fusion applies it to wind and solar alike, and
#     for wind it is all-True in practice rather than by construction;
#   * the +/-7 day window does NOT wrap around the year end, so day 0 draws on
#     days 0-6 only. That is fusion's behaviour and reproducing it matters more
#     than fixing it here.
#
# NOTE ON PLACEMENT: this must be computed BEFORE the DROP_NAN_CURVES filter
# below, which reindexes F_tr_ without reindexing t_tr_. Computed after it,
# `F_tr_[_idx_days_, :]` raises IndexError as soon as a single curve is dropped
# (boolean index of len(t_tr_) against a shorter array) - verified. It fails
# loudly rather than misaligning, but only because the two lengths disagree;
# keep it here where the rows and the day coordinates still correspond.
_IDX_HOURS_ = np.empty((F_ts_.shape[0], T_DAY), dtype=bool)

for _d in range(F_ts_.shape[0]):
    _idx_days_ = np.absolute(t_tr_ - _d) < 7
    _IDX_HOURS_[_d, :] = (np.sum(F_tr_[_idx_days_, :], axis = 0)
                          + np.sum(E_tr_biased_[_idx_days_, :], axis = 0)) > 1.

# ---- the error pool of the CORRECTED curve, at this tau ----------------------
#
# The band has to be calibrated to the estimator it dresses, so the pool is the
# residual of the damped-persistence forecast, not of the raw day-ahead curve.
# Every training curve is put through exactly the test-time procedure: its own
# last prefix observation gives its own anchor, damped by the same phi^k.
#
#     b_i    = f_i(tau-1) - e_i(tau-1)        (a ratio for solar)
#     e^_i   = clip(e_i (+|x) b_i phi^k, 0, CAP)
#     d_i(s) = f_i(s) - e^_i(s),  s >= tau
#
# The pool is therefore tau-dependent and each run builds its own. Note the
# shape the decay gives it: tight at k = 0, where the anchor has just removed
# the level error, widening toward the raw day-ahead climatology as phi^k dies.
# The old constant-offset pool was flat in lead by construction.

F_tr_ = np.asarray(F_tr_, dtype=np.float32)
E_tr_ = np.asarray(E_tr_, dtype=np.float32)

if DROP_NAN_CURVES:
    # A curve needs a usable prefix as well as a usable tail: a partly missing
    # prefix would silently fall back to a neutral anchor and contaminate the
    # pool with un-corrected residuals.
    keep_ = np.isfinite(F_tr_).all(axis = 1) & np.isfinite(E_tr_).all(axis = 1)
    F_tr_ = F_tr_[keep_, :]
    E_tr_ = E_tr_[keep_, :]

# Vectorised form of _anchor over the whole pool. Valid only because every
# surviving prefix is finite, so 'the last usable interval' is column tau - 1
# for every row; the assertions below check both the anchor and the damped
# curve against the scalar routines actually used at test time.
DECAY_ = _decay(E_tr_.shape[1] - time)

if time > 0:
    _y_ = F_tr_[:, time - 1].astype(np.float64)
    _e_ = E_tr_[:, time - 1].astype(np.float64)
else:
    _y_ = np.zeros(F_tr_.shape[0])
    _e_ = np.zeros(F_tr_.shape[0])

_w32_ = DECAY_.astype(np.float32)[None, :]

if OFFSET_MODE == 'add':
    B_POOL_ = (_y_ - _e_) if time > 0 else np.zeros(F_tr_.shape[0])
    E_TR_CORR_ = np.clip(E_tr_[:, time:]
                         + B_POOL_.astype(np.float32)[:, None] * _w32_,
                         0., CAP)
else:
    B_POOL_ = (np.where(_e_ > 1e-6,
                        np.clip(_y_ / np.maximum(_e_, 1e-6), *RATIO_CLIP), 1.)
               if time > 0 else np.ones(F_tr_.shape[0]))
    E_TR_CORR_ = np.clip(E_tr_[:, time:]
                         * np.exp(np.log(np.maximum(B_POOL_, 1e-12)
                                         ).astype(np.float32)[:, None] * _w32_),
                         0., CAP)

for _i in (0, F_tr_.shape[0] // 2, F_tr_.shape[0] - 1):
    assert np.isclose(B_POOL_[_i],
                      _anchor(F_tr_[_i, :time], E_tr_[_i, :time]),
                      rtol = 1e-5, atol = 1e-7), (
        f'vectorised anchor disagrees with _anchor at row {_i}'
    )
    assert np.allclose(E_TR_CORR_[_i, :],
                       _apply_offset(E_tr_[_i, time:], B_POOL_[_i]),
                       rtol = 1e-4, atol = 1e-5), (
        f'vectorised damping disagrees with _apply_offset at row {_i}'
    )


D_POOL_ = (F_tr_[:, time:] - E_TR_CORR_).astype(np.float32)

assert D_POOL_.shape[0] >= N_SAMPLES, (
    f'error pool has {D_POOL_.shape[0]} usable curves, fewer than N_SAMPLES.'
)

if RANK == 0:
    print(f'Error pool: {D_POOL_.shape[0]} curves x {D_POOL_.shape[1]} intervals, '
          f'CAP = {CAP:.4f}, dressing = {DRESS}, offset = {OFFSET_MODE}, '
          f'anchor = {ANCHOR}, beta = {BETA:.4f} /h '
          f'(half-life {DECAY_HALFLIFE:g} intervals) '
          f'(phi = {PHI:.4f}), '
          f'mean |b| = {np.mean(np.absolute(B_POOL_)):.4f}', flush = True)

# Evaluation mask: fusion's seasonal active-interval mask, precomputed above.
# Sliced to the remaining horizon the same way the old daylight mask was.
def EVAL_MASK_(f_hat_, day):
    return _IDX_HOURS_[day, T_DAY - len(f_hat_):]


if RANK == 0:
    print(f'Evaluation mask: fusion idx_hours_, '
          f'active fraction {float(np.mean(_IDX_HOURS_)):.4f}, '
          f'per-day range [{_IDX_HOURS_.sum(axis = 1).min()}, '
          f'{_IDX_HOURS_.sum(axis = 1).max()}] of {T_DAY} intervals', flush = True)

# Nothing is fitted and the one constant (DECAY_HALFLIFE) is fixed a priori, so
# there is no validation split to protect: the reference is evaluated on the same
# held-out assets as the FDU model's test stage.
processes_test_ = [(asset, j) for asset in range(10, 20) for j in range(0, 360)]


## ----------------------------- SCORING PASS ----------------------------------

if RANK == 0:
    print('----- TEST -----')

pit_, psr_, prob_, det_, stat_, func_ = _run_parallel_mpi(_data, processes_test_, time)

if RANK == 0:

    # ---- aggregated scoring rules and PIT ------------------------------------
    ks_ = np.array([_KS(pit_[:, j:(j + LEAD)].flatten()) for j in INTERVALS])
    ks_labels_ = [f'S{i}' for i in range(len(INTERVALS))]

    psr_ = psr_.astype(float)

    row_ = {
        'initialization': init,
        'time': time,

        'ES_test': float(np.mean(psr_[:, 3])),
        'WIS_test': float(np.mean(psr_[:, 4])),

        # The reference itself: the offset-corrected day-ahead curve.
        'RMSE_test': float(np.mean(psr_[:, 5])),
        'MAE_test': float(np.mean(psr_[:, 6])),

        # The dressed ensemble's median, for column parity with the other runs.
        'RMSE_median_test': float(np.mean(psr_[:, 7])),
        'MAE_median_test': float(np.mean(psr_[:, 8])),

        # The no-update null: the same curve at every tau, so its trend across
        # tau is the horizon-composition effect in isolation.
        'RMSE_noupdate_test': float(np.mean(psr_[:, 9])),
        'MAE_noupdate_test': float(np.mean(psr_[:, 10])),

        # Mean and mean-absolute ANCHOR, i.e. the correction at lead 0, as a
        # sanity check that it is doing something and not saturating the clip.
        # The correction actually applied at lead k is this times phi^k.
        'offset_mean': float(np.mean(psr_[:, 11])),
        'offset_abs_mean': float(np.mean(np.absolute(psr_[:, 11]))),

        'KS_test': float(np.mean(ks_)),

        **{ks_labels_[i] + '_test': float(ks_[i]) for i in range(len(INTERVALS))},

        # Fixed settings, recorded so a results file is self-describing.
        'model': 'day-ahead-damped-persistence',
        'point_forecast': 'day-ahead',
        'offset_mode': OFFSET_MODE,
        'anchor': ANCHOR,
        'decay_halflife': DECAY_HALFLIFE,
        'decay_beta_per_hour': BETA,
        'decay_rate': RATE,
        'decay_phi': PHI,
        'ratio_clip': str(RATIO_CLIP) if OFFSET_MODE == 'mul' else '',
        'dressing': DRESS,
        'eval_mask': 'ffc idx_hours_ (|t_tr_ - day| < 7, biased day-ahead)',
        'n_samples': N_SAMPLES,
        'error_pool_curves': int(D_POOL_.shape[0]),
        'unbiased': unbiased,
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
              'FCS_day', 'SCP_day', 'width_mean_day', 'n_excursions',
              'len_excursion', 'frac_out', 'frac_degenerate']:
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
        frac_out = ('frac_out', 'mean'),
        frac_degenerate = ('frac_degenerate', 'mean'),
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
    for c in ['RMSE', 'MAE', 'MBE', 'RMSE_day', 'MAE_day', 'MBE_day', 'frac_boundary']:
        det_df[c] = pd.to_numeric(det_df[c], errors = 'coerce')

    det_agg_ = det_df.groupby(['time', 'point_forecast']).agg(
        {'RMSE': 'mean', 'MAE': 'mean', 'MBE': 'mean',
         'RMSE_day': 'mean', 'MAE_day': 'mean', 'MBE_day': 'mean',
         'frac_boundary': 'mean'}
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
