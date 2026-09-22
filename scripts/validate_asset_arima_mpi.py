"""
AutoARIMA curve-completion benchmark for the intra-day functional dynamic
update (FDU) experiments: one dynamic harmonic regression per asset, validated
on the training year and applied to the test year.
================================================================================

Companion to ``validate_asset_ffc_mpi.py``, ``validate_asset_prophet_mpi.py``
and ``validate_asset_daref_mpi.py``. Same MPI layout and the same output CSV
schema, so all four result sets drop into the same tables.

What this is
------------
The benchmark that reads a partially observed day and completes it using no
exogenous information at all: no weather, no day-ahead forecast, no neighbours,
only the realised generation of the same asset up to the update time. It is the
counterpart to the day-ahead reference, and the pair brackets the proposed
method:

    daref   day-ahead forecast, no memory of today beyond a scalar offset
    arima   today's memory in full, no day-ahead forecast
    FDU     both, plus the neighbourhood

so ``FDU - daref`` is what the neighbourhood adds and ``FDU - arima`` is what
the day-ahead forecast adds. It also answers the referee question Prophet
cannot: *did you try a standard statistical time-series model?* Prophet has no
autoregressive component and is structurally unable to use the morning; ARIMA
is, which is why this benchmark needs no persistence bolt-on.

  Data layout      AutoARIMA is a time-series model and wants a long, gapless,
                   chronologically ordered series. The loader hands back the
                   test year as ``(n_days, T_DAY, n_assets)``, so the series for
                   asset a is that cube raveled day-major,

                       y_a[d * T_DAY + s] = F[d, s, a],

                   i.e. interval 1 of day 1 through interval 288 of day 365 in
                   order. See ``_year_series`` and the reshape assertions in
                   the LOAD DATA section; nothing in this script indexes the
                   stacked ``(n_curves, T)`` training pool by row, because the
                   loader does not document whether that pool is day-major or
                   asset-major and a wrong guess would silently scramble time.

  Year bridge      The scored year is 2018 but the series must not start cold.
                   The last ``BRIDGE_DAYS`` days of 2017 are prepended to each
                   asset's 2018 series, so day 0 of 2018 has exactly as much
                   history as day 200 does. The bridge is history only: no day
                   of 2017 is ever scored, and ``_ts_index`` is the single
                   place that converts a 2018 day index into an offset in the
                   bridged series.

  Model            Dynamic harmonic regression (Hyndman & Athanasopoulos,
                   *Forecasting: Principles and Practice*, 3rd ed., Sec. 12.1):
                   an ARIMA(p,d,q) error process on deterministic Fourier terms
                   for the 288-interval daily cycle,

                       y_t = sum_k [a_k sin(2 pi k t / 288)
                                  + b_k cos(2 pi k t / 288)] + n_t,
                       n_t ~ ARIMA(p, d, q).

                   A textbook SARIMA with s = 288 is not usable - the seasonal
                   state is enormous - and Fourier terms are the standard remedy
                   for a long seasonal period.

  One per asset    ONE model per asset, not one pooled model. Each asset gets
                   its own order, its own parameters and its own selected
                   FOURIER_K and HISTORY_DAYS. That model is then reused at
                   every update time tau: tau changes what has been observed,
                   not what the model is.

  Validation       Rolling-origin cross-validation on 2017 selects FOURIER_K and
                   HISTORY_DAYS per asset; 2018 is scored once, at the end, with
                   the selection frozen. The folds are contiguous day-aligned
                   blocks, never shuffled, and every validation block sits
                   strictly AFTER its training block with a gap of at least
                   ``max(HISTORY_DAYS_GRID)`` days, so a validation day's
                   history never overlaps the block its parameters came from -
                   the same separation the 2017/2018 split gives at test time.
                   The ARIMA order itself is chosen by Hyndman-Khandakar
                   AutoARIMA (AICc) once per (asset, K); the CV chooses only
                   what AICc cannot see.

  Selection score  Mean weighted interval score, ``_weighted_interval_score``
                   over ALPHAS - the same WIS that daref writes as ``WIS_test``,
                   computed by the same function on the same objects. RMSE, MAE
                   and ES are logged next to it in the per-asset hyper file so
                   the selection can be shown to agree with the point scores.

  Completion       For each test case the fitted model is APPLIED, NEVER
                   REFITTED, to the operationally available history - the last
                   HISTORY_DAYS whole days of that asset followed by today's
                   observations up to tau - and simulated forward to midnight OF
                   THAT SAME DAY. Kalman filtering is what makes this a
                   completion rather than a forecast from cold: the state at tau
                   carries the morning. The horizon is always ``T_DAY - tau``
                   steps and stops at the day boundary; the scored tail never
                   enters the filter.

  Band             N_SAMPLES simulated WHOLE paths, clipped to [0, CAP]; the
                   band at nominal alpha is their marginal quantiles at each
                   interval. Whole paths rather than per-horizon prediction
                   intervals, because simultaneous coverage needs the temporal
                   correlation of the errors.

Reporting conventions follow ``region_comparison_protocol.md`` and
``finding_zone_rmse_pointforecast_mixup.md``:

  * two point summaries are scored on every case, the simulated median and the
    simulated mean, each carrying its own ``point_forecast`` label.
  * NO SCORE IS MASKED. ``validate_asset_ffc_mpi.py`` is the reference this
    benchmark has to be comparable to, and it scores the whole remaining
    horizon: FIS, FCS, SCP, RMSE, MAE and MBE are all computed on the full
    ``F_ts_[day, time:, asset]``, and its results files carry no ``_day``
    columns at all. Fusion's only mask, ``idx_hours_`` (built from
    ``idx_days_ = |t_tr_ - day| < 7``), is a MODEL INPUT passed to
    ``_fdu.fit(..., interval_mask = ...)``, not an evaluation mask - it tells
    the FDU model which intraday intervals are active in the season around the
    test day, and it never touches a score. So this script reports the same
    scoring surface fusion does. The daylight-restricted ``RMSE_day`` /
    ``MAE_day`` / ``MBE_day`` / ``FCS_day`` / ``SCP_day`` /
    ``width_mean_day`` columns that ``validate_asset_daref_mpi.py`` and
    ``validate_asset_prophet_mpi.py`` carry are deliberately absent here; for
    solar, read ``frac_boundary`` and ``frac_degenerate`` instead.

Usage
-----
    mpirun -n <N> python validate_asset_arima_mpi.py \
        <resource> <method> <time> <init> <unbiased> <description>

    resource     'wind' | 'solar'
    method       tag used in output file names, e.g. 'arima'
    time         cutoff tau in 5-minute intervals (wind 72/144/216,
                 solar 120/132/144/168), or 'all' to score every cutoff of the
                 resource in ONE run. 'all' is the intended mode: the model is
                 per asset and independent of tau, so one pass fits once and
                 scores every cutoff, writing exactly the files that separate
                 per-tau invocations would have written.
    init         run index. Repeated inits measure Monte-Carlo noise in the
                 simulated paths only; the fit and the selection are
                 deterministic.
    unbiased     0 | 1, passed to the loader for parity ONLY. This benchmark
                 never touches the day-ahead array, so the flag cannot change
                 its numbers.
    description  free-form tag used in output file names

MPI layout
----------
Two stages, because they balance differently.

    Stage 1  ranks split ASSETS. Each rank cross-validates and fits only its own
             assets, then every rank ends up with every asset's selection via
             ``allgather``. Run with ``-n <= len(ASSETS)`` here and no rank
             idles; more ranks than assets simply leaves the surplus waiting.
    Stage 2  ranks split (asset, day) cases, exactly as the other three
             scripts do, so scoring uses every rank whatever -n is.

The selection is cached per asset under ``params/arima/``; later invocations
skip stage 1 entirely and cost only the scoring pass.
"""

import os, sys, json, time as _time, warnings

sys.path.append('/home/gterren/dynamic_update/functional_forecast_dynamic_update/')

from pathlib import Path

import pandas as pd
import numpy as np

from mpi4py import MPI
from functools import partial
from time import sleep

from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsforecast.models import AutoARIMA

from src import loader
from src.utils import (_KS,
                       _weighted_interval_score,
                       _simultaneous_coverage,
                       _coverage_score,
                       _interval_score,
                       _empirical_PIT,
                       _energy_score)

warnings.filterwarnings('ignore')

VALIDATION = Path('/home/gterren/dynamic_update/validation')
PARAM = Path('/home/gterren/dynamic_update/params')
DATA = Path('/home/gterren/dynamic_update/data')

# -------------------------------- SETTINGS ------------------------------------

# ============================== USER VARIABLE =================================
#
# The assets this benchmark runs on. These are the FDU model's held-out test
# assets: the FDU run validates its hyperparameters on assets 0-9 and tests on
# 10-19, so scoring the benchmark on 10-19 puts both methods on identical
# asset-days. Change this to run a subset while debugging - e.g. [10] for a
# single-asset smoke test - and change it back before producing table numbers.
#
# One model is fitted per asset in this list, and stage 1 parallelises over it.
ASSETS = list(range(10, 20))

# Days of the TEST year scored, matching the other three scripts exactly.
DAYS = list(range(0, 360))
# ==============================================================================

# Number of intervals per day (5-minute resolution).
T_DAY = 288

# Calendar years behind the loader's training / testing pickles. The training
# year is where the cross-validation lives; the testing year is scored.
YEAR_TR = 2017
YEAR_TS = 2018

# ---- cross-validation grid ---------------------------------------------------
#
# FOURIER_K sets how much of the daily cycle is deterministic rather than left
# to the ARMA errors. It is the one quantity AutoARIMA's AICc cannot see -- its
# cost falls at long lead, where a one-step in-sample criterion has no purchase
# -- so it is the only thing cross-validated here. The order (p, d, q) is NOT in
# this grid: d comes from unit-root testing and (p, q) from AICc, both inside
# Hyndman-Khandakar AutoARIMA, once per (asset, K).
FOURIER_K_GRID = (4, 6, 10)

# HISTORY_DAYS: whole days of past observations the Kalman filter runs over
# before it reaches the update time.
#
# FIXED, NOT CROSS-VALIDATED (2026-09-21). This is a sufficiency threshold, not
# a tuned quantity. The filter must reconstruct the innovation sequence, which
# is recovered by inverting the MA polynomial and so depends on the whole past;
# starting it HISTORY_DAYS back with zero innovations truncates that sum, and
# the truncation error decays as |MA root|^-k. Across the ten wind assets the
# slowest decay is a half-life of 3.6 intervals (asset 19, |MA root| = 1.21), so
# ten half-lives is about three hours. Three days is roughly 24x that.
#
# The previous grid (3, 7, 14) days sampled only the flat part of that curve:
# validation WIS moved by at most 4.4% across it, with no consistent direction
# (six assets picked 3 d, one 7 d, three 14 d), so the selection was responding
# to noise among values that were all already sufficient. Cross-validating it
# implied a tuning that was not happening, and the 14-day cases cost about 4.4x the
# filtering work per case for nothing.
HISTORY_DAYS = 3

# Kept as a one-element grid so the CV machinery, the cache key and the results
# schema are unchanged; the loop simply runs once.
HISTORY_DAYS_GRID = (HISTORY_DAYS,)

# Rolling-origin folds on the training year. Validation blocks are spread
# through the year so the selection is not made on one season.
N_FOLDS = 4

# Whole days in each fold's parameter-estimation block, and in the final fit.
FIT_DAYS = 60

# Validation days scored per fold. The CV cost is
# len(FOURIER_K_GRID) * N_FOLDS * len(HISTORY_DAYS_GRID) * CV_VAL_DAYS * len(TAUS)
# completions per asset -- now a third of what it was, HISTORY_DAYS being fixed.
# completions per asset, so this is the knob that controls it.
CV_VAL_DAYS = 8

# Simulated paths inside the CV loop only. The selection compares combinations
# against each other, so it tolerates more Monte-Carlo noise than the reported
# scores do.
N_SAMPLES_CV = 200

# Set False to reuse the cached per-asset selection without re-validating, or
# to fall back to CV_DEFAULT when no cache exists.
CROSS_VALIDATE = True

# Used when CROSS_VALIDATE is False and nothing is cached.
CV_DEFAULT = {'fourier_k': 10, 'history_days': HISTORY_DAYS}

# ---- test-time settings ------------------------------------------------------

# Number of simulated paths behind every reported band. Raised from 200: at
# alpha = 0.1 the 5% and 95% quantiles rest on about ten order statistics with
# 200 draws, which is visible as ragged band edges and as Monte-Carlo noise in
# SCP. Keep this equal to N_SAMPLES in validate_asset_daref_mpi.py.
N_SAMPLES = 1000

# Whole days of the TRAINING year prepended to each asset's test-year series, so
# that day 0 of the test year has the same history as any other day. Must be at
# least HISTORY_DAYS; asserted below.
BRIDGE_DAYS = HISTORY_DAYS

# Days of the training year AutoARIMA searches over. Shorter than FIT_DAYS keeps
# the stepwise search affordable; the parameters are re-estimated on the full
# FIT_DAYS block afterwards.
ORDER_SEARCH_DAYS = 20

# Caps on the Hyndman-Khandakar search. (5,1,4) on a 20-day block is more
# structure than a benchmark whose role in the paper is 'a standard statistical
# time-series model' should carry, and the high orders are exactly where the
# unconstrained re-estimation used to leave the admissible region. Passed to
# statsforecast only if the installed version accepts them; the route string
# records which happened.
MAX_P, MAX_Q = 3, 3

# Estimate under stationarity and invertibility constraints. AutoARIMA selects
# the order UNDER these constraints, so re-estimating without them lets the
# coefficients leave the region the order was chosen in - which is how one asset
# acquired an explosive AR whose simulated paths diverged, clipped to [0, CAP],
# and produced the best apparent coverage in the file from the worst model.
# _fit_block and _filter_block MUST agree here, or stored parameters are read
# back under a different parameterisation.
ENFORCE = True

# A fit is admissible only if every AR and MA root lies outside the unit circle
# by this margin. statsmodels is asked to enforce it; this is the check that the
# enforcement actually held, because a constrained optimiser can still stop on
# the boundary.
ROOT_MARGIN = 1e-3

# Draw the initial state explicitly from its filtered distribution, one
# simulate() call per path. VERIFIED REDUNDANT: statsmodels already draws the
# initial state when simulating from the end of the sample. Left in as a
# cross-check; the default vectorised route is the one to run.
STATE_UNCERTAINTY = False

# Seed for the simulation draw. Mixed with (asset, day, time) so every case is
# reproducible and no two cases share a draw.
SEED = 20260915

# ------------------------------------------------------------------------------

assert BRIDGE_DAYS >= max(HISTORY_DAYS_GRID), (
    f'BRIDGE_DAYS = {BRIDGE_DAYS} is shorter than the longest candidate history '
    f'{max(HISTORY_DAYS_GRID)}; early test days would get a truncated history.'
)


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


# One row of the probabilistic results table. Unmasked, over the whole remaining
# horizon, exactly as validate_asset_ffc_mpi.py scores.
def _score_row(time, asset, day, alpha, f_hat_, lo_, up_):

    assert lo_.shape == f_hat_.shape and up_.shape == f_hat_.shape, (
        f'band {lo_.shape}/{up_.shape} vs actual {f_hat_.shape}'
    )

    FIS = _interval_score(f_hat_, lo_, up_, alpha).mean()
    FCS = _coverage_score(f_hat_, lo_, up_)
    SCP = _simultaneous_coverage(f_hat_, lo_, up_)

    width_ = up_ - lo_

    out_ = (f_hat_ < lo_) | (f_hat_ > up_)
    frac_out = float(np.mean(out_))

    n_exc, len_exc = _excursion_stats(f_hat_, lo_, up_)

    # Band saturation, measured on the BAND (frac_boundary in the other scripts
    # is measured on the realisation and is a different quantity).
    frac_degenerate = float(np.mean(width_ <= 1e-6))

    return [time, asset, day, alpha, 'ECDF',
            FIS, FCS, SCP,
            float(np.mean(width_)), float(np.median(width_)),
            n_exc, len_exc, frac_out, frac_degenerate]


PROB_COLUMNS = ['time', 'asset', 'day', 'alpha', 'distance',
                'FIS', 'FCS', 'SCP',
                'width_mean', 'width_median',
                'n_excursions', 'len_excursion',
                'frac_out', 'frac_degenerate']

DET_COLUMNS = ['time', 'asset', 'day', 'distance', 'point_forecast',
               'RMSE', 'MAE', 'MBE',
               'frac_boundary']


# ------------------------------ MODEL PIECES ----------------------------------
#
# Deterministic Fourier terms for the daily cycle. t is the absolute interval
# index, so the terms line up across the history, the prefix and the simulated
# tail without any bookkeeping at the day boundary. Because the period is T_DAY
# and every day holds exactly T_DAY intervals, t and t mod T_DAY give identical
# terms; the absolute index is used so that a history spanning a day boundary
# needs no special case.
def _fourier(t_, K, period = T_DAY):

    t_ = np.asarray(t_, dtype=float)[:, None]
    k_ = np.arange(1, K + 1, dtype=float)[None, :]

    arg_ = 2. * np.pi * k_ * t_ / float(period)

    return np.concatenate([np.sin(arg_), np.cos(arg_)], axis = 1)


# A constant is a level with d = 0 and a drift with d > 0. A drift on a capacity
# factor series is not wanted, so the trend is dropped whenever the model
# differences.
def _trend_for(order):
    return 'c' if int(order[1]) == 0 else 'n'


# Hyndman-Khandakar order selection for one asset at one K. The route actually
# taken is returned so the results file records it.
#
# Only ImportError falls through to the next selector. A selector that is
# installed but fails is a different event from one that is absent, and
# collapsing the two is how a modelling failure gets reported as a missing
# dependency - or, worse, as a selected order.
def _select_order(y_, X_):

    try:
        from statsforecast.models import AutoARIMA

    except ImportError as e_sf:

        try:
            import pmdarima as pm

        except ImportError as e_pm:
            return FALLBACK_ORDER, (f'fallback (statsforecast: {e_sf!r}; '
                                    f'pmdarima: {e_pm!r})')

        m = pm.auto_arima(np.asarray(y_, dtype=float),
                          X = np.asarray(X_, dtype=float),
                          max_p = MAX_P, max_q = MAX_Q,
                          seasonal = False, stepwise = True,
                          suppress_warnings = True, error_action = 'ignore')

        return tuple(int(v) for v in m.order), 'pmdarima.auto_arima'

    # Pass the order caps only if this statsforecast accepts them, following the
    # same resolve-rather-than-pin convention as _sim_seed_kw. An uncapped run
    # is legal but says so in the route, because it is not the same experiment.
    import inspect
    _accepts = inspect.signature(AutoARIMA.__init__).parameters
    _kw = {k: v for k, v in (('max_p', MAX_P), ('max_q', MAX_Q)) if k in _accepts}
    _tag = (f'max_p={MAX_P}, max_q={MAX_Q}' if len(_kw) == 2
            else 'CAPS UNSUPPORTED by this statsforecast')

    m = AutoARIMA(season_length = 1, **_kw)
    m = m.fit(y = np.asarray(y_, dtype=float), X = np.asarray(X_, dtype=float))

    # statsforecast stores the order as arma = (p, q, P, Q, m, d, D).
    arma_ = m.model_['arma']

    if len(arma_) < 6:
        raise RuntimeError(f'unexpected arma tuple {arma_}')

    order = (int(arma_[0]), int(arma_[5]), int(arma_[1]))

    return order, f'statsforecast.AutoARIMA ({_tag})'



def _fit_block(y_, t_, K, order):

    mod = SARIMAX(np.asarray(y_, dtype=float),
                  exog = _fourier(t_, K),
                  order = tuple(int(v) for v in order),
                  trend = _trend_for(order),
                  enforce_stationarity = ENFORCE,
                  enforce_invertibility = ENFORCE)

    return mod.fit(disp = False)


# Rebuild a results object from stored parameters without re-optimising, so that
# stage 2 can reconstruct every asset's model from the small dict stage 1
# broadcast rather than shipping a fitted object over MPI.
def _filter_block(y_, t_, K, order, params_):

    mod = SARIMAX(np.asarray(y_, dtype=float),
                  exog = _fourier(t_, K),
                  order = tuple(int(v) for v in order),
                  trend = _trend_for(order),
                  enforce_stationarity = ENFORCE,
                  enforce_invertibility = ENFORCE)

    return mod.filter(np.asarray(params_, dtype=float))


# Moduli of the AR and MA roots of a fitted model, smallest first. Read off
# res.param_names rather than by position, so the slice cannot silently drift
# when the trend term or the number of Fourier regressors changes. Seasonal
# terms are named 'ar.S.L*' and are excluded by construction.
def _ar_ma_roots(res):

    names_ = list(res.param_names)
    pr_ = np.asarray(res.params, dtype=float)

    ar_ = np.array([pr_[i] for i, n in enumerate(names_) if n.startswith('ar.L')])
    ma_ = np.array([pr_[i] for i, n in enumerate(names_) if n.startswith('ma.L')])

    def _min_root(c_, sign):
        # AR: 1 - p1 z - p2 z^2 ...  (sign = -1)
        # MA: 1 + t1 z + t2 z^2 ...  (sign = +1)
        if c_.size == 0:
            return float('inf')
        return float(np.min(np.abs(np.roots(np.r_[sign * c_[::-1], 1.]))))

    return _min_root(ar_, -1.), _min_root(ma_, +1.)


# Orders tried, in order, when the selected one will not produce an admissible
# fit. Reductions are appended to order_route and never applied silently.
def _order_ladder(order):

    p, d, q = (int(v) for v in order)

    out_ = []
    for c in [(p, d, q), (min(p, 2), d, min(q, 2)), (1, d, 1), (0, d, 1), (1, d, 0)]:
        if c not in out_ and (c[0] or c[2]):
            out_.append(c)

    return out_


# Fit at `order`, stepping down the ladder until the result is admissible.
#
# THE POINT OF THIS FUNCTION. AutoARIMA searches under stationarity and
# invertibility constraints; this keeps the re-estimated coefficients under them
# too, AND VERIFIES IT rather than trusting the optimiser, because a constrained
# fit can still stop on the boundary. An inadmissible model is not a slightly
# worse model: an AR root inside the unit circle makes the simulated paths
# diverge, the clip to [0, CAP] then turns the band into the whole physical
# range, and the asset scores near-perfect coverage with a meaningless forecast.
def _fit_admissible(y_, t_, K, order):

    last_ = 'no candidate tried'

    for cand_ in _order_ladder(order):

        try:
            res = _fit_block(y_, t_, K, cand_)
            ar_r, ma_r = _ar_ma_roots(res)

        except Exception as e:
            last_ = f'{cand_} failed to fit: {e!r}'
            continue

        if min(ar_r, ma_r) > 1. + ROOT_MARGIN:
            note = ('' if tuple(cand_) == tuple(order)
                    else f'; REDUCED from {tuple(order)} to {tuple(cand_)}')
            return res, tuple(cand_), note, ar_r, ma_r

        last_ = (f'{cand_} inadmissible: min|AR root| = {ar_r:.4f}, '
                 f'min|MA root| = {ma_r:.4f}')

    raise RuntimeError(f'no admissible fit for order {tuple(order)} '
                       f'(last attempt: {last_})')


# statsmodels renamed simulate()'s seed argument from 'random_state' to 'rng'
# (0.15). Resolve it once rather than pinning a version.
def _sim_seed_kw(res, seed):

    global _SIM_KW

    if _SIM_KW is None:
        import inspect
        params_ = inspect.signature(res.simulate).parameters
        _SIM_KW = 'rng' if 'rng' in params_ else 'random_state'

    return {_SIM_KW: int(seed)}


_SIM_KW = None

# Apply the fitted parameters to one case's history and simulate the remaining
# horizon as whole paths. Nothing here refits: `apply(..., refit = False)` runs
# the Kalman filter with the same parameters over new data, which is what
# carries the observed morning into the state at tau.
def _complete(res, y_hist_, t_hist_, t_fut_, K, rng, n_samples):

    H = len(t_fut_)

    # A dead or fully-clipped asset gives a singular filter; the physically
    # right answer is the degenerate zero ensemble, as in the Prophet benchmark.
    if float(np.nanmax(np.absolute(y_hist_))) < 1e-9:
        return np.zeros((n_samples, H), dtype=float)

    res_ = res.apply(endog = np.asarray(y_hist_, dtype=float),
                     exog = _fourier(t_hist_, K),
                     refit = False)

    if STATE_UNCERTAINTY:
        mu_ = np.asarray(res_.predicted_state[:, -1], dtype=float)
        S_ = np.asarray(res_.predicted_state_cov[:, :, -1], dtype=float)
        S_ = 0.5 * (S_ + S_.T)
        S_ = S_ + 1e-10 * np.eye(S_.shape[0])

        L_ = np.linalg.cholesky(S_)

        M_ = np.empty((n_samples, H), dtype=float)

        for k in range(n_samples):
            s0_ = mu_ + L_ @ rng.standard_normal(mu_.shape[0])
            M_[k, :] = np.asarray(
                res_.simulate(nsimulations = H, anchor = 'end',
                              exog = _fourier(t_fut_, K),
                              initial_state = s0_,
                              **_sim_seed_kw(res_, rng.integers(0, 2**31 - 1)))
            ).ravel()[:H]
    else:
        sim_ = res_.simulate(nsimulations = H, anchor = 'end',
                             exog = _fourier(t_fut_, K),
                             repetitions = n_samples,
                             **_sim_seed_kw(res_, rng.integers(0, 2**31 - 1)))

        M_ = np.asarray(sim_).reshape(H, -1).T[:n_samples, :]

    assert M_.shape == (n_samples, H), M_.shape

    return np.clip(M_, 0., CAP)


# --------------------------- SERIES CONSTRUCTION ------------------------------
#
# The single place a (day, interval) pair becomes an offset into a raveled
# series, and the single place the bridge length is added. Everything else
# calls these two, so the day-major convention is stated once.

def _year_series(F_, asset):
    """Ravel one asset out of a (n_days, T_DAY, n_assets) cube, day-major.

    Returns a (n_days * T_DAY,) series running interval 0 of day 0 through
    interval T_DAY - 1 of the last day, with no gaps and no reordering.
    """

    F_ = np.asarray(F_, dtype=float)

    assert F_.ndim == 3 and F_.shape[1] == T_DAY, (
        f'expected (n_days, {T_DAY}, n_assets); got {F_.shape}'
    )

    y_ = F_[:, :, asset].ravel(order = 'C')

    assert y_.size == F_.shape[0] * T_DAY, (y_.size, F_.shape)

    return y_


def _ts_index(day, s):
    """Offset of (test-year day, interval s) inside the BRIDGED series."""

    return BRIDGE_DAYS * T_DAY + day * T_DAY + s


def _history_window(y_, day, tau, history_days, bridged):
    """The operationally available history ending at (day, tau), and its index.

    Whole days strictly before the scored day, then today up to but excluding
    tau. The tail being scored is never included - the slice stops at tau by
    construction, not by a filter that could be got wrong.
    """

    end = _ts_index(day, tau) if bridged else (day * T_DAY + tau)
    start = end - history_days * T_DAY - tau

    assert start >= 0, (
        f'history for day {day} at tau {tau} would start at {start}; '
        f'the bridge is too short or the fold starts too early'
    )

    t_ = np.arange(start, end, dtype=float)
    y_hist_ = np.asarray(y_[start:end], dtype=float)

    assert len(y_hist_) == history_days * T_DAY + tau, (len(y_hist_), history_days, tau)
    assert len(t_) == len(y_hist_), (len(t_), len(y_hist_))

    # The horizon runs from tau to the end of THIS day and no further.
    t_fut_ = np.arange(end, end + (T_DAY - tau), dtype=float)

    return y_hist_, t_, t_fut_


# ------------------------- CROSS-VALIDATION ON 2017 ---------------------------
#
# Rolling origin, contiguous blocks, no shuffling. Fold f validates on
# CV_VAL_DAYS days starting at roughly (f + 1) / (N_FOLDS + 1) of the training
# year, and estimates its parameters on the FIT_DAYS days that end GAP_DAYS
# before that, so the history a validation day filters over lies entirely
# outside the block its parameters came from - mirroring the test year, where
# the parameters come from 2017 and the history from 2018.
#
# DELIBERATELY DECOUPLED from HISTORY_DAYS. Only GAP_DAYS >= HISTORY_DAYS is
# required for causality; 14 days is a margin, and it keeps the fold boundaries
# identical to those already verified rather than moving them with a parameter
# that is no longer tuned. A larger gap is never wrong.
GAP_DAYS = 14


def _cv_folds(n_days_tr):

    folds_ = []

    for f in range(N_FOLDS):

        val_start = int((f + 1) * n_days_tr / (N_FOLDS + 1))
        train_end = val_start - GAP_DAYS
        train_start = max(0, train_end - FIT_DAYS)

        val_days_ = list(range(val_start,
                               min(val_start + CV_VAL_DAYS, n_days_tr)))

        # A validation day needs max(HISTORY_DAYS_GRID) whole days before it,
        # and the training block needs to be long enough to identify the model.
        if train_end - train_start < 2 * max(FOURIER_K_GRID) or not val_days_:
            continue
        if val_start < max(HISTORY_DAYS_GRID):
            continue

        folds_.append({'train': (train_start, train_end), 'val': val_days_})

    assert folds_, 'no usable CV folds; check FIT_DAYS, GAP_DAYS and N_FOLDS'

    return folds_


def _score_wis(f_hat_, M_):
    """WIS of one completion - the same score, on the same objects, that
    validate_asset_daref_mpi.py aggregates into WIS_test."""

    f_median_ = np.median(M_, axis = 0)

    _upper, _lower = {}, {}
    for alpha in ALPHAS:
        _lower[f'{alpha}'] = np.quantile(M_, alpha / 2., axis = 0)
        _upper[f'{alpha}'] = np.quantile(M_, 1. - alpha / 2., axis = 0)

    wis = float(_weighted_interval_score(f_hat_, f_median_, _lower, _upper, ALPHAS).mean())
    rmse, mae, _ = _point_errors(f_hat_, f_median_)

    return wis, rmse, mae


def _cross_validate_asset(asset, Y17_):
    """Select FOURIER_K and HISTORY_DAYS for one asset on the training year.

    Y17_ is that asset's raveled training-year series. Returns the selection and
    the full score table, so the choice is auditable rather than a bare number.
    """

    n_days_tr = Y17_.size // T_DAY
    folds_ = _cv_folds(n_days_tr)

    # (K, history_days) -> list of per-case WIS / RMSE / MAE
    scores_ = {(K, hd): [[], [], []]
               for K in FOURIER_K_GRID for hd in HISTORY_DAYS_GRID}

    orders_ = {}

    for K in FOURIER_K_GRID:

        # The order is structural: selected once per (asset, K), on the first
        # fold's training block, then held fixed across folds. Only the
        # parameters are re-estimated per fold.
        a0, b0 = folds_[0]['train']
        s0 = max(a0, b0 - ORDER_SEARCH_DAYS)

        y0_ = Y17_[s0 * T_DAY:b0 * T_DAY]
        t0_ = np.arange(s0 * T_DAY, b0 * T_DAY, dtype=float)

        order, route = _select_order(y0_, _fourier(t0_, K))
        orders_[K] = (order, route)

        for fold_ in folds_:

            a, b = fold_['train']

            y_fit_ = Y17_[a * T_DAY:b * T_DAY]
            t_fit_ = np.arange(a * T_DAY, b * T_DAY, dtype=float)

            # The CV must score the same estimator the test year gets, so the
            # folds are fitted under the same constraints and the same ladder.
            # A fold that cannot produce an admissible fit is dropped and
            # counted, never silently scored with an unstable model.
            try:
                res = _fit_admissible(y_fit_, t_fit_, K, order)[0]
            except Exception as e:
                CV_FAILURES.append((f'{asset}-K{K}-fold{fold_["train"]}', repr(e)))
                continue

            for hd in HISTORY_DAYS_GRID:
                for day in fold_['val']:
                    for tau in TAUS:

                        # try:
                        y_hist_, t_hist_, t_fut_ = _history_window(
                            Y17_, day, tau, hd, bridged = False
                        )

                        f_hat_ = Y17_[day * T_DAY + tau:(day + 1) * T_DAY]

                        assert len(f_hat_) == len(t_fut_), (len(f_hat_), len(t_fut_))

                        if not (np.isfinite(y_hist_).all() and np.isfinite(f_hat_).all()):
                            continue

                        rng = np.random.default_rng(
                            (SEED * 7_919) ^ (int(asset) * 1_000_033
                                                + int(day) * 1_009 + int(tau) * 31 + K * 7 + hd)
                        )

                        M_ = _complete(res, y_hist_, t_hist_, t_fut_, K,
                                        rng, N_SAMPLES_CV)

                        wis, rmse, mae = _score_wis(f_hat_, M_)

                        if np.isfinite(wis):
                            scores_[(K, hd)][0].append(wis)
                            scores_[(K, hd)][1].append(rmse)
                            scores_[(K, hd)][2].append(mae)

                        # except Exception as e:
                        #     CV_FAILURES.append((f'{asset}-{K}-{hd}-{day}-{tau}', repr(e)))

    # ---- pick the minimum-WIS combination ------------------------------------
    table_ = []
    for (K, hd), (wis_, rmse_, mae_) in scores_.items():
        if not wis_:
            continue
        table_.append({'asset': asset, 'fourier_k': K, 'history_days': hd,
                       'n_cases': len(wis_),
                       'WIS_val': float(np.mean(wis_)),
                       'RMSE_val': float(np.mean(rmse_)),
                       'MAE_val': float(np.mean(mae_)),
                       'arima_order': str(orders_[K][0]),
                       'order_route': orders_[K][1]})

    assert table_, f'asset {asset}: every CV combination failed'

    best_ = min(table_, key = lambda r: r['WIS_val'])

    return best_, table_, orders_


CV_FAILURES = []


# Modal value of a per-asset selection, for the scalar summary columns.
#
# NOT a median. The selections live on a small discrete grid and there is an
# even number of assets, so np.median averages the two middle values and can
# return a point the grid does not contain - it reported fourier_k = 8 against
# a grid of (4, 6, 10), naming a model no asset was ever fitted with. Ties go
# to the larger value; the per-asset dicts beside these columns remain the
# authoritative record.
def _modal(values_):

    from collections import Counter

    counts_ = Counter(int(v) for v in values_)

    return max(counts_.items(), key = lambda kv: (kv[1], kv[0]))[0]


# Why a cached selection must not be reused, or None if it is fine.
#
# The cache key encodes the candidate grids and the fold structure, not the
# execution environment or the quality of the fit. Without this guard, a
# selection made where no order selector was installed is inherited verbatim by
# every later run - including runs where one IS installed - and the stale
# fallback route string is written into the results file as though it described
# the current environment.
def _cache_reject_reason(c_):

    if str(c_.get('order_route', '')).startswith('fallback'):
        return 'order_route is a fallback'

    if 'ar_root_min' not in c_ or 'ma_root_min' not in c_:
        return 'predates the admissibility check'

    _m = min(float(c_['ar_root_min']), float(c_['ma_root_min']))

    if _m <= 1. + ROOT_MARGIN:
        return (f"inadmissible fit (min|AR| = {c_['ar_root_min']:.4f}, "
                f"min|MA| = {c_['ma_root_min']:.4f})")

    return None


def _prepare_asset(asset, Y17_):
    """Select, fit and package one asset's model.

    Returns a small JSON-serialisable dict: enough for any rank to rebuild the
    fitted model with `_filter_block`, without an MPI transfer of a statsmodels
    object.
    """

    cache_path = CACHE / (f'{resource}_asset{asset:03d}_'
                          f'F{"-".join(map(str, FOURIER_K_GRID))}_'
                          f'H{"-".join(map(str, HISTORY_DAYS_GRID))}_'
                          f'D{FIT_DAYS}_N{N_FOLDS}.json')

    if cache_path.exists():
        _c = json.loads(cache_path.read_text())
        _why = _cache_reject_reason(_c)

        if _why is None:
            return _c, _c.get('cv_table')

        print(f'[Rank {RANK}] asset {asset}: discarding cached selection '
              f'({_why}); re-selecting.', flush = True)

    t0 = _time.time()

    if CROSS_VALIDATE:
        best_, table_, orders_ = _cross_validate_asset(asset, Y17_)
        K = int(best_['fourier_k'])
        hd = int(best_['history_days'])
        order = tuple(orders_[K][0])
        route = orders_[K][1]
        wis_val = float(best_['WIS_val'])
    else:
        K = int(CV_DEFAULT['fourier_k'])
        hd = int(CV_DEFAULT['history_days'])
        table_ = []

        a = max(0, (Y17_.size // T_DAY) - FIT_DAYS)
        b = Y17_.size // T_DAY
        s0 = max(a, b - ORDER_SEARCH_DAYS)

        order, route = _select_order(
            Y17_[s0 * T_DAY:b * T_DAY],
            _fourier(np.arange(s0 * T_DAY, b * T_DAY, dtype=float), K)
        )
        wis_val = float('nan')

    # ---- the model that scores the test year ---------------------------------
    #
    # Estimated on the LAST FIT_DAYS days of the training year: the block
    # closest to the scored year, and the only block whose relationship to the
    # test data is the same for every test day.
    n_days_tr = Y17_.size // T_DAY
    a = max(0, n_days_tr - FIT_DAYS)

    y_fit_ = Y17_[a * T_DAY:n_days_tr * T_DAY]
    t_fit_ = np.arange(a * T_DAY, n_days_tr * T_DAY, dtype=float)

    res, order, _note, ar_root_min, ma_root_min = _fit_admissible(
        y_fit_, t_fit_, K, order)

    route = route + _note

    spec_ = {
        'asset': int(asset),
        'fourier_k': K,
        'history_days': hd,
        'order': [int(v) for v in order],
        'order_route': route,
        # Recorded so the cache guard, the results file and a reader can all
        # check admissibility without refitting. inf (no AR or no MA term) is
        # capped so the value stays valid JSON.
        'ar_root_min': float(min(ar_root_min, 999.)),
        'ma_root_min': float(min(ma_root_min, 999.)),
        'fit_day_start': int(a),
        'fit_day_end': int(n_days_tr),
        'params': [float(v) for v in np.asarray(res.params, dtype=float)],
        'llf': float(res.llf),
        'WIS_val': wis_val,
        'cv_seconds': float(_time.time() - t0),
        'cv_table': table_,
    }

    cache_path.write_text(json.dumps(spec_))

    return spec_, table_


# ------------------------------- EXPERIMENTS ----------------------------------

# One test case: filter this asset's own model onto its recent history and
# today's prefix, then simulate to the end of the same day.
def _run_reference(process_, _data, time):

    asset, day = process_

    file_name = f'{asset}-{day}-{time}'

    F_tr_, F_ts_ = _data['ac']

    # try:
    f_hat_ = np.asarray(F_ts_[day, time:, asset], dtype=float)

    res = RES_BY_ASSET_[asset]
    K = SPEC_BY_ASSET_[asset]['fourier_k']
    hd = SPEC_BY_ASSET_[asset]['history_days']

    # ---- the operationally available history -----------------------------
    # Whole days strictly before the scored day, then today up to tau, read
    # out of the bridged series so the first days of the test year are not
    # short-changed. The tail being scored never enters.
    y_hist_, t_hist_, t_fut_ = _history_window(
        Y_BRIDGED_[asset], day, time, hd, bridged = True
    )

    assert len(t_fut_) == len(f_hat_), (len(t_fut_), len(f_hat_))
    assert np.isfinite(y_hist_).all(), 'history contains non-finite values'
    assert len(y_hist_) > 2 * K, (
        f'history of {len(y_hist_)} points is too short to identify '
        f'{2 * K} Fourier terms'
    )

    rng = np.random.default_rng(
        (SEED * 1_000_003) ^ (int(asset) * 1_000_033 + int(day) * 1_009 + int(time))
    )

    M_ = _complete(res, y_hist_, t_hist_, t_fut_, K, rng, N_SAMPLES)

    # ---- bands and scores, identical to the other scripts -----------------
    f_median_ = np.median(M_, axis = 0)
    f_mean_ = M_.mean(axis = 0)

    _upper, _lower = {}, {}
    for alpha in ALPHAS:
        _lower[f'{alpha}'] = np.quantile(M_, alpha / 2., axis = 0)
        _upper[f'{alpha}'] = np.quantile(M_, 1. - alpha / 2., axis = 0)

    pit_ = _empirical_PIT(f_hat_, M_.T, seed = 1234)

    es = _energy_score(M_, f_hat_)
    wis = _weighted_interval_score(f_hat_, f_median_, _lower, _upper, ALPHAS).mean()

    rmse_md, mae_md, _ = _point_errors(f_hat_, f_median_)
    rmse_mn, mae_mn, _ = _point_errors(f_hat_, f_mean_)

    psr_ = np.array([time, asset, day, es, wis,
                        rmse_md, mae_md, rmse_mn, mae_mn])

    prob_ = [
        _score_row(time, asset, day, alpha,
                    f_hat_, _lower[f'{alpha}'], _upper[f'{alpha}'])
        for alpha in ALPHAS
    ]

    frac_boundary = float(np.mean((f_hat_ <= 1e-6) | (f_hat_ >= CAP - 1e-6)))

    det_ = []
    for curve_, label in [(f_median_, 'median'), (f_mean_, 'mean')]:
        r, m, b = _point_errors(f_hat_, curve_)
        det_.append([time, asset, day, 'ECDF', label,
                        r, m, b, frac_boundary])

    stat_ = pd.DataFrame(np.std(M_, axis = 0)).T
    stat_.columns = [f'H{str(i+1).zfill(2)}' for i in range(len(stat_.columns))]
    stat_['time'] = time
    stat_['region'] = asset
    stat_['day'] = day

    func_ = []
    for curve_, label in [(f_median_, 'median'), (f_mean_, 'mean'),
                            (f_hat_, 'actual')]:
        _df = pd.DataFrame([curve_], columns = [f'H{i:02d}' for i in range(len(curve_))])
        _df['type'] = label
        func_.append(_df)

    func_ = pd.concat(func_, axis = 0)
    func_['time'] = time
    func_['asset'] = asset
    func_['day'] = day

    # except Exception as e:
    #     FAILURES.append((file_name, repr(e)))
    #     return None

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
time_arg = sys.argv[3]
init = int(sys.argv[4])
unbiased = bool(int(sys.argv[5]))
description = sys.argv[6]

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

ALPHAS = [0.1, 0.2, 0.3, 0.4]

# Cutoff tau -> (LEAD, INTERVALS) for the PIT block summaries. Unchanged from
# the other three scripts; held as a table rather than an if-chain because this
# script can score every cutoff of a resource in one run.
TAU_GRID = {
    'wind': {
        72:  (36, [0, 36, 72, 108, 144, 180]),
        144: (24, [0, 24, 48,  72,  96, 120]),
        216: (12, [0, 12, 24,  36,  48,  60]),
    },
    'solar': {
        120: (12, [0, 12, 24, 36, 48, 60]),
        132: (9,  [0,  9, 18, 27, 36, 45]),
        144: (9,  [0,  9, 18, 27, 36, 45]),
        168: (6,  [0,  6, 12, 18, 24, 30]),
    },
}

if resource not in TAU_GRID:
    exit("Invalid resource. Must be one of ['wind', 'solar'].")

if str(time_arg).lower() == 'all':
    TAUS = sorted(TAU_GRID[resource].keys())
else:
    _t = int(time_arg)
    if _t not in TAU_GRID[resource]:
        exit(f'Invalid time horizon. Must be one of '
             f'{sorted(TAU_GRID[resource].keys())} or "all".')
    TAUS = [_t]

if RANK == 0:
    print(resource, method, TAUS, init, unbiased, description)
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
    path_to_training_data = DATA / f'preprocessed_{resource}_{YEAR_TR}.pkl',
    path_to_testing_data = DATA / f'preprocessed_{resource}_{YEAR_TS}.pkl',
    T = T_DAY,
)

# The loader returns the TRAINING set as a stacked pool of curves -- (n_curves,
# T), one row per (day, asset) pair -- and keeps the (n_days, T, n_assets)
# layout only for the TEST set.
assert F_tr_.ndim == 2, (
    f'F_tr_ is expected to be a stacked curve pool (n_curves, T); got {F_tr_.shape}.'
)
assert F_ts_.ndim == 3 and F_ts_.shape[1] == T_DAY, (
    f'F_ts_ {F_ts_.shape} expected (n_days, {T_DAY}, n_assets).'
)

# The day-ahead forecast arrays are deliberately NOT placed in `_data`: this
# benchmark is defined by not having access to them, and keeping them out makes
# that impossible to violate by accident.
_data = {
    'ac': [F_tr_, F_ts_],
    'dates': [t_tr_, t_ts_],
    'grid': [dt_, dx_],
}

# Physical upper bound used to clip the simulated paths. Training data only, and
# the same expression the other three scripts use.
CAP = float(max(np.nanmax(F_tr_), 1e-6))

# NO EVALUATION MASK. `validate_asset_ffc_mpi.py` scores the whole remaining
# horizon and carries no `_day` columns, so this benchmark does the same and the
# two are compared on identical quantities. See the note in the module docstring
# for why fusion's `idx_hours_` is not an evaluation mask.


## --------------------- THE TRAINING YEAR AS A CUBE ----------------------------
#
# The cross-validation needs 2017 in chronological order, per asset. The stacked
# pool F_tr_ cannot supply that: the loader does not document whether it is
# stored day-major or asset-major, and `validate_asset_prophet_mpi.py` says so
# explicitly ("nothing assumed about whether the pool is stored day-major or
# asset-major"). Guessing wrong would scramble time silently and the model would
# still fit, which is the worst kind of bug.
#
# So the training year is re-loaded through the TEST slot of the same loader
# call, where the (n_days, T, n_assets) layout is documented and is the one
# `F_ts_[day, s, asset]` already relies on. The fallback below reconstructs the
# cube from the pool only if that second call fails, and it infers the stacking
# order from `t_tr_` rather than assuming one.

def _reconstruct_training_cube(F_tr_, t_tr_, n_assets):
    """Last resort: fold the stacked pool back into (n_days, T, n_assets).

    `t_tr_` carries one day coordinate per curve. If the pool is day-major it is
    non-decreasing with each day repeated n_assets times; if asset-major it
    cycles through the days n_assets times. Those two are distinguishable, so
    the order is detected rather than assumed, and anything else is refused.
    """

    F_tr_ = np.asarray(F_tr_, dtype=float)
    d_ = np.asarray(t_tr_, dtype=float).ravel()

    n_curves, T = F_tr_.shape

    assert d_.size == n_curves, (d_.size, n_curves)
    assert n_curves % n_assets == 0, (
        f'{n_curves} curves is not a whole number of days for {n_assets} assets'
    )

    n_days = n_curves // n_assets

    if np.all(np.diff(d_) >= 0):
        # day-major: [d0]*n_assets, [d1]*n_assets, ...
        C_ = F_tr_.reshape(n_days, n_assets, T).transpose(0, 2, 1)
        order = 'day-major'
    else:
        # asset-major: [d0..dN] repeated n_assets times
        blocks_ = d_.reshape(n_assets, n_days)
        assert np.all(np.diff(blocks_, axis = 1) >= 0), (
            'training pool is neither day-major nor asset-major by t_tr_; '
            'refusing to guess the chronological order'
        )
        C_ = F_tr_.reshape(n_assets, n_days, T).transpose(1, 2, 0)
        order = 'asset-major'

    return np.ascontiguousarray(C_), order


CUBE_ROUTE = None

# try:
_out17 = loader.preprocessed_dataset(
    unbiased = unbiased,
    path_to_training_data = DATA / f'preprocessed_{resource}_{YEAR_TR}.pkl',
    path_to_testing_data = DATA / f'preprocessed_{resource}_{YEAR_TR}.pkl',
    T = T_DAY,
)

F17_ = np.asarray(_out17[1], dtype=float)                # the TEST slot

assert F17_.ndim == 3 and F17_.shape[1] == T_DAY, F17_.shape
assert F17_.shape[2] == F_ts_.shape[2], (
    f'training cube has {F17_.shape[2]} assets, test cube has {F_ts_.shape[2]}'
)
assert F17_.shape[0] * F17_.shape[2] == F_tr_.shape[0], (
    f'training cube {F17_.shape} does not account for the {F_tr_.shape[0]} '
    f'curves in the stacked pool'
)

CUBE_ROUTE = 'loader second call'

del _out17

# except Exception as _e_cube:

#     F17_, _order = _reconstruct_training_cube(F_tr_, t_tr_, F_ts_.shape[2])
#     CUBE_ROUTE = f'reconstructed from pool ({_order}); second load failed: {_e_cube!r}'

assert max(ASSETS) < F_ts_.shape[2], (
    f'ASSETS asks for asset {max(ASSETS)} but the test cube has only '
    f'{F_ts_.shape[2]}'
)
assert F17_.shape[0] > FIT_DAYS + BRIDGE_DAYS, (
    f'training year has {F17_.shape[0]} days, too few for FIT_DAYS = {FIT_DAYS} '
    f'and BRIDGE_DAYS = {BRIDGE_DAYS}'
)
assert max(DAYS) < F_ts_.shape[0], (
    f'DAYS asks for day {max(DAYS)} but the test year has {F_ts_.shape[0]} days'
)


## ------------------- LONG SERIES, PER ASSET, WITH THE BRIDGE ------------------
#
# Y17_BY_ASSET_[a]   the training year, chronological, used by the CV
# Y_BRIDGED_[a]      BRIDGE_DAYS of the training year followed by the whole test
#                    year, used at test time. Only the test-year part is ever
#                    scored; the prefix exists so that day 0 has a history.

Y17_BY_ASSET_ = {}
Y_BRIDGED_ = {}

for _a in ASSETS:

    y17_ = _year_series(F17_, _a)
    y18_ = _year_series(F_ts_, _a)

    Y17_BY_ASSET_[_a] = y17_
    Y_BRIDGED_[_a] = np.concatenate([y17_[-BRIDGE_DAYS * T_DAY:], y18_])

    # The one alignment that everything downstream rests on: the bridged series
    # read at _ts_index(day, s) must be the same number the scoring code reads
    # at F_ts_[day, s, asset]. Checked on the corners and the middle of the year
    # rather than trusted.
    for _d in (0, F_ts_.shape[0] // 2, F_ts_.shape[0] - 1):
        for _s in (0, T_DAY // 2, T_DAY - 1):
            _lhs = Y_BRIDGED_[_a][_ts_index(_d, _s)]
            _rhs = float(F_ts_[_d, _s, _a])
            assert (_lhs == _rhs) or (np.isnan(_lhs) and np.isnan(_rhs)), (
                f'bridged series misaligned for asset {_a} at day {_d}, '
                f'interval {_s}: {_lhs} vs {_rhs}'
            )

    assert Y_BRIDGED_[_a].size == (BRIDGE_DAYS + F_ts_.shape[0]) * T_DAY

if RANK == 0:
    print(f'Series: training year {F17_.shape[0]} days ({CUBE_ROUTE}), '
          f'test year {F_ts_.shape[0]} days, bridge {BRIDGE_DAYS} days, '
          f'{len(ASSETS)} assets, CAP = {CAP:.4f}', flush = True)


## ------------ STAGE 1: ONE MODEL PER ASSET, VALIDATED ON THE TRAINING YEAR ----
#
# Ranks split ASSETS, so each asset's cross-validation is paid exactly once, and
# the resulting specs are allgathered so that every rank can rebuild every
# model for stage 2.

CACHE = PARAM / 'arima'

if RANK == 0:
    CACHE.mkdir(parents = True, exist_ok = True)

COMM.Barrier()

if SIZE > len(ASSETS) and RANK == 0:
    print(f'[Note] -n {SIZE} exceeds {len(ASSETS)} assets: stage 1 leaves '
          f'{SIZE - len(ASSETS)} ranks idle. Stage 2 uses all of them.',
          flush = True)

# np.array_split hands the surplus ranks empty arrays when SIZE > len(ASSETS),
# so no special case is needed: those ranks simply do no stage-1 work.
_local_assets_ = np.array_split(np.array(ASSETS), SIZE)[RANK]

_local_specs_ = []
_local_tables_ = []

for _a in _local_assets_:

    _a = int(_a)
    _t0 = _time.time()

    _spec, _table = _prepare_asset(_a, Y17_BY_ASSET_[_a])

    _local_specs_.append(_spec)
    if _table:
        _local_tables_.extend(_table)

    print(f'[Rank {RANK}] asset {_a}: K = {_spec["fourier_k"]}, '
          f'history = {_spec["history_days"]} d, order = {tuple(_spec["order"])} '
          f'({_spec["order_route"]}), WIS_val = {_spec["WIS_val"]:.5f}, '
          f'{_time.time() - _t0:.1f} s', flush = True)

if CV_FAILURES:
    from collections import Counter
    _kinds = Counter(msg for _, msg in CV_FAILURES)
    print(f'[Rank {RANK}] CV: {len(CV_FAILURES)} failed cases. Modes:', flush = True)
    for _msg, _n in _kinds.most_common(5):
        print(f'[Rank {RANK}]   {_n:5d} x {_msg}', flush = True)

_all_specs_ = COMM.allgather(_local_specs_)
_all_tables_ = COMM.gather(_local_tables_, root = 0)

SPEC_BY_ASSET_ = {}
for _chunk in _all_specs_:
    for _spec in _chunk:
        SPEC_BY_ASSET_[int(_spec['asset'])] = _spec

assert set(SPEC_BY_ASSET_.keys()) == set(ASSETS), (
    f'stage 1 produced specs for {sorted(SPEC_BY_ASSET_.keys())}, '
    f'expected {ASSETS}'
)

# Final gate: nothing inadmissible reaches the test year. _fit_admissible has
# already stepped down the ladder, so an asset arriving here inadmissible means
# the ladder was exhausted - which is a result to investigate, not to average.
_bad_ = {_a: (SPEC_BY_ASSET_[_a]['ar_root_min'], SPEC_BY_ASSET_[_a]['ma_root_min'])
         for _a in ASSETS
         if min(SPEC_BY_ASSET_[_a]['ar_root_min'],
                SPEC_BY_ASSET_[_a]['ma_root_min']) <= 1. + ROOT_MARGIN}

assert not _bad_, (
    f'inadmissible fits reached stage 2, asset: (min|AR root|, min|MA root|) '
    f'{_bad_}. An AR root inside the unit circle gives diverging simulated '
    f'paths and a band that is the whole [0, CAP] range.'
)

# Every rank rebuilds every model from the stored parameters. `filter` runs the
# Kalman filter at fixed parameters - no optimisation - so this is cheap and
# gives byte-identical models on every rank.
RES_BY_ASSET_ = {}

for _a in ASSETS:

    _spec = SPEC_BY_ASSET_[_a]

    _y17 = Y17_BY_ASSET_[_a]
    _a0, _b0 = int(_spec['fit_day_start']), int(_spec['fit_day_end'])

    RES_BY_ASSET_[_a] = _filter_block(
        _y17[_a0 * T_DAY:_b0 * T_DAY],
        np.arange(_a0 * T_DAY, _b0 * T_DAY, dtype=float),
        int(_spec['fourier_k']),
        tuple(_spec['order']),
        _spec['params'],
    )

if RANK == 0:

    print('----- MODELS -----')
    for _a in ASSETS:
        _s = SPEC_BY_ASSET_[_a]
        print(f'  asset {_a:3d}  K = {_s["fourier_k"]:2d}  '
              f'history = {_s["history_days"]:2d} d  '
              f'order = {tuple(_s["order"])}  '
              f'min|AR| = {_s["ar_root_min"]:7.3f}  '
              f'min|MA| = {_s["ma_root_min"]:7.3f}  '
              f'WIS_val = {_s["WIS_val"]:.5f}  '
              f'llf = {_s["llf"]:.1f}', flush = True)

    # The per-asset selection, written once so the choice is auditable.
    _sel_rows = [{'asset': _a,
                  'fourier_k': SPEC_BY_ASSET_[_a]['fourier_k'],
                  'history_days': SPEC_BY_ASSET_[_a]['history_days'],
                  'arima_order': str(tuple(SPEC_BY_ASSET_[_a]['order'])),
                  'order_route': SPEC_BY_ASSET_[_a]['order_route'],
                  'ar_root_min': SPEC_BY_ASSET_[_a]['ar_root_min'],
                  'ma_root_min': SPEC_BY_ASSET_[_a]['ma_root_min'],
                  'WIS_val': SPEC_BY_ASSET_[_a]['WIS_val'],
                  'llf': SPEC_BY_ASSET_[_a]['llf'],
                  'fit_day_start': SPEC_BY_ASSET_[_a]['fit_day_start'],
                  'fit_day_end': SPEC_BY_ASSET_[_a]['fit_day_end'],
                  'cv_seconds': SPEC_BY_ASSET_[_a]['cv_seconds'],
                  'initialization': init}
                 for _a in ASSETS]

    _sel_path = VALIDATION / f'{resource}/{resource}_{method}_asset-selection-{description}.csv'
    _sel_df = _read_csv_safe(_sel_path)
    _sel_df = pd.concat([_sel_df, pd.DataFrame(_sel_rows)], ignore_index = True)
    _sel_df.to_csv(_sel_path, index = False)
    print(f'Saved per-asset selection to {_sel_path}')

    # The full CV score table, so the selection can be shown to be a minimum and
    # not a coin flip between neighbouring combinations.
    _flat_tables_ = [r for chunk in (_all_tables_ or []) for r in (chunk or [])]

    if _flat_tables_:
        _cv_path = VALIDATION / f'{resource}/{resource}_{method}_asset-cv-{description}.csv'
        _cv_df = _read_csv_safe(_cv_path)
        _cv_new = pd.DataFrame(_flat_tables_)
        _cv_new['initialization'] = init
        _cv_df = pd.concat([_cv_df, _cv_new], ignore_index = True)
        _cv_df.to_csv(_cv_path, index = False)
        print(f'Saved CV table to {_cv_path}')


## ----------------------------- SCORING PASS ----------------------------------
#
# One pass per cutoff. The model does not change with tau - tau changes what has
# been observed, not what the model is - so every cutoff reuses the models built
# above and writes exactly the files a separate per-tau invocation would have.

processes_test_ = [(asset, j) for asset in ASSETS for j in DAYS]

for time in TAUS:

    LEAD, INTERVALS = TAU_GRID[resource][time]

    if RANK == 0:
        print(f'----- TEST tau = {time} -----', flush = True)

    pit_, psr_, prob_, det_, stat_, func_ = _run_parallel_mpi(_data, processes_test_, time)

    if RANK != 0:
        continue

    if pit_ is None:
        print(f'[Rank 0] tau = {time}: every case failed; nothing written.', flush = True)
        continue

    # ---- aggregated scoring rules and PIT ------------------------------------
    ks_ = np.array([_KS(pit_[:, j:(j + LEAD)].flatten()) for j in INTERVALS])
    ks_labels_ = [f'S{i}' for i in range(len(INTERVALS))]

    psr_ = psr_.astype(float)

    row_ = {
        'initialization': init,
        'time': time,

        'ES_test': float(np.mean(psr_[:, 3])),
        'WIS_test': float(np.mean(psr_[:, 4])),

        # The reference itself: the simulated median.
        'RMSE_test': float(np.mean(psr_[:, 5])),
        'MAE_test': float(np.mean(psr_[:, 6])),

        # The simulated MEAN, for column parity with the other runs.
        #
        # RENAMED 2026-09-19. These were written as RMSE_median_test /
        # MAE_median_test, but psr_[:, 7] is rmse_mn and psr_[:, 8] is mae_mn -
        # the mean. RMSE_test above is already the median, consistent with
        # point_forecast. Any table built against the old names labelled the
        # mean as the median; see finding_zone_rmse_pointforecast_mixup.md.
        'RMSE_mean_test': float(np.mean(psr_[:, 7])),
        'MAE_mean_test': float(np.mean(psr_[:, 8])),

        'KS_test': float(np.mean(ks_)),

        **{ks_labels_[i] + '_test': float(ks_[i]) for i in range(len(INTERVALS))},

        # Settings, recorded so a results file is self-describing. The tuned
        # quantities are per asset, so both the summary and the full mapping are
        # written; `{resource}_{method}_asset-selection-*.csv` holds the same
        # thing one row per asset.
        'model': 'dynamic-harmonic-regression-per-asset',
        'point_forecast': 'median',
        'fourier_k': _modal([SPEC_BY_ASSET_[a]['fourier_k'] for a in ASSETS]),
        'history_days': _modal([SPEC_BY_ASSET_[a]['history_days'] for a in ASSETS]),
        'fourier_k_by_asset': json.dumps(
            {str(a): SPEC_BY_ASSET_[a]['fourier_k'] for a in ASSETS}),
        'history_days_by_asset': json.dumps(
            {str(a): SPEC_BY_ASSET_[a]['history_days'] for a in ASSETS}),
        'arima_order_by_asset': json.dumps(
            {str(a): str(tuple(SPEC_BY_ASSET_[a]['order'])) for a in ASSETS}),
        'order_route': SPEC_BY_ASSET_[ASSETS[0]]['order_route'],
        'order_route_by_asset': json.dumps(
            {str(a): SPEC_BY_ASSET_[a]['order_route'] for a in ASSETS}),

        # Admissibility of the fitted models, so a reader can check it without
        # refitting. Both must exceed 1; see ROOT_MARGIN and _fit_admissible.
        'ar_root_min_by_asset': json.dumps(
            {str(a): round(float(SPEC_BY_ASSET_[a]['ar_root_min']), 4)
             for a in ASSETS}),
        'ma_root_min_by_asset': json.dumps(
            {str(a): round(float(SPEC_BY_ASSET_[a]['ma_root_min']), 4)
             for a in ASSETS}),
        'enforce_stationarity': ENFORCE,
        'enforce_invertibility': ENFORCE,
        'max_p': MAX_P,
        'max_q': MAX_Q,
        'root_margin': ROOT_MARGIN,
        'WIS_val_mean': float(np.nanmean([SPEC_BY_ASSET_[a]['WIS_val'] for a in ASSETS])),
        'cross_validated': CROSS_VALIDATE,
        'n_folds': N_FOLDS,
        'cv_val_days': CV_VAL_DAYS,
        'gap_days': GAP_DAYS,
        'fit_days': FIT_DAYS,
        'bridge_days': BRIDGE_DAYS,
        'year_train': YEAR_TR,
        'year_test': YEAR_TS,
        'cube_route': CUBE_ROUTE,
        'state_uncertainty': STATE_UNCERTAINTY,
        'n_samples': N_SAMPLES,
        'n_assets': len(ASSETS),
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
              'n_excursions', 'len_excursion', 'frac_out', 'frac_degenerate']:
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
    for c in ['RMSE', 'MAE', 'MBE', 'frac_boundary']:
        det_df[c] = pd.to_numeric(det_df[c], errors = 'coerce')

    det_agg_ = det_df.groupby(['time', 'point_forecast']).agg(
        {'RMSE': 'mean', 'MAE': 'mean', 'MBE': 'mean',
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

if RANK == 0:
    print('----- DONE -----')
