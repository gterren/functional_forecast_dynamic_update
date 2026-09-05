import sys

PROJECT = "/Users/Guille/Desktop/dynamic_update/functional_forecast_dynamic_update"

sys.path.insert(0, PROJECT)

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

from skfda import FDataGrid
from skfda.exploratory.depth import IntegratedDepth, ModifiedBandDepth

from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib import cm

from src import loader, plotter
from datetime import timedelta

from src.config import DATA, DEPTH, IMAGES, VALIDATION, TABLES
from src.fda import _fDepth, _fQuantile
from src.fdu import functional_dynamic_update
from src.utils import KS, get_band_fraction, get_hyper, mask_intervals

from src.fdu import functional_dynamic_update

# Loading color palette
palette_ = pd.read_csv(DATA / "palette.csv")

# Loading Texas map
_TX = gpd.read_file(DATA / "maps/TX/State.shp")

T = 48
method = 'fusion'
resource = 'wind'
aggregation = 'zone'

# exp_description="unbiased-025-C0"
# _init = {6: 1, 12: 3, 18: 4}

# exp_description="unbiased-025-C1"
# _init = {6: 3, 12: 1, 18: 4}

# exp_description="unbiased-025-C2"
# _init = {6: 2, 12: 5, 18: 5}

# exp_description="biased-025-C0"
# _init = {6: 3, 12: 5, 18: 2}

# exp_description="biased-025-C1"
# _init = {6: 1, 12: 4, 18: 1}

# exp_description="biased-025-C2"
# _init = {6: 4, 12: 4, 18: 1}

# exp_description="unbiased-025-C1"
# _init = {6: 5, 12: 1, 18: 2}

exp_description="unbiased-025-C1-1"
_init = {6: 5, 12: 4, 18: 3}

exp_description="unbiased-025-C1-6"
_init = {6: 4, 12: 4, 18: 4}

day = 176
zone = 1
interval = 18

alpha_ = [0.1, 0.2, 0.3, 0.4]
_depth = ModifiedBandDepth()

scale = 10

hyper_, envelope_ = loader.hyperparameters(
    _init,
    resource,
    method,
    aggregation,
    exp_description,
    path_to_validation=VALIDATION,
)

hyper_.loc['length_scale_f'] = 1./hyper_.loc['tau']
hyper_.loc['length_scale_e'] = hyper_.loc['length_scale_f']*hyper_.loc['rho_e']
hyper_.loc['length_scale_d'] = hyper_.loc['length_scale_f']*hyper_.loc['rho_d']
print(hyper_)

dates_ = np.random.default_rng(1234).permutation(np.arange(360))[180:]

(F_tr_, F_ts_, 
 E_tr_, E_ts_, 
 d_tr_, d_ts_, 
 t_tr_, t_ts_, 
 X_tr_, X_ts_, 
 dt_, regions_, region
) = loader.processed_zone_dataset(
    path_to_data = DATA / "gridstatus/processed_ERCOT_wind_data_v4.pkl",
    zone = zone,
    N_test_samples = 360,
    unbiased = True,
)

print(F_tr_.shape, F_ts_.shape)
print(E_tr_.shape, E_ts_.shape)
print(d_tr_.shape, d_ts_.shape)
print(t_tr_.shape, t_ts_.shape)
print(X_tr_.shape, X_ts_.shape)
print(dt_.shape)
print(region)
print(regions_)

dx_ = np.array([
    (
        t_ts + timedelta(minutes=0)
    ).strftime('%b %-d %-I%p').replace('AM', 'am').replace('PM', 'pm') 
    for t_ts in t_ts_[day, interval]
])

# ===============================================

file_name = f"{resource}_zone-{region}_{day}_{interval}"
print(file_name)

_fdu = functional_dynamic_update(
    _distances = {'temporal': 'seasonal_equinox', 'spatial': 'graph', 'fusion': 'None'},
    name = region,
    date = dx_[24],
)
print(_fdu.name, _fdu.date)

F_tr_p_ = F_tr_[:, interval, :]
E_tr_p_ = np.concatenate([np.zeros((E_tr_.shape[0], 24)), E_tr_[:, interval, :]], axis = 1)
print(F_tr_p_.shape, E_tr_p_.shape)

_fdu.fit(
    F_tr_p_, E_tr_p_, dt_, 
    X_ = X_tr_, 
    t_ = d_tr_, 
    n_samples_per_hour=1,
)

f_ = F_ts_[day, interval, :24]
f_hat_ = F_ts_[day, interval, 24:]
e_ = E_ts_[day, interval, ...]
e_p_ = np.concatenate([np.zeros(24,), e_, ], axis = 0)
print(f_.shape, f_hat_.shape, e_.shape, e_p_.shape)

M_ = _fdu.predict(
    f_, e_p_, X_ts_[day], d_ts_[day],
    clique_order = get_hyper(hyper_, "clique_order", interval),
    forget_rate_f = get_hyper(hyper_, "forget_rate_f", interval),
    forget_rate_e = get_hyper(hyper_, "forget_rate_e", interval),
    lookahead_rate = get_hyper(hyper_, "lookahead_rate", interval),
    length_scale_f = get_hyper(hyper_, "length_scale_f", interval),
    length_scale_e = get_hyper(hyper_, "length_scale_e", interval),
    length_scale_d = get_hyper(hyper_, "length_scale_d", interval),
    sigma = get_hyper(hyper_, "sigma", interval),
    nu = get_hyper(hyper_, "nu", interval),
    kappa_0 = get_hyper(hyper_, "kappa", interval),
    kappa = get_hyper(hyper_, "kappa", interval),
    p_fusion = get_hyper(hyper_, "p_fusion", interval),
)
print(_fdu.xi, _fdu.r)

# Plotting variables
e_p_[:24] = np.nan
E_tr_p_[:, :24] = np.nan

# # Calculate confidence intervals from Directional Quantiles
# f_median_ext_, _upper, _lower = _fdu.ecdf_confidence_bands(
#     _fdu.M_ext_, 
#     alpha_,
# )

f_wmedian_ext_, _wupper, _wlower = _fdu.weighted_ecdf_confidence_region(
    _fdu.M_ext_, 
    _fdu.w_prime_,
    alpha_,
)

_fig, _ax = plt.subplots(
    figsize=(7.5, 2.25), 
    constrained_layout=True
)

plotter.plot_envelope(
    _fig, 
    _ax, 
    palette_,
    _wupper,
    _wlower, 
    f_wmedian_ext_, 
    f_, 
    f_hat_, 
    e_p_, 
    dx_,
    dt_ = dt_ + 60, 
    interval = 24,
    n = 720,
    color_med=palette_.loc[3, "ibm"],
    CR=r"$\mathcal{{R}}^{{mar}}_{{{}}}$",
    label = r"$\hat{\mu}_{med} (s)$",
    range_=[0, 71],
    legend_1=False,
    legend_2=True,
)

_ax.legend(
    frameon = False,
    loc = (0.0125, .225),
    #loc = 'lower center',
    fontsize = 13,
    columnspacing = 0.25,
    handletextpad = 0.125,
    labelspacing = 0.125,
    ncol = 1,
)

plt.savefig(IMAGES / f"median_mar_regions-{file_name}.pdf")

plt.show()

# Samples in each confidence band
score = 'FCS'
distance = "fknn"

J_ = _fdu.focal_curve_envelope(
    None, 
    _fdu.M_ext_, 
    distance,
)
print(J_.shape)

k_ = get_band_fraction(
    envelope_, 
    alpha_, 
    interval, 
    distance, 
    score,
)

f_focal_ext_, _upper, _lower = _fdu.focal_envelope_confidence_region(
    alpha_, 
    k_,
)

_fig, _ax = plt.subplots(
    figsize=(7.5, 2.25), 
    constrained_layout=True,
)

plotter.plot_envelope(
    _fig, 
    _ax, 
    palette_,
    _upper, 
    _lower, 
    f_focal_ext_, 
    f_, 
    f_hat_, 
    e_p_, 
    dx_,
    dt_ = dt_ + 60, 
    interval = 24,
    n = 720,
    color_med=palette_.loc[4, "ibm"],
    label=r"$\hat{\mu}_{focal} (s)$",
    CR=r"$\mathcal{{R}}^{{prj}}_{{{}}}$",
    range_=[0, 71],
    legend_1=False,
    legend_2=True,
)

_ax.legend(
    frameon = False,
    loc = (0.245, .75),
    #loc = 'lower center',
    fontsize = 13,
    columnspacing = 0.25,
    handletextpad = 0.125,
    labelspacing = 0.125,
    ncol = 5,
)

plt.savefig(IMAGES / f"focal_prj_regions-{file_name}.pdf")

plt.show()

_fig, _ax = plt.subplots(
    figsize=(2, 4), 
    constrained_layout=True,
)

print(t_ts_[day, interval, 12])

plotter.scenarios_frequency_dates(
    _fig, 
    _ax, 
    palette_,
    _fdu.idx_fed_, 
    _fdu.idx_x_,
    t_tr_[:, interval, 12], 
    t_ts_[day, interval, 12],
    scale = scale,
    colorbar = True,
)

plt.savefig(IMAGES / f"temporal_neighbors-{file_name}.pdf")

plt.show()

# Plot
_fig, _ax = plt.subplots(
    figsize=(3.75, 2), 
    constrained_layout=True,
)

plotter.plot_zone_neighborhood(
    _fig, 
    _ax, 
    palette_,
    X_tr_, 
    regions_, 
    _fdu.idx_x_, 
    region,
)

plt.savefig(IMAGES / f"spatial_neighbors-{file_name}.pdf")

plt.show()

distance = 'MBD'

depth_score_, depth_rank_ = _fdu.get_depth(_depth, E_tr_p_[_fdu.idx_x_, :])

# Calculate confidence intervals from Directional Quantiles
f_deepest_ext_, _upper, _lower = _fdu.functional_boxplot(
    E_tr_p_[_fdu.idx_x_, :], 
    depth_score_,
)

_fig, _ax = plt.subplots(
    figsize=(7.5, 2.25), 
    constrained_layout=True
)

plotter.plot_enhanced_functional_boxplot(
    _fig, _ax, palette_,
    _upper, _lower, f_deepest_ext_, f_, f_hat_, e_p_, dx_,
    dt_ = dt_ + 60, 
    interval = 24,
    n = 720,
    range_=[0, 71],
    CR=r"$\mathcal{{R}}^{{bxp}}_{{{}}}$",
    legend_1=False,
    legend_2=False,
)

_ax.legend(
    frameon = False,
    loc = (0.3675, .55),
    #loc = 'lower center',
    fontsize = 13,
    columnspacing = 0.25,
    handletextpad = 0.125,
    labelspacing = 0.125,
    ncol = 1,
)

plt.savefig(IMAGES / f"{distance}_box_forecast-{file_name}.pdf")

plt.show()

depth_score_, depth_rank_ = _fdu.get_depth(_depth, F_tr_p_[_fdu.idx_x_, :])

# Calculate confidence intervals from Directional Quantiles
f_deepest_ext_, _upper, _lower = _fdu.functional_boxplot(
    F_tr_p_[_fdu.idx_x_, :], 
    depth_score_,
)

_fig, _ax = plt.subplots(
    figsize=(7.5, 2.25), 
    constrained_layout=True,
)

plotter.plot_enhanced_functional_boxplot(
    _fig, _ax, palette_,
    _upper, _lower, f_deepest_ext_, f_, f_hat_, e_p_, dx_,
    dt_ = dt_ + 60, 
    interval = 24,
    n = 720,
    range_=[0, 71],
    CR=r"$\mathcal{{R}}^{{bxp}}_{{{}}}$",
    legend_1=False,
    legend_2=False,
)

_ax.legend(
    frameon = False,
    loc = (0.3675, .55),
    #loc = 'lower center',
    fontsize = 13,
    columnspacing = 0.25,
    handletextpad = 0.125,
    labelspacing = 0.125,
    ncol = 1,
)

plt.savefig(IMAGES / f"{distance}_box_forecast-{file_name}.pdf")

plt.show()

depth_score_, depth_rank_ = _fdu.get_depth(_depth, _fdu.M_)

# Calculate confidence intervals from Directional Quantiles
f_deepest_ext_, _upper, _lower = _fdu.functional_boxplot(
    M_, 
    depth_score_,
)

_fig, _ax = plt.subplots(
    figsize=(7.5, 2.25), 
    constrained_layout=True,
)

plotter.plot_enhanced_functional_boxplot(
    _fig, _ax, palette_,
    _upper, _lower, f_deepest_ext_, f_, f_hat_, e_p_, dx_,
    dt_ = dt_ + 60, 
    interval = 24,
    n = 720,
    range_=[0, 71],
    CR=r"$\mathcal{{R}}^{{bxp}}_{{\alpha={}}}$",
    legend_1=False,
    legend_2=False,
)

_ax.legend(
    frameon = False,
    loc = (0.0125, .775),
    #loc = 'lower center',
    fontsize = 13,
    columnspacing = 0.25,
    handletextpad = 0.125,
    labelspacing = 0.125,
    ncol = 5,
)

plt.savefig(IMAGES / f"{distance}_box_fused-{file_name}.pdf")

plt.show()

INTERVALS = [0, 6, 12, 18, 24, 30]

pit_ = loader.pit(
    _init, 
    resource, 
    method, 
    aggregation, 
    interval, 
    exp_description,
    VALIDATION,
)

INTERVAL = INTERVALS[0]
LEAD = 24
print((INTERVAL), (INTERVAL + LEAD))

_fig, _ax = plt.subplots(
    figsize = (2.25, 2.25), 
    constrained_layout = True
)

plotter.plot_pit(
    _fig, _ax, pit_[:, INTERVAL:(INTERVAL + LEAD)].flatten(), 
    v = 0.4,
    bins = 10,
    xlabel = 'PIT (from 6pm to 6pm + 24h)',
)

plt.savefig(IMAGES / f"PIT-{resource}-{INTERVAL}-{LEAD}-{interval}.pdf")

plt.show()

INTERVAL = INTERVALS[4]
LEAD = 24
print((INTERVAL), (INTERVAL + LEAD))

_fig, _ax = plt.subplots(
    figsize = (2.25, 2.25), 
    constrained_layout = True
)

plotter.plot_pit(
    _fig, _ax, pit_[:, INTERVAL:(INTERVAL + LEAD)].flatten(), 
    v = 0.4,
    bins = 10,
    xlabel = 'PIT (from 6pm + 24h to 6pm + 48h)',
)

plt.savefig(IMAGES / f"PIT-{resource}-{INTERVAL}-{LEAD}-{interval}.pdf")

plt.show()

day = 215
alpha_ = [0.1, 0.2, 0.3, 0.4]
_depth = ModifiedBandDepth()

F_curves_ = []
E_curves_ = []
M_curves_ = []
D_curves_ = []
G_curves_ = []
dxs_ = []
dts_ = []

#for i, day in enumerate([213, 214, 215, 216, 217, 218, 219, 220, 221]):
i = 0
for interval in range(0, 24, 1):

    dx_ = np.array([
        (
            t_ts + timedelta(minutes=0)
        ).strftime('%b %-d %-I%p').replace('AM', 'am').replace('PM', 'pm') 
        for t_ts in t_ts_[day, interval]
    ])

    F_tr_p_ = F_tr_[:, interval, :]
    E_tr_p_ = np.concatenate([np.zeros((E_tr_.shape[0], 24)), E_tr_[:, interval, :]], axis = 1)
    #print(F_tr_p_.shape, E_tr_p_.shape)
    
    _fdu.fit(
        F_tr_p_, E_tr_p_, dt_, 
        X_ = X_tr_, 
        t_ = d_tr_, 
        n_samples_per_hour=1,
    )

    f_ = F_ts_[day, interval, :24]
    f_hat_ = F_ts_[day, interval, 24:]
    e_ = E_ts_[day, interval, ...]
    e_p_ = np.concatenate([np.zeros(24,), e_, ], axis = 0)
    #print(f_.shape, f_hat_.shape, e_.shape, e_p_.shape)
    
    M_ = _fdu.predict(
        f_, e_p_, X_ts_[day], d_ts_[day],
        clique_order = get_hyper(hyper_, "clique_order", interval),
        forget_rate_f = get_hyper(hyper_, "forget_rate_f", interval),
        forget_rate_e = get_hyper(hyper_, "forget_rate_e", interval),
        lookahead_rate = get_hyper(hyper_, "lookahead_rate", interval),
        length_scale_f = get_hyper(hyper_, "length_scale_f", interval),
        length_scale_e = get_hyper(hyper_, "length_scale_e", interval),
        length_scale_d = get_hyper(hyper_, "length_scale_d", interval),
        sigma = get_hyper(hyper_, "sigma", interval),
        nu = get_hyper(hyper_, "nu", interval),
        kappa_0 = get_hyper(hyper_, "kappa", interval),
        kappa = get_hyper(hyper_, "kappa", interval),
        p_fusion = get_hyper(hyper_, "p_fusion", interval),
    )

    f_wmedian_ext_, _wupper, _wlower = _fdu.weighted_ecdf_confidence_region(
        _fdu.M_ext_, 
        _fdu.w_prime_,
        alpha_,
    )
        
    k_ = get_band_fraction(
        envelope_, 
        alpha_, 
        interval, 
        "MBD", 
        "FCS",
    )
    
    f_deepest_ext_, _upper, _lower = _fdu.adjusted_functional_confidence_region(
        _depth, 
        _fdu.M_ext_, 
        alpha_, 
        k_,
    )
    
    k_ = get_band_fraction(
        envelope_, 
        alpha_, 
        interval, 
        "fknn", 
        "FCS",
    )
    
    J_ = _fdu.focal_curve_envelope(
        None, 
        _fdu.M_ext_, 
        "fknn",
    )
    
    f_focal_ext_, _upper, _lower = _fdu.focal_envelope_confidence_region(
        alpha_, 
        k_,
    )

    e_ext_ = np.concatenate([f_wmedian_ext_[0][np.newaxis], e_], axis = 0)
    M_curves_.append(f_wmedian_ext_)
    E_curves_.append(e_ext_)
    D_curves_.append(f_deepest_ext_)
    F_curves_.append(f_hat_)
    G_curves_.append(f_focal_ext_)

    dts_.append(dt_ + interval*60 + i*24*60)
    dxs_.append(dx_)
        
print(len(F_curves_), len(M_curves_), len(D_curves_))

_fig, _ax = plt.subplots(
    figsize = (7.5, 2.25), 
    constrained_layout = True,
)

plotter.plot_zonal_dynamic_update(
    _fig, 
    _ax, 
    palette_, 
    F_curves_, 
    G_curves_, 
    dxs_, 
    dts_, 
    range_=[0, 71],
    n = 220,
)

plt.savefig(IMAGES / f"focal_update-{file_name}.pdf")

plt.show()