import sys

PROJECT = "/Users/Guille/Desktop/dynamic_update/functional_forecast_dynamic_update"

sys.path.insert(0, PROJECT)

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from skfda import FDataGrid
from skfda.exploratory.depth import IntegratedDepth, ModifiedBandDepth

from matplotlib_scalebar.scalebar import ScaleBar

from src import loader, plotter

from src.config import DATA, DEPTH, IMAGES, VALIDATION, TABLES
from src.fda import _fDepth, _fQuantile
from src.utils import KS, get_band_fraction, get_hyper, mask_intervals

from src.fdu import functional_dynamic_update

# Loading color palette
palette_ = pd.read_csv(DATA / "palette.csv")

# Loading Texas map
_TX = gpd.read_file(DATA / "maps/TX/State.shp")

T = 288
method = "fusion"
resource = "wind"
aggregation = "asset"

# wind unbiased-025 {72: 1, 144: 4, 216: 1}
# wind biased-025 {72: 2, 144: 1, 216: 2}
exp_description="unbiased-025-1"
_init = {72: 1, 144: 2, 216: 1}

exp_description="unbiased-025-2"
_init = {72: 1, 144: 3, 216: 3}

exp_description="unbiased-025-3"
_init = {72: 1, 144: 2, 216: 2}

exp_description="unbiased-025-6"
_init = {72: 3, 144: 4, 216: 1}

# exp_description="biased-025"
# _init = {72: 5, 144: 2, 216: 1}

# Blanco Canyon 18 112 in 176
# Black Jack Creek Wind 17 289
day = 176
asset = 18              
interval = 144

alpha_ = [0.1, 0.2, 0.3, 0.4]
_depth = ModifiedBandDepth()

scale = 10

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
    unbiased = True,
    path_to_training_data = DATA / "preprocessed_wind_2017.pkl",
    path_to_testing_data = DATA / "preprocessed_wind_2018.pkl",
    T = T,
)

print(F_tr_.shape, F_ts_.shape)
print(E_tr_.shape, E_ts_.shape)
print(X_tr_.shape, X_ts_.shape)
print(T_tr_.shape, T_ts_.shape)
print(assets_tr_.shape, assets_ts_.shape)
print(t_tr_.shape, t_ts_.shape)
print(dt_.shape, dx_.shape)

E_tr_lin_, E_ts_lin_ = loader.processed_dataset(
    unbiased = True,
    path_to_training_data = DATA / "linear_preprocessed_wind_2017.pkl",
    path_to_testing_data = DATA / "linear_preprocessed_wind_2018.pkl",
    T = T,
)

print(E_tr_lin_.shape, E_ts_lin_.shape)

E_tr_biased_, E_ts_biased_ = loader.processed_dataset(
    unbiased = False,
    path_to_training_data = DATA / "preprocessed_wind_2017.pkl",
    path_to_testing_data = DATA / "preprocessed_wind_2018.pkl",
    T = T,
)

print(E_tr_biased_.shape, E_ts_biased_.shape)

hyper_, envelope_ = loader.hyperparameters(
    _init,
    resource,
    method,
    aggregation,
    exp_description,
    path_to_validation = VALIDATION,
)

hyper_.loc['length_scale_f'] = 1./hyper_.loc['tau']
hyper_.loc['length_scale_e'] = hyper_.loc['length_scale_f']*hyper_.loc['rho_e']
hyper_.loc['length_scale_d'] = hyper_.loc['length_scale_f']*hyper_.loc['rho_d']
print(hyper_)
# print(envelope_)

stats_, funcs_ = loader.get_asset_stats(
    _init, 
    resource, 
    method, 
    aggregation, 
    exp_description, 
    T, 
    VALIDATION, 
)
# print(stats_)
# print(funcs_)

file_name = f"{resource}_asset-{asset}_{day}_{interval}"
print(file_name)

_fdu = functional_dynamic_update(
    _distances = {"temporal": "seasonal_equinox", 
                  "spatial": "haversine", 
                  "fusion": "dynamic"},
    name = assets_ts_[asset],
    date = T_ts_[day, interval],
)

# Filter solar hours with loading solar set
interval_mask = mask_intervals(
    F_tr_, 
    E_tr_biased_, 
    t_tr_, 
    day, 
    day_window = 7,
)

_fdu.fit(
    F_tr_,
    E_tr_lin_,
    dt_,
    X_ = X_tr_,
    t_ = t_tr_,
    interval_mask = interval_mask,
    n_samples_per_hour = 12,
)

M_ = _fdu.predict(
    F_ts_[day, :interval, asset], 
    E_ts_lin_[day, :, asset], 
    X_ts_[asset, :], 
    t_ts_[day],
    forget_rate_f = get_hyper(hyper_, "forget_rate_f", interval),
    forget_rate_e = get_hyper(hyper_, "forget_rate_e", interval),
    lookahead_rate = get_hyper(hyper_, "lookahead_rate", interval),
    length_scale_f = get_hyper(hyper_, "length_scale_f", interval),
    length_scale_e = get_hyper(hyper_, "length_scale_e", interval),
    length_scale_d = get_hyper(hyper_, "length_scale_d", interval),
    sigma = get_hyper(hyper_, "sigma", interval),
    nu = get_hyper(hyper_, "nu", interval),
    kappa_0 = get_hyper(hyper_, "kappa_0", interval),
    kappa = get_hyper(hyper_, "kappa", interval),
    p_fusion = get_hyper(hyper_, "p_fusion", interval),
)
print(_fdu.xi, _fdu.ess)

# Plotting variables
e_biased_ = E_ts_biased_[day, :, asset]
f_hat_ = F_ts_[day, interval:, asset]
e_lin_ = E_ts_lin_[day, :, asset]
x_ts_ = X_ts_[asset, :]
e_ = E_ts_[day, :, asset]
f_ = F_ts_[day, :interval, asset]

# Calculate confidence intervals from Directional Quantiles
M_int_, M_int_ds_ = _fdu.functional_downsampling(
    subsample=12, 
    n_basis=int(1.333*(T - interval)/12),
)

distance = 'MBD'

depth_score_, depth_rank_ = _fdu.get_depth(
    _depth, 
    M_int_,
)

_fig, _ax = plt.subplots(
    figsize=(7.5, 2.25), 
    constrained_layout=True
)

plotter.plot_depth(
    _fig, _ax, palette_,
    M_, f_, f_hat_, e_biased_, depth_score_, dx_, 
    dt_ = dt_ + 5, 
    interval = interval,
    n = 120,
    range_ = [0, 287],
    colorbar_pos = [500, 75, 150, 5],
    colorbar = True,
    labels_1 = True,
)

_ax.legend(
    frameon = False,
    loc = (0.0125, .775),
    #loc = 'lower center',
    fontsize = 13,
    columnspacing = 0.25,
    handletextpad = 0.125,
    labelspacing = 0.125,
    ncol = 1,
)

plt.savefig(IMAGES / f"{distance}_ranking-{file_name}.pdf")

plt.show()

# Samples in each confidence band
score = 'FCS'
distance = 'MBD'
_depth = ModifiedBandDepth()

f_deepest_ext_, _upper, _lower = _fdu.functional_confidence_region(
    _depth, 
    M_int_, 
    alpha_, 
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
    f_deepest_ext_, 
    f_, 
    f_hat_, 
    e_biased_, 
    dx_, 
    dt_ = dt_ + 5, 
    interval = interval,
    n = 120,
    color_med = palette_.loc[2, "ibm"],
    label = r"$\hat{\mu}_{deep} (s)$",
    CR=r"$\mathcal{{R}}^{{fun}}_{{{}}}$",
    range_ =[0, 287],
    legend_1 = False,
    legend_2 = True,
)

_ax.legend(
    frameon = False,
    loc = (0.0125, .2),
    #loc = 'lower center',
    fontsize = 13,
    columnspacing = 0.25,
    handletextpad = 0.125,
    labelspacing = 0.125,
    ncol = 1,
)

plt.savefig(IMAGES / f"{distance}_fun_regions-{file_name}.pdf")

plt.show()

# Samples in each confidence band
score = 'FCS'
distance = 'MBD'
_depth = ModifiedBandDepth()

k_ = get_band_fraction(
    envelope_, 
    alpha_, 
    interval, 
    distance, 
    score,
)

f_deepest_ext_, _upper, _lower = _fdu.adjusted_functional_confidence_region(
    _depth, 
    M_int_, 
    alpha_, 
    k_,
)

_fig, _ax = plt.subplots(
    figsize=(7.5, 2.25), 
    constrained_layout=True
)

plotter.plot_envelope(
    _fig, 
    _ax, 
    palette_,
    _upper, 
    _lower, 
    f_deepest_ext_, 
    f_, 
    f_hat_, 
    e_biased_, 
    dx_, 
    dt_ = dt_ + 5, 
    interval = interval,
    n = 120,
    color_med = palette_.loc[2, "ibm"],
    label = r"$\hat{\mu}_{deep} (s)$",
    CR=r"$\mathcal{{R}}^{{adj}}_{{{}}}$",
    range_ =[0, 287],
    legend_1 = False,
    legend_2 = True,
)

_ax.legend(
    frameon = False,
    loc = (0.0125, .15),
    #loc = 'lower center',
    fontsize = 13,
    columnspacing = 0.25,
    handletextpad = 0.125,
    labelspacing = 0.125,
    ncol = 1,
)

plt.savefig(IMAGES / f"{distance}_adj_regions-{file_name}.pdf")

plt.show()

f_median_ext_, _upper, _lower = _fdu.ecdf_confidence_bands(
    M_int_, 
    alpha_,
)

f_wmedian_ext_, _wupper, _wlower = _fdu.weighted_ecdf_confidence_region(
    M_int_, 
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
    e_biased_, 
    dx_,
    dt_ = dt_ + 5, 
    interval = interval,
    n = 120,
    color_med = palette_.loc[3, "ibm"],
    CR=r"$\mathcal{{R}}^{{mar}}_{{{}}}$",
    label = r"$\hat{\mu}_{med} (s)$",
    range_ = [0, 287],
    legend_1 = False,
    legend_2 = True,
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
    M_int_, 
    dist = distance,
    max_iter = 100,
)

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
    constrained_layout=True
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
    e_biased_, 
    dx_,
    dt_ = dt_ + 5, 
    interval = interval,
    n = 120,
    color_med = palette_.loc[4, "ibm"],
    label = r"$\hat{\mu}_{focal} (s)$",
    CR=r"$\mathcal{{R}}^{{prj}}_{{{}}}$",
    range_ = [0, 287],
    legend_1 = False,
    legend_2 = True,
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
    figsize=(7.5, 2.25), 
    constrained_layout=True
)

plotter.plot_density_heatmap(
    _fig, 
    _ax, 
    palette_,
    M_, 
    f_wmedian_ext_, 
    f_deepest_ext_, 
    f_focal_ext_,
    f_, 
    f_hat_, 
    e_biased_, 
    dx_, 
    dt_ = dt_ + 5,
    interval = interval,
    range_ = [0, 287],
    colorbar_pos = [640, 75, 150, 5],
    colorbar = True,
    legend_1 = False,
    legend_2 = False,
)

_ax.legend(
    frameon = False,
    loc = (0.0125, .5),
    #loc = 'lower center',
    fontsize = 13,
    columnspacing = 0.25,
    handletextpad = 0.125,
    labelspacing = 0.125,
    ncol = 1,
)

plt.savefig(IMAGES / f"heatmap-{file_name}.pdf")

plt.show()


_fig, _ax = plt.subplots(
    figsize=(2, 4), 
    constrained_layout=True,
)

plotter.scenarios_frequency_dates(
    _fig, 
    _ax, 
    palette_,
    _fdu.idx_fed_, 
    _fdu.idx_x_,
    T_tr_[:, interval], 
    _fdu.date,
    scale = scale,
    colorbar = True,
)

plt.savefig(IMAGES / f"temporal_neighbors-{file_name}.pdf")

plt.show()


_fig, _ax = plt.subplots(figsize=(7.5, 6.25), constrained_layout=True)

plotter.selected_scenarios_heatmap(
    _fig, 
    _ax, 
    palette_,
    _fdu._haversine_dist, 
    _fdu.idx_fed_, 
    _fdu.idx_x_local_,
    _fdu.d_x_,
    _fdu.x_, 
    T_tr_[:, interval], 
    X_ts_,
    _fdu.date,
    colorbar = True,
)

plt.savefig(IMAGES / f"spatiotemporal_heatmap-{file_name}.pdf")

plt.show()

_fig, _ax = plt.subplot_mosaic(
    [["A", "B"], ["C", "."]],
    figsize=(6.5, 7.5),
    layout="constrained",
    gridspec_kw=dict(width_ratios=(1, 0.25), height_ratios=(1, 0.25)),
)

plotter.plot_filtered_scenarios(
    _fig, 
    _ax["A"], 
    palette_,
    _fdu.idx_fed_, 
    _fdu.idx_x_, 
    _fdu.idx_x_local_, 
    _fdu.d_x_,
    t_tr_, 
    t_ts_[day], 
    _fdu.r,
)

plotter.plot_distance_histogram(
    _fig, 
    _ax["B"], 
    palette_,
    _fdu.idx_fed_, 
    _fdu.idx_x_local_, 
    _fdu.d_x_, 
    _fdu.r,
)

plotter.plot_dates_histogram(
    _fig, 
    _ax["C"], 
    palette_,
    _fdu.idx_fed_, 
    _fdu.idx_x_local_, 
    _fdu.d_x_,
    t_tr_, 
    t_ts_[day], 
)

_ax["A"].legend(
    frameon = False,
    ncol = 1,
    columnspacing = 0.625,
    handletextpad = 0.25,
    loc = (0.0, 1.025),
    fontsize = 15,
)

plt.savefig(IMAGES / f"spatiotemporal_neighbors-{file_name}.pdf")

plt.show()


_fig, _ax = plt.subplots(
    1, 1, figsize=(10, 10), layout="constrained"
)

# Add an inset globe showing Texas
plotter.globe_inset(
    _fig, _ax, _TX, 
    x0 = 0.005, 
    y0 = 0.7, 
    width = 0.25, 
    height = 0.25,
)

_fig, _ax = plotter.hillshade(_fig, _ax, _TX)

plotter.plot_hillshade_frequency_map(
    _fig, _ax, palette_, _TX, _fdu, X_tr_, X_ts_, 
    _fdu.x_, 
    _fdu.idx_fed_,
    _fdu.idx_x_, 
    _fdu.r,
)

# Add compass (North arrow)
x = 0.1675
y = 0.2575
arrow_length = 0.06  # relative axes coords

_ax.annotate(
    "N",
    xy = (x, y),
    xytext = (x, y - arrow_length),
    arrowprops = dict(facecolor = "k", 
                      arrowstyle = "wedge,tail_width=1,shrink_factor=.5"),
    ha = "center",
    va = "center",
    fontsize = 14,
    xycoords = _ax.transAxes,
)

scalebar = ScaleBar(
    dx = 100,  # scaling factor
    units = "km",
    dimension = "si-length",  # could also be 'imperial-length.'
    location = "lower left",
    length_fraction = 0.09,  # fraction of the axis for bar length
    color = "black",
    box_color = "None",
    box_alpha = 1.0,
    border_pad = 9,
    font_properties = {"size": 14},
)

_ax.add_artist(scalebar)

_ax.legend(
    frameon = False, 
    bbox_to_anchor = (0.625, 0.9875), 
    ncol = 1, 
    fontsize = 14
)

plt.savefig(IMAGES / f"geographical_neighbors-{file_name}.pdf")

plt.show()


slices_ = [6, 12, 24, 48, 96]

_fig, _ax = plt.subplots(
    figsize=(9, 2.25), 
    constrained_layout=True
)

plotter.plot_heatmap_slices(
    _fig, 
    _ax, 
    palette_, 
    _fdu.M_, 
    f_, 
    f_hat_, 
    e_biased_, 
    dx_, 
    dt_ = dt_ + 5, 
    interval = interval,
    slices_ = slices_,
    range_ = [0, 287],
    n = 120,
    colorbar = True,
)

_ax.legend(
    loc = (0.0625, 1), 
    frameon = False, 
    fontsize = 12, 
    ncol = 4,
)

plt.savefig(IMAGES / f"heatmap_detail-{file_name}.pdf")

plt.show()

WKDs_ = _fdu.weighted_kernel_density_estimation(
    100*M_, 
    weights = _fdu.w_fed_[_fdu.idx_x_local_],
    algorithm = "auto", 
    kernel = "gaussian",
)

KDs_ = _fdu.kernel_density_estimation(
    100*M_, 
    algorithm = "auto", 
    kernel = "gaussian",
)

_fig, _ax = plt.subplots(
    1, len(slices_), 
    figsize = (9, 2.25), 
    sharey = True, 
    layout = "constrained"
)

plotter.plot_histogram_cuts(
    _fig, 
    _ax, 
    palette_, 
    WKDs_, 
    M_, 
    f_, 
    f_hat_, 
    e_biased_, 
    dx_, 
    dt_, 
    interval, 
    slices_)

_ax[0].legend(
    frameon = False, 
    fontsize = 12, 
    loc = (0, -.4),
)

plt.savefig(IMAGES / f"marginal_detail-{file_name}.pdf")

plt.show()


# Calculate confidence intervals from Directional Quantiles
depth_score_, depth_rank_ = _fdu.get_depth(_depth, E_tr_biased_[_fdu.idx_x_, :])

f_deepest_ext_, _upper, _lower = _fdu.functional_boxplot(
    E_tr_biased_[_fdu.idx_x_, :], 
    depth_score_,
)

_fig, _ax = plt.subplots(
    figsize=(7.5, 2.25), 
    constrained_layout=True,
)

plotter.plot_enhanced_functional_boxplot(
    _fig, 
    _ax, 
    palette_,
    _upper, 
    _lower, 
    f_deepest_ext_, 
    f_, 
    f_hat_, 
    e_biased_, 
    dx_,
    dt_ = dt_ + 5, 
    interval = interval,
    CR=r"$\mathcal{{R}}^{{bxp}}_{{\alpha={}}}$",
    n = 120,
    range_ = [0, 287],
    legend_1 = True,
    legend_2 = False,
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

# Calculate confidence intervals from Directional Quantiles
depth_score_, depth_rank_ = _fdu.get_depth(_depth, F_tr_[_fdu.idx_x_, :])

f_deepest_ext_, _upper, _lower = _fdu.functional_boxplot(
    F_tr_[_fdu.idx_x_, :], 
    depth_score_,
)

_fig, _ax = plt.subplots(
    figsize=(7.5, 2.25), 
    constrained_layout=True,
)

plotter.plot_enhanced_functional_boxplot(
    _fig, _ax, palette_,
    _upper, _lower, f_deepest_ext_, f_, f_hat_, e_biased_, dx_,
    dt_ = dt_ + 5, 
    interval = interval,
    CR=r"$\mathcal{{R}}^{{bxp}}_{{\alpha={}}}$",
    n = 120,
    range_ = [0, 287],
    legend_1 = False,
    legend_2 = False,
)

_ax.legend(
    frameon = False,
    loc = (0.25, .625),
    #loc = 'lower center',
    fontsize = 13,
    columnspacing = 0.25,
    handletextpad = 0.125,
    labelspacing = 0.125,
    ncol = 3,
)

plt.savefig(IMAGES / f"{distance}_box_realized-{file_name}.pdf")

plt.show()

# Calculate confidence intervals from Directional Quantiles
depth_score_, depth_rank_ = _fdu.get_depth(_depth, M_)

f_deepest_ext_, _upper, _lower = _fdu.functional_boxplot(
    M_, 
    depth_score_,
)

_fig, _ax = plt.subplots(
    figsize=(7.5, 2.25), 
    constrained_layout=True,
)

plotter.plot_enhanced_functional_boxplot(
    _fig, 
    _ax, 
    palette_,
    _upper, 
    _lower, 
    f_deepest_ext_, 
    f_, 
    f_hat_, 
    e_biased_, 
    dx_,
    dt_ = dt_ + 5, 
    interval = interval,
    CR=r"$\mathcal{{R}}^{{bxp}}_{{\alpha={}}}$",
    n = 120,
    range_ = [0, 287],
    legend_1 = False,
    legend_2 = True,
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

F_curves_ = []
M_curves_ = []
D_curves_ = []
for interval in range(6, 282, 6):

    M_ = _fdu.predict(
        F_ts_[day, :interval, asset],
        E_ts_lin_[day, :, asset],
        X_ts_[asset, :],
        t_ts_[day],
        forget_rate_f = get_hyper(hyper_, "forget_rate_f", interval),
        forget_rate_e = get_hyper(hyper_, "forget_rate_e", interval),
        lookahead_rate = get_hyper(hyper_, "lookahead_rate", interval),
        length_scale_f = get_hyper(hyper_, "length_scale_f", interval),
        length_scale_e = get_hyper(hyper_, "length_scale_e", interval),
        length_scale_d = get_hyper(hyper_, "length_scale_d", interval),
        sigma = get_hyper(hyper_, "sigma", interval),
        nu = get_hyper(hyper_, "nu", interval),
        kappa_0 = get_hyper(hyper_, "kappa_0", interval),
        kappa = get_hyper(hyper_, "kappa", interval),
        p_fusion = get_hyper(hyper_, "p_fusion", interval),
    )

    # Samples in each confidence band    
    M_int_, M_int_ds_ = _fdu.functional_downsampling(
        subsample=12, 
        n_basis=int(1.333*(T - interval)/12)
    )

    f_wmedian_ext_, _wupper, _wlower = _fdu.weighted_ecdf_confidence_bands(
        M_int_, 
        _fdu.w_prime_,
        alpha_
    )
    
    k_ = get_band_fraction(
        envelope_, 
        alpha_, 
        interval, 
        dist = 'MBD', 
        score = 'FCS',
    )
    
    f_deepest_ext_, _upper, _lower = _fdu.depth_confidence_bands(
        _depth, 
        M_int_, 
        alpha_, 
        k_
    )

    M_curves_.append(f_wmedian_ext_[1:])
    F_curves_.append(_fdu.f_focal_)
    D_curves_.append(f_deepest_ext_[1:])

print(len(F_curves_), len(M_curves_), len(D_curves_))

_fig, _ax = plt.subplots(
    figsize=(7.5, 2.25), 
    constrained_layout=True,
)

plotter.plot_dynamic_update(
    _fig, _ax, palette_, 
    F_curves_, f_, f_hat_, e_biased_, dx_, dt_ + 5,
    label = r"$\hat{\mu}_{focal}$",
    range_ = [0, 287],
    legend = True,
    colorbar = False,
)

_ax.legend(
    frameon = False,
    loc = (0.1875, .775),
    #loc = 'lower center',
    fontsize = 13,
    columnspacing = 0.25,
    handletextpad = 0.125,
    labelspacing = 0.125,
    ncol = 3,
)

plt.savefig(IMAGES / f"focal_update-{file_name}.pdf")

plt.show()