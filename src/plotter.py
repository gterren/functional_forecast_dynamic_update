import ee, geemap, datetime, calendar

import numpy as np
import geopandas as gpd

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib import cm
from matplotlib.transforms import ScaledTranslation
from matplotlib.ticker import FuncFormatter

from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams["legend.handlelength"] = 1
plt.rcParams["legend.handleheight"] = 1.125
plt.rcParams["font.family"] = "Avenir"


def plot_histogram_cuts(
    _fig, 
    _ax, 
    palette_, 
    KDs_, 
    M_, 
    f_, 
    f_hat_, 
    e_, 
    dx_, 
    dt_, 
    interval, 
    slices_=[]
):

    tau_ = dt_[:interval]
    s_ = dt_[interval:]

    x_ = np.linspace(0, 100, 1000)[:, np.newaxis]

    for slice, i in zip(slices_, range(len(slices_))):
        z_ = np.exp(KDs_[slice].score_samples(x_))

        _ax[i].axvline(
            100 * f_hat_[slice],
            color=palette_.loc[0, "ibm"],
            lw=2,
            ls="--",
            # label="CF (actual)",
            zorder=10,
        )

        _ax[i].axvline(
            100 * e_[interval + slice],
            color="k",
            lw=2,
            # label="CF (forecast)",
            zorder=10,
        )

        _ax[i].hist(
            100 * M_[:, slice],
            bins=20,
            range=(0, 100),
            density=True,
            color=palette_.loc[3, "ibm"],
            clip_on=True,
            zorder=8,
            edgecolor="w",
            linewidth=0.5,
        )

        _ax[i].plot(
            x_, z_, 
            label="KDE (update)", 
            color=palette_.loc[1, "ibm"], 
            lw=2, 
            zorder=9
        )

        _ax[i].set_title(dx_[interval:][slice], size=12)
        _ax[i].set_xlim(0, 100)
        _ax[i].set_ylim(0,)

        _ax[i].tick_params(axis="both", labelsize=12)

    _ax[0].set_ylabel("EDF", size=12)
    _ax[2].set_xlabel("Capacity Factor (%)", size=12)



def plot_heatmap_slices(
    _fig, 
    _ax, 
    palette_, 
    M_, 
    f_, 
    f_hat_, 
    e_, 
    dx_, 
    dt_, 
    interval, 
    slices_=[], 
    range_=[],
    colorbar=True
):

    tau_ = dt_[:interval]
    s_ = dt_[interval:]

    Z_ = []
    for i in range(M_.shape[1]):
        a_, b_ = np.histogram(
            100.* M_[:, i], 
            bins=25, 
            range=(0, 100), 
            density=True
        )
        
        Z_.append(a_)

    Z_ = np.stack(Z_).T
    X_, Y_ = np.meshgrid(dt_[interval:], (b_[1:] + b_[:-1])/2.0)

    _cmap = sns.color_palette("rocket_r", as_cmap=True)
    _ax.pcolormesh(X_, Y_, Z_, cmap=_cmap)

    _ax.plot(
        tau_,
        100 * f_,
        c=palette_.loc[0, "ibm"],
        clip_on=False,
        lw=2.0,
        label="CF (actual)",
        zorder=9,
    )

    _ax.plot(
        s_,
        100 * f_hat_,
        c=palette_.loc[0, "ibm"],
        clip_on=False,
        lw=2.0,
        ls="--",
        zorder=9,
    )

    _ax.plot(dt_, 
             100 * e_, 
             c="k", 
             lw=2.0, 
             label="CF (forecast)", 
             clip_on=False, 
             zorder=8
    )

    _ax.axvline(
        dt_[interval], 
        color="k", 
        lw=0.75, 
        ls="--", 
        label="Detail"
    )

    for slice in slices_:
        _ax.axvline(
            dt_[interval + slice], 
            color="k", 
            lw=0.75, 
            ls="--"
        )

    _ax.fill_between(
        tau_,
        100 * np.ones(tau_.shape),
        100 * np.zeros(tau_.shape),
        color="lightgray",
        alpha=0.5,
    )

    _ax.axvline(
        dt_[interval], 
        color="k", 
        label="Event (update)", 
        linewidth=0.75, 
        zorder=11
    )

    _ax.set_xticks(dt_[24::24], dx_[24::24], rotation=0)
    _ax.set_ylabel("Capacity Factor (%)", size=14)

    _ax.tick_params(axis="both", labelsize=12)

    _ax.set_ylim(0, 101)
    _ax.set_xlim(dt_[0], dt_[-1])

    if colorbar:
        cbar = _fig.colorbar(
            cm.ScalarMappable(cmap=_cmap),
            cax=_ax.inset_axes([40, 77.5, 150, 5], transform=_ax.transData),
            orientation="horizontal",
            extend="max",
        )

        cbar.set_ticks([0, 1], labels=["low", "high"], fontsize=12)
        cbar.ax.tick_params(length=0)

        cbar.ax.set_title("EDF", rotation=0)


def plot_updates(
    _fig, 
    _ax, 
    palette_, 
    M_, 
    f_, 
    f_hat_, 
    e_, 
    w_, 
    idx_, 
    dx_, 
    dt_, 
    interval, 
    range_=[]
):

    tau_ = dt_[:interval]
    s_ = dt_[interval:]
    z_ = (w_ - w_[idx_].min()) / (w_[idx_].max() - w_[idx_].min())

    w_selected_ = w_[idx_]

    idx_w_ = np.argsort(w_selected_)
    idx_ = idx_[idx_w_]

    _cmap = sns.color_palette("rocket", as_cmap=True)
    _norm = plt.Normalize(0.0, 1)

    _ax.plot([], [], lw=0.75, label=r"$\hat{\mu}_i(s)$", c="k")

    _ax.plot(
        tau_,
        100.0 * f_,
        c=palette_.loc[0, "ibm"],
        zorder=10,
        # label="CF (ac)",
        clip_on=False,
        lw=2,
    )

    _ax.plot(
        s_,
        100.0 * f_hat_,
        c=palette_.loc[0, "ibm"],
        zorder=10,
        lw=2,
        clip_on=False,
        ls="--",
    )

    _ax.plot(
        dt_,
        100.0 * e_,
        lw=2,
        zorder=9,
        # label="CF (fc)",
        clip_on=False,
        c="k",
    )

    for i, j in zip(idx_, range(idx_.shape[0])):
        _ax.plot(
            dt_[interval:],
            100 * M_[j, :],
            # c=_cmap(_norm(z_[i])),
            c=_cmap(_norm(j / idx_.shape[0])),
            lw=0.5,
            zorder=8,
        )

    _ax.fill_between(
        tau_,
        100 * np.ones(tau_.shape),
        100 * np.zeros(tau_.shape),
        color="lightgray",
        alpha=0.5,
        zorder=1,
    )

    _ax.axvline(
        dt_[interval],
        color="k",
        lw=1.0,
        # label="Event (update)",
        zorder=11,
    )

    _ax.set_xticks(dt_[24::24], dx_[24::24], rotation=0)
    # ax_[2].set_yticks(size = 12)
    _ax.set_ylim(0.0, 100.0)
    _ax.set_xlim(dt_[0], dt_[-1])
    _ax.set_ylabel("Capacity Factor (%)", size=14)

    _ax.tick_params(axis="both", labelsize=12)

    cbar = _fig.colorbar(
        cm.ScalarMappable(_norm, _cmap),
        cax=_ax.inset_axes([45, 80, 150, 5], transform=_ax.transData),
        orientation="horizontal",
    )

    cbar.set_ticks([0, 1], labels=[1, M_.shape[0]], size=12)

    # cbar.ax.tick_params(length=0)

    cbar.ax.set_title("Similarity Rank", rotation=0, size=12)

def plot_neighbors(
    _fig,
    _ax, 
    palette_,
    F_tr_, 
    w_, 
    idx_, 
    f_, 
    f_hat_, 
    e_, 
    dx_, 
    dt_, 
    interval, 
    range_=[]
):

    tau_ = dt_[:interval]
    s_ = dt_[interval:]

    z_ = (w_ - w_[idx_].min()) / (w_[idx_].max() - w_[idx_].min())
    idx_ = idx_[np.argsort(w_[idx_])]

    _cmap = sns.color_palette("rocket", as_cmap=True)
    _norm = plt.Normalize(0.0, 1.0)

    _ax.plot(
        [], [], 
        lw=0.75, 
        label=r"$\mathcal{F}_{\star} \left([\tau, s] \right)$", 
        c="k"
    )

    _ax.plot(
        tau_,
        100 * f_,
        c=palette_.loc[0, "ibm"],
        zorder=10,
        # label="CF (ac)",
        lw=2,
        clip_on=False,
    )

    _ax.plot(
        s_,
        100 * f_hat_,
        c=palette_.loc[0, "ibm"],
        zorder=10,
        lw=2,
        ls="--",
        clip_on=False,
    )

    _ax.plot(
        dt_,
        100 * e_,
        lw=2,
        zorder=9,
        # label="CF (fc)",
        c="k",
        clip_on=False,
    )

    for i, j in zip(idx_, range(idx_.shape[0])):
        _ax.plot(
            dt_,
            100 * F_tr_[i, :],
            c=_cmap(_norm(j / idx_.shape[0])),
            # c=_cmap(_norm(z_[i])),
            lw=0.5,
            zorder=8,
        )

    _ax.fill_between(
        tau_,
        100 * np.ones(tau_.shape),
        100 * np.zeros(tau_.shape),
        color="lightgray",
        alpha=0.5,
        zorder=1,
    )

    _ax.axvline(dt_[interval], color="k", lw=1.0, zorder=11)

    _ax.set_xticks(dt_[24::24], dx_[24::24], rotation=0)
    _ax.set_ylim(0.0, 101)
    _ax.set_xlim(dt_[0], dt_[-1])
    _ax.set_ylabel("Capacity Factor (%)", size=14)

    _ax.tick_params(axis="both", labelsize=12)


def plot_forecasts(
    _fig,
    _ax,
    palette_,
    E_tr_,
    w_,
    idx_,
    f_,
    f_hat_,
    e_,
    dx_,
    dt_,
    interval,
    range_=[],
    legend=False,
):

    tau_ = dt_[:interval]
    s_ = dt_[interval:]

    z_ = (w_ - w_[idx_].min()) / (w_[idx_].max() - w_[idx_].min())
    idx_ = idx_[np.argsort(w_[idx_])]

    _cmap = sns.color_palette("rocket", as_cmap=True)
    _norm = plt.Normalize(0, 1)

    _ax.plot(
        [], [], 
        lw=0.75, 
        label=r"$E_{\star} \left([\tau, s]\right)$", 
        c="k"
    )

    _ax.plot(
        tau_,
        100.0 * f_,
        c=palette_.loc[0, "ibm"],
        clip_on=False,
        zorder=10,
        # label="CF (ac)",
        lw=2,
    )

    _ax.plot(
        s_,
        100.0 * f_hat_,
        c=palette_.loc[0, "ibm"],
        clip_on=False,
        zorder=10,
        lw=2,
        ls="--",
    )

    _ax.plot(
        dt_,
        100.0 * e_,
        clip_on=False,
        lw=1.5,
        zorder=9,
        # label="CF (fc)",
        c="k",
    )

    for i, j in zip(idx_, range(idx_.shape[0])):
        _ax.plot(
            dt_,
            100 * E_tr_[i, :],
            # c=_cmap(_norm(z_[i])),
            c=_cmap(_norm(j / idx_.shape[0])),
            lw=0.5,
            zorder=8,
        )

    _ax.fill_between(
        tau_,
        100 * np.ones(tau_.shape),
        100 * np.zeros(tau_.shape),
        color="lightgray",
        alpha=0.5,
        zorder=1,
    )

    _ax.axvline(dt_[interval], color="k", lw=1.0, zorder=11)

    _ax.set_xticks(dt_[24::24], dx_[24::24], rotation=0)

    _ax.set_ylim(0.0, 101)
    _ax.set_xlim(dt_[0], dt_[-1])
    _ax.set_ylabel("Capacity Factor (%)", size=14)

    _ax.tick_params(axis="both", labelsize=12)

def plot_envelop(
    _fig, _ax, palette_, _upper, _lower, m_, f_, f_hat_, 
    e_, dt_, dx_, interval, color, label,
    CR = 'CR',
    range_ = [],
    n = 120,
    legend_1 = True, 
    legend_2 = True
):

    dt   = dt_[1] - dt_[0]
    tau_ = dt_[:interval]
    s_   = dt_[interval:]

    _ax.plot(
        dt_, 100*e_, 
        c = "k",
        lw = 2, 
        clip_on = True,
        label = "CF (forecast)" if legend_1 else None, 
        zorder = 8
    )

    _ax.plot(
        tau_, 100*f_, 
        c = palette_.loc[0, "ibm"], 
        clip_on = True,
        label = "CF (actual)" if legend_1 else None, 
        lw = 2, 
        zorder = 9
    )
    
    _ax.plot(
        s_, 100*f_hat_, 
        clip_on = True,
        c = palette_.loc[0, "ibm"], 
        ls = "--", 
        lw = 2, 
        zorder = 9
    )

    _ax.plot(
        s_, 100*m_[1:], 
        c = color, 
        label = label if legend_2 else None,
        lw = 2, 
        zorder = 10
    )
    
    colors_ = ["lightgray", "darkgray", "gray", "dimgray"]
    for color, key, i in zip(
        colors_, _upper.keys(), range(len(_upper.keys()))
    ):

        u_ = _upper[key]
        l_ = _lower[key]
        cr = int((1. - float(key))*100)

        _ax.fill_between(
            dt_[-u_.shape[0]:], 100*u_, 100*l_,
            color  = color,
            label  = f"{cr}% {CR}" if legend_2 else None,
            zorder = i + 1
        )

    _ax.fill_between(
        tau_, 100 * np.ones(tau_.shape), 100 * np.zeros(tau_.shape),
        color = "lightgray",
        alpha = 0.5
    )

    _ax.axvline(
        dt_[interval - 1], 
        color = "k", 
        linewidth = 0.75, 
        #label = "Event (update)",
        zorder = 11
    )
    
    idx_ = (dt_ % n) == 0
    idx_[1] = False
    idx_[-1] = False
    _ax.set_xticks(dt_[idx_], dx_[idx_], rotation=0)
    
    _ax.tick_params(axis = "both", 
                    labelsize = 12)
    
    _ax.set_ylim(0, 100)
    _ax.set_xlim(dt_[0], dt_[-1])
    _ax.set_ylabel("Capacity Factor (%)", size = 14)

def plot_forecast_parameters(
    _fig, _ax, palette_, phi_, psi_, eta_, f_, 
    f_hat_, e_, dx_, dt_, interval,
    range_ = [],
    labels_1 = False,
    labels_2 = False
):

    tau_ = dt_[:interval]
    s_   = dt_[interval:]
    dt   = dt_[1] - dt_[0]

    _ax.plot(
        tau_ + dt, 100. * f_, 
        c=palette_.loc[0, "ibm"], 
        label="CF (actual)" if labels_1 else None, 
        lw=1.5, 
        zorder=5
    )

    _ax.plot(s_ + dt, 100. * f_hat_, 
             c=palette_.loc[0, "ibm"], 
             lw=1.5, 
             ls="--", 
             zorder=5)

    _ax.plot(
        dt_ + dt, 100. * e_, 
        lw=1.5, 
        label="CF (forecast)" if labels_1 else None, 
        c="k", 
        zorder=4
    )

    _ax.axvline(
        dt_[interval - 1] + dt, 
        color="k", 
        lw=0.75,
        label="Event (update)" if labels_1 else None, 
        zorder=6
    )

    _ax.plot(
        tau_ + dt, 100. * phi_,
        c=palette_.loc[3, "ibm"],
        lw=3,
        label=r"$\phi_{\varepsilon_f} (\tau)$" if labels_2 else None,
        zorder=2
    )
    
    _ax.plot(
        tau_ + dt, 100. * psi_[:interval],
        c=palette_.loc[1, "ibm"],
        lw=3,
        label=r"$\phi_{\varepsilon_e} (\tau)$" if labels_2 else None,
        zorder=2
    )

    _ax.plot(
        s_ + dt, 100. * psi_[interval:],
        c=palette_.loc[2, "ibm"],
        lw=3,
        label=r"$\psi_{\eta} (s)$" if labels_2 else None,
        zorder=2
    )

    _ax.plot(
        s_ + dt, 100. * eta_,
        c=palette_.loc[4, "ibm"],
        lw=3,
        label=r"$\sigma_{\alpha} (s)$" if labels_2 else None,
        zorder=2
    )

    _ax.fill_between(
        tau_ + dt, 100. * np.ones(tau_.shape), 100. * np.zeros(tau_.shape),
        color="lightgray",
        alpha=0.5,
        zorder=1
    )

    _ax.set_xticks(dt_[24::24], dx_[24::24], rotation=0)
    _ax.set_ylabel("Capacity Factor (%)", size=14)
    _ax.tick_params(axis="both", labelsize=12)
    _ax.set_ylim(0, 100)
    _ax.set_xlim(dt_[range_[0]], dt_[range_[1]])


def plot_depth(
    _fig, _ax, palette_, M_, f_, f_hat_, 
    e_, w_, dx_, dt_, interval,
    range_ = [],
    colorbar = True
):

    tau_ = dt_[:interval]
    s_   = dt_[interval:]
    dt   = dt_[1] - dt_[0]

    idx_  = np.argsort(w_)
    _cmap = sns.color_palette("rocket", as_cmap=True)
    _norm = plt.Normalize(0., 1)

    _ax.plot(
        [], [], 
        lw = 0.75,
        label = r"$\hat{\mu}_i(s)$", 
        c = "k"
    )

    for i in range(idx_.shape[0]):
        _ax.plot(s_ + dt, 100. * M_[idx_[i], :],
                 c = _cmap(_norm(i/idx_.shape[0])), 
                 lw = 0.75, 
                 zorder = 0, 
                 clip_on = False)

    _ax.plot(dt_[:f_.shape[0]] + dt, 100. * f_,
             c = palette_.loc[0, "ibm"],
             #label = "CF (ac)",
             zorder = 9,
             lw = 2, 
             clip_on = False)

    _ax.plot(
        dt_[-f_hat_.shape[0]:] + dt, 100. * f_hat_,
        c = palette_.loc[0, "ibm"],
        zorder = 9,
        lw = 2,
        ls = "--", 
        clip_on = False
    )

    _ax.plot(
        dt_ + dt, 100. * e_, 
        lw = 2, 
        #label = "CF (fc)", 
        zorder = 8,
        c = "k", 
        clip_on = False
    )

    _ax.axvline(
        dt_[f_.shape[0] - 1] + dt, 
        color = "k", 
        lw = 1, 
        #label  = "Event (update)", 
        zorder = 10
    )
        
    _ax.fill_between(
        tau_ + dt, 100 * np.ones(tau_.shape), 100 * np.zeros(tau_.shape),
        color = "lightgray",
        alpha = 0.5,
        zorder = 1
    )
    
    _ax.set_xticks(dt_[24::24], dx_[24::24], rotation=0)
    # ax_[2].set_yticks(size = 12)
    _ax.set_ylim(0., 101)
    _ax.set_xlim(dt_[0], dt_[-1])
    _ax.set_ylabel("Capacity Factor (%)", size=14)
    
    _ax.tick_params(axis = "both", 
                    labelsize = 12)
    
    if colorbar:
        cbar = _fig.colorbar(
            cm.ScalarMappable(_norm, 
                              sns.color_palette("rocket_r", as_cmap = True)),
            cax= _ax.inset_axes([45, 80, 150, 5], transform = _ax.transData),
            orientation = "horizontal"
        )
    
        cbar.set_ticks([0, 1], 
                       labels = [1, M_.shape[0]], 
                       size = 12)
            
        cbar.ax.set_title("Depth Rank", 
                          rotation = 0, 
                          size = 12)


def plot_enhanced_functional_boxplot(
    _fig, _ax, palette_, _upper, _lower, m_, 
    f_, f_hat_, e_, dt_, dx_, interval,
    range_ = [],
    legend_1 = True, 
    legend_2 = True
):

    tau_ = dt_[:interval]
    s_   = dt_[interval:]
    dt   = dt_[1] - dt_[0]

    _ax.plot(
        dt_ + dt, 100 * e_, 
        c = "k", 
        lw = 2, 
        zorder = 8,
        #label = "CF (forecast)" if legend_1 else None
        clip_on = False
    )

    _ax.plot(
        tau_ + dt, 100 * f_, 
        c = palette_.loc[0, "ibm"], 
        #label = "CF (actual)" if legend_1 else None, 
        zorder = 9,
        lw = 2, 
        clip_on = False
    )

    _ax.plot(
        s_ + dt, 100 * f_hat_, 
        c  = palette_.loc[0, "ibm"], 
        lw = 2, 
        ls = "--", 
        zorder = 9,
        clip_on = False
    )
    #m_p_ = np.concatenate([f_[-1] * np.ones((1,)), m_], axis=0)

    _ax.plot(
        dt_[-m_.shape[0]:] + dt, 100 * m_, 
        c = palette_.loc[2, "ibm"], 
        ls = '-',
        zorder = 4,
        label = r"$\bar{\mu} (s)$" if legend_1 else None,
        lw = 2
    )
    
    u_ = _upper['max']
    l_ = _lower['min']

    _ax.plot(
        dt_[-u_.shape[0]:] + dt, 100 * u_, 
        c = 'k', 
        lw = .75, 
        zorder = 7
    )
    
    _ax.plot(
        dt_[-l_.shape[0]:] + dt, 100 * l_, 
        c = 'k', 
        lw = .75, 
        zorder = 7, 
        label = "min-max" if legend_2 else None
    )

    _upper.pop('max')
    _lower.pop('min')

    colors_ = ["lightgray", "darkgray", "gray"]

    for color, key, i in zip(
        colors_, _upper.keys(), range(len(_upper.keys()))
    ):
        u_ = _upper[key]
        l_ = _lower[key]
        cr = int((1. - float(key)) * 100)
        
        _ax.fill_between(
            dt_[-u_.shape[0]:] + dt, 100*u_, 100*l_,
            color = color,
            label = f"{cr}% CR" if legend_2 else None,
            zorder = i + 1
        )

    _ax.axvline(
        dt_[interval - 1] + dt, 
        color = "k", 
        linewidth = 0.75, 
        #label = "Event (update)",
        zorder = 10
    )

    _ax.fill_between(
        tau_ + dt, 100*np.ones(tau_.shape), 100*np.zeros(tau_.shape),
        color = "lightgray",
         alpha = 0.5
    )
    
    _ax.tick_params(axis = "both", labelsize = 12)
    
    _ax.set_xticks(dt_[24::24], dx_[24::24], rotation=0)
    # ax_[2].set_yticks(size = 12)
    _ax.set_ylim(0., 101.)
    _ax.set_xlim(dt_[0], dt_[-1])
    _ax.set_ylabel("Capacity Factor (%)", size=14)

def plot_density_heatmap(
    _fig, _ax, palette_, M_, f_median_, f_deepest_, 
    f_focal_, f_, f_hat_, e_, dx_, dt_, interval, 
    range_ = [0, 287],
    colorbar = True,
    legend_1 = True,
    legend_2 = True
):

    tau_ = dt_[:interval]
    s_   = dt_[interval:]
    dt   = dt_[1] - dt_[0]

    Z_ = []
    for i in range(M_.shape[1]):
        a_, b_ = np.histogram(
            100 * M_[:, i], 
            bins = 25, 
            range = (0, 100), 
            density = True
        )
        
        Z_.append(a_)

    Z_ = np.stack(Z_).T
    X_, Y_ = np.meshgrid(dt_[interval:], (b_[1:] + b_[:-1]) / 2.0)

    _cmap = sns.color_palette("rocket_r", as_cmap = True)
    _ax.pcolormesh(X_ + dt, Y_, Z_, cmap = _cmap, alpha = 1.)

    _ax.plot(
        s_ + dt, 100 * f_median_[1:], 
        c = palette_.loc[3, "ibm"], 
        label = r"$\bar{\mu} (s)$" if legend_2 else None,
        lw = 2, 
        zorder = 10
    )

    _ax.plot(
        s_ + dt, 100 * f_median_[1:], 
        c = 'k', 
        lw = .25,
        alpha = 0.5,
        zorder = 11
    )
    
    _ax.plot(
        s_ + dt, 100 * f_deepest_[1:], 
        c = palette_.loc[2, "ibm"], 
        label = r"$\bar{f} (s)$" if legend_2 else None,
        lw = 2, 
        zorder = 10
    )
    
    _ax.plot(
        s_ + dt, 100 * f_deepest_[1:], 
        c = 'k', 
        lw = .25, 
        alpha = 0.5,
        zorder = 11
    )
    
    _ax.plot(
        s_ + dt, 100 * f_focal_[1:], 
        c = palette_.loc[4, "ibm"], 
        label = r"$\tilde{f} (s)$" if legend_2 else None,
        lw = 2, 
        zorder = 10
    )

    _ax.plot(
        s_ + dt, 100 * f_focal_[1:], 
        c = 'k', 
        lw = .25, 
        alpha = 0.5,
        zorder = 11
    )
    
    _ax.plot(
        tau_ + dt, 100 * f_,
        c = palette_.loc[0, "ibm"],
        zorder = 9,
        lw = 2,
        label = "CF (actual)" if legend_1 else None, 
        clip_on = False
    )

    _ax.plot(
        s_ + dt, 100 * f_hat_, 
        c = palette_.loc[0, "ibm"], 
        zorder = 9,
        lw = 2, 
        ls = "--", 
        clip_on = False
    )

    _ax.plot(
        dt_ + dt, 100 * e_, 
        c = "k", 
        lw = 2, 
        label = "CF (forecast)" if legend_1 else None, 
        zorder = 8,
        clip_on = False
    )

    _ax.axvline(
        dt_[interval - 1] + dt, 
        color = "k", 
        linewidth = 0.75, 
        label = "Event (update)" if legend_1 else None,
    zorder = 10
    )

    _ax.fill_between(
        tau_ + dt, 100 * np.ones(tau_.shape), 100 * np.zeros(tau_.shape),
        color = "lightgray",
        alpha = 0.5
    )
    
    _ax.set_xticks(dt_[24::24], dx_[24::24], rotation = 0)
    _ax.set_ylabel("Capacity Factor (%)", size = 14)

    _ax.tick_params(axis = "both", labelsize = 12)

    _ax.set_ylim(0, 100)
    _ax.set_xlim(dt_[0], dt_[-1])

    if colorbar:
        cbar = _fig.colorbar(
            cm.ScalarMappable(cmap=_cmap),
            cax = _ax.inset_axes([50, 80, 150, 5], transform=_ax.transData),
            orientation = "horizontal",
            extend = "max")
    
        cbar.set_ticks([0, 1], labels = ["low", "high"], fontsize = 12)
        
        cbar.ax.tick_params(length=0)
    
        cbar.ax.set_title("EDF", rotation=0)
    

def plot_frequency_map(
    _fig, 
    _ax, 
    palette_, 
    _TX, 
    _fdu, 
    x_tr_, 
    x_ts_, 
    x_, 
    idx_neighbors_, 
    idx_temporal_, 
    idx_spatial_, 
    sigma
):

    x_tr_p_, z_tr_p_ = np.unique(
        x_tr_[idx_neighbors_, :], 
        return_counts=True, 
        axis=0
    )

    _cmap = sns.color_palette("rocket_r", as_cmap=True)
    _norm = plt.Normalize(1, z_tr_p_.max())

    _TX.plot(
        ax=_ax, 
        facecolor="lightgray", 
        edgecolor="white", 
        zorder=0
    )

    _ax.plot(
        x_ts_[:, 0],
        x_ts_[:, 1],
        c="gray",
        alpha=1,
        ms=6,
        marker="o",
        mec="w",
        ls="none",
        mew=1.0,
        zorder=2,
        clip_on=False,
        label="No neighboring assets",
    )

    _ax.plot(
        x_[0],
        x_[1],
        c=_cmap(_norm(z_tr_p_.max() / 2.0)),
        alpha=0.75,
        ms=6,
        ls="none",
        marker="o",
        mec="w",
        mew=1.0,
        zorder=0,
        clip_on=False,
        label="Neighboring assets",
    )

    _ax.plot(
        x_[0],
        x_[1],
        c=_cmap(_norm(z_tr_p_.max() / 4.0)),
        alpha=0.75,
        ms=6,
        ls="none",
        marker="o",
        mec="k",
        mew=1.0,
        zorder=0,
        clip_on=False,
        label="Selected neighboring assets",
    )

    for i in np.arange(x_tr_p_.shape[0], dtype=int)[np.argsort(z_tr_p_)]:
        if (x_tr_p_[i, 0] != x_[0]) | (x_tr_p_[i, 1] != x_[1]):
            _ax.plot(
                x_tr_p_[i, 0],
                x_tr_p_[i, 1],
                c=_cmap(_norm(z_tr_p_[i])),
                ms=6,
                ls="none",
                marker="o",
                mec="w",
                mew=0.75,
                zorder=6,
                clip_on=False,
            )
        else:
            _ax.plot(
                x_tr_p_[i, 0],
                x_tr_p_[i, 1],
                c=_cmap(_norm(z_tr_p_[i])),
                ms=14,
                ls="none",
                marker="d",
                mec="w",
                mew=1.0,
                zorder=4,
                label="Asset (update)",
                clip_on=False,
            )

    x_tr_p_ = np.unique(x_tr_[idx_spatial_, :], axis=0)
    for i in range(x_tr_p_.shape[0]):
        if (x_tr_p_[i, 0] != x_[0]) | (x_tr_p_[i, 1] != x_[1]):
            _ax.plot(
                x_tr_p_[i, 0],
                x_tr_p_[i, 1],
                ms=6,
                ls="none",
                marker="o",
                c="none",
                mec="k",
                mew=0.75,
                zorder=7,
                clip_on=False,
            )
        else:
            _ax.plot(
                x_tr_p_[i, 0], x_tr_p_[i, 1],
                ms=14,
                ls="none",
                marker="d",
                c="none",
                mec="lime",
                mew=0.75,
                zorder=5,
                # label = 'Asset (update)',
                clip_on=False,
            )

    minx, miny, maxx, maxy = _TX.total_bounds
    X_, Y_ = np.meshgrid(np.linspace(minx, maxx, 1000), np.linspace(miny, maxy, 1000))

    XX_ = np.concatenate([X_.flatten()[:, np.newaxis], Y_.flatten()[:, np.newaxis]], axis=1)
    Z_ = _fdu._haversine_dist(XX_, x_).reshape(X_.shape)

    contours = _ax.contour(
        X_, Y_, Z_,
        levels=[sigma],
        colors="k",
        linewidths=0.5,
        linestyles="dashed",
        zorder=11,
    )

    cbar = _fig.colorbar(
        cm.ScalarMappable(cmap=_cmap),
        cax=_ax.inset_axes([-97.75, 35.5, 2.0, 0.25],transform=_ax.transData),
        orientation="horizontal",
    )

    cbar.set_ticks([0, 1], labels=[1, int(z_tr_p_.max())], fontsize=14)

    cbar.ax.tick_params(length=0)

    cbar.ax.set_title("Neighbors", rotation=0, fontsize=12)

    _ax.set_axis_off()


def plot_dynamic_update(
    _fig,
    _ax,
    palette_,
    F_curves_,
    f_,
    f_hat_,
    e_,
    dx_,
    dt_,
    legend=False,
    colorbar=True,
    label=r"$\bar{f} (s)$",
    range_=[],
):

    f_p_ = np.concatenate([f_, f_hat_], axis=0)

    _cmap = sns.color_palette("rocket_r", as_cmap=True)
    _norm = plt.Normalize(0.0, len(F_curves_))

    _ax.plot(
        [], [], 
        lw=0.75, 
        label=label if legend else None, 
        c="k"
    )

    _ax.plot(
        dt_,
        100 * f_p_,
        c=palette_.loc[0, "ibm"],
        zorder=10,
        label="CF (actual)" if legend else None,
        lw=2,
        clip_on=False,
    )

    _ax.plot(
        dt_,
        100 * e_,
        lw=2,
        zorder=9,
        label="CF (forecast)" if legend else None,
        c="k",
        clip_on=False,
    )

    for i in range(len(F_curves_)):
        focal_curve_ = F_curves_[i]

        _ax.plot(
            dt_[-focal_curve_.shape[0] :],
            100 * focal_curve_,
            c=_cmap(_norm(i)),
            lw=0.75,
            zorder=8,
        )

    _ax.set_xticks(dt_[24::24], dx_[24::24], rotation=0)
    _ax.set_ylim(0.0, 101)
    _ax.set_xlim(dt_[0], dt_[-1])
    _ax.set_ylabel("Capacity Factor (%)", size=14)

    _ax.tick_params(axis="both", labelsize=12)

    if colorbar:
        cbar = _fig.colorbar(
            cm.ScalarMappable(_norm, _cmap),
            cax=_ax.inset_axes([60, 80, 150, 5], transform=_ax.transData),
            orientation="horizontal",
        )

        cbar.set_ticks([0, len(F_curves_)], labels=["1:00", "23:00"], size=12)

        # cbar.ax.tick_params(length=0)

        cbar.ax.set_title("Event (update)", rotation=0, size=12)



def hillshade(_fig, _ax, _shapefile):

    # Initialize Earth Engine
    ee.Initialize()

    # Get the bounding box from
    bounds = _shapefile.total_bounds  # [minx, miny, maxx, maxy]
    roi = geemap.geopandas_to_ee(
        _shapefile
    )  # Convert GeoDataFrame to EE FeatureCollection

    _DEM = ee.Image("USGS/SRTMGL1_003").select("elevation")
    # hillshade = ee.Terrain.hillshade(_DEM, 180, 22.5)
    hillshade = ee.Terrain.hillshade(_DEM, 315, 21)
    # hillshade = ee.Terrain.hillshade(_DEM, 315, 20).unitScale(0, 255).pow(1.4).multiply(255)

    # Download hillshade as a NumPy array
    hillshade_np = geemap.ee_to_numpy(
        hillshade.clip(roi), region=roi, scale=1000
    ).astype(float)

    hs = hillshade_np.squeeze()
    hs[hs == 0] = np.nan

    # Create a colormap and set NaNs (bad values) to transparent
    _cmap = cm.get_cmap("Greys_r").copy()
    # _cmap = cm.get_cmap("gray").copy()
    _cmap.set_bad(color=(0, 0, 0, 0))  # RGBA: transparent

    _ax.imshow(
        hs,
        cmap=_cmap,
        extent=[bounds[0], bounds[2], bounds[1], bounds[3]],
        origin="upper",
        vmin=1,
    )

    return _fig, _ax


def plot_hillshade_frequency_map(
    _fig,
    _ax,
    palette_,
    shape_,
    _fdu,
    x_tr_,
    x_ts_,
    x_,
    idx_neighbors_,
    idx_temporal_,
    idx_spatial_,
    sigma,
):
    
    idx_ = idx_temporal_
    
    y_tr_ = x_tr_[idx_, :]
    x_tr_p_, inv_idx_, z_tr_p_ = np.unique(
        y_tr_, 
        return_inverse=True, 
        return_counts=True, 
        axis=0
    )

    z_tr_pp_ = z_tr_p_[inv_idx_]

    # Step 3: expand back to full x_tr_
    z_tr_ = np.zeros(x_tr_.shape[0], dtype=int)
    z_tr_[idx_] = z_tr_pp_
    print(z_tr_.shape, z_tr_p_.shape, y_tr_.shape, x_tr_p_.shape)

    _cmap = sns.color_palette("rocket_r", as_cmap=True)
    _norm = plt.Normalize(1, z_tr_p_.max())

    shape_.plot(
        ax=_ax, 
        facecolor="None", 
        edgecolor="w"
    )

    # _ax.scatter(x_[0], x_[1],
    #             c="lime",
    #             lw=1,
    #             ec="k",
    #             s=110,
    #             zorder=1,
    #             marker="o",
    #             clip_on=False,
    #             label="Asset (update)")

    _ax.plot(
        x_[0],
        x_[1],
        c="k",
        marker="None",
        ls="--",
        clip_on=False,
        label="Distance threshold",
    )

    _ax.plot(
        x_ts_[:, 0],
        x_ts_[:, 1],
        c="lightgray",
        alpha=0.75,
        ms=7.5,
        marker="o",
        mec="w",
        ls="none",
        mew=0.75,
        zorder=2,
        clip_on=False,
        label="No neighboring assets",
    )

    _ax.plot(
        x_[0],
        x_[1],
        c=_cmap(_norm(z_tr_p_.max() / 2.0)),
        alpha=0.75,
        ms=7.5,
        ls="none",
        marker="o",
        mec="w",
        mew=1.0,
        zorder=0,
        clip_on=False,
        label="Neighboring assets",
    )

    _ax.plot(
        x_[0],
        x_[1],
        c=_cmap(_norm(z_tr_p_.max() / 4.0)),
        alpha=0.75,
        ms=7.5,
        ls="none",
        marker="o",
        mec="k",
        mew=1.0,
        zorder=0,
        clip_on=False,
        label="Selected neighboring assets",
    )

    for i in np.arange(x_tr_p_.shape[0], dtype=int)[np.argsort(z_tr_p_)]:
        if (x_tr_p_[i, 0] != x_[0]) | (x_tr_p_[i, 1] != x_[1]):
            _ax.plot(
                x_tr_p_[i, 0],
                x_tr_p_[i, 1],
                c=_cmap(_norm(z_tr_p_[i])),
                ms=6,
                ls="none",
                marker="o",
                mec="w",
                mew=1.0,
                zorder=6,
                clip_on=False,
            )
        else:
            _ax.plot(
                x_tr_p_[i, 0],
                x_tr_p_[i, 1],
                c=_cmap(_norm(z_tr_p_[i])),
                ms=14,
                ls="none",
                marker="d",
                mec="w",
                mew=1.0,
                zorder=4,
                label="Asset (update)",
                clip_on=False,
            )

    x_tr_p_ = np.unique(x_tr_[idx_spatial_, :], axis=0)
    for i in range(x_tr_p_.shape[0]):
        if (x_tr_p_[i, 0] != x_[0]) | (x_tr_p_[i, 1] != x_[1]):
            _ax.plot(
                x_tr_p_[i, 0],
                x_tr_p_[i, 1],
                ms=6,
                ls="none",
                marker="o",
                c="none",
                mec="k",
                mew=1.0,
                zorder=7,
                clip_on=False,
            )
        else:
            _ax.plot(
                x_tr_p_[i, 0],
                x_tr_p_[i, 1],
                ms=14,
                ls="none",
                marker="d",
                c="none",
                mec="lime",
                mew=1.0,
                zorder=5,
                # label = 'Asset (update)',
                clip_on=False,
            )

    minx, miny, maxx, maxy = shape_.total_bounds
    X_, Y_ = np.meshgrid(
        np.linspace(minx, maxx, 1000), 
        np.linspace(miny, maxy, 1000)
    )

    XX_ = np.concatenate([X_.flatten()[:, np.newaxis], Y_.flatten()[:, np.newaxis]], axis=1)
    Z_ = _fdu._haversine_dist(XX_, x_).reshape(X_.shape)

    contours = _ax.contour(
        X_,
        Y_,
        Z_,
        levels=[sigma],
        colors="k",
        linewidths=1.0,
        linestyles="dashed",
        zorder=11,
    )

    cbar = _fig.colorbar(
        cm.ScalarMappable(cmap=_cmap),
        cax=_ax.inset_axes([-103, 27.85, 2.0, 0.25], transform=_ax.transData),
        orientation="horizontal",
    )

    cbar.set_ticks([0, 1], labels=[1, int(z_tr_p_.max())], fontsize=14)

    cbar.ax.tick_params(length=0)

    cbar.ax.set_title("Neighbors", rotation=0, fontsize=14)

    _ax.set_axis_off()


def globe_inset(_fig, _ax, _TX, x0, y0, width, height):
    """
    Add an orthographic globe inset centered on Texas.

    Parameters
    ----------
    _ax : matplotlib Axes / GeoAxes
        Parent axes where the inset will be placed.
    _TX : GeoDataFrame
        Texas geometry.
    x0, y0 : float
        Lower-left corner of the inset in parent axes coordinates (0–1).
    width, height: float
        Size of the inset, also in parent axes coordinates (0–1).
    """

    # --- Data prep ---
    _world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    _world = _world.to_crs(epsg=4326)
    _TX = _TX.to_crs(epsg=4326)

    shape_centroid = _TX.geometry.unary_union.centroid

    ortho = ccrs.Orthographic(
        central_longitude=shape_centroid.x, 
        central_latitude=shape_centroid.y
    )

    # Use inset_axes *function*, not _ax.inset_axes
    # bbox_to_anchor=(x0, y0, width, height) is in _ax.transAxes
    _ax_inset = inset_axes(
        _ax,
        width="100%",  # 100% of the bbox width
        height="100%",  # 100% of the bbox height
        bbox_to_anchor=(x0, y0, width, height),
        bbox_transform=_ax.transAxes,
        axes_class=GeoAxes,
        axes_kwargs=dict(map_projection=ortho),
        borderpad=0.0,
    )

    # --- Drawing ---
    _ax_inset.add_geometries(
        _world.geometry,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="k",
        linewidth=0.25,
    )

    _ax_inset.add_geometries(
        _TX.geometry,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="r",
        zorder=10,
        linewidth=0.75,
    )

    gl = _ax_inset.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,  # no labels – small inset
        linewidth=0.4,
        color="white",
        alpha=0.825,
        xlocs=range(-180, 181, 30),
        ylocs=range(-90, 91, 30),
    )

    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    _ax_inset.coastlines(linewidth=0.5)
    _ax_inset.add_feature(cfeature.OCEAN, zorder=0)
    _ax_inset.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1)
    _ax_inset.add_feature(cfeature.BORDERS, linewidth=0.4, zorder=2)
    _ax_inset.set_global()


def plot_density_threshold(_fig, _ax, palette_, d_f_, d_e_, w_f_, w_e_, xi):

    _ax.plot(
        d_f_[np.argsort(d_f_)],
        w_f_[np.argsort(d_f_)],
        c=palette_.loc[2, "ibm"],
        lw=3,
        label="f",
        clip_on=False,
    )

    _ax.plot(
        d_e_[np.argsort(d_e_)],
        w_e_[np.argsort(d_e_)],
        c=palette_.loc[4, "ibm"],
        lw=3,
        label="e",
        clip_on=False,
    )

    _ax.axhline(
        xi, 
        ls="--", 
        color="k", 
        lw=1.0, 
        label=r"$\xi$", 
        zorder=10
    )

    _ax.set_ylabel(r"$\omega^{f,e}$", size=12)
    _ax.set_xlabel(r"$r_{f,e}$", size=12)

    _ax.tick_params(axis="both", labelsize=12)

    _ax.legend(frameon=False, fontsize=12)

    _ax.set_ylim(0, 1)
    _ax.set_xlim(0, 1)


def plot_functional_neighborhood(
    _fig, 
    _ax, 
    _ax_top, 
    _ax_left, 
    palette_, 
    idx_neighbors_, 
    w_f_, 
    w_e_, 
    xi
):

    _ax.scatter(
        w_f_, 
        w_e_, 
        color="gray", 
        s=0.25, 
        zorder=1, 
        clip_on=False
    )

    _ax.scatter(
        w_f_[idx_neighbors_],
        w_e_[idx_neighbors_],
        c=palette_.loc[0, "ibm"],
        s=0.25,
        zorder=2,
        clip_on=False,
    )

    _ax.axhline(xi, ls="--", color="k", lw=1.0, label=r"$\xi$", zorder=10)

    _ax.axvline(xi, ls="--", color="k", lw=1.0, label=r"$\xi$", zorder=10)

    _ax.set_ylabel(r"$\omega^{e}$", size=12)
    _ax.set_xlabel(r"$\omega^{f}$", size=12)

    _ax.tick_params(axis="both", labelsize=12)

    _ax.set_ylim(0, 1)
    _ax.set_xlim(0, 1)

    _ax_top.hist(w_f_, bins=25, range=(0, 1), color="gray", density=True)

    _ax_top.tick_params(axis="both", labelsize=12)

    _ax_top.set_xlim(0, 1)

    _ax_left.hist(
        w_e_,
        bins=25,
        range=(0, 1),
        color="gray",
        density=True,
        orientation="horizontal",
    )

    _ax_left.set_ylim(0, 1)

    _ax_left.tick_params(axis="both", labelsize=12)

    _ax.tick_params(axis="y", labelleft=False)

    _ax_top.tick_params(axis="x", labelbottom=False)

    _ax_left.tick_params(axis="y", labelleft=False)


def plot_selected_functional_neighbors(
    _fig, 
    _ax, 
    palette_, 
    idx_neighbors_, 
    idx_temporal_, 
    idx_spatial_, 
    w_f_, 
    w_e_, 
    w_, 
    xi
):

    _ax.scatter(
        w_f_,
        w_e_,
        c="gray",
        s=10,
        alpha=1.0,
        lw=0.25,
        ec="k",
        label="Scenario",
        clip_on=True,
    )

    _ax.scatter(
        w_f_[idx_neighbors_],
        w_e_[idx_neighbors_],
        c=palette_.loc[0, "ibm"],
        s=10,
        alpha=1.0,
        lw=0.25,
        ec="k",
        label="Neighbor",
        clip_on=False,
    )

    _ax.scatter(
        w_f_[idx_spatial_],
        w_e_[idx_spatial_],
        c=palette_.loc[3, "ibm"],
        s=10,
        lw=0.25,
        ec="k",
        alpha=1.0,
        label="Selected neighbor",
        clip_on=False,
    )

    _ax.axline((1, 1), slope=1, lw=1, c="k")

    _ax.set_ylabel(r"$\omega^{e}$", size=12)
    _ax.set_xlabel(r"$\omega^{f}$", size=12)

    _ax.tick_params(axis="both", labelsize=12)

    _ax.set_xlim(xi, 1)
    _ax.set_ylim(xi, 1)

    _ax.axhline(xi, ls="--", color="k", lw=1.0, zorder=10)

    _ax.axvline(xi, ls="--", color="k", lw=1.0, zorder=10)


def _check_limit(x):
    if x > 365:
        x = x - 365
    if x < 1:
        x = x + 365
    return x


def plot_filtered_scenarios(
    fig,
    _ax,
    palette_,
    idx_neighbors_,
    idx_temporal_,
    idx_spatial_,
    d_h_,
    t_tr_,
    t_ts,
    sigma,
    gamma,
    gamma_prime,
):

    _ax.scatter(
        t_tr_[idx_neighbors_],
        d_h_[idx_neighbors_],
        s=75,
        c="darkgray",
        lw=0.5,
        edgecolor="w",
        clip_on=False,
        zorder=4,
        label="Neighbors",
    )

    _ax.scatter(
        t_tr_[idx_temporal_],
        d_h_[idx_temporal_],
        s=75,
        c=palette_.loc[3, "ibm"],
        lw=0.5,
        edgecolor="w",
        clip_on=False,
        zorder=5,
        label="Temporal Neighbors",
    )

    _ax.scatter(
        t_tr_[idx_spatial_],
        d_h_[idx_spatial_],
        s=75,
        c=palette_.loc[3, "ibm"],
        lw=0.5,
        edgecolor="k",
        clip_on=False,
        zorder=5,
        label="Spatial Neighbors",
    )

    _ax.axvline(
        t_ts, 
        color=palette_.loc[0, "ibm"], 
        ls="--", 
        lw=4, 
        zorder=6
    )

    if gamma_prime != 0:
        _ax.axvline(
            _check_limit(t_ts + gamma), 
            c="k", 
            ls="--", 
            lw=1.5, 
            zorder=10
        )

        _ax.axvline(
            _check_limit(t_ts - gamma),
            c="k",
            ls="--",
            lw=1.5,
            zorder=10,
            label="Thresholds",
        )

    if sigma != 0:
        _ax.axhline(sigma, c="k", ls="--", lw=1.5, zorder=10)

    # _ax.set_ylabel(r"$|| \mathbf{x}_\star - \mathbf{x}_n ||_\mathrm{H}$", size=14)
    # _ax.set_xlabel(r"$|| d_\star - d_n ||_\mathrm{p}$", size=16)
    # _ax.set_xlabel(r"Year Day", size=16)
    _ax.set_ylabel(r"Distance (km)", size=18)
    _ax.set_xlim(1, 365)
    _ax.set_ylim(0, d_h_.max())
    _ax.set_xticks([], [])

    _ax.tick_params(axis="both", labelsize=16)

    _ax.set_xlim(1, 365)
    _ax.invert_yaxis()


def plot_dates_histogram(
    _fig,
    _ax,
    palette_,
    idx_neighbors_,
    idx_temporal_,
    idx_spatial_,
    d_h_,
    t_tr_,
    t_ts,
    gamma,
    gamma_prime,
):

    _ax.hist(
        t_tr_[idx_neighbors_],
        bins=50,
        range=(1, 365),
        color="darkgray",
        edgecolor="w",
        lw=0.5,
    )

    _ax.hist(
        t_tr_[idx_temporal_],
        bins=50,
        range=(1, 365),
        alpha=0.5,
        color=palette_.loc[3, "ibm"],
        edgecolor="w",
        lw=0.5,
    )

    _ax.hist(
        t_tr_[idx_spatial_],
        bins=50,
        range=(0, 365),
        alpha=0.5,
        color=palette_.loc[3, "ibm"],
        edgecolor="k",
        lw=0.5,
    )

    if gamma_prime != 0:
        _ax.axvline(
            _check_limit(t_ts - gamma), 
            c="k", 
            ls="--", 
            lw=1.5, 
            zorder=2, 
            label="d: day"
        )

        _ax.axvline(
            _check_limit(t_ts + gamma), 
            c="k", 
            ls="--", 
            lw=1.5, 
            zorder=2, 
            label="d: day"
        )

    _ax.set_xlabel(r"Year Day", size=18)
    _ax.set_ylabel(r"Neighbors", size=18)
    _ax.set_xlim(1, 365)
    _ax.tick_params(axis="both", labelsize=16)
    # _ax.set_xticks([], [])


def plot_distance_histogram(
    _fig, 
    _ax, 
    palette_, 
    idx_neighbors_, 
    idx_temporal_, 
    idx_spatial_, 
    d_h_, 
    sigma
):

    d_max = d_h_.max()

    _ax.hist(
        d_h_[idx_neighbors_],
        bins=50,
        range=(0, d_max),
        color="darkgray",
        edgecolor="w",
        lw=0.5,
        orientation="horizontal",
    )

    _ax.hist(
        d_h_[idx_temporal_],
        bins=50,
        range=(0, d_h_.max()),
        alpha=0.5,
        color=palette_.loc[3, "ibm"],
        edgecolor="w",
        lw=0.5,
        orientation="horizontal",
    )

    _ax.hist(
        d_h_[idx_spatial_],
        bins=50,
        range=(0, d_h_.max()),
        alpha=0.5,
        color=palette_.loc[3, "ibm"],
        edgecolor="k",
        lw=1,
        orientation="horizontal",
    )

    if sigma != 0:
        _ax.axhline(
            sigma, 
            c="k", 
            ls="--", 
            lw=1.5, 
            zorder=10
        )

    _ax.xaxis.set_label_position("top")
    _ax.xaxis.tick_top()
    # _ax.set_xlabel(r"Distance (km)", size=16)
    _ax.set_xlabel(r"Neighbors", size=18)
    _ax.set_ylim(0, d_max)
    _ax.tick_params(axis="both", labelsize=16, rotation=270)
    _ax.set_yticks([], [])
    _ax.invert_yaxis()
    # _ax.legend(frameon=False, ncol=1, fontsize = 12)

def selected_scenarios_heatmap(
    _fig, 
    _ax, 
    palette_,
    _haversine_dist, 
    idx_, 
    d_h_, 
    x_, 
    t_tr_, 
    x_ts_, 
    t_ts, 
    colorbar=True,
    N = 12,
    delta = 15,
):

    d_ = _haversine_dist(x_ts_, x_)
    d_sort_ = np.sort(d_)[:-7]

    tops_ = [int((i + 1) * delta) for i in range(N)]
    intervals_ = [d_sort_[i - 1] for i in tops_]

    _date = datetime.datetime.strptime(t_ts, "%Y-%m-%d %H:%M:%S")
    m_a = _date.timetuple().tm_mon - 1
    day = _date.timetuple().tm_mday - 1
    _, n_days = calendar.monthrange(_date.year, m_a + 1)

    m_tr_ = np.stack(
        [datetime.datetime.strptime(t_tr_[i], "%Y-%m-%d %H:%M:%S").timetuple().tm_mon - 1
         for i in range(t_tr_.shape[0])]
    )

    K = 0
    heatmap_ = np.zeros((N + 1, m_tr_.max() + 1))
    for d_h, m in zip(d_h_[idx_], m_tr_[idx_]):
        heatmap_[np.searchsorted(intervals_, d_h), m] += 1
        K += 1

    h_max = int(heatmap_.max())
    heatmap_[heatmap_ == 0.0] = np.nan

    _cmap = sns.color_palette("Spectral_r", as_cmap=True)
    _cmap.set_bad(color="lightgray")  # RGBA: transparent

    month_ = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    _ax.matshow(heatmap_[:-1, :], cmap=_cmap, vmin=0.0, extent=[0, 12, N, 0])

    _ax.set_xticks(range(len(month_)), [] * len(month_))
    _ax.set_yticks(range(len(tops_)), [] * len(tops_))

    _ax.set_ylabel("Top Spatial Neighboring Asset", size=18)

    # if xlabel:
    #     _ax.set_xlabel("Month", size=16)
    _ax.xaxis.set_label_position("top")

    _ax.set_xticks(
        np.arange(len(month_), dtype=int) + 0.5,
        month_,
        rotation=45,
        minor=True,
        size=18,
    )

    _ax.set_yticks(np.arange(len(tops_), dtype=int) + 0.5, tops_, minor=True, size=16)

    _ax.tick_params(which="major", bottom=False, left=False, top=False)

    _ax.tick_params(which="minor", bottom=False)

    _ax.grid(which="major", color="k", linestyle="-", linewidth=1.5)

    _ax.axvline(
        m_a + day / n_days, 
        color=palette_.loc[0, "ibm"], 
        ls="--", 
        lw=3, 
        zorder=6
    )

    if colorbar:
        cbar = _fig.colorbar(
            cm.ScalarMappable(cmap=_cmap),
            cax=_ax.inset_axes([3.5, 13.125, 5.0, 0.5], transform=_ax.transData),
            orientation="horizontal",
        )

        cbar.set_ticks([0, 1], labels=[1, h_max], fontsize=16)

        cbar.ax.tick_params(length=0)

        cbar.ax.set_title("Neighbors", rotation=0, fontsize=16)



def scenarios_frequency_dates(
    _fig,
    _ax,
    palette_,
    idx_neighbors_,
    idx_temporal_,
    idx_spatial_,
    t_tr_,
    t_ts,
    scale=2.5,
    colorbar=True,
):

    _date = datetime.datetime.strptime(t_ts, "%Y-%m-%d %H:%M:%S")
    m_a = _date.timetuple().tm_mon - 1
    day = _date.timetuple().tm_mday - 1
    _, n_days = calendar.monthrange(_date.year, m_a + 1)

    m_tr_ = np.stack(
        [datetime.datetime.strptime(t_tr_[i], "%Y-%m-%d %H:%M:%S").timetuple().tm_mon - 1 
         for i in range(t_tr_.shape[0])]
    )

    month_name_ = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    months_, counts_ = np.unique(m_tr_[idx_neighbors_], return_counts=True)

    counts_1_ = np.zeros(12)
    for month, count in zip(months_, counts_):
        counts_1_[month] = count

    months_, counts_ = np.unique(m_tr_[idx_temporal_], return_counts=True)

    counts_2_ = np.zeros(12)
    for month, count in zip(months_, counts_):
        counts_2_[month] = count

    months_ = np.arange(12, dtype=int)
    jitter_strength = 0.5
    x_ = 0.0 + np.random.normal(0, jitter_strength, size=months_.shape[0]) * 0.0

    counts_2_p_ = counts_2_.copy()
    counts_2_p_[counts_2_ == 0] = np.nan

    _cmap = sns.color_palette("viridis", as_cmap=True)
    _cmap.set_bad(color="lightgray")  # RGBA: transparent
    _norm = plt.Normalize(1, counts_2_.max())

    for i in range(x_.shape[0]):
        # Plot
        _ax.scatter(
            x_[i],
            months_[i],
            s=counts_1_[i] * scale,
            color=_cmap(_norm(counts_2_p_[i])),
            ec="k",
            vmin=1,
            vmax=counts_2_.max(),
            lw=0.75,
            clip_on=False,
            zorder=6,
        )

    _ax.set_xlim(-5, 5)

    _ax.set_xticks(x_, [] * len(x_))

    _ax.set_yticks(np.arange(len(month_name_), dtype=int), month_name_, size=14)

    _ax.tick_params(axis="y", length=0)
    _ax.tick_params(axis="x", length=0)

    _ax.axhline(
        m_a + day / n_days, color=palette_.loc[0, "ibm"], ls="--", lw=2.5, zorder=6
    )

    cbar = _fig.colorbar(
        cm.ScalarMappable(cmap=_cmap),
        cax=_ax.inset_axes([7.5, 4, 2, 3.25], transform=_ax.transData),
    )

    cbar.set_ticks([0, 1], labels=[1, int(counts_2_.max())], fontsize=12)

    cbar.ax.set_title("Neighbors", rotation=0, fontsize=12)
    _ax.set_xlim(-4, 4)

    sns.despine(left=True, bottom=True)



