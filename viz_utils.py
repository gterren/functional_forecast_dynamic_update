import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from matplotlib import cm
from sklearn.neighbors import KernelDensity

from ffc_utils import _silverman_rule
from ffc_utils import _haversine_dist

def _plot_forecast_parameters(_ax, palette_, f_, e_, f_hat_, phi_, psi_, eta_, dx_, dt_, t):

    tau_ = dt_[:t]
    s_ = dt_[t:]

    _ax.plot(tau_, 100.0 * f_, 
             c=palette_.loc[0, "ibm"], 
             label="CF (ac)", 
             lw=1.5, 
             zorder=5)

    _ax.plot(s_, 100.0 * f_hat_, 
             c=palette_.loc[0, "ibm"], 
             lw=1.5, 
             ls="--", 
             zorder=5)

    _ax.plot(dt_, 100.0 * e_, 
             lw=1.5, 
             label="CF (fc)", 
             c="k", 
             zorder=4)

    _ax.fill_between(tau_, 100.0 * np.ones(tau_.shape), 100.0 * np.zeros(tau_.shape),
                     color="lightgray",
                     alpha=0.25,
                     zorder=1)

    _ax.plot(tau_, 100.0 * phi_,
             c=palette_.loc[3, "ibm"],
             lw=3,
             label=r"$\phi_{\varepsilon_f} (\tau)$",
             zorder=2)

    _ax.plot(tau_, 100.0 * psi_[:t],
             c=palette_.loc[1, "ibm"],
             lw=3,
             label=r"$\phi_{\varepsilon_e} (\tau)$",
             zorder=2)

    _ax.plot(s_, 100.0 * psi_[t:],
             c=palette_.loc[2, "ibm"],
             label=r"$\psi_{\beta} (s)$",
             lw=3,
             zorder=2)

    _ax.plot(s_, 100.0 * eta_,
             c=palette_.loc[4, "ibm"],
             lw=3,
             label=r"$\sigma_{\alpha} (s) $",
             zorder=2)
    
    _ax.axvline(dt_[t - 1], 
                color="k", 
                lw=0.75, 
                label="event (du)", 
                zorder=6)


    _ax.set_xticks(dt_[::12], dx_[::12], rotation=22.5)
    _ax.set_ylabel("Capacity Factor (%)", size=14)

    _ax.tick_params(axis="both", 
                    labelsize=12)
    
    _ax.set_ylim(0.0, 100.0)
    _ax.set_xlim(dt_[0], dt_[-1])

    _ax.legend(frameon=False, 
               ncol=4, 
               loc="upper left")

def _plot_fneighborhood(_ax, palette_, d_f_, d_e_, w_f_, w_e_, w_, idx_0_, idx_1_, idx_2_, xi):

    _ax["a"].plot(d_f_[np.argsort(d_f_)], w_f_[np.argsort(d_f_)],
                  c=palette_.loc[0, "ibm"],
                  lw=2.5,
                  label="f")

    _ax["a"].plot(d_e_[np.argsort(d_e_)], w_e_[np.argsort(d_e_)],
                  c=palette_.loc[3, "ibm"],
                  lw=2.5,
                  label="e")

    _ax["a"].axhline(xi, 
                     ls="--", 
                     color="k", 
                     lw=1.0, 
                     label=r"$\xi$", 
                     zorder=10)

    _ax["a"].set_ylabel(r"$\varphi_{\lambda} (r)$", size=14)
    _ax["a"].set_xlabel(r"$r$", size=14)

    _ax["a"].tick_params(axis="both", 
                         labelsize=12)
    
    _ax["a"].legend(frameon=False, 
                    fontsize=12)
    
    _ax["a"].set_xlim(0, 1)
    _ax["a"].set_ylim(0, 1)

    _ax["e"].scatter(w_f_, w_e_, 
                     c="gray", 
                     s=35, 
                     alpha=1.0, 
                     lw=0.25, 
                     ec="k", 
                     label="Scen.")

    _ax["e"].scatter(w_f_[idx_1_], w_e_[idx_1_],
                     c=palette_.loc[0, "ibm"],
                     s=35,
                     alpha=1.0,
                     lw=0.25,
                     ec="k",
                     label="Neighbors")

    c_ = [palette_.loc[2, "ibm"], palette_.loc[4, "ibm"]]
    # for i, j in zip(idx_2_, idx_0_[idx_2_]):
    _ax["e"].scatter(w_f_[idx_2_], w_e_[idx_2_],
                     c=palette_.loc[3, "ibm"],
                     s=35,
                     lw=0.25,
                     ec="k",
                     alpha=1.0,
                     label="Filtered-out")

    _ax["e"].axline((1, 1), 
                    slope=1, 
                    lw=1, 
                    c="k")

    _ax["e"].set_ylabel(r"$\varphi_{\lambda_e} (r_{e})$", size=14)
    _ax["e"].set_xlabel(r"$\varphi_{\lambda_f} (r_{f})$", size=14)

    _ax["e"].tick_params(axis="both", 
                         labelsize=12)

    _ax["e"].set_xlim(w_f_[idx_2_].min() * 0.9985, w_f_[idx_2_].max() * 1.0015)
    _ax["e"].set_ylim(w_e_[idx_2_].min() * 0.9985, w_e_[idx_2_].max() * 1.0015)

    _ax["e"].axhline(xi, 
                     ls="--", 
                     color="k", 
                     lw=1.0, 
                     zorder=10)
    
    _ax["e"].axvline(xi, 
                     ls="--", 
                     color="k", 
                     lw=1.0, 
                     zorder=10)

    _ax["e"].legend(frameon=False, 
                    fontsize=12, 
                    ncol=3)

    _ax["c"].tick_params(axis="x", 
                         labelbottom=False)
    
    _ax["d"].tick_params(axis="y", 
                         labelleft=False)

    _ax["b"].scatter(w_f_, w_e_, 
                     color="gray", 
                     s=0.25, 
                     zorder=1)

    _ax["b"].scatter(w_f_[idx_1_], w_e_[idx_1_], 
                     c=palette_.loc[0, "ibm"], 
                     s=0.25, 
                     zorder=2)

    _ax["b"].axhline(xi, 
                     ls="--", 
                     color="k", 
                     lw=1.0, 
                     label=r"$\xi$", 
                     zorder=10)

    _ax["b"].axvline(xi, 
                     ls="--", 
                     color="k", 
                     lw=1.0, 
                     label=r"$\xi$", 
                     zorder=10)

    _ax["b"].set_ylabel(r"$\varphi_{\lambda_e} (r_{e})$", size=14)
    _ax["b"].set_xlabel(r"$\varphi_{\lambda_f} (r_{f})$", size=14)

    _ax["b"].tick_params(axis="both", 
                         labelsize=12)
    _ax["b"].set_xlim(0, 1)
    _ax["b"].set_ylim(0, 1)

    _ax["c"].hist(w_f_, 
                  bins=50, 
                  color="gray", 
                  density=True)

    _ax["c"].tick_params(axis="both", 
                         labelsize=12)


    _ax["c"].set_xlim(0, 1)

    _ax["d"].hist(w_e_, 
                  bins=50, 
                  color="gray", 
                  density=True, 
                  orientation="horizontal")

    _ax["d"].tick_params(axis="both", 
                         labelsize=12)
    
    _ax["d"].set_ylim(0, 1)

def _plot_forecasts(_ax, palette_, E_tr_, f_, e_, f_hat_, w_, idx_, dx_, dt_, t):

    tau_ = dt_[:t]
    s_ = dt_[t:]

    z_ = (w_ - w_[idx_].min()) / (w_[idx_].max() - w_[idx_].min())
    idx_ = idx_[np.argsort(w_[idx_])]
    _cmap = sns.color_palette("flare", as_cmap=True)
    _norm = plt.Normalize(z_[idx_].min(), z_[idx_].max())

    _ax.plot(tau_, 100.0 * f_,
             c=palette_.loc[0, "ibm"],
             clip_on=False,
             zorder=10,
             #label="CF (ac)",
             lw=2)

    _ax.plot(s_, 100.0 * f_hat_,
             c=palette_.loc[0, "ibm"],
             clip_on=False,
             zorder=10,
             lw=2,
             ls="--")

    _ax.plot(dt_, 100.0 * e_, 
             clip_on=False, 
             lw=2, 
             zorder=9, 
             #label="CF (fc)", 
             c="k")

    _ax.plot([], [], 
             label=r"$E_{\kappa_{\xi}}(t)$", 
             c="orange")

    for i, j in zip(idx_, range(idx_.shape[0])):
        _ax.plot(dt_, 100 * E_tr_[i, :], 
                 c=_cmap(_norm(z_[i])), 
                 lw=0.75, 
                 zorder=8)

    _ax.fill_between(tau_, 100 * np.ones(tau_.shape), 100 * np.zeros(tau_.shape),
                     color="lightgray",
                     alpha=0.25,
                     zorder=1)

    _ax.axvline(dt_[t - 1], 
                color="k", 
                lw=1.0, 
                zorder=11)

    _ax.set_xticks(dt_[::12], dx_[::12], rotation=22.5)

    _ax.set_ylim(0.0, 100.0)
    _ax.set_xlim(dt_[0], dt_[-1])
    _ax.set_ylabel("Capacity Factor (%)", size=14)

    _ax.tick_params(axis="both", 
                    labelsize=12)

    _ax.legend(frameon=False, 
               fontsize=12, 
               ncol=10, loc = 'upper center')

def _plot_scenarios(_ax, palette_, F_tr_, f_, e_, f_hat_, w_, idx_, dx_, dt_, t):

    tau_ = dt_[:t]
    s_ = dt_[t:]

    z_ = (w_ - w_[idx_].min()) / (w_[idx_].max() - w_[idx_].min())
    idx_ = idx_[np.argsort(w_[idx_])]
    _cmap = sns.color_palette("flare", as_cmap=True)
    _norm = plt.Normalize(z_[idx_].min(), z_[idx_].max())
    print(tau_.shape, s_.shape, f_.shape)
    _ax.plot(tau_, 100.0 * f_, 
             c=palette_.loc[0, "ibm"], 
             zorder=10, 
             #label="CF (ac)", 
             lw=2)

    _ax.plot(s_, 100.0 * f_hat_, 
             c=palette_.loc[0, "ibm"], 
             zorder=10, 
             lw=2, 
             ls="--")

    _ax.plot(dt_, 100.0 * e_, 
             lw=2, 
             zorder=9, 
             #label="CF (fc)", 
             c="k")

    _ax.plot([], [], 
             label=r"$\mathcal{F}_{\kappa_{\xi}}(t)$", 
             c="orange")

    for i, j in zip(idx_, range(idx_.shape[0])):
        _ax.plot(dt_, 100 * F_tr_[i, :], 
                 c=_cmap(_norm(z_[i])), 
                 lw=0.75, 
                 zorder=8)

    _ax.fill_between(tau_, 100 * np.ones(tau_.shape), 100 * np.zeros(tau_.shape),
                     color="lightgray",
                     alpha=0.25,
                     zorder=1)

    _ax.axvline(dt_[t - 1], 
                color="k", 
                lw=1.0, 
                zorder=11)

    _ax.set_xticks(dt_[::12], dx_[::12], rotation=22.5)
    _ax.set_ylim(0.0, 100.0)
    _ax.set_xlim(dt_[0], dt_[-1])
    _ax.set_ylabel("Capacity Factor (%)", size=14)

    _ax.tick_params(axis="both", 
                    labelsize=12)

    _ax.legend(frameon=False,
               fontsize=12, 
               ncol=10, loc = 'upper center')

def _plot_updates(_fig, _ax, palette_, M_, f_, e_, f_hat_, w_, idx_, dx_, dt_, t):

    tau_ = dt_[:t]
    s_ = dt_[t:]

    z_ = (w_ - w_[idx_].min()) / (w_[idx_].max() - w_[idx_].min())
    idx_ = idx_[np.argsort(w_[idx_])]
    _cmap = sns.color_palette("flare", as_cmap=True)
    _norm = plt.Normalize(z_[idx_].min(), z_[idx_].max())

    _ax.plot([], [], 
             label=r"$\mu(s)$", 
             c="orange")

    for i, j in zip(idx_, range(M_.shape[0])):
        _ax.plot(dt_[t:], 100 * M_[j, :],
                 c=_cmap(_norm(z_[i])), 
                 lw=0.75, 
                 zorder=8)

    _ax.fill_between(tau_, 100 * np.ones(tau_.shape), 100 * np.zeros(tau_.shape),
                     color="lightgray",
                     alpha=0.375,
                     zorder=1)

    _ax.plot(tau_, 100.0 * f_,
             c=palette_.loc[0, "ibm"],
             zorder=10,
             label="CF (ac)",
             clip_on=False,
             lw=2)

    _ax.plot(s_, 100.0 * f_hat_,
             c=palette_.loc[0, "ibm"],
             zorder=10,
             lw=2,
             clip_on=False,
             ls="--")

    _ax.plot(dt_, 100.0 * e_, 
             lw=2, 
             zorder=9, 
             label="CF (fc)", 
             clip_on=False, 
             c="k")
    
    _ax.axvline(dt_[t - 1], 
                color="k", 
                lw=1.0, 
                label="event (du)", 
                zorder=8)
    
    _ax.set_xticks(dt_[::12], dx_[::12], rotation=22.5)
    # ax_[2].set_yticks(size = 12)
    _ax.set_ylim(0.0, 100.0)
    _ax.set_xlim(dt_[0], dt_[-1])
    _ax.set_ylabel("Capacity Factor (%)", size=14)
    _ax.tick_params(axis="both", 
                    labelsize=12)
    
    _ax.legend(frameon=False, 
               fontsize=12, 
               ncol=1)

    cbar = _fig.colorbar(cm.ScalarMappable(_norm, _cmap),
                         cax=_ax.inset_axes([180, 75, 100, 7.5], transform=_ax.transData),
                         orientation="horizontal",
                         extend="both")

    cbar.set_ticks([0, 1], labels=["low", "high"])
    cbar.ax.tick_params(length=0)
    cbar.ax.set_title("Similarity", rotation=0)

def _plot_histogram_cuts(_ax, palette_, M_, e_, f_hat_, dx_, dt_, t, cut,
                         legend = False,
                         xlabel = False):

    tau_ = dt_[:t]
    s_   = dt_[t:]

    x_ = np.linspace(0, 100, 1000)[:, np.newaxis]

    _KD = KernelDensity(bandwidth=_silverman_rule(100 * M_[:, cut]),
                        algorithm="auto",
                        kernel="gaussian").fit(100 * M_[:, cut][:, np.newaxis])

    _ax.axvline(100.0 * f_hat_[cut],
                color=palette_.loc[0, "ibm"],
                lw=2,
                ls="--",
                label="CF (ac)",
                zorder=10)

    _ax.axvline(100.0 * e_[t + cut], 
                color="k", 
                lw=2, 
                label="CF (fc)", 
                zorder=10)

    _ax.hist(100.0 * M_[:, cut],
             bins=25,
             range=(0, 100),
             density=True,
             edgecolor='w', 
             linewidth=.5,
             color=palette_.loc[3, "ibm"],
             zorder=8)

    _ax.plot(x_, np.exp(_KD.score_samples(x_)),
             label="KDE (du)",
             color=palette_.loc[1, "ibm"],
             lw=3,
             zorder=9)

    _ax.set_title(dx_[t:][cut])
    _ax.set_xlim(0, 100)
    _ax.set_ylim(0,)

    if xlabel:
        _ax.set_xlabel("Capacity Factor (%)", size=14)

    _ax.tick_params(axis="both", 
                    labelsize=12)

    _ax.set_ylabel("EDF", size=14)

    if legend:
        _ax.legend(frameon=False, 
                    fontsize=12)

def _plot_density_heatmap(_fig, _ax, palette_, M_, f_, e_, f_hat_, dx_, dt_, t, cuts_):

    tau_ = dt_[:t]
    s_ = dt_[t:]

    Z_ = []
    for i in range(M_.shape[1]):
        a_, b_ = np.histogram(100.0 * M_[:, i], 
                              bins=25, 
                              range=(0, 100), 
                              density=True)
        Z_.append(a_)

    Z_ = np.stack(Z_).T
    X_, Y_ = np.meshgrid(dt_[t:], (b_[1:] + b_[:-1]) / 2.0)

    _cmap = sns.color_palette("rocket_r", as_cmap=True)
    _ax.pcolormesh(X_, Y_, Z_, cmap=_cmap)

    _ax.plot(dt_, 100.0 * e_, 
             c="k", 
             lw=2.0, 
             label="CF (fc)", 
             clip_on=False)

    _ax.plot(tau_, 100.0 * f_,
             c=palette_.loc[0, "ibm"],
             clip_on=False,
             lw=2.0,
             label="CF (ac)")

    _ax.plot(s_, 100.0 * f_hat_, 
             c=palette_.loc[0, "ibm"], 
             clip_on=False, 
             lw=2.0, 
             ls="--")

    _ax.fill_between(tau_, 100.0 * np.ones(tau_.shape), 100.0 * np.zeros(tau_.shape),
                     color="lightgray",
                     alpha=0.375)

    _ax.axvline(dt_[t - 1], 
                color="k", 
                linewidth=0.75, 
                label="event (du)")

    _ax.axvline(dt_[t], 
                color="k", 
                lw=0.75, 
                ls="--", 
                label="detail")

    for cut in cuts_:
        _ax.axvline(dt_[t + cut], 
                    color="k", 
                    lw=0.75, 
                    ls="--")

    _ax.set_xticks(dt_[::12], dx_[::12], rotation=22.5)
    _ax.set_ylabel("Capacity Factor (%)", size=14)

    _ax.tick_params(axis="both", 
                    labelsize=12)

    _ax.set_ylim(0.0, 100.0)
    _ax.set_xlim(dt_[0], dt_[-1])

    _ax.legend(frameon=False, 
               fontsize=12, 
               ncol=1)

    cbar = _fig.colorbar(cm.ScalarMappable(cmap=_cmap),
                         cax=_ax.inset_axes([180, 75, 100, 7.5], transform=_ax.transData),
                         orientation="horizontal",
                         extend="max")

    cbar.set_ticks([0, 1], labels=["low", "high"], fontsize=12)
    cbar.ax.tick_params(length=0)

    cbar.ax.set_title("EDF", 
                      rotation=0, 
                      fontsize=14)

def _plot_frequency_map(_fig, _ax, TX_, x_tr_, x_ts_, x_, idx_1_, idx_2_, sigma):

    x_tr_p_, z_tr_p_ = np.unique(x_tr_[idx_1_, :], return_counts=True, axis=0)

    _cmap = sns.color_palette("rocket_r", as_cmap=True)
    _norm = plt.Normalize(0.0, z_tr_p_.max())

    TX_.plot(ax=_ax, 
             facecolor="lightgray", 
             edgecolor="white", 
             zorder=0)

    _ax.scatter(x_[0], x_[1],
                c="lime",
                lw=1,
                ec="k",
                s=100.,
                zorder=1,
                marker="o",
                label="Asset (du)")

    if sigma != 0:
        _ax.scatter(x_[0], x_[1],
                    c="none",
                    lw=0.5,
                    ec="k",
                    ls = '--',
                    s=sigma**2,
                    zorder=1,
                    marker="o")

    _ax.plot(x_[0], x_[1],
             c="k",
             marker="None",
             ls="--",
             label="Distance threshold")
    
    _ax.plot(x_ts_[:, 0], x_ts_[:, 1],
             c="gray",
             alpha=0.75,
             ms=6,
             marker="o",
             mec="w",
             ls="none",
             mew=1.0,
             zorder=2,
             label="Assets wo/ neighbors")

    _ax.plot(x_[0], x_[1],
             c=_cmap(_norm(z_tr_p_.max() / 2.0)),
             alpha=0.75,
             ms=6,
             ls="none",
             marker="o",
             mec="w",
             mew=1.0,
             zorder=0,
             label="Assets w/ neighbors")

    _ax.plot(x_[0], x_[1],
             c=_cmap(_norm(z_tr_p_.max() / 4.0)),
             alpha=0.75,
             ms=6,
             ls="none",
             marker="o",
             mec="k",
             mew=1.0,
             zorder=0,
             label="Assets w/ neighbors after filtering-out")

    for i in np.arange(x_tr_p_.shape[0], dtype=int)[np.argsort(z_tr_p_)]:
        _ax.plot(x_tr_p_[i, 0], x_tr_p_[i, 1],
                 c=_cmap(_norm(z_tr_p_[i])),
                 alpha=0.75,
                 ms=6,
                 ls="none",
                 marker="o",
                 mec="w",
                 mew=1.0,
                 zorder=4)
    
    for i in range(x_tr_[idx_2_, :].shape[0]):
        _ax.plot(x_tr_[idx_2_[i], 0], x_tr_[idx_2_[i], 1],
                 alpha=0.75,
                 ms=5,
                 ls="none",
                 marker="o",
                 c = 'none', 
                 mec="k",
                 mew=1.0,
                 zorder=5)

    cbar = _fig.colorbar(cm.ScalarMappable(cmap=_cmap),
                         cax=_ax.inset_axes([-97.75, 35.5, 2.0, 0.25], transform=_ax.transData),
                         orientation="horizontal",
                         extend="both")

    cbar.set_ticks([0, 1], 
                   labels=["low", "high"], 
                   fontsize=12)
    
    cbar.ax.tick_params(length=0)
    
    cbar.ax.set_title("Frequency", 
                      rotation=0, 
                      fontsize=12)

    _ax.legend(frameon=False,
               bbox_to_anchor=(0.475, 0.25),
               ncol=1,
               fontsize=12)

    _ax.set_axis_off()

def _selected_scenarios_heatmap(_ax, T_tr_, d_h_, x_ts_, x_, idx_1_, t_ts):

    N     = 12
    delta = 15

    d_      = _haversine_dist(x_, x_ts_)
    d_sort_ = np.sort(d_)[:-7]

    tops_      = [int((i + 1) * delta) for i in range(N)]
    intervals_ = [d_sort_[i - 1] for i in tops_]

    m_a   = datetime.datetime.strptime(t_ts, "%Y-%m-%d").timetuple().tm_mon - 1
    m_tr_ = np.stack([datetime.datetime.strptime(T_tr_[i], "%Y-%m-%d").timetuple().tm_mon - 1
                      for i in range(T_tr_.shape[0])])

    K        = 0
    heatmap_ = np.zeros((N + 1, m_tr_.max() + 1))
    for d_h, m in zip(d_h_[idx_1_], m_tr_[idx_1_]):
        heatmap_[np.searchsorted(intervals_, d_h), m] += 1
        K += 1

    _cmap = sns.color_palette("rocket_r", as_cmap=True)

    month_ = ["Jan",
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
              "Dec"]

    print(K, month_[m_a])

    _ax.matshow(heatmap_[:-1, :], 
                cmap = _cmap, 
                vmin=0.0, 
                extent=[0, 12, N, 0])

    _ax.set_xticks(range(len(month_)), [] * len(month_))
    _ax.set_yticks(range(len(tops_)), [] * len(tops_))

    _ax.set_ylabel("Top Neighboring Assets", size=16)
    _ax.set_xlabel("Month", size=16)
    _ax.xaxis.set_label_position("top")

    _ax.set_xticks(np.arange(len(month_), dtype=int) + 0.5, month_,
                   rotation=45,
                   minor=True,
                   size=14)

    _ax.set_yticks(np.arange(len(tops_), dtype=int) + 0.5, tops_,
                   minor=True,
                   size=14)

    _ax.tick_params(which="major", 
                    bottom=False, 
                    left=False, 
                    top=False)
    
    _ax.tick_params(which="minor", 
                    bottom=False)

    _ax.grid(which="major", 
             color="k", 
             linestyle="-", 
             linewidth=1.5)

def _plot_scenario_filtering(_ax, palette_, d_h_, t_tr_, idx_1_, idx_2_, t_ts, Gamma, gamma, sigma):

    def __check_limit(x):
        if x > 365:
            x = x - 365
        if x < 1:
            x = x + 365
        return x

    _ax.scatter(t_tr_[idx_1_], d_h_[idx_1_],
                s=75,
                c="darkgray",
                lw=0.5,
                edgecolor="w",
                clip_on=False,
                zorder=4,
                label="Neighbors")

    _ax.scatter(t_tr_[idx_2_], d_h_[idx_2_],
                s=75,
                c=palette_.loc[3, "ibm"],
                lw=0.5,
                edgecolor="k",
                clip_on=False,
                zorder=5,
                label="Selected")
    
    if Gamma != 0:  
        _ax.axvline(__check_limit(t_ts + gamma), 
                    c="k", 
                    ls='--', 
                    lw=1.5, 
                    zorder=10)
                    
        _ax.axvline(__check_limit(t_ts - gamma), 
                    c="k", 
                    ls='--', 
                    lw=1.5, 
                    zorder=10, 
                    label="Thresholds")
            
    if sigma != 0:
        _ax.axhline(sigma, 
                    c="k", 
                    ls='--', 
                    lw=1.5, 
                    zorder=10)

    #_ax.set_ylabel(r"$|| \mathbf{x}_\star - \mathbf{x}_n ||_\mathrm{H}$", size=14)
    #_ax.set_xlabel(r"$|| d_\star - d_n ||_\mathrm{p}$", size=16)
    _ax.set_xlabel(r"Year Day", size=16)
    _ax.set_ylabel(r"Distance (km)", size=16)

    _ax.tick_params(axis="both", 
                    labelsize=14)
    #sns.despine(ax=_ax, offset = 5, trim = True)
    _ax.set_xlim(1, 365)

def _plot_dates_histogram(_ax, palette_, t_tr_, idx_1_, idx_2_, t_ts, Gamma, gamma):


    def __check_limit(x):
        if x > 365:
            x = x - 365
        if x < 1:
            x = x + 365
        return x

    _ax.hist(t_tr_[idx_1_], 
             bins=50, 
             range=(1, 365), 
             color="darkgray", 
             edgecolor="w", 
             lw=0.5)

    _ax.hist(t_tr_[idx_2_],
             bins=50,
             range=(1, 365),
             alpha=0.5,
             color=palette_.loc[3, "ibm"],
             edgecolor="k",
             lw=0.5)
    
    if Gamma != 0:  
        _ax.axvline(__check_limit(t_ts - gamma), 
                    c="k", 
                    ls="--", 
                    lw=1.5, 
                    zorder=2, 
                    label="d: day")
        
        _ax.axvline(__check_limit(t_ts + gamma), 
                    c="k", 
                    ls="--", 
                    lw=1.5, 
                    zorder=2, 
                    label="d: day")
        
    #_ax.set_xlabel(r"Year Day", size=16)
    _ax.set_ylabel(r"Neighbors", size=16)
    _ax.set_xlim(1, 365)
    _ax.tick_params(axis="both", labelsize=14)
    #_ax.legend(frameon=False, ncol=1, fontsize = 12)
    _ax.set_xticks([], [])

def _plot_distance_histogram(_ax, palette_, d_h_, idx_1_, idx_2_, d_max = 1200):

    d_max = d_h_.max()
    
    _ax.hist(d_h_[idx_1_], 
             bins=50, 
             range=(0, d_max), 
             color="darkgray", 
             edgecolor="w", 
             lw=0.5, 
             orientation="horizontal")
    
    _ax.hist(d_h_[idx_2_],
             bins=50,
             range=(0, d_max),    
             alpha=0.5,
             color=palette_.loc[3, "ibm"],
             edgecolor="k",
             lw=0.5, 
             orientation="horizontal")
    
    _ax.axhline(d_h_[idx_2_].max(), 
                c="k", 
                ls="--", 
                lw=1.5, 
                zorder=2, 
                label="r: max distance")
    
    #_ax.set_xlabel(r"Distance (km)", size=16)
    _ax.set_xlabel(r"Neighbors", size=16)
    #_ax.set_xlim(0, d_max)

    _ax.tick_params(axis="both", 
                    labelsize=14, 
                    rotation = 270)
    
    _ax.set_yticks([], [])

def _plot_envelop(_ax, palette_, _upper, _lower, f_, f_hat_, dt_, dx_, tau_, s_, t, 
                  legend = True):

    s_p_ = np.concatenate([tau_[-1] * np.ones((1,)), s_], axis=0)

    _ax.plot(tau_, 100 * f_, 
            c       = palette_.loc[0, "ibm"], 
            label   = "CF (ac)", 
            lw      = 2.0, 
            clip_on = False)
    
    _ax.axvline(dt_[t - 1], 
                color     = "k", 
                linewidth = 1.0, 
                label     = "event (du)", 
                zorder    = 10)
    
    _ax.plot(s_, 100.0 * f_hat_, 
             c       = palette_.loc[0, "ibm"], 
             ls      = "--", 
             lw      = 2.0, 
             clip_on = False)

    for key, i in zip(_upper.keys(), range(len(_upper.keys()))):

        upper_ = np.concatenate([f_, _upper[key]], axis=0)[f_.shape[0] - 1:]
        lower_ = np.concatenate([f_, _lower[key]], axis=0)[f_.shape[0] - 1:]

        _ax.plot(s_p_, 100. * upper_, 
                 lw = 0.375, 
                 c  = "lightgray")
        
        _ax.plot(s_p_, 100. * lower_, 
                 lw = 0.375, 
                 c  = "lightgray")

        alpha = float(key)
        ci    = (1. - alpha)*100.
        
        _ax.fill_between(s_p_, 100. * upper_, 100. * lower_,
                        color  = "gray",
                        alpha  = 0.15 + i * 0.15,
                        label  = f"{ci}%",
                        zorder = 1)
            
    _ax.fill_between(tau_, 100 * np.ones(tau_.shape), 100 * np.zeros(tau_.shape),
                     color = "lightgray",
                     alpha = 0.25)

    _ax.set_xticks(dt_[::12], dx_[::12], rotation=22.5)
    
    if legend: 
        _ax.legend(frameon  = False, 
                   loc      = 'upper left',
                   fontsize = 12,
                   ncol     = 1)
        
    _ax.tick_params(axis      = "both", 
                    labelsize = 12)
    
    _ax.set_ylim(0.0, 100.0)
    _ax.set_xlim(dt_[0], dt_[-1])
    _ax.set_ylabel("Capacity Factor (%)", size = 14)