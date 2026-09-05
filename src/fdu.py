import numpy as np

from sklearn.neighbors import KernelDensity

from skfda.representation.basis import Fourier, BSpline
from skfda import FDataGrid

from statsmodels.distributions.empirical_distribution import ECDF

class functional_dynamic_update:
    
    def __init__(self, 
                 _distances = {'spatial': 'None', 
                               'temporal': 'None', 
                               'fusion': 'None'},
                 name = 'noname', 
                 date = 'nodate'):

        self._distances = _distances
        self.name = name
        self.date = date

    # Define exponential growth function
    def _exponential_growth(self, t, growth_rate, n_samples_per_hour):
        tau_ = np.linspace(t - 1, 0, t)/n_samples_per_hour
        return np.exp(np.log(0.5)*tau_/growth_rate)

    # Define exponential decay function
    def _exponential_decay(self, S, decay_rate, n_samples_per_hour):
        s_ = np.linspace(0, S - 1, S)/n_samples_per_hour
        return np.exp(np.log(0.5)*s_/decay_rate)    
    
    # def _logistic(x_, k):
    #     return 1. - 1.0 / (1.0 + np.exp(np.log(999) * x_ / (k*60/2)))

    # Linear Inverse Exponential (LIE) function
    def _linear_inverse_exponential(
        self, 
        x_, 
        T, 
        nu, 
        trust_rate, 
        n_samples_per_hour,
        eta = 2.5, 
        alpha = 1.
    ):
        minutes_per_sample = 60 / n_samples_per_hour
        T_minutes = T * minutes_per_sample
        nu_minutes = nu * 60
        x_ = x_ - T_minutes + nu_minutes
        x_ = eta * x_ / (nu_minutes - minutes_per_sample)
        
        y_ = np.where(
            x_ > 0, 
            -x_, 
            -alpha * (np.exp(x_) - 1)
        )
        
        return trust_rate*((y_ + eta) / (eta + alpha))

    # Calculate weighted (w_) distance between X_ and x_
    def _weighted_euclidian_dist(self, X_, x_, 
                                 w_ = [],
                                 normalize = True):
        if len(w_) == 0:
            w_ = np.ones(x_.shape)/x_.shape[0]
        # Normalize weights
        if normalize:
            w_ = w_ / w_.sum()
        # Calculate weighted Ecludian distance
        d_ = np.zeros((X_.shape[0],))
        for i in range(X_.shape[0]):
            d_[i] = w_.T @ (X_[i, :] - x_) ** 2
        return d_

    
    def _haversine_dist(self, x_1_, x_2_):
        """
        Calculate the distance between two points on Earth using the Haversine formula.
    
        Args:s
            x_1_ (float): Longitude and latitude of the first point in degrees.
            x_2_ (float): Longitude and latitude of the second point in degrees.
    
        Returns:
            float: Distance between the two points in kilometers.
        """
        # Radius of Earth in kilometers
        R = 6371  

        # Latitude and longitude distance in radians
        dlat_ = np.deg2rad(x_1_[:, 1]) - np.deg2rad(x_2_[1])
        dlon_ = np.deg2rad(x_1_[:, 0]) - np.deg2rad(x_2_[0])

        # Haversine distance
        theta = (np.sin(dlat_/2)**2 
                 + (np.cos(np.deg2rad(x_2_[1])) 
                    * np.cos(np.deg2rad(x_1_[:, 1]))*np.sin(dlon_/2)**2))
        
        return 2.*R*np.arcsin(np.sqrt(theta))


    def _graph_dist(self, x_1_):
        return x_1_.astype(int)
        
    # # Equinoxes - Solstices seasonal distance
    # def _seasonal_dist(self, d_1, d_2, gamma = None):

    #     # Periodic distance to rank samples by day of the year
    #     def __periodic_dist(d, gamma, 
    #                         alpha = 0.5,
    #                         day_to_degree = 360/365, 
    #                         degree_to_rad = np.pi/180):

    #         return np.sin(alpha*degree_to_rad*day_to_degree*(d - gamma))**2
        
    #     # --- seasonal membership ---
    #     # Soltices day of the year: 172 (summer), and 355 (winter)
    #     # Equinox day of the year: 80 (spring), and 266 (fall)
    #     def __seasonal_membership(d_2):
    
    #         d_s = np.min([__periodic_dist(172, d_2), __periodic_dist(355, d_2)])
    #         d_e = np.min([__periodic_dist(80, d_2), __periodic_dist(266, d_2)]) 
    #         D_ = np.array([d_s, d_e])
    #         m_ = np.zeros_like(D_)
    #         m_[np.argmin(D_)] = 1
        
    #         return m_
            
    #     # Soltices or Equinox
    #     m_ = __seasonal_membership(d_2)

    #     if gamma is None:
    #         # Calculate period distance
    #         return (m_[0] * __periodic_dist(d_1, d_2, alpha = 0.5) 
    #                 + m_[1] * __periodic_dist(d_1, d_2, alpha = 1))
    #     else:
    #         # Calculate period threshold
    #         return (m_[0] * __periodic_dist(d_1, d_2 + gamma, alpha = 0.5) 
    #                 + m_[1] * __periodic_dist(d_1, d_2 + gamma/2., alpha = 1))

  
    # # --- seasonal membership ---
    # # Solstices day of the year: 172 (summer), and 355 (winter)
    # # Equinox day of the year: 80 (spring), and 266 (fall)
    # def _equinox_dist(self, d1, d2, d_prime, thr):

    #     if d_prime < thr:
    #         self.period = 1
    #         return self._periodic_dist(d1, d2, alpha = self.period)
    #     else:
    #         self.period = 1/2
    #         return self._periodic_dist(d1, d2, alpha = self.period)    

    # # Periodic distance to rank samples by day of the year
    # def _periodic_dist(self, d, gamma, 
    #                    alpha = 0.5,
    #                    day_to_degree = 360/365, 
    #                    degree_to_rad = np.pi/180):

    #     return np.sin(alpha*degree_to_rad*day_to_degree*(d - gamma))**2
        
    # # Equinoxes - Solstices seasonal distance
    # def _seasonal_equinox_dist(self, d1, d2, 
    #                            gamma = None, 
    #                            thr = 25, 
    #                            scale = 2):
    #     rho = gamma
    #     d_prime_spring = np.absolute(80 - d2)
    #     d_prime_fall = np.absolute(266 - d2)
    #     d_prime = np.min([d_prime_spring, d_prime_fall]) 

    #     if rho is None:
    #         return self._equinox_dist(d1, d2, d_prime, thr)
            
    #     else:
    #         self.window = np.absolute(rho - (d_prime/scale))
    #         return self._equinox_dist(d1, d2 + self.window, d_prime, thr)

    # # Seasonal distance
    # def _seasonal_dist(self, d1, d2, 
    #                    gamma = None):
        
    #     if gamma is None:
    #         return self._periodic_dist(d1, d2)
            
    #     else:
    #         return self._periodic_dist(d1, d2 + gamma)
        
    # kernel based on distance (d_)
    def _kernel(self, r_, length_scale):
        return np.exp(-length_scale*np.sqrt(r_))

    # Filter by similarity distance
    def _similarity_filter(self, w_, kappa):

        # Initialize index
        idx_ = np.arange(w_.shape[0], dtype=int)

        xi  = w_[np.argsort(w_)[::-1]][kappa]
        # Similarity threshold
        idx_neighbors_ = idx_[w_ >= xi]

        # Only apply filter if enough samples
        if idx_neighbors_.shape[0] < kappa:
            return idx_.copy(), True, xi
        else:
            return idx_neighbors_, False, xi

    # Filter by temporal distance
    def _temporal_filter(self, idx_neighbors_, d_temporal_, gamma, kappa):
        if d_temporal_ is None:
            idx_temporal_ = idx_neighbors_.copy()
        else:
            idx_temporal_ = idx_neighbors_[d_temporal_[idx_neighbors_] <= gamma]

        # Only apply filter if enough samples
        if idx_temporal_.shape[0] < kappa:
            return idx_neighbors_.copy(), True
        else:
            return idx_temporal_, False

    def _spatial_metric_filter(self, idx_temporal_, d_spatial_, kappa):
        sigma = None
    
        if idx_temporal_.shape[0] >= kappa:
            idx_sorted = np.argsort(d_spatial_[idx_temporal_])
            idx_spatial_ = idx_temporal_[idx_sorted[:kappa]]
            sigma = d_spatial_[idx_spatial_].max()
        else:
            idx_spatial_ = idx_temporal_.copy()
    
        return idx_spatial_, sigma

    def _spatial_clique_filter(
        self, idx_temporal_, w_, d_spatial_, clique_order, kappa):
        
        sigma = None
        
        # Nodes within clique neighborhood
        idx_clique_ = idx_temporal_[d_spatial_[idx_temporal_] <= clique_order]
    
        if idx_clique_.shape[0] >= kappa:
            idx_sorted = np.argsort(w_[idx_clique_])[::-1]
            return idx_clique_[idx_sorted[:kappa]], sigma
        else:
            idx_spatial_ = idx_temporal_.copy()
            
        return idx_spatial_, sigma

    # δ(d∗)=23.45∘⋅sin(365360∘(284+d∗))
    def _solar_declination(
        self, 
        d, 
        day_to_degree = 360/365, 
        degree_to_rad = np.pi/180
    ):
        return np.sin(degree_to_rad*day_to_degree*(284 + d))

    def _seasonal_distance(
        self,
        d_, 
        d_star
    ):
        r = (self._solar_declination(d_) - self._solar_declination(d_star))**2
        return r/4.

    # Calculate the neighborhood's focal curve
    def _focal_curve(self, M_, w_prime_):
        return M_.T @ w_prime_

    # Fuse neighboring curves with day-ahead forecasts
    def _fuse_curves(
        self, 
        F_tr_, 
        E_tr_, 
        e_ts_, 
        eta_, 
        idx_spatial_, 
        S, 
        t, 
        sigma, 
        kappa, 
        p_fusion,  
        eps = 1e-8
    ):

        # Initialize variables
        M_ = np.zeros((kappa, S))
        F_ = np.zeros((kappa, S))
        E_ = np.zeros((kappa, S))
        m_0_ = np.zeros((kappa, 1))
        
        for i, j in zip(idx_spatial_, range(kappa)):

            # Partial actual and day-ahead curves to fuse
            F_[j, :] = F_tr_[i, t:] 
            E_[j, :] = E_tr_[i, t:] 
            e_ = e_ts_[t:] 

            # Last observations of neighboring curves 
            m_0_[j] = F_tr_[i, t - 1]

            # Probability of fusion rejection
            if self._distances['fusion'] == 'dynamic':
                # Calculate the probability of fusing
                p0 = (np.sum(F_[j, :] == 0) + np.sum(F_[j, :] == 1))/F_[j, :].shape[0]
                pf = p_fusion * (1. - sigma) / np.clip(1. - p0, eps, 1. - eps)

                # Probability of fusion rejection
                if np.random.uniform(0., 1., size=1)[0] < pf:
                    M_[j, :] = F_[j, :]
                else:
                    M_[j, :] = F_[j, :] * (1. - eta_) + E_[j, :] * eta_   
                    #M_[j, :] = F_[j, :] * (1. - eta_) + e_ * eta_   

            else:
                if np.random.uniform(0., 1., size=1)[0] < p_fusion:
                    M_[j, :] = (F_[j, :] * (1. - eta_)) + (E_[j, :] * eta_)
                    #M_[j, :] = F_[j, :] * (1. - eta_) + e_ * eta_
                else:
                    M_[j, :] = F_[j, :]

        return M_, m_0_

    def fit(self, F_, E_, dt_, 
            n_samples_per_hour = 1,
            X_ = None, 
            t_ = None, 
            interval_mask = None):
        
        # Collection of real-time actual curves
        self.F_ = F_
        
        # Collection of forecasted curves
        self.E_ = E_

        # Temporal structure in the curves
        self.dt_ = dt_
        self.T = self.dt_.shape[0]
        self.n_samples_per_hour = n_samples_per_hour
        # Temporal filter to apply to intervals
        if interval_mask is None:
            interval_mask = np.ones(self.dt_.shape, dtype = bool)
        self.interval_mask = interval_mask

        # Curves and asset dates
        self.t_ = t_

        # Curves and asset locations
        self.X_ = X_

    def predict(
        self, 
        f_ = None, 
        e_ = None,
        x_ = None,
        t = None,
        clique_order = None,
        forget_rate_f = 1.,
        forget_rate_e = .5,
        lookahead_rate = 6,
        length_scale_f = 10,
        length_scale_e = 10,
        length_scale_d = 10,
        length_scale_x = 10,
        nu = 6,
        sigma = 0.5,
        kappa_0 = 400,
        kappa = 100,
        p_fusion = 1.,
        normalize_distance = True,
    ):

        # Partially observed curve
        if f_.all() != None:
            self.f_ = f_

        # Forecasted curve
        if e_.all() != None:
            self.e_ = e_

        # Asset location
        if x_.all() != None:
            self.x_ = x_

        # Day of the year
        self.t = t

        # Functional hyperparameters
        self.forget_rate_f = forget_rate_f
        self.forget_rate_e = forget_rate_e
        self.lookahead_rate = lookahead_rate
        self.length_scale_f = length_scale_f
        self.length_scale_e = length_scale_e
        self.length_scale_d = length_scale_d
        self.length_scale_x = length_scale_x
        self.nu = nu
        self.sigma = sigma
        self.p_fusion = p_fusion

        # Spatial distance parameters
        self.clique_order = clique_order
        if self.clique_order != None:
            self.clique_order = int(self.clique_order)

        # Neighborhood hyperparameters
        self.kappa_0 = int(kappa_0)
        self.kappa = int(kappa)

        # Interval
        self.interval = self.f_.shape[0]
        self.S = self.T - self.interval
        
        self.tau_ = self.dt_[:self.interval]
        self.s_ = self.dt_[self.interval:]

        # phi: importance weights based on past time distance
        self.phi_ = self._exponential_growth(
            self.interval, 
            self.forget_rate_f, 
            self.n_samples_per_hour
        )
        
        # Mask intervals
        self.phi_[~self.interval_mask[:self.interval]] = 0.
        
        # psi: importance weights based on past and future
        # time distance
        psi_minus_ = self._exponential_growth(
            self.interval, 
            self.forget_rate_e, 
            self.n_samples_per_hour
        )
        
        psi_plus_ = self._exponential_decay(
            self.S, 
            self.lookahead_rate, 
            self.n_samples_per_hour
        )
        
        self.psi_ = np.concatenate(
            [psi_minus_, psi_plus_], axis = 0
        )

        # Mask intervals
        self.psi_[~self.interval_mask] = 0.
        
        # eta: importance weights based on future time 
        # distance
        self.eta_ = self._linear_inverse_exponential(
            self.s_[::-1], 
            self.T, 
            self.nu, 
            self.sigma, 
            self.n_samples_per_hour
        )

        # d: Euclidean functional similarity distance between
        # samples weighted by importance weights
        self.d_f_ = self._weighted_euclidian_dist(
            self.F_[:, :self.interval], 
            self.f_, 
            w_ = self.phi_,
            normalize=normalize_distance,
        )
        
        self.d_e_ = self._weighted_euclidian_dist(
            self.E_, 
            self.e_, 
            w_ = self.psi_,
            normalize=normalize_distance,
        )
        
       # d: Temporal distance between samples
        if self._distances['temporal'] == 'seasonal':
            
            self.d_d_ = self._seasonal_dist(self.t_, self.t)
            
        elif self._distances['temporal'] == 'seasonal_equinox':
            
            self.d_d_ = self._seasonal_distance(self.t_, self.t)
            
        else:
            
            self.d_d_ = None

        # w: partially observed curve similarity
        self.w_f_ = self._kernel(
            self.d_f_, 
            self.length_scale_f
        )

        # w: DA forecast similarity
        self.w_e_ = self._kernel(
            self.d_e_, 
            self.length_scale_e
        )

        # w: Temporal similarity
        self.w_d_ = self._kernel(
            self.d_d_, 
            self.length_scale_d
        )

        # # Functional Neighborhood
        # self.w_fed_ = np.min(
        #     np.stack([self.w_f_, self.w_e_, self.w_d_]), axis = 0
        # )

        # # Rescale so nearest neighbor = 1 (prevents underflow at large
        # # length scales from deciding the ranking). Guard against total
        # # underflow: if all weights are 0, raise instead of producing NaNs
        # # (caught by the try/except in _run_ffc as a failed process).
        # w_max = self.w_fed_.max()
        # if w_max > 0.:
        #     self.w_fed_ /= w_max
        # else:
        #     raise FloatingPointError(
        #         "w_fed_ underflowed to zero for all candidates "
        #         "(length scales too large for the distance ranges)"
        #     )

        # self.idx_fed_ = np.argsort(self.w_fed_)[::-1][:self.kappa_0]
        
        # Functional Neighborhood — log-weights (identical to min-of-RBF formulation,
        # but immune to underflow at large length scales)
        self.log_w_fed_ = -np.max(
            np.stack([
                self.length_scale_f * np.sqrt(self.d_f_),
                self.length_scale_e * np.sqrt(self.d_e_),
                self.length_scale_d * np.sqrt(self.d_d_),
            ]), axis = 0
        )
        self.arg_fed_ = np.argmax(
            np.stack([
                self.length_scale_f * np.sqrt(self.d_f_),
                self.length_scale_e * np.sqrt(self.d_e_),
                self.length_scale_d * np.sqrt(self.d_d_),
            ]), axis = 0
        )

        # # spatial: Euclidean spatial distance between samples
        # if self._distances['spatial'] == 'euclidean':
            
        #     self.d_x_ = self._weighted_euclidian_dist(
        #         self.X_[self.idx_fed_, :], self.x_
        #     )
            
        # # spatial: Haversine spatial distance between samples
        # elif self._distances['spatial'] == 'haversine':
            
        #     self.d_x_ = self._haversine_dist(
        #         self.X_[self.idx_fed_, :], self.x_
        #     )
            
        # # spatial: Graph spatial distance between samples
        # elif self._distances['spatial'] == 'graph':
        #     self.d_x_ = self._graph_dist(self.X_[self.idx_fed_, 1])
            
        # else:
        #     self.d_x = None

        # # w: Spatial similarity
        # self.w_x_ = self._kernel(
        #     self.d_x_, 
        #     self.length_scale_x
        # )

        # # Spatiotemporal Neighborhood
        # self.w_dx_ = np.min(
        #     np.stack([self.w_d_, self.w_x_]), axis = 0
        # )

        # # Spatially nearest kappa candidates; exact ties in d_x_
        # # (curves from the same asset share one location) are broken
        # # by functional similarity, most similar first.
        # self.idx_x_local_ = np.lexsort(
        #     (-self.w_fed_[self.idx_fed_], self.d_x_)
        # )[:self.kappa]

        # self.idx_x_ = self.idx_fed_[self.idx_x_local_]
        
        # # Normalized the weight of each neighboring curve 
        # self.w_prime_  = self.w_fed_[self.idx_x_]
        # self.w_prime_ /= self.w_fed_[self.idx_x_].sum()
        # #self.w_prime_prime = self.w_[self.idx_spatial_]

        # self.xi = self.w_fed_[self.idx_fed_].min()
        # self.r = self.d_x_[self.idx_x_local_].max()
        # self.t_max = self.t_[self.idx_fed_].max()
        # self.t_min = self.t_[self.idx_fed_].min()

        if (self._distances['spatial'] == 'haversine'):

            # linear-scale weights, rescaled so nearest neighbor = 1 (for inspection/xi)
            self.w_fed_ = np.exp(self.log_w_fed_ - self.log_w_fed_.max())
            self.idx_fed_ = np.argsort(self.log_w_fed_)[::-1][:self.kappa_0]
       
            self.d_x_ = self._haversine_dist(
                self.X_[self.idx_fed_, :], self.x_
            )

            self.idx_x_local_ = np.lexsort(
                (-self.log_w_fed_[self.idx_fed_], self.d_x_)
            )[:self.kappa]
    
            # map local positions (within idx_fed_) back to global sample indices
            self.idx_x_ = self.idx_fed_[self.idx_x_local_]
    
            lw_ = self.log_w_fed_[self.idx_x_]
            self.w_prime_  = np.exp(lw_ - lw_.max())
            self.w_prime_ /= self.w_prime_.sum()
    
            self.r = self.d_x_[self.idx_x_local_].max()
            self.t_max = self.t_[self.idx_fed_].max()
            self.d_max = self.d_d_[self.idx_fed_].max()
            
        elif (self._distances['spatial'] == 'graph'):
            # linear-scale weights, rescaled so nearest neighbor = 1 (for inspection/xi)
            self.w_fed_ = np.exp(self.log_w_fed_ - self.log_w_fed_.max())

            # Stage I (spatial FIRST): graph distance from the target node to every
            # sample's asset; keep only samples within the clique-order threshold
            self.d_x_ = self._graph_dist(self.X_[:, 1])
            idx_clique_ = np.where(self.d_x_ <= self.clique_order)[0]
        
            # Stage II: within the clique, rank by fed similarity and keep kappa_0
            order_ = np.argsort(self.log_w_fed_[idx_clique_])[::-1]
            self.idx_fed_ = idx_clique_[order_[:self.kappa_0]]
        
            # analog set: kappa best by fed weight, ties broken by graph distance
            # (lexsort: last key is primary — flipped w.r.t. the haversine branch,
            # since graph distances are discrete and produce many ties)
            self.idx_x_local_ = np.lexsort(
                (self.d_x_[self.idx_fed_], -self.log_w_fed_[self.idx_fed_])
            )[:self.kappa]
        
            # map local positions (within idx_fed_) back to global sample indices
            self.idx_x_ = self.idx_fed_[self.idx_x_local_]
        
            lw_ = self.log_w_fed_[self.idx_x_]
            self.w_prime_  = np.exp(lw_ - lw_.max())
            self.w_prime_ /= self.w_prime_.sum()
            # diagnostics, mirroring the haversine branch
            # self.xi    = self.log_w_fed_[self.idx_fed_].min() - self.log_w_fed_.max()
            self.r     = self.d_x_[self.idx_x_].max()          # max clique order used
            self.t_max = self.t_[self.idx_fed_].max()
            self.d_max = self.d_d_[self.idx_fed_].max()

        self.ess = 1.0 / np.sum(self.w_prime_ ** 2)
        self.xi = np.exp(self.log_w_fed_[self.idx_fed_].min())   # log relative similarity, 0 = best
        # Fuse neighboring curves with DA forecasts
        self.M_, self.m_0_ = self._fuse_curves(
            self.F_, 
            self.E_, 
            self.e_, 
            self.eta_, 
            self.idx_x_, 
            self.S, 
            self.interval, 
            self.sigma, 
            self.kappa, 
            self.p_fusion
        )

        # Neighborhood focal curve
        self.f_focal_ = self._focal_curve(
            self.M_, 
            self.w_prime_
        )
        
        self.f_0 = self.f_[-1]
        self.f_0_ = np.ones(self.m_0_.shape)*self.f_0
        self.M_ext_ = np.concatenate([self.f_0_, self.M_], axis = 1)
        
        return self.M_

    # ------------------------------------------------------------------
    # Adaptive B-spline basis for the forecast horizon
    # ------------------------------------------------------------------
    def _effective_support(self, n_points):
        """
        Mask and index of the effective support of the extended forecast
        horizon grid.

        The extended horizon grid is t_ = dt_[-(n_points + 1):], i.e., the
        anchor (the last observation, at interval - 1) followed by the
        S = n_points future instants; its mask is the tail of
        self.interval_mask. The anchor always belongs to the support: every
        fused curve is pinned to f_0 there.
        """
        mask_ = np.asarray(
            self.interval_mask[-(n_points + 1):], dtype = bool
        ).copy()
        mask_[0] = True

        idx_ = np.flatnonzero(mask_)

        # Degenerate mask (no active instant in the horizon): fall back to
        # the full grid so the expansion is always well posed
        if idx_.shape[0] < 2:
            mask_ = np.ones(n_points + 1, dtype = bool)
            idx_ = np.flatnonzero(mask_)

        return mask_, np.arange(idx_[0], idx_[-1] + 1)

    def _effective_horizon(self, mask_, n_points, mask_mode = 'horizon'):
        """
        Number of active samples in the extended forecast horizon, S_eff.

        Two equivalent-for-wind definitions:

        'horizon' (default) counts the active instants of the horizon
        directly, S_eff = interval_mask[interval - 1:].sum(). It is exact for
        any mask.

        'day' uses the closed form implied by the masked-out fraction of the
        horizon, omega = (T - interval_mask.sum()) / (T - interval),

            S_eff = (T - interval) (1 - omega) + 1
                  = interval_mask.sum() - interval + 1,

        a function of interval_mask.sum(), T and interval alone. It is exact
        when the active window starts at the beginning of the day -- always
        true for wind, where omega = 0 and S_eff = T - interval + 1 -- but it
        charges the horizon with the *whole day's* inactive samples, so for
        solar it understates the remaining daylight and can even turn
        negative late in the day. It is kept for reference and clamped to at
        least one sample.
        """
        if mask_mode == 'day':
            S_eff = int(self.interval_mask.sum()) - (self.T - n_points) + 1
        else:
            S_eff = int(mask_.sum())

        return max(1, S_eff)

    def _adaptive_basis(
        self,
        t_,
        n_points,
        basis_per_hour = 4./3.,
        order = 4,
        n_basis = None,
        min_basis = None,
        max_basis = None,
        mask_mode = 'horizon',
        eps = 1e-9,
    ):
        """
        B-spline basis whose degrees of freedom and knot placement adapt to
        the forecast update time and to the interval mask.

        The number of knot spans is proportional to the *effective* duration
        of the horizon -- the active samples S_eff of the extended horizon
        over the sampling rate -- so that the number of spans per active hour,

            basis_per_hour = (n_basis - order + 1) n_samples_per_hour / S_eff,

        and hence the knot spacing and the smoothing resolution, is invariant
        to the update time and to the energy feature:

            m       = ceil(basis_per_hour * S_eff / n_samples_per_hour)
            n_basis = m + order - 1.

        With the default basis_per_hour = 4/3 (a 45-minute knot span) a wind
        asset at interval = 144 (12 h of horizon at 12 samples/h) gives
        n_basis = 20, reproducing the previously hard-coded value, while
        interval = 216 (6 h) gives 12 and interval = 72 (18 h) gives 28 --
        the same degrees of freedom per unit of active time in all three
        cases, instead of a knot spacing that ranged from 21 to 64 minutes.

        Knots sit at equally spaced quantiles of the cumulative count of
        active instants inside the support, so they are uniform when the mask
        is inactive (wind) and concentrate on daylight when it is (solar),
        rather than spending degrees of freedom on the flat night.

        Passing n_basis overrides the count but keeps the mask-aware knots.
        """
        order = int(order)

        mask_, idx_support_ = self._effective_support(n_points)
        m_support_ = mask_[idx_support_]
        n_support = idx_support_.shape[0]

        S_eff = self._effective_horizon(
            mask_, n_points, mask_mode = mask_mode
        )
        hours_eff = S_eff / self.n_samples_per_hour

        # Number of knot spans
        if n_basis is None:
            m = int(np.ceil(basis_per_hour*hours_eff))
        else:
            m = int(n_basis) - order + 1

        if min_basis is not None:
            m = max(m, int(min_basis) - order + 1)
        if max_basis is not None:
            m = min(m, int(max_basis) - order + 1)

        # At least one span, and no more coefficients than support points
        m = int(np.clip(m, 1, max(1, n_support - order + 1)))

        # Knots at equally spaced quantiles of active time
        t_support_ = t_[idx_support_]
        u_ = np.cumsum(m_support_.astype(float))
        # Strictly increasing so the quantile map is well defined; the ramp
        # is negligible, so no knot is placed inside an inactive run
        u_ = u_ + eps*np.arange(u_.shape[0])

        knots_ = np.interp(
            np.linspace(u_[0], u_[-1], m + 1), u_, t_support_
        )
        knots_[0] = t_support_[0]
        knots_[-1] = t_support_[-1]

        # Fall back to a uniform grid if the quantile map collapsed
        if np.any(np.diff(knots_) <= 0.):
            knots_ = np.linspace(t_support_[0], t_support_[-1], m + 1)

        _basis = BSpline(knots = list(knots_), order = order)

        # Diagnostics
        self.idx_support_ = idx_support_
        self.knots_ = knots_
        self.n_basis = _basis.n_basis
        self.S_eff = S_eff
        self.hours_eff = hours_eff
        self.knot_spacing = hours_eff/m
        self.basis_per_active_hour = m/hours_eff

        return _basis, idx_support_

    # Downsample collection of curves
    def functional_downsampling(
        self,
        subsample,
        n_basis = None,
        basis_per_hour = 4./3.,
        order = 4,
        min_basis = None,
        max_basis = None,
        mask_mode = 'horizon'
    ):
        """
        Smooth the fused curves on an adaptive B-spline basis and re-evaluate
        them on the original and on a downsampled grid.

        n_basis defaults to None: the number of basis functions follows from
        the length of the horizon and from the interval mask through
        _adaptive_basis, so the knot spacing -- the smoothing resolution -- is
        the same at every forecast update time and for both energy features.
        Passing an integer restores a fixed number of degrees of freedom,
        still with mask-aware knots.
        """

        dt_ = self.dt_
        M_ = self.M_

        n_samples, n_points = M_.shape

        f_0_ = np.ones(self.m_0_.shape)*self.f_0
        M_ext_ = np.concatenate([f_0_, M_], axis = 1)

        # Extended horizon grid and downsampled grid
        t_ = dt_[-n_points-1:]
        t_ds_ = np.linspace(t_[1], t_[-1], int(n_points/subsample))

        # Basis adapted to the interval and to the interval mask
        _basis, idx_support_ = self._adaptive_basis(
            t_,
            n_points,
            basis_per_hour = basis_per_hour,
            order = order,
            n_basis = n_basis,
            min_basis = min_basis,
            max_basis = max_basis,
            mask_mode = mask_mode,
        )

        t_support_ = t_[idx_support_]

        # Expansion on the effective support
        _fd = FDataGrid(
            data_matrix = [M_ext_[i, idx_support_] for i in range(n_samples)],
            grid_points = t_support_
        )
        _fd_int = _fd.to_basis(_basis)

        # Re-evaluate on the original grid; instants outside the effective
        # support keep their observed value (the flat night, for solar)
        M_int_ = M_ext_.copy()
        M_int_support_ = _fd_int.to_grid(t_support_)
        M_int_[:, idx_support_] = np.stack(
            [M_int_support_.data_matrix[i] for i in range(n_samples)]
        )[..., 0]
        self.M_int_ = M_int_

        # Re-evaluate on the downsampled grid
        in_support_ = (t_ds_ >= t_support_[0]) & (t_ds_ <= t_support_[-1])
        M_int_ds_ = np.stack([
            np.interp(t_ds_, t_, M_ext_[i, :]) for i in range(n_samples)
        ])

        if in_support_.any():
            M_int_ds_support_ = _fd_int.to_grid(t_ds_[in_support_])
            M_int_ds_[:, in_support_] = np.stack(
                [M_int_ds_support_.data_matrix[i] for i in range(n_samples)]
            )[..., 0]

        self.M_int_ds_ = M_int_ds_

        return self.M_int_, self.M_int_ds_
    
    # def focal_curve_envelope(
    #         self,
    #         _depth,
    #         X_,
    #         dist,
    #         max_iter = 100,
    #         idx_focal = 0
    # ):
    #     """
    #     Envelope algorithm to obtain functional neighborhoods.
    
    #     Parameters
    #     ----------
    #     data: dict
    #         Dictionary with keys:
    #             - "x": np.ndarray of grid points (n_points,)
    #             - "y": np.ndarray of function values (n_points, n_curves)
    #     focal: int or str
    #         Index (or column name) of the focal curve to envelope.
    #     plot : bool, optional
    #         Whether to plot the selected curves in each iteration.
    #     max_iter: int, optional
    #         Maximum number of iterations before stopping.
    
    #     Returns
    #     -------
    #     dict
    #         Dictionary with key 'Jordered' containing the ordered list
    #         of selected curve indices.
    #     """

    #     self.dist = dist

    #     self.f_focal_ext_ = np.insert(self.f_focal_, 0, self.f_0)

    #     X_ext_ = np.concatenate([
    #         self.f_focal_ext_[np.newaxis, :], X_], axis = 0)[:, 1:].T
        
    #     dt_ = self.dt_[-X_ext_.shape[0]:]

    #     if self.dist == 'fknn':
    #         #dist_ = np.insert(self.w_prime_, 0, 0)
    #         d_fknn_ = -self.log_w_fed_[self.idx_x_]     # >= 0, smaller = more similar
    #         dist_   = np.insert(d_fknn_, 0, 0.)
    #     else:
    #         dist_ = self.dist

        
    #     # Compute depth to find the focal curve
    #     _fd_filtered = FDataGrid(
    #         data_matrix = X_ext_.T,
    #         grid_points = dt_
    #     )
        
    #     #filtered_depth_ = _depth(_fd_filtered)
    #     # idx_focal       = np.argsort(-filtered_depth_)[0]
    #     f_ = X_ext_[:, idx_focal]
    
    #     if isinstance(dist_, str):
    #         # Distances from curves to the focal curve
    #         if dist_ == 'sup':
    #             dist_ = np.max(np.abs(X_ext_.T - f_), axis = 1)
    #         elif dist_ == 'l2':
    #             dist_ = np.sum((X_ext_.T - f_)**2, axis = 1)
                
    #     # Initialize
    #     idx_subsample = []
    #     idx_          = [i for i in range(X_ext_.shape[1]) if i != idx_focal]
    #     iter_depth    = [0]
    #     iteration     = 0
    
    #     while len(idx_) > 1:
            
    #         # New iteration
    #         iteration += 1
            
    #         # Sort curves by distance
    #         idx_sorted_dist_   = [idx_[i] for i in np.argsort(dist_[idx_])]
    #         idx_iter_subsample = [idx_sorted_dist_[0]]
    #         idx_candidates     = idx_sorted_dist_[1:]
    #         # Iterative envelope selection
    #         remaining_points = set(dt_)
    #         while remaining_points and idx_candidates:
    #             idx_next = idx_candidates[0]
    #             combined = idx_iter_subsample + [idx_next]
                
    #             # Check if envelopes
    #             sign_ = np.sign(X_ext_[:, combined].T - f_)
    #             Ji_   = np.where(np.abs(np.sum(sign_, axis = 0)) < len(combined))[0]
                
    #             if len(remaining_points - set(dt_[Ji_]) ) == len(remaining_points):
    #                 # Does not envelope
    #                 idx_candidates.pop(0)
    #             else:
    #                 remaining_points -= set(dt_[Ji_])
                    
    #                 idx_iter_subsample.append(idx_next)
                    
    #                 idx_candidates = [c for c in idx_candidates if c not in idx_iter_subsample]
    #                 idx_           = [c for c in idx_ if c not in idx_iter_subsample]
    
    #         # Compute functional depth 
    #         _fd_subset = FDataGrid(
    #             data_matrix = (X_ext_[:, [idx_focal] + idx_subsample  + idx_iter_subsample]).T,
    #             grid_points = dt_
    #         )
        
    #         depth_             = _depth(_fd_subset)
    #         idx_depth_         = np.argsort(-depth_)
    #         idx_ordered_depth_ = np.array(
    #             [idx_focal] + idx_subsample + idx_iter_subsample
    #         )[idx_depth_]
    
    #         # How deep is the new set of curves?
    #         depth_percentile = 1 - np.where(idx_ordered_depth_ == idx_focal)[0][0]/(len(idx_ordered_depth_) - 1)
            
    #         iter_depth.append(depth_percentile)
    
    #         # Accept subsample if depth improves
    #         if max(iter_depth[:-1]) <= iter_depth[-1]:
    #             idx_subsample.extend(idx_iter_subsample)
                
    #         # Stop if there are no more candidate curves
    #         if not idx_candidates:
    #             break
                
    #         # Stop if max_iter is reached
    #         if iteration >= max_iter:
    #             break
    
    #     # Selected curves
    #     idx_sel_ = [idx_focal] + idx_subsample
        
    #     # Final selected curves ordered by depth
    #     _fd_filtered = FDataGrid(
    #         data_matrix = X_ext_[:, idx_sel_].T,
    #         grid_points = dt_
    #     )
            
    #     filtered_depth_             = _depth(_fd_filtered)
    #     idx_filtered_depth_         = np.argsort(-filtered_depth_)
    #     idx_ordered_filtered_depth_ = np.array(idx_sel_)[idx_filtered_depth_]
        
    #     # Select curves for the focal curve envelop
    #     self.J_ = X_ext_[:, idx_ordered_filtered_depth_].T
        
    #     return self.J_

    def focal_curve_envelope(
            self,
            _depth,
            X_,
            dist,
            n_layers = None,
            max_iter = 100,
            idx_focal = 0
    ):
        """
        Envelope algorithm to obtain functional neighborhoods.
    
        A layer is a minimal subset of curves that brackets the focal curve at
        every grid point. After `n_layers` accepted layers, at every grid point at
        least `n_layers` curves lie above and at least `n_layers` below the focal
        curve, so `n_layers` acts as the band's sharpness level.
    
        `n_layers` replaces the previous depth-percentile acceptance rule, which
        could never reject: self.f_focal_ is a weighted mean of the curves in X_,
        hence always the deepest curve of any subset containing it, so the
        percentile was pinned at 1.0 and every layer was accepted. The loop then
        ran to exhaustion and J_ contained the whole pool, making this method
        equivalent to depth-based trimming.
    
        Parameters
        ----------
        _depth: callable or None
            Functional depth used to order the selected curves. Pass None to skip
            the ordering (valid when the caller takes k = 1, i.e. min/max over all
            of J_).
        X_: np.ndarray, (n_curves, n_points + 1)
            Curves to select from, first column the common anchor f_0.
        dist: str or np.ndarray
            'fknn', 'sup', 'l2', or a precomputed distance vector.
        n_layers: int, optional
            Number of enveloping layers to accept. None means max_iter.
        max_iter: int, optional
            Hard cap on iterations.
        idx_focal: int, optional
            Column of X_ext_ holding the focal curve.
    
        Returns
        -------
        np.ndarray, (n_selected, n_points)
            Selected curves, ordered by decreasing depth when _depth is given.
        """
        self.dist = dist
        self.f_focal_ext_ = np.insert(self.f_focal_, 0, self.f_0)
        X_ext_ = np.concatenate([
            self.f_focal_ext_[np.newaxis, :], X_], axis = 0)[:, 1:].T
    
        dt_ = self.dt_[-X_ext_.shape[0]:]
        f_  = X_ext_[:, idx_focal]
    
        if self.dist == 'fknn':
            dist_ = np.insert(-self.w_prime_, 0, 0)
            #d_fknn_ = -self.log_w_fed_[self.idx_x_]     # >= 0, smaller = more similar
            #dist_   = np.insert(d_fknn_, 0, 0.)
        else:
            dist_ = self.dist
    
        if isinstance(dist_, str):
            # Distances from curves to the focal curve
            if dist_ == 'sup':
                dist_ = np.max(np.abs(X_ext_.T - f_), axis = 1)
            elif dist_ == 'l2':
                dist_ = np.sum((X_ext_.T - f_)**2, axis = 1)
    
        # Initialize
        idx_subsample = []
        idx_          = [i for i in range(X_ext_.shape[1]) if i != idx_focal]
        iteration     = 0
    
        # Number of enveloping layers to peel off
        n_layers = max_iter if n_layers is None else int(n_layers)
        self.layer_sizes_ = []
    
        while len(idx_) > 1:
    
            # New iteration
            iteration += 1
    
            # Sort curves by distance
            idx_sorted_dist_   = [idx_[i] for i in np.argsort(dist_[idx_])]
            idx_iter_subsample = [idx_sorted_dist_[0]]
            idx_candidates     = idx_sorted_dist_[1:]
    
            # Iterative envelope selection
            remaining_points = set(dt_)
            while remaining_points and idx_candidates:
                idx_next = idx_candidates[0]

                combined = idx_iter_subsample + [idx_next]
    
                # Check if envelopes
                sign_ = np.sign(X_ext_[:, combined].T - f_)
                Ji_   = np.where(np.abs(np.sum(sign_, axis = 0)) < len(combined))[0]
    
                if len(remaining_points - set(dt_[Ji_])) == len(remaining_points):
                    # Does not envelope
                    idx_candidates.pop(0)
                else:
                    remaining_points -= set(dt_[Ji_])
    
                    idx_iter_subsample.append(idx_next)
    
                    idx_candidates = [c for c in idx_candidates if c not in idx_iter_subsample]
                    idx_           = [c for c in idx_ if c not in idx_iter_subsample]
    
            # The layer did not close: the curves left cannot bracket the focal
            # curve at every grid point, so drop the partial layer and stop
            if remaining_points:
                break
    
            # Accept the layer
            idx_subsample.extend(idx_iter_subsample)
            self.layer_sizes_.append(len(idx_iter_subsample))
    
            # Stop once n_layers layers have been accepted
            if iteration >= n_layers:
                break
    
            # Stop if there are no more candidate curves
            if not idx_candidates:
                break
    
            # Stop if max_iter is reached
            if iteration >= max_iter:
                break
    
        # Selected curves
        idx_sel_ = [idx_focal] + idx_subsample

        self.J_ = X_ext_[:, idx_sel_].T
        if _depth is not None:
            # Final selected curves ordered by depth
            _fd_filtered = FDataGrid(
                data_matrix = self.J_,
                grid_points = dt_,
            )
    
            filtered_depth_ = _depth(_fd_filtered)
            self.J_ = self.J_[np.argsort(-filtered_depth_), :]

        return self.J_
        
    # Confidence bands from focal-curve envelop
    #def _focal_envelop_confidence_bands(J_, alpha_, k_, f_0):    
    def focal_envelope_confidence_region(
        self,
        alpha_,
        k_
    ):
    
        self._upper_envelop_confidence_bands = {}
        self._lower_envelop_confidence_bands = {}
        for i in range(len(alpha_)):

            N = min(self.J_.shape[0], max(2, int(k_[i]*self.J_.shape[0])))

            self._upper_envelop_confidence_bands[f'{alpha_[i]}'] = np.insert(
                np.max(self.J_[:N, :], axis = 0),
                0,
                self.f_0
            )
            self._lower_envelop_confidence_bands[f'{alpha_[i]}'] = np.insert(
                np.min(self.J_[:N, :], axis = 0),
                0,
                self.f_0
            )
            
        return (self.f_focal_ext_, 
                self._upper_envelop_confidence_bands, 
                self._lower_envelop_confidence_bands)
        
    # Functional boxplot from a smooth functional depth metric
    def functional_boxplot(self, X_, depth_score_, alpha_ = [0.25, 0.5, 0.75]):
    
        idx_ = np.argsort(depth_score_)[::-1]
    
        self._upper_functional_boxplot = {}
        self._lower_functional_boxplot = {}

        for i in range(len(alpha_)):
            X_sel_ = X_[idx_[:-int(X_.shape[0] * alpha_[i])],]

            #X_sel_ = X_[idx_[:min(X_.shape[0], max(2, int(X_.shape[0]*(1. - alpha_[i]))))], ]

            self._upper_functional_boxplot[f'{alpha_[i]}'] = np.max(X_sel_, axis = 0)
            self._lower_functional_boxplot[f'{alpha_[i]}'] = np.min(X_sel_, axis = 0)
    
            self._upper_functional_boxplot[f'{alpha_[i]}'][
                self._lower_functional_boxplot[f'{alpha_[i]}'] > 1
            ] = 1

            self._lower_functional_boxplot[f'{alpha_[i]}'][
                self._lower_functional_boxplot[f'{alpha_[i]}'] < 0
            ] = 0
            
        self._upper_functional_boxplot['max'] = np.max(X_, axis = 0)
        self._lower_functional_boxplot['min'] = np.min(X_, axis = 0)

        self.f_deepest_ = X_[idx_[0],]

        # Whiskers: envelope of the curves that stay inside the fences, the
        # central region inflated by factor x its pointwise width
        alpha_whisker = 0.5
        factor = 1.5 
        atol = 1e-3
        X_sel_ = X_[idx_[:-int(X_.shape[0] * alpha_whisker)],]
        u_ = np.max(X_sel_, axis = 0)
        l_ = np.min(X_sel_, axis = 0)
        w_ = factor * (u_ - l_)

        out_ = np.any((X_ > u_ + w_ + atol) | (X_ < l_ - w_ - atol), axis = 1)
        X_in_ = X_ if out_.all() else X_[~out_,]

        self._upper_functional_boxplot['whisker'] = np.max(X_in_, axis = 0)
        self._lower_functional_boxplot['whisker'] = np.min(X_in_, axis = 0)
        self.idx_outliers_ = np.flatnonzero(out_)
        
        return (self.f_deepest_, 
                self._upper_functional_boxplot, 
                self._lower_functional_boxplot)
        

    def _eQuantile(self, _ECDF, q_):
        """
        Calculates quantiles from an ECDF.
    
        Args:
        _ECDF: function from statsmodels api
        q_: A list or numpy array of quantiles to calculate (values between 0 and 1).
    
        Returns:
        q_: A dictionary where keys are the input quantiles and values are the corresponding
        Quantile values from the ECDF.
        """
    
        return np.array([_ECDF.x[np.searchsorted(_ECDF.y, q)] for q in q_])
        
            
    # Derive confidence bands from a functional depth metric
    def ecdf_confidence_bands(self, X_, alpha_):    

        self._upper_ecdf_confidence_bands = {}
        self._lower_ecdf_confidence_bands = {}
        for i in range(len(alpha_)):

            self._upper_ecdf_confidence_bands[f'{alpha_[i]}'] = np.stack([
                self._eQuantile(ECDF(X_[:, j]), [1. - alpha_[i]/2.])
                for j in range(X_.shape[1])
            ])[:, 0]
            
            self._lower_ecdf_confidence_bands[f'{alpha_[i]}'] = np.stack([
                self._eQuantile(ECDF(X_[:, j]), [alpha_[i]/2.])
                for j in range(X_.shape[1])
            ])[:, 0]
            
    
        self.f_median_ext_ = np.median(X_, axis = 0)
    
        return self.f_median_ext_, self._upper_ecdf_confidence_bands, self._lower_ecdf_confidence_bands

    # def _eval_depth(self, _depth, X_):
        
    #        # Compute functional depth
    #     _F = FDataGrid(
    #         data_matrix = X_,
    #         grid_points = self.dt_[-X_.shape[1]:]
    #     )
        
    #     score_ = _depth(_F)
    #     rank_ = np.argsort(score_)[::-1]
    #     return score_, rank_

    # Confidence bands from depth function
    def get_depth(
            self,
            _depth,
            X_
    ):
    
        # Compute functional depth
        _F = FDataGrid(
            data_matrix = X_,
            grid_points = self.dt_[-X_.shape[1]:]
        )
        
        depth_score_ = _depth(_F)

        return depth_score_, np.argsort(depth_score_)[::-1]
        
    # Confidence bands from depth function
    def functional_confidence_region(
            self,
            _depth,
            X_,
            alpha_,
    ):
    
        # Compute functional depth
        _F = FDataGrid(
            data_matrix = X_,
            grid_points = self.dt_[-X_.shape[1]:]
        )
        
        depth_score_ = _depth(_F)
        idx_ = np.argsort(depth_score_)[::-1]

        self._upper_depth_confidence_bands = {}
        self._lower_depth_confidence_bands = {}
        for i in range(len(alpha_)):
            N = int(X_.shape[0]*(1. - alpha_[i]))
            self._upper_depth_confidence_bands[f'{alpha_[i]}'] = np.max(X_[idx_[:N], :], axis = 0)
            self._lower_depth_confidence_bands [f'{alpha_[i]}'] = np.min(X_[idx_[:N], :], axis = 0)
        
        # Deepest curve
        self.f_deepest_ext_ = X_[idx_[0], :]
    
        return self.f_deepest_ext_, self._upper_depth_confidence_bands, self._lower_depth_confidence_bands 

    # Confidence bands from depth function
    def adjusted_functional_confidence_region(
            self,
            _depth,
            X_,
            alpha_,
            k_
    ):

        # Compute functional depth
        _F = FDataGrid(
            data_matrix = X_,
            grid_points = self.dt_[-X_.shape[1]:]
        )
        
        depth_score_ = _depth(_F)
        idx_ = np.argsort(depth_score_)[::-1]
        
        self._upper_depth_confidence_bands = {}
        self._lower_depth_confidence_bands = {}
        for i in range(len(alpha_)):
            N = min(X_.shape[0], max(2, int(k_[i]*X_.shape[0])))
            self._upper_depth_confidence_bands[f'{alpha_[i]}'] = np.max(X_[idx_[:N], :], axis = 0)
            self._lower_depth_confidence_bands [f'{alpha_[i]}'] = np.min(X_[idx_[:N], :], axis = 0)
        
        # Deepest curve
        self.f_deepest_ext_ = X_[idx_[0], :]
    
        return self.f_deepest_ext_, self._upper_depth_confidence_bands, self._lower_depth_confidence_bands 

    def _weighted_std(
            self,
            x,
            weights
    ):
        """Weighted standard deviation."""
        x = np.asarray(x)
        weights = np.asarray(weights)
        
        w_mean = np.average(x, weights=weights)
        variance = np.average((x - w_mean) ** 2, weights=weights)
        return np.sqrt(variance)

    def _weighted_quantile(
            self,
            x,
            quantiles,
            weights
    ):
        """Weighted quantile(s), e.g. quantiles=[0.25, 0.75] for Q1/Q3."""
        x = np.asarray(x)
        weights = np.asarray(weights)
        sorter = np.argsort(x)
    
        x_sorted = x[sorter]
        w_sorted = weights[sorter]
    
        cum_weights = np.cumsum(w_sorted) - 0.5 * w_sorted
        cum_weights /= np.sum(w_sorted)
    
        return np.interp(quantiles, cum_weights, x_sorted)
    
    def _weighted_silverman_bandwidth(
            self,
            x,
            weights
    ):
        """Silverman's Rule of Thumb with sample weights."""
        x = np.asarray(x)
        weights = np.asarray(weights)

        n_eff = (weights.sum() ** 2) / np.sum(weights ** 2)  # effective sample size
        sigma = self._weighted_std(x, weights)
        q1, q3 = self._weighted_quantile(x, [0.25, 0.75], weights)
        iqr = q3 - q1
    
        A = min(sigma, iqr / 1.34)
        h = 0.9 * A * n_eff ** (-1 / 5)
        return h

    # Derive confidence bands from a functional depth metric
    def weighted_ecdf_confidence_region(
            self,
            X_,
            weights_,
            alpha_
    ):

        self._upper_wecdf_confidence_bands = {}
        self._lower_wecdf_confidence_bands = {}

        weights_ = np.asarray(weights_, dtype = float)/np.sum(weights_)

        for i in range(len(alpha_)):

            self._upper_wecdf_confidence_bands[f'{alpha_[i]}'] = np.stack([
                self._weighted_quantile(X_[:, j], [1. - alpha_[i]/2.], weights_)
                for j in range(X_.shape[1])
            ])[:, 0]

            self._lower_wecdf_confidence_bands[f'{alpha_[i]}'] = np.stack([
                self._weighted_quantile(X_[:, j], [alpha_[i]/2.], weights_)
                for j in range(X_.shape[1])
            ])[:, 0]
            
        self.f_wmedian_ext_ =  np.stack([
            self._weighted_quantile(X_[:, j], [0.5], weights_)
            for j in range(X_.shape[1])
        ])[:, 0]       
                
        return (self.f_wmedian_ext_, 
                self._upper_wecdf_confidence_bands, 
                self._lower_wecdf_confidence_bands)
        
    # Silverman's Rule
    def _silverman_bandwidth(self, x_):
        IQR = np.percentile(x_, 75) - np.percentile(x_, 25)
        return 0.9 * min(np.std(x_), IQR / 1.34) * x_.shape[0] ** (-1 / 5)

    def kernel_density_estimation(
        self, 
        M_, 
        algorithm = "auto", 
        kernel = "gaussian"
    ):
        _KDS = []
        for i in range(M_.shape[1]):
            _KD = KernelDensity(
                bandwidth = self._silverman_bandwidth(M_[:, i]), 
                algorithm = algorithm, 
                kernel = kernel
            ).fit(M_[:, i][:, np.newaxis])
            
            _KDS.append(_KD)

        return _KDS
        
    def weighted_kernel_density_estimation(
        self, 
        M_, 
        weights = 'None',
        algorithm = "auto", 
        kernel = "gaussian"
    ):
        
        _KDS = []
        #weights /= weights.sum()
        for i in range(M_.shape[1]):
            h = self._weighted_silverman_bandwidth(M_[:, i], weights)
            _KD = KernelDensity(
                bandwidth = h + 0.01, 
                algorithm = algorithm, 
                kernel = kernel
            ).fit(M_[:, i][:, np.newaxis], sample_weight = weights)
            
            _KDS.append(_KD)

        return _KDS
