import numpy as np

from sklearn.neighbors import KernelDensity

from skfda.representation.basis import Fourier, BSpline
from skfda import FDataGrid

from statsmodels.distributions.empirical_distribution import ECDF

class functional_dynamic_update:
    
    def __init__(self, 
                 _distances = {'spatial': 'None', 'temporal': 'None'},
                 name = 'noname', 
                 date = 'nodate'):

        self._distances = _distances
        self.name = name
        self.date = date

    # Define exponential growth function
    def _exponential_growth(self, t, growth_rate):
        tau_ = np.linspace(t - 1, 0, t)
        return np.exp(np.log(0.5)*tau_/(growth_rate*12))

    # Define exponential decay function
    def _exponential_decay(self, S, decay_rate):
        s_ = np.linspace(0, S - 1, S)
        return np.exp(np.log(0.5)*s_/(decay_rate*12))    
    
    # def _logistic(x_, k):
    #     return 1. - 1.0 / (1.0 + np.exp(np.log(999) * x_ / (k*60/2)))

    # Linear Inverse Exponential (LIE) function
    def _linear_inverse_exponential(self, x_, T, nu, trust_rate, 
                                    k = 2.5, 
                                    alpha = 1.):
        
        x_ = x_ - T*5 + nu*5*12
        x_ = k*x_/(nu*5*12 - 5)
        y_ = np.where(x_ > 0, -x_, -alpha*(np.exp(x_) - 1))
        y_ = (y_ + k)/(k + alpha)
        return trust_rate*y_

    # Calculate weighted (w_) distance between X_ and x_
    def _weighted_euclidian_dist(self, X_, x_, w_ = []):
        if len(w_) == 0:
            w_ = np.ones(x_.shape)
        w_ = w_ / w_.sum()
        d_ = np.zeros((X_.shape[0],))
        for i in range(X_.shape[0]):
            d_[i] = w_.T @ (X_[i, :] - x_) ** 2
        return d_
    
    def _haversine_dist(self, x_1_, x_2_):
        """
        Calculate the distance between two points on Earth using the Haversine formula.
    
        Args:
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
        
    # # Equinoxes - Soltices seasonal distance
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

  
    # --- seasonal membership ---
    # Soltices day of the year: 172 (summer), and 355 (winter)
    # Equinox day of the year: 80 (spring), and 266 (fall)
    def _equinox_dist(self, d1, d2, d_prime, thr):

        if d_prime < thr:
            return self._periodic_dist(d1, d2, alpha = 1)
        else:
            return self._periodic_dist(d1, d2, alpha = 0.5)    

    # Periodic distance to rank samples by day of the year
    def _periodic_dist(self, d, gamma, 
                       alpha = 0.5,
                       day_to_degree = 360/365, 
                       degree_to_rad = np.pi/180):

        return np.sin(alpha*degree_to_rad*day_to_degree*(d - gamma))**2
        
    # Equinoxes - Soltices seasonal distance
    def _seasonal_equinox_dist(self, d1, d2, 
                               gamma = None, 
                               thr = 25, 
                               scale = 2):
        
        d_prime_spring = np.absolute(80 - d2)
        d_prime_fall   = np.absolute(266 - d2)
        d_prime        = np.min([d_prime_spring, d_prime_fall]) 

        if gamma is None:
            return self._equinox_dist(d1, d2, d_prime, thr)
            
        else:
            window = np.absolute(gamma - (d_prime/scale))
            return self._equinox_dist(d1, d2 + window, d_prime, thr)

    # Seasonal distance
    def _seasonal_dist(self, d1, d2, 
                       gamma = None):
        
        if gamma is None:
            return self._periodic_dist(d1, d2)
            
        else:
            return self._periodic_dist(d1, d2 + gamma)
        
    # Radial Basis function kernel based on distance (d_)
    def _rbf_kernel(self, d_, length_scale):
        return np.exp(-d_ / length_scale)
        
    # # Filtering neighboring curves above a threshold
    # def _neighborhood_filtering(self, w_, d_spatial_, d_temporal_, xi, gamma, clique_order, kappa):
    
    #     sigma = None
    #     # Filter by similarity
    #     idx_ = np.arange(w_.shape[0], dtype = int)
    #     idx_neighbors_ = idx_[w_ >= xi]

    #     if d_temporal_ is None:
    #         idx_temporal_ = idx_neighbors_.copy()
    #     else:
    #         idx_temporal_ = idx_[idx_neighbors_][d_temporal_[idx_neighbors_] <= gamma]
            
    #     if (idx_temporal_.shape[0] < kappa) or (d_spatial_ is None):  
    #         # Increase similarity threshold 
    #         idx_spatial_ = idx_[w_ >= np.sort(w_)[::-1][kappa]][:kappa]
    
    #     else: 
    #         # Rank neighbors by Haversine distance
    #         idx_spatial_rank_ = np.argsort(d_spatial_[idx_temporal_])
    
    #         # Select the kappa_max closest neibors
    #         idx_spatial_ = idx_temporal_[idx_spatial_rank_][:kappa]
    
    #         # What is the distance threshold?
    #         sigma = d_h_[idx_spatial_].max()
    
    #     return idx_neighbors_, idx_temporal_, idx_spatial_, sigma

    # # Filtering neighboring curves above a threshold
    # def _neighborhood_filtering(self, w_, d_spatial_, d_temporal_, xi, clique_order, gamma, kappa):
    
    #     sigma = None
    #     idx_ = np.arange(w_.shape[0], dtype=int)
    
    #     # 1. Probability filter
    #     idx_neighbors_ = idx_[w_ >= xi]

    #     # 2. Temporal filter
    #     if d_temporal_ is not None:
    #         #print('Temporal filer...')
    #         idx_temporal_ = idx_neighbors_[d_temporal_[idx_neighbors_] <= gamma]
    #     else:
    #         idx_temporal_ = idx_neighbors_.copy()
    #     #print(idx_temporal_.shape)
    #     # 3. Spatial / graph filtering
    #     if d_spatial_ is not None:
    #         #print('Spatial filer...')
    #         if clique_order is None:
    #             #print('Metric filer...')
    #             # --- Metric space (continuous distance) ---
    #             if idx_temporal_.shape[0] >= kappa:
    #                 idx_sorted = np.argsort(d_spatial_[idx_temporal_])
    #                 idx_spatial_ = idx_temporal_[idx_sorted[:kappa]]
    #                 sigma = d_spatial_[idx_spatial_].max()
    #             else:
    #                 idx_spatial_ = idx_temporal_.copy()
    #         else:
    #             #print('Clique filer...')
    #             # --- Graph space (clique neighborhood) ---
    #             # keep only nodes within clique_order hops
    #             idx_clique_ = idx_temporal_[d_spatial_[idx_temporal_] <= clique_order]

    #             if idx_clique_.shape[0] >= kappa:
    #                 # choose top-k by probability inside clique
    #                 idx_sorted = np.argsort(w_[idx_clique_])[::-1]
    #                 idx_spatial_ = idx_clique_[idx_sorted[:kappa]]
    #             else:
    #                 # not enough nodes in clique → fallback to temporal set
    #                 idx_sorted = np.argsort(w_[idx_temporal_])[::-1]
    #                 idx_spatial_ = idx_temporal_[idx_sorted[:kappa]]
    
    #     else:
    #         # No spatial info → just use probability
    #         idx_sorted = np.argsort(w_[idx_temporal_])[::-1]
    #         idx_spatial_ = idx_temporal_[idx_sorted[:kappa]]
    
    #     return idx_neighbors_, idx_temporal_, idx_spatial_, sigma

    # def _graph_neighborhood_filtering(self, w_, d_spatial_, d_temporal_, xi, gamma, clique_order, kappa):

    #     sigma = None
    #     # Filter by similarity
    #     idx_ = np.arange(w_.shape[0], dtype = int)

    #     if d_temporal_ is None:
    #         idx_temporal_ = idx_neighbors_.copy()
    #     else:
    #         idx_temporal_ = idx_[idx_neighbors_][d_temporal_[idx_neighbors_] <= gamma]
            
    #     if (idx_temporal_.shape[0] < kappa) or (d_spatial_ is None):  
    #         # Increase similarity threshold 
    #         idx_spatial_ = idx_[w_ >= np.sort(w_)[::-1][kappa]][:kappa]
    
    #     else: 
    #         idx_spatial_ = idx_[idx_temporal_][w_[idx_temporal_] >= np.sort(w_[idx_temporal_])[::-1][kappa]][:kappa]

    #     return idx_neighbors_, idx_temporal_, idx_spatial_, sigma

    def _similarity_filter(self, w_, xi, kappa):
        idx_ = np.arange(w_.shape[0], dtype=int)
        idx_neighbors_ = idx_[w_ >= xi]
        if idx_neighbors_.shape[0] < kappa:
            return idx_.copy()
        else:
            return idx_neighbors_

    def _temporal_filter(self, idx_neighbors_, d_temporal_, gamma, kappa):
        if d_temporal_ is None:
            idx_temporal_ = idx_neighbors_.copy()
        else:
            idx_temporal_ = idx_neighbors_[d_temporal_[idx_neighbors_] <= gamma]

        if idx_temporal_.shape[0] < kappa:
            idx_temporal_ = idx_neighbors_.copy()

        return idx_temporal_

    def _spatial_metric_filter(self, idx_temporal_, d_spatial_, kappa):
        sigma = None
    
        if idx_temporal_.shape[0] >= kappa:
            idx_sorted = np.argsort(d_spatial_[idx_temporal_])
            idx_spatial_ = idx_temporal_[idx_sorted[:kappa]]
            sigma = d_spatial_[idx_spatial_].max()
        else:
            idx_spatial_ = idx_temporal_.copy()
    
        return idx_spatial_, sigma

    def _spatial_clique_filter(self, idx_temporal_, w_, d_spatial_, clique_order, kappa):
        sigma = None
        
        # Nodes within clique neighborhood
        idx_clique_ = idx_temporal_[d_spatial_[idx_temporal_] <= clique_order]
    
        if idx_clique_.shape[0] >= kappa:
            idx_sorted = np.argsort(w_[idx_clique_])[::-1]
            return idx_clique_[idx_sorted[:kappa]], sigma
        else:
            idx_spatial_ = idx_temporal_.copy()
            
        return idx_spatial_, sigma

    # Calculate the neighborhood's focal curve
    def _focal_curve(self, M_, w_prime_):
        return M_.T @ w_prime_

    # Fuse neighboring curves with day-ahead forecasts
    def _fuse_curves(self, F_tr_, E_tr_, eta_, idx_spatial_, S, t, trust_rate, kappa, p_fusion,  
                     eps = 1e-8):

        # Initialize variables
        M_ = np.zeros((kappa, S))
        F_ = np.zeros((kappa, S))
        E_ = np.zeros((kappa, S))
        m_0_ = np.zeros((kappa, 1))
        
        for i, j in zip(idx_spatial_, range(kappa)):

            # Partial actual and day-ahead curves to fuse
            F_[j, :] = F_tr_[i, t:] 
            E_[j, :] = E_tr_[i, t:] 

            # Last observations of neighboring curves 
            m_0_[j] = F_tr_[i, t - 1]

            # Calculate the probability of fusing
            p0 = np.sum(F_[j, :] == 0)/F_[j, :].shape[0]
            pf = p_fusion * (1. - trust_rate) / np.clip(1. - p0, eps, 1. - eps)

            # Probability of fusion rejection
            if np.random.uniform(0., 1., size=1)[0] < pf:
                M_[j, :] = F_[j, :]
            else:
                M_[j, :] = F_[j, :] * (1. - eta_)  + E_[j, :] * eta_
        
        return M_, m_0_

    def fit(self, F_, E_, dt_, 
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

        # Temporal filter to apply to intervals
        if interval_mask is None:
            interval_mask = np.ones(self.dt_.shape, dtype = bool)
        self.interval_mask = interval_mask

        # Curves and asset dates
        self.t_ = t_

        # Curves and asset locations
        self.X_ = X_

    def predict(self, 
                f_ = None, 
                e_ = None,
                x_ = None,
                t = None,
                clique_order = None,
                forget_rate_f = 1.,
                forget_rate_e = .5,
                lookup_rate = .05,
                length_scale_f = .1,
                length_scale_e = .75,
                trust_rate = 0.0175,
                nu = 340,
                gamma = 30,
                xi = 0.99,
                kappa = 500,
                p_fusion = 1.):

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
        self.lookup_rate_e = lookup_rate
        self.length_scale_f = length_scale_f
        self.length_scale_e = length_scale_e
        self.trust_rate_e = trust_rate
        self.nu = nu
        self.p_fusion = p_fusion

        # Spatial distance parameters
        self.clique_order = clique_order
        if self.clique_order != None:
            self.clique_order = int(self.clique_order)

        # Neighborhood hyperparameters
        self.gamma = int(gamma)
        self.xi = xi
        self.kappa = int(kappa)
        
        # Interval
        self.interval = self.f_.shape[0]
        self.S = self.T - self.interval
        
        self.tau_ = self.dt_[:self.interval]
        self.s_ = self.dt_[self.interval:]

       # d: Temporal distance between samples 
        if self._distances['temporal'] == 'seasonal':
            self.d_temporal_ = self._seasonal_dist(self.t_, self.t)
        elif self._distances['temporal'] == 'seasonal_equinox':
            self.d_temporal_ = self._seasonal_equinox_dist(self.t_, self.t)
        else:
            self.d_temporal_ = None
            
        # spatial: Euclidean spatial distance between samples
        if self._distances['spatial'] == 'euclidean':
            self.d_spatial_ = self._weighted_euclidian_dist(self.X_, self.x_)
        # spatial: Haverise spatial distance between samples
        elif self._distances['spatial'] == 'haversine':
            self.d_spatial_ = self._haversine_dist(self.X_, self.x_)
        # spatial: Graph spatial distance between samples
        elif self._distances['spatial'] == 'graph':
            self.d_spatial_ = self._graph_dist(self.X_[:, 1])
        else:
            self.d_spatial_ = None
            
        # Calculate the temporal threshold 
        if (self.t_ is None) or (self.t is None):
            self.gamma = None
            self.gamma_prime = None
        else:
            self.gamma_prime = self._seasonal_equinox_dist(self.t, self.t, self.gamma)

        # phi: importance weights based on past time distance
        self.phi_ = self._exponential_growth(self.interval, self.forget_rate_f)
        # Mask intervals
        self.phi_[~self.interval_mask[:self.interval]] = 0.
        
        # psi: importance weights based on past and future time distance
        psi_minus_ = self._exponential_growth(self.interval, self.forget_rate_e)
        psi_plus_ = self._exponential_decay(self.S, self.lookup_rate_e)
        self.psi_ = np.concatenate([psi_minus_, psi_plus_], axis = 0)

        # Mask intervals
        self.psi_[~self.interval_mask] = 0.
        
        # eta: importance weights based on future time distance
        #eta_ = _logistic(s_ - t*5 - nu*60., trust_rate)
        self.eta_ = self._linear_inverse_exponential(self.s_[::-1], self.T*12, self.nu, self.trust_rate_e)

        # d: Euclidean similarity distance between samples weighted by importance weights
        self.d_f_ = self._weighted_euclidian_dist(self.F_[:, :self.interval], self.f_, w_ = self.phi_)
        self.d_e_ = self._weighted_euclidian_dist(self.E_, self.e_, w_ = self.psi_)

        # w: normalized weights distance across observations based on the exponential link function
        self.w_f_ = self._rbf_kernel(self.d_f_, self.length_scale_f)
        self.w_e_ = self._rbf_kernel(self.d_e_, self.length_scale_e)
        
        # Neighbor by partially observed curve or forecasted curve
        self.w_ = np.min(np.stack([self.w_f_, self.w_e_]), axis = 0)

        # Similarity filter
        self.idx_neighbors_ = self._similarity_filter(self.w_, self.xi, self.kappa)

        # Temporal filter
        self.idx_temporal_ = self._temporal_filter(self.idx_neighbors_, self.d_temporal_, self.gamma_prime, self.kappa)

        # spatial filter: Euclidean or Haversine spatial distance between samples
        if (self._distances['spatial'] == 'euclidean') or (self._distances['spatial'] == 'haversine'):
              self.idx_spatial_, self.sigma = self._spatial_metric_filter(self.idx_temporal_, self.d_spatial_, self.kappa)

        # spatial filter: Graph spatial distance between samples
        if self._distances['spatial'] == 'graph':
            self.idx_spatial_, self.sigma = self._spatial_clique_filter(self.idx_temporal_, self.w_, self.d_spatial_, self.clique_order, self.kappa)

        # If no filter
        if self.idx_spatial_.shape[0] > self.kappa:
            idx_ = np.argsort(self.w_[self.idx_temporal_])[::-1]
            self.idx_spatial_ = self.idx_temporal_[idx_[:self.kappa]]

        # # Filter neighbors by temporal (seasonal) and spatial (Haversine) distance
        # (self.idx_neighbors_, 
        # self.idx_temporal_, 
        # self.idx_spatial_, 
        # self.sigma) = self._neighborhood_filtering(self.w_,
        #                                            self.d_spatial_, 
        #                                            self.d_temporal_, 
        #                                            self.xi, 
        #                                            self.clique_order,
        #                                            self.gamma_prime, 
        #                                            self.kappa)

        # Fuse neighboring curves with day-ahead forecasts
        self.M_, self.m_0_ = self._fuse_curves(self.F_, 
                                               self.E_, 
                                               self.eta_, 
                                               self.idx_spatial_, 
                                               self.S, 
                                               self.interval, 
                                               self.trust_rate_e, 
                                               self.kappa, 
                                               self.p_fusion)
        
        # Normalized the weight of each neighboring curve 
        self.w_prime_ = self.w_[self.idx_spatial_]/self.w_[self.idx_spatial_].sum()
        self.w_prime_prime = self.w_[self.idx_spatial_]

        # Neighborhood focal curve
        self.f_focal_ = self._focal_curve(self.M_, self.w_prime_)
        
        self.f_0 = self.f_[-1]
        self.f_0_ = np.ones(self.m_0_.shape)*self.f_0
        self.M_ext_ = np.concatenate([self.f_0_, self.M_], axis = 1)
        
        return self.M_

    # Downsample collection of curves
    def functional_downsampling(self, subsample, n_basis = 20):

        dt_ = self.dt_
        M_ = self.M_

        n_samples, n_points = M_.shape

        f_0_   = np.ones(self.m_0_.shape)*self.f_0
        M_ext_ = np.concatenate([f_0_, M_], axis = 1)
    
        # Ensure the length is divisible by subsample
        dt    = dt_[1] - dt_[0]
        t_    = dt_[-n_points-1:]
        t_ds_ = np.linspace(t_[1], t_[-1], int(n_points/subsample))
    
        # Create an FDataGrid object
        data_ = [M_ext_[i, :] for i in range(n_samples)]
        _fd = FDataGrid(data_matrix = data_,  grid_points = t_)
        _fd_int = _fd.to_basis(BSpline(n_basis = n_basis))
        
        # Interpolate first (useful if data are unevenly spaced or need smoothing)
        M_int_ = _fd_int.to_grid(t_)
        self.M_int_ = np.stack([M_int_.data_matrix[i] for i in range(n_samples)])[..., 0]
        
        # Re-evaluate existing data
        M_int_ds_ = _fd_int.to_grid(t_ds_)
        self.M_int_ds_ = np.stack([M_int_ds_.data_matrix[i] for i in range(n_samples)])[..., 0]

        return self.M_int_, self.M_int_ds_

    
    def focal_curve_envelope(self, _depth, X_, dist, 
                             max_iter = 100,
                             idx_focal = 0):
        """
        Envelope algorithm to obtain functional neighborhoods.
    
        Parameters
        ----------
        data: dict
            Dictionary with keys:
                - "x": np.ndarray of grid points (n_points,)
                - "y": np.ndarray of function values (n_points, n_curves)
        focal: int or str
            Index (or column name) of the focal curve to envelope.
        plot : bool, optional
            Whether to plot the selected curves in each iteration.
        max_iter: int, optional
            Maximum number of iterations before stopping.
    
        Returns
        -------
        dict
            Dictionary with key 'Jordered' containing the ordered list
            of selected curve indices.
        """

        self.dist = dist

        self.f_focal_ext_ = np.insert(self.f_focal_, 0, self.f_0)

        X_ext_ = np.concatenate([self.f_focal_ext_[np.newaxis, :], X_], axis = 0)[:, 1:].T
        dt_ = self.dt_[-X_ext_.shape[0]:]

        if self.dist == 'fknn':
            dist_ = np.insert(self.w_prime_, 0, 0)
        else:
            dist_ = self.dist

        
        # Compute depth to find the focal curve
        _fd_filtered = FDataGrid(data_matrix = X_ext_.T,
                                 grid_points = dt_)
        
        #filtered_depth_ = _depth(_fd_filtered)
        # idx_focal       = np.argsort(-filtered_depth_)[0]
        f_ = X_ext_[:, idx_focal]
    
        if isinstance(dist_, str):
            # Distances from curves to the focal curve
            if dist_ == 'sup':
                dist_ = np.max(np.abs(X_ext_.T - f_), axis = 1)
            elif dist_ == 'l2':
                dist_ = np.sum((X_ext_.T - f_)**2, axis = 1)
                
        # Initialize
        idx_subsample = []
        idx_          = [i for i in range(X_ext_.shape[1]) if i != idx_focal]
        iter_depth    = [0]
        iteration     = 0
    
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
                
                if len(remaining_points - set(dt_[Ji_]) ) == len(remaining_points):
                    # Does not envelope
                    idx_candidates.pop(0)
                else:
                    remaining_points -= set(dt_[Ji_])
                    
                    idx_iter_subsample.append(idx_next)
                    
                    idx_candidates = [c for c in idx_candidates if c not in idx_iter_subsample]
                    idx_           = [c for c in idx_ if c not in idx_iter_subsample]
    
            # Compute functional depth 
            _fd_subset = FDataGrid(data_matrix = (X_ext_[:, [idx_focal] 
                                                     + idx_subsample 
                                                     + idx_iter_subsample]).T, grid_points = dt_)
        
            depth_             = _depth(_fd_subset)
            idx_depth_         = np.argsort(-depth_)
            idx_ordered_depth_ = np.array([idx_focal] + idx_subsample + idx_iter_subsample)[idx_depth_]
    
            # How deep is the new set of curves?
            depth_percentile = 1 - np.where(idx_ordered_depth_ == idx_focal)[0][0]/(len(idx_ordered_depth_) - 1)
            
            iter_depth.append(depth_percentile)
    
            # Accept subsample if depth improves
            if max(iter_depth[:-1]) <= iter_depth[-1]:
                idx_subsample.extend(idx_iter_subsample)
                
            # Stop if there are no more candidate curves
            if not idx_candidates:
                break
                
            # Stop if max_iter is reached
            if iteration >= max_iter:
                break
    
        # Selected curves
        idx_sel_ = [idx_focal] + idx_subsample
        
        # Final selected curves ordered by depth
        _fd_filtered = FDataGrid(data_matrix = X_ext_[:, idx_sel_].T,
                                 grid_points = dt_)
            
        filtered_depth_             = _depth(_fd_filtered)
        idx_filtered_depth_         = np.argsort(-filtered_depth_)
        idx_ordered_filtered_depth_ = np.array(idx_sel_)[idx_filtered_depth_]
        
        # Select curves for the focal curve envelop
        self.J_ = X_ext_[:, idx_ordered_filtered_depth_].T
        
        return self.J_

    # Confidence bands from focal-curve envelop
    #def _focal_envelop_confidence_bands(J_, alpha_, k_, f_0):    
    def focal_envelop_confidence_bands(self, alpha_, k_):    
    
        self._upper_envelop_confidence_bands = {}
        self._lower_envelop_confidence_bands = {}
        for i in range(len(alpha_)):
            N = int(k_[i]*self.J_.shape[0])
            #print(N, self.J_[:N, :].shape)
            
            self._upper_envelop_confidence_bands[f'{alpha_[i]}'] = np.insert(
                np.max(self.J_[:N, :], axis = 0), 0, self.f_0
            )
            self._lower_envelop_confidence_bands[f'{alpha_[i]}'] = np.insert(
                np.min(self.J_[:N, :], axis = 0), 0, self.f_0
            )
            
        return self.f_focal_ext_, self._upper_envelop_confidence_bands, self._lower_envelop_confidence_bands
        
    # Functional boxplot from a smooth functional depth metric
    def functional_boxplot(self, X_, depth_score_, alpha_ = [0.25, 0.5, 0.75]):
    
        idx_ = np.argsort(depth_score_)[::-1]
    
        self._upper_functional_boxplot = {}
        self._lower_functional_boxplot = {}
        for i in range(len(alpha_)):
            X_sel_ = X_[idx_[:-int(X_.shape[0] * alpha_[i])],]
            
            self._upper_functional_boxplot[f'{alpha_[i]}'] = np.max(X_sel_, axis = 0)
            self._lower_functional_boxplot[f'{alpha_[i]}'] = np.min(X_sel_, axis = 0)
    
            self._upper_functional_boxplot[f'{alpha_[i]}'][self._lower_functional_boxplot[f'{alpha_[i]}'] > 1] = 1
            self._lower_functional_boxplot[f'{alpha_[i]}'][self._lower_functional_boxplot[f'{alpha_[i]}'] < 0] = 0
            
        self._upper_functional_boxplot['max'] = np.max(X_, axis = 0)
        self._lower_functional_boxplot['min'] = np.min(X_, axis = 0)

        self.f_deepest_ = X_[idx_[0],]

        return self.f_deepest_, self._upper_functional_boxplot, self._lower_functional_boxplot
        

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
                self._eQuantile(ECDF(X_[:, j]), [1. - alpha_[i]/2.]) for j in range(X_.shape[1])
            ])[:, 0]
            
            self._lower_ecdf_confidence_bands[f'{alpha_[i]}'] = np.stack([
                self._eQuantile(ECDF(X_[:, j]), [alpha_[i]/2.]) for j in range(X_.shape[1])
            ])[:, 0]
            
    
        self.f_median_ext_ = np.median(X_, axis = 0)
    
        return self.f_median_ext_, self._upper_ecdf_confidence_bands, self._lower_ecdf_confidence_bands


    # Confidence bands from depth function
    def depth_confidence_bands(self, _depth, X_, alpha_, k_):    
    
        # Compute functional depth
        _F = FDataGrid(data_matrix = X_,
                       grid_points = self.dt_[-X_.shape[1]:])
        
        depth_score_ = _depth(_F)
        idx_ = np.argsort(depth_score_)[::-1]
    
        self._upper_depth_confidence_bands = {}
        self._lower_depth_confidence_bands = {}
        for i in range(len(alpha_)):
            N = int(k_[i]*X_.shape[0])
            self._upper_depth_confidence_bands[f'{alpha_[i]}'] = np.max(X_[idx_[:N], :], axis = 0)
            self._lower_depth_confidence_bands [f'{alpha_[i]}'] = np.min(X_[idx_[:N], :], axis = 0)
        
        # Deepest curve
        self.f_deepest_ext_ = X_[idx_[0], :]
    
        return self.f_deepest_ext_, self._upper_depth_confidence_bands, self._lower_depth_confidence_bands 

    # Silverman's Rule
    def _silverman_rule(self, x_):
        IQR = np.percentile(x_, 75) - np.percentile(x_, 25)
        return 0.9 * min(np.std(x_), IQR / 1.34) * x_.shape[0] ** (-1 / 5)

    def kernel_density_estimation(self, M_, 
                                  algorithm = "auto", 
                                  kernel = "gaussian"):
        _KDS = []
        for i in range(M_.shape[1]):
            _KD = KernelDensity(bandwidth = self._silverman_rule(M_[:, i]), 
                                algorithm = algorithm, 
                                kernel = kernel).fit(M_[:, i][:, np.newaxis])
            _KDS.append(_KD)

        return _KDS
    