import sys

PROJECT = "/Users/Guille/Desktop/dynamic_update/functional_forecast_dynamic_update"

sys.path.insert(0, PROJECT)

import numpy as np

from src import loader
from src.config import DATA, DEPTH, IMAGES, VALIDATION, TABLES
from src.utils import KS

T = 288
method = "fusion"
resource = "solar"
aggregation = "asset"

# ===============================================

exp_description="unbiased-025-6"
#_init = {120: 1, 132: 2, 144: 2, 168: 3}
_init = {120: 4, 132: 2, 144: 3, 168: 3}

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

# ===============================================

df_fmt, latex = loader.error_scores(
    _init,
    resource, 
    method,
    aggregation,
    exp_description,
    VALIDATION
)
print(df_fmt)

(TABLES / f"error_{resource}_{aggregation}_{exp_description}.tex").write_text(latex, encoding = "utf-8")

# ===============================================

df_fmt, latex = loader.envelope_scores(
    envelope_, 
    alphas = [0.1, 0.2],
)
print(df_fmt['FCS'])

(TABLES / f"envelope_{resource}_{aggregation}_{exp_description}.tex").write_text(latex, encoding = "utf-8")

# ===============================================

LEAD = 12

df_fmt, latex = loader.ks_by_block(
    _init, 
    resource, 
    method, 
    aggregation, 
    exp_description,
    path_to_validation = VALIDATION,
    intervals = [120, 132, 144, 168],
    lead = LEAD,
    starts = np.arange(71, 215, LEAD),
    _KS = KS,
)
print(df_fmt)

(TABLES / f"ks_{resource}_{aggregation}_{exp_description}.tex").write_text(latex, encoding = "utf-8")

# ===============================================
# ===============================================

LEAD = 24

exp_description = 'unbiased'
method = 'prophet'
_init = {120: 1, 132: 1, 144: 1, 168: 1}

df_fmt, latex = loader.ks_by_block(
    _init, 
    resource, 
    method, 
    aggregation, 
    exp_description,
    path_to_validation = VALIDATION,
    intervals = [120, 132, 144, 168],
    lead = LEAD,
    starts = np.arange(0, 287, LEAD),
    _KS = KS,
)
print(df_fmt)

(TABLES / f"ks_{resource}_{aggregation}_{exp_description}.tex").write_text(latex, encoding = "utf-8")

# ===============================================

df_fmt, latex = loader.error_scores(
    _init,
    resource, 
    method,
    aggregation,
    exp_description,
    VALIDATION,
)
print(df_fmt)

(TABLES / f"error_{resource}_{aggregation}_{exp_description}.tex").write_text(latex, encoding = "utf-8")

# ===============================================
# ===============================================

LEAD = 24

exp_description = 'unbiased'
method = 'prophet'
_init = {120: 1, 132: 1, 144: 1, 168: 1}

df_fmt, latex = loader.ks_by_block(
    _init, 
    resource, 
    method, 
    aggregation, 
    exp_description,
    path_to_validation = VALIDATION,
    intervals = [120, 132, 144, 168],
    lead = LEAD,
    starts = np.arange(0, 287, LEAD),
    _KS = KS,
)
print(df_fmt)

(TABLES / f"ks_{resource}_{aggregation}_{method}.tex").write_text(latex, encoding = "utf-8")

# ===============================================

df_fmt, latex = loader.error_scores(
    _init,
    resource, 
    method,
    aggregation,
    exp_description,
    VALIDATION,
)
print(df_fmt)

(TABLES / f"error_{resource}_{aggregation}_{method}.tex").write_text(latex, encoding = "utf-8")

# ===============================================
# ===============================================

LEAD = 24

exp_description = 'unbiased'
method = 'daref'
_init = {120: 1, 132: 1, 144: 1, 168: 1}

df_fmt, latex = loader.ks_by_block(
    _init, 
    resource, 
    method, 
    aggregation, 
    exp_description,
    path_to_validation = VALIDATION,
    intervals = [120, 132, 144, 168],
    lead = LEAD,
    starts = np.arange(0, 287, LEAD),
    _KS = KS,
)
print(df_fmt)

(TABLES / f"ks_{resource}_{aggregation}_{method}.tex").write_text(latex, encoding = "utf-8")

# ===============================================

df_fmt, latex = loader.error_scores(
    _init,
    resource, 
    method,
    aggregation,
    exp_description,
    VALIDATION,
)
print(df_fmt)

(TABLES / f"error_{resource}_{aggregation}_{method}.tex").write_text(latex, encoding = "utf-8")
