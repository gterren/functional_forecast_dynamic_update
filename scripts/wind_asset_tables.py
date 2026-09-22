import sys

PROJECT = "/Users/Guille/Desktop/dynamic_update/functional_forecast_dynamic_update"

sys.path.insert(0, PROJECT)

import numpy as np

from src import loader
from src.config import DATA, DEPTH, IMAGES, VALIDATION, TABLES
from src.utils import KS

T = 288
method = "fusion"
resource = "wind"
aggregation = "asset"

# ===============================================

exp_description="biased-025-7"
_init = {72: 1, 144: 1, 216: 1}

exp_description="unbiased-025-7"
_init = {72: 2, 144: 2, 216: 3}

# exp_description="unbiased-025-7"
# _init = {72: 3, 144: 4, 216: 2}

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
    VALIDATION,
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

# exp_description="biased-025-6"
# _init = {72: 3, 144: 4, 216: 1}

# hyper_, envelope_ = loader.hyperparameters(
#     _init,
#     resource,
#     method,
#     aggregation,
#     exp_description,
#     path_to_validation = VALIDATION,
# )

# hyper_.loc['length_scale_f'] = 1./hyper_.loc['tau']
# hyper_.loc['length_scale_e'] = hyper_.loc['length_scale_f']*hyper_.loc['rho_e']
# hyper_.loc['length_scale_d'] = hyper_.loc['length_scale_f']*hyper_.loc['rho_d']
# print(hyper_)

# df_fmt, latex = loader.error_scores(
#     _init,
#     resource, 
#     method,
#     aggregation,
#     exp_description,
#     VALIDATION,
# )
# print(df_fmt)

# (TABLES / f"error_{resource}_{aggregation}_{exp_description}.tex").write_text(latex, encoding = "utf-8")

# df_fmt, latex = loader.envelope_scores(
#     envelope_, 
#     alphas = [0.1, 0.2],
# )
# print(df_fmt)

# (TABLES / f"envelope_{resource}_{aggregation}_{exp_description}.tex").write_text(latex, encoding = "utf-8")

# ===============================================

LEAD = 24

df_fmt, latex = loader.ks_by_block(
    _init, 
    resource, 
    method, 
    aggregation, 
    exp_description,
    path_to_validation = VALIDATION,
    intervals = [72, 144, 216],
    lead = LEAD,
    starts = np.arange(0, 287, LEAD),
    _KS = KS,
)
print(df_fmt)

(TABLES / f"ks_{resource}_{aggregation}_{exp_description}.tex").write_text(latex, encoding = "utf-8")

# ===============================================
# ===============================================

LEAD = 24

exp_description = 'unbiased'
method = 'arima'
_init = {72: 1, 144: 1, 216: 1}

df_fmt, latex = loader.ks_by_block(
    _init, 
    resource, 
    method, 
    aggregation, 
    exp_description,
    path_to_validation = VALIDATION,
    intervals = [72, 144, 216],
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
_init = {72: 1, 144: 1, 216: 1}

df_fmt, latex = loader.ks_by_block(
    _init, 
    resource, 
    method, 
    aggregation, 
    exp_description,
    path_to_validation = VALIDATION,
    intervals = [72, 144, 216],
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
