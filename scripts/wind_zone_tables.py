import sys

PROJECT = "/Users/Guille/Desktop/dynamic_update/functional_forecast_dynamic_update"

sys.path.insert(0, PROJECT)

import numpy as np

from src import loader
from src.config import DATA, DEPTH, IMAGES, VALIDATION, TABLES
from src.utils import KS

T = 48
method = 'fusion'
resource = 'wind'
aggregation = 'zone'

# ===============================================

# exp_description="unbiased-025-C0-6"
# _init = {6: 4, 12: 4, 18: 4}

# hyper_, envelope_ = loader.hyperparameters(
#     _init,
#     resource,
#     method,
#     aggregation,
#     exp_description,
#     path_to_validation=VALIDATION,
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
#     VALIDATION)
# print(df_fmt)

# (TABLES / f"error_{resource}_{aggregation}_{exp_description}.tex").write_text(latex, encoding = "utf-8")

# df_fmt, latex = loader.envelope_scores(
#     envelope_, 
#     alphas = [0.1, 0.2],
# )
# print(df_fmt)

# (TABLES / f"envelope_{resource}_{aggregation}_{exp_description}.tex").write_text(latex, encoding = "utf-8")

# ===============================================
# ===============================================

# exp_description="unbiased-025-C0-6"
# _init = {6: 4, 12: 4, 18: 1}
# exp_description="unbiased-025-C1-6"
# _init = {6: 4, 12: 4, 18: 4}
exp_description="unbiased-025-C1-6"
_init = {6: 2, 12: 4, 18: 4}
# exp_description="unbiased-025-C2-6"
# _init = {6: 4, 12: 1, 18: 3}

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

# ===============================================

df_fmt, latex = loader.error_scores(
    _init,
    resource, 
    method,
    aggregation,
    exp_description,
    VALIDATION)
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

LEAD = 6

df_fmt, latex = loader.ks_by_block(
    _init, 
    resource, 
    method, 
    aggregation, 
    exp_description,
    path_to_validation = VALIDATION,
    intervals = [6, 12, 18],
    lead = LEAD,
    starts = np.arange(0, 47, LEAD),
    _KS = KS,
)
print(df_fmt)

(TABLES / f"ks_{resource}_{aggregation}_{exp_description}.tex").write_text(latex, encoding = "utf-8")

# ===============================================
# ===============================================

# exp_description="unbiased-025-C2-6"
# _init = {6: 4, 12: 4, 18: 4}

# hyper_, envelope_ = loader.hyperparameters(
#     _init,
#     resource,
#     method,
#     aggregation,
#     exp_description,
#     path_to_validation=VALIDATION,
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
#     VALIDATION)
# print(df_fmt)

# (TABLES / f"error_{resource}_{aggregation}_{exp_description}.tex").write_text(latex, encoding = "utf-8")

# df_fmt, latex = loader.envelope_scores(
#     envelope_, 
#     alphas = [0.1, 0.2],
# )
# print(df_fmt)

# (TABLES / f"envelope_{resource}_{aggregation}_{exp_description}.tex").write_text(latex, encoding = "utf-8")

# ===============================================
