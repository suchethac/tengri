"""diffsed: Differentiable SED fitting with IFT star formation history priors.

A modular, fully differentiable JAX pipeline:
PSD-governed GP → SFH → DSPS SED → photometry/spectroscopy.
"""

# Enable float64 — required for cosmological distance calculations
# (dL^2 at z>0.01 overflows float32)
import jax
jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"

from diffsed.models.sfh.psd_models import psd_drw, drw_acf, drw_variance
from diffsed.models.sfh.gp_sfh import (
    gp_from_xi, generate_gp_fourier, generate_gp_batch,
    compute_sqrt_power_drw, make_log_age_grid,
)
from diffsed.models.sfh.mean_sfh import double_powerlaw, delayed_tau, constant_sfh
from diffsed.models.dust.charlot_fall import charlot_fall
from diffsed.models.sps.dsps_wrapper import load_ssp_data, SSPData
from diffsed.forward_model import ForwardModel, ModelConfig, generate_mock

__all__ = [
    "psd_drw",
    "drw_acf",
    "drw_variance",
    "gp_from_xi",
    "generate_gp_fourier",
    "generate_gp_batch",
    "compute_sqrt_power_drw",
    "make_log_age_grid",
    "double_powerlaw",
    "delayed_tau",
    "constant_sfh",
    "charlot_fall",
    "load_ssp_data",
    "SSPData",
    "ForwardModel",
    "ModelConfig",
    "generate_mock",
]
