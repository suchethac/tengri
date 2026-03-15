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

# --- New high-level API ---
from diffsed.distributions import Uniform, Gaussian, LogUniform, LogNormal, StudentT, Fixed
from diffsed.param_spec import ParamSpec
from diffsed.model import Model
from diffsed.fitter import Fitter
from diffsed.posterior import Posterior
from diffsed.raytrace_jax import sample_raytrace
from diffsed.hierarchical import HierarchicalFitter, HierarchicalResult
from diffsed.models.observation.filters import load_filter_set

__all__ = [
    # High-level API
    "Model",
    "ParamSpec",
    "Fitter",
    "Posterior",
    "Uniform",
    "Gaussian",
    "LogUniform",
    "LogNormal",
    "StudentT",
    "Fixed",
    # Samplers
    "sample_raytrace",
    # Hierarchical
    "HierarchicalFitter",
    "HierarchicalResult",
    # Data loading
    "load_ssp_data",
    "load_filter_set",
    "SSPData",
    # Low-level (kept for internal use and advanced users)
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
    "ForwardModel",
    "ModelConfig",
    "generate_mock",
]

# Plotting utilities
from diffsed.plotting import (setup_style, plot_sfh, plot_sfh_comparison,
                               plot_sed_fit, plot_spectrum_fit,
                               safe_corner, plot_corner_comparison,
                               diagnostics_table, COLORS, SDSS_WAVE_EFF,
                               SPECTRAL_FEATURES)
