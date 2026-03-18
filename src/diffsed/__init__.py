"""diffsed: Differentiable SED fitting with IFT star formation history priors.

A modular, fully differentiable JAX pipeline:
PSD-governed GP → SFH → DSPS SED → photometry/spectroscopy.
"""

# Enable float64 — required for cosmological distance calculations
# (dL^2 at z>0.01 overflows float32)
import jax

jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"

# --- New high-level API ---
from diffsed.distributions import Fixed, Gaussian, LogNormal, LogUniform, StudentT, Uniform
from diffsed.fitter import Fitter
from diffsed.forward_model import ForwardModel, ModelConfig, generate_mock
from diffsed.hierarchical import HierarchicalFitter, HierarchicalResult
from diffsed.model import Model
from diffsed.models.dust.charlot_fall import charlot_fall
from diffsed.models.observation.filters import load_filter_set
from diffsed.models.sfh.gp_sfh import (
    compute_sqrt_power_drw,
    generate_gp_batch,
    generate_gp_fourier,
    gp_from_xi,
    make_log_age_grid,
)
from diffsed.models.sfh.mean_sfh import constant_sfh, delayed_tau, double_powerlaw
from diffsed.models.sfh.psd_models import drw_acf, drw_variance, psd_drw
from diffsed.models.sps.dsps_wrapper import SSPData, load_ssp_data
from diffsed.param_spec import ParamSpec
from diffsed.posterior import Posterior
from diffsed.raytrace_jax import sample_raytrace
from diffsed.vi_config import VIConfig

__all__ = [
    "Fitter",  # High-level API
    "Fixed",  # High-level API
    "ForwardModel",  # Low-level (internal use and advanced users)
    "Gaussian",  # High-level API
    "HierarchicalFitter",  # Hierarchical
    "HierarchicalResult",  # Hierarchical
    "LogNormal",  # High-level API
    "LogUniform",  # High-level API
    "Model",  # High-level API
    "ModelConfig",  # Low-level (internal use and advanced users)
    "ParamSpec",  # High-level API
    "Posterior",  # High-level API
    "SSPData",  # Data loading
    "StudentT",  # High-level API
    "Uniform",  # High-level API
    "VIConfig",  # High-level API
    "charlot_fall",  # Low-level (internal use and advanced users)
    "compute_sqrt_power_drw",  # Low-level (internal use and advanced users)
    "constant_sfh",  # Low-level (internal use and advanced users)
    "delayed_tau",  # Low-level (internal use and advanced users)
    "double_powerlaw",  # Low-level (internal use and advanced users)
    "drw_acf",  # Low-level (internal use and advanced users)
    "drw_variance",  # Low-level (internal use and advanced users)
    "generate_gp_batch",  # Low-level (internal use and advanced users)
    "generate_gp_fourier",  # Low-level (internal use and advanced users)
    "generate_mock",  # Low-level (internal use and advanced users)
    "gp_from_xi",  # Low-level (internal use and advanced users)
    "load_filter_set",  # Data loading
    "load_ssp_data",  # Data loading
    "make_log_age_grid",  # Low-level (internal use and advanced users)
    "psd_drw",  # Low-level (internal use and advanced users)
    "sample_raytrace",  # Samplers
]

# Plotting utilities
from diffsed.plotting import (
    COLORS,
    SDSS_WAVE_EFF,
    SPECTRAL_FEATURES,
    diagnostics_table,
    plot_corner_comparison,
    plot_sed_fit,
    plot_sfh,
    plot_sfh_comparison,
    plot_spectrum_fit,
    safe_corner,
    setup_style,
)
