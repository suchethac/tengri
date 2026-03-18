"""diffsed: Differentiable SED fitting with IFT star formation history priors.

A modular, fully differentiable JAX pipeline:
PSD-governed GP → SFH → DSPS SED → photometry/spectroscopy.
"""

# Enable float64 — required for cosmological distance calculations
# (dL^2 at z>0.01 overflows float32)
import jax

jax.config.update("jax_enable_x64", True)

# Enable persistent XLA compilation cache — avoids re-compilation
# across sessions/restarts. ~10x first-call speedup on subsequent runs.
jax.config.update("jax_compilation_cache_dir", "/tmp/diffsed_jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

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
from diffsed.models.sfh.mean_sfh import (
    AGEMAX_YR,
    constant_sfh,
    delayed_exponential_sfh,
    delayed_tau,
    double_powerlaw,
    dpl,
    exponential_sfh,
    lnorm,
    norm,
    snorm,
    triweight_burst,
    tsnorm,
)
from diffsed.models.sfh.psd_models import drw_acf, drw_variance, psd_drw
from diffsed.models.sfh.registry import (
    FIELD_MODEL_REGISTRY,
    SFH_REGISTRY,
    compute_field_gp,
    resolve_sfh,
)
from diffsed.models.sps.dsps_wrapper import SSPData, load_ssp_data
from diffsed.param_spec import ParamSpec
from diffsed.posterior import Posterior
from diffsed.raytrace_jax import sample_raytrace
from diffsed.vi_config import VIConfig

__all__ = [
    # High-level API
    "AGEMAX_YR",
    # Registry
    "FIELD_MODEL_REGISTRY",
    "SFH_REGISTRY",
    "Fitter",
    "Fixed",
    # Low-level
    "ForwardModel",
    "Gaussian",
    "HierarchicalFitter",
    "HierarchicalResult",
    "LogNormal",
    "LogUniform",
    "Model",
    "ModelConfig",
    "ParamSpec",
    "Posterior",
    "SSPData",
    "StudentT",
    "Uniform",
    "VIConfig",
    "charlot_fall",
    "compute_field_gp",
    "compute_sqrt_power_drw",
    # SFH models
    "constant_sfh",
    "delayed_exponential_sfh",
    "delayed_tau",
    "double_powerlaw",
    "dpl",
    "drw_acf",
    "drw_variance",
    "exponential_sfh",
    "generate_gp_batch",
    "generate_gp_fourier",
    "generate_mock",
    "gp_from_xi",
    "lnorm",
    "load_filter_set",
    "load_ssp_data",
    "make_log_age_grid",
    "norm",
    "psd_drw",
    "resolve_sfh",
    "sample_raytrace",
    "snorm",
    "triweight_burst",
    "tsnorm",
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
