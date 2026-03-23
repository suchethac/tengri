"""tengri: Differentiable SED fitting with IFT star formation history priors.

A modular, fully differentiable JAX pipeline:
PSD-governed GP → SFH → DSPS SED → photometry/spectroscopy.
"""

# Enable float64 — required for cosmological distance calculations
# (dL^2 at z>0.01 overflows float32)
import jax

jax.config.update("jax_enable_x64", True)

# Enable persistent XLA compilation cache — avoids re-compilation
# across sessions/restarts. ~10x first-call speedup on subsequent runs.
jax.config.update("jax_compilation_cache_dir", "/tmp/tengri_jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

__version__ = "0.1.0"

# --- New high-level API ---
from tengri.core.mock import MockData, generate_mock
from tengri.core.model import Model
from tengri.core.noise import (
    compute_effective_noise,
    compute_std_inv,
    has_noise_model,
    uses_student_t,
    variable_noise_hamiltonian,
)
from tengri.core.param_spec import ParamSpec
from tengri.core.prediction import (
    DerivedQuantities,
    EmissionLines,
    Prediction,
    SEDQuantities,
    SFHQuantities,
)
from tengri.distributions import Fixed, Gaussian, LogNormal, LogUniform, StudentT, Uniform
from tengri.inference.fitter import Fitter
from tengri.inference.hierarchical import HierarchicalFitter, HierarchicalResult
from tengri.inference.posterior import Posterior
from tengri.inference.raytrace import sample_raytrace
from tengri.inference.vi_config import VIConfig
from tengri.models.dust.attenuation import two_component_dust
from tengri.models.observation.filters import load_filter_set
from tengri.models.observation.noise_config import NoiseConfig
from tengri.models.observation.observation import Observation
from tengri.models.observation.photometry_config import Photometry
from tengri.models.observation.spectroscopy_config import SpectroscopyConfig
from tengri.models.sfh.gp_sfh import (
    compute_sqrt_power_drw,
    generate_gp_batch,
    generate_gp_fourier,
    gp_from_xi,
    make_log_age_grid,
)
from tengri.models.sfh.mean_sfh import (
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
from tengri.models.sfh.psd_models import drw_acf, drw_variance, psd_drw
from tengri.models.sfh.registry import (
    FIELD_MODEL_REGISTRY,
    SFH_REGISTRY,
    compute_field_gp,
    resolve_sfh,
)
from tengri.models.sps.dsps_wrapper import (
    SSPData,
    effective_metallicity,
    has_alpha_grid,
    interpolate_met_alpha,
    load_ssp_data,
    salaris_feh_from_mh,
    salaris_mh_from_feh,
)

__all__ = [
    # High-level API
    "AGEMAX_YR",
    # Registry
    "FIELD_MODEL_REGISTRY",
    "SFH_REGISTRY",
    "DerivedQuantities",
    "EmissionLines",
    "Fitter",
    "Fixed",
    "Gaussian",
    "HierarchicalFitter",
    "HierarchicalResult",
    "LogNormal",
    "LogUniform",
    "MockData",
    "Model",
    "NoiseConfig",
    "Observation",
    "ParamSpec",
    "Photometry",
    "Posterior",
    "Prediction",
    "SEDQuantities",
    "SFHQuantities",
    "SSPData",
    "SpectroscopyConfig",
    "StudentT",
    "Uniform",
    "VIConfig",
    "compute_effective_noise",
    "compute_field_gp",
    "compute_sqrt_power_drw",
    "compute_std_inv",
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
    "has_noise_model",
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
    "two_component_dust",
]

# Plotting utilities
from tengri.plotting import (
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
