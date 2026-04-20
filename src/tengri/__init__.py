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
# ~/.cache persists across reboots (unlike /tmp on macOS).
import os as _os

jax.config.update(
    "jax_compilation_cache_dir",
    _os.path.join(_os.path.expanduser("~"), ".cache", "tengri_jax_cache"),
)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

__version__ = "0.1.0"

# --- Exception hierarchy ---
from tengri.components.agn.agn_config import AGNConfig
from tengri.components.dust.attenuation import two_component_dust
from tengri.components.igm.dla import dla_transmission, dla_transmission_obs
from tengri.components.sfh.gp_sfh import (
    compute_sqrt_power_drw,
    generate_gp_batch,
    generate_gp_fourier,
    gp_from_xi,
    make_log_age_grid,
)
from tengri.components.sfh.mean_sfh import (
    AGEMAX_YR,
    constant_sfh,
    delayed_exponential_sfh,
    delayed_tau,
    double_powerlaw,
    dpl,
    exponential_sfh,
    gaussian_sfh,
    lnorm,
    lognormal_sfh,
    norm,
    skewnormal_sfh,
    snorm,
    triweight_burst,
    truncated_skewnormal_sfh,
    tsnorm,
)
from tengri.components.sfh.psd_models import drw_acf, drw_variance, psd_drw
from tengri.components.sfh.registry import (
    FIELD_MODEL_REGISTRY,
    SFH_REGISTRY,
    compute_field_gp,
    resolve_sfh,
)
from tengri.components.sps.dsps_wrapper import (
    SSPData,
    effective_metallicity,
    has_alpha_grid,
    interpolate_met_alpha,
    load_ssp_data,
    salaris_feh_from_mh,
    salaris_mh_from_feh,
)
from tengri.forward.convenience import catalog_summary, fit_batch
from tengri.forward.prediction import (
    DerivedQuantities,
    EmissionLines,
    Prediction,
    SEDQuantities,
    SFHQuantities,
)
from tengri.forward.result import SEDResult
from tengri.forward.sed_model import Model, PriorPredictive, SEDModel
from tengri.inference.backends.mcmc.raytrace import sample_raytrace
from tengri.inference.fitter import Fitter
from tengri.inference.hierarchical import (
    HierarchicalFitter,
    HierarchicalResult,
    PopulationFitter,
    PopulationPosterior,
)
from tengri.inference.posterior import Posterior
from tengri.inference.vi_config import VIConfig
from tengri.observation.filters import load_filter_set
from tengri.observation.line_flux_data import LineFluxData
from tengri.observation.line_list import LineCatalog, LineList

# --- New high-level API ---
from tengri.observation.mock import MockData, generate_mock
from tengri.observation.noise import (
    compute_effective_noise,
    compute_std_inv,
    has_noise_model,
    uses_student_t,
    variable_noise_hamiltonian,
)
from tengri.observation.noise_model import NoiseConfig, NoiseModel
from tengri.observation.observation import Observation
from tengri.observation.photometry_config import Photometry
from tengri.observation.spectral_indices import SpectralIndexData, SpectralIndexDef
from tengri.observation.spectroscopy import Spectroscopy, SpectroscopyConfig
from tengri.parameters.parameters import Parameters, ParamSpec
from tengri.parameters.priors import Fixed, Gaussian, LogNormal, LogUniform, StudentT, Uniform
from tengri.runtime.exceptions import (
    BackendError,
    ConfigError,
    InferenceError,
    ParameterError,
    TengriError,
    TengriIOError,
)
from tengri.runtime.settings import (
    DustConfig,
    ModelConfig,
    MultiwavelengthConfig,
    NebularConfig,
    SFHConfig,
)


def posteriors_to_dataframe(results: list, params: list[str] | None = None):
    """Summarise a list of Posteriors into a pandas DataFrame.

    Requires ``pandas`` (``pip install pandas``).

    Parameters
    ----------
    results : list of Posterior
        Output of ``model.fit_catalog()`` or any list of Posterior objects.
    params : list of str or None
        Parameter names to include. Default: all scalar free parameters,
        excluding ``psd_xi``.

    Returns
    -------
    pandas.DataFrame
        One row per galaxy, columns: ``{param}_median``, ``{param}_lo68``,
        ``{param}_hi68`` for each requested parameter.

    Examples
    --------
    >>> df = tengri.posteriors_to_dataframe(results, params=["met_logzsol", "dust_tau_bc"])
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "posteriors_to_dataframe() requires pandas: pip install pandas"
        ) from None

    import numpy as np

    rows = []
    for result in results:
        row: dict = {}

        if result.samples is None:
            # MAP: use point estimates
            for name, val in result.params.items():
                if name == "psd_xi":
                    continue
                if params is not None and name not in params:
                    continue
                row[f"{name}_value"] = float(np.mean(np.array(val)))
        else:
            # Sampling: use median + 68% CI
            for name, arr in result.samples.items():
                if name == "psd_xi":
                    continue
                if params is not None and name not in params:
                    continue
                arr_np = np.array(arr)
                if arr_np.ndim != 1:
                    continue
                row[f"{name}_median"] = float(np.median(arr_np))
                row[f"{name}_lo68"] = float(np.percentile(arr_np, 16))
                row[f"{name}_hi68"] = float(np.percentile(arr_np, 84))

        rows.append(row)

    return pd.DataFrame(rows)


# ── Convenient namespace aliases ──────────────────────────────────────
# Usage: from tengri import agn; agn.unified_nlr_blr(...)
# Or:    from tengri.agn import unified_nlr_blr
import sys

from tengri import components as _components

agn = _components.agn
dust = _components.dust
nebular = _components.nebular
sfh = _components.sfh
sps = _components.sps
igm = _components.igm
radio = _components.radio
xray = _components.xray

# Register module aliases for convenient short imports (Pattern 3: from tengri.agn import ...)
sys.modules["tengri.agn"] = agn
sys.modules["tengri.dust"] = dust
sys.modules["tengri.nebular"] = nebular
sys.modules["tengri.sfh"] = sfh
sys.modules["tengri.sps"] = sps
sys.modules["tengri.igm"] = igm
sys.modules["tengri.radio"] = radio
sys.modules["tengri.xray"] = xray

# Observation layer shortcut (already exists in imports above)
# observation module is imported separately below


__all__ = [
    # High-level API
    "AGEMAX_YR",
    # Registry
    "FIELD_MODEL_REGISTRY",
    "SFH_REGISTRY",
    "AGNConfig",
    # Exceptions
    "BackendError",
    "ConfigError",
    "DerivedQuantities",
    "DustConfig",
    "EmissionLines",
    "Fitter",
    "Fixed",
    "Gaussian",
    "HierarchicalFitter",
    "HierarchicalResult",
    "InferenceError",
    "LineCatalog",
    "LineFluxData",
    "LineList",
    "LogNormal",
    "LogUniform",
    "MockData",
    "Model",
    "ModelConfig",
    "MultiwavelengthConfig",
    "NebularConfig",
    "NoiseConfig",
    "NoiseModel",
    "Observation",
    "ParamSpec",
    "ParameterError",
    "Parameters",
    "Photometry",
    "PopulationFitter",
    "PopulationPosterior",
    "Posterior",
    "Prediction",
    "PriorPredictive",
    "SEDModel",
    "SEDQuantities",
    "SFHConfig",
    "SFHQuantities",
    "SSPData",
    "SpectralIndexData",
    "SpectralIndexDef",
    "Spectroscopy",
    "SpectroscopyConfig",
    "StudentT",
    "TengriError",
    "TengriIOError",
    "Uniform",
    "VIConfig",
    # Convenient namespace aliases for component access
    "agn",
    # SFH models
    "catalog_summary",
    "compute_effective_noise",
    "compute_field_gp",
    "compute_sqrt_power_drw",
    "compute_std_inv",
    "constant_sfh",
    "delayed_exponential_sfh",
    "delayed_tau",
    "dla_transmission",
    "dla_transmission_obs",
    "double_powerlaw",
    "dpl",
    "drw_acf",
    "drw_variance",
    "dust",
    "effective_metallicity",
    "exponential_sfh",
    "fit_batch",
    "gaussian_sfh",
    "generate_gp_batch",
    "generate_gp_fourier",
    "generate_mock",
    "gp_from_xi",
    "has_noise_model",
    "igm",
    "lnorm",
    "load_filter_set",
    "load_ssp_data",
    "lognormal_sfh",
    "make_log_age_grid",
    "nebular",
    "norm",
    "observation",
    "posteriors_to_dataframe",
    "psd_drw",
    "radio",
    "resolve_sfh",
    "sample_raytrace",
    "sfh",
    "skewnormal_sfh",
    "snorm",
    "sps",
    "triweight_burst",
    "truncated_skewnormal_sfh",
    "tsnorm",
    "two_component_dust",
    "uses_student_t",
    "variable_noise_hamiltonian",
    "xray",
]

# Plotting utilities
# Import observation module for namespace alias (already in imports above, adding as alias)
from tengri import observation
from tengri.analysis.plotting import (
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
