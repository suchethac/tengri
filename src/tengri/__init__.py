"""tengri: Differentiable SED fitting with IFT star formation history priors.

A modular, fully differentiable JAX pipeline:
PSD-governed GP → SFH → DSPS SED → photometry/spectroscopy.
"""

# Enable float64 — required for cosmological distance calculations
# (dL^2 at z>0.01 overflows float32)
import jax

jax.config.update("jax_enable_x64", True)

# Enable persistent XLA compilation cache. Universal speedup across
# notebook restarts, slurm tasks, benchmark workers — first compile is
# persisted to disk, every later process loads it in ~100 ms.
# Override location via TENGRI_JAX_CACHE_DIR; opt out via
# TENGRI_DISABLE_JAX_CACHE=1. See tengri.utils.jax_cache for details.
import logging as _logging

from tengri.utils.jax_cache import (
    cache_size_bytes,
    clear_cache,
    enable_persistent_cache,
    is_cache_enabled,
)

try:
    enable_persistent_cache()
except Exception as _cache_err:  # never break import
    _logging.getLogger(__name__).warning("Failed to enable persistent JAX cache: %s", _cache_err)

__version__ = "0.1.0"

# --- Exception hierarchy ---
# --- New high-level API ---
from tengri._logo import LOGO, LOGO_BANNER, print_logo
from tengri.analysis.mock import MockData, generate_mock
from tengri.citations import (
    Bibliography,
    Citation,
    citations_bibtex,
    citations_report,
    cite,
    cite_all,
    cites,
    collect_citations,
    paper_citation,
    print_bibtex,
    print_citations,
    print_paper_citation,
)
from tengri.components.dust.attenuation import two_component_dust
from tengri.components.igm.dla import dla_transmission, dla_transmission_obs
from tengri.components.sfh import (
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
    snorm_burst,
    snorm_burst_sfh,
    snorm_trunc_burst_sfh,
    spline_sfh,
    triweight_burst,
    truncated_skewnormal_sfh,
    tsnorm,
    tsnorm_burst,
)
from tengri.components.sfh.gp_sfh import (
    compute_sqrt_power_drw,
    generate_gp_batch,
    generate_gp_fourier,
    gp_from_xi,
    make_log_age_grid,
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
from tengri.config.exceptions import (
    BackendError,
    ConfigError,
    InferenceError,
    ParameterError,
    TengriError,
    TengriIOError,
)
from tengri.config.settings import (
    AGNConfig,
    DustConfig,
    MultiwavelengthConfig,
    NebularConfig,
    SEDModelConfig,
    SFHConfig,
)
from tengri.facade import Galaxy, doctor
from tengri.forward.convenience import catalog_summary, fit_batch
from tengri.forward.prediction import (
    DerivedQuantities,
    EmissionLines,
    Prediction,
    SEDQuantities,
    SFHQuantities,
)
from tengri.forward.result import SEDResult
from tengri.forward.sed_model import PriorPredictive, SEDModel
from tengri.inference.backends.mcmc.raytrace import sample_raytrace
from tengri.inference.catalog_fitter import CatalogFitter, CatalogPosterior
from tengri.inference.fitter import Fitter
from tengri.inference.hierarchical import (
    PopulationFitter,
    PopulationPosterior,
)
from tengri.inference.posterior import Posterior
from tengri.inference.vi_config import VIConfig
from tengri.observation.filters import load_filter_set
from tengri.observation.line_flux_data import LineFluxData
from tengri.observation.line_list import LineList
from tengri.observation.noise import (
    compute_effective_noise,
    compute_std_inv,
    exp_squared_kernel,
    gp_noise_covariance,
    has_noise_model,
    matern32_kernel,
    uses_student_t,
    variable_noise_hamiltonian,
)
from tengri.observation.noise_model import NoiseModel
from tengri.observation.observation import Observation
from tengri.observation.photometry_config import Photometry
from tengri.observation.spectral_indices import SpectralIndexData, SpectralIndexDef
from tengri.observation.spectroscopy import Spectroscopy
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed, Gaussian, LogNormal, LogUniform, StudentT, Uniform
from tengri.results import FitResult, Provenance
from tengri.utils import jit_logging


def posteriors_to_dataframe(results: list, params: list[str] | None = None):
    """Summarise a list of Posteriors into a pandas DataFrame.

    Requires ``pandas`` (``pip install pandas``).

    Parameters
    ----------
    results : list of Posterior
        Output of ``model.fit_batch()`` or any list of Posterior objects.
    params : list of str or None
        Parameter names to include. Default: all scalar free parameters,
        excluding ``psd_xi``.

    Returns
    -------
    pandas.DataFrame
        One row per galaxy, columns: ``{param}_median``, ``{param}_lo68``,
        ``{param}_hi68`` for each requested parameter.

    Notes
    -----
    **JIT-compatible**: no — pure Python, requires pandas library.

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

from tengri import components as _components, preprocessing, presets

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

# Filter discovery helpers
from tengri import filters as _filters_module

filters = _filters_module
sys.modules["tengri.filters"] = filters

# I/O layer
from tengri import io

sys.modules["tengri.io"] = io

# New namespace hierarchy (Phase 1, see docs/dev/api_migration_v0.x.md)
# These are pure re-exports: no behavioural change, just clearer locations.
#   tengri.cosmology — Planck18 + distance/age helpers (was tengri.utils.cosmology)
#   tengri.units     — F_nu/L_nu/AB-mag conversions (was utils.{conversions,magnitudes})
#   tengri.plot      — plotting helpers (was tengri.analysis.plotting)
from tengri import citations, cosmology, pipeline, plot, units

# Phase 6 (2026-05): the advertised top-level surface.
#
# This list is the *recommended* import paths. Everything in it is
# either a user-facing class/facade (Galaxy, SEDModel, Fitter, ...) or
# a subpackage namespace (tengri.sfh, tengri.dust, tengri.cosmology, ...).
# Implementation details (noise kernels, branding strings, individual
# citation helpers, single-purpose loaders) are no longer advertised
# but remain importable for backward compatibility — see
# docs/dev/api_migration_v0.x.md for the full story.
__all__ = [
    "AGNConfig",
    "BackendError",
    "CatalogFitter",
    "CatalogPosterior",
    "ConfigError",
    "DustConfig",
    "FitResult",
    "Fitter",
    "Fixed",
    "Galaxy",
    "Gaussian",
    "InferenceError",
    "LineFluxData",
    "LineList",
    "LogNormal",
    "LogUniform",
    "MockData",
    "NebularConfig",
    "NoiseModel",
    "Observation",
    "ParameterError",
    "Parameters",
    "Photometry",
    "PopulationFitter",
    "PopulationPosterior",
    "Posterior",
    "Provenance",
    "SEDModel",
    "SEDModelConfig",
    "SFHConfig",
    "SpectralIndexData",
    "SpectralIndexDef",
    "Spectroscopy",
    "StudentT",
    "TengriError",
    "TengriIOError",
    "Uniform",
    "VIConfig",
    "agn",
    "cache_size_bytes",
    "citations",
    "clear_cache",
    "cosmology",
    "doctor",
    "dust",
    "enable_persistent_cache",
    "filters",
    "generate_mock",
    "igm",
    "io",
    "is_cache_enabled",
    "nebular",
    "observation",
    "pipeline",
    "plot",
    "posteriors_to_dataframe",
    "preprocessing",
    "presets",
    "radio",
    "sfh",
    "sps",
    "units",
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
