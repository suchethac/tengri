"""tengri: Differentiable SED fitting with IFT star formation history priors.

A modular, fully differentiable JAX pipeline:
PSD-governed GP → SFH → DSPS SED → photometry/spectroscopy.
"""

# Silence the JAX/absl/XLA C++ chatter that prints on first JAX touch:
#
#   "Empty bitcode string provided for eigen. Optimizations relying on …"
#   "Assume version compatibility. PjRt-IFRT does not track …"
#
# These come from absl logging at min level WARNING, fired by routine
# JAX subsystem initialization. Setting the env var BEFORE importing
# jax suppresses them; setting absl's stderr_threshold afterwards
# catches anything that slips through. Users who want them back can
# set ``TENGRI_VERBOSE_JAX=1``.
import os as _os

if not _os.environ.get("TENGRI_VERBOSE_JAX"):
    _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    _os.environ.setdefault("ABSL_LOG_LEVEL", "ERROR")

# Enable float64 — required for cosmological distance calculations
# (dL^2 at z>0.01 overflows float32)
import jax

jax.config.update("jax_enable_x64", True)

if not _os.environ.get("TENGRI_VERBOSE_JAX"):
    try:
        from absl import logging as _absl_logging

        _absl_logging.set_verbosity(_absl_logging.ERROR)
        _absl_logging.set_stderrthreshold(_absl_logging.ERROR)
    except Exception:  # absl optional / not yet wired
        pass

# Enable persistent XLA compilation cache. Universal speedup across
# notebook restarts, slurm tasks, benchmark workers — first compile is
# persisted to disk, every later process loads it in ~100 ms.
# Override location via TENGRI_JAX_CACHE_DIR; opt out via
# TENGRI_DISABLE_JAX_CACHE=1. See tengri.utils.jax_cache for details.
import logging as _logging

from tengri.inference.jit_engine import clear_shared_caches, lean, persistent


def gc() -> None:
    """Drop tengri JIT caches + JAX caches + run Python GC.

    Shorthand for ``clear_shared_caches(drop_xla=True)``. Use between
    notebook cells when iterating on hyperparameters or model variants —
    each ``jax.jit`` over a new closure leaks compile artefacts that
    ``Fitter.run(lean=True)`` does not see.

    Examples
    --------
    >>> for tau_gyr in (0.1, 0.3, 1.0, 3.0):
    ...     spec = make_spec(tau_gyr)
    ...     result = fit(spec)
    ...     tengri.gc()  # one line of cleanup between configurations
    """
    clear_shared_caches(drop_xla=True)


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

# Test/dev escape hatch: disable JIT globally so each test pays trace-only cost
# instead of full compile cost. Useful when running large pytest suites where
# per-test compile dominates wallclock (each test cold-compiles a fresh fused
# kernel). Tests that explicitly assert JIT-traceability still work because
# ``jax.jit`` becomes a transparent no-op rather than disappearing.
#
# Usage:
#     TENGRI_DISABLE_JIT=1 pytest tests/ -q
#
# Caveat: a few tests may surface latent eager-mode bugs (closures over Python
# floats that previously got promoted by tracers). Treat those as real findings.
if _os.environ.get("TENGRI_DISABLE_JIT", "").lower() in ("1", "true", "yes"):
    jax.config.update("jax_disable_jit", True)

__version__ = "0.1.0"

# --- Exception hierarchy ---
# --- New high-level API ---
# ── Convenient namespace aliases ──────────────────────────────────────
# Usage: from tengri import agn; agn.unified_nlr_blr(...)
# Or:    from tengri.agn import unified_nlr_blr
import sys

from tengri import components as _components, preprocessing, presets, recipes
from tengri._data_setup import download_ssp, list_known_ssps
from tengri._logo import LOGO, LOGO_BANNER, print_logo
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
from tengri.components import register_component
from tengri.components.dust.attenuation import two_component_dust
from tengri.components.igm.dla import dla_transmission, dla_transmission_obs
from tengri.components.stellar.sfh import (
    AGEMAX_YR,
    constant,
    delayed_exponential,
    delayed_tau,
    double_powerlaw,
    dpl,
    exponential,
    gaussian,
    lnorm,
    lognormal,
    norm,
    skewnormal,
    snorm,
    snorm_burst,
    snorm_trunc_burst,
    spline,
    triweight_burst,
    truncated_skewnormal,
    tsnorm,
    tsnorm_burst,
)
from tengri.components.stellar.sfh.gp_sfh import (
    compute_sqrt_power_drw,
    generate_gp_batch,
    generate_gp_fourier,
    gp_from_xi,
    make_log_age_grid,
)
from tengri.components.stellar.sfh.psd_models import drw_acf, drw_variance, psd_drw
from tengri.components.stellar.sfh.registry import (
    FIELD_MODEL_REGISTRY,
    SFH_REGISTRY,
    compute_field_gp,
    resolve_sfh,
)
from tengri.components.stellar.sps.dsps_wrapper import (
    SSPData,
    effective_metallicity,
    has_alpha_grid,
    interpolate_met_alpha,
    load_ssp,
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
from tengri.facade import Galaxy, doctor
from tengri.forward._kernels import (
    COMPOSITIONAL_ONLY as COMPOSITIONAL_ONLY_KERNEL_STRATEGY,
    DEFAULT as DEFAULT_KERNEL_STRATEGY,
    EXACT_ONLY as EXACT_ONLY_KERNEL_STRATEGY,
    LOW_MEMORY as LOW_MEMORY_KERNEL_STRATEGY,
    KernelStrategy,
    NoCompatibleKernelError,
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
from tengri.forward.sed_model import PriorPredictive, SEDModel
from tengri.inference.backends.mcmc.raytrace import sample_raytrace
from tengri.observation.filters import load_filter_set
from tengri.observation.noise import (
    PoissonNoiseLikelihood,
    StudentTLikelihood,
    compute_effective_noise,
    compute_std_inv,
    exp_squared_kernel,
    gp_noise_covariance,
    has_noise_model,
    matern32_kernel,
    uses_student_t,
    variable_noise_hamiltonian,
)
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed, Gaussian, LogNormal, LogUniform, StudentT, Uniform
from tengri.parameters.sentinels import FIXED, FREE
from tengri.utils import jit_logging

agn = _components.agn
dust = _components.dust
nebular = _components.nebular
# sfh/sps were folded into stellar in Phase II-2.1; the top-level
# convenience aliases continue to resolve to the canonical location.
sfh = _components.stellar.sfh
sps = _components.stellar.sps
stellar = _components.stellar
igm = _components.igm
radio = _components.radio
xray = _components.xray

# Register module aliases for convenient short imports (Pattern 3: from tengri.agn import ...)
sys.modules["tengri.agn"] = agn
sys.modules["tengri.dust"] = dust
sys.modules["tengri.nebular"] = nebular
sys.modules["tengri.sfh"] = sfh
sys.modules["tengri.sps"] = sps
sys.modules["tengri.stellar"] = stellar
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

# New namespace hierarchy (Phase 1+2, see docs/dev/api_migration_v0.x.md)
# These are pure re-exports: no behavioural change, just clearer locations.
#   tengri.cosmology — Planck18 + distance/age helpers
#   tengri.units     — F_nu/L_nu/AB-mag conversions
#   tengri.plot      — plotting helpers
#   tengri.results   — FitResult, Posterior, MockData, generate_mock, ...
#   tengri.inference — Fitter, CatalogFitter, PopulationFitter, VIConfig, ...
#   tengri.config    — *Config dataclasses, exceptions
#   tengri.observation — Photometry, Spectroscopy, NoiseModel, ...
from tengri import (
    citations,
    config,
    cosmology,
    inference,
    pipeline,
    plot,
    results,
    units,
)

# Introspection façade — public registry lookups
from tengri._tutorials import examples, explain, tutorial
from tengri.registry import (
    cite_components,
    describe,
    help,
    list_agn_models,
    list_all,
    list_components,
    list_dust_emission_models,
    list_dust_laws,
    list_filters,
    list_inference_methods,
    list_nebular_backends,
    list_plots,
    list_sfh_models,
    print_components_bibtex,
    search,
    suggest_parameters,
    summary,
)

# Phase 6 (2026-05): the advertised top-level surface.
#
# This list is the *recommended* import paths. Everything in it is
# either a user-facing class/facade (Galaxy, SEDModel, Fitter, ...) or
# a subpackage namespace (tengri.sfh, tengri.dust, tengri.cosmology, ...).
# Implementation details (noise kernels, branding strings, individual
# citation helpers, single-purpose loaders) are no longer advertised
# but remain importable for backward compatibility — see
# docs/dev/api_migration_v0.x.md for the full story.
# Top-level surface, sorted alphabetically (ruff RUF022).
# Buckets:
#   Core classes:    Galaxy, Parameters, SEDModel
#   Physics modules: agn, dust, igm, nebular, radio, sfh, sps, stellar, xray
#   Layer modules:   citations, config, cosmology, filters, inference, io,
#                    observation, pipeline, plot, preprocessing, presets,
#                    results, units
#   Registry verbs:  describe, help, list_*, summary
#   Runtime verbs:   cache_size_bytes, clear_cache, doctor,
#                    enable_persistent_cache, is_cache_enabled
#   Exceptions:      *Error, TengriIOError
#   Priors:          Fixed, Gaussian, LogNormal, LogUniform, StudentT, Uniform
__all__ = [
    "BackendError",
    "ConfigError",
    "Fixed",
    "Galaxy",
    "Gaussian",
    "InferenceError",
    "LogNormal",
    "LogUniform",
    "ParameterError",
    "Parameters",
    "SEDModel",
    "StudentT",
    "TengriError",
    "TengriIOError",
    "Uniform",
    "agn",
    "cache_size_bytes",
    "citations",
    "cite_components",
    "clear_cache",
    "clear_shared_caches",
    "config",
    "cosmology",
    "describe",
    "doctor",
    "download_ssp",
    "dust",
    "enable_persistent_cache",
    "examples",
    "explain",
    "filters",
    "gc",
    "help",
    "igm",
    "inference",
    "io",
    "is_cache_enabled",
    "lean",
    "list_agn_models",
    "list_all",
    "list_components",
    "list_dust_emission_models",
    "list_dust_laws",
    "list_filters",
    "list_inference_methods",
    "list_known_ssps",
    "list_nebular_backends",
    "list_plots",
    "list_sfh_models",
    "nebular",
    "observation",
    "persistent",
    "pipeline",
    "plot",
    "preprocessing",
    "presets",
    "radio",
    "register_component",
    "results",
    "search",
    "sfh",
    "sps",
    "stellar",
    "summary",
    "tutorial",
    "units",
    "xray",
]


# ──────────────────────────────────────────────────────────────────
# Re-export the primary user-facing classes at the top level.
#
# These objects are advertised in ``tengri.__dir__`` and ``tengri.help()``
# as the main entry points, so they must be importable as
# ``tengri.Fitter`` / ``tengri.Posterior`` / ``tengri.Photometry`` etc.
# **without** triggering a DeprecationWarning.  The canonical paths
# (``tengri.inference.Fitter``, ``tengri.results.Posterior``, …) remain
# valid; this is just an additional re-export, not a relocation.
# ──────────────────────────────────────────────────────────────────
from tengri.config.settings import (
    AGNConfig,
    DustConfig,
    NebularConfig,
    SEDModelConfig,
    SFHConfig,
)
from tengri.inference.catalog_fitter import CatalogFitter
from tengri.inference.fitter import Fitter
from tengri.inference.hierarchical import PopulationFitter
from tengri.inference.vi_config import VIConfig
from tengri.observation.instrument import Instrument, list_instruments
from tengri.observation.line_list import LineList
from tengri.observation.noise_model import NoiseModel
from tengri.observation.observation import Observation
from tengri.observation.photometry_config import Photometry
from tengri.observation.spectroscopy import Spectroscopy
from tengri.results import (
    CatalogPosterior,
    FitResult,
    MockData,
    PopulationPosterior,
    Posterior,
    Provenance,
    generate_mock,
    posteriors_to_dataframe,
)

# Internal-only relocation shim — names that genuinely moved and still
# warn when accessed via the old path. The user-facing API above is
# *not* in this dict, so accessing those is silent and supported.
_RELOCATED: dict[str, tuple[str, str]] = {
    "LineFluxData": ("tengri.observation", "LineFluxData"),
    "SpectralIndexDef": ("tengri.observation", "SpectralIndexDef"),
    "SpectralIndexData": ("tengri.observation", "SpectralIndexData"),
}


def __getattr__(name: str):
    """Resolve relocated symbols with a DeprecationWarning (PEP 562)."""
    if name in _RELOCATED:
        import importlib

        from tengri._deprecated import deprecated_attribute

        module_path, attr = _RELOCATED[name]
        value = getattr(importlib.import_module(module_path), attr)
        return deprecated_attribute(
            value,
            old_name=f"tengri.{name}",
            new_name=f"{module_path}.{attr}",
        )
    # Set ``name`` and ``obj`` so Python's built-in "Did you mean: …"
    # suggestion (PEP 657, fired when formatting the traceback) kicks in
    # against the curated ``__dir__``. Without these attributes a custom
    # __getattr__ swallows the suggestion mechanism.
    import sys as _sys

    raise AttributeError(
        f"module 'tengri' has no attribute {name!r}",
        name=name,
        obj=_sys.modules[__name__],
    )


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

# ──────────────────────────────────────────────────────────────────
# Curated tab-completion surface.
#
# `tengri.<TAB>` should give a fresh user ~30 obvious entry points,
# not the 175-item kitchen sink of every public symbol.  Everything
# remains accessible via attribute access — only `dir(tengri)` is
# filtered.  `__all__` (above) keeps `from tengri import *` working
# unchanged.
# ──────────────────────────────────────────────────────────────────
_CURATED_DIR = (
    # 1.  First-impression discovery
    "help",
    "summary",
    "describe",
    "list_agn_models",
    "list_dust_emission_models",
    "list_dust_laws",
    "list_sfh_models",
    "list_nebular_backends",
    "list_filters",
    "list_plots",
    "list_components",
    "list_inference_methods",
    "list_all",
    "search",
    "suggest_parameters",
    "cite_components",
    "tutorial",
    "examples",
    "explain",
    # 2.  Build a fit
    "Parameters",
    "SEDModel",
    "Fitter",
    "Observation",
    "Photometry",
    "Spectroscopy",
    "NoiseModel",
    "LineList",
    "Instrument",
    "list_instruments",
    # 3.  Priors / distributions
    "Uniform",
    "Gaussian",
    "LogUniform",
    "LogNormal",
    "Fixed",
    "StudentT",
    # 4.  Result types
    "Posterior",
    "PopulationFitter",
    "PopulationPosterior",
    # 5.  Convenience
    "generate_mock",
    "doctor",
    "cite",
    "print_citations",
    "print_logo",
    "register_component",
    "__version__",
)


def __dir__() -> list[str]:
    """Curated tab-completion list — keeps `tengri.<TAB>` readable.

    Everything else remains accessible via attribute access; only the
    completion surface is trimmed.
    """
    return list(_CURATED_DIR)
