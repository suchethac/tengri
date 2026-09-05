# SPDX-License-Identifier: BSD-3-Clause
"""tengri: Differentiable SED fitting with IFT star formation history priors.

A modular, fully differentiable JAX pipeline:
PSD-governed GP → SFH → DSPS SED → photometry/spectroscopy.

Where the public API lives
==========================

There are three surfaces, and they are deliberately not the same set. This
section used to claim ``__all__`` was a tiered map of everything below, which
was wrong in both directions: it advertised eleven names ``__all__`` excludes
and omitted sixty-five it contains (#1283).

**1. The canonical import path is the sub-namespace.** For the objects a fit is
made of, import from the layer that owns them::

    from tengri.observation import Photometry, Spectroscopy, NoiseModel, Observation
    from tengri.inference import Fitter, Posterior, VIConfig

These are also reachable as ``tengri.Photometry`` and appear in
``dir(tengri)``, so tab-completion finds them; that spelling is tolerated, not
canonical. See ``docs/dev/api_migration_v0.x.md``.

**2. ``__all__`` is the star-import surface**, kept deliberately narrower than
"everything public" so that ``from tengri import *`` does not dump the whole
package into a namespace. Names above are excluded on purpose: their absence
from ``import *`` is the design, not a defect.

**3. ``dir(tengri)`` is the discovery surface**, a curated subset sized for
tab-completion rather than completeness.

Building a first fit
====================

::

    import tengri
    from tengri.observation import Photometry, Observation, NoiseModel

    obs = Observation(photometry=Photometry.from_names([...]))
    sed = tengri.SEDModel.build(
        ssp_data=ssp, observation=obs, **tengri.recipes.star_forming_photometry()
    )
    fwd = tengri.ForwardModel.build(sed=sed, observation=obs)
    post = fwd.fit(flux, flux_err, method="vi")  # canonical entry point

``tengri.help()`` prints the same path with the discovery verbs alongside it.

What is in the package
======================

**Model construction:** ``SEDModel``, ``ForwardModel``, ``recipes``,
``builders``, ``Parameters``, ``parse_groups``, and the ``FREE``/``DEFAULT``
sentinels (``DEFAULT`` legal only as ``Fixed(DEFAULT)``) with the
seven distributions (``Uniform``, ``Gaussian``, ``Fixed``, ``LogUniform``,
``LogNormal``, ``StudentT``, ``Laplace``).

**Physics namespaces:** ``agn``, ``dust``, ``nebular``, ``stellar``, ``sfh``,
``sps``, ``igm``, ``radio``, ``xray``.

**Layer modules:** ``observation``, ``filters``, ``inference``, ``cosmology``,
``units``, ``plot``, ``citations``, ``config``, ``io``, ``pipeline``,
``presets``, ``preprocessing``, ``results``.

**Toolkit:** precompute configs (``WavePrecomp``, ``FeaturePrecomp``,
``SpectrumPrecomp``), population and spatial models, SSP loading, instruments,
line lists, spectral indices, ``fit_batch``, ``generate_mock``.

**Introspection:** ``help``, ``summary``, ``describe``, ``search``, ``explain``,
``examples``, ``tutorial``, ``doctor``, every ``list_*`` and ``describe_*``,
``cite_components``, ``suggest_parameters``.

**Exceptions:** ``TengriError`` and its subclasses ``ParameterError``,
``ConfigError``, ``BackendError``, ``InferenceError``, ``TengriIOError``.

``tengri.list_all()`` enumerates every registry live; prefer it to any list
written down here, which can only go stale.
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

# Enable float64 by DEFAULT: required for cosmological distance calculations
# (dL^2 at z>0.01 overflows float32).
#
# Default, not decree (#1840). This used to be an unconditional
# ``jax.config.update("jax_enable_x64", True)``, which silently discarded
# ``JAX_ENABLE_X64=0``: the documented JAX way to ask for float32. The
# environment and the live config then disagreed with no warning, so every
# float32 probe, benchmark and bug report that selected float32 that way ran
# in float64 and reported float32 as healthy. A measurement that cannot fail
# is not a measurement.
#
# JAX has already applied the variable by the time this module is imported, so
# honoring it is simply a matter of not overriding it. When the user has asked
# for float32 we say so once, because losing d_L^2 to overflow is a silent,
# catastrophic failure mode and nobody should meet it unwarned.
import jax

_X64_REQUEST = _os.environ.get("JAX_ENABLE_X64")

_x64_guard_installed = False


def _install_x64_guard() -> None:
    """Hold ``jax_enable_x64`` off for the duration of tengri's own import (#1880).

    Restoring the flag at the end of the import is not enough. Five DSPS modules
    flip x64 on early and transitively, and every module-scope ``jnp`` constant
    evaluated after that point is allocated as **float64**: 70 of them, tree
    wide. :func:`_reassert_x64_preference` puts the flag back but cannot
    un-allocate an array.

    On CPU that is only a doubled footprint: with x64 off, JAX's promotion caps
    results at float32 regardless of a float64 operand, so the numbers are
    unaffected. On a backend with **no** float64 the allocation itself raises
    (``MLX does not support float64``), and ``import tengri`` fails outright:
    measured on Apple MPS via ``jax-mps``.

    So while the user has explicitly asked for float32, refuse to turn x64 on at
    all. Only the enabling direction is blocked; anything turning it *off*, and
    every other config key, passes through untouched.
    """
    global _x64_guard_installed
    if _x64_guard_installed:
        return
    _original_update = jax.config.update

    def _guarded_update(name, value, *args, **kwargs):
        if name == "jax_enable_x64" and value:
            return  # the user asked for float32; keep the import in float32
        return _original_update(name, value, *args, **kwargs)

    jax.config.update = _guarded_update
    _guarded_update._tengri_original = _original_update  # type: ignore[attr-defined]
    _x64_guard_installed = True


def _remove_x64_guard() -> None:
    """Restore ``jax.config.update`` once tengri's import has finished."""
    global _x64_guard_installed
    if not _x64_guard_installed:
        return
    original = getattr(jax.config.update, "_tengri_original", None)
    if original is not None:
        jax.config.update = original
    _x64_guard_installed = False


if _X64_REQUEST is None:
    jax.config.update("jax_enable_x64", True)
elif _X64_REQUEST.strip().lower() in {"0", "false", "no", "off"}:
    import warnings as _warnings

    _install_x64_guard()

    _warnings.warn(
        f"JAX_ENABLE_X64={_X64_REQUEST}: tengri is honoring your request for "
        "float32 and is NOT enabling 64-bit precision. Cosmological distances "
        "are the known hazard (d_L^2 at z > 0.01 overflows float32) so any "
        "code that forms d_L^2 directly will produce inf. tengri's own "
        "projection avoids it by applying (1+z)/(4*pi*d_L^2) as a log10 "
        "offset, but third-party code may not. Unset JAX_ENABLE_X64 to "
        "restore the float64 default.",
        UserWarning,
        stacklevel=2,
    )
else:
    jax.config.update("jax_enable_x64", True)

if not _os.environ.get("TENGRI_VERBOSE_JAX"):
    try:
        from absl import logging as _absl_logging

        _absl_logging.set_verbosity(_absl_logging.ERROR)
        _absl_logging.set_stderrthreshold(_absl_logging.ERROR)
    except Exception:  # absl optional / not yet wired
        pass

# Enable persistent XLA compilation cache. Universal speedup across
# notebook restarts, slurm tasks, benchmark workers: first compile is
# persisted to disk, every later process loads it in ~100 ms.
# Override location via TENGRI_JAX_CACHE_DIR; opt out via
# TENGRI_DISABLE_JAX_CACHE=1. See tengri.utils.jax_cache for details.
import logging as _logging

from tengri.inference.jit_engine import clear_shared_caches, lean, persistent


def gc() -> None:
    """Drop tengri JIT caches + JAX caches + run Python GC.

    Shorthand for ``clear_shared_caches(drop_xla=True)``. Use between
    notebook cells when iterating on hyperparameters or model variants;
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

from tengri import builders, components as _components, preprocessing, presets, recipes
from tengri._completion import curated_dir
from tengri._data_setup import data_path, download_ssp, list_available_ssps, list_known_ssps
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
from tengri.components.dust.astrodust_hd23 import load_astrodust_hd23
from tengri.components.dust.attenuation import two_component_dust
from tengri.components.dust.draine2021_pah import (
    load_pahspec_draine2021,
    select_pahspec_axes,
)
from tengri.components.igm import (
    igm_transmission,
    igm_transmission_madau,
    igm_transmission_meiksin06,
)
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
from tengri.components.stellar.sfh.nonparametric import make_agebins_from_zred
from tengri.components.stellar.sfh.psd_models import drw_acf, drw_variance, psd_drw
from tengri.components.stellar.sfh.registry import (
    FIELD_MODEL_REGISTRY,
    SFH_REGISTRY,
    compute_field_gp,
    resolve_sfh,
)
from tengri.components.stellar.sps.dsps_wrapper import (
    SSPData,
    compute_surviving_mass,
    effective_metallicity,
    has_alpha_grid,
    interpolate_mass_remaining,
    interpolate_met_alpha,
    load_ssp,
    load_ssp_data,
    predict_surviving_mass,
    salaris_feh_from_mh,
    salaris_mh_from_feh,
)
from tengri.components.stellar.sps.mass_remaining import compute_mass_remaining_fraction
from tengri.config.exceptions import (
    BackendError,
    ConfigError,
    DeadFitError,
    InferenceError,
    ParameterError,
    TengriError,
    TengriIOError,
)
from tengri.facade import Galaxy, doctor
from tengri.observation.spectral_indices import (
    STANDARD_COMPOSITE_INDICES,
    STANDARD_INDICES,
    CompositeIndexDef,
    SpectralIndexData,
    SpectralIndexDef,
    measure_index_jax,
)
from tengri.observation.spectrum import apply_lsf, velocity_broaden


class _KernelsRemoved:
    """Stand-in raised on access of removed ``KernelStrategy`` / ``NoCompatibleKernelError``.

    The kernel-adapter family (``tengri.forward._kernels``) was removed in
    2026-05. Importing the old names from ``tengri`` returns this stand-in;
    any attempt to call, instantiate, or subscript it raises ImportError
    with a migration message.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def _raise(self, *_args, **_kwargs):
        raise ImportError(
            f"{self._name} was removed in Phase 6 (kernel adapter deletion). "
            "The structural-cache opt-in for fast photometry is now "
            "``approx=WavePrecomp(...)`` at build time; the JIT-safe "
            "forward path is ``model.predict_observables_jit(params)``. "
            f"{self._name} has no replacement: drop it."
        )

    __call__ = _raise
    __getitem__ = _raise
    __getattr__ = _raise


KernelStrategy = _KernelsRemoved("KernelStrategy")
NoCompatibleKernelError = _KernelsRemoved("NoCompatibleKernelError")
from tengri.components.sed_model_component import SEDModelComponent
from tengri.components.spatial import Exponential, FlatSlab, Sersic
from tengri.forward.convenience import catalog_summary, fit_batch
from tengri.forward.forward_model import ForwardModel
from tengri.forward.population import Population
from tengri.forward.population_sed_model import PopulationSEDModel
from tengri.forward.prediction import (
    DerivedQuantities,
    EmissionLines,
    Prediction,
    SEDQuantities,
    SFHQuantities,
)
from tengri.forward.result import SEDResult
from tengri.forward.sed_model import (
    FeaturePrecomp,
    PriorPredictive,
    SEDModel,
    SpectrumPrecomp,
    WavePrecomp,
)
from tengri.forward.spatial_model import SpatialModel, SpatialSEDModel
from tengri.inference.backends.mcmc.raytrace import sample_raytrace
from tengri.inference.bma import bma_resample, bma_weights
from tengri.observation.filters import (
    list_registered_filters,
    list_synthetic_bands,
    load_alma_band,
    load_custom_filter,
    load_filter,
    load_filter_from_dsps_file,
    load_filter_from_dsps_transmission_curve,
    load_filter_set,
    load_tophat_filter,
    register_filter,
    register_filter_from_file,
    unregister_filter,
)
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
from tengri.observation.photometry import FilterConvention, FilterCurve, list_filter_conventions
from tengri.parameters.groups import parse_groups
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import (
    Fixed,
    Gaussian,
    Laplace,
    LogNormal,
    LogUniform,
    StudentT,
    Uniform,
)
from tengri.parameters.registry import (
    ParameterRecord,
    describe_parameter,
    list_parameters,
    recipe_parameters,
)
from tengri.parameters.sentinels import DEFAULT, FREE
from tengri.protocols import ComponentIOError, DerivedKey, DerivedState, ForwardState
from tengri.utils import jit_logging
from tengri.utils.batching import vmap_chunked

agn = _components.agn
dust = _components.dust
nebular = _components.nebular
# sfh/sps were folded into stellar; the top-level convenience aliases
# continue to resolve to the canonical location.
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

# Filter discovery helpers (canonical path: tengri.observation.filters)
from tengri.observation import filters

sys.modules["tengri.filters"] = filters

# I/O layer
from tengri import io

sys.modules["tengri.io"] = io

# New namespace hierarchy (Phase 1+2, see docs/dev/api_migration_v0.x.md)
# These are pure re-exports: no behavioral change, just clearer locations.
#   tengri.cosmology: Planck18 + distance/age helpers
#   tengri.units: F_nu/L_nu/AB-mag conversions
#   tengri.plot: plotting helpers
#   tengri.results: FitResult, Posterior, MockData, generate_mock, ...
#   tengri.inference: Fitter, CatalogFitter, PopulationFitter, VIConfig, ...
#   tengri.config: *Config dataclasses, exceptions
#   tengri.observation: Photometry, Spectroscopy, NoiseModel, ...
#
# ``plot`` is deliberately absent: it is resolved lazily in ``__getattr__``
# below, because importing it pulls matplotlib into every ``import tengri``.
from tengri import (
    citations,
    config,
    cosmology,
    inference,
    measure,
    pipeline,
    results,
    units,
)

# Introspection façade: public registry lookups
from tengri._tutorials import examples, explain, tutorial
from tengri.registry import (
    cite_components,
    describe,
    describe_agn_block,
    describe_agn_model,
    describe_dust_emission_model,
    describe_dust_law,
    describe_inference_method,
    describe_nebular_backend,
    describe_property,
    describe_recipe,
    describe_sfh_model,
    help,
    list_age_kernels,
    list_agn_blocks,
    list_agn_models,
    list_all,
    list_components,
    list_dust_emission_models,
    list_dust_laws,
    list_dust_models,
    list_filters,
    list_igm_models,
    list_inference_methods,
    list_metallicity_modes,
    list_nebular_backends,
    list_plots,
    list_properties,
    list_radio_blocks,
    list_radio_models,
    list_recipes,
    list_sfh_models,
    list_shock_models,
    list_xray_models,
    print_components_bibtex,
    search,
    suggest_parameters,
    summary,
)

# The advertised top-level surface (api_migration_v0.x.md Phase 6, 2026-05).
#
# This list is the *recommended* import paths. Everything in it is
# either a user-facing class/facade (Galaxy, SEDModel, Fitter, ...) or
# a subpackage namespace (tengri.sfh, tengri.dust, tengri.cosmology, ...).
# Implementation details (noise kernels, branding strings, individual
# citation helpers, single-purpose loaders) are no longer advertised
# but remain importable for backward compatibility: see
# docs/dev/api_migration_v0.x.md for the full story.
# Top-level surface, sorted alphabetically (ruff RUF022).
# Buckets:
#   Core classes:    Galaxy, Parameters, SEDModel
#   Physics modules: agn, dust, igm, nebular, radio, sfh, sps, stellar, xray
#   Layer modules:   citations, config, cosmology, filters, inference, io,
#                    observation, pipeline, plot, preprocessing, presets,
#                    results, units
#   Registry verbs:  describe, help, list_*, summary
#   Runtime verbs:   clear_cache, doctor
#   Exceptions:      *Error, TengriIOError
#   Priors:          Fixed, Gaussian, LogNormal, LogUniform, StudentT, Uniform
#
# Cache / JIT machinery (gc, lean, persistent, clear_shared_caches,
# cache_size_bytes, enable_persistent_cache, is_cache_enabled) is
# intentionally NOT advertised. The one entry point at the top level is
# ``tengri.clear_cache()``; the rest live in ``tengri.utils.jax_cache``
# and ``tengri.inference.jit_engine`` for callers who need them.
# RUF022 (alphabetical __all__) is disabled on purpose: the list is grouped into
# the tiers documented in the module docstring, and sorting it would destroy the
# map. tests/contract/test_public_api_tiers.py enforces that the tiers partition
# __all__ exactly, which is the invariant that actually matters here.
__all__ = [  # noqa: RUF022
    # ========== Tier 1: CORE (user-facing classes for a first fit) ==========
    # Sentinels & distributions
    "FREE",
    "DEFAULT",
    "Fixed",
    "Gaussian",
    "Laplace",
    "LogNormal",
    "LogUniform",
    "StudentT",
    "Uniform",
    # Model construction & parameters
    "SEDModel",
    "PriorPredictive",
    "Parameters",
    "ParameterRecord",
    "parse_groups",
    # Observations: the instrument-schema family, re-promoted (#1338)
    "Observation",
    "Photometry",
    "Spectroscopy",
    "NoiseModel",
    "LineList",
    # Inference
    # Results
    # High-level facade
    "Galaxy",
    "doctor",
    # Core utilities
    "load_ssp",
    "SSPData",
    # ========== Tier 2: PHYSICS SUBMODULES (namespaces) ==========
    "agn",
    "dust",
    "nebular",
    "stellar",
    "sfh",
    "sps",
    "igm",
    "radio",
    "xray",
    "observation",
    "filters",
    # ========== Tier 3: TOOLKIT (analysis, batching, construction) ==========
    # Model builders
    "builders",
    "recipes",
    "register_component",
    # Bayesian model averaging
    "bma_resample",
    "bma_weights",
    # Utilities
    "measure",
    "vmap_chunked",
    "FeaturePrecomp",
    "WavePrecomp",
    "SpectrumPrecomp",
    # Population & spatial models
    "Population",
    "PopulationSEDModel",
    "SpatialModel",
    "SpatialSEDModel",
    "ForwardModel",
    "SEDResult",
    # Spatial components
    "Exponential",
    "FlatSlab",
    "Sersic",
    # Data loading
    "download_ssp",
    "list_available_ssps",
    "list_known_ssps",
    "load_ssp_data",
    "data_path",
    # Components & physics
    "FilterConvention",
    "Data",
    "CompositeIndexDef",
    "SpectralIndexDef",
    "SpectralIndexData",
    # Specialized loaders
    "load_astrodust_hd23",
    "load_pahspec_draine2021",
    "apply_lsf",
    "velocity_broaden",
    "igm_transmission",
    "igm_transmission_madau",
    "igm_transmission_meiksin06",
    # Spectral indices
    "measure_index_jax",
    "STANDARD_INDICES",
    "STANDARD_COMPOSITE_INDICES",
    # Noise kernels & utilities
    "exp_squared_kernel",
    "gp_noise_covariance",
    "matern32_kernel",
    "fit_batch",
    # Catalog fitting: the astronomer-facing noun (#1317)
    "Catalog",
    "compute_mass_remaining_fraction",
    "recipe_parameters",
    # ========== Tier 4: INTROSPECTION (discovery & diagnostics) ==========
    "help",
    "summary",
    "describe",
    "search",
    "explain",
    "examples",
    "tutorial",
    # Component discovery
    "list_age_kernels",
    "list_agn_blocks",
    "list_agn_models",
    "list_all",
    "list_components",
    "list_dust_emission_models",
    "list_dust_laws",
    "list_dust_models",
    "list_filter_conventions",
    "list_filters",
    "list_registered_filters",
    "list_synthetic_bands",
    "load_alma_band",
    "load_custom_filter",
    "load_filter_from_dsps_file",
    "load_filter_from_dsps_transmission_curve",
    "load_tophat_filter",
    "register_filter",
    "register_filter_from_file",
    "unregister_filter",
    "list_igm_models",
    "list_inference_methods",
    "list_metallicity_modes",
    "list_nebular_backends",
    "list_parameters",
    "list_plots",
    "list_properties",
    "list_radio_blocks",
    "list_radio_models",
    "list_recipes",
    "list_sfh_models",
    "list_shock_models",
    "list_xray_models",
    # Component description
    "describe_agn_block",
    "describe_agn_model",
    "describe_dust_emission_model",
    "describe_dust_law",
    "describe_inference_method",
    "describe_nebular_backend",
    "describe_parameter",
    "describe_property",
    "describe_recipe",
    "describe_sfh_model",
    # Citations & help
    "cite_components",
    "print_components_bibtex",
    "suggest_parameters",
    # ========== Tier 5: EXCEPTIONS ==========
    "TengriError",
    "ParameterError",
    "ConfigError",
    "BackendError",
    "InferenceError",
    "DeadFitError",
    "TengriIOError",
    # ========== Layer modules (optional imports, see docs) ==========
    "citations",
    "config",
    "cosmology",
    "inference",
    "io",
    "pipeline",
    "plot",
    "preprocessing",
    "presets",
    "results",
    "units",
    # ========== Utilities (advanced / cache management) ==========
    "clear_cache",
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
# Plotting utilities
# Import observation module for namespace alias (already in imports above, adding as alias)
from tengri import observation

# NOTE: the plotting names that used to be imported here are now resolved
# lazily in ``__getattr__`` (see ``_LAZY_PLOTTING``). They remain in
# ``__all__`` and ``tengri.<name>`` still works; only the *timing* changed.
from tengri.inference.catalog import Catalog
from tengri.inference.catalog_fitter import CatalogFitter
from tengri.inference.fitter import Fitter
from tengri.inference.hierarchical import PopulationFitter
from tengri.inference.information import ParameterInformation, parameter_information
from tengri.inference.vi_config import VIConfig
from tengri.observation.data import Data
from tengri.observation.instrument import Instrument, list_instruments
from tengri.observation.line_list import LineList
from tengri.observation.noise_model import NoiseModel
from tengri.observation.observation import Observation
from tengri.observation.photometry_config import Photometry
from tengri.observation.spectroscopy import Spectroscopy
from tengri.results import (
    CatalogPosterior,
    FitRecord,
    FitResult,
    MockData,
    PopulationPosterior,
    Posterior,
    generate_mock,
    posteriors_to_dataframe,
)

# ──────────────────────────────────────────────────────────────────
# Curated tab-completion surface.
#
# `tengri.<TAB>` should give a fresh user ~30 obvious entry points,
# not the 175-item kitchen sink of every public symbol.  Everything
# remains accessible via attribute access: only `dir(tengri)` is
# filtered.  `__all__` (above) keeps `from tengri import *` working
# unchanged.
# ──────────────────────────────────────────────────────────────────
_CURATED_DIR = (
    # 1.  First-impression discovery
    "help",
    "summary",
    "describe",
    "list_age_kernels",
    "list_agn_blocks",
    "list_agn_models",
    "list_dust_emission_models",
    "list_dust_laws",
    "list_dust_models",
    "list_igm_models",
    "list_radio_models",
    "list_radio_blocks",
    "list_shock_models",
    "list_sfh_models",
    "list_metallicity_modes",
    "list_nebular_backends",
    "list_xray_models",
    "list_filters",
    "list_plots",
    "list_components",
    "list_inference_methods",
    "list_recipes",
    "list_all",
    "search",
    "suggest_parameters",
    "cite_components",
    "print_components_bibtex",
    "describe_agn_block",
    "describe_recipe",
    "tutorial",
    "examples",
    "explain",
    # 2.  Build a fit
    "load_ssp",
    "load_ssp_data",
    "download_ssp",
    "list_known_ssps",
    "SSPData",
    "Parameters",
    "parse_groups",
    "SEDModel",
    "ForwardModel",
    "builders",
    "recipes",
    "FeaturePrecomp",
    "WavePrecomp",
    # ``Fitter`` is deliberately absent: it is the cache-reuse mechanism, not a
    # taught noun (api_migration_v0.x.md). It stays importable: no public API
    # is removed: it just must not be what tab-completion suggests first. The
    # canonical multi-galaxy entry point is ``Catalog``, below (#1455).
    "Catalog",
    "fit_batch",
    "Observation",
    "Photometry",
    "Spectroscopy",
    "NoiseModel",
    "LineList",
    "Instrument",
    "list_instruments",
    # 3.  Priors / distributions (+ the FREE/DEFAULT build sentinels)
    "FREE",
    "DEFAULT",
    "Uniform",
    "Gaussian",
    "LogUniform",
    "LogNormal",
    "Fixed",
    "StudentT",
    # 4.  Result types (+ hierarchical / spatial model classes)
    "Posterior",
    "ParameterInformation",
    "parameter_information",
    "PopulationSEDModel",
    # ``PopulationFitter`` is deliberately absent too, and the reason is worth
    # stating because two documents look like they disagree. It is the
    # canonical *name* for the class (NAMING_CONTRACT, vs the retired
    # ``HierarchicalFitter``), AND its taught construction form
    # ``PopulationFitter(model_factory, galaxies, ...)`` raises a
    # DeprecationWarning pointing at ForwardModel + PopulationSEDModel (#211,
    # #1319). Both are true at different levels; a constructor that warns is
    # not a fresh-user entry point.
    "PopulationPosterior",
    "SpatialSEDModel",
    # 5.  Convenience
    "generate_mock",
    "doctor",
    "cite",
    "print_citations",
    "print_logo",
    "register_component",
    "cosmology",
    "units",
    "plot",
    "__version__",
)


__dir__ = curated_dir(_CURATED_DIR)


# Backward-compatibility shims for renamed public symbols. Each old name
# resolves once, emits a DeprecationWarning pointing at the new name, and
# then forwards. Will be removed in v1.0.
_RENAMED_SYMBOLS = {
    "Provenance": ("FitRecord", "tengri.FitRecord"),
    "DerivedBundle": ("DerivedState", "tengri.protocols.DerivedState"),
    "PipelineContractError": ("ComponentIOError", "tengri.protocols.ComponentIOError"),
}

# Config dataclasses demoted from the top-level namespace (2026-07, #887):
# the nested-dict grammar (``SEDModel.build``) is the one construction
# surface; the dataclasses are internal lowering artifacts. Still importable
# from ``tengri.config`` without a warning for internal/expert use.
_DEMOTED_CONFIGS = frozenset(
    {"AGNConfig", "DustConfig", "NebularConfig", "SEDModelConfig", "SFHConfig"}
)

# Plotting names resolved on first access rather than at import (#1852).
#
# Importing them eagerly pulled matplotlib into *every* ``import tengri``,
# including inference runs, CI shards and slurm tasks that never draw
# anything. Measured on a clean install: ``import tengri`` 2.43 s, of which
# ``import matplotlib.pyplot`` alone is 0.77 s -- 32%, paid by everyone.
#
# This does not shrink the install. matplotlib stays a hard dependency
# because nifty8 requires it, so pip fetches it either way; what changes is
# when it is loaded. ``tengri.plot_sed_fit`` and ``tengri.plot`` behave
# exactly as before, one attribute lookup later.
_LAZY_PLOTTING = frozenset(
    {
        "COLORS",
        "SDSS_WAVE_EFF",
        "SPECTRAL_FEATURES",
        "diagnostics_table",
        "plot_corner_comparison",
        "plot_sed_fit",
        "plot_sfh",
        "plot_sfh_comparison",
        "plot_spectrum_fit",
        "safe_corner",
        "setup_style",
    }
)


def __getattr__(name: str) -> object:
    if name == "plot":
        import tengri.plot as _plot

        globals()["plot"] = _plot  # cache: later lookups skip __getattr__
        return _plot
    if name in _LAZY_PLOTTING:
        import tengri.analysis.plotting as _plotting

        obj = getattr(_plotting, name)
        globals()[name] = obj
        return obj
    if name in _RENAMED_SYMBOLS:
        new_name, new_path = _RENAMED_SYMBOLS[name]
        from tengri._deprecated import deprecated_attribute

        new_obj = globals()[new_name]
        return deprecated_attribute(new_obj, old_name=f"tengri.{name}", new_name=new_path)
    if name in _DEMOTED_CONFIGS:
        import warnings

        from tengri.config import settings as _settings

        warnings.warn(
            f"tengri.{name} is internal: build models with the nested-dict "
            f"grammar (SEDModel.build(...)) instead. For expert use import it "
            f"from tengri.config ({name} stays there without a warning). "
            f"Top-level access will be removed in tengri v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(_settings, name)
    raise AttributeError(f"module 'tengri' has no attribute {name!r}")


def _reassert_x64_preference() -> None:
    """Restore an explicit ``JAX_ENABLE_X64=0`` that an import clobbered (#1840).

    Six DSPS modules run a bare ``jax.config.update("jax_enable_x64", True)`` at
    import time (``dsps.sed.__init__``, ``ssp_weights``, ``stellar_sed``,
    ``stellar_age_weights``, ``metallicity_weights``, ``dust.blackbody``). They
    are pulled in transitively by tengri's own imports, so declining to force
    x64 at the top of this module is necessary but not sufficient: by the time
    the package finishes importing, the flag is on again regardless.

    This runs last, after every transitive import has had its say, and puts the
    flag back where the user asked for it. Measured before the fix:
    ``JAX_ENABLE_X64=0 python -c "import tengri; ..."`` reported
    ``jax_enable_x64 = True`` and a ``float64`` default dtype.

    A DSPS module imported *lazily* later: several tengri functions import
    from ``dsps`` inside the function body: can still flip the flag mid-run.
    That is upstream behavior this package cannot intercept; the supported
    route for a float32 session remains
    ``jax.config.update("jax_enable_x64", False)`` immediately before the work,
    which is what :func:`tengri.utils.devices.setup_jax` does, and what the
    float32 tests use via the ``jax.enable_x64(False)`` context manager.
    """
    request = _os.environ.get("JAX_ENABLE_X64")
    if request is None or request.strip().lower() not in {"0", "false", "no", "off"}:
        return
    import jax as _jax

    if _jax.config.jax_enable_x64:
        _jax.config.update("jax_enable_x64", False)


# Order matters: drop the guard FIRST so the reassert below can still call
# through to the real jax.config.update, then restore the flag (#1880, #1840).
_remove_x64_guard()
_reassert_x64_preference()
