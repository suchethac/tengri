# SPDX-License-Identifier: BSD-3-Clause
"""Tests guarding the shape of the public API surface.

These tests catch accidental top-level pollution and accidental removal
of public symbols. They are intentionally permissive at Phase 1 — the
allowed-list shrinks in later phases as deprecation shims are added and
old top-level names move into subpackages.

If you intentionally add or remove a public symbol, update both
``tengri.__all__`` AND the ``ALLOWED_TOP_LEVEL`` set below in the same
commit, with a corresponding entry in ``docs/dev/api_migration_v0.x.md``.
"""

from __future__ import annotations

import pytest

import tengri

pytestmark = pytest.mark.contract
# Names allowed in tengri.__all__ as of Phase 2 (2026-05).
#
# Phase 2 moved result classes, observation classes, fitters, and config
# dataclasses into sub-namespaces (`tengri.results`, `tengri.observation`,
# `tengri.inference`, `tengri.config`). Old top-level names still resolve
# via a `__getattr__` deprecation shim — they belong in
# DEMOTED_BUT_IMPORTABLE below, not here.
#
# Adding a new top-level symbol requires editing this set AND adding a
# row in `docs/dev/api_migration_v0.x.md`.
ALLOWED_TOP_LEVEL: frozenset[str] = frozenset(
    {
        # ── Core classes ────────────────────────────────────────────
        "ForwardModel",
        "Galaxy",
        "Parameters",
        "parse_groups",
        "Population",
        "PopulationSEDModel",
        "SEDModel",
        "SpatialModel",
        "SpatialSEDModel",
        "WavePrecomp",
        "SpectrumPrecomp",
        "FeaturePrecomp",
        # ── Object model — the measurement record (#1321) ───────────
        # The razor: Observation is the instrument schema, Data the
        # per-galaxy record it is validated against at fit(). Data and the
        # instrument-schema family are both advertised top-level (#1338).
        "Data",
        # ── Object model — the instrument-schema family, re-promoted (#1338) ──
        "Observation",
        "Photometry",
        "Spectroscopy",
        "NoiseModel",
        "LineList",
        # ── Catalog fitting — the astronomer-facing noun (#1317) ────
        "Catalog",
        # ── Spatial profile components ──────────────────────────────
        "Exponential",
        "FilterConvention",
        "FlatSlab",
        "Sersic",
        # ── SEDComponent extension surface — demoted; see DEMOTED_BUT_IMPORTABLE
        # ── Priors (parameters/) ─────────────────────────────────────
        "Fixed",
        "Gaussian",
        "Laplace",
        "LogNormal",
        "LogUniform",
        "StudentT",
        "Uniform",
        # ── Sentinels (parameters/sentinels.py) ──────────────────────
        # Used by every recipe and 100+ tests via `from tengri import DEFAULT, FREE`.
        # The FIXED sentinel was removed; `Fixed(DEFAULT)` replaces it.
        "DEFAULT",
        "FREE",
        # ── Forward-model outputs and helpers ────────────────────────
        "PriorPredictive",
        "SEDResult",
        # ── Top-level convenience verbs (additional) ─────────────────
        "fit_batch",
        # ── Data discovery helpers ───────────────────────────────────
        "data_path",
        # ── Exceptions ──────────────────────────────────────────────
        "BackendError",
        "ConfigError",
        "InferenceError",
        "ParameterError",
        "TengriError",
        "TengriIOError",
        # ── Cache helpers ───────────────────────────────────────────
        # Only the single entry point is advertised; the rest live in
        # tengri.utils.jax_cache / tengri.inference.jit_engine.
        "clear_cache",
        # ── Bayesian model averaging ────────────────────────────────
        "bma_resample",
        "bma_weights",
        # ── Top-level convenience verbs ─────────────────────────────
        "cite_components",
        "doctor",
        "register_component",
        "search",
        # ── SSP data setup ──────────────────────────────────────────
        "download_ssp",
        "list_available_ssps",
        "list_known_ssps",
        # ── SSP loaders (closes #496) ───────────────────────────────
        "load_ssp",
        "load_ssp_data",
        "SSPData",
        # ── Dust-emission template loaders (closes #803) ────────────
        "load_astrodust_hd23",
        "load_pahspec_draine2021",
        # ── Component helpers (closes #497 / #498) ──────────────────
        # The BAGPIPES reproduction notebook (PR #493) needed direct
        # callable access to these — tying them to a full SEDModel
        # build for a curve-only check is unnecessary friction.
        "igm_transmission",
        "igm_transmission_madau",
        "igm_transmission_meiksin06",
        "velocity_broaden",
        "apply_lsf",
        # ── GP-noise kernels + spectral indices (closes #511) ───────
        "exp_squared_kernel",
        "matern32_kernel",
        "gp_noise_covariance",
        "SpectralIndexDef",
        "SpectralIndexData",
        "STANDARD_INDICES",
        "measure_index_jax",
        # ── Model-free measurement namespace (#1047, API Phase 3) ───
        # The astronomer-facing façade over the measurement engines. The bare
        # ``measure_index_jax`` above stays importable for one deprecation
        # cycle (Phase 5, #1049) — ``tengri.measure.spectral_index`` supersedes it.
        "measure",
        # ── Composite spectral indices (closes #505) ────────────────
        "CompositeIndexDef",
        "STANDARD_COMPOSITE_INDICES",
        # ── Per-age stellar mass-remaining curve (closes #447) ──────
        "compute_mass_remaining_fraction",
        # ── Memory-bounded batched evaluation (#1048, API Phase 4) ─
        "vmap_chunked",
        # ── Registry introspection ──────────────────────────────────
        "describe",
        "describe_agn_block",
        "describe_agn_model",
        "describe_dust_emission_model",
        "describe_dust_law",
        "describe_inference_method",
        "describe_nebular_backend",
        "describe_parameter",
        "describe_recipe",
        "describe_sfh_model",
        "examples",
        "explain",
        "help",
        # property catalog introspection (API consolidation #1045/#1043)
        "describe_property",
        "list_properties",
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
        "list_nebular_backends",
        "list_parameters",
        "list_plots",
        "list_metallicity_modes",
        "list_radio_blocks",
        "list_radio_models",
        "list_recipes",
        "list_sfh_models",
        "list_shock_models",
        "list_xray_models",
        "ParameterRecord",
        "print_components_bibtex",
        "recipe_parameters",
        "suggest_parameters",
        "summary",
        "tutorial",
        # ── Subpackage namespaces (canonical import paths) ──────────
        "agn",
        "builders",
        "citations",
        "config",
        "cosmology",
        "dust",
        "filters",
        "igm",
        "inference",
        "io",
        "nebular",
        "observation",
        "pipeline",
        "plot",
        "preprocessing",
        "presets",
        "radio",
        "recipes",
        "results",
        "sfh",
        "sps",
        "stellar",
        "units",
        "xray",
    }
)


# Names that are still *importable* via `from tengri import X` but are
# intentionally NOT advertised in `__all__`. They emit no warning today
# (Phase 6); a future phase may add DeprecationWarning shims that point
# users to the canonical subpackage paths. Listed here so the test can
# distinguish "removed" from "demoted".
DEMOTED_BUT_IMPORTABLE: frozenset[str] = frozenset(
    {
        # User-facing extension surfaces — importable but not advertised in
        # __all__ (custom filters / custom physics-block base class). Examples
        # use ``from tengri import FilterCurve`` / ``SEDModelComponent``.
        "FilterCurve",
        "SEDModelComponent",
        # Branding
        "LOGO",
        "LOGO_BANNER",
        "print_logo",
        # Individual citation helpers — use `from tengri import citations` instead
        "Bibliography",
        "Citation",
        "citations_bibtex",
        "citations_report",
        "cite",
        "cite_all",
        "cites",
        "collect_citations",
        "paper_citation",
        "print_bibtex",
        "print_citations",
        "print_paper_citation",
        # (exp_squared_kernel / matern32_kernel / gp_noise_covariance were
        # re-promoted to top-level as of #511 — they're the standard noise-model
        # kernels and every spectroscopy fit needs them.)
        # Single-purpose filter loaders — importable but not advertised; the
        # gallery uses `from tengri import load_filter` / `load_filter_set` (#802).
        # (load_ssp_data is RE-PROMOTED to top-level as of #496 — every reproduction
        # notebook needs it, so back into ALLOWED_TOP_LEVEL it goes.)
        "load_filter",
        "load_filter_set",
        # PAHspec axis-selection helper — importable but not advertised; paired
        # with the advertised `load_pahspec_draine2021` loader (#803).
        "select_pahspec_axes",
        # Cache machinery — use `tengri.utils.jax_cache.*` /
        # `tengri.inference.jit_engine.*` instead
        "cache_size_bytes",
        "clear_shared_caches",
        "enable_persistent_cache",
        "gc",
        "is_cache_enabled",
        "lean",
        "persistent",
        # SEDComponent extension surface — demoted to `tengri.protocols.*`
        # Both renamed names and back-compat aliases resolve here.
        "ComponentIOError",
        "DerivedKey",
        "DerivedState",
        "ForwardState",
        # Back-compat aliases for renamed classes — emit DeprecationWarning
        "DerivedBundle",
        "PipelineContractError",
        "Provenance",
        # ── Phase 2 (2026-05) — relocated to sub-namespaces ─────────
        # Resolve via `__getattr__` deprecation shim, emit DeprecationWarning.
        # Result classes → tengri.results
        "FitRecord",
        "FitResult",
        "MockData",
        "Posterior",
        "CatalogPosterior",
        "PopulationPosterior",
        "generate_mock",
        "posteriors_to_dataframe",
        # Fitters / inference → tengri.inference
        "Fitter",
        "CatalogFitter",
        "PopulationFitter",
        "VIConfig",
        # Configs → tengri.config
        "AGNConfig",
        "DustConfig",
        "NebularConfig",
        "SEDModelConfig",
        "SFHConfig",
        # (The Observation instrument-schema family — Observation, Photometry,
        # Spectroscopy, NoiseModel, LineList — was re-promoted to __all__ /
        # ALLOWED_TOP_LEVEL in #1338; it is no longer demoted.)
        # LineFluxData / SpectralIndex{Def,Data} no longer resolvable at
        # the top level — import from `tengri.observation` directly. The
        # back-compat shim was removed in the alias cleanup pass.
    }
)


@pytest.mark.contract
def test_all_is_within_allowed_top_level() -> None:
    """No accidental top-level pollution.

    Anything in ``__all__`` must be in :data:`ALLOWED_TOP_LEVEL`. Adding
    a new top-level symbol requires a deliberate edit here AND an entry
    in the migration doc.
    """
    extra = set(tengri.__all__) - ALLOWED_TOP_LEVEL
    assert not extra, (
        f"Unexpected names in tengri.__all__: {sorted(extra)}. "
        f"Either move them into a subpackage or add them to ALLOWED_TOP_LEVEL "
        f"with a migration-doc entry."
    )


@pytest.mark.contract
def test_all_names_are_actually_resolvable() -> None:
    """Every name in ``__all__`` must exist on the module."""
    missing = [name for name in tengri.__all__ if not hasattr(tengri, name)]
    assert not missing, f"Names listed in __all__ but not present: {missing}"


@pytest.mark.contract
def test_demoted_names_still_importable() -> None:
    """Phase 6 demoted names must still resolve for back-compat.

    They are no longer in ``__all__`` but ``from tengri import X``
    must continue to work until a later phase adds explicit
    DeprecationWarning shims.
    """
    missing = [name for name in DEMOTED_BUT_IMPORTABLE if not hasattr(tengri, name)]
    assert not missing, (
        f"Demoted names that broke back-compat: {missing}. "
        f"Either restore them or add a DeprecationWarning shim."
    )


@pytest.mark.contract
def test_demoted_and_advertised_are_disjoint() -> None:
    """A name cannot be both advertised and demoted."""
    overlap = ALLOWED_TOP_LEVEL & DEMOTED_BUT_IMPORTABLE
    assert not overlap, f"name in both ALLOWED_TOP_LEVEL and DEMOTED_BUT_IMPORTABLE: {overlap}"


@pytest.mark.contract
def test_new_subpackages_resolve() -> None:
    """``tengri.plot`` / ``.cosmology`` / ``.units`` re-export identical objects.

    Surface protected: the Phase-1 shim subpackages. Every name in each
    shim's ``__all__`` must be the SAME object as in its canonical source
    module — a diverging copy (stale import, accidental wrapper) would let
    the two paths drift apart silently.
    """
    import types

    import tengri.analysis.plotting as _plotting
    import tengri.cosmology
    import tengri.parameters.translate as _parameters_translate
    import tengri.plot
    import tengri.units
    import tengri.utils.conversions as _conversions
    import tengri.utils.cosmology as _cosmology
    import tengri.utils.magnitudes as _magnitudes
    import tengri.utils.physics_constants as _physics_constants

    shim_sources = [
        (tengri.cosmology, (_cosmology,)),
        (tengri.units, (_conversions, _magnitudes, _physics_constants, _parameters_translate)),
        (tengri.plot, (_plotting,)),
    ]
    for shim, sources in shim_sources:
        diverged = []
        for name in shim.__all__:
            obj = getattr(shim, name)
            if isinstance(obj, types.ModuleType):
                continue
            if not any(getattr(src, name, None) is obj for src in sources):
                diverged.append(name)
        assert not diverged, (
            f"{shim.__name__}: names not identical to their canonical source module: {diverged}"
        )


# ── SEDModel predict-surface ratchet (cleanup PR-2, 2026-07) ────────
# The predict_* surface accreted to 28 methods session-by-session; the
# diet froze it. Shrink-only: removing a method updates this list in the
# same PR (forcing the conversation); ADDING a public predict_* method
# fails here — new derived quantities belong on the lazy Prediction
# wrapper (model.predict(params).sed / .sfh / .lines / ...), not on
# SEDModel.

# The 29 methods split three ways (#1290). Keeping the classification in the
# code — rather than in a comment — lets the tests below assert each group's
# distinct property: sanctioned ones are documented, deprecated ones warn and
# are excluded from rendered docs, and the middle group is the actual backlog.

#: The three the naming contract sanctions (NAMING_CONTRACT §4b, CLAUDE.md).
CONTRACT_PREDICT_METHODS = frozenset(
    {
        "predict",  # rich + cached; one forward pass
        "predict_photometry",  # lean, JIT/vmap-safe; the inference hot path
        "predict_properties",  # the one jit/vmap surface for derived quantities
    }
)

#: Live, un-deprecated, but outside the sanctioned three. This is the backlog:
#: each either earns a place in the contract or moves to the lazy Prediction
#: wrapper. Shrinking this set is the point of the ratchet.
UNSANCTIONED_PREDICT_METHODS = frozenset(
    {
        # core forward passes with production callers
        "predict_state",
        "predict_observables",  # bypasses the WavePrecomp LUT: ~16.5x slower
        "predict_observables_jit",
        # likelihood-facing channels (inference callers)
        "predict_spectrum",
        "predict_line_fluxes",
        "predict_line_ratios",
        "predict_spectral_indices",
        # batch conveniences
        "predict_photometry_batch",
        "predict_spectrum_batch",
        # interactive convenience
        "predict_sfh",
    }
)

#: Deprecated shims: emit DeprecationWarning, removal at v1.0. Every one of
#: these must also appear in the ``:exclude-members:`` list of the SEDModel
#: autoclass directive — a test below binds the two together, because the
#: correspondence is hand-maintained and was previously unguarded.
DEPRECATED_PREDICT_METHODS = frozenset(
    {
        "predict_rest_sed",
        "predict_obs_sed",
        "predict_magnitudes",
        "predict_derived",
        "predict_sfh_quantities",
        "predict_sed_quantities",
        "predict_photometry_components",
        "predict_spectrum_components",
        "predict_sfh_quantities_components",
        "predict_sed_quantities_components",
        "predict_hbeta",
        "predict_emission_lines",
        "predict_luminosity",
        "predict_ionizing_quantities",
        "predict_radio_quantities",
        "predict_xray_quantities",
    }
)

ALLOWED_PREDICT_METHODS = (
    CONTRACT_PREDICT_METHODS | UNSANCTIONED_PREDICT_METHODS | DEPRECATED_PREDICT_METHODS
)


@pytest.mark.contract
def test_sedmodel_predict_surface_does_not_grow() -> None:
    """No new public predict_* methods on SEDModel (shrink-only ratchet)."""
    from tengri.forward.sed_model import SEDModel

    actual = {
        name
        for name in dir(SEDModel)
        if name.startswith("predict") and callable(getattr(SEDModel, name))
    }
    new = actual - ALLOWED_PREDICT_METHODS
    assert not new, (
        f"New public predict_* method(s) on SEDModel: {sorted(new)}. "
        f"Derived quantities belong on the Prediction wrapper "
        f"(model.predict(params).<accessor>); if a new forward channel is "
        f"genuinely needed, update ALLOWED_PREDICT_METHODS deliberately."
    )
