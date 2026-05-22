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
        "Population",
        "SEDModel",
        "WavePrecomp",
        # ── SEDComponent extension surface (Phase II protocols) ─────
        "DerivedBundle",
        "DerivedKey",
        "ForwardState",
        "PipelineContractError",
        # ── Priors (parameters/) ─────────────────────────────────────
        "Fixed",
        "Gaussian",
        "LogNormal",
        "LogUniform",
        "StudentT",
        "Uniform",
        # ── Exceptions ──────────────────────────────────────────────
        "BackendError",
        "ConfigError",
        "InferenceError",
        "ParameterError",
        "TengriError",
        "TengriIOError",
        # ── Cache helpers ───────────────────────────────────────────
        "cache_size_bytes",
        "clear_cache",
        "enable_persistent_cache",
        "is_cache_enabled",
        # ── Top-level convenience verbs ─────────────────────────────
        "cite_components",
        "clear_shared_caches",
        "doctor",
        "gc",
        "lean",
        "persistent",
        "register_component",
        "search",
        # ── SSP data setup ──────────────────────────────────────────
        "download_ssp",
        "list_known_ssps",
        # ── Registry introspection ──────────────────────────────────
        "describe",
        "describe_parameter",
        "examples",
        "explain",
        "help",
        "list_agn_models",
        "list_all",
        "list_components",
        "list_dust_emission_models",
        "list_dust_laws",
        "list_filters",
        "list_inference_methods",
        "list_nebular_backends",
        "list_parameters",
        "list_plots",
        "list_sfh_models",
        "ParameterRecord",
        "recipe_parameters",
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
        # Noise kernel helpers — use `tengri.observation.noise.*` instead
        "exp_squared_kernel",
        "gp_noise_covariance",
        "matern32_kernel",
        # Single-purpose loaders — use `tengri.observation.load_filter_set` /
        # `tengri.sps.load_ssp_data` instead
        "load_filter_set",
        "load_ssp_data",
        # ── Phase 2 (2026-05) — relocated to sub-namespaces ─────────
        # Resolve via `__getattr__` deprecation shim, emit DeprecationWarning.
        # Result classes → tengri.results
        "FitResult",
        "MockData",
        "Provenance",
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
        # Observation classes → tengri.observation
        "Photometry",
        "Spectroscopy",
        "NoiseModel",
        "Observation",
        "LineList",
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
    """Phase 1 introduces ``tengri.plot``, ``.cosmology``, ``.units``."""
    assert hasattr(tengri, "plot")
    assert hasattr(tengri, "cosmology")
    assert hasattr(tengri, "units")

    # And they expose what we documented.
    assert hasattr(tengri.cosmology, "PLANCK18")
    assert hasattr(tengri.cosmology, "luminosity_distance")
    assert hasattr(tengri.units, "fnu_to_jy")
    assert hasattr(tengri.units, "ab_mag_to_fnu")
    assert hasattr(tengri.plot, "plot_sed_fit")
