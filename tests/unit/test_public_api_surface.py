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

# Names allowed in tengri.__all__ as of Phase 6 (2026-05).
#
# This is intentionally tighter than what's *importable* — implementation
# helpers (noise kernels, branding strings, individual citation
# functions, single-purpose loaders) remain accessible via direct
# import but are no longer advertised. Adding a new top-level symbol
# requires editing this set AND adding a row in
# `docs/dev/api_migration_v0.x.md`.
ALLOWED_TOP_LEVEL: frozenset[str] = frozenset(
    {
        # ── Top-level user verbs (the seven a typical user types) ────
        "Galaxy",
        "SEDModel",
        "Parameters",
        "Fitter",
        "Posterior",
        "Observation",
        "NoiseModel",
        # ── Population / catalog inference ───────────────────────────
        "CatalogFitter",
        "CatalogPosterior",
        "PopulationFitter",
        "PopulationPosterior",
        "VIConfig",
        # ── Priors (parameters/) ─────────────────────────────────────
        "Fixed",
        "Gaussian",
        "LogNormal",
        "LogUniform",
        "StudentT",
        "Uniform",
        # ── Configuration ────────────────────────────────────────────
        "AGNConfig",
        "DustConfig",
        "NebularConfig",
        "SEDModelConfig",
        "SFHConfig",
        # ── Exceptions ──────────────────────────────────────────────
        "BackendError",
        "ConfigError",
        "InferenceError",
        "ParameterError",
        "TengriError",
        "TengriIOError",
        # ── Observation containers ──────────────────────────────────
        "LineFluxData",
        "LineList",
        "Photometry",
        "SpectralIndexData",
        "SpectralIndexDef",
        "Spectroscopy",
        # ── Result / mock containers ─────────────────────────────────
        "FitResult",
        "MockData",
        "Provenance",
        # ── Cache helpers ───────────────────────────────────────────
        "cache_size_bytes",
        "clear_cache",
        "enable_persistent_cache",
        "is_cache_enabled",
        # ── Convenience top-level functions ─────────────────────────
        "doctor",
        "generate_mock",
        "posteriors_to_dataframe",
        # ── Subpackage namespaces (canonical import paths) ──────────
        "agn",
        "citations",
        "cosmology",
        "dust",
        "filters",
        "igm",
        "io",
        "nebular",
        "observation",
        "pipeline",
        "plot",
        "preprocessing",
        "presets",
        "radio",
        "sfh",
        "sps",
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
    }
)


@pytest.mark.unit
def test_all_is_a_list_of_strings() -> None:
    """``tengri.__all__`` must be a list/tuple of strings (introspection)."""
    assert hasattr(tengri, "__all__"), "tengri must define __all__"
    assert isinstance(tengri.__all__, (list, tuple))
    for name in tengri.__all__:
        assert isinstance(name, str), f"non-string in __all__: {name!r}"


@pytest.mark.unit
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


@pytest.mark.unit
def test_all_names_are_actually_resolvable() -> None:
    """Every name in ``__all__`` must exist on the module."""
    missing = [name for name in tengri.__all__ if not hasattr(tengri, name)]
    assert not missing, f"Names listed in __all__ but not present: {missing}"


@pytest.mark.unit
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


@pytest.mark.unit
def test_demoted_and_advertised_are_disjoint() -> None:
    """A name cannot be both advertised and demoted."""
    overlap = ALLOWED_TOP_LEVEL & DEMOTED_BUT_IMPORTABLE
    assert not overlap, f"name in both ALLOWED_TOP_LEVEL and DEMOTED_BUT_IMPORTABLE: {overlap}"


@pytest.mark.unit
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
