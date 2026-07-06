# SPDX-License-Identifier: BSD-3-Clause
"""Lock the advertised top-level public surface of `tengri`.

Phase 2 of the API cleanup (2026-05) trimmed `tengri.__all__` and moved
result classes, observation classes, fitters, and config dataclasses
into sub-namespaces (`tengri.results`, `tengri.observation`,
`tengri.inference`, `tengri.config`). This test pins the new layout and
asserts the deprecation shims still resolve old import paths.

Plan: docs/dev/api_migration_v0.x.md, ~/.claude/plans/i-want-you-to-lazy-acorn.md
"""

from __future__ import annotations

import warnings

import pytest

import tengri

pytestmark = pytest.mark.contract
# Canonical locations of names that used to live at the top level. Each
# entry pins that the new path still resolves cleanly (no DeprecationWarning).
_RELOCATED: dict[str, tuple[str, str]] = {
    "LineFluxData": ("tengri.observation", "LineFluxData"),
    # (SpectralIndexDef / SpectralIndexData re-promoted to top-level as of #511.)
}

# Frozen target list. Adding to or removing from this list is a deliberate
# public-API change and should be reviewed.
EXPECTED_ALL = frozenset(
    {
        # Core
        "Exponential",
        "FilterConvention",
        "FlatSlab",
        "ForwardModel",
        "Galaxy",
        "Parameters",
        "parse_groups",
        "Population",
        "PopulationSEDModel",
        "SEDModel",
        "Sersic",
        "SpatialModel",
        "SpatialSEDModel",
        "WavePrecomp",
        "SpectrumPrecomp",
        # Component-extension surface — demoted to `tengri.protocols.*`,
        # no longer advertised at top level (importable for back-compat).
        # Physics modules
        "agn",
        "builders",
        "dust",
        "igm",
        "nebular",
        "radio",
        "sfh",
        "sps",
        "stellar",
        "xray",
        # Layer modules
        "citations",
        "config",
        "cosmology",
        "filters",
        "inference",
        "io",
        "observation",
        "pipeline",
        "plot",
        "preprocessing",
        "presets",
        "results",
        "units",
        # Registry verbs
        "describe",
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
        "list_agn_models",
        "list_all",
        "list_components",
        "list_dust_emission_models",
        "list_dust_laws",
        "list_filter_conventions",
        "list_filters",
        "list_igm_models",
        "list_inference_methods",
        "list_nebular_backends",
        "list_parameters",
        "list_plots",
        "list_radio_models",
        "list_recipes",
        "list_sfh_models",
        "list_xray_models",
        "ParameterRecord",
        "recipe_parameters",
        "search",
        "summary",
        "tutorial",
        # Runtime verbs (cache machinery demoted to tengri.utils.jax_cache /
        # tengri.inference.jit_engine — clear_cache is the only top-level entry)
        "cite_components",
        "clear_cache",
        "doctor",
        "download_ssp",
        "list_available_ssps",
        "list_known_ssps",
        # SSP loaders (closes #496)
        "load_ssp",
        "load_ssp_data",
        "SSPData",
        # Dust-emission template loaders (closes #803)
        "load_astrodust_hd23",
        "load_pahspec_draine2021",
        # Component helpers (closes #497 / #498)
        "igm_transmission",
        "igm_transmission_madau",
        "velocity_broaden",
        "apply_lsf",
        # GP-noise kernels + spectral-index helpers (closes #511)
        "exp_squared_kernel",
        "matern32_kernel",
        "gp_noise_covariance",
        "SpectralIndexDef",
        "SpectralIndexData",
        "STANDARD_INDICES",
        "measure_index_jax",
        # Composite spectral indices (closes #505)
        "CompositeIndexDef",
        "STANDARD_COMPOSITE_INDICES",
        # Per-age stellar mass-remaining curve (closes #447)
        "compute_mass_remaining_fraction",
        "register_component",
        # Exceptions
        "BackendError",
        "ConfigError",
        "InferenceError",
        "ParameterError",
        "TengriError",
        "TengriIOError",
        # Priors
        "Fixed",
        "Gaussian",
        "Laplace",
        "LogNormal",
        "LogUniform",
        "StudentT",
        "Uniform",
        # Sentinels (used in every recipe + 100+ tests)
        "FIXED",
        "FREE",
        # Forward-model outputs and helpers
        "PriorPredictive",
        "SEDResult",
        # Top-level convenience verbs
        "fit_batch",
        # Data discovery helpers
        "data_path",
    }
)


def test_all_matches_expected() -> None:
    """`tengri.__all__` must equal the frozen target. Drift = deliberate review."""
    actual = set(tengri.__all__)
    extra = actual - EXPECTED_ALL
    missing = EXPECTED_ALL - actual
    assert not extra, f"Unexpected names in tengri.__all__: {sorted(extra)}"
    assert not missing, f"Missing from tengri.__all__: {sorted(missing)}"


def test_all_entries_resolve_without_warning() -> None:
    """Every entry in `__all__` must resolve cleanly (no DeprecationWarning)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for name in tengri.__all__:
            getattr(tengri, name)
        deprecations = [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and any(rel in str(w.message) for rel in _RELOCATED)
        ]
    assert not deprecations, (
        f"Top-level __all__ entries triggered relocation warnings: "
        f"{[str(w.message) for w in deprecations]}"
    )


@pytest.mark.parametrize("old_name", sorted(_RELOCATED))
def test_relocated_symbols_no_longer_at_top_level(old_name: str) -> None:
    """Names moved into sub-namespaces must no longer resolve at ``tengri.<name>``."""
    with pytest.raises(AttributeError):
        getattr(tengri, old_name)


@pytest.mark.parametrize(
    ("module_path", "attr"),
    sorted({(mod, attr) for (mod, attr) in _RELOCATED.values()}),
)
def test_canonical_paths_resolve_cleanly(module_path: str, attr: str) -> None:
    """The new canonical path must resolve without any DeprecationWarning."""
    import importlib

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module(module_path)
        value = getattr(module, attr)
    assert value is not None
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not deprecations, (
        f"{module_path}.{attr} emitted DeprecationWarning: "
        f"{[str(w.message) for w in deprecations]}"
    )


def test_unknown_attribute_raises_attribute_error() -> None:
    """The shim must not silently swallow typos."""
    with pytest.raises(AttributeError, match="has no attribute 'NotARealName'"):
        tengri.NotARealName  # noqa: B018
