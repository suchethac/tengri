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
    "SpectralIndexDef": ("tengri.observation", "SpectralIndexDef"),
    "SpectralIndexData": ("tengri.observation", "SpectralIndexData"),
}

# Frozen target list. Adding to or removing from this list is a deliberate
# public-API change and should be reviewed.
EXPECTED_ALL = frozenset(
    {
        # Core
        "ForwardModel",
        "Galaxy",
        "Parameters",
        "Population",
        "SEDModel",
        "WavePrecomp",
        # Component / pipeline contract (astronomer-facing extension surface)
        "DerivedBundle",
        "DerivedKey",
        "ForwardState",
        "PipelineContractError",
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
        "search",
        "summary",
        "tutorial",
        # Runtime verbs
        "cache_size_bytes",
        "cite_components",
        "clear_cache",
        "clear_shared_caches",
        "doctor",
        "download_ssp",
        "enable_persistent_cache",
        "gc",
        "is_cache_enabled",
        "lean",
        "list_known_ssps",
        "persistent",
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
        "LogNormal",
        "LogUniform",
        "StudentT",
        "Uniform",
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
