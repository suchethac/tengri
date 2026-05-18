"""Conformance suite for every registered inference backend.

Iterates ``_BACKENDS`` and verifies each entry satisfies the
``BackendEntry`` contract described in ADR-0009:

- ``name`` matches its registry key.
- ``tier`` is one of ``"primary"`` / ``"experimental"``.
- ``runner`` is callable.
- ``requires`` declares only importable packages (or is reported as
  ``missing_dep`` by :func:`~tengri.inference._strategy.resolve_status`).
- ``legacy_fitter`` defaults to ``True`` (out-of-tree backends keep
  working); every in-tree entry sets it ``False``.

Because the suite is parametrised over the live registry, adding a
new backend in ``_registration.py`` makes it appear here automatically
— no test-file edits required.

No SSP data is needed: tests inspect the registry's metadata, not
actually run inference. The end-to-end smoke that proves dispatch
plumbing works lives in ``test_inference_context.py``.
"""

from __future__ import annotations

import importlib

import pytest

from tengri.inference._backend_registry import _BACKENDS, BackendEntry, all_backends
from tengri.inference._strategy import BackendStatus, resolve_status

# Build the parametrisation list at import time so test IDs are
# stable (``mcmc_nuts``, ``vi_nonlinear_fast``, ...).
_REGISTERED_NAMES = sorted(_BACKENDS)
_UNIQUE_ENTRIES = all_backends()


# ── per-name contract ────────────────────────────────────────────────


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
def test_entry_has_callable_runner(name: str) -> None:
    entry = _BACKENDS[name]
    assert callable(entry.runner), f"{name}: runner is not callable"


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
def test_entry_tier_is_known(name: str) -> None:
    entry = _BACKENDS[name]
    assert entry.tier in {"primary", "experimental"}, (
        f"{name}: tier={entry.tier!r} not in {{primary, experimental}}"
    )


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
def test_entry_short_doc_present(name: str) -> None:
    """Each backend must self-describe via ``short_doc`` for
    :func:`tengri.list_inference_methods`."""
    entry = _BACKENDS[name]
    assert entry.short_doc, f"{name}: short_doc is empty"


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
def test_requires_entries_are_strings(name: str) -> None:
    entry = _BACKENDS[name]
    assert isinstance(entry.requires, tuple), (
        f"{name}: requires must be a tuple, got {type(entry.requires).__name__}"
    )
    for pkg in entry.requires:
        assert isinstance(pkg, str) and pkg, (
            f"{name}: requires entry {pkg!r} is not a non-empty string"
        )


# ── in-tree backends are all migrated ────────────────────────────────


@pytest.mark.parametrize("entry", _UNIQUE_ENTRIES, ids=lambda e: e.name)
def test_in_tree_backends_use_context(entry: BackendEntry) -> None:
    """Every backend registered in ``_registration.py`` must opt into
    the InferenceContext protocol (``legacy_fitter=False``).

    The flag remains on ``BackendEntry`` for out-of-tree compatibility,
    but no shipped backend should rely on it.
    """
    assert entry.legacy_fitter is False, (
        f"{entry.name}: legacy_fitter is True; in-tree backends must "
        f"migrate to the InferenceContext protocol (ADR-0009)."
    )


# ── status reporting ─────────────────────────────────────────────────


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
def test_resolve_status_returns_known_value(name: str) -> None:
    entry = _BACKENDS[name]
    status = resolve_status(entry)
    assert isinstance(status, BackendStatus), (
        f"{name}: resolve_status returned {type(status).__name__}, expected BackendStatus"
    )


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
def test_status_missing_dep_matches_importability(name: str) -> None:
    """If ``resolve_status`` reports ``missing_dep``, at least one
    ``requires`` entry must actually fail to import — and vice versa."""
    entry = _BACKENDS[name]
    status = resolve_status(entry)

    importable = True
    for pkg in entry.requires:
        try:
            importlib.import_module(pkg)
        except ImportError:
            importable = False
            break

    if status == BackendStatus.MISSING_DEP:
        assert not importable, (
            f"{name}: status=missing_dep but all of {entry.requires} import cleanly. Stale check?"
        )
    elif importable:
        assert status != BackendStatus.MISSING_DEP, (
            f"{name}: all deps importable but status={status.value}"
        )


# ── registry coverage ────────────────────────────────────────────────


def test_registry_is_non_empty() -> None:
    """Sanity: ``inference/__init__.py`` must have triggered
    ``_registration`` at package import time."""
    assert len(_BACKENDS) > 0, (
        "Backend registry is empty. ``inference/__init__.py`` is "
        "expected to import ``_registration`` for its side effects."
    )


def test_at_least_one_primary_per_family() -> None:
    """Quick smoke: MAP, MCMC, and VI must each have a primary entry
    so users have a clear default for each problem class."""
    primary_names = {e.name for e in all_backends() if e.tier == "primary"}
    assert "map" in primary_names, "missing primary backend: map"
    assert any(n.startswith("mcmc") for n in primary_names), "missing primary MCMC backend"
    assert any(n.startswith("vi") for n in primary_names), "missing primary VI backend"


def test_canonical_aliases_share_runner() -> None:
    """``vi`` and ``vi_nonlinear`` are documented as aliases —
    they must dispatch to the same runner so ``fitter.run("vi")`` and
    ``fitter.run("vi_nonlinear")`` cannot diverge."""
    assert _BACKENDS["vi"].runner is _BACKENDS["vi_nonlinear"].runner, (
        "'vi' and 'vi_nonlinear' must share a runner (registered as aliases)."
    )
