"""Unit tests for the KernelStrategy selection logic.

These tests exercise the Python-level orchestration in isolation, using
plain object stand-ins for ``SEDModelState`` so we don't pay the cost of
building a real ``SEDModel`` (which needs SSP data on disk).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tengri.forward._kernels import (
    COMPOSITIONAL_ONLY,
    DEFAULT,
    EXACT_ONLY,
    LOW_MEMORY,
    KernelStrategy,
    NoCompatibleKernelError,
)

# ── fixtures ────────────────────────────────────────────────────────


def _state(
    *,
    with_filters: bool = True,
    with_rest_wavelength: bool = True,
    with_precomp_photometry: bool = True,
    with_precomp_spectroscopy: bool = True,
    with_precomp_ztable: bool = False,
    z_fixed: float | None = 0.05,
    wave_obs: Any = None,
) -> SimpleNamespace:
    """Build a bare-bones state stand-in covering every attribute the
    adapter predicates inspect."""
    precomputed = SimpleNamespace(
        photometry=object() if with_precomp_photometry else None,
        spectroscopy=object() if with_precomp_spectroscopy else None,
        photometry_ztable=object() if with_precomp_ztable else None,
    )
    return SimpleNamespace(
        filter_waves=[object()] if with_filters else None,
        rest_wavelength=object() if with_rest_wavelength else None,
        precomputed=precomputed,
        z_fixed=z_fixed,
        wave_obs=wave_obs,
    )


# ── strategy selection ─────────────────────────────────────────────


def test_default_selects_compositional_photometry_first():
    state = _state()
    names = [k.name for k in DEFAULT.select(state, product="photometry")]
    assert names[0] == "compositional_photometry"
    assert "hybrid_photometry" in names
    assert "exact_rest_sed" not in names  # different product


def test_low_memory_skips_hybrid():
    state = _state()
    names = [k.name for k in LOW_MEMORY.select(state, product="photometry")]
    assert "hybrid_photometry" not in names
    assert names[0] == "compositional_photometry"


def test_exact_only_yields_nothing_for_photometry():
    # exact_rest_sed produces rest_sed, not photometry — selecting for
    # photometry under EXACT_ONLY yields an empty iterator.
    state = _state()
    names = [k.name for k in EXACT_ONLY.select(state, product="photometry")]
    assert names == []


def test_exact_only_yields_exact_for_rest_sed():
    state = _state()
    names = [k.name for k in EXACT_ONLY.select(state, product="rest_sed")]
    assert names == ["exact_rest_sed"]


def test_compositional_only_for_spectrum():
    state = _state()
    names = [k.name for k in COMPOSITIONAL_ONLY.select(state, product="spectrum")]
    assert names == ["compositional_spectrum"]


# ── mode shortcuts ─────────────────────────────────────────────────


def test_mode_hybrid_yields_only_hybrid_photometry():
    state = _state()
    names = [k.name for k in DEFAULT.select(state, product="photometry", requested_mode="hybrid")]
    assert names == ["hybrid_photometry"]  # ztable disabled (no precompute_ztable)


def test_mode_hybrid_ztable_when_table_present():
    state = _state(with_precomp_ztable=True, z_fixed=None, with_precomp_photometry=False)
    names = [k.name for k in DEFAULT.select(state, product="photometry", requested_mode="hybrid")]
    assert names == ["hybrid_photometry_ztable"]


def test_mode_compositional():
    state = _state()
    names = [
        k.name for k in DEFAULT.select(state, product="photometry", requested_mode="compositional")
    ]
    assert names == ["compositional_photometry"]


def test_mode_exact_only_valid_for_rest_sed():
    state = _state()
    assert [k.name for k in DEFAULT.select(state, product="rest_sed", requested_mode="exact")] == [
        "exact_rest_sed"
    ]
    # For photometry / spectrum, exact mode resolves to no adapters.
    assert list(DEFAULT.select(state, product="photometry", requested_mode="exact")) == []


# ── predicates ─────────────────────────────────────────────────────


def test_missing_filters_blocks_compositional_photometry():
    state = _state(with_filters=False)
    names = [k.name for k in DEFAULT.select(state, product="photometry")]
    assert "compositional_photometry" not in names


def test_free_z_blocks_fixed_hybrid_photometry():
    state = _state(z_fixed=None, with_precomp_ztable=False)
    names = [k.name for k in DEFAULT.select(state, product="photometry")]
    assert "hybrid_photometry" not in names
    assert "hybrid_photometry_ztable" not in names  # no table either


def test_free_z_with_ztable_enables_ztable_kernel():
    state = _state(z_fixed=None, with_precomp_ztable=True)
    names = [k.name for k in DEFAULT.select(state, product="photometry")]
    assert "hybrid_photometry_ztable" in names


def test_missing_spectroscopy_blocks_spectrum_kernels():
    state = _state(with_precomp_spectroscopy=False, wave_obs=None)
    names = [k.name for k in DEFAULT.select(state, product="spectrum")]
    assert "compositional_spectrum" not in names
    assert "hybrid_spectrum" not in names


def test_tabulated_sfh_blocks_hybrid_photometry():
    state = _state()
    params = {"sfh_t_gyr": object(), "sfh_sfr": object()}
    names = [k.name for k in DEFAULT.select(state, product="photometry", params=params)]
    assert "hybrid_photometry" not in names
    assert "compositional_photometry" in names


# ── error path ────────────────────────────────────────────────────


def test_first_or_raise_returns_first():
    state = _state()
    adapter = DEFAULT.first_or_raise(state, product="photometry")
    assert adapter.name == "compositional_photometry"


def test_first_or_raise_raises_when_no_match():
    state = _state(with_filters=False, with_precomp_photometry=False, z_fixed=None)
    with pytest.raises(NoCompatibleKernelError):
        DEFAULT.first_or_raise(state, product="photometry")


def test_strategy_is_frozen_and_hashable():
    import dataclasses

    s = KernelStrategy()
    assert hash(s) == hash(KernelStrategy())
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.preferred = ()  # type: ignore[misc]


def test_unknown_adapter_name_in_strategy_is_silently_skipped():
    # Robustness: a typo in preferred names shouldn't crash the iteration.
    s = KernelStrategy(preferred=("does_not_exist", "compositional_photometry"))
    state = _state()
    names = [k.name for k in s.select(state, product="photometry")]
    assert names == ["compositional_photometry"]
