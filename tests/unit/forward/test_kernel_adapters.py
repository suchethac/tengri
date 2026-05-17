"""Unit tests for the kernel adapter wrappers.

Adapters are thin wrappers around the seven existing ``build_*`` builders.
These tests verify two properties:

1. ``is_compatible`` returns the right boolean for representative state shapes.
2. ``build`` delegates to the underlying function (verified by patching).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from tengri.forward._kernels import (
    ALL_ADAPTERS,
    CompositionalPhotometryKernel,
    CompositionalRestSEDKernel,
    CompositionalSpectrumKernel,
    ExactRestSEDKernel,
    HybridPhotometryKernel,
    HybridPhotometryZTableKernel,
    HybridSpectrumKernel,
    Kernel,
    adapters_by_name,
)


def _state(**overrides):
    precomp = SimpleNamespace(
        photometry=overrides.pop("precomp_phot", object()),
        spectroscopy=overrides.pop("precomp_spec", object()),
        photometry_ztable=overrides.pop("precomp_ztable", None),
    )
    defaults = {
        "filter_waves": [object()],
        "rest_wavelength": object(),
        "z_fixed": 0.05,
        "wave_obs": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(precomputed=precomp, **defaults)


# ── registry ───────────────────────────────────────────────────────


def test_all_adapters_satisfy_kernel_protocol():
    for adapter in ALL_ADAPTERS:
        assert isinstance(adapter, Kernel)


def test_adapter_names_are_unique():
    names = [a.name for a in ALL_ADAPTERS]
    assert len(names) == len(set(names))


def test_adapters_by_name_is_complete():
    by_name = adapters_by_name()
    assert {a.name for a in ALL_ADAPTERS} == set(by_name)


# ── is_compatible (state-level) ────────────────────────────────────


def test_exact_rest_sed_is_always_compatible():
    assert ExactRestSEDKernel().is_compatible(SimpleNamespace())


def test_compositional_rest_sed_needs_rest_wavelength():
    assert CompositionalRestSEDKernel().is_compatible(_state())
    assert not CompositionalRestSEDKernel().is_compatible(_state(rest_wavelength=None))


def test_compositional_photometry_needs_filters():
    adapter = CompositionalPhotometryKernel()
    assert adapter.is_compatible(_state())
    assert not adapter.is_compatible(_state(filter_waves=None))
    assert not adapter.is_compatible(_state(rest_wavelength=None))


def test_compositional_spectrum_accepts_wave_obs_or_precompute():
    adapter = CompositionalSpectrumKernel()
    assert adapter.is_compatible(_state())  # has precomp_spec
    assert adapter.is_compatible(_state(precomp_spec=None, wave_obs=object()))
    assert not adapter.is_compatible(_state(precomp_spec=None, wave_obs=None))


def test_hybrid_photometry_needs_precompute_and_fixed_z():
    adapter = HybridPhotometryKernel()
    assert adapter.is_compatible(_state())
    assert not adapter.is_compatible(_state(precomp_phot=None))
    assert not adapter.is_compatible(_state(z_fixed=None))


def test_hybrid_photometry_ztable_needs_table():
    adapter = HybridPhotometryZTableKernel()
    assert not adapter.is_compatible(_state())
    assert adapter.is_compatible(_state(precomp_ztable=object()))


def test_hybrid_spectrum_needs_precompute_and_fixed_z():
    adapter = HybridSpectrumKernel()
    assert adapter.is_compatible(_state())
    assert not adapter.is_compatible(_state(precomp_spec=None))
    assert not adapter.is_compatible(_state(z_fixed=None))


# ── is_compatible_with_params (param-level) ────────────────────────


def test_hybrid_adapters_block_tabulated_sfh():
    params = {"sfh_t_gyr": object()}
    assert not HybridPhotometryKernel().is_compatible_with_params(params)
    assert not HybridPhotometryZTableKernel().is_compatible_with_params(params)
    assert not HybridSpectrumKernel().is_compatible_with_params(params)


def test_compositional_adapters_accept_tabulated_sfh():
    params = {"sfh_t_gyr": object()}
    assert CompositionalPhotometryKernel().is_compatible_with_params(params)
    assert CompositionalSpectrumKernel().is_compatible_with_params(params)
    assert CompositionalRestSEDKernel().is_compatible_with_params(params)


def test_empty_params_compatible_everywhere():
    for adapter in ALL_ADAPTERS:
        assert adapter.is_compatible_with_params({})


# ── build() delegates ───────────────────────────────────────────────


def test_exact_rest_sed_build_delegates():
    sentinel = object()
    with patch("tengri.forward._kernels._adapters.build_exact_sed") as m:
        m.return_value = sentinel
        result = ExactRestSEDKernel().build(state="state_obj")
    m.assert_called_once_with("state_obj")
    assert result is sentinel


def test_compositional_photometry_build_passes_model():
    sentinel = object()
    with patch("tengri.forward._kernels._adapters.build_fused_tier2_photometry") as m:
        m.return_value = sentinel
        result = CompositionalPhotometryKernel().build(state="state_obj", model="model_obj")
    m.assert_called_once_with("state_obj", "model_obj")
    assert result is sentinel


def test_hybrid_photometry_ztable_build_delegates():
    sentinel = object()
    with patch("tengri.forward._kernels._adapters.build_hybrid_photometry_ztable") as m:
        m.return_value = sentinel
        result = HybridPhotometryZTableKernel().build(state="s", model="m")
    m.assert_called_once_with("s", "m")
    assert result is sentinel
