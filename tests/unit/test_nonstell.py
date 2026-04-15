"""Tests for tengri.forward.nonstell (NonStellarSlot registry + build_nonstell_fn factory).

Tests cover:
- collect_nonstell() returns the correct ordered slots given model flags
- build_nonstell_fn() round-trip: pure-stellar model returns stellar_sed unchanged
- build_nonstell_fn() returns a callable with correct output shape
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.forward.nonstell import NonStellarSlot, build_nonstell_fn, collect_nonstell

# ---------------------------------------------------------------------------
# Helpers — minimal model stubs so tests run without SSP data
# ---------------------------------------------------------------------------


def _make_model(
    *,
    nebular_backend=None,
    shock_enabled=False,
    dust_emission_model=None,
    agn_model=None,
    agn_parametric=False,
    radio_enabled=False,
    xray_enabled=False,
    neb_dust="bc",
    dust_model="charlot_fall",
    radio_sfr_mode="ir",
    radio_include_freefree=True,
    redshift=0.1,
    ssp_wave=None,
    rest_wave=None,
):
    """Return a lightweight namespace that looks like a Model to nonstell helpers."""
    if ssp_wave is None:
        ssp_wave = np.linspace(1e3, 1e4, 64)
    if rest_wave is None:
        rest_wave = ssp_wave  # same object → _needs_extension = False

    model = SimpleNamespace(
        _nebular_backend=nebular_backend,
        _shock_enabled=shock_enabled,
        _dust_emission_model=dust_emission_model,
        _agn_model=agn_model,
        _agn_parametric=agn_parametric,
        _radio_enabled=radio_enabled,
        _xray_enabled=xray_enabled,
        _neb_dust=neb_dust,
        _dust_model=dust_model,
        _radio_sfr_mode=radio_sfr_mode,
        _radio_include_freefree=radio_include_freefree,
        _redshift=redshift,
        ssp_log_ages_yr=np.log10(np.logspace(6, 10, 64)),
        # rest_wavelength is an identity reference (same object as ssp_wave)
        _rest_wavelength=rest_wave,
        ssp_data=SimpleNamespace(ssp_wave=ssp_wave),
    )
    return model


def _identity_dust_law(wave, **kw):
    """No-op dust law: transmission = 1 everywhere."""
    return jnp.zeros_like(jnp.asarray(wave))


# ---------------------------------------------------------------------------
# collect_nonstell — registry tests
# ---------------------------------------------------------------------------


class TestCollectNonstell:
    def test_all_disabled_returns_empty(self):
        model = _make_model()
        slots = collect_nonstell(model)
        assert slots == []

    def test_nebular_only_has_free_params(self):
        backend = MagicMock()
        backend.has_free_params = True
        model = _make_model(nebular_backend=backend)
        slots = collect_nonstell(model)
        assert len(slots) == 1
        assert slots[0].name == "nebular"
        assert slots[0].on_ssp_grid is True

    def test_nebular_without_free_params_not_registered(self):
        backend = MagicMock()
        backend.has_free_params = False
        model = _make_model(nebular_backend=backend)
        assert collect_nonstell(model) == []

    def test_shock_slot(self):
        model = _make_model(shock_enabled=True)
        slots = collect_nonstell(model)
        assert any(s.name == "shock" for s in slots)

    def test_dust_ir_slot(self):
        model = _make_model(dust_emission_model="dl07")
        slots = collect_nonstell(model)
        assert any(s.name == "dust_ir" for s in slots)
        assert all(not s.on_ssp_grid for s in slots if s.name == "dust_ir")

    def test_agn_slot(self):
        model = _make_model(agn_model="kubota_done_full")
        slots = collect_nonstell(model)
        assert any(s.name == "agn" for s in slots)

    def test_radio_slot(self):
        model = _make_model(radio_enabled=True)
        slots = collect_nonstell(model)
        assert any(s.name == "radio" for s in slots)

    def test_xray_slot(self):
        model = _make_model(xray_enabled=True)
        slots = collect_nonstell(model)
        assert any(s.name == "xray" for s in slots)

    def test_canonical_order_respected(self):
        """All-enabled slots must appear in canonical order: nebular < shock < dust_ir < agn < radio < xray."""  # noqa: E501
        backend = MagicMock()
        backend.has_free_params = True
        model = _make_model(
            nebular_backend=backend,
            shock_enabled=True,
            dust_emission_model="dl07",
            agn_model="qsogen",
            radio_enabled=True,
            xray_enabled=True,
        )
        names = [s.name for s in collect_nonstell(model)]
        expected = ["nebular", "shock", "dust_ir", "agn", "radio", "xray"]
        assert names == expected

    def test_partial_set_preserves_order(self):
        """Radio + xray only → still in the correct relative order."""
        model = _make_model(radio_enabled=True, xray_enabled=True)
        names = [s.name for s in collect_nonstell(model)]
        assert names == ["radio", "xray"]


# ---------------------------------------------------------------------------
# build_nonstell_fn — factory tests
# ---------------------------------------------------------------------------


class TestBuildNonstellFn:
    def _make_pure_stellar_fn(self):
        """Build nonstell_fn with all components disabled."""
        ssp_wave = np.linspace(1e3, 1e4, 64)
        model = _make_model(ssp_wave=ssp_wave, rest_wave=ssp_wave)
        fn = build_nonstell_fn(model, _identity_dust_law, _identity_dust_law, ssp_wave, ssp_wave)
        return fn, ssp_wave

    def test_returns_callable(self):
        fn, _ = self._make_pure_stellar_fn()
        assert callable(fn)

    def test_pure_stellar_passthrough(self):
        """With no non-stellar components, nonstell_fn must return stellar_sed unchanged."""
        fn, _ssp_wave = self._make_pure_stellar_fn()
        weights = jnp.ones(64)
        stellar_sed = jnp.linspace(1e30, 1e33, 64)
        stellar_intr = stellar_sed  # identical (no dust)
        p = {"log_z_abs": -1.848, "tau_bc": 0.0, "tau_diff": 0.0}

        result = fn(weights, p, stellar_sed, stellar_intr)

        assert result.shape == stellar_sed.shape
        np.testing.assert_allclose(np.array(result), np.array(stellar_sed), rtol=1e-6)

    def test_output_shape_matches_rest_wave(self):
        """Output shape must match the rest_wave grid, not ssp_wave."""
        ssp_wave = np.linspace(1e3, 1e4, 64)
        model = _make_model(ssp_wave=ssp_wave, rest_wave=ssp_wave)
        fn = build_nonstell_fn(model, _identity_dust_law, _identity_dust_law, ssp_wave, ssp_wave)
        weights = jnp.ones(64)
        sed = jnp.ones(64) * 1e32
        result = fn(weights, {}, sed, sed)
        assert result.shape == (64,)

    def test_pure_stellar_returns_float64(self):
        fn, _ = self._make_pure_stellar_fn()
        weights = jnp.ones(64)
        sed = jnp.ones(64, dtype=jnp.float64) * 1e32
        result = fn(weights, {}, sed, sed)
        assert result.dtype == jnp.float64


# ---------------------------------------------------------------------------
# NonStellarSlot dataclass
# ---------------------------------------------------------------------------


class TestNonStellarSlot:
    def test_defaults(self):
        slot = NonStellarSlot("nebular")
        assert slot.name == "nebular"
        assert slot.dust_mode == "none"
        assert slot.on_ssp_grid is True

    def test_frozen(self):
        slot = NonStellarSlot("agn", on_ssp_grid=False)
        with pytest.raises((TypeError, AttributeError)):
            slot.name = "radio"  # type: ignore[misc]

    def test_equality(self):
        a = NonStellarSlot("shock", dust_mode="diff")
        b = NonStellarSlot("shock", dust_mode="diff")
        assert a == b
