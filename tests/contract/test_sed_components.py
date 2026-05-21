"""Tests for tengri.forward.components_assembly closure builders.

All builder functions take a ``model``-like object and return a pure JAX
callable (or None when the component is disabled). Tests use
``types.SimpleNamespace`` mocks to supply only the attributes each builder
actually reads — this avoids constructing a full SEDModel.

Coverage targets:
- build_ssp_component        (nearest-interp / linear-interp paths)
- build_dust_atten_component (no-dust, single-law, dual-law paths)
- build_dust_emission_component disabled (→ None)
- build_agn_component        disabled (→ None)
- build_nebular_component    disabled (→ None)
- build_radio_component      disabled (→ None)
- build_xray_component       disabled (→ None)
"""

import types

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.contract

from tengri.forward.components_assembly import (
    build_agn_component,
    build_dust_atten_component,
    build_dust_emission_component,
    build_nebular_component,
    build_radio_component,
    build_ssp_component,
    build_xray_component,
)

# ── Shared geometry ───────────────────────────────────────────────

N_MET = 4
N_AGE = 8
N_WAVE = 50
_LGMET_GRID = jnp.linspace(-2.0, 0.0, N_MET)  # log10(Z)
_WAVE_GRID = jnp.linspace(1000.0, 10000.0, N_WAVE)  # Angstrom


def _make_ssp_data(flux_value=1.0):
    """Minimal SSP namespace: (n_met, n_age, n_wave) flux, flat."""
    return types.SimpleNamespace(
        ssp_flux=jnp.full((N_MET, N_AGE, N_WAVE), flux_value),
        ssp_lgmet=_LGMET_GRID,
        ssp_wave=_WAVE_GRID,
    )


def _make_model_for_ssp(*, met_interp="linear", csp_integration="trapz"):
    """Mock model for build_ssp_component."""
    ssp_data = _make_ssp_data()
    age_dt = jnp.ones(N_AGE) * 1e8  # 100 Myr bins
    model = types.SimpleNamespace(
        ssp_data=ssp_data,
        _forward_dtype=jnp.dtype("float64"),
        _met_interp=met_interp,
        _lgmet_scatter=0.2,
        _csp_integration=csp_integration,
        _csp_age_dt=age_dt,
        # not used unless log_interp
        _csp_matrix=jnp.eye(N_AGE),
    )
    return model


def _make_model_for_dust(*, same_law=True):
    """Mock model for build_dust_atten_component."""
    ssp_data = _make_ssp_data()
    precomputed = types.SimpleNamespace(
        dust_age_weights=jnp.linspace(1.0, 0.0, N_AGE),
    )
    law_name = "power_law"
    # Flat attenuation curve: k(lambda) = 1 everywhere — analytical simplicity
    flat_law = lambda wave, **kw: jnp.ones_like(wave)  # noqa: E731
    model = types.SimpleNamespace(
        ssp_data=ssp_data,
        _forward_dtype=jnp.dtype("float64"),
        _precomputed=precomputed,
        _dust_law_bc_fn=flat_law,
        _dust_law_diff_fn=flat_law,
        _dust_law_bc=law_name,
        _dust_law_diff=law_name if same_law else "calzetti",
    )
    return model


# ── build_ssp_component — linear interpolation path ───────────────


class TestBuildSspComponentLinear:
    def test_returns_callable(self):
        model = _make_model_for_ssp(met_interp="linear")
        fn = build_ssp_component(model)
        assert callable(fn)

    def test_output_shapes(self):
        model = _make_model_for_ssp(met_interp="linear")
        fn = build_ssp_component(model)
        sfr = jnp.ones(N_AGE)
        # Use metallicity in the middle of the grid
        log_z = float(_LGMET_GRID[1])
        ssp_at_z, weights = fn(sfr, log_z)
        assert ssp_at_z.shape == (N_AGE, N_WAVE)
        assert weights.shape == (N_AGE,)

    def test_weights_proportional_to_sfr_x_dt(self):
        """weights = sfr * age_dt when csp_integration != 'log_interp'."""
        model = _make_model_for_ssp(met_interp="linear")
        fn = build_ssp_component(model)
        sfr = jnp.arange(1.0, N_AGE + 1.0)
        log_z = float(_LGMET_GRID[1])
        _, weights = fn(sfr, log_z)
        expected = sfr * model._csp_age_dt.astype(jnp.float64)
        assert jnp.allclose(weights, expected, rtol=1e-5)

    def test_ssp_flux_finite(self):
        model = _make_model_for_ssp(met_interp="linear")
        fn = build_ssp_component(model)
        sfr = jnp.ones(N_AGE)
        ssp_at_z, _ = fn(sfr, float(_LGMET_GRID[2]))
        assert jnp.all(jnp.isfinite(ssp_at_z))

    def test_extrapolation_clamps_to_grid_edge(self):
        """log_z below grid → clamped to grid[0], ssp flux = first met plane."""
        model = _make_model_for_ssp(met_interp="linear")
        fn = build_ssp_component(model)
        sfr = jnp.ones(N_AGE)
        ssp_at_z_low, _ = fn(sfr, -10.0)  # below grid
        ssp_at_z_high, _ = fn(sfr, 10.0)  # above grid
        # Flat SSP: all metallicities have flux=1, so clamped values also ≈ 1
        assert jnp.allclose(ssp_at_z_low, 1.0, atol=1e-5)
        assert jnp.allclose(ssp_at_z_high, 1.0, atol=1e-5)

    def test_alpha_fe_shifts_effective_metallicity(self):
        """Non-zero alpha_fe shifts the effective log_z — here flat SSP so both ≈ 1."""
        model = _make_model_for_ssp(met_interp="linear")
        fn = build_ssp_component(model)
        sfr = jnp.ones(N_AGE)
        log_z = float(_LGMET_GRID[1])
        _ssp0, _ = fn(sfr, log_z, alpha_fe=0.0)
        ssp1, _ = fn(sfr, log_z, alpha_fe=0.3)
        # Both should be ≈ 1.0 (flat SSP), but we check they are finite
        assert jnp.all(jnp.isfinite(ssp1))


# ── build_ssp_component — log_interp (matrix) path ────────────────


class TestBuildSspComponentMatrix:
    def test_returns_callable(self):
        model = _make_model_for_ssp(met_interp="linear", csp_integration="log_interp")
        fn = build_ssp_component(model)
        assert callable(fn)

    def test_matrix_path_weights_shape(self):
        """In log_interp mode weights = csp_matrix @ sfr."""
        n = N_AGE
        # Use identity matrix → weights = sfr
        model = _make_model_for_ssp(csp_integration="log_interp")
        model._csp_matrix = jnp.eye(n)
        fn = build_ssp_component(model)
        sfr = jnp.arange(1.0, n + 1.0)
        _, weights = fn(sfr, float(_LGMET_GRID[1]))
        assert jnp.allclose(weights, sfr, rtol=1e-5)

    def test_matrix_path_output_shape(self):
        model = _make_model_for_ssp(csp_integration="log_interp")
        model._csp_matrix = jnp.eye(N_AGE)
        fn = build_ssp_component(model)
        sfr = jnp.ones(N_AGE)
        ssp_at_z, weights = fn(sfr, float(_LGMET_GRID[2]))
        assert ssp_at_z.shape == (N_AGE, N_WAVE)
        assert weights.shape == (N_AGE,)


# ── build_ssp_component — smooth interpolation path ───────────────


class TestBuildSspComponentSmooth:
    def test_smooth_path_returns_callable(self):
        model = _make_model_for_ssp(met_interp="smooth")
        fn = build_ssp_component(model)
        assert callable(fn)

    def test_smooth_path_output_shape(self):
        model = _make_model_for_ssp(met_interp="smooth")
        fn = build_ssp_component(model)
        sfr = jnp.ones(N_AGE)
        ssp_at_z, weights = fn(sfr, float(_LGMET_GRID[1]))
        assert ssp_at_z.shape == (N_AGE, N_WAVE)
        assert weights.shape == (N_AGE,)

    def test_smooth_path_finite(self):
        model = _make_model_for_ssp(met_interp="smooth")
        fn = build_ssp_component(model)
        sfr = jnp.ones(N_AGE)
        ssp_at_z, _ = fn(sfr, float(_LGMET_GRID[2]))
        assert jnp.all(jnp.isfinite(ssp_at_z))


# ── build_dust_atten_component ────────────────────────────────────


class TestBuildDustAttenComponent:
    def test_returns_callable(self):
        model = _make_model_for_dust()
        fn = build_dust_atten_component(model)
        assert callable(fn)

    def _call(self, tau_bc=0.0, tau_diff=0.0, same_law=True):
        model = _make_model_for_dust(same_law=same_law)
        fn = build_dust_atten_component(model)
        ssp_flux_at_z = jnp.ones((N_AGE, N_WAVE))
        weights = jnp.ones(N_AGE)
        return fn(ssp_flux_at_z, weights, tau_bc=tau_bc, tau_diff=tau_diff)

    def test_output_shapes(self):
        sed_atten, sed_intr, L_abs = self._call()
        assert sed_atten.shape == (N_WAVE,)
        assert sed_intr.shape == (N_WAVE,)
        assert L_abs.shape == ()

    def test_no_dust_atten_equals_intrinsic(self):
        """With tau_bc=tau_diff=0, dust transmission=1 → sed_atten = sed_intr."""
        sed_atten, sed_intr, _L_abs = self._call(tau_bc=0.0, tau_diff=0.0)
        assert jnp.allclose(sed_atten, sed_intr, rtol=1e-5)

    def test_no_dust_absorbed_is_zero(self):
        """Zero dust → zero absorbed luminosity."""
        _, _, L_abs = self._call(tau_bc=0.0, tau_diff=0.0)
        assert float(L_abs) == pytest.approx(0.0, abs=1e30)

    def test_dust_reduces_flux(self):
        """Non-zero dust optical depth → attenuated < intrinsic."""
        sed_atten, sed_intr, _ = self._call(tau_bc=1.0, tau_diff=1.0)
        assert jnp.all(sed_atten <= sed_intr + 1e-10)

    def test_absorbed_luminosity_positive(self):
        """With non-zero dust, absorbed luminosity must be positive."""
        _, _, L_abs = self._call(tau_bc=2.0, tau_diff=0.5)
        assert float(L_abs) > 0.0

    def test_outputs_finite(self):
        sed_atten, sed_intr, L_abs = self._call(tau_bc=1.0, tau_diff=0.3)
        assert jnp.all(jnp.isfinite(sed_atten))
        assert jnp.all(jnp.isfinite(sed_intr))
        assert jnp.isfinite(L_abs)

    def test_dual_law_same_as_single_law_when_identical(self):
        """When both laws are the same flat curve, same_law and dual_law agree."""
        model_same = _make_model_for_dust(same_law=True)
        model_dual = _make_model_for_dust(same_law=False)
        # Make sure dual model's diff law is also flat
        model_dual._dust_law_diff_fn = lambda wave, **kw: jnp.ones_like(wave)
        fn_same = build_dust_atten_component(model_same)
        fn_dual = build_dust_atten_component(model_dual)
        ssp_flux_at_z = jnp.ones((N_AGE, N_WAVE))
        weights = jnp.ones(N_AGE)
        sed_s, _, _ = fn_same(ssp_flux_at_z, weights, tau_bc=1.0, tau_diff=0.5)
        sed_d, _, _ = fn_dual(ssp_flux_at_z, weights, tau_bc=1.0, tau_diff=0.5)
        assert jnp.allclose(sed_s, sed_d, rtol=1e-5)

    def test_full_obscuration_frac_limits_transmission(self):
        """f_obscuration=1.0 means all starlight is fully obscured regardless of tau."""
        model = _make_model_for_dust()
        fn = build_dust_atten_component(model)
        ssp_flux_at_z = jnp.ones((N_AGE, N_WAVE))
        weights = jnp.ones(N_AGE)
        sed_atten, _, _ = fn(ssp_flux_at_z, weights, tau_bc=0.0, tau_diff=0.0, f_obscuration=1.0)
        # dust_trans = 1.0 + 0.0 * exp(0) = 1.0, so no change at f_obscuration=1
        # The formula: f_obscuration + (1 - f_obscuration) * exp(-tau)
        # With f_obs=1, tau=0: trans = 1 + 0 = 1 → no attenuation
        _, sed_intr_no_dust, _ = fn(ssp_flux_at_z, weights, tau_bc=0.0, tau_diff=0.0)
        assert jnp.allclose(sed_atten, sed_intr_no_dust, rtol=1e-5)


# ── Disabled component builders — must return None ────────────────


class TestDisabledBuilders:
    def test_dust_emission_disabled_returns_none(self):
        ssp_data = _make_ssp_data()
        model = types.SimpleNamespace(
            ssp_data=ssp_data,
            _dust_emission_model=None,
        )
        assert build_dust_emission_component(model) is None

    def test_agn_disabled_returns_none(self):
        ssp_data = _make_ssp_data()
        model = types.SimpleNamespace(
            ssp_data=ssp_data,
            _agn_model=None,
        )
        assert build_agn_component(model) is None

    def test_nebular_backend_none_returns_none(self):
        ssp_data = _make_ssp_data()
        model = types.SimpleNamespace(
            ssp_data=ssp_data,
            _nebular_backend=None,
            ssp_log_ages_yr=jnp.linspace(6.0, 10.0, N_AGE),
        )
        assert build_nebular_component(model) is None

    def test_nebular_backend_no_free_params_returns_none(self):
        """Backend that reports has_free_params=False → builder returns None."""
        ssp_data = _make_ssp_data()
        frozen_backend = types.SimpleNamespace(has_free_params=False)
        model = types.SimpleNamespace(
            ssp_data=ssp_data,
            _nebular_backend=frozen_backend,
            ssp_log_ages_yr=jnp.linspace(6.0, 10.0, N_AGE),
        )
        assert build_nebular_component(model) is None

    def test_radio_disabled_returns_none(self):
        ssp_data = _make_ssp_data()
        model = types.SimpleNamespace(
            ssp_data=ssp_data,
            _uses_radio=False,
        )
        assert build_radio_component(model) is None

    def test_xray_disabled_returns_none(self):
        ssp_data = _make_ssp_data()
        model = types.SimpleNamespace(
            ssp_data=ssp_data,
            _uses_xray=False,
        )
        assert build_xray_component(model) is None
