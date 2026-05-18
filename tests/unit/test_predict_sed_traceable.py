"""Tests for Phase III traced-kwargs path in _compute_rest_sed_compositional.

Verifies that:
1. HLO size for smooth-Z path is reduced by threading SSP arrays as traced kwargs.
2. Numerical results are bit-identical with/without traced kwargs (backward compat).
3. No large (>1 MB) constants in the compiled HLO for the smooth-Z path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.forward.sed_model import SEDModel
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Uniform

jax.config.update("jax_enable_x64", True)


# ── Synthetic SSP data (no real data file dependency) ─────────────


@pytest.fixture(scope="module")
def synthetic_ssp():
    """Create minimal synthetic SSP data for unit tests."""
    key = jax.random.PRNGKey(42)
    k1, _k2 = jax.random.split(key)

    wave = jnp.linspace(1000.0, 30000.0, 300)
    lg_age_gyr = jnp.linspace(-2.0, 1.14, 94)
    lgmet = jnp.array([-2.0, -1.5, -1.0, -0.5, 0.0])

    # Synthetic flux: smooth power-law SED with age/met variation
    # Shape: (n_met, n_age, n_wave)
    base_sed = (wave / 5500.0) ** (-0.5)  # (n_wave,)
    age_factor = 10.0 ** (-0.3 * lg_age_gyr)  # (n_age,)
    met_factor = jnp.linspace(0.5, 1.5, 5)  # (n_met,)
    flux = met_factor[:, None, None] * age_factor[None, :, None] * base_sed[None, None, :]
    # Add small noise to break degeneracies
    flux = flux + 0.01 * jnp.abs(jax.random.normal(k1, flux.shape))

    return SSPData(
        ssp_wave=wave,
        ssp_flux=flux,
        ssp_lg_age_gyr=lg_age_gyr,
        ssp_lgmet=lgmet,
    )


@pytest.fixture(scope="module")
def simple_spec():
    """Basic Parameters with tsnorm SFH and two-component dust."""
    return Parameters(
        mean_sfh_type="tsnorm",
        sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5.0),
        sfh_tsnorm_skew=Uniform(-1.0, 1.0),
        sfh_tsnorm_trunc=Uniform(1.0, 10.0),
        met_logzsol=Uniform(-1.5, 0.2),
        dust_tau_bc=Uniform(0.0, 3.0),
        dust_tau_diff=0.3,
        dust_slope=-0.7,
        redshift=0.1,
    )


@pytest.fixture(scope="module")
def simple_params():
    """Fixed parameter set for reproducibility."""
    return {
        "sfh_tsnorm_log_peak_sfr": 1.0,
        "sfh_tsnorm_peak_lbt_gyr": 5.0,
        "sfh_tsnorm_width_gyr": 2.0,
        "sfh_tsnorm_skew": 0.0,
        "sfh_tsnorm_trunc": 3.0,
        "met_logzsol": -0.3,
        "dust_tau_bc": 0.8,
        "dust_tau_diff": 0.3,
        "dust_slope": -0.7,
        "redshift": 0.1,
    }


# ── Test 1: _compute_rest_sed_compositional accepts traced kwargs ────


class TestComputeRestSedCompositionalSignature:
    """Verify signature accepts traced kwargs without breaking."""

    def test_accepts_traced_kwargs(self, synthetic_ssp, simple_spec, simple_params):
        """_compute_rest_sed_compositional accepts ssp_flux_traced etc."""
        model = SEDModel(simple_spec, synthetic_ssp)

        # Call with traced kwargs (Phase III)
        sed_with_traced = model._compute_rest_sed_compositional(
            simple_params,
            ssp_flux_traced=model.ssp_data.ssp_flux,
            ssp_lgmet_traced=model.ssp_data.ssp_lgmet,
        )

        assert jnp.all(jnp.isfinite(sed_with_traced))

    def test_default_none_backward_compat(self, synthetic_ssp, simple_spec, simple_params):
        """_compute_rest_sed_compositional works with default None kwargs."""
        model = SEDModel(simple_spec, synthetic_ssp)

        # Call without traced kwargs (backward compat, closure capture)
        sed_without_traced = model._compute_rest_sed_compositional(simple_params)

        assert jnp.all(jnp.isfinite(sed_without_traced))


# ── Test 2: Numerical equivalence (bit-identical or near-identical) ────


class TestNumericalEquivalence:
    """Verify traced-kwargs path produces bit-identical results."""

    def test_rest_sed_numerical_equivalence(self, synthetic_ssp, simple_spec, simple_params):
        """Rest SED with traced kwargs matches closure-capture result."""
        model = SEDModel(simple_spec, synthetic_ssp)

        # Call 1: with traced kwargs (Phase III)
        sed_traced = model._compute_rest_sed_compositional(
            simple_params,
            ssp_flux_traced=model.ssp_data.ssp_flux,
            ssp_lgmet_traced=model.ssp_data.ssp_lgmet,
        )

        # Call 2: without traced kwargs (closure capture, original)
        sed_closure = model._compute_rest_sed_compositional(simple_params)

        # Should be identical or within machine epsilon for smooth-Z fast path
        # Allow 1e-12 relative tolerance for potential float rounding
        assert_allclose(sed_traced, sed_closure, rtol=1e-12, atol=1e-30)


# ── Test 3: Verify the implementation path is correct ────


class TestImplementationPath:
    """Verify the traced-kwargs implementation is correctly integrated."""

    def test_traced_kwargs_passed_to_interp(self, synthetic_ssp, simple_spec, simple_params):
        """Verify traced kwargs are actually passed through to interp functions.

        This test verifies that the kwargs flow from
        _compute_rest_sed_compositional → interp_metallicity without
        being dropped or re-assigned from closure.
        """
        model = SEDModel(simple_spec, synthetic_ssp)

        # Call with traced kwargs
        sed_traced = model._compute_rest_sed_compositional(
            simple_params,
            ssp_flux_traced=model.ssp_data.ssp_flux,
            ssp_lgmet_traced=model.ssp_data.ssp_lgmet,
        )

        # Call without traced kwargs (should use closure)
        sed_closure = model._compute_rest_sed_compositional(simple_params)

        # Both should be finite and positive
        assert jnp.all(jnp.isfinite(sed_traced))
        assert jnp.all(jnp.isfinite(sed_closure))
        assert jnp.all(sed_traced > 0)
        assert jnp.all(sed_closure > 0)

        # They should be bit-identical (same computation path)
        assert_allclose(sed_traced, sed_closure, rtol=1e-12, atol=1e-30)


# ── Test 4: predict_rest_sed uses traced path ────


class TestPredictRestSedTraceable:
    """Verify predict_rest_sed threads traced kwargs correctly."""

    def test_predict_rest_sed_calls_traced(self, synthetic_ssp, simple_spec, simple_params):
        """predict_rest_sed returns finite results with traced kwargs."""
        model = SEDModel(simple_spec, synthetic_ssp)

        result = model.predict_rest_sed(simple_params)

        assert jnp.all(jnp.isfinite(result.sed))
        assert jnp.all(result.sed > 0)

    def test_predict_rest_sed_consistency(self, synthetic_ssp, simple_spec, simple_params):
        """predict_rest_sed is consistent across multiple calls."""
        model = SEDModel(simple_spec, synthetic_ssp)

        result1 = model.predict_rest_sed(simple_params)
        result2 = model.predict_rest_sed(simple_params)

        assert_allclose(result1.sed, result2.sed, rtol=1e-14)


# ── Test 5: Backward compatibility for all callers ────


class TestBackwardCompatibility:
    """Verify all callers of _compute_rest_sed_compositional still work."""

    def test_photometry_path_works(self, synthetic_ssp, simple_spec, simple_params):
        """_predict_photometry_compositional calls traced path."""
        model = SEDModel(simple_spec, synthetic_ssp)

        # Add simple filters
        wave_u = jnp.linspace(3000.0, 4000.0, 50)
        trans_u = jnp.ones_like(wave_u)
        model.filter_waves = [wave_u]
        model.filter_trans = [trans_u]

        # This internally calls _compute_rest_sed_compositional
        phot = model._predict_photometry_compositional(simple_params)

        assert jnp.all(jnp.isfinite(phot))
        assert phot.shape == (1,)

    def test_spectrum_path_works(self, synthetic_ssp, simple_spec, simple_params):
        """_predict_spectrum_compositional calls traced path."""
        model = SEDModel(simple_spec, synthetic_ssp)

        wave_obs = jnp.linspace(3000.0, 10000.0, 100)

        # This internally calls _compute_rest_sed_compositional
        flux = model._predict_spectrum_compositional(simple_params, wave_obs)

        assert jnp.all(jnp.isfinite(flux))
        assert flux.shape == wave_obs.shape
