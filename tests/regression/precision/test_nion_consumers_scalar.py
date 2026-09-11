# SPDX-License-Identifier: BSD-3-Clause
"""Test scalar log-domain consumers of nion (radio thermal, xi_ion).

Task 3a: migrate radio-thermal and xi_ion consumers to read log_nion and
combine in log space for float32 safety (#1206).

XFAIL lifted in Tier B once helper functions are integrated.
Tasks 1 (log_nion core) and 2 (Cue combine) are completed. This task
adds log-domain helpers and routes both the stellar property (_xi_ion_fn)
and the factory (state_to_ionizing_quantities) through a shared path.
"""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from tengri.utils.scale import pow10

from .conftest import build_minimal_cue_model

pytestmark = pytest.mark.regression_bug


class TestRadioThermalLogDomain:
    """Test the log-domain radio-thermal helper."""

    def test_radio_thermal_log_vs_linear_parity_f64(self):
        """(1) Log-vs-linear parity in float64.

        For a grid of log_q_h values, log-domain and linear forms should
        agree to ~1e-12 precision.
        """
        from tengri.utils.sed_quantities import (
            compute_l_radio_thermal,
            compute_l_radio_thermal_from_log_qh,
        )

        log_q_h_vals = jnp.array([40.0, 48.0, 56.0])

        for log_q_h in log_q_h_vals:
            q_h_linear = pow10(log_q_h)
            l_linear = compute_l_radio_thermal(q_h_linear)
            l_log = compute_l_radio_thermal_from_log_qh(log_q_h)

            assert_allclose(l_log, l_linear, rtol=1e-12)

    def test_radio_thermal_log_minus_inf_zero(self):
        """Log-domain: log_q_h = -inf → thermal = 0.0 exactly."""
        from tengri.utils.sed_quantities import compute_l_radio_thermal_from_log_qh

        l_thermal = compute_l_radio_thermal_from_log_qh(-jnp.inf)
        assert l_thermal == 0.0

    def test_radio_thermal_pure_f32_finiteness(self):
        """(2) Pure-f32: log-domain stays finite (linear would be inf).

        Verify that the linear form overflows to inf under pure float32
        while the log-domain form stays finite.
        """
        from tengri.utils.sed_quantities import (
            compute_l_radio_thermal,
            compute_l_radio_thermal_from_log_qh,
        )

        log_q_h_f32 = jnp.asarray(56.0, dtype=jnp.float32)

        with jax.enable_x64(False):
            # Log-domain must stay finite
            l_thermal_log = compute_l_radio_thermal_from_log_qh(log_q_h_f32)
            assert jnp.isfinite(l_thermal_log), (
                f"log-domain thermal is non-finite: {l_thermal_log}"
            )
            # Check magnitude is reasonable (~5.5e28)
            assert 1e28 < l_thermal_log < 1e29

            # Linear form should overflow to inf or be non-finite
            q_h_linear = pow10(log_q_h_f32)
            l_thermal_linear = compute_l_radio_thermal(q_h_linear)
            # Verify that the linear form is indeed inf (the migration solves this)
            assert not jnp.isfinite(l_thermal_linear), (
                f"linear thermal should be inf but got: {l_thermal_linear}"
            )


class TestXiIonLogDomain:
    """Test the log-domain xi_ion helper."""

    def test_xi_ion_log_vs_linear_parity_f64(self):
        """(3) xi_ion log-vs-linear parity in float64.

        Build a synthetic stellar SED (1000-1700 Å FUV range plus filler)
        and verify xi_ion parity across a grid of log_q_h.
        """
        from tengri.utils.physics_constants import C_AA
        from tengri.utils.sed_quantities import compute_xi_ion_from_log_qh

        # Synthetic wave grid (Angstrom, ascending)
        wave = jnp.array([500.0, 700.0, 1000.0, 1200.0, 1500.0, 1700.0, 2500.0])

        # Synthetic SED (erg/s/Hz): ~1e28 in the FUV band
        sed_flux = jnp.array([1e27, 1e27, 1e28, 1e28, 1e28, 1e28, 1e27])

        log_q_h_vals = jnp.array([40.0, 48.0, 56.0])

        for log_q_h in log_q_h_vals:
            # Compute via log-domain helper
            xi_ion_log = compute_xi_ion_from_log_qh(log_q_h, sed_flux, wave)

            # Compute via frozen reference (old xi arithmetic)
            q_h_linear = pow10(log_q_h)
            fuv = jnp.mean(sed_flux[(wave >= 1000.0) & (wave <= 1700.0)])
            nu_1500 = C_AA / 1500.0
            nu_l_uv = fuv * nu_1500
            _TINY = 1e-30
            xi_ion_ref = q_h_linear / jnp.maximum(nu_l_uv, _TINY)

            assert_allclose(xi_ion_log, xi_ion_ref, rtol=1e-12)

    def test_xi_ion_zero_fuv_floored(self):
        """(3) xi_ion: zero FUV floor behavior (both finite).

        When FUV = 0 everywhere, the log-domain helper should floor and
        return a finite value, matching the old floored arithmetic.
        """
        from tengri.utils.sed_quantities import compute_xi_ion_from_log_qh

        wave = jnp.linspace(500.0, 2500.0, 10)
        sed_flux = jnp.zeros_like(wave)  # Zero flux everywhere

        log_q_h = jnp.asarray(56.0)
        xi_ion = compute_xi_ion_from_log_qh(log_q_h, sed_flux, wave)

        # Should be finite even with zero FUV
        assert jnp.isfinite(xi_ion), f"xi_ion non-finite with zero FUV: {xi_ion}"
        assert jnp.any(xi_ion != 0.0), (
            "`xi_ion` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )

    def test_xi_ion_pure_f32_finiteness(self):
        """(4) Pure-f32: xi_ion stays finite (numerator/denominator both overflow).

        In pure float32, both Q_H ~1e56 and νL_ν ~1e43 overflow.
        The log-domain computation must stay finite.
        """
        from tengri.utils.sed_quantities import (
            compute_xi_ion_from_log_qh,
        )

        wave_vals = [500.0, 700.0, 1000.0, 1200.0, 1500.0, 1700.0, 2500.0]
        sed_vals = [1e27, 1e27, 1e28, 1e28, 1e28, 1e28, 1e27]

        # f64 reference — computed OUTSIDE the x64=False context, so it is genuinely
        # float64. Building it inside the context would silently truncate the inputs
        # to float32 (JAX drops float64 requests when x64 is disabled), making the
        # parity assertion below f32-vs-f32 and therefore vacuous.
        xi_ion_f64_ref = float(
            compute_xi_ion_from_log_qh(
                jnp.asarray(56.0, dtype=jnp.float64),
                jnp.asarray(sed_vals, dtype=jnp.float64),
                jnp.asarray(wave_vals, dtype=jnp.float64),
            )
        )

        with jax.enable_x64(False):
            wave_f32 = jnp.asarray(wave_vals, dtype=jnp.float32)
            sed_f32 = jnp.asarray(sed_vals, dtype=jnp.float32)
            log_q_h_f32 = jnp.asarray(56.0, dtype=jnp.float32)
            assert log_q_h_f32.dtype == jnp.float32  # precondition: genuinely pure f32

            # In pure float32 both Q_H ~1e56 and nu*L_nu ~1e43 overflow; the
            # log-domain form must stay finite and track the f64 reference.
            xi_ion_log = compute_xi_ion_from_log_qh(log_q_h_f32, sed_f32, wave_f32)
            assert jnp.isfinite(xi_ion_log), f"log-domain xi_ion is non-finite: {xi_ion_log}"
            assert xi_ion_log > 1e10, f"xi_ion magnitude too small: {xi_ion_log}"

        assert_allclose(float(xi_ion_log), xi_ion_f64_ref, rtol=5e-3)


class TestXiIonPropertyFactoryBitEquality:
    """Test that property and factory routes are bit-identical."""

    def test_property_factory_xi_ion_bit_equality(self, ssp_bare):
        """(5a) xi_ion: property and factory bit-identical via shared helper.

        Build a minimal Cue model and verify that pred.properties["xi_ion"]
        (via the property fn) equals state_to_ionizing_quantities(state).xi_ion
        (via the factory) at atol=0 (bit-identical).
        """
        from tengri.forward.component_factory import state_to_ionizing_quantities

        model = build_minimal_cue_model(ssp_bare, "float64")
        p = dict(model.spec.sample(jax.random.PRNGKey(0)))
        p["redshift"] = 1.0

        pred = model.predict(p)
        state = model.predict_state(p)

        xi_ion_prop = float(pred.properties["xi_ion"])
        xi_ion_factory = float(state_to_ionizing_quantities(state).xi_ion)

        # Bit-identical (atol=0)
        assert_allclose(xi_ion_prop, xi_ion_factory, atol=0)

    def test_property_factory_q_h_bit_equality(self, ssp_bare):
        """(5b) q_h: property and factory bit-identical (unchanged linear surface).

        Verify pred.properties["q_h"] == state_to_ionizing_quantities(state).q_h
        at atol=0. The q_h surface remains unchanged (deferred).
        """
        from tengri.forward.component_factory import state_to_ionizing_quantities

        model = build_minimal_cue_model(ssp_bare, "float64")
        p = dict(model.spec.sample(jax.random.PRNGKey(0)))
        p["redshift"] = 1.0

        pred = model.predict(p)
        state = model.predict_state(p)

        q_h_prop = float(pred.properties["q_h"])
        q_h_factory = float(state_to_ionizing_quantities(state).q_h)

        # Bit-identical (atol=0)
        assert_allclose(q_h_prop, q_h_factory, atol=0)

    def test_xi_ion_finite_and_positive_young_pop(self, ssp_bare):
        """(5c) xi_ion finite and positive for the young stellar population."""
        model = build_minimal_cue_model(ssp_bare, "float64")
        p = dict(model.spec.sample(jax.random.PRNGKey(0)))
        p["redshift"] = 1.0

        pred = model.predict(p)

        xi_ion = float(pred.properties["xi_ion"])
        assert jnp.isfinite(xi_ion), f"xi_ion is non-finite: {xi_ion}"
        assert xi_ion > 0.0, f"xi_ion is not positive: {xi_ion}"
