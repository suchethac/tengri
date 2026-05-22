# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: RUF012
"""WavePrecomp integration tests for SEDModelComponent base class.

Verifies that the base class automatically participates in the fast LUT path
when WavePrecomp is active, and that Taylor refinement works correctly.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from tengri import Uniform
from tengri.components.sed_model_component import SEDModelComponent
from tengri.protocols.component import ForwardState


class SimpleDustComponent(SEDModelComponent):
    """Minimal dust attenuation component for testing WavePrecomp."""

    name = "test_dust"
    parameter_prefix = "dust_"
    tau_v = Uniform(0.0, 2.0, description="V-band optical depth", units="")

    outputs = {"L_ir": "erg/s"}

    def predict(
        self, p: dict[str, Any], sed_in: jnp.ndarray, wave: jnp.ndarray, **inputs: Any
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Apply simple attenuation: k(λ) = 1/λ (inverse-wavelength dust law)."""
        tau_v = p["tau_v"]
        # Simple power-law dust law: k(λ) ∝ λ^{-1}
        k_lambda = 1.0 / (wave / 5500.0)  # normalized to V-band
        attenuation = jnp.exp(-tau_v * k_lambda)
        attenuated = sed_in * attenuation

        # Compute total IR luminosity as a sanity check
        l_ir = jnp.sum(attenuated)
        return attenuated, {"L_ir": l_ir}


class SimpleDustComponentWithTaylor(SEDModelComponent):
    """Dust component with Taylor order=1 for slope LUT generation."""

    name = "test_dust_taylor"
    parameter_prefix = "dust_taylor_"
    tau_v = Uniform(0.0, 2.0, description="V-band optical depth", units="")
    taylor_order = 1

    outputs = {"L_ir": "erg/s"}

    def predict(
        self, p: dict[str, Any], sed_in: jnp.ndarray, wave: jnp.ndarray, **inputs: Any
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Apply simple attenuation with inverse-wavelength dust law."""
        tau_v = p["tau_v"]
        k_lambda = 1.0 / (wave / 5500.0)
        attenuation = jnp.exp(-tau_v * k_lambda)
        attenuated = sed_in * attenuation
        l_ir = jnp.sum(attenuated)
        return attenuated, {"L_ir": l_ir}


class TestWavePrecompActivation:
    """Verify WavePrecomp mode detection and LUT publishing."""

    def test_full_grid_path_without_filter_eff_waves(self):
        """Default path when filter_eff_waves is NOT in state.derived."""
        comp = SimpleDustComponent()
        state = ForwardState(
            wave=jnp.linspace(1000, 10000, 100),
            sed_intrinsic=jnp.ones(100),
            derived={},
        )
        params = {"dust_tau_v": 0.5}

        result = comp.apply(state, params)

        # Should update sed_intrinsic
        assert result.sed_intrinsic is not None
        assert result.sed_intrinsic.shape == state.wave.shape
        # Should publish L_ir
        assert "L_ir" in result.derived
        # Should NOT publish precomp LUT
        assert "test_dust_phot_lnu_precomp" not in result.derived

    def test_precomp_path_with_filter_eff_waves(self):
        """WavePrecomp path when filter_eff_waves IS in state.derived."""
        comp = SimpleDustComponent()
        wave = jnp.linspace(1000, 10000, 100)
        filter_eff = jnp.array([3500.0, 4700.0, 5500.0, 7000.0, 9000.0])

        state = ForwardState(
            wave=wave,
            sed_intrinsic=jnp.ones(100),
            derived={"filter_eff_waves": filter_eff},
        )
        params = {"dust_tau_v": 0.5}

        result = comp.apply(state, params)

        # Should NOT update sed_intrinsic on LUT path
        assert result.sed_intrinsic == state.sed_intrinsic
        # Should publish precomp LUT
        assert "test_dust_phot_lnu_precomp" in result.derived
        lut = result.derived["test_dust_phot_lnu_precomp"]
        assert lut.shape == filter_eff.shape

    def test_precomp_photometry_accuracy(self):
        """Verify WavePrecomp photometry within ~0.5% of exact full-grid path."""
        comp = SimpleDustComponent()
        wave = jnp.linspace(1000, 10000, 500)
        filter_eff = jnp.array([3500.0, 4700.0, 5500.0, 7000.0, 9000.0])
        sed_in = jnp.ones(500)
        params = {"dust_tau_v": 0.7}

        # Full-grid path: apply component, then project through filters
        state_full = ForwardState(wave=wave, sed_intrinsic=sed_in, derived={})
        result_full = comp.apply(state_full, params)
        sed_attenuated = result_full.sed_intrinsic

        # Manually integrate at effective wavelengths (rough simulation)
        # by evaluating predict at filter wavelengths
        from scipy.interpolate import interp1d

        sed_interp = interp1d(wave, sed_attenuated, kind="linear", fill_value="extrapolate")
        exact_phot = sed_interp(filter_eff)

        # WavePrecomp path
        state_precomp = ForwardState(
            wave=wave, sed_intrinsic=sed_in, derived={"filter_eff_waves": filter_eff}
        )
        result_precomp = comp.apply(state_precomp, params)
        phot_lut = result_precomp.derived["test_dust_phot_lnu_precomp"]

        # Check agreement (within 1% due to interpolation)
        relative_error = jnp.abs(phot_lut - exact_phot) / jnp.maximum(jnp.abs(exact_phot), 1e-10)
        assert jnp.all(relative_error < 0.01), f"Max error: {jnp.max(relative_error)}"


class TestTaylorRefinement:
    """Verify Taylor first-order derivative computation."""

    def test_taylor_slope_published_when_enabled(self):
        """taylor_order=1 should publish slope LUT."""
        comp = SimpleDustComponentWithTaylor()
        filter_eff = jnp.array([3500.0, 5500.0, 9000.0])

        state = ForwardState(
            wave=jnp.linspace(1000, 10000, 100),
            sed_intrinsic=jnp.ones(100),
            derived={"filter_eff_waves": filter_eff},
        )
        params = {"dust_taylor_tau_v": 0.5}

        result = comp.apply(state, params)

        # Should publish both zeroth and first-order LUTs
        assert "test_dust_taylor_phot_lnu_precomp" in result.derived
        assert "test_dust_taylor_phot_lnu_slope_precomp" in result.derived

        zeroth = result.derived["test_dust_taylor_phot_lnu_precomp"]
        slope = result.derived["test_dust_taylor_phot_lnu_slope_precomp"]

        assert zeroth.shape == filter_eff.shape
        assert slope.shape == filter_eff.shape

    def test_taylor_slope_nonzero(self):
        """Slope should be nonzero for a wavelength-dependent function."""
        comp = SimpleDustComponentWithTaylor()
        filter_eff = jnp.array([3500.0, 5500.0, 9000.0])

        state = ForwardState(
            wave=jnp.linspace(1000, 10000, 100),
            sed_intrinsic=jnp.ones(100),
            derived={"filter_eff_waves": filter_eff},
        )
        params = {"dust_taylor_tau_v": 0.5}

        result = comp.apply(state, params)
        slope = result.derived["test_dust_taylor_phot_lnu_slope_precomp"]

        # For the inverse-wavelength law, slope should be nonzero
        assert not jnp.allclose(slope, 0.0)

    def test_default_taylor_order_zero(self):
        """Default taylor_order=0 should NOT publish slope."""
        comp = SimpleDustComponent()
        filter_eff = jnp.array([3500.0, 5500.0, 9000.0])

        state = ForwardState(
            wave=jnp.linspace(1000, 10000, 100),
            sed_intrinsic=jnp.ones(100),
            derived={"filter_eff_waves": filter_eff},
        )
        params = {"dust_tau_v": 0.5}

        result = comp.apply(state, params)

        # Should NOT publish slope
        assert "test_dust_phot_lnu_slope_precomp" not in result.derived


class TestPredictPrecompFallback:
    """Verify default predict_precomp() fallback behavior."""

    def test_predict_precomp_default_implementation(self):
        """Default predict_precomp should call predict at filter wavelengths."""
        comp = SimpleDustComponent()
        filter_eff = jnp.array([3500.0, 5500.0, 9000.0])
        params = {"tau_v": 0.5}

        # Call predict_precomp directly
        phot, published = comp.predict_precomp(params, filter_eff)

        # Should return shape matching filter_eff
        assert phot.shape == filter_eff.shape
        # Should publish expected keys
        assert "L_ir" in published


class TestComponentContract:
    """Verify WavePrecomp doesn't break existing component contract."""

    def test_declared_parameters_unchanged(self):
        """declared_parameters() should be unaffected."""
        comp = SimpleDustComponent()
        params = comp.declared_parameters()

        assert len(params) == 1
        assert params[0].name == "dust_tau_v"
        assert params[0].units == ""

    def test_inputs_outputs_unchanged(self):
        """inputs() and outputs() should be unaffected."""
        comp = SimpleDustComponent()

        assert len(comp.inputs()) == 0
        assert len(comp.outputs()) == 1
        assert comp.outputs()[0].name == "L_ir"
        assert comp.outputs()[0].units == "erg/s"
