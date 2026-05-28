# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: RUF012
"""WavePrecomp integration tests for SEDModelComponent base class.

Verifies that the base class automatically participates in the fast LUT path
when WavePrecomp is active, and that Taylor refinement works correctly.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pytest

from tengri import Uniform
from tengri.components.sed_model_component import SEDModelComponent
from tengri.protocols.component import ForwardState

pytestmark = pytest.mark.contract


class SimpleDustComponent(SEDModelComponent):
    """Minimal additive dust-emission component for testing WavePrecomp.

    Uses an additive emission template (not multiplicative attenuation) so the
    default :meth:`SEDModelComponent.predict_precomp` — which passes
    ``sed_in=zeros`` — exercises a meaningful code path.
    """

    name = "test_dust"
    parameter_prefix = "dust_"
    tau_v = Uniform(
        0.0, 2.0, default=0.3, description="Emission amplitude", units="dimensionless"
    )

    outputs = {"L_ir": "erg/s"}

    def predict(
        self, p: dict[str, Any], sed_in: jnp.ndarray, wave: jnp.ndarray, **inputs: Any
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Add a wavelength-dependent emission template scaled by ``tau_v``."""
        tau_v = p["tau_v"]
        # Inverse-wavelength emission template, normalized at V-band
        template = 1.0 / (wave / 5500.0)
        emission = tau_v * template
        sed_out = sed_in + emission

        # Total IR luminosity as a sanity check
        l_ir = jnp.sum(emission)
        return sed_out, {"L_ir": l_ir}


class SimpleDustComponentWithTaylor(SEDModelComponent):
    """Additive dust-emission component with Taylor order=1 for slope LUTs."""

    name = "test_dust_taylor"
    parameter_prefix = "dust_taylor_"
    tau_v = Uniform(
        0.0, 2.0, default=0.3, description="Emission amplitude", units="dimensionless"
    )
    taylor_order = 1

    outputs = {"L_ir": "erg/s"}

    def predict(
        self, p: dict[str, Any], sed_in: jnp.ndarray, wave: jnp.ndarray, **inputs: Any
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Same additive inverse-wavelength template as :class:`SimpleDustComponent`."""
        tau_v = p["tau_v"]
        template = 1.0 / (wave / 5500.0)
        emission = tau_v * template
        sed_out = sed_in + emission
        l_ir = jnp.sum(emission)
        return sed_out, {"L_ir": l_ir}


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

        # Should NOT update sed_intrinsic on LUT path (same array object passed through)
        assert result.sed_intrinsic is state.sed_intrinsic
        # Should publish precomp LUT
        assert "test_dust_phot_lnu_precomp" in result.derived
        lut = result.derived["test_dust_phot_lnu_precomp"]
        assert lut.shape == filter_eff.shape

    def test_precomp_photometry_accuracy(self):
        """Verify WavePrecomp photometry within ~1% of exact full-grid path.

        Default :meth:`predict_precomp` evaluates :meth:`predict` with
        ``sed_in=zeros``, so the comparison uses a zero baseline SED — the
        precomp LUT then equals the emission template at filter wavelengths,
        which is what the full-grid path also produces at those points.
        """
        comp = SimpleDustComponent()
        wave = jnp.linspace(1000, 10000, 500)
        filter_eff = jnp.array([3500.0, 4700.0, 5500.0, 7000.0, 9000.0])
        sed_in = jnp.zeros(500)
        params = {"dust_tau_v": 0.7}

        # Full-grid path: apply component, then interpolate to filter wavelengths
        state_full = ForwardState(wave=wave, sed_intrinsic=sed_in, derived={})
        result_full = comp.apply(state_full, params)
        sed_full = result_full.sed_intrinsic

        from scipy.interpolate import interp1d

        sed_interp = interp1d(wave, sed_full, kind="linear", fill_value="extrapolate")
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
        assert params[0].units == "dimensionless"

    def test_inputs_outputs_unchanged(self):
        """inputs() and outputs() should be unaffected."""
        comp = SimpleDustComponent()

        assert len(comp.inputs()) == 0
        assert len(comp.outputs()) == 1
        assert comp.outputs()[0].name == "L_ir"
        assert comp.outputs()[0].units == "erg/s"
