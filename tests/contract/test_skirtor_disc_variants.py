# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for SKIRTOR disc spectrum selection and delta modulation.

Exercises the new disc_type and delta parameters added to SKIRTORTorus:
- disc_type selects between SKIRTOR / Schartmann2005 / ADAF+disc (static, build-time).
- delta modulates the disc spectral slope (dynamic, free parameter).

This test verifies that the disc_cigale functions are JIT-compatible and that
delta produces expected slope changes. It uses synthetic fixtures to avoid
requiring SKIRTOR templates.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._grad_parity import assert_grad_matches_fd

pytestmark = pytest.mark.contract

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def wave_grid():
    """Synthetic wavelength grid covering UV to mid-IR."""
    return np.logspace(2, 6, 256)  # 100 Å to 1 µm in log spacing


class TestDiscSpectrumFunctions:
    """Test the disc_cigale functions for JIT safety and basic properties."""

    def test_skirtor_disk_spectrum_jit(self, wave_grid):
        """SKIRTOR disc spectrum is JIT-compatible."""
        from tengri.components.agn.disc_cigale import skirtor_disk_spectrum

        # Convert to nanometers for the function
        wave_nm = wave_grid / 10.0
        delta = 0.0

        spec = skirtor_disk_spectrum(wave_nm, delta=delta)
        assert spec.shape == wave_nm.shape
        chex.assert_tree_all_finite(spec)

        # JIT it
        jit_fn = jax.jit(skirtor_disk_spectrum)
        spec_jit = jit_fn(wave_nm, delta=delta)
        chex.assert_trees_all_close(spec, spec_jit, rtol=1e-6)

    def test_schartmann2005_disk_spectrum_jit(self, wave_grid):
        """Schartmann 2005 disc spectrum is JIT-compatible."""
        from tengri.components.agn.disc_cigale import schartmann2005_disk_spectrum

        wave_nm = wave_grid / 10.0
        delta = 0.0

        spec = schartmann2005_disk_spectrum(wave_nm, delta=delta)
        assert spec.shape == wave_nm.shape
        chex.assert_tree_all_finite(spec)

        jit_fn = jax.jit(schartmann2005_disk_spectrum)
        spec_jit = jit_fn(wave_nm, delta=delta)
        chex.assert_trees_all_close(spec, spec_jit, rtol=1e-6)

    def test_adaf_disk_spectrum_jit(self, wave_grid):
        """ADAF disc spectrum is JIT-compatible."""
        from tengri.components.agn.disc_cigale import adaf_disk_spectrum

        wave_nm = wave_grid / 10.0
        delta = 0.0

        spec = adaf_disk_spectrum(wave_nm, delta=delta)
        assert spec.shape == wave_nm.shape
        chex.assert_tree_all_finite(spec)

        jit_fn = jax.jit(adaf_disk_spectrum)
        spec_jit = jit_fn(wave_nm, delta=delta)
        chex.assert_trees_all_close(spec, spec_jit, rtol=1e-6)


class TestDeltaParameterEffect:
    """Test that delta parameter modulates disc slope as expected."""

    @pytest.mark.bounds
    def test_delta_modulates_skirtor_slope(self, wave_grid):
        """Positive delta shallows the SKIRTOR mid-IR slope (Stalevski+2012).

        The SKIRTOR disc mid-IR slope is alpha = -1.5 + delta. A larger delta
        makes the (negative) slope shallower, so more flux is redistributed to
        the mid-IR (the unit-area-normalized spectrum rises there).
        """
        from tengri.components.agn.disc_cigale import skirtor_disk_spectrum

        wave_nm = wave_grid / 10.0

        # Test range
        spec_delta_neg = skirtor_disk_spectrum(wave_nm, delta=-0.5)
        spec_delta_zero = skirtor_disk_spectrum(wave_nm, delta=0.0)
        spec_delta_pos = skirtor_disk_spectrum(wave_nm, delta=0.5)

        # All should be finite and positive
        chex.assert_tree_all_finite(spec_delta_neg)
        chex.assert_tree_all_finite(spec_delta_zero)
        chex.assert_tree_all_finite(spec_delta_pos)
        assert jnp.all(spec_delta_neg > 0)
        assert jnp.all(spec_delta_zero > 0)
        assert jnp.all(spec_delta_pos > 0)

        # Mid-IR region: 1000-5000 nm. Higher delta → shallower slope → more flux.
        mid_ir_slice = (wave_nm > 1000) & (wave_nm < 5000)
        mid_ir_flux_neg = jnp.mean(spec_delta_neg[mid_ir_slice])
        mid_ir_flux_zero = jnp.mean(spec_delta_zero[mid_ir_slice])
        mid_ir_flux_pos = jnp.mean(spec_delta_pos[mid_ir_slice])

        # Monotonic in delta: delta_pos > delta_zero > delta_neg.
        assert mid_ir_flux_pos > mid_ir_flux_zero, (
            f"delta=0.5 should have more mid-IR flux than delta=0: "
            f"{mid_ir_flux_pos} vs {mid_ir_flux_zero}"
        )
        assert mid_ir_flux_zero > mid_ir_flux_neg, (
            f"delta=0 should have more mid-IR flux than delta=-0.5: "
            f"{mid_ir_flux_zero} vs {mid_ir_flux_neg}"
        )

    @pytest.mark.bounds
    def test_delta_modulates_schartmann_slope(self, wave_grid):
        """Positive delta should steepen Schartmann mid-IR slope.

        Schartmann 2005 also uses α_mid = -1.5 + delta.
        """
        from tengri.components.agn.disc_cigale import schartmann2005_disk_spectrum

        wave_nm = wave_grid / 10.0

        spec_delta_zero = schartmann2005_disk_spectrum(wave_nm, delta=0.0)
        spec_delta_pos = schartmann2005_disk_spectrum(wave_nm, delta=0.3)

        mid_ir_slice = (wave_nm > 1000) & (wave_nm < 5000)
        mid_ir_flux_zero = jnp.mean(spec_delta_zero[mid_ir_slice])
        mid_ir_flux_pos = jnp.mean(spec_delta_pos[mid_ir_slice])

        assert mid_ir_flux_pos > mid_ir_flux_zero, (
            f"Positive delta should raise mid-IR flux (shallower slope): "
            f"{mid_ir_flux_pos} (delta=0.3) vs {mid_ir_flux_zero} (delta=0)"
        )

    @pytest.mark.bounds
    def test_delta_bounds_adaf_spectrum(self, wave_grid):
        """ADAF delta is the ADAF->thin-disc blend weight over [0, 1].

        delta=0 is pure ADAF, delta=1 is the thin disc, and intermediate
        values are genuine blends. Out-of-range delta stays finite (the blend
        weight is clipped to [0, 1]).
        """
        from tengri.components.agn.disc_cigale import adaf_disk_spectrum

        wave_nm = wave_grid / 10.0

        spec_adaf = adaf_disk_spectrum(wave_nm, delta=0.0)
        spec_mid = adaf_disk_spectrum(wave_nm, delta=0.5)
        spec_disc = adaf_disk_spectrum(wave_nm, delta=1.0)

        for s in (spec_adaf, spec_mid, spec_disc):
            chex.assert_tree_all_finite(s)
            assert jnp.all(s >= 0)

        # delta blends ADAF -> disc, so the endpoints differ and the midpoint
        # is a genuine mixture (not equal to either endpoint).
        assert not jnp.allclose(spec_adaf, spec_disc)
        assert not jnp.allclose(spec_mid, spec_adaf)

        # Out-of-range delta must remain finite (blend weight clipped to [0, 1]).
        chex.assert_tree_all_finite(adaf_disk_spectrum(wave_nm, delta=-0.5))
        chex.assert_tree_all_finite(adaf_disk_spectrum(wave_nm, delta=1.5))


class TestSKIRTORTorusParameterDiscovery:
    """Test that SKIRTORTorus exposes the delta disc-shape parameter."""

    def test_delta_parameter_in_registry(self):
        """agn_delta should be auto-discovered from SKIRTORTorus class."""
        from tengri.components.agn.skirtor_model import SKIRTORTorus

        component = SKIRTORTorus()
        # Check that delta is a declared distribution
        assert hasattr(component, "delta"), "delta attribute not found"
        assert component.delta is not None, "delta is None"

    def test_disk_type_in_config(self):
        """disk_type should be configurable in SKIRTORTorusConfig."""
        from tengri.components.agn.skirtor_model import SKIRTORTorus, SKIRTORTorusConfig

        # Default is 0: the SKIRTOR-intrinsic disc, which matches tengri's
        # tabulated disc bit-for-bit (CIGALE's module default is 1; tengri
        # documents disk_type=1 to recover Schartmann).
        config_default = SKIRTORTorusConfig()
        assert config_default.disk_type == 0, (
            f"Expected disk_type=0 (SKIRTOR-intrinsic), got {config_default.disk_type}"
        )

        # Should be configurable
        config_0 = SKIRTORTorusConfig(disk_type=0)
        assert config_0.disk_type == 0

        config_2 = SKIRTORTorusConfig(disk_type=2)
        assert config_2.disk_type == 2

        # Should be usable in component
        component_0 = SKIRTORTorus(config=config_0)
        assert component_0.config.disk_type == 0


class TestDiscTypeDispatch:
    """Test that disc_type parameter is static (non-traced) and JIT-safe."""

    def test_disc_type_dispatch_static(self):
        """disk_type should be usable in control flow without tracing issues.

        This tests that the component can be built with different disk_type
        values and that they don't create tracing problems.
        """
        from tengri.components.agn.skirtor_model import SKIRTORTorus, SKIRTORTorusConfig

        # Build components with different disk_type values
        for disk_type in [0, 1, 2]:
            config = SKIRTORTorusConfig(disk_type=disk_type)
            component = SKIRTORTorus(config=config)
            assert component.config.disk_type == disk_type

            # The component should be JIT-compatible in theory
            # (actual predict() needs templates, but the config is static)


class TestDiscReshapeAffectsPredict:
    """Predict-level guard: disk_type / delta MUST change the SED.

    Catches the regression where the parameters were declared but never wired
    into ``predict`` (a no-op). Requires the SKIRTOR template grid; skips if
    absent.
    """

    @pytest.fixture(scope="class")
    def ready_components(self):
        from tengri.components.agn.skirtor import _find_skirtor_grid
        from tengri.components.agn.skirtor_model import (
            SKIRTORTorus,
            SKIRTORTorusConfig,
        )

        grid_path = _find_skirtor_grid()
        if not grid_path:
            pytest.skip("SKIRTOR template grid not available")
        wave = jnp.geomspace(1e3, 1e7, 400)

        def make(disk_type):
            comp = SKIRTORTorus(
                config=SKIRTORTorusConfig(grid_path=grid_path, disk_type=disk_type)
            )
            # The forward pipeline populates ``data`` during a mutable build
            # phase; the component is frozen here, so set it directly (test-only).
            object.__setattr__(comp, "data", comp.load(wave))
            return comp

        return {"wave": wave, "make": make}

    def _params(self, delta):
        return dict(
            log_lbol=12.0,
            tau_skirtor=7.0,
            p_skirtor=1.0,
            q_skirtor=1.0,
            oa_skirtor=40.0,
            cos_inc=0.9,
            # The component's param dict is keyed by prefix-stripped attribute
            # names, so renaming agn_frac_agn -> agn_band_frac (#1296) renamed
            # this key too.
            band_frac=0.5,
            polar_ebv=0.0,
            polar_temperature=100.0,
            polar_beta=1.6,
            delta=delta,
        )

    @pytest.mark.contract
    def test_default_finite_and_positive(self, ready_components):
        wave = ready_components["wave"]
        comp = ready_components["make"](0)
        sed, _ = comp.predict(self._params(0.0), jnp.zeros_like(wave), wave)
        chex.assert_tree_all_finite(sed)
        assert float(jnp.sum(sed)) > 0.0

    @pytest.mark.contract
    def test_delta_changes_sed(self, ready_components):
        """A non-zero delta must visibly re-tilt the disc SED."""
        wave = ready_components["wave"]
        comp = ready_components["make"](0)
        base, _ = comp.predict(self._params(0.0), jnp.zeros_like(wave), wave)
        tilted, _ = comp.predict(self._params(0.8), jnp.zeros_like(wave), wave)
        rel = float(jnp.max(jnp.abs(tilted - base)) / (jnp.max(jnp.abs(base)) + 1e-30))
        assert rel > 1e-3, f"delta is a no-op (max frac change {rel:.2e})"

    @pytest.mark.contract
    def test_disk_type_changes_sed(self, ready_components):
        """Switching disk_type (0 SKIRTOR -> 1 Schartmann) must change the SED."""
        wave = ready_components["wave"]
        c0 = ready_components["make"](0)
        c1 = ready_components["make"](1)
        sed0, _ = c0.predict(self._params(0.0), jnp.zeros_like(wave), wave)
        sed1, _ = c1.predict(self._params(0.0), jnp.zeros_like(wave), wave)
        rel = float(jnp.max(jnp.abs(sed1 - sed0)) / (jnp.max(jnp.abs(sed0)) + 1e-30))
        assert rel > 1e-3, f"disk_type is a no-op (max frac change {rel:.2e})"

    @pytest.mark.gradient
    def test_delta_gradient_finite(self, ready_components):
        wave = ready_components["wave"]
        comp = ready_components["make"](0)

        def scalar(delta):
            sed, _ = comp.predict(self._params(delta), jnp.zeros_like(wave), wave)
            return jnp.sum(sed)

        g = assert_grad_matches_fd(scalar, 0.2)
        assert jnp.isfinite(g)
