"""Tests for simple_agn: power-law disc + single-temperature torus."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.bounds


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


@pytest.fixture()
def wavelength():
    """Broad wavelength grid from radio (1 cm) to hard X-ray (1 A)."""
    return jnp.logspace(0, 8, 500)  # 1 A to 10^8 A (= 1 cm)


@pytest.fixture()
def optical_wavelength():
    """Optical/UV wavelength grid."""
    return jnp.logspace(2.5, 5.0, 200)  # 316 A to 100,000 A


class TestSimpleAgn:
    """Tests for simple_agn: power-law disc + single-temperature torus."""

    def test_finite_nonneg(self, wavelength):
        """simple_agn produces finite, non-negative SED."""
        from tengri.components.agn.unified import simple_agn

        l_nu = simple_agn(wavelength, agn_log_lbol=44.0)
        chex.assert_tree_all_finite(l_nu)
        assert jnp.all(l_nu >= 0.0)
        chex.assert_equal_shape([l_nu, wavelength])

    def test_registered_as_simple(self):
        """'simple' appears in the AGN_MODELS registry."""
        from tengri.components.agn.unified import AGN_MODELS

        assert "simple" in AGN_MODELS

    def test_agn_frac_scales_linearly(self, wavelength):
        """agn_frac multiplies the whole SED linearly."""
        from tengri.components.agn.unified import simple_agn

        l1 = simple_agn(wavelength, agn_log_lbol=44.0, agn_frac=0.1)
        l2 = simple_agn(wavelength, agn_log_lbol=44.0, agn_frac=0.2)
        # Exclude wavelengths where both SEDs fall below float64 underflow (~1e-300);
        # X-ray regime can reach 1e-40 making the ratio numerically indeterminate.
        significant = l1 > 1e-50
        assert jnp.any(significant), "No significant SED values found"
        ratio = l2[significant] / l1[significant]
        assert jnp.allclose(ratio, 2.0, rtol=0.01)

    def test_torus_frac_nonzero_adds_ir(self, wavelength):
        """Non-zero agn_torus_frac adds torus IR flux near the BB peak.

        At fixed total luminosity, a torus fraction of 0.5 routes half the
        power into a 1000 K BB (peak ~29,000 Å).  The IR sum near that peak
        must exceed the disc-only (torus_frac=0) case where the powerlaw disc
        carries all the power but is fainter at 1000 K BB wavelengths.
        """
        from tengri.components.agn.disc import powerlaw_disc
        from tengri.components.agn.unified import simple_agn

        # Near the 1000 K BB peak (Wien: λ_peak = 2.898e7/1000 Å ≈ 29,000 Å)
        nir_mask = (wavelength > 2e4) & (wavelength < 5e4)
        # disc only at half power
        l_disc_half = powerlaw_disc(wavelength, agn_log_lbol=44.0, agn_frac=0.5)
        # disc (half power) + torus (half power, 1000 K BB peak in NIR)
        l_with_torus = simple_agn(
            wavelength,
            agn_log_lbol=44.0,
            agn_frac=1.0,
            agn_torus_frac=0.5,
            agn_T_torus=1000.0,
        )
        assert float(jnp.sum(l_with_torus[nir_mask])) > float(jnp.sum(l_disc_half[nir_mask]))

    def test_flatter_alpha_more_uv(self, wavelength):
        """Flatter disc slope (less-negative alpha) puts more power in UV.

        L_nu ∝ nu^alpha; flatter alpha means relatively more emission at
        high frequencies (UV) compared to a steep, red power law.
        """
        from tengri.components.agn.unified import simple_agn

        uv_mask = (wavelength > 1000.0) & (wavelength < 3000.0)
        # Use torus_frac=0 so we isolate the disc spectrum
        l_steep = simple_agn(
            wavelength, agn_log_lbol=44.0, agn_frac=1.0, agn_alpha=-2.0, agn_torus_frac=0.0
        )
        l_flat = simple_agn(
            wavelength, agn_log_lbol=44.0, agn_frac=1.0, agn_alpha=-0.5, agn_torus_frac=0.0
        )
        assert float(jnp.sum(l_flat[uv_mask])) > float(jnp.sum(l_steep[uv_mask]))

    def test_hotter_torus_peaks_at_shorter_wavelength(self, wavelength):
        """Hotter torus temperature shifts the IR peak to shorter wavelengths (Wien).

        Wien's law: λ_peak = b/T, so T_hot → shorter peak.
        """
        from tengri.components.agn.unified import simple_agn

        ir = wavelength[(wavelength > 5e3) & (wavelength < 2e7)]
        if ir.shape[0] < 10:
            pytest.skip("wavelength grid too sparse for IR peak test")

        l_cold = simple_agn(
            ir, agn_log_lbol=44.0, agn_frac=1.0, agn_torus_frac=1.0, agn_T_torus=400.0
        )
        l_hot = simple_agn(
            ir, agn_log_lbol=44.0, agn_frac=1.0, agn_torus_frac=1.0, agn_T_torus=2000.0
        )

        peak_cold = float(ir[jnp.argmax(l_cold)])
        peak_hot = float(ir[jnp.argmax(l_hot)])
        assert peak_hot < peak_cold, (
            f"Expected hotter torus peak at shorter λ; got peak_hot={peak_hot:.1f} Å, "
            f"peak_cold={peak_cold:.1f} Å"
        )

    def test_jit_compatible(self, wavelength):
        """simple_agn is JIT-compilable."""
        from tengri.components.agn.unified import simple_agn

        @jax.jit
        def _run(wave):
            return simple_agn(wave, agn_log_lbol=44.0)

        chex.assert_tree_all_finite(_run(wavelength))

    def test_gradient_wrt_lbol(self, optical_wavelength):
        """FD check: ∂(∑SED)/∂agn_log_lbol for simple_agn."""
        from tengri.components.agn.unified import simple_agn

        def loss_fn(lbol):
            return jnp.sum(simple_agn(optical_wavelength, agn_log_lbol=lbol))

        g = float(jax.grad(loss_fn)(44.0))
        np.testing.assert_allclose(
            g,
            fd_grad(loss_fn, 44.0, eps=0.01),
            rtol=1e-3,
            err_msg="simple_agn: FD check ∂/∂agn_log_lbol",
        )
