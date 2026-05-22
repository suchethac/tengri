"""Tests for standard_agn: multi-color disc + two-temperature torus."""

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


class TestStandardAgn:
    """Tests for standard_agn: multi-color disc + two-temperature torus."""

    def test_finite_nonneg(self, wavelength):
        """standard_agn produces finite, non-negative SED."""
        from tengri.components.agn.unified import standard_agn

        l_nu = standard_agn(wavelength, agn_log_lbol=44.0)
        chex.assert_tree_all_finite(l_nu)
        assert jnp.all(l_nu >= 0.0)
        chex.assert_equal_shape([l_nu, wavelength])

    def test_registered_as_standard(self):
        """'standard' appears in the AGN_MODELS registry."""
        from tengri.components.agn.unified import AGN_MODELS

        assert "standard" in AGN_MODELS

    def test_agn_frac_scales_linearly(self, wavelength):
        """agn_frac multiplies the whole SED linearly."""
        from tengri.components.agn.unified import standard_agn

        l1 = standard_agn(wavelength, agn_log_lbol=44.0, agn_frac=0.1)
        l2 = standard_agn(wavelength, agn_log_lbol=44.0, agn_frac=0.2)
        # Exclude wavelengths where both SEDs fall below float64 underflow (~1e-300);
        # X-ray regime can reach 1e-40 making the ratio numerically indeterminate.
        significant = l1 > 1e-50
        assert jnp.any(significant), "No significant SED values found"
        ratio = l2[significant] / l1[significant]
        assert jnp.allclose(ratio, 2.0, rtol=0.01)

    def test_higher_ledd_bluer_disc(self, optical_wavelength):
        """Higher Eddington ratio gives a hotter disc → more far-UV emission.

        T_in ∝ mdot^{1/4} at fixed M_BH (mdot ∝ L_Edd * ratio ∝ M_BH * l_edd_ratio).
        At fixed L_bol both discs peak in the EUV (<100 Å), but the hotter disc
        extends further into the far-UV.  Below ~500 Å the high-Eddington SED
        exceeds the low-Eddington one; integrating a broader window mixes in
        the optical bump where the cooler disc wins.
        """
        from tengri.components.agn.unified import standard_agn

        # Far-UV (<500 Å): hotter disc emits more (further from the Wien cutoff)
        far_uv_mask = (optical_wavelength > 300.0) & (optical_wavelength < 500.0)
        # Isolate disc by setting torus_frac=0; vary only Eddington ratio
        l_low = standard_agn(
            optical_wavelength,
            agn_log_lbol=44.0,
            agn_frac=1.0,
            agn_log_ledd=-2.0,
            agn_torus_frac=0.0,
        )
        l_high = standard_agn(
            optical_wavelength,
            agn_log_lbol=44.0,
            agn_frac=1.0,
            agn_log_ledd=-0.5,
            agn_torus_frac=0.0,
        )
        assert jnp.any(far_uv_mask), "No wavelengths in far-UV window"
        # Higher Eddington ratio → hotter disc → more flux at far-UV wavelengths
        assert float(jnp.sum(l_high[far_uv_mask])) > float(jnp.sum(l_low[far_uv_mask]))

    def test_two_temperature_torus_has_near_ir_excess(self, wavelength):
        """Two-temperature torus produces near-IR emission from the hot component.

        With a hot (1200 K) and warm (300 K) component, there is more near-IR
        (1–5 μm) emission than a cool single-temperature torus would produce.
        """
        from tengri.components.agn.unified import standard_agn

        # Near-IR: 1–5 μm = 10,000–50,000 Å
        nir_mask = (wavelength > 1e4) & (wavelength < 5e4)
        l_standard = standard_agn(
            wavelength,
            agn_log_lbol=44.0,
            agn_frac=1.0,
            agn_T_hot=1200.0,
            agn_T_warm=300.0,
            agn_frac_hot=0.3,
            agn_torus_frac=1.0,
        )
        # Set hot component to near-zero to simulate a cooler single-temperature torus
        l_cold_only = standard_agn(
            wavelength,
            agn_log_lbol=44.0,
            agn_frac=1.0,
            agn_T_hot=350.0,
            agn_T_warm=300.0,
            agn_frac_hot=0.0,
            agn_torus_frac=1.0,
        )
        assert float(jnp.sum(l_standard[nir_mask])) > float(jnp.sum(l_cold_only[nir_mask]))

    def test_jit_compatible(self, wavelength):
        """standard_agn is JIT-compilable."""
        from tengri.components.agn.unified import standard_agn

        @jax.jit
        def _run(wave):
            return standard_agn(wave, agn_log_lbol=44.0)

        chex.assert_tree_all_finite(_run(wavelength))

    def test_gradient_wrt_lbol(self, optical_wavelength):
        """FD check: ∂(∑SED)/∂agn_log_lbol for standard_agn."""
        from tengri.components.agn.unified import standard_agn

        def loss_fn(lbol):
            return jnp.sum(standard_agn(optical_wavelength, agn_log_lbol=lbol))

        g = float(jax.grad(loss_fn)(44.0))
        np.testing.assert_allclose(
            g,
            fd_grad(loss_fn, 44.0, eps=0.01),
            rtol=1e-3,
            err_msg="standard_agn: FD check ∂/∂agn_log_lbol",
        )

    def test_gradient_wrt_torus_frac(self, optical_wavelength):
        """FD check: ∂(∑SED)/∂agn_torus_frac for standard_agn."""
        from tengri.components.agn.unified import standard_agn

        def loss_fn(frac):
            return jnp.sum(
                standard_agn(optical_wavelength, agn_log_lbol=44.0, agn_torus_frac=frac)
            )

        g = float(jax.grad(loss_fn)(0.5))
        np.testing.assert_allclose(
            g,
            fd_grad(loss_fn, 0.5, eps=0.01),
            rtol=1e-3,
            err_msg="standard_agn: FD check ∂/∂agn_torus_frac",
        )
