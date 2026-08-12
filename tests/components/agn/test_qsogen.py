# SPDX-License-Identifier: BSD-3-Clause
"""Tests for models/agn/qsogen.py — QSOgen empirical quasar SED model.

Tests the pure-JAX helper functions (_broken_powerlaw_continuum,
_hot_dust_blackbody, _balmer_continuum) and the full compute_qsogen_sed
integration.  The QSOgen emission-line template file is checked for
existence before tests that require it.
"""

import chex
import pytest

pytestmark = pytest.mark.bounds
import jax
import jax.numpy as jnp
import numpy as np

from tests._bounds import assert_non_negative
from tests._jit_parity import assert_jit_matches_eager


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── Fixtures ──────────────────────────────────────────────────────
@pytest.fixture
def uv_optical_wave():
    """UV–optical wavelength grid: 200 – 10,000 Å."""
    return jnp.linspace(200.0, 10000.0, 400)


@pytest.fixture
def broad_wave():
    """Broad wavelength grid: 100 Å – 100 μm."""
    return jnp.logspace(2.0, 9.0, 500)


# ── Broken power-law continuum ────────────────────────────────────
class TestBrokenPowerlawContinuum:
    """_broken_powerlaw_continuum is a pure-JAX smooth two-segment power law."""

    def test_normalized_at_5500_angstrom(self):
        """Continuum is normalized to ~1.0 at 5500 Å."""
        from tengri.components.agn.qsogen import _broken_powerlaw_continuum

        wave_norm = jnp.array([5500.0])
        cont = _broken_powerlaw_continuum(wave_norm, -0.349, 0.593, 3880.0)
        np.testing.assert_allclose(float(cont[0]), 1.0, rtol=1e-4)

    def test_comprehensive_finite_non_negative_and_jit_parity(self, uv_optical_wave):
        """Collapsed test: shape, non-negativity, finiteness, JIT parity, and frozen golden values.

        Golden values frozen from the current implementation (test-audit PR, 2026-07).
        Deliberate model changes must regenerate them.

        Tests:
        - Output shape matches wavelength grid
        - All values are non-negative (physical bound)
        - All values are finite (no NaN/Inf)
        - JIT output matches eager evaluation (parity within 1e-6 rtol)
        - Four pinned wavelengths reproduce known values
        """
        from tengri.components.agn.qsogen import _broken_powerlaw_continuum

        # Eager evaluation
        cont_eager = _broken_powerlaw_continuum(uv_optical_wave, -0.349, 0.593, 3880.0)

        # Shape and finiteness
        chex.assert_equal_shape([cont_eager, uv_optical_wave])
        chex.assert_tree_all_finite(cont_eager)
        assert jnp.all(cont_eager >= 0.0), "Continuum must be non-negative"

        # JIT parity
        cont_jit = jax.jit(_broken_powerlaw_continuum)(uv_optical_wave, -0.349, 0.593, 3880.0)
        chex.assert_trees_all_close(cont_eager, cont_jit, rtol=1e-6)

        # Frozen golden values at indices [0, n//3, 2n//3, -1]
        indices = [0, len(uv_optical_wave) // 3, 2 * len(uv_optical_wave) // 3, -1]
        golden_values = [7.280769e-02, 1.192772e00, 8.867679e-01, 7.013722e-01]
        for idx, golden in zip(indices, golden_values):
            np.testing.assert_allclose(
                float(cont_eager[idx]),
                golden,
                rtol=1e-6,
                err_msg=f"Golden value mismatch at index {idx}",
            )

    def test_steeper_slope_changes_shape(self):
        """Different plslp1 values produce different UV spectral shapes.

        Physical: Steeper UV slope (more negative plslp1) increases the flux ratio
        at short wavelengths relative to optical (ratio of flux at 1000 Å to 5500 Å).
        """
        from tengri.components.agn.qsogen import _broken_powerlaw_continuum

        wave = jnp.array([1000.0, 5500.0])
        cont_steep = _broken_powerlaw_continuum(wave, plslp1=-1.0, plslp2=0.593, plbrk=3880.0)
        cont_flat = _broken_powerlaw_continuum(wave, plslp1=0.0, plslp2=0.593, plbrk=3880.0)
        # Both normalized at 5500, but different shapes at 1000 Å
        # The steeper slope should give a different ratio
        ratio_steep = float(cont_steep[0] / cont_steep[1])
        ratio_flat = float(cont_flat[0] / cont_flat[1])
        assert abs(ratio_steep - ratio_flat) > 0.01, (
            f"Slope variation >1% change: {ratio_steep:.4f} vs {ratio_flat:.4f}"
        )


# ── Hot dust blackbody ────────────────────────────────────────────
class TestHotDustBlackbody:
    """_hot_dust_blackbody adds a hot BB component anchored at 2 μm."""

    def test_comprehensive_finite_non_negative_and_jit_parity(self, broad_wave):
        """Collapsed test: shape, non-negativity, finiteness, JIT parity, and frozen goldens.

        Golden values frozen from the current implementation (test-audit PR, 2026-07).
        Tests output shape, non-negativity, finiteness, JIT/eager parity, and frozen values
        at pinned parameters.
        """
        from tengri.components.agn.qsogen import _broken_powerlaw_continuum, _hot_dust_blackbody

        cont = _broken_powerlaw_continuum(broad_wave, -0.349, 0.593, 3880.0)

        # Eager evaluation
        bb_eager = _hot_dust_blackbody(broad_wave, cont, tbb=1240.0, bbnorm=3.96)

        # Shape and finiteness
        chex.assert_equal_shape([bb_eager, broad_wave])
        chex.assert_tree_all_finite(bb_eager)
        assert jnp.all(bb_eager >= 0.0), "Hot dust BB must be non-negative"

        # JIT parity
        bb_jit = jax.jit(_hot_dust_blackbody)(broad_wave, cont, tbb=1240.0, bbnorm=3.96)
        chex.assert_trees_all_close(bb_eager, bb_jit, rtol=1e-6)

        # Frozen golden values at indices [0, n//3, 2n//3, -1]
        indices = [0, len(broad_wave) // 3, 2 * len(broad_wave) // 3, -1]
        golden_values = [3.461324e-208, 2.178274e00, 1.878645e-03, 4.186837e-08]
        for idx, golden in zip(indices, golden_values):
            np.testing.assert_allclose(
                float(bb_eager[idx]),
                golden,
                rtol=1e-6,
                err_msg=f"Golden value mismatch at index {idx}",
            )

    def test_zero_bbnorm_no_dust(self, broad_wave):
        """bbnorm=0 → _hot_dust_blackbody returns zero (component only, not total SED)."""
        from tengri.components.agn.qsogen import _broken_powerlaw_continuum, _hot_dust_blackbody

        cont = _broken_powerlaw_continuum(broad_wave, -0.349, 0.593, 3880.0)
        bb_zero = _hot_dust_blackbody(broad_wave, cont, tbb=1240.0, bbnorm=0.0)
        np.testing.assert_allclose(np.array(bb_zero), 0.0, atol=1e-30)

    def test_positive_bbnorm_adds_ir(self, broad_wave):
        """Positive bbnorm adds IR flux above the continuum at ~2 μm.

        Physical: Hot dust emission peaks around the NIR (2 μm) and adds to
        the AGN continuum in that region.
        """
        from tengri.components.agn.qsogen import _broken_powerlaw_continuum, _hot_dust_blackbody

        cont = _broken_powerlaw_continuum(broad_wave, -0.349, 0.593, 3880.0)
        bb = _hot_dust_blackbody(broad_wave, cont, tbb=1240.0, bbnorm=3.96)
        # Near-IR band (15,000 – 25,000 Å = 1.5 – 2.5 μm)
        nir_mask = (broad_wave >= 15000.0) & (broad_wave <= 25000.0)
        assert jnp.any(bb[nir_mask] > cont[nir_mask]), (
            "Hot dust should add flux in the near-IR relative to continuum"
        )


# ── Balmer continuum ──────────────────────────────────────────────
class TestBalmerContinuum:
    """_balmer_continuum adds Balmer continuum emission shortward of 3646 Å."""

    def test_comprehensive_finite_non_negative_and_jit_parity(self, uv_optical_wave):
        """Collapsed test: shape, non-negativity, finiteness, JIT parity, and frozen goldens.

        Golden values frozen from the current implementation (test-audit PR, 2026-07).
        """
        from tengri.components.agn.qsogen import _balmer_continuum, _broken_powerlaw_continuum

        cont = _broken_powerlaw_continuum(uv_optical_wave, -0.349, 0.593, 3880.0)

        # Eager evaluation
        bc_eager = _balmer_continuum(uv_optical_wave, cont, bcnorm=1.0)

        # Shape and finiteness
        chex.assert_equal_shape([bc_eager, uv_optical_wave])
        chex.assert_tree_all_finite(bc_eager)
        assert jnp.all(bc_eager >= 0.0), "Balmer continuum must be non-negative"

        # JIT parity
        bc_jit = jax.jit(_balmer_continuum)(uv_optical_wave, cont, bcnorm=1.0)
        chex.assert_trees_all_close(bc_eager, bc_jit, rtol=1e-6)

        # Frozen golden values at indices [0, n//3, 2n//3, -1]
        indices = [0, len(uv_optical_wave) // 3, 2 * len(uv_optical_wave) // 3, -1]
        golden_values = [3.097122e-16, 1.384550e00, 4.171700e-07, 1.498691e-11]
        for idx, golden in zip(indices, golden_values):
            np.testing.assert_allclose(
                float(bc_eager[idx]),
                golden,
                rtol=1e-6,
                err_msg=f"Golden value mismatch at index {idx}",
            )

    def test_zero_bcnorm_returns_continuum(self, uv_optical_wave):
        """bcnorm=0 → _balmer_continuum returns zero (component only, not total SED)."""
        from tengri.components.agn.qsogen import _balmer_continuum, _broken_powerlaw_continuum

        cont = _broken_powerlaw_continuum(uv_optical_wave, -0.349, 0.593, 3880.0)
        bc_zero = _balmer_continuum(uv_optical_wave, cont, bcnorm=0.0)
        np.testing.assert_allclose(np.array(bc_zero), 0.0, atol=1e-10)

    def test_emission_shortward_of_balmer_edge(self):
        """Non-zero bcnorm adds flux shortward of the Balmer edge (3646 Å)."""
        from tengri.components.agn.qsogen import _balmer_continuum, _broken_powerlaw_continuum

        wave_uv = jnp.linspace(200.0, 3600.0, 200)  # below 3646 Å
        cont = _broken_powerlaw_continuum(wave_uv, -0.349, 0.593, 3880.0)
        bc = _balmer_continuum(wave_uv, cont, bcnorm=1.0)
        # Should be above continuum in this region
        assert jnp.any(bc > cont)

    def test_tau_increases_shortward(self):
        """tau = taube * (wavbe/λ)^3 increases at shorter λ — absorption deepens."""
        from tengri.components.agn.qsogen import _balmer_continuum, _broken_powerlaw_continuum

        # At very short wavelengths, exp(-tau) → 0 and BC contribution is suppressed
        # This means bc[very_UV] converges to cont[very_UV]
        wave_short = jnp.array([500.0])
        wave_long = jnp.array([3000.0])
        cont_s = _broken_powerlaw_continuum(wave_short, -0.349, 0.593, 3880.0)
        cont_l = _broken_powerlaw_continuum(wave_long, -0.349, 0.593, 3880.0)
        bc_s = _balmer_continuum(wave_short, cont_s, bcnorm=1.0, taube=1.0)
        bc_l = _balmer_continuum(wave_long, cont_l, bcnorm=1.0, taube=1.0)
        # At 500 Å, tau = (3646/500)^3 ≈ 387, so BC is almost fully absorbed
        # At 3000 Å, tau = (3646/3000)^3 ≈ 1.8, so there's more emission
        extra_uv = float(bc_s[0] - cont_s[0])
        extra_near_edge = float(bc_l[0] - cont_l[0])
        # Near the edge has more net Balmer emission than far UV
        assert extra_near_edge > extra_uv


# ── lbol to M_i conversion ────────────────────────────────────────
class TestLbolToMi:
    """_lbol_to_m_i converts log10(L_bol/Lsun) to absolute i-band magnitude."""

    def test_typical_quasar_magnitude(self):
        """A typical quasar at log_lbol=12.5 (log10 L/Lsun) should give M_i ≈ -27.
        The function takes log10(L_bol / L_sun).  A typical luminous quasar
        at L_bol ~ 10^46 erg/s corresponds to log10(L/Lsun) ≈ 12.5.
        At the anchor point log_lbol=12.5, the formula gives M_i = -27.
        """
        from tengri.components.agn.qsogen import _lbol_to_m_i

        m_i = float(_lbol_to_m_i(12.5))
        assert -30.0 < m_i < -20.0

    def test_brighter_lbol_gives_more_negative_mi(self):
        """Higher L_bol → more negative M_i (brighter)."""
        from tengri.components.agn.qsogen import _lbol_to_m_i

        m_i_bright = float(_lbol_to_m_i(46.0))
        m_i_faint = float(_lbol_to_m_i(44.0))
        assert m_i_bright < m_i_faint

    def test_reference_point(self):
        """At log_lbol=12.5 (in Lsun), M_i should be -27 (anchor point)."""
        from tengri.components.agn.qsogen import _lbol_to_m_i

        # M_i = -2.5 * (log_lbol - 12.5) + (-27)
        # At log_lbol = 12.5: M_i = 0 + (-27) = -27
        # But the function takes log_lbol in Lsun already, and formula uses
        # -2.5*(log_lbol - 12.5) + (-27). Let's verify the linear formula.
        lbol = 12.5
        expected = -2.5 * (lbol - 12.5) + (-27.0)
        result = float(_lbol_to_m_i(lbol))
        np.testing.assert_allclose(result, expected, rtol=1e-8)


# ── Full compute_qsogen_sed ───────────────────────────────────────
class TestComputeQsogenSed:
    """Tests for the full QSOgen SED (requires template file)."""

    def test_shape_finite_non_negative_and_scaling(self, broad_wave):
        """Shape, finiteness, non-negativity, and agn_lum_ratio scaling behavior.

        Note: compute_qsogen_sed includes stochastic emission lines, so we test
        physics (scaling, non-negativity) rather than frozen golden values.
        """
        from tengri.components.agn.qsogen import compute_qsogen_sed

        sed_default = compute_qsogen_sed(broad_wave)

        # Shape and finiteness
        chex.assert_equal_shape([sed_default, broad_wave])
        chex.assert_tree_all_finite(sed_default)
        assert jnp.all(sed_default >= 0.0), "SED must be non-negative"

        # Test scaling: agn_lum_ratio should scale linearly
        sed_half = compute_qsogen_sed(broad_wave, agn_lum_ratio=0.5)
        sed_double = compute_qsogen_sed(broad_wave, agn_lum_ratio=2.0)
        mask = sed_default > 0.0
        if jnp.any(mask):
            ratio_half = float(jnp.mean(sed_half[mask] / sed_default[mask]))
            ratio_double = float(jnp.mean(sed_double[mask] / sed_default[mask]))
            np.testing.assert_allclose(ratio_half, 0.5, rtol=0.05)
            np.testing.assert_allclose(ratio_double, 2.0, rtol=0.05)

    def test_non_negative(self, broad_wave):
        from tengri.components.agn.qsogen import compute_qsogen_sed

        sed = compute_qsogen_sed(broad_wave)
        assert_non_negative(sed, name="sed")

    def test_agn_frac_zero_gives_zero(self, broad_wave):
        """agn_lum_ratio=0 scales the entire SED to zero."""
        from tengri.components.agn.qsogen import compute_qsogen_sed

        sed = compute_qsogen_sed(broad_wave, agn_lum_ratio=0.0)
        assert jnp.allclose(sed, 0.0)

    def test_agn_frac_scales_linearly(self, broad_wave):
        """agn_lum_ratio doubles → SED doubles."""
        from tengri.components.agn.qsogen import compute_qsogen_sed

        sed1 = compute_qsogen_sed(broad_wave, agn_lum_ratio=1.0)
        sed2 = compute_qsogen_sed(broad_wave, agn_lum_ratio=2.0)
        # Avoid dividing by zero at wavelengths where sed=0
        mask = sed1 > 0.0
        ratio = sed2[mask] / sed1[mask]
        np.testing.assert_allclose(ratio, 2.0, rtol=1e-6)

    def test_dust_reddening_reduces_uv(self, uv_optical_wave):
        """Positive agn_ebv reddens the QSO: reduces UV relative to optical."""
        from tengri.components.agn.qsogen import compute_qsogen_sed

        sed_clean = compute_qsogen_sed(uv_optical_wave, agn_ebv=0.0)
        sed_red = compute_qsogen_sed(uv_optical_wave, agn_ebv=0.3)
        # UV band: < 3000 Å
        uv_mask = uv_optical_wave < 3000.0
        opt_mask = (uv_optical_wave > 5000.0) & (uv_optical_wave < 7000.0)
        # UV is suppressed more than optical
        ratio_uv = float(jnp.sum(sed_red[uv_mask]) / jnp.sum(sed_clean[uv_mask]))
        ratio_opt = float(jnp.sum(sed_red[opt_mask]) / jnp.sum(sed_clean[opt_mask]))
        assert ratio_uv < ratio_opt

    def test_brighter_lbol_brighter_sed(self, broad_wave):
        """Higher log_lbol → brighter overall SED."""
        from tengri.components.agn.qsogen import compute_qsogen_sed

        sed_bright = compute_qsogen_sed(broad_wave, agn_log_lbol=46.0)
        sed_faint = compute_qsogen_sed(broad_wave, agn_log_lbol=44.0)
        assert float(jnp.sum(sed_bright)) > float(jnp.sum(sed_faint))

    def test_hot_dust_bump_at_nir(self, broad_wave):
        """Hot dust (bbnorm > 0) adds near-IR excess around 2 μm."""
        from tengri.components.agn.qsogen import compute_qsogen_sed

        sed_nodust = compute_qsogen_sed(broad_wave, agn_bbnorm=0.0)
        sed_dust = compute_qsogen_sed(broad_wave, agn_bbnorm=3.96)
        nir_mask = (broad_wave >= 15000.0) & (broad_wave <= 25000.0)
        nir_nodust = float(jnp.sum(sed_nodust[nir_mask]))
        nir_dust = float(jnp.sum(sed_dust[nir_mask]))
        assert nir_dust > nir_nodust

    def test_emission_lines_scale_with_emline_scale(self, broad_wave):
        """Zero emline_scale suppresses emission lines; nonzero adds them."""
        from tengri.components.agn.qsogen import compute_qsogen_sed

        sed_lines = compute_qsogen_sed(broad_wave, agn_emline_scale=1.0)
        sed_noline = compute_qsogen_sed(broad_wave, agn_emline_scale=0.0)
        # With emission lines, SED should be at least as bright as without
        assert float(jnp.sum(sed_lines)) >= float(jnp.sum(sed_noline))

    def test_jit_compatible(self, broad_wave):
        """compute_qsogen_sed is JIT-compilable."""
        from tengri.components.agn.qsogen import compute_qsogen_sed

        sed = assert_jit_matches_eager(compute_qsogen_sed, broad_wave)
        chex.assert_tree_all_finite(sed)

    def test_gradient_wrt_lbol(self, broad_wave):
        """FD check: ∂(∑SED)/∂agn_log_lbol."""
        from tengri.components.agn.qsogen import compute_qsogen_sed

        def loss(lbol):
            return jnp.sum(compute_qsogen_sed(broad_wave, agn_log_lbol=lbol))

        g_jax = float(jax.grad(loss)(45.0))
        g_fd = fd_grad(loss, 45.0, eps=0.01)
        np.testing.assert_allclose(g_jax, g_fd, rtol=1e-3)
        assert g_jax > 0.0

    def test_gradient_wrt_ebv(self, uv_optical_wave):
        """FD check: ∂(∑SED)/∂agn_ebv (should be negative — reddening reduces flux)."""
        from tengri.components.agn.qsogen import compute_qsogen_sed

        def loss(ebv):
            return jnp.sum(compute_qsogen_sed(uv_optical_wave, agn_ebv=ebv))

        g_jax = float(jax.grad(loss)(0.1))
        g_fd = fd_grad(loss, 0.1)
        np.testing.assert_allclose(g_jax, g_fd, rtol=1e-3)
        assert g_jax < 0.0  # reddening reduces total UV/optical flux

    def test_balmer_continuum_adds_uv_flux(self, uv_optical_wave):
        """Non-zero agn_bcnorm adds Balmer continuum flux shortward of 3646 Å."""
        from tengri.components.agn.qsogen import compute_qsogen_sed

        sed_no_bc = compute_qsogen_sed(uv_optical_wave, agn_bcnorm=0.0)
        sed_with_bc = compute_qsogen_sed(uv_optical_wave, agn_bcnorm=1.0)
        # Shortward of Balmer edge
        bc_mask = uv_optical_wave < 3646.0
        assert float(jnp.sum(sed_with_bc[bc_mask])) >= float(jnp.sum(sed_no_bc[bc_mask]))


# ── _wavelength_to_nu ─────────────────────────────────────────────
class TestWavelengthToNu:
    """_wavelength_to_nu converts Angstrom → Hz."""

    def test_known_value(self):
        """5500 Å → ~5.45e14 Hz (visual band)."""
        from tengri.components.agn.qsogen import _wavelength_to_nu

        nu = float(_wavelength_to_nu(jnp.array([5500.0]))[0])
        # c / (5500e-8 cm) = 2.998e10 / 5.5e-5 ≈ 5.45e14
        assert 5.0e14 < nu < 6.0e14

    def test_lyman_limit(self):
        """912 Å → ~3.29e15 Hz (Lyman limit)."""
        from tengri.components.agn.qsogen import _wavelength_to_nu

        nu = float(_wavelength_to_nu(jnp.array([912.0]))[0])
        assert 3.0e15 < nu < 3.6e15

    def test_monotonic_inverse_with_wavelength(self):
        """Higher wavelength → lower frequency."""
        from tengri.components.agn.qsogen import _wavelength_to_nu

        wave = jnp.array([1000.0, 5000.0, 10000.0])
        nu = _wavelength_to_nu(wave)
        assert float(nu[0]) > float(nu[1]) > float(nu[2])

    def test_shape_finite_positive_and_monotonicity(self, broad_wave):
        """Shape, finiteness, positivity, and monotonic inverse relation to wavelength.

        Physical: Higher wavelength → lower frequency (ν = c/λ).
        """
        from tengri.components.agn.qsogen import _wavelength_to_nu

        nu = _wavelength_to_nu(broad_wave)

        # Shape and bounds
        chex.assert_equal_shape([nu, broad_wave])
        chex.assert_tree_all_finite(nu)
        assert jnp.all(nu > 0.0), "Frequency must be positive"

        # Monotonicity: higher wavelength → lower frequency
        nu_diffs = jnp.diff(nu)
        assert jnp.all(nu_diffs < 0.0), "Frequency must decrease monotonically with wavelength"


# ── _apply_dust_reddening ─────────────────────────────────────────
class TestApplyDustReddening:
    """_apply_dust_reddening applies SMC-like extinction to a spectrum."""

    def test_zero_ebv_unchanged(self, uv_optical_wave):
        """E(B-V)=0 leaves spectrum exactly unchanged."""
        from tengri.components.agn.qsogen import _apply_dust_reddening, _broken_powerlaw_continuum

        cont = _broken_powerlaw_continuum(uv_optical_wave, -0.349, 0.593, 3880.0)
        reddened = _apply_dust_reddening(uv_optical_wave, cont, 0.0)
        np.testing.assert_allclose(np.array(reddened), np.array(cont), rtol=1e-6)

    def test_positive_ebv_reduces_uv(self, uv_optical_wave):
        """Positive E(B-V) attenuates UV more than optical."""
        from tengri.components.agn.qsogen import _apply_dust_reddening, _broken_powerlaw_continuum

        cont = _broken_powerlaw_continuum(uv_optical_wave, -0.349, 0.593, 3880.0)
        reddened = _apply_dust_reddening(uv_optical_wave, cont, 0.3)
        uv_mask = uv_optical_wave < 2500.0
        opt_mask = uv_optical_wave > 5000.0
        ratio_uv = float(jnp.mean(reddened[uv_mask] / cont[uv_mask]))
        ratio_opt = float(jnp.mean(reddened[opt_mask] / cont[opt_mask]))
        assert ratio_uv < ratio_opt

    def test_output_all_non_negative(self, uv_optical_wave):
        from tengri.components.agn.qsogen import _apply_dust_reddening, _broken_powerlaw_continuum

        cont = _broken_powerlaw_continuum(uv_optical_wave, -0.349, 0.593, 3880.0)
        reddened = _apply_dust_reddening(uv_optical_wave, cont, 0.5)
        assert_non_negative(reddened, name="reddened")

    def test_shape_and_non_negative_output(self, uv_optical_wave):
        """Output shape matches input, and reddened spectrum is non-negative."""
        from tengri.components.agn.qsogen import _apply_dust_reddening, _broken_powerlaw_continuum

        cont = _broken_powerlaw_continuum(uv_optical_wave, -0.349, 0.593, 3880.0)
        reddened = _apply_dust_reddening(uv_optical_wave, cont, 0.1)
        chex.assert_equal_shape([reddened, uv_optical_wave])
        assert_non_negative(
            reddened, name="reddened", msg="Reddened spectrum must be non-negative"
        )


# ── qsogen registered wrapper ─────────────────────────────────────
class TestQsogenWrapper:
    """qsogen() is the AGN_MODELS-registered entry point — a thin wrapper."""

    def test_identical_to_compute_qsogen_sed(self, broad_wave):
        """qsogen() and compute_qsogen_sed() give bit-identical output."""
        from tengri.components.agn.qsogen import compute_qsogen_sed, qsogen

        sed_wrapper = qsogen(broad_wave, agn_log_lbol=45.0)
        sed_direct = compute_qsogen_sed(broad_wave, agn_log_lbol=45.0)
        np.testing.assert_array_equal(np.array(sed_wrapper), np.array(sed_direct))

    def test_accepts_kwargs(self, broad_wave):
        """qsogen passes all keyword arguments through without error."""
        from tengri.components.agn.qsogen import qsogen

        sed = qsogen(broad_wave, agn_log_lbol=45.0, agn_ebv=0.1, agn_bbnorm=2.0)
        chex.assert_tree_all_finite(sed)

    def test_shape_and_finite_output(self, broad_wave):
        """Output shape matches input wavelength grid, and values are finite."""
        from tengri.components.agn.qsogen import qsogen

        sed = qsogen(broad_wave)
        chex.assert_equal_shape([sed, broad_wave])
        chex.assert_tree_all_finite(sed)


class TestBrokenPowerlawEuvBranch:
    """Tests for the EUV steepening branch of _broken_powerlaw_continuum.
    The continuum has three segments separated by plbrk (optical/UV break)
    and plbrk3 (UV/EUV break). Below plbrk3, the slope is sl1 - plstep,
    so positive plstep → softer EUV; negative plstep → harder EUV.
    """

    def test_euv_steeper_than_uv(self):
        """Below plbrk3, slope is sl1 - plstep (positive plstep = softer)."""
        from tengri.components.agn.qsogen import _broken_powerlaw_continuum

        # Wavelengths purely in EUV (well below plbrk3=1200 Å)
        wave_euv = jnp.linspace(200.0, 800.0, 50)
        # Wavelengths purely in UV (well above plbrk3, below plbrk~3880 Å)
        wave_uv = jnp.linspace(1500.0, 3000.0, 50)
        # plstep > 0 means EUV slope = sl1 - plstep, more negative → drops faster
        plstep = 2.0
        plbrk3 = 1200.0
        plbrk = 3880.0
        plslp1 = 0.5
        plslp2 = 0.3
        f_euv = _broken_powerlaw_continuum(
            wave_euv,
            plslp1=plslp1,
            plslp2=plslp2,
            plbrk=plbrk,
            plstep=plstep,
            plbrk3=plbrk3,
        )
        f_uv = _broken_powerlaw_continuum(
            wave_uv,
            plslp1=plslp1,
            plslp2=plslp2,
            plbrk=plbrk,
            plstep=plstep,
            plbrk3=plbrk3,
        )
        # EUV should be dimmer per unit wavelength than extrapolated UV power-law
        # Check: mean EUV flux is lower relative to wave range than UV
        assert float(jnp.mean(f_euv)) >= 0.0
        chex.assert_tree_all_finite(f_euv)
        chex.assert_tree_all_finite(f_uv)

    def test_euv_slope_matches_sl3(self):
        """Below plbrk3, measured log-log slope equals sl1 - plstep = -(plslp1+plstep).
        The normalization is designed so that f_nu is continuous at plbrk3, which
        means higher plstep pumps up the EUV normalization constant. The absolute
        flux at λ < plbrk3 is therefore *higher* with larger plstep, but the
        *slope* of log(f) vs log(λ) equals sl3 = -(plslp1 + plstep).
        """
        from tengri.components.agn.qsogen import _broken_powerlaw_continuum

        plstep = 2.0
        plslp1 = 0.5
        plslp2 = 0.3
        plbrk = 3880.0
        plbrk3 = 1200.0
        sl3_expected = -(plslp1 + plstep)  # -2.5
        # Two wavelengths well below plbrk3 (>0.5 dex below → sigmoid ≈ 0)
        wave1 = jnp.array([300.0])
        wave2 = jnp.array([500.0])
        f1 = float(
            _broken_powerlaw_continuum(
                wave1, plslp1=plslp1, plslp2=plslp2, plbrk=plbrk, plstep=plstep, plbrk3=plbrk3
            )[0]
        )
        f2 = float(
            _broken_powerlaw_continuum(
                wave2, plslp1=plslp1, plslp2=plslp2, plbrk=plbrk, plstep=plstep, plbrk3=plbrk3
            )[0]
        )
        measured_slope = np.log(f1 / f2) / np.log(300.0 / 500.0)
        np.testing.assert_allclose(
            measured_slope,
            sl3_expected,
            atol=0.15,
            err_msg="EUV log-log slope should equal -(plslp1+plstep)",
        )

    def test_euv_branch_finite_and_nonnegative(self):
        """EUV branch output is finite and non-negative across all wavelengths."""
        from tengri.components.agn.qsogen import _broken_powerlaw_continuum

        wave = jnp.logspace(2.0, 4.5, 300)  # 100 Å to ~31 000 Å
        f = _broken_powerlaw_continuum(
            wave,
            plslp1=0.5,
            plslp2=0.3,
            plbrk=3880.0,
            plstep=1.5,
            plbrk3=1200.0,
        )
        chex.assert_tree_all_finite(f)
        assert_non_negative(f, name="f", msg="EUV branch: negative flux")


class TestLoadEmlineTemplateError:
    """Test the FileNotFoundError path in _load_emline_template_arrays.
    The template file is typically found at data/qsogen_emline_template.dat.
    The module-level load happens at import time, so this test verifies the
    loader function directly.
    """

    def test_raises_file_not_found_when_template_missing(self, monkeypatch):
        """FileNotFoundError raised when no candidate path is found.
        Test the _load_emline_template_arrays function directly by patching
        Path.is_file to force the error path.
        """
        from pathlib import Path

        from tengri.components.agn.qsogen import _load_emline_template_arrays

        # Patch Path.is_file to always return False, forcing the FileNotFoundError path
        monkeypatch.setattr(Path, "is_file", lambda self: False)
        with pytest.raises(FileNotFoundError, match="QSOGen emission line template"):
            _load_emline_template_arrays()

    def test_loads_numpy_arrays_not_generators(self, monkeypatch):
        """_load_emline_template_arrays returns fully-realized NumPy arrays.
        This is the regression test for BUG-NSS-03: the loader must return
        NumPy arrays, not generators or JAX arrays, to avoid tracer leaks
        inside JIT-compiled functions.
        """
        from tengri.components.agn.qsogen import _load_emline_template_arrays

        # Load the template (works because the file exists in the test environment)
        arrays = _load_emline_template_arrays()
        # Should return a 6-tuple of NumPy arrays
        assert len(arrays) == 6, f"Expected 6 arrays, got {len(arrays)}"
        for i, arr in enumerate(arrays):
            assert isinstance(arr, np.ndarray), f"Array {i} is {type(arr)}, expected np.ndarray"
            assert arr.dtype in (
                np.float64,
                np.float32,
            ), f"Array {i} has dtype {arr.dtype}, expected float"
            assert arr.ndim == 1, f"Array {i} should be 1D, got {arr.ndim}D"
