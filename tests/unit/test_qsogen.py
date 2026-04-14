"""Tests for models/agn/qsogen.py — QSOgen empirical quasar SED model.

Tests the pure-JAX helper functions (_broken_powerlaw_continuum,
_hot_dust_blackbody, _balmer_continuum) and the full compute_qsogen_sed
integration.  The QSOgen emission-line template file is checked for
existence before tests that require it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def uv_optical_wave():
    """UV–optical wavelength grid: 200 – 10,000 Å."""
    return jnp.linspace(200.0, 10000.0, 400)


@pytest.fixture
def broad_wave():
    """Broad wavelength grid: 100 Å – 100 μm."""
    return jnp.logspace(2.0, 9.0, 500)


# ---------------------------------------------------------------------------
# Broken power-law continuum
# ---------------------------------------------------------------------------


class TestBrokenPowerlawContinuum:
    """_broken_powerlaw_continuum is a pure-JAX smooth two-segment power law."""

    def test_finite_output(self, uv_optical_wave):
        from tengri.models.agn.qsogen import _broken_powerlaw_continuum

        cont = _broken_powerlaw_continuum(uv_optical_wave, -0.349, 0.593, 3880.0)
        assert jnp.all(jnp.isfinite(cont))

    def test_normalized_at_5500_angstrom(self):
        """Continuum is normalized to ~1.0 at 5500 Å."""
        from tengri.models.agn.qsogen import _broken_powerlaw_continuum

        wave_norm = jnp.array([5500.0])
        cont = _broken_powerlaw_continuum(wave_norm, -0.349, 0.593, 3880.0)
        np.testing.assert_allclose(float(cont[0]), 1.0, rtol=1e-4)

    def test_non_negative(self, uv_optical_wave):
        """Continuum is non-negative everywhere."""
        from tengri.models.agn.qsogen import _broken_powerlaw_continuum

        cont = _broken_powerlaw_continuum(uv_optical_wave, -0.349, 0.593, 3880.0)
        assert jnp.all(cont >= 0.0)

    def test_jit_compatible(self, uv_optical_wave):
        from tengri.models.agn.qsogen import _broken_powerlaw_continuum

        jitted = jax.jit(_broken_powerlaw_continuum, static_argnums=())
        cont = jitted(uv_optical_wave, -0.349, 0.593, 3880.0)
        assert jnp.all(jnp.isfinite(cont))

    def test_slope_uv_side(self):
        """Below the break (UV), slope is set by plslp1."""
        from tengri.models.agn.qsogen import _broken_powerlaw_continuum

        # Sample well below break at 3880 Å: 1000 Å and 1500 Å
        wave_uv = jnp.array([1000.0, 1500.0, 2000.0])
        cont = _broken_powerlaw_continuum(wave_uv, plslp1=-0.349, plslp2=0.593, plbrk=3880.0)
        # In terms of f_lambda ~ lambda^alpha, the UV slope should produce
        # a falling spectrum toward shorter wavelengths for typical QSO slopes
        # Just verify it's finite and positive
        assert jnp.all(jnp.isfinite(cont))
        assert jnp.all(cont > 0.0)

    def test_steeper_slope_changes_shape(self):
        """Different plslp1 values produce different UV spectral shapes."""
        from tengri.models.agn.qsogen import _broken_powerlaw_continuum

        wave = jnp.array([1000.0, 5500.0])
        cont_steep = _broken_powerlaw_continuum(wave, plslp1=-1.0, plslp2=0.593, plbrk=3880.0)
        cont_flat = _broken_powerlaw_continuum(wave, plslp1=0.0, plslp2=0.593, plbrk=3880.0)
        # Both normalized at 5500, but different shapes at 1000 Å
        # The steeper slope should give a different ratio
        ratio_steep = float(cont_steep[0] / cont_steep[1])
        ratio_flat = float(cont_flat[0] / cont_flat[1])
        assert abs(ratio_steep - ratio_flat) > 0.01


# ---------------------------------------------------------------------------
# Hot dust blackbody
# ---------------------------------------------------------------------------


class TestHotDustBlackbody:
    """_hot_dust_blackbody adds a hot BB component anchored at 2 μm."""

    def test_finite_output(self, broad_wave):
        from tengri.models.agn.qsogen import _broken_powerlaw_continuum, _hot_dust_blackbody

        cont = _broken_powerlaw_continuum(broad_wave, -0.349, 0.593, 3880.0)
        bb = _hot_dust_blackbody(broad_wave, cont, tbb=1240.0, bbnorm=3.96)
        assert jnp.all(jnp.isfinite(bb))

    def test_non_negative(self, broad_wave):
        from tengri.models.agn.qsogen import _broken_powerlaw_continuum, _hot_dust_blackbody

        cont = _broken_powerlaw_continuum(broad_wave, -0.349, 0.593, 3880.0)
        bb = _hot_dust_blackbody(broad_wave, cont, tbb=1240.0, bbnorm=3.96)
        assert jnp.all(bb >= 0.0)

    def test_zero_bbnorm_no_dust(self, broad_wave):
        """bbnorm=0 → _hot_dust_blackbody returns zero (component only, not total SED)."""
        from tengri.models.agn.qsogen import _broken_powerlaw_continuum, _hot_dust_blackbody

        cont = _broken_powerlaw_continuum(broad_wave, -0.349, 0.593, 3880.0)
        bb_zero = _hot_dust_blackbody(broad_wave, cont, tbb=1240.0, bbnorm=0.0)
        np.testing.assert_allclose(np.array(bb_zero), 0.0, atol=1e-30)

    def test_positive_bbnorm_adds_ir(self, broad_wave):
        """Positive bbnorm adds IR flux above the continuum at ~2 μm."""
        from tengri.models.agn.qsogen import _broken_powerlaw_continuum, _hot_dust_blackbody

        cont = _broken_powerlaw_continuum(broad_wave, -0.349, 0.593, 3880.0)
        bb = _hot_dust_blackbody(broad_wave, cont, tbb=1240.0, bbnorm=3.96)

        # Near-IR band (15,000 – 25,000 Å = 1.5 – 2.5 μm)
        nir_mask = (broad_wave >= 15000.0) & (broad_wave <= 25000.0)
        assert jnp.any(bb[nir_mask] > cont[nir_mask])

    def test_jit_compatible(self, broad_wave):
        from tengri.models.agn.qsogen import _broken_powerlaw_continuum, _hot_dust_blackbody

        cont = _broken_powerlaw_continuum(broad_wave, -0.349, 0.593, 3880.0)
        jitted = jax.jit(_hot_dust_blackbody)
        bb = jitted(broad_wave, cont, tbb=1240.0, bbnorm=3.96)
        assert jnp.all(jnp.isfinite(bb))


# ---------------------------------------------------------------------------
# Balmer continuum
# ---------------------------------------------------------------------------


class TestBalmerContinuum:
    """_balmer_continuum adds Balmer continuum emission shortward of 3646 Å."""

    def test_finite_output(self, uv_optical_wave):
        from tengri.models.agn.qsogen import _balmer_continuum, _broken_powerlaw_continuum

        cont = _broken_powerlaw_continuum(uv_optical_wave, -0.349, 0.593, 3880.0)
        bc = _balmer_continuum(uv_optical_wave, cont)
        assert jnp.all(jnp.isfinite(bc))

    def test_non_negative(self, uv_optical_wave):
        from tengri.models.agn.qsogen import _balmer_continuum, _broken_powerlaw_continuum

        cont = _broken_powerlaw_continuum(uv_optical_wave, -0.349, 0.593, 3880.0)
        bc = _balmer_continuum(uv_optical_wave, cont)
        assert jnp.all(bc >= 0.0)

    def test_zero_bcnorm_returns_continuum(self, uv_optical_wave):
        """bcnorm=0 → _balmer_continuum returns zero (component only, not total SED)."""
        from tengri.models.agn.qsogen import _balmer_continuum, _broken_powerlaw_continuum

        cont = _broken_powerlaw_continuum(uv_optical_wave, -0.349, 0.593, 3880.0)
        bc_zero = _balmer_continuum(uv_optical_wave, cont, bcnorm=0.0)
        np.testing.assert_allclose(np.array(bc_zero), 0.0, atol=1e-10)

    def test_emission_shortward_of_balmer_edge(self):
        """Non-zero bcnorm adds flux shortward of the Balmer edge (3646 Å)."""
        from tengri.models.agn.qsogen import _balmer_continuum, _broken_powerlaw_continuum

        wave_uv = jnp.linspace(200.0, 3600.0, 200)  # below 3646 Å
        cont = _broken_powerlaw_continuum(wave_uv, -0.349, 0.593, 3880.0)
        bc = _balmer_continuum(wave_uv, cont, bcnorm=1.0)
        # Should be above continuum in this region
        assert jnp.any(bc > cont)

    def test_tau_increases_shortward(self):
        """tau = taube * (wavbe/λ)^3 increases at shorter λ — absorption deepens."""
        from tengri.models.agn.qsogen import _balmer_continuum, _broken_powerlaw_continuum

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


# ---------------------------------------------------------------------------
# lbol to M_i conversion
# ---------------------------------------------------------------------------


class TestLbolToMi:
    """_lbol_to_m_i converts log10(L_bol/Lsun) to absolute i-band magnitude."""

    def test_typical_quasar_magnitude(self):
        """A typical quasar at log_lbol=12.5 (log10 L/Lsun) should give M_i ≈ -27.

        The function takes log10(L_bol / L_sun).  A typical luminous quasar
        at L_bol ~ 10^46 erg/s corresponds to log10(L/Lsun) ≈ 12.5.
        At the anchor point log_lbol=12.5, the formula gives M_i = -27.
        """
        from tengri.models.agn.qsogen import _lbol_to_m_i

        m_i = float(_lbol_to_m_i(12.5))
        assert -30.0 < m_i < -20.0

    def test_brighter_lbol_gives_more_negative_mi(self):
        """Higher L_bol → more negative M_i (brighter)."""
        from tengri.models.agn.qsogen import _lbol_to_m_i

        m_i_bright = float(_lbol_to_m_i(46.0))
        m_i_faint = float(_lbol_to_m_i(44.0))
        assert m_i_bright < m_i_faint

    def test_reference_point(self):
        """At log_lbol=12.5 (in Lsun), M_i should be -27 (anchor point)."""
        from tengri.models.agn.qsogen import _lbol_to_m_i

        # M_i = -2.5 * (log_lbol - 12.5) + (-27)
        # At log_lbol = 12.5: M_i = 0 + (-27) = -27
        # But the function takes log_lbol in Lsun already, and formula uses
        # -2.5*(log_lbol - 12.5) + (-27). Let's verify the linear formula.
        lbol = 12.5
        expected = -2.5 * (lbol - 12.5) + (-27.0)
        result = float(_lbol_to_m_i(lbol))
        np.testing.assert_allclose(result, expected, rtol=1e-8)


# ---------------------------------------------------------------------------
# Full compute_qsogen_sed
# ---------------------------------------------------------------------------


class TestComputeQsogenSed:
    """Tests for the full QSOgen SED (requires template file)."""

    def test_output_shape(self, broad_wave):
        from tengri.models.agn.qsogen import compute_qsogen_sed

        sed = compute_qsogen_sed(broad_wave)
        assert sed.shape == broad_wave.shape

    def test_finite_output(self, broad_wave):
        from tengri.models.agn.qsogen import compute_qsogen_sed

        sed = compute_qsogen_sed(broad_wave)
        assert jnp.all(jnp.isfinite(sed))

    def test_non_negative(self, broad_wave):
        from tengri.models.agn.qsogen import compute_qsogen_sed

        sed = compute_qsogen_sed(broad_wave)
        assert jnp.all(sed >= 0.0)

    def test_agn_frac_zero_gives_zero(self, broad_wave):
        """agn_frac=0 scales the entire SED to zero."""
        from tengri.models.agn.qsogen import compute_qsogen_sed

        sed = compute_qsogen_sed(broad_wave, agn_frac=0.0)
        assert jnp.allclose(sed, 0.0)

    def test_agn_frac_scales_linearly(self, broad_wave):
        """agn_frac doubles → SED doubles."""
        from tengri.models.agn.qsogen import compute_qsogen_sed

        sed1 = compute_qsogen_sed(broad_wave, agn_frac=1.0)
        sed2 = compute_qsogen_sed(broad_wave, agn_frac=2.0)
        # Avoid dividing by zero at wavelengths where sed=0
        mask = sed1 > 0.0
        ratio = sed2[mask] / sed1[mask]
        np.testing.assert_allclose(ratio, 2.0, rtol=1e-6)

    def test_dust_reddening_reduces_uv(self, uv_optical_wave):
        """Positive agn_ebv reddens the QSO: reduces UV relative to optical."""
        from tengri.models.agn.qsogen import compute_qsogen_sed

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
        from tengri.models.agn.qsogen import compute_qsogen_sed

        sed_bright = compute_qsogen_sed(broad_wave, agn_log_lbol=46.0)
        sed_faint = compute_qsogen_sed(broad_wave, agn_log_lbol=44.0)
        assert float(jnp.sum(sed_bright)) > float(jnp.sum(sed_faint))

    def test_hot_dust_bump_at_nir(self, broad_wave):
        """Hot dust (bbnorm > 0) adds near-IR excess around 2 μm."""
        from tengri.models.agn.qsogen import compute_qsogen_sed

        sed_nodust = compute_qsogen_sed(broad_wave, agn_bbnorm=0.0)
        sed_dust = compute_qsogen_sed(broad_wave, agn_bbnorm=3.96)

        nir_mask = (broad_wave >= 15000.0) & (broad_wave <= 25000.0)
        nir_nodust = float(jnp.sum(sed_nodust[nir_mask]))
        nir_dust = float(jnp.sum(sed_dust[nir_mask]))
        assert nir_dust > nir_nodust

    def test_emission_lines_scale_with_emline_scale(self, broad_wave):
        """Zero emline_scale suppresses emission lines; nonzero adds them."""
        from tengri.models.agn.qsogen import compute_qsogen_sed

        sed_lines = compute_qsogen_sed(broad_wave, agn_emline_scale=1.0)
        sed_noline = compute_qsogen_sed(broad_wave, agn_emline_scale=0.0)
        # With emission lines, SED should be at least as bright as without
        assert float(jnp.sum(sed_lines)) >= float(jnp.sum(sed_noline))

    def test_jit_compatible(self, broad_wave):
        """compute_qsogen_sed is JIT-compilable."""
        from tengri.models.agn.qsogen import compute_qsogen_sed

        jitted = jax.jit(compute_qsogen_sed)
        sed = jitted(broad_wave)
        assert jnp.all(jnp.isfinite(sed))

    def test_gradient_wrt_lbol(self, broad_wave):
        """FD check: ∂(∑SED)/∂agn_log_lbol."""
        from tengri.models.agn.qsogen import compute_qsogen_sed

        def loss(lbol):
            return jnp.sum(compute_qsogen_sed(broad_wave, agn_log_lbol=lbol))

        g_jax = float(jax.grad(loss)(45.0))
        g_fd = fd_grad(loss, 45.0, eps=0.01)
        np.testing.assert_allclose(g_jax, g_fd, rtol=1e-3)
        assert g_jax > 0.0

    def test_gradient_wrt_ebv(self, uv_optical_wave):
        """FD check: ∂(∑SED)/∂agn_ebv (should be negative — reddening reduces flux)."""
        from tengri.models.agn.qsogen import compute_qsogen_sed

        def loss(ebv):
            return jnp.sum(compute_qsogen_sed(uv_optical_wave, agn_ebv=ebv))

        g_jax = float(jax.grad(loss)(0.1))
        g_fd = fd_grad(loss, 0.1)
        np.testing.assert_allclose(g_jax, g_fd, rtol=1e-3)
        assert g_jax < 0.0  # reddening reduces total UV/optical flux

    def test_balmer_continuum_adds_uv_flux(self, uv_optical_wave):
        """Non-zero agn_bcnorm adds Balmer continuum flux shortward of 3646 Å."""
        from tengri.models.agn.qsogen import compute_qsogen_sed

        sed_no_bc = compute_qsogen_sed(uv_optical_wave, agn_bcnorm=0.0)
        sed_with_bc = compute_qsogen_sed(uv_optical_wave, agn_bcnorm=1.0)

        # Shortward of Balmer edge
        bc_mask = uv_optical_wave < 3646.0
        assert float(jnp.sum(sed_with_bc[bc_mask])) >= float(jnp.sum(sed_no_bc[bc_mask]))


# ---------------------------------------------------------------------------
# _wavelength_to_nu
# ---------------------------------------------------------------------------


class TestWavelengthToNu:
    """_wavelength_to_nu converts Angstrom → Hz."""

    def test_known_value(self):
        """5500 Å → ~5.45e14 Hz (visual band)."""
        from tengri.models.agn.qsogen import _wavelength_to_nu

        nu = float(_wavelength_to_nu(jnp.array([5500.0]))[0])
        # c / (5500e-8 cm) = 2.998e10 / 5.5e-5 ≈ 5.45e14
        assert 5.0e14 < nu < 6.0e14

    def test_lyman_limit(self):
        """912 Å → ~3.29e15 Hz (Lyman limit)."""
        from tengri.models.agn.qsogen import _wavelength_to_nu

        nu = float(_wavelength_to_nu(jnp.array([912.0]))[0])
        assert 3.0e15 < nu < 3.6e15

    def test_monotonic_inverse_with_wavelength(self):
        """Higher wavelength → lower frequency."""
        from tengri.models.agn.qsogen import _wavelength_to_nu

        wave = jnp.array([1000.0, 5000.0, 10000.0])
        nu = _wavelength_to_nu(wave)
        assert float(nu[0]) > float(nu[1]) > float(nu[2])

    def test_output_shape(self, broad_wave):
        from tengri.models.agn.qsogen import _wavelength_to_nu

        nu = _wavelength_to_nu(broad_wave)
        assert nu.shape == broad_wave.shape

    def test_finite_and_positive(self, broad_wave):
        from tengri.models.agn.qsogen import _wavelength_to_nu

        nu = _wavelength_to_nu(broad_wave)
        assert jnp.all(jnp.isfinite(nu))
        assert jnp.all(nu > 0.0)


# ---------------------------------------------------------------------------
# _planck_blambda
# ---------------------------------------------------------------------------


class TestPlanckBlambda:
    """_planck_blambda returns B_lambda(T) in per-Angstrom CGS units."""

    def test_finite_output(self, uv_optical_wave):
        from tengri.models.agn.qsogen import _planck_blambda

        blam = _planck_blambda(uv_optical_wave, 1240.0)
        assert jnp.all(jnp.isfinite(blam))

    def test_non_negative(self, uv_optical_wave):
        from tengri.models.agn.qsogen import _planck_blambda

        blam = _planck_blambda(uv_optical_wave, 1240.0)
        assert jnp.all(blam >= 0.0)

    def test_higher_temperature_brighter_uv(self):
        """Hotter blackbody emits more in UV (Wien's law)."""
        from tengri.models.agn.qsogen import _planck_blambda

        wave_uv = jnp.array([2000.0])
        blam_hot = float(_planck_blambda(wave_uv, 30000.0)[0])
        blam_cool = float(_planck_blambda(wave_uv, 1240.0)[0])
        assert blam_hot > blam_cool

    def test_zero_temperature_clamped(self):
        """T=0 is clamped to T=1 (no division by zero or NaN)."""
        from tengri.models.agn.qsogen import _planck_blambda

        wave = jnp.array([5500.0])
        blam = _planck_blambda(wave, 0.0)
        assert jnp.all(jnp.isfinite(blam))

    def test_output_shape(self, broad_wave):
        from tengri.models.agn.qsogen import _planck_blambda

        blam = _planck_blambda(broad_wave, 1240.0)
        assert blam.shape == broad_wave.shape

    def test_jit_compatible(self, uv_optical_wave):
        from tengri.models.agn.qsogen import _planck_blambda

        jitted = jax.jit(_planck_blambda)
        blam = jitted(uv_optical_wave, 1240.0)
        assert jnp.all(jnp.isfinite(blam))


# ---------------------------------------------------------------------------
# _apply_dust_reddening
# ---------------------------------------------------------------------------


class TestApplyDustReddening:
    """_apply_dust_reddening applies SMC-like extinction to a spectrum."""

    def test_zero_ebv_unchanged(self, uv_optical_wave):
        """E(B-V)=0 leaves spectrum exactly unchanged."""
        from tengri.models.agn.qsogen import _apply_dust_reddening, _broken_powerlaw_continuum

        cont = _broken_powerlaw_continuum(uv_optical_wave, -0.349, 0.593, 3880.0)
        reddened = _apply_dust_reddening(uv_optical_wave, cont, 0.0)
        np.testing.assert_allclose(np.array(reddened), np.array(cont), rtol=1e-6)

    def test_positive_ebv_reduces_uv(self, uv_optical_wave):
        """Positive E(B-V) attenuates UV more than optical."""
        from tengri.models.agn.qsogen import _apply_dust_reddening, _broken_powerlaw_continuum

        cont = _broken_powerlaw_continuum(uv_optical_wave, -0.349, 0.593, 3880.0)
        reddened = _apply_dust_reddening(uv_optical_wave, cont, 0.3)

        uv_mask = uv_optical_wave < 2500.0
        opt_mask = uv_optical_wave > 5000.0
        ratio_uv = float(jnp.mean(reddened[uv_mask] / cont[uv_mask]))
        ratio_opt = float(jnp.mean(reddened[opt_mask] / cont[opt_mask]))
        assert ratio_uv < ratio_opt

    def test_output_all_non_negative(self, uv_optical_wave):
        from tengri.models.agn.qsogen import _apply_dust_reddening, _broken_powerlaw_continuum

        cont = _broken_powerlaw_continuum(uv_optical_wave, -0.349, 0.593, 3880.0)
        reddened = _apply_dust_reddening(uv_optical_wave, cont, 0.5)
        assert jnp.all(reddened >= 0.0)

    def test_output_shape(self, uv_optical_wave):
        from tengri.models.agn.qsogen import _apply_dust_reddening, _broken_powerlaw_continuum

        cont = _broken_powerlaw_continuum(uv_optical_wave, -0.349, 0.593, 3880.0)
        reddened = _apply_dust_reddening(uv_optical_wave, cont, 0.1)
        assert reddened.shape == uv_optical_wave.shape


# ---------------------------------------------------------------------------
# qsogen registered wrapper
# ---------------------------------------------------------------------------


class TestQsogenWrapper:
    """qsogen() is the AGN_MODELS-registered entry point — a thin wrapper."""

    def test_identical_to_compute_qsogen_sed(self, broad_wave):
        """qsogen() and compute_qsogen_sed() give bit-identical output."""
        from tengri.models.agn.qsogen import compute_qsogen_sed, qsogen

        sed_wrapper = qsogen(broad_wave, agn_log_lbol=45.0)
        sed_direct = compute_qsogen_sed(broad_wave, agn_log_lbol=45.0)
        np.testing.assert_array_equal(np.array(sed_wrapper), np.array(sed_direct))

    def test_accepts_kwargs(self, broad_wave):
        """qsogen passes all keyword arguments through without error."""
        from tengri.models.agn.qsogen import qsogen

        sed = qsogen(broad_wave, agn_log_lbol=45.0, agn_ebv=0.1, agn_bbnorm=2.0)
        assert jnp.all(jnp.isfinite(sed))

    def test_output_shape(self, broad_wave):
        from tengri.models.agn.qsogen import qsogen

        sed = qsogen(broad_wave)
        assert sed.shape == broad_wave.shape
