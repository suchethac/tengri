"""Tests for the ADAF + truncated disc model (disc.adaf_disc + unified.adaf_agn)."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def wavelength():
    """Broad wavelength grid from radio (1 cm) to hard X-ray (1 A)."""
    return jnp.logspace(0, 8, 500)  # 1 A to 10^8 A (= 1 cm)


@pytest.fixture()
def optical_wavelength():
    """Optical/UV wavelength grid."""
    return jnp.logspace(2.5, 5.0, 200)  # 316 A to 100,000 A


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestAdafDisc:
    """Tests for the low-level adaf_disc function."""

    def test_finite_sed(self, wavelength):
        """ADAF produces finite SED values everywhere."""
        from tengri.models.agn.disc import adaf_disc

        l_nu = adaf_disc(
            wavelength,
            agn_log_lbol=42.0,
            agn_frac=0.1,
            agn_log_mbh=8.0,
            agn_log_ledd=-3.0,
            agn_r_tr=100.0,
        )
        assert jnp.all(jnp.isfinite(l_nu))
        assert l_nu.shape == wavelength.shape

    def test_non_negative(self, wavelength):
        """ADAF SED is non-negative everywhere."""
        from tengri.models.agn.disc import adaf_disc

        l_nu = adaf_disc(
            wavelength,
            agn_log_lbol=42.0,
            agn_frac=0.1,
            agn_log_mbh=8.0,
            agn_log_ledd=-3.0,
        )
        assert jnp.all(l_nu >= 0.0)

    def test_peaks_at_longer_wavelengths_than_standard_disc(self, wavelength):
        """ADAF SED peaks at longer wavelengths than a standard thin disc.

        The ADAF synchrotron peak is in the radio/mm regime (~300 um),
        while the standard disc peaks in the UV.
        """
        from tengri.models.agn.disc import adaf_disc, multicolor_disc

        l_adaf = adaf_disc(
            wavelength,
            agn_log_lbol=42.0,
            agn_frac=1.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-3.0,
            agn_r_tr=100.0,
        )
        l_disc = multicolor_disc(
            wavelength,
            agn_log_lbol=42.0,
            agn_frac=1.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-1.0,
        )

        # Find peak wavelengths
        peak_adaf = wavelength[jnp.argmax(l_adaf * wavelength)]
        peak_disc = wavelength[jnp.argmax(l_disc * wavelength)]

        # ADAF should peak at longer wavelength (lower frequency)
        assert peak_adaf > peak_disc

    def test_truncation_radius_affects_uv(self, optical_wavelength):
        """Larger truncation radius reduces UV emission from outer disc.

        A larger r_tr means the thin disc starts further out (cooler),
        producing less UV/optical emission.
        """
        from tengri.models.agn.disc import adaf_disc

        # UV band: 1000-3000 A
        uv_mask = (optical_wavelength > 1000.0) & (optical_wavelength < 3000.0)

        l_small_tr = adaf_disc(
            optical_wavelength,
            agn_log_lbol=42.0,
            agn_frac=1.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-3.0,
            agn_r_tr=30.0,
        )
        l_large_tr = adaf_disc(
            optical_wavelength,
            agn_log_lbol=42.0,
            agn_frac=1.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-3.0,
            agn_r_tr=500.0,
        )

        # Larger r_tr -> less UV (hotter inner disc is truncated further out)
        uv_small = jnp.sum(l_small_tr[uv_mask])
        uv_large = jnp.sum(l_large_tr[uv_mask])
        assert uv_small > uv_large

    def test_adaf_faint_at_high_ledd(self, wavelength):
        """At high L/L_Edd, ADAF component is faint relative to disc.

        The ADAF radiative efficiency scales as r_isco/r_tr, so when
        r_tr is small (high accretion) the ADAF is more efficient but
        the disc dominates.
        """
        from tengri.models.agn.disc import adaf_disc

        # Low Eddington ratio: ADAF regime
        l_low = adaf_disc(
            wavelength,
            agn_log_lbol=42.0,
            agn_frac=1.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-4.0,
            agn_r_tr=300.0,
        )

        # Higher Eddington ratio: disc-dominated
        l_high = adaf_disc(
            wavelength,
            agn_log_lbol=44.0,
            agn_frac=1.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-1.0,
            agn_r_tr=10.0,
        )

        # Higher L_bol should produce brighter overall SED.
        # Compare bolometric luminosities via frequency integral — raw sum
        # over a log-spaced wavelength grid is NOT a bolometric proxy because
        # L_nu * dnu gains a nu factor, biasing toward radio-peaking SEDs.
        c_aa_per_s = 2.99792458e18  # c in Angstrom/s
        nu = c_aa_per_s / wavelength  # Hz, descending when wavelength ascending
        sort_idx = jnp.argsort(nu)
        lbol_high = jnp.trapezoid(l_high[sort_idx], nu[sort_idx])
        lbol_low = jnp.trapezoid(l_low[sort_idx], nu[sort_idx])
        assert lbol_high > lbol_low, (
            f"Higher L_bol SED not brighter: {lbol_high:.3e} vs {lbol_low:.3e} Lsun"
        )

    def test_agn_frac_scaling(self, wavelength):
        """agn_frac linearly scales the output."""
        from tengri.models.agn.disc import adaf_disc

        l_full = adaf_disc(wavelength, agn_log_lbol=42.0, agn_frac=1.0)
        l_half = adaf_disc(wavelength, agn_log_lbol=42.0, agn_frac=0.5)

        ratio = l_full / jnp.maximum(l_half, 1e-100)
        # Should be ~2 everywhere (within numerical precision)
        assert jnp.allclose(ratio, 2.0, rtol=0.01, atol=1e-30)


# ---------------------------------------------------------------------------
# JIT and gradient compatibility
# ---------------------------------------------------------------------------


class TestAdafJitGrad:
    """JIT compilation and gradient tests."""

    def test_jit_compatible(self, wavelength):
        """adaf_disc is JIT-compilable."""
        from tengri.models.agn.disc import adaf_disc

        @jax.jit
        def _run(wave):
            return adaf_disc(wave, agn_log_lbol=42.0, agn_frac=0.1)

        result = _run(wavelength)
        assert jnp.all(jnp.isfinite(result))

    def test_gradient_wrt_lbol(self, optical_wavelength):
        """Gradient of adaf_disc w.r.t. agn_log_lbol is finite."""
        from tengri.models.agn.disc import adaf_disc

        def _loss(log_lbol):
            return jnp.sum(adaf_disc(optical_wavelength, agn_log_lbol=log_lbol, agn_frac=0.1))

        grad = jax.grad(_loss)(42.0)
        assert jnp.isfinite(grad)

    def test_gradient_wrt_r_tr(self, optical_wavelength):
        """Gradient of adaf_disc w.r.t. agn_r_tr is finite."""
        from tengri.models.agn.disc import adaf_disc

        def _loss(r_tr):
            return jnp.sum(
                adaf_disc(
                    optical_wavelength,
                    agn_log_lbol=42.0,
                    agn_frac=0.1,
                    agn_r_tr=r_tr,
                )
            )

        grad = jax.grad(_loss)(100.0)
        assert jnp.isfinite(grad)

    def test_gradient_wrt_delta(self, optical_wavelength):
        """Gradient of adaf_disc w.r.t. agn_adaf_delta is finite."""
        from tengri.models.agn.disc import adaf_disc

        def _loss(delta):
            return jnp.sum(
                adaf_disc(
                    optical_wavelength,
                    agn_log_lbol=42.0,
                    agn_frac=0.1,
                    agn_adaf_delta=delta,
                )
            )

        grad = jax.grad(_loss)(0.01)
        assert jnp.isfinite(grad)

    def test_gradient_wrt_beta(self, optical_wavelength):
        """Gradient of adaf_disc w.r.t. agn_adaf_beta is finite."""
        from tengri.models.agn.disc import adaf_disc

        def _loss(beta):
            return jnp.sum(
                adaf_disc(
                    optical_wavelength,
                    agn_log_lbol=42.0,
                    agn_frac=0.1,
                    agn_adaf_beta=beta,
                )
            )

        grad = jax.grad(_loss)(0.5)
        assert jnp.isfinite(grad)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestAdafRegistry:
    """Tests that ADAF is properly registered in the AGN model registry."""

    def test_registered_as_adaf(self):
        """'adaf' is in AGN_MODELS registry."""
        from tengri.models.agn.unified import AGN_MODELS

        assert "adaf" in AGN_MODELS

    def test_get_agn_model_adaf(self):
        """resolve_agn_model('adaf') returns a callable."""
        from tengri.models.agn.unified import resolve_agn_model

        model_fn = resolve_agn_model("adaf")
        assert callable(model_fn)

    def test_registered_model_runs(self, optical_wavelength):
        """The registered 'adaf' model produces finite output."""
        from tengri.models.agn.unified import resolve_agn_model

        model_fn = resolve_agn_model("adaf")
        l_nu = model_fn(optical_wavelength, agn_log_lbol=42.0)
        assert jnp.all(jnp.isfinite(l_nu))
        assert l_nu.shape == optical_wavelength.shape

    def test_adaf_in_unified_disc_fns(self, optical_wavelength):
        """'adaf' disc type works in unified_agn combiner."""
        from tengri.models.agn.unified import unified_agn

        l_nu = unified_agn(
            optical_wavelength,
            agn_log_lbol=42.0,
            disc_model="adaf",
            torus_model="simple",
        )
        assert jnp.all(jnp.isfinite(l_nu))
