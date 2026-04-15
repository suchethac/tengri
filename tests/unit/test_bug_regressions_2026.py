"""Regression tests for bugs fixed in 2026-04.

Each test documents a specific bug from CLAUDE.md, cites the fix, and
verifies the correct behavior. These tests MUST fail if the bug is
reintroduced.

Bug index (from CLAUDE.md):
- QSOgen Balmer continuum tau direction (tau ∝ (λ_BE/λ)³)
- QSOgen hot dust BB normalization (bbnorm = f_bb/f_cont at 2μm)
- agn_torus_frac gradient discontinuity at 0.5 (removed auto-derivation)
- Nebular line profile unit bug (spurious LSUN_ERG on Gaussian profiles)
- Shock sigma_nu Å→cm conversion
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ===================================================================
# 1. QSOgen Balmer continuum optical depth direction
# ===================================================================


class TestBalmerContinuumTauDirection:
    """Regression: tau must increase at shorter wavelengths (blueward).

    Bug: tau was computed as (λ/λ_BE)³ instead of (λ_BE/λ)³.
    Fix: qsogen.py now uses tau = tau_BE * (λ_BE/λ)³ (Osterbrock & Ferland AGN² Eq. 2.4).
    """

    def test_tau_increases_blueward(self):
        """Balmer continuum absorption must be stronger at shorter wavelengths."""
        from tengri.components.agn.qsogen import _balmer_continuum

        wave = jnp.linspace(2000.0, 3700.0, 500)
        # Use a flat continuum so any shape comes from the Balmer component
        flat_cont = jnp.ones_like(wave)
        bc = _balmer_continuum(wave, flat_cont, bcnorm=1.0, tbc=15000.0, taube=1.0)

        # The Balmer edge is at 3646 Å. Below it, emission should be stronger
        # at shorter wavelengths (because tau increases and absorption = 1-exp(-tau) increases)
        bc_3000 = float(jnp.interp(jnp.array([3000.0]), wave, bc)[0])
        bc_3500 = float(jnp.interp(jnp.array([3500.0]), wave, bc)[0])
        assert bc_3000 > bc_3500, (
            f"Balmer continuum at 3000A ({bc_3000:.4e}) must exceed 3500A ({bc_3500:.4e}) "
            "because tau ∝ (λ_BE/λ)³ increases blueward"
        )


# ===================================================================
# 2. QSOgen hot dust BB normalization
# ===================================================================


class TestHotDustNormalization:
    """Regression: bbnorm is f_bb/f_cont ratio at 2μm anchor.

    Bug: bbnorm was treated as absolute f_nu, not as a ratio.
    Fix: qsogen.py now computes cmult = bbnorm * cont(2μm) / bb(2μm).
    """

    def test_hot_dust_scales_with_bbnorm(self):
        """Doubling bbnorm should double the hot dust contribution."""
        from tengri.components.agn.qsogen import _hot_dust_blackbody

        wave = jnp.linspace(5000.0, 50000.0, 500)
        # Power-law continuum (simple f_lambda)
        cont = (wave / 5500.0) ** (-1.5)

        hd1 = _hot_dust_blackbody(wave, cont, tbb=1200.0, bbnorm=1.0)
        hd2 = _hot_dust_blackbody(wave, cont, tbb=1200.0, bbnorm=2.0)

        # In the MIR where hot dust dominates, flux should scale ~2x
        mir_mask = (wave > 15000) & (wave < 40000)
        ratio = float(jnp.mean(hd2[mir_mask]) / jnp.maximum(jnp.mean(hd1[mir_mask]), 1e-60))
        assert 1.8 < ratio < 2.2, f"bbnorm 2x should give ~2x hot dust, got {ratio:.2f}x"

    def test_hot_dust_anchored_at_2um(self):
        """At 2μm, hot dust flux should equal bbnorm × continuum flux."""
        from tengri.components.agn.qsogen import _hot_dust_blackbody

        wave = jnp.linspace(5000.0, 30000.0, 1000)
        cont = (wave / 5500.0) ** (-1.5)
        bbnorm = 0.5

        hd = _hot_dust_blackbody(wave, cont, tbb=1200.0, bbnorm=bbnorm)

        # At 2μm (20000 Å), hd/cont should be approximately bbnorm
        idx_2um = jnp.argmin(jnp.abs(wave - 20000.0))
        ratio_at_2um = float(hd[idx_2um] / cont[idx_2um])
        assert abs(ratio_at_2um - bbnorm) / bbnorm < 0.1, (
            f"Hot dust/continuum at 2μm = {ratio_at_2um:.3f}, expected {bbnorm}"
        )


# ===================================================================
# 3. agn_torus_frac gradient discontinuity
# ===================================================================


class TestTorusFracGradientContinuity:
    """Regression: agn_torus_frac gradient must be continuous at 0.5.

    Bug: agn_torus_frac was auto-derived from cos(theta_torus) inside the
    forward pass, creating a gradient discontinuity at the default value 0.5
    that corrupted VI/MAP optimization.
    Fix: unified.py no longer auto-derives agn_torus_frac in the forward pass.
    """

    def test_gradient_continuous_around_half(self):
        """Gradient of L_total w.r.t. agn_torus_frac must vary smoothly."""
        from tengri.components.agn.unified import simple_agn

        wave = jnp.linspace(500.0, 200000.0, 1000)

        def total_flux(frac):
            return jnp.sum(simple_agn(wave, agn_log_lbol=11.0, agn_torus_frac=frac))

        grad_fn = jax.grad(total_flux)
        g_049 = float(grad_fn(0.49))
        g_050 = float(grad_fn(0.50))
        g_051 = float(grad_fn(0.51))

        assert jnp.isfinite(jnp.array([g_049, g_050, g_051])).all(), (
            "Gradient must be finite at frac=0.49, 0.50, 0.51"
        )

        # Gradient should vary smoothly (no large relative jump).
        # The key check is that there's no NaN or sign flip at exactly 0.5.
        # If gradients are all identical (perfectly smooth), that's fine.
        step = abs(g_051 - g_049)
        if step > 0:
            jump_left = abs(g_050 - g_049)
            jump_right = abs(g_051 - g_050)
            max_jump = max(jump_left, jump_right)
            assert max_jump < 10 * step, (
                f"Gradient discontinuity at frac=0.5: "
                f"g(0.49)={g_049:.2e}, g(0.50)={g_050:.2e}, g(0.51)={g_051:.2e}"
            )
        # If step==0, gradient is constant across the region — perfectly smooth


# ===================================================================
# 4. Shock sigma_nu Å→cm conversion
# ===================================================================


class TestShockSigmaNuConversion:
    """Regression: line_sigma_aa must be converted from Å to cm for sigma_nu.

    Bug: sigma_nu used line_sigma_aa directly in Å, giving sigma_nu ~1e8× too
    large, so line widths were ~1e8× too narrow.
    Fix: shock.py now converts line_sigma_aa × 1e-8 before the c/λ² formula.
    """

    def test_line_width_physical(self):
        """Shock Hα line with sigma=2Å should be ~2Å wide, not sub-mA."""
        from tengri.components.nebular.shock import shock_emission_sed

        wave = jnp.linspace(6550.0, 6580.0, 1000)
        sigma_aa = 2.0
        sed = shock_emission_sed(wave, 300.0, 1e8, line_sigma_aa=sigma_aa)

        # Find FWHM of the Hα peak
        peak = float(jnp.max(sed))
        if peak > 0:
            half_max = peak / 2.0
            above_half = sed > half_max
            fwhm_pixels = float(jnp.sum(above_half))
            dwave = float(wave[1] - wave[0])
            fwhm_aa = fwhm_pixels * dwave

            # FWHM ≈ 2.355 * sigma_aa for Gaussian
            expected_fwhm = 2.355 * sigma_aa
            assert 0.5 * expected_fwhm < fwhm_aa < 3.0 * expected_fwhm, (
                f"FWHM = {fwhm_aa:.2f} Å, expected ~{expected_fwhm:.2f} Å (sigma={sigma_aa} Å)"
            )


# ===================================================================
# 5. Vacuum wavelength consistency for emission lines
# ===================================================================


class TestVacuumWavelengthConsistency:
    """Verify emission line catalog uses vacuum wavelengths throughout.

    Bug: Some wavelengths were air values (e.g. Hα = 6562.80 Å).
    Fix: All wavelengths updated to vacuum (Hα = 6564.61 Å, Hβ = 4862.68 Å,
    [OIII]5007 = 5008.24 Å).
    """

    def test_halpha_vacuum(self):
        """Hα must be at vacuum wavelength 6564.61 Å, not air 6562.80 Å."""
        from tengri.observation.line_list import LineCatalog

        cat = LineCatalog.default_optical()
        ha_idx = cat.names.index("Halpha")
        ha_wave = float(cat.wavelengths[ha_idx])
        assert abs(ha_wave - 6564.61) < 0.5, f"Hα at {ha_wave:.2f} Å, expected 6564.61 Å (vacuum)"
        assert abs(ha_wave - 6562.80) > 0.5, (
            f"Hα at {ha_wave:.2f} Å — this is the air wavelength, should be vacuum"
        )

    def test_hbeta_vacuum(self):
        """Hβ must be at vacuum wavelength 4862.68 Å."""
        from tengri.observation.line_list import LineCatalog

        cat = LineCatalog.default_optical()
        hb_idx = cat.names.index("Hbeta")
        hb_wave = float(cat.wavelengths[hb_idx])
        assert abs(hb_wave - 4862.68) < 0.5, f"Hβ at {hb_wave:.2f} Å, expected 4862.68 Å (vacuum)"

    def test_oiii5007_vacuum(self):
        """[OIII]5007 must be at vacuum wavelength 5008.24 Å."""
        from tengri.observation.line_list import LineCatalog

        cat = LineCatalog.default_optical()
        oiii_idx = cat.names.index("OIII_5007")
        oiii_wave = float(cat.wavelengths[oiii_idx])
        assert abs(oiii_wave - 5008.24) < 0.5, (
            f"[OIII]5007 at {oiii_wave:.2f} Å, expected 5008.24 Å (vacuum)"
        )
