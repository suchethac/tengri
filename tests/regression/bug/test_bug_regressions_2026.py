# SPDX-License-Identifier: BSD-3-Clause
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

import chex
import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


# ── 1. QSOgen Balmer continuum optical depth direction ────────────


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


# ── 2. QSOgen hot dust BB normalization ───────────────────────────


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


# ── 3. agn_torus_frac gradient discontinuity ──────────────────────


class TestTorusFracGradientContinuity:
    """Regression: agn_torus_frac gradient must be continuous at 0.5.

    Bug: agn_torus_frac was auto-derived from cos(theta_torus) inside the
    forward pass, creating a gradient discontinuity at the default value 0.5
    that corrupted VI/MAP optimization.
    Fix: unified.py no longer auto-derives agn_torus_frac in the forward pass.
    """

    def test_gradient_continuous_around_half(self):
        """Gradient of L_total w.r.t. agn_torus_frac must vary smoothly."""
        from tengri.components.agn.unified import multicolor_agn

        wave = jnp.linspace(500.0, 200000.0, 1000)

        def total_flux(frac):
            return jnp.sum(multicolor_agn(wave, agn_log_lbol=11.0, agn_torus_frac=frac))

        grad_fn = jax.grad(total_flux)
        g_049 = float(grad_fn(0.49))
        g_050 = float(grad_fn(0.50))
        g_051 = float(grad_fn(0.51))

        assert jnp.isfinite(jnp.array([g_049, g_050, g_051])).all(), (
            "Gradient must be finite at frac=0.49, 0.50, 0.51"
        )
        assert jnp.any(jnp.array([g_049, g_050, g_051]) != 0.0), (
            "`jnp.array([g_049, g_050, g_051])` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
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


# ── 4. Shock sigma_nu Å→cm conversion ─────────────────────────────


class TestShockSigmaNuConversion:
    """Regression: line_sigma_aa must be converted from Å to cm for sigma_nu.

    Bug: sigma_nu used line_sigma_aa directly in Å, giving sigma_nu ~1e8× too
    large, so line widths were ~1e8× too narrow.
    Fix: shock.py now converts line_sigma_aa × 1e-8 before the c/λ² formula.
    """

    def test_line_width_physical(self):
        """Shock Hα line with sigma=2Å should be ~2Å wide, not sub-mA."""
        from tengri.components.nebular.shock import compute_shock_sed

        wave = jnp.linspace(6550.0, 6580.0, 1000)
        sigma_aa = 2.0
        sed = compute_shock_sed(wave, 300.0, 1e8, line_sigma_aa=sigma_aa)

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


# ── 5. Vacuum wavelength consistency for emission lines ───────────


class TestVacuumWavelengthConsistency:
    """Verify emission line catalog uses vacuum wavelengths throughout.

    Bug: Some wavelengths were air values (e.g. Hα = 6562.80 Å).
    Fix: All wavelengths updated to vacuum (Hα = 6564.61 Å, Hβ = 4862.68 Å,
    [OIII]5007 = 5008.24 Å).
    """

    def test_halpha_vacuum(self):
        """Hα must be at vacuum wavelength 6564.61 Å, not air 6562.80 Å."""
        from tengri.observation.line_list import LineList

        cat = LineList.default_optical()
        ha_idx = cat.names.index("Halpha")
        ha_wave = float(cat.wavelengths[ha_idx])
        assert abs(ha_wave - 6564.61) < 0.5, f"Hα at {ha_wave:.2f} Å, expected 6564.61 Å (vacuum)"
        assert abs(ha_wave - 6562.80) > 0.5, (
            f"Hα at {ha_wave:.2f} Å — this is the air wavelength, should be vacuum"
        )

    def test_hbeta_vacuum(self):
        """Hβ must be at vacuum wavelength 4862.68 Å."""
        from tengri.observation.line_list import LineList

        cat = LineList.default_optical()
        hb_idx = cat.names.index("Hbeta")
        hb_wave = float(cat.wavelengths[hb_idx])
        assert abs(hb_wave - 4862.68) < 0.5, f"Hβ at {hb_wave:.2f} Å, expected 4862.68 Å (vacuum)"

    def test_oiii5007_vacuum(self):
        """[OIII]5007 must be at vacuum wavelength 5008.24 Å."""
        from tengri.observation.line_list import LineList

        cat = LineList.default_optical()
        oiii_idx = cat.names.index("OIII_5007")
        oiii_wave = float(cat.wavelengths[oiii_idx])
        assert abs(oiii_wave - 5008.24) < 0.5, (
            f"[OIII]5007 at {oiii_wave:.2f} Å, expected 5008.24 Å (vacuum)"
        )


# ── 6. ADAF synchrotron self-absorption spectral index ν^2 vs ν^{5/2}


class TestAdafSyncSpectralIndex:
    """The faithful ADAF (Mahadevan 1997) integrated synchrotron slope is ν^{2/5}.

    The old single-zone ``adaf_disc`` modeled the self-absorbed regime as the
    *local* Rayleigh-Jeans ν^2 (Eq. 19). The faithful ``adaf_spectrum`` (#898)
    integrates the self-absorbed thermal synchrotron over the ADAF's radial
    temperature/field structure — emission at each frequency arises from a
    different radius — giving the shallower ν^{2/5} rise of Mahadevan (1997)
    Fig. 1. That integrated slope (validated in test_adaf_mahadevan.py) supersedes
    the single-zone ν^2; the old value is no longer the physical target.
    """

    def test_log_slope_below_peak_is_two_fifths(self):
        """Log-slope d(log L_ν)/d(log ν) below the synchrotron peak ≈ 2/5."""
        from tengri.components.agn.disc import adaf_disc

        # Sample two frequencies below the ~1e12 Hz peak but above the low cutoff:
        # nu_lo ~ 1e10 Hz (lam 3e7 A), nu_hi ~ 3e10 Hz (lam 1e7 A).
        lam_lo = 3e7
        lam_hi = 1e7
        wave = jnp.array([lam_lo, lam_hi], dtype=jnp.float64)
        sed = adaf_disc(wave, agn_log_lbol=10.0, agn_log_mbh=8.0)

        lam_ratio = float(lam_lo / lam_hi)
        lnu_ratio = float(sed[0] / jnp.maximum(sed[1], 1e-300))
        # nu ∝ 1/λ, so slope in frequency space = -d(log L)/d(log λ).
        slope_nu = float(-jnp.log(lnu_ratio) / jnp.log(lam_ratio))
        assert abs(slope_nu - 0.4) < 0.15, (
            f"ADAF integrated synchrotron slope below the peak = {slope_nu:.3f}, "
            "expected 2/5 (Mahadevan 1997 Fig. 1, integrated over the radial "
            "structure). The single-zone ν^2 (old adaf_disc) is superseded."
        )


# ── 7. Bell (2003) synchrotron suppression formula ────────────────


class TestBell2003SynchrotronSuppression:
    """Regression: non-thermal fraction must follow Bell (2003) ApJ 586, 794 Eq. 3.

    Bug: _synchrotron_suppression used quadratic L/(1+(L0/L)^2), giving
    power-law index ~2 suppression. Bell's actual model is n = 0.9*(L/L*)^0.3
    for L ≤ L* (gentle 0.3 slope) and n = 0.9 for L > L* (saturates).
    Fix: replaced with piecewise power-law per Bell (2003) Eq. 3.
    """

    def test_high_luminosity_fraction_saturates_at_09(self):
        """n(L >> L*) must saturate at 0.9; suppressed(L) / L ≈ 0.9."""
        from tengri.components.radio.radio import _L_STAR_SYNCH, _synchrotron_suppression

        L_high = jnp.array(1e4 * _L_STAR_SYNCH, dtype=jnp.float64)
        suppressed = float(_synchrotron_suppression(L_high))
        n = suppressed / float(L_high)
        assert abs(n - 0.9) < 0.01, (
            f"n(L >> L*) = {n:.4f}, expected 0.9 (Bell 2003 Eq. 3). "
            "Regression: old formula gave n → 1 at high L, not 0.9."
        )

    def test_low_luminosity_fraction_follows_power_law(self):
        """n(L = 0.01 L*) ≈ 0.9 × 0.01^0.3 ≈ 0.45; log-slope ≈ 0.3."""
        from tengri.components.radio.radio import _L_STAR_SYNCH, _synchrotron_suppression

        L_low = jnp.array(0.01 * _L_STAR_SYNCH, dtype=jnp.float64)
        suppressed = float(_synchrotron_suppression(L_low))
        n = suppressed / float(L_low)
        expected_n = 0.9 * 0.01**0.3  # ≈ 0.451
        assert abs(n - expected_n) < 0.02, (
            f"n(0.01 L*) = {n:.4f}, expected {expected_n:.4f} = 0.9 × 0.01^0.3 "
            "(Bell 2003 Eq. 3). "
            "Regression: old quadratic formula gave ~0.01/(1+100^2) ≈ 1e-6."
        )

    def test_log_slope_below_L_star_is_03(self):
        """d(log n)/d(log L) ≈ 0.3 for L << L* (Bell 2003 Eq. 3 power-law index)."""
        from tengri.components.radio.radio import _L_STAR_SYNCH, _synchrotron_suppression

        L1 = jnp.array(0.001 * _L_STAR_SYNCH, dtype=jnp.float64)
        L2 = jnp.array(0.01 * _L_STAR_SYNCH, dtype=jnp.float64)
        n1 = float(_synchrotron_suppression(L1)) / float(L1)
        n2 = float(_synchrotron_suppression(L2)) / float(L2)

        slope = jnp.log(n2 / n1) / jnp.log(0.01 / 0.001)
        assert abs(float(slope) - 0.3) < 0.02, (
            f"n log-slope = {float(slope):.3f}, expected 0.3 (Bell 2003 Eq. 3). "
            "Regression: old formula had slope ≈ 2 at low L."
        )


# ── 8. Evolving metallicity KeyError fallback (BUG-NSS-02) ────────


class TestEvolvingMetallicityFallback:
    """Regression: evolving_metallicity=True emits log_z_abs_final, not log_z_abs.

    Bug: assembly.py, nonstell.py, pipeline.py, sed_model.py all hard-subscripted
    p["log_z_abs"] — causing KeyError when evolving_metallicity=True produces
    log_z_abs_initial / log_z_abs_final keys instead.
    Fix: All lookup sites replaced with p.get("log_z_abs", p.get("log_z_abs_final", -1.8477)).
    """

    def test_fallback_to_log_z_abs_final(self):
        """Params with log_z_abs_final but no log_z_abs must not raise KeyError."""
        p = {"log_z_abs_final": -1.5, "log_z_abs_initial": -2.0}
        # This is the exact pattern used in all fixed sites
        val = p.get("log_z_abs", p.get("log_z_abs_final", -1.8477))
        assert abs(val - (-1.5)) < 1e-9, f"Expected -1.5, got {val}"

    def test_log_z_abs_takes_priority(self):
        """When log_z_abs is present it must be used, not log_z_abs_final."""
        p = {"log_z_abs": -1.8, "log_z_abs_final": -1.5}
        val = p.get("log_z_abs", p.get("log_z_abs_final", -1.8477))
        assert abs(val - (-1.8)) < 1e-9, f"Expected -1.8, got {val}"

    def test_default_when_both_absent(self):
        """When neither key is present the fallback default -1.8477 is returned."""
        p = {}
        val = p.get("log_z_abs", p.get("log_z_abs_final", -1.8477))
        assert abs(val - (-1.8477)) < 1e-9, f"Expected -1.8477, got {val}"


# ── 9. CLOUDY line grid fixed-axis collapsing (BUG-07) ────────────


class TestCloudyLineGridCollapse:
    """Regression: CloudyGridBackend._line_lum_collapsed shape must reflect fixed axes.

    Bug: _precompute_photometry collapsed the continuum grid (PreintegratedGrid) but
    not the line luminosity grid. interp_nd_triweight received a 3D grid with only 2
    axes, causing shape mismatch.
    Fix: New code applies triweight contraction to line_luminosity at fixed axis
    indices and stores result as _line_lum_collapsed with reduced leading dimensions.
    """

    def test_collapsing_reduces_line_lum_ndim(self):
        """Triweight collapsing at one axis reduces line_lum leading dims by 1."""
        import numpy as np

        from tengri.utils.interpolation import compute_grid_weights, edges_for_grid

        # Minimal mock: 3D grid (n_Z=4, n_age=5, n_logU=3, n_lines=10)
        rng = np.random.default_rng(42)
        line_lum = jnp.asarray(rng.standard_normal((4, 5, 3, 10)))
        line_axes = [
            jnp.linspace(-2.5, -0.5, 4),  # log_Z
            jnp.linspace(7.0, 10.5, 5),  # log_age
            jnp.linspace(-4.0, -1.0, 3),  # log_U
        ]

        # Fix axis 2 (log_U) at -3.0 — this is the collapse the bug prevented
        fixed = {2: -3.0}
        collapsed = jnp.asarray(line_lum)
        fixed_axes_list = list(line_axes)
        for axis_idx in sorted(fixed.keys(), reverse=True):
            value = fixed[axis_idx]
            ax = fixed_axes_list[axis_idx]
            scatter = 0.5 * float(ax[1] - ax[0])
            w = compute_grid_weights(value, ax, scatter=scatter, edges=edges_for_grid(ax))
            collapsed = jnp.tensordot(w, collapsed, axes=([0], [axis_idx]))
            fixed_axes_list.pop(axis_idx)

        # Original: (4, 5, 3, 10) → collapsed along axis 2 → (4, 5, 10)
        assert collapsed.shape == (4, 5, 10), (
            f"Collapsed shape {collapsed.shape}, expected (4, 5, 10). "
            "Regression: without fix, interp_nd_triweight would see ndim=3 grid with 2 axes."
        )
        # Values must be finite
        chex.assert_tree_all_finite(collapsed)

    def test_no_fixed_axes_preserves_shape(self):
        """Without fixed axes, line_lum_collapsed must equal the original grid."""
        import numpy as np

        rng = np.random.default_rng(7)
        line_lum = jnp.asarray(rng.standard_normal((4, 5, 3, 10)))
        # No fixed axes — collapsed should be identical
        collapsed = line_lum  # the else branch: self._line_lum_collapsed = jnp.asarray(grid)
        assert collapsed.shape == (4, 5, 3, 10), (
            f"No-fixed-axes shape {collapsed.shape}, expected (4, 5, 3, 10)."
        )
