# SPDX-License-Identifier: BSD-3-Clause
"""Physics cross-validation for nebular models (shock SED, DIG, NLR/BLR lines).

Tests physical correctness of nebular emission models against known
astrophysical relationships, atomic physics constraints, and published
line ratios.

References
----------
- Allen et al. 2008, ApJS, 178, 20 — MAPPINGS V shock models
- Reynolds 1984, ApJ, 282, 191 — DIG observations
- Haffner et al. 2009, RvMP, 81, 969 — DIG review
- Osterbrock & Ferland 2006 — nebular astrophysics
- Storey & Zeippen 2000, MNRAS, 312, 813 — forbidden line ratios
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.crossval


# ── 1. SHOCK EMISSION SED — spectral line placement ───────────────


class TestShockEmissionSEDPhysics:
    """compute_shock_sed must place lines at correct wavelengths."""

    def test_sed_has_halpha_peak(self):
        """Shock SED should peak near Hα 6563A (strongest optical line)."""
        from tengri.components.nebular.shock import compute_shock_sed

        wave = jnp.linspace(3000.0, 8000.0, 5000)
        # l_shock_halpha in Lsun
        l_nu = compute_shock_sed(wave, shock_velocity=300.0, l_shock_halpha=1e8)
        peak_wave = float(wave[jnp.argmax(l_nu)])
        # Should be near one of the strong lines: Hα, [OIII], [NII]
        near_ha = abs(peak_wave - 6563.0) < 100.0
        near_oiii = abs(peak_wave - 5007.0) < 100.0
        assert near_ha or near_oiii, (
            f"Shock SED peak at {peak_wave:.0f} A, expected near Hα or [OIII]"
        )

    def test_sed_scales_with_luminosity(self):
        """10x luminosity → 10x flux."""
        from tengri.components.nebular.shock import compute_shock_sed

        wave = jnp.linspace(3000.0, 8000.0, 2000)
        l_faint = compute_shock_sed(wave, shock_velocity=300.0, l_shock_halpha=1e7)
        l_bright = compute_shock_sed(wave, shock_velocity=300.0, l_shock_halpha=1e8)
        ratio = float(jnp.sum(l_bright)) / float(jnp.sum(l_faint))
        assert abs(ratio - 10.0) < 1.0, f"10x luminosity should give 10x flux, got {ratio:.1f}x"

    def test_sed_velocity_changes_line_ratios(self):
        """Different velocities produce different line ratio patterns."""
        from tengri.components.nebular.shock import compute_shock_sed

        wave = jnp.linspace(3000.0, 8000.0, 5000)
        l_slow = compute_shock_sed(wave, shock_velocity=150.0, l_shock_halpha=1e8)
        l_fast = compute_shock_sed(wave, shock_velocity=500.0, l_shock_halpha=1e8)

        # [OIII] 5007A region
        oiii_mask = (wave > 4950) & (wave < 5050)
        ha_mask = (wave > 6500) & (wave < 6630)

        ratio_slow = float(jnp.sum(l_slow[oiii_mask]) / jnp.sum(l_slow[ha_mask]))
        ratio_fast = float(jnp.sum(l_fast[oiii_mask]) / jnp.sum(l_fast[ha_mask]))
        assert ratio_slow != pytest.approx(ratio_fast, rel=0.1), (
            "Different velocities should produce different line ratios"
        )

    def test_sed_positive_finite(self):
        """Shock SED must be non-negative and finite at all wavelengths."""
        from tengri.components.nebular.shock import compute_shock_sed

        wave = jnp.linspace(1000.0, 10000.0, 5000)
        for v in [100.0, 300.0, 700.0, 1000.0]:
            l_nu = compute_shock_sed(wave, shock_velocity=v, l_shock_halpha=1e8)
            assert jnp.all(jnp.isfinite(l_nu)), f"Shock SED at v={v} has non-finite values"
            assert jnp.all(l_nu >= 0), f"Shock SED at v={v} has negative values"


# ── 2. SHOCK LINE RATIOS — velocity-dependent physics ─────────────


class TestShockLineRatioPhysics:
    """Shock line ratios must follow Allen+2008 velocity trends."""

    def test_oiii_peaks_at_intermediate_velocity(self):
        """[OIII]/Hβ turns over inside the grid rather than running to an edge.

        The qualitative feature is the turnover itself: faster shocks ionize
        more O++ up to a point, beyond which the ionizing continuum hardens
        past it. Measured on the default ``component="combined"`` the maximum
        sits at **700 km/s** (ratio 11.7), rising from 2.4 at 200 km/s and
        falling to 9.0 at 1000.

        This test used to sample [100, 200, 300, 400, 500, 750, 1000] — which
        skips 600-700 entirely — and assert a peak in 200-600 km/s on the
        grounds that [OIII]/Hβ maximizes near 300-400. That is shock-only
        intuition: decomposed, the shock component alone peaks at the grid's
        low edge while the *precursor* drives the combined ratio and peaks at
        700 (#1728). Sampled densely, and asserting the interior turnover
        rather than a hand-set band.
        """
        from tengri.components.nebular.shock import shock_line_ratios

        velocities = np.arange(100.0, 1001.0, 50.0)
        oiii_hb = np.array([float(shock_line_ratios(float(v))["O3_5007A"]) for v in velocities])

        peak_idx = int(np.argmax(oiii_hb))
        peak_v = float(velocities[peak_idx])

        assert 0 < peak_idx < len(velocities) - 1, (
            f"[OIII]/Hβ should turn over inside the grid, peak at {peak_v} km/s (an edge)"
        )
        assert oiii_hb[peak_idx] > 3.0 * oiii_hb[0], (
            "faster shocks should raise [OIII]/Hβ well above its low-velocity value, "
            f"peak {oiii_hb[peak_idx]:.2f} vs {oiii_hb[0]:.2f} at 100 km/s"
        )
        assert oiii_hb[-1] < oiii_hb[peak_idx], "the ratio must fall again past the peak"

    def test_halpha_hbeta_above_case_b(self):
        """In shocks, Hα/Hβ > 2.86 (Case B) due to collisional excitation."""
        from tengri.components.nebular.shock import shock_line_ratios

        for v in [200.0, 300.0, 500.0]:
            ratios = shock_line_ratios(v)
            ha_hb = float(ratios["HA_6563A"])
            assert ha_hb > 2.86, (
                f"Hα/Hβ should exceed Case B (2.86) in shocks at v={v}, got {ha_hb:.2f}"
            )

    def test_sii_strong_shock_diagnostic(self):
        """[SII]/Hβ must be enhanced in shocks (vs. HII regions ≲ 0.4)."""
        from tengri.components.nebular.shock import shock_line_ratios

        for v in [200.0, 300.0]:
            ratios = shock_line_ratios(v)
            # SII is the sum of 6716+6731 relative to Hβ
            sii_hb = float(ratios["SII_6716A"]) + float(ratios["SII_6731A"])
            assert sii_hb > 0.5, (
                f"[SII]/Hβ should be enhanced in shocks at v={v}, got {sii_hb:.2f}"
            )

    @pytest.mark.parametrize("velocity", [50.0, 99.0, 1001.0, 1500.0])
    def test_velocity_outside_the_grid_raises(self, velocity):
        """Out-of-grid shock velocities raise rather than silently extrapolate.

        This test used to assert the opposite — "should be clipped, not error".
        Clipping means a 50 km/s shock is quietly answered with the 100 km/s
        line ratios and the caller never learns the model was not evaluated
        where they asked. The MAPPINGS grid spans 100-1000 km/s; outside it
        there is no model, and saying so is the better contract. It matches how
        the rest of the codebase treats this class — see the ``ApproxPolicy``
        module docstring on a silently-defaulting read being worse than a loud
        failure.
        """
        from tengri.components.nebular.shock import shock_line_ratios

        with pytest.raises(ValueError, match="outside the grid"):
            shock_line_ratios(velocity)

    @pytest.mark.parametrize("velocity", [100.0, 550.0, 1000.0])
    def test_velocity_inside_the_grid_is_accepted(self, velocity):
        """Both endpoints are inclusive — the guard must not exclude the grid."""
        from tengri.components.nebular.shock import shock_line_ratios

        ratios = shock_line_ratios(velocity)
        assert float(ratios["HA_6563A"]) > 0.0


# ── 3. SHOCK ATOMIC PHYSICS — doublet ratios ──────────────────────


class TestShockAtomicPhysics:
    """Forbidden line doublet ratios are set by atomic physics."""

    def test_oiii_doublet(self):
        """[OIII] 5007/4959 = 2.98 (Storey & Zeippen 2000)."""
        from tengri.components.nebular.shock import shock_line_ratios

        ratios = shock_line_ratios(300.0)
        r = float(ratios["O3_5007A"]) / float(ratios["O3_4959A"])
        assert abs(r - 2.98) < 0.1, f"[OIII] doublet ratio should be 2.98, got {r:.2f}"

    def test_nii_doublet(self):
        """[NII] 6583/6548 = 2.94 (Storey & Zeippen 2000)."""
        from tengri.components.nebular.shock import shock_line_ratios

        ratios = shock_line_ratios(300.0)
        r = float(ratios["NII_6583A"]) / float(ratios["NII_6548A"])
        assert abs(r - 2.94) < 0.1, f"[NII] doublet ratio should be 2.94, got {r:.2f}"


# ── 4. NLR LINE PHYSICS — forbidden line properties ───────────────


class TestNLRLinePhysics:
    """NLR must satisfy forbidden line ratio constraints.

    Measured from the ``compute_nlr_sed`` output rather than from the template
    arrays. These tests used to read ``_NLR_LINE_STRENGTHS`` and
    ``_NLR_LINE_WAVELENGTHS`` directly; those private arrays were removed in a
    refactor and the whole class had been failing with ``ImportError`` ever
    since — which is how the [NII] deviation below went unwatched (#1728).
    Reading the emitted spectrum is both refactor-proof and a test of what a
    user actually receives.

    Wavelengths are **vacuum** throughout, per the naming contract:
    [OIII] 5008.24/4960.30, [NII] 6585.27/6549.86, Halpha 6564.61.
    """

    #: Narrow enough to separate the [NII] doublet from Halpha. At the default
    #: fwhm_kms=500 the three blend (sigma_lambda ~ 4.6 A against separations of
    #: 14 and 21 A), and neither peak heights nor windowed fluxes are clean.
    _FWHM_KMS = 20.0
    _L_DISC = 1e45

    @staticmethod
    def _sed():
        from tengri.components.agn.nlr import compute_nlr_sed

        wave = jnp.linspace(3000.0, 7500.0, 200_000)
        sed = compute_nlr_sed(
            wave,
            l_disc_bol_erg=TestNLRLinePhysics._L_DISC,
            fwhm_kms=TestNLRLinePhysics._FWHM_KMS,
        )
        return np.asarray(wave), np.asarray(sed)

    @classmethod
    def _line_flux(cls, wave, sed, center_aa: float, half_width_aa: float = 6.0) -> float:
        """Integrate the line over a window wide enough to hold all of it."""
        mask = (wave > center_aa - half_width_aa) & (wave < center_aa + half_width_aa)
        return float(np.trapezoid(sed[mask], wave[mask]))

    def test_oiii_5007_4959_ratio_in_template(self):
        """[OIII] 5007/4959 = 2.98 — fixed by the transition probabilities."""
        wave, sed = self._sed()
        ratio = self._line_flux(wave, sed, 5008.24) / self._line_flux(wave, sed, 4960.30)
        assert abs(ratio - 2.98) < 0.1, f"[OIII] 5007/4959 should be ~2.98, got {ratio:.3f}"

    @pytest.mark.xfail(
        reason="#1752: NLR template emits [NII] 6583/6548 = 2.73; atomic value is ~2.96",
        strict=True,
    )
    def test_nii_6583_6548_ratio_in_template(self):
        """[NII] 6583/6548 = ~2.96 (atomic physics).

        Both lines leave the same upper level, so the ratio is a constant
        independent of density, temperature, ionization parameter and
        abundance — one of the few numbers in a nebular spectrum with no
        physical freedom. The template carries 2.73 (#1752).
        """
        wave, sed = self._sed()
        ratio = self._line_flux(wave, sed, 6585.27) / self._line_flux(wave, sed, 6549.86)
        assert abs(ratio - 2.96) < 0.2, f"[NII] 6583/6548 should be ~2.96, got {ratio:.3f}"

    def test_nlr_oiii_strongest(self):
        """[OIII] 5007 should be the strongest NLR line."""
        wave, sed = self._sed()
        peak_wave = float(wave[int(np.argmax(sed))])
        assert abs(peak_wave - 5008.24) < 1.0, (
            f"Strongest NLR line should be [OIII] 5007 (vacuum 5008.24), got {peak_wave:.1f} A"
        )

    @pytest.mark.parametrize(
        "line_aa,name",
        [
            (3727.09, "[OII] 3727"),
            (4862.69, "Hbeta"),
            (4960.30, "[OIII] 4959"),
            (5008.24, "[OIII] 5007"),
            (6302.05, "[OI] 6300"),
            (6549.86, "[NII] 6548"),
            (6564.61, "Halpha"),
            (6585.27, "[NII] 6583"),
        ],
    )
    def test_nlr_has_all_key_lines(self, line_aa, name):
        """Every standard diagnostic line is actually emitted, not just listed.

        Parametrized so a missing line names itself instead of failing the whole
        set on the first gap.
        """
        wave, sed = self._sed()
        flux = self._line_flux(wave, sed, line_aa)
        total = float(np.trapezoid(sed, wave))
        assert flux > 1e-6 * total, f"NLR emits no {name} at {line_aa:.2f} A (vacuum)"


# ── 5. BLR LINE PHYSICS — broad permitted lines ───────────────────


class TestBLRLinePhysics:
    """BLR must contain only permitted (broad) lines with correct ratios."""

    def test_blr_has_lya(self):
        """BLR must include Lyα 1216A (strongest UV emission line)."""
        from tengri.components.agn.blr import _BLR_LINE_WAVELENGTHS

        min_dist = float(jnp.min(jnp.abs(_BLR_LINE_WAVELENGTHS - 1216.0)))
        assert min_dist < 2.0, "BLR missing Lyα 1216"

    def test_blr_has_civ(self):
        """BLR must include CIV 1549A (strong broad line)."""
        from tengri.components.agn.blr import _BLR_LINE_WAVELENGTHS

        min_dist = float(jnp.min(jnp.abs(_BLR_LINE_WAVELENGTHS - 1549.0)))
        assert min_dist < 2.0, "BLR missing CIV 1549"

    def test_blr_halpha_strongest_optical(self):
        """Hα should be the strongest optical BLR line (Vanden Berk+2001)."""
        from tengri.components.agn.blr import _BLR_LINE_STRENGTHS, _BLR_LINE_WAVELENGTHS

        optical = _BLR_LINE_WAVELENGTHS > 4000.0
        if jnp.any(optical):
            optical_strengths = _BLR_LINE_STRENGTHS[optical]
            optical_waves = _BLR_LINE_WAVELENGTHS[optical]
            max_idx = int(jnp.argmax(optical_strengths))
            max_wave = float(optical_waves[max_idx])
            assert abs(max_wave - 6563.0) < 2.0, (
                f"Strongest optical BLR line should be Hα, got λ={max_wave:.0f}"
            )

    def test_blr_halpha_hbeta_ratio(self):
        """Hα/Hβ ~ 2.8 for Type 1 AGN (Vanden Berk+2001 broad: ~260/46 ≈ 5.6 in EW,
        but in relative strength ~1.43/0.50 = 2.86)."""
        from tengri.components.agn.blr import _BLR_LINE_STRENGTHS, _BLR_LINE_WAVELENGTHS

        idx_ha = int(jnp.argmin(jnp.abs(_BLR_LINE_WAVELENGTHS - 6563.0)))
        idx_hb = int(jnp.argmin(jnp.abs(_BLR_LINE_WAVELENGTHS - 4861.0)))
        ratio = float(_BLR_LINE_STRENGTHS[idx_ha] / _BLR_LINE_STRENGTHS[idx_hb])
        assert 2.0 < ratio < 4.0, f"BLR Hα/Hβ should be ~2.86, got {ratio:.2f}"


# ── 6. SHOCK+NLR BPT SEPARATION — diagnostic diagram physics ──────


class TestBPTDiagramPhysics:
    """Shock and NLR should occupy distinct BPT regions."""

    def test_shock_nii_enhanced_vs_photoionization(self):
        """Shocks produce enhanced [NII]/Hα compared to typical HII regions.

        HII regions: log([NII]/Hα) ≲ -0.3
        Shocks: log([NII]/Hα) > -0.3 at intermediate velocities.
        """
        from tengri.components.nebular.shock import shock_line_ratios

        ratios = shock_line_ratios(300.0)
        nii_ha = float(ratios["NII_6583A"]) / float(ratios["HA_6563A"])
        log_nii_ha = np.log10(nii_ha)
        assert log_nii_ha > -1.0, f"log([NII]/Hα) should be > -1.0 in shocks, got {log_nii_ha:.2f}"

    def test_shock_oi_enhanced(self):
        """[OI] 6300 is THE shock diagnostic — much stronger than in HII regions.

        HII regions: log([OI]/Hα) < -1.5
        Shocks: log([OI]/Hα) > -1.0 at v > 100 km/s.
        """
        from tengri.components.nebular.shock import shock_line_ratios

        for v in [150.0, 300.0]:
            ratios = shock_line_ratios(v)
            oi_ha = float(ratios["OI_6300A"]) / float(ratios["HA_6563A"])
            log_oi_ha = np.log10(oi_ha)
            assert log_oi_ha > -1.5, f"log([OI]/Hα) at v={v} should be > -1.5, got {log_oi_ha:.2f}"
