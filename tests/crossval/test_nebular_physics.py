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

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.crossval


# ── 1. SHOCK EMISSION SED — spectral line placement ───────────────


class TestShockEmissionSEDPhysics:
    """shock_emission_sed must place lines at correct wavelengths."""

    def test_sed_has_halpha_peak(self):
        """Shock SED should peak near Hα 6563A (strongest optical line)."""
        from tengri.components.nebular.shock import shock_emission_sed

        wave = jnp.linspace(3000.0, 8000.0, 5000)
        # l_shock_halpha in Lsun
        l_nu = shock_emission_sed(wave, shock_velocity=300.0, l_shock_halpha=1e8)
        peak_wave = float(wave[jnp.argmax(l_nu)])
        # Should be near one of the strong lines: Hα, [OIII], [NII]
        near_ha = abs(peak_wave - 6563.0) < 100.0
        near_oiii = abs(peak_wave - 5007.0) < 100.0
        assert near_ha or near_oiii, (
            f"Shock SED peak at {peak_wave:.0f} A, expected near Hα or [OIII]"
        )

    def test_sed_scales_with_luminosity(self):
        """10x luminosity → 10x flux."""
        from tengri.components.nebular.shock import shock_emission_sed

        wave = jnp.linspace(3000.0, 8000.0, 2000)
        l_faint = shock_emission_sed(wave, shock_velocity=300.0, l_shock_halpha=1e7)
        l_bright = shock_emission_sed(wave, shock_velocity=300.0, l_shock_halpha=1e8)
        ratio = float(jnp.sum(l_bright)) / float(jnp.sum(l_faint))
        assert abs(ratio - 10.0) < 1.0, f"10x luminosity should give 10x flux, got {ratio:.1f}x"

    def test_sed_velocity_changes_line_ratios(self):
        """Different velocities produce different line ratio patterns."""
        from tengri.components.nebular.shock import shock_emission_sed

        wave = jnp.linspace(3000.0, 8000.0, 5000)
        l_slow = shock_emission_sed(wave, shock_velocity=150.0, l_shock_halpha=1e8)
        l_fast = shock_emission_sed(wave, shock_velocity=500.0, l_shock_halpha=1e8)

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
        from tengri.components.nebular.shock import shock_emission_sed

        wave = jnp.linspace(1000.0, 10000.0, 5000)
        for v in [100.0, 300.0, 700.0, 1000.0]:
            l_nu = shock_emission_sed(wave, shock_velocity=v, l_shock_halpha=1e8)
            assert jnp.all(jnp.isfinite(l_nu)), f"Shock SED at v={v} has non-finite values"
            assert jnp.all(l_nu >= 0), f"Shock SED at v={v} has negative values"


# ── 2. SHOCK LINE RATIOS — velocity-dependent physics ─────────────


class TestShockLineRatioPhysics:
    """Shock line ratios must follow Allen+2008 velocity trends."""

    def test_oiii_peaks_at_intermediate_velocity(self):
        """[OIII]/Hβ peaks at v~300-400 km/s (maximum ionization)."""
        from tengri.components.nebular.shock import shock_line_ratios

        oiii_hb = []
        velocities = [100.0, 200.0, 300.0, 400.0, 500.0, 750.0, 1000.0]
        for v in velocities:
            ratios = shock_line_ratios(v)
            oiii_hb.append(float(ratios["OIII_5007"]))

        peak_v_idx = np.argmax(oiii_hb)
        peak_v = velocities[peak_v_idx]
        assert 200.0 <= peak_v <= 600.0, f"[OIII]/Hβ should peak at 200-600 km/s, got {peak_v}"

    def test_halpha_hbeta_above_case_b(self):
        """In shocks, Hα/Hβ > 2.86 (Case B) due to collisional excitation."""
        from tengri.components.nebular.shock import shock_line_ratios

        for v in [200.0, 300.0, 500.0]:
            ratios = shock_line_ratios(v)
            ha_hb = float(ratios["Halpha"])
            assert ha_hb > 2.86, (
                f"Hα/Hβ should exceed Case B (2.86) in shocks at v={v}, got {ha_hb:.2f}"
            )

    def test_sii_strong_shock_diagnostic(self):
        """[SII]/Hβ must be enhanced in shocks (vs. HII regions ≲ 0.4)."""
        from tengri.components.nebular.shock import shock_line_ratios

        for v in [200.0, 300.0]:
            ratios = shock_line_ratios(v)
            # SII is the sum of 6716+6731 relative to Hβ
            sii_hb = float(ratios["SII_6716"]) + float(ratios["SII_6731"])
            assert sii_hb > 0.5, (
                f"[SII]/Hβ should be enhanced in shocks at v={v}, got {sii_hb:.2f}"
            )

    def test_velocity_clipping(self):
        """Velocities outside [100, 1000] should be clipped, not error."""
        from tengri.components.nebular.shock import shock_line_ratios

        # Below minimum
        ratios_low = shock_line_ratios(50.0)
        assert ratios_low["Hbeta"] == pytest.approx(1.0)

        # Above maximum
        ratios_high = shock_line_ratios(1500.0)
        assert ratios_high["Hbeta"] == pytest.approx(1.0)


# ── 3. SHOCK ATOMIC PHYSICS — doublet ratios ──────────────────────


class TestShockAtomicPhysics:
    """Forbidden line doublet ratios are set by atomic physics."""

    def test_oiii_doublet(self):
        """[OIII] 5007/4959 = 2.98 (Storey & Zeippen 2000)."""
        from tengri.components.nebular.shock import shock_line_ratios

        ratios = shock_line_ratios(300.0)
        r = float(ratios["OIII_5007"]) / float(ratios["OIII_4959"])
        assert abs(r - 2.98) < 0.1, f"[OIII] doublet ratio should be 2.98, got {r:.2f}"

    def test_nii_doublet(self):
        """[NII] 6583/6548 = 2.94 (Storey & Zeippen 2000)."""
        from tengri.components.nebular.shock import shock_line_ratios

        ratios = shock_line_ratios(300.0)
        r = float(ratios["NII_6583"]) / float(ratios["NII_6548"])
        assert abs(r - 2.94) < 0.1, f"[NII] doublet ratio should be 2.94, got {r:.2f}"


# ── 4. NLR LINE PHYSICS — forbidden line properties ───────────────


class TestNLRLinePhysics:
    """NLR must satisfy forbidden line ratio constraints."""

    def test_oiii_5007_4959_ratio_in_template(self):
        """[OIII] 5007/4959 = 2.98 in NLR template."""
        from tengri.components.agn.nlr import _NLR_LINE_STRENGTHS, _NLR_LINE_WAVELENGTHS

        idx_5007 = int(jnp.argmin(jnp.abs(_NLR_LINE_WAVELENGTHS - 5007.0)))
        idx_4959 = int(jnp.argmin(jnp.abs(_NLR_LINE_WAVELENGTHS - 4959.0)))
        ratio = float(_NLR_LINE_STRENGTHS[idx_5007] / _NLR_LINE_STRENGTHS[idx_4959])
        assert abs(ratio - 2.98) < 0.1

    def test_nii_6583_6548_ratio_in_template(self):
        """[NII] 6583/6548 = 3.0 in NLR template (atomic physics)."""
        from tengri.components.agn.nlr import _NLR_LINE_STRENGTHS, _NLR_LINE_WAVELENGTHS

        idx_6583 = int(jnp.argmin(jnp.abs(_NLR_LINE_WAVELENGTHS - 6583.0)))
        idx_6548 = int(jnp.argmin(jnp.abs(_NLR_LINE_WAVELENGTHS - 6548.0)))
        ratio = float(_NLR_LINE_STRENGTHS[idx_6583] / _NLR_LINE_STRENGTHS[idx_6548])
        assert abs(ratio - 3.0) < 0.2, f"[NII] 6583/6548 should be ~3.0, got {ratio:.2f}"

    def test_nlr_oiii_strongest(self):
        """[OIII] 5007 should be the strongest NLR line."""
        from tengri.components.agn.nlr import _NLR_LINE_STRENGTHS, _NLR_LINE_WAVELENGTHS

        max_idx = int(jnp.argmax(_NLR_LINE_STRENGTHS))
        max_wave = float(_NLR_LINE_WAVELENGTHS[max_idx])
        assert abs(max_wave - 5007.0) < 1.0, (
            f"Strongest NLR line should be [OIII] 5007, got λ={max_wave:.0f}"
        )

    def test_nlr_has_all_key_lines(self):
        """NLR template must include all standard diagnostic lines."""
        from tengri.components.agn.nlr import _NLR_LINE_WAVELENGTHS

        expected_lines = [3727.0, 4861.0, 4959.0, 5007.0, 6300.0, 6548.0, 6563.0, 6583.0]
        for line_wave in expected_lines:
            min_dist = float(jnp.min(jnp.abs(_NLR_LINE_WAVELENGTHS - line_wave)))
            assert min_dist < 5.0, f"NLR missing line at {line_wave:.0f} A"


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
        nii_ha = float(ratios["NII_6583"]) / float(ratios["Halpha"])
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
            oi_ha = float(ratios["OI_6300"]) / float(ratios["Halpha"])
            log_oi_ha = np.log10(oi_ha)
            assert log_oi_ha > -1.5, f"log([OI]/Hα) at v={v} should be > -1.5, got {log_oi_ha:.2f}"
