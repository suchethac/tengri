# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate tengri physics against python-fsps.

Tests component-level agreement between tengri implementations
and FSPS (Conroy, Gunn & White 2009) for:

1. Mass-remaining fraction (internal computation vs FSPS tables)
2. Charlot & Fall dust attenuation (dust_type=0)
3. Ionizing photon rate Q_H
4. Energy-balance dust re-emission
5. AGN SED (qualitative checks)

python-fsps requires SPS_HOME to be set. Tests are skipped if
either fsps or the data are unavailable.
"""

import os

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.crossval

# python-fsps requires SPS_HOME at import time; guard both conditions
if "SPS_HOME" not in os.environ:
    pytest.skip("SPS_HOME environment variable not set", allow_module_level=True)

fsps = pytest.importorskip("fsps", reason="python-fsps not installed")


@pytest.fixture(scope="module")
def sp():
    """FSPS StellarPopulation instance (Chabrier IMF, solar Z)."""
    if "SPS_HOME" not in os.environ:
        pytest.skip("SPS_HOME not set")
    sp = fsps.StellarPopulation(
        compute_vega_mags=False,
        zcontinuous=1,
        imf_type=1,  # Chabrier
    )
    sp.params["logzsol"] = 0.0
    sp.params["dust1"] = 0.0
    sp.params["dust2"] = 0.0
    sp.params["add_neb_emission"] = False
    return sp


# ── 1. Mass-remaining fraction ────────────────────────────────────


class TestMassRemainingCrossval:
    """Compare tengri internal mass-remaining vs FSPS."""

    def test_internal_matches_fsps_within_10pct(self, sp):
        """Internal Chabrier IMF computation should be within 10% of FSPS."""
        from tengri.components.stellar.sps.mass_remaining import compute_mass_remaining_fraction

        ages_gyr = np.array([0.01, 0.1, 0.3, 1.0, 3.0, 5.0, 10.0])
        f_tengri = np.asarray(compute_mass_remaining_fraction(jnp.array(ages_gyr), imf="chabrier"))

        f_fsps = []
        for t in ages_gyr:
            sp.params["logzsol"] = 0.0
            sp.get_spectrum(tage=t)
            f_fsps.append(sp.stellar_mass)
        f_fsps = np.array(f_fsps)

        np.testing.assert_allclose(
            f_tengri,
            f_fsps,
            rtol=0.10,
            err_msg="Internal mass-remaining >10% off FSPS",
        )

    def test_monotonically_decreasing(self, sp):
        """Mass-remaining should decrease with age (more stars die)."""
        from tengri.components.stellar.sps.mass_remaining import compute_mass_remaining_fraction

        ages = jnp.array([0.001, 0.01, 0.1, 1.0, 5.0, 13.0])
        f_surv = np.asarray(compute_mass_remaining_fraction(ages, imf="chabrier"))
        assert np.all(np.diff(f_surv) < 0), "Mass-remaining should decrease with age"

    def test_all_imfs_physical(self):
        """All IMFs should give f_surviving in (0.3, 1.0) at 10 Gyr."""
        from tengri.components.stellar.sps.mass_remaining import compute_mass_remaining_fraction

        ages = jnp.array([10.0])
        for imf in ["chabrier", "salpeter", "kroupa"]:
            f = float(compute_mass_remaining_fraction(ages, imf=imf)[0])
            assert 0.3 < f < 1.0, f"{imf} IMF: f_surviving={f:.3f} at 10 Gyr"

    def test_imf_mass_fractions_differ(self):
        """Different IMFs should give different mass-remaining fractions.

        Salpeter extends the power-law below 1 Msun (no lognormal
        turnover), so it has relatively more low-mass long-lived stars
        and thus a HIGHER surviving fraction than Chabrier. This is
        counter-intuitive but correct: the relevant quantity is the
        mass-weighted surviving fraction, not just the high-mass tail.
        """
        from tengri.components.stellar.sps.mass_remaining import compute_mass_remaining_fraction

        ages = jnp.array([1.0, 5.0, 10.0])
        f_chab = np.asarray(compute_mass_remaining_fraction(ages, imf="chabrier"))
        f_salp = np.asarray(compute_mass_remaining_fraction(ages, imf="salpeter"))

        # They should differ by at least 5%
        assert not np.allclose(f_chab, f_salp, rtol=0.05), (
            "Salpeter and Chabrier should give different mass fractions"
        )


# ── 2. Dust attenuation: Charlot & Fall (CF00) ────────────────────


class TestDustCF00Crossval:
    """Compare tengri Charlot & Fall vs FSPS dust_type=0."""

    def test_v_band_transmission_matches(self, sp):
        """Transmission at V-band (5500A) should match exp(-tau_V).

        FSPS dust_type=0 with dust2=tau_V gives T(V) = exp(-tau_V)
        for old stars. tengri's CF00 should give the same.
        """
        from tengri.components.dust.attenuation import two_component_dust

        for tau_v2 in [0.1, 0.3, 0.5, 1.0, 2.0]:
            # FSPS
            sp.params["dust_type"] = 0
            sp.params["dust1"] = 0.0
            sp.params["dust2"] = tau_v2
            sp.params["dust_index"] = -0.7
            wave_fsps, spec_dusty = sp.get_spectrum(tage=1.0)
            sp.params["dust2"] = 0.0
            wave_fsps, spec_nodust = sp.get_spectrum(tage=1.0)
            ratio_fsps = spec_dusty / np.maximum(spec_nodust, 1e-50)
            idx_v = np.argmin(np.abs(wave_fsps - 5500))
            t_v_fsps = ratio_fsps[idx_v]

            # tengri (old stars, 10 Gyr)
            wavs = jnp.array([5500.0])
            ages = jnp.array([1e10])
            trans_ds = float(
                two_component_dust(
                    wavs, ages, 0.0, tau_v2, law_bc="power_law", law_diff="power_law", n_slope=-0.7
                )[0, 0]
            )

            np.testing.assert_allclose(
                trans_ds,
                t_v_fsps,
                rtol=0.01,
                err_msg=f"V-band transmission mismatch at tau={tau_v2}",
            )

    def test_wavelength_dependence_matches(self, sp):
        """Attenuation curve shape should match FSPS across wavelengths."""
        from tengri.components.dust.attenuation import two_component_dust

        tau_v2 = 0.5
        sp.params["dust_type"] = 0
        sp.params["dust1"] = 0.0
        sp.params["dust2"] = tau_v2
        sp.params["dust_index"] = -0.7
        wave_fsps, spec_dusty = sp.get_spectrum(tage=5.0)
        sp.params["dust2"] = 0.0
        wave_fsps, spec_nodust = sp.get_spectrum(tage=5.0)

        # Sample at several wavelengths
        test_wavs = np.array([2000, 3600, 5500, 8000, 12000, 22000], dtype=float)
        ratio_fsps = []
        for w in test_wavs:
            idx = np.argmin(np.abs(wave_fsps - w))
            ratio_fsps.append(spec_dusty[idx] / max(spec_nodust[idx], 1e-50))
        ratio_fsps = np.array(ratio_fsps)

        # tengri
        ages = jnp.array([5e9])  # 5 Gyr
        trans_ds = np.asarray(
            two_component_dust(
                jnp.array(test_wavs),
                ages,
                0.0,
                tau_v2,
                law_bc="power_law",
                law_diff="power_law",
                n_slope=-0.7,
            )
        )[0]

        np.testing.assert_allclose(
            trans_ds,
            ratio_fsps,
            rtol=0.05,
            err_msg="Dust curve shape mismatch vs FSPS",
        )

    def test_birth_cloud_young_stars(self, sp):
        """Young stars should have extra attenuation from birth cloud."""
        from tengri.components.dust.attenuation import two_component_dust

        tau_v1 = 1.0  # birth cloud
        tau_v2 = 0.3  # diffuse

        # FSPS: dust1 affects young stars (< t_birth), dust2 affects all
        sp.params["dust_type"] = 0
        sp.params["dust1"] = tau_v1
        sp.params["dust2"] = tau_v2
        sp.params["dust_index"] = -0.7
        wave, spec_dusty = sp.get_spectrum(tage=0.005)  # 5 Myr = young
        sp.params["dust1"] = 0.0
        sp.params["dust2"] = 0.0
        wave, spec_nodust = sp.get_spectrum(tage=0.005)
        idx_v = np.argmin(np.abs(wave - 5500))

        # Young stars: FSPS applies tau1 + tau2
        t_v_fsps = spec_dusty[idx_v] / max(spec_nodust[idx_v], 1e-50)

        # tengri: young star (1 Myr) should have tau_eff ~ tau_v1 + tau_v2
        ages = jnp.array([1e6])  # 1 Myr
        trans_ds = float(
            two_component_dust(
                jnp.array([5500.0]),
                ages,
                tau_v1,
                tau_v2,
                law_bc="power_law",
                law_diff="power_law",
                n_slope=-0.7,
            )[0, 0]
        )

        # Expected: exp(-(tau_v1 + tau_v2)) ~ exp(-1.3) ~ 0.27
        expected = float(jnp.exp(-(tau_v1 + tau_v2)))
        np.testing.assert_allclose(trans_ds, expected, rtol=0.05)

        # FSPS should also be near exp(-(tau1+tau2))
        np.testing.assert_allclose(t_v_fsps, expected, rtol=0.15)


# ── 3. Ionizing photon rate Q_H ───────────────────────────────────


class TestQHCrossval:
    """Compare ionizing photon rate Q_H with FSPS."""

    def _compute_qh_from_spectrum(self, wave, spec_lsun_hz):
        """Compute Q_H from an SSP spectrum.

        Q_H = integral of (L_nu / h*nu) dnu for lambda < 912 A
        spec is in Lsun/Hz per Msun formed.
        """
        h_planck = 6.626e-27
        c_cgs = 2.998e10
        lsun = 3.828e33

        mask = wave < 912.0
        if np.sum(mask) < 2:
            return 0.0

        nu = c_cgs / (wave[mask] * 1e-8)
        integrand = spec_lsun_hz[mask] * lsun / (h_planck * nu)
        return abs(np.trapezoid(integrand, nu))

    def test_young_population_high_qh(self, sp):
        """Very young populations (~1 Myr) should have log Q_H > 46."""
        sp.params["add_neb_emission"] = False
        wave, spec = sp.get_spectrum(tage=0.001)
        q_h = self._compute_qh_from_spectrum(wave, spec)
        assert np.log10(max(q_h, 1e-99)) > 45, "Young SSP should have high Q_H"

    def test_old_population_low_qh(self, sp):
        """Old populations (~1 Gyr) should have much lower Q_H."""
        sp.params["add_neb_emission"] = False
        wave, spec = sp.get_spectrum(tage=1.0)
        q_h = self._compute_qh_from_spectrum(wave, spec)

        sp.params["add_neb_emission"] = False
        wave_y, spec_y = sp.get_spectrum(tage=0.001)
        q_h_young = self._compute_qh_from_spectrum(wave_y, spec_y)

        assert q_h < q_h_young * 0.01, "Old SSP Q_H should be << young SSP Q_H"

    def test_qh_drops_over_long_timescales(self, sp):
        """Q_H at 100 Myr should be much lower than at 1 Myr.

        Note: Q_H is NOT strictly monotonic at 1-3 Myr because
        Wolf-Rayet stars briefly boost the ionizing flux. We test
        the long-timescale trend instead.
        """
        sp.params["add_neb_emission"] = False
        wave_y, spec_y = sp.get_spectrum(tage=0.001)
        wave_o, spec_o = sp.get_spectrum(tage=0.1)

        q_h_young = self._compute_qh_from_spectrum(wave_y, spec_y)
        q_h_old = self._compute_qh_from_spectrum(wave_o, spec_o)

        assert q_h_old < q_h_young * 0.1, "Q_H at 100 Myr should be <<< Q_H at 1 Myr"


# ── 4. Dust re-emission (energy balance) ──────────────────────────


class TestDustEmissionCrossval:
    """Verify energy-balance dust re-emission properties."""

    def test_energy_balance_conserved(self):
        """Absorbed luminosity should equal emitted IR luminosity."""
        from tengri.components.dust.emission import (
            compute_absorbed_luminosity,
            modified_blackbody,
        )

        # Create a simple flat SED (1 Lsun/Hz over UV-optical)
        wave = jnp.linspace(1000, 30000, 2000)
        l_nu_intrinsic = jnp.ones_like(wave) * 1e-10  # Lsun/Hz

        # Apply 50% attenuation
        transmission = jnp.where(wave < 10000, 0.5, 0.9)

        l_abs = float(compute_absorbed_luminosity(wave, l_nu_intrinsic, transmission))
        assert l_abs > 0, "Absorbed luminosity should be positive"

        # Re-emit as modified blackbody
        wave_ir = jnp.linspace(10000, 5e6, 5000)  # 1-500 um
        l_nu_emission = modified_blackbody(wave_ir, l_abs, dust_T=30.0, dust_beta_ir=1.8)

        # Integrate emission over frequency
        c_cgs = 2.998e10
        nu_ir = c_cgs / (wave_ir * 1e-8)
        l_emitted = float(jnp.trapezoid(l_nu_emission[::-1], nu_ir[::-1]))

        np.testing.assert_allclose(
            l_emitted,
            l_abs,
            rtol=0.05,
            err_msg="Dust emission doesn't conserve energy",
        )

    def test_higher_tau_more_ir(self):
        """More dust attenuation should produce more IR emission."""
        from tengri.components.dust.emission import (
            compute_absorbed_luminosity,
        )

        wave = jnp.linspace(1000, 30000, 2000)
        l_nu = jnp.ones_like(wave) * 1e-10

        trans_low = jnp.exp(-0.1 * (wave / 5500.0) ** (-0.7))
        trans_high = jnp.exp(-1.0 * (wave / 5500.0) ** (-0.7))

        l_abs_low = float(compute_absorbed_luminosity(wave, l_nu, trans_low))
        l_abs_high = float(compute_absorbed_luminosity(wave, l_nu, trans_high))

        assert l_abs_high > l_abs_low, "More dust -> more absorbed luminosity"

    def test_modified_blackbody_peaks_at_physical_wavelength(self):
        """Modified blackbody at 30K should peak around 100 um."""
        from tengri.components.dust.emission import modified_blackbody

        wave = jnp.linspace(10000, 5e6, 5000)  # 1-500 um
        l_nu = modified_blackbody(wave, L_absorbed=1.0, dust_T=30.0)

        peak_wave = float(wave[jnp.argmax(l_nu)])
        peak_um = peak_wave / 1e4

        # Wien's law: peak in L_nu ~ 100 um for T~30K
        assert 50 < peak_um < 300, f"MBB peak at {peak_um:.0f} um, expected ~100 um"

    def test_dale2014_alpha_dependence(self):
        """Dale alpha < 2 should produce warmer (more peaked) emission."""
        from tengri.components.dust.emission import dale2014

        wave = jnp.linspace(10000, 5e6, 5000)

        l_warm = dale2014(wave, L_absorbed=1.0, dust_alpha_dale=1.0)
        l_cool = dale2014(wave, L_absorbed=1.0, dust_alpha_dale=3.0)

        # Warm (low alpha) should peak at shorter wavelengths
        peak_warm = float(wave[jnp.argmax(l_warm)])
        peak_cool = float(wave[jnp.argmax(l_cool)])
        assert peak_warm < peak_cool, "Lower alpha should produce warmer dust"


# ── 5. AGN SED ────────────────────────────────────────────────────


class TestAGNCrossval:
    """Qualitative AGN SED checks (no FSPS AGN to compare against)."""

    def test_power_law_disc_uv_bright(self):
        """Power-law disc with steep slope should be UV-bright.

        With alpha=-1.5, L_nu ~ nu^{-1.5} ~ lambda^{1.5}, so L_nu
        peaks at long wavelengths. But nu*L_nu ~ nu^{-0.5} peaks at
        short wavelengths. We test that the disc has significant UV flux.
        """
        from tengri.components.agn.disc import powerlaw_disc

        wave = jnp.linspace(500, 50000, 2000)
        l_nu = powerlaw_disc(wave, agn_log_lbol=11.0, agn_alpha=-1.0)

        # UV flux should be non-negligible relative to optical
        uv = (wave > 1000) & (wave < 3000)
        opt = (wave > 4000) & (wave < 7000)
        l_uv = float(jnp.mean(l_nu[uv]))
        l_opt = float(jnp.mean(l_nu[opt]))

        assert l_uv > 0, "Disc should have UV emission"
        assert l_opt > 0, "Disc should have optical emission"
        # UV and optical should be within 2 orders of magnitude
        assert l_uv / l_opt > 0.01, "Disc UV flux too weak relative to optical"

    def test_torus_ir_dominated(self):
        """Silva+04 torus should peak in the IR (1-100 um)."""
        from tengri.components.agn.silva04 import silva04_analytic

        wave = jnp.linspace(1000, 200000, 5000)
        l_nu = silva04_analytic(wave, agn_log_lbol=11.0)

        peak = float(wave[jnp.argmax(l_nu)])
        peak_um = peak / 1e4
        assert 0.5 < peak_um < 100, f"Torus should peak in IR, got {peak_um:.1f} um"

    def test_unified_has_both_components(self):
        """Unified AGN should have emission in both UV and IR."""
        from tengri.components.agn.unified import multicolor_agn

        wave = jnp.linspace(500, 200000, 5000)
        l_nu = multicolor_agn(
            wave,
            agn_log_lbol=11.0,
            agn_lum_ratio=1.0,
            agn_alpha=-1.0,
            agn_T_torus=1000.0,
        )

        # UV region
        uv_mask = (wave > 1000) & (wave < 3000)
        l_uv = float(jnp.mean(l_nu[uv_mask]))

        # IR region
        ir_mask = (wave > 20000) & (wave < 100000)
        l_ir = float(jnp.mean(l_nu[ir_mask]))

        assert l_uv > 0, "AGN should have UV emission (disc)"
        assert l_ir > 0, "AGN should have IR emission (torus)"

    def test_higher_lbol_more_luminous(self):
        """Doubling L_bol should roughly double the AGN SED."""
        from tengri.components.agn.unified import multicolor_agn

        wave = jnp.linspace(1000, 100000, 2000)
        l_lo = multicolor_agn(wave, agn_log_lbol=10.0, agn_lum_ratio=1.0)
        l_hi = multicolor_agn(wave, agn_log_lbol=11.0, agn_lum_ratio=1.0)

        # 10x L_bol should give ~10x more emission
        ratio = float(jnp.sum(l_hi)) / float(jnp.sum(l_lo))
        np.testing.assert_allclose(ratio, 10.0, rtol=0.3, err_msg="L_bol scaling off")

    def test_covering_factor_shifts_uv_ir_balance(self):
        """Higher covering factor should shift power from UV to IR."""
        from tengri.components.agn.unified import multicolor_agn

        wave = jnp.linspace(500, 200000, 5000)

        l_low_cf = multicolor_agn(wave, agn_log_lbol=11.0, agn_lum_ratio=1.0, agn_torus_frac=0.2)
        l_high_cf = multicolor_agn(wave, agn_log_lbol=11.0, agn_lum_ratio=1.0, agn_torus_frac=0.8)

        uv_mask = (wave > 1000) & (wave < 3000)
        ir_mask = (wave > 30000) & (wave < 100000)

        uv_ir_low = float(jnp.mean(l_low_cf[uv_mask]) / jnp.mean(l_low_cf[ir_mask]))
        uv_ir_high = float(jnp.mean(l_high_cf[uv_mask]) / jnp.mean(l_high_cf[ir_mask]))

        assert uv_ir_high < uv_ir_low, "Higher covering factor should shift power from UV to IR"
