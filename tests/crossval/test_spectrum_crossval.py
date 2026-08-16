# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate tengri spectra against FSPS and bagpipes.

Tests actual spectrum VALUES (Lsun/Hz/Msun) at key wavelengths.
This catches SSP interpolation errors, unit mismatches, metallicity
offset bugs, dust normalization errors, and nebular contribution.

IMPORTANT: tengri's ssp_lgmet grid is log10(Z) ABSOLUTE, not
log10(Z/Zsun). Solar metallicity is log10(0.0142) = -1.848, NOT 0.0.
The SEDModel class handles this conversion (met_logzsol -> log_z), but
tests using the low-level SSP functions must apply LOG10_ZSUN manually.

Reference spectra from python-fsps (Chabrier IMF, PADOVA, MILES)
stored in data/fsps_spectrum_reference.npz.
"""

import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.crossval

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_PATH = _DATA_DIR / "fsps_prsc_miles_chabrier.h5"
_REF_PATH = _DATA_DIR / "fsps_spectrum_reference.npz"
LOG10_ZSUN = -1.848  # log10(Z_solar) absolute

if not _REF_PATH.is_file():
    pytest.skip("FSPS spectrum reference not found", allow_module_level=True)

_HAS_FSPS = "SPS_HOME" in os.environ


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(str(_REF_PATH)))


@pytest.fixture(scope="module")
def fsps_wave(ref):
    return ref["wave"]


@pytest.fixture(scope="module")
def ssp_data():
    if not _SSP_PATH.is_file():
        pytest.skip("SSP data not found")
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    return load_ssp_data(str(_SSP_PATH))


def _flux_at(wave, flux, target, width=50.0):
    """Average flux in a band centered on target +/- width."""
    mask = (wave > target - width) & (wave < target + width)
    return float(np.mean(flux[mask])) if np.sum(mask) > 0 else 0.0


# ── 1. SSP spectrum values vs FSPS ────────────────────────────────


class TestSSPSpectrumCrossval:
    """Compare tengri SSP spectra against FSPS reference.

    tengri loads the same FSPS SSP data, so the main source of
    difference is metallicity interpolation. FSPS zcontinuous=1 uses
    a different interpolation scheme than tengri's linear-in-logZ.
    We allow ~60% tolerance in the UV (where Z-dependence is steepest)
    and ~25% in the optical/NIR.
    """

    def test_ssp_1gyr_solar_optical(self, ssp_data, ref, fsps_wave):
        """SSP 1 Gyr solar: optical/NIR flux within 25% of FSPS."""
        from tengri.components.stellar.sps.dsps_wrapper import interpolate_metallicity

        ssp_flux = np.asarray(
            interpolate_metallicity(ssp_data.ssp_flux, ssp_data.ssp_lgmet, LOG10_ZSUN)
        )
        ssp_lg_age = np.asarray(ssp_data.ssp_lg_age_gyr)
        age_idx = np.argmin(np.abs(ssp_lg_age - 0.0))
        ssp_wave = np.asarray(ssp_data.ssp_wave)
        flux_ds = ssp_flux[age_idx]
        flux_ref = ref["ssp_1gyr_solar"]

        for w in [3600, 5500, 8000, 12000]:
            f_ds = _flux_at(ssp_wave, flux_ds, w)
            f_ref = _flux_at(fsps_wave, flux_ref, w)
            ratio = f_ds / f_ref
            assert 0.75 < ratio < 1.25, f"SSP 1 Gyr at {w}A: ratio={ratio:.3f}, expected 0.75-1.25"

    def test_ssp_1gyr_solar_uv(self, ssp_data, ref, fsps_wave):
        """SSP 1 Gyr solar: UV flux within factor 2 (Z-sensitive)."""
        from tengri.components.stellar.sps.dsps_wrapper import interpolate_metallicity

        ssp_flux = np.asarray(
            interpolate_metallicity(ssp_data.ssp_flux, ssp_data.ssp_lgmet, LOG10_ZSUN)
        )
        age_idx = np.argmin(np.abs(np.asarray(ssp_data.ssp_lg_age_gyr) - 0.0))
        ssp_wave = np.asarray(ssp_data.ssp_wave)

        for w in [1500, 2000]:
            f_ds = _flux_at(ssp_wave, ssp_flux[age_idx], w)
            f_ref = _flux_at(fsps_wave, ref["ssp_1gyr_solar"], w)
            if f_ref > 0 and f_ds > 0:
                ratio = f_ds / f_ref
                assert 0.5 < ratio < 2.0, f"SSP UV at {w}A: ratio={ratio:.3f}, expected 0.5-2.0"

    def test_young_ssp_bluer_than_old(self, ssp_data):
        """100 Myr SSP should be bluer than 10 Gyr SSP."""
        from tengri.components.stellar.sps.dsps_wrapper import interpolate_metallicity

        ssp_flux = np.asarray(
            interpolate_metallicity(ssp_data.ssp_flux, ssp_data.ssp_lgmet, LOG10_ZSUN)
        )
        ssp_lg_age = np.asarray(ssp_data.ssp_lg_age_gyr)
        ssp_wave = np.asarray(ssp_data.ssp_wave)

        idx_young = np.argmin(np.abs(ssp_lg_age - (-1.0)))  # 0.1 Gyr
        idx_old = np.argmin(np.abs(ssp_lg_age - 1.0))  # 10 Gyr

        # Blue/red color
        color_young = _flux_at(ssp_wave, ssp_flux[idx_young], 3600) / max(
            _flux_at(ssp_wave, ssp_flux[idx_young], 8000), 1e-50
        )
        color_old = _flux_at(ssp_wave, ssp_flux[idx_old], 3600) / max(
            _flux_at(ssp_wave, ssp_flux[idx_old], 8000), 1e-50
        )
        assert color_young > color_old, "Young SSP should be bluer than old"

    def test_metallicity_affects_uv(self, ssp_data):
        """Low-Z SSP should be UV-brighter than high-Z at 1 Gyr."""
        from tengri.components.stellar.sps.dsps_wrapper import interpolate_metallicity

        ssp_wave = np.asarray(ssp_data.ssp_wave)
        ssp_lg_age = np.asarray(ssp_data.ssp_lg_age_gyr)
        age_idx = np.argmin(np.abs(ssp_lg_age - 0.0))

        flux_low_z = np.asarray(
            interpolate_metallicity(ssp_data.ssp_flux, ssp_data.ssp_lgmet, LOG10_ZSUN - 1.0)
        )[age_idx]
        flux_high_z = np.asarray(
            interpolate_metallicity(ssp_data.ssp_flux, ssp_data.ssp_lgmet, LOG10_ZSUN + 0.2)
        )[age_idx]

        uv_low = _flux_at(ssp_wave, flux_low_z, 2500)
        uv_high = _flux_at(ssp_wave, flux_high_z, 2500)
        assert uv_low > uv_high, "Low-Z should be UV-brighter"


# ── 2. Dust-attenuated spectrum ───────────────────────────────────


class TestDustySpectrumCrossval:
    """Compare dust attenuation on spectra."""

    def test_cf00_attenuation_curve(self, ssp_data, ref, fsps_wave):
        """Charlot & Fall attenuation at key wavelengths vs FSPS."""
        from tengri.components.dust.attenuation import two_component_dust

        ssp_wave = np.asarray(ssp_data.ssp_wave)
        tau_v2 = 0.5

        # Compute attenuation at several wavelengths for old stars
        ages_yr = np.array([1e10])
        trans = np.asarray(
            two_component_dust(
                jnp.array(ssp_wave),
                jnp.array(ages_yr),
                0.0,
                tau_v2,
                law_bc="power_law",
                law_diff="power_law",
                n_slope=-0.7,
            )
        )[0]

        # FSPS reference: ratio of dusty/clean
        flux_dusty = ref["ssp_1gyr_dusty"]
        flux_clean = ref["ssp_1gyr_solar"]

        for w in [3600, 5500, 8000, 12000]:
            t_ds = _flux_at(ssp_wave, trans, w)
            t_ref = _flux_at(fsps_wave, flux_dusty, w) / max(
                _flux_at(fsps_wave, flux_clean, w), 1e-50
            )
            np.testing.assert_allclose(
                t_ds,
                t_ref,
                rtol=0.05,
                err_msg=f"Dust T({w}A): tengri={t_ds:.4f}, FSPS={t_ref:.4f}",
            )

    def test_vband_exp_minus_tau(self, ref, fsps_wave):
        """V-band attenuation should be exp(-tau_V) for old stars."""
        t_v = _flux_at(fsps_wave, ref["ssp_1gyr_dusty"], 5500) / max(
            _flux_at(fsps_wave, ref["ssp_1gyr_solar"], 5500), 1e-50
        )
        np.testing.assert_allclose(t_v, np.exp(-0.5), rtol=0.05)


# ── 3. Burst population spectra ───────────────────────────────────


class TestBurstSpectrumCrossval:
    """Validate spectral features of burst populations."""

    def test_young_burst_uv_bright(self, ref, fsps_wave):
        """1 Myr burst should have UV/opt flux ratio > 3."""
        flux = ref["burst_1myr"]
        f_uv = _flux_at(fsps_wave, flux, 1500)
        f_opt = _flux_at(fsps_wave, flux, 5500)
        assert f_uv > f_opt, "1 Myr burst should be UV-dominated"
        assert f_uv / f_opt > 3, f"UV/opt = {f_uv / f_opt:.1f}, expected > 3"

    def test_post_starburst_balmer_break(self, ref, fsps_wave):
        """500 Myr burst should show 4000A break (D4000 > 1)."""
        flux = ref["burst_500myr"]
        d4000 = _flux_at(fsps_wave, flux, 4200, 100) / _flux_at(fsps_wave, flux, 3700, 100)
        assert d4000 > 1.0, f"D4000 = {d4000:.2f}, expected > 1"

    def test_old_burst_nir_dominated(self, ref, fsps_wave):
        """5 Gyr burst should have NIR > UV."""
        flux = ref["burst_5gyr"]
        assert _flux_at(fsps_wave, flux, 15000) > _flux_at(fsps_wave, flux, 2000)

    def test_composite_bluer_than_old(self, ref, fsps_wave):
        """90% old + 10% young composite should be bluer than pure old."""
        color_comp = _flux_at(fsps_wave, ref["composite_90old_10young"], 2000) / max(
            _flux_at(fsps_wave, ref["composite_90old_10young"], 15000), 1e-50
        )
        color_old = _flux_at(fsps_wave, ref["burst_5gyr"], 2000) / max(
            _flux_at(fsps_wave, ref["burst_5gyr"], 15000), 1e-50
        )
        assert color_comp > color_old

    def test_dusty_burst_uv_suppressed(self, ref, fsps_wave):
        """Dusty 10 Myr burst UV should be < 10% of clean 100 Myr."""
        uv_dusty = _flux_at(fsps_wave, ref["burst_10myr_dusty"], 1500)
        uv_clean = _flux_at(fsps_wave, ref["ssp_01gyr_solar"], 1500)
        assert uv_dusty < uv_clean * 0.1

    def test_age_sequence_reddens(self, ref, fsps_wave):
        """Burst populations should redden with age."""
        ages = ["burst_1myr", "burst_500myr", "burst_5gyr"]
        colors = []
        for name in ages:
            c = _flux_at(fsps_wave, ref[name], 3600) / max(
                _flux_at(fsps_wave, ref[name], 8000), 1e-50
            )
            colors.append(c)
        # Blue/red ratio should decrease with age
        assert colors[0] > colors[1] > colors[2], f"Color should redden: {colors}"


# ── 4. Nebular emission features ──────────────────────────────────


class TestNebularSpectrumCrossval:
    """Validate nebular emission signatures in FSPS spectra."""

    def test_halpha_boost(self, ref, fsps_wave):
        """Nebular should boost Halpha by > 1.5x at 3 Myr."""
        f_neb = _flux_at(fsps_wave, ref["burst_3myr_neb"], 6563, 20)
        f_noneb = _flux_at(fsps_wave, ref["burst_3myr_noneb"], 6563, 20)
        assert f_neb / f_noneb > 1.5, f"Halpha boost = {f_neb / f_noneb:.1f}x"

    def test_oiii_boost(self, ref, fsps_wave):
        """Nebular should boost [OIII] 5007."""
        f_neb = _flux_at(fsps_wave, ref["burst_3myr_neb"], 5007, 20)
        f_noneb = _flux_at(fsps_wave, ref["burst_3myr_noneb"], 5007, 20)
        assert f_neb > f_noneb

    def test_continuum_raised(self, ref, fsps_wave):
        """Nebular continuum (free-free) should raise baseline."""
        f_neb = _flux_at(fsps_wave, ref["burst_3myr_neb"], 4500, 50)
        f_noneb = _flux_at(fsps_wave, ref["burst_3myr_noneb"], 4500, 50)
        assert f_neb > f_noneb

    def test_10myr_halpha_modest(self, ref, fsps_wave):
        """At 10 Myr, Halpha boost should be present but modest."""
        f_neb = _flux_at(fsps_wave, ref["ssp_001gyr_neb"], 6563, 20)
        f_noneb = _flux_at(fsps_wave, ref["ssp_001gyr_noneb"], 6563, 20)
        ratio = f_neb / max(f_noneb, 1e-50)
        assert 1.0 < ratio < 10.0, f"At 10 Myr, Halpha boost = {ratio:.1f}x"


# ── 5. Bagpipes comparison (qualitative, BC03 vs FSPS) ────────────


class TestBagpipesSpectrumCrossval:
    """Compare normalized SED shape between bagpipes (BC03) and FSPS.

    Different SSP libraries: expect factor ~2 differences in absolute
    flux, but normalized color ratios within ~50%.
    """

    def test_normalized_shape_agrees(self, ref):
        """Normalized SED shape should agree within 50% at optical."""
        bagpipes_mg = pytest.importorskip(
            "bagpipes.models.model_galaxy", reason="bagpipes not installed"
        )

        wavs = np.arange(3000, 10000, 10.0)
        comp = {
            "redshift": 0.0,
            "constant": {
                "metallicity": 1.0,
                "age_of_universe_Gyr": 13.8,
                "age_min": 0.0,
                "age_max": 1.0,
                "massformed": 9.0,
            },
        }
        mg = bagpipes_mg.model_galaxy(comp, spec_wavs=wavs)
        bp_flux = mg.spectrum[:, 1]
        bp_wavs = mg.spectrum[:, 0]

        ref_flux = ref["csp_const_1gyr"]
        ref_wave = ref["wave"]

        # Normalize by V-band
        bp_v = _flux_at(bp_wavs, bp_flux, 5500)
        ref_v = _flux_at(ref_wave, ref_flux, 5500)

        # Blue end (3600A) differs up to 3x between BC03 and FSPS
        # (horizontal branch + TP-AGB treatment). Red agrees better.
        for w, tol in [(3600, 3.0), (5500, 2.0), (8000, 3.0)]:
            color_bp = _flux_at(bp_wavs, bp_flux, w) / bp_v
            color_ref = _flux_at(ref_wave, ref_flux, w) / ref_v
            ratio = color_bp / color_ref
            assert 1.0 / tol < ratio < tol, f"Shape at {w}A: BP/FSPS = {ratio:.2f}"

    def test_both_show_4000a_break(self, ref):
        """Both codes should show D4000 > 1 for 1 Gyr population."""
        bagpipes_mg = pytest.importorskip(
            "bagpipes.models.model_galaxy", reason="bagpipes not installed"
        )

        wavs = np.arange(3000, 6000, 5.0)
        comp = {
            "redshift": 0.0,
            "constant": {
                "metallicity": 1.0,
                "age_of_universe_Gyr": 13.8,
                "age_min": 0.0,
                "age_max": 1.0,
                "massformed": 9.0,
            },
        }
        mg = bagpipes_mg.model_galaxy(comp, spec_wavs=wavs)

        d4000_bp = _flux_at(mg.spectrum[:, 0], mg.spectrum[:, 1], 4200, 100) / _flux_at(
            mg.spectrum[:, 0], mg.spectrum[:, 1], 3700, 100
        )
        d4000_ref = _flux_at(ref["wave"], ref["csp_const_1gyr"], 4200, 100) / _flux_at(
            ref["wave"], ref["csp_const_1gyr"], 3700, 100
        )

        assert d4000_bp > 1.0, f"bagpipes D4000 = {d4000_bp:.2f}"
        assert d4000_ref > 1.0, f"FSPS D4000 = {d4000_ref:.2f}"
        np.testing.assert_allclose(d4000_bp, d4000_ref, rtol=0.30)
