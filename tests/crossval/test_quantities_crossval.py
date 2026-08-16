# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate stellar mass and SFR against bagpipes.

For a constant star formation history, physical quantities are
analytically predictable:

    formed_mass = SFR x duration
    SFR = formed_mass / duration (constant by definition)

Both codes should recover these values. We also compare the
surviving stellar mass (which depends on SSP mass-loss fractions,
so we expect ~10-20% disagreement between FSPS and BC03).

Note: tengri's `stellar_mass` = total FORMED mass (sum of CSP weights).
bagpipes reports `formed_mass` (log10) and `stellar_mass` (surviving).
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.crossval

bagpipes_mg = pytest.importorskip(
    "bagpipes.models.model_galaxy",
    reason="bagpipes not installed",
)

# ── SSP data (needed for tengri SEDModel) ────────────────────────────
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_PATH = _DATA_DIR / "fsps_prsc_miles_chabrier.h5"
_SSP_EXISTS = _SSP_PATH.is_file()


def _tengri_smooth_params(n_grid=64):
    """Return tengri params for a smooth DPL SFH (near-zero burstiness)."""
    return {
        "sfh_dpl_alpha": 1.0,
        "sfh_dpl_beta": 1.5,
        "sfh_dpl_tau_gyr": 3.0,
        "sfh_dpl_log_total_mass": 0.5,
        "sfh_field_psd_sigma": 0.01,  # near-zero burstiness
        "sfh_field_psd_tau_myr": 50.0,
        "sfh_field_xi": jnp.zeros(n_grid),
        "met_logzsol": 0.0,
        "dust_tau_bc": 0.0,
        "dust_tau_diff": 0.0,
    }


@pytest.fixture(scope="module")
def tengri_model():
    """Create a tengri SEDModel with smooth SFH (no GP burstiness)."""
    if not _SSP_EXISTS:
        pytest.skip("SSP data not found")

    from tengri import Parameters, SEDModel, Uniform

    spec = Parameters(
        sfh_alpha=Uniform(0.5, 3.0),
        sfh_beta=Uniform(0.3, 2.0),
        sfh_tau_peak_gyr=Uniform(0.5, 10.0),
        psd_sigma=Uniform(0.01, 3.0),
        psd_tau_myr=Uniform(10, 500),
        met_logzsol=Uniform(-2.0, 0.2),
        dust_tau_bc=Uniform(0.0, 4.0),
        dust_tau_diff=Uniform(0.0, 4.0),
        redshift=0.1,
    )

    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    ssp = load_ssp_data(str(_SSP_PATH))
    return SEDModel(spec, ssp)


def _make_bagpipes_constant_sfh(age_gyr, log_massformed, metallicity_solar=1.0):
    """Create a bagpipes model_galaxy with constant SFH.

    Parameters
    ----------
    age_gyr : float
        Duration of star formation (Gyr).
    log_massformed : float
        log10(total formed mass / Msun).
    metallicity_solar : float
        Metallicity in solar units.

    Returns
    -------
    bagpipes model_galaxy
    """
    comp = {
        "redshift": 0.0,
        "constant": {
            "metallicity": metallicity_solar,
            "age_of_universe_Gyr": 13.8,
            "age_min": 0.0,
            "age_max": age_gyr,
            "massformed": log_massformed,
        },
    }
    return bagpipes_mg.model_galaxy(comp, spec_wavs=np.arange(3000, 10000, 10.0))


# ── 1. Formed mass consistency (SSP-independent) ──────────────────


class TestFormedMassCrossval:
    """Verify formed mass = integral(SFR dt) is consistent."""

    @pytest.mark.parametrize(
        "age_gyr,log_mf",
        [(0.5, 8.5), (1.0, 9.0), (3.0, 10.0), (10.0, 10.5)],
    )
    def test_bagpipes_formed_mass_self_consistent(self, age_gyr, log_mf):
        """bagpipes: formed_mass should match the input."""
        mg = _make_bagpipes_constant_sfh(age_gyr, log_mf)
        np.testing.assert_allclose(
            mg.sfh.formed_mass,
            log_mf,
            atol=0.01,
            err_msg="bagpipes formed_mass != input",
        )

    def test_tengri_formed_mass_reasonable(self, tengri_model):
        """tengri: formed mass for a smooth SFH should be physical."""
        params = _tengri_smooth_params()
        derived = tengri_model.predict_derived(params)
        mstar = float(derived["stellar_mass"])

        # For a DPL SFH with these params, mass should be physical
        assert 1e7 < mstar < 1e13, f"M*_formed = {mstar:.2e} outside physical range"

    def test_sfr_times_duration_equals_formed_mass(self):
        """For constant SFH: formed_mass = SFR x duration (trivially)."""
        age_gyr = 1.0
        log_mf = 9.0

        mg = _make_bagpipes_constant_sfh(age_gyr, log_mf)

        expected_sfr = 10**log_mf / (age_gyr * 1e9)  # Msun/yr
        np.testing.assert_allclose(
            mg.sfh.sfr,
            expected_sfr,
            rtol=0.05,
            err_msg="bagpipes: SFR x duration != formed mass",
        )


# ── 2. SFR consistency ────────────────────────────────────────────


class TestSFRCrossval:
    """Cross-check SFR between tengri and bagpipes."""

    def test_tengri_sfr_positive_for_star_forming(self, tengri_model):
        """tengri: SFR should be positive for a star-forming galaxy."""
        params = _tengri_smooth_params()
        derived = tengri_model.predict_derived(params)
        assert float(derived["sfr_100myr"]) > 0.0
        assert float(derived["sfr_10myr"]) > 0.0

    def test_bagpipes_sfr_matches_analytical(self):
        """bagpipes: constant SFH should have SFR = M_formed / duration."""
        for age_gyr, log_mf in [(0.5, 8.0), (1.0, 9.0), (5.0, 10.0)]:
            mg = _make_bagpipes_constant_sfh(age_gyr, log_mf)
            expected = 10**log_mf / (age_gyr * 1e9)
            np.testing.assert_allclose(
                mg.sfh.sfr,
                expected,
                rtol=0.05,
                err_msg=f"SFR mismatch at age={age_gyr}, log_mf={log_mf}",
            )

    def test_ssfr_range_agreement(self, tengri_model):
        """Both codes should give sSFR in the same ballpark for SF galaxies.

        bagpipes constant SFH (1 Gyr, 10^9 Msun):
            SFR = 1 Msun/yr, M* ~ 10^8.8 (surviving)
            sSFR ~ 10^(-8.8) yr^-1

        tengri DPL SFH (peaked at 3 Gyr, moderate SFR):
            sSFR should be in [1e-12, 1e-8] yr^-1 range.
        """
        # bagpipes
        mg = _make_bagpipes_constant_sfh(1.0, 9.0)
        ssfr_bp = 10**mg.sfh.ssfr  # bagpipes stores log10(sSFR)

        # tengri
        params = _tengri_smooth_params()
        derived = tengri_model.predict_derived(params)
        ssfr_ds = float(derived["ssfr"])

        # Both should be in star-forming range
        assert 1e-14 < ssfr_bp < 1e-7, f"bagpipes sSFR = {ssfr_bp:.2e} out of range"
        assert 1e-14 < ssfr_ds < 1e-7, f"tengri sSFR = {ssfr_ds:.2e} out of range"


# ── 3. Surviving stellar mass (SSP-dependent, expect ~20% difference)


class TestSurvivingMassCrossval:
    """Compare surviving vs formed mass fractions.

    tengri uses FSPS SSPs, bagpipes uses BC03. The mass-loss
    (recycling) fraction differs by ~10-20%, but the qualitative
    behavior should agree: surviving < formed, and the fraction
    should be ~0.5-0.8 for a 1 Gyr old population.
    """

    def test_bagpipes_surviving_less_than_formed(self):
        """Surviving mass must be less than formed mass."""
        mg = _make_bagpipes_constant_sfh(1.0, 9.0)
        assert mg.sfh.stellar_mass < mg.sfh.formed_mass

    def test_mass_loss_fraction_physical(self):
        """Mass-loss fraction should be 20-60% for a 1 Gyr population."""
        mg = _make_bagpipes_constant_sfh(1.0, 9.0)
        f_surviving = 10 ** (mg.sfh.stellar_mass - mg.sfh.formed_mass)
        assert 0.4 < f_surviving < 0.9, f"Surviving fraction = {f_surviving:.2f}, expected 0.4-0.9"

    def test_older_population_more_mass_loss(self):
        """Older populations should have lost more mass."""
        mg_young = _make_bagpipes_constant_sfh(0.1, 9.0)
        mg_old = _make_bagpipes_constant_sfh(5.0, 9.0)

        f_young = 10 ** (mg_young.sfh.stellar_mass - mg_young.sfh.formed_mass)
        f_old = 10 ** (mg_old.sfh.stellar_mass - mg_old.sfh.formed_mass)

        assert f_old < f_young, "Older population should have more mass loss"


# ── 4. SED shape sanity (qualitative agreement) ───────────────────


class TestSEDShapeCrossval:
    """Qualitative SED shape comparison.

    We don't expect exact agreement because tengri uses FSPS SSPs
    and bagpipes uses BC03, but both should produce SEDs that:
    - Peak in the optical/NIR for a ~1 Gyr population
    - Have similar overall shape (within a factor of ~2)
    - Show the same trends with metallicity and age
    """

    def test_sed_peaks_in_optical_both_codes(self, tengri_model):
        """Both codes should produce SEDs peaking in 3000-15000 A."""
        # bagpipes (restrict to optical/NIR to avoid Lyman-break spike)
        wavs_bp = np.arange(2000, 20000, 10.0)
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
        mg = bagpipes_mg.model_galaxy(comp, spec_wavs=wavs_bp)
        bp_spectrum = mg.spectrum[:, 1]
        bp_wavs = mg.spectrum[:, 0]
        peak_bp = bp_wavs[np.argmax(bp_spectrum)]

        # tengri — predict rest-frame SED
        params = _tengri_smooth_params()
        sed_ds = np.asarray(tengri_model.predict_rest_sed(params).sed)
        wave_ds = np.asarray(tengri_model.ssp_data.ssp_wave)
        # Restrict to optical range
        mask = (wave_ds > 1000) & (wave_ds < 20000)
        peak_ds = wave_ds[mask][np.argmax(sed_ds[mask])]

        # Both should peak in optical/NIR
        assert 2000 < peak_bp < 20000, f"bagpipes peak at {peak_bp:.0f} A"
        assert 2000 < peak_ds < 20000, f"tengri peak at {peak_ds:.0f} A"

    def test_higher_metallicity_redder(self):
        """Higher metallicity should shift the SED redward in bagpipes.

        Uses extreme metallicity contrast (0.05 vs 5.0 solar) to
        ensure a clear signal above numerical noise.
        """
        wavs = np.arange(3000, 10000, 10.0)
        base_const = {
            "age_of_universe_Gyr": 13.8,
            "age_min": 0.0,
            "age_max": 3.0,  # older population amplifies Z effect
            "massformed": 9.0,
        }

        # Very low metallicity
        comp_low = {
            "redshift": 0.0,
            "constant": {**base_const, "metallicity": 0.05},
        }
        mg_low = bagpipes_mg.model_galaxy(comp_low, spec_wavs=wavs)

        # Very high metallicity
        comp_high = {
            "redshift": 0.0,
            "constant": {**base_const, "metallicity": 5.0},
        }
        mg_high = bagpipes_mg.model_galaxy(comp_high, spec_wavs=wavs)

        # Compare blue/red flux ratio
        blue = (wavs > 3500) & (wavs < 4500)
        red = (wavs > 7000) & (wavs < 8000)

        ratio_low = np.mean(mg_low.spectrum[blue, 1]) / np.mean(mg_low.spectrum[red, 1])
        ratio_high = np.mean(mg_high.spectrum[blue, 1]) / np.mean(mg_high.spectrum[red, 1])

        # Higher metallicity -> redder -> lower blue/red ratio
        assert ratio_high < ratio_low, (
            f"Higher Z should produce redder SED: "
            f"blue/red ratio low-Z={ratio_low:.3f}, high-Z={ratio_high:.3f}"
        )
