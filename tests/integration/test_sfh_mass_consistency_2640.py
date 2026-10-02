# SPDX-License-Identifier: BSD-3-Clause
"""Test that the SFR history integrates to the declared formed mass (#2640).

Verifies that ∫ sfr_history d(lookback), restricted to the support
``[0, age(z)]``, equals ``10**log_mstar_formed`` for the composite
``dpl + field`` SFH whose onset exceeds ``age(z)`` -- the literal #2640
reproducer. The published SFR history must carry the same mass-conserving
rescale as the age weights so the integral matches the formed mass.

The published history is bounded to ``[0, age(z)]`` (it has a node exactly at
``age(z)``) and carries the factor that makes its own plain trapezoid equal the
formed mass, so the identity is tested on the published arrays with no mask.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Parameters, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.cosmology import age_at_z
from tengri.parameters.priors import Fixed

# ── Paths to real SSP data ────────────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_NO_NEB = _DATA_DIR / "fsps_prsc_miles_chabrier.h5"

_SSP_FILES_EXIST = _SSP_NO_NEB.is_file()
pytestmark = pytest.mark.integration
_skip_no_ssp = pytest.mark.skipif(
    not _SSP_FILES_EXIST,
    reason="SSP data files not found in data/",
)

# The issue's own reproducer: age(z=0.1) = 12.4417 Gyr < 13.81 Gyr, so the
# dpl's onset genuinely exceeds the support: the history must be windowed to
# [0, age(z)] before it is normalized to the formed mass.
_AGE_GYR = 13.81
_Z = 0.1


def _trapz_inside_support(sfr, lbt_yr, age_z_yr):
    """Plain integral of the published history, after checking its support.

    The history has a node exactly at age(z) and every older node collapses
    onto it: no node lies beyond age(z) (to the grid's float32 resolution) and
    the trapezoid area over cells starting at age(z) is exactly zero.
    """
    sfr = np.asarray(sfr)
    lbt = np.asarray(lbt_yr)
    assert lbt.max() <= age_z_yr * (1.0 + 1e-6)
    beyond = lbt[:-1] >= age_z_yr * (1.0 - 1e-6)
    area_beyond = np.sum((0.5 * (sfr[1:] + sfr[:-1]) * np.diff(lbt))[beyond])
    assert area_beyond == 0.0
    return float(np.trapezoid(sfr, lbt))


@pytest.fixture(scope="module")
def ssp():
    return load_ssp_data(str(_SSP_NO_NEB))


@pytest.fixture(scope="module")
def spec_dpl_field():
    """DPL + field composite whose onset exceeds age(z=0.1) (#2640 reproducer)."""
    return Parameters(
        mean_sfh_type=["dpl", "field"],
        sfh_dpl_age_gyr=Fixed(_AGE_GYR),
        n_grid=256,
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
        sfh_dpl_log_total_mass=Uniform(9.0, 12.0),
        sfh_field_psd_sigma=Uniform(0.01, 3.0),
        sfh_field_psd_tau_myr=Uniform(10.0, 500.0),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        met_logzsol=Fixed(0.0),
        redshift=Fixed(_Z),
    )


@pytest.fixture(scope="module")
def model_dpl_field(ssp, spec_dpl_field):
    return SEDModel(spec_dpl_field, ssp)


@_skip_no_ssp
class TestSFRHistoryFormationMass:
    """Verify sfr_history integral = declared formed mass (#2640)."""

    def test_zero_field_integral_matches_formed_mass(self, model_dpl_field, spec_dpl_field):
        """∫ sfr_history (support) = 10**log_mstar_formed for zero field."""
        n_grid = spec_dpl_field.n_grid
        declared_mass = 5.0e10
        age_z_yr = float(age_at_z(_Z)) * 1e9

        params = {
            "sfh_dpl_alpha": 1.0,
            "sfh_dpl_beta": 1.5,
            "sfh_dpl_tau_gyr": 3.0,
            "sfh_dpl_log_total_mass": np.log10(declared_mass),
            "sfh_field_psd_sigma": 2.0,
            "sfh_field_psd_tau_myr": 50.0,
            "sfh_field_xi": jnp.zeros(n_grid),
        }

        st = model_dpl_field.predict_state(params)
        lbt = np.asarray(st.derived["sfh_grid_lbt_yr"])
        sfr = np.asarray(st.derived["sfr_history"])
        formed_mass = 10 ** float(st.derived["log_mstar_formed"])

        trapz_sfr = _trapz_inside_support(sfr, lbt, age_z_yr)

        np.testing.assert_allclose(
            trapz_sfr,
            formed_mass,
            rtol=1e-6,
            err_msg=(
                f"SFR history integral does not match formed mass (#2640): "
                f"∫ sfr_history (support) = {trapz_sfr:.6e}, formed = {formed_mass:.6e}"
            ),
        )

    def test_zero_field_integral_with_varying_mass(self, model_dpl_field, spec_dpl_field):
        """∫ sfr_history (support) = 10**log_mstar_formed for varying declared masses."""
        n_grid = spec_dpl_field.n_grid
        age_z_yr = float(age_at_z(_Z)) * 1e9

        for declared_mass in [1.0e10, 5.0e10, 1.0e11]:
            params = {
                "sfh_dpl_alpha": 1.0,
                "sfh_dpl_beta": 1.5,
                "sfh_dpl_tau_gyr": 3.0,
                "sfh_dpl_log_total_mass": np.log10(declared_mass),
                "sfh_field_psd_sigma": 0.01,  # Near-zero to avoid per-draw mass variation
                "sfh_field_psd_tau_myr": 50.0,
                "sfh_field_xi": jnp.zeros(n_grid),
            }

            st = model_dpl_field.predict_state(params)
            lbt = np.asarray(st.derived["sfh_grid_lbt_yr"])
            sfr = np.asarray(st.derived["sfr_history"])
            formed_mass = 10 ** float(st.derived["log_mstar_formed"])

            trapz_sfr = _trapz_inside_support(sfr, lbt, age_z_yr)

            np.testing.assert_allclose(
                trapz_sfr,
                formed_mass,
                rtol=1e-6,
                err_msg=(
                    f"SFR history integral mismatch at declared mass {declared_mass:.2e}: "
                    f"∫ sfr_history (support) = {trapz_sfr:.6e}, formed = {formed_mass:.6e}"
                ),
            )

    def test_random_field_integral_matches_formed_mass(self, model_dpl_field, spec_dpl_field):
        """∫ sfr_history (support) = 10**log_mstar_formed for a bursty field draw.

        The GP field modulates the published history multiplicatively on top
        of the shape's own normalization, so the rescale factor must be
        measured on the published grid itself: a factor measured on a
        different grid leaves the integral off by a factor of 2.55 here.
        """
        n_grid = spec_dpl_field.n_grid
        age_z_yr = float(age_at_z(_Z)) * 1e9
        key = jax.random.PRNGKey(42)
        xi = jax.random.normal(key, shape=(n_grid,))

        params = {
            "sfh_dpl_alpha": 1.0,
            "sfh_dpl_beta": 1.5,
            "sfh_dpl_tau_gyr": 3.0,
            "sfh_dpl_log_total_mass": np.log10(5.0e10),
            "sfh_field_psd_sigma": 2.0,
            "sfh_field_psd_tau_myr": 50.0,
            "sfh_field_xi": xi,
        }

        st = model_dpl_field.predict_state(params)
        lbt = np.asarray(st.derived["sfh_grid_lbt_yr"])
        sfr = np.asarray(st.derived["sfr_history"])
        formed_mass = 10 ** float(st.derived["log_mstar_formed"])

        trapz_sfr = _trapz_inside_support(sfr, lbt, age_z_yr)

        np.testing.assert_allclose(
            trapz_sfr,
            formed_mass,
            rtol=1e-5,
            err_msg=(
                f"SFR history integral does not match formed mass for a bursty "
                f"field (#2640): ∫ sfr_history (support) = {trapz_sfr:.6e}, "
                f"formed = {formed_mass:.6e}, ratio = {trapz_sfr / formed_mass:.6f}"
            ),
        )

    def test_gp_field_changes_sfr_history_not_formed_mass(self, model_dpl_field, spec_dpl_field):
        """The field modulates SFR history but preserves declared formed mass."""
        n_grid = spec_dpl_field.n_grid
        key = jax.random.PRNGKey(42)
        xi = jax.random.normal(key, shape=(n_grid,))

        params = {
            "sfh_dpl_alpha": 1.0,
            "sfh_dpl_beta": 1.5,
            "sfh_dpl_tau_gyr": 3.0,
            "sfh_dpl_log_total_mass": np.log10(5.0e10),
            "sfh_field_psd_sigma": 2.0,
            "sfh_field_psd_tau_myr": 50.0,
            "sfh_field_xi": xi,
        }

        # Compare to zero-xi (smooth) version
        params_smooth = {**params, "sfh_field_xi": jnp.zeros(n_grid)}

        st_bursty = model_dpl_field.predict_state(params)
        st_smooth = model_dpl_field.predict_state(params_smooth)

        # Both should form the same declared mass (no change to formed mass)
        formed_bursty = 10 ** float(st_bursty.derived["log_mstar_formed"])
        formed_smooth = 10 ** float(st_smooth.derived["log_mstar_formed"])
        np.testing.assert_allclose(
            formed_bursty,
            formed_smooth,
            rtol=1e-12,
            err_msg=(
                f"Field should not change declared mass: "
                f"bursty formed = {formed_bursty:.4e}, "
                f"smooth formed = {formed_smooth:.4e}"
            ),
        )

        # But the SFR histories should differ (field modulates the shape)
        sfr_bursty = np.asarray(st_bursty.derived["sfr_history"])
        sfr_smooth = np.asarray(st_smooth.derived["sfr_history"])
        assert not np.allclose(sfr_bursty, sfr_smooth, atol=0), (
            "Field should modulate the SFR history shape"
        )

    def test_sfr_averages_use_rescaled_history(self, model_dpl_field, spec_dpl_field):
        """sfr_100myr/sfr_10myr use the rescaled published history (independent recomputation)."""
        from tengri.components.stellar.sfh.sfr_window import time_weighted_sfr

        n_grid = spec_dpl_field.n_grid
        declared_mass = 5.0e10

        params = {
            "sfh_dpl_alpha": 1.0,
            "sfh_dpl_beta": 1.5,
            "sfh_dpl_tau_gyr": 3.0,
            "sfh_dpl_log_total_mass": np.log10(declared_mass),
            "sfh_field_psd_sigma": 2.0,
            "sfh_field_psd_tau_myr": 50.0,
            "sfh_field_xi": jnp.zeros(n_grid),
        }

        st = model_dpl_field.predict_state(params)
        lbt = jnp.asarray(st.derived["sfh_grid_lbt_yr"])
        sfr = jnp.asarray(st.derived["sfr_history"])

        for window_yr, key in [(1e7, "sfr_10myr"), (1e8, "sfr_100myr")]:
            recomputed = float(time_weighted_sfr(sfr, lbt, window_yr))
            direct = float(st.derived[key])
            np.testing.assert_allclose(
                recomputed,
                direct,
                rtol=1e-10,
                err_msg=(
                    f"{key} does not match an independent recomputation from the "
                    f"published (rescaled) history: direct = {direct:.6e}, "
                    f"recomputed = {recomputed:.6e}"
                ),
            )
