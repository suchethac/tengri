# SPDX-License-Identifier: BSD-3-Clause
"""Surviving stellar mass uses each node's own metallicity, not single-Z interpolation.

Issue #2613: surviving_mass should use Σ_age Σ_Z w(age, Z) · m_rem(age, Z)
instead of interpolating the mass-remaining table at the youngest stars'
metallicity for the whole population. The defect causes 2–4% errors on
Z-dependent grids (BAGPIPES and FSPS both use exact per-node evaluation).
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri import DEFAULT, Fixed, SEDModel, SSPData, load_ssp_data

pytestmark = pytest.mark.conservation


@pytest.fixture
def ssp_with_z_dependent_mass_remaining(synthetic_ssp_wide):
    """Build a Z-dependent mass-remaining table for testing.

    The test-grid's default mass_remaining is None. Here we construct a
    synthetic one that varies with Z (to trigger the defect) using a simple
    physically-motivated model: older ages and higher Z have higher survival
    fractions (less mass loss in metal-rich/older stars).

    The table: m_rem(age, Z) = f_age(age) · (1 + 0.1·(lgmet − mean_lgmet))
    clipped to [0.3, 1].
    """
    base = synthetic_ssp_wide
    n_met = base.ssp_lgmet.shape[0]
    n_age = base.ssp_lg_age_gyr.shape[0]

    # Age function: declines from 0.95 (young) to 0.55 (old)
    f_age = jnp.linspace(0.95, 0.55, n_age)

    # Metallicity modulation: varies linearly with Z
    # At each Z, multiply the age profile by (1 + 0.1 * (lgmet - mean))
    lgmet_centered = base.ssp_lgmet - base.ssp_lgmet.mean()
    z_factor = 1.0 + 0.1 * lgmet_centered  # shape (n_met,)

    # Build table: (n_met, n_age)
    table = z_factor[:, None] * f_age[None, :]
    table = jnp.clip(table, 0.3, 1.0)

    return SSPData(
        ssp_wave=base.ssp_wave,
        ssp_flux=base.ssp_flux,
        ssp_lg_age_gyr=base.ssp_lg_age_gyr,
        ssp_lgmet=base.ssp_lgmet,
        ssp_mass_remaining=table,
    )


class TestSurvivingMassUsesEachNodesMetallicity:
    """Surviving mass = Σ_age Σ_Z w(age, Z) · m_rem(age, Z), exact sum.

    Every metallicity history mode (delta, two_step, bins, etc.) should compute
    the exact joint-weight contraction, not interpolate the table at a single Z.
    """

    @pytest.mark.parametrize(
        "met_mode",
        [
            "delta_z",
            ("two_step", {"logzsol_old": -0.5, "logzsol_young": 0.2, "step_age_gyr": 1.0}),
            (
                "two_step_reversed",
                {"logzsol_old": 0.2, "logzsol_young": -0.5, "step_age_gyr": 1.0},
            ),
            "bins",
            "bins_continuity",
        ],
    )
    def test_surviving_mass_equals_joint_weight_contraction(
        self, ssp_with_z_dependent_mass_remaining, met_mode
    ):
        """10**log_mstar_surviving must equal formed · Σ(joint_weights · m_rem).

        Tests that the surviving mass uses all joint weights, not just a
        single Z's interpolation. On a Z-dependent table, this catches the
        defect (#2613) with 2–4% error.

        Parameters
        ----------
        met_mode : str or tuple
            If str: "delta_z" uses solar.
            If tuple: ("mode_name", {params_dict}) specifies mode and params.
        """
        ssp = ssp_with_z_dependent_mass_remaining
        n_met = ssp.ssp_lgmet.shape[0]
        n_age = ssp.ssp_lg_age_gyr.shape[0]

        # Parse met_mode
        if isinstance(met_mode, str):
            if met_mode == "delta_z":
                met_spec = {"type": "delta", "logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)}
            elif met_mode == "bins":
                # Explicit six-bin ladder with distinct per-bin metallicities
                met_spec = {
                    "type": "bins",
                    "met_bin_0": Fixed(-0.7),
                    "met_bin_1": Fixed(-0.5),
                    "met_bin_2": Fixed(-0.2),
                    "met_bin_3": Fixed(0.0),
                    "met_bin_4": Fixed(0.2),
                    "met_bin_5": Fixed(0.3),
                }
            elif met_mode == "bins_continuity":
                # Explicit bins_continuity with distinct metallicities
                met_spec = {
                    "type": "bins_continuity",
                    "met_logzsol_base": Fixed(-0.7),
                    "met_d_log_z_0": Fixed(0.2),
                    "met_d_log_z_1": Fixed(0.2),
                    "met_d_log_z_2": Fixed(0.2),
                    "met_d_log_z_3": Fixed(0.1),
                    "met_d_log_z_4": Fixed(0.1),
                }
            else:
                raise ValueError(f"Unknown mode: {met_mode}")
        else:
            mode_name, params = met_mode
            if mode_name in ("two_step", "two_step_reversed"):
                met_spec = {
                    "type": "two_step",
                    "logzsol_old": Fixed(params["logzsol_old"]),
                    "logzsol_young": Fixed(params["logzsol_young"]),
                    "step_age_gyr": Fixed(params["step_age_gyr"]),
                    "all_params": Fixed(DEFAULT),
                }
            else:
                raise ValueError(f"Unknown mode: {mode_name}")

        # Build model with delayed-tau SFH, all parameters fixed to defaults
        model = SEDModel.build(
            ssp_data=ssp,
            observation=None,
            sfh={
                "type": "delayed",
                "tau_gyr": Fixed(3.0),
                "age_gyr": Fixed(12.0),
                "log_total_mass": Fixed(10.0),
                "all_params": Fixed(DEFAULT),
            },
            met=met_spec,
            redshift=Fixed(0.0),
        )

        # Predict
        params = {}  # No free params
        state = model.predict_state(params)

        # Extract derived quantities
        log_mstar_formed = state.derived["log_mstar_formed"]
        log_mstar_surviving = state.derived["log_mstar_surviving"]
        joint_weights = state.derived["joint_weights"]  # shape (n_met, n_age)

        # For bins/bins_continuity: verify the mass-weighted metallicity differs between ages
        if met_mode in ("bins", "bins_continuity"):
            # Per-age mass-weighted metallicity
            jw_per_age = jnp.sum(joint_weights, axis=0)  # sum over Z for each age
            z_weights = joint_weights / jnp.sum(joint_weights)  # normalized (n_met, n_age)
            age_z_means = jnp.sum(z_weights * ssp.ssp_lgmet[:, None], axis=0)  # (n_age,)
            z_spread = jnp.ptp(age_z_means)  # peak-to-peak
            assert float(z_spread) > 0.3, (
                f"For {met_mode}: mass-weighted Z spread {float(z_spread):.4f} dex "
                f"must exceed 0.3 dex to test Z-dependence"
            )

        # Normalize joint weights to sum to 1
        jw_norm = joint_weights / jnp.sum(joint_weights)

        # Compute exact surviving mass: Σ_age Σ_Z w(age, Z) · m_rem(age, Z)
        exact_surv_frac = float(jnp.sum(jw_norm * ssp.ssp_mass_remaining))

        # Published surviving mass
        published_surv = 10.0 ** float(log_mstar_surviving)
        formed = 10.0 ** float(log_mstar_formed)

        # The test: published surviving mass should equal exact contraction
        # Allow some tolerance for floating-point rounding
        expected_surviving = formed * exact_surv_frac
        assert_allclose(
            published_surv,
            expected_surviving,
            rtol=1e-6,
            err_msg=(
                f"Surviving mass {published_surv:.6e} != "
                f"exact joint-weight sum {expected_surviving:.6e} "
                f"(ratio {published_surv / expected_surviving - 1:+.4%})"
            ),
        )

    def test_surviving_mass_is_order_dependent(self, ssp_with_z_dependent_mass_remaining):
        """Two metallicity histories with reversed order give different surviving masses.

        For a two_step history with metal-poor old stars vs metal-rich old stars,
        the surviving mass differs because the table m_rem(age, Z) varies with Z.
        This test verifies that both histories are evaluated correctly (at their
        respective node Z, not at a single "young" Z), by checking that they give
        different results and that the difference follows the table's Z-dependence.
        """
        ssp = ssp_with_z_dependent_mass_remaining
        formed_mass = 1e10

        # Build two models: metal-poor old / metal-rich young, and reversed
        def run_two_step(logzsol_old, logzsol_young):
            model = SEDModel.build(
                ssp_data=ssp,
                observation=None,
                sfh={
                    "type": "delayed",
                    "tau_gyr": Fixed(3.0),
                    "age_gyr": Fixed(12.0),
                    "log_total_mass": Fixed(np.log10(formed_mass)),
                    "all_params": Fixed(DEFAULT),
                },
                met={
                    "type": "two_step",
                    "logzsol_old": Fixed(logzsol_old),
                    "logzsol_young": Fixed(logzsol_young),
                    "step_age_gyr": Fixed(1.0),
                    "all_params": Fixed(DEFAULT),
                },
                redshift=Fixed(0.0),
            )
            state = model.predict_state({})
            return 10.0 ** float(state.derived["log_mstar_surviving"])

        # Test case: old poor, young rich
        surv_poor_old = run_two_step(logzsol_old=-0.5, logzsol_young=0.2)
        # Reversed: old rich, young poor
        surv_rich_old = run_two_step(logzsol_old=0.2, logzsol_young=-0.5)

        # The synthetic table must have Z-dependence by construction
        table_range = ssp.ssp_mass_remaining.ptp(axis=0)  # variation per age
        table_z_spread = float(table_range.max())
        assert table_z_spread > 0.0, "Synthetic table must be Z-dependent for this test"

        # They should be different (assuming the mass table varies with Z)
        assert surv_poor_old != surv_rich_old, (
            "Two reversed two_step histories should give different surviving masses "
            "when the ssp_mass_remaining table depends on Z"
        )

        # At older ages (higher indices), the table is lower (0.55 range).
        # Higher Z → higher m_rem, so rich-old should have more survivors.
        assert surv_rich_old > surv_poor_old, (
            f"With higher-Z old stars, surviving mass {surv_rich_old:.3e} "
            f"should exceed lower-Z case {surv_poor_old:.3e}, "
            f"matching the table's Z-dependence"
        )

    def test_surviving_mass_on_tracked_grid_fsps(self):
        """Surviving mass on tracked Z-dependent grid equals joint-weight contraction.

        Use the shipped fsps_prsc_miles_chabrier.h5 grid which has a Z-dependent
        mass_remaining table. This validates the fix on real data, not just
        a synthetic table.
        """
        # Build path relative to test file (like tests/integration/test_derived_quantities.py)
        grid_path = Path(__file__).resolve().parents[3] / "data" / "fsps_prsc_miles_chabrier.h5"
        if not grid_path.is_file():
            pytest.skip(f"Tracked SSP grid not found at {grid_path}")

        ssp = load_ssp_data(str(grid_path))

        # Build model with two_step SFH, all parameters fixed to defaults
        model = SEDModel.build(
            ssp_data=ssp,
            observation=None,
            sfh={
                "type": "delayed",
                "tau_gyr": Fixed(3.0),
                "age_gyr": Fixed(12.0),
                "log_total_mass": Fixed(10.0),
                "all_params": Fixed(DEFAULT),
            },
            met={
                "type": "two_step",
                "logzsol_old": Fixed(-0.5),
                "logzsol_young": Fixed(0.2),
                "step_age_gyr": Fixed(1.0),
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.0),
        )

        # Predict
        params = {}
        state = model.predict_state(params)

        # Extract derived quantities
        log_mstar_formed = state.derived["log_mstar_formed"]
        log_mstar_surviving = state.derived["log_mstar_surviving"]
        joint_weights = state.derived["joint_weights"]

        # Normalize joint weights to sum to 1
        jw_norm = joint_weights / jnp.sum(joint_weights)

        # The table the loader resolved for this registered grid (the companion, #2751).
        # It is NOT the grid file's embedded ssp_mass_remaining: that table differs from the
        # companion by up to 0.132, so contracting over it moved the ratio from 0 to +1.489%.
        table = np.asarray(ssp.ssp_mass_remaining)
        assert ssp.mass_remaining_source.startswith("companion:")

        # Compute exact surviving mass: Σ_age Σ_Z w(age, Z) · m_rem(age, Z)
        exact_surv_frac = float(jnp.sum(jw_norm * table))

        # Published surviving mass
        published_surv = 10.0 ** float(log_mstar_surviving)
        formed = 10.0 ** float(log_mstar_formed)

        # The test: published surviving mass should equal exact contraction
        expected_surviving = formed * exact_surv_frac
        assert_allclose(
            published_surv,
            expected_surviving,
            rtol=1e-6,
            err_msg=(
                f"Surviving mass {published_surv:.6e} != "
                f"exact joint-weight sum {expected_surviving:.6e} "
                f"(ratio {published_surv / expected_surviving - 1:+.4%})"
            ),
        )
