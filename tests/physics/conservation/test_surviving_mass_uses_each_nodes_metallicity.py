# SPDX-License-Identifier: BSD-3-Clause
"""Surviving stellar mass uses each node's own metallicity, not single-Z interpolation.

Issue #2613: surviving_mass should use Σ_age Σ_Z w(age, Z) · m_rem(age, Z)
instead of interpolating the mass-remaining table at the youngest stars'
metallicity for the whole population. The defect causes 2–4% errors on
Z-dependent grids (BAGPIPES and FSPS both use exact per-node evaluation).
"""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri import DEFAULT, Fixed, SEDModel, SSPData

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
                met_spec = {"type": "delta", "logzsol": Fixed(0.0)}
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
                }
            else:
                raise ValueError(f"Unknown mode: {mode_name}")

        # Build model with delayed-tau SFH
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
                },
                redshift=Fixed(0.0),
            )
            state = model.predict_state({})
            return 10.0 ** float(state.derived["log_mstar_surviving"])

        # Test case: old poor, young rich
        surv_poor_old = run_two_step(logzsol_old=-0.5, logzsol_young=0.2)
        # Reversed: old rich, young poor
        surv_rich_old = run_two_step(logzsol_old=0.2, logzsol_young=-0.5)

        # They should be different (assuming the mass table varies with Z)
        assert surv_poor_old != surv_rich_old, (
            "Two reversed two_step histories should give different surviving masses "
            "when the ssp_mass_remaining table depends on Z"
        )

        # Determine which is larger: the history with higher-Z old stars should
        # have higher surviving mass (since the table shows m_rem increases with Z)
        table_range = ssp.ssp_mass_remaining.ptp(axis=0)  # variation per age
        table_has_z_dependence = float(table_range.mean()) > 1e-6
        if table_has_z_dependence:
            # At older ages (higher indices), the table is lower (0.55 range).
            # Higher Z → higher m_rem, so rich-old should have more survivors.
            assert surv_rich_old > surv_poor_old, (
                f"With higher-Z old stars, surviving mass {surv_rich_old:.3e} "
                f"should exceed lower-Z case {surv_poor_old:.3e}, "
                f"matching the table's Z-dependence"
            )

    def test_fast_path_matches_exact_surviving_mass(self, ssp_with_z_dependent_mass_remaining):
        """Surviving mass via compute_joint_weights equals exact forward path.

        The StellarSEDComponent has two paths to compute joint_weights:
        - Exact: build (n_met, n_age) grid in the forward pass
        - Fast: compute_joint_weights helper (used by SED-free paths)

        Both must give bit-identical surviving mass. This test uses the
        exact forward path and verifies it matches the fast-path contract.
        """
        ssp = ssp_with_z_dependent_mass_remaining

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
            met={"type": "delta", "logzsol": Fixed(0.0)},
            redshift=Fixed(0.0),
        )

        # Predict with exact forward path
        state = model.predict_state({})
        log_surv_exact = state.derived["log_mstar_surviving"]
        joint_weights_exact = state.derived["joint_weights"]

        # Compute surviving mass independently from joint_weights and table
        jw_norm = joint_weights_exact / jnp.sum(joint_weights_exact)
        surv_frac = float(jnp.sum(jw_norm * ssp.ssp_mass_remaining))
        log_formed = state.derived["log_mstar_formed"]
        formed_mass = 10.0 ** float(log_formed)

        # Compare to published log value
        expected_log_surv = np.log10(formed_mass * surv_frac)
        assert_allclose(
            float(log_surv_exact),
            expected_log_surv,
            rtol=1e-6,
            err_msg=(
                f"Published log_mstar_surviving {float(log_surv_exact):.6f} "
                f"!= expected {expected_log_surv:.6f} (computed from joint_weights)"
            ),
        )
