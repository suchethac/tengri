# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for BUG-29: XRB uses formed mass, not surviving mass.

See ADR / docs/known_bugs.md for full context.
"""

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


class TestBug29MstarSurvivingMass:
    """sed_pipeline.py:753 — XRB must use surviving stellar mass, not formed mass.

    Lehmer+2010 / Mineo+2012 XRB calibrations are normalized to surviving
    stellar mass (living stars + remnants). Using total formed mass overestimates
    XRB L_X by ~30-50% for old stellar populations.

    Fix: sed_pipeline.py now calls compute_surviving_mass(weights,
    interpolate_mass_remaining(...)) and exposes both mstar_formed and
    mstar_surviving in the output dict.
    """

    def test_surviving_mass_less_than_formed_for_old_population(self):
        """Surviving mass must be < formed mass for a purely old SSP.

        For a 10 Gyr population with Kroupa IMF, f_surv ≈ 0.6 (B&C03).
        compute_surviving_mass(weights, f_surv * ones) < sum(weights).
        """
        from tengri.components.stellar.sps.dsps_wrapper import compute_surviving_mass

        weights = jnp.ones(50) * 1e9  # 50 age bins, 1e9 Msun each
        # Simulate old population: f_surv = 0.6 uniformly
        mass_remaining = jnp.full(50, 0.6)
        surviving = float(compute_surviving_mass(weights, mass_remaining))
        formed = float(jnp.sum(weights))
        assert surviving < formed, (
            f"Surviving mass {surviving:.3e} should be < formed mass {formed:.3e}"
        )
        assert abs(surviving / formed - 0.6) < 1e-6, (
            f"Expected surviving/formed = 0.6, got {surviving / formed:.4f}"
        )

    def test_interpolate_mass_remaining_shape(self):
        """interpolate_mass_remaining returns shape (n_age,) for a scalar log_z."""
        from tengri.components.stellar.sps.dsps_wrapper import interpolate_mass_remaining

        n_met, n_age = 4, 20
        # Synthetic mass-remaining grid: decreases with age (older → less survives)
        ssp_mass_remaining = jnp.ones((n_met, n_age)) * jnp.linspace(0.95, 0.50, n_age)
        ssp_lgmet = jnp.linspace(-2.0, 0.3, n_met)
        mr = interpolate_mass_remaining(ssp_mass_remaining, ssp_lgmet, log_z=-1.0)
        assert mr.shape == (n_age,), f"Expected shape ({n_age},), got {mr.shape}"
        assert jnp.all(mr > 0.0), "Mass-remaining fractions must be positive"
        assert jnp.all(mr <= 1.0), "Mass-remaining fractions must be <= 1"

    def test_orchestrator_exposes_surviving_mass(self, synthetic_ssp_wide):
        """The orchestrator path must compute and expose surviving stellar mass.

        Regression: #29. The StellarSEDComponent must publish both mstar_formed
        and mstar_surv into the state.derived dict via compute_surviving_mass.

        Test: Build a model, get state, and verify:
        1. Both mstar_formed and mstar_surv are published
        2. mstar_surv < mstar_formed (mass is lost to stellar evolution)
        3. The ratio is consistent with age-dependent mass-remaining fractions
        """
        import jax

        from tengri import FREE, Fixed, SEDModel, SSPData

        # ``synthetic_ssp_wide`` carries no ``ssp_mass_remaining``, so on it the
        # orchestrator publishes ``log_mstar_surviving`` as NaN. An earlier version
        # of this test guarded every assertion below with
        # ``if jnp.isfinite(log_mstar_surviving):`` and so passed while checking
        # nothing at all about surviving mass -- the entire subject of #29.
        # Supply the grid the claim needs rather than skipping around its absence.
        base = synthetic_ssp_wide
        n_met = base.ssp_lgmet.shape[0]
        n_age = base.ssp_lg_age_gyr.shape[0]
        # Decreasing with age: a young population keeps nearly all its mass, a
        # 13.8 Gyr one keeps ~0.55 (B&C03, Kroupa IMF).
        frac = jnp.linspace(0.98, 0.55, n_age)
        ssp = SSPData(
            ssp_wave=base.ssp_wave,
            ssp_flux=base.ssp_flux,
            ssp_lg_age_gyr=base.ssp_lg_age_gyr,
            ssp_lgmet=base.ssp_lgmet,
            ssp_mass_remaining=jnp.broadcast_to(frac[None, :], (n_met, n_age)),
        )

        model = SEDModel.build(
            ssp_data=ssp,
            observation=None,
            sfh={"type": "dpl", "all_params": FREE},
            met={"logzsol": Fixed(0.0)},
            redshift=Fixed(0.0),
        )
        params = model.spec.sample(jax.random.PRNGKey(0))
        state = model.predict_state(params)

        assert "log_mstar_formed" in state.derived, (
            "log_mstar_formed must be published in ForwardState.derived"
        )
        assert "log_mstar_surviving" in state.derived, (
            "log_mstar_surviving must be published in ForwardState.derived"
        )

        log_formed = state.derived["log_mstar_formed"]
        log_surviving = state.derived["log_mstar_surviving"]

        # Unconditional: a non-finite value here is the failure, not a reason to skip.
        assert jnp.isfinite(log_formed), f"log_mstar_formed must be finite, got {log_formed}"
        assert jnp.isfinite(log_surviving), (
            f"log_mstar_surviving is {log_surviving}: this SSP supplies "
            f"ssp_mass_remaining, so a non-finite value means the surviving-mass "
            f"path never ran"
        )

        f_surv = float(10.0 ** (log_surviving - log_formed))
        assert 0.0 < f_surv < 1.0, (
            f"surviving fraction must lie in (0, 1); got {f_surv:.4f}. Equal masses "
            f"mean the mass-remaining weighting was never applied -- the #29 defect"
        )
        # Tighter than (0, 1): an SFH-weighted mean of a profile bounded by
        # [0.55, 0.98] cannot fall outside those bounds. This is what separates
        # "a surviving fraction was applied" from "this particular grid was used".
        assert float(frac.min()) <= f_surv <= float(frac.max()), (
            f"surviving fraction {f_surv:.4f} lies outside the range the supplied "
            f"mass-remaining grid spans ([{float(frac.min()):.2f}, "
            f"{float(frac.max()):.2f}]): the published value is not a weighted mean "
            f"of this grid"
        )
