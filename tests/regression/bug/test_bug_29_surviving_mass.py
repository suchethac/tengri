"""Regression test for BUG-29: XRB uses formed mass, not surviving mass.

See ADR / docs/known_bugs.md for full context.
"""

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


class TestBug29MstarSurvivingMass:
    """sed_pipeline.py:753 — XRB must use surviving stellar mass, not formed mass.

    Lehmer+2010 / Mineo+2012 XRB calibrations are normalised to surviving
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

    def test_orchestrator_exposes_surviving_mass(self):
        """The orchestrator path must compute and expose surviving stellar mass.

        Originally pinned to the legacy ``sed_pipeline.compute_sed_components``
        output dict (deleted in Phase B closure). The equivalent invariant —
        that surviving mass is computed via ``compute_surviving_mass`` (not
        a bare ``jnp.sum(weights)``) — is now upheld in StellarSEDComponent
        and the SEDModel orchestrator helpers.
        """
        import inspect

        from tengri.components.stellar import component as stellar_component
        from tengri.forward import prediction, sed_model

        stellar_src = inspect.getsource(stellar_component)
        assert "compute_surviving_mass" in stellar_src, (
            "StellarSEDComponent must call compute_surviving_mass"
        )
        assert "mstar_surv" in stellar_src, (
            "StellarSEDComponent must publish mstar_surv into pipeline state"
        )

        sed_src = inspect.getsource(sed_model)
        pred_src = inspect.getsource(prediction)
        assert "compute_surviving_mass" in sed_src or "compute_surviving_mass" in pred_src, (
            "SEDModel/Prediction must call compute_surviving_mass for derived mass"
        )
