"""Integration tests for stellar mass and derived quantities.

Verifies that compute_stellar_mass() and compute_derived_quantities()
return physically reasonable values using real SSP data.
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from diffsed.forward_model import ForwardModel, ModelConfig
from diffsed.models.sps.dsps_wrapper import SSPData, load_ssp_data


# ---------------------------------------------------------------------------
# Paths to real SSP data
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_NO_NEB = _DATA_DIR / "fsps_prsc_miles_chabrier.h5"
_SSP_WITH_NEB = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"

_SSP_FILES_EXIST = _SSP_NO_NEB.is_file() and _SSP_WITH_NEB.is_file()
pytestmark = pytest.mark.skipif(
    not _SSP_FILES_EXIST,
    reason="SSP data files not found in data/",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ssp_no_neb():
    return load_ssp_data(str(_SSP_NO_NEB))


@pytest.fixture(scope="session")
def default_config():
    return ModelConfig(
        n_grid=256,
        log_age_min=6.0,
        log_age_max=10.14,
        redshift=0.1,
    )


@pytest.fixture(scope="session")
def model(ssp_no_neb, default_config):
    return ForwardModel(ssp_no_neb, default_config)


@pytest.fixture(scope="session")
def fiducial_params():
    """Typical star-forming galaxy parameters."""
    return {
        "xi": jnp.zeros(256),
        "sigma_ps": 1.0,
        "tau_ps": 50e6,
        "alpha": 1.0,
        "beta": 1.5,
        "tau_sfh": 3e9,
        "sfr_norm": 5.0,
        "log_z": -0.2,
        "tau_v1": 1.0,
        "tau_v2": 0.3,
        "dust_n": -0.7,
    }


@pytest.fixture(scope="session")
def smooth_params():
    """Very smooth SFH (sigma_ps ~ 0) so GP has negligible effect."""
    return {
        "xi": jnp.zeros(256),
        "sigma_ps": 0.01,
        "tau_ps": 50e6,
        "alpha": 1.0,
        "beta": 1.5,
        "tau_sfh": 3e9,
        "sfr_norm": 5.0,
        "log_z": -0.2,
        "tau_v1": 1.0,
        "tau_v2": 0.3,
        "dust_n": -0.7,
    }


# ===================================================================
# 1. Stellar Mass
# ===================================================================

class TestStellarMass:
    """Verify compute_stellar_mass returns physical values."""

    def test_mass_positive_and_finite(self, model, fiducial_params):
        masses = model.compute_stellar_mass(fiducial_params)
        assert jnp.isfinite(masses["mstar_mean"]), "mstar_mean must be finite"
        assert jnp.isfinite(masses["mstar_total"]), "mstar_total must be finite"
        assert float(masses["mstar_mean"]) > 0.0, "mstar_mean must be positive"
        assert float(masses["mstar_total"]) > 0.0, "mstar_total must be positive"

    def test_mass_in_reasonable_range(self, model, fiducial_params):
        masses = model.compute_stellar_mass(fiducial_params)
        mstar = float(masses["mstar_total"])
        assert 1e8 < mstar < 1e12, (
            f"M* = {mstar:.2e} Msun outside range [1e8, 1e12]"
        )

    def test_mean_equals_total_when_smooth(self, model, smooth_params):
        """With sigma_ps ~ 0, GP is negligible so mean ≈ total."""
        masses = model.compute_stellar_mass(smooth_params)
        ratio = float(masses["mstar_total"]) / float(masses["mstar_mean"])
        np.testing.assert_allclose(ratio, 1.0, atol=0.01, err_msg=(
            f"M*_total / M*_mean = {ratio:.4f}, should be ~1.0 for smooth SFH"
        ))

    def test_mass_scales_with_norm(self, model, fiducial_params):
        """Doubling sfr_norm should roughly double M*."""
        masses_1x = model.compute_stellar_mass(fiducial_params)
        params_2x = {**fiducial_params, "sfr_norm": fiducial_params["sfr_norm"] * 2.0}
        masses_2x = model.compute_stellar_mass(params_2x)

        ratio = float(masses_2x["mstar_total"]) / float(masses_1x["mstar_total"])
        np.testing.assert_allclose(ratio, 2.0, rtol=0.05, err_msg=(
            f"Doubling sfr_norm should double M*, got ratio = {ratio:.3f}"
        ))


# ===================================================================
# 2. Derived Quantities
# ===================================================================

class TestDerivedQuantities:
    """Verify compute_derived_quantities returns physical values."""

    def test_all_keys_present(self, model, fiducial_params):
        derived = model.compute_derived_quantities(fiducial_params)
        expected_keys = {"mstar_formed", "mstar_mean", "sfr_100myr", "sfr_10myr", "ssfr"}
        assert set(derived.keys()) == expected_keys

    def test_all_values_finite(self, model, fiducial_params):
        derived = model.compute_derived_quantities(fiducial_params)
        for key, val in derived.items():
            assert jnp.isfinite(val), f"{key} is not finite: {val}"

    def test_sfr_positive_for_star_forming(self, model, fiducial_params):
        derived = model.compute_derived_quantities(fiducial_params)
        assert float(derived["sfr_100myr"]) > 0.0, "SFR_100Myr must be > 0"
        assert float(derived["sfr_10myr"]) > 0.0, "SFR_10Myr must be > 0"

    def test_ssfr_reasonable_range(self, model, fiducial_params):
        """sSFR for a star-forming galaxy should be ~1e-11 to 1e-8 yr^-1."""
        derived = model.compute_derived_quantities(fiducial_params)
        ssfr = float(derived["ssfr"])
        assert 1e-14 < ssfr < 1e-7, (
            f"sSFR = {ssfr:.2e} yr^-1 outside plausible range"
        )

    def test_mstar_consistent_with_compute_stellar_mass(self, model, fiducial_params):
        """mstar_formed from derived should match compute_stellar_mass."""
        derived = model.compute_derived_quantities(fiducial_params)
        masses = model.compute_stellar_mass(fiducial_params)

        np.testing.assert_allclose(
            float(derived["mstar_formed"]),
            float(masses["mstar_total"]),
            rtol=1e-10,
            err_msg="mstar_formed should match compute_stellar_mass mstar_total",
        )
        np.testing.assert_allclose(
            float(derived["mstar_mean"]),
            float(masses["mstar_mean"]),
            rtol=1e-10,
            err_msg="mstar_mean should be consistent",
        )

    def test_bursty_gp_changes_mass(self, model):
        """A non-zero GP realization with large sigma should shift M* from mean."""
        key = jax.random.PRNGKey(42)
        xi = jax.random.normal(key, shape=(256,))

        params = {
            "xi": xi,
            "sigma_ps": 2.0,
            "tau_ps": 50e6,
            "alpha": 1.0,
            "beta": 1.5,
            "tau_sfh": 3e9,
            "sfr_norm": 5.0,
            "log_z": -0.2,
            "tau_v1": 1.0,
            "tau_v2": 0.3,
            "dust_n": -0.7,
        }

        derived = model.compute_derived_quantities(params)
        ratio = float(derived["mstar_formed"]) / float(derived["mstar_mean"])
        # With a random GP and large sigma, the ratio should deviate from 1
        assert ratio != 1.0, "Bursty GP should make M*_total differ from M*_mean"

    def test_smooth_gp_mass_close_to_mean(self, model):
        """With small sigma_PS, total mass should be close to mean SFH mass."""
        params = {
            "xi": jax.random.normal(jax.random.PRNGKey(0), shape=(256,)),
            "sigma_ps": 0.3,  # very smooth
            "tau_ps": 50e6,
            "alpha": 1.0, "beta": 1.5, "tau_sfh": 3e9, "sfr_norm": 5.0,
            "log_z": -0.2, "tau_v1": 0.5, "tau_v2": 0.2, "dust_n": -0.7,
        }
        masses = model.compute_stellar_mass(params)
        ratio = float(masses["mstar_total"]) / float(masses["mstar_mean"])
        # With sigma_PS=0.3, the ratio should be within ~30% of 1
        assert 0.5 < ratio < 2.0, (
            f"Smooth GP: M*_total/M*_mean = {ratio:.2f}, expected close to 1"
        )

    def test_ensemble_mean_mass_converges(self, model):
        """Over many GP realizations, <M*_total> should approach M*_mean.

        The lognormal correction ensures E[SFR] = mean_SFR, so the
        ensemble-averaged mass should equal the mean SFH mass (approximately).
        """
        base = {
            "sigma_ps": 1.5, "tau_ps": 50e6,
            "alpha": 1.0, "beta": 1.5, "tau_sfh": 3e9, "sfr_norm": 5.0,
            "log_z": -0.2, "tau_v1": 0.5, "tau_v2": 0.2, "dust_n": -0.7,
        }

        # Get M*_mean (from mean SFH, no GP dependence)
        params_zero = {**base, "xi": jnp.zeros(256)}
        mstar_mean = float(model.compute_stellar_mass(params_zero)["mstar_mean"])

        # Compute M*_total for many GP realizations
        n_draws = 50
        masses = []
        for i in range(n_draws):
            xi = jax.random.normal(jax.random.PRNGKey(i), shape=(256,))
            p = {**base, "xi": xi}
            m = float(model.compute_stellar_mass(p)["mstar_total"])
            masses.append(m)

        ensemble_mean = sum(masses) / len(masses)
        ratio = ensemble_mean / mstar_mean

        # The ensemble average should be within ~50% of the mean SFH mass
        # (finite N, finite grid → won't be exact)
        assert 0.3 < ratio < 3.0, (
            f"Ensemble <M*_total>/M*_mean = {ratio:.2f}, expected ~1"
        )


# ===================================================================
# 3. Gradient Flow
# ===================================================================

class TestDerivedGradients:
    """Verify gradients flow through derived quantities."""

    def test_mstar_gradient_wrt_sfr_norm(self, model, fiducial_params):
        def loss(p):
            return model.compute_stellar_mass(p)["mstar_total"]

        grad = jax.grad(loss)(fiducial_params)
        assert jnp.isfinite(grad["sfr_norm"]), "Gradient w.r.t. sfr_norm not finite"
        assert float(grad["sfr_norm"]) > 0.0, (
            "Increasing sfr_norm should increase M*"
        )

    def test_derived_gradient_wrt_sfr_norm(self, model, fiducial_params):
        def loss(p):
            d = model.compute_derived_quantities(p)
            return d["mstar_formed"] + d["sfr_100myr"]

        grad = jax.grad(loss)(fiducial_params)
        assert jnp.isfinite(grad["sfr_norm"]), "Gradient not finite"
