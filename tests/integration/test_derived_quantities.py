"""Integration tests for stellar mass and derived quantities.

Verifies that Model.predict_sfh_quantities() and Model.predict_derived()
return physically reasonable values using real SSP data.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tengri import Model, ParamSpec, Uniform
from tengri.components.sps.dsps_wrapper import load_ssp_data
from tengri.parameters.priors import Fixed

# ── Paths to real SSP data ────────────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_NO_NEB = _DATA_DIR / "fsps_prsc_miles_chabrier.h5"

_SSP_FILES_EXIST = _SSP_NO_NEB.is_file()
pytestmark = pytest.mark.skipif(
    not _SSP_FILES_EXIST,
    reason="SSP data files not found in data/",
)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def ssp():
    return load_ssp_data(str(_SSP_NO_NEB))


@pytest.fixture(scope="session")
def spec():
    """ParamSpec with DPL + field SFH model."""
    return ParamSpec(
        mean_sfh_type=["dpl", "field"],
        n_grid=256,
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
        sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.0),
        sfh_field_psd_sigma=Uniform(0.01, 3.0),
        sfh_field_psd_tau_myr=Uniform(10.0, 500.0),
        met_logzsol=Uniform(-2.0, 0.2),
        dust_tau_bc=Uniform(0.0, 4.0),
        dust_tau_diff=Uniform(0.0, 3.0),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
    )


@pytest.fixture(scope="session")
def model(ssp, spec):
    return Model(spec, ssp)


@pytest.fixture(scope="session")
def fiducial_params(spec):
    """Typical star-forming galaxy parameters using public names."""
    n_grid = spec.n_grid
    return {
        "sfh_field_xi": jnp.zeros(n_grid),
        "sfh_dpl_alpha": 1.0,
        "sfh_dpl_beta": 1.5,
        "sfh_dpl_tau_gyr": 3.0,
        "sfh_dpl_log_peak_sfr": np.log10(5.0),
        "sfh_field_psd_sigma": 1.0,
        "sfh_field_psd_tau_myr": 50.0,
        "met_logzsol": -0.2,
        "dust_tau_bc": 1.0,
        "dust_tau_diff": 0.3,
        "dust_slope": -0.7,
        "redshift": 0.1,
    }


@pytest.fixture(scope="session")
def smooth_params(spec):
    """Very smooth SFH (psd_sigma ~ 0) so GP has negligible effect."""
    n_grid = spec.n_grid
    return {
        "sfh_field_xi": jnp.zeros(n_grid),
        "sfh_dpl_alpha": 1.0,
        "sfh_dpl_beta": 1.5,
        "sfh_dpl_tau_gyr": 3.0,
        "sfh_dpl_log_peak_sfr": np.log10(5.0),
        "sfh_field_psd_sigma": 0.01,
        "sfh_field_psd_tau_myr": 50.0,
        "met_logzsol": -0.2,
        "dust_tau_bc": 1.0,
        "dust_tau_diff": 0.3,
        "dust_slope": -0.7,
        "redshift": 0.1,
    }


# ── 1. Stellar Mass (via predict_sfh_quantities) ──────────────────


class TestStellarMass:
    """Verify predict_sfh_quantities returns physical mass values."""

    def test_mass_positive_and_finite(self, model, fiducial_params):
        sfh = model.predict_sfh_quantities(fiducial_params)
        assert jnp.isfinite(sfh.stellar_mass), "stellar_mass must be finite"
        assert float(sfh.stellar_mass) > 0.0, "stellar_mass must be positive"

    def test_mass_in_reasonable_range(self, model, fiducial_params):
        sfh = model.predict_sfh_quantities(fiducial_params)
        mstar = float(sfh.stellar_mass)
        assert 1e6 < mstar < 1e14, f"M* = {mstar:.2e} Msun outside plausible range [1e6, 1e14]"

    def test_smooth_gp_mass_consistent(self, model, smooth_params):
        """With psd_sigma ~ 0, GP is negligible so mass should be stable."""
        sfh = model.predict_sfh_quantities(smooth_params)
        assert jnp.isfinite(sfh.stellar_mass)
        assert float(sfh.stellar_mass) > 0.0

    def test_mass_scales_with_peak_sfr(self, model, fiducial_params):
        """Doubling peak SFR should roughly double M*."""
        sfh_1x = model.predict_sfh_quantities(fiducial_params)

        params_2x = {
            **fiducial_params,
            "sfh_dpl_log_peak_sfr": (fiducial_params["sfh_dpl_log_peak_sfr"] + np.log10(2.0)),
        }
        sfh_2x = model.predict_sfh_quantities(params_2x)

        ratio = float(sfh_2x.stellar_mass) / float(sfh_1x.stellar_mass)
        np.testing.assert_allclose(
            ratio,
            2.0,
            rtol=0.1,
            err_msg=(f"Doubling peak SFR should ~double M*, got ratio = {ratio:.3f}"),
        )


# ── 2. Derived Quantities (via predict_derived) ───────────────────


class TestDerivedQuantities:
    """Verify predict_derived returns physical values."""

    def test_all_keys_present(self, model, fiducial_params):
        derived = model.predict_derived(fiducial_params)
        expected_keys = {
            "stellar_mass",
            "stellar_mass_surviving",
            "sfr_100myr",
            "sfr_10myr",
            "ssfr",
        }
        assert set(derived.keys()) == expected_keys

    def test_all_values_finite(self, model, fiducial_params):
        derived = model.predict_derived(fiducial_params)
        for key, val in derived.items():
            if val is not None:  # stellar_mass_surviving can be None
                assert jnp.isfinite(val), f"{key} is not finite: {val}"

    def test_sfr_positive_for_star_forming(self, model, fiducial_params):
        derived = model.predict_derived(fiducial_params)
        assert float(derived["sfr_100myr"]) > 0.0, "SFR_100Myr must be > 0"
        assert float(derived["sfr_10myr"]) > 0.0, "SFR_10Myr must be > 0"

    def test_ssfr_reasonable_range(self, model, fiducial_params):
        """sSFR for a star-forming galaxy should be ~1e-14 to 1e-7 yr^-1."""
        derived = model.predict_derived(fiducial_params)
        ssfr = float(derived["ssfr"])
        assert 1e-16 < ssfr < 1e-6, f"sSFR = {ssfr:.2e} yr^-1 outside plausible range"

    def test_mstar_consistent_between_methods(self, model, fiducial_params):
        """predict_derived and predict_sfh_quantities should agree on mass."""
        derived = model.predict_derived(fiducial_params)
        sfh = model.predict_sfh_quantities(fiducial_params)

        np.testing.assert_allclose(
            float(derived["stellar_mass"]),
            float(sfh.stellar_mass),
            rtol=1e-6,
            err_msg="stellar_mass should match between methods",
        )

    def test_bursty_gp_changes_mass(self, model, spec):
        """A non-zero GP realization with large sigma should change M*."""
        n_grid = spec.n_grid
        key = jax.random.PRNGKey(42)
        xi = jax.random.normal(key, shape=(n_grid,))

        params = {
            "sfh_field_xi": xi,
            "sfh_dpl_alpha": 1.0,
            "sfh_dpl_beta": 1.5,
            "sfh_dpl_tau_gyr": 3.0,
            "sfh_dpl_log_peak_sfr": np.log10(5.0),
            "sfh_field_psd_sigma": 2.0,
            "sfh_field_psd_tau_myr": 50.0,
            "met_logzsol": -0.2,
            "dust_tau_bc": 1.0,
            "dust_tau_diff": 0.3,
            "dust_slope": -0.7,
            "redshift": 0.1,
        }

        # Compare to zero-xi version
        params_zero = {**params, "sfh_field_xi": jnp.zeros(n_grid)}
        sfh_bursty = model.predict_sfh_quantities(params)
        sfh_smooth = model.predict_sfh_quantities(params_zero)

        # With random xi and large sigma, masses should differ
        ratio = float(sfh_bursty.stellar_mass) / float(sfh_smooth.stellar_mass)
        assert ratio != 1.0, "Bursty GP should change stellar mass"

    def test_smooth_gp_mass_close_to_zero_xi(self, model, spec):
        """With small sigma, mass should be close to zero-xi value."""
        n_grid = spec.n_grid
        xi = jax.random.normal(jax.random.PRNGKey(0), shape=(n_grid,))

        params = {
            "sfh_field_xi": xi,
            "sfh_dpl_alpha": 1.0,
            "sfh_dpl_beta": 1.5,
            "sfh_dpl_tau_gyr": 3.0,
            "sfh_dpl_log_peak_sfr": np.log10(5.0),
            "sfh_field_psd_sigma": 0.3,
            "sfh_field_psd_tau_myr": 50.0,
            "met_logzsol": -0.2,
            "dust_tau_bc": 0.5,
            "dust_tau_diff": 0.2,
            "dust_slope": -0.7,
            "redshift": 0.1,
        }

        params_zero = {**params, "sfh_field_xi": jnp.zeros(n_grid)}
        sfh_xi = model.predict_sfh_quantities(params)
        sfh_zero = model.predict_sfh_quantities(params_zero)

        ratio = float(sfh_xi.stellar_mass) / float(sfh_zero.stellar_mass)
        assert 0.5 < ratio < 2.0, f"Smooth GP: mass ratio = {ratio:.2f}, expected close to 1"

    def test_ensemble_mean_mass_converges(self, model, spec):
        """Over many GP realizations, <M*> should approach zero-xi M*.

        The lognormal correction ensures E[SFR] = mean_SFR.
        """
        n_grid = spec.n_grid
        base = {
            "sfh_dpl_alpha": 1.0,
            "sfh_dpl_beta": 1.5,
            "sfh_dpl_tau_gyr": 3.0,
            "sfh_dpl_log_peak_sfr": np.log10(5.0),
            "sfh_field_psd_sigma": 1.5,
            "sfh_field_psd_tau_myr": 50.0,
            "met_logzsol": -0.2,
            "dust_tau_bc": 0.5,
            "dust_tau_diff": 0.2,
            "dust_slope": -0.7,
            "redshift": 0.1,
        }

        # Get baseline mass with zero xi
        params_zero = {**base, "sfh_field_xi": jnp.zeros(n_grid)}
        mstar_base = float(model.predict_sfh_quantities(params_zero).stellar_mass)

        # Compute mass for many GP realizations
        n_draws = 50
        masses = []
        for i in range(n_draws):
            xi = jax.random.normal(jax.random.PRNGKey(i), shape=(n_grid,))
            p = {**base, "sfh_field_xi": xi}
            m = float(model.predict_sfh_quantities(p).stellar_mass)
            masses.append(m)

        ensemble_mean = sum(masses) / len(masses)
        ratio = ensemble_mean / mstar_base

        # Ensemble average should be within ~3x of the zero-xi mass
        assert 0.3 < ratio < 3.0, f"Ensemble <M*>/M*_base = {ratio:.2f}, expected ~1"


# ── 3. Gradient Flow ──────────────────────────────────────────────


class TestDerivedGradients:
    """Verify gradients flow through derived quantities."""

    def test_mstar_gradient_wrt_peak_sfr(self, model, fiducial_params):
        def loss(p):
            return model.predict_sfh_quantities(p).stellar_mass

        grad = jax.grad(loss)(fiducial_params)
        g = grad["sfh_dpl_log_peak_sfr"]

        # Test with finite difference for peak_sfr
        def loss_scalar(log_peak_sfr):
            params = dict(fiducial_params)
            params["sfh_dpl_log_peak_sfr"] = log_peak_sfr
            return float(model.predict_sfh_quantities(params).stellar_mass)

        grad_jax = float(g)
        x0 = float(fiducial_params["sfh_dpl_log_peak_sfr"])
        grad_fd = (loss_scalar(x0 + 1e-4) - loss_scalar(x0 - 1e-4)) / (2.0 * 1e-4)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            atol=1e-10,
            err_msg=f"log_peak_sfr: autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
        )
        assert grad_jax > 0.0, "Increasing log_peak_sfr should increase M*"

    def test_sfr_gradient_wrt_peak_sfr(self, model, fiducial_params):
        def loss(p):
            sfh = model.predict_sfh_quantities(p)
            return sfh.stellar_mass + sfh.sfr_100myr

        grad = jax.grad(loss)(fiducial_params)
        g = grad["sfh_dpl_log_peak_sfr"]

        # Test with finite difference for peak_sfr
        def loss_scalar(log_peak_sfr):
            params = dict(fiducial_params)
            params["sfh_dpl_log_peak_sfr"] = log_peak_sfr
            sfh = model.predict_sfh_quantities(params)
            return float(sfh.stellar_mass + sfh.sfr_100myr)

        grad_jax = float(g)
        x0 = float(fiducial_params["sfh_dpl_log_peak_sfr"])
        grad_fd = (loss_scalar(x0 + 1e-4) - loss_scalar(x0 - 1e-4)) / (2.0 * 1e-4)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            atol=1e-10,
            err_msg=f"log_peak_sfr: autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
        )
