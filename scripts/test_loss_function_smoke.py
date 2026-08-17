"""Quick smoke test for corrected loss function with IFT transforms.

Verifies that the loss function works with all prior types and produces
finite gradients.
"""

import jax
import jax.numpy as jnp
import jax.random as jr

jax.config.update("jax_enable_x64", True)

from pathlib import Path
from tengri import (
    SEDModel,
    Parameters,
    Uniform,
    Gaussian,
    LogUniform,
    LogNormal,
    StudentT,
    Fitter,
    load_ssp_data,
    load_filter_set,
)


def test_loss_with_different_priors():
    """Test that loss function works with all prior types."""
    print("Testing loss function with different prior types...")

    # Load SSP data
    data_dir = Path(__file__).resolve().parents[1] / "data"
    ssp_file = data_dir / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
    if not ssp_file.is_file():
        print("  ⚠️  SSP data not found, skipping test")
        return

    ssp = load_ssp_data(str(ssp_file))
    filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

    # Test with different priors for one parameter each
    prior_configs = {
        "Uniform": {
            "spec_kwargs": {
                "mean_sfh_type": "tsnorm",
                "sfh_tsnorm_skew": Uniform(-0.5, 0.5),
                "sfh_tsnorm_peak_lbt_gyr": Uniform(0.5, 10.0),
                "sfh_tsnorm_width_gyr": Uniform(0.1, 3.0),
                "sfh_tsnorm_trunc": Uniform(0.01, 1.0),
                "sfh_tsnorm_log_total_mass": Uniform(7.0, 12.5),
                "met_logzsol": Uniform(-1.5, 0.2),
                "dust_tau_bc": Uniform(0.0, 2.0),
                "dust_tau_diff": 0.3,
                "dust_slope": -0.7,
                "redshift": 1.0,
            },
            "true_params": {
                "sfh_tsnorm_skew": 0.1,
                "sfh_tsnorm_peak_lbt_gyr": 3.0,
                "sfh_tsnorm_width_gyr": 1.0,
                "sfh_tsnorm_trunc": 0.5,
                "sfh_tsnorm_log_total_mass": 1.0,
                "met_logzsol": -0.2,
                "dust_tau_bc": 0.5,
                "dust_tau_diff": 0.3,
                "dust_slope": -0.7,
                "redshift": 1.0,
            },
        },
        "Gaussian": {
            "spec_kwargs": {
                "mean_sfh_type": "tsnorm",
                "sfh_tsnorm_skew": Gaussian(0.0, 0.2, -0.5, 0.5),
                "sfh_tsnorm_peak_lbt_gyr": Uniform(0.5, 10.0),
                "sfh_tsnorm_width_gyr": Uniform(0.1, 3.0),
                "sfh_tsnorm_trunc": Uniform(0.01, 1.0),
                "sfh_tsnorm_log_total_mass": Uniform(7.0, 12.5),
                "met_logzsol": Uniform(-1.5, 0.2),
                "dust_tau_bc": Uniform(0.0, 2.0),
                "dust_tau_diff": 0.3,
                "dust_slope": -0.7,
                "redshift": 1.0,
            },
            "true_params": {
                "sfh_tsnorm_skew": 0.1,
                "sfh_tsnorm_peak_lbt_gyr": 3.0,
                "sfh_tsnorm_width_gyr": 1.0,
                "sfh_tsnorm_trunc": 0.5,
                "sfh_tsnorm_log_total_mass": 1.0,
                "met_logzsol": -0.2,
                "dust_tau_bc": 0.5,
                "dust_tau_diff": 0.3,
                "dust_slope": -0.7,
                "redshift": 1.0,
            },
        },
        "LogUniform": {
            "spec_kwargs": {
                "mean_sfh_type": "tsnorm",
                "sfh_tsnorm_skew": Uniform(-0.5, 0.5),
                "sfh_tsnorm_peak_lbt_gyr": Uniform(0.5, 10.0),
                "sfh_tsnorm_width_gyr": Uniform(0.1, 3.0),
                "sfh_tsnorm_trunc": Uniform(0.01, 1.0),
                "sfh_tsnorm_log_total_mass": Uniform(7.0, 12.5),
                "met_logzsol": Uniform(-1.5, 0.2),
                "dust_tau_bc": Uniform(0.0, 2.0),
                "dust_tau_diff": LogUniform(0.01, 2.0),
                "dust_slope": -0.7,
                "redshift": 1.0,
            },
            "true_params": {
                "sfh_tsnorm_skew": 0.1,
                "sfh_tsnorm_peak_lbt_gyr": 3.0,
                "sfh_tsnorm_width_gyr": 1.0,
                "sfh_tsnorm_trunc": 0.5,
                "sfh_tsnorm_log_total_mass": 1.0,
                "met_logzsol": -0.2,
                "dust_tau_bc": 0.5,
                "dust_tau_diff": 0.3,
                "dust_slope": -0.7,
                "redshift": 1.0,
            },
        },
        "LogNormal": {
            "spec_kwargs": {
                "mean_sfh_type": "tsnorm",
                "sfh_tsnorm_skew": Uniform(-0.5, 0.5),
                "sfh_tsnorm_peak_lbt_gyr": Uniform(0.5, 10.0),
                "sfh_tsnorm_width_gyr": Uniform(0.1, 3.0),
                "sfh_tsnorm_trunc": Uniform(0.01, 1.0),
                "sfh_tsnorm_log_total_mass": Uniform(7.0, 12.5),
                "met_logzsol": Uniform(-1.5, 0.2),
                "dust_tau_bc": Uniform(0.0, 2.0),
                "dust_tau_diff": LogNormal(-1.0, 0.5, 0.01, 2.0),
                "dust_slope": -0.7,
                "redshift": 1.0,
            },
            "true_params": {
                "sfh_tsnorm_skew": 0.1,
                "sfh_tsnorm_peak_lbt_gyr": 3.0,
                "sfh_tsnorm_width_gyr": 1.0,
                "sfh_tsnorm_trunc": 0.5,
                "sfh_tsnorm_log_total_mass": 1.0,
                "met_logzsol": -0.2,
                "dust_tau_bc": 0.5,
                "dust_tau_diff": 0.3,
                "dust_slope": -0.7,
                "redshift": 1.0,
            },
        },
        "StudentT": {
            "spec_kwargs": {
                "mean_sfh_type": "tsnorm",
                "sfh_tsnorm_skew": StudentT(0.0, 0.2, 3.0, -0.5, 0.5),
                "sfh_tsnorm_peak_lbt_gyr": Uniform(0.5, 10.0),
                "sfh_tsnorm_width_gyr": Uniform(0.1, 3.0),
                "sfh_tsnorm_trunc": Uniform(0.01, 1.0),
                "sfh_tsnorm_log_total_mass": Uniform(7.0, 12.5),
                "met_logzsol": Uniform(-1.5, 0.2),
                "dust_tau_bc": Uniform(0.0, 2.0),
                "dust_tau_diff": 0.3,
                "dust_slope": -0.7,
                "redshift": 1.0,
            },
            "true_params": {
                "sfh_tsnorm_skew": 0.1,
                "sfh_tsnorm_peak_lbt_gyr": 3.0,
                "sfh_tsnorm_width_gyr": 1.0,
                "sfh_tsnorm_trunc": 0.5,
                "sfh_tsnorm_log_total_mass": 1.0,
                "met_logzsol": -0.2,
                "dust_tau_bc": 0.5,
                "dust_tau_diff": 0.3,
                "dust_slope": -0.7,
                "redshift": 1.0,
            },
        },
    }

    key = jr.PRNGKey(42)

    for name, config in prior_configs.items():
        print(f"\n  Testing {name} prior...")

        # Create param spec
        spec = Parameters(**config["spec_kwargs"])

        # Create model
        model = SEDModel(spec, ssp, filters=filters)

        # Generate mock observation
        key, subkey = jr.split(key)
        obs = model.mock(config["true_params"], snr=20.0, key=subkey)

        # Create fitter
        fitter = Fitter(model, obs.flux_obs, obs.noise)

        # Build loss function
        loss_fn = fitter._build_loss_fn()
        data_args = fitter._data_args

        # Initialize parameters
        key, subkey = jr.split(key)
        params_unbounded = fitter._initialize_unbounded(subkey)

        # Compute loss
        loss = loss_fn(params_unbounded, data_args)

        # Compute gradient
        grad_fn = jax.grad(lambda p: loss_fn(p, data_args))
        grads = grad_fn(params_unbounded)

        # Check that loss and gradients are finite
        assert jnp.isfinite(loss), f"{name}: loss is not finite"
        assert loss > 0, f"{name}: loss should be positive"

        grad_norms = {k: float(jnp.linalg.norm(v)) for k, v in grads.items()}
        assert all(jnp.isfinite(v) for v in grad_norms.values()), (
            f"{name}: some gradients are not finite"
        )

        print(f"    ✓ loss = {loss:.2f}")
        print(f"    ✓ max |grad| = {max(grad_norms.values()):.2e}")

    print("\n✓ All prior types work correctly!\n")


if __name__ == "__main__":
    print()
    print("=" * 70)
    print("  Loss Function Smoke Test (IFT Transforms)")
    print("=" * 70)
    print()

    test_loss_with_different_priors()

    print("=" * 70)
    print("  ✓ All smoke tests passed!")
    print("=" * 70)
    print()
