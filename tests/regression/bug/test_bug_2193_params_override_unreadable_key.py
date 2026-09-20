# SPDX-License-Identifier: BSD-3-Clause
r"""params_override rejects a key the likelihood does not read (#2193).

When `has_noise_model(spec)` is False (no noise parameter free or Fixed nonzero),
the built likelihood does not wire `f_cal_param="noise_frac_cal"` and cannot read
it. Attempting to override it at fit time is silently accepted, with no error and
no warning — a confusing contrast to typo rejection. The override has zero effect
on the fit.

**Measured behavior before the fix:**

| `params_override` | result |
|---|---|
| `{'noise_frac_cal': 0.25}` (default spec) | ACCEPTED — loss unchanged |
| `{'not_a_real_parameter_at_all': 1.23}` | ValueError: not a valid parameter name |
| `{'sfh_delayed_log_total_masss': 10.0}` (typo) | ValueError: not a valid parameter name |

**The fix:** when `has_noise_model(spec)` is False and the override key is a `noise_*`
parameter, raise ValueError with a message that explains the mechanism and suggests
declaring it in the spec instead.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Fixed, ForwardModel, Observation, Photometry
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.observation.noise_model import NoiseModel
from tengri.observation.photometry import FilterCurve


def build_default_model():
    """Build a default model with no noise parameters declared.

    Synthetic SSP, SDSS 5-band photometry, double power law SFH, two-component
    Calzetti dust, z=0.1. The spec has noise_frac_cal at Fixed(0.0), so
    has_noise_model(spec) returns False and the likelihood is plain Gaussian.
    """
    from tengri import FREE

    wave = jnp.linspace(3000.0, 10000.0, 60)
    ages = jnp.linspace(-1.0, 1.14, 12)
    lgmet = jnp.array([-1.5, -0.5, 0.0])
    flux_grid = jnp.abs(jnp.ones((3, 12, 60))) * 1e-3 + 1e-5
    ssp = SSPData(ssp_wave=wave, ssp_flux=flux_grid, ssp_lg_age_gyr=ages, ssp_lgmet=lgmet)
    curves = tuple(
        FilterCurve(wave=jnp.linspace(lo, hi, 30), trans=jnp.ones(30) * 0.5, name=f"b{i}")
        for i, (lo, hi) in enumerate([(3500.0, 4500.0), (5000.0, 6500.0), (7500.0, 9000.0)])
    )
    obs = Observation(photometry=Photometry(filters=curves))

    from tengri import SEDModel

    sed = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FREE},
        dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FREE},
        redshift=Fixed(0.1),
    )
    truth = {
        "sfh_dpl_log_total_mass": 10.0,
        "sfh_dpl_age_gyr": 1.0,
        "sfh_dpl_alpha": 1.0,
        "sfh_dpl_beta": 1.0,
        "sfh_dpl_tau_gyr": 1.0,
        "dust_tau_bc": 0.3,
        "dust_tau_diff": 0.2,
    }
    data = jnp.asarray(np.asarray(sed.predict_photometry(truth)))
    noise = jnp.asarray(0.05 * np.abs(np.asarray(data)))

    forward = ForwardModel.build(sed=sed, observation=obs)
    return forward, data, noise, sed


def build_noise_model():
    """Build a model with noise_frac_cal declared at Fixed(0.05).

    Same as default_model, but with noise_frac_cal explicitly declared
    in the spec. Now has_noise_model(spec) returns True and the likelihood
    wires with f_cal_param="noise_frac_cal".
    """
    from tengri import FREE

    wave = jnp.linspace(3000.0, 10000.0, 60)
    ages = jnp.linspace(-1.0, 1.14, 12)
    lgmet = jnp.array([-1.5, -0.5, 0.0])
    flux_grid = jnp.abs(jnp.ones((3, 12, 60))) * 1e-3 + 1e-5
    ssp = SSPData(ssp_wave=wave, ssp_flux=flux_grid, ssp_lg_age_gyr=ages, ssp_lgmet=lgmet)
    curves = tuple(
        FilterCurve(wave=jnp.linspace(lo, hi, 30), trans=jnp.ones(30) * 0.5, name=f"b{i}")
        for i, (lo, hi) in enumerate([(3500.0, 4500.0), (5000.0, 6500.0), (7500.0, 9000.0)])
    )
    obs = Observation(
        photometry=Photometry(filters=curves),
        noise=NoiseModel(calibration_floor=0.05),  # Declare noise_frac_cal at nonzero value
    )

    from tengri import SEDModel

    sed = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FREE},
        dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FREE},
        redshift=Fixed(0.1),
    )
    truth = {
        "sfh_dpl_log_total_mass": 10.0,
        "sfh_dpl_age_gyr": 1.0,
        "sfh_dpl_alpha": 1.0,
        "sfh_dpl_beta": 1.0,
        "sfh_dpl_tau_gyr": 1.0,
        "dust_tau_bc": 0.3,
        "dust_tau_diff": 0.2,
    }
    data = jnp.asarray(np.asarray(sed.predict_photometry(truth)))
    noise = jnp.asarray(0.05 * np.abs(np.asarray(data)))

    forward = ForwardModel.build(sed=sed, observation=obs)
    return forward, data, noise, sed


class TestParamsOverrideUnreadableKey:
    """params_override rejects a key the built likelihood cannot read (#2193)."""

    pytestmark = pytest.mark.regression_bug

    def test_noise_param_unreadable_default_spec_raises(self):
        """Override a noise_* parameter when has_noise_model(spec) is False raises."""
        forward, data, noise, _ = build_default_model()

        # The default spec has noise_frac_cal at Fixed(0.0), so
        # has_noise_model(spec) is False and the likelihood does not read it.
        # Attempting to override it should raise with a message naming the
        # declaration route.
        with pytest.raises(ValueError, match=r"Observation\(noise=NoiseModel"):
            forward.fit(data, noise, method="map", params={"noise_frac_cal": 0.25}, n_steps=10)

    def test_noise_param_readable_declared_nonzero_accepts(self):
        """Control: override noise_frac_cal when declared at Fixed(0.05) is accepted."""
        forward, data, noise, _ = build_noise_model()

        # The spec has noise_frac_cal declared at Fixed(0.05), so
        # has_noise_model(spec) is True and the likelihood reads it.
        # Attempting to override it should succeed without raising.
        try:
            result = forward.fit(
                data, noise, method="map", params={"noise_frac_cal": 0.25}, n_steps=10
            )
            # If we get here, the override was accepted.
            assert "noise_frac_cal" in result.params
        except ValueError as e:
            if "does not read" in str(e):
                msg = f"Unexpected rejection when noise_frac_cal is declared: {e}"
                pytest.fail(msg)
            raise

    def test_typo_parameter_rejected_as_before(self):
        """Typo in parameter name still raises the existing 'not a valid parameter name' error."""
        forward, data, noise, _ = build_default_model()

        with pytest.raises(ValueError, match="not a valid parameter name"):
            forward.fit(
                data, noise, method="map", params={"sfh_dpl_log_total_masss": 10.0}, n_steps=10
            )
