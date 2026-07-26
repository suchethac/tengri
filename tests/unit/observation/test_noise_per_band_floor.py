# SPDX-License-Identifier: BSD-3-Clause
"""#919: calibration_floor accepts a per-band array — sigma_eff must
apply each band's own floor: sqrt(sigma^2 + (f_b * model_b)^2)."""

import jax.numpy as jnp
import numpy as np
import pytest


def test_per_band_floor_shapes_and_values():
    """Per-band calibration floors apply element-wise in sigma_eff formula."""
    from tengri.observation.noise import compute_effective_noise

    # Per-band calibration floors
    f_cal_array = jnp.array([0.02, 0.10, 0.05])
    model_flux = jnp.array([1.0, 1.0, 2.0])
    sigma_obs = jnp.array([0.1, 0.1, 0.1])

    # Compute effective noise with per-band floor
    sigma_eff = compute_effective_noise(sigma_obs, model_flux, f_cal_array)

    # Expected: each band uses its own floor
    expected = np.sqrt(sigma_obs**2 + (f_cal_array * model_flux) ** 2)
    np.testing.assert_allclose(np.asarray(sigma_eff), expected, rtol=1e-12)


def test_noise_model_accepts_array_floor():
    """NoiseModel.calibration_floor accepts per-band array."""
    from tengri.observation import NoiseModel

    floor_array = jnp.array([0.02, 0.10, 0.05])
    nm = NoiseModel(calibration_floor=floor_array)
    assert jnp.array_equal(nm.calibration_floor, floor_array)


def test_noise_model_validates_array_length(simple_observation):
    """Per-band array length must match number of filters."""
    from tengri.observation import NoiseModel

    # Wrong length should raise ValueError
    floor_array = jnp.array([0.02, 0.10])  # 2 elements, but simple_observation has 3 bands
    nm = NoiseModel(calibration_floor=floor_array)

    # Validation happens when checking length
    with pytest.raises(ValueError, match="got 2, expected 3"):
        nm.validate_array_length(n_filters=3)


def test_noise_model_get_params_with_scalar_floor():
    """Scalar calibration_floor produces single noise_frac_cal parameter."""
    from tengri.observation import NoiseModel
    from tengri.parameters.priors import Fixed, Uniform

    # Scalar float
    nm_float = NoiseModel(calibration_floor=0.05)
    params = nm_float.get_params()
    assert "noise_frac_cal" in params
    assert isinstance(params["noise_frac_cal"], Fixed)

    # Scalar Distribution
    nm_dist = NoiseModel(calibration_floor=Uniform(0.01, 0.15))
    params = nm_dist.get_params()
    assert "noise_frac_cal" in params
    assert isinstance(params["noise_frac_cal"], Uniform)


def test_noise_model_get_params_with_array_floor():
    """Array calibration_floor is fixed, not added as free parameter."""
    from tengri.observation import NoiseModel

    floor_array = jnp.array([0.02, 0.10, 0.05])
    nm = NoiseModel(calibration_floor=floor_array)
    params = nm.get_params()

    # Array floors are fixed, so no noise_frac_cal parameter entry
    # (the array is used directly in compute_effective_noise)
    assert "noise_frac_cal" not in params
