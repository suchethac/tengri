# SPDX-License-Identifier: BSD-3-Clause
"""Contract test for #2296: params dicts refuse Fixed keys.

https://github.com/suchethac/tengri/issues/2296
"""

import pytest
import jax.random as jr
import numpy as np

from tengri import SEDModel, FREE, Fixed
from tengri.config.exceptions import ParameterError
from tengri.inference.catalog import Catalog


pytestmark = pytest.mark.contract


@pytest.fixture
def model_with_fixed_redshift(synthetic_ssp, synthetic_obs):
    """Build a model with Fixed redshift and FREE parameters."""
    return SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=synthetic_obs,
        sfh={'type': 'dpl', 'alpha': FREE, 'beta': FREE},
        redshift=Fixed(0.05),
    )


@pytest.fixture
def free_params_dict(model_with_fixed_redshift):
    """Build a dict with only free parameters."""
    return {name: 0.5 for name in model_with_fixed_redshift.spec.free_params}


@pytest.fixture
def params_with_fixed_override(free_params_dict):
    """Build a dict with a Fixed key override."""
    return {**free_params_dict, "redshift": 0.05}


@pytest.mark.parametrize("method_name", [
    "predict",
    "predict_photometry",
    "predict_properties",
    "predict_line_fluxes",
    "predict_state",
])
def test_fixed_key_refused_by_entry_points(model_with_fixed_redshift, free_params_dict, params_with_fixed_override, method_name):
    """Each entry point refuses Fixed keys in params."""
    model = model_with_fixed_redshift
    method = getattr(model, method_name)

    # Free-only params should work
    try:
        if method_name == "predict_properties":
            # predict_properties requires names parameter
            result = method(free_params_dict, names=["stellar_mass"])
        else:
            result = method(free_params_dict)
        assert result is not None, f"{method_name} with free params should succeed"
    except TypeError as e:
        # Some methods may not exist on this model configuration
        if "not callable" in str(e) or "has no attribute" in str(e):
            pytest.skip(f"{method_name} not available on this model")
        raise

    # Fixed key override should raise ParameterError
    with pytest.raises(ParameterError) as exc_info:
        if method_name == "predict_properties":
            method(params_with_fixed_override, names=["stellar_mass"])
        else:
            method(params_with_fixed_override)

    error_msg = str(exc_info.value)
    assert "redshift" in error_msg, f"Error should name the Fixed key 'redshift'"
    assert ("#2296" in error_msg or "Fixed" in error_msg or "overrides" in error_msg), \
        f"Error should mention Fixed override (got: {error_msg})"


def test_catalog_predict_refuses_fixed_keys(model_with_fixed_redshift, free_params_dict, params_with_fixed_override):
    """Catalog.predict refuses Fixed keys in param_table."""
    model = model_with_fixed_redshift

    # Create a catalog
    cat = Catalog(model, [])

    # Create param_table with correct shape (N,) for each param
    free_table = {name: np.full((10,), 0.5) for name in free_params_dict.keys()}

    # Free-only params should work
    result = cat.predict(free_table)
    assert result is not None
    assert result.shape == (10, len(model.observation.photometry.filters))

    # Fixed key override should raise ParameterError
    fixed_table = {name: np.full((10,), val) for name, val in params_with_fixed_override.items()}

    with pytest.raises(ParameterError) as exc_info:
        cat.predict(fixed_table)

    error_msg = str(exc_info.value)
    assert "redshift" in error_msg
    assert ("#2296" in error_msg or "Fixed" in error_msg or "overrides" in error_msg)


def test_fixed_key_omitted_works(model_with_fixed_redshift, free_params_dict):
    """Params dict without the Fixed key works fine."""
    model = model_with_fixed_redshift

    # Should not raise
    output = model.predict(free_params_dict)
    assert output is not None


def test_spec_sample_returns_only_free_keys(model_with_fixed_redshift):
    """spec.sample() returns only free parameters."""
    model = model_with_fixed_redshift

    sampled = model.spec.sample(jr.PRNGKey(0))

    # All sampled keys should be free
    for key in sampled.keys():
        assert key in model.spec.free_params, f"Sampled key {key} is not free"

    # All free params should be in the sample
    for free_key in model.spec.free_params:
        assert free_key in sampled, f"Free param {free_key} missing from sample"

    # Should be able to feed sampled dict directly into predict
    output = model.predict(sampled)
    assert output is not None


