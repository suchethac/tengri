# SPDX-License-Identifier: BSD-3-Clause
"""Contract test for #2296: params dicts refuse Fixed keys.

https://github.com/suchethac/tengri/issues/2296
"""

import pytest
import jax.random as jr

from tengri import SEDModel, FREE, Fixed
from tengri.config.exceptions import ParameterError


pytestmark = pytest.mark.contract


@pytest.fixture
def model_with_fixed_redshift(synthetic_ssp):
    """Build a forward-only model with Fixed redshift and FREE SFH."""
    return SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=None,
        sfh={'type': 'dpl', 'alpha': FREE},
        redshift=Fixed(0.05),
    )


def test_fixed_key_in_params_raises(model_with_fixed_redshift):
    """(a) A params dict containing a Fixed key raises ParameterError naming the key and pin."""
    model = model_with_fixed_redshift

    # Build params dict with a Fixed key (redshift is Fixed at 0.05)
    free_params = {name: 0.5 for name in model.spec.free_params}
    params_with_fixed = {**free_params, "redshift": 0.05}

    # predict should raise ParameterError
    with pytest.raises(ParameterError) as exc_info:
        model.predict(params_with_fixed)

    error_msg = str(exc_info.value)
    assert "redshift" in error_msg
    assert "Fixed" in error_msg or "pinned" in error_msg


def test_fixed_key_omitted_works(model_with_fixed_redshift):
    """(b) Params dict without the Fixed key works fine."""
    model = model_with_fixed_redshift

    # Build params dict with only free keys
    free_params = {name: 0.5 for name in model.spec.free_params}

    # Should not raise
    output = model.predict(free_params)
    assert output is not None


def test_spec_sample_returns_only_free_keys(model_with_fixed_redshift):
    """(c) spec.sample() returns only free parameters."""
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


