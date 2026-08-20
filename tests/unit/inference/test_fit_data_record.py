# SPDX-License-Identifier: BSD-3-Clause
"""#1321: fit() accepts a Data record; bare arrays stay as sugar."""

import jax
import jax.numpy as jnp
import pytest

from tengri import FIXED, Data, ForwardModel, SEDModel
from tengri.inference import fitter as fitter_mod


@pytest.fixture
def mock_flux(synthetic_ssp_wide, synthetic_tophat_obs):
    """Generate synthetic photometry with ~5% noise using FIXED parameters."""
    sed = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "all_params": FIXED},
        dust={"law": "power_law", "type": "two_component", "all_params": FIXED},
        neb={"type": "none"},
    )
    # Get fixed parameters
    params = sed.spec.get_fixed_values()
    pred = sed.predict_photometry(params)
    noise = 0.05 * pred + 1e-15  # Add small floor
    return pred, noise


def test_bare_arrays_and_data_record_agree(synthetic_ssp_wide, synthetic_tophat_obs, mock_flux):
    """Bare arrays and Data record should produce the same fit result."""
    sed = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl"},
        dust={"law": "power_law", "type": "two_component", "all_params": FIXED},
        neb={"type": "none"},
    )
    fwd = ForwardModel.build(sed=sed)
    flux, err = mock_flux
    key = jax.random.PRNGKey(0)
    p_arrays = fwd.fit(flux, err, method="map", key=key)
    p_record = fwd.fit(Data(photometry=(flux, err)), method="map", key=key)
    # Compare the parameter dictionaries (they should match)
    for key_name in p_arrays.params:
        assert jnp.allclose(p_arrays.params[key_name], p_record.params[key_name])


def test_data_censor_reaches_data_mask(
    synthetic_ssp_wide, synthetic_tophat_obs, mock_flux, monkeypatch
):
    """Censor flags from Data should route to data_mask in Fitter."""
    seen = {}
    orig = fitter_mod.Fitter.__init__

    def spy(self, model, data=None, noise=None, **kw):
        seen.update(kw)
        return orig(self, model, data=data, noise=noise, **kw)

    monkeypatch.setattr(fitter_mod.Fitter, "__init__", spy)
    sed = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl"},
        dust={"law": "power_law", "type": "two_component", "all_params": FIXED},
        neb={"type": "none"},
    )
    fwd = ForwardModel.build(sed=sed)
    flux, err = mock_flux
    fwd.fit(
        Data(photometry=(flux, err), censor=jnp.array([0, 1, 0, 0, 0])),
        method="map",
        key=jax.random.PRNGKey(0),
    )
    assert "data_mask" in seen and seen["data_mask"] is not None


def test_data_plus_noise_kwarg_is_an_error(synthetic_ssp_wide, synthetic_tophat_obs, mock_flux):
    """Passing both Data and noise should raise TypeError."""
    sed = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl"},
        dust={"law": "power_law", "type": "two_component", "all_params": FIXED},
        neb={"type": "none"},
    )
    fwd = ForwardModel.build(sed=sed)
    flux, err = mock_flux
    with pytest.raises(TypeError, match=r"Data.*noise"):
        fwd.fit(Data(photometry=(flux, err)), err, method="map", key=jax.random.PRNGKey(0))
