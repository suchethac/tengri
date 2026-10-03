# SPDX-License-Identifier: BSD-3-Clause
"""Contract: ``generate_mock(noise=...)`` draws from a catalog's observed errors (#2628, item 6).

CIGALE's ``mock_flag`` replaces each band with a draw from
``N(best-fit flux, |observed error|)`` (pcigale ``managers/observations.py``,
``generate_mock``); ``generate_mock(model, params, key, noise=sigma_obs)`` does the
same for tengri's forward model, where ``snr`` instead sets ``sigma = flux / snr``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from tengri import DEFAULT, Fixed, SEDModel
from tengri.analysis.mock import generate_mock

pytestmark = pytest.mark.contract

_FLUX = jnp.array([2.0e-29, 5.0e-29, 1.0e-28, 3.0e-28])
_SIGMA_OBS = jnp.array([1.0e-30, -4.0e-30, 2.0e-29, 5.0e-30])  # one negative (CIGALE limit sign)


class _FixedFluxModel:
    """The documented ``generate_mock`` contract: any object with ``predict_photometry``."""

    def predict_photometry(self, params):
        return _FLUX


def test_noise_is_the_absolute_observed_error_and_replaces_snr():
    mock = generate_mock(
        _FixedFluxModel(), {}, key=jax.random.PRNGKey(0), snr=3.0, noise=_SIGMA_OBS
    )
    np.testing.assert_allclose(mock["noise"], np.abs(np.asarray(_SIGMA_OBS)), rtol=0, atol=0)
    np.testing.assert_allclose(mock["flux_true"], np.asarray(_FLUX), rtol=0, atol=0)


def test_draw_formula_is_exact():
    """A fixed key draws flux_obs = flux_true + |sigma_obs| * N(0,1) exactly."""
    key = jax.random.PRNGKey(42)
    mock = generate_mock(_FixedFluxModel(), {}, key=key, noise=_SIGMA_OBS)
    expected = np.asarray(_FLUX) + np.abs(np.asarray(_SIGMA_OBS)) * np.asarray(
        jax.random.normal(key, shape=np.asarray(_FLUX).shape)
    )
    np.testing.assert_allclose(mock["flux_obs"], expected, rtol=1e-12, atol=0)


def test_draws_have_the_observed_standard_deviation():
    """1000 draws per band: (flux_obs - flux_true) / sigma_obs is N(0, 1) (Kolmogorov-Smirnov)."""
    keys = jax.random.split(jax.random.PRNGKey(7), 1000)
    draws = jax.vmap(
        lambda k: generate_mock(_FixedFluxModel(), {}, key=k, noise=_SIGMA_OBS)["flux_obs"]
    )(keys)
    z = (np.asarray(draws) - np.asarray(_FLUX)) / np.abs(np.asarray(_SIGMA_OBS))
    for band in range(z.shape[1]):
        assert stats.kstest(z[:, band], "norm").pvalue > 0.01
        assert abs(z[:, band].std() - 1.0) < 0.1


def test_default_snr_path_is_unchanged():
    mock = generate_mock(_FixedFluxModel(), {}, key=jax.random.PRNGKey(0), snr=20.0)
    np.testing.assert_allclose(mock["noise"], np.asarray(_FLUX) / 20.0, rtol=1e-12)


def test_noise_shape_mismatch_is_refused():
    with pytest.raises(ValueError, match="noise has shape"):
        generate_mock(_FixedFluxModel(), {}, key=jax.random.PRNGKey(0), noise=jnp.ones(3))


def test_noise_shape_mismatch_same_size_different_shape_is_refused():
    """Shape (n_bands, 1) has the same size as (n_bands,) but different shape."""
    with pytest.raises(ValueError, match="noise has shape"):
        generate_mock(_FixedFluxModel(), {}, key=jax.random.PRNGKey(0), noise=jnp.ones((4, 1)))


def test_observed_noise_through_a_real_forward_model(synthetic_ssp_wide, synthetic_tophat_obs):
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
        neb={"type": "none"},
        redshift=Fixed(0.1),
    )
    truth = model.predict_photometry({})
    sigma = 0.05 * jnp.abs(truth)
    mock = generate_mock(model, {}, key=jax.random.PRNGKey(3), noise=sigma)
    np.testing.assert_allclose(mock["flux_true"], np.asarray(truth), rtol=1e-12)
    np.testing.assert_allclose(mock["noise"], np.asarray(sigma), rtol=1e-12)
    assert mock["flux_obs"].shape == truth.shape
    assert not np.array_equal(np.asarray(mock["flux_obs"]), np.asarray(truth))
