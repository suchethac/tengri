# SPDX-License-Identifier: BSD-3-Clause
"""Regression: spectroscopy/joint fitting broke when ``predict_spectrum`` was
called without an explicit ``wave_obs``.

The inference loss (``inference/loss_functions.py:_build_prediction``) calls
``model.predict_spectrum(params)`` with no grid. Before the fix,
``predict_spectrum`` / ``predict_spectrum_components`` only resolved the grid
from ``_precomputed.spectroscopy`` or ``_wave_obs`` and otherwise raised
``ValueError("No wavelength grid")`` — even though the model was built from an
``Observation(spectroscopy=Spectroscopy(wave_obs=...))`` that carries the grid.
The JIT path (``predict_observables_jit``) already fell back to
``observation.spectroscopy.wave_obs``; the eager path did not, so every
spectroscopy-only and joint fit crashed while ``predict()`` kept working —
hiding the breakage behind stale committed notebook outputs.

Fix: ``predict_spectrum``, ``predict_spectrum_components``, and the public
``SEDModel.wave_obs`` property now fall back to the configured Observation's
spectroscopy grid. See branch ``cs/olpish-docs`` (docs polish, 2026-05-31).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.forward.sed_model import SEDModel
from tengri.observation.observation import Observation
from tengri.observation.spectroscopy import Spectroscopy
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed, Uniform

_DATA_DIR = Path(__file__).resolve().parents[2].parent / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_SSP_EXISTS = _SSP_FILE.is_file()

# Both the taxonomy marker and the data guard apply.
pytestmark = [
    pytest.mark.regression_bug,
    pytest.mark.skipif(not _SSP_EXISTS, reason="SSP data not found"),
]


@pytest.fixture(scope="module")
def ssp():
    return load_ssp_data(str(_SSP_FILE))


@pytest.fixture(scope="module")
def base_spec():
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
        sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
        met_logzsol=Fixed(0.0),
        dust_tau_bc=Uniform(0.0, 1.0),
        redshift=Fixed(0.5),
    )


@pytest.fixture(scope="module")
def wave_obs_grid():
    """100-pixel observed-frame grid (3000-7500 A rest at z=0.5)."""
    z = 0.5
    return jnp.linspace(3000.0 * (1.0 + z), 7500.0 * (1.0 + z), 100)


def test_predict_spectrum_falls_back_to_observation_grid(ssp, base_spec, wave_obs_grid):
    """``predict_spectrum(params)`` without ``wave_obs`` uses the Observation grid."""
    model = SEDModel(
        base_spec, ssp, observation=Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs_grid))
    )
    params = base_spec.sample(jax.random.PRNGKey(0))

    flux = model.predict_spectrum(params)  # used to raise ValueError
    assert flux.shape == (100,)
    assert jnp.all(jnp.isfinite(flux))


def test_predict_spectrum_components_falls_back(ssp, base_spec, wave_obs_grid):
    """``predict_spectrum_components(params)`` without ``wave_obs`` uses the Observation grid."""
    model = SEDModel(
        base_spec, ssp, observation=Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs_grid))
    )
    params = base_spec.sample(jax.random.PRNGKey(1))

    flux = model.predict_spectrum_components(params)
    assert flux.shape == (100,)
    assert jnp.all(jnp.isfinite(flux))


def test_wave_obs_property_resolves_observation_grid(ssp, base_spec, wave_obs_grid):
    """The public ``wave_obs`` accessor reports the grid the model predicts on."""
    model = SEDModel(
        base_spec, ssp, observation=Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs_grid))
    )
    assert model.wave_obs is not None
    assert jnp.array_equal(model.wave_obs, wave_obs_grid)


def test_no_spectroscopy_still_raises(ssp, base_spec):
    """The fallback is gated on ``can_do_spectroscopy``: a photometry-only model
    (no grid anywhere) still raises rather than silently inventing a grid."""
    from tengri.observation.photometry_config import Photometry

    model = SEDModel(
        base_spec,
        ssp,
        observation=Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"])),
    )
    params = base_spec.sample(jax.random.PRNGKey(2))
    with pytest.raises(ValueError, match="No wavelength grid"):
        model.predict_spectrum(params)
