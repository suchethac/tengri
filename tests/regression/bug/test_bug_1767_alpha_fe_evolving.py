# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #1767 - alpha_fe_evolving=True used to raise NotImplementedError.

#1767: `alpha_fe_evolving=True` was a documented setting that could not build a model: the
stellar component accepted only a scalar met_alpha_fe. The per-age ramp is now carried by the
stellar component (alpha bracket per SSP age, #2250). The flat-kwarg form builds and predicts on
a 4-D ([alpha/Fe]-axis) grid, and a 3-D grid is refused by name. The grammar, the lookup tables
and the exact composition are covered in tests/contract/test_alpha_fe_evolving.py.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import Parameters, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.config.exceptions import ParameterError
from tengri.observation import Observation, Photometry

pytestmark = pytest.mark.regression_bug


def _grid(n_alpha: int) -> SSPData:
    """A small Z x [alpha/Fe] x age x wave grid; the alpha axis deepens a trough."""
    wave = jnp.linspace(3000.0, 9000.0, 60)
    ages = jnp.linspace(-1.0, 1.1, 5)
    base = jnp.full((3, 5, 60), 1e-3) * (1.0 + 0.5 * jnp.sin(wave / 400.0))
    nodes = jnp.linspace(0.0, 0.6, n_alpha)
    flux = jnp.stack([base * (1.0 - 0.1 * (1.0 + a)) for a in np.asarray(nodes)], axis=1)
    if n_alpha == 1:
        flux = flux[:, 0]
        return SSPData(
            ssp_wave=wave,
            ssp_flux=flux,
            ssp_lg_age_gyr=ages,
            ssp_lgmet=jnp.array([-1.5, -0.5, 0.0]),
        )
    return SSPData(
        ssp_wave=wave,
        ssp_flux=flux,
        ssp_lg_age_gyr=ages,
        ssp_lgmet=jnp.array([-1.5, -0.5, 0.0]),
        ssp_alpha_fe=nodes,
    )


def _spec(ssp: SSPData) -> Parameters:
    return Parameters(
        mean_sfh_type="delayed",
        alpha_fe_evolving=True,
        sfh_delayed_log_total_mass=10.0,
        sfh_delayed_tau_gyr=2.0,
        sfh_delayed_age_gyr=3.0,
        met_logzsol=0.0,
        met_alpha_fe_old=0.3,
        met_alpha_fe_young=0.0,
        redshift=0.1,
        dust_tau_bc=tengri.Fixed(0.0),
        dust_tau_diff=tengri.Fixed(0.0),
    )


def test_alpha_fe_evolving_builds_and_predicts_on_a_4d_grid():
    """alpha_fe_evolving=True builds on a 4-D grid and predicts finite flux (#1767)."""
    ssp = _grid(4)
    observation = Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g"]))
    model = SEDModel(_spec(ssp), ssp, observation=observation)
    flux = np.asarray(model.predict_photometry({}))
    assert flux.shape == (2,)
    assert np.all(np.isfinite(flux)) and np.all(flux > 0.0)


def test_alpha_fe_evolving_on_a_3d_grid_is_refused_by_name():
    """A 3-D grid has no [alpha/Fe] axis, so the ramp is refused by name, not silently (#1767)."""
    ssp = _grid(1)
    observation = Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g"]))
    with pytest.raises(ParameterError, match="alpha_fe_evolving=True needs"):
        SEDModel(_spec(ssp), ssp, observation=observation)
