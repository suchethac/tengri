#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""On a 3-D SSP grid ``met_logzsol`` is the total metallicity, with no alpha fold (#3411).

Before #3411 the ``"delta"`` branch read ``met_logzsol`` as [Fe/H] and shifted it
by 0.75 * [alpha/Fe] on a 3-D grid, so a model at ``met_logzsol = x`` and
``met_alpha_fe = 0.4`` was the same model as one at ``x + 0.3``. Both facts are
pinned here: photometry at a given ``met_logzsol`` depends only on that value,
and a 0.3 dex change in it changes the photometry. If the fold came back, the
second assertion would fail, because the metallicity would be read twice.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.parameters import Fixed
from tengri.utils.physics_constants import LOG10_ZSUN

pytestmark = pytest.mark.contract

X = -0.6


@pytest.fixture(scope="module")
def ssp3d():
    """A 3-D grid: 3 total-Z nodes, each brighter than the last, 20 ages, 300 wavelengths."""
    wave = jnp.linspace(3000.0, 10000.0, 300)
    ages = jnp.linspace(-1.0, 1.14, 20)
    base = 1e-3 * (1.0 + 0.2 * jnp.sin(jnp.linspace(0.0, 6.0, 300)))
    flux = jnp.stack([base * 10.0 ** (0.3 * m) for m in range(3)])[:, None, :] * jnp.ones(
        (1, 20, 1)
    )
    return SSPData(
        ssp_wave=wave,
        ssp_flux=flux,
        ssp_lg_age_gyr=ages,
        ssp_lgmet=jnp.array([-1.5, -0.5, 0.0]) + LOG10_ZSUN,
    )


@pytest.fixture(scope="module")
def observation():
    return tengri.Observation(photometry=tengri.Photometry.from_names(["sdss_g", "sdss_r"]))


def _photometry(ssp, observation, logzsol):
    model = SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "tau_gyr": Fixed(2.0),
            "age_gyr": Fixed(3.0),
            "log_total_mass": Fixed(10.0),
        },
        met={"logzsol": Fixed(logzsol), "alpha_fe": Fixed(0.0)},
        redshift=Fixed(0.1),
        neb={"type": "ssp"},
    )
    return np.asarray(model.predict_photometry({}), dtype=np.float64)


def test_the_same_met_logzsol_gives_the_same_photometry(ssp3d, observation):
    first = _photometry(ssp3d, observation, X)
    again = _photometry(ssp3d, observation, X)
    assert np.array_equal(first, again)


def test_a_shift_in_met_logzsol_is_not_absorbed_by_an_alpha_fold(ssp3d, observation):
    """A 0.3 dex shift (= 0.75 * 0.4) must move the photometry: no fold reads it back."""
    at_x = _photometry(ssp3d, observation, X)
    shifted = _photometry(ssp3d, observation, X + 0.75 * 0.4)
    assert not np.array_equal(at_x, shifted)
    assert np.max(np.abs(shifted - at_x) / np.abs(at_x)) > 1e-3
