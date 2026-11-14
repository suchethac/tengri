#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""On a 3-D SSP grid ``met_logzsol`` is the total metallicity, with no alpha fold (#3411).

The build refuses a non-zero ``met_alpha_fe`` on a grid without an [alpha/Fe] axis,
so the refusal alone cannot show that the stellar component ignores the value. These
tests hand [alpha/Fe] = 0.4 to the two places that compute the metallicity weights,
past the refusal: the compiled forward (through its fixed-value argument) and
``compute_joint_weights``. Each must give the alpha = 0 result bit for bit. A term
``0.75 * met_alpha_fe`` added to the metallicity in either place moves the photometry
by a factor 1.54 on this grid.
"""

from __future__ import annotations

import jax
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
ALPHA = 0.4


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


def _model(ssp, observation):
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "tau_gyr": Fixed(2.0),
            "age_gyr": Fixed(3.0),
            "log_total_mass": Fixed(10.0),
        },
        met={"logzsol": Fixed(X), "alpha_fe": Fixed(0.0)},
        redshift=Fixed(0.1),
        neb={"type": "ssp"},
    )


def _forward(model, **fixed_overrides):
    """Photometry from the compiled forward with some fixed values replaced."""
    fixed = dict(model.spec.get_fixed_values())
    fixed.update({name: jnp.asarray(value) for name, value in fixed_overrides.items()})
    kernel = model._get_or_build_predict_observables_jit()
    out = kernel(
        {}, fixed, model.ssp_data, model._template_data_for_jit(), model._ztable_data_for_jit()
    )
    return np.asarray(getattr(out, "photometry", out), dtype=np.float64)


def test_alpha_reaching_apply_does_not_shift_the_metallicity(ssp3d, observation):
    model = _model(ssp3d, observation)
    assert np.array_equal(_forward(model, met_alpha_fe=ALPHA), _forward(model))


def test_alpha_reaching_the_joint_weights_does_not_shift_the_metallicity(ssp3d, observation):
    model = _model(ssp3d, observation)
    stellar = model._build_component_chain()[0]
    fixed = dict(model.spec.get_fixed_values())
    at_zero = stellar.compute_joint_weights(fixed, ssp3d)
    enhanced = stellar.compute_joint_weights({**fixed, "met_alpha_fe": jnp.asarray(ALPHA)}, ssp3d)
    for a, b in zip(jax.tree_util.tree_leaves(enhanced), jax.tree_util.tree_leaves(at_zero)):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_the_metallicity_the_fold_would_have_reached_gives_other_photometry(ssp3d, observation):
    """The control: this grid tells ``met_logzsol`` from ``met_logzsol + 0.75 * 0.4``."""
    model = _model(ssp3d, observation)
    shifted = _forward(model, met_logzsol=X + 0.75 * ALPHA)
    assert not np.allclose(shifted, _forward(model), rtol=1e-3, atol=0.0)
