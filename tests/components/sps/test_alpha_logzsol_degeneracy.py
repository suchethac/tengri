# SPDX-License-Identifier: BSD-3-Clause
"""On a 3D SSP grid, [alpha/Fe] and [Z/H] are one parameter wearing two names.

``DegenerateParameterPairWarning`` tells the user that freeing ``met_alpha_fe``
beside ``met_logzsol`` on a grid with no [alpha/Fe] axis buys nothing, because
alpha enters only as an additive shift of the effective metallicity::

    log_z_eff = met_logzsol + _ALPHA_TO_Z_COEFF * met_alpha_fe

``tests/components/sps/test_alpha_fe.py`` already pins the coefficient and the
behavior of :func:`effective_metallicity` itself. What is pinned here is the
consequence a user actually meets: that a **built model's photometry** is
exactly invariant along that direction. The two are not the same claim -- the
correction could be applied correctly by the function and still be wired into
the pipeline somewhere that breaks the invariance, and then the warning would
be telling users something untrue.

The pair of tests is the point. Flatness alone is worthless as evidence: a
probe whose filters cannot see [alpha/Fe] at all would report a perfect
invariance and prove nothing. The second test is the control arm, and it must
stay in the same file, on the same model and the same filters, so the two
cannot drift apart.

A meaningful free [alpha/Fe] needs a 4D grid carrying the axis
(``has_alpha_grid``); no such grid is bundled, which is why this file asserts
the degeneracy rather than testing around it.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import _ALPHA_TO_Z_COEFF, has_alpha_grid
from tengri.config.exceptions import DegenerateParameterPairWarning

pytestmark = pytest.mark.conservation

#: Tracked in the repository, so its absence is a real failure, not a skip.
SSP_NAME = "fsps_prsc_miles_chabrier"

#: Optical through near-IR: where an [alpha/Fe] shift actually moves a SED.
#: The control arm below proves these filters can see it.
FILTERS = ("sdss_g", "sdss_r", "sdss_i", "2mass_j")

#: Held fixed while [alpha/Fe] is traded against it.
LOGZSOL_REF = -0.3

#: Generous against float noise, and six orders below what the control arm
#: measures, so the two verdicts cannot be confused.
FLAT_TOL_PCT = 1e-8


def _build():
    ssp = tengri.load_ssp(SSP_NAME)
    assert not has_alpha_grid(ssp), (
        f"{SSP_NAME} now carries an [alpha/Fe] axis; the degeneracy this file "
        "pins applies only to 3D grids, so rewrite it against the 4D path."
    )
    obs = Observation(photometry=Photometry.from_names(list(FILTERS)))
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
        },
        met={"logzsol": Uniform(-1.0, 0.2), "alpha_fe": Uniform(-0.5, 1.0)},
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        redshift=Fixed(0.1),
    )


@pytest.fixture(scope="module")
def model_and_params():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _build()
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    return model, params


def _photometry(model, params, logzsol, alpha_fe):
    return np.asarray(
        model.predict_photometry(
            {
                **params,
                "met_logzsol": jnp.asarray(float(logzsol)),
                "met_alpha_fe": jnp.asarray(float(alpha_fe)),
            }
        )
    )


def test_build_warns_that_the_pair_is_degenerate():
    """The user is told, at build time, that the second parameter buys nothing."""
    with pytest.warns(DegenerateParameterPairWarning, match="degenerate"):
        _build()


@pytest.mark.parametrize("alpha_fe", [-0.4, -0.2, 0.2, 0.4])
def test_photometry_is_invariant_along_the_degenerate_direction(model_and_params, alpha_fe):
    """Trading alpha against logzsol at fixed log_z_eff changes no flux."""
    model, params = model_and_params
    const = LOGZSOL_REF + _ALPHA_TO_Z_COEFF * 0.0
    reference = _photometry(model, params, LOGZSOL_REF, 0.0)

    traded = _photometry(model, params, const - _ALPHA_TO_Z_COEFF * alpha_fe, alpha_fe)

    worst = float(np.max(np.abs((traded - reference) / reference))) * 100.0
    assert worst < FLAT_TOL_PCT, (
        f"photometry moved {worst:.3e}% along met_logzsol + "
        f"{_ALPHA_TO_Z_COEFF} * met_alpha_fe = {const}, which "
        "DegenerateParameterPairWarning promises is flat"
    )


@pytest.mark.parametrize(("alpha_fe", "floor_pct"), [(-0.2, 1.0), (0.2, 1.0), (0.4, 5.0)])
def test_photometry_moves_across_the_degenerate_direction(model_and_params, alpha_fe, floor_pct):
    """Control arm: these filters DO see [alpha/Fe], so the flatness means something."""
    model, params = model_and_params
    reference = _photometry(model, params, LOGZSOL_REF, 0.0)

    moved = _photometry(model, params, LOGZSOL_REF, alpha_fe)

    worst = float(np.max(np.abs((moved - reference) / reference))) * 100.0
    assert worst > floor_pct, (
        f"[alpha/Fe]={alpha_fe} moved photometry only {worst:.4f}%, below the "
        f"{floor_pct}% floor. These filters can no longer see the parameter, so "
        "the invariance test above proves nothing."
    )
