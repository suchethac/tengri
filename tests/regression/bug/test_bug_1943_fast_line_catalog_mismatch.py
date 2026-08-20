# SPDX-License-Identifier: BSD-3-Clause
"""The fast line branch must not pair one catalog's waves with another's lums (#1943).

``SEDModel.predict_line_fluxes`` takes its fast branch whenever a nebular grid
table exists, and sets **both** arrays from the grid::

    all_waves = jnp.asarray(grid.wavelengths)  # the requested targets
    all_lums = reconstruct_nebular_line_lums(nion, params, grid)

The shared redden tail (#1877) then replaces ``all_lums`` with the state's
published ``log_line_lums_attenuated`` — and leaves ``all_waves`` alone. That
array is indexed on the backend's FULL catalog (128 entries for Cue), so the
target match walks the 3-entry grid axis and reads the first three entries of
the 128-entry catalog: the far-UV lines at 923-937 A, returned under the labels
Halpha / Hbeta / [OIII].

Measured on this fixture: Halpha ``9.15e-12`` against ``3.96e-16``, a constant
**23,093x**, which closes against the catalog mismatch itself
(``10**atten[62] / 10**atten[0]`` = 2.479e44 / 1.074e40 = 23,090).

``tolerance_aa`` cannot catch it: the *wavelengths* match the targets exactly.
Only the luminosities come from the wrong array — which is why this shipped.

**The dust-free control is not optional.** Without dust the nebular component
takes its fast grid branch, publishes no attenuated catalog, and the fallback
screen runs against the grid's own axis — correctly paired. A test that only
exercised the dust-free arm would pass against the defect.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

_LINES = ("Halpha", "Hbeta", "OIII_5007")
_WAVES = jnp.array([6564.61, 4862.71, 5008.24])

#: Tolerance is *derived*, not read off the failure. The fast and exact paths are
#: not bit-identical even when correct -- the grid is an interpolant -- and the
#: builder's own warning bounds that at ~1.3 % worst-case for the collisionally
#: excited lines. 2 % clears that ceiling with margin while sitting 50x below the
#: defect's 99.996 % deviation (a factor 2.3e4 in the flux), so this threshold
#: cannot be satisfied by the bug and does not fail on honest interpolation error.
_TOL_PCT = 2.0

_SSP_CANDIDATES = ["data/fsps_prsc_miles_chabrier.h5", "data/ssp_prsc_bc03_chabrier.h5"]


def _ssp_path():
    return next((p for p in _SSP_CANDIDATES if Path(p).is_file()), None)


@pytest.fixture(scope="module")
def ssp():
    from tengri import load_ssp_data

    path = _ssp_path()
    if path is None or not Path("data/cue_weights.npz").is_file():
        pytest.skip("No bare-stellar SSP / Cue weights available.")
    return load_ssp_data(path)


def _build(ssp, approx, obs, *, dust_attenuation: bool):
    """A dusty (or not) Cue model over ``obs``."""
    from tengri import FIXED, FREE, Fixed, SEDModel

    dust_block = (
        {
            "type": "two_component",
            "law": "calzetti",
            "all_params": FIXED,
        }
        if dust_attenuation
        else {"type": "none"}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust_attenuation=dust_block,
            neb={"type": "cue", "all_params": FIXED},
            redshift=Fixed(0.1),
            approx=approx,
        )


def _phot_obs():
    from tengri import Observation, Photometry

    return Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]))


@pytest.fixture(scope="module")
def fixtures(ssp):
    """(observation carrying a line channel, params) per dust setting.

    The grid's wavelength axis is populated FROM the observation's line channel,
    so a model built without one has an empty grid and
    ``predict_line_fluxes`` raises ``argmin of an empty sequence`` — a real
    failure, but not the one this file is about. The fixture therefore carries
    the line channel, which is also what the shipped fit does.
    """
    from tengri import Observation
    from tengri.observation.line_flux_data import LineFluxData

    out = {}
    for dust in (True, False):
        base = _build(ssp, None, _phot_obs(), dust_attenuation=dust)
        params = base.spec.sample(jax.random.PRNGKey(0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lf = np.asarray(base.measure_line_fluxes(params, approx=False))[:3]
        obs = Observation(
            photometry=_phot_obs().photometry,
            line_fluxes=LineFluxData(
                names=_LINES,
                fluxes=jnp.asarray(lf),
                errors=jnp.asarray(np.abs(lf) * 0.05 + 1e-30),
                wavelengths=_WAVES,
            ),
        )
        out[dust] = (obs, params)
    return out


def _lut_model(ssp, obs, *, dust_attenuation: bool):
    """The model a line fit resolves to under the default ``approx='auto'``."""
    from tengri import FeaturePrecomp

    return _build(ssp, FeaturePrecomp(), obs, dust_attenuation=dust_attenuation)


def _fluxes(model, params, *, state):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(model.predict_line_fluxes(params, _WAVES, state=state))


@pytest.mark.parametrize("dust", [True, False])
def test_a_supplied_state_does_not_change_the_line_fluxes(ssp, fixtures, dust):
    """``state=`` is a performance argument, not a physics one.

    ``loss_functions._build_prediction`` passes a state whenever any feature
    channel needs the full forward. Passing it must not change the answer -- it
    exists so the forward runs once, not so the result differs.
    """
    obs, params = fixtures[dust]
    model = _lut_model(ssp, obs, dust_attenuation=dust)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        state = model.predict_state(params)

    stateless = _fluxes(model, params, state=None)
    stateful = _fluxes(model, params, state=state)

    dev = np.abs((stateful - stateless) / np.abs(stateless)) * 100.0
    worst = int(np.argmax(dev))
    assert dev[worst] < _TOL_PCT, (
        f"passing a state changed the line fluxes: {_LINES[worst]} "
        f"{stateless[worst]:.6e} -> {stateful[worst]:.6e} ({dev[worst]:.4f}%). "
        "The fast branch sets all_waves from the grid and the redden tail then "
        "takes all_lums from the state's full catalog (#1943)."
    )


@pytest.mark.parametrize("dust", [True, False])
def test_the_lut_line_fluxes_match_the_exact_path(ssp, fixtures, dust):
    """And the value a supplied state produces must be the exact one.

    Guards the direction the previous test cannot: both arms agreeing on the
    *wrong* number would satisfy it.
    """
    obs, params = fixtures[dust]
    exact_m = _build(ssp, None, obs, dust_attenuation=dust)
    lut_m = _lut_model(ssp, obs, dust_attenuation=dust)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exact = _fluxes(exact_m, params, state=exact_m.predict_state(params))
        lut = _fluxes(lut_m, params, state=lut_m.predict_state(params))

    dev = np.abs((lut - exact) / np.abs(exact)) * 100.0
    worst = int(np.argmax(dev))
    assert dev[worst] < _TOL_PCT, (
        f"the LUT path disagrees with exact on {_LINES[worst]}: "
        f"{exact[worst]:.6e} -> {lut[worst]:.6e} ({dev[worst]:.4f}%)"
    )


def test_the_two_catalogs_really_do_differ_in_length(ssp, fixtures):
    """Pins the precondition, so a future refactor cannot make this file vacuous.

    If the grid axis and the backend catalog ever become the same length, the
    two tests above would pass for a reason unrelated to the fix, and this one
    fails to say so.
    """
    obs, params = fixtures[True]
    model = _lut_model(ssp, obs, dust_attenuation=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        state = model.predict_state(params)

    grid_waves = np.asarray(model._nebular_grid_table.wavelengths)
    state_waves = np.asarray(state.derived["line_waves"])
    assert grid_waves.size != state_waves.size, (
        "grid axis and backend catalog now have the same length; the mismatch "
        "this file guards can no longer be expressed by index confusion, so "
        "re-derive the guard rather than deleting it"
    )
    assert "log_line_lums_attenuated" in state.derived, (
        "a dusty chain stopped publishing the attenuated catalog; the mixing "
        "this file guards is unreachable and the fixture needs revisiting"
    )
