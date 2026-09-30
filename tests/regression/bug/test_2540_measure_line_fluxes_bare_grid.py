# SPDX-License-Identifier: BSD-3-Clause
"""#2540: measure_line_fluxes must refuse on a grid with no nebular emission.

``predict_line_fluxes`` already refuses when the configured nebular backend
publishes no discrete line catalog (BakedIn / no backend). ``measure_line_fluxes``
instead measures the model's own rest-frame SED via continuum-subtraction, so it
silently "worked" on a bare-stellar grid paired with the BakedIn backend
(``neb={'type': 'ssp'}`` or the model's default), returning a small
stellar-continuum/absorption number indistinguishable from ``neb={'type':
'none'}`` -- with no indication the nebular emission it is meant to measure was
never there.
"""

from __future__ import annotations

import jax
import pytest

import tengri
from tengri import DEFAULT, Fixed, Observation, Photometry
from tengri.observation.line_flux_data import LineFluxData

pytestmark = pytest.mark.regression_bug


def _build(neb):
    ssp = tengri.load_ssp("fsps_prsc_miles_chabrier", download=False)
    lines = LineFluxData.from_dict({"Halpha": (1.2e-16, 0.1e-16)})
    obs = Observation(photometry=Photometry.from_names(["des_g", "des_r"]), line_fluxes=lines)
    return tengri.SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        redshift=Fixed(0.1),
        sfh={
            "type": "const",
            "all_params": Fixed(DEFAULT),
            "start_gyr": Fixed(1.0),
            "end_gyr": Fixed(0.0),
        },
        neb=neb,
    )


def test_measure_line_fluxes_refuses_when_backend_is_bakedin_on_unstamped_grid():
    """``fsps_prsc_miles_chabrier`` is 'unknown' (unstamped) status, not 'included'.

    The default shipped bare-stellar grid carries no nebular metadata stamp,
    so ``BakedInBackend`` (``neb={'type': 'ssp'}``) has no way to supply the
    emission its name implies. This must WARN, not silently succeed and not
    raise (the grid genuinely cannot be proven bare -- #2362).
    """
    import warnings as _warnings

    m = _build({"type": "ssp"})
    params = dict(m.spec.sample(jax.random.PRNGKey(0)))
    _warnings.simplefilter("always")
    with pytest.warns(UserWarning, match="contributes zero nebular flux of its own"):
        m.measure_line_fluxes(params, [tengri.observation.line_measurement.DESI_LINES[0]])


def test_measure_line_fluxes_still_works_for_cue_backend():
    """A real additive backend (Cue) must keep working, unaffected."""
    m = _build({"type": "cue", "all_params": Fixed(DEFAULT)})
    params = dict(m.spec.sample(jax.random.PRNGKey(0)))
    result = m.measure_line_fluxes(params, [tengri.observation.line_measurement.DESI_LINES[0]])
    assert result.shape == (1,)
