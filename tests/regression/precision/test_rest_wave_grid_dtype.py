# SPDX-License-Identifier: BSD-3-Clause
r"""The rest-wavelength grid must carry the session's precision (#1206, #1439).

Thirteen sites in ``components/`` decide which arithmetic to run by asking
``wave.dtype == jnp.float32`` — the AGN discs (x6), X-ray (x2), radio, the
nebular shock model, and others. Each has a float32-safe log-domain branch and
a plain float64 branch, and each picks between them from that one dtype.

``SEDModel`` built the grid two ways: ``make_union_grid`` when some component
contributes its own wavelength coverage, and otherwise ``ssp_data.ssp_wave``
verbatim — which is the float64 array the HDF5 loader produced, regardless of
``jax.enable_x64(False)``. So in a pure-float32 session the grid's dtype
depended on **which components happened to be in the model**, and with a
float64 grid every one of those thirteen gates failed *open* together: the
float64 branch ran while the arithmetic was float32.

Measured before the fix: a composable AGN with a ``multicolor`` disc and no
torus (nothing contributes a wing, so the grid stayed float64) evaluated the
disc at the true ``10**11 * L_sun`` = 3.8e44 — past float32's 3.4e38 — and
returned ``sed_agn`` non-finite at **all 5994** grid points, poisoning
``sed_intrinsic`` and every band. Adding a SKIRTOR torus forced a union grid
and the identical model was clean.

That configuration-dependence is why no float32 test caught it:
``test_agn_disc_float32_inventory`` builds every one of its twelve discs with
``torus="skirtor"``, so the torus-less path — the minimal AGN a user writes —
had never been measured.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform

pytestmark = pytest.mark.regression_bug

_SFH = {
    "type": "delayed",
    "all_params": FIXED,
    "log_total_mass": Uniform(9.0, 11.0),
    "tau_gyr": 1.0,
    "age_gyr": 5.0,
}
_DUST = {
    "type": "two_component",
    "law": "calzetti",
    "all_params": FIXED,
    "tau_diff": 0.3,
    "tau_bc": 0.0,
}
_DISC = {"type": "multicolor", "all_params": FIXED}


def _params(dtype):
    return {
        "sfh_delayed_log_total_mass": jnp.asarray(10.0, dtype=dtype),
        "agn_log_lbol": jnp.asarray(11.0, dtype=dtype),
    }


def _model(ssp, agn=None):
    obs = Observation(photometry=Photometry.from_names(["sdss_r", "wise_w1"]))
    groups = dict(sfh=_SFH, dust_attenuation=_DUST)
    if agn is not None:
        groups["agn"] = agn
    return SEDModel.build(ssp_data=ssp, observation=obs, redshift=Fixed(0.1), **groups)


#: With a torus the grid is rebuilt by ``make_union_grid`` (which already
#: honored the session precision); without one it used to come straight from
#: the SSP. Both must now agree, which is the point of parametrizing.
_AGN_CASES = {
    "no_torus": {
        "type": "composable",
        "all_params": FIXED,
        "disc": _DISC,
        "log_lbol": Uniform(9.0, 12.0),
    },
    "skirtor": {
        "type": "composable",
        "all_params": FIXED,
        "disc": _DISC,
        "torus": {"type": "skirtor", "all_params": FIXED},
        "log_lbol": Uniform(9.0, 12.0),
    },
}


@pytest.mark.parametrize("agn_key", [None, *sorted(_AGN_CASES)])
def test_rest_grid_is_float32_in_a_float32_session(ssp_bare, agn_key):
    """The grid's dtype must follow the session, not the component list."""
    agn = None if agn_key is None else _AGN_CASES[agn_key]
    with jax.enable_x64(False):
        wave = _model(ssp_bare, agn)._rest_wavelength
    assert wave.dtype == jnp.float32, (
        f"rest grid is {wave.dtype} in a pure-float32 session (agn={agn_key}) — "
        "every `wave.dtype == jnp.float32` gate in components/ will fail open"
    )


@pytest.mark.parametrize("agn_key", [None, *sorted(_AGN_CASES)])
def test_rest_grid_is_float64_under_x64(ssp_bare, agn_key):
    """The other direction: x64 must still get a float64 grid (no behavior change)."""
    agn = None if agn_key is None else _AGN_CASES[agn_key]
    with jax.enable_x64(True):
        wave = _model(ssp_bare, agn)._rest_wavelength
    assert wave.dtype == jnp.float64, f"rest grid is {wave.dtype} under x64"


@pytest.mark.parametrize("agn_key", sorted(_AGN_CASES))
def test_agn_sed_is_finite_in_float32_with_and_without_a_torus(ssp_bare, agn_key):
    """The defect the dtype caused, pinned end-to-end.

    Asserts float64 finiteness first: if the reference were already broken this
    comparison would pass for the wrong reason.
    """
    agn = _AGN_CASES[agn_key]
    with jax.enable_x64(True):
        ref = np.asarray(
            _model(ssp_bare, agn).predict_state(_params(jnp.float64)).derived["sed_agn"],
            dtype=np.float64,
        )
    assert np.all(np.isfinite(ref)), f"setup: float64 sed_agn is not finite ({agn_key})"

    with jax.enable_x64(False):
        got = np.asarray(
            _model(ssp_bare, agn).predict_state(_params(jnp.float32)).derived["sed_agn"],
            dtype=np.float64,
        )

    n_bad = int((~np.isfinite(got)).sum())
    assert n_bad == 0, (
        f"{n_bad}/{got.size} sed_agn points are non-finite in pure float32 "
        f"(agn={agn_key}) while float64 is clean — a dtype gate fell through "
        "to the float64 branch (#1439)"
    )
    bright = np.abs(ref) > 1e-6 * np.abs(ref).max()
    rel = np.abs(got[bright] - ref[bright]) / np.abs(ref[bright])
    assert rel.max() < 1e-2, f"float32 sed_agn disagrees with float64 by {rel.max():.2e}"
