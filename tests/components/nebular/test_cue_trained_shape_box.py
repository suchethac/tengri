# SPDX-License-Identifier: BSD-3-Clause
"""Cue's ionizing-shape parameters are held inside its training box and warned about.

Li et al. 2025 (ApJ 986, 9, arXiv:2405.04598), Table 1, lists the range each of
the seven ionizing-shape parameters was trained on (the training set is drawn
uniformly inside it). tengri clamps the SSP-derived shape to that box before the
network with a straight-through clamp and warns at build when an SSP's young
composite shape leaves it. Tolerance: the table values are exact; in-range
output is bit-identical (``==``).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import load_ssp_data
from tengri.components.nebular.cue import CueBackend
from tengri.components.nebular.ionizing_spectrum import (
    _CLIP_RANGES,
    clamp_ionspec_straight_through,
)
from tengri.config.exceptions import CueIonizingShapeWarning

pytestmark = pytest.mark.regression_paper

_BARE = "data/fsps_prsc_miles_chabrier.h5"
_CUE_WEIGHTS = "data/cue_weights.npz"

# Li et al. 2025, Table 1, rows 1-7 (alpha_HeII .. log F_HI/F_HeI).
_TABLE_1 = {
    "ionspec_index1": (1.0, 42.0),
    "ionspec_index2": (-0.3, 30.0),
    "ionspec_index3": (-1.1, 14.0),
    "ionspec_index4": (-1.7, 8.0),
    "ionspec_logLratio1": (-0.1, 10.1),
    "ionspec_logLratio2": (-0.5, 1.9),
    "ionspec_logLratio3": (-0.4, 2.2),
}


def _ssp():
    if not Path(_BARE).is_file():
        pytest.skip(f"missing bare SSP {_BARE}")
    return load_ssp_data(_BARE)


def _shape_warnings(ssp):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        be = CueBackend(_CUE_WEIGHTS, ssp_data=ssp)
    return be, [w for w in caught if issubclass(w.category, CueIonizingShapeWarning)]


def test_clip_ranges_equal_li2025_table1():
    """The clamp bounds are the published training range, all seven."""
    assert dict(_CLIP_RANGES) == _TABLE_1


def test_clamp_in_range_is_bit_identical_with_unit_gradient():
    """Inside the box value and gradient pass through unchanged."""
    mid = jnp.asarray([(lo + hi) / 2 for lo, hi in _TABLE_1.values()])
    np.testing.assert_array_equal(np.asarray(clamp_ionspec_straight_through(mid)), np.asarray(mid))
    g = jax.grad(lambda x: jnp.sum(clamp_ionspec_straight_through(x) * jnp.arange(1.0, 8.0)))(mid)
    np.testing.assert_array_equal(np.asarray(g), np.arange(1.0, 8.0))


def test_clamp_out_of_range_moves_to_bound_and_keeps_gradient():
    """Outside the box the value is the bound; the gradient is not zeroed."""
    lo = jnp.asarray([v[0] for v in _TABLE_1.values()])
    hi = jnp.asarray([v[1] for v in _TABLE_1.values()])
    np.testing.assert_array_equal(
        np.asarray(clamp_ionspec_straight_through(hi + 5.0)), np.asarray(hi)
    )
    np.testing.assert_array_equal(
        np.asarray(clamp_ionspec_straight_through(lo - 5.0)), np.asarray(lo)
    )
    g = jax.grad(lambda x: jnp.sum(clamp_ionspec_straight_through(x)))(hi + 5.0)
    np.testing.assert_array_equal(np.asarray(g), np.ones(7))


def test_dense_fsps_does_not_warn_and_weighted_shape_is_unclamped():
    """A densely sampled FSPS Lyman continuum is in range: no warning, no change."""
    ssp = _ssp()
    be, caught = _shape_warnings(ssp)
    assert caught == []
    ages = np.asarray(be._ssp_log_age_yr)
    w = jnp.asarray(np.where(ages <= 7.5, np.gradient(10.0**ages), 0.0))
    kw = dict(neb_logU=-2.0)
    raw = be._weighted_cue_params_raw(w, jnp.asarray(ages), -1.7, **kw)
    out = be._compute_weighted_cue_params(w, jnp.asarray(ages), -1.7, **kw)
    for k in _TABLE_1:
        assert float(out[k]) == float(raw[k])


def test_out_of_box_shape_warns_and_is_clamped():
    """A Lyman continuum with no HeII/OII flux leaves the box: warn, then clamp."""
    ssp = _ssp()
    wave = np.asarray(ssp.ssp_wave)
    flux = np.array(ssp.ssp_flux)
    flux[..., wave < 504.0] *= 1e-40  # segments 1-2 vanish: logLratio3 explodes
    be, caught = _shape_warnings(ssp._replace(ssp_flux=jnp.asarray(flux)))
    assert len(caught) == 1
    msg = str(caught[0].message)
    assert "ionspec_logLratio3" in msg
    assert "[-0.4, 2.2]" in msg
    assert "fsps_mist_c3k_a_chabrier" in msg
    ages = np.asarray(be._ssp_log_age_yr)
    w = jnp.asarray(np.where(ages <= 7.5, np.gradient(10.0**ages), 0.0))
    raw = be._weighted_cue_params_raw(w, jnp.asarray(ages), -1.7)
    out = be._compute_weighted_cue_params(w, jnp.asarray(ages), -1.7)
    assert float(raw["ionspec_logLratio3"]) > _TABLE_1["ionspec_logLratio3"][1]
    assert float(out["ionspec_logLratio3"]) == _TABLE_1["ionspec_logLratio3"][1]
