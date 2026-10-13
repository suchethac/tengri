# SPDX-License-Identifier: BSD-3-Clause
"""Cue's ionizing-shape parameters are held inside its training box and warned about.

The Cue reference code (github.com/efburnham/cue, ``emulator.py`` docstring and
``utils.py`` clip bounds) gives the range of each of the seven ionizing-shape
parameters. It equals Li et al. 2025 (ApJ 986, 9, arXiv:2405.04598), Table 1,
except index3 [-1, 14] (paper -1.1) and logLratio1 [-1, 10.1] (paper -0.1); the
paper's own AGN coverage (alpha 1.2-2, logLratio1 -0.86 to -0.26) needs the code
values. tengri clamps the SSP-derived shape to that box before the
network with a straight-through clamp and warns at build when an SSP's young
composite shape leaves it. Tolerance: the table values are exact; in-range
output is bit-identical (``==``).
"""

from __future__ import annotations

import itertools
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

# Cue reference code bounds (see module docstring for the two Table 1 differences).
_TABLE_1 = {
    "ionspec_index1": (1.0, 42.0),
    "ionspec_index2": (-0.3, 30.0),
    "ionspec_index3": (-1.0, 14.0),
    "ionspec_index4": (-1.7, 8.0),
    "ionspec_logLratio1": (-1.0, 10.1),
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


def test_clip_ranges_equal_cue_reference_bounds():
    """The clamp bounds are the Cue reference code's training range, all seven."""
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
    raw = be._weighted_cue_params_raw(w, jnp.asarray(ages), -1.7)
    out = be._compute_weighted_cue_params(w, jnp.asarray(ages), -1.7, **kw)
    for k in _TABLE_1:
        assert float(out[k]) == float(raw[k])
    # neb_logU reaches Cue as gas_logu through the clamped wrapper
    assert float(out["gas_logu"]) == kw["neb_logU"]


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


def test_clamp_non_finite_input():
    """+/-inf lands on the bound, nan stays nan (a plain clip's behavior)."""
    lo = np.asarray([v[0] for v in _TABLE_1.values()])
    hi = np.asarray([v[1] for v in _TABLE_1.values()])
    np.testing.assert_array_equal(
        np.asarray(clamp_ionspec_straight_through(jnp.full(7, jnp.inf))), hi
    )
    np.testing.assert_array_equal(
        np.asarray(clamp_ionspec_straight_through(jnp.full(7, -jnp.inf))), lo
    )
    assert np.all(np.isnan(np.asarray(clamp_ionspec_straight_through(jnp.full(7, jnp.nan)))))


def test_warning_is_emitted_once_per_ssp():
    """A repeated build of the same out-of-box SSP does not repeat the warning."""
    ssp = _ssp()
    flux = np.array(ssp.ssp_flux)
    flux[..., np.asarray(ssp.ssp_wave) < 504.0] *= 1e-39  # distinct fingerprint
    bad = ssp._replace(ssp_flux=jnp.asarray(flux))
    assert len(_shape_warnings(bad)[1]) == 1
    assert len(_shape_warnings(bad)[1]) == 0


def test_agn_lines_monotonic_in_alpha_pl():
    """AGN [OIII]/Hb and HeII/Hb vary monotonically with alpha_pl in -1.2..-2.5.

    Guards the logLratio1 floor: at -0.1 the whole -1.2..-2.35 range collapsed
    to one value and HeII/Hb went flat then non-monotonic. Hard spectra
    (shallower alpha_pl) give stronger high-ionization lines.
    """
    from tengri.components.nebular.agn_nebular import agn_ionspec_from_alpha_pl

    alphas = [-1.2, -1.5, -1.7, -2.0, -2.5]
    lr1 = [float(agn_ionspec_from_alpha_pl(a)["ionspec_logLratio1"]) for a in alphas]
    assert all(b > a for a, b in itertools.pairwise(lr1)), lr1  # strictly rising, unclipped
    assert lr1[0] == pytest.approx(-0.860, abs=1e-3)

    be = CueBackend(_CUE_WEIGHTS)
    ratios = []
    for a in alphas:
        shape = {k: float(v) for k, v in agn_ionspec_from_alpha_pl(a).items()}
        wave, lum = be.predict_nebular_line_luminosities(
            gas_logu=-2.5, gas_logn=3.0, gas_logz=0.0, gas_logno=0.0, gas_logco=0.0,
            gas_logqion=53.0, cloudyfsps_only=False, **shape,
        )  # fmt: skip
        wave, lum = np.asarray(wave), np.asarray(lum)

        def line(c, wave=wave, lum=lum):
            return float(lum[np.argmin(np.abs(wave - c))])

        ratios.append((line(5008.24) / line(4862.68), line(4687.0) / line(4862.68)))
    o3, he2 = zip(*ratios)
    assert all(b < a for a, b in itertools.pairwise(o3)), o3
    assert all(b < a for a, b in itertools.pairwise(he2)), he2
