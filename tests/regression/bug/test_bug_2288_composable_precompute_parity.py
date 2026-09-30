# SPDX-License-Identifier: BSD-3-Clause
r"""Composable precompute LUT accuracy is measured against the exact recipe (#2288).

``interp_nd_triweight`` is a kernel smoother, not an interpolant: a lookup at a
grid node does NOT reproduce the stored table entry (node parity is a valid
invariant for the nebular grid, not for this LUT). The valid invariant here is
an accuracy *measurement* of lookup vs the exact recipe evaluation, with the
measured bound stated honestly:

Measured on the documented standard 21-node ``agn_grahsp_log_l5100`` axis
(grahsp_sbpl disc + grahsp torus, toy Gaussian filter at 5500 A):

- grid-center node: agreement to ~0 (the contract test
  ``test_agn_composable_precompute.py::test_parity_at_grid_center`` pins 5%);
- edge node (one-sided kernel): 8.7% relative error;
- interior midpoints: ramp to a plateau at 15.9% max — the photometry is
  exponential in the axis coordinate, so the triweight kernel's relative bias
  saturates at a constant set by the grid step (Jensen bias of averaging an
  exponential), far above node error but well under the 50% refusal rule.

The earlier draft of this fix wrapped these numbers in a grid-size tolerance
ladder (rtol 1.0/0.3/0.15) with warn-instead-of-raise fallbacks and ran the
check inside ``precompute()`` itself; that check was dropped as invalid — a
ladder invented to make a wrong invariant pass. This test pins the honest
bound instead, at test time.

The corruption guard corrupts the ENGAGED store: ``build_lookup`` reads
``preint["_preint"]`` (a ``PreintegratedGrid``) on the uncollapsed path, and
the top-level ``"grid_phot"`` key is an inspection copy — corrupting the copy
changes nothing a caller can observe (the attachment-vs-engagement trap; the
earlier draft's corruption test made exactly that mistake).
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn._params import DEFAULT_AGN_LOG_LBOL
from tengri.components.agn.blocks import Recipe, composable_precompute
from tengri.components.agn.blocks.runner import composable_agn_l_nu
from tests._data_skip import requires_grahsp

pytestmark = pytest.mark.regression_bug

#: Interior-midpoint bound: measured max 0.159 on the standard 21-node axis
#: (plateau of the triweight kernel's Jensen bias on an exponential-in-
#: coordinate quantity); pinned with ~1.6x headroom, well under the 50% rule.
_INTERIOR_MAX_REL_ERR = 0.25

_WAVE_REST = np.logspace(2.0, 6.0, 1500, dtype=np.float64)
_AXIS_GRID = np.linspace(43.0, 46.0, 21)


def _toy_filter():
    wave = np.linspace(4500.0, 6500.0, 200)
    trans = np.exp(-0.5 * ((wave - 5500.0) / 300.0) ** 2)
    return wave, trans


def _standard_recipe():
    return Recipe.from_selectors(
        disc="grahsp_sbpl",
        torus="grahsp",
        attenuation="none",
        axis_params=("agn_grahsp_log_l5100",),
    )


def _standard_precompute():
    fw, ft = _toy_filter()
    return composable_precompute.precompute(
        filter_waves=[fw],
        filter_trans=[ft],
        redshift=0.0,
        parameters=None,
        recipe=_standard_recipe(),
        axis_grids={"agn_grahsp_log_l5100": _AXIS_GRID},
        wave_rest=_WAVE_REST,
    )


def _exact_photometry(l5100_value: float) -> float:
    """The recipe evaluated exactly, integrated through the toy filter.

    Same reference path as the contract test's ``test_parity_at_grid_center``:
    the runner on the same wave grid, trapezoidal <F_nu> through the filter.
    """
    fw, ft = _toy_filter()
    l_nu = composable_agn_l_nu(
        jnp.asarray(_WAVE_REST),
        agn_disc_block="grahsp_sbpl",
        agn_nlr_block="none",
        agn_blr_block="none",
        agn_feii_block="none",
        agn_torus_block="grahsp",
        agn_attenuation_block="none",
        agn_log_lbol=DEFAULT_AGN_LOG_LBOL,
        agn_grahsp_log_l5100=float(l5100_value),
    )
    f_nu = np.asarray(l_nu)
    c_aa_per_s = 2.99792458e18
    nu = c_aa_per_s / _WAVE_REST
    order = np.argsort(nu)
    trans_interp = np.interp(_WAVE_REST, fw, ft, left=0.0, right=0.0)
    return float(
        np.trapezoid((f_nu * trans_interp / nu)[order], nu[order])
        / np.trapezoid((trans_interp / nu)[order], nu[order])
    )


def _max_interior_rel_err(lookup) -> float:
    mids = (_AXIS_GRID[:-1] + _AXIS_GRID[1:]) / 2.0
    errs = []
    for v in mids:
        lut = float(lookup(jnp.array(1.0), jnp.array(float(v)))[0])
        exact = _exact_photometry(v)
        errs.append(abs(lut - exact) / abs(exact))
    return max(errs)


@requires_grahsp
def test_interior_accuracy_bounded_on_standard_grid():
    """Max lookup-vs-exact error at interior midpoints stays under the measured bound."""
    out = _standard_precompute()
    lookup = composable_precompute.build_lookup(out)

    max_err = _max_interior_rel_err(lookup)
    assert max_err < _INTERIOR_MAX_REL_ERR, (
        f"interior midpoint max relative error {max_err:.3f} exceeds the pinned "
        f"bound {_INTERIOR_MAX_REL_ERR} (measured 0.159 on this grid at pin time)"
    )
    # The bound must also stay a measurement, not a vacuously huge ceiling:
    # the kernel's plateau is real, so the measured max cannot be tiny either.
    assert max_err > 0.05, (
        f"interior max {max_err:.4f} is far below the known kernel plateau (~0.16); "
        "the measurement is probably no longer evaluating the exponential regime"
    )


@requires_grahsp
def test_corruption_of_engaged_grid_is_detected():
    """Corrupting the ENGAGED preintegrated grid pushes the measurement past the bound.

    This is the designed-dead probe for the accuracy measurement: a x5
    corruption of one node's stored photometry is far above the kernel's own
    ~16% bias, so the same measurement that passes on the honest grid must
    fail on the corrupted one. Corrupting the top-level ``grid_phot``
    inspection copy would NOT be detected — the lookup never reads it on this
    path — so this test pins engagement, not attachment.
    """
    out = _standard_precompute()

    p = out["_preint"]
    phot = np.array(p.phot)
    phot[10] *= 5.0  # corrupt every filter at node 10 — well above kernel bias
    out_corrupt = {**out, "_preint": dataclasses.replace(p, phot=jnp.asarray(phot))}

    lookup = composable_precompute.build_lookup(out_corrupt)
    max_err = _max_interior_rel_err(lookup)
    assert max_err > _INTERIOR_MAX_REL_ERR, (
        f"x5 corruption of the engaged grid produced max error {max_err:.3f}, "
        f"below the bound {_INTERIOR_MAX_REL_ERR} — the measurement is not "
        "reading the engaged data path"
    )
