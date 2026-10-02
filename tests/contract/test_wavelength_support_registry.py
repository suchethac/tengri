# SPDX-License-Identifier: BSD-3-Clause
"""Contract (#2564): every registered emitter declares its wavelength support.

The master rest-frame grid is the union of the SSP grid and each attached
component's *declared* native grid (``tengri.forward.wavelength_extension``).
A block that emits outside, or finer than, the SSP grid but declares nothing is
silently sampled on the SSP grid: the IR peak moves, the submm tail is cut and
energy-normalised blocks are renormalised over the truncated window. So every
registered dust-emission model and AGN torus block must be in exactly one of

* a tabulated declaration (template axis),
* an analytic declaration (synthetic support grid), or
* an explicit grid-less list (no emission of its own),

and so must every AGN disc block (template axis, analytic range in
``_ANALYTIC_DISC_RANGE_AA``, or ``_GRIDLESS_DISC`` with the measured reason). A
newly registered block therefore fails this test until someone decides which
it is.
"""

from __future__ import annotations

import pytest

from tengri.components.agn.blocks._protocol import AGN_BLOCKS
from tengri.components.dust.emission.emission import DUST_EMISSION_MODELS
from tengri.forward import wavelength_extension as we

pytestmark = pytest.mark.contract

def _torus_declared(name: str) -> bool:
    return name in we._AGN_TORUS_TEMPLATES or name in we._ANALYTIC_TORUS or name in we._GRIDLESS_TORUS


def _dust_declared(name: str) -> bool:
    return (
        name in we._DUST_EMISSION_TEMPLATES
        or name in we._ANALYTIC_DUST_EMISSION
        or name in we._GRIDLESS_DUST_EMISSION
    )


def test_every_dust_emission_model_declares_support():
    missing = sorted(n for n in DUST_EMISSION_MODELS if not _dust_declared(n))
    assert not missing, (
        f"dust-emission models without declared wavelength support: {missing}. Add each to "
        "_DUST_EMISSION_TEMPLATES, _ANALYTIC_DUST_EMISSION or _GRIDLESS_DUST_EMISSION (#2564)."
    )


def test_every_torus_block_declares_support():
    missing = sorted(n for n in AGN_BLOCKS["torus"] if not _torus_declared(n))
    assert not missing, (
        f"torus blocks without declared wavelength support: {missing}. Add each to "
        "_AGN_TORUS_TEMPLATES, _ANALYTIC_TORUS or _GRIDLESS_TORUS (#2564)."
    )


def _disc_declared(name: str) -> bool:
    return (
        name in we._AGN_DISC_TEMPLATES or name in we._ANALYTIC_DISC_RANGE_AA or name in we._GRIDLESS_DISC
    )


def test_every_disc_block_declares_support():
    missing = sorted(n for n in AGN_BLOCKS["disc"] if not _disc_declared(n))
    assert not missing, (
        f"disc blocks without declared wavelength support: {missing}. Add each to _AGN_DISC_TEMPLATES, "
        "_ANALYTIC_DISC_RANGE_AA or _GRIDLESS_DISC after measuring the energy outside the SSP window (#2564)."
    )


def test_disc_declarations_are_consistent():
    n_decl = [set(we._AGN_DISC_TEMPLATES), set(we._ANALYTIC_DISC_RANGE_AA), set(we._GRIDLESS_DISC)]
    assert not (n_decl[0] & n_decl[1]) and not (n_decl[0] & n_decl[2]) and not (n_decl[1] & n_decl[2])
    stale = sorted(set().union(*n_decl) - set(AGN_BLOCKS["disc"]))
    assert not stale, f"declared disc names that are not registered blocks: {stale}"
    for name in we._GRIDLESS_DISC:
        assert we.native_wave_agn_disc(name) is None
    for name, (lo, hi) in we._ANALYTIC_DISC_RANGE_AA.items():
        w = we.native_wave_agn_disc(name)
        assert w is not None and w[0] == pytest.approx(lo) and w[-1] == pytest.approx(hi)


def test_no_torus_block_is_both_declared_and_gridless():
    assert not (set(we._AGN_TORUS_TEMPLATES) | set(we._ANALYTIC_TORUS)) & set(we._GRIDLESS_TORUS)
