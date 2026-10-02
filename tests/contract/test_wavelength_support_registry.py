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

and every AGN disc block must either declare support or appear in
``_DISC_WITHOUT_DECLARED_SUPPORT`` below with the reason. A newly registered
block therefore fails this test until someone decides which it is.
"""

from __future__ import annotations

import pytest

from tengri.components.agn.blocks._protocol import AGN_BLOCKS
from tengri.components.dust.emission.emission import DUST_EMISSION_MODELS
from tengri.forward import wavelength_extension as we

pytestmark = pytest.mark.contract

# Discs that declare nothing, with the measured reason (default parameters,
# L_bol = 1e11 Lsun, energy outside the SSP window 91 A - 160 um; #2564).
# Analytic discs evaluate on whatever grid they are handed, so a declaration
# would be an analytic support range: a separate, larger change with its own
# regression risk (the disc normalisation changes for every model using them).
_DISC_WITHOUT_DECLARED_SUPPORT = {
    "none": "no emission",
    "adaf": "analytic X-ray ADAF, 99 % of the energy below 91 A (physically an X-ray block)",
    "adaf_lopez2024": "analytic, 1 % beyond 160 um",
    "grahsp_sbpl": "analytic broken power law, energy inside the SSP window",
    "kubota_done": "analytic, 21 % of the energy below 91 A (follow-up: analytic EUV support)",
    "multicolor": "analytic, 2.5 % below 91 A",
    "powerlaw": "analytic power law, 39 % beyond 160 um by construction",
    "qsogen": "analytic, energy inside the SSP window",
    "richards2006": "energy outside the SSP window < 0.1 %",
    "schartmann2005": "analytic, 0.3 % below 91 A",
    "schartmann2005_skirtor_atten": "analytic, 0.3 % below 91 A",
    "skirtor": "analytic, 15 % below 91 A (follow-up: analytic EUV support)",
    "slone_netzer": "template axis 450 A - 3e7 A, no energy outside the SSP window",
}


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


def test_every_disc_block_declares_support_or_is_listed_with_a_reason():
    undeclared = {n for n in AGN_BLOCKS["disc"] if n not in we._AGN_DISC_TEMPLATES}
    unlisted = sorted(undeclared - set(_DISC_WITHOUT_DECLARED_SUPPORT))
    assert not unlisted, (
        f"disc blocks with neither a declared native grid nor a listed reason: {unlisted} (#2564). "
        "Measure the energy outside the SSP window and declare, or list with the measurement."
    )


def test_disc_reason_list_has_no_stale_entries():
    stale = sorted(
        n
        for n in _DISC_WITHOUT_DECLARED_SUPPORT
        if n not in AGN_BLOCKS["disc"] or n in we._AGN_DISC_TEMPLATES
    )
    assert not stale, f"entries that are unregistered or now declared, remove them: {stale}"


def test_no_torus_block_is_both_declared_and_gridless():
    assert not (set(we._AGN_TORUS_TEMPLATES) | set(we._ANALYTIC_TORUS)) & set(we._GRIDLESS_TORUS)
