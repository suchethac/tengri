# SPDX-License-Identifier: BSD-3-Clause
"""Equivalence gate — monolithic AGN models vs their composable presets.

The monolithic AGN forward functions (``silva04_agn``, ``multicolor_agn``, …)
still live in :mod:`tengri.components.agn.unified` as the reference physics.
Their public model names were retired from the ``AGN_MODELS`` dispatch (#916)
and now route through composable-block *presets* via
:func:`tengri.components.agn.unified.resolve_agn_model`. This gate pins each
preset to the monolithic SED it claims to reproduce.

Why this gate exists
--------------------
The retired ``tests/unit/agn/test_monolithic_to_composable_equivalence.py``
compared ``resolve_agn_model(name)`` against a *hand-written* composable call —
composable on **both** sides — so it verified nothing about the monolithic
physics and passed as a tautology. That tautology is exactly why the #941
``agn_norm='independent'`` regression (which mis-scaled every disc+torus preset
by an ``agn_frac`` factor, ~2×/99% off) merged green. Here the reference side is
the true monolithic function, so such a divergence fails instead.

Per-model tolerances
--------------------
Presets that reproduce the monolithic normalization exactly get a tight gate
(``rtol = 1e-5``). Presets with a *known* block-level gap — the composable
disc/torus block does not yet reproduce the monolithic disc/torus bit-for-bit —
are marked ``xfail(strict=True)`` with the tracking issue, never silently
loosened. The gap stays visible, and the day a block is fixed the ``xfail``
flips to ``xpass`` and the strict marker fails, forcing the marker's removal.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.components.agn import richards2006, unified as U
from tengri.components.agn.unified import resolve_agn_model

pytestmark = pytest.mark.regression_paper

# 1000 Å – ~3 mm: spans the UV/optical disc and the mid/far-IR torus so a
# torus-only or disc-only normalization error cannot hide off the grid.
_WAVE = np.logspace(3.0, 5.5, 120)
_KW = dict(agn_log_lbol=11.0, agn_frac=0.5)
_TIGHT = 1e-5

# Known block-level gaps: the composable block does not yet reproduce the
# monolithic disc/torus bit-for-bit. Tracked, xfail(strict) — not loosened.
#
# NOTE (#898): adaf is deliberately NOT in this equivalence set. Its composable
# block is now the *faithful* Mahadevan 1997 model, which has SURPASSED the
# monolithic ``adaf_agn`` (built on the old ``adaf_disc`` that misapplied Eq. 49).
# "Equivalence with monolithic" is therefore the wrong success criterion for
# adaf — a strict-xfail here would keep "passing as expected failure" forever,
# guarding the buggy side. Instead see ``test_adaf_surpasses_deprecated_monolithic``
# below (documented-divergence) and the physics gate in test_adaf_mahadevan.py.
_KUBOTA_GAP = "kubota_done disc block normalization ~39% off monolithic (#944)"
_SKIRTOR_GAP = (
    "monolithic skirtor uses CIGALE's joint disc+torus energy balance; the "
    "powerlaw-disc + skirtor-torus composition differs ~96% (#944)"
)
_UNLRBLR_GAP = "analytic NLR/BLR composition ~59% off monolithic (#944)"


def _gap(reason: str) -> list:
    return [pytest.mark.xfail(reason=reason, strict=True)]


# preset name -> monolithic reference callable, with an xfail mark on the
# models carrying a documented block-level gap.
_CASES = [
    pytest.param("multicolor_agn", U.multicolor_agn, id="multicolor_agn"),
    pytest.param("silva04", U.silva04_agn, id="silva04"),
    pytest.param("cat3d_wind", U.cat3d_wind_agn, id="cat3d_wind"),
    pytest.param("richards2006", richards2006, id="richards2006"),
    pytest.param("kubota_done", U.kubota_done_full_agn, marks=_gap(_KUBOTA_GAP), id="kubota_done"),
    pytest.param(
        "kubota_done_full", U.kubota_done_full_agn, marks=_gap(_KUBOTA_GAP), id="kubota_done_full"
    ),
    pytest.param("skirtor", U.skirtor_agn, marks=_gap(_SKIRTOR_GAP), id="skirtor"),
    pytest.param(
        "unified_nlr_blr", U.unified_nlr_blr, marks=_gap(_UNLRBLR_GAP), id="unified_nlr_blr"
    ),
]


@pytest.mark.parametrize("preset_name,monolithic_fn", _CASES)
def test_preset_reproduces_monolithic(preset_name, monolithic_fn):
    """``resolve_agn_model(name)`` must reproduce the monolithic ``name`` SED.

    The reference is the genuine monolithic forward function; the candidate is
    the composable-preset routing that replaced it. A tight ``rtol`` gate
    catches any preset that silently normalizes, selects, or scales the SED
    differently from the physics it advertises.
    """
    try:
        ref = np.asarray(monolithic_fn(_WAVE, **_KW))
    except (FileNotFoundError, OSError) as exc:
        # Grid/template-gated references (relagn, grahsp) skip cleanly in CI;
        # a data-only catch never masks a physics or wiring regression.
        pytest.skip(f"{preset_name}: monolithic reference data unavailable ({exc})")

    got = np.asarray(resolve_agn_model(preset_name)(_WAVE, **_KW))
    denom = np.maximum(np.abs(ref), 1e-30)
    max_rel = float(np.max(np.abs((ref - got) / denom)))
    assert max_rel < _TIGHT, (
        f"{preset_name}: composable preset diverges from monolithic by "
        f"max_rel={max_rel:.3e} (tol {_TIGHT:.0e}). The preset no longer "
        f"reproduces the physics it claims — check agn_norm / block selectors."
    )


def test_adaf_surpasses_deprecated_monolithic():
    """The faithful composable ADAF has SURPASSED the monolithic reference (#898).

    The monolithic ``adaf_disc`` misapplied Mahadevan 1997 Eq. 49 (radiative
    luminosity scaled by L_bol, not L_Edd) and bundled an ad-hoc truncated disc;
    the composable ``disc='adaf'`` block is now the faithful ``adaf_spectrum``.
    "Equivalence with monolithic" is thus the wrong success criterion — so this
    is a *documented-divergence* gate: the faithful model must differ from the
    deprecated one by a wide margin (it does not merely re-normalize; the whole
    spectral shape and the L_bol->mdot relation changed). Correctness of the
    faithful physics is pinned independently in test_adaf_mahadevan.py.

    The deprecated monolithic ``adaf_disc`` / ``adaf_agn`` remain only as the old
    reference and are slated for removal in the monolithic-retirement follow-up.
    """
    from tengri.components.agn.adaf import adaf_spectrum

    # Wide grid (UV -> mm) so the ADAF synchrotron peak is on-grid, where the two
    # disc physics differ most.
    wave = np.logspace(3.0, 8.0, 400)
    faithful = np.asarray(adaf_spectrum(wave, agn_log_lbol=11.0, agn_log_mbh=8.0))
    deprecated = np.asarray(
        U.adaf_disc(wave, agn_log_lbol=11.0, agn_log_mbh=8.0, agn_log_ledd=-3.0)
    )
    denom = np.maximum(np.abs(deprecated), 1e-30)
    max_rel = float(np.max(np.abs((faithful - deprecated) / denom)))
    assert max_rel > 0.5, (
        "The faithful ADAF (adaf_spectrum) should differ substantially from the "
        f"deprecated monolithic adaf_disc, but max_rel={max_rel:.3e} — has the "
        "block silently regressed to the old physics?"
    )


def test_skirtor_stalevski_routes_to_raw_template():
    """``skirtor_stalevski`` is un-composable: it must return the raw RT total.

    The published Stalevski (2016) SKIRTOR *total* SED is computed jointly by
    radiative transfer (disc + torus + scattering) and is provably not a
    disc-block + torus-block sum, so it routes directly to the monolithic
    template rather than a preset. This pins that routing (regression for the
    #941 attempt to express it as ``disc=skirtor + torus=skirtor``).
    """
    from tengri.components.agn.skirtor import _find_skirtor_grid

    if _find_skirtor_grid() is None:
        pytest.skip("SKIRTOR grid not available (data-gated)")
    fn = resolve_agn_model("skirtor_stalevski")
    assert fn is U.skirtor_stalevski_agn, (
        "skirtor_stalevski must route to the monolithic raw-total function, "
        "not a composable preset (the raw RT total is not disc+torus)."
    )
