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
The retired ``tests/regression/agn/test_monolithic_composable_equivalence.py``
compared ``resolve_agn_model(name)`` against a *hand-written* composable call —
composable on **both** sides — so it verified nothing about the monolithic
physics and passed as a tautology. That tautology is exactly why the #941
``agn_norm='independent'`` regression (which mis-scaled every disc+torus preset
by an ``agn_lum_ratio`` factor, ~2×/99% off) merged green. Here the reference side is
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
_KW = dict(agn_log_lbol=11.0, agn_lum_ratio=0.5)
_TIGHT = 1e-5

# Known block-level gaps: the composable block does not yet reproduce the
# monolithic disc/torus bit-for-bit. Tracked, xfail(strict) — not loosened.
#
# adaf is an EXACT case (no xfail): as of #898 the monolithic ``adaf_agn`` and
# the composable ``disc='adaf'`` preset both use the faithful Mahadevan 1997
# ``adaf_spectrum``, so the preset reproduces the monolithic to the Type-1/2
# mask floor. (The Phase-1 documented-divergence gate was retired once the
# monolithic path was unified onto the faithful physics.)
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
    pytest.param("adaf", U.adaf_agn, id="adaf"),
    pytest.param("richards2006", richards2006, id="richards2006"),
    # Was _KUBOTA_GAP (#944, "~39% off monolithic"). Not a block normalization
    # bug: the monolithic kubota_done_disc defaulted agn_cos_inc to 0.5 while the
    # composable block defaulted it to cos(30 deg), and this test supplies
    # neither. Unifying both onto the declared default closed the gap to the
    # tight rtol, so the strict xfail flipped to xpass and the marker goes.
    pytest.param("kubota_done", U.kubota_done_full_agn, id="kubota_done"),
    pytest.param("kubota_done_full", U.kubota_done_full_agn, id="kubota_done_full"),
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
