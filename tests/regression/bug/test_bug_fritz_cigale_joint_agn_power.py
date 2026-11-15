# SPDX-License-Identifier: BSD-3-Clause
"""Under ``cigale_joint`` the tied Fritz torus + polar dust carry ``agn_power`` at every psi.

Owner ruling 2026-11-13, narrowing the 2026-10-13 one (#3070, #2851) for
``agn_norm='cigale_joint'`` with the Fritz torus and the ``agn_ir_frac`` tie only. pCIGALE's
``fritz2006`` divides the dust
(torus plus polar blackbody) by its own integral at the chosen ``psy`` and multiplies it by
``agn_power``, so the AGN dust total does not depend on the viewing angle (Fritz et al. 2006,
MNRAS 366, 767; ``pcigale/sed_modules/fritz2006.py``, ``norm = 1/trapz(dust)``). tengri kept the
library's per-record anisotropy ``a(psi)/a_bar``, so the total read 0.85x, 1.07x and 1.25x of
``agn_power`` at the three CIGALE-reproduction (section 9e) cases. The torus shape per sightline is
unchanged; the scalar is removed.

The untied path (``agn_ir_frac = 0``) and ``agn_norm='independent'`` keep the anisotropy (#2851).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.blocks._protocol import collect_block_templates
from tengri.components.agn.blocks.runner import composable_agn_l_nu
from tengri.utils.physics_constants import L_SUN

pytestmark = pytest.mark.regression_bug

_C_AA_PER_S = 2.99792458e18
_LOG_LBOL = 11.0
_TORUS_FRAC = 0.3
_WAVE = np.geomspace(100.0, 1.0e7, 6000)
#: The three section 9e Fritz cases of reproduction/cigale/01_cigale.py: (tau, full opening, psi).
_CASES = ((1.0, 60.0, 0.001), (1.0, 100.0, 40.1), (6.0, 60.0, 89.99))
_SELECTORS = {
    "agn_disc_block": "schartmann2005",
    "agn_nlr_block": "none",
    "agn_blr_block": "none",
    "agn_feii_block": "none",
    "agn_torus_block": "fritz",
    "agn_attenuation_block": "polar_dust",
}


def _pieces_over_budget(case, *, agn_norm="cigale_joint", ir_frac=0.3):
    """(torus, polar) power over ``agn_torus_frac L_bol`` for one (tau, oa, psi) case."""
    tau, oa_full, psi = case
    templates = collect_block_templates(
        {k[len("agn_") : -len("_block")]: v for k, v in _SELECTORS.items()}
    )
    params = {
        "agn_norm": agn_norm,
        "agn_cos_inc": float(np.sin(np.radians(psi))),
        "agn_torus_frac": _TORUS_FRAC,
        "agn_ir_frac": ir_frac,
        "agn_polar_ebv": 0.03,
        "agn_fritz_tau": tau,
        "agn_fritz_oa": (180.0 - oa_full) / 2.0,
    }
    with warnings.catch_warnings(), jax.enable_x64(True):
        warnings.simplefilter("ignore")
        wave = jnp.asarray(_WAVE, dtype=jnp.float64)
        _, comps = composable_agn_l_nu(
            wave,
            _LOG_LBOL,
            return_components=True,
            template_state=templates,
            **_SELECTORS,
            **params,
        )
        nu = _C_AA_PER_S / wave
        budget = _TORUS_FRAC * 10.0**_LOG_LBOL * L_SUN
        return tuple(float(-jnp.trapezoid(comps[k], nu)) / budget for k in ("torus", "polar"))


def _dust_over_budget(case, **kwargs):
    """Torus + polar over ``agn_torus_frac L_bol``."""
    return sum(_pieces_over_budget(case, **kwargs))


def _or_skip(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except FileNotFoundError as exc:
        pytest.skip(f"library data absent: {exc}")


@pytest.mark.parametrize("case", _CASES)
def test_tied_fritz_dust_carries_agn_power_at_every_inclination(case):
    """Torus + polar = ``agn_power`` to 1e-4 under cigale_joint (0.85x to 1.25x before)."""
    assert _or_skip(_dust_over_budget, case) == pytest.approx(1.0, rel=1e-4)


def test_untied_fritz_keeps_its_library_anisotropy():
    """Without the tie the sightline power still follows the library (#2851, unchanged)."""
    ratios = [_or_skip(_dust_over_budget, c, ir_frac=0.0) for c in (_CASES[0], _CASES[2])]
    assert abs(ratios[0] / ratios[1] - 1.0) > 0.05


def test_independent_norm_keeps_its_library_anisotropy():
    """``agn_norm='independent'`` is outside the ruling: the anisotropy stays."""
    ratios = [
        _or_skip(_dust_over_budget, c, agn_norm="independent", ir_frac=0.0)
        for c in (_CASES[0], _CASES[2])
    ]
    assert abs(ratios[0] / ratios[1] - 1.0) > 0.05


#: (torus, polar) over ``agn_torus_frac L_bol`` on main before the ruling (origin/main 27bb1034a,
#: measured with this module's harness),
#: untied (``agn_ir_frac = 0``), per case in ``_CASES`` order. The ruling must not move them.
_MAIN_UNTIED = {
    "cigale_joint": (
        (0.5028672850298646, 0.20171428371622885),
        (0.9664195825767926, 0.10573270572657494),
        (1.234594484644917, 0.2017142811211918),
    ),
    "independent": (
        (0.6299348318422044, 0.36097809835624367),
        (1.08068376907645, 0.16890566404557464),
        (1.54655930547788, 0.36097809835624367),
    ),
}


@pytest.mark.parametrize("agn_norm", ["cigale_joint", "independent"])
@pytest.mark.parametrize("index", range(len(_CASES)))
def test_untied_torus_and_polar_totals_are_main_values(agn_norm, index):
    """Untied torus and polar totals are bit-for-bit what main emitted (1e-10)."""
    got = _or_skip(_pieces_over_budget, _CASES[index], agn_norm=agn_norm, ir_frac=0.0)
    assert got == pytest.approx(_MAIN_UNTIED[agn_norm][index], rel=1e-10)
