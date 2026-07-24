# SPDX-License-Identifier: BSD-3-Clause
"""Line-energy conservation of the composable AGN runner (#929).

The NLR/BLR/FeII line regions reprocess disc ionizing photons, so under
``agn_norm='conserving'`` they must be *debited* from the disc, not stacked on
top of a full-luminosity disc. Adding the lines without debiting inflates the
total above ``L_bol`` — the same leak class #916 fixed for the torus, now for
the line covering fraction (the Σf ledger, spec Phase 1.x).

This is the invariant #929 introduces. Synthesizer's ``UnifiedAGN`` conserves
the same way: the disc is dimmed by the covering fraction and the lines are the
reprocessed remainder, so ``∫total = L_bol``.
"""

import numpy as np
import pytest

from tengri.components.agn.blocks.registry import composable
from tengri.utils.physics_constants import L_SUN

pytestmark = pytest.mark.conservation

_WAVE = np.geomspace(1e2, 1e7, 2000)  # Å — disc UV through torus IR
_C_AA_PER_S = 2.99792458e18


def _band_energy(l_nu, wave):
    """Integrate L_nu over frequency → band-limited bolometric [erg/s]."""
    nu = _C_AA_PER_S / wave
    order = np.argsort(nu)
    return np.trapezoid(np.asarray(l_nu)[order], nu[order])


def _compose(**overrides):
    kw = dict(
        agn_disc_block="powerlaw",
        agn_nlr_block="none",
        agn_blr_block="none",
        agn_feii_block="none",
        agn_torus_block="none",
        agn_norm="conserving",
        agn_lum_ratio=1.0,
    )
    kw.update(overrides)
    return _band_energy(composable(_WAVE, 45.0, **kw), _WAVE)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_conserving_debits_disc_for_nlr_blr_lines():
    """Under ``conserving``, analytic NLR+BLR lines do not inflate the total
    above ``L_bol`` — the disc is debited for the line covering fraction."""
    l_bol_erg = 10.0**45 * L_SUN
    e = _compose(agn_nlr_block="analytic", agn_blr_block="analytic")
    assert e == pytest.approx(l_bol_erg, rel=0.005), (
        f"conserving + nlr + blr total {e:.4e} erg/s != L_bol {l_bol_erg:.4e} "
        f"(ratio {e / l_bol_erg:.5f}) — disc not debited for line energy"
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_conserving_line_debit_leaves_disc_only_unchanged():
    """The line debit is a no-op when no line blocks are active — a bare disc
    under ``conserving`` still integrates to exactly ``L_bol``."""
    l_bol_erg = 10.0**45 * L_SUN
    assert _compose() == pytest.approx(l_bol_erg, rel=1e-3)
