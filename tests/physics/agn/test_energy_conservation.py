"""Energy conservation of the composable AGN runner (conserving policy).

The disc carries the intrinsic bolometric luminosity; the torus reprocesses a
fraction of it. Moving energy from disc to torus via ``agn_torus_frac`` must not
change the *total* emitted energy — bolometric is conserved. This is the
invariant PR #916 violated (the composable path added the torus on top of a
full-luminosity disc, inflating the total by 0.5-4x).
"""

import numpy as np
import pytest

from tengri.components.agn.blocks.registry import composable

pytestmark = pytest.mark.conservation

_WAVE = np.geomspace(1e2, 1e7, 2000)  # Å — wide enough for disc UV + torus IR
_C_AA_PER_S = 2.99792458e18


def _band_energy(l_nu, wave):
    """Integrate L_nu over frequency to get the band-limited bolometric [erg/s]."""
    nu = _C_AA_PER_S / wave
    order = np.argsort(nu)
    return np.trapezoid(np.asarray(l_nu)[order], nu[order])


@pytest.mark.parametrize("torus", ["silva04", "cat3d_wind", "simple", "two_temperature"])
def test_conserving_policy_is_invariant_under_torus_frac(torus):
    """Under ``agn_norm='conserving'`` the total emitted energy is invariant as
    ``agn_torus_frac`` shifts energy from disc to torus."""
    e0 = _band_energy(
        composable(
            _WAVE,
            45.0,
            agn_disc_block="powerlaw",
            agn_torus_block=torus,
            agn_norm="conserving",
            agn_frac=1.0,
            agn_torus_frac=0.0,
        ),
        _WAVE,
    )
    for tf in (0.3, 0.6, 0.9):
        e = _band_energy(
            composable(
                _WAVE,
                45.0,
                agn_disc_block="powerlaw",
                agn_torus_block=torus,
                agn_norm="conserving",
                agn_frac=1.0,
                agn_torus_frac=tf,
            ),
            _WAVE,
        )
        assert e == pytest.approx(e0, rel=0.02), (
            f"{torus}: energy not conserved at torus_frac={tf} "
            f"({e:.3e} vs {e0:.3e}) — disc not debited"
        )
