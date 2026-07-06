"""Energy semantics of each ``agn_norm`` policy — the foundation contract.

Pins what each cross-block normalization policy guarantees so the layer can
never silently regress. In particular: ``cigale_joint`` used to LEAK energy
(add the torus on top, ratio ~1.6) for every torus except SKIRTOR, because its
energy tie was only wired for the SKIRTOR template. It now falls back to the
``agn_torus_frac`` disc debit for non-SKIRTOR tori, so it conserves for every
torus. ``independent`` is the deliberately non-conserving comparison policy.
"""

import numpy as np
import pytest

from tengri.components.agn.blocks.registry import composable

pytestmark = pytest.mark.conservation

_WAVE = np.geomspace(1e2, 1e7, 2000)
_C_AA_PER_S = 2.99792458e18


def _band_energy(l_nu):
    nu = _C_AA_PER_S / _WAVE
    order = np.argsort(nu)
    return np.trapezoid(np.asarray(l_nu)[order], nu[order])


def _energy(norm, torus, tf):
    return _band_energy(
        composable(
            _WAVE,
            45.0,
            agn_disc_block="powerlaw",
            agn_torus_block=torus,
            agn_norm=norm,
            agn_frac=1.0,
            agn_torus_frac=tf,
        )
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("torus", ["silva04", "cat3d_wind"])
def test_conserving_conserves(torus):
    """`conserving` keeps total energy fixed as it moves disc -> torus."""
    assert _energy("conserving", torus, 0.6) == pytest.approx(
        _energy("conserving", torus, 0.0), rel=0.02
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("torus", ["silva04", "cat3d_wind"])
def test_cigale_joint_conserves_for_nonskirtor_tori(torus):
    """Regression: `cigale_joint` used to leak (ratio ~1.6) for non-SKIRTOR
    tori. It now falls back to the `agn_torus_frac` disc debit and conserves.
    (SKIRTOR uses the separate agn_power x R tie — see the #556 CIGALE parity
    tests.)"""
    e0 = _energy("cigale_joint", torus, 0.0)
    e6 = _energy("cigale_joint", torus, 0.6)
    assert e6 == pytest.approx(e0, rel=0.02), (
        f"cigale_joint leaks energy for non-SKIRTOR torus {torus!r}: "
        f"E(0.6)/E(0.0) = {e6 / e0:.3f} (should be ~1.0)"
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_independent_does_not_conserve():
    """`independent` is the non-conserving comparison policy: the torus is
    added on top, so total energy grows with `agn_torus_frac`."""
    e0 = _energy("independent", "silva04", 0.0)
    e6 = _energy("independent", "silva04", 0.6)
    assert e6 > 1.3 * e0, f"independent unexpectedly conserved: E(0.6)/E(0.0) = {e6 / e0:.3f}"
