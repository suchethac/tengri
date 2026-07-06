"""Energy conservation of the composable AGN runner (conserving policy).

The disc carries the intrinsic bolometric luminosity; the torus reprocesses a
fraction of it. Moving energy from disc to torus via ``agn_torus_frac`` must not
change the *total* emitted energy — bolometric is conserved. This is the
invariant PR #916 violated (the composable path added the torus on top of a
full-luminosity disc, inflating the total by 0.5-4x).

The torus list is derived from the registry so every non-self-contained torus
inherits the invariant automatically — new tori are covered, and deprecated
toys (``simple``, ``two_temperature``) drop out when removed rather than leaving
a stale hand-picked list. Data-gated tori (whose template h5 is absent in CI)
skip gracefully.
"""

import numpy as np
import pytest

from tengri.components.agn.blocks._protocol import AGN_BLOCKS
from tengri.components.agn.blocks.registry import composable
from tengri.components.agn.blocks.runner import _SELF_CONTAINED_TORI
from tengri.utils.physics_constants import L_SUN

pytestmark = pytest.mark.conservation

_WAVE = np.geomspace(1e2, 1e7, 2000)  # Å — wide enough for disc UV + torus IR
_C_AA_PER_S = 2.99792458e18

# Every non-self-contained torus must conserve energy under the conserving policy.
_TORI = sorted(set(AGN_BLOCKS["torus"]) - set(_SELF_CONTAINED_TORI))


def _band_energy(l_nu, wave):
    """Integrate L_nu over frequency to get the band-limited bolometric [erg/s]."""
    nu = _C_AA_PER_S / wave
    order = np.argsort(nu)
    return np.trapezoid(np.asarray(l_nu)[order], nu[order])


def _compose_or_skip(torus, tf):
    try:
        return composable(
            _WAVE,
            45.0,
            agn_disc_block="powerlaw",
            agn_torus_block=torus,
            agn_norm="conserving",
            agn_frac=1.0,
            agn_torus_frac=tf,
        )
    except (FileNotFoundError, OSError) as exc:
        # data-gated tori (template h5 absent in CI). Narrow on purpose: a
        # ValueError/KeyError is a real bug we want to surface, not skip.
        pytest.skip(f"torus '{torus}' unavailable: {type(exc).__name__}: {exc}")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("torus", _TORI)
def test_conserving_policy_is_invariant_under_torus_frac(torus):
    """Under ``agn_norm='conserving'`` the total emitted energy is invariant as
    ``agn_torus_frac`` shifts energy from disc to torus, and it equals L_bol."""
    e0 = _band_energy(_compose_or_skip(torus, 0.0), _WAVE)

    # Absolute anchor: at agn_log_lbol=45 the disc-only (torus_frac=0) total is
    # ~10^45 L_sun in erg/s. This catches a global mis-scale (e.g. disc AND
    # torus both inflated 2x) that the relative invariance below cannot see.
    l_bol_erg = 10.0**45 * L_SUN
    assert e0 == pytest.approx(l_bol_erg, rel=0.1), (
        f"{torus}: disc-only band energy {e0:.3e} erg/s != L_bol {l_bol_erg:.3e}"
    )

    # Relative invariance: energy is conserved as it moves disc -> torus.
    for tf in (0.3, 0.6, 0.9):
        e = _band_energy(_compose_or_skip(torus, tf), _WAVE)
        assert e == pytest.approx(e0, rel=0.02), (
            f"{torus}: energy not conserved at torus_frac={tf} "
            f"({e:.3e} vs {e0:.3e}) — disc not debited"
        )
