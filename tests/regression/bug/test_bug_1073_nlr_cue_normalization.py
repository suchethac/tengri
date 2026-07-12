"""Regression: the Cue AGN-NLR line luminosities were L_SUN too bright (#1073).

``CueBackend.predict_nebular_line_luminosities`` returns erg/s, but
:func:`agn_nlr_cue` promises L_sun — the unit its Feltre and Synthesizer
siblings return, and the unit every consumer converts back from. Returning
erg/s there scaled the NLR lines by an extra ``L_SUN`` (~3.8e33), which the
Gaussian rendering turned into an [O III] 5007 peak of ~1e60 erg/s/Hz.

The bug was scale-only, so it was invisible to the line-ratio and
monotonicity tests that already covered this path. The assertions below are
absolute: reprocessed lines cannot outshine the accretion luminosity that
powers them.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from tengri.components.agn.nlr_cloudy import compute_nlr_sed_cue
from tengri.components.nebular.agn_nebular import agn_nlr_cue
from tengri.utils.physics_constants import L_SUN

pytestmark = pytest.mark.regression_bug

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_CUE_WEIGHTS_PATH = _DATA_DIR / "cue_weights.npz"
requires_cue = pytest.mark.skipif(
    not _CUE_WEIGHTS_PATH.exists(),
    reason="Cue weights not found at data/cue_weights.npz",
)

_L_ACC_ERG = 1e45
_COVERING = 0.1


@pytest.fixture(scope="module")
def cue_backend():
    from tengri.components.nebular.cue import CueBackend

    return CueBackend(str(_CUE_WEIGHTS_PATH))


@requires_cue
@pytest.mark.conservation
def test_nlr_lines_cannot_outshine_the_intercepted_accretion_luminosity(cue_backend):
    """Sum of NLR lines <= the accretion luminosity the NLR intercepts.

    The pre-#1073 return was ~3.8e33x this bound.
    """
    _wav, lum_lsun = agn_nlr_cue(cue_backend, l_acc_erg=_L_ACC_ERG, covering_fraction=_COVERING)
    total_erg = float(jnp.sum(jnp.asarray(lum_lsun)) * L_SUN)

    assert total_erg <= _COVERING * _L_ACC_ERG, (
        f"NLR lines total {total_erg:.3e} erg/s > intercepted "
        f"{_COVERING * _L_ACC_ERG:.3e} erg/s — energy is being created (#1073)"
    )
    # Not vacuous in the other direction: a real NLR reprocesses a percent-level
    # share of what it intercepts, so the lines must not be ~zero either.
    assert total_erg > 1e-4 * _COVERING * _L_ACC_ERG


@requires_cue
def test_oiii_5007_lands_at_a_physical_luminosity(cue_backend):
    """[O III] 5007 is a few percent of L_acc, not 1e33x it (#1073)."""
    wav, lum_lsun = agn_nlr_cue(cue_backend, l_acc_erg=_L_ACC_ERG, covering_fraction=_COVERING)
    wav = jnp.asarray(wav)
    oiii = jnp.asarray(lum_lsun)[jnp.argmin(jnp.abs(wav - 5007.0))] * L_SUN
    oiii_erg = float(oiii)

    # A luminous AGN NLR: [O III] is a small fraction of the intercepted power.
    assert 1e-6 * _L_ACC_ERG < oiii_erg < _COVERING * _L_ACC_ERG, (
        f"[O III] 5007 = {oiii_erg:.3e} erg/s is unphysical for "
        f"L_acc = {_L_ACC_ERG:.1e} erg/s (#1073 rendered ~1e60)"
    )


@requires_cue
def test_rendered_nlr_sed_is_finite_and_physical(cue_backend):
    """End-to-end: the L_nu the forward model sees, not just the line list.

    Integrating the rendered L_nu over frequency must recover the line power,
    so the same L_SUN inflation shows up here as a ~1e60 erg/s/Hz peak.
    """
    wavelength = jnp.linspace(4000.0, 7000.0, 3001)
    sed = compute_nlr_sed_cue(
        wavelength,
        l_disc_bol_erg=_L_ACC_ERG,
        covering_fraction=_COVERING,
    )

    assert jnp.all(jnp.isfinite(sed))
    peak = float(jnp.max(sed))
    # Every erg the lines carry must come out of the intercepted luminosity;
    # a Gaussian of width ~1e12 Hz therefore peaks far below 1e40 erg/s/Hz.
    assert peak < 1e40, f"NLR L_nu peaks at {peak:.3e} erg/s/Hz — pre-#1073 gave ~1e60"
    assert peak > 0.0


@requires_cue
def test_normalization_is_linear_in_l_acc(cue_backend):
    """The fix is a constant factor, so the L_acc scaling must be untouched."""
    _w1, lum_1 = agn_nlr_cue(cue_backend, l_acc_erg=1e44, covering_fraction=_COVERING)
    _w2, lum_2 = agn_nlr_cue(cue_backend, l_acc_erg=1e45, covering_fraction=_COVERING)

    ratio = float(jnp.sum(jnp.asarray(lum_2)) / jnp.sum(jnp.asarray(lum_1)))
    assert 5.0 < ratio < 20.0, f"10x L_acc gave {ratio:.2f}x lines"
