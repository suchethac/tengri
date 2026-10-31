# SPDX-License-Identifier: BSD-3-Clause
r"""Energy conservation of the composable AGN runner (conserving policy).

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

A torus whose library keeps its inclination anisotropy emits
:math:`f L_{\rm bol}\,a(\mu)/\bar a` along a sightline, so its share of the ledger is
:math:`f L_{\rm bol}` on the sphere mean, not along one sightline: every AGN component's
published pieces close on ``agn_log_lbol`` on the sphere mean, and the line of sight is a
derived output (owner ruling 2026-10-13). For those tori the invariant is asserted on the
angle-integrated total, and the sightline is asserted to carry exactly :math:`a(\mu)/\bar a`
of the torus share.
"""

import inspect

import numpy as np
import pytest
from tests.physics._template_torus_reference import LIBRARIES, RawTorus

from tengri._data_setup import find_data
from tengri.components.agn._params import DEFAULT_AGN_COS_INC
from tengri.components.agn.blocks._protocol import AGN_BLOCKS
from tengri.components.agn.blocks.registry import composable
from tengri.components.agn.blocks.runner import _SELF_CONTAINED_TORI
from tengri.utils.physics_constants import L_SUN

pytestmark = pytest.mark.conservation

# Å. Covers the power-law disc's normalization band (10 A - 1e8 A): the grid 1e2 - 1e7 A holds
# only 0.7822 of its power, which is what a band-limited energy ledger would then report.
_WAVE = np.geomspace(10.0, 1e8, 8000)
_C_AA_PER_S = 2.99792458e18

# Every non-self-contained torus must conserve energy under the conserving policy.
_TORI = sorted(set(AGN_BLOCKS["torus"]) - set(_SELF_CONTAINED_TORI))

#: Tori whose library keeps its inclination anisotropy -> (the block's sightline parameter, the
#: block parameters giving the library point on its other axes, in the axis order of
#: ``tests/physics/_template_torus_reference.py``). Every one reads the AGN's one inclination,
#: ``agn_cos_inc``, which the disc reads too.
_ANISOTROPIC_TORI = {
    "fritz": (
        "agn_cos_inc",
        (
            "agn_fritz_r_ratio",
            "agn_fritz_tau",
            "agn_fritz_beta",
            "agn_fritz_gamma",
            "agn_fritz_oa",
        ),
    ),
    "nenkova_agnfitter": ("agn_cos_inc", ()),
    "nenkova_agnfitter_2p": ("agn_cos_inc", ("agn_oa_nenkova",)),
    "nenkova_agnfitter_3p": ("agn_cos_inc", ("agn_oa_nenkova", "agn_tv_nenkova")),
    "skirtor_agnfitter_1p": ("agn_cos_inc", ()),
}

#: Gauss-Legendre order of the sphere mean in ``mu = cos i`` over [0, 1]. The anisotropy is a
#: PCHIP profile with kinks at the library nodes, so the rule converges algebraically; at 16 the
#: torus sphere mean is ``f L_bol`` to 7.5e-5 (NK0_mean_1p, the worst of the five), far inside
#: the 2 % invariance tolerance.
_MU_ORDER = 16

#: Bound on the sightline torus power against ``f L_bol a(mu)/mean a`` of the raw records.
#: Measured at the default sightline: 3.7e-7 (Fritz) to 8.8e-7 (NK0_mean_2p).
_LOS_RTOL = 1e-5

_TORUS_FRACS = (0.3, 0.6, 0.9)


def _band_energy(l_nu, wave):
    """Integrate L_nu over frequency to get the band-limited bolometric [erg/s]."""
    nu = _C_AA_PER_S / wave
    order = np.argsort(nu)
    return np.trapezoid(np.asarray(l_nu)[order], nu[order])


def _compose_or_skip(torus, tf, **sightline):
    try:
        return composable(
            _WAVE,
            45.0,
            agn_disc_block="powerlaw",
            agn_torus_block=torus,
            agn_norm="conserving",
            agn_lum_ratio=1.0,
            agn_torus_frac=tf,
            **sightline,
        )
    except (FileNotFoundError, OSError) as exc:
        # data-gated tori (template h5 absent in CI). Narrow on purpose: a
        # ValueError/KeyError is a real bug we want to surface, not skip.
        pytest.skip(f"torus '{torus}' unavailable: {type(exc).__name__}: {exc}")


def _block_default(torus, name):
    return inspect.signature(AGN_BLOCKS["torus"][torus]).parameters[name].default


def _sightline_kwargs(torus, mu):
    """The keyword placing the observer at ``mu = cos i``, for the torus and the disc alike."""
    return {_ANISOTROPIC_TORI[torus][0]: float(mu)}


def _default_mu(torus):
    """``cos i`` of the sightline a call without a sightline keyword is seen along."""
    return float(DEFAULT_AGN_COS_INC)


def _torus_power(torus, tf, e_disc, **sightline):
    """Band energy of the torus piece: the total minus the disc debited by ``1 - tf``.

    Along a fixed sightline the powerlaw disc depends on ``agn_torus_frac`` only through the
    conserving debit, so ``e_disc`` (the ``tf = 0`` total on that sightline) times ``1 - tf``
    is the disc piece.
    """
    return _band_energy(_compose_or_skip(torus, tf, **sightline), _WAVE) - (1.0 - tf) * e_disc


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("torus", _TORI)
def test_conserving_policy_is_invariant_under_torus_frac(torus):
    """Under ``agn_norm='conserving'`` the total emitted energy is invariant as
    ``agn_torus_frac`` shifts energy from disc to torus, and it equals L_bol.

    For an isotropic torus one sightline is the total. For a torus that keeps its library's
    anisotropy the invariant is the angle-integrated total: the debited disc plus the torus
    power averaged over the sphere in ``mu = cos i`` (owner ruling 2026-10-13, the ledger
    closes on ``agn_log_lbol`` on the sphere mean). Its sightline is asserted in
    :func:`test_anisotropic_torus_sightline_carries_its_anisotropy`.
    """
    e0 = _band_energy(_compose_or_skip(torus, 0.0), _WAVE)

    # Absolute anchor: at agn_log_lbol=45 the disc-only (torus_frac=0) total is
    # ~10^45 L_sun in erg/s. This catches a global mis-scale (e.g. disc AND
    # torus both inflated 2x) that the relative invariance below cannot see.
    l_bol_erg = 10.0**45 * L_SUN
    assert e0 == pytest.approx(l_bol_erg, rel=0.1), (
        f"{torus}: disc-only band energy {e0:.3e} erg/s != L_bol {l_bol_erg:.3e}"
    )

    if torus not in _ANISOTROPIC_TORI:
        # Relative invariance: energy is conserved as it moves disc -> torus.
        for tf in _TORUS_FRACS:
            e = _band_energy(_compose_or_skip(torus, tf), _WAVE)
            assert e == pytest.approx(e0, rel=0.02), (
                f"{torus}: energy not conserved at torus_frac={tf} "
                f"({e:.3e} vs {e0:.3e}) — disc not debited"
            )
        return

    x, w = np.polynomial.legendre.leggauss(_MU_ORDER)
    mus, weights = 0.5 * (x + 1.0), 0.5 * w
    e_disc = [
        _band_energy(_compose_or_skip(torus, 0.0, **_sightline_kwargs(torus, m)), _WAVE)
        for m in mus
    ]
    for tf in _TORUS_FRACS:
        torus_mean = sum(
            wt * _torus_power(torus, tf, ed, **_sightline_kwargs(torus, m))
            for m, wt, ed in zip(mus, weights, e_disc, strict=True)
        )
        e = (1.0 - tf) * e0 + torus_mean
        assert e == pytest.approx(e0, rel=0.02), (
            f"{torus}: angle-integrated energy not conserved at torus_frac={tf} "
            f"({e:.3e} vs {e0:.3e}) — disc not debited by the sphere-mean torus power"
        )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("torus", sorted(_ANISOTROPIC_TORI))
def test_anisotropic_torus_sightline_carries_its_anisotropy(torus):
    """Along the default sightline the torus piece is ``f L_bol a(mu)/mean a``.

    The disc debit is the angle-integrated share ``1 - f`` whatever the sightline, so the
    sightline total differs from the angle-integrated one by exactly the torus piece's
    anisotropy, ``a(mu)/mean a`` of the raw library records (an independent h5py + SciPy
    reader, ``tests/physics/_template_torus_reference.py``).
    """
    filename = LIBRARIES[torus][0]
    path = find_data(filename)
    if path is None:
        pytest.skip(f"template data unavailable ({filename})")
    raw = RawTorus(torus, str(path))
    point = tuple(float(_block_default(torus, k)) for k in _ANISOTROPIC_TORI[torus][1])
    mu = _default_mu(torus)
    ratio = float(raw.power(point, mu)) / raw.sphere_mean(point)
    # The default sightline must see the anisotropy, or this test cannot fail.
    assert abs(ratio - 1.0) > 0.1, f"{torus}: a(mu)/mean a = {ratio:.4f} at mu = {mu:.4f}"

    l_bol_erg = 10.0**45 * L_SUN
    e_disc = _band_energy(_compose_or_skip(torus, 0.0), _WAVE)
    for tf in _TORUS_FRACS:
        los = _torus_power(torus, tf, e_disc)
        assert los == pytest.approx(tf * l_bol_erg * ratio, rel=_LOS_RTOL), (
            f"{torus}: sightline torus power {los:.6e} at torus_frac={tf} is not "
            f"f L_bol a(mu)/mean a = {tf * l_bol_erg * ratio:.6e}"
        )
