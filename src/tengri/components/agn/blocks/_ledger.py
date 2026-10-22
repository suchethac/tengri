# SPDX-License-Identifier: BSD-3-Clause
r"""Energy ledger of the composable AGN runner (NAMING_CONTRACT section 4c, "Composable runner").

The composable runner publishes, for each piece of the SED it emits (the disc, the torus, the
emission lines and the polar dust), two powers, each a ``log_L_agn_<piece>`` [dex re erg/s]
with its ``L_agn_<piece>_lsun`` [Lsun] sibling:

- ``log_L_agn_<piece>``: the angle-integrated power, the sphere mean
  :math:`\int_0^1 P(\mu)\,d\mu` of the piece's emitted power :math:`P(\mu)` along
  :math:`\mu = \cos i` (C1);
- ``log_L_agn_<piece>_los``: the frequency integral of the piece as emitted along the model's
  sightline ``agn_cos_inc``, so it equals the integral of that piece of the SED (C3).

``log_L_agn_los`` is the emitted line-of-sight sum (the integral of ``sed_agn``). The power a
piece carries but does not emit is published as ``log_L_agn_<piece>_unemitted``: the disc and
broad-line light the torus screen, the polar dust or an attenuation block removes (D1 publishes
it; the torus does not yet re-emit it), and the torus light a foreground attenuation block
removes. On a conserving recipe the sphere-mean pieces plus the unemitted ones add up to the
accretion power ``agn_lum_ratio * 10**agn_log_lbol L_sun`` (C2), and ``agn_ledger_closed`` is 1.

The sphere means are Gauss-Legendre over :math:`\mu`, of the runner's own emitted pieces
re-evaluated at the quadrature sightlines (one ``vmap``), except the torus of a library that
keeps its flux ratio between sightlines, whose sphere mean is its sightline power over the
library's anisotropy :math:`a(\mu)/\bar a` (``TORUS_ANISOTROPY_BLOCKS``), exact to the library
interpolant.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.agn._publication import (
    PUBLICATION_HI_AA,
    PUBLICATION_LO_AA,
    log10_emitted_bolometric,
    with_lsun_siblings,
)
from tengri.utils.scale import log10_add, log10_weighted_sum

__all__ = [
    "COMPOSABLE_LEDGER_OUTPUTS",
    "LEDGER_CLOSED_KEY",
    "LEDGER_KEYS",
    "LEDGER_PIECES",
    "NON_CONSERVING_DISCS",
    "NON_CONSERVING_TORI",
    "ledger_derived_keys",
    "ledger_from_evaluations",
    "ledger_policy_closes",
    "ledger_quadrature",
    "ledger_wave",
    "non_conserving_reason",
]

#: Published piece name -> key of the runner's ``components`` dict holding its SED.
LEDGER_PIECES: dict[str, str] = {
    "disc": "disc",
    "torus": "torus",
    "lines": "lines",
    "polar_dust": "polar",
}

#: Pieces whose budget can be partly removed before it leaves the AGN.
_UNEMITTED_PIECES: tuple[str, ...] = ("disc", "torus", "lines")

#: Key of the published ledger flag: 1.0 when the pieces close on the accretion power, else 0.0.
LEDGER_CLOSED_KEY = "agn_ledger_closed"

#: Every ``log_L_agn_*`` key the ledger publishes [dex re erg/s] (each with its ``_lsun`` sibling).
LEDGER_KEYS: tuple[str, ...] = (
    "log_L_agn_los",
    *(f"log_L_agn_{p}" for p in LEDGER_PIECES),
    *(f"log_L_agn_{p}_los" for p in LEDGER_PIECES),
    *(f"log_L_agn_{p}_unemitted" for p in _UNEMITTED_PIECES),
)

#: Every key the ledger publishes: the powers, their ``Lsun`` siblings and the flag.
COMPOSABLE_LEDGER_OUTPUTS: tuple[str, ...] = (
    *LEDGER_KEYS,
    *(f"{k[len('log_') :]}_lsun" for k in LEDGER_KEYS),
    LEDGER_CLOSED_KEY,
)

_PIECE_PROSE: dict[str, str] = {
    "disc": "disc (with its corona)",
    "torus": "torus dust",
    "lines": "NLR + BLR + FeII emission lines",
    "polar_dust": "polar-dust re-emission",
}


def _key_description(key: str) -> str:
    """One-line description of a ledger ``log_L_agn_*`` key."""
    if key == "log_L_agn_los":
        return (
            "log10 of the emitted AGN SED integrated over frequency along agn_cos_inc "
            "[dex re erg/s], the line-of-sight sum of the ledger pieces"
        )
    stem = key[len("log_L_agn_") :]
    for suffix, prose in (
        (
            "_unemitted",
            "carried by the {p} but removed before it leaves the AGN (torus screen, "
            "polar dust, attenuation block, disc reddening), integrated over all directions",
        ),
        ("_los", "of the {p} as emitted along agn_cos_inc, the integral of that piece of the SED"),
        ("", "of the {p} as emitted, integrated over all directions (sphere mean over cos i)"),
    ):
        if stem.endswith(suffix) and stem[: len(stem) - len(suffix)] in _PIECE_PROSE:
            piece = stem[: len(stem) - len(suffix)]
            return "log10 power [dex re erg/s] " + prose.format(p=_PIECE_PROSE[piece])
    raise KeyError(key)


def ledger_derived_keys() -> tuple:
    """The ledger's published keys as ``DerivedKey`` declarations (composable model only).

    Returns
    -------
    tuple of DerivedKey
        One per entry of :data:`COMPOSABLE_LEDGER_OUTPUTS`.
    """
    from tengri.protocols.component import DerivedKey

    keys = [
        DerivedKey(k, "dex", f"{_key_description(k)}; composable model only (#3076)")
        for k in LEDGER_KEYS
    ]
    keys += [
        DerivedKey(f"{k[len('log_') :]}_lsun", "Lsun", f"{k} in solar luminosities")
        for k in LEDGER_KEYS
    ]
    keys.append(
        DerivedKey(
            LEDGER_CLOSED_KEY,
            "",
            "1.0 when the angle-integrated pieces plus the unemitted ones add up to the "
            "accretion power agn_lum_ratio * 10**agn_log_lbol L_sun, else 0.0 (self-contained "
            "quasar templates, agn_norm='independent', 'cigale_joint' with lines or an active "
            "agn_ir_frac tie, polar dust beside a library torus with anisotropy); composable "
            "model only",
        )
    )
    return tuple(keys)


#: Torus blocks whose single template carries disc, torus and lines together on their own scale
#: (the GRAHSP and QSOgen empirical quasar SEDs): nothing in them is debited from the accretion
#: power, so the ledger does not close (decision 3 of the 2026-10-13 rulings).
NON_CONSERVING_TORI: frozenset[str] = frozenset({"grahsp", "grahsp_mn12", "qsogen"})

#: Disc blocks the ledger cannot close on ``agn_log_lbol``: the GRAHSP discs are normalized by
#: ``l5100`` (their parameter), the QSOgen disc is an empirical template on its own scale, and
#: ``schartmann2005_skirtor_atten`` applies the SKIRTOR library's disc attenuation inside the
#: block, so the light it removes is in no published piece.
NON_CONSERVING_DISCS: frozenset[str] = frozenset(
    {"grahsp_sbpl", "grahsp_netzer", "qsogen", "schartmann2005_skirtor_atten"}
)

#: Gauss-Legendre panels uniform in ``cos i`` and points per panel of the sphere means.
_SPHERE_PANELS: int = 16
_SPHERE_ORDER: int = 3

#: Log-uniform node count of the ledger grid over the publication range (its spacing outside
#: the fine window), and the fine window [Angstrom] (1.2e-4 in ln lambda, 35 km/s).
_LEDGER_NODES: int = 8000
_FINE_WINDOW_AA: tuple[float, float, int] = (9.0e2, 3.0e5, 48001)


def ledger_quadrature(
    n_panels: int = _SPHERE_PANELS, order: int = _SPHERE_ORDER
) -> tuple[np.ndarray, np.ndarray]:
    r"""Points and weights for :math:`\int_0^1 f(\mu)\,d\mu`, composite Gauss-Legendre.

    Parameters
    ----------
    n_panels : int, optional
        Panels of equal width in :math:`\mu = \cos i`.
    order : int, optional
        Points per panel.

    Returns
    -------
    mu : ndarray, shape (n_panels * order,)
        Sightlines :math:`\cos i` [dimensionless], within ``(0, 1)``.
    weight : ndarray, shape (n_panels * order,)
        Weights [dimensionless], summing to 1.

    Notes
    -----
    A host (numpy) constant. No point sits on ``mu = 0`` or ``mu = 1``, where a library's
    sightline factor can be singular.
    """
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(0.0, 1.0, n_panels + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    mu = (0.5 * (hi - lo) * x + 0.5 * (hi + lo)).ravel()
    weight = (0.5 * (hi - lo) * w).ravel()
    return mu, weight / weight.sum()


def ledger_wave(dtype) -> jnp.ndarray:
    """The fixed wavelength grid [Angstrom] the ledger integrates the emitted pieces on.

    Parameters
    ----------
    dtype : dtype
        Floating dtype of the grid.

    Returns
    -------
    ndarray, shape (n_ledger,)
        Ascending wavelengths: log-uniform over the publication range of ``_publication`` at
        615 nodes per decade, and 1.2e-4 in ln lambda (35 km/s) over 900 A to 3e5 A.

    Notes
    -----
    The fine window resolves the narrow lines (a 300 km/s line integrates exactly to rounding)
    and the exponential edges a reddening or screen cuts into the ultraviolet disc (measured:
    1e-3 on an SMC-screened disc at the publication spacing, 1e-6 on this grid). It is three
    log-uniform pieces meeting at the window edges, because the line blocks paint each line
    conserving its flux over the cells of the grid they are given: publication nodes
    interleaved inside the window moved the line integral by 3e-5.
    """
    lo, hi, n = _FINE_WINDOW_AA
    per_dex = (_LEDGER_NODES - 1) / np.log10(PUBLICATION_HI_AA / PUBLICATION_LO_AA)
    n_lo = int(np.ceil(per_dex * np.log10(lo / PUBLICATION_LO_AA))) + 1
    n_hi = int(np.ceil(per_dex * np.log10(PUBLICATION_HI_AA / hi))) + 1
    nodes = np.concatenate(
        [
            np.geomspace(PUBLICATION_LO_AA, lo, n_lo)[:-1],
            np.geomspace(lo, hi, n),
            np.geomspace(hi, PUBLICATION_HI_AA, n_hi)[1:],
        ]
    )
    return jnp.asarray(nodes, dtype=dtype)


def non_conserving_reason(
    *, agn_disc_block: str, agn_torus_block: str, agn_norm: str | None
) -> str | None:
    """Why a recipe's ledger cannot close on the accretion power, or ``None``.

    Covers the recipes documented as non-conserving (decision 3 of the 2026-10-13 rulings):
    the self-contained quasar templates, the discs normalized off ``agn_log_lbol`` and the
    ``'independent'`` policy. ``'cigale_joint'`` with lines or an active ``agn_ir_frac`` tie is
    CIGALE's own bookkeeping and is reported by the flag alone.

    Parameters
    ----------
    agn_disc_block, agn_torus_block : str
        Disc and torus selectors.
    agn_norm : str or None
        Normalization policy; ``None`` when the caller does not know it.

    Returns
    -------
    str or None
        A one-sentence reason, or ``None`` when the recipe can conserve.
    """
    if agn_torus_block in NON_CONSERVING_TORI:
        return (
            f"the {agn_torus_block!r} torus is a self-contained quasar template on its own "
            "scale, not debited from the accretion power"
        )
    if agn_disc_block in ("grahsp_sbpl", "grahsp_netzer"):
        return f"the {agn_disc_block!r} disc is normalized by its l5100, not by agn_log_lbol"
    if agn_disc_block == "qsogen":
        return "the 'qsogen' disc is an empirical template on its own scale"
    if agn_disc_block == "schartmann2005_skirtor_atten":
        return (
            "the 'schartmann2005_skirtor_atten' disc removes light inside the block that no "
            "published piece carries"
        )
    if agn_norm == "independent":
        return "agn_norm='independent' keeps every piece on its own luminosity scale"
    return None


def ledger_policy_closes(
    *,
    agn_disc_block: str,
    agn_nlr_block: str,
    agn_blr_block: str,
    agn_feii_block: str,
    agn_torus_block: str,
    agn_attenuation_block: str,
    agn_norm: str,
) -> bool:
    """Whether the recipe's normalization debits every piece from the accretion power.

    Parameters
    ----------
    agn_disc_block, agn_nlr_block, agn_blr_block, agn_feii_block, agn_torus_block, \
agn_attenuation_block : str
        Block selectors.
    agn_norm : str
        Normalization policy.

    Returns
    -------
    bool
        ``True`` for ``'conserving'``, and for ``'cigale_joint'`` without a line block (it adds
        the lines on top, as CIGALE adds nebular emission); ``False`` for the recipes of
        :func:`non_conserving_reason`, and for the ``polar_dust`` block beside a torus whose
        library keeps its anisotropy (``TORUS_ANISOTROPY_BLOCKS``): the runner splits the dust
        budget between the torus and the polar dust on the torus's sightline power, not its
        angle-integrated power, so the pieces miss the accretion power by the anisotropy
        (measured +2.5% for ``nenkova_agnfitter`` at ``cos i = 0.8``). The CIGALE
        ``agn_ir_frac`` tie of the ``skirtor`` and ``fritz`` tori, which normalizes on
        ``agn_power`` instead, is a traced condition the runner applies on top.
    """
    from tengri.components.agn.blocks._protocol import TORUS_ANISOTROPY_BLOCKS

    if non_conserving_reason(
        agn_disc_block=agn_disc_block, agn_torus_block=agn_torus_block, agn_norm=agn_norm
    ):
        return False
    if agn_attenuation_block == "polar_dust" and agn_torus_block in TORUS_ANISOTROPY_BLOCKS:
        return False
    has_lines = any(b != "none" for b in (agn_nlr_block, agn_blr_block, agn_feii_block))
    return agn_norm == "conserving" or (agn_norm == "cigale_joint" and not has_lines)


def _clamped_difference(log_a, log_b):
    """``log10(10**a - 10**b)`` where ``a > b``, else ``-inf`` (nothing left over)."""
    return jnp.where(log_a > log_b, log10_add(log_a, log_b, sign_b=-1.0), -jnp.inf)


def _log_power(lnu, wave):
    """``log10`` of the frequency integral of an emitted ``L_nu`` on ``wave`` [dex re erg/s]."""
    return log10_emitted_bolometric(lnu, wave)


def ledger_from_evaluations(
    components: dict,
    wave,
    weights,
    *,
    log_disc_intrinsic,
    log_lines_intrinsic,
    log_torus_intrinsic=None,
    log_torus_anisotropy=None,
    closed,
) -> dict:
    r"""The published ledger from the runner's pieces at the sightline and the quadrature points.

    Parameters
    ----------
    components : dict
        The runner's ``components`` evaluated on ``wave`` at ``agn_cos_inc`` (row 0) and at the
        quadrature sightlines (rows 1:): ``"disc"``, ``"torus"``, ``"lines"``, ``"polar"`` ->
        ndarray, shape (1 + n_mu, n_ledger) [erg/s/Hz].
    wave : array_like, shape (n_ledger,)
        :func:`ledger_wave` [Angstrom].
    weights : array_like, shape (n_mu,)
        :func:`ledger_quadrature` weights [dimensionless].
    log_disc_intrinsic : float
        ``log10`` of the angle-integrated power of the disc before any obscuration [dex re
        erg/s].
    log_lines_intrinsic : float
        ``log10`` of the line power before any obscuration [dex re erg/s]; ``-inf`` without
        lines.
    log_torus_intrinsic : float, optional
        ``log10`` of the angle-integrated torus power before an attenuation block [dex re
        erg/s]; ``None`` when nothing attenuates the torus.
    log_torus_anisotropy : float, optional
        :math:`\log_{10}(a(\mu)/\bar a)` of a library torus [dex]; ``None`` for a torus that
        carries the same power along every sightline.
    closed : float
        1.0 when the recipe conserves the accretion power, else 0.0.

    Returns
    -------
    dict
        :data:`LEDGER_KEYS` [dex re erg/s] with their ``L_agn_*_lsun`` [Lsun] siblings, and
        :data:`LEDGER_CLOSED_KEY`.

    Notes
    -----
    .. math::

        L_p = \sum_k w_k P_p(\mu_k), \qquad
        L_{p,\,\rm unemitted} = \max(L_{p,\,\rm intrinsic} - L_p,\ 0),

    with :math:`P_p(\mu)` the frequency integral of piece :math:`p` emitted along
    :math:`\mu` (trapezoid in :math:`\ln\nu` on ``wave``) and :math:`(\mu_k, w_k)` the
    quadrature. The torus of a library with anisotropy takes
    :math:`L_{\rm torus} = P_{\rm torus}(\mu_{\rm obs})\,\bar a/a(\mu_{\rm obs})`.

    **JIT/grad/vmap-safe**: yes. Every power is formed in log space (peak-factored integrals,
    base-10 log-sum-exp), so the ledger is finite in pure float32 (NAMING_CONTRACT 4c).
    """
    wave = jnp.asarray(wave)
    w = jnp.asarray(weights, dtype=wave.dtype)
    row_powers = jax.vmap(_log_power, in_axes=(0, None))
    logs = {
        piece: row_powers(jnp.asarray(components[key]), wave)
        for piece, key in LEDGER_PIECES.items()
    }
    total_los = sum(jnp.asarray(components[key])[0] for key in LEDGER_PIECES.values())
    out = {"log_L_agn_los": _log_power(total_los, wave)}
    for piece, row_logs in logs.items():
        out[f"log_L_agn_{piece}_los"] = row_logs[0]
        out[f"log_L_agn_{piece}"] = log10_weighted_sum(row_logs[1:], w)
    if log_torus_anisotropy is not None:
        out["log_L_agn_torus"] = logs["torus"][0] - log_torus_anisotropy
    intrinsic = {
        "disc": log_disc_intrinsic,
        "lines": log_lines_intrinsic,
        "torus": -jnp.inf if log_torus_intrinsic is None else log_torus_intrinsic,
    }
    for piece in _UNEMITTED_PIECES:
        out[f"log_L_agn_{piece}_unemitted"] = _clamped_difference(
            jnp.asarray(intrinsic[piece], dtype=wave.dtype), out[f"log_L_agn_{piece}"]
        )
    out = with_lsun_siblings(out)
    out[LEDGER_CLOSED_KEY] = jnp.asarray(closed, dtype=wave.dtype)
    return out
