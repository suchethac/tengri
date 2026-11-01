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
broad-line light the torus screen, the polar dust or an attenuation block removes (the torus
does not yet re-emit it), and the torus light a foreground attenuation block removes. On a
conserving recipe the sphere-mean pieces plus the unemitted ones add up to the accretion power
``agn_lum_ratio * 10**agn_log_lbol L_sun`` (C2), and ``agn_ledger_closed`` is 1.

The sphere means are Gauss-Legendre over :math:`\mu`, of the runner's own emitted pieces
re-evaluated at the quadrature sightlines (one ``vmap``), except the torus of a library that
keeps its flux ratio between sightlines, whose sphere mean is its sightline power over the
library's anisotropy :math:`a(\mu)/\bar a` (``TORUS_ANISOTROPY_BLOCKS``), exact to the library
interpolant. The frequency integrals are trapezoids in :math:`\ln\nu` on :func:`ledger_wave`,
which carries a node pair at every spectral edge of the selected blocks
(``spectral_edges.block_edges``), so they converge at second order.

The ledger is computed on request, as properties of the model (the AGN component's
``energy_ledger``), never in the forward pass.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.agn._publication import (
    PUBLICATION_HI_AA,
    PUBLICATION_LO_AA,
    log10_emitted_bolometric,
    publication_wave,
    with_lsun_siblings,
)
from tengri.utils.physics_constants import C_AA
from tengri.utils.scale import log10_add, log10_weighted_sum

__all__ = [
    "COMPOSABLE_LEDGER_OUTPUTS",
    "LEDGER_CLOSED_KEY",
    "LEDGER_KEYS",
    "LEDGER_PIECES",
    "NON_CONSERVING_DISCS",
    "NON_CONSERVING_TORI",
    "POWER_KEY",
    "SPHERE_ORDER",
    "ledger_from_evaluations",
    "ledger_policy_closes",
    "ledger_property_specs",
    "ledger_sightlines",
    "ledger_wave",
    "log10_signed_sphere_mean",
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

#: The AGN dust power the torus and polar dust are normalized on (CIGALE's ``agn_power``).
POWER_KEY = "log_L_agn_power"

#: Every key the ledger publishes: the powers, their ``Lsun`` siblings and the flag.
COMPOSABLE_LEDGER_OUTPUTS: tuple[str, ...] = (
    *LEDGER_KEYS,
    POWER_KEY,
    *(f"{k[len('log_') :]}_lsun" for k in (*LEDGER_KEYS, POWER_KEY)),
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


def ledger_property_specs() -> dict[str, tuple[str, str]]:
    """Units and one-line description of every key the ledger publishes.

    Returns
    -------
    dict
        ``{key: (units, doc)}`` for :data:`COMPOSABLE_LEDGER_OUTPUTS`, the order the property
        catalog lists them in.
    """
    specs: dict[str, tuple[str, str]] = {}
    for k in LEDGER_KEYS:
        specs[k] = ("dex", f"{_key_description(k)}; composable AGN only")
    specs[POWER_KEY] = (
        "dex",
        "log10 of the AGN dust power the runner normalizes the torus and the polar dust on "
        "[dex re erg/s], the torus block's power along agn_cos_inc (CIGALE's agn_power, the "
        "anchor of the agn_ir_frac tie); composable AGN with a torus only",
    )
    for k in (*LEDGER_KEYS, POWER_KEY):
        specs[f"{k[len('log_') :]}_lsun"] = ("Lsun", f"{k} in solar luminosities")
    specs[LEDGER_CLOSED_KEY] = (
        "",
        "1.0 when the angle-integrated pieces plus the unemitted ones add up to the "
        "accretion power agn_lum_ratio * 10**agn_log_lbol L_sun, else 0.0 (self-contained "
        "quasar templates, agn_norm='independent', 'cigale_joint' with lines or an active "
        "agn_ir_frac tie, polar dust beside a library torus with anisotropy); composable AGN only",
    )
    return specs


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

#: Panels uniform in ``cos i`` the sightline quadrature starts from, and Gauss-Legendre points per
#: panel. The library's inclination nodes and the Type-1/2 transitions are added as panel edges.
_SPHERE_BASE_PANELS: int = 8
SPHERE_ORDER: int = 4

#: Offsets [units of the transition width] of the panel edges laid about each Type-1/2 sigmoid
#: (``polar_dust.type1_weight``), whose width in ``cos i`` is ``TYPE_TRANSITION_WIDTH``.
_SIGMOID_PANEL_OFFSETS: tuple[float, ...] = (-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0)

#: Log-uniform node count of the ledger grid over the publication range (its spacing outside
#: the fine window), and the fine window [Angstrom] (1.2e-4 in ln lambda, 35 km/s).
_LEDGER_NODES: int = 32000
_FINE_WINDOW_AA: tuple[float, float, int] = (9.0e2, 3.0e5, 48001)

#: Half-width, relative, of the node pair put on each side of a spectral edge.
_EDGE_PAIR_REL: float = 1.0e-9


def ledger_wave(dtype, edges_aa=()) -> jnp.ndarray:
    """The fixed wavelength grid [Angstrom] the ledger integrates the emitted pieces on.

    Parameters
    ----------
    dtype : dtype
        Floating dtype of the grid.
    edges_aa : sequence of float, optional
        Spectral edges of the selected blocks [Angstrom] (``spectral_edges.block_edges``); a
        node pair ``edge * (1 -/+ 1e-9)`` is put on each.

    Returns
    -------
    ndarray, shape (n_ledger,)
        ``_publication.publication_wave`` built in the trace: 2460 log nodes per decade, the fine
        window 900 A to 3e5 A at 1.2e-4 in ln lambda, and the edge pairs.

    Notes
    -----
    With a node on each side of every discontinuity the trapezoid converges at second order;
    without, at first order, with an error set by where the step falls between nodes
    (measured on the multicolor disc's 30 A cut: 6.8e-5 of its power on this grid without the
    pair, 5e-7 with it). The fine window resolves the narrow lines and the exponential edges a
    reddening or screen cuts into the ultraviolet disc.
    """
    edges = np.asarray(edges_aa, dtype=np.float64)
    edges = edges[(edges > PUBLICATION_LO_AA) & (edges < PUBLICATION_HI_AA)]
    pairs = np.concatenate([edges * (1.0 - _EDGE_PAIR_REL), edges * (1.0 + _EDGE_PAIR_REL)])
    return publication_wave(
        dtype, extra_aa=pairs, n_nodes=_LEDGER_NODES, fine_window=_FINE_WINDOW_AA, traced=True
    )


def ledger_sightlines(library_cos_nodes, sigmoid_centers, dtype):
    r"""Sightlines and weights for the sphere means :math:`\int_0^1 f(\mu)\,d\mu`.

    Parameters
    ----------
    library_cos_nodes : array_like, shape (n_lib,)
        The torus library's inclination nodes in :math:`\cos i` (host values; empty without a
        library), where its interpolated sightline factors have their panel breaks.
    sigmoid_centers : sequence of array_like
        :math:`\cos i` at the center of each active Type-1/2 sigmoid (may be traced).
    dtype : dtype
        Evaluation dtype.

    Returns
    -------
    mu : ndarray, shape (n_mu,)
        Sightlines :math:`\cos i` [dimensionless].
    weight : ndarray, shape (n_mu,)
        Weights [dimensionless], summing to 1.

    Notes
    -----
    ``_torus_inclination.inclination_quadrature_in`` (Gauss-Legendre of ``SPHERE_ORDER`` points
    on each panel) on the union of uniform panels, the library's nodes and edges laid about each
    sigmoid, so every panel is a smooth stretch of the integrand. **JIT/vmap-safe**: yes; the
    panel count is static and the sigmoid edges may be traced.
    """
    from tengri.components.agn._torus_inclination import inclination_quadrature_in
    from tengri.components.agn.polar_dust import TYPE_TRANSITION_WIDTH

    host = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, _SPHERE_BASE_PANELS + 1)[1:-1],
                np.clip(np.asarray(library_cos_nodes, dtype=np.float64), 0.0, 1.0),
            ]
        )
    )
    host = host[(host > 0.0) & (host < 1.0)]
    nodes = jnp.asarray(host, dtype=dtype)
    if sigmoid_centers:
        offsets = jnp.asarray(_SIGMOID_PANEL_OFFSETS, dtype=dtype) * TYPE_TRANSITION_WIDTH
        # Edges laid past an end are clipped onto it by the quadrature (an empty panel).
        laid = [jnp.asarray(c, dtype=dtype) + offsets for c in sigmoid_centers]
        nodes = jnp.sort(jnp.concatenate([nodes, *laid]))
    if not isinstance(nodes, jax.core.Tracer):
        # Coinciding edges (the polar dust follows the torus angle, a laid edge clipped onto an
        # end) make empty panels of zero weight; concrete nodes drop them, as the quadrature
        # refuses ties it can see.
        nodes = np.unique(np.asarray(nodes))
        nodes = jnp.asarray(nodes[(nodes > 0.0) & (nodes < 1.0)], dtype=dtype)
    return inclination_quadrature_in(nodes, "cos_inc", SPHERE_ORDER)


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


def log10_signed_sphere_mean(rows, wave, weights):
    r"""``log10`` of the sphere mean of the power of a signed spectrum, ``-inf`` if not positive.

    Parameters
    ----------
    rows : array_like, shape (n_mu, n_ledger)
        A spectrum per sightline [erg/s/Hz] that may change sign node by node (the intrinsic
        minus the emitted disc: a library sightline factor ``2 cos i`` exceeds one on the
        near-face-on rows of an isotropic disc).
    wave : array_like, shape (n_ledger,)
        :func:`ledger_wave` [Angstrom].
    weights : array_like, shape (n_mu,)
        :func:`ledger_sightlines` weights [dimensionless].

    Returns
    -------
    ndarray, shape ()
        :math:`\log_{10}\sum_k w_k \int R_k\,d\nu` [dex re erg/s].

    Notes
    -----
    **JIT/grad/vmap-safe**: yes. Peak-factored as ``_publication.log10_emitted_bolometric``
    (one peak for every row, held under ``stop_gradient``), so no erg/s-scale value forms in
    float32; the sign survives the sum, unlike a per-row logarithm.
    """
    rows = jnp.asarray(rows)
    nu = C_AA / jnp.asarray(wave)
    peak = jax.lax.stop_gradient(jnp.max(jnp.abs(rows), initial=0.0))
    safe_peak = jnp.where(peak > 0, peak, jnp.ones_like(peak))
    per_row = -jnp.trapezoid(rows / safe_peak * nu, jnp.log(nu), axis=-1)
    total = jnp.sum(jnp.asarray(weights, dtype=rows.dtype) * per_row)
    positive = total > 0
    return jnp.where(
        positive, jnp.log10(jnp.where(positive, total, 1.0)) + jnp.log10(safe_peak), -jnp.inf
    )


def _clamped_difference(log_a, log_b):
    """``log10(10**a - 10**b)`` where ``a > b``, else ``-inf`` (nothing left over)."""
    return jnp.where(log_a > log_b, log10_add(log_a, log_b, sign_b=-1.0), -jnp.inf)


def ledger_from_evaluations(
    components: dict,
    wave,
    weights,
    *,
    log_disc_intrinsic,
    log_lines_intrinsic,
    log_disc_unemitted=None,
    log_lines_unemitted=None,
    log_torus_intrinsic=None,
    log_torus_anisotropy=None,
    log_power=None,
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
        :func:`ledger_sightlines` weights [dimensionless].
    log_disc_intrinsic : float
        ``log10`` of the angle-integrated power of the disc before any obscuration [dex re
        erg/s].
    log_lines_intrinsic : float
        ``log10`` of the line power before any obscuration [dex re erg/s]; ``-inf`` without
        lines.
    log_disc_unemitted, log_lines_unemitted : float, optional
        ``log10`` of the unemitted disc and lines, integrated from the per-node difference
        between the intrinsic and the emitted spectra [dex re erg/s]; ``None`` forms them as
        the difference of the intrinsic and emitted powers.
    log_torus_intrinsic : float, optional
        ``log10`` of the angle-integrated torus power before an attenuation block [dex re
        erg/s]; ``None`` when nothing attenuates the torus.
    log_torus_anisotropy : float, optional
        :math:`\log_{10}(a(\mu)/\bar a)` of a library torus [dex]; ``None`` for a torus that
        carries the same power along every sightline.
    log_power : float, optional
        ``log10`` of the AGN dust power the torus and polar dust are normalized on [dex re
        erg/s]; ``None`` (published ``-inf``) without a torus.
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
    row_powers = jax.vmap(log10_emitted_bolometric, in_axes=(0, None))
    logs = {
        piece: row_powers(jnp.asarray(components[key]), wave)
        for piece, key in LEDGER_PIECES.items()
    }
    total_los = sum(jnp.asarray(components[key])[0] for key in LEDGER_PIECES.values())
    out = {"log_L_agn_los": log10_emitted_bolometric(total_los, wave)}
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
    direct = {"disc": log_disc_unemitted, "lines": log_lines_unemitted, "torus": None}
    for piece in _UNEMITTED_PIECES:
        out[f"log_L_agn_{piece}_unemitted"] = (
            _clamped_difference(
                jnp.asarray(intrinsic[piece], dtype=wave.dtype), out[f"log_L_agn_{piece}"]
            )
            if direct[piece] is None
            else jnp.asarray(direct[piece], dtype=wave.dtype)
        )
    out[POWER_KEY] = jnp.asarray(-jnp.inf if log_power is None else log_power, dtype=wave.dtype)
    out = with_lsun_siblings(out)
    out[LEDGER_CLOSED_KEY] = jnp.asarray(closed, dtype=wave.dtype)
    return out
