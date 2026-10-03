# SPDX-License-Identifier: MIT
r"""The one young/old mixture of age-windowed dust screens (any number of screens).

Every attenuator that splits a stellar population by age (the Charlot & Fall
birth-cloud/diffuse pair of ``two_component``; the N windows of ``age_binned``)
reduces to the same arithmetic.  Sorted window edges :math:`b_0 < \dots < b_{m-1}`
cut the age axis into :math:`m + 1` intervals :math:`j`; the transmission in
interval :math:`j` is the product of the screens whose windows cover it,

.. math::

    T_j(\lambda) = f_{\rm obs} + (1 - f_{\rm obs})
        \exp\!\Bigl[-\sum_{i \in {\rm cover}(j)} \tau_i\,k_i(\lambda)\Bigr],

and the transmission of an SSP node :math:`a` is the *mixture of the stellar
populations* the age kernel put on it,

.. math::

    T_a(\lambda) = \sum_j F_j(a)\,T_j(\lambda),
    \qquad F_j(a) = G_j(a) - G_{j-1}(a),

with :math:`G_j(a)` the fraction of node :math:`a`'s formed mass younger than
:math:`b_j` (:func:`tengri.components.stellar.age_boundary`), :math:`G_{-1} = 0`
and :math:`G_{m} = 1`.  The :math:`F_j` telescope: they sum to one at every
node.  A node whose mass lies wholly in one interval reproduces that
interval's transmission exactly; a node that straddles an edge gets the
mass-weighted mixture of the two populations' transmissions (a mixture of
*transmissions*, not of optical depths).

The nebular continuum and lines are weighted by where the ionizing photons come
from instead of by mass: :math:`T_{\rm neb} = \sum_j q_j T_j` with
:math:`q_j = \sum_a F_j(a) L_{\rm LyC}(a) / \sum_a L_{\rm LyC}(a)`
(:func:`ionizing_interval_weights`).

Notes
-----
**JIT/grad/vmap-compatible**: the window tuple and the covering matrix are
static Python data; everything else is pure ``jnp``.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

__all__ = [
    "interval_cover",
    "interval_fractions",
    "interval_optical_depth",
    "interval_transmission",
    "ionizing_interval_weights",
    "lyc_interval_transmissions",
    "mix_intervals",
    "nebular_interval_weights",
    "weighted_interval_transmission",
    "window_boundaries",
]

#: One screen's age window, ``(lo_yr, hi_yr)``; ``None`` is unbounded on that side.
Window = tuple[float | None, float | None]


def window_boundaries(windows: Sequence[Window]) -> tuple[float, ...]:
    """Sorted, de-duplicated finite window edges [yr].

    Parameters
    ----------
    windows : sequence of (float or None, float or None)
        Per-screen ``(lo_yr, hi_yr)`` age windows.

    Returns
    -------
    tuple of float
        The age boundaries the stellar component must publish younger-than
        fractions for, in ascending order.
    """
    return tuple(sorted({float(e) for w in windows for e in w if e is not None}))


def interval_cover(
    windows: Sequence[Window], boundaries: Sequence[float]
) -> tuple[tuple[bool, ...], ...]:
    """Which screen covers which age interval, ``[screen][interval]``.

    Interval ``j`` spans ``[boundaries[j-1], boundaries[j])`` with the outer
    edges at 0 and infinity.  A screen covers it when its window contains it.

    Parameters
    ----------
    windows : sequence of (float or None, float or None)
        Per-screen ``(lo_yr, hi_yr)`` windows; every finite edge must be in
        ``boundaries``.
    boundaries : sequence of float
        Ascending interval edges [yr].

    Returns
    -------
    tuple of tuple of bool
        ``cover[i][j]``, static.
    """
    b = tuple(boundaries)
    n_int = len(b) + 1
    out = []
    for lo, hi in windows:
        row = []
        for j in range(n_int):
            lo_ok = lo is None or (j > 0 and float(lo) <= b[j - 1])
            hi_ok = hi is None or (j < len(b) and float(hi) >= b[j])
            row.append(bool(lo_ok and hi_ok))
        out.append(tuple(row))
    return tuple(out)


def interval_fractions(younger_fraction: jnp.ndarray) -> jnp.ndarray:
    """Per-node formed-mass fraction in each age interval.

    Parameters
    ----------
    younger_fraction : ndarray, shape (n_boundary, n_age)
        ``G_j(a)``: fraction of node ``a``'s mass younger than boundary ``j``
        (ascending boundaries), in ``[0, 1]``.

    Returns
    -------
    ndarray, shape (n_boundary + 1, n_age)
        ``F_j(a) = G_j - G_{j-1}`` [dimensionless]; sums to one over ``j``.
    """
    g = jnp.asarray(younger_fraction)
    n_age = g.shape[1]
    padded = jnp.concatenate([jnp.zeros((1, n_age), g.dtype), g, jnp.ones((1, n_age), g.dtype)])
    return padded[1:] - padded[:-1]


def interval_optical_depth(
    tau_k: Sequence[jnp.ndarray], cover: Sequence[Sequence[bool]]
) -> jnp.ndarray:
    """Optical depth of every age interval: the sum over covering screens.

    Parameters
    ----------
    tau_k : sequence of ndarray, shape (n_wave,)
        ``tau_i * k_i(lambda)`` per screen, in screen order [dimensionless].
    cover : sequence of sequence of bool
        :func:`interval_cover` output.

    Returns
    -------
    ndarray, shape (n_interval, n_wave)
        Optical depth per interval [dimensionless]; zero where no screen
        covers the interval.
    """
    rows = []
    for j in range(len(cover[0])):
        acc = None
        for i, t in enumerate(tau_k):
            if cover[i][j]:
                acc = t if acc is None else acc + t
        rows.append(jnp.zeros_like(tau_k[0]) if acc is None else acc)
    return jnp.stack(rows)


def interval_transmission(tau_int: jnp.ndarray, f_obscuration=0.0) -> jnp.ndarray:
    r"""Per-interval transmission :math:`f_{\rm obs} + (1 - f_{\rm obs})\,e^{-\tau}`.

    Parameters
    ----------
    tau_int : ndarray, shape (n_interval, n_wave)
        Optical depth per interval [dimensionless].
    f_obscuration : float or ndarray, optional
        Unattenuated-sightline fraction [dimensionless, in [0, 1]].

    Returns
    -------
    ndarray, shape (n_interval, n_wave)
        Transmission in ``[0, 1]``.
    """
    f = jnp.asarray(f_obscuration)
    return f + (1.0 - f) * jnp.exp(-tau_int)


def mix_intervals(fractions: jnp.ndarray, per_interval: jnp.ndarray) -> jnp.ndarray:
    """Mixture over intervals: ``sum_j F_j(a) X_j(...)``.

    Parameters
    ----------
    fractions : ndarray, shape (n_interval, n_age)
        :func:`interval_fractions` output.
    per_interval : ndarray, shape (n_interval, ...)
        Any per-interval quantity (transmission, gate, ...).

    Returns
    -------
    ndarray, shape (n_age, ...)
        The mixture at every SSP node.
    """
    per_interval = jnp.asarray(per_interval)
    # Elementwise multiply-and-sum, not an einsum: the interval axis is 2-3 long, and
    # a dot_general with a contraction that short, batched by vmap inside a larger
    # jitted graph, miscompiles on XLA CPU (measured: a uniform 1e-4 scale error).
    f = fractions.reshape(fractions.shape + (1,) * (per_interval.ndim - 1))
    return jnp.sum(f * per_interval[:, None, ...], axis=0)


def weighted_interval_transmission(weights: jnp.ndarray, per_interval: jnp.ndarray) -> jnp.ndarray:
    """``sum_j q_j X_j(...)`` for interval weights ``q`` that do not depend on node.

    Parameters
    ----------
    weights : ndarray, shape (n_interval,)
        Interval weights summing to one.
    per_interval : ndarray, shape (n_interval, ...)
        Per-interval quantity.

    Returns
    -------
    ndarray, shape (...)
        The weighted sum.
    """
    return jnp.tensordot(weights, per_interval, axes=(0, 0))


def nebular_interval_weights(derived, fractions: jnp.ndarray, *, from_grid: bool = False):
    """Interval weights of the nebular screen (youngest-interval limit with no nebular source).

    The ionizing-luminosity weights need the per-age LyC quadrature, which a
    model with no photoionized nebular source never uses: it publishes neither
    ``lyc_transmission`` nor a line catalog.  The decision is static (the keys
    are present in ``derived`` or they are not), so such a model pays nothing
    for the weights.

    Parameters
    ----------
    derived : mapping
        ``state.derived``.
    fractions : ndarray, shape (n_interval, n_age)
        :func:`interval_fractions` output.
    from_grid : bool, optional
        The nebular continuum comes from the per-Q_H grid (a nebular source).

    Returns
    -------
    ndarray, shape (n_interval,)
        :func:`ionizing_interval_weights` when a nebular source exists, else
        all weight on the youngest interval.
    """
    has_source = (
        from_grid
        or derived.get("lyc_transmission") is not None
        or derived.get("line_waves") is not None
    )
    if has_source:
        return ionizing_interval_weights(fractions, derived["log_L_lyc_age"])
    return jnp.zeros(fractions.shape[0], fractions.dtype).at[0].set(1.0)


def ionizing_interval_weights(fractions: jnp.ndarray, log_L_lyc_age: jnp.ndarray) -> jnp.ndarray:
    r"""Share of the ionizing luminosity produced in each age interval.

    .. math::

        q_j = \frac{\sum_a F_j(a)\, L_{\rm LyC}(a)}{\sum_a L_{\rm LyC}(a)}

    Parameters
    ----------
    fractions : ndarray, shape (n_interval, n_age)
        :func:`interval_fractions` output.
    log_L_lyc_age : ndarray, shape (n_age,)
        ``log10`` of the per-age ionizing luminosity [dex, erg/s]; ``-inf``
        where an age holds no ionizing light.

    Returns
    -------
    ndarray, shape (n_interval,)
        ``q_j`` in ``[0, 1]``, summing to one [dimensionless].  With no
        ionizing light anywhere the youngest interval takes all the weight
        (the young-limit screen).

    Notes
    -----
    **JIT/grad/vmap-compatible**; the peak is factored out in the log domain
    so no erg-scale linear intermediate forms (float32-safe).  Weighted by the
    published energy-integrated luminosity, ``log_L_lyc_age``.
    """
    log_l = jnp.asarray(log_L_lyc_age)
    finite = jnp.isfinite(log_l)
    peak = jnp.max(jnp.where(finite, log_l, -jnp.inf))
    has_light = jnp.isfinite(peak)
    safe_peak = jnp.where(has_light, peak, 0.0)
    w = jnp.where(finite, 10.0 ** jnp.where(finite, log_l - safe_peak, 0.0), 0.0)
    total = jnp.sum(w)
    safe_total = jnp.where(total > 0.0, total, 1.0)
    q = jnp.einsum("ja,a->j", fractions, w) / safe_total
    youngest = jnp.zeros_like(q).at[0].set(1.0)
    return jnp.where(has_light & (total > 0.0), q, youngest)


def lyc_interval_transmissions(
    t_int: jnp.ndarray,
    *,
    lyc_t: jnp.ndarray,
    mode: str,
    geometry: str,
    f_esc,
    f_obscuration,
    ionizing: jnp.ndarray,
    covered_raw: jnp.ndarray,
    hole_raw: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Per-interval observed and intrinsic factors once the gas reprocesses ionizing photons.

    The HII gas that reprocesses the Lyman continuum surrounds the youngest
    stars, so interval 0 (stars younger than the youngest window edge) is the
    population it acts on; ``mode='all'`` (FSPS/CIGALE) applies it to every
    interval.  With a #2529 escape geometry a covering fraction ``f_esc`` of
    the young population's light bypasses the birth-cloud screen through a
    hole and never meets the gas.

    Parameters
    ----------
    t_int : ndarray, shape (n_interval, n_wave)
        Dust transmission per interval (:func:`interval_transmission`).
    lyc_t : ndarray, shape (n_wave,)
        Stellar Lyman-continuum survival ``where(lambda < 912, neb_fesc, 1)``
        [dimensionless].
    mode : str
        ``'young'`` or ``'all'`` (``lyc_reprocessed_by``).
    geometry : str
        ``'screened'``, ``'birth_cloud_holes'`` or ``'clear'``
        (``lyc_escape_geometry``); a hole needs ``mode='young'``.
    f_esc : float or ndarray
        Escaping covering fraction ``neb_fesc`` [dimensionless]; read only
        for a hole geometry.
    f_obscuration : float or ndarray
        Unattenuated-sightline fraction [dimensionless].
    ionizing : ndarray, shape (n_wave,)
        1 below the Lyman edge, 0 above (:func:`tengri.components.lyc.ionizing_mask`).
    covered_raw, hole_raw : ndarray, shape (n_wave,)
        Raw transmissions (no ``f_obscuration``) of the young population's
        screened sightline and of its hole sightline.

    Returns
    -------
    observed, intrinsic : ndarray, shape (n_interval, n_wave)
        What each interval's light keeps after dust and gas, and what the
        gas alone leaves of it (the pre-screen side of the energy-balance
        difference) [dimensionless].

    Notes
    -----
    **JIT/grad/vmap-compatible**: ``mode`` and ``geometry`` are static strings.
    """
    from tengri.components.lyc import hole_young_transmission

    ones = jnp.ones_like(lyc_t)
    n_int = t_int.shape[0]
    if mode == "all":
        gate = jnp.broadcast_to(lyc_t, t_int.shape)
        return t_int * gate, gate
    if geometry == "screened":
        gate = jnp.stack([lyc_t] + [ones] * (n_int - 1))
        return t_int * gate, gate
    f_esc = jnp.asarray(f_esc)
    f_obs = jnp.asarray(f_obscuration)
    young_raw = hole_young_transmission(covered_raw * (1.0 - ionizing), hole_raw, f_esc)
    observed = t_int.at[0].set(f_obs + (1.0 - f_obs) * young_raw)
    intrinsic = jnp.stack([1.0 - (1.0 - f_esc) * ionizing] + [ones] * (n_int - 1))
    return observed, intrinsic
