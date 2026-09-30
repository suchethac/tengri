# SPDX-License-Identifier: BSD-3-Clause
"""The one Lyman-continuum module: a single edge, a single step model.

Every consumer of the hydrogen-ionizing continuum (Q_H photon-rate
integrals, the nebular fesc mask, the dust energy-balance LyC exclusion)
shares one physical edge, :data:`LYMAN_LIMIT_AA` = 911.76 Å (re-exported
from :data:`tengri.utils.physics_constants.LYMAN_LIMIT_AA`), and one model
of how a tabulated SED behaves across a grid cell that straddles it.

**The step model.** A wavelength grid cell :math:`[\\lambda_a, \\lambda_b]`
with :math:`\\lambda_a < \\lambda_\\mathrm{edge} \\le \\lambda_b` is *not*
linearly interpolated across the edge. Real stellar and nebular continua
carry a near-discontinuous drop at the Lyman limit (#537): the ionizing
side is well-approximated as constant at the last-ionizing-node value
:math:`L(\\lambda_a)` all the way to the edge, and the non-ionizing side is
constant at the first-non-ionizing-node value :math:`L(\\lambda_b)` from the
edge onward:

.. math::

    L(\\lambda) = \\begin{cases}
        L(\\lambda_a) & \\lambda_a \\le \\lambda \\le \\lambda_\\mathrm{edge} \\\\
        L(\\lambda_b) & \\lambda_\\mathrm{edge} < \\lambda \\le \\lambda_b
    \\end{cases}

Every grid cell not straddling the edge is ordinary piecewise-linear
(trapezoid) as usual. This is the #537 Q_H "rectangle" correction
(:func:`tengri.components.stellar.component._integrate_nion_log10`),
generalized to a shared primitive: a per-node boolean mask
``wave < edge_aa`` (:func:`ionizing_mask`) is *exact* for a step-model
integral or interpolant once the bracket cell is treated this way — the
exactness comes from the edge-aware quadrature/interpolation in this
module, not from moving or reweighting grid nodes.

Provides
--------
- :data:`LYMAN_LIMIT_AA` — the one edge, re-exported from
  :mod:`tengri.utils.physics_constants`.
- :func:`ionizing_mask` — the shared per-node boolean mask.
- :func:`edge_trapezoid` — step-model-exact definite integral (whole grid,
  ionizing side only, or non-ionizing side only), in either the wavelength
  or frequency quadrature variable.
- :func:`edge_interp` — step-model-exact interpolation (linear away from
  the edge, a step in the bracket cell).
- :func:`lyc_shares` — the #2436 additive ionizing-photon-budget split
  (escape / HII-dust / photoionization), moved here from
  :mod:`tengri.components.nebular._recombination_coeffs`.

Notes
-----
**JIT-compatible**: yes for all four callables — pure ``jnp`` arithmetic
with no Python branching on traced values. ``variable``, ``side``, and
``axis`` in :func:`edge_trapezoid` are static (non-traced) Python
arguments; pass them via ``functools.partial`` or list them in
``static_argnames`` when wrapping in :func:`jax.jit`.

**Gradient-safe**: yes, with respect to the sampled values ``y`` (or
``sed_lnu`` in the Q_H case) — the bracket-cell weights depend only on
``wave`` and ``edge_aa``, which are ordinarily static, so the map from
``y`` to the integral/interpolant is linear and differentiates cleanly.
"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.utils.physics_constants import C_AA, LYMAN_LIMIT_AA

__all__ = [
    "LYMAN_LIMIT_AA",
    "edge_interp",
    "edge_trapezoid",
    "ionizing_mask",
    "lyc_shares",
]


def ionizing_mask(wave: jnp.ndarray, edge_aa: float = LYMAN_LIMIT_AA) -> jnp.ndarray:
    r"""Per-node hydrogen-ionizing mask, exact under the step model.

    Parameters
    ----------
    wave : array_like, shape (n_wave,)
        Wavelength grid, ascending. [Angstrom]
    edge_aa : float, optional
        Lyman edge [Angstrom]. Default :data:`LYMAN_LIMIT_AA` (911.76 Å).

    Returns
    -------
    ndarray, shape (n_wave,)
        Boolean mask, ``wave < edge_aa``.

    Notes
    -----
    **JIT-compatible**: yes.

    A node exactly at the edge (``wave == edge_aa``) is classified
    non-ionizing (the mask is strict ``<``), matching the step model's
    convention :math:`\lambda_a < \lambda_\mathrm{edge} \le \lambda_b`: the
    edge itself belongs to the non-ionizing (``b``) side of its bracket
    cell. This mask is exact — not an approximation to be corrected — only
    because :func:`edge_trapezoid` and :func:`edge_interp` handle the
    bracket cell with the step model rather than a plain trapezoid/linear
    rule; a naive consumer that masks with this array and then integrates
    or interpolates the result with an ordinary rule re-introduces the
    #537 partial-bin error this module exists to remove.
    """
    return jnp.asarray(wave) < edge_aa


def _quadrature_coordinate(wave: jnp.ndarray, variable: str) -> jnp.ndarray:
    """One-sentence: map a wavelength grid to the requested quadrature axis."""
    if variable == "nu":
        return C_AA / wave
    if variable == "wave":
        return wave
    raise ValueError(f"variable must be 'nu' or 'wave', got {variable!r}")


def edge_trapezoid(
    y: jnp.ndarray,
    wave: jnp.ndarray,
    *,
    variable: str = "nu",
    side: str = "all",
    edge_aa: float = LYMAN_LIMIT_AA,
    axis: int = -1,
) -> jnp.ndarray:
    r"""Step-model-exact definite integral of a tabulated SED across the Lyman edge.

    Integrates a function :math:`y(\lambda)` sampled on ``wave``, treating
    the grid cell straddling ``edge_aa`` with the step model (module
    docstring) instead of linear interpolation: every other cell is the
    ordinary trapezoid rule.

    Parameters
    ----------
    y : array_like, shape (..., n_wave)
        Sampled function values along ``axis``. Any physical quantity
        (e.g. :math:`L_\nu/\nu` for a Q_H integrand); units are the
        caller's.
    wave : array_like, shape (n_wave,)
        Wavelength grid, ascending. [Angstrom]
    variable : {"nu", "wave"}, optional
        Quadrature variable. ``"wave"`` integrates in :math:`\lambda`
        directly; ``"nu"`` integrates in frequency,
        :math:`\nu = c/\lambda` (:data:`tengri.utils.physics_constants.C_AA`
        for :math:`c` in Å/s). Because ``wave`` is ascending, :math:`\nu` is
        descending; every cell width is taken as :math:`|\Delta x|`
        (sign-normalized to a positive measure) rather than reversing the
        node order, so the returned integral is non-negative whenever
        ``y`` is non-negative on the selected side. Default ``"nu"``.
    side : {"all", "ionizing", "nonionizing"}, optional
        Which side of the edge to integrate. ``"ionizing"`` is
        :math:`\lambda < \mathtt{edge\_aa}`; ``"nonionizing"`` is
        :math:`\lambda \ge \mathtt{edge\_aa}`; ``"all"`` is their sum
        (exactly, to round-off — see Notes). Default ``"all"``.
    edge_aa : float, optional
        Lyman edge [Angstrom]. Default :data:`LYMAN_LIMIT_AA` (911.76 Å).
    axis : int, optional
        Axis of ``y`` along which ``wave`` runs. Default -1.

    Returns
    -------
    ndarray, shape (...)
        The integral, with ``axis`` reduced away.

    Notes
    -----
    **JIT-compatible**: yes — pure ``jnp`` arithmetic, static-shape
    throughout (only index *values*, never array *shapes*, depend on
    ``wave``/``edge_aa``). ``variable``, ``side``, and ``axis`` are static
    Python arguments, not traced values.

    **Gradient-safe**: yes with respect to ``y`` — linear in ``y`` for
    fixed ``wave``/``edge_aa``, so the gradient is finite and exact
    everywhere ``wave`` has no repeated nodes.

    Away from the bracket cell (the one cell, if any, with
    :math:`\lambda_a < \mathtt{edge\_aa} \le \lambda_b`), every panel is
    the ordinary trapezoid, :math:`\tfrac{1}{2}(y_i + y_{i+1})|x_{i+1} -
    x_i|`. The bracket cell contributes
    :math:`y_a\,|x_\mathrm{edge} - x_a|` to the ionizing side and
    :math:`y_b\,|x_b - x_\mathrm{edge}|` to the non-ionizing side (the step
    model's two rectangles), where :math:`x = \lambda` or :math:`c/\lambda`
    per ``variable``. Because ``"ionizing"`` and ``"nonionizing"`` partition
    every panel and every bracket-cell rectangle with no overlap and no
    gap, ``edge_trapezoid(..., side="ionizing") + edge_trapezoid(...,
    side="nonionizing") == edge_trapezoid(..., side="all")`` holds by
    construction, to floating-point round-off.

    **Edge outside the grid**: if ``edge_aa <= wave[0]`` the whole grid is
    non-ionizing (no bracket cell exists; ``side="ionizing"`` returns
    exactly 0); if ``edge_aa > wave[-1]`` the whole grid is ionizing
    (``side="nonionizing"`` returns exactly 0). Both are handled by the
    same panel classification, not a special-cased branch.

    **Reference**: generalizes the #537 partial-bin Lyman correction
    (originally hand-derived once for :math:`Q_H` in
    :mod:`tengri.components.stellar.component`) into a shared primitive so
    every LyC-adjacent quantity uses the same edge-aware quadrature.
    """
    y = jnp.asarray(y)
    wave = jnp.asarray(wave)
    if side not in ("all", "ionizing", "nonionizing"):
        raise ValueError(f"side must be 'all', 'ionizing', or 'nonionizing', got {side!r}")

    y = jnp.moveaxis(y, axis, -1)

    x = _quadrature_coordinate(wave, variable)
    x_edge = _quadrature_coordinate(jnp.asarray(edge_aa), variable)

    y_lo = y[..., :-1]
    y_hi = y[..., 1:]
    dx = jnp.abs(x[1:] - x[:-1])
    panel_area = 0.5 * (y_lo + y_hi) * dx

    wave_lo = wave[:-1]
    wave_hi = wave[1:]
    fully_ionizing = wave_hi < edge_aa
    fully_nonionizing = wave_lo >= edge_aa
    is_bracket = ~fully_ionizing & ~fully_nonionizing

    ordinary_ionizing = jnp.where(fully_ionizing, panel_area, 0.0)
    ordinary_nonionizing = jnp.where(fully_nonionizing, panel_area, 0.0)

    x_lo = x[:-1]
    x_hi = x[1:]
    bracket_ionizing = jnp.where(is_bracket, y_lo * jnp.abs(x_edge - x_lo), 0.0)
    bracket_nonionizing = jnp.where(is_bracket, y_hi * jnp.abs(x_hi - x_edge), 0.0)

    ionizing_total = jnp.sum(ordinary_ionizing + bracket_ionizing, axis=-1)
    nonionizing_total = jnp.sum(ordinary_nonionizing + bracket_nonionizing, axis=-1)

    if side == "ionizing":
        return ionizing_total
    if side == "nonionizing":
        return nonionizing_total
    return ionizing_total + nonionizing_total


def edge_interp(
    x_new: jnp.ndarray,
    wave: jnp.ndarray,
    y: jnp.ndarray,
    *,
    edge_aa: float = LYMAN_LIMIT_AA,
) -> jnp.ndarray:
    r"""Step-model-exact interpolation of a tabulated SED across the Lyman edge.

    Ordinary linear interpolation (:func:`jnp.interp` semantics, zero
    outside the grid) everywhere except the one grid cell straddling
    ``edge_aa``, where the step model (module docstring) applies: the
    interpolated value is the last-ionizing-node value below the edge and
    the first-non-ionizing-node value at or above it, never a linear ramp
    between them.

    Parameters
    ----------
    x_new : array_like
        Query points. [Angstrom]
    wave : array_like, shape (n_wave,)
        Wavelength grid, ascending. [Angstrom]
    y : array_like, shape (n_wave,)
        Sampled function values at ``wave``.
    edge_aa : float, optional
        Lyman edge [Angstrom]. Default :data:`LYMAN_LIMIT_AA` (911.76 Å).

    Returns
    -------
    ndarray, shape of ``x_new``
        Interpolated values: ``0`` outside ``[wave[0], wave[-1]]`` (matching
        :func:`jnp.interp`'s ``left=0, right=0``), the step-model value in
        the bracket cell, ordinary linear interpolation elsewhere.

    Notes
    -----
    **JIT-compatible**: yes — pure ``jnp`` arithmetic and gather-by-index
    (dynamic index *values*, static shapes).

    **Gradient-safe**: yes with respect to ``y`` — linear in ``y`` for
    fixed ``wave``/``edge_aa``/``x_new``.

    In the bracket cell :math:`[\lambda_a, \lambda_b]`
    (:math:`\lambda_a < \mathtt{edge\_aa} \le \lambda_b`):

    .. math::

        \hat y(\lambda) = \begin{cases}
            y_a & \lambda < \mathtt{edge\_aa} \\
            y_b & \lambda \ge \mathtt{edge\_aa}
        \end{cases}

    If ``edge_aa`` is outside the grid range (no bracket cell exists), this
    reduces exactly to ordinary linear interpolation everywhere.
    """
    x_new = jnp.asarray(x_new)
    wave = jnp.asarray(wave)
    y = jnp.asarray(y)
    n_wave = wave.shape[0]

    linear = jnp.interp(x_new, wave, y, left=0.0, right=0.0)

    n_ion = jnp.sum(ionizing_mask(wave, edge_aa).astype(jnp.int32))
    has_bracket = (n_ion > 0) & (n_ion < n_wave)
    idx_a = jnp.clip(n_ion - 1, 0, n_wave - 1)
    idx_b = jnp.clip(n_ion, 0, n_wave - 1)
    wave_a = wave[idx_a]
    wave_b = wave[idx_b]
    y_a = y[idx_a]
    y_b = y[idx_b]

    in_bracket = has_bracket & (x_new >= wave_a) & (x_new <= wave_b)
    step_value = jnp.where(x_new < edge_aa, y_a, y_b)
    return jnp.where(in_bracket, step_value, linear)


def lyc_shares(
    neb_fesc: jnp.ndarray | float, neb_fdust_frac: jnp.ndarray | float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""Split the ionizing-photon budget into its three additive shares (#2436).

    Owner ruling (#2436): ``neb_fesc + f_dust <= 1`` is a precondition of the
    additive per-photon budget this module's own docstring derives (``f_esc``
    escapes, ``f_dust`` heats HII-region dust, ``1 - f_esc - f_dust``
    photoionizes -- see
    :func:`tengri.components.nebular._recombination_coeffs.lyc_dust_escape_factor`'s
    "Per-photon LyC budget" section). Declaring ``f_dust`` as its own
    independent ``Uniform(0, 1)`` parameter (the retired ``neb_fdust``) let a
    caller pick ``neb_fesc=0.7, neb_fdust=0.7``, an impossible 1.4 of the
    budget, with nothing to catch it before it reached
    :func:`lyc_dust_escape_factor`'s own silent ``jnp.clip``.

    ``neb_fdust_frac`` instead parametrizes the fraction of the
    NON-escaping budget (``1 - neb_fesc``) that HII-region dust absorbs, so
    the three shares

    .. math::

        f_\mathrm{esc} &= \mathtt{neb\_fesc} \\
        f_\mathrm{dust} &= \mathtt{neb\_fdust\_frac} \, (1 - \mathtt{neb\_fesc}) \\
        f_\mathrm{gas} &= (1 - \mathtt{neb\_fdust\_frac})(1 - \mathtt{neb\_fesc})

    sum to exactly 1 for ANY ``(neb_fesc, neb_fdust_frac) \in [0, 1]^2`` --
    the whole prior box is physical, with no clamp needed downstream. This is
    the ONE place that splits the budget; every consumer that used to read
    the retired absolute ``neb_fdust`` (:func:`lyc_dust_escape_factor`'s
    ``f_dust`` callers in ``cue.py``/``cloudy_grid.py``/``cloudy_cb19.py``,
    the #2539 HII-dust LyC credit in ``components/nebular/component.py``) now
    calls this function first and reads the absolute ``f_dust``/``f_gas`` it
    returns.

    Parameters
    ----------
    neb_fesc : array_like or float
        Ionizing photon escape fraction [dimensionless, in [0, 1]].
    neb_fdust_frac : array_like or float
        Fraction of the NON-escaping ionizing budget absorbed by dust inside
        the HII region [dimensionless, in [0, 1]].

    Returns
    -------
    f_esc, f_dust, f_gas : tuple of ndarray
        The three additive shares of the ionizing-photon budget
        (escape, HII-region dust absorption, photoionization), summing to 1
        to round-off for any input in ``[0, 1]^2``.

    Notes
    -----
    **JIT/grad-safe**: pure ``jnp`` arithmetic, no branching; the gradient
    wrt either input is finite and nonzero everywhere on the open box.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> f_esc, f_dust, f_gas = lyc_shares(0.0, 0.0)
    >>> float(f_esc), float(f_dust), float(f_gas)
    (0.0, 0.0, 1.0)
    >>> f_esc, f_dust, f_gas = lyc_shares(0.3, 0.5)
    >>> round(float(f_dust), 4), round(float(f_gas), 4)
    (0.35, 0.35)
    >>> float(f_esc + f_dust + f_gas)
    1.0
    """
    f_esc = jnp.asarray(neb_fesc)
    frac = jnp.asarray(neb_fdust_frac)
    non_escaping = 1.0 - f_esc
    f_dust = frac * non_escaping
    f_gas = (1.0 - frac) * non_escaping
    return f_esc, f_dust, f_gas
