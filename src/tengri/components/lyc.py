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
(``tengri.components.stellar.component._integrate_nion_log10``),
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
- :func:`edge_bracket_values` — the bracket cell's two step values
  (``y_a``, ``y_b``), for callers building their OWN exact quadrature on a
  grid finer than ``wave`` (e.g. a photometric filter's own nodes).
- :func:`lyc_shares` — the #2436 additive ionizing-photon-budget split
  (escape / HII-dust / photoionization), moved here from
  ``tengri.components.nebular._recombination_coeffs``.

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

import jax
import jax.numpy as jnp

from tengri.utils.physics_constants import C_AA, LYMAN_LIMIT_AA
from tengri.utils.scale import _not_computable, log10_magnitude

__all__ = [
    "LYC_ESCAPE_GEOMETRIES",
    "LYMAN_LIMIT_AA",
    "credited_log10_lyc",
    "edge_bracket_values",
    "edge_interp",
    "edge_trapezoid",
    "escape_geometry_transmission",
    "hole_young_transmission",
    "ionizing_mask",
    "log10_age_sum_lyc",
    "log10_lyc_luminosity",
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


def edge_bracket_values(
    wave: jnp.ndarray,
    y: jnp.ndarray,
    edge_aa: float = LYMAN_LIMIT_AA,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""The two step-model endpoint values of the bracket cell at ``edge_aa``.

    Returns ``(y_a, y_b)``: the value :func:`edge_interp` would give for a
    query point infinitesimally below ``edge_aa`` and one exactly at
    ``edge_aa``, respectively. Delegating to :func:`edge_interp` (via
    :func:`jnp.nextafter` for the "infinitesimally below" query) rather than
    re-deriving the bracket search keeps this consistent, by construction,
    with every degenerate case :func:`edge_interp` already handles (``0``
    when ``edge_aa`` is outside ``wave``'s domain on the queried side, the
    ordinary linear value when there is no bracket cell at all).

    Used to build an exact quadrature across the edge on a grid FINER than
    ``wave`` (e.g. a photometric filter's own nodes subdividing ``wave``'s
    bracket cell, #2447): inserting ``edge_aa`` into that finer grid as a
    zero-width node pair tagged with ``(y_a, y_b)`` lets an ORDINARY
    (non-edge-aware) trapezoid integrate the step exactly, because a
    smooth, independently-varying multiplicative factor (a filter's
    transmission x bandpass weight) evaluated at the edge is then correctly
    paired with each side's own constant value, rather than averaged across
    the whole bracket width as :func:`edge_trapezoid` applied to an
    already-multiplied integrand would do.

    Parameters
    ----------
    wave : array_like, shape (n_wave,)
        Wavelength grid, ascending. [Angstrom]
    y : array_like, shape (n_wave,)
        Sampled function values at ``wave``.
    edge_aa : float, optional
        Lyman edge [Angstrom]. Default :data:`LYMAN_LIMIT_AA` (911.76 Å).

    Returns
    -------
    y_a, y_b : ndarray
        The ionizing-side and non-ionizing-side step values (scalars, or
        matching ``y``'s leading shape if ``y`` carries one).

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes, linear in ``y``.
    """
    edge_aa = jnp.asarray(edge_aa)
    # ``nextafter`` has no differentiation rule; the exact offset is a
    # non-differentiable implementation detail anyway (``y_a`` is piecewise
    # constant in ``edge_aa``, like every other discrete bracket-index
    # lookup in this module), so the query point it builds is detached from
    # the gradient tape while ``edge_aa`` itself (used for the comparisons
    # inside :func:`edge_interp`) is not.
    just_below = jnp.nextafter(jax.lax.stop_gradient(edge_aa), -jnp.inf)
    y_a = edge_interp(just_below, wave, y, edge_aa=edge_aa)
    y_b = edge_interp(edge_aa, wave, y, edge_aa=edge_aa)
    return y_a, y_b


def log10_lyc_luminosity(
    lnu: jnp.ndarray,
    wave: jnp.ndarray,
    *,
    log10_scale: jnp.ndarray | float = 0.0,
    edge_aa: float = LYMAN_LIMIT_AA,
    axis: int = -1,
) -> jnp.ndarray:
    r"""log10 of the Lyman-continuum LUMINOSITY, peak-factored for float32 safety.

    THE single source of the LyC luminosity integral:

    .. math::

        L_{\rm LyC} = \int_{\lambda < \lambda_\mathrm{edge}} L_\nu(\lambda)\,d\nu

    via :func:`edge_trapezoid` (``variable="nu"``, ``side="ionizing"``), which
    applies the step model at the Lyman edge (module docstring) instead of a
    hard mask. This is a LUMINOSITY (erg/s), not a photon RATE -- it must not
    be confused with the Q_H integrand
    :math:`\int L_\nu/(h\nu)\,d\nu` (photons/s) that
    ``tengri.components.stellar.component._integrate_nion_log10`` computes;
    the two integrands differ by a factor of :math:`h\nu` under the integral
    and are not interchangeable (#2539 G1/G2).

    The computation normalizes ``lnu`` by its peak (|axis|-wise, so each
    leading-axis row -- e.g. each SSP age -- is normalized independently),
    integrates in linear-normalized space, and restores the peak and
    ``log10_scale`` as log10 OFFSETS, keeping every intermediate within
    float32 range (mirrors ``tengri.components.stellar.component.
    _integrate_nion_log10``, #1206).

    Parameters
    ----------
    lnu : array_like, shape (..., n_wave)
        Rest-frame :math:`L_\nu` [erg/s/Hz], sampled on ``wave`` along
        ``axis``. Any leading (batch) shape, e.g. ``(n_age, n_wave)``.
    wave : array_like, shape (n_wave,)
        Wavelength grid, ascending [Angstrom]; must span the Lyman limit (a
        few points above 911.76 A suffice -- the boundary bin needs the
        first non-ionizing point).
    log10_scale : array_like or float, optional
        Log10-scale offset [dex] added to the result, broadcast against the
        ``axis``-reduced shape. Default 0.0 (``lnu`` is already on an
        absolute scale). Used for mass-scaling a per-Msun cube:
        ``log10_scale = log10(total_mass) + log10(L_sun / (erg/s))``, so the
        ``total_mass x L_sun`` linear product (float32-unsafe, #1206) is
        never materialized.
    edge_aa : float, optional
        Lyman edge [Angstrom]. Default :data:`LYMAN_LIMIT_AA` (911.76 Å).
    axis : int, optional
        Axis of ``lnu`` along which ``wave`` runs. Default -1. Every other
        axis is a batch axis: no ``vmap`` is needed, ``edge_trapezoid``
        already broadcasts over leading axes.

    Returns
    -------
    ndarray, shape (...)
        :math:`\log_{10}(L_{\rm LyC} / (\mathrm{erg/s}))` [dex], with
        ``axis`` reduced away. ``-inf`` when the slice has no ionizing flux
        (powers back to exactly 0.0 via :func:`tengri.utils.scale.pow10`).
        ``+inf`` when the input is corrupt (non-finite); see
        :func:`tengri.utils.scale.log10_magnitude`.

    Notes
    -----
    **JIT-compatible**: yes -- pure ``jnp`` arithmetic; ``log10_scale``,
    ``edge_aa``, and ``axis`` may be static or traced (``axis`` must be
    static for ``jax.jit``/``vmap``).

    **Gradient-safe**: yes, with respect to ``lnu`` -- the peak factor is
    detached (``jax.lax.stop_gradient``) and cancels analytically against
    the log10(peak) added back, so the gradient is that of the unfactored
    integral; linear in ``lnu`` for fixed ``wave``/``edge_aa``.

    :func:`tengri.forward.energy_balance.bolometric_lyc_log10` is a thin
    wrapper around this function (``axis=-1``, no mass scaling) that also
    reports the integral's sign, for callers combining it with another
    signed term via :func:`tengri.utils.scale.log10_add`.
    """
    lnu = jnp.asarray(lnu)
    wave = jnp.asarray(wave)
    # stop_gradient: pure factorization constant (mirrors _integrate_nion_log10,
    # #1436); log10(peak) is added back below, so the peak cancels analytically.
    peak = jax.lax.stop_gradient(jnp.max(jnp.abs(lnu), axis=axis, keepdims=True, initial=0.0))
    peak = jnp.where(peak > 0, peak, jnp.ones_like(peak))
    ell = lnu / peak  # O(1) normalized L_nu
    norm = edge_trapezoid(ell, wave, variable="nu", side="ionizing", edge_aa=edge_aa, axis=axis)
    peak_reduced = jnp.squeeze(peak, axis=axis)
    # log10_magnitude keeps "no ionizing flux" (-inf) apart from "the input
    # was corrupt" (+inf) -- see that function's docstring and #1527.
    log10_norm = log10_magnitude(norm)
    offsets = jnp.log10(peak_reduced) + log10_scale
    # -inf + finite is -inf (true zero) and +inf + finite is +inf (corrupt), so
    # both sentinels survive the offset addition unchanged; only a +inf peak
    # (lnu itself non-finite) could turn one into NaN, and that is itself corrupt.
    return jnp.where(_not_computable(log10_norm), jnp.inf, log10_norm + offsets)


def lyc_shares(
    neb_fesc: jnp.ndarray | float, neb_fdust_frac: jnp.ndarray | float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""Split the ionizing-photon budget into its three additive shares (#2436).

    Owner ruling (#2436): ``neb_fesc + f_dust <= 1`` is a precondition of the
    additive per-photon budget this module's own docstring derives (``f_esc``
    escapes, ``f_dust`` heats HII-region dust, ``1 - f_esc - f_dust``
    photoionizes -- see
    ``tengri.components.nebular._recombination_coeffs.lyc_dust_escape_factor``'s
    "Per-photon LyC budget" section). Declaring ``f_dust`` as its own
    independent ``Uniform(0, 1)`` parameter (the retired ``neb_fdust``) let a
    caller pick ``neb_fesc=0.7, neb_fdust=0.7``, an impossible 1.4 of the
    budget, with nothing to catch it before it reached
    ``lyc_dust_escape_factor``'s own silent ``jnp.clip``.

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
    the retired absolute ``neb_fdust`` (``lyc_dust_escape_factor``'s
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


#: Allowed values of the ``dust_attenuation['lyc_escape_geometry']`` structural
#: key (#2529): whether the escaping fraction ``f_esc`` (read off
#: :func:`lyc_shares`'s first return, ``neb_fesc``) bypasses the birth-cloud
#: dust screen through a geometric hole, and if so what the hole itself still
#: crosses. ``'screened'`` (default) is the pre-#2529 behavior, bit-identical:
#: ``f_esc`` never touches the dust screen, only the nebular-reprocessing
#: budget. See :func:`escape_geometry_transmission`.
LYC_ESCAPE_GEOMETRIES: tuple[str, ...] = ("screened", "birth_cloud_holes", "clear")


def hole_young_transmission(
    T_covered: jnp.ndarray, T_hole: jnp.ndarray, f_esc: jnp.ndarray | float
) -> jnp.ndarray:
    r"""The young population's transmission with a covering-fraction hole (#2529).

    .. math::

        T_\mathrm{young} = (1 - f_\mathrm{esc})\,T_\mathrm{covered}
            + f_\mathrm{esc}\,T_\mathrm{hole}

    The one place the hole formula is written: the exact path, the energy
    balance table and the photometry table all build it through this.

    Parameters
    ----------
    T_covered : array_like
        Raw transmission of the screened sightline [dimensionless, in [0, 1]].
    T_hole : array_like
        Raw transmission of the hole sightline [dimensionless, in [0, 1]].
    f_esc : array_like or float
        Escaping covering fraction ``neb_fesc`` [dimensionless, in [0, 1]].

    Returns
    -------
    ndarray
        Young-population transmission, broadcast shape of the inputs.

    Notes
    -----
    **JIT/grad/vmap-compatible**: pure arithmetic.
    """
    f_esc = jnp.asarray(f_esc)
    return (1.0 - f_esc) * jnp.asarray(T_covered) + f_esc * jnp.asarray(T_hole)


def escape_geometry_transmission(
    y_age: jnp.ndarray,
    T_bc: jnp.ndarray,
    T_diff: jnp.ndarray,
    f_esc: jnp.ndarray | float,
    geometry: str,
) -> jnp.ndarray:
    r"""Population-mixture dust transmission with a #2529 LyC escape geometry.

    Every SSP node holds a mixture of two stellar populations: the fraction
    :math:`y(a)` of its formed mass younger than the birth-cloud lifetime
    (:mod:`tengri.components.stellar.age_boundary`), seen through the birth
    cloud *and* the diffuse ISM, and the rest, seen through the diffuse ISM
    only.  Without a hole the node's transmission is the mixture

    .. math::

        T(\lambda, a) = y(a)\,T_\mathrm{bc}(\lambda)\,T_\mathrm{diff}(\lambda)
            + [1 - y(a)]\,T_\mathrm{diff}(\lambda).

    #2529 observed that a hole in a birth cloud is a covering-fraction split
    of the *young population* into two sub-beams with DIFFERENT screens: a
    covering fraction ``f_esc`` of the young population's light (at every
    wavelength, continuum and Lyman continuum alike -- "a hole in a birth
    cloud is geometric, not wavelength selective") bypasses the birth-cloud
    screen through a hole that still crosses :math:`T_\mathrm{hole}(\lambda)`;
    the complementary :math:`1 - f_\mathrm{esc}` stays on the screened
    sightline:

    .. math::

        T(\lambda, a) = y(a)\bigl[(1 - f_\mathrm{esc})\,T_\mathrm{bc}\,T_\mathrm{diff}
            + f_\mathrm{esc}\,T_\mathrm{hole}\bigr] + [1 - y(a)]\,T_\mathrm{diff}

    with :math:`T_\mathrm{hole} = T_\mathrm{diff}` for ``'birth_cloud_holes'``
    (FSPS ``frac_obrun``-like: the hole still crosses the general diffuse
    ISM) or :math:`T_\mathrm{hole} = 1` for ``'clear'`` (Synthesizer
    ``fesc``-like: a fully clear sightline, no dust at all). The old
    population (:math:`y(a) \to 0`) reduces to :math:`T_\mathrm{diff}`
    identically regardless of ``f_esc``/``geometry`` -- old stars have no
    birth cloud for a hole to be in.

    ``T_bc``/``T_diff`` are the RAW, un-weighted per-screen transmissions
    (``exp(-tau_v1 k_bc)`` / ``exp(-tau_v2 k_diff)``, no ``f_obscuration``
    floor): the caller applies the shared ``f_obscuration`` affine wrap
    (``f_obs + (1 - f_obs) * (...)``) to this function's OUTPUT once, so a
    nonzero obscuration floor is not double-applied to the escaping sub-beam
    specifically.

    This function returns the DUST-SCREEN-ONLY transmission (what the
    energy-balance integral attenuates). It does not include the separate
    Lyman-continuum nebular-reprocessing gate (``neb_fesc``'s OTHER role,
    already handled by :func:`lyc_shares`): the covered (``1 - f_esc``)
    sub-beam's ionizing photons still reach the HII gas and are reprocessed
    there, so a caller building the OBSERVED stellar continuum passes
    ``T_bc * (1 - ionizing_mask(wave))`` as ``T_bc`` (the gate lives on the
    covered young sub-beam only; see ``DustSEDComponent.apply`` §2a).

    Parameters
    ----------
    y_age : array_like
        Birth-cloud weight, :math:`y(a) \in [0, 1]` (1 = fully young, 0 =
        fully old). Broadcasts against ``T_bc``/``T_diff``.
    T_bc : array_like
        Raw birth-cloud-only transmission, :math:`\exp(-\tau_\mathrm{bc}
        k_\mathrm{bc}(\lambda))` [dimensionless, in [0, 1]]. No
        ``f_obscuration``.
    T_diff : array_like
        Raw diffuse-ISM-only transmission, :math:`\exp(-\tau_\mathrm{diff}
        k_\mathrm{diff}(\lambda))` [dimensionless, in [0, 1]]. No
        ``f_obscuration``.
    f_esc : array_like or float
        Escaping covering fraction of the young population
        [dimensionless, in [0, 1]] -- :func:`lyc_shares`'s ``f_esc``
        (``neb_fesc``). May be a traced, runtime (fit) value.
    geometry : str
        One of :data:`LYC_ESCAPE_GEOMETRIES` other than ``'screened'``:
        ``'birth_cloud_holes'`` or ``'clear'``. Static Python string, never
        traced (resolved once at build time); ``'screened'`` is a caller
        error here -- the caller should skip this function entirely for
        ``'screened'`` and reuse its plain mixture call instead.

    Returns
    -------
    ndarray
        Dust-screen-only transmission, same broadcast shape as
        ``y_age * T_bc * T_diff`` [dimensionless, in [0, 1]].

    Raises
    ------
    ValueError
        If ``geometry`` is not ``'birth_cloud_holes'`` or ``'clear'``.

    Notes
    -----
    **JIT-compatible**: yes; ``geometry`` must be a static (non-traced)
    Python string.

    **Gradient-safe**: yes, with respect to ``y_age``, ``T_bc``, ``T_diff``,
    ``f_esc`` -- pure ``jnp`` arithmetic, no branching on traced values.

    Identities (owner ruling #2529, pinned by
    ``tests/regression/bug/test_2529_lyc_escape_geometry.py``):

    - ``f_esc = 0``: :math:`T = y(a)\,T_\mathrm{bc}\,T_\mathrm{diff} +
      [1 - y(a)]\,T_\mathrm{diff}` regardless of ``geometry`` --
      construction-exact (the ``T_hole`` term is multiplied by
      ``f_esc = 0``), so ``'birth_cloud_holes'`` and ``'clear'`` agree with
      each other AND with the plain ``'screened'`` mixture.
    - ``f_esc = 1``, ``'clear'``: :math:`T = y(a) + (1 - y(a))\,
      T_\mathrm{diff}(\lambda)`; at :math:`y(a) = 1`, :math:`T = 1`
      (unattenuated).
    - ``f_esc = 1``, ``'birth_cloud_holes'``: :math:`T = T_\mathrm{diff}
      (\lambda)` for ANY :math:`y(a)` (``T_hole = T_diff`` makes both
      populations see the diffuse screen), matching FSPS ``frac_obrun``: an
      OB star outside its birth cloud sees only the diffuse screen.
    - ``y(a) = 0``: :math:`T = T_\mathrm{diff}(\lambda)` to machine
      precision for any ``f_esc``/``geometry`` -- old stars have no birth
      cloud.

    References
    ----------
    FSPS ``frac_obrun`` (Conroy, Gunn & White 2009; Conroy & Gunn 2010),
    ``src/add_dust.f90`` (``ADD_DUST``, installed FSPS Fortran source)::

        cspi = csp1 * EXP(-pset%dust1*(spec_lambda/5500.)**(pset%dust1_index))*&
             (1-pset%frac_obrun) + csp1*pset%frac_obrun + &
             csp2 * EXP(-pset%dust3*tau_diff)
        ...
        specdust  = (1-pset%frac_nodust) * cspi*diff_dust + cspi*pset%frac_nodust

    (``csp1`` the young/birth-cloud spectrum, ``dust1`` its birth-cloud
    optical depth, ``diff_dust`` the diffuse-ISM screen applied
    UNCONDITIONALLY afterward to the combined ``cspi``): a ``frac_obrun``
    covering fraction of the young population bypasses the birth-cloud
    term specifically (added back in unattenuated by it, ``csp1*frac_obrun``)
    while still being multiplied by the diffuse screen two lines later --
    exactly the ``'birth_cloud_holes'`` geometry here (``T_hole = T_diff``).

    Synthesizer's ``fesc`` (installed Python source),
    ``emission_models/stellar/models.py``'s ``TransmittedEmissionWithEscaped``
    builds the escaped term as
    ``StellarEmissionModel(apply_to=incident, transformer=EscapedFraction(),
    fesc=fesc)``, and ``emission_models/transformers/escape_fraction.py``'s
    ``EscapedFraction._transform`` reads ``return emission.scale(fesc, ...)``
    applied to the UNDUSTED, UNREPROCESSED ``incident`` spectrum: the
    escaped fraction of the young population sees neither dust nor nebular
    reprocessing at all -- exactly the ``'clear'`` geometry here
    (``T_hole = 1``).

    tengri implements the same two reference models (credited, not ported)
    as the ``'birth_cloud_holes'`` / ``'clear'`` geometries respectively,
    unified with its own young/old population mixture and the existing
    ``neb_fesc``/:func:`lyc_shares` budget.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> y_age = jnp.array([1.0, 0.0])
    >>> T_bc = jnp.array([0.5, 0.5])
    >>> T_diff = jnp.array([0.8, 0.8])
    >>> T = escape_geometry_transmission(y_age, T_bc, T_diff, 1.0, "birth_cloud_holes")
    >>> [round(float(v), 6) for v in T]
    [0.8, 0.8]
    """
    if geometry not in ("birth_cloud_holes", "clear"):
        raise ValueError(
            f"escape_geometry_transmission: geometry must be one of "
            f"('birth_cloud_holes', 'clear') -- 'screened' is handled by the "
            f"caller's plain mixture call, never routed through this "
            f"function; got {geometry!r}."
        )
    y_age = jnp.asarray(y_age)
    T_bc = jnp.asarray(T_bc)
    T_diff = jnp.asarray(T_diff)
    f_esc = jnp.asarray(f_esc)
    T_hole = T_diff if geometry == "birth_cloud_holes" else jnp.ones_like(T_diff)
    t_young = hole_young_transmission(T_bc * T_diff, T_hole, f_esc)
    return y_age * t_young + (1.0 - y_age) * T_diff


def log10_age_sum_lyc(log_L_lyc_age, weights=None):
    r"""log10 of the age-summed Lyman-continuum luminosity.

    Combines per-age ionizing luminosities in log-space via a weighted sum:

    .. math::

        L_{\rm LyC,total} = \sum_{\rm age} w_{\rm age} \times 10^{\log L_{\rm LyC,age}}

    returning :math:`\log_{10} L_{\rm LyC,total}` safely without overflow/underflow.

    Parameters
    ----------
    log_L_lyc_age : array_like, shape (n_age,)
        Per-age Lyman-continuum luminosity in log10 [erg/s]. Ordinarily from
        :func:`edge_trapezoid` on a per-age ionizing SED slice, matching the
        denominator of :meth:`integral method's single-sourced Q_H pathway
        (#1206, #537).
    weights : array_like, shape (n_age,), optional
        Per-age weights (typically mass fractions or escape-fraction indicators).
        Default (None): uniform weights (implicit ``1.0`` per age); the sum is
        equivalent to log10(sum(10^log_L_lyc_age)).

    Returns
    -------
    ndarray, shape ()
        log10 of the weighted sum [erg/s]. ``-inf`` if every (weighted) entry
        has no ionizing luminosity, following the sentinel contract of
        :func:`edge_trapezoid` and :func:`bolometric_lyc_log10`. ``+inf`` if
        any entry of ``log_L_lyc_age`` is corrupt (non-finite), even one with
        zero weight -- a corrupt age can never read back as "contributed
        nothing" (the #1527 failure class :func:`log10_magnitude` exists to
        avoid).

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes, including at a weight of
    exactly 0 and when every entry is ``-inf`` -- the *double-where* idiom
    (twice over, see :func:`tengri.forward.energy_balance.
    log10_fdust_lyc_credit`): the exponent and the outer ``log10`` argument
    are each clamped to a finite dummy before the transcendental, with the
    ``-inf``/``+inf`` sentinel restored afterwards by an independent
    ``jnp.where``, so a zero weight still carries the correct, nonzero
    gradient onto the other, non-zero-weighted ages (``d/dw[w * c] = c``
    at ``w = 0``, not the 0 a single clamp-and-select would give).

    Used by :class:`tengri.components.nebular.component.NebularSEDComponent`
    to sum over all ages (#2539) and by
    :class:`tengri.components.dust.two_component.DustSEDComponent`
    (two_component path) to sum over young ages only
    (lyc_reprocessed_by='young', #2539 item 2).
    """
    log_L_lyc_age = jnp.asarray(log_L_lyc_age)
    weights = jnp.ones_like(log_L_lyc_age) if weights is None else jnp.asarray(weights)

    # Corrupt (+inf/NaN) entries are substituted with -inf for the finite-path
    # arithmetic below; `any_corrupt` (not this substitution) is what reports
    # them in the return value -- see the Returns section above.
    corrupt = _not_computable(log_L_lyc_age)
    any_corrupt = jnp.any(corrupt)
    safe_log = jnp.where(corrupt, -jnp.inf, log_L_lyc_age)

    # log10(sum(w * 10^x)) = max_log + log10(sum(w * 10^(x - max_log))),
    # peak-factored (the logsumexp max trick) so 10**(x - max_log) never
    # overflows. First double-where: max_log_safe substitutes a finite dummy
    # (0.0) for the "every entry is -inf" case, so the exponent is never
    # literally (-inf - -inf); the real -inf sentinel is restored by the
    # `jnp.isfinite(max_log)` branch of the outer where below.
    max_log = jnp.max(safe_log, initial=-jnp.inf)
    max_log_safe = jnp.where(jnp.isfinite(max_log), max_log, 0.0)
    sum_weighted = jnp.sum(weights * jnp.power(10.0, safe_log - max_log_safe))
    # Second double-where: the brightest (max_log) age can carry zero weight
    # while a dimmer age carries all of it, so sum_weighted == 0 is reachable
    # even when max_log is finite -- jnp.log10(0) is -inf, not NaN, but its
    # gradient is singular there, the same removable-zero trap
    # log10_fdust_lyc_credit's docstring derives for a plain jnp.log10. Clamp
    # the argument to a finite dummy (1.0) before the log; select the -inf
    # sentinel afterwards with an independent, constant-off-branch where.
    sum_safe = jnp.where(sum_weighted > 0.0, sum_weighted, 1.0)
    log_sum = jnp.where(
        jnp.isfinite(max_log) & (sum_weighted > 0.0),
        max_log + jnp.log10(sum_safe),
        -jnp.inf,
    )
    result = jnp.where(any_corrupt, jnp.inf, log_sum)

    return result


def credited_log10_lyc(derived, young_fraction, mode: str):
    r"""log10 of the ionizing luminosity the HII-region dust credit is taken over.

    The credited luminosity must be the Lyman continuum of the SAME stellar
    population the nebular escape/dust factor was applied to: the whole
    population under ``lyc_reprocessed_by='all'`` (FSPS/CIGALE), the youngest
    interval (stars younger than the youngest window edge, weighted by each
    node's formed-mass fraction there) under ``'young'``.  Crediting the whole
    population in the latter case would credit dust for old-star ionizing
    light that never reached any gas.

    Parameters
    ----------
    derived : mapping
        ``state.derived`` carrying stellar's ``log_L_lyc`` (whole population),
        ``lnu_age_ion`` / ``ssp_wave_ion`` (per-age ionizing slice, per Msun),
        ``log_stellar_mass_scale`` and ``log_L_lyc_age``.
    young_fraction : ndarray, shape (n_age,)
        Fraction of each SSP node's formed mass in the youngest interval
        [dimensionless].
    mode : str
        ``'young'`` or ``'all'``.

    Returns
    -------
    ndarray, shape () or None
        ``log10(L_LyC / (erg/s))``, or ``None`` when ``mode='all'`` and the
        nebular component published no whole-population key.

    Notes
    -----
    **JIT/grad/vmap-compatible.**  The ``'young'`` branch weights-and-sums the
    per-age ionizing slice over ages first (linear) and integrates once, which
    equals ``log10_age_sum_lyc(log_L_lyc_age, weights=young_fraction)`` to
    round-off but pays for the edge-aware quadrature a single time.
    """
    if mode == "all":
        return derived.get("log_L_lyc")
    lnu_age_ion = derived.get("lnu_age_ion")
    ssp_wave_ion = derived.get("ssp_wave_ion")
    if lnu_age_ion is not None:
        young_lnu = jnp.sum(young_fraction[:, None] * lnu_age_ion, axis=0)
        return log10_lyc_luminosity(
            young_lnu, ssp_wave_ion, log10_scale=derived["log_stellar_mass_scale"]
        )
    return log10_age_sum_lyc(derived["log_L_lyc_age"], weights=young_fraction)
