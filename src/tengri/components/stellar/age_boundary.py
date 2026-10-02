# SPDX-License-Identifier: MIT
r"""Exact per-SSP-node formed-mass fractions younger than an age boundary.

Everything in tengri that splits a stellar population by age (the birth-cloud
screen of ``two_component``, every window of ``age_binned``, the young-only
Lyman-continuum credit) needs the same quantity: for each SSP template age
node :math:`a` and each age boundary :math:`b_k`, the fraction of the mass
that the age kernel deposits on node :math:`a` that formed *less than*
:math:`b_k` ago.  This module is the one definition of that fraction.

Survival function
-----------------
A mass parcel of age :math:`t` is "younger than :math:`b` (dispersed with
width :math:`w`)" with weight

.. math::

    S(t; b, w) = \begin{cases}
        \mathbf{1}[t < b] & w = 0 \quad (\text{hard step}),\\[2pt]
        \sigma\!\left(-\dfrac{\log_{10} t - \log_{10} b}{w}\right) & w > 0 ,
    \end{cases}

with :math:`b` in yr and :math:`w` in dex.  The step is the default; the
log-logistic spread is opt-in (``transition_width_dex > 0``).

Exact node fractions
--------------------
The age kernel turns a star-formation history into node weights
:math:`W_a = \sum_i m_i\,\kappa_{ia}` (parcel masses :math:`m_i`, kernel
shares :math:`\kappa_{ia}`).  The young mass on node :math:`a` is the *same*
kernel applied to :math:`m_i \bar S_i`, where :math:`\bar S_i` is
:math:`S` averaged over the parcel's own time cell:

.. math::

    F_a(b) = \frac{\sum_i m_i \bar S_i\,\kappa_{ia}}{\sum_i m_i\,\kappa_{ia}},
    \qquad
    \bar S_i = \frac{1}{t_i^{+} - t_i^{-}} \int_{t_i^{-}}^{t_i^{+}} S(t)\,dt .

For the step :math:`\bar S_i` is the exact share of the cell younger than
:math:`b` (piecewise linear in :math:`b`); for the smooth law it is a
composite Gauss-Legendre quadrature in :math:`v = \ln(t/b)/(w\ln 10)`
accurate to better than :math:`10^{-9}` (measured in
``tests/physics/limit/test_age_boundary_fraction.py``).  A grid node that
happens to straddle :math:`b` therefore contributes exactly the share of its
mass younger than :math:`b`, instead of being all-or-nothing (step) or
sampled at one age (smooth).

Reference codes
---------------
Implements the same model as CF00 and FSPS (``dust_tesc``, which split the
continuous age integral at the birth-cloud lifetime), pcigale
(``separation_age``, a step on 1-Myr SSPs: the same thing at 1-Myr
resolution) and bagpipes (which splits the age bin straddling ``t_bc`` by the
fraction of the bin younger than ``t_bc``).

Notes
-----
**JIT/grad/vmap-compatible**: static boundary tuple and width, pure ``jnp``.
Gradients flow through the parcel masses and cell edges; the boundary and
width are build-time constants.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "age_boundary_younger_fraction_cic",
    "age_boundary_younger_fraction_dsps",
    "cic_cell_edges",
    "survival_cell_mean",
    "validate_age_boundaries",
]

#: |v| beyond which the logistic is 0 or 1 to double precision (e^-40 ~ 4e-18).
_V_CLIP = 40.0
#: Gauss-Legendre points per panel and panels per cell for the smooth law.
_GL_POINTS = 8
_N_PANEL = 32
_GL_X, _GL_W = np.polynomial.legendre.leggauss(_GL_POINTS)


def validate_age_boundaries(boundaries_yr: Sequence[float], width_dex: float) -> tuple[float, ...]:
    """Validate a static age-boundary tuple and its dispersal width.

    Parameters
    ----------
    boundaries_yr : sequence of float
        Boundary ages [yr]; each must be finite and positive.
    width_dex : float
        Dispersal width [dex]; ``0`` selects the hard step, ``> 0`` the
        log-logistic spread.

    Returns
    -------
    tuple of float
        ``boundaries_yr`` as a tuple of Python floats.

    Raises
    ------
    ValueError
        On a non-positive or non-finite boundary, or a negative or
        non-finite width.
    """
    out = tuple(float(b) for b in boundaries_yr)
    for b in out:
        if not (math.isfinite(b) and b > 0.0):
            raise ValueError(f"age boundary must be finite and > 0 yr, got {b!r}")
    if not (math.isfinite(float(width_dex)) and float(width_dex) >= 0.0):
        raise ValueError(f"transition_width_dex must be finite and >= 0, got {width_dex!r}")
    return out


def cic_cell_edges(age):
    """Lower and upper lookback edges of every trapezoid parcel cell [yr].

    The cells are the ones ``_cic_parcels`` in the stellar component
    integrates: parcel ``i`` carries ``SFR_i * (hi_i - lo_i)``.

    Parameters
    ----------
    age : array_like, shape (n,)
        Ascending lookback ages of the (lookback-0-extended) integrand [yr].

    Returns
    -------
    lo, hi : ndarray, shape (n,)
        Cell edges [yr]; ``hi[i] == lo[i + 1]``, ``lo[0] == age[0]``,
        ``hi[-1] == age[-1]``.
    """
    age = jnp.asarray(age)
    d = jnp.diff(age)
    lo = jnp.concatenate([age[:1], age[1:] - 0.5 * d])
    hi = jnp.concatenate([age[:-1] + 0.5 * d, age[-1:]])
    return lo, hi


def survival_cell_mean(lo, hi, boundary_yr: float, width_dex: float):
    r"""Mean of :math:`S(t; b, w)` over cells :math:`[lo, hi]`, uniform in :math:`t`.

    Parameters
    ----------
    lo, hi : array_like, shape (n,)
        Cell edges [yr], ``hi >= lo >= 0``.
    boundary_yr : float
        Boundary age ``b`` [yr], static.
    width_dex : float
        Dispersal width ``w`` [dex], static; ``0`` is the exact step.

    Returns
    -------
    ndarray, shape (n,)
        :math:`\bar S` in ``[0, 1]`` [dimensionless].  Zero-width cells return
        :math:`S` at their edge.

    Notes
    -----
    **JIT/grad/vmap-compatible.**  Step: ``clip((b - lo) / (hi - lo), 0, 1)``.
    Smooth: the cell is split at :math:`b e^{\pm 40 c}` (``c = w ln 10``),
    outside which :math:`S` is 1 or 0 to round-off, and the remainder is
    integrated with 32 panels of 8-point Gauss-Legendre in
    :math:`v = \ln(t/b)/c`, panel half-width at most 1.25 against the
    logistic's nearest pole at :math:`|{\rm Im}\,v| = \pi`.
    """
    lo = jnp.asarray(lo)
    hi = jnp.asarray(hi)
    width = hi - lo
    positive = width > 0.0
    safe_width = jnp.where(positive, width, 1.0)
    b = float(boundary_yr)
    if float(width_dex) == 0.0:
        share = jnp.clip((b - lo) / safe_width, 0.0, 1.0)
        return jnp.where(positive, share, jnp.where(lo < b, 1.0, 0.0).astype(share.dtype))

    c = float(width_dex) * math.log(10.0)
    t_lo = b * math.exp(-c * _V_CLIP)
    t_hi = b * math.exp(c * _V_CLIP)
    young_floor = jnp.maximum(jnp.minimum(hi, t_lo) - lo, 0.0)
    a = jnp.clip(lo, t_lo, t_hi)
    z = jnp.clip(hi, t_lo, t_hi)
    v_a = jnp.log(a / b) / c
    v_z = jnp.log(z / b) / c
    frac = (jnp.arange(_N_PANEL, dtype=lo.dtype) + 0.5) / _N_PANEL  # panel centers in [0, 1]
    half = 0.5 / _N_PANEL
    span = (v_z - v_a)[:, None, None]
    v = (
        v_a[:, None, None]
        + span * frac[None, :, None]
        + span * half * jnp.asarray(_GL_X, dtype=lo.dtype)[None, None, :]
    )
    integrand = jax.nn.sigmoid(-v) * (b * c) * jnp.exp(c * v)
    transition = jnp.sum(integrand * jnp.asarray(_GL_W, dtype=lo.dtype)[None, None, :], axis=-1)
    transition = jnp.sum(transition, axis=-1) * (v_z - v_a) * half
    smooth = (young_floor + transition) / safe_width
    s_edge = jax.nn.sigmoid(-jnp.log(jnp.maximum(lo, jnp.finfo(lo.dtype).tiny) / b) / c)
    return jnp.where(positive, jnp.clip(smooth, 0.0, 1.0), s_edge)


def _safe_fraction(young, total):
    """``young / total`` per node, 0 where the node holds no mass, clipped to [0, 1]."""
    has_mass = total > 0.0
    frac = jnp.where(has_mass, young / jnp.where(has_mass, total, 1.0), 0.0)
    return jnp.clip(frac, 0.0, 1.0)


def age_boundary_younger_fraction_cic(
    contrib, idx, f, age, n_age: int, boundaries_yr: Sequence[float], width_dex: float
):
    r"""Per-node younger-than-boundary mass fraction for the cloud-in-cell kernel.

    Parameters
    ----------
    contrib, idx, f, age
        ``(contrib, idx, f, _, age)`` from ``_cic_parcels``: parcel masses
        [Msun], lower bracketing SSP node, log-age share on the upper node,
        and the extended lookback grid [yr].
    n_age : int
        Number of SSP age nodes.
    boundaries_yr : sequence of float
        Static boundary ages [yr].
    width_dex : float
        Dispersal width [dex]; ``0`` is the hard step.

    Returns
    -------
    ndarray, shape (n_boundary, n_age)
        :math:`F_a(b_k)` in ``[0, 1]`` [dimensionless]: the share of the mass
        on node ``a`` formed less than ``b_k`` ago.  0 where the node holds no
        mass.

    Notes
    -----
    **JIT/grad/vmap-compatible.**  The young mass is the CIC scatter of
    ``contrib * S_bar`` through the same ``(idx, f)`` as the node weights, so
    the same kernel produces numerator and denominator.  Every metallicity
    treatment (delta, lognormal MDF, per-age table) distributes each parcel
    over ``met`` with rows summing to one, so the age marginal needs no
    metallicity axis.
    """
    lo, hi = cic_cell_edges(age)
    s_bar = jnp.stack([survival_cell_mean(lo, hi, b, width_dex) for b in boundaries_yr])
    young_c = s_bar * contrib[None, :]
    zeros = jnp.zeros((s_bar.shape[0], n_age), dtype=contrib.dtype)
    young = (
        zeros.at[:, idx].add(young_c * (1.0 - f)[None, :]).at[:, idx + 1].add(young_c * f[None, :])
    )
    total = (
        jnp.zeros(n_age, dtype=contrib.dtype)
        .at[idx]
        .add(contrib * (1.0 - f))
        .at[idx + 1]
        .add(contrib * f)
    )
    return _safe_fraction(young, total[None, :])


def age_boundary_younger_fraction_dsps(
    gal_t_table,
    gal_sfr_table,
    ssp_lg_age_gyr,
    t_obs_gyr,
    boundaries_yr: Sequence[float],
    width_dex: float,
    youngest_multiplier,
):
    r"""Per-node younger-than-boundary mass fraction for DSPS's histogram kernel.

    DSPS assigns node :math:`a` the mass formed between the lookbacks of its
    log-midpoint bin edges :math:`e_a < e_{a+1}`:
    :math:`W_a = M_b(e_a) - M_b(e_{a+1})`, with :math:`M_b(x)` the mass formed
    *before* lookback :math:`x` (a cumulative function interpolated in
    :math:`\log_{10} M` against :math:`\log_{10} t`).  The young share of the
    bin is the same differences of the same :math:`M_b`.

    Step: :math:`M_b(e_a) - M_b(\mathrm{clip}(b, e_a, e_{a+1}))`.  Smooth,
    by parts:
    :math:`S(e_a) M_b(e_a) - S(e_{a+1}) M_b(e_{a+1}) - \int M_b\,\sigma(v)\sigma(-v)\,dv`,
    integrated with the panel Gauss-Legendre rule of :func:`survival_cell_mean`.

    Parameters
    ----------
    gal_t_table, gal_sfr_table : array_like
        The (cosmic time [Gyr], SFR [Msun/yr]) table handed to DSPS.
    ssp_lg_age_gyr : array_like, shape (n_age,)
        ``log10`` SSP ages [Gyr].
    t_obs_gyr : float
        Cosmic age at the observation [Gyr].
    boundaries_yr : sequence of float
        Static boundary ages [yr].
    width_dex : float
        Dispersal width [dex]; ``0`` is the hard step.
    youngest_multiplier : array_like, shape (n_age,)
        The #821 youngest-bin multiplier (``mult`` on the youngest finite
        node, 1 elsewhere): DSPS clips that bin's lower edge at ``e_lo > 0``
        and tengri restores the ``[0, e_lo]`` sliver at constant SFR.  The
        restored mass is younger than ``b`` for the share of ``[0, e_lo]``
        below ``b``.

    Returns
    -------
    ndarray, shape (n_boundary, n_age)
        :math:`F_a(b_k)` in ``[0, 1]`` [dimensionless].

    Notes
    -----
    **JIT/grad/vmap-compatible.**  Uses DSPS's private
    ``_calc_logsm_table_from_sfh_table``, ``_get_lg_age_bin_edges`` and
    ``_get_lgt_birth`` so the cumulative function is the one DSPS weights
    from; a DSPS rename fails the tests loudly.
    """
    from tengri._x64_hold import hold_x64_preference

    with hold_x64_preference():
        from dsps.constants import SFR_MIN
        from dsps.sed.stellar_age_weights import (
            _calc_logsm_table_from_sfh_table,
            _get_lg_age_bin_edges,
            _get_lgt_birth,
        )

    lgt_table = jnp.log10(gal_t_table)
    logsm_table = _calc_logsm_table_from_sfh_table(gal_t_table, gal_sfr_table, SFR_MIN)
    lg_edges = _get_lg_age_bin_edges(ssp_lg_age_gyr)  # (n_age + 1,), log10 Gyr

    def mass_older_than(lg_x_gyr):
        return 10.0 ** jnp.interp(_get_lgt_birth(t_obs_gyr, lg_x_gyr), lgt_table, logsm_table)

    e_gyr = 10.0**lg_edges
    m_edge = mass_older_than(lg_edges)  # (n_age + 1,)
    total = m_edge[:-1] - m_edge[1:]
    mult = jnp.asarray(youngest_multiplier)
    extra = (mult - 1.0) * total
    e_lo_yr = e_gyr[:-1] * 1e9
    rows = []
    for b in boundaries_yr:
        b_gyr = b * 1e-9
        if float(width_dex) == 0.0:
            clipped = jnp.clip(b_gyr, e_gyr[:-1], e_gyr[1:])
            core = m_edge[:-1] - mass_older_than(jnp.log10(clipped))
        else:
            c = float(width_dex) * math.log(10.0)
            v_edge = jnp.clip(
                jnp.log(jnp.maximum(e_gyr, jnp.finfo(e_gyr.dtype).tiny) / b_gyr) / c,
                -_V_CLIP,
                _V_CLIP,
            )
            s_edge = jax.nn.sigmoid(-v_edge)
            v_a, v_z = v_edge[:-1], v_edge[1:]
            frac = (jnp.arange(_N_PANEL, dtype=v_edge.dtype) + 0.5) / _N_PANEL
            half = 0.5 / _N_PANEL
            span = (v_z - v_a)[:, None, None]
            v = (
                v_a[:, None, None]
                + span * frac[None, :, None]
                + span * half * jnp.asarray(_GL_X, dtype=v_edge.dtype)[None, None, :]
            )
            m_v = mass_older_than(math.log10(b_gyr) + float(width_dex) * v)
            dens = jax.nn.sigmoid(v) * jax.nn.sigmoid(-v)
            by_parts = jnp.sum(m_v * dens * jnp.asarray(_GL_W, dtype=v_edge.dtype), axis=-1)
            by_parts = jnp.sum(by_parts, axis=-1) * (v_z - v_a) * half
            core = s_edge[:-1] * m_edge[:-1] - s_edge[1:] * m_edge[1:] - by_parts
        s_ext = survival_cell_mean(jnp.zeros_like(e_lo_yr), e_lo_yr, b, width_dex)
        young = core + extra * s_ext
        rows.append(_safe_fraction(young, (total + extra)[None, :])[0])
    return jnp.stack(rows)
