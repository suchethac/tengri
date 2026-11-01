# SPDX-License-Identifier: BSD-3-Clause
"""Young-fraction coordinates for the continuity SFH (stick-breaking from the youngest bin).

The continuity SFH (:func:`~tengri.components.stellar.sfh.nonparametric.continuity`)
samples the log10 SFR ratios :math:`r_k = \\log_{10}(\\mathrm{SFR}_k/\\mathrm{SFR}_{k+1})`.
For a dwarf whose youngest-bin mass is pinned by the photometry, the posterior in
those coordinates is a curved ridge. This module provides the alternative
coordinates: stick-breaking mass fractions taken from the youngest bin,
:math:`f_k = M_k / R_k` with :math:`R_k = M - \\sum_{j<k} M_j`. The fraction
:math:`f_0 = M_0/M` is the photometrically pinned young mass, so it is one axis.

The map is a bijection between :math:`(0, 1)^6` and :math:`\\mathbb{R}^6`. The
prior that the ratios carry, a Student-t density on each :math:`r_k`, is carried
to the fractions as its exact pullback, so both coordinate systems describe the
same posterior. The pullback needs the Jacobian of the map, which is bidiagonal,
and the reference density of the standardized seam, which is uniform on each
fraction.

Notes
-----
The shipped registry declares the ratio prior without a truncation. The
truncated form, the Student-t restricted to :math:`|r_k| \\le c`, is available
through the ``lo`` and ``hi`` fields of :class:`YoungFractionPrior`; outside the
box the density is :math:`-\\infty`, which is a refusal, not a floor.

References
----------
.. [1] J. Leja et al., "How to Measure Galaxy Star Formation Histories. II.
   Nonparametric Models," ApJ, 876, 3 (2019). arXiv:1811.03637.
   https://doi.org/10.3847/1538-4357/ab133c
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.special import betainc, log_ndtr
from jax.scipy.stats import t as student_t

from tengri.components.stellar.sfh.nonparametric import (
    DEFAULT_BIN_EDGES_GYR,
    continuity,
)
from tengri.utils.host_array import device_table, host_array

N_YOUNG_FRACTIONS = 6
YOUNG_FRACTION_PREFIX = "sfh_cont_young_frac_"
YOUNG_FRACTION_NAMES = tuple(f"{YOUNG_FRACTION_PREFIX}{k}" for k in range(N_YOUNG_FRACTIONS))

_LN10 = math.log(10.0)


def log_width_steps(bin_edges_gyr) -> jnp.ndarray:
    r"""Per-ratio log10 width steps :math:`\Delta_k = \log_{10}(\Delta t_{k+1}/\Delta t_k)`.

    Parameters
    ----------
    bin_edges_gyr : array_like, shape (7,)
        Edges of the seven bins [Gyr], ascending.

    Returns
    -------
    ndarray, shape (6,)
        :math:`\Delta_k` [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes for traced edges; the edges are static in practice.
    """
    widths = jnp.diff(jnp.asarray(bin_edges_gyr))
    return jnp.log10(widths[1:] / widths[:-1])


def flat_young_fractions(bin_edges_gyr) -> jnp.ndarray:
    r"""Young fractions of the flat SFH, where every ratio is zero.

    Each fraction is :math:`f_k = \Delta t_k / \sum_{j \ge k} \Delta t_j`.

    Parameters
    ----------
    bin_edges_gyr : array_like, shape (7,)
        Edges of the seven bins [Gyr], ascending.

    Returns
    -------
    ndarray, shape (6,)
        Fractions in (0, 1) [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes. Used for the registry default and for the SFH
    default when a fraction is not supplied.
    """
    widths = jnp.diff(jnp.asarray(bin_edges_gyr))
    tail = jnp.cumsum(widths[::-1])[::-1]
    return (widths / tail)[:-1]


def ratios_from_young_fractions(f, delta) -> jnp.ndarray:
    r"""Map young fractions to log10 SFR ratios.

    .. math::

        r_k = \log_{10}\frac{f_k}{1 - f_k} - \log_{10} f_{k+1} + \Delta_k,
        \qquad f_6 \equiv 1,

    with :math:`\Delta_k` from :func:`log_width_steps`.

    Parameters
    ----------
    f : array_like, shape (6,)
        Young fractions :math:`f_k` in (0, 1) [dimensionless].
    delta : array_like, shape (6,)
        Width steps :math:`\Delta_k` [dimensionless].

    Returns
    -------
    ndarray, shape (6,)
        Ratios :math:`r_k` [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes in the open box (0, 1)^6; a
    fraction at exactly 0 or 1 gives a non-finite ratio, as the logarithm does.
    """
    f = jnp.asarray(f)
    log_f = jnp.log(f)
    log_1m_f = jnp.log1p(-f)
    log_f_next = jnp.concatenate([log_f[1:], jnp.zeros_like(log_f[:1])])
    return (log_f - log_1m_f - log_f_next) / _LN10 + jnp.asarray(delta)


def young_fractions_from_ratios(r, delta) -> jnp.ndarray:
    r"""Invert :func:`ratios_from_young_fractions`.

    The map is triangular in the oldest-first order: :math:`f_5` follows from
    :math:`r_5` alone, then each :math:`f_k` from :math:`r_k` and :math:`f_{k+1}`.

    Parameters
    ----------
    r : array_like, shape (6,)
        Ratios :math:`r_k` [dimensionless].
    delta : array_like, shape (6,)
        Width steps :math:`\Delta_k` [dimensionless].

    Returns
    -------
    ndarray, shape (6,)
        Young fractions in (0, 1) [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes (a fixed-length Python loop, unrolled under trace).
    """
    z = (jnp.asarray(r) - jnp.asarray(delta)) * _LN10
    log_f_next = jnp.zeros_like(z[:1])[0]
    out = [None] * N_YOUNG_FRACTIONS
    for k in reversed(range(N_YOUNG_FRACTIONS)):
        logit = z[k] + log_f_next
        out[k] = jax.nn.sigmoid(logit)
        log_f_next = jax.nn.log_sigmoid(logit)
    return jnp.stack(out)


def _student_t_cdf(x, df, scale):
    """Student-t CDF with zero location, through the regularized incomplete beta function."""
    t_std = x / scale
    tail = 0.5 * betainc(0.5 * df, 0.5, df / (df + t_std * t_std))
    return jnp.where(t_std < 0.0, tail, 1.0 - tail)


def _student_t_log_normalizer(df, scale, lo, hi) -> jnp.ndarray:
    """Log of the probability mass the Student-t keeps on ``[lo, hi]``."""
    return jnp.log(_student_t_cdf(hi, df, scale) - _student_t_cdf(lo, df, scale))


@dataclass(frozen=True)
class YoungFractionPrior:
    r"""Exact pullback of the continuity ratio prior to the young fractions.

    Parameters
    ----------
    delta : tuple of float, length 6
        Width steps :math:`\Delta_k` from :func:`log_width_steps` [dimensionless].
    df : float
        Student-t degrees of freedom of the ratio prior [dimensionless].
    scale : float
        Student-t scale of the ratio prior [dex].
    lo, hi : float or None
        Support of the ratio prior in dex. ``None`` leaves that side untruncated.

    Notes
    -----
    The log-density of the fractions, with respect to Lebesgue measure on
    :math:`(0,1)^6`, is

    .. math::

        \log p(f) = \sum_k \left[\log t_{\nu,\sigma}(r_k(f)) - \log Z\right]
        - \sum_k \left[\log\ln 10 + \log f_k + \log(1 - f_k)\right],

    where :math:`Z` is the truncation mass and the last sum is
    :math:`\log|\det \partial r/\partial f|`. It is :math:`-\infty` where any
    :math:`r_k` leaves ``[lo, hi]``.

    The seam's latents are iid N(0, 1) and map to fractions by
    :math:`f = \Phi(\xi)`, the uniform reference, so the term contributed to the
    negative log-prior of the latents is :math:`-\log p(f(\xi))`. Under the seam
    prior :math:`\phi(\xi)` this gives an induced density of exactly
    :math:`p(f)` on the fractions.

    **JIT-compatible**: yes; a frozen dataclass of static floats.
    **Gradient-safe**: the latent form uses :func:`jax.scipy.special.log_ndtr`,
    so :math:`\log f` and :math:`\log(1-f)` stay finite for large :math:`|\xi|`.
    """

    delta: tuple
    df: float = 2.0
    scale: float = 0.3
    lo: float | None = None
    hi: float | None = None

    def log_density_from_latents(self, xi) -> jnp.ndarray:
        r"""Log-density of the fractions :math:`f = \Phi(\xi)` at standardized latents ``xi``.

        Parameters
        ----------
        xi : array_like, shape (6,)
            Standardized latents, one per young fraction [dimensionless].

        Returns
        -------
        ndarray, scalar
            :math:`\log p(f(\xi))` [nats], :math:`-\infty` outside the truncation box.

        Notes
        -----
        **JIT-compatible**: yes. **Gradient-safe**: yes; the box test is a
        ``jnp.where`` on finite inputs, and the excluded branch is never ``nan``.
        """
        xi = jnp.asarray(xi)
        delta = jnp.asarray(self.delta, dtype=xi.dtype)
        log_f = log_ndtr(xi)
        log_1m_f = log_ndtr(-xi)
        log_f_next = jnp.concatenate([log_f[1:], jnp.zeros_like(log_f[:1])])
        r = (log_f - log_1m_f - log_f_next) / _LN10 + delta
        lo = -jnp.inf if self.lo is None else self.lo
        hi = jnp.inf if self.hi is None else self.hi
        logp = student_t.logpdf(r, self.df, loc=0.0, scale=self.scale)
        if self.lo is not None or self.hi is not None:
            logp = logp - N_YOUNG_FRACTIONS * _student_t_log_normalizer(
                self.df, self.scale, lo, hi
            )
        log_jac = -jnp.sum(math.log(_LN10) + log_f + log_1m_f)
        total = jnp.sum(logp) + log_jac
        inside = jnp.all((r >= lo) & (r <= hi))
        return jnp.where(inside, total, -jnp.inf)


def young_fraction_prior_from_edges(bin_edges_gyr, *, df=2.0, scale=0.3, lo=None, hi=None):
    """Build the :class:`YoungFractionPrior` for a bin ladder.

    Parameters
    ----------
    bin_edges_gyr : array_like, shape (7,)
        Edges of the seven bins [Gyr], ascending.
    df, scale : float, optional
        Student-t degrees of freedom and scale of the ratio prior (defaults
        match the registry declaration of ``sfh_cont_ratio_k``).
    lo, hi : float or None, optional
        Truncation of the ratio prior [dex]; ``None`` leaves that side open.

    Returns
    -------
    YoungFractionPrior
    """
    delta = tuple(float(x) for x in host_array(log_width_steps(bin_edges_gyr)))
    return YoungFractionPrior(delta=delta, df=df, scale=scale, lo=lo, hi=hi)


def young_fraction_latents(params_unbounded) -> jnp.ndarray:
    """Stack the six young-fraction latents of a latent dict in declared order.

    Parameters
    ----------
    params_unbounded : dict of str to array_like
        Standardized latents keyed by parameter name.

    Returns
    -------
    ndarray, shape (6,)
        Latents ``sfh_cont_young_frac_0`` .. ``_5``.
    """
    return jnp.stack([jnp.asarray(params_unbounded[name]) for name in YOUNG_FRACTION_NAMES])


def young_fraction_prior_for_spec(spec) -> YoungFractionPrior | None:
    """The young-fraction prior a spec needs on the standardized seam, or ``None``.

    Parameters
    ----------
    spec : Parameters
        Model parameter spec. Read: ``mean_sfh_type``, ``free_params``,
        ``bin_edges_gyr``.

    Returns
    -------
    YoungFractionPrior or None
        ``None`` unless the continuity SFH runs in young-fraction coordinates with
        its fractions free.

    Raises
    ------
    ConfigError
        If the young-fraction SFH has some but not all of its fractions free. A
        partly fixed block has no exact pullback on the free part.
    """
    from tengri.config.exceptions import ConfigError

    if not _is_young_fraction_spec(spec):
        return None
    free = set(spec.free_params)
    present = [name for name in YOUNG_FRACTION_NAMES if name in free]
    if not present:
        return None
    if len(present) != N_YOUNG_FRACTIONS:
        missing = [name for name in YOUNG_FRACTION_NAMES if name not in free]
        raise ConfigError(
            "sfh coordinates='young_fraction' needs all six young fractions free "
            f"(all_params=FREE or each named); fixed here: {missing}. The young-fraction "
            "prior is an exact pullback of the ratio prior only over the full block."
        )
    edges = spec.bin_edges_gyr
    if edges is None:
        edges = host_array(device_table(DEFAULT_BIN_EDGES_GYR))
    return young_fraction_prior_from_edges(edges)


def _is_young_fraction_spec(spec) -> bool:
    """Whether a spec's SFH type is the young-fraction continuity."""
    sfh = getattr(spec, "mean_sfh_type", None)
    types = sfh if isinstance(sfh, (list, tuple)) else [sfh]
    return "continuity_young_fraction" in types


def continuity_young_fraction(
    age_yr: jnp.ndarray,
    log_total_mass: float = 10.0,
    bin_edges_gyr: jnp.ndarray | None = None,
    **young_frac_kwargs,
) -> jnp.ndarray:
    r"""Continuity SFH parameterized by young fractions instead of ratios.

    The fractions :math:`f_k` set the bin masses by stick-breaking from the
    youngest bin, :math:`M_k = f_k R_k`. The SFH is then the
    :func:`~tengri.components.stellar.sfh.nonparametric.continuity` SFH at the
    ratios of :func:`ratios_from_young_fractions`, so the two parameterizations
    describe the same family of histories.

    Parameters
    ----------
    age_yr : array_like, shape (n_age,)
        Lookback time grid [yr].
    log_total_mass : float, optional
        log10 of total stellar mass formed [Msun]. Default 10.0.
    bin_edges_gyr : array_like, shape (7,), optional
        Bin edges [Gyr]. Default: the seven-bin ladder ``DEFAULT_BIN_EDGES_GYR``.
    **young_frac_kwargs
        ``young_frac_0`` .. ``young_frac_5``, the young fractions in (0, 1)
        [dimensionless]. A missing fraction takes its flat-SFH value.

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr], non-negative.

    Raises
    ------
    TypeError
        If an unknown keyword argument is supplied.

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes in the open box.
    Pairs with :meth:`YoungFractionPrior.log_density_from_latents`, which carries
    the Student-t ratio prior to these coordinates.
    """
    if bin_edges_gyr is None:
        bin_edges_gyr = device_table(DEFAULT_BIN_EDGES_GYR)
    n_bins = bin_edges_gyr.shape[0] - 1
    valid = {f"young_frac_{k}" for k in range(n_bins - 1)}
    unknown = sorted(k for k in young_frac_kwargs if k not in valid)
    if unknown:
        raise TypeError(
            f"continuity_young_fraction() got unexpected keyword argument(s) {unknown}; "
            f"the {n_bins}-bin grid accepts 'young_frac_0'..'young_frac_{n_bins - 2}' only."
        )
    flat = flat_young_fractions(bin_edges_gyr)
    f = jnp.stack([young_frac_kwargs.get(f"young_frac_{k}", flat[k]) for k in range(n_bins - 1)])
    ratios = ratios_from_young_fractions(f, log_width_steps(bin_edges_gyr))
    ratio_kwargs = {f"ratio_{k}": ratios[k] for k in range(n_bins - 1)}
    return continuity(age_yr, log_total_mass, bin_edges_gyr, **ratio_kwargs)
