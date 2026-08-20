# SPDX-License-Identifier: BSD-3-Clause
"""Probability distribution objects for parameter specification.

Each distribution defines a prior for a single model parameter.
Used by Parameters for both mock generation (sampling) and inference (priors).

All methods are JAX-compatible for use inside JIT-compiled functions.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np


def _norm_cdf_float(x: float) -> float:
    """Standard normal CDF Φ(x) for a Python float; handles ±inf exactly."""
    if math.isinf(x):
        return 0.0 if x < 0 else 1.0
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# Guard for probabilities entering inverse CDFs (matches Uniform.standardize).
_P_EPS = 1e-7

# ── Base class ────────────────────────────────────────────────────


class Distribution:
    """Base class for parameter distributions.

    A Distribution defines a prior for a single model parameter. Supports both
    sampling (for mock generation) and probability evaluation (for inference).

    Subclasses must implement ``bounds``, ``sample()``, and ``log_prob()``.
    For standardized inference with reparameterization, subclasses should also
    implement ``unstandardize()`` and ``standardize()``.

    Parameters
    ----------
    (None — this is an abstract base class)

    Attributes
    ----------
    bounds : tuple[float, float]
        (lo, hi) — lower and upper bounds on the parameter value.
    description : str
        Human-readable summary of the quantity, surfaced by
        ``describe_parameter`` and ``spec.summary()``. Empty when unset.
    units : str
        Physical units of the quantity, e.g. ``"erg/s"``, ``"yr"``,
        ``"Msun/yr"``. Empty when dimensionless or unset.

    Methods
    -------
    sample(key : jax.Array) → ndarray
        Draw a random sample from the prior distribution.
    log_prob(x : ndarray) → ndarray
        Evaluate the log probability density at parameter value x.
    unstandardize(xi : ndarray) → ndarray
        Map standardized latent variable ξ ~ N(0,1) to physical parameter space.
    standardize(theta : ndarray) → ndarray
        Map physical parameter to standardized latent variable ξ ~ N(0,1).

    Notes
    -----
    **Standardized inference**: The unstandardize method absorbs the prior into
    the forward model via a change-of-variables. If the map h(ξ) satisfies
    P(h(ξ))|dh/dξ| = φ(ξ) (where φ is the standard normal density), then the
    loss becomes H = ½χ² + ½ξᵀξ with no extra per-parameter prior penalty terms.
    This Jacobian cancellation is exact (not an approximation).

    **Posterior geometry**: This reparameterization makes the posterior landscape
    isotropic in the prior directions, with unit-scale curvature from the prior.
    This benefits all samplers: variational (geoVI/MGVI), MCMC, and gradient-based
    optimization.

    **Abstract base class**: Distribution is an abstract base class defining the
    interface for all prior distributions. Subclasses must implement ``bounds``,
    ``sample()``, ``log_prob()``, ``unstandardize()``, and ``standardize()``.
    All methods are JIT-compatible when implemented using JAX primitives.

    Examples
    --------
    Creating custom priors:

    >>> from tengri import Uniform, Gaussian, Fixed
    >>> # Shorthand notation (resolved in Parameters)
    >>> p1 = Uniform(0, 1)
    >>> p2 = Gaussian(mu=0, sigma=0.5)
    >>> p3 = Fixed(0.0)
    >>> # Sample and evaluate
    >>> import jax.random
    >>> key = jax.random.PRNGKey(0)
    >>> sample = p1.sample(key)
    >>> log_p = p1.log_prob(sample)
    """

    # Class-level defaults so that *every* distribution answers to
    # ``.description`` and ``.units``. Before these existed, four of the
    # seven subclasses raised AttributeError, which is why consumers read
    # them as ``getattr(prior, "units", "")`` — a fail-open guard whose only
    # job was to paper over an incomplete base class. Subclasses that accept
    # the arguments shadow these with instance attributes.
    description: str = ""
    units: str = ""

    @property
    def is_fixed(self) -> bool:
        """Return True if this is a Fixed distribution, False otherwise.

        Returns
        -------
        bool
            True if this is a Fixed distribution, False otherwise.
        """
        return False

    @property
    def default(self) -> float | str | None:
        """Physically-motivated default value when this knob is marked FIXED
        without an explicit user-supplied value.

        Returns None if no default was registered at construction. Downstream
        code (``parameters/groups.py``) raises ``ParameterDefaultMissingError``
        if a registry entry is converted to ``Fixed`` via the ``'*': FIXED``
        wildcard but the distribution carries no default.

        For ``Fixed(value)``, ``default`` is the value itself — see
        ``Fixed.default``.

        Returns
        -------
        float, str, or None
            The registered default, or None if no default was set.
        """
        return getattr(self, "_default", None)

    def _register_default(self, default: float | None) -> None:
        """Validate ``default`` against this distribution's bounds and store it.

        Called by every concrete subclass ``__init__`` after ``bounds`` is set.
        Centralizing the validation here keeps the per-subclass boilerplate to
        a single line.

        Parameters
        ----------
        default : float or None
            Physically-motivated default value, or ``None`` if no default is
            registered.

        Raises
        ------
        ValueError
            If ``default`` is finite-numeric and falls outside ``self.bounds``.
        """
        if default is None:
            self._default = None
            return
        lo, hi = self.bounds
        # Skip range check for distributions without numeric bounds (Fixed with
        # string value would return (None, None); never reaches here in practice
        # because Fixed overrides ``default``).
        if lo is None or hi is None:
            self._default = default
            return
        if not (lo <= float(default) <= hi):
            raise ValueError(
                f"{type(self).__name__}: default={default} is outside bounds "
                f"[{lo}, {hi}]. Defaults must lie within the prior support so "
                f"the FIXED-fallback value is consistent with the prior."
            )
        self._default = float(default)

    @property
    def bounds(self) -> tuple[float, float]:
        """Lower and upper bounds [lo, hi] for this distribution.

        Returns
        -------
        tuple[float, float]
            Lower and upper bounds (lo, hi).
        """
        raise NotImplementedError

    def sample(self, key: jax.Array) -> jnp.ndarray:
        """Draw a random sample from the prior distribution.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key for random sampling.

        Returns
        -------
        ndarray
            A single sample drawn from the prior distribution.
        """
        raise NotImplementedError

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate log probability density at parameter value x.

        Parameters
        ----------
        x : float or array_like
            Parameter value in physical (unstandardized) space.

        Returns
        -------
        float
            Log probability density at x.
        """
        raise NotImplementedError

    def unstandardize(self, xi: jnp.ndarray) -> jnp.ndarray:
        """Map standardized latent ξ ~ N(0,1) → physical parameter.

        Must be JAX-differentiable. This is the core method for
        standardized inference — it absorbs the prior into the
        forward model so the loss is always ½χ² + ½ξᵀξ.

        The transform h(ξ) is chosen so that P(h(ξ)) |dh/dξ| = φ(ξ),
        where φ is the standard normal density. This Jacobian cancellation
        is exact (not an approximation) and is what eliminates all
        per-parameter prior penalty terms from the loss function.

        The geometric consequence: every parameter has unit-scale
        curvature from the prior, making the posterior landscape
        isotropic in the prior directions. This benefits all samplers,
        not just the variational ones (geoVI/MGVI) that require it.

        Parameters
        ----------
        xi : float or array_like
            Standardized latent value, typically from a standard normal
            distribution.

        Returns
        -------
        float or ndarray
            Physical-space parameter value.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement unstandardize()")

    def standardize(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Map physical parameter → standardized latent ξ.

        Inverse of unstandardize. Used for initialization from
        physical parameter values (e.g., from a MAP solution).

        Parameters
        ----------
        theta : float or array_like
            Physical-space parameter value.

        Returns
        -------
        float or ndarray
            Standardized latent-space value.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement standardize()")


# ── Concrete distributions ────────────────────────────────────────


class Uniform(Distribution):
    """Uniform prior on [lo, hi].

    A flat probability density on the interval [lo, hi]. Commonly used for
    bounded astrophysical quantities with little prior knowledge. Reparameterizes
    via the Gaussian CDF to ensure differentiability and automatic bound
    satisfaction.

    Parameters
    ----------
    lo : float
        Lower bound (inclusive).
    hi : float
        Upper bound (inclusive). Must satisfy hi > lo.

    Attributes
    ----------
    lo : float
        Lower bound of the distribution.
    hi : float
        Upper bound of the distribution.
    bounds : tuple[float, float]
        ``(lo, hi)`` convenience tuple.

    Raises
    ------
    ValueError
        If lo >= hi.

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    **Standardization**: Maps ξ ~ N(0,1) to θ via the Gaussian CDF:

    .. math::

        \\theta = lo + (hi - lo) \\cdot \\Phi(\\xi)

    where Φ(ξ) = 0.5 * (1 + erf(ξ / sqrt(2))) is the standard normal CDF.
    At ξ = 0 (prior center), Φ(0) = 0.5, so θ = (lo + hi) / 2.
    This ensures an N(0,1) latent yields a genuine uniform prior on [lo, hi],
    not a midpoint-peaked one. Automatic bound satisfaction and smooth gradients.

    Examples
    --------
    >>> import jax.random
    >>> from tengri import Uniform
    >>> prior = Uniform(0, 1)
    >>> key = jax.random.PRNGKey(0)
    >>> sample = prior.sample(key)
    >>> print(f"Sample: {sample:.4f}")  # Will be in [0, 1)
    >>> log_prob = prior.log_prob(0.5)
    >>> print(f"log p(0.5): {log_prob:.4f}")  # ≈ 0.0 (log(1) = 0)
    """

    def __init__(
        self,
        lo: float,
        hi: float,
        description: str = "",
        *,
        units: str = "",
        default: float | None = None,
    ):
        if lo >= hi:
            raise ValueError(f"Uniform requires lo < hi, got lo={lo}, hi={hi}")
        self._lo = float(lo)
        self._hi = float(hi)
        self.description = description
        self.units = units
        self._register_default(default)

    @property
    def lo(self) -> float:
        """Lower bound of the uniform distribution.

        Returns
        -------
        float
            Lower bound value.
        """
        return self._lo

    @property
    def hi(self) -> float:
        """Upper bound of the uniform distribution.

        Returns
        -------
        float
            Upper bound value.
        """
        return self._hi

    @property
    def bounds(self) -> tuple[float, float]:
        """Lower and upper bounds [lo, hi].

        Returns
        -------
        tuple[float, float]
            Bounds as (lo, hi) tuple.
        """
        return (self._lo, self._hi)

    def sample(self, key: jax.Array) -> jnp.ndarray:
        """Draw one sample uniformly from [lo, hi].

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key for random sampling.

        Returns
        -------
        ndarray
            A single sample uniformly distributed in [lo, hi].
        """
        return jax.random.uniform(key, minval=self._lo, maxval=self._hi)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return log probability: -log(hi-lo) inside bounds, -inf outside.

        Parameters
        ----------
        x : float or array_like
            Parameter value in physical space.

        Returns
        -------
        float
            Log probability density at x.
        """
        in_bounds = (x >= self._lo) & (x <= self._hi)
        return jnp.where(in_bounds, -jnp.log(self._hi - self._lo), -jnp.inf)

    def unstandardize(self, xi: jnp.ndarray) -> jnp.ndarray:
        """ξ ~ N(0,1) → Uniform(lo, hi) via Gaussian CDF.

        Uses the standard normal CDF Φ(ξ) = 0.5 * (1 + erf(ξ / sqrt(2))) to map
        N(0,1) latent variables to uniform on (lo, hi). At ξ=0, Φ(0) = 0.5,
        so θ = midpoint of [lo, hi]. This ensures an N(0,1) latent yields a
        genuine uniform prior, not a midpoint-peaked one.

        Parameters
        ----------
        xi : float or array_like
            Standardized latent value from standard normal distribution.

        Returns
        -------
        float or ndarray
            Physical-space parameter in [lo, hi].
        """
        # Standard normal CDF: Φ(ξ) = 0.5 * (1 + erf(ξ / sqrt(2)))
        phi_xi = 0.5 * (1.0 + jax.scipy.special.erf(xi / jnp.sqrt(2.0)))
        return self._lo + (self._hi - self._lo) * phi_xi

    def standardize(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Uniform(lo, hi) → ξ via inverse Gaussian CDF.

        Inverts the unstandardize map: given θ ∈ (lo, hi), returns ξ such that
        θ = lo + (hi-lo) * Φ(ξ). Uses the inverse error function erfinv.

        Parameters
        ----------
        theta : float or array_like
            Physical-space parameter value in [lo, hi].

        Returns
        -------
        float or ndarray
            Standardized latent-space value.
        """
        # Normalize to [0, 1]
        p = (theta - self._lo) / (self._hi - self._lo)
        # Clip to avoid numerical issues at the extremes
        p = jnp.clip(p, 1e-7, 1.0 - 1e-7)
        # Inverse of Φ: Φ^{-1}(p) = sqrt(2) * erfinv(2p - 1)
        return jnp.sqrt(2.0) * jax.scipy.special.erfinv(2.0 * p - 1.0)

    def __repr__(self) -> str:
        return f"Uniform({self._lo}, {self._hi})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Uniform) and self._lo == other._lo and self._hi == other._hi


class Gaussian(Distribution):
    """Gaussian (normal) prior, optionally clipped to [lo, hi].

    A bell-curve probability density centered at μ with standard deviation σ.
    Useful when prior information suggests a most-probable value with uncertainty.
    Optional bounds allow truncation to physical ranges.

    Parameters
    ----------
    mu : float
        Mean of the Gaussian distribution.
    sigma : float
        Standard deviation. Must be positive.
    lo : float, optional
        Lower truncation bound. Default: -∞ (no lower truncation).
    hi : float, optional
        Upper truncation bound. Default: +∞ (no upper truncation).

    Attributes
    ----------
    mu : float
        Mean of the distribution.
    sigma : float
        Standard deviation of the distribution.
    lo : float
        Lower truncation bound (``-inf`` if unbounded).
    hi : float
        Upper truncation bound (``+inf`` if unbounded).
    bounds : tuple[float, float]
        ``(lo, hi)`` convenience tuple.

    Raises
    ------
    ValueError
        If sigma <= 0 or lo >= hi.

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    **Standardization**: Unbounded case maps ξ ~ N(0,1) to θ = μ + σ·ξ.
    With finite bounds, uses the truncated-normal inverse CDF:

    .. math::

        \\theta = \\mu + \\sigma \\cdot \\Phi^{-1}\\left[\\Phi(\\alpha) +
        \\Phi(\\xi)\\,(\\Phi(\\beta) - \\Phi(\\alpha))\\right]

    with α = (lo−μ)/σ, β = (hi−μ)/σ — the exact inverse-CDF pushforward
    (Knollmüller & Enßlin 2019, arXiv:1901.11033, Eqs. 18–25). Clipping
    would instead censor: point-masses at the bounds and zero gradient
    outside them.

    **Normalization**: When lo, hi are finite, the density is normalized over
    [lo, hi] (``log_prob`` includes the −log[Φ(β)−Φ(α)] mass term), not over
    the full real line. Use this for physically bounded quantities.

    Examples
    --------
    >>> import jax.random
    >>> from tengri import Gaussian
    >>> prior = Gaussian(mu=-0.3, sigma=0.2)  # metallicity
    >>> key = jax.random.PRNGKey(0)
    >>> sample = prior.sample(key)
    >>> print(f"Sample: {sample:.3f}")  # Typically near -0.3
    >>> log_prob = prior.log_prob(-0.3)
    >>> print(f"log p(μ): {log_prob:.4f}")  # Maximum at mean
    """

    def __init__(
        self,
        mu: float,
        sigma: float,
        lo: float = float("-inf"),
        hi: float = float("inf"),
        description: str = "",
        *,
        units: str = "",
        default: float | None = None,
    ):
        if sigma <= 0:
            raise ValueError(f"Gaussian requires sigma > 0, got {sigma}")
        if lo >= hi:
            raise ValueError(f"Gaussian requires lo < hi, got lo={lo}, hi={hi}")
        self._mu = float(mu)
        self._sigma = float(sigma)
        self._lo = float(lo)
        self._hi = float(hi)
        # Truncation constants (Python floats, fixed at construction):
        # Φ((lo−μ)/σ), Φ((hi−μ)/σ) and the mass Z = Φ(β) − Φ(α) in [lo, hi].
        self._cdf_lo = _norm_cdf_float((self._lo - self._mu) / self._sigma)
        self._cdf_hi = _norm_cdf_float((self._hi - self._mu) / self._sigma)
        self._truncated = self._cdf_lo > 0.0 or self._cdf_hi < 1.0
        self.description = description
        self.units = units
        self._register_default(default)

    @property
    def mu(self) -> float:
        """Mean of the Gaussian distribution.

        Returns
        -------
        float
            Mean value.
        """
        return self._mu

    @property
    def sigma(self) -> float:
        """Standard deviation of the Gaussian distribution.

        Returns
        -------
        float
            Standard deviation value.
        """
        return self._sigma

    @property
    def lo(self) -> float:
        """Lower truncation bound.

        Returns
        -------
        float
            Lower truncation bound (-inf if unbounded).
        """
        return self._lo

    @property
    def hi(self) -> float:
        """Upper truncation bound.

        Returns
        -------
        float
            Upper truncation bound (+inf if unbounded).
        """
        return self._hi

    @property
    def bounds(self) -> tuple[float, float]:
        """Lower and upper truncation bounds [lo, hi].

        Returns
        -------
        tuple[float, float]
            Bounds as (lo, hi) tuple.
        """
        return (self._lo, self._hi)

    def sample(self, key: jax.Array) -> jnp.ndarray:
        """Draw a random sample from the Gaussian distribution.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key for random sampling.

        Returns
        -------
        ndarray
            A single sample from N(mu, sigma²) truncated to [lo, hi].
        """
        # Inverse-CDF sampling through the standardization pushforward —
        # one source of truth with unstandardize(), exactly truncated
        # (no clip point-masses at the bounds).
        return self.unstandardize(jax.random.normal(key))

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate log probability density, returning -inf outside bounds.

        Parameters
        ----------
        x : float or array_like
            Parameter value in physical space.

        Returns
        -------
        float
            Log probability density at x.
        """
        # Normalized truncated-normal density: the constants matter for
        # nested-sampling evidence, which compares priors across models.
        z_mass = self._cdf_hi - self._cdf_lo
        lp = (
            -0.5 * ((x - self._mu) / self._sigma) ** 2
            - math.log(self._sigma)
            - 0.5 * math.log(2.0 * math.pi)
            - math.log(z_mass)
        )
        in_bounds = (x >= self._lo) & (x <= self._hi)
        return jnp.where(in_bounds, lp, -jnp.inf)

    def unstandardize(self, xi: jnp.ndarray) -> jnp.ndarray:
        """ξ ~ N(0,1) → N(μ,σ²) exactly truncated to [lo, hi].

        Unbounded case is the identity-affine map θ = μ + σξ. With finite
        bounds, uses the truncated-normal inverse CDF
        θ = μ + σ·Φ⁻¹(Φ(α) + Φ(ξ)·(Φ(β) − Φ(α))) — a smooth bijection onto
        (lo, hi), unlike clipping, which piles point-masses at the bounds
        and zeroes the gradient outside them.

        Parameters
        ----------
        xi : float or array_like
            Standardized latent value from standard normal distribution.

        Returns
        -------
        float or ndarray
            Physical-space parameter in [lo, hi].
        """
        if not self._truncated:
            return self._mu + self._sigma * xi
        phi_xi = 0.5 * (1.0 + jax.scipy.special.erf(xi / jnp.sqrt(2.0)))
        p = self._cdf_lo + phi_xi * (self._cdf_hi - self._cdf_lo)
        p = jnp.clip(p, _P_EPS, 1.0 - _P_EPS)
        return self._mu + self._sigma * jax.scipy.special.ndtri(p)

    def standardize(self, theta: jnp.ndarray) -> jnp.ndarray:
        """N(μ,σ²) → ξ (inverse of the truncation-aware map).

        Parameters
        ----------
        theta : float or array_like
            Physical-space parameter value.

        Returns
        -------
        float or ndarray
            Standardized latent-space value.
        """
        if not self._truncated:
            return (theta - self._mu) / self._sigma
        p = 0.5 * (1.0 + jax.scipy.special.erf((theta - self._mu) / (self._sigma * jnp.sqrt(2.0))))
        u = (p - self._cdf_lo) / (self._cdf_hi - self._cdf_lo)
        u = jnp.clip(u, _P_EPS, 1.0 - _P_EPS)
        return jnp.sqrt(2.0) * jax.scipy.special.erfinv(2.0 * u - 1.0)

    def __repr__(self) -> str:
        parts = [f"mu={self._mu}", f"sigma={self._sigma}"]
        if self._lo != float("-inf"):
            parts.append(f"lo={self._lo}")
        if self._hi != float("inf"):
            parts.append(f"hi={self._hi}")
        return f"Gaussian({', '.join(parts)})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Gaussian)
            and self._mu == other._mu
            and self._sigma == other._sigma
            and self._lo == other._lo
            and self._hi == other._hi
        )


class LogUniform(Distribution):
    """Uniform in log10 space on [lo, hi].

    A prior that places equal probability in logarithmic intervals, resulting
    in power-law density in linear space. Useful for quantities with logarithmic
    uncertainties, such as star formation rates, timescales, and luminosities.

    Parameters
    ----------
    lo : float
        Lower bound. Must be strictly positive.
    hi : float
        Upper bound. Must be greater than lo.
    description : str, optional
        Human-readable summary of the quantity, surfaced by
        ``describe_parameter`` and ``spec.summary()``.
    units : str, optional
        Physical units, e.g. ``"erg/s"``. Empty for dimensionless quantities.

    Attributes
    ----------
    lo : float
        Lower bound of the distribution.
    hi : float
        Upper bound of the distribution.
    bounds : tuple[float, float]
        ``(lo, hi)`` convenience tuple.

    Raises
    ------
    ValueError
        If lo <= 0 or lo >= hi.

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    The probability density in linear space is:

    .. math::

        p(x) = \\frac{1}{x \\cdot \\ln(10) \\cdot \\log_{10}(hi / lo)}

    where x ∈ [lo, hi]. Samples drawn from this distribution are equally
    spaced in log10 space: log10(x) ~ U(log10(lo), log10(hi)).

    **Standardization**: Maps ξ ~ N(0,1) to θ via the Gaussian CDF in log
    space (the exact inverse-CDF pushforward required by the standardized
    parameterization; Knollmüller & Enßlin 2019 [1]_, Eqs. 18–25):

    .. math::

        \\theta = \\exp\\left[\\ln(lo) + (\\ln(hi) - \\ln(lo))
        \\cdot \\Phi(\\xi)\\right]

    where Φ(ξ) = 0.5 * (1 + erf(ξ / sqrt(2))) is the standard normal CDF.
    This ensures an N(0,1) latent yields a genuine log-uniform prior on
    [lo, hi] — matching ``log_prob`` and ``sample`` — not a midpoint-peaked
    one. (A sigmoid map here would silently substitute a logit-normal
    prior in log space, biasing weakly-constrained scale parameters toward
    the geometric midpoint and compressing the tails.)

    References
    ----------
    .. [1] Knollmüller, J. & Enßlin, T. A., "Metric Gaussian Variational
       Inference", arXiv:1901.11033.

    Examples
    --------
    >>> import jax.random
    >>> from tengri import LogUniform
    >>> prior = LogUniform(1e-2, 1e2)  # ~4 orders of magnitude
    >>> key = jax.random.PRNGKey(0)
    >>> sample = prior.sample(key)
    >>> print(f"Sample: {sample:.3e}")
    >>> log_prob = prior.log_prob(1.0)  # Center of log space
    >>> print(f"log p(1.0): {log_prob:.4f}")
    """

    def __init__(
        self,
        lo: float,
        hi: float,
        description: str = "",
        *,
        units: str = "",
        default: float | None = None,
    ):
        if lo <= 0:
            raise ValueError(f"LogUniform requires lo > 0, got {lo}")
        if lo >= hi:
            raise ValueError(f"LogUniform requires lo < hi, got lo={lo}, hi={hi}")
        self._lo = float(lo)
        self._hi = float(hi)
        self.description = description
        self.units = units
        self._register_default(default)

    @property
    def lo(self) -> float:
        """Lower bound of the log-uniform distribution.

        Returns
        -------
        float
            Lower bound value.
        """
        return self._lo

    @property
    def hi(self) -> float:
        """Upper bound of the log-uniform distribution.

        Returns
        -------
        float
            Upper bound value.
        """
        return self._hi

    @property
    def bounds(self) -> tuple[float, float]:
        """Lower and upper bounds [lo, hi].

        Returns
        -------
        tuple[float, float]
            Bounds as (lo, hi) tuple.
        """
        return (self._lo, self._hi)

    def sample(self, key: jax.Array) -> jnp.ndarray:
        """Draw one sample log-uniformly from [lo, hi].

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key for random sampling.

        Returns
        -------
        ndarray
            A single sample log-uniformly distributed in [lo, hi].
        """
        log_lo = jnp.log10(self._lo)
        log_hi = jnp.log10(self._hi)
        log_val = jax.random.uniform(key, minval=log_lo, maxval=log_hi)
        return 10.0**log_val

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return log probability: -log(x * log(hi/lo)) inside bounds, -inf outside.

        Parameters
        ----------
        x : float or array_like
            Parameter value in physical space.

        Returns
        -------
        float
            Log probability density at x.
        """
        in_bounds = (x >= self._lo) & (x <= self._hi)
        lp = -jnp.log(x * jnp.log(self._hi / self._lo))
        return jnp.where(in_bounds, lp, -jnp.inf)

    def unstandardize(self, xi: jnp.ndarray) -> jnp.ndarray:
        """ξ ~ N(0,1) → LogUniform(lo, hi) via Gaussian CDF in log space.

        Uses the standard normal CDF Φ(ξ) = 0.5 * (1 + erf(ξ / sqrt(2))) so
        that an N(0,1) latent yields a genuine log-uniform prior — the exact
        inverse-CDF pushforward θ = F⁻¹(Φ(ξ)) required for the ½ξᵀξ prior
        term in the standardized information Hamiltonian.

        Parameters
        ----------
        xi : float or array_like
            Standardized latent value from standard normal distribution.

        Returns
        -------
        float or ndarray
            Physical-space parameter in [lo, hi].
        """
        log_lo = jnp.log(self._lo)
        log_hi = jnp.log(self._hi)
        # Standard normal CDF: Φ(ξ) = 0.5 * (1 + erf(ξ / sqrt(2)))
        phi_xi = 0.5 * (1.0 + jax.scipy.special.erf(xi / jnp.sqrt(2.0)))
        return jnp.exp(log_lo + (log_hi - log_lo) * phi_xi)

    def standardize(self, theta: jnp.ndarray) -> jnp.ndarray:
        """LogUniform(lo, hi) → ξ via inverse Gaussian CDF in log space.

        Inverts the unstandardize map: given θ ∈ (lo, hi), returns ξ such
        that θ = exp(ln lo + (ln hi − ln lo) · Φ(ξ)).

        Parameters
        ----------
        theta : float or array_like
            Physical-space parameter value in [lo, hi].

        Returns
        -------
        float or ndarray
            Standardized latent-space value.
        """
        log_lo = jnp.log(self._lo)
        log_hi = jnp.log(self._hi)
        p = (jnp.log(theta) - log_lo) / (log_hi - log_lo)
        # Clip to avoid numerical issues at the extremes
        p = jnp.clip(p, 1e-7, 1.0 - 1e-7)
        # Inverse of Φ: Φ^{-1}(p) = sqrt(2) * erfinv(2p - 1)
        return jnp.sqrt(2.0) * jax.scipy.special.erfinv(2.0 * p - 1.0)

    def __repr__(self) -> str:
        return f"LogUniform({self._lo}, {self._hi})"

    def __eq__(self, other) -> bool:
        return isinstance(other, LogUniform) and self._lo == other._lo and self._hi == other._hi


class LogNormal(Distribution):
    """Log-normal prior: log(θ) ~ N(μ, σ²).

    A prior suitable for positive-definite quantities with multiplicative
    uncertainty, such as timescales, amplitudes, and scale factors. The log
    of the parameter is normally distributed.

    Parameters
    ----------
    mu : float, optional
        Mean of log(θ). Default: 0.0.
    sigma : float, optional
        Standard deviation of log(θ). Must be positive. Default: 1.0.
    lo : float, optional
        Lower truncation bound. Default: 0.0 (ensures θ > 0).
    hi : float, optional
        Upper truncation bound. Default: +∞ (no upper truncation).
    description : str, optional
        Human-readable summary of the quantity, surfaced by
        ``describe_parameter`` and ``spec.summary()``.
    units : str, optional
        Physical units, e.g. ``"erg/s"``. Empty for dimensionless quantities.

    Attributes
    ----------
    mu : float
        Mean of log(theta).
    sigma : float
        Standard deviation of log(theta).
    bounds : tuple[float, float]
        ``(lo, hi)`` convenience tuple.

    Raises
    ------
    ValueError
        If sigma <= 0.

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    The probability density in linear space is:

    .. math::

        p(\\theta) = \\frac{1}{\\theta \\sigma \\sqrt{2\\pi}} \\exp\\left(
          -\\frac{(\\ln \\theta - \\mu)^2}{2\\sigma^2}
        \\right)

    for θ ∈ [lo, hi]. When truncated, the density is renormalized over the
    interval.

    **Standardization**: Maps ξ ~ N(0,1) to θ via:

    .. math::

        \\theta = \\text{clip}(\\exp(\\mu + \\sigma \\cdot \\xi), lo, hi)

    Examples
    --------
    >>> import jax.random
    >>> from tengri import LogNormal
    >>> # PSD timescale: log(tau_yr) centered at 8, width 0.5 dex
    >>> prior = LogNormal(mu=8, sigma=0.5)
    >>> key = jax.random.PRNGKey(0)
    >>> sample = prior.sample(key)
    >>> print(f"Sample (yr): {sample:.3e}")
    >>> log_prob = prior.log_prob(1e8)
    >>> print(f"log p(1e8): {log_prob:.4f}")
    """

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 1.0,
        lo: float = 0.0,
        hi: float = float("inf"),
        description: str = "",
        *,
        units: str = "",
        default: float | None = None,
    ):
        if sigma <= 0:
            raise ValueError(f"LogNormal requires sigma > 0, got {sigma}")
        self._mu = float(mu)
        self._sigma = float(sigma)
        self._lo = float(lo)
        self._hi = float(hi)
        self.description = description
        self.units = units
        # Truncation constants in log space (lo = 0 → log lo = −inf → Φ = 0).
        log_lo = math.log(self._lo) if self._lo > 0 else float("-inf")
        log_hi = math.log(self._hi) if math.isfinite(self._hi) else float("inf")
        self._cdf_lo = _norm_cdf_float((log_lo - self._mu) / self._sigma)
        self._cdf_hi = _norm_cdf_float((log_hi - self._mu) / self._sigma)
        self._truncated = self._cdf_lo > 0.0 or self._cdf_hi < 1.0
        self._register_default(default)

    @property
    def mu(self) -> float:
        """Mean of log(theta).

        Returns
        -------
        float
            Mean of the logarithm of the parameter.
        """
        return self._mu

    @property
    def sigma(self) -> float:
        """Standard deviation of log(theta).

        Returns
        -------
        float
            Standard deviation of the logarithm of the parameter.
        """
        return self._sigma

    @property
    def bounds(self) -> tuple[float, float]:
        """Lower and upper truncation bounds [lo, hi].

        Returns
        -------
        tuple[float, float]
            Bounds as (lo, hi) tuple.
        """
        return (self._lo, self._hi)

    def sample(self, key: jax.Array) -> jnp.ndarray:
        """Draw a random sample from the log-normal distribution.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key for random sampling.

        Returns
        -------
        ndarray
            A single sample from LogNormal(mu, sigma²) truncated to [lo, hi].
        """
        # Inverse-CDF sampling through the standardization pushforward —
        # one source of truth with unstandardize(), exactly truncated.
        return self.unstandardize(jax.random.normal(key))

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate log probability density, returning -inf outside bounds.

        Normalized over [lo, hi] (the constants matter for nested-sampling
        evidence, which compares priors across models).

        Parameters
        ----------
        x : float or array_like
            Parameter value in physical space.

        Returns
        -------
        float
            Log probability density at x.
        """
        z_mass = self._cdf_hi - self._cdf_lo
        lp = (
            -jnp.log(x)
            - 0.5 * ((jnp.log(x) - self._mu) / self._sigma) ** 2
            - math.log(self._sigma)
            - 0.5 * math.log(2.0 * math.pi)
            - math.log(z_mass)
        )
        in_bounds = (x >= self._lo) & (x <= self._hi)
        return jnp.where(in_bounds, lp, -jnp.inf)

    def unstandardize(self, xi: jnp.ndarray) -> jnp.ndarray:
        """ξ ~ N(0,1) → LogNormal(μ, σ²) exactly truncated to [lo, hi].

        Unbounded case is θ = exp(μ + σξ). With finite bounds, applies the
        truncated-normal inverse CDF in log space (see ``Gaussian`` for the
        formula) — a smooth bijection onto (lo, hi) instead of clipping.

        Parameters
        ----------
        xi : float or array_like
            Standardized latent value from standard normal distribution.

        Returns
        -------
        float or ndarray
            Physical-space parameter in [lo, hi].
        """
        if not self._truncated:
            return jnp.exp(self._mu + self._sigma * xi)
        phi_xi = 0.5 * (1.0 + jax.scipy.special.erf(xi / jnp.sqrt(2.0)))
        p = self._cdf_lo + phi_xi * (self._cdf_hi - self._cdf_lo)
        p = jnp.clip(p, _P_EPS, 1.0 - _P_EPS)
        return jnp.exp(self._mu + self._sigma * jax.scipy.special.ndtri(p))

    def standardize(self, theta: jnp.ndarray) -> jnp.ndarray:
        """LogNormal → ξ (inverse of the truncation-aware map).

        Parameters
        ----------
        theta : float or array_like
            Physical-space parameter value.

        Returns
        -------
        float or ndarray
            Standardized latent-space value.
        """
        z = (jnp.log(jnp.maximum(theta, 1e-30)) - self._mu) / self._sigma
        if not self._truncated:
            return z
        p = 0.5 * (1.0 + jax.scipy.special.erf(z / jnp.sqrt(2.0)))
        u = (p - self._cdf_lo) / (self._cdf_hi - self._cdf_lo)
        u = jnp.clip(u, _P_EPS, 1.0 - _P_EPS)
        return jnp.sqrt(2.0) * jax.scipy.special.erfinv(2.0 * u - 1.0)

    def __repr__(self) -> str:
        parts = [f"mu={self._mu}", f"sigma={self._sigma}"]
        if self._lo > 0:
            parts.append(f"lo={self._lo}")
        if self._hi < float("inf"):
            parts.append(f"hi={self._hi}")
        return f"LogNormal({', '.join(parts)})"

    def __eq__(self, other) -> bool:
        # Truncation is part of the prior's identity (#1292). Comparing only
        # (mu, sigma) reported LogNormal(0, 1, hi=10) == LogNormal(0, 1, hi=1e9)
        # -- and prior equality is what the builder-vs-dict and to_groups
        # round-trip contracts use to prove two construction paths agree, so a
        # blind spot here makes those tests vacuously green.
        return (
            isinstance(other, LogNormal)
            and self._mu == other._mu
            and self._sigma == other._sigma
            and self._lo == other._lo
            and self._hi == other._hi
        )


class StudentT(Distribution):
    """Student's t prior with heavier tails than Gaussian.

    A robust prior with longer tails, useful for parameters that may exhibit
    outlier-like behavior. Commonly used in BAGPIPES-style SED fitting for
    down-weighting extreme values while remaining flexible.

    Parameters
    ----------
    mu : float, optional
        Location (center) of the distribution. Default: 0.0.
    sigma : float, optional
        Scale parameter. Must be positive. Default: 1.0.
    df : float, optional
        Degrees of freedom. Controls tail weight:

        - df → ∞ gives Gaussian (heaviest concentration at center)
        - df = 3 gives a moderately heavy-tailed prior
        - df = 1 gives Cauchy (extremely heavy tails)

        Default: 3.0.
    lo : float, optional
        Lower truncation bound. Default: -∞ (no lower truncation).
    hi : float, optional
        Upper truncation bound. Default: +∞ (no upper truncation).
    description : str, optional
        Human-readable summary of the quantity, surfaced by
        ``describe_parameter`` and ``spec.summary()``.
    units : str, optional
        Physical units, e.g. ``"erg/s"``. Empty for dimensionless quantities.

    Attributes
    ----------
    bounds : tuple[float, float]
        ``(lo, hi)`` truncation bounds.

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    The probability density follows a Student's t distribution with the
    standard normalization. For finite df, it has heavier tails than a
    Gaussian.

    **Standardization**: Exact quantile pushforward
    θ = μ + σ·F⁻¹_t,df(Φ(ξ)) (Knollmüller & Enßlin 2019, arXiv:1901.11033,
    Eqs. 18–25). Closed forms for df = 1 (Cauchy: tan(π(p−½))) and df = 2
    ((2p−1)/√(2p(1−p))); other df use a monotone quantile table built at
    construction from the incomplete-beta CDF and interpolated with
    ``jnp.interp`` (the NIFTy interpolation-operator pattern). Finite
    truncation bounds are applied in CDF space, giving a smooth bijection
    onto (lo, hi).

    Heavy tails are preserved exactly — the previous variance-matched
    Gaussian approximation silently discarded them, which matters for the
    Leja et al. (2019) [1]_ continuity-SFH log-SFR-ratio priors
    (StudentT(0, 0.3, df=2)) whose tails control burst flexibility.

    References
    ----------
    .. [1] Leja, J., et al. 2019, ApJ, 876, 3, "How to Measure Galaxy Star
       Formation Histories. II.", arXiv:1811.03637, doi:10.3847/1538-4357/ab133c

    Examples
    --------
    >>> import jax.random
    >>> from tengri import StudentT
    >>> prior = StudentT(mu=0, sigma=1, df=3)  # Robust prior
    >>> key = jax.random.PRNGKey(0)
    >>> sample = prior.sample(key)
    >>> print(f"Sample: {sample:.4f}")
    """

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 1.0,
        df: float = 3.0,
        lo: float = float("-inf"),
        hi: float = float("inf"),
        description: str = "",
        *,
        units: str = "",
        default: float | None = None,
    ):
        self._mu = float(mu)
        self._sigma = float(sigma)
        self._df = float(df)
        self._lo = float(lo)
        self._hi = float(hi)
        self.description = description
        self.units = units
        # Quantile machinery: closed forms for df ∈ {1, 2}; otherwise a
        # monotone (F, z) table for interpolation, built once here (the
        # NIFTy interpolation-operator pattern — no scipy dependency).
        if self._df not in (1.0, 2.0):
            self._cdf_grid, self._z_grid = self._build_cdf_table(self._df)
        else:
            self._cdf_grid = self._z_grid = None
        # Truncation constants in t-CDF space (Python floats).
        self._pcdf_lo = self._t_cdf_float((self._lo - self._mu) / self._sigma)
        self._pcdf_hi = self._t_cdf_float((self._hi - self._mu) / self._sigma)
        self._truncated = self._pcdf_lo > 0.0 or self._pcdf_hi < 1.0
        # Normalization: ln Γ((ν+1)/2) − ln Γ(ν/2) − ½ln(νπ) − ln σ − ln Z.
        self._log_norm = (
            math.lgamma((self._df + 1.0) / 2.0)
            - math.lgamma(self._df / 2.0)
            - 0.5 * math.log(self._df * math.pi)
            - math.log(self._sigma)
            - math.log(self._pcdf_hi - self._pcdf_lo)
        )
        self._register_default(default)

    @staticmethod
    def _build_cdf_table(df: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Monotone (CDF, z) table for the standard t distribution.

        z on a tan-spaced grid reaching the extreme quantiles heavy tails
        need; CDF via the regularized incomplete beta function
        F(z) = 1 − ½ I_x(ν/2, ½), x = ν/(ν+z²) for z ≥ 0 (symmetric below).
        """
        u = np.linspace(-0.5 * np.pi + 1e-4, 0.5 * np.pi - 1e-4, 4097)
        z = np.tan(u) * max(1.0, np.sqrt(df))
        x = df / (df + z**2)
        upper_half = np.asarray(
            1.0 - 0.5 * jax.scipy.special.betainc(df / 2.0, 0.5, jnp.asarray(x))
        )
        cdf = np.where(z >= 0, upper_half, 1.0 - upper_half)
        cdf = np.maximum.accumulate(cdf)
        return jnp.asarray(cdf), jnp.asarray(z)

    def _t_cdf_float(self, z: float) -> float:
        """Standard-t CDF at a Python float; handles ±inf exactly."""
        if math.isinf(z):
            return 0.0 if z < 0 else 1.0
        if self._df == 1.0:
            return 0.5 + math.atan(z) / math.pi
        if self._df == 2.0:
            return 0.5 * (1.0 + z / math.sqrt(2.0 + z * z))
        return float(jnp.interp(z, self._z_grid, self._cdf_grid))

    def _t_quantile(self, p: jnp.ndarray) -> jnp.ndarray:
        """Standard-t quantile F⁻¹(p): closed form for df ∈ {1, 2}, else table.

        The df∉{1,2} branch is piecewise-linear (``jnp.interp``), so its
        gradient is discontinuous at the 4097 knots — fine for MAP/NUTS in
        practice, but the only df used in-repo are 1 and 2 (both closed-form
        above), so this branch is currently never exercised. Swap to a
        monotone-cubic interpolation if a fittable df∉{1,2} is introduced.
        """
        if self._df == 1.0:
            return jnp.tan(jnp.pi * (p - 0.5))
        if self._df == 2.0:
            return (2.0 * p - 1.0) / jnp.sqrt(2.0 * p * (1.0 - p))
        return jnp.interp(p, self._cdf_grid, self._z_grid)

    @property
    def bounds(self) -> tuple[float, float]:
        """Lower and upper truncation bounds [lo, hi].

        Returns
        -------
        tuple[float, float]
            Bounds as (lo, hi) tuple.
        """
        return (self._lo, self._hi)

    def sample(self, key: jax.Array) -> jnp.ndarray:
        """Draw a random sample from the Student's t distribution.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key for random sampling.

        Returns
        -------
        ndarray
            A single sample from Student's t distribution truncated to [lo, hi].
        """
        # Inverse-CDF sampling through the standardization pushforward —
        # one source of truth with unstandardize(), exactly truncated.
        return self.unstandardize(jax.random.normal(key))

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate log probability density, returning -inf outside bounds.

        Parameters
        ----------
        x : float or array_like
            Parameter value in physical space.

        Returns
        -------
        float
            Log probability density at x.
        """
        # Normalized (truncated) Student-t density — constants matter for
        # nested-sampling evidence.
        z = (x - self._mu) / self._sigma
        lp = -0.5 * (self._df + 1) * jnp.log(1 + z**2 / self._df) + self._log_norm
        in_bounds = (x >= self._lo) & (x <= self._hi)
        return jnp.where(in_bounds, lp, -jnp.inf)

    def unstandardize(self, xi: jnp.ndarray) -> jnp.ndarray:
        """ξ ~ N(0,1) → t-distributed via the exact quantile pushforward.

        θ = μ + σ·F⁻¹_t,df(p) with p = Φ(ξ) mapped through the truncation
        bounds in CDF space. Closed-form quantiles for df ∈ {1, 2}
        (df = 2 is the Leja+2019 / Tacchella+2022 continuity-SFH ratio
        prior); other df interpolate the construction-time CDF table.

        Parameters
        ----------
        xi : float or array_like
            Standardized latent value from standard normal distribution.

        Returns
        -------
        float or ndarray
            Physical-space parameter in [lo, hi].
        """
        phi_xi = 0.5 * (1.0 + jax.scipy.special.erf(xi / jnp.sqrt(2.0)))
        p = self._pcdf_lo + phi_xi * (self._pcdf_hi - self._pcdf_lo)
        p = jnp.clip(p, _P_EPS, 1.0 - _P_EPS)
        return self._mu + self._sigma * self._t_quantile(p)

    def standardize(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Physical value → ξ (inverse of the exact quantile pushforward).

        Parameters
        ----------
        theta : float or array_like
            Physical-space parameter value.

        Returns
        -------
        float or ndarray
            Standardized latent-space value.
        """
        z = (theta - self._mu) / self._sigma
        if self._df == 1.0:
            p = 0.5 + jnp.arctan(z) / jnp.pi
        elif self._df == 2.0:
            p = 0.5 * (1.0 + z / jnp.sqrt(2.0 + z * z))
        else:
            p = jnp.interp(z, self._z_grid, self._cdf_grid)
        u = (p - self._pcdf_lo) / (self._pcdf_hi - self._pcdf_lo)
        u = jnp.clip(u, _P_EPS, 1.0 - _P_EPS)
        return jnp.sqrt(2.0) * jax.scipy.special.erfinv(2.0 * u - 1.0)

    def __repr__(self) -> str:
        return f"StudentT(mu={self._mu}, sigma={self._sigma}, df={self._df})"

    def __eq__(self, other) -> bool:
        # StudentT was the only one of the seven distributions without an
        # __eq__ (#1292), so it fell back to identity: two Student-t priors
        # built from the same numbers compared unequal, and Student-t could
        # never take part in the construction-path equivalence contracts.
        return (
            isinstance(other, StudentT)
            and self._mu == other._mu
            and self._sigma == other._sigma
            and self._df == other._df
            and self._lo == other._lo
            and self._hi == other._hi
        )


def _laplace_cdf_float(x: float, mu: float, b: float) -> float:
    """Laplace CDF at a Python float; handles ±inf exactly."""
    if math.isinf(x):
        return 0.0 if x < 0 else 1.0
    z = (x - mu) / b
    return 0.5 * math.exp(z) if z < 0 else 1.0 - 0.5 * math.exp(-z)


class Laplace(Distribution):
    r"""Laplace (double-exponential) prior — a sparsity/robustness prior.

    Heavier-tailed than a Gaussian and peaked at the location, the Laplace
    prior is the continuous analog of an L1 penalty (LASSO): it pulls weakly
    constrained parameters toward ``mu`` while tolerating occasional large
    excursions. Useful for coefficients expected to be near a default with a
    few genuine departures (e.g. per-band calibration offsets, sparse
    additive components).

    Parameters
    ----------
    mu : float
        Location (median and mode).
    b : float
        Scale (diversity). Must be positive; variance is ``2 b^2``.
    lo : float, optional
        Lower truncation bound. Default: ``-inf``.
    hi : float, optional
        Upper truncation bound. Default: ``+inf``.
    description : str, optional
        Human-readable summary of the quantity, surfaced by
        ``describe_parameter`` and ``spec.summary()``.
    units : str, optional
        Physical units, e.g. ``"erg/s"``. Empty for dimensionless quantities.

    Attributes
    ----------
    bounds : tuple[float, float]
        ``(lo, hi)`` truncation bounds.

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    The density is

    .. math::

        p(x) = \\frac{1}{2b}\\exp\\!\\left(-\\frac{|x-\\mu|}{b}\\right).

    **Standardization**: exact inverse-CDF pushforward
    :math:`\\theta = \\mu - b\\,\\mathrm{sgn}(p-\\tfrac12)\\,
    \\ln(1-2|p-\\tfrac12|)` with :math:`p = \\Phi(\\xi)` mapped through the
    truncation bounds in CDF space (Knollmüller & Enßlin 2019 [1]_,
    Eqs. 18-25). The quantile is closed-form, so no interpolation table is
    needed.

    Examples
    --------
    >>> import jax.random
    >>> from tengri import Laplace
    >>> prior = Laplace(mu=0.0, b=0.1)  # sparse calibration offset
    >>> sample = prior.sample(jax.random.PRNGKey(0))

    References
    ----------
    .. [1] Knollmüller, J. & Enßlin, T. A., "Metric Gaussian Variational
       Inference", arXiv:1901.11033.
    """

    def __init__(
        self,
        mu: float = 0.0,
        b: float = 1.0,
        lo: float = float("-inf"),
        hi: float = float("inf"),
        description: str = "",
        *,
        units: str = "",
        default: float | None = None,
    ):
        if b <= 0:
            raise ValueError(f"Laplace requires b > 0, got {b}")
        if lo >= hi:
            raise ValueError(f"Laplace requires lo < hi, got lo={lo}, hi={hi}")
        self._mu = float(mu)
        self._b = float(b)
        self._lo = float(lo)
        self._hi = float(hi)
        self.description = description
        self.units = units
        self._cdf_lo = _laplace_cdf_float(self._lo, self._mu, self._b)
        self._cdf_hi = _laplace_cdf_float(self._hi, self._mu, self._b)
        self._truncated = self._cdf_lo > 0.0 or self._cdf_hi < 1.0
        self._register_default(default)

    @property
    def bounds(self) -> tuple[float, float]:
        """Lower and upper truncation bounds [lo, hi].

        Returns
        -------
        tuple[float, float]
            Bounds as (lo, hi) tuple.
        """
        return (self._lo, self._hi)

    def sample(self, key: jax.Array) -> jnp.ndarray:
        """Draw one sample from the (truncated) Laplace via inverse-CDF.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key for random sampling.

        Returns
        -------
        ndarray
            A single sample from Laplace(mu, b) truncated to [lo, hi].
        """
        # One source of truth with unstandardize(); exactly truncated.
        return self.unstandardize(jax.random.normal(key))

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return the normalized (truncated) Laplace log-density, -inf outside.

        Parameters
        ----------
        x : float or array_like
            Parameter value in physical space.

        Returns
        -------
        float
            Log probability density at x.
        """
        z_mass = self._cdf_hi - self._cdf_lo
        lp = -jnp.abs(x - self._mu) / self._b - math.log(2.0 * self._b) - math.log(z_mass)
        in_bounds = (x >= self._lo) & (x <= self._hi)
        return jnp.where(in_bounds, lp, -jnp.inf)

    def _quantile(self, p: jnp.ndarray) -> jnp.ndarray:
        """Standard Laplace(mu, b) quantile F^{-1}(p), closed form."""
        q = p - 0.5
        return self._mu - self._b * jnp.sign(q) * jnp.log1p(-2.0 * jnp.abs(q))

    def unstandardize(self, xi: jnp.ndarray) -> jnp.ndarray:
        """ξ ~ N(0,1) → Laplace(mu, b) exactly truncated to [lo, hi].

        Parameters
        ----------
        xi : float or array_like
            Standardized latent value from standard normal distribution.

        Returns
        -------
        float or ndarray
            Physical-space parameter in [lo, hi].
        """
        phi_xi = 0.5 * (1.0 + jax.scipy.special.erf(xi / jnp.sqrt(2.0)))
        p = self._cdf_lo + phi_xi * (self._cdf_hi - self._cdf_lo)
        p = jnp.clip(p, _P_EPS, 1.0 - _P_EPS)
        return self._quantile(p)

    def standardize(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Laplace(mu, b) → ξ (inverse of the truncation-aware map).

        Parameters
        ----------
        theta : float or array_like
            Physical-space parameter value.

        Returns
        -------
        float or ndarray
            Standardized latent-space value.
        """
        z = (theta - self._mu) / self._b
        p = jnp.where(z < 0, 0.5 * jnp.exp(z), 1.0 - 0.5 * jnp.exp(-z))
        u = (p - self._cdf_lo) / (self._cdf_hi - self._cdf_lo)
        u = jnp.clip(u, _P_EPS, 1.0 - _P_EPS)
        return jnp.sqrt(2.0) * jax.scipy.special.erfinv(2.0 * u - 1.0)

    def __repr__(self) -> str:
        parts = [f"mu={self._mu}", f"b={self._b}"]
        if self._lo != float("-inf"):
            parts.append(f"lo={self._lo}")
        if self._hi != float("inf"):
            parts.append(f"hi={self._hi}")
        return f"Laplace({', '.join(parts)})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Laplace)
            and self._mu == other._mu
            and self._b == other._b
            and self._lo == other._lo
            and self._hi == other._hi
        )


class Fixed(Distribution):
    """Fixed (non-free) parameter with a constant value.

    Represents a parameter that is not sampled or inferred. Used for holding
    model settings constant during fitting, or for categorical parameters
    that don't vary. Fixed parameters contribute zero to the likelihood.

    Parameters
    ----------
    value : float, int, or str
        The fixed value. Can be numeric (for quantitative parameters) or
        string (for categorical choices, e.g. "solar" for shock abundance).

    Returns
    -------
    Fixed
        Fixed instance with the given value.

    Attributes
    ----------
    value : float or str
        The constant value returned by ``sample()`` and ``unstandardize()``.
    bounds : tuple[float, float]
        Always ``(-inf, +inf)`` — Fixed parameters have no bounds.

    Notes
    -----
    **JIT-compatible**: yes — ``unstandardize()`` returns the constant value
    regardless of the latent variable ξ.

    **Inference**: Fixed parameters are excluded from the inference set.
    They do not appear in the posterior and do not contribute to the loss
    or gradients.

    Examples
    --------
    >>> from tengri import Fixed
    >>> # Numerical fixed value
    >>> redshift = Fixed(0.1)
    >>> print(redshift.sample(None))
    0.1
    >>> # Categorical fixed value
    >>> shock_abundance = Fixed("solar")
    >>> print(shock_abundance.sample(None))
    solar
    """

    def __init__(self, value: float | str, description: str = "", *, units: str = ""):
        self._value = value if isinstance(value, str) else float(value)
        self.description = description
        self.units = units

    @property
    def value(self) -> float | str:
        """The fixed value (numeric or string).

        Returns
        -------
        float or str
            The constant fixed value.
        """
        return self._value

    @property
    def is_fixed(self) -> bool:
        """Return True — this is a fixed (non-free) parameter.

        Returns
        -------
        bool
            Always True for Fixed distributions.
        """
        return True

    @property
    def default(self) -> float | str:
        """Return the fixed value — for ``Fixed``, value and default coincide.

        Returns
        -------
        float or str
            The fixed value.
        """
        return self._value

    @property
    def bounds(self) -> tuple[float, float] | tuple[None, None]:
        """Return (value, value) for numeric, or (None, None) for string.

        Returns
        -------
        tuple[float, float] or tuple[None, None]
            For numeric values: (value, value); for string values: (None, None).
        """
        if isinstance(self._value, str):
            return (None, None)
        return (self._value, self._value)

    def sample(self, key: jax.Array) -> jnp.ndarray | str:
        """Return the fixed value (ignores random key).

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key (ignored for fixed parameters).

        Returns
        -------
        float, int, or str
            The constant fixed value.
        """
        if isinstance(self._value, str):
            return self._value
        return jnp.array(self._value)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return 0.0 (fixed parameters have zero log-likelihood contribution).

        Parameters
        ----------
        x : float or array_like
            Parameter value (ignored for fixed distributions).

        Returns
        -------
        float
            Always 0.0.
        """
        return jnp.array(0.0)

    def unstandardize(self, xi: jnp.ndarray) -> jnp.ndarray | str:
        """Fixed: always returns the fixed value (ignores ξ).

        Parameters
        ----------
        xi : float or array_like
            Standardized latent value (ignored for fixed parameters).

        Returns
        -------
        float, int, or str
            The constant fixed value.
        """
        if isinstance(self._value, str):
            return self._value
        return jnp.array(self._value)

    def standardize(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Fixed: returns 0 (no latent variable needed).

        Parameters
        ----------
        theta : float or array_like
            Physical-space parameter value (ignored for fixed parameters).

        Returns
        -------
        float
            Always 0.0.
        """
        return jnp.array(0.0)

    def __repr__(self) -> str:
        if isinstance(self._value, str):
            return f"Fixed({self._value!r})"
        return f"Fixed({self._value})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Fixed) and self._value == other._value


# ── Shorthand resolution ──────────────────────────────────────────


def resolve_shorthand(val) -> Distribution:
    """Convert shorthand notation to a Distribution object.

    Provides convenient shorthand for common prior specifications, allowing
    Parameters to accept scalar, tuple, or explicit Distribution objects.

    Parameters
    ----------
    val : float, int, tuple, or Distribution
        Parameter specification:

        - Scalar int/float → Fixed(value)
        - Tuple (lo, hi) → Uniform(lo, hi)
        - Distribution instance → returned unchanged

    Returns
    -------
    Distribution
        Resolved distribution object.

    Raises
    ------
    TypeError
        If val is not a supported type.

    Notes
    -----
    This function enables the concise Parameters syntax, e.g.::

        Parameters(
            redshift=0.1,  # → Fixed(0.1)
            met_logzsol=(-2, 0.5),  # → Uniform(-2, 0.5)
            dust_tau_bc=Uniform(0, 4),  # → Uniform(0, 4)
            neb_logU=Gaussian(-3, 0.5),  # → Gaussian(-3, 0.5)
        )

    Examples
    --------
    >>> from tengri.parameters.priors import resolve_shorthand
    >>> resolve_shorthand(0.1)
    Fixed(0.1)
    >>> resolve_shorthand((0, 1))
    Uniform(0, 1)
    >>> from tengri import Uniform
    >>> resolve_shorthand(Uniform(0, 1))
    Uniform(0, 1)
    """
    if isinstance(val, Distribution):
        return val
    if isinstance(val, (int, float)):
        return Fixed(val)
    if isinstance(val, tuple) and len(val) == 2:
        return Uniform(val[0], val[1])
    raise TypeError(
        f"Cannot resolve {val!r} to a Distribution. "
        f"Expected float (Fixed), tuple (lo, hi) (Uniform), "
        f"or a Distribution instance."
    )
