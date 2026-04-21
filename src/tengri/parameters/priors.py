"""Probability distribution objects for parameter specification.

Each distribution defines a prior for a single model parameter.
Used by Parameters for both mock generation (sampling) and inference (priors).

All methods are JAX-compatible for use inside JIT-compiled functions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

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

    @property
    def is_fixed(self) -> bool:
        """Return True if this is a Fixed distribution, False otherwise."""
        return False

    @property
    def bounds(self) -> tuple[float, float]:
        """Lower and upper bounds [lo, hi] for this distribution."""
        raise NotImplementedError

    def sample(self, key: jax.Array) -> jnp.ndarray:
        """Draw a random sample from the prior distribution."""
        raise NotImplementedError

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate log probability density at parameter value x."""
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
        """
        raise NotImplementedError(f"{type(self).__name__} must implement unstandardize()")

    def standardize(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Map physical parameter → standardized latent ξ.

        Inverse of unstandardize. Used for initialization from
        physical parameter values (e.g., from a MAP solution).
        """
        raise NotImplementedError(f"{type(self).__name__} must implement standardize()")

    def to_nifty_prior(self):
        """Convert to a NIFTy.re prior transform (optional).

        Returns a callable that maps ξ → θ, compatible with
        NIFTy's CorrelatedFieldMaker and optimize_kl.
        Returns None if nifty8.re is not installed.
        """
        import importlib.util

        if importlib.util.find_spec("nifty8") is None:
            return None
        # Default: wrap our unstandardize as a callable
        return self.unstandardize


# ── Concrete distributions ────────────────────────────────────────


class Uniform(Distribution):
    """Uniform prior on [lo, hi].

    A flat probability density on the interval [lo, hi]. Commonly used for
    bounded astrophysical quantities with little prior knowledge. Reparameterizes
    via sigmoid to ensure differentiability and automatic bound satisfaction.

    Parameters
    ----------
    lo : float
        Lower bound (inclusive).
    hi : float
        Upper bound (inclusive). Must satisfy hi > lo.

    Raises
    ------
    ValueError
        If lo >= hi.

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    **Standardization**: Maps ξ ~ N(0,1) to θ via the sigmoid function:

    .. math::

        \\theta = lo + (hi - lo) \\cdot \\sigma(\\xi)

    where σ(ξ) = 1 / (1 + exp(-ξ)). At ξ = 0 (prior center), θ = (lo + hi) / 2.
    This ensures automatic bound satisfaction and smooth gradients.

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

    def __init__(self, lo: float, hi: float):
        if lo >= hi:
            raise ValueError(f"Uniform requires lo < hi, got lo={lo}, hi={hi}")
        self._lo = float(lo)
        self._hi = float(hi)

    @property
    def lo(self) -> float:
        """Lower bound of the uniform distribution."""
        return self._lo

    @property
    def hi(self) -> float:
        """Upper bound of the uniform distribution."""
        return self._hi

    @property
    def bounds(self) -> tuple[float, float]:
        """Lower and upper bounds [lo, hi]."""
        return (self._lo, self._hi)

    def sample(self, key: jax.Array) -> jnp.ndarray:
        """Draw one sample uniformly from [lo, hi]."""
        return jax.random.uniform(key, minval=self._lo, maxval=self._hi)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return log probability: -log(hi-lo) inside bounds, -inf outside."""
        in_bounds = (x >= self._lo) & (x <= self._hi)
        return jnp.where(in_bounds, -jnp.log(self._hi - self._lo), -jnp.inf)

    def unstandardize(self, xi: jnp.ndarray) -> jnp.ndarray:
        """ξ ~ N(0,1) → Uniform(lo, hi) via sigmoid.

        At ξ=0 (prior center), sigmoid(0) = 0.5, so θ = midpoint of [lo, hi].
        At ξ=±3 (~99.7% of N(0,1) mass), θ covers ~95% of [lo, hi].
        The sigmoid naturally respects bounds without clipping.
        """
        return self._lo + (self._hi - self._lo) * jax.nn.sigmoid(xi)

    def standardize(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Uniform(lo, hi) → ξ via logit."""
        u = (theta - self._lo) / (self._hi - self._lo)
        u = jnp.clip(u, 1e-6, 1 - 1e-6)
        return jnp.log(u / (1 - u))  # logit

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

    Raises
    ------
    ValueError
        If sigma <= 0 or lo >= hi.

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    **Standardization**: Maps ξ ~ N(0,1) to θ via:

    .. math::

        \\theta = \\text{clip}(\\mu + \\sigma \\cdot \\xi, lo, hi)

    **Normalization**: When lo, hi are finite, the density is normalized over [lo, hi],
    not over the full real line. Use this for physically bounded quantities.

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
        self, mu: float, sigma: float, lo: float = float("-inf"), hi: float = float("inf")
    ):
        if sigma <= 0:
            raise ValueError(f"Gaussian requires sigma > 0, got {sigma}")
        if lo >= hi:
            raise ValueError(f"Gaussian requires lo < hi, got lo={lo}, hi={hi}")
        self._mu = float(mu)
        self._sigma = float(sigma)
        self._lo = float(lo)
        self._hi = float(hi)

    @property
    def mu(self) -> float:
        """Mean of the Gaussian distribution."""
        return self._mu

    @property
    def sigma(self) -> float:
        """Standard deviation of the Gaussian distribution."""
        return self._sigma

    @property
    def lo(self) -> float:
        """Lower truncation bound."""
        return self._lo

    @property
    def hi(self) -> float:
        """Upper truncation bound."""
        return self._hi

    @property
    def bounds(self) -> tuple[float, float]:
        """Lower and upper truncation bounds [lo, hi]."""
        return (self._lo, self._hi)

    def sample(self, key: jax.Array) -> jnp.ndarray:
        """Draw a random sample from the Gaussian distribution."""
        raw = self._mu + self._sigma * jax.random.normal(key)
        return jnp.clip(raw, self._lo, self._hi)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate log probability density, returning -inf outside bounds."""
        lp = -0.5 * ((x - self._mu) / self._sigma) ** 2
        in_bounds = (x >= self._lo) & (x <= self._hi)
        return jnp.where(in_bounds, lp, -jnp.inf)

    def unstandardize(self, xi: jnp.ndarray) -> jnp.ndarray:
        """ξ ~ N(0,1) → N(μ,σ²) clipped to [lo, hi]."""
        return jnp.clip(self._mu + self._sigma * xi, self._lo, self._hi)

    def standardize(self, theta: jnp.ndarray) -> jnp.ndarray:
        """N(μ,σ²) → ξ."""
        return (theta - self._mu) / self._sigma

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

    **Standardization**: Maps ξ ~ N(0,1) to θ via sigmoid in log space:

    .. math::

        \\theta = 10^{\\log_{10}(lo) + (\\log_{10}(hi) - \\log_{10}(lo)) \\cdot \\sigma(\\xi)}

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

    def __init__(self, lo: float, hi: float):
        if lo <= 0:
            raise ValueError(f"LogUniform requires lo > 0, got {lo}")
        if lo >= hi:
            raise ValueError(f"LogUniform requires lo < hi, got lo={lo}, hi={hi}")
        self._lo = float(lo)
        self._hi = float(hi)

    @property
    def lo(self) -> float:
        """Lower bound of the log-uniform distribution."""
        return self._lo

    @property
    def hi(self) -> float:
        """Upper bound of the log-uniform distribution."""
        return self._hi

    @property
    def bounds(self) -> tuple[float, float]:
        """Lower and upper bounds [lo, hi]."""
        return (self._lo, self._hi)

    def sample(self, key: jax.Array) -> jnp.ndarray:
        """Draw one sample log-uniformly from [lo, hi]."""
        log_lo = jnp.log10(self._lo)
        log_hi = jnp.log10(self._hi)
        log_val = jax.random.uniform(key, minval=log_lo, maxval=log_hi)
        return 10.0**log_val

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return log probability: -log(x * log(hi/lo)) inside bounds, -inf outside."""
        in_bounds = (x >= self._lo) & (x <= self._hi)
        lp = -jnp.log(x * jnp.log(self._hi / self._lo))
        return jnp.where(in_bounds, lp, -jnp.inf)

    def unstandardize(self, xi: jnp.ndarray) -> jnp.ndarray:
        """ξ ~ N(0,1) → LogUniform(lo, hi) via sigmoid in log space."""
        log_lo = jnp.log(self._lo)
        log_hi = jnp.log(self._hi)
        return jnp.exp(log_lo + (log_hi - log_lo) * jax.nn.sigmoid(xi))

    def standardize(self, theta: jnp.ndarray) -> jnp.ndarray:
        """LogUniform(lo, hi) → ξ via logit in log space."""
        log_lo = jnp.log(self._lo)
        log_hi = jnp.log(self._hi)
        u = (jnp.log(theta) - log_lo) / (log_hi - log_lo)
        u = jnp.clip(u, 1e-6, 1 - 1e-6)
        return jnp.log(u / (1 - u))

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
        self, mu: float = 0.0, sigma: float = 1.0, lo: float = 0.0, hi: float = float("inf")
    ):
        if sigma <= 0:
            raise ValueError(f"LogNormal requires sigma > 0, got {sigma}")
        self._mu = float(mu)
        self._sigma = float(sigma)
        self._lo = float(lo)
        self._hi = float(hi)

    @property
    def mu(self) -> float:
        """Mean of log(theta)."""
        return self._mu

    @property
    def sigma(self) -> float:
        """Standard deviation of log(theta)."""
        return self._sigma

    @property
    def bounds(self) -> tuple[float, float]:
        """Lower and upper truncation bounds [lo, hi]."""
        return (self._lo, self._hi)

    def sample(self, key: jax.Array) -> jnp.ndarray:
        """Draw a random sample from the log-normal distribution."""
        log_val = self._mu + self._sigma * jax.random.normal(key)
        return jnp.clip(jnp.exp(log_val), self._lo, self._hi)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate log probability density, returning -inf outside bounds."""
        lp = -jnp.log(x) - 0.5 * ((jnp.log(x) - self._mu) / self._sigma) ** 2
        in_bounds = (x >= self._lo) & (x <= self._hi)
        return jnp.where(in_bounds, lp, -jnp.inf)

    def unstandardize(self, xi: jnp.ndarray) -> jnp.ndarray:
        """ξ ~ N(0,1) → exp(μ + σ·ξ), clipped to [lo, hi]."""
        return jnp.clip(jnp.exp(self._mu + self._sigma * xi), self._lo, self._hi)

    def standardize(self, theta: jnp.ndarray) -> jnp.ndarray:
        """LogNormal → ξ."""
        return (jnp.log(jnp.maximum(theta, 1e-30)) - self._mu) / self._sigma

    def __repr__(self) -> str:
        parts = [f"mu={self._mu}", f"sigma={self._sigma}"]
        if self._lo > 0:
            parts.append(f"lo={self._lo}")
        if self._hi < float("inf"):
            parts.append(f"hi={self._hi}")
        return f"LogNormal({', '.join(parts)})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, LogNormal) and self._mu == other._mu and self._sigma == other._sigma
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

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    The probability density follows a Student's t distribution with the
    standard normalisation. For finite df, it has heavier tails than a
    Gaussian.

    **Standardization**: Uses a Gaussian approximation with variance scaling:

    .. math::

        \\theta = \\text{clip}(\\mu + \\sigma \\cdot \\sqrt{df/(df-2)} \\cdot \\xi,
          lo, hi)

    This is valid for df > 2. For df ≤ 2, a fallback scale of 3 is used.

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
    ):
        self._mu = float(mu)
        self._sigma = float(sigma)
        self._df = float(df)
        self._lo = float(lo)
        self._hi = float(hi)

    @property
    def bounds(self) -> tuple[float, float]:
        """Lower and upper truncation bounds [lo, hi]."""
        return (self._lo, self._hi)

    def sample(self, key: jax.Array) -> jnp.ndarray:
        """Draw a random sample from the Student's t distribution."""
        # t = normal / sqrt(chi2/df)
        k1, k2 = jax.random.split(key)
        z = jax.random.normal(k1)
        chi2 = jax.random.gamma(k2, self._df / 2) * 2
        t = z / jnp.sqrt(chi2 / self._df)
        return jnp.clip(self._mu + self._sigma * t, self._lo, self._hi)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate log probability density, returning -inf outside bounds."""
        z = (x - self._mu) / self._sigma
        lp = -0.5 * (self._df + 1) * jnp.log(1 + z**2 / self._df)
        in_bounds = (x >= self._lo) & (x <= self._hi)
        return jnp.where(in_bounds, lp, -jnp.inf)

    def unstandardize(self, xi: jnp.ndarray) -> jnp.ndarray:
        """ξ ~ N(0,1) → t-distributed via Gaussian approximation.

        For df>2, a Gaussian with matched variance is a reasonable
        approximation for the bulk of the distribution.
        """
        # Scale factor: Var(t) = df/(df-2) for df>2
        scale = jnp.where(
            self._df > 2, jnp.sqrt(self._df / (self._df - 2)), 3.0
        )  # fallback for df<=2
        return jnp.clip(self._mu + self._sigma * scale * xi, self._lo, self._hi)

    def standardize(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Map a physical parameter value to a standardized coordinate via the Student-t scale."""
        scale = jnp.where(self._df > 2, jnp.sqrt(self._df / (self._df - 2)), 3.0)
        return (theta - self._mu) / (self._sigma * scale)

    def __repr__(self) -> str:
        return f"StudentT(mu={self._mu}, sigma={self._sigma}, df={self._df})"


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

    def __init__(self, value: float | str):
        self._value = value if isinstance(value, str) else float(value)

    @property
    def value(self) -> float | str:
        """The fixed value (numeric or string)."""
        return self._value

    @property
    def is_fixed(self) -> bool:
        """Return True — this is a fixed (non-free) parameter."""
        return True

    @property
    def bounds(self) -> tuple[float, float] | tuple[None, None]:
        """Return (value, value) for numeric, or (None, None) for string."""
        if isinstance(self._value, str):
            return (None, None)
        return (self._value, self._value)

    def sample(self, key: jax.Array) -> jnp.ndarray | str:
        """Return the fixed value (ignores random key)."""
        if isinstance(self._value, str):
            return self._value
        return jnp.array(self._value)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return 0.0 (fixed parameters have zero log-likelihood contribution)."""
        return jnp.array(0.0)

    def unstandardize(self, xi: jnp.ndarray) -> jnp.ndarray | str:
        """Fixed: always returns the fixed value (ignores ξ)."""
        if isinstance(self._value, str):
            return self._value
        return jnp.array(self._value)

    def standardize(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Fixed: returns 0 (no latent variable needed)."""
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
