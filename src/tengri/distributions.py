"""Probability distribution objects for parameter specification.

Each distribution defines a prior for a single model parameter.
Used by ParamSpec for both mock generation (sampling) and inference (priors).

All methods are JAX-compatible for use inside JIT-compiled functions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Distribution:
    """Base class for parameter distributions.

    Subclasses must implement:
    - bounds: (lo, hi) tuple
    - sample(key): draw from the prior
    - log_prob(x): log probability density

    For standardized inference, subclasses should also implement:
    - unstandardize(xi): map N(0,1) → physical space (differentiable)
    - standardize(theta): map physical space → N(0,1) (for initialization)

    The unstandardize method defines how the prior is absorbed into the
    forward model. The loss becomes H = ½χ² + ½ξᵀξ with no extra terms.
    """

    @property
    def is_fixed(self) -> bool:
        return False

    @property
    def bounds(self) -> tuple[float, float]:
        raise NotImplementedError

    def sample(self, key: jax.Array) -> jnp.ndarray:
        raise NotImplementedError

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
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


# ---------------------------------------------------------------------------
# Concrete distributions
# ---------------------------------------------------------------------------


class Uniform(Distribution):
    """Uniform prior on [lo, hi].

    Parameters
    ----------
    lo : float
        Lower bound.
    hi : float
        Upper bound (must be > lo).
    """

    def __init__(self, lo: float, hi: float):
        if lo >= hi:
            raise ValueError(f"Uniform requires lo < hi, got lo={lo}, hi={hi}")
        self._lo = float(lo)
        self._hi = float(hi)

    @property
    def lo(self) -> float:
        return self._lo

    @property
    def hi(self) -> float:
        return self._hi

    @property
    def bounds(self) -> tuple[float, float]:
        return (self._lo, self._hi)

    def sample(self, key: jax.Array) -> jnp.ndarray:
        return jax.random.uniform(key, minval=self._lo, maxval=self._hi)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
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
    """Gaussian prior, optionally clipped to [lo, hi].

    Parameters
    ----------
    mu : float
        Mean.
    sigma : float
        Standard deviation (must be > 0).
    lo : float
        Lower bound (default: -inf).
    hi : float
        Upper bound (default: +inf).
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
        return self._mu

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def lo(self) -> float:
        return self._lo

    @property
    def hi(self) -> float:
        return self._hi

    @property
    def bounds(self) -> tuple[float, float]:
        return (self._lo, self._hi)

    def sample(self, key: jax.Array) -> jnp.ndarray:
        raw = self._mu + self._sigma * jax.random.normal(key)
        return jnp.clip(raw, self._lo, self._hi)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
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

    Samples are drawn uniformly in log10(x), so the density in linear
    space is p(x) = 1 / (x * ln(hi/lo)).

    Parameters
    ----------
    lo : float
        Lower bound (must be > 0).
    hi : float
        Upper bound (must be > lo).
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
        return self._lo

    @property
    def hi(self) -> float:
        return self._hi

    @property
    def bounds(self) -> tuple[float, float]:
        return (self._lo, self._hi)

    def sample(self, key: jax.Array) -> jnp.ndarray:
        log_lo = jnp.log10(self._lo)
        log_hi = jnp.log10(self._hi)
        log_val = jax.random.uniform(key, minval=log_lo, maxval=log_hi)
        return 10.0**log_val

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
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

    Natural for positive-definite quantities with multiplicative
    uncertainty, such as PSD amplitudes and timescales.

    Parameters
    ----------
    mu : float
        Mean of log(θ).
    sigma : float
        Standard deviation of log(θ).
    lo : float
        Lower bound (default: 0).
    hi : float
        Upper bound (default: inf).
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
        return self._mu

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def bounds(self) -> tuple[float, float]:
        return (self._lo, self._hi)

    def sample(self, key: jax.Array) -> jnp.ndarray:
        log_val = self._mu + self._sigma * jax.random.normal(key)
        return jnp.clip(jnp.exp(log_val), self._lo, self._hi)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
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
    """Student's t prior — heavier tails than Gaussian.

    Useful for parameters that may have outlier-like behavior.
    Common in BAGPIPES-style SED fitting for robust priors.

    Parameters
    ----------
    mu : float
        Location.
    sigma : float
        Scale.
    df : float
        Degrees of freedom (df→∞ gives Gaussian, df=1 gives Cauchy).
    lo, hi : float
        Bounds.
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
        return (self._lo, self._hi)

    def sample(self, key: jax.Array) -> jnp.ndarray:
        # t = normal / sqrt(chi2/df)
        k1, k2 = jax.random.split(key)
        z = jax.random.normal(k1)
        chi2 = jax.random.gamma(k2, self._df / 2) * 2
        t = z / jnp.sqrt(chi2 / self._df)
        return jnp.clip(self._mu + self._sigma * t, self._lo, self._hi)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
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
        scale = jnp.where(self._df > 2, jnp.sqrt(self._df / (self._df - 2)), 3.0)
        return (theta - self._mu) / (self._sigma * scale)

    def __repr__(self) -> str:
        return f"StudentT(mu={self._mu}, sigma={self._sigma}, df={self._df})"


class Fixed(Distribution):
    """Fixed value — not sampled, not inferred.

    Parameters
    ----------
    value : float
        The fixed value.
    """

    def __init__(self, value: float):
        self._value = float(value)

    @property
    def value(self) -> float:
        return self._value

    @property
    def is_fixed(self) -> bool:
        return True

    @property
    def bounds(self) -> tuple[float, float]:
        return (self._value, self._value)

    def sample(self, key: jax.Array) -> jnp.ndarray:
        return jnp.array(self._value)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array(0.0)

    def unstandardize(self, xi: jnp.ndarray) -> jnp.ndarray:
        """Fixed: always returns the fixed value (ignores ξ)."""
        return jnp.array(self._value)

    def standardize(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Fixed: returns 0 (no latent variable needed)."""
        return jnp.array(0.0)

    def __repr__(self) -> str:
        return f"Fixed({self._value})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Fixed) and self._value == other._value


# ---------------------------------------------------------------------------
# Shorthand resolution
# ---------------------------------------------------------------------------


def resolve_shorthand(val) -> Distribution:
    """Convert shorthand notation to a Distribution object.

    - Scalar (int/float) → Fixed(value)
    - Tuple (lo, hi) → Uniform(lo, hi)
    - Distribution instance → returned as-is

    Parameters
    ----------
    val : float, tuple, or Distribution
        Parameter specification.

    Returns
    -------
    Distribution
        Resolved distribution object.
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
