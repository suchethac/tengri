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
    """Base class for parameter distributions."""

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

    def __init__(self, mu: float, sigma: float,
                 lo: float = float("-inf"), hi: float = float("inf")):
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

    def __repr__(self) -> str:
        parts = [f"mu={self._mu}", f"sigma={self._sigma}"]
        if self._lo != float("-inf"):
            parts.append(f"lo={self._lo}")
        if self._hi != float("inf"):
            parts.append(f"hi={self._hi}")
        return f"Gaussian({', '.join(parts)})"

    def __eq__(self, other) -> bool:
        return (isinstance(other, Gaussian)
                and self._mu == other._mu and self._sigma == other._sigma
                and self._lo == other._lo and self._hi == other._hi)


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
        return 10.0 ** log_val

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        in_bounds = (x >= self._lo) & (x <= self._hi)
        lp = -jnp.log(x * jnp.log(self._hi / self._lo))
        return jnp.where(in_bounds, lp, -jnp.inf)

    def __repr__(self) -> str:
        return f"LogUniform({self._lo}, {self._hi})"

    def __eq__(self, other) -> bool:
        return isinstance(other, LogUniform) and self._lo == other._lo and self._hi == other._hi


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
