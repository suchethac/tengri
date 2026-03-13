"""Parameter transforms for bounded/unbounded optimization.

Inspired by diffsky's utility_funcs.py pattern: all physical parameters
live in bounded space, but samplers/optimizers work in unbounded space.
Sigmoid transforms connect the two, with well-defined gradients everywhere.

This is the same idea as NIFTy's standardization: the sampler explores
xi ~ N(0, I) in unbounded space, and the forward model maps to physical
parameters via differentiable bijections.
"""

import jax
import jax.numpy as jnp


@jax.jit
def sigmoid(x: jnp.ndarray, x0: float, k: float,
            ymin: float, ymax: float) -> jnp.ndarray:
    """Smooth bounded transform: R -> (ymin, ymax).

    Parameters
    ----------
    x : array
        Unbounded input.
    x0 : float
        Midpoint of transition.
    k : float
        Steepness (0.1 is typical).
    ymin, ymax : float
        Output bounds.

    Returns
    -------
    array
        Bounded output in (ymin, ymax).
    """
    return ymin + (ymax - ymin) * jax.nn.sigmoid(k * (x - x0))


@jax.jit
def inverse_sigmoid(y: jnp.ndarray, x0: float, k: float,
                    ymin: float, ymax: float) -> jnp.ndarray:
    """Inverse of sigmoid: (ymin, ymax) -> R."""
    lnarg = (ymax - ymin) / (y - ymin) - 1.0
    return x0 - jnp.log(lnarg) / k


@jax.jit
def softplus(x: jnp.ndarray) -> jnp.ndarray:
    """Softplus: smooth approximation to max(0, x)."""
    return jax.nn.softplus(x)


@jax.jit
def log_softplus(x: jnp.ndarray) -> jnp.ndarray:
    """Log of softplus (numerically stable)."""
    return jnp.logaddexp(x, 0.0)


@jax.jit
def to_bounded(u_param: jnp.ndarray, lo: float, hi: float) -> jnp.ndarray:
    """Map unbounded parameter to (lo, hi) via sigmoid.

    Convenience wrapper with default steepness k=0.1, centered at midpoint.
    """
    mid = 0.5 * (lo + hi)
    return sigmoid(u_param, mid, 0.1, lo, hi)


@jax.jit
def to_unbounded(param: jnp.ndarray, lo: float, hi: float) -> jnp.ndarray:
    """Map bounded parameter from (lo, hi) to R."""
    mid = 0.5 * (lo + hi)
    return inverse_sigmoid(param, mid, 0.1, lo, hi)
