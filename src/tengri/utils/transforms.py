# SPDX-License-Identifier: BSD-3-Clause
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
def sigmoid(x: jnp.ndarray, x0: float, k: float, ymin: float, ymax: float) -> jnp.ndarray:
    """Smooth bounded transform: R -> (ymin, ymax).

    Parameters
    ----------
    x: array
        Unbounded input.
    x0: float
        Midpoint of transition.
    k: float
        Steepness (0.1 is typical).
    ymin, ymax: float
        Output bounds.

    Returns
    -------
    array
        Bounded output in (ymin, ymax).
    """
    return ymin + (ymax - ymin) * jax.nn.sigmoid(k * (x - x0))


@jax.jit
def inverse_sigmoid(y: jnp.ndarray, x0: float, k: float, ymin: float, ymax: float) -> jnp.ndarray:
    """Inverse of the sigmoid transform: (ymin, ymax) -> R.

    Parameters
    ----------
    y: array_like
        Bounded input in (ymin, ymax).
    x0: float
        Midpoint of the forward sigmoid.
    k: float
        Steepness of the forward sigmoid.
    ymin, ymax: float
        Bounds used in the forward sigmoid.

    Returns
    -------
    ndarray
        Unbounded output in R.
    """
    lnarg = (ymax - ymin) / (y - ymin) - 1.0
    return x0 - jnp.log(lnarg) / k


@jax.jit
def softplus(x: jnp.ndarray) -> jnp.ndarray:
    """Softplus: smooth approximation to max(0, x).

    Parameters
    ----------
    x: array_like
        Input.

    Returns
    -------
    ndarray
        log(1 + exp(x)).
    """
    return jax.nn.softplus(x)


@jax.jit
def to_bounded(u_param: jnp.ndarray, lo: float, hi: float) -> jnp.ndarray:
    """Map unbounded parameter to (lo, hi) via Gaussian CDF.

    Uses the standard normal CDF Φ(u) = 0.5 * (1 + erf(u / sqrt(2))) to map
    N(0,1) latent variables to Uniform(lo, hi). At u=0, Φ(0) = 0.5,
    so θ maps to the midpoint. This ensures that an N(0,1) latent yields a
    genuine uniform prior on the bounded interval, not a midpoint-peaked one.

    Parameters
    ----------
    u_param: array_like
        Unbounded input.
    lo, hi: float
        Lower and upper bounds of the target interval.

    Returns
    -------
    ndarray
        Bounded output in (lo, hi).

    Notes
    -----
    JIT-compatible, differentiable everywhere. The Gaussian-CDF
    standardization (vs logistic sigmoid) ensures Uniform(lo,hi) is a flat
    prior under N(0,1) latent, fundamental for standardized inference.
    """
    # Standard normal CDF: Φ(u) = 0.5 * (1 + erf(u / sqrt(2)))
    phi_u = 0.5 * (1.0 + jax.scipy.special.erf(u_param / jnp.sqrt(2.0)))
    return lo + (hi - lo) * phi_u


@jax.jit
def to_unbounded(param: jnp.ndarray, lo: float, hi: float) -> jnp.ndarray:
    """Map bounded parameter from (lo, hi) to R via inverse Gaussian CDF.

    Inverts the to_bounded map: given θ ∈ (lo, hi), returns u such that
    θ = lo + (hi-lo) * Φ(u). Uses the inverse error function erfinv.

    Parameters
    ----------
    param: array_like
        Bounded input in (lo, hi).
    lo, hi: float
        Lower and upper bounds.

    Returns
    -------
    ndarray
        Unbounded output in R.

    Notes
    -----
    JIT-compatible, differentiable. Clipping to (1e-7, 1-1e-7) prevents
    numerical issues at the extremes.
    """
    # Normalize to [0, 1]
    p = (param - lo) / (hi - lo)
    # Clip to avoid numerical issues at the extremes
    p = jnp.clip(p, 1e-7, 1.0 - 1e-7)
    # Inverse of Φ: Φ^{-1}(p) = sqrt(2) * erfinv(2p - 1)
    return jnp.sqrt(2.0) * jax.scipy.special.erfinv(2.0 * p - 1.0)


@jax.jit
def log_det_jacobian_to_bounded(u_param: jnp.ndarray, lo: float, hi: float) -> jnp.ndarray:
    """Log determinant of Jacobian: d(bounded)/d(unbounded).

    For the Gaussian-CDF transform θ = lo + (hi - lo) * Φ(u),
    the Jacobian is: dθ/du = (hi - lo) * φ(u),
    where φ(u) is the standard normal PDF.

    This is the log|dθ/du| term needed when transforming densities:
        log p(θ) = log p(u) - log|dθ/du|

    Parameters
    ----------
    u_param: array
        Unbounded parameter.
    lo, hi: float
        Bounds of the transformed parameter.

    Returns
    -------
    array
        log|dθ/du| = log(hi - lo) + log_phi(u)

    Notes
    -----
    where log_phi(u) = -0.5*u**2 - 0.5*log(2π) is the log standard normal PDF.
    JIT-compatible and differentiable everywhere.
    """
    width = hi - lo
    # Log of standard normal PDF: log φ(u) = -0.5*u^2 - 0.5*log(2π)
    log_phi = -0.5 * u_param**2 - 0.5 * jnp.log(2.0 * jnp.pi)
    return jnp.log(width) + log_phi


@jax.jit
def log_det_jacobian_to_unbounded(param: jnp.ndarray, lo: float, hi: float) -> jnp.ndarray:
    """Log determinant of Jacobian: d(unbounded)/d(bounded).

    This is the inverse Jacobian: du/dθ = 1 / (dθ/du)

    Parameters
    ----------
    param: array_like
        Bounded parameter in (lo, hi).
    lo, hi: float
        Bounds.

    Returns
    -------
    array
        log|du/dθ| = -log|dθ/du|

    Notes
    -----
    Evaluated at u = to_unbounded(param, lo, hi). JIT-compatible.
    """
    # Compute u first
    u = to_unbounded(param, lo, hi)
    return -log_det_jacobian_to_bounded(u, lo, hi)
