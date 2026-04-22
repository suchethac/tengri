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
def inverse_sigmoid(y: jnp.ndarray, x0: float, k: float, ymin: float, ymax: float) -> jnp.ndarray:
    """Inverse of the sigmoid transform: (ymin, ymax) -> R.

    Parameters
    ----------
    y : array_like
        Bounded input in (ymin, ymax).
    x0 : float
        Midpoint of the forward sigmoid.
    k : float
        Steepness of the forward sigmoid.
    ymin, ymax : float
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
    x : array_like
        Input.

    Returns
    -------
    ndarray
        log(1 + exp(x)).
    """
    return jax.nn.softplus(x)


@jax.jit
def log_softplus(x: jnp.ndarray) -> jnp.ndarray:
    """Log of softplus, numerically stable.

    Parameters
    ----------
    x : array_like
        Input.

    Returns
    -------
    ndarray
        log(log(1 + exp(x))), computed as logaddexp(x, 0).
    """
    return jnp.logaddexp(x, 0.0)


@jax.jit
def to_bounded(u_param: jnp.ndarray, lo: float, hi: float) -> jnp.ndarray:
    """Map unbounded parameter to (lo, hi) via sigmoid.

    Centered at x0=0 so that u=0 maps to the midpoint of (lo, hi).
    With k=1.0, u in [-3, +3] covers ~95% of the bounded range,
    matching the scale of standard-normal latent variables.

    Parameters
    ----------
    u_param : array_like
        Unbounded input.
    lo, hi : float
        Lower and upper bounds of the target interval.

    Returns
    -------
    ndarray
        Bounded output in (lo, hi).
    """
    return sigmoid(u_param, 0.0, 1.0, lo, hi)


@jax.jit
def to_unbounded(param: jnp.ndarray, lo: float, hi: float) -> jnp.ndarray:
    """Map bounded parameter from (lo, hi) to R.

    Parameters
    ----------
    param : array_like
        Bounded input in (lo, hi).
    lo, hi : float
        Lower and upper bounds.

    Returns
    -------
    ndarray
        Unbounded output in R.
    """
    return inverse_sigmoid(param, 0.0, 1.0, lo, hi)


@jax.jit
def log_det_jacobian_to_bounded(u_param: jnp.ndarray, lo: float, hi: float) -> jnp.ndarray:
    """Log determinant of Jacobian: d(bounded)/d(unbounded).

    For the sigmoid transform θ = lo + (hi - lo) * sigmoid(u),
    the Jacobian is: dθ/du = (hi - lo) * sigmoid(u) * (1 - sigmoid(u))

    This is the log|dθ/du| term needed when transforming densities:
        p(θ) = p(u) * |du/dθ| = p(u) / |dθ/du|
        log p(θ) = log p(u) - log|dθ/du|

    Parameters
    ----------
    u_param : array
        Unbounded parameter.
    lo, hi : float
        Bounds of the transformed parameter.

    Returns
    -------
    array
        log|dθ/du| = log(hi - lo) + log(sigmoid(u)) + log(1 - sigmoid(u))

    Notes
    -----
    With k=1.0 (as used in to_bounded), this simplifies to:
        log|dθ/du| = log(hi - lo) + u - 2*log(1 + exp(u))
                    = log(hi - lo) + u - 2*softplus(u)
    """
    width = hi - lo
    # log(sig * (1 - sig)) = log(sig) + log(1 - sig)
    # Numerically stable:
    # log(sigmoid(u)) = -softplus(-u)
    # log(1 - sigmoid(u)) = -softplus(u)
    log_jac = jnp.log(width) - jax.nn.softplus(-u_param) - jax.nn.softplus(u_param)
    return log_jac


@jax.jit
def log_det_jacobian_to_unbounded(param: jnp.ndarray, lo: float, hi: float) -> jnp.ndarray:
    """Log determinant of Jacobian: d(unbounded)/d(bounded).

    This is the inverse Jacobian: du/dθ = 1 / (dθ/du)

    Parameters
    ----------
    param : array_like
        Bounded parameter in (lo, hi).
    lo, hi : float
        Bounds.

    Returns
    -------
    array
        log|du/dθ| = -log|dθ/du|
    """
    # Compute u first
    u = to_unbounded(param, lo, hi)
    return -log_det_jacobian_to_bounded(u, lo, hi)
