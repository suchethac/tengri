###**************************************
### Ray Tracing Sampler
### Original HMC Implementation: Copyright (C) 2024, Martin Marek
### See original source at https://github.com/martin-marek/mini-hmc-jax
### Additional Changes (sample_raytracer, UpdateV, raytracer_leapfrog):
###   Copyright (C) 2025, Peter Behroozi
###
### Licensed under the Apache License, Version 2.0 (the "License");
### you may not use this file except in compliance with the License.
### You may obtain a copy of the License at
###
###     http://www.apache.org/licenses/LICENSE-2.0
###
### Unless required by applicable law or agreed to in writing, software
### distributed under the License is distributed on an "AS IS" BASIS,
### WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
### See the License for the specific language governing permissions and
### limitations under the License.
###**************************************

"""Ray Tracing Sampler (Behroozi 2025).

A physics-inspired MCMC sampler that propagates light rays through a
medium where the refractive index n(x) = L(x)^{1/(D-1)}. Snell's law
bends rays toward high-likelihood regions, naturally producing fair
posterior samples.

Key advantages over HMC/NUTS:
- Orders of magnitude more resilient to stochastic/noisy gradients
- No energy conservation issues (constant speed propagation)
- Can cross arbitrary likelihood barriers
- Simple Metropolis correction

Reference: Behroozi (2025), arXiv:2510.25824
Source: https://bitbucket.org/pbehroozi/ray-tracing-sampler/src/main/
"""

import operator as op

import jax
import jax.numpy as jnp
from jax.tree_util import tree_leaves, tree_map, tree_reduce

__all__ = ["sample_hamiltonian", "sample_raytrace"]


def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None:
        treedef = jax.tree_util.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)


def normal_like_tree(rng_key, target, mean=0, std=1):
    keys_tree = random_split_like_tree(rng_key, target)
    return tree_map(
        lambda l, k: mean + std * jax.random.normal(k, l.shape, l.dtype),
        target,
        keys_tree,
    )


def ifelse(cond, val_true, val_false):
    return jax.lax.cond(cond, lambda x: x[0], lambda x: x[1], (val_true, val_false))


def ScatterV(momentum, refresh_rate, dt, key):
    key, normal_key = jax.random.split(key, 2)
    f = jnp.exp(-jnp.abs(refresh_rate * dt))
    fn = jnp.sqrt(1.0 - f * f)
    diffusion = normal_like_tree(normal_key, momentum)
    momentum = tree_map(lambda m, d: f * m + fn * d, momentum, diffusion)
    return momentum, key


# -------------------------------------------------------------------
# HMC leapfrog variants (for comparison)
# -------------------------------------------------------------------


def hmc_leapfrog_refresh(params, momentum, log_prob_fn, step_size, n_steps, refresh_rate, key):
    """HMC leapfrog with partial momentum refresh."""
    kinetic_energy_diff = 0

    def step(i, args):
        params, momentum, kinetic_energy_diff, key = args
        momentum, key = ScatterV(momentum, refresh_rate, step_size / 2.0, key)
        params = tree_map(lambda p, m: p + 0.5 * m * step_size, params, momentum)
        momentum_dot = tree_reduce(op.add, tree_map(lambda x: (x**2).sum(), tree_leaves(momentum)))
        grad = jax.grad(log_prob_fn)(params)
        momentum = tree_map(lambda m, g: m + step_size * g, momentum, grad)
        new_momentum_dot = tree_reduce(
            op.add, tree_map(lambda x: (x**2).sum(), tree_leaves(momentum))
        )
        kinetic_energy_diff += 0.5 * (momentum_dot - new_momentum_dot)
        params = tree_map(lambda p, m: p + 0.5 * m * step_size, params, momentum)
        momentum, key = ScatterV(momentum, refresh_rate, step_size / 2.0, key)
        return params, momentum, kinetic_energy_diff, key

    new_params, new_momentum, kinetic_energy_diff, key = jax.lax.fori_loop(
        0, n_steps, step, (params, momentum, kinetic_energy_diff, key)
    )
    return new_params, new_momentum, -kinetic_energy_diff, key


def hmc_leapfrog_norefresh(params, momentum, log_prob_fn, step_size, n_steps, refresh_rate, key):
    """HMC leapfrog without momentum refresh."""
    momentum_dot = tree_reduce(op.add, tree_map(lambda x: (x**2).sum(), tree_leaves(momentum)))

    def step(i, args):
        params, momentum = args
        params = tree_map(lambda p, m: p + 0.5 * m * step_size, params, momentum)
        grad = jax.grad(log_prob_fn)(params)
        momentum = tree_map(lambda m, g: m + step_size * g, momentum, grad)
        params = tree_map(lambda p, m: p + 0.5 * m * step_size, params, momentum)
        return params, momentum

    new_params, new_momentum = jax.lax.fori_loop(0, n_steps, step, (params, momentum))
    new_momentum_dot = tree_reduce(
        op.add, tree_map(lambda x: (x**2).sum(), tree_leaves(new_momentum))
    )
    kinetic_energy_diff = 0.5 * (momentum_dot - new_momentum_dot)
    return new_params, new_momentum, -kinetic_energy_diff, key


# -------------------------------------------------------------------
# Ray tracing core: UpdateV (Snell's law in parameter space)
# -------------------------------------------------------------------


def UpdateV(momentum, grad, D, step_size):
    """Update velocity direction via Snell's law refraction.

    Implements Eq. 23 from Behroozi (2025):
    tan(θ_f/2) = tan(θ_i/2) * exp(-Δs * |∇ln(n)|)

    Matches the C implementation guard: ``if (!mnorm || !gnorm) return 0;``
    When either the momentum or gradient has zero norm, the velocity is
    unchanged and delta_ln_L = 0.

    Parameters
    ----------
    momentum : array
        Current velocity vector.
    grad : array
        Gradient of log-probability (∝ grad(ln n)).
    D : int
        Number of dimensions.
    step_size : float
        Integration step size.

    Returns
    -------
    tuple
        (new_momentum, delta_ln_L, norm_v, norm_g, theta_i, theta_f, f_v, f_n)
    """
    norm_v = jnp.sqrt(tree_reduce(op.add, tree_map(lambda x: (x**2).sum(), tree_leaves(momentum))))
    norm_g = jnp.sqrt(tree_reduce(op.add, tree_map(lambda x: (x**2).sum(), tree_leaves(grad))))

    # Guard: if either norm is zero, no refraction occurs.
    # Matches C code: if (!mnorm || !gnorm) return 0;
    # Without this, unit_g = grad/0 → NaN, poisoning the entire chain.
    # This matters for KDK at stationary points (e.g. mode of a Gaussian).
    zero = jnp.float64(0.0) if momentum.dtype == jnp.float64 else jnp.float32(0.0)
    zero_result = (momentum, zero, norm_v, norm_g, zero, zero, jnp.ones_like(zero), zero)

    def _refract(_):
        unit_v = tree_map(lambda m, n: m / n, momentum, norm_v)
        unit_g = tree_map(lambda g, n: g / n, grad, norm_g)

        sub_vec = tree_map(lambda v, g: v - g, unit_v, unit_g)
        sub_vec_norm = jnp.sqrt(
            tree_reduce(op.add, tree_map(lambda x: (x**2).sum(), tree_leaves(sub_vec)))
        )
        add_vec = tree_map(lambda v, g: v + g, unit_v, unit_g)
        add_vec_norm = jnp.sqrt(
            tree_reduce(op.add, tree_map(lambda x: (x**2).sum(), tree_leaves(add_vec)))
        )

        theta_i = 2.0 * jnp.arctan2(sub_vec_norm, add_vec_norm)
        theta_f = 2.0 * jnp.arctan2(
            sub_vec_norm,
            add_vec_norm * jnp.exp(norm_g * norm_v * step_size / (D - 1.0)),
        )

        f_v = jax.lax.cond(
            jnp.sin(theta_i) == 0,
            lambda _: 1.0,
            lambda _: jnp.sin(theta_f) / jnp.sin(theta_i),
            operand=None,
        )
        f_n = (jnp.cos(theta_f) - f_v * jnp.cos(theta_i)) * norm_v

        new_momentum = tree_map(lambda m, ug, fv, fn: fv * m + fn * ug, momentum, unit_g, f_v, f_n)

        delta_ln_L = jax.lax.cond(
            jnp.sin(theta_i) == 0,
            lambda _: norm_v * norm_g * step_size * jnp.cos(theta_i),
            lambda _: (1.0 - D) * jnp.log(f_v),
            operand=None,
        )
        return (new_momentum, delta_ln_L, norm_v, norm_g, theta_i, theta_f, f_v, f_n)

    return jax.lax.cond(
        (norm_v == 0) | (norm_g == 0),
        lambda _: zero_result,
        _refract,
        operand=None,
    )


# -------------------------------------------------------------------
# Ray tracing leapfrog integrators
# -------------------------------------------------------------------


def raytracer_leapfrog_refresh(
    params, momentum, log_prob_fn, step_size, n_steps, refresh_rate, key
):
    """Ray tracing DKD leapfrog with partial momentum refresh."""
    ln_L = 0

    def step(i, args):
        params, momentum, ln_L, key = args
        momentum, key = ScatterV(momentum, refresh_rate, step_size / 2.0, key)
        params = tree_map(lambda p, m: p + 0.5 * m * step_size, params, momentum)
        grad = jax.grad(log_prob_fn)(params)
        momentum, delta_ln_L, *_ = UpdateV(momentum, grad, momentum.size, step_size)
        ln_L += delta_ln_L
        params = tree_map(lambda p, m: p + 0.5 * m * step_size, params, momentum)
        momentum, key = ScatterV(momentum, refresh_rate, step_size / 2.0, key)
        return params, momentum, ln_L, key

    new_params, new_momentum, new_ln_L, key = jax.lax.fori_loop(
        0, n_steps, step, (params, momentum, ln_L, key)
    )
    return new_params, new_momentum, new_ln_L, key


def raytracer_leapfrog_norefresh(
    params, momentum, log_prob_fn, step_size, n_steps, refresh_rate, key
):
    """Ray tracing DKD leapfrog without momentum refresh."""
    ln_L = 0

    def step(i, args):
        params, momentum, ln_L = args
        params = tree_map(lambda p, m: p + 0.5 * m * step_size, params, momentum)
        grad = jax.grad(log_prob_fn)(params)
        momentum, delta_ln_L, *_ = UpdateV(momentum, grad, momentum.size, step_size)
        ln_L += delta_ln_L
        params = tree_map(lambda p, m: p + 0.5 * m * step_size, params, momentum)
        return params, momentum, ln_L

    new_params, new_momentum, new_ln_L = jax.lax.fori_loop(
        0, n_steps, step, (params, momentum, ln_L)
    )
    return new_params, new_momentum, new_ln_L, key


# -------------------------------------------------------------------
# KDK (Kick-Drift-Kick) integrators
# -------------------------------------------------------------------


def raytracer_kdk_norefresh(params, momentum, log_prob_fn, step_size, n_steps, refresh_rate, key):
    """Ray tracing KDK leapfrog without momentum refresh.

    KDK: Kick(dt/2) at position → Drift(dt) → Kick(dt/2) at new position.
    Both DKD and KDK are second-order palindromic integrators with valid
    radiance tracking. The half-kicks use δ = dt/2 in the Snell's law formula.
    """
    ln_L = 0
    half_dt = step_size * 0.5

    def step(i, args):
        params, momentum, ln_L = args
        # Half kick at current position
        grad = jax.grad(log_prob_fn)(params)
        momentum, delta_ln_L, *_ = UpdateV(momentum, grad, momentum.size, half_dt)
        ln_L += delta_ln_L
        # Full drift
        params = tree_map(lambda p, m: p + m * step_size, params, momentum)
        # Half kick at new position
        grad = jax.grad(log_prob_fn)(params)
        momentum, delta_ln_L, *_ = UpdateV(momentum, grad, momentum.size, half_dt)
        ln_L += delta_ln_L
        return params, momentum, ln_L

    new_params, new_momentum, new_ln_L = jax.lax.fori_loop(
        0, n_steps, step, (params, momentum, ln_L)
    )
    return new_params, new_momentum, new_ln_L, key


def hmc_kdk_norefresh(params, momentum, log_prob_fn, step_size, n_steps, refresh_rate, key):
    """HMC KDK leapfrog without momentum refresh."""
    momentum_dot = tree_reduce(op.add, tree_map(lambda x: (x**2).sum(), tree_leaves(momentum)))

    def step(i, args):
        params, momentum = args
        # Half kick
        grad = jax.grad(log_prob_fn)(params)
        momentum = tree_map(lambda m, g: m + 0.5 * step_size * g, momentum, grad)
        # Full drift
        params = tree_map(lambda p, m: p + m * step_size, params, momentum)
        # Half kick
        grad = jax.grad(log_prob_fn)(params)
        momentum = tree_map(lambda m, g: m + 0.5 * step_size * g, momentum, grad)
        return params, momentum

    new_params, new_momentum = jax.lax.fori_loop(0, n_steps, step, (params, momentum))

    new_momentum_dot = tree_reduce(
        op.add, tree_map(lambda x: (x**2).sum(), tree_leaves(new_momentum))
    )
    kinetic_energy_diff = 0.5 * (momentum_dot - new_momentum_dot)
    return new_params, new_momentum, -kinetic_energy_diff, key


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------


def sample_hamiltonian(
    key,
    params_init,
    log_prob_fn,
    n_steps,
    n_leapfrog_steps,
    step_size,
    refresh_rate=0,
    metro_check=1,
    sample_hmc=True,
):
    """HMC sampling (convenience wrapper around sample_raytrace)."""
    return sample_raytrace(
        key,
        params_init,
        log_prob_fn,
        n_steps,
        n_leapfrog_steps,
        step_size,
        refresh_rate,
        metro_check,
        sample_hmc,
    )


def sample_raytrace(
    key,
    params_init,
    log_prob_fn,
    n_steps,
    n_leapfrog_steps,
    step_size,
    refresh_rate=0,
    metro_check=1,
    sample_hmc=False,
    integrator="dkd",
):
    """Run Ray Tracing Sampler and return the full Markov chain.

    Parameters
    ----------
    key : PRNGKey
        Random key.
    params_init : array
        Initial parameter values (flat 1D array).
    log_prob_fn : callable
        Function mapping params → scalar log probability.
    n_steps : int
        Number of MCMC steps (samples to collect).
    n_leapfrog_steps : int
        Leapfrog integration steps per trajectory.
    step_size : float
        Leapfrog step size. Recommended: ~0.03 * sqrt(D).
    refresh_rate : float
        Partial momentum refresh rate. 0 = no refresh.
    metro_check : int
        1 = apply Metropolis correction, 0 = skip.
    sample_hmc : bool
        If True, use HMC instead of ray tracing.
    integrator : str
        Leapfrog integrator scheme: ``"dkd"`` (Drift-Kick-Drift, default)
        or ``"kdk"`` (Kick-Drift-Kick). KDK is the time-reversed partner
        of DKD. Both are symplectic and second-order. DKD matches
        Behroozi's reference implementation.

    Returns
    -------
    chain : array, shape (n_steps, D)
        Parameter samples.
    log_likelihood : array, shape (n_steps,)
        Log-likelihood at each accepted sample.
    accept_prob : array, shape (n_steps,)
        Acceptance probability at each step.
    """
    if params_init.size < 2 and not sample_hmc:
        sample_hmc = True

    integrator = integrator.lower()
    if integrator not in ("dkd", "kdk"):
        raise ValueError(f"Unknown integrator: {integrator!r}. Use 'dkd' or 'kdk'.")
    if integrator == "kdk":
        if sample_hmc:
            leapfrog_func = hmc_kdk_norefresh
        else:
            leapfrog_func = raytracer_kdk_norefresh
    else:
        leapfrog_func = raytracer_leapfrog_norefresh
        if sample_hmc:
            if refresh_rate:
                leapfrog_func = hmc_leapfrog_refresh
            else:
                leapfrog_func = hmc_leapfrog_norefresh
        elif refresh_rate:
            leapfrog_func = raytracer_leapfrog_refresh

    # Cache log-likelihood in scan carry to avoid redundant evaluation.
    # After accept: lnl = new_lnl = log_prob(new_params) = log_prob(params).
    # After reject: lnl = old_lnl = log_prob(old_params) = log_prob(params).
    # Saves one forward model evaluation per MCMC step.
    init_lnl = log_prob_fn(params_init)

    def ray_step_fn(carry, x):
        params, key, old_lnl = carry
        key, normal_key, uniform_key = jax.random.split(key, 3)

        momentum = normal_like_tree(normal_key, params)
        delta_ln_L = 0

        new_params, _new_momentum, delta_ln_L, key = leapfrog_func(
            params,
            momentum,
            log_prob_fn,
            step_size,
            n_leapfrog_steps,
            refresh_rate,
            key,
        )

        # Metropolis-Hastings correction (old_lnl from carry, not recomputed)
        new_lnl = log_prob_fn(new_params)
        log_likelihood_diff = new_lnl - old_lnl
        log_accept_prob = log_likelihood_diff - delta_ln_L
        log_accept_prob = jnp.nan_to_num(log_accept_prob, nan=-jnp.inf)
        accept_prob = jnp.minimum(1.0, jnp.exp(log_accept_prob))
        accept_prob = jnp.maximum(accept_prob, 1.0 - metro_check)
        accept = jax.random.uniform(uniform_key) < accept_prob
        params = ifelse(accept, new_params, params)
        lnl = ifelse(accept, new_lnl, old_lnl)
        return (params, key, lnl), (params, accept_prob, lnl)

    _, (chain, accept_prob, log_likelihood) = jax.lax.scan(
        ray_step_fn,
        (params_init, key, init_lnl),
        xs=None,
        length=n_steps,
    )

    return chain, log_likelihood, accept_prob
