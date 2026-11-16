# SPDX-License-Identifier: BSD-3-Clause
"""Guards and mode-finding shared by the flat-target samplers.

Every guard refuses a silent failure by name: a chain that never moved, an expansion about
a non-mode, a non-finite tuning, a live set that cannot span the space.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _require_moving_chain(chain, method):
    """Refuse a retained chain whose draws are all identical.

    #1530's lesson generalized past its raytrace origin: MAP-echo draws look
    like a plausible answer. ``_require_finite_tuning`` catches the MCLMC
    starved-tuner *cause*; this is the effect-side net for every MCMC driver
    whatever produced it, a chain that never moved is not a posterior.
    """
    import numpy as np

    if not bool(np.all(np.asarray(chain[1:]) == np.asarray(chain[0]))):
        return
    from tengri.inference.hierarchical import DegenerateChainError

    raise DegenerateChainError(
        f"{method}: every retained draw is identical to the first, the "
        f"chain never moved, so this is the initialization echoed "
        f"{int(chain.shape[0])} times, not a posterior. Check the sampler's "
        f"tuning diagnostics (step size, acceptance, warmup length) rather "
        f"than using these draws."
    )


def adam_map_ascent(logp, init, map_steps, map_learning_rate) -> jax.Array:
    """Adam gradient ascent on ``logp`` from ``init``, shared by map and laplace.

    One implementation so the two drivers cannot drift: laplace IS this
    ascent plus a verified-mode covariance.
    """
    import optax

    opt = optax.adam(map_learning_rate)
    state = opt.init(init)

    def _step(carry, _):
        pos, st = carry
        g = jax.grad(logp)(pos)
        updates, st = opt.update(jax.tree.map(lambda x: -x, g), st, pos)
        return (optax.apply_updates(pos, updates), st), None

    (best, _), _ = jax.lax.scan(_step, (init, state), None, length=map_steps)
    return jnp.asarray(best)


def _require_converged_mode(grad_at_mode, method, map_steps, *, tol):
    """Refuse a Laplace expansion about a point that is not a mode (#1537).

    The single-galaxy laplace expanded about non-modes with no grad=0 check
    and returned plausible wrong answers, curvature measured off a mode is
    a covariance for the wrong distribution. The gradient's infinity norm at
    the reached point must be small; otherwise raise naming the knob.
    """
    import numpy as np

    gmax = float(np.max(np.abs(np.asarray(grad_at_mode))))
    if gmax <= tol:
        return
    raise RuntimeError(
        f"{method}: the point reached after map_steps={map_steps} is not a "
        f"mode, max |grad log_prob| = {gmax:.3g} exceeds tol={tol:.3g}, and "
        f"a Laplace covariance measured off a mode is a plausible wrong "
        f"answer (#1537). Raise map_steps (or adjust map_learning_rate) "
        f"until the ascent converges, or loosen laplace_grad_tol if this "
        f"tolerance is genuinely too strict for your problem's scale."
    )


def _require_psd_curvature(chol, method):
    """Refuse a non-negative-definite curvature at the expansion point.

    ``jnp.linalg.cholesky`` returns NaNs, not an exception, when the negative
    Hessian is not positive definite, which means the reached point is a
    saddle or worse, not a maximum. Sampling from those NaNs would return a
    posterior of NaNs (or garbage that passes a finite-check downstream);
    refuse by name instead.
    """
    import numpy as np

    if bool(np.all(np.isfinite(np.asarray(chol)))):
        return
    raise RuntimeError(
        f"{method}: the negative Hessian at the reached point is not "
        f"positive definite (its Cholesky factor carries non-finite "
        f"entries), so the point is a saddle rather than a mode and no "
        f"Gaussian covariance exists there. Raise map_steps so the ascent "
        f"reaches an actual maximum (#1537)."
    )


def _require_finite_tuning(L, step_size, method, n_warmup):
    """Refuse a non-finite (L, step size) tuning instead of sampling with it.

    A starved MCLMC tuner produces NaN parameters, and the chain then never
    moves, measured on the 2-galaxy D=516 fixture at ``n_warmup=60``:
    ``L=nan``, ``step_size=nan``, 60 post-tuning draws all equal to the init
    point. That output LOOKS like a plausible populated posterior, which is
    the #1530 failure mode; #1569 made raytrace's version of it a loud
    ``DegenerateChainError``, and this is the MCLMC family's version. The
    fraction-based tuning phases starve below a few hundred steps; the same
    fixture tunes finite at ``n_warmup=500``.
    """
    import numpy as np

    if np.isfinite(float(L)) and np.isfinite(float(step_size)):
        return
    from tengri.inference.hierarchical import DegenerateChainError

    raise DegenerateChainError(
        f"{method}: the (L, step size) tuner returned non-finite values "
        f"(L={float(L)!r}, step_size={float(step_size)!r}) after "
        f"n_warmup={n_warmup} tuning steps, and a chain run with them never "
        f"moves, every draw would be a copy of the initialization, which "
        f"looks like a posterior and is not one. The fraction-based tuning "
        f"phases starve at short warmup; raise n_warmup to a few hundred "
        f"(the single-galaxy default is 500; n_warmup=500 tunes this family "
        f"finite on the reference 2-galaxy problem)."
    )


def _require_nondegenerate_live_set(n_live, n_dim, method):
    """Refuse an NSS live set that cannot span the parameter space.

    HRSS draws its slice directions from the live points' empirical
    covariance, and ``n_live`` points give that matrix rank at most
    ``n_live - 1``. With ``n_live <= D`` every direction lies in a proper
    subspace and the orthogonal complement is NEVER explored, the returned
    samples are confined to a hyperplane while passing every finite-check
    downstream. That is silent bias, not slowness, so it is refused by name
    rather than priced as a warning.
    """
    if n_live > n_dim:
        return
    raise ValueError(
        f"{method}: nss_n_live={n_live} live points cannot span a D={n_dim} "
        f"parameter space, the HRSS direction covariance has rank at most "
        f"n_live-1, so slice directions never leave a proper subspace and "
        f"the samples are silently biased. Raise nss_n_live above D (cost "
        f"grows with both), or use mcmc_nuts / mcmc_hmc, which need no live "
        f"set. Stochastic-field hierarchies put every field latent into D "
        f"(n_grid per galaxy); a coarser n_grid at build time shrinks D."
    )
