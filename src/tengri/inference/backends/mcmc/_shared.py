# SPDX-License-Identifier: BSD-3-Clause
"""Shared MCMC infrastructure: kernel getters, scan functions, logdensity helpers.

Internal, imported by per-sampler modules. Not part of the public API.

Compilation strategy
--------------------
Every sampler exposes two JIT-compiled entry points:

``_<method>_full_scan``
    Outer JIT wrapping BlackJAX window adaptation **and** the chain
    (burn-in + sampling) in a single XLA program.  Used for the cold
    path (no cached adaptation).  The kernel, e.g. the NUTS
    ``lax.while_loop`` tree builder, is compiled exactly once instead
    of once per phase.  Returns ``(positions, divergent, step_size,
    inv_mass_matrix)`` so the caller can cache the adaptation params
    after the call.

``_<method>_chain_scan``
    Outer JIT wrapping burn-in + sampling only.  Used for the warm
    path (adaptation params cached from a prior run).  Replaces the
    old separate burnin_scan + sample_scan pair, halving the number of
    distinct JIT compilations.

``data_args`` is always a **traced** argument (never closed over) so
the compiled XLA program is galaxy-agnostic: switching to a new galaxy
with the same model never triggers recompilation.

``logdensity_fn_2arg`` is a **static** argument because JAX uses
function identity as part of the compilation cache key.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.metadata

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from packaging.requirements import Requirement

from tengri.config.exceptions import BackendError, DeadFitError
from tengri.inference._model_cache import _default_owner as _model_cache_owner

# ---------------------------------------------------------------------------
# BlackJAX version floor validation
# ---------------------------------------------------------------------------

#: Memoization flag to ensure _check_blackjax_floor() runs only once.
_blackjax_floor_checked = False


def _get_blackjax_floor_from_source():
    """Parse blackjax floor from source tree's pyproject.toml, if available.

    Resolves the source root through ``tengri._data_setup.package_data_dirs()``
    (#1431: no component anchors on parent-directory counting itself) and reads
    the pyproject.toml declaration. Returns the floor version string
    (e.g. "1.6") or None if no source pyproject found.

    Returns
    -------
    str or None
        The floor version extracted from pyproject.toml, or None if source
        is not available or blackjax is not declared.
    """
    try:
        # Locate the source root through the sanctioned seam (#1431: components
        # must not anchor on parent-directory counting themselves).
        from tengri._data_setup import package_data_dirs

        pyproject_path = None
        for candidate in package_data_dirs():
            if (candidate / "pyproject.toml").exists():
                pyproject_path = candidate / "pyproject.toml"
                break
        if pyproject_path is None:
            return None

        # Parse the pyproject.toml
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                # tomli not available; fall back to metadata
                return None

        with open(pyproject_path, "rb") as f:
            try:
                pyproject = tomllib.load(f)
            except Exception:
                # Unparseable pyproject; fall back to metadata
                return None

        # Check [project.optional-dependencies] nuts for blackjax
        optional_deps = pyproject.get("project", {}).get("optional-dependencies", {})
        nuts_deps = optional_deps.get("nuts", [])

        for dep_str in nuts_deps:
            req = Requirement(dep_str)
            if req.name == "blackjax":
                # Extract >= constraint
                for spec in req.specifier:
                    if spec.operator == ">=":
                        return spec.version
                break

        # Also check [project.dependencies] in case blackjax moved there
        deps = pyproject.get("project", {}).get("dependencies", [])
        for dep_str in deps:
            req = Requirement(dep_str)
            if req.name == "blackjax":
                for spec in req.specifier:
                    if spec.operator == ">=":
                        return spec.version
                break

        return None

    except Exception:
        # Any exception reading source: fall back to metadata
        return None


def _get_blackjax_floor_from_metadata():
    """Parse blackjax floor from dist metadata.

    Reads from importlib.metadata.requires("tengri") and extracts the
    blackjax>=X.Y constraint. Returns the floor version string or None
    if not found.

    Returns
    -------
    str or None
        The floor version extracted from metadata, or None if not found
        or metadata unavailable.
    """
    try:
        requires = importlib.metadata.requires("tengri")
        if requires is None:
            return None

        for req_str in requires:
            req = Requirement(req_str)
            if req.name == "blackjax":
                for spec in req.specifier:
                    if spec.operator == ">=":
                        return spec.version
                break

        return None

    except importlib.metadata.PackageNotFoundError:
        return None


def _check_blackjax_floor():
    """Validate blackjax version against the declared floor in pyproject.

    Derives the floor by checking (in order):
    1. The source tree's pyproject.toml (for editable installs)
    2. The dist metadata (for wheel installs)

    Compares against the installed version and raises with context on violation:
    the installed version, the declared floor, the consequence (frozen chains
    from silent blackjax API changes), and the exact remedy (pip install -U).

    Skips silently if the floor cannot be resolved: a floor you cannot read
    is not a floor you can enforce.

    Raises
    ------
    BackendError
        When installed blackjax version is below the declared floor.
    """
    global _blackjax_floor_checked
    if _blackjax_floor_checked:
        return
    _blackjax_floor_checked = True

    # Try to get floor from source tree first, then metadata
    floor_version = _get_blackjax_floor_from_source()
    if floor_version is None:
        floor_version = _get_blackjax_floor_from_metadata()

    if floor_version is None:
        # No floor found; skip silently
        return

    # Get the installed blackjax version
    try:
        installed_version_str = importlib.metadata.version("blackjax")
    except importlib.metadata.PackageNotFoundError:
        # blackjax not installed; let the normal import error surface
        return

    # Parse both as tuples of integers for comparison
    def parse_version(v_str):
        """Parse X.Y.Z string to tuple of ints for comparison."""
        try:
            return tuple(int(x) for x in v_str.split(".")[:3])
        except (ValueError, AttributeError):
            return None

    installed = parse_version(installed_version_str)
    floor = parse_version(floor_version)

    if installed is None or floor is None:
        # Unparseable version; let it slide
        return

    if installed < floor:
        raise BackendError(
            f"blackjax {installed_version_str} does not satisfy "
            f"blackjax>={floor_version}. Samplers on an unsupported blackjax can fail "
            f"silently: a frozen chain, not an error (issue #1999). "
            f"Remedy: pip install -U 'blackjax>={floor_version}'"
        )


# ---------------------------------------------------------------------------
# Kernel getters (cached in Python so we don't rebuild on every JIT call)
# ---------------------------------------------------------------------------


@functools.cache
def _get_nuts_kernel():
    """Build and cache the BlackJAX NUTS kernel."""
    _check_blackjax_floor()
    import blackjax.mcmc.nuts

    return blackjax.mcmc.nuts.build_kernel()


@functools.cache
def _get_hmc_kernel():
    """Build and cache the BlackJAX HMC kernel."""
    _check_blackjax_floor()
    import blackjax.mcmc.hmc

    return blackjax.mcmc.hmc.build_kernel()


@functools.cache
def _get_dynamic_hmc_kernel():
    """Build and cache the BlackJAX dynamic HMC kernel."""
    _check_blackjax_floor()
    import blackjax.mcmc.dynamic_hmc

    return blackjax.mcmc.dynamic_hmc.build_kernel()


@functools.cache
def _get_ghmc_kernel():
    """Build and cache the BlackJAX GHMC kernel."""
    _check_blackjax_floor()
    import blackjax.mcmc.ghmc

    return blackjax.mcmc.ghmc.build_kernel()


#: ELBO draws used by Pathfinder to pick the best Gaussian along the L-BFGS path,
#: when Pathfinder seeds the NUTS warmup. BlackJAX's ``pathfinder_adaptation`` calls
#: ``vi.pathfinder.approximate(key, logdensity, position)`` with no ``num_samples``,
#: so it silently takes the library default of 200. Each draw is a full forward-model
#: evaluation, and 200 of them vmapped drove a 7-parameter photometry fit past 25 GB
#: (OOM-killing the slow test tier). 25 matches Stan's ``num_elbo_draws``; path
#: selection is a low-precision decision, so the draws buy accuracy nowhere.
_PATHFINDER_ELBO_DRAWS = 25


@contextlib.contextmanager
def _bounded_pathfinder_elbo_draws(n_draws: int | None = None):
    """Cap the ELBO draws BlackJAX's ``pathfinder_adaptation`` uses internally.

    ``pathfinder_adaptation`` exposes no knob for them (its ``**extra_parameters``
    are forwarded to the sampling algorithm, not to ``approximate``), so the only
    seam is the module attribute it resolves at call time. Rebinding it for the
    duration of the warmup is scoped, restored on any exit, and touches nothing
    else -- ``approximate`` is invoked once, at trace time, from a single thread.

    Rebind ``blackjax.vi.pathfinder`` (the **module**), which is what
    ``pathfinder_adaptation`` resolves. ``blackjax.pathfinder`` is a separate
    ``GeneratePathfinderAPI`` instance holding the same function object; patching that
    one instead would be a silent no-op here.

    ``n_draws=None`` reads :data:`_PATHFINDER_ELBO_DRAWS` **at call time**, not at import,
    so a test may lower it by patching the module attribute.

    Remove this when BlackJAX lets the caller pass ``num_samples`` through.
    """
    from blackjax import vi

    draws = _PATHFINDER_ELBO_DRAWS if n_draws is None else n_draws
    original = vi.pathfinder.approximate

    @functools.wraps(original)
    def _capped(rng_key, logdensity_fn, initial_position, num_samples=draws, **kwargs):
        return original(rng_key, logdensity_fn, initial_position, num_samples, **kwargs)

    vi.pathfinder.approximate = _capped
    try:
        yield
    finally:
        vi.pathfinder.approximate = original


# ---------------------------------------------------------------------------
# NUTS
# ---------------------------------------------------------------------------
# static_argnums legend for every scan:
#   logdensity_fn_2arg → function identity is the JIT cache key
#   n_warmup           → lax.scan length inside window_adaptation
#   max_doublings      → NUTS tree parameter (compile-time constant)
#   n_burnin           → used to slice chain_keys at trace time
#   use_dense          → warmup constructor kwarg (bool)
#   target_accept_rate → warmup constructor kwarg (float, rarely changed)
#   use_pathfinder_warmup → picks pathfinder_adaptation vs window_adaptation (bool)

#: Default NUTS tree-depth cap, shared by every entry point (single-fit
#: ``run_nuts``, the catalog engine, the batched-vmap path, and the prewarm
#: compile) so the compiled program and the fit that follows cannot disagree.
#: Before this constant existed the prewarm path hardcoded its own ``10``,
#: correct only by coincidence with the signature defaults it never saw.
#:
#: The cap is 10, the BlackJAX/Stan convention, and measured to stay there. Lowering
#: the cap looks like a huge win on the heavy-tailed StudentT SFR-ratio
#: geometry of the nonparametric SFHs and is a trap: on a 19-band continuity
#: fit (D=9, 500 warmup + 500 samples, CPU, 2026-08-18) cap 6 cut the wall
#: 118 s → 11 s but collapsed min-ESS 93 → 5; per *effective* sample it is
#: strictly worse (1.99 vs 1.27 s/ESS). ``dense_mass_matrix=True`` was the
#: recommendation here and no longer is: re-measured it buys wall time at the
#: cost of 8.8 divergences per run against 3.3 for the diagonal. On that
#: geometry the genuine fixes are bin edges that stop at the age of the
#: universe (#1975) and a longer fixed-length trajectory
#: (``mcmc_hmc``, ``n_leapfrog_steps=150``). Saturation of a deep cap
#: is surfaced by ``NUTSTreeDepthWarning`` and the ``tree_depth_*``
#: diagnostics every NUTS fit now reports; a deliberate low cap for a
#: wall-bounded quick look is one kwarg, taken knowingly.
DEFAULT_MAX_NUM_DOUBLINGS = 10


def total_draws(diagnostics: dict, n_samples: int | None = None) -> int:
    """Kept draws across every chain of an MCMC result.

    Every backend records ``n_samples`` *per chain* and ``n_chains`` beside it,
    while ``n_divergent`` is summed over the flattened
    ``(n_chains * n_samples,)`` divergence record. Any comparison of a
    divergence count with a draw count must compare against this total, not
    ``n_samples``: the dead-fit guard's ``n_divergent == n_samples`` was false
    for every multi-chain run (2400 != 600 on a 4-chain fit) and
    ``convergence_check`` reported 400% divergences (#2087).

    Parameters
    ----------
    diagnostics : dict
        A ``Posterior.diagnostics`` mapping. ``n_chains`` defaults to 1 when
        absent (single-chain paths and hand-built posteriors).
    n_samples : int, optional
        Per-chain draw count to use instead of ``diagnostics["n_samples"]``,
        for callers that already resolved a fallback.
    """
    if n_samples is None:
        n_samples = diagnostics["n_samples"]
    return int(n_samples) * int(diagnostics.get("n_chains", 1))


#: Divergent fraction over the final window of warmup at which a
#: window-adaptation backend refuses to sample (#2088). Healthy fits at
#: target_accept 0.85 end warmup near 0; the heavy-tailed nonparametric-SFH
#: fits measured up to ~0.2; a posterior the sampler cannot enter measured 1.0.
DEAD_WARMUP_DIVERGENCE_FRAC = 0.9
#: The final window is this fraction of ``n_warmup`` ...
DEAD_WARMUP_WINDOW_FRAC = 0.1
#: ... and never fewer than this many steps.
DEAD_WARMUP_MIN_WINDOW = 10


def _dead_warmup_window(n_flags: int, n_warmup: int) -> int:
    return min(n_flags, max(DEAD_WARMUP_MIN_WINDOW, round(DEAD_WARMUP_WINDOW_FRAC * n_warmup)))


def final_window_divergence_frac(warmup_divergent, n_warmup: int) -> float | None:
    """Divergent fraction over the final window of the warmup record.

    Parameters
    ----------
    warmup_divergent : array_like of bool, shape (n_warmup,), or None
        Per-step ``is_divergent`` flags from window adaptation, or ``None``
        for a caller holding no record at all.
    n_warmup : int
        Warmup length the window is sized from.

    Returns
    -------
    float or None
        ``None`` when there is nothing to measure: no flags, an empty record
        (a warmup that ran no steps), or fewer steps than
        ``DEAD_WARMUP_MIN_WINDOW`` — a record that short cannot fill the
        minimum window and carries no verdict. BlackJAX opens dual averaging
        at ``mu = log(10 * step_size)``, so the first proposals are made at
        roughly twice the initial step size whatever the posterior and take
        five or six rejections to collapse; a sub-window record is that
        opening burst and nothing else, on a healthy posterior as much as on
        a dead one (#2088). Callers must treat ``None`` as "not measured"
        rather than as a fraction of zero — the backends omit the
        ``warmup_divergence_frac`` diagnostic entirely in that case.
    """
    if warmup_divergent is None:
        return None
    flags = np.asarray(warmup_divergent, dtype=bool)
    if flags.size < DEAD_WARMUP_MIN_WINDOW:
        return None
    window = _dead_warmup_window(flags.size, n_warmup)
    return float(flags[-window:].mean())


def refuse_dead_warmup(
    frac: float | None, *, sampler: str, step_size: float, n_warmup: int, n_samples: int
) -> None:
    """Raise :class:`DeadFitError` when the final warmup window is (nearly) all divergent.

    The refusal seam of #2088: NUTS, HMC and dynamic HMC call it once warmup
    has returned, before the adaptation is cached and before the sampling
    scan compiles. A ``frac`` of ``None`` means nothing was measured, so
    there is nothing to refuse on, and it returns quietly. The backends do
    not call this at all when they reuse a cached adaptation.
    """
    if frac is None or frac < DEAD_WARMUP_DIVERGENCE_FRAC:
        return

    window = _dead_warmup_window(n_warmup, n_warmup)
    raise DeadFitError(
        f"{sampler} warmup ended dead: {frac:.0%} of its final {window} adaptation steps "
        f"diverged at the adapted step size {step_size:.3g}, so the sampler rejects "
        f"essentially every proposal and {n_samples} draws would only return a frozen "
        f"posterior. Sampling was refused and the adaptation was not cached. This is a "
        f"posterior problem, not a tuning one: the measured trigger was data 1000x too "
        f"faint (a wrong AB zero point) that pushed the stellar mass to its prior edge, "
        f"where the bounded transform runs to infinity. Check the data units and scale, "
        f"the prior bounds against the MAP initialization, and that the initial log "
        f"posterior is finite, before re-tuning.",
        warmup_divergence_frac=frac,
        step_size=step_size,
    )


@functools.partial(jax.jit, static_argnums=(3, 5, 6, 7, 8, 9))
def _nuts_full_scan(
    init_flat,
    warmup_key,
    chain_keys,
    logdensity_fn_2arg,
    data_args,
    n_warmup,
    max_doublings,
    use_dense,
    target_accept_rate,
    use_pathfinder_warmup: bool = False,
):
    """Outer JIT: BlackJAX NUTS window adaptation + burn-in + sampling.

    Compiles the NUTS tree-building kernel once for the full chain.
    Returns adaptation params so the caller can populate the Python-side
    cache after the call.

    Parameters
    ----------
    init_flat : ndarray, shape (D,)
        Initial position in unbounded latent space.
    warmup_key : PRNGKey
        Random key for window adaptation.
    chain_keys : ndarray, shape (n_chain, 2)
        Pre-split keys for the chain. ``n_chain = n_burnin + n_samples``;
        burnin is discarded by the *caller* via Python slicing rather
        than inside JIT, so changing ``n_burnin`` while keeping
        ``n_chain`` constant does not trigger recompilation.
    logdensity_fn_2arg : callable (static)
        ``log_p(position, data_args)``, galaxy-agnostic log-posterior.
    data_args : pytree (traced)
        Observed data tensors; changing these does NOT trigger recompilation.
    n_warmup : int (static)
        Window adaptation steps.
    max_doublings : int (static)
        Maximum NUTS tree doublings.
    use_dense : bool (static)
        Dense vs diagonal mass matrix. Ignored when ``use_pathfinder_warmup``
        is True (Pathfinder always returns a full inverse-covariance matrix
        from its L-BFGS Hessian approximation).
    target_accept_rate : float (static)
        Target acceptance rate for dual averaging.
    use_pathfinder_warmup : bool (static)
        When True, use ``blackjax.adaptation.pathfinder_adaptation`` in
        place of ``blackjax.window_adaptation``. Pathfinder runs L-BFGS
        to locate the posterior mode and derives an inverse mass matrix
        from the Hessian approximation, followed by a short dual-averaging
        step-size refinement. Typically 3-10x faster than window adaptation
        on high-dimensional problems (D>~30). Default False.

    Returns
    -------
    positions : ndarray, shape (n_chain, D)
    divergent : ndarray, shape (n_chain,)
    expansions : ndarray, shape (n_chain,)
        Per-iteration NUTS trajectory-expansion count (tree depth). All
        three include the burnin prefix; caller slices ``[n_burnin:]``.
    step_size : scalar
    inv_mass_matrix : ndarray, shape (D,) or (D, D)
    """
    import blackjax

    def ld_1arg(pos):
        return logdensity_fn_2arg(pos, data_args)

    # blackjax retains every warmup step's adaptation info by default, and warns
    # that this "can result in excessive memory usage if the information is
    # unused". We discard it below, so keep nothing (#1028).
    from blackjax.adaptation.base import get_filter_adapt_info_fn

    _drop_adapt_info = get_filter_adapt_info_fn()

    if use_pathfinder_warmup:
        from blackjax.adaptation.pathfinder_adaptation import pathfinder_adaptation

        warmup = pathfinder_adaptation(
            blackjax.nuts,
            ld_1arg,
            target_acceptance_rate=target_accept_rate,
            adaptation_info_fn=_drop_adapt_info,
        )
        with _bounded_pathfinder_elbo_draws():
            (state, parameters), _ = warmup.run(warmup_key, init_flat, num_steps=n_warmup)
    else:
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            ld_1arg,
            is_mass_matrix_diagonal=not use_dense,
            target_acceptance_rate=target_accept_rate,
            adaptation_info_fn=_drop_adapt_info,
        )
        (state, parameters), _ = warmup.run(warmup_key, init_flat, num_steps=n_warmup)
    step_size = parameters["step_size"]
    inv_mass_matrix = parameters["inverse_mass_matrix"]

    kernel = _get_nuts_kernel()

    def _step(s, k):
        """Advance NUTS one step: position, divergence flag, tree depth."""
        s, info = kernel(k, s, ld_1arg, step_size, inv_mass_matrix, max_doublings)
        return s, (s.position, info.is_divergent, info.num_trajectory_expansions)

    _, (positions, divergent, expansions) = jax.lax.scan(_step, state, chain_keys)
    return positions, divergent, expansions, step_size, inv_mass_matrix


@functools.partial(jax.jit, static_argnums=(2, 4, 5, 6, 7))
def _nuts_warmup_only(
    init_flat,
    warmup_key,
    logdensity_fn_2arg,
    data_args,
    n_warmup,
    use_dense,
    target_accept_rate,
    use_pathfinder_warmup: bool = False,
):
    """BlackJAX NUTS window adaptation only, returns tuned (step_size, inv_mass).

    The warmup half of :func:`_nuts_full_scan`, split out for the same reason
    :func:`_hmc_warmup_only` was: so the fresh and cached-adaptation paths end
    in the *same* sampling call. While warmup and sampling were fused here, a
    first fit ran ``_nuts_full_scan`` and every later fit on the same model ran
    a sampling-only scan against the cached parameters, structurally different
    computations, so one pinned ``key`` produced two different posteriors. HMC
    had already been split this way and was reproducible; NUTS was not.

    Same static-arg / traced-``data_args`` contract as :func:`_nuts_full_scan`.

    Returns
    -------
    step_size : scalar
    inv_mass_matrix : ndarray, shape (D,) or (D, D)
    warmup_divergent : ndarray of bool, shape (n_warmup,)
        Per-step ``is_divergent`` flags from the adaptation, for the
        dead-warmup refusal (#2088). Both the window and the pathfinder
        adaptation report it.
    """
    import blackjax

    def ld_1arg(pos):
        return logdensity_fn_2arg(pos, data_args)

    # Keep only the per-step divergence flags (n_warmup bools) for the
    # dead-warmup refusal (#2088); everything else blackjax would retain is the
    # memory cost #1028 removed.
    from blackjax.adaptation.base import get_filter_adapt_info_fn

    _keep_divergence_flags = get_filter_adapt_info_fn(info_keys={"is_divergent"})

    if use_pathfinder_warmup:
        from blackjax.adaptation.pathfinder_adaptation import pathfinder_adaptation

        warmup = pathfinder_adaptation(
            blackjax.nuts,
            ld_1arg,
            target_acceptance_rate=target_accept_rate,
            adaptation_info_fn=_keep_divergence_flags,
        )
        with _bounded_pathfinder_elbo_draws():
            (_, parameters), info = warmup.run(warmup_key, init_flat, num_steps=n_warmup)
    else:
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            ld_1arg,
            is_mass_matrix_diagonal=not use_dense,
            target_acceptance_rate=target_accept_rate,
            adaptation_info_fn=_keep_divergence_flags,
        )
        (_, parameters), info = warmup.run(warmup_key, init_flat, num_steps=n_warmup)

    warmup_divergent = jnp.asarray(info.info.is_divergent)
    return parameters["step_size"], parameters["inverse_mass_matrix"], warmup_divergent


@functools.partial(jax.jit, static_argnums=(2, 6))
def _nuts_chain_scan(
    state,
    chain_keys,
    logdensity_fn_2arg,
    data_args,
    step_size,
    inv_mass_matrix,
    max_doublings,
):
    """Outer JIT: NUTS burn-in + sampling with pre-computed adaptation params.

    Used when adaptation params are already cached.  Combines the old
    ``_nuts_burnin_scan`` + ``_nuts_sample_scan`` into a single compiled
    program.

    Parameters
    ----------
    state : NUTSState
        Initial chain state (from ``blackjax.mcmc.nuts.init``).
    chain_keys : ndarray, shape (n_chain, 2)
        Pre-split keys; caller slices ``[n_burnin:]`` Python-side.
    logdensity_fn_2arg : callable (static)
    data_args : pytree (traced)
    step_size : scalar (traced)
    inv_mass_matrix : ndarray (traced)
    max_doublings : int (static)

    Returns
    -------
    positions : ndarray, shape (n_chain, D)
    divergent : ndarray, shape (n_chain,)
    expansions : ndarray, shape (n_chain,)
        Per-iteration NUTS trajectory-expansion count (tree depth).
        Caller slices ``[n_burnin:]`` on all three.
    """

    def ld(pos):
        return logdensity_fn_2arg(pos, data_args)

    kernel = _get_nuts_kernel()

    def _step(s, k):
        """Advance NUTS one step: position, divergence flag, tree depth."""
        s, info = kernel(k, s, ld, step_size, inv_mass_matrix, max_doublings)
        return s, (s.position, info.is_divergent, info.num_trajectory_expansions)

    _, (positions, divergent, expansions) = jax.lax.scan(_step, state, chain_keys)
    return positions, divergent, expansions


# ---------------------------------------------------------------------------
# HMC
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(2, 4, 5, 6, 7))
def _hmc_warmup_only(
    init_flat,
    warmup_key,
    logdensity_fn_2arg,
    data_args,
    n_warmup,
    n_leapfrog,
    use_dense,
    target_accept_rate,
):
    """BlackJAX HMC window adaptation only, returns tuned (step_size, inv_mass).

    The warmup half of :func:`_hmc_full_scan`, split out so a *fresh*
    ``run_hmc`` call can adapt once and then dispatch sampling through the
    single- or multi-chain path (vmap/parallel) that honors ``n_chains``,
    without the old behavior of silently sampling a single chain on the first
    call. Same static-arg / traced-``data_args`` contract as
    :func:`_hmc_full_scan`.

    Returns
    -------
    step_size : scalar
    inv_mass_matrix : ndarray, shape (D,) or (D, D)
    warmup_divergent : ndarray of bool, shape (n_warmup,)
        Per-step ``is_divergent`` flags from the adaptation, for the
        dead-warmup refusal (#2088).
    """
    import blackjax
    from blackjax.adaptation.base import get_filter_adapt_info_fn

    def ld_1arg(pos):
        return logdensity_fn_2arg(pos, data_args)

    warmup = blackjax.window_adaptation(
        blackjax.hmc,
        ld_1arg,
        is_mass_matrix_diagonal=not use_dense,
        target_acceptance_rate=target_accept_rate,
        num_integration_steps=n_leapfrog,
        adaptation_info_fn=get_filter_adapt_info_fn(info_keys={"is_divergent"}),
    )
    (_, parameters), info = warmup.run(warmup_key, init_flat, num_steps=n_warmup)
    warmup_divergent = jnp.asarray(info.info.is_divergent)
    return parameters["step_size"], parameters["inverse_mass_matrix"], warmup_divergent


@functools.partial(jax.jit, static_argnums=(3, 5, 6, 7, 8))
def _hmc_full_scan(
    init_flat,
    warmup_key,
    chain_keys,
    logdensity_fn_2arg,
    data_args,
    n_warmup,
    n_leapfrog,
    use_dense,
    target_accept_rate,
):
    """Outer JIT: BlackJAX HMC window adaptation + sampling chain.

    Wraps warmup and the leapfrog scan in a single ``jax.jit`` so the
    HMC kernel is compiled once. Burnin discard is done by the caller
    Python-side, so changing ``n_burnin`` while keeping ``n_chain``
    constant does not trigger recompilation.

    Parameters
    ----------
    init_flat : ndarray, shape (D,)
        Initial position in unbounded latent space.
    warmup_key : PRNGKey
        Random key for warmup adaptation.
    chain_keys : ndarray, shape (n_chain, 2)
        Pre-split keys (``n_chain = n_burnin + n_samples``).
    logdensity_fn_2arg : callable (static)
        ``log_p(position, data_args)``, galaxy-agnostic log-posterior.
    data_args : pytree (traced)
        Observed data tensors; changing these does NOT trigger recompilation.
    n_warmup : int (static)
        Window adaptation steps.
    n_leapfrog : int (static)
        Leapfrog integration steps per HMC proposal.
    use_dense : bool (static)
        Dense vs diagonal mass matrix.
    target_accept_rate : float (static)
        Target acceptance rate for dual averaging.

    Returns
    -------
    positions : ndarray, shape (n_chain, D)
    divergent : ndarray, shape (n_chain,)
        Caller slices ``[n_burnin:]``.
    step_size : scalar
    inv_mass_matrix : ndarray, shape (D,) or (D, D)
    """
    import blackjax

    def ld_1arg(pos):
        return logdensity_fn_2arg(pos, data_args)

    warmup = blackjax.window_adaptation(
        blackjax.hmc,
        ld_1arg,
        is_mass_matrix_diagonal=not use_dense,
        target_acceptance_rate=target_accept_rate,
        num_integration_steps=n_leapfrog,
    )
    (state, parameters), _ = warmup.run(warmup_key, init_flat, num_steps=n_warmup)
    step_size = parameters["step_size"]
    inv_mass_matrix = parameters["inverse_mass_matrix"]

    kernel = _get_hmc_kernel()

    def _step(s, k):
        """Advance HMC sampler by one step, returning position and divergence flag."""
        s, info = kernel(k, s, ld_1arg, step_size, inv_mass_matrix, n_leapfrog)
        return s, (s.position, info.is_divergent)

    _, (positions, divergent) = jax.lax.scan(_step, state, chain_keys)
    return positions, divergent, step_size, inv_mass_matrix


@functools.partial(jax.jit, static_argnums=(2, 6))
def _hmc_chain_scan(
    state,
    chain_keys,
    logdensity_fn_2arg,
    data_args,
    step_size,
    inv_mass_matrix,
    n_leapfrog,
):
    """Outer JIT: HMC burn-in + sampling with pre-computed adaptation params.

    Parameters
    ----------
    state : HMCState
        Initial chain state (from ``blackjax.mcmc.hmc.init``).
    chain_keys : ndarray, shape (n_chain, 2)
        Pre-split keys; caller slices ``[n_burnin:]`` Python-side.
    logdensity_fn_2arg : callable (static)
        ``log_p(position, data_args)``.
    data_args : pytree (traced)
        Observed data tensors.
    step_size : scalar (traced)
        Step size from warmup adaptation.
    inv_mass_matrix : ndarray (traced), shape (D,) or (D, D)
        Inverse mass matrix from warmup adaptation.
    n_leapfrog : int (static)
        Leapfrog integration steps per proposal.

    Returns
    -------
    positions : ndarray, shape (n_chain, D)
    divergent : ndarray, shape (n_chain,)
        Caller slices ``[n_burnin:]``.
    """

    def ld(pos):
        return logdensity_fn_2arg(pos, data_args)

    kernel = _get_hmc_kernel()

    def _step(s, k):
        """Advance HMC sampler by one step, returning position and divergence flag."""
        s, info = kernel(k, s, ld, step_size, inv_mass_matrix, n_leapfrog)
        return s, (s.position, info.is_divergent)

    _, (positions, divergent) = jax.lax.scan(_step, state, chain_keys)
    return positions, divergent


# ---------------------------------------------------------------------------
# Dynamic HMC
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(4, 6, 7, 8))
def _dynamic_hmc_full_scan(
    init_flat,
    warmup_key,
    dhmc_init_key,
    chain_keys,
    logdensity_fn_2arg,
    data_args,
    n_warmup,
    use_dense,
    target_accept_rate,
):
    """Outer JIT: HMC warmup + dynamic HMC init + sampling chain.

    Uses HMC window adaptation to tune step size and mass matrix, then
    initializes a dynamic HMC state inside the same JIT so the kernel is
    compiled once. Burnin discard is done by the caller Python-side, so
    changing ``n_burnin`` while keeping ``n_chain`` constant does not
    trigger recompilation.

    Parameters
    ----------
    init_flat : ndarray, shape (D,)
        Initial position in unbounded latent space.
    warmup_key : PRNGKey
        Random key for HMC window adaptation.
    dhmc_init_key : PRNGKey (traced)
        Random key for ``dynamic_hmc.init`` (requires a random generator arg).
    chain_keys : ndarray, shape (n_chain, 2)
        Pre-split keys; caller slices ``[n_burnin:]`` Python-side.
    logdensity_fn_2arg : callable (static)
        ``log_p(position, data_args)``, galaxy-agnostic log-posterior.
    data_args : pytree (traced)
        Observed data tensors; changing these does NOT trigger recompilation.
    n_warmup : int (static)
        Window adaptation steps.
    use_dense : bool (static)
        Dense vs diagonal mass matrix for HMC warmup.
    target_accept_rate : float (static)
        Target acceptance rate for dual averaging.

    Returns
    -------
    positions : ndarray, shape (n_chain, D)
    divergent : ndarray, shape (n_chain,)
        Caller slices ``[n_burnin:]``.
    step_size : scalar
    inv_mass_matrix : ndarray, shape (D,) or (D, D)
    """
    import blackjax

    def ld_1arg(pos):
        return logdensity_fn_2arg(pos, data_args)

    warmup = blackjax.window_adaptation(
        blackjax.hmc,
        ld_1arg,
        is_mass_matrix_diagonal=not use_dense,
        target_acceptance_rate=target_accept_rate,
        num_integration_steps=10,
    )
    (_, parameters), _ = warmup.run(warmup_key, init_flat, num_steps=n_warmup)
    step_size = parameters["step_size"]
    inv_mass_matrix = parameters["inverse_mass_matrix"]

    state = blackjax.mcmc.dynamic_hmc.init(init_flat, ld_1arg, dhmc_init_key)
    kernel = _get_dynamic_hmc_kernel()

    def _step(s, k):
        """Advance dynamic HMC sampler by one step, returning position and divergence flag."""
        s, info = kernel(k, s, ld_1arg, step_size, inv_mass_matrix)
        return s, (s.position, info.is_divergent)

    _, (positions, divergent) = jax.lax.scan(_step, state, chain_keys)
    return positions, divergent, step_size, inv_mass_matrix


@functools.partial(jax.jit, static_argnums=(2,))
def _dynamic_hmc_chain_scan(
    state,
    chain_keys,
    logdensity_fn_2arg,
    data_args,
    step_size,
    inv_mass_matrix,
):
    """Outer JIT: dynamic HMC sampling with pre-computed adaptation params.

    Parameters
    ----------
    state : DynamicHMCState
        Initial chain state (from ``blackjax.mcmc.dynamic_hmc.init``).
    chain_keys : ndarray, shape (n_chain, 2)
        Pre-split keys; caller slices ``[n_burnin:]`` Python-side.
    logdensity_fn_2arg : callable (static)
        ``log_p(position, data_args)``.
    data_args : pytree (traced)
        Observed data tensors.
    step_size : scalar (traced)
        Step size from HMC window adaptation.
    inv_mass_matrix : ndarray (traced), shape (D,) or (D, D)
        Inverse mass matrix from HMC window adaptation.

    Returns
    -------
    positions : ndarray, shape (n_chain, D)
    divergent : ndarray, shape (n_chain,)
        Caller slices ``[n_burnin:]``.
    """

    def ld(pos):
        return logdensity_fn_2arg(pos, data_args)

    kernel = _get_dynamic_hmc_kernel()

    def _step(s, k):
        """Advance dynamic HMC sampler by one step, returning position and divergence flag."""
        s, info = kernel(k, s, ld, step_size, inv_mass_matrix)
        return s, (s.position, info.is_divergent)

    _, (positions, divergent) = jax.lax.scan(_step, state, chain_keys)
    return positions, divergent


# ---------------------------------------------------------------------------
# GHMC
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(4, 6, 7, 8, 9))
def _ghmc_full_scan(
    init_flat,
    warmup_key,
    ghmc_init_key,
    chain_keys,
    logdensity_fn_2arg,
    data_args,
    n_warmup,
    target_accept_rate,
    alpha,
    delta,
):
    """Outer JIT: HMC warmup + GHMC init + sampling chain.

    GHMC requires a diagonal mass matrix (momentum generator constraint),
    so HMC warmup always uses diagonal regardless of the ``dense_mass_matrix``
    flag. Returns step size and diagonal momentum inverse scale. Burnin
    discard happens caller-side.

    Parameters
    ----------
    init_flat : ndarray, shape (D,)
        Initial position in unbounded latent space.
    warmup_key : PRNGKey
        Random key for HMC window adaptation.
    ghmc_init_key : PRNGKey (traced)
        Random key for ``ghmc.init`` momentum initialization.
    chain_keys : ndarray, shape (n_chain, 2)
        Pre-split keys; caller slices ``[n_burnin:]`` Python-side.
    logdensity_fn_2arg : callable (static)
        ``log_p(position, data_args)``, galaxy-agnostic log-posterior.
    data_args : pytree (traced)
        Observed data tensors; changing these does NOT trigger recompilation.
    n_warmup : int (static)
        HMC window adaptation steps.
    target_accept_rate : float (static)
        Target acceptance rate for HMC dual averaging.
    alpha : float (static)
        Momentum persistence (0=full refresh, 1=no refresh).
    delta : float (static)
        Step size scaling in the GHMC proposal.

    Returns
    -------
    positions : ndarray, shape (n_chain, D)
    divergent : ndarray, shape (n_chain,)
        Caller slices ``[n_burnin:]``.
    step_size : scalar
    momentum_inv_scale : ndarray, shape (D,)
    """
    import blackjax

    def ld_1arg(pos):
        return logdensity_fn_2arg(pos, data_args)

    warmup = blackjax.window_adaptation(
        blackjax.hmc,
        ld_1arg,
        is_mass_matrix_diagonal=True,
        target_acceptance_rate=target_accept_rate,
        num_integration_steps=10,
    )
    (_, parameters), _ = warmup.run(warmup_key, init_flat, num_steps=n_warmup)
    step_size = parameters["step_size"]
    momentum_inv_scale = parameters["inverse_mass_matrix"]

    # Keyword args: blackjax reordered ghmc.init's (rng_key, logdensity_fn)
    # between 1.3 and 1.6, keywords are correct on both.
    state = blackjax.mcmc.ghmc.init(
        position=init_flat, logdensity_fn=ld_1arg, rng_key=ghmc_init_key
    )
    kernel = _get_ghmc_kernel()

    def _step(s, k):
        """Advance GHMC sampler by one step, returning position and divergence flag."""
        s, info = kernel(k, s, ld_1arg, step_size, momentum_inv_scale, alpha, delta)
        return s, (s.position, info.is_divergent)

    _, (positions, divergent) = jax.lax.scan(_step, state, chain_keys)
    return positions, divergent, step_size, momentum_inv_scale


@functools.partial(jax.jit, static_argnums=(2, 8, 9))
def _ghmc_chain_scan(
    state,
    chain_keys,
    logdensity_fn_2arg,
    data_args,
    step_size,
    momentum_inv_scale,
    alpha,
    delta,
    alpha_static,
    delta_static,
):
    """Outer JIT: GHMC sampling chain with pre-computed adaptation params.

    Parameters
    ----------
    state : GHMCState
        Initial chain state (from ``blackjax.mcmc.ghmc.init``).
    chain_keys : ndarray, shape (n_chain, 2)
        Pre-split keys; caller slices ``[n_burnin:]`` Python-side.
    logdensity_fn_2arg : callable (static)
        ``log_p(position, data_args)``.
    data_args : pytree (traced)
        Observed data tensors.
    step_size : scalar (traced)
        Step size from HMC window adaptation.
    momentum_inv_scale : ndarray (traced), shape (D,)
        Diagonal inverse mass matrix from HMC window adaptation.
    alpha : float (traced)
        Momentum persistence passed to the GHMC kernel.
    delta : float (traced)
        Step size scaling passed to the GHMC kernel.
    alpha_static : float (static)
        Mirrors ``alpha`` as a static arg; belongs in the XLA cache key
        because it controls the momentum-refresh geometry. Pass the same value.
    delta_static : float (static)
        Mirrors ``delta`` as a static arg. Pass the same value as ``delta``.

    Returns
    -------
    positions : ndarray, shape (n_chain, D)
    divergent : ndarray, shape (n_chain,)
        Caller slices ``[n_burnin:]``.
    """

    def ld(pos):
        return logdensity_fn_2arg(pos, data_args)

    kernel = _get_ghmc_kernel()

    def _step(s, k):
        """Advance GHMC sampler by one step, returning position and divergence flag."""
        s, info = kernel(k, s, ld, step_size, momentum_inv_scale, alpha, delta)
        return s, (s.position, info.is_divergent)

    _, (positions, divergent) = jax.lax.scan(_step, state, chain_keys)
    return positions, divergent


# ---------------------------------------------------------------------------
# GHMC / MEADS -- the adaptation BlackJAX ships *for* generalized HMC
# ---------------------------------------------------------------------------

#: Default adaptation-ensemble size when the caller says ``"auto"``.
#:
#: MEADS (Hoffman & Sountsov 2022, Algorithm 3) derives the step size and the
#: damping from **cross-chain** statistics: a maximum-eigenvalue estimate of the
#: preconditioned gradient matrix and of the centered position matrix, taken over
#: the chains in one fold. Both estimators are ratios of sums over the ensemble,
#: so their variance is set by the *number of chains per fold*, not by the number
#: of warmup steps. An ensemble of four chains split into four folds leaves one
#: chain per fold, and both estimators are then undefined.
#:
#: 32 with the paper's ``num_folds=4`` gives eight chains per fold, the smallest
#: size that estimates a D <= 8 covariance's leading eigenvalue at all. The paper
#: itself runs 128. This is a floor chosen for a single-galaxy fit on a laptop,
#: not an optimum: raise it when the posterior is wider than the ensemble can
#: resolve.
_MEADS_DEFAULT_ENSEMBLE = 32

#: Smallest chains-per-fold MEADS is allowed to run with.
#:
#: Two is the absolute floor for ``meads_adaptation``'s ``maximum_eigenvalue``,
#: whose unbiased ratio estimator divides by ``n * (n - 1)`` over the chains in a
#: fold: ``n = 1`` is a division by zero and ``n = 2`` is a single off-diagonal
#: term. Anything at or near that floor is refused rather than silently run,
#: because the failure mode is not a crash. It is an adapted step size drawn from
#: noise, which is indistinguishable from the hand-set constants MEADS replaced.
_MEADS_MIN_CHAINS_PER_FOLD = 4

#: Ensemble dispersion around the seed position [dimensionless, latent units].
#:
#: Deliberately ~500x :func:`_vmap_chains`'s ``jitter_scale=1e-3``, and the two
#: are not interchangeable. ``_vmap_chains`` jitters only to decorrelate chains
#: that already have a tuned step size; MEADS *reads the ensemble spread as the
#: posterior scale*. Seeded from a 1e-3 ball, the per-fold ``position_sigma`` is
#: 1e-3, the preconditioned gradients are 1e-3 of their true size, and MEADS's
#: ``min(0.5 / sqrt(lambda_max), 1.0)`` saturates at the 1.0 cap on the first
#: step: a step size chosen by the clamp rather than by the posterior. BlackJAX's
#: own docstring makes the same point from the other end ("use a dispersed,
#: full-rank initialization"; a rank-deficient one measured R-hat ~ 5). 0.5 is
#: O(1) in a latent space whose priors are unit-scaled by construction.
_MEADS_JITTER_SCALE = 0.5


def _resolve_meads_ensemble(n_ensemble, n_chains, n_folds):
    """Resolve the MEADS adaptation-ensemble size to a concrete, legal int.

    Pure Python, evaluated before any trace, so the ensemble width stays a
    static shape.

    The ensemble is **not** the same axis as ``n_chains``. ``n_chains`` is how
    many chains' draws land in the returned ``Posterior`` (the
    :func:`_vmap_chains` axis); the ensemble is how many chains MEADS runs
    *during warmup* to estimate the cross-chain statistics it adapts from. They
    are reconciled rather than duplicated: the ensemble is always a superset, and
    the sampling chains are seeded from the first ``n_chains`` of the ensemble's
    warmed-up final states, so nothing is run twice and no warmup is discarded.

    Keeping them separate is what lets ``n_chains=1`` -- the signature default,
    and what every catalog fit uses -- still get a genuinely adapted step size.
    Tying the ensemble to ``n_chains`` would have made the default configuration
    the one case where MEADS degenerates to noise.

    Parameters
    ----------
    n_ensemble : int or ``"auto"``
        ``"auto"`` resolves to ``max(_MEADS_DEFAULT_ENSEMBLE, n_chains)``. An
        explicit int is honored except that it is still raised to ``n_chains``
        and rounded up to a multiple of ``n_folds``.
    n_chains : int
        Sampling chains; the ensemble can never be smaller.
    n_folds : int
        MEADS folds K. BlackJAX requires ``num_chains`` divisible by K, so the
        resolved size is rounded up.

    Returns
    -------
    int
        Ensemble size, a multiple of ``n_folds``.

    Raises
    ------
    ValueError
        If ``n_folds < 1``, if ``n_ensemble`` is a string other than ``"auto"``,
        or if an explicit ``n_ensemble`` is too small to give
        :data:`_MEADS_MIN_CHAINS_PER_FOLD` chains per fold. Refused, not clamped:
        silently growing an ensemble a caller pinned would hide the memory cost
        of the second vmap axis they were trying to bound.
    """
    n_folds = int(n_folds)
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    n_chains = max(1, int(n_chains))

    if isinstance(n_ensemble, str):
        if n_ensemble != "auto":
            raise ValueError(f"n_ensemble must be an int or 'auto', got {n_ensemble!r}")
        resolved = max(_MEADS_DEFAULT_ENSEMBLE, n_chains)
    else:
        resolved = int(n_ensemble)
        floor = n_folds * _MEADS_MIN_CHAINS_PER_FOLD
        if resolved < floor:
            raise ValueError(
                f"n_ensemble={resolved} gives {resolved / n_folds:.2g} chains per fold "
                f"across n_folds={n_folds}. MEADS adapts the step size and the damping "
                f"from cross-chain statistics within each fold, so fewer than "
                f"{_MEADS_MIN_CHAINS_PER_FOLD} chains per fold does not estimate them, "
                f"it fabricates them: BlackJAX's maximum-eigenvalue estimator divides "
                f"by n*(n-1) over the chains in a fold. Pass n_ensemble >= {floor} (the "
                f"default 'auto' gives {_MEADS_DEFAULT_ENSEMBLE}), or lower n_folds. "
                "This is separate from n_chains, which stays whatever you asked for."
            )
        resolved = max(resolved, n_chains)

    remainder = resolved % n_folds
    if remainder:
        resolved += n_folds - remainder
    return resolved


@functools.partial(jax.jit, static_argnums=(3, 5, 6, 7, 8, 9, 10, 11, 12))
def _ghmc_meads_scan(
    init_flat,
    warmup_key,
    chain_keys,
    logdensity_fn_2arg,
    data_args,
    n_warmup,
    n_ensemble,
    n_folds,
    alpha_override,
    delta_override,
    jitter_scale,
    low_rank_rank,
    low_rank_window_fraction,
):
    """Outer JIT: MEADS ensemble adaptation + GHMC sampling, one XLA program.

    Replaces the ``blackjax.window_adaptation`` path GHMC used to borrow from
    HMC. Window adaptation dual-averages a step size against a *target acceptance
    rate* for a reversible Metropolis step; generalized HMC has no such step (it
    uses a non-reversible slice update) and its mixing is governed by the damping
    ``alpha``, which window adaptation cannot see and therefore left at a
    hand-set constant. MEADS derives both from the ensemble.

    The warmup ensemble's final states are *reused* as the sampling chains'
    initial states rather than discarded and re-seeded from ``init_flat``. That
    is why this is one function and not two: MEADS has no separate warmup phase
    by construction, so throwing its ensemble away would mean paying for warmup
    and then starting cold anyway.

    Parameters
    ----------
    init_flat : ndarray, shape (D,)
        Seed position in unbounded latent space; the ensemble is dispersed
        around it [dimensionless].
    warmup_key : PRNGKey (traced)
        Split into the ensemble-dispersion key and the MEADS adaptation key.
    chain_keys : ndarray, shape (n_chains, n_iter, 2)
        Pre-split per-chain keys. ``n_chains`` is read off this array's leading
        axis and must not exceed ``n_ensemble``; the caller guarantees that via
        :func:`_resolve_meads_ensemble`.
    logdensity_fn_2arg : callable (static)
        ``log_p(position, data_args)``, the galaxy-agnostic log-posterior.
    data_args : pytree (traced)
        Observed data tensors; changing these does NOT trigger recompilation.
    n_warmup : int (static)
        MEADS adaptation steps. One leapfrog per step per ensemble chain, so the
        gradient budget is exactly ``n_warmup * n_ensemble`` and flat -- unlike a
        NUTS warmup, whose cost per step is posterior-dependent.
    n_ensemble : int (static)
        Adaptation-ensemble width; see :func:`_resolve_meads_ensemble`.
    n_folds : int (static)
        MEADS folds K.
    alpha_override, delta_override : float or None (static)
        ``None`` uses the adapted value [dimensionless]. A float pins it, which
        is what the old ``alpha=0.8, delta=0.65`` defaults did unconditionally.
    jitter_scale : float (static)
        Ensemble dispersion; see :data:`_MEADS_JITTER_SCALE` [dimensionless].
    low_rank_rank : int or None (static)
        MEADS-LRD. ``None`` (the default, and BlackJAX's) adapts the *diagonal*
        momentum metric from each fold. An int instead adapts a rank-``k``
        ``LowRankInverseMassMatrix`` from the pooled ensemble, which is the
        obvious thing to reach for given the 1e5-1e8 latent condition numbers
        ``inference/preconditioning.py`` documents. **Measured, and it does not
        help**: at ``rank = D`` (i.e. a full dense metric) nb05's best split-R-hat
        is 1.81 and nb01's 1.46, against a bar of 1.01 -- see
        ``bench/reports/2026-08-30_ghmc_meads_adaptation.md``. It is exposed so
        the experiment can be re-run without editing source, and it is off by
        default because a knob that does not help must not be a default.
    low_rank_window_fraction : float (static)
        Only read when ``low_rank_rank`` is set. Fraction of warmup, counted
        from the end, over which the low-rank covariance accumulates.

    Returns
    -------
    positions : ndarray, shape (n_chains, n_iter, D)
    divergent : ndarray, shape (n_chains, n_iter)
        Caller trims ``[:, n_burnin:]`` and flattens, matching
        :func:`_vmap_chains`.
    step_size : ndarray, shape ()
    momentum_inv_scale : ndarray, shape (D,)
        Adapted diagonal momentum scale. GHMC's momentum generator treats this as
        a diagonal vector, which is why GHMC is always diagonal-metric regardless
        of ``dense_mass_matrix``.
    alpha, delta : ndarray, shape ()
        The adapted (or overridden) damping and slice translation.

    Notes
    -----
    JIT: this *is* the jitted entry point. Not vmappable over galaxies as-is --
    the batched catalog path still uses :func:`_ghmc_full_scan`.

    References
    ----------
    .. [1] M. D. Hoffman and P. Sountsov, "Tuning-Free Generalized Hamiltonian
       Monte Carlo", Proceedings of the 25th International Conference on
       Artificial Intelligence and Statistics (AISTATS), PMLR 151:7799-7813,
       2022. https://proceedings.mlr.press/v151/hoffman22a.html
    """
    import blackjax
    from blackjax.adaptation.base import get_filter_adapt_info_fn

    def ld_1arg(pos):
        """Bind the traced data to the galaxy-agnostic log-posterior."""
        return logdensity_fn_2arg(pos, data_args)

    jitter_key, adapt_key = jax.random.split(warmup_key)
    ensemble = init_flat[None, :] + jitter_scale * jax.random.normal(
        jitter_key, (n_ensemble, init_flat.shape[0]), dtype=init_flat.dtype
    )

    # ``get_filter_adapt_info_fn()`` with no arguments keeps *nothing*. The
    # default, ``return_all_adapt_info``, stacks every ensemble state and every
    # HMCInfo for every warmup step -- n_warmup * n_ensemble * D floats plus the
    # momentum and the proposal, which at 300 x 32 is already larger than the
    # posterior being estimated, and is pure overhead here: the adapted values
    # come back in ``parameters``, not in the info.
    warmup = blackjax.meads_adaptation(
        ld_1arg,
        num_chains=n_ensemble,
        num_folds=n_folds,
        adaptation_info_fn=get_filter_adapt_info_fn(),
        low_rank_rank=low_rank_rank,
        low_rank_window_fraction=low_rank_window_fraction,
    )
    (last_states, parameters), _ = warmup.run(adapt_key, ensemble, num_steps=n_warmup)

    step_size = parameters["step_size"]
    momentum_inv_scale = parameters["momentum_inverse_scale"]
    alpha = parameters["alpha"] if alpha_override is None else jnp.asarray(alpha_override)
    delta = parameters["delta"] if delta_override is None else jnp.asarray(delta_override)

    n_chains = chain_keys.shape[0]
    states = jax.tree.map(lambda leaf: leaf[:n_chains], last_states)
    kernel = _get_ghmc_kernel()

    def _step(s, k):
        """Advance GHMC by one step, returning position and divergence flag."""
        s, info = kernel(k, s, ld_1arg, step_size, momentum_inv_scale, alpha, delta)
        return s, (s.position, info.is_divergent)

    def _chain(state, keys):
        """Run one chain's scan; vmapped over the sampling chains."""
        return jax.lax.scan(_step, state, keys)[1]

    positions, divergent = jax.vmap(_chain)(states, chain_keys)
    return positions, divergent, step_size, momentum_inv_scale, alpha, delta


# ---------------------------------------------------------------------------
# ChEES-HMC (cross-chain adapted trajectory length, lock-step preserved)
# ---------------------------------------------------------------------------

#: Default ChEES adaptation-ensemble width.
#:
#: ChEES's trajectory-length gradient is built from *cross-chain* centered
#: positions (``p - p.mean(axis=0)`` inside ``chees_adaptation``'s
#: ``compute_parameters``), so an ensemble of one centers to exactly zero and the
#: adapted length never moves off its initial value -- adapted in name only, and
#: silently. 32 matches :data:`_MEADS_DEFAULT_ENSEMBLE` so the two ensemble
#: samplers cost the same per warmup step and their rows stay comparable; the
#: ChEES paper itself runs hundreds.
_CHEES_DEFAULT_ENSEMBLE = 32

#: Smallest ensemble ChEES is allowed to run with.
#:
#: Below this the centered-position matrix is rank-deficient enough that the
#: trajectory-length gradient is dominated by its own sampling noise. Refused
#: rather than clamped, for the reason :func:`_resolve_meads_ensemble` states:
#: an adapted value drawn from noise is indistinguishable from a hand-set one,
#: and the whole claim of this backend is that the length is learned.
_CHEES_MIN_ENSEMBLE = 4

#: **Adaptation-ensemble** dispersion around the seed position [dimensionless].
#:
#: A criterion-estimation parameter, and nothing else. ChEES's criterion is the
#: change in the *cross-chain* expected square, so a dispersed ensemble already
#: carries a large expected square before the sampler moves anything: the
#: criterion is then dominated by the initial spread rather than by what the
#: trajectory achieved, and the optimizer settles on a shorter length. BlackJAX's
#: own ``chees_adaptation`` docstring records the same effect from the other end
#: ("dispersion inflates the cross-chain jump-distance criterion, driving the
#: adapted trajectory length down"). **This dial wants to be tight.**
#:
#: It is deliberately NOT :data:`_MEADS_JITTER_SCALE`, whose job is the opposite:
#: MEADS reads the ensemble spread *as* the posterior scale and needs an O(1)
#: dispersion to have a scale at all.
#:
#: It is also deliberately not :data:`_CHEES_CHAIN_JITTER_SCALE`, which is the
#: diagnostic dial. Collapsing the two is what makes "tight enough to adapt" and
#: "wide enough for R-hat to mean something" look like a trade-off; they are two
#: different chain sets and only one of them is measured by R-hat.
_CHEES_JITTER_SCALE = 0.1

#: **Sampling-chain** overdispersion around the seed position [dimensionless].
#:
#: A diagnostic parameter, and nothing else. Split R-hat only detects
#: non-convergence when its chains start overdispersed relative to the posterior;
#: chains started at one point can share a non-equilibrium basin and still score
#: a clean R-hat, which is the failure BlackJAX's docstring warns about
#: ("initializing all chains at a single point ... can produce clean R-hat that is
#: structurally blind to same-basin non-equilibrium"). **This dial wants to be
#: wide**, and it is free to be, because the sampling chains are not what the
#: ChEES criterion is estimated from.
#:
#: ``None`` (the default) does not use this at all: the sampling chains are
#: seeded from the adaptation ensemble's own warmed final states, which after
#: warmup are distributed roughly *according to the posterior* -- dispersed, but
#: not over-dispersed, and correlated with the ensemble that tuned them. A float
#: instead seeds them independently around the seed position, which is what makes
#: R-hat a real test rather than a consistency check.
_CHEES_CHAIN_JITTER_SCALE = 0.5

#: Adam learning rate on ChEES's ``log`` trajectory length [dimensionless].
#:
#: ``chees_adaptation.run`` takes an ``optax.GradientTransformation`` and has no
#: default. 0.05 is the value the ChEES authors' own reference implementation
#: uses (TFP's ``ChEESAdaptation`` examples); it is exposed as a parameter
#: because a learning rate with no default in the library is a knob someone will
#: eventually need to move, not a constant.
_CHEES_LEARNING_RATE = 0.05

#: Initial step size handed to ChEES's dual averaging [latent units].
#:
#: Only a starting point: ``chees_adaptation`` dual-averages it against
#: ``target_acceptance_rate`` over the whole warmup, so this sets where the
#: search starts, not where it lands. 0.1 is small enough not to divergence-storm
#: a D <= 30 latent posterior on the first step and large enough that dual
#: averaging is not still climbing when warmup ends.
_CHEES_INIT_STEP_SIZE = 0.1


def _resolve_chees_ensemble(n_ensemble, n_chains):
    """Resolve the ChEES adaptation-ensemble size to a concrete, legal int.

    Pure Python, evaluated before any trace, so the ensemble width stays a static
    shape.

    **The ensemble axis is chains-within-galaxy, not galaxies-within-batch.**
    That is the load-bearing decision of this backend and it is deliberate: ChEES
    adapts one trajectory length from the ensemble's pooled cross-chain
    statistics, so an ensemble spanning *galaxies* would tune a single ``L``
    against a mixture of different posteriors and -- worse -- would make each
    galaxy's draws depend on which other galaxies happened to share its batch.
    Per-galaxy posteriors must be independent of batch composition; only
    chains-within-galaxy keeps that true. The cost is that the ensemble is an
    inner vmap under any future outer galaxy vmap rather than a reuse of it.

    Like :func:`_resolve_meads_ensemble`, the ensemble is a **superset** of the
    sampling chains: ``n_ensemble`` chains adapt, and the first ``n_chains`` of
    their warmed-up final states are the ones whose draws land in the returned
    posterior. Nothing is run twice and no warmup is discarded. Keeping the two
    axes separate is what lets ``n_chains=1`` -- the signature default, and what
    every catalog fit uses -- still get a genuinely adapted trajectory length.

    Parameters
    ----------
    n_ensemble : int or ``"auto"``
        ``"auto"`` resolves to ``max(_CHEES_DEFAULT_ENSEMBLE, n_chains)``. An
        explicit int is honored except that it is still raised to ``n_chains``.
    n_chains : int
        Sampling chains; the ensemble can never be smaller.

    Returns
    -------
    int
        Ensemble size.

    Raises
    ------
    ValueError
        If ``n_ensemble`` is a string other than ``"auto"``, or if an explicit
        ``n_ensemble`` is below :data:`_CHEES_MIN_ENSEMBLE`.
    """
    n_chains = max(1, int(n_chains))
    if isinstance(n_ensemble, str):
        if n_ensemble != "auto":
            raise ValueError(f"n_ensemble must be an int or 'auto', got {n_ensemble!r}")
        return max(_CHEES_DEFAULT_ENSEMBLE, n_chains)

    resolved = int(n_ensemble)
    if resolved < _CHEES_MIN_ENSEMBLE:
        raise ValueError(
            f"n_ensemble={resolved} is below the floor of {_CHEES_MIN_ENSEMBLE}. ChEES "
            "adapts the trajectory length from cross-chain centered positions, so a "
            "one-chain ensemble centers to exactly zero and the length never moves off "
            "its initial value -- adapted in name only, and silently. Pass n_ensemble "
            f">= {_CHEES_MIN_ENSEMBLE} (the default 'auto' gives "
            f"{_CHEES_DEFAULT_ENSEMBLE}). This is separate from n_chains, which stays "
            "whatever you asked for."
        )
    return max(resolved, n_chains)


@functools.partial(jax.jit, static_argnums=(3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
def _chees_scan(
    init_flat,
    warmup_key,
    chain_keys,
    logdensity_fn_2arg,
    data_args,
    n_warmup,
    n_ensemble,
    n_chains,
    n_iter,
    jitter_scale,
    jitter_amount,
    target_accept_rate,
    max_leapfrog_steps,
    learning_rate,
    mass_matrix_estimation,
    chain_jitter=None,
):
    """Outer JIT: ChEES cross-chain adaptation + dynamic-HMC sampling, one program.

    ChEES-HMC [1]_ tunes the *trajectory length* by maximizing the Change in the
    Estimator of the Expected Square across an ensemble of chains, then jitters
    that one length per step with a quasi-random Halton sequence. Every chain
    still integrates for a number of leapfrog steps drawn from the same
    distribution at the same iteration, so the ensemble stays lock-step on an
    accelerator -- which is the whole reason to prefer it to NUTS, whose vmapped
    chains run at the speed of the deepest tree.

    Underneath is an ordinary ``dynamic_hmc`` kernel: a full Metropolis accept
    step and a dual-averaged step size. That matters, and it is the difference
    between this and :func:`_ghmc_meads_scan`. MEADS derives its momentum metric
    from the adapting ensemble's own per-fold standard deviation, which closes a
    loop -- wider ensemble to larger momentum to longer excursions to wider
    ensemble -- that acceptance cannot object to, because energy really is
    conserved under the same inflated metric that produced the excursions
    (measured: ``bench/reports/2026-08-30_ghmc_meads_adaptation.md``). ChEES with
    ``mass_matrix_estimation=None`` has no counterpart: the metric is the
    identity, fixed for the whole run, and the geometry comes from
    :mod:`tengri.inference.preconditioning`'s analytic ``J^T N^-1 J + I`` instead
    of from the ensemble. **Do not switch that default on lightly.**

    The warmup ensemble's final states are reused as the sampling chains' initial
    states rather than discarded, for the reason :func:`_ghmc_meads_scan` gives:
    the adaptation has no separate warmup phase to throw away.

    Parameters
    ----------
    init_flat : ndarray, shape (D,)
        Seed position in unbounded latent space; the ensemble is dispersed around
        it [dimensionless].
    warmup_key : PRNGKey (traced)
        Split into the ensemble-dispersion key and the ChEES adaptation key.
    chain_keys : ndarray, shape (n_chains, n_iter, 2)
        Pre-split per-chain sampling keys.
    logdensity_fn_2arg : callable (static)
        ``log_p(position, data_args)``, the galaxy-agnostic log-posterior.
    data_args : pytree (traced)
        Observed data tensors; changing these does NOT trigger recompilation.
    n_warmup : int (static)
        ChEES adaptation steps.
    n_ensemble : int (static)
        Adaptation-ensemble width; see :func:`_resolve_chees_ensemble`.
    n_chains, n_iter : int (static)
        Sampling chains and iterations per chain. Passed separately from
        ``chain_keys.shape`` because they also size the Halton sequence's bit
        budget, which is a Python-level ``np.log2`` and must be static.
    jitter_scale : float (static)
        **Adaptation-ensemble** dispersion; see :data:`_CHEES_JITTER_SCALE`
        [dimensionless]. A criterion-estimation dial, and it wants to be tight.
    jitter_amount : float (static)
        Fraction of the adapted trajectory length that is jittered, in ``[0, 1]``
        [dimensionless]. BlackJAX's default of 1.0 draws each step's length
        uniformly from ``(0, L]``.
    target_accept_rate : float (static)
        Dual-averaging target [dimensionless]. ChEES's own default is **0.651**,
        not NUTS's 0.8: the optimal acceptance rate for a *fixed*-length HMC
        proposal, which is what each ChEES step is.
    max_leapfrog_steps : int (static)
        Hard cap on the leapfrog count per proposal. Bounds the worst case of a
        warmup step, whose cost is ``n_ensemble * L`` gradients.
    learning_rate : float (static)
        Adam step on ``log`` trajectory length; see :data:`_CHEES_LEARNING_RATE`.
    mass_matrix_estimation : str or None (static)
        ``None`` (the default, and BlackJAX's) pins ``inverse_mass_matrix`` to
        ones for the whole run. ``"diagonal"`` instead estimates it from the
        ensemble -- see the warning in :func:`run_chees`.
    chain_jitter : float or None (static)
        **Sampling-chain** overdispersion; see
        :data:`_CHEES_CHAIN_JITTER_SCALE` [dimensionless]. ``None`` seeds the
        sampling chains from the adaptation ensemble's warmed final states. A
        float instead seeds them independently around ``init_flat``, which
        decouples the diagnostic dial from the criterion dial: the ensemble can
        be tight enough to adapt a long trajectory while the chains R-hat
        actually sees are overdispersed enough for R-hat to be a test rather
        than a consistency check.

    Returns
    -------
    positions : ndarray, shape (n_chains, n_iter, D)
    divergent : ndarray, shape (n_chains, n_iter)
        Caller trims ``[:, n_burnin:]`` and flattens, matching
        :func:`_vmap_chains`.
    step_size : ndarray, shape ()
        Dual-averaged step size [latent units].
    inverse_mass_matrix : ndarray, shape (D,)
        Ones unless ``mass_matrix_estimation="diagonal"``.
    n_leapfrog : ndarray, shape ()
        The adapted trajectory length in leapfrog steps, before jitter. This is
        the number the phase exists to learn.

    Notes
    -----
    JIT: this *is* the jitted entry point. Not vmappable over galaxies as-is --
    the batched catalog path is Phase 3's and still runs NUTS/HMC.

    References
    ----------
    .. [1] M. D. Hoffman, A. Radul and P. Sountsov, "An Adaptive-MCMC Scheme for
       Setting Trajectory Lengths in Hamiltonian Monte Carlo", Proceedings of the
       24th International Conference on Artificial Intelligence and Statistics
       (AISTATS), PMLR 130:3907-3915, 2021.
       https://proceedings.mlr.press/v130/hoffman21a.html
    """
    import blackjax
    import blackjax.mcmc.dynamic_hmc as _dynamic_hmc
    import optax
    from blackjax.adaptation.base import get_filter_adapt_info_fn

    def ld_1arg(pos):
        """Bind the traced data to the galaxy-agnostic log-posterior."""
        return logdensity_fn_2arg(pos, data_args)

    jitter_key, adapt_key = jax.random.split(warmup_key)
    ensemble = init_flat[None, :] + jitter_scale * jax.random.normal(
        jitter_key, (n_ensemble, init_flat.shape[0]), dtype=init_flat.dtype
    )

    # ``get_filter_adapt_info_fn()`` with no arguments keeps *nothing*, for the
    # reason ``_ghmc_meads_scan`` states: the default stacks every ensemble state
    # and every HMCInfo for every warmup step, which at 500 x 32 is far larger
    # than the posterior being estimated and is pure overhead -- the adapted
    # values come back in ``parameters``.
    warmup = blackjax.chees_adaptation(
        ld_1arg,
        num_chains=n_ensemble,
        jitter_amount=jitter_amount,
        target_acceptance_rate=target_accept_rate,
        max_leapfrog_steps=max_leapfrog_steps,
        adaptation_info_fn=get_filter_adapt_info_fn(),
        mass_matrix_estimation=mass_matrix_estimation,
    )
    (last_states, parameters), _ = warmup.run(
        adapt_key,
        ensemble,
        _CHEES_INIT_STEP_SIZE,
        optax.adam(learning_rate),
        num_steps=n_warmup,
        # The Halton jitter's bit budget is sized from ``num_steps +
        # max_sampling_steps`` at trace time. Leaving this at BlackJAX's default
        # 1000 while sampling more than that silently wraps the sequence, so the
        # jitter stops being low-discrepancy exactly on the long runs where it
        # matters most.
        max_sampling_steps=n_iter,
    )

    step_size = parameters["step_size"]
    inverse_mass_matrix = parameters["inverse_mass_matrix"]
    (n_leapfrog,) = parameters["integration_steps_params"]

    # The kernel must be rebuilt from the adaptation's OWN
    # ``next_random_arg_fn`` / ``integration_steps_fn``: they carry the Halton
    # counter and the jitter law that ``last_states.random_generator_arg`` is
    # already partway through. Building a fresh default kernel here would reset
    # the jitter to a different sequence than the one the states were adapted
    # under.
    kernel = _dynamic_hmc.build_kernel(
        next_random_arg_fn=parameters["next_random_arg_fn"],
        integration_steps_fn=parameters["integration_steps_fn"],
    )

    if chain_jitter is None:
        # Reuse the ensemble's warmed final states. Nothing is discarded and the
        # chains start in the typical set -- but they are also correlated with
        # the very ensemble that tuned the sampler, so R-hat over them is closer
        # to a consistency check than to an independent test.
        states = jax.tree.map(lambda leaf: leaf[:n_chains], last_states)
    else:
        # Seed the sampling chains independently and OVERDISPERSED. Split R-hat
        # only detects non-convergence when its chains start wider than the
        # posterior, so this is what makes the diagnostic real. The cost is a
        # cold start, which is what ``n_burnin`` pays for; the adaptation
        # (step size, trajectory length) is kept either way, so no warmup is
        # thrown away -- only the warmed positions are.
        # Folded off the warmup key rather than split from a sampling key, so
        # the chains' starting offsets are independent of their own first moves.
        chain_init_key = jax.random.fold_in(warmup_key, 0xC4A15)
        starts = init_flat[None, :] + chain_jitter * jax.random.normal(
            chain_init_key, (n_chains, init_flat.shape[0]), dtype=init_flat.dtype
        )
        states = jax.vmap(lambda p: _dynamic_hmc.init(p, ld_1arg, n_warmup))(starts)

    def _step(s, k):
        """Advance dynamic HMC by one step, returning position and divergence."""
        s, info = kernel(
            k,
            s,
            ld_1arg,
            step_size,
            inverse_mass_matrix,
            (n_leapfrog,),
        )
        return s, (s.position, info.is_divergent)

    def _chain(state, keys):
        """Run one chain's scan; vmapped over the sampling chains."""
        return jax.lax.scan(_step, state, keys)[1]

    positions, divergent = jax.vmap(_chain)(states, chain_keys)
    return positions, divergent, step_size, inverse_mass_matrix, n_leapfrog


@functools.partial(jax.jit, static_argnums=(2, 7, 8, 9))
def _chees_cached_chain_scan(
    init_flat,
    chain_keys,
    logdensity_fn_2arg,
    data_args,
    step_size,
    inverse_mass_matrix,
    n_leapfrog,
    jitter_amount,
    n_iter,
    chain_jitter=None,
):
    """Sample dynamic HMC from a *cached* ChEES adaptation, chains started cold.

    The cache stores three small arrays -- step size, diagonal metric, trajectory
    length -- and deliberately not the warmed ensemble, which is ``n_ensemble x
    D`` of live sampler state. So this path starts the chains from ``init_flat``
    rather than from a warmed state, and ``n_burnin`` is what pays for that. It
    is the reason a cached ChEES call is not bit-identical to the first one, and
    :func:`~tengri.inference.backends.mcmc.chees.run_chees` says so in a comment
    rather than leaving it to be discovered.

    Parameters
    ----------
    init_flat : ndarray, shape (D,)
        Seed position [dimensionless, latent units]; chains are jittered around
        it by :func:`_vmap_chains`'s decorrelation scale, not the adaptation
        ensemble's.
    chain_keys : ndarray, shape (n_chains, n_iter, 2)
        Pre-split per-chain keys.
    logdensity_fn_2arg : callable (static)
        ``log_p(position, data_args)``.
    data_args : pytree (traced)
        Observed data tensors.
    step_size : ndarray, shape ()
    inverse_mass_matrix : ndarray, shape (D,)
    n_leapfrog : ndarray, shape ()
        The three adapted quantities, from the cache.
    jitter_amount : float (static)
        Halton jitter fraction, matching the adaptation's.
    n_iter : int (static)
        Iterations per chain; also sizes the Halton bit budget.
    chain_jitter : float or None (static)
        Sampling-chain dispersion around ``init_flat`` [dimensionless]. ``None``
        uses :func:`_vmap_chains`'s decorrelation scale of 1e-3, which is enough
        to keep two chains from being the same chain and nowhere near enough for
        split R-hat to be a real test. Pass
        :data:`_CHEES_CHAIN_JITTER_SCALE` for that. The knob is threaded here as
        well as into :func:`_chees_scan` because a parameter honored on the cold
        path and silently dropped on the cached one is the failure
        :func:`_adaptation_cache_key`'s docstring records.

    Returns
    -------
    positions : ndarray, shape (n_chains, n_iter, D)
    divergent : ndarray, shape (n_chains, n_iter)
    """
    import blackjax.mcmc.dynamic_hmc as _dynamic_hmc

    def ld_1arg(pos):
        """Bind the traced data to the galaxy-agnostic log-posterior."""
        return logdensity_fn_2arg(pos, data_args)

    n_chains = chain_keys.shape[0]
    max_bits = int(np.ceil(np.log2(n_iter + 2)))

    def _next_random_arg_fn(i):
        return i + 1

    def _jitter_gn(i):
        return _dynamic_hmc.halton_sequence(i, max_bits) * jitter_amount + (1.0 - jitter_amount)

    def _integration_steps_fn(random_generator_arg, num_leapfrog_steps):
        return jnp.asarray(
            jnp.ceil(_jitter_gn(random_generator_arg) * num_leapfrog_steps), dtype=int
        )

    kernel = _dynamic_hmc.build_kernel(
        next_random_arg_fn=_next_random_arg_fn,
        integration_steps_fn=_integration_steps_fn,
    )

    # Folded rather than split off ``chain_keys[0, 0]``: that key is also chain
    # 0's first proposal key, so splitting it would correlate a chain's starting
    # offset with its own first move. ``fold_in`` leaves the sampling stream
    # untouched.
    jitter_key = jax.random.fold_in(chain_keys[0, 0], 0xC4EE5)
    scale = 1e-3 if chain_jitter is None else chain_jitter
    jitter = scale * jax.random.normal(
        jitter_key, (n_chains, init_flat.shape[0]), dtype=init_flat.dtype
    )
    positions0 = init_flat[None, :] + jitter
    states = jax.vmap(lambda p: _dynamic_hmc.init(p, ld_1arg, 0))(positions0)

    def _step(s, k):
        """Advance dynamic HMC by one step, returning position and divergence."""
        s, info = kernel(k, s, ld_1arg, step_size, inverse_mass_matrix, (n_leapfrog,))
        return s, (s.position, info.is_divergent)

    def _chain(state, keys):
        """Run one chain's scan; vmapped over the sampling chains."""
        return jax.lax.scan(_step, state, keys)[1]

    return jax.vmap(_chain)(states, chain_keys)


# ---------------------------------------------------------------------------
# Elliptical slice (no warmup: the exact-prior ellipse needs no tuning)
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(2,))
def _ess_full_scan(
    init_flat,
    chain_keys,
    loglikelihood_fn_2arg,
    data_args,
):
    """Outer JIT: BlackJAX elliptical slice init + sampling chain.

    ESS proposes along ellipses drawn from the exact N(0, I) prior, which
    guarantees acceptance with no step size, mass matrix, or warmup to tune
    (Murray, Adams & MacKay 2010 [1]_). It therefore takes the log-LIKELIHOOD
    only, handing it the full posterior would double-count the prior the
    ellipse already encodes. Burn-in discard is done by the caller
    Python-side, so changing ``n_burnin`` while keeping ``n_chain`` constant
    does not trigger recompilation.

    Parameters
    ----------
    init_flat : ndarray, shape (D,)
        Initial position in unbounded latent space, where every coordinate
        carries an iid standard-normal prior.
    chain_keys : ndarray, shape (n_chain, 2)
        Pre-split keys (``n_chain = n_burnin + n_samples``).
    loglikelihood_fn_2arg : callable (static)
        ``log_L(position, data_args)``, the Gaussian data term ALONE, no
        prior. Taking the data as a traced argument keeps one compiled
        program serving every catalog, same as the other scans here.
    data_args : pytree (traced)
        Observed data tensors; changing these does NOT trigger recompilation.

    Returns
    -------
    positions : ndarray, shape (n_chain, D)
        Caller slices ``[n_burnin:]``.
    subiters : ndarray, shape (n_chain,)
        Ellipse-shrinkage iterations per step, ESS's only tuning-free
        diagnostic; caller slices ``[n_burnin:]``.

    References
    ----------
    .. [1] Murray, I., Adams, R. P., & MacKay, D. J. C. 2010, "Elliptical
       slice sampling", Proceedings of AISTATS 2010, JMLR W&CP 9, 541-548,
       arXiv:1001.0175.
    """
    import blackjax

    def ll(pos):
        return loglikelihood_fn_2arg(pos, data_args)

    n_dim = init_flat.shape[0]
    ess = blackjax.elliptical_slice(ll, mean=jnp.zeros(n_dim), cov=jnp.eye(n_dim))
    state = ess.init(init_flat)

    def _step(s, k):
        """Advance ESS by one step, returning position and subiteration count."""
        s, info = ess.step(k, s)
        return s, (s.position, info.subiter)

    _, (positions, subiters) = jax.lax.scan(_step, state, chain_keys)
    return positions, subiters


# ---------------------------------------------------------------------------
# MCLMC scans (no burn-in phase; adaptation tunes L and step_size jointly)
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(2, 5))
def _mclmc_sample_scan(state, keys, kernel, L, step_size, logdensity_fn, inverse_mass_matrix):
    """JIT-compiled MCLMC sampling scan over multiple steps.

    Requires blackjax >= 1.6, whose kernel takes ``logdensity_fn`` and
    ``inverse_mass_matrix`` per step rather than baking them in at
    ``build_kernel`` time (#1177). Arguments are passed by keyword so a
    future position change cannot silently misbind them.
    """

    def _step(s, k):
        """Advance MCLMC sampler by one step, returning updated state and position."""
        s, _info = kernel(
            rng_key=k,
            state=s,
            logdensity_fn=logdensity_fn,
            inverse_mass_matrix=inverse_mass_matrix,
            L=L,
            step_size=step_size,
        )
        return s, s.position

    return jax.lax.scan(_step, state, keys)


@functools.partial(jax.jit, static_argnums=(2, 5))
def _adjusted_mclmc_sample_scan(
    state, keys, kernel, step_size, n_integration_steps, logdensity_fn, inverse_mass_matrix
):
    """JIT-compiled adjusted MCLMC sampling scan over multiple steps.

    Requires blackjax >= 1.6 (see :func:`_mclmc_sample_scan`).
    """

    def _step(s, k):
        """Advance adjusted MCLMC sampler by one step, returning position and divergence flag."""
        s, info = kernel(
            rng_key=k,
            state=s,
            logdensity_fn=logdensity_fn,
            step_size=step_size,
            # blackjax >= 1.6 unpacks this as ``(num_integration_steps,) =
            # integration_steps_params``, a bare scalar raises
            # "iteration over a 0-d array".
            integration_steps_params=(n_integration_steps,),
            inverse_mass_matrix=inverse_mass_matrix,
        )
        return s, (s.position, info.is_divergent)

    return jax.lax.scan(_step, state, keys)


# ---------------------------------------------------------------------------
# Logdensity and adaptation cache helpers
# ---------------------------------------------------------------------------


def _get_flat_logdensity(fitter, init_params):
    """Return (log_posterior_flat_2arg, unravel_fn, init_flat, data_args).

    The returned ``log_posterior_flat_2arg(position, data_args)`` takes
    ``data_args`` as a **traced** JAX argument (not closed over), so
    the compiled XLA program is reused across galaxies sharing the
    same model structure.  The function is cached on the **Model**
    (not fitter) so that multiple Fitters with different data share
    the same compiled code.
    """
    cache_key = fitter._engine_cache_key()
    model = fitter.model
    cache = _model_cache_owner.get_or_compile_model(model).setdefault("flat_logdensity", {})

    if cache_key not in cache:
        logdensity_2arg = fitter._get_or_build_logdensity_fn()
        _, unravel_fn = ravel_pytree(init_params)

        def log_posterior_flat_2arg(position, data_args):
            """Evaluate log posterior from a flat position vector by unraveling to pytree first."""
            return logdensity_2arg(unravel_fn(position), data_args)

        cache[cache_key] = (
            log_posterior_flat_2arg,
            unravel_fn,
        )

    logdensity_flat, unravel_fn = cache[cache_key]
    init_flat, _ = ravel_pytree(init_params)
    return logdensity_flat, unravel_fn, init_flat, fitter._data_args


def _adaptation_cache_key(fitter, method_key):
    """Key an adaptation entry by engine shape, method, **and target data**.

    ``_engine_cache_key`` identifies the compiled *engine shape*, data length,
    free-parameter names, feature channels, and deliberately says nothing
    about the data values. Two galaxies with the same band count therefore
    share it. Without the data in the key, the ordinary catalog loop that
    reuses one model hands every galaxy the first galaxy's step size and mass
    matrix: adaptation tuned on another target's posterior geometry.

    This is the defect ``tengri.inference._sample_utils._data_fingerprint``
    was written for (issue #1529), where the *MAP* cache seeded every fit from
    the first galaxy's optimum and killed six of eight NUTS fits. The
    adaptation cache is that cache's sibling and was never given the same
    guard; ``ghmc`` and ``mclmc`` additionally keyed on nothing but a bare
    method name. Hashing the data separates targets while keeping the intended
    win, a genuine refit of the same target still hits, because this keys on
    content rather than identity.

    ``method_key`` is the backend's own tuple and must carry every setting that
    *produces* the adaptation, warmup length and target acceptance rate as
    well as the structural choices. Leave one out and that knob goes quiet on a
    model that already holds an entry: measured while tuning the quickstart, a
    500 / 1000 / 1500 warmup sweep returned byte-identical diagnostics three
    times over, which reads as "warmup length does not matter here" rather than
    "your knob was ignored".
    """
    from tengri.inference._sample_utils import _data_fingerprint

    return (fitter._engine_cache_key(), method_key, _data_fingerprint(fitter))


def _get_cached_adaptation(fitter, method_key):
    """Retrieve cached adaptation parameters by method key, or None if not cached."""
    mc = _model_cache_owner.get_or_compile_model(fitter.model)
    cache = mc.get("adaptation")
    if cache is None:
        return None
    return cache.get(_adaptation_cache_key(fitter, method_key))


def _set_cached_adaptation(fitter, method_key, params):
    """Store adaptation parameters on the Model for cross-fitter reuse."""
    cache = _model_cache_owner.get_or_compile_model(fitter.model).setdefault("adaptation", {})
    cache[_adaptation_cache_key(fitter, method_key)] = params


def _vmap_chains(
    init_state_fn,
    chain_scan_fn,
    *,
    init_flat,
    chain_key,
    n_chains,
    n_iter,
    n_burnin,
    jitter_scale=1e-3,
):
    """Run ``n_chains`` independent MCMC chains in parallel via ``jax.vmap``.

    Shared multi-chain plumbing used by HMC / NUTS / dHMC / GHMC / MCLMC.
    Each chain starts from ``init_flat + jitter`` and shares the same
    cached adaptation; the vmap dispatches them across XLA SIMD lanes
    (CPU) or accelerator cores (GPU/TPU). Per-chain burnin is discarded
    before the ``(n_chains, n_iter, ...)`` → flattened reshape so
    ``n_burnin`` correctly applies to *each* chain.

    Parameters
    ----------
    init_state_fn : callable(init_pos_for_chain) -> chain_state
        Builds the sampler's initial state from a chain's starting position.
        May close over the log-density callable.
    chain_scan_fn : callable(chain_state, chain_keys) -> outputs
        Runs the per-chain scan (burn-in + sampling). Outputs may be a
        single jnp.ndarray (e.g. positions) or a tuple where the leading
        axis is iterations (e.g. ``(positions, divergent)``).
    init_flat : jnp.ndarray, shape (D,)
        Reference initial position; each chain is jittered around this.
    chain_key : PRNGKey
        Splits into per-chain init jitter, per-chain init keys, and the
        flat key block fed to ``chain_scan_fn``.
    n_chains : int
        Number of chains to vmap (≥ 2; callers handle the single-chain case).
    n_iter : int
        Iterations per chain (caller passes ``n_burnin + n_samples``).
    n_burnin : int
        Per-chain burn-in to discard before flatten.
    jitter_scale : float, default 1e-3
        Gaussian jitter scale applied to ``init_flat`` for each chain.

    Returns
    -------
    same shape as ``chain_scan_fn`` output, with the leading
    ``(n_chains, n_iter)`` dimensions burnin-discarded and flattened to
    ``(n_chains * (n_iter - n_burnin),)``.
    """
    keys = jax.random.split(chain_key, n_chains + 2)
    new_chain_key, jitter_key, init_key_seed = keys[0], keys[1], keys[2]
    per_chain_init_keys = jax.random.split(init_key_seed, n_chains)
    jitter = jitter_scale * jax.random.normal(jitter_key, shape=(n_chains, init_flat.shape[0]))
    init_flat_batch = init_flat[None, :] + jitter

    # init_state_fn may be unary (just init_pos) or binary (init_pos, init_key).
    import inspect as _inspect

    arity = len(_inspect.signature(init_state_fn).parameters)
    if arity == 1:
        states = jax.vmap(init_state_fn)(init_flat_batch)
    else:
        states = jax.vmap(init_state_fn)(init_flat_batch, per_chain_init_keys)

    per_chain_keys = jax.random.split(new_chain_key, n_chains * n_iter)
    per_chain_keys = per_chain_keys.reshape(n_chains, n_iter, 2)
    out = jax.vmap(chain_scan_fn)(states, per_chain_keys)

    def _trim_and_flatten(arr):
        if n_burnin > 0:
            arr = arr[:, n_burnin:]
        if arr.ndim >= 3:
            return arr.reshape(-1, *arr.shape[2:])
        return arr.reshape(-1)

    if isinstance(out, tuple):
        return tuple(_trim_and_flatten(a) for a in out)
    return _trim_and_flatten(out)


def _parallel_chains(
    init_state_fn,
    chain_scan_fn,
    *,
    init_flat,
    chain_key,
    n_chains,
    n_iter,
    n_burnin,
    jitter_scale=1e-3,
):
    """Run ``n_chains`` MCMC chains on separate devices via ``jax.pmap``.

    True parallelism variant of :func:`_vmap_chains`: instead of SIMD-batching
    the chains into one device's kernel (which costs ~``n_chains``× a single
    chain on CPU), this maps them across physical devices so ``n_chains``
    chains run in ~one chain's wall time. Same jitter / burn-in / flatten
    contract and same return shape as :func:`_vmap_chains`, so callers can
    swap the two.

    On CPU, JAX exposes one device by default; obtaining ``n_chains`` devices
    needs ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` set **before**
    JAX initializes. Raises :class:`RuntimeError` when fewer than ``n_chains``
    devices are visible so callers can fall back to :func:`_vmap_chains`.

    Parameters
    ----------
    init_state_fn, chain_scan_fn, init_flat, chain_key, n_chains, n_iter, \
    n_burnin, jitter_scale
        Identical to :func:`_vmap_chains`.

    Returns
    -------
    Same shape as :func:`_vmap_chains`.

    Raises
    ------
    RuntimeError
        If ``jax.device_count() < n_chains``.
    """
    n_dev = jax.device_count()
    if n_dev < n_chains:
        raise RuntimeError(
            f"chain_method='parallel' needs at least n_chains={n_chains} JAX devices, "
            f"found {n_dev}. On CPU, set "
            f"XLA_FLAGS=--xla_force_host_platform_device_count={n_chains} before importing "
            f"jax / tengri, or use chain_method='vmap'."
        )
    devices = jax.devices()[:n_chains]

    keys = jax.random.split(chain_key, n_chains + 2)
    new_chain_key, jitter_key, init_key_seed = keys[0], keys[1], keys[2]
    per_chain_init_keys = jax.random.split(init_key_seed, n_chains)
    jitter = jitter_scale * jax.random.normal(jitter_key, shape=(n_chains, init_flat.shape[0]))
    init_flat_batch = init_flat[None, :] + jitter

    import inspect as _inspect

    arity = len(_inspect.signature(init_state_fn).parameters)
    if arity == 1:
        states = jax.pmap(init_state_fn, devices=devices)(init_flat_batch)
    else:
        states = jax.pmap(init_state_fn, devices=devices)(init_flat_batch, per_chain_init_keys)

    per_chain_keys = jax.random.split(new_chain_key, n_chains * n_iter)
    per_chain_keys = per_chain_keys.reshape(n_chains, n_iter, 2)
    out = jax.pmap(chain_scan_fn, devices=devices)(states, per_chain_keys)

    def _trim_and_flatten(arr):
        if n_burnin > 0:
            arr = arr[:, n_burnin:]
        if arr.ndim >= 3:
            return arr.reshape(-1, *arr.shape[2:])
        return arr.reshape(-1)

    if isinstance(out, tuple):
        return tuple(_trim_and_flatten(a) for a in out)
    return _trim_and_flatten(out)


def _sequential_chains(
    init_state_fn,
    chain_scan_fn,
    *,
    init_flat,
    chain_key,
    n_chains,
    n_iter,
    n_burnin,
    jitter_scale=1e-3,
):
    """Run ``n_chains`` MCMC chains one at a time, the memory-frugal executor.

    Same jitter / burn-in / flatten contract and return shape as
    :func:`_vmap_chains`, but the chains are looped in Python instead of
    SIMD-batched. ``chain_scan_fn`` is a single-chain program, compiled once on
    the first call and reused, so **peak memory stays at one chain's** rather
    than ``n_chains`` × (the vmap path compiles the whole batch into one XLA
    program, which is what OOMs a dense-mass multi-chain fit on modest RAM).
    The trade-off is wall time: ~``n_chains`` × a single chain's sampling. This
    is the default for a converged multi-chain fit that must run on cheap
    hardware; use ``chain_method="vmap"`` / ``"parallel"`` when RAM / devices
    allow.

    Parameters
    ----------
    init_state_fn, chain_scan_fn, init_flat, chain_key, n_chains, n_iter, \
    n_burnin, jitter_scale
        Identical to :func:`_vmap_chains`.

    Returns
    -------
    Same shape as :func:`_vmap_chains`.
    """
    keys = jax.random.split(chain_key, n_chains + 2)
    new_chain_key, jitter_key, init_key_seed = keys[0], keys[1], keys[2]
    per_chain_init_keys = jax.random.split(init_key_seed, n_chains)
    jitter = jitter_scale * jax.random.normal(jitter_key, shape=(n_chains, init_flat.shape[0]))
    init_flat_batch = init_flat[None, :] + jitter
    per_chain_keys = jax.random.split(new_chain_key, n_chains * n_iter)
    per_chain_keys = per_chain_keys.reshape(n_chains, n_iter, 2)

    import inspect as _inspect

    arity = len(_inspect.signature(init_state_fn).parameters)

    per_chain_out = []
    for c in range(n_chains):
        if arity == 1:
            state = init_state_fn(init_flat_batch[c])
        else:
            state = init_state_fn(init_flat_batch[c], per_chain_init_keys[c])
        out = chain_scan_fn(state, per_chain_keys[c])
        if n_burnin > 0:
            out = jax.tree_util.tree_map(lambda a: a[n_burnin:], out)
        jax.block_until_ready(out)  # free this chain's trace before the next
        per_chain_out.append(out)

    # Stack over chains, then flatten (n_chains, per_chain, ...) -> (total, ...).
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *per_chain_out)

    def _flatten(arr):
        if arr.ndim >= 2:
            return arr.reshape(-1, *arr.shape[2:])
        return arr.reshape(-1)

    if isinstance(stacked, tuple):
        return tuple(_flatten(a) for a in stacked)
    return _flatten(stacked)
