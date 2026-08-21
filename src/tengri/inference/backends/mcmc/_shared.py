# SPDX-License-Identifier: BSD-3-Clause
"""Shared MCMC infrastructure: kernel getters, scan functions, logdensity helpers.

Internal — imported by per-sampler modules. Not part of the public API.

Compilation strategy
--------------------
Every sampler exposes two JIT-compiled entry points:

``_<method>_full_scan``
    Outer JIT wrapping BlackJAX window adaptation **and** the chain
    (burn-in + sampling) in a single XLA program.  Used for the cold
    path (no cached adaptation).  The kernel — e.g. the NUTS
    ``lax.while_loop`` tree builder — is compiled exactly once instead
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
from jax.flatten_util import ravel_pytree
from packaging.requirements import Requirement

from tengri.config.exceptions import BackendError
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
            f"silently — a frozen chain, not an error (issue #1999). "
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
#: Before this constant existed the prewarm path hardcoded its own ``10`` —
#: correct only by coincidence with the signature defaults it never saw.
#:
#: 10 (the BlackJAX/Stan convention) — and measured to stay there. Lowering
#: the cap looks like a huge win on the heavy-tailed StudentT SFR-ratio
#: geometry of the nonparametric SFHs and is a trap: on a 19-band continuity
#: fit (D=9, 500 warmup + 500 samples, CPU, 2026-08-18) cap 6 cut the wall
#: 118 s → 11 s but collapsed min-ESS 93 → 5 — per *effective* sample it is
#: strictly worse (1.99 vs 1.27 s/ESS). ``dense_mass_matrix=True`` was the
#: recommendation here and no longer is: re-measured it buys wall time at the
#: cost of 12 divergences per run against 2 for the diagonal. On that
#: geometry the genuine fixes are bin edges that stop at the age of the
#: universe (#1975) and a longer fixed-length trajectory
#: (``mcmc_hmc``, ``n_leapfrog_steps=150``). Saturation of a deep cap
#: is surfaced by ``NUTSTreeDepthWarning`` and the ``tree_depth_*``
#: diagnostics every NUTS fit now reports; a deliberate low cap for a
#: wall-bounded quick look is one kwarg, taken knowingly.
DEFAULT_MAX_NUM_DOUBLINGS = 10


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
        ``log_p(position, data_args)`` — galaxy-agnostic log-posterior.
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
    """BlackJAX NUTS window adaptation only — returns tuned (step_size, inv_mass).

    The warmup half of :func:`_nuts_full_scan`, split out for the same reason
    :func:`_hmc_warmup_only` was: so the fresh and cached-adaptation paths end
    in the *same* sampling call. While warmup and sampling were fused here, a
    first fit ran ``_nuts_full_scan`` and every later fit on the same model ran
    a sampling-only scan against the cached parameters — structurally different
    computations, so one pinned ``key`` produced two different posteriors. HMC
    had already been split this way and was reproducible; NUTS was not.

    Same static-arg / traced-``data_args`` contract as :func:`_nuts_full_scan`.

    Returns
    -------
    step_size : scalar
    inv_mass_matrix : ndarray, shape (D,) or (D, D)
    """
    import blackjax

    def ld_1arg(pos):
        return logdensity_fn_2arg(pos, data_args)

    # Discard per-step adaptation info; blackjax retains it all by default and
    # warns about the memory cost when it goes unused (#1028).
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
            (_, parameters), _ = warmup.run(warmup_key, init_flat, num_steps=n_warmup)
    else:
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            ld_1arg,
            is_mass_matrix_diagonal=not use_dense,
            target_acceptance_rate=target_accept_rate,
            adaptation_info_fn=_drop_adapt_info,
        )
        (_, parameters), _ = warmup.run(warmup_key, init_flat, num_steps=n_warmup)

    return parameters["step_size"], parameters["inverse_mass_matrix"]


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
    """BlackJAX HMC window adaptation only — returns tuned (step_size, inv_mass).

    The warmup half of :func:`_hmc_full_scan`, split out so a *fresh*
    ``run_hmc`` call can adapt once and then dispatch sampling through the
    single- or multi-chain path (vmap/parallel) that honors ``n_chains`` —
    without the old behavior of silently sampling a single chain on the first
    call. Same static-arg / traced-``data_args`` contract as
    :func:`_hmc_full_scan`.

    Returns
    -------
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
    (_, parameters), _ = warmup.run(warmup_key, init_flat, num_steps=n_warmup)
    return parameters["step_size"], parameters["inverse_mass_matrix"]


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
        ``log_p(position, data_args)`` — galaxy-agnostic log-posterior.
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
        ``log_p(position, data_args)`` — galaxy-agnostic log-posterior.
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
        ``log_p(position, data_args)`` — galaxy-agnostic log-posterior.
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
    # between 1.3 and 1.6 — keywords are correct on both.
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
    only — handing it the full posterior would double-count the prior the
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
        ``log_L(position, data_args)`` — the Gaussian data term ALONE, no
        prior. Taking the data as a traced argument keeps one compiled
        program serving every catalog, same as the other scans here.
    data_args : pytree (traced)
        Observed data tensors; changing these does NOT trigger recompilation.

    Returns
    -------
    positions : ndarray, shape (n_chain, D)
        Caller slices ``[n_burnin:]``.
    subiters : ndarray, shape (n_chain,)
        Ellipse-shrinkage iterations per step — ESS's only tuning-free
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
            # integration_steps_params`` — a bare scalar raises
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

    ``_engine_cache_key`` identifies the compiled *engine shape* — data length,
    free-parameter names, feature channels — and deliberately says nothing
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
    win — a genuine refit of the same target still hits, because this keys on
    content rather than identity.

    ``method_key`` is the backend's own tuple and must carry every setting that
    *produces* the adaptation — warmup length and target acceptance rate as
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
    """Run ``n_chains`` MCMC chains one at a time — the memory-frugal executor.

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
