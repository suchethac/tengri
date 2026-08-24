# SPDX-License-Identifier: BSD-3-Clause
"""Shared post-processing utilities for all inference backends.

Every sampler needs the same three operations after collecting samples:
convert flat unbounded positions → physical dicts, summarize with a
point estimate, and optionally warm-start from a MAP.  Centralizing
them here removes ~50 duplicated blocks across the backends.
"""

from __future__ import annotations

import hashlib
import warnings
from collections import OrderedDict
from collections.abc import Callable, Mapping
from threading import Lock
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from jax import Array

_MAP_INIT_STEPS = 1000
#: Number of vmap'd ADAM restarts for the MCMC MAP-init seed. The standardized
#: prior is genuinely uniform, so a single random init can land in a poor basin;
#: a handful of parallel restarts (keep-best) seeds all chains at a good point so
#: NUTS/HMC converge in a short warm-up. Kept modest so the seed stays ~1 s.
_MAP_INIT_RESTARTS = 8

# Bound the cache so a long session iterating over many fitters does not
# accumulate compiled vmaps unboundedly. LRU-evict on insert.
_VMAP_TO_PHYSICAL_CACHE: OrderedDict = OrderedDict()
_VMAP_TO_PHYSICAL_CACHE_LOCK = Lock()
_VMAP_TO_PHYSICAL_CACHE_MAX = 8


def _vmap_samples_to_physical(
    samples: Array,
    unravel_fn: Callable[[Array], dict],
    to_physical_fn: Callable[[dict], dict],
) -> dict[str, Array]:
    """Convert a batch of flat unbounded samples to physical parameter dicts.

    The compiled vmap is cached on the fitter (recovered from the bound
    method's ``__self__``) keyed on ``(id(fitter), n_dim)``. Without
    this cache every ``Fitter.run`` re-traces and recompiles the
    conversion (~700 ms photometry, more on spec). ``unravel_fn``
    captured by the cache entry is the one from the first call,
    structurally identical to subsequent ones for the same fitter, so
    reusing it is safe.
    """
    fitter = getattr(to_physical_fn, "__self__", None)
    if fitter is None:
        # No bound-method fitter → can't cache safely; fall back to per-call vmap.
        return jax.vmap(lambda p: to_physical_fn(unravel_fn(p)))(samples)

    n_dim = int(samples.shape[-1])
    key = (id(fitter), n_dim)
    with _VMAP_TO_PHYSICAL_CACHE_LOCK:
        cached = _VMAP_TO_PHYSICAL_CACHE.get(key)
        if cached is None:
            # Capture the unravel_fn closure from THIS call; structurally
            # equivalent to any future unravel_fn for the same fitter.
            unravel_captured = unravel_fn
            to_phys_captured = to_physical_fn

            @jax.jit
            def _convert_one(flat_pos: Array) -> dict[str, Array]:
                return to_phys_captured(unravel_captured(flat_pos))

            cached = jax.vmap(_convert_one)
            _VMAP_TO_PHYSICAL_CACHE[key] = cached
            while len(_VMAP_TO_PHYSICAL_CACHE) > _VMAP_TO_PHYSICAL_CACHE_MAX:
                _VMAP_TO_PHYSICAL_CACHE.popitem(last=False)
        else:
            _VMAP_TO_PHYSICAL_CACHE.move_to_end(key)
    return cached(samples)


def _clear_vmap_to_physical_cache() -> None:
    """Drop the cached vmap conversions (called by ``tengri.gc()``)."""
    with _VMAP_TO_PHYSICAL_CACHE_LOCK:
        _VMAP_TO_PHYSICAL_CACHE.clear()


def _mean_params(samples_phys: dict[str, Array]) -> dict[str, Array]:
    """Return per-parameter posterior means."""
    return {k: jnp.mean(v, axis=0) for k, v in samples_phys.items()}


#: Per-fitter inputs that change the posterior and so must invalidate a cached
#: MAP. Everything else in ``data_args`` (spectroscopic covariance, line fluxes)
#: hangs off ``model.observation``, which is already the cache key.
_FINGERPRINTED_ATTRS = ("data", "noise", "data_mask", "presence", "_runtime_redshift")


def _data_fingerprint(fitter: Any) -> str:
    """Content hash of the observations this fitter was constructed against.

    The MAP cache lives in a per-model namespace keyed on the model object,
    which says nothing about the data. Reusing one model across a catalog,
    the ordinary loop, then hands every galaxy the first galaxy's MAP as its
    starting point (issue #1529). Hashing the data separates targets while
    keeping the intended win: a genuine refit of the same target still hits,
    including across sessions, because this keys on content and not identity.

    Parameters
    ----------
    fitter: Fitter
        Fitter whose ``data``/``noise`` (and optional mask, presence and
        runtime redshift) identify the target.

    Returns
    -------
    str
        Hex digest. Cost is linear in the data bytes, negligible beside the
        thousands of forward passes a MAP run would otherwise repeat.
    """
    digest = hashlib.blake2b(digest_size=16)
    for name in _FINGERPRINTED_ATTRS:
        value = getattr(fitter, name, None)
        if value is None:
            digest.update(b"\x00none")
            continue
        array = np.asarray(value)
        digest.update(f"{name}|{array.shape}|{array.dtype}|".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class _ParamsShim:
    """Minimal posterior-like wrapper so a plain mapping can seed a fit."""

    __slots__ = ("params",)

    def __init__(self, params):
        self.params = params


def _as_posterior_like(fitter: Any, init_from: Any) -> Any:
    """Accept a starting point as either a posterior-like object or a mapping.

    ``init_from`` reached ``_unbounded_from_posterior`` directly, which reads
    ``.params`` off it, so the obvious spelling -- a dict of the parameters you
    want to start at -- died on ``AttributeError: 'dict' object has no
    attribute 'params'``, a message about this function's internals rather than
    about what the caller passed. Every other parameter surface in tengri
    (``predict``, ``predict_photometry``, ``spec.get_fixed_values``) takes a
    plain ``dict[str, float]``; this one is now no exception (issue #1854).

    Validation matches what the conversion actually consumes.
    ``Fitter._unbounded_from_posterior`` reads only the *free* names and starts
    any it does not find at the standardized ``0.0`` -- the prior center -- so:

    * **Fixed parameters are accepted and ignored.** A ``Posterior.params``
      carries them, and ``dict(map_result.params)`` is the obvious thing to
      hand back; refusing those keys would reject the commonest input.
    * **A partial mapping is accepted**, with every unnamed free parameter
      starting at its prior center. That is pre-existing behavior, not a new
      promise -- seeding two of seven axes and leaving five at the center is a
      legitimate but *weak* start, and mixing may suffer accordingly.
    * **An unrecognized name is refused.** Silently ignoring a misspelled key
      would start that axis somewhere the caller did not choose while the fit
      looked perfectly healthy -- the failure mode
      ``approx.get("n_subbnads", 0)`` already cost this codebase once.
    * **A mapping naming no free parameter at all is refused**, because it
      cannot move the starting point and is therefore always a mistake.

    Parameters
    ----------
    fitter: Fitter
        Supplies the free and fixed parameter names for validation.
    init_from: Mapping or posterior-like
        Starting point. A mapping is wrapped; anything exposing ``.params``
        (a :class:`~tengri.inference.posterior.Posterior`, a MAP result) is
        returned unchanged.

    Returns
    -------
    posterior-like
        An object with a ``.params`` mapping.

    Raises
    ------
    ParameterError
        If a mapping names an unrecognized parameter, names no free parameter,
        or if ``init_from`` is neither a mapping nor posterior-like.
    """
    if hasattr(init_from, "params"):
        return init_from

    from tengri.config.exceptions import ParameterError

    if isinstance(init_from, Mapping):
        free = list(getattr(fitter.spec, "free_params", ()))
        try:
            fixed = set(fitter.spec.get_fixed_values())
        except Exception:
            fixed = set()
        known = set(free) | fixed | {"psd_xi"}

        unknown = sorted(name for name in init_from if name not in known)
        if unknown:
            raise ParameterError(
                f"init_from names {len(unknown)} parameter(s) this model does not "
                f"have: {unknown}. Free parameters are: {free}. A starting point "
                "for a name the model never reads would be silently ignored, so "
                "it is refused here instead."
            )
        supplied = [name for name in free if name in init_from]
        if not supplied:
            raise ParameterError(
                "init_from names no free parameter, so it cannot move the "
                f"starting point. Free parameters are: {free}."
            )
        missing = [name for name in free if name not in init_from]
        if missing:
            # Legal, but weak, and invisible otherwise: those axes start at the
            # prior center while the fit reports nothing unusual. A two-of-seven
            # seed has been measured mixing to split R-hat ~1e15 on this path.
            warnings.warn(
                f"init_from seeds {len(supplied)} of {len(free)} free parameters; "
                f"{missing} will start at the prior center. Partial starts mix "
                "poorly, check split R-hat, or pass every free parameter (a MAP "
                "result supplies them all).",
                UserWarning,
                stacklevel=3,
            )
        return _ParamsShim(dict(init_from))

    raise ParameterError(
        f"init_from must be a mapping of parameter values or a result with a "
        f".params attribute, not {type(init_from).__name__}. Pass a dict like "
        "{'dust_tau_bc': 0.3}, or the result of "
        "forward.fit(..., method='map', key=key)."
    )


def _maybe_map_init(
    fitter: Any,
    key: Array,
    init_from: Any,
    verbose: bool,
    n_map_steps: int = _MAP_INIT_STEPS,
) -> tuple[dict, Array]:
    """Return (init_params_unbounded, updated_key).

    Resolution order:

    1. Explicit ``init_from`` from caller, converted to unbounded.
    2. Cached MAP point on the model, **if it was fit to this same data**
       (populated by a previous fit or by :meth:`Fitter.load_cache`), used
       directly, no fresh MAP run.
    3. Run a short MAP optimization to find a good starting point.
       Caches the result so subsequent calls / sessions can skip step 3.

    Step 2 is gated on :func:`_data_fingerprint`. Without that gate a loop that
    reuses one model across a catalog silently starts every galaxy from the
    first galaxy's MAP, which killed six of eight NUTS fits with R-hat up to
    10.74 and zero divergences (issue #1529).
    """
    if init_from is not None:
        return fitter._unbounded_from_posterior(_as_posterior_like(fitter, init_from)), key

    # Cached MAP point (from a prior fit or load_cache) for THIS data?

    from tengri.inference._model_cache import _default_owner as _model_cache_owner

    # Advance the key before branching, so a cache hit and a cache miss hand the
    # sampler the same stream. Previously the hit returned ``key`` untouched
    # while the miss returned it split for the MAP run, which made the chain
    # depend on whether a *previous* fit had happened to populate the cache,
    # invisible to the caller, and enough to give two identical ``fit`` calls
    # with one ``key`` different posteriors.
    key, map_key = jax.random.split(key)

    mc = _model_cache_owner.get_or_compile_model(fitter.model)
    fingerprint = _data_fingerprint(fitter)
    cached_map = mc.get("map_params_physical")
    if cached_map is not None and mc.get("map_data_fingerprint") != fingerprint:
        # Same model, different target: the cached point belongs to another
        # galaxy and would seed this fit somewhere its sampler may not recover
        # from. Fall through to a fresh MAP.
        cached_map = None
    if cached_map is not None:
        # Build a minimal posterior-like shim to reuse _unbounded_from_posterior.
        class _Shim:
            def __init__(self, params):
                self.params = params

        if verbose:
            print("  MAP initialization: reusing cached MAP point")
        return fitter._unbounded_from_posterior(_Shim(cached_map)), key

    # Pre-warm: one eager forward call resolves a tracing pathology in
    # ``Uniform.unstandardize`` that crashes the MAP-init JIT trace on
    # models whose forward pass has never been exercised in Python (e.g.
    # ``Fitter(model, flux_from_disk, noise).run('mcmc_hmc')`` with no
    # prior ``model.predict_photometry(...)`` or ``model.mock(...)``).
    # See issue #262 for the narrow reproducer and traceback.
    _prewarm_forward(fitter)
    if verbose:
        print(f"  MAP initialization ({n_map_steps} steps)...")
    map_result = fitter._run_map(
        key=map_key, n_steps=n_map_steps, n_restarts=_MAP_INIT_RESTARTS, verbose=False
    )
    # Cache the physical MAP params so future runs/sessions skip MAP. Stamped
    # with the data it was fit to, so it can never be reused for another target.
    mc["map_params_physical"] = {k: jnp.asarray(v) for k, v in map_result.params.items()}
    mc["map_data_fingerprint"] = fingerprint
    init_params = fitter._unbounded_from_posterior(map_result)
    if verbose:
        print(f"  MAP init done (loss={map_result.diagnostics['final_loss']:.2f})")
    return init_params, key


def _prewarm_forward(fitter: Any) -> None:
    """Run one eager forward pass to materialize lazy unbounded→physical state.

    Hit by issue #262: when ``Fitter`` is constructed against externally
    sourced data (e.g. ``np.load(npz)``) and the model has never had a
    Python-side forward pass on it, the first JIT trace through
    ``Uniform.unstandardize`` keeps a traced array escaping into
    ``float()``. Calling ``predict_photometry`` once eagerly before
    JIT tracing materializes the relevant arrays and resolves the
    leak. Best-effort: if the call fails for any other reason, swallow
    silently and let the real path raise.
    """
    try:
        sample_params = {}
        for name in fitter._free_names:
            dist = fitter.spec.get_distribution(name)
            if hasattr(dist, "low") or hasattr(dist, "lo"):
                lo = getattr(dist, "low", getattr(dist, "lo", 0.0))
                hi = getattr(dist, "high", getattr(dist, "hi", 1.0))
                sample_params[name] = float(0.5 * (lo + hi))
            elif hasattr(dist, "mean"):
                sample_params[name] = float(dist.mean)
            else:
                sample_params[name] = 0.0
        for name, value in fitter._fixed_values.items():
            sample_params[name] = float(value)
        fitter.model.predict_photometry(sample_params)
    except Exception:
        # Pre-warm is a soft guarantee; the real path will surface
        # any genuine failure with a richer traceback.
        pass
