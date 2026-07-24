# SPDX-License-Identifier: BSD-3-Clause
"""Independent per-galaxy catalog inference with optional K-way parallelism.

Unlike :class:`PopulationFitter`, galaxies share no parameters.
:class:`CatalogFitter` supports every method :class:`Fitter` accepts; for
``native_vi_linear`` and ``native_vi_nonlinear`` it vmaps K galaxies per
``lax.map`` step so the compiled XLA graph stays O(1) in N while K galaxies
execute simultaneously on-device.
"""

from __future__ import annotations

import functools
import math
import time
import warnings
from dataclasses import dataclass, field

__all__ = ["CatalogPosterior"]

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from tengri.inference._sample_utils import _mean_params, _vmap_samples_to_physical


def _compute_summaries(samples, percentiles=None, reducers=None):
    """Compute percentiles and reducer statistics for a sample dict.

    Parameters
    ----------
    samples : dict
        Parameter samples, keys are parameter names, values are (n_samples,) arrays.
    percentiles : tuple, optional
        Percentile values to compute. Default (16, 50, 84).
    reducers : dict, optional
        Additional reducers {name: callable} (e.g., {"mean": jnp.mean}).

    Returns
    -------
    percentiles_dict : dict
        Keys are parameter names, values are (n_pct,) arrays.
    summary_dict : dict
        Nested dict: {reducer_name: {param_name: scalar value}}.
    """
    if percentiles is None:
        percentiles = (16, 50, 84)
    if reducers is None:
        reducers = {}

    percentiles_dict = {}
    summary_dict = {}

    # Compute percentiles
    for name, samples_arr in samples.items():
        percentiles_dict[name] = np.percentile(np.asarray(samples_arr), percentiles)

    # Compute reducers
    for reducer_name, reducer_fn in reducers.items():
        summary_dict[reducer_name] = {}
        for name, samples_arr in samples.items():
            summary_dict[reducer_name][name] = float(reducer_fn(np.asarray(samples_arr)))

    return percentiles_dict, summary_dict


def _resolve_n_padded(n_gal: int, K: int, n_pad: int | str | None) -> int:
    """Resolve the padded catalog size for the catalog VI engine.

    The native VI engine requires the catalog to be a multiple of K (so
    galaxies can be reshaped to ``(n_chunks, K, ...)`` for the inner
    vmap). ``n_pad`` lets the caller pad further so different catalog
    sizes share an XLA compile cache key.

    Parameters
    ----------
    n_gal : int
        Real galaxy count.
    K : int
        ``forward_chunk_size`` (>=1).
    n_pad : int, "auto", or None
        ``None`` → multiple-of-K minimum (existing behavior).
        ``"auto"`` → next power of 2 (also at least multiple of K).
        ``int`` → exact target (must be ``>= n_gal``).

    Returns
    -------
    int
        Padded catalog size, always a multiple of ``K`` and ``>= n_gal``.

    Raises
    ------
    ValueError
        If ``n_pad`` is an int below ``n_gal``, or an invalid string.
    """
    base = math.ceil(n_gal / K) * K  # multiple-of-K floor
    if n_pad is None:
        return base
    if isinstance(n_pad, str):
        if n_pad != "auto":
            raise ValueError(f"n_pad must be 'auto', None, or int; got {n_pad!r}")
        # Smallest power of 2 >= n_gal, then round up to multiple of K.
        pow2 = 1 if n_gal <= 1 else 1 << (n_gal - 1).bit_length()
        return max(base, math.ceil(pow2 / K) * K)
    target = int(n_pad)
    if target < n_gal:
        raise ValueError(f"n_pad={target} must be >= n_galaxies={n_gal} (cannot drop galaxies)")
    return max(base, math.ceil(target / K) * K)


@dataclass
class CatalogPosterior:
    """Container for N independent per-galaxy posteriors.

    Parameters
    ----------
    posteriors : list of Posterior
        One result per galaxy, in the same order as the input catalog.
    method : str
        Inference method used.
    wall_time_s : float
        Total wall-clock time for all galaxies. [s]
    n_galaxies : int
        Number of galaxies.
    diagnostics : dict
        Method-specific diagnostics (e.g. ``mean_n_iterations`` for native VI).
    percentiles : dict or None
        Per-galaxy percentile summaries when store="summary". Keys are property
        names, values are (n_galaxies, n_percentiles) arrays. None for store="full".
    summary : dict or None
        Per-galaxy summary statistics when store="summary". Keys are reducer names
        (e.g., "mean", "std"), values are dicts mapping property names to
        (n_galaxies,) arrays. None for store="full".
    store : str
        Storage mode: "full" keeps all samples, "summary" computes percentiles
        and reducers and drops samples to save memory.

    Raises
    ------
    IndexError
        If ``__getitem__`` is called with an out-of-range index.

    Examples
    --------
    >>> result = cat.run("native_vi_linear", key=jax.random.PRNGKey(0))
    >>> result[0].params  # first galaxy
    >>> for post in result:
    ...     ...  # iterate over all galaxies
    """

    posteriors: list
    method: str = ""
    wall_time_s: float = 0.0
    n_galaxies: int = 0
    diagnostics: dict = field(default_factory=dict)
    percentiles: dict | None = None
    summary: dict | None = None
    store: str = "full"

    def __getitem__(self, i):
        return self.posteriors[i]

    def __iter__(self):
        return iter(self.posteriors)

    def __len__(self):
        return len(self.posteriors)

    @functools.cached_property
    def properties(self):
        """The property catalog over the galaxy axis.

        Contract §1 — **same names, more axes**. The keys are the ones a single
        :class:`~tengri.inference.posterior.Posterior` answers to; the leading
        axis is now the galaxy.

        Returns
        -------
        CatalogProperties
            Dict-like: ``[name]`` -> shape ``(n_galaxies, n_samples)`` when every
            galaxy has the same number of draws (or ``(n_galaxies,)`` for MAP
            fits), else a list of per-galaxy arrays. ``ci(name)`` gives the
            per-galaxy credible interval, shape ``(n_galaxies, 3)``.

        Examples
        --------
        >>> cat = fitter.run("map")  # doctest: +SKIP
        >>> cat.properties["stellar_mass"].shape  # doctest: +SKIP
        (500,)
        """
        return CatalogProperties(self)

    def to_table(self) -> dict:
        """Export posteriors as a table dict (round-trips through ingest_catalog).

        Returns
        -------
        dict[str, np.ndarray]
            Dict mapping property names to (N,) arrays over galaxies. Includes
            medians from the per-galaxy posteriors and percentile columns when
            percentiles are stored (e.g., "stellar_mass_p16", "stellar_mass_p50",
            "stellar_mass_p84" for percentiles=[16, 50, 84]).

        Notes
        -----
        The returned dict is a duck-type match for the input to
        :func:`~tengri.inference.catalog_ingest.ingest_catalog`, enabling
        round-trip workflows: ``Catalog(..., table).fit() -> cat.to_table()``.
        """
        table = {}

        # If percentiles are stored (store='summary' case), use them preferentially
        if self.percentiles:
            # Extract medians from percentile columns
            first_name = next(iter(self.percentiles.keys()))
            n_pct = self.percentiles[first_name].shape[1]

            # Infer percentile values from the shape (assumes standard 16/50/84 or similar)
            pct_values = [16, 50, 84]
            if n_pct == 1:
                pct_values = [50]
            elif n_pct == 2:
                pct_values = [16, 84]
            elif n_pct > 3:
                # Fallback: evenly spaced
                pct_values = np.linspace(0, 100, n_pct).astype(int).tolist()

            # Add median (index 1 for [16, 50, 84]) as the main column
            for name, percentile_array in self.percentiles.items():
                # Median is typically at index 1 (50th percentile)
                median_idx = min(1, n_pct - 1)  # fallback to 0 if only 1 pct
                table[name] = percentile_array[:, median_idx]

                # Add all percentile columns
                for i, pct in enumerate(pct_values[:n_pct]):
                    col_name = f"{name}_p{pct}"
                    table[col_name] = percentile_array[:, i]

        else:
            # store='full' case: try to get properties, with graceful fallback
            try:
                props = self.properties
                for name in props:
                    # Get the per-galaxy values
                    vals = props[name]
                    if isinstance(vals, list):
                        # Ragged posteriors — convert to array
                        medians = [np.median(np.asarray(v)) if np.ndim(v) > 0 else v for v in vals]
                        vals = np.array(medians)
                    else:
                        # Stacked posteriors
                        if vals.ndim > 1:
                            # Has sample axis — take median along it
                            vals = np.median(vals, axis=1)
                    table[name] = vals
            except (RuntimeError, KeyError, AttributeError):
                # No model available; fall back to parameter median
                if self.posteriors:
                    # Just include the parameter values as a fallback
                    first_post = self.posteriors[0]
                    if first_post.params:
                        for param_name in first_post.params:
                            vals = np.array([p.params[param_name] for p in self.posteriors])
                            table[param_name] = np.asarray(vals)

        return table

    def __repr__(self) -> str:
        return (
            f"CatalogPosterior(n_galaxies={self.n_galaxies}, "
            f"method='{self.method}', wall_time={self.wall_time_s:.1f}s)"
        )


class CatalogProperties:
    """The property catalog lifted over the galaxy axis of a :class:`CatalogPosterior`.

    A ``CatalogPosterior`` is a *list of independent* ``Posterior`` objects, not
    a batched array — each galaxy was fit separately and may carry a different
    number of draws. So the lift here is a **stack over galaxies**, not a vmap:
    each galaxy's own (already chunk-vmapped) property array is gathered, and the
    results are stacked when their shapes agree and returned as a list when they
    do not. Ragged posteriors are a fact of catalog fitting, and silently padding
    or truncating them would be worse than handing back the list.
    """

    def __init__(self, catalog):
        object.__setattr__(self, "_catalog", catalog)
        object.__setattr__(self, "_cache", {})

    def _posteriors(self):
        return object.__getattribute__(self, "_catalog").posteriors

    def __getitem__(self, name: str):
        cache = object.__getattribute__(self, "_cache")
        if name in cache:
            return cache[name]

        per_galaxy = [p.properties[name] for p in self._posteriors()]
        shapes = {np.shape(v) for v in per_galaxy}
        value = np.stack([np.asarray(v) for v in per_galaxy]) if len(shapes) == 1 else per_galaxy

        cache[name] = value
        return value

    def __contains__(self, name: str) -> bool:
        posts = self._posteriors()
        return bool(posts) and name in posts[0].properties

    def __iter__(self):
        posts = self._posteriors()
        return iter(posts[0].properties.keys() if posts else [])

    def keys(self):
        """Available property names — identical to the per-galaxy ones."""
        return list(self)

    def to_dict(self, names=None) -> dict:
        """Export properties as a plain dict keyed by name."""
        if names is None:
            names = list(self)
        return {name: self[name] for name in names}

    def ci(self, name: str, level: float = 0.68) -> np.ndarray:
        """Per-galaxy credible interval.

        Returns
        -------
        ndarray, shape (n_galaxies, 3)
            ``(lo, median, hi)`` per galaxy.
        """
        return np.array([p.properties.ci(name, level=level) for p in self._posteriors()])

    def __setattr__(self, name, value):
        raise AttributeError("CatalogProperties is read-only")

    def __repr__(self):
        return (
            f"<CatalogProperties: {len(self.keys())} properties over "
            f"{len(self._posteriors())} galaxies>"
        )


class _CatalogFitterOriginal:
    """Per-galaxy catalog inference with optional K-way on-device parallelism.

    Wraps all :class:`~tengri.inference.fitter.Fitter` inference methods with a
    single ``run(method, ...)`` entry point. For ``native_vi_linear`` and
    ``native_vi_nonlinear``, setting ``forward_chunk_size=K`` vmaps K galaxies
    per ``lax.map`` iteration so K galaxies execute in parallel on the
    accelerator while the XLA graph remains O(1) in the catalog size N.

    Parameters
    ----------
    model : SEDModel
        Forward model shared across all galaxies.
    galaxies : list of dict
        Each dict must contain ``'flux_obs'`` array, shape ``(n_data,)`` [erg/s/Hz],
        and ``'noise'`` array, shape ``(n_data,)`` [erg/s/Hz] (per-band 1-sigma errors).
        For ``forward_chunk_size > 1`` with native methods, all galaxies must
        have the same ``n_data``.
    data_type : str
        ``"photometry"`` (default), ``"spectroscopy"``, or ``"joint"``.

    Raises
    ------
    ValueError
        If ``forward_chunk_size > 1`` and galaxies have different ``n_data``
        (raised by :meth:`_validate_uniform_data`).

    Notes
    -----
    The ``signal_response`` (forward model) is built once from the first galaxy
    and shared across all galaxies — it does not capture any galaxy-specific
    data. The per-galaxy ``data`` and ``noise`` vectors are runtime arguments
    to the catalog VI engines, enabling ``jax.vmap`` over the catalog batch.

    For non-native methods (e.g. ``vi_nonlinear``, ``map``, ``mcmc``),
    ``CatalogFitter`` delegates to sequential :class:`Fitter` instances.
    JAX's XLA persistent cache means only the first galaxy pays the compilation cost.

    **Not JIT-compatible at the Python level** — ``CatalogFitter`` is a Python
    orchestrator; the individual catalog VI engine callables it dispatches to are
    JIT-compiled and vmap-compatible.

    Examples
    --------
    >>> cat = CatalogFitter(model, galaxies)
    >>> result = cat.run("native_vi_linear", key=jax.random.PRNGKey(0), forward_chunk_size=4)
    >>> result[0].params  # first galaxy posterior
    """

    _NATIVE_VMAPPABLE: frozenset = frozenset({"native_vi_linear", "native_vi_nonlinear"})
    #: Sampling methods that honor ``forward_chunk_size`` by vmapping K galaxies'
    #: NUTS/HMC chains per ``lax.map`` step (per-galaxy warmup, diagonal mass).
    _MCMC_VMAPPABLE: frozenset = frozenset({"mcmc_nuts", "mcmc_hmc"})

    def __init__(self, model, galaxies, data_type="photometry"):
        from tengri.inference.jit_engine import CompileCache

        self.model = model
        self.galaxies = list(galaxies)
        self.n_galaxies = len(self.galaxies)
        self.data_type = data_type
        self._dummy_fitter = None
        self._catalog_linear_engine = None
        self._catalog_nonlinear_engine = None
        # Create a single CompileCache for all per-galaxy Fitter instances.
        # This prevents cross-galaxy cache evictions: each galaxy's compile
        # stays in the same bounded cache, not in competing global singletons.
        self.cache = CompileCache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        method="native_vi_linear",
        *,
        key,
        forward_chunk_size=1,
        n_pad: int | str | None = None,
        devices=None,
        store: str | None = None,
        percentiles: tuple | None = None,
        reducers: dict | None = None,
        **kwargs,
    ):
        """Fit all galaxies independently.

        Parameters
        ----------
        method : str
            Any method accepted by :class:`~tengri.inference.fitter.Fitter`.
            ``native_vi_linear`` (default) and ``native_vi_nonlinear`` support
            ``forward_chunk_size``-based on-device parallelism.
        key : jax.random.PRNGKey
            Base random key; per-galaxy keys are derived via ``jax.random.split``.
        forward_chunk_size : int
            K galaxies evaluated in parallel per ``lax.map`` step.
            Only applies to ``native_vi_linear`` / ``native_vi_nonlinear``.
            ``K=1`` (default) = sequential; ``K=N`` = fully vmapped.
        n_pad : int, "auto", or None
            Pad the catalog up to this many galaxies before running. The
            extra slots are dummy galaxies whose results are discarded
            after the run; their only purpose is to make the XLA program
            shape match a previously-cached compile so different catalog
            sizes share an artifact.

            - ``None`` (default) — pad only to the next multiple of K
              (existing behavior).
            - ``"auto"`` — pad to the next power of 2.
            - ``int`` — pad to exactly this many galaxies (must be
              ``>= n_galaxies``).

            Only applies to native methods. Ignored with a warning for
            sequential paths (each galaxy is fit in its own jit, so
            shape-bucketing has no effect).
        store : {"full", "summary"} or None
            Storage mode for posterior samples. ``None`` (default) auto-selects:
            ``"full"`` if N <= 1000, else ``"summary"`` with a warning.
            ``"full"`` retains all samples. ``"summary"`` computes percentiles
            and reducer statistics per property, then drops samples.
        percentiles : tuple, optional
            Percentiles to compute when store="summary". Default (16, 50, 84).
        reducers : dict, optional
            Additional reducer functions {name: callable} to apply per property
            (e.g., {"mean": jnp.mean, "std": jnp.std}). With store="full", these
            are ignored.
        **kwargs
            Forwarded to the underlying inference method.

        Returns
        -------
        CatalogPosterior

        Raises
        ------
        ValueError
            If ``forward_chunk_size > 1`` and galaxies have heterogeneous ``n_data``,
            or if ``n_pad`` is an int below ``n_galaxies``.
        UserWarning
            If ``forward_chunk_size != 1`` is passed for a non-native method (ignored).

        Notes
        -----
        Padding is safe in :class:`CatalogFitter` because the catalog VI
        engine is fully per-galaxy: each galaxy's ``run_fn`` operates on
        only its own ``(init_pos, key, data, noise)`` with no cross-galaxy
        reduction. Dummy padded galaxies converge to their own (irrelevant)
        posteriors and are trimmed off the result. The same trick is
        **not** safe for :class:`PopulationFitter`, where the hierarchical
        population field couples all galaxies — there, rely on
        :func:`tengri.enable_persistent_cache` instead.
        """
        from tengri.inference.fitter import resolve_method

        # Auto-select store mode based on catalog size if not specified
        if store is None:
            if self.n_galaxies <= 1000:
                store = "full"
            else:
                store = "summary"
                warnings.warn(
                    f"Catalog has {self.n_galaxies} galaxies > 1000; "
                    f"automatically switching to store='summary' to save memory. "
                    f"Samples will be dropped after computing percentiles. "
                    f"Pass store='full' to retain all samples.",
                    UserWarning,
                    stacklevel=2,
                )

        if percentiles is None:
            percentiles = (16, 50, 84)

        resolved = resolve_method(method)
        # Per-galaxy fixed-value overrides (e.g. redshift) are threaded only through
        # the sequential path today; the batched native/MCMC paths stack flux/noise
        # and would SILENTLY DROP per-galaxy redshift. Fail loudly instead.
        if (resolved in self._NATIVE_VMAPPABLE or resolved in self._MCMC_VMAPPABLE) and any(
            "redshift" in g for g in self.galaxies
        ):
            raise NotImplementedError(
                f"Per-galaxy redshift is not yet supported for batched method "
                f"{method!r}. Use method='map' (the Catalog default; sequential) for "
                f"per-galaxy redshifts — batched per-galaxy redshift threading is a "
                f"follow-up. See #1317."
            )
        if resolved in self._NATIVE_VMAPPABLE:
            return self._run_native(
                resolved,
                key=key,
                forward_chunk_size=forward_chunk_size,
                n_pad=n_pad,
                store=store,
                percentiles=percentiles,
                reducers=reducers,
                **kwargs,
            )
        elif resolved in self._MCMC_VMAPPABLE:
            return self._run_native_mcmc(
                resolved,
                key=key,
                forward_chunk_size=forward_chunk_size,
                n_pad=n_pad,
                devices=devices,
                store=store,
                percentiles=percentiles,
                reducers=reducers,
                **kwargs,
            )
        else:
            if devices is not None:
                warnings.warn(
                    f"devices={devices!r} is ignored for method={method!r}. "
                    "Multi-device sharding is currently supported for "
                    "mcmc_nuts / mcmc_hmc only.",
                    UserWarning,
                    stacklevel=2,
                )
            if forward_chunk_size != 1:
                warnings.warn(
                    f"forward_chunk_size={forward_chunk_size} is ignored for "
                    f"method={method!r}. Only native_vi_linear and "
                    "native_vi_nonlinear support chunked parallelism.",
                    UserWarning,
                    stacklevel=2,
                )
            if n_pad is not None:
                warnings.warn(
                    f"n_pad={n_pad!r} is ignored for method={method!r}. "
                    "Sequential per-galaxy fits don't benefit from "
                    "shape-bucketing — each galaxy is its own jit.",
                    UserWarning,
                    stacklevel=2,
                )
            return self._run_sequential(
                method,
                key=key,
                store=store,
                percentiles=percentiles,
                reducers=reducers,
                **kwargs,
            )

    # ------------------------------------------------------------------
    # Internal: native vmapped path
    # ------------------------------------------------------------------

    def _get_dummy_fitter(self):
        if self._dummy_fitter is None:
            from tengri.inference.fitter import Fitter

            g = self.galaxies[0]
            self._dummy_fitter = Fitter(
                self.model, g["flux_obs"], g["noise"], data_type=self.data_type, cache=self.cache
            )
        return self._dummy_fitter

    def _get_catalog_linear_engine(self):
        if self._catalog_linear_engine is None:
            from tengri.inference.backends.vi.native import (
                build_native_vi_catalog_linear_engine,
            )
            from tengri.inference.jit_engine import get_or_build_signal_response

            fitter = self._get_dummy_fitter()
            sr, _ = get_or_build_signal_response(fitter)
            dummy_pos = fitter._initialize_unbounded(jax.random.PRNGKey(0))
            _flat0, unravel = ravel_pytree(dummy_pos)

            def _flatten(p):
                return ravel_pytree(p)[0]

            run_fn, draw_fn, hamiltonian_fn = build_native_vi_catalog_linear_engine(
                sr, _flatten, unravel
            )
            self._catalog_linear_engine = (run_fn, draw_fn, hamiltonian_fn, _flatten, unravel)
        return self._catalog_linear_engine

    def _get_catalog_nonlinear_engine(self):
        if self._catalog_nonlinear_engine is None:
            from tengri.inference.backends.vi.native import (
                build_native_vi_catalog_nonlinear_engine,
            )
            from tengri.inference.jit_engine import get_or_build_signal_response

            fitter = self._get_dummy_fitter()
            sr, _ = get_or_build_signal_response(fitter)
            dummy_pos = fitter._initialize_unbounded(jax.random.PRNGKey(0))
            _flat0, unravel = ravel_pytree(dummy_pos)

            def _flatten(p):
                return ravel_pytree(p)[0]

            run_fn, draw_fn, hamiltonian_fn = build_native_vi_catalog_nonlinear_engine(
                sr, _flatten, unravel
            )
            self._catalog_nonlinear_engine = (run_fn, draw_fn, hamiltonian_fn, _flatten, unravel)
        return self._catalog_nonlinear_engine

    def _validate_uniform_data(self):
        """Raise if galaxies have different n_data (required for vmap)."""
        n_data = len(self.galaxies[0]["flux_obs"])
        for i, g in enumerate(self.galaxies[1:], 1):
            if len(g["flux_obs"]) != n_data:
                raise ValueError(
                    f"forward_chunk_size > 1 requires all galaxies to have the same "
                    f"number of data points. Galaxy 0 has {n_data}, galaxy {i} has "
                    f"{len(g['flux_obs'])}. Use forward_chunk_size=1 for heterogeneous catalogs."
                )
        return n_data

    def _run_native(
        self,
        method_tag,
        *,
        key,
        forward_chunk_size=1,
        n_pad: int | str | None = None,
        store: str = "full",
        percentiles: tuple = (16, 50, 84),
        reducers: dict | None = None,
        n_iterations=20,
        n_samples=3,
        n_posterior_samples=500,
        kl_rtol=1e-2,
        verbose=True,
    ):
        from tengri.inference.posterior import Posterior

        t0 = time.time()
        K = max(1, int(forward_chunk_size))
        n_gal = self.n_galaxies
        n_padded = _resolve_n_padded(n_gal, K, n_pad)
        n_pad_extra = n_padded - n_gal

        n_data = self._validate_uniform_data()

        if method_tag == "native_vi_linear":
            run_fn, draw_fn, _hamiltonian_fn, flatten, unflatten = (
                self._get_catalog_linear_engine()
            )
        else:
            run_fn, draw_fn, _hamiltonian_fn, flatten, unflatten = (
                self._get_catalog_nonlinear_engine()
            )

        fitter = self._get_dummy_fitter()
        dummy_flat = flatten(fitter._initialize_unbounded(jax.random.PRNGKey(0)))
        d_params = dummy_flat.shape[0]

        if verbose:
            tag = "MGVI" if method_tag == "native_vi_linear" else "geoVI"
            print(
                f"CatalogFitter ({tag}): {n_gal} galaxies, "
                f"K={K}, D={d_params}, {n_iterations} max iters"
            )

        # Stack data and noise: each galaxy may have different noise values
        # but must have the same number of data points for vmap.
        all_data_orig = jnp.stack([jnp.asarray(g["flux_obs"]) for g in self.galaxies])
        all_noise_orig = jnp.stack([jnp.asarray(g["noise"]) for g in self.galaxies])

        if n_pad_extra > 0:
            all_data = jnp.concatenate([all_data_orig, jnp.zeros((n_pad_extra, n_data))], axis=0)
            # Use noise=1 for padded slots to avoid division by zero; their outputs are trimmed.
            all_noise = jnp.concatenate([all_noise_orig, jnp.ones((n_pad_extra, n_data))], axis=0)
        else:
            all_data = all_data_orig
            all_noise = all_noise_orig

        # Random per-galaxy init (sequential MAP would be N×compilation cost)
        init_keys = jax.random.split(key, n_padded)
        all_init = jnp.stack([flatten(fitter._initialize_unbounded(k)) for k in init_keys[:n_gal]])
        if n_pad_extra > 0:
            all_init = jnp.concatenate([all_init, jnp.zeros((n_pad_extra, d_params))], axis=0)

        run_keys = jax.random.split(jax.random.fold_in(key, 1), n_padded)

        if verbose:
            print("  Compiling JIT engine (first call only)...")

        # --- lax.map(batch_size=K): scan over n_padded/K vmaps of size K. ---
        # batch_size=K handles non-divisible n_padded internally, but n_pad is
        # still useful for amortizing XLA compile cost across catalog sizes
        # (e.g., always pad to power-of-2 to reuse the persistent cache).
        def run_one(args):
            ini, k, d, n = args
            return run_fn(ini, k, d, n, n_iterations, n_samples, kl_rtol)

        all_best_flat, all_n_iters = jax.lax.map(
            run_one,
            (all_init, run_keys, all_data, all_noise),
            batch_size=K,
        )
        all_best_flat = all_best_flat[:n_gal]
        all_n_iters = jnp.asarray(all_n_iters)[:n_gal]
        jax.block_until_ready(all_best_flat)

        if verbose:
            mean_iters = float(jnp.mean(all_n_iters))
            print(f"  Optimization done. Mean iterations: {mean_iters:.1f}")
            print(f"  Drawing {n_posterior_samples} posterior samples per galaxy...")

        # --- Draw posterior samples per galaxy (sequential over galaxies) ---
        draw_key = jax.random.fold_in(key, 99999)
        is_nonlinear = method_tag == "native_vi_nonlinear"
        n_draw = max(1, n_posterior_samples // 2) if is_nonlinear else n_posterior_samples

        posteriors = []
        for i in range(n_gal):
            pos_i = all_best_flat[i]
            noise_i = all_noise_orig[i]
            draw_keys_i = jax.random.split(jax.random.fold_in(draw_key, i), n_draw)

            residuals_i = draw_fn(pos_i, draw_keys_i, noise_i)
            if is_nonlinear:
                residuals_i = residuals_i[:n_posterior_samples]

            converged_dict = unflatten(pos_i)
            sample_dicts = []
            for s in range(residuals_i.shape[0]):
                res = unflatten(residuals_i[s])
                combined = {k: converged_dict[k] + res[k] for k in converged_dict}
                sample_dicts.append(fitter._to_physical(combined))

            samples_phys = {k: jnp.stack([sd[k] for sd in sample_dicts]) for k in sample_dicts[0]}
            best_params = _mean_params(samples_phys)

            # Compute percentiles and summary if store="summary"
            percentiles_i = None
            summary_i = None
            samples_to_store = samples_phys
            if store == "summary":
                percentiles_i, summary_i = _compute_summaries(samples_phys, percentiles, reducers)
                samples_to_store = None  # Drop samples to save memory

            post_i = Posterior(
                samples=samples_to_store,
                params=best_params,
                method=f"CatalogFitter/{method_tag}",
                wall_time_s=time.time() - t0,
                diagnostics={"n_iterations": int(all_n_iters[i])},
                loss_history=None,
                _model=self.model,
            )

            # Attach summaries if computed
            if percentiles_i is not None:
                post_i._percentiles_stats_ = percentiles_i
            if summary_i is not None:
                post_i._summary_stats_ = summary_i

            posteriors.append(post_i)

        wall = time.time() - t0
        if verbose:
            print(f"  Done in {wall:.1f}s ({wall / n_gal:.2f}s/galaxy)")

        # Stack percentiles and summary across galaxies if they exist
        cat_percentiles = None
        cat_summary = None
        if store == "summary" and posteriors and hasattr(posteriors[0], "_percentiles_stats_"):
            cat_percentiles = {}
            first_post = posteriors[0]
            for name in first_post._percentiles_stats_:
                cat_percentiles[name] = np.stack([p._percentiles_stats_[name] for p in posteriors])

            if hasattr(posteriors[0], "_summary_stats_") and posteriors[0]._summary_stats_:
                cat_summary = {}
                for reducer_name in posteriors[0]._summary_stats_:
                    cat_summary[reducer_name] = {}
                    for name in posteriors[0]._summary_stats_[reducer_name]:
                        cat_summary[reducer_name][name] = np.array(
                            [p._summary_stats_[reducer_name][name] for p in posteriors]
                        )

        return CatalogPosterior(
            posteriors=posteriors,
            method=method_tag,
            wall_time_s=wall,
            n_galaxies=n_gal,
            diagnostics={"mean_n_iterations": float(jnp.mean(all_n_iters))},
            percentiles=cat_percentiles,
            summary=cat_summary,
            store=store,
        )

    # ------------------------------------------------------------------
    # Internal: vectorized sampling path (mcmc_nuts / mcmc_hmc)
    # ------------------------------------------------------------------

    def _run_native_mcmc(
        self,
        method_tag,
        *,
        key,
        forward_chunk_size=1,
        n_pad: int | str | None = None,
        devices=None,
        store: str = "full",
        percentiles: tuple = (16, 50, 84),
        reducers: dict | None = None,
        n_warmup=300,
        n_burnin=100,
        n_samples=1000,
        max_num_doublings=10,
        n_leapfrog_steps=10,
        target_accept_rate=0.85,
        dense_mass_matrix=False,
        verbose=True,
    ):
        """Vectorized per-galaxy NUTS/HMC sampling via ``lax.map(batch_size=K)``.

        Each galaxy runs its own BlackJAX window adaptation and chain inside a
        single JIT'd program; K galaxies execute per ``lax.map`` step so the
        compiled graph stays O(1) in the catalog size while K chains run in
        parallel on the accelerator. Returns one :class:`Posterior` per galaxy,
        each carrying posterior ``samples`` — the same public contract as the
        sequential path, minus the N serial warmups.

        Diagonal mass matrix is the default (``dense_mass_matrix=False``): each
        galaxy is low-D and the parallelism is *width over galaxies*, so a
        diagonal mass keeps the vmap flat and dodges the dense-mass warmup
        memory spike (see :func:`...mcmc.nuts.run_nuts`).

        When ``devices`` is given (``"all"`` or a device list), the galaxy axis
        is sharded across those devices via ``jax.shard_map`` — each device runs
        ``lax.map`` on its own slice of the catalog with no cross-device
        reduction (galaxies are independent). Bit-parity with the single-device
        path holds up to float round-off.
        """
        from tengri.inference.backends.mcmc.catalog import build_catalog_mcmc_engine
        from tengri.inference.posterior import Posterior

        sampler = "nuts" if method_tag == "mcmc_nuts" else "hmc"
        t0 = time.time()
        K = max(1, int(forward_chunk_size))
        n_gal = self.n_galaxies
        dev_list = self._resolve_devices(devices)
        n_dev = len(dev_list) if dev_list else 1
        if n_dev > 1:
            # Pad to a multiple of both K (the lax.map chunk) and n_dev (even
            # shards) so every device gets an equal, K-divisible slice.
            unit = math.lcm(K, n_dev)
            n_padded = math.ceil(n_gal / unit) * unit
        else:
            n_padded = _resolve_n_padded(n_gal, K, n_pad)
        n_pad_extra = n_padded - n_gal
        n_data = self._validate_uniform_data()

        fitter = self._get_dummy_fitter()
        run_one, unravel_fn = build_catalog_mcmc_engine(
            fitter,
            sampler,
            n_warmup=n_warmup,
            n_burnin=n_burnin,
            n_samples=n_samples,
            max_num_doublings=max_num_doublings,
            n_leapfrog=n_leapfrog_steps,
            target_accept_rate=target_accept_rate,
            use_dense=bool(dense_mass_matrix),
        )

        dummy_flat = ravel_pytree(fitter._initialize_unbounded(jax.random.PRNGKey(0)))[0]
        d_params = dummy_flat.shape[0]

        if verbose:
            tag = "NUTS" if sampler == "nuts" else "HMC"
            print(
                f"CatalogFitter ({tag}): {n_gal} galaxies, K={K}, D={d_params}, "
                f"{n_warmup} warmup + {n_samples} samples"
            )

        # Stack per-galaxy data; pad with dummy galaxies (trimmed after).
        all_data_orig = jnp.stack([jnp.asarray(g["flux_obs"]) for g in self.galaxies])
        all_noise_orig = jnp.stack([jnp.asarray(g["noise"]) for g in self.galaxies])
        if n_pad_extra > 0:
            all_data = jnp.concatenate([all_data_orig, jnp.zeros((n_pad_extra, n_data))], axis=0)
            all_noise = jnp.concatenate([all_noise_orig, jnp.ones((n_pad_extra, n_data))], axis=0)
        else:
            all_data, all_noise = all_data_orig, all_noise_orig

        init_keys = jax.random.split(key, n_padded)
        all_init = jnp.stack(
            [ravel_pytree(fitter._initialize_unbounded(k))[0] for k in init_keys[:n_gal]]
        )
        if n_pad_extra > 0:
            all_init = jnp.concatenate([all_init, jnp.zeros((n_pad_extra, d_params))], axis=0)
        gal_keys = jax.random.split(jax.random.fold_in(key, 1), n_padded)

        def _run_one(args):
            ini, gk, d, n = args
            return run_one(ini, gk, d, n)

        xs = (all_init, gal_keys, all_data, all_noise)
        if n_dev > 1:
            all_positions, all_divergent = self._sharded_vmap(run_one, xs, dev_list)
        else:
            all_positions, all_divergent = jax.lax.map(_run_one, xs, batch_size=K)
        all_positions = all_positions[:n_gal]
        all_divergent = all_divergent[:n_gal]
        jax.block_until_ready(all_positions)

        posteriors = []
        for i in range(n_gal):
            samples_phys = _vmap_samples_to_physical(
                all_positions[i], unravel_fn, fitter._to_physical
            )
            best_params = _mean_params(samples_phys)
            n_div = int(jnp.sum(all_divergent[i]))

            # Compute percentiles and summary if store="summary"
            percentiles_i = None
            summary_i = None
            samples_to_store = samples_phys
            if store == "summary":
                percentiles_i, summary_i = _compute_summaries(samples_phys, percentiles, reducers)
                samples_to_store = None  # Drop samples to save memory

            post_i = Posterior(
                samples=samples_to_store,
                params=best_params,
                method=f"CatalogFitter/{method_tag}",
                wall_time_s=time.time() - t0,
                diagnostics={"n_divergent": n_div, "n_samples": n_samples},
                loss_history=None,
                _model=self.model,
            )

            # Attach summaries if computed
            if percentiles_i is not None:
                post_i._percentiles_stats_ = percentiles_i
            if summary_i is not None:
                post_i._summary_stats_ = summary_i

            posteriors.append(post_i)

        wall = time.time() - t0
        if verbose:
            print(f"  Done in {wall:.1f}s ({wall / n_gal:.2f}s/galaxy)")

        # Stack percentiles and summary across galaxies if they exist
        cat_percentiles = None
        cat_summary = None
        if store == "summary" and posteriors and hasattr(posteriors[0], "_percentiles_stats_"):
            cat_percentiles = {}
            first_post = posteriors[0]
            for name in first_post._percentiles_stats_:
                cat_percentiles[name] = np.stack([p._percentiles_stats_[name] for p in posteriors])

            if hasattr(posteriors[0], "_summary_stats_") and posteriors[0]._summary_stats_:
                cat_summary = {}
                for reducer_name in posteriors[0]._summary_stats_:
                    cat_summary[reducer_name] = {}
                    for name in posteriors[0]._summary_stats_[reducer_name]:
                        cat_summary[reducer_name][name] = np.array(
                            [p._summary_stats_[reducer_name][name] for p in posteriors]
                        )

        return CatalogPosterior(
            posteriors=posteriors,
            method=method_tag,
            wall_time_s=wall,
            n_galaxies=n_gal,
            diagnostics={
                "vectorized": True,
                "sampler": sampler,
                "forward_chunk_size": K,
                "n_devices": n_dev,
                "n_divergent_total": int(jnp.sum(all_divergent)),
                "n_warmup": n_warmup,
                "n_samples": n_samples,
            },
            percentiles=cat_percentiles,
            summary=cat_summary,
            store=store,
        )

    @staticmethod
    def _resolve_devices(devices):
        """Normalize the ``devices`` argument to a device list or ``None``.

        ``None`` → single-device path; ``"all"`` → every ``jax.devices()``; a
        list/tuple of devices → those devices.
        """
        if devices is None:
            return None
        if isinstance(devices, str):
            if devices == "all":
                return list(jax.devices())
            raise ValueError(f"devices must be None, 'all', or a device list; got {devices!r}")
        return list(devices)

    @staticmethod
    def _sharded_vmap(run_one, xs, dev_list):
        """``jax.vmap(run_one)`` over a galaxy axis sharded across ``dev_list``.

        Per-galaxy fits are independent, so this is pure data parallelism with
        no cross-device reduction: the leading (galaxy) axis of every input is
        sharded over the devices and GSPMD distributes the vmapped program —
        each device samples its own slice of the catalog. GSPMD auto-partitioning
        is used rather than ``shard_map`` because BlackJAX's NUTS tree-builder
        contains ``lax.cond`` branches that trip ``shard_map``'s manual
        varying-axis tracking. The leading axis must be divisible by
        ``len(dev_list)`` (the caller pads to a multiple of ``lcm(K, n_dev)``).
        Returns the gathered ``(positions, divergent)``.
        """
        mesh = jax.sharding.Mesh(np.asarray(dev_list, dtype=object), ("gal",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("gal"))
        xs_sharded = jax.device_put(xs, sharding)
        run_all = jax.jit(jax.vmap(run_one))
        return run_all(*xs_sharded)

    # ------------------------------------------------------------------
    # Internal: sequential fallback (all other methods)
    # ------------------------------------------------------------------

    def _run_sequential(
        self,
        method,
        *,
        key,
        store: str = "full",
        percentiles: tuple = (16, 50, 84),
        reducers: dict | None = None,
        verbose=True,
        **kwargs,
    ):
        from tengri.inference.fitter import Fitter

        t0 = time.time()
        keys = jax.random.split(key, self.n_galaxies)
        posteriors = []

        # Smart lean (Fitter.run default) keeps the L3 entry that matches
        # (compile_signature, method) across runs and drops only stale
        # entries. For a same-shape catalog every galaxy hits the cache;
        # for a mixed-shape catalog the prior entry is dropped before
        # the next compile, bounding peak RAM. No persistent() wrap
        # needed — the wrap was strictly worse for mixed-shape loops.
        for i, galaxy in enumerate(self.galaxies):
            if verbose:
                print(f"  Galaxy {i + 1}/{self.n_galaxies}...", end="\r", flush=True)
            # Per-galaxy redshift (or any fixed-value override) reaches the forward
            # pass via the #1329 params-override seam, not just the reported params.
            override = {"redshift": galaxy["redshift"]} if "redshift" in galaxy else None
            fitter_i = Fitter(
                self.model,
                galaxy["flux_obs"],
                galaxy["noise"],
                data_type=self.data_type,
                cache=self.cache,
                params_override=override,
            )
            post_i = fitter_i.run(method, key=keys[i], verbose=False, **kwargs)

            # Compute percentiles and summary if store="summary"
            if store == "summary" and post_i.samples is not None:
                percentiles_i, summary_i = _compute_summaries(
                    post_i.samples, percentiles, reducers
                )
                post_i._percentiles_stats_ = percentiles_i
                if summary_i:
                    post_i._summary_stats_ = summary_i
                post_i.samples = None  # Drop samples to save memory

            posteriors.append(post_i)

        wall = time.time() - t0
        if verbose:
            per = wall / self.n_galaxies
            print(f"  {self.n_galaxies} galaxies done in {wall:.1f}s ({per:.2f}s/galaxy)")

        # Stack percentiles and summary across galaxies if they exist
        cat_percentiles = None
        cat_summary = None
        if store == "summary" and posteriors and hasattr(posteriors[0], "_percentiles_stats_"):
            cat_percentiles = {}
            first_post = posteriors[0]
            for name in first_post._percentiles_stats_:
                cat_percentiles[name] = np.stack([p._percentiles_stats_[name] for p in posteriors])

            if hasattr(posteriors[0], "_summary_stats_") and posteriors[0]._summary_stats_:
                cat_summary = {}
                for reducer_name in posteriors[0]._summary_stats_:
                    cat_summary[reducer_name] = {}
                    for name in posteriors[0]._summary_stats_[reducer_name]:
                        cat_summary[reducer_name][name] = np.array(
                            [p._summary_stats_[reducer_name][name] for p in posteriors]
                        )

        return CatalogPosterior(
            posteriors=posteriors,
            method=method,
            wall_time_s=wall,
            n_galaxies=self.n_galaxies,
            percentiles=cat_percentiles,
            summary=cat_summary,
            store=store,
        )


def __getattr__(name: str):
    """Emit a one-shot deprecation warning for CatalogFitter import."""
    if name == "CatalogFitter":
        warnings.warn(
            "CatalogFitter is deprecated and will be removed in tengri v1.0; "
            "use Catalog (from tengri.inference.catalog or tengri) instead. "
            "Catalog wraps CatalogFitter with table-in/table-out access and "
            "eager validation.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _CatalogFitterOriginal
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
