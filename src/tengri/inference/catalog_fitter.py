"""Independent per-galaxy catalog inference with optional K-way parallelism.

Unlike :class:`PopulationFitter`, galaxies share no parameters.
:class:`CatalogFitter` supports every method :class:`Fitter` accepts; for
``native_vi_linear`` and ``native_vi_nonlinear`` it vmaps K galaxies per
``lax.map`` step so the compiled XLA graph stays O(1) in N while K galaxies
execute simultaneously on-device.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field

__all__ = ["CatalogFitter", "CatalogPosterior"]

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from tengri.inference._sample_utils import _mean_params


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

    def __getitem__(self, i):
        return self.posteriors[i]

    def __iter__(self):
        return iter(self.posteriors)

    def __len__(self):
        return len(self.posteriors)

    def __repr__(self) -> str:
        return (
            f"CatalogPosterior(n_galaxies={self.n_galaxies}, "
            f"method='{self.method}', wall_time={self.wall_time_s:.1f}s)"
        )


class CatalogFitter:
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

    def __init__(self, model, galaxies, data_type="photometry"):
        self.model = model
        self.galaxies = list(galaxies)
        self.n_galaxies = len(self.galaxies)
        self.data_type = data_type
        self._dummy_fitter = None
        self._catalog_linear_engine = None
        self._catalog_nonlinear_engine = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        method="native_vi_linear",
        *,
        key,
        forward_chunk_size=1,
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
        **kwargs
            Forwarded to the underlying inference method.

        Returns
        -------
        CatalogPosterior

        Raises
        ------
        ValueError
            If ``forward_chunk_size > 1`` and galaxies have heterogeneous ``n_data``.
        UserWarning
            If ``forward_chunk_size != 1`` is passed for a non-native method (ignored).
        """
        from tengri.inference.fitter import resolve_method

        resolved = resolve_method(method)
        if resolved in self._NATIVE_VMAPPABLE:
            return self._run_native(
                resolved,
                key=key,
                forward_chunk_size=forward_chunk_size,
                **kwargs,
            )
        else:
            if forward_chunk_size != 1:
                warnings.warn(
                    f"forward_chunk_size={forward_chunk_size} is ignored for "
                    f"method={method!r}. Only native_vi_linear and "
                    "native_vi_nonlinear support chunked parallelism.",
                    UserWarning,
                    stacklevel=2,
                )
            return self._run_sequential(method, key=key, **kwargs)

    # ------------------------------------------------------------------
    # Internal: native vmapped path
    # ------------------------------------------------------------------

    def _get_dummy_fitter(self):
        if self._dummy_fitter is None:
            from tengri.inference.fitter import Fitter

            g = self.galaxies[0]
            self._dummy_fitter = Fitter(
                self.model, g["flux_obs"], g["noise"], data_type=self.data_type
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
        n_padded = math.ceil(n_gal / K) * K
        n_chunks = n_padded // K
        n_pad = n_padded - n_gal

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

        if n_pad > 0:
            all_data = jnp.concatenate([all_data_orig, jnp.zeros((n_pad, n_data))], axis=0)
            # Use noise=1 for padded slots to avoid division by zero; their outputs are trimmed.
            all_noise = jnp.concatenate([all_noise_orig, jnp.ones((n_pad, n_data))], axis=0)
        else:
            all_data = all_data_orig
            all_noise = all_noise_orig

        # Random per-galaxy init (sequential MAP would be N×compilation cost)
        init_keys = jax.random.split(key, n_padded)
        all_init = jnp.stack([flatten(fitter._initialize_unbounded(k)) for k in init_keys[:n_gal]])
        if n_pad > 0:
            all_init = jnp.concatenate([all_init, jnp.zeros((n_pad, d_params))], axis=0)

        run_keys = jax.random.split(jax.random.fold_in(key, 1), n_padded)

        if verbose:
            print("  Compiling JIT engine (first call only)...")

        # --- Chunked vmap: lax.map over n_chunks, vmap over K per chunk ---
        def run_chunk(args):
            init_c, key_c, data_c, noise_c = args
            return jax.vmap(
                lambda ini, k, d, n: run_fn(ini, k, d, n, n_iterations, n_samples, kl_rtol)
            )(init_c, key_c, data_c, noise_c)

        chunked = (
            all_init.reshape(n_chunks, K, d_params),
            run_keys.reshape(n_chunks, K, *run_keys.shape[1:]),
            all_data.reshape(n_chunks, K, n_data),
            all_noise.reshape(n_chunks, K, n_data),
        )
        all_best_flat, all_n_iters = jax.lax.map(run_chunk, chunked)
        all_best_flat = all_best_flat.reshape(n_padded, d_params)[:n_gal]
        all_n_iters = jnp.asarray(all_n_iters).reshape(n_padded)[:n_gal]
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

            posteriors.append(
                Posterior(
                    samples=samples_phys,
                    params=best_params,
                    method=f"CatalogFitter/{method_tag}",
                    wall_time_s=time.time() - t0,
                    diagnostics={"n_iterations": int(all_n_iters[i])},
                    loss_history=None,
                    _model=self.model,
                )
            )

        wall = time.time() - t0
        if verbose:
            print(f"  Done in {wall:.1f}s ({wall / n_gal:.2f}s/galaxy)")

        return CatalogPosterior(
            posteriors=posteriors,
            method=method_tag,
            wall_time_s=wall,
            n_galaxies=n_gal,
            diagnostics={"mean_n_iterations": float(jnp.mean(all_n_iters))},
        )

    # ------------------------------------------------------------------
    # Internal: sequential fallback (all other methods)
    # ------------------------------------------------------------------

    def _run_sequential(self, method, *, key, verbose=True, **kwargs):
        from tengri.inference.fitter import Fitter

        t0 = time.time()
        keys = jax.random.split(key, self.n_galaxies)
        posteriors = []

        for i, galaxy in enumerate(self.galaxies):
            if verbose:
                print(f"  Galaxy {i + 1}/{self.n_galaxies}...", end="\r", flush=True)
            fitter_i = Fitter(
                self.model, galaxy["flux_obs"], galaxy["noise"], data_type=self.data_type
            )
            post_i = fitter_i.run(method, key=keys[i], verbose=False, **kwargs)
            posteriors.append(post_i)

        wall = time.time() - t0
        if verbose:
            per = wall / self.n_galaxies
            print(f"  {self.n_galaxies} galaxies done in {wall:.1f}s ({per:.2f}s/galaxy)")

        return CatalogPosterior(
            posteriors=posteriors,
            method=method,
            wall_time_s=wall,
            n_galaxies=self.n_galaxies,
        )
