"""Helper functions for the explore performance notebooks.

Provides standardized timing, batch fitting, and plotting utilities
so the notebooks stay clean and focus on results.
"""

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri import Fitter


# ── Timing ────────────────────────────────────────────────────────


def time_fit(fitter, method, *, key=None, init_from=None,
             verbose=False, **kwargs):
    """Time a single fit, return (result, wall_time_s)."""
    if key is None:
        key = jax.random.PRNGKey(42)
    t0 = time.perf_counter()
    result = fitter.run(method, key=key, init_from=init_from,
                        verbose=verbose, **kwargs)
    dt = time.perf_counter() - t0
    return result, dt


def time_compile(fitter, *, verbose=False):
    """Time JIT engine compilation, return wall_time_s."""
    t0 = time.perf_counter()
    fitter.compile(verbose=verbose)
    return time.perf_counter() - t0


# ── Mock generation ───────────────────────────────────────────────


def make_mock_galaxy(model, spec, wave_obs, *, key, snr=30.0,
                     true_overrides=None):
    """Generate a mock galaxy spectrum.

    Returns (true_params, mock) tuple.
    """
    true_params = {**spec.sample(key)}
    if true_overrides:
        for k, v in true_overrides.items():
            true_params[k] = jnp.array(v)
    mock = model.mock_spectrum(true_params, wave_obs, snr=snr, key=key)
    return true_params, mock


def make_mock_batch(model, spec, wave_obs, n_galaxies, *,
                    base_key, snr=30.0):
    """Generate N mock galaxies with different random seeds.

    Returns (list_of_true_params, list_of_mocks).
    """
    keys = jax.random.split(base_key, n_galaxies)
    truths, mocks = [], []
    for k in keys:
        true_i, mock_i = make_mock_galaxy(
            model, spec, wave_obs, key=k, snr=snr,
        )
        truths.append(true_i)
        mocks.append(mock_i)
    return truths, mocks


# ── Batch fitting ─────────────────────────────────────────────────


def fit_batch_sequential(model, mocks, method, *,
                         base_key=None, verbose=True, **kwargs):
    """Fit N galaxies sequentially, timing each.

    Returns (results, times) lists.
    """
    if base_key is None:
        base_key = jax.random.PRNGKey(42)
    keys = jax.random.split(base_key, len(mocks))

    results, times = [], []
    for i, (mock_i, ki) in enumerate(zip(mocks, keys)):
        f_i = Fitter(model, mock_i.flux_obs, mock_i.noise)
        r_i, dt = time_fit(f_i, method, key=ki, verbose=False,
                           **kwargs)
        results.append(r_i)
        times.append(dt)
        if verbose:
            print(f"  Galaxy {i+1:>2d}/{len(mocks)}: {dt:.1f}s")

    return results, times


# ── Plotting ──────────────────────────────────────────────────────


def plot_batch_timing(times, compile_time=None, color="C0",
                      label="", cached_time=None, figdir=None,
                      filename=None):
    """Two-panel plot: per-galaxy time + amortized cost curve."""
    n = len(times)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: per-galaxy bar chart
    ax1.bar(range(1, n + 1), times, color=color)
    if cached_time is not None:
        ax1.axhline(cached_time, color="grey", ls=":", lw=1,
                     label=f"Cached: {cached_time:.0f}s")
        ax1.legend(fontsize=8)
    ax1.set_xlabel("Galaxy")
    ax1.set_ylabel("Wall time [s]")
    ax1.set_title(f"Per-galaxy fit time ({label})")

    # Right: amortized cost
    cum = np.cumsum(times)
    if compile_time is not None:
        amort = (compile_time + cum) / np.arange(1, n + 1)
        ax2.plot(range(1, n + 1), amort, "o-", color=color,
                 label="Amortized (+ compile)")
    marginal = cum / np.arange(1, n + 1)
    ax2.plot(range(1, n + 1), marginal, "s--", color="grey",
             label=f"Marginal: {np.mean(times):.1f}s")
    ax2.set_xlabel("Galaxies fitted")
    ax2.set_ylabel("Cost per galaxy [s]")
    ax2.set_title("Compile cost amortizes to zero")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, None)

    fig.tight_layout()
    if figdir and filename:
        import os
        plt.savefig(os.path.join(figdir, filename), dpi=150)
    plt.show()
    return fig


def plot_sfh_gallery(model, results, truths, times=None,
                     indices=None, color="C0", label="",
                     figdir=None, filename=None):
    """Grid of SFH recovery plots for a batch of galaxies."""
    from _plot_style import plot_sfh

    if indices is None:
        # Pick evenly spaced galaxies
        n = len(results)
        indices = np.linspace(0, n - 1, min(4, n), dtype=int)

    ncols = min(len(indices), 4)
    nrows = (len(indices) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 4 * nrows),
                             squeeze=False)

    for i, idx in enumerate(indices):
        ax = axes[i // ncols][i % ncols]
        plot_sfh(model, results[idx], true_params=truths[idx],
                 ax=ax, method="geoVI", label=label, color=color)
        title = f"Galaxy {idx + 1}"
        if times is not None:
            title += f" — {times[idx]:.1f}s"
        ax.set_title(title)

    for j in range(len(indices), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.tight_layout()
    if figdir and filename:
        import os
        plt.savefig(os.path.join(figdir, filename),
                    dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# ── Batch vmap (requires data-as-argument refactor) ───────────────


def fit_batch_vmap(fitter, mocks, *, base_key=None,
                   n_iterations=10, n_seeds=1, chunk_size=20,
                   verbose=True):
    """Fit N galaxies via jax.vmap over the compiled native_geovi.

    Processes galaxies in chunks of ``chunk_size`` to avoid OOM.
    Requires the engine to be cached on the Model (data-as-argument
    refactor). Falls back to sequential if vmap is not available.

    Parameters
    ----------
    fitter : Fitter
        A compiled Fitter (call fitter.compile() first).
        Used to access the cached engine on the Model.
    mocks : list of mock objects
        Each must have .flux_obs and .noise attributes.
    base_key : PRNGKey, optional
    n_iterations : int
    n_seeds : int
    chunk_size : int
        Max galaxies per vmap call. Default 50. Reduce if OOM.
        On CPU with D=135 and 200 spectral pixels, 50 uses ~2 GB.

    Returns
    -------
    results : list of Posterior
    wall_time : float
    """
    if base_key is None:
        base_key = jax.random.PRNGKey(42)

    n_gal = len(mocks)
    keys = jax.random.split(base_key, n_gal)

    # Check if the engine supports data_args as argument
    engine = getattr(fitter, "_jit_sampler", None)
    has_data_args = (
        engine is not None
        and hasattr(fitter, "_data_args")
    )

    if has_data_args:
        # Build batched data_args
        batch_data_args = {
            "data": jnp.stack([m.flux_obs for m in mocks]),
            "noise": jnp.stack([m.noise for m in mocks]),
            "noise_inv": jnp.stack(
                [1.0 / m.noise**2 for m in mocks]),
            "sqrt_noise_inv": jnp.stack(
                [jnp.sqrt(1.0 / m.noise**2) for m in mocks]),
            "n_data": fitter._data_args["n_data"],
        }

        # Get init positions (MAP for each galaxy)
        if verbose:
            print(f"fit_batch_vmap: {n_gal} galaxies, "
                  f"n_iter={n_iterations}")
            print("  Initializing MAP per galaxy...")

        init_flats = []
        flatten = engine["flatten"]
        for i, (mock_i, ki) in enumerate(zip(mocks, keys)):
            f_i = Fitter(fitter.model, mock_i.flux_obs, mock_i.noise)
            r_map = f_i.run("map", n_steps=500, key=ki, verbose=False)
            init_params = f_i._unbounded_from_posterior(r_map)
            init_flats.append(flatten(init_params))

        batch_init = jnp.stack(init_flats)
        batch_keys_opt = jnp.stack(
            [jax.random.fold_in(ki, 999) for ki in keys])

        run_fn = engine.get("run_evi_geovi")
        if run_fn is None:
            if verbose:
                print("  No run_evi_geovi in engine, "
                      "falling back to sequential")
            return fit_batch_sequential(
                fitter.model, mocks, "native_geovi",
                base_key=base_key, verbose=verbose,
                n_iterations=n_iterations, n_seeds=n_seeds,
            )

        # vmap over (init_pos, key, data_args)
        def _fit_one(init_pos, opt_key, da):
            return run_fn(
                init_pos, opt_key, da,
                n_iterations=n_iterations,
                n_samples=3,
                kl_rtol=1e-2,
                sample_mode="nonlinear_update",
            )

        vmapped_fit = jax.vmap(
            _fit_one,
            in_axes=(0, 0, {
                "data": 0, "noise": 0,
                "noise_inv": 0, "sqrt_noise_inv": 0,
                "n_data": None,
            }),
        )

        # Process in chunks to avoid OOM
        n_chunks = (n_gal + chunk_size - 1) // chunk_size
        if verbose:
            print(f"  Running vmapped native_geovi "
                  f"({n_chunks} chunk(s) of ≤{chunk_size})...")

        from tengri.inference.posterior import Posterior
        unflatten = engine["unflatten"]

        results = []
        t0 = time.perf_counter()

        for c in range(n_chunks):
            lo = c * chunk_size
            hi = min(lo + chunk_size, n_gal)
            chunk_init = batch_init[lo:hi]
            chunk_keys = batch_keys_opt[lo:hi]
            chunk_da = {
                k: v[lo:hi] if v.ndim > 0 and v.shape[0] == n_gal
                else v
                for k, v in batch_data_args.items()
            }

            chunk_converged, chunk_n_iters = vmapped_fit(
                chunk_init, chunk_keys, chunk_da,
            )

            for i in range(hi - lo):
                flat_i = chunk_converged[i]
                params_i = fitter._to_physical(unflatten(flat_i))
                results.append(Posterior(
                    samples=None,
                    params=params_i,
                    method="native_geovi (vmap)",
                    wall_time_s=0.0,
                ))

            if verbose and n_chunks > 1:
                print(f"    Chunk {c+1}/{n_chunks}: "
                      f"{hi - lo} galaxies")

        wall_time = time.perf_counter() - t0
        for r in results:
            r.wall_time_s = wall_time / n_gal

        if verbose:
            print(f"  Done: {n_gal} galaxies in {wall_time:.1f}s "
                  f"({wall_time / n_gal:.2f}s/galaxy)")

        return results, wall_time

    else:
        # Fallback: sequential fitting
        if verbose:
            print("fit_batch_vmap: data-as-argument refactor not "
                  "available, falling back to sequential")
        results, times = fit_batch_sequential(
            fitter.model, mocks, "native_geovi",
            base_key=base_key, verbose=verbose,
            n_iterations=n_iterations, n_seeds=n_seeds,
        )
        return results, sum(times)


def print_timing_summary(rows):
    """Print a formatted timing table.

    rows: list of (label, time_seconds) tuples.
    """
    print("=" * 50)
    print(f"{'Scenario':<35s} {'Time':>10s}")
    print("-" * 50)
    for label, t in rows:
        if isinstance(t, float):
            print(f"{label:<35s} {t:>9.1f}s")
        else:
            print(f"{label:<35s} {t:>10s}")
    print("=" * 50)
