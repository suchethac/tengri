# SPDX-License-Identifier: BSD-3-Clause
"""Convenience methods delegated from SEDModel.

Extracted from core/model.py to keep model.py focused on the forward model.
Each function takes (model, ...) where model is an SEDModel instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tengri.inference._backend_registry import DEFAULT_METHOD
from tengri.parameters.defaults import UNSET as _UNSET

if TYPE_CHECKING:
    from tengri.forward.sed_model import MockData, PriorPredictive, SEDModel


# ── Mock data generation ──────────────────────────────────────────


def _assemble_mock(params, flux_true, noise, key) -> MockData:
    """Assemble a :class:`MockData`, adding Gaussian noise when ``key`` is given."""
    from tengri.forward.sed_model import MockData

    if key is not None:
        flux_obs = flux_true + noise * jax.random.normal(key, shape=flux_true.shape)
    else:
        flux_obs = flux_true

    return MockData(
        flux_true=flux_true,
        flux_obs=flux_obs,
        noise=noise,
        params=params,
    )


def mock(model: SEDModel, params, snr=20.0, key=None) -> MockData:
    """Generate mock photometric observation with Gaussian noise.

    Parameters
    ----------
    model : SEDModel
        Forward model instance.
    params : dict
        Parameter values (public-space names).
    snr : float, optional
        Signal-to-noise ratio. Default 20.0.
    key : PRNGKey, optional
        JAX random key for noise generation. If None, returns noiseless observation.

    Returns
    -------
    MockData
        Mock observation with ``flux_true`` [erg/s/cm²/Hz],
        ``flux_obs`` [erg/s/cm²/Hz], ``noise`` [erg/s/cm²/Hz],
        and ``params`` dict.

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.
    """
    flux_true = model.predict_photometry(params)
    # No abs() here (unlike mock_spectrum), inherited difference kept bit-exact.
    return _assemble_mock(params, flux_true, flux_true / snr, key)


def mock_spectrum(model: SEDModel, params, wave_obs, snr=30.0, key=None) -> MockData:
    """Generate mock spectroscopic observation with Gaussian noise.

    Parameters
    ----------
    model : SEDModel
        Forward model instance.
    params : dict
        Parameter values (public-space names).
    wave_obs : array_like, shape (n_pix,)
        Observed wavelength grid [Angstrom].
    snr : float, optional
        Signal-to-noise ratio per pixel. Default 30.0.
    key : PRNGKey, optional
        JAX random key for noise generation. If None, returns noiseless observation.

    Returns
    -------
    MockData
        Mock observation with ``flux_true`` [erg/s/cm²/Hz],
        ``flux_obs`` [erg/s/cm²/Hz], ``noise`` [erg/s/cm²/Hz],
        and ``params`` dict.

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.
    """
    flux_true = model.predict_spectrum(params, wave_obs)
    return _assemble_mock(params, flux_true, jnp.abs(flux_true) / snr, key)


def mock_batch(model: SEDModel, params_batch, snr=20.0, key=None) -> MockData:
    """Generate batch of mock observations with Gaussian noise.

    Parameters
    ----------
    model : SEDModel
        Forward model instance.
    params_batch : dict of arrays
        Each value has shape (N, ...) with leading batch dimension.
    snr : float, optional
        Signal-to-noise ratio. Default 20.0.
    key : PRNGKey, optional
        JAX random key for noise generation. If None, returns noiseless observations.

    Returns
    -------
    MockData
        Mock observation batch with ``flux_true`` shape (N, n_filters) [erg/s/cm²/Hz],
        ``flux_obs`` shape (N, n_filters) [erg/s/cm²/Hz], ``noise`` shape (N, n_filters)
        [erg/s/cm²/Hz], and ``params`` dict of arrays.

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.
    """
    from tengri.forward.sed_model import MockData

    first_key = next(iter(params_batch))
    n_batch = params_batch[first_key].shape[0]

    def _get_single(i):
        """Extract i-th galaxy parameters from batched dict."""
        return {k: v[i] for k, v in params_batch.items()}

    if key is not None:
        noise_keys = jax.random.split(key, n_batch)
    else:
        noise_keys = [None] * n_batch

    results = [mock(model, _get_single(i), snr=snr, key=noise_keys[i]) for i in range(n_batch)]

    return MockData(
        flux_true=jnp.stack([r.flux_true for r in results]),
        flux_obs=jnp.stack([r.flux_obs for r in results]),
        noise=jnp.stack([r.noise for r in results]),
        params=params_batch,
    )


# ── Batch predictions (vmap over galaxies) ────────────────────────


def _threaded_batch(model, method_name, params_batch, ssp_data, template_data):
    """vmap ``method_name`` over the batch axis with the grids broadcast, not baked.

    ``jax.vmap(model.predict_photometry)`` closure-captures ``model``, so the SSP
    grid reaches the trace as a constant and a caller who wraps the batch helper
    in their own ``jax.jit`` has no channel to pass it in, the #1753 gap, one
    level up (#1793). Making the grids *arguments* of the vmapped function, with
    ``in_axes=None`` so they are shared rather than mapped, puts them on the same
    threading footing as the scalar surfaces.

    Resolution goes through ``_resolve_threaded_data`` so the override policy is
    stated once; ``ForwardModel`` delegates that to its inner SED.
    """
    resolve = getattr(model, "_resolve_threaded_data", None)
    if resolve is None:  # pragma: no cover - ForwardModel delegates; SEDModel defines it
        inner = model._inner_sed_for_delegation()
        resolve = inner._resolve_threaded_data
    ssp, templates = resolve(ssp_data, template_data)
    bound = getattr(model, method_name)

    def _one(params, ssp_arg, template_arg):
        return bound(params, ssp_data=ssp_arg, template_data=template_arg)

    return jax.vmap(_one, in_axes=(0, None, None))(params_batch, ssp, templates)


def predict_photometry_batch(model: SEDModel, params_batch, *, ssp_data=None, template_data=None):
    """Compute photometry for a batch of galaxies via vmap.

    Parameters
    ----------
    model : SEDModel
        Forward model instance.
    params_batch : dict of arrays
        Each value has shape (N, ...) with leading batch dimension.
    ssp_data, template_data : Any | None, keyword-only, optional
        The JIT-threading channel (#1793). ``None`` (default) uses the model's
        own arrays, which is correct for every ordinary call. Pass them only
        when wrapping this helper in your own JAX transform, so the grid enters
        as an argument instead of being frozen into your compiled program.

    Returns
    -------
    ndarray, shape (N, n_filters)
        Photometric flux density for each galaxy [erg/s/cm²/Hz].

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.
    See :func:`jax.vmap` for vmap internals.

    The grids are broadcast with ``in_axes=None``: one shared table across the
    batch, never a per-galaxy copy.
    """
    return _threaded_batch(model, "predict_photometry", params_batch, ssp_data, template_data)


def predict_spectrum_batch(model: SEDModel, params_batch, *, ssp_data=None, template_data=None):
    """Compute spectra for a batch of galaxies via vmap.

    Requires ``precompute_spectroscopy()`` to have been called.

    Parameters
    ----------
    model : SEDModel
        Forward model instance.
    params_batch : dict of arrays
        Each value has shape (N, ...) with leading batch dimension.
    ssp_data, template_data : Any | None, keyword-only, optional
        The JIT-threading channel, see :func:`predict_photometry_batch` (#1793).

    Returns
    -------
    ndarray, shape (N, n_pix)
        Spectral flux density for each galaxy [erg/s/cm²/Hz].

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.
    See :func:`jax.vmap` for vmap internals.
    """
    return _threaded_batch(model, "predict_spectrum", params_batch, ssp_data, template_data)


# ── Prior predictive check ────────────────────────────────────────


def prior_predictive(model: SEDModel, n: int = 500, seed: int = 42) -> PriorPredictive:
    """Sample prior and evaluate forward model on each draw.

    Returns a ``PriorPredictive`` object with draw arrays and convenience
    methods for prior predictive checking before inference.

    Parameters
    ----------
    model : SEDModel
        Forward model instance.
    n : int, optional
        Number of prior draws. Default 500.
    seed : int, optional
        Random seed for reproducibility. Default 42.

    Returns
    -------
    PriorPredictive
        Object with attributes:

        - ``flux``: ndarray, shape (n, n_filters) or None, photometry draws [erg/s/cm²/Hz]
        - ``sfh``: ndarray, shape (n, n_grid), SFR on internal time grid [Msun/yr]
        - ``params``: dict of arrays, shape (n,), parameter draws

    Notes
    -----
    **JIT-compatible**: no, uses Python-level for-loop to handle vmap failures gracefully.
    """
    from tengri.forward.sed_model import PriorPredictive

    key = jax.random.PRNGKey(seed)
    params_batch = model.spec.sample_batch(key, n)

    # SFH draws
    sfh_batch = jax.vmap(model.predict_sfh)(params_batch)

    # Photometry draws (if filters present)
    flux_batch = None
    if model.filter_waves is not None:
        try:
            flux_batch = jax.vmap(model.predict_photometry)(params_batch)
        except (TypeError, ValueError, RuntimeError, AttributeError):
            # TypeError: predict_photometry isn't vmappable
            # ValueError: params_batch structure incompatible with vmap
            # RuntimeError: JAX compilation error
            # AttributeError: predict_photometry doesn't behave as expected
            flux_batch = None

    return PriorPredictive(
        flux=flux_batch,
        sfh=sfh_batch,
        params=params_batch,
        _model=model,
    )


# ── Catalog fitting ───────────────────────────────────────────────


def fit_batch_map_vmap(
    model: SEDModel,
    fluxes,
    noises,
    *,
    n_steps: int = 500,
    learning_rate: float = 0.02,
    seed: int = 0,
    verbose: bool = True,
):
    """Batched MAP inference via vmap across galaxies (photometry-only).

    One XLA program compiles once; Adam runs in parallel over all galaxies.
    Typical speedup on catalogs is 10-50x vs sequential MAP fitting.
    Photometry-only: same model (redshift, filters, parameters) for every galaxy.

    Parameters
    ----------
    model : SEDModel
        Forward model with ``Observation(photometry=...)``.
    fluxes : array_like, shape (N, n_filters)
        Per-galaxy observed fluxes [erg/s/cm²/Hz].
    noises : array_like, shape (N, n_filters)
        Per-galaxy 1-sigma noise [erg/s/cm²/Hz].
    n_steps : int, optional
        Adam iterations per galaxy. Default 500.
    learning_rate : float, optional
        Adam learning rate for optimization. Default 0.02.
    seed : int, optional
        PRNG seed for Adam initialization. Default 0.
    verbose : bool, optional
        Print batch statistics and wall time. Default True.

    Returns
    -------
    dict[str, ndarray]
        Physical-space MAP point estimates. Each value has shape (N,)
        (or (N, n_grid) for stochastic xi). Keys match ``model.spec.free_params``.

    Notes
    -----
    **JIT-compatible**: no, uses Python-level vmap setup and optax integration.
    """
    import time

    try:
        import optax
    except ImportError as err:  # pragma: no cover - optional dep
        raise ImportError("fit_batch_map_vmap requires optax") from err

    from tengri.inference.fitter import Fitter
    from tengri.inference.loss_functions import build_loss_fn
    from tengri.utils.transforms import to_bounded

    fluxes = jnp.asarray(fluxes)
    noises = jnp.asarray(noises)
    if fluxes.shape != noises.shape:
        raise ValueError(f"fluxes.shape {fluxes.shape} != noises.shape {noises.shape}")
    if fluxes.ndim != 2:
        raise ValueError(f"fluxes must be 2-D (N, n_filters); got shape {fluxes.shape}")
    n_gal = int(fluxes.shape[0])

    # Build a template Fitter on the first galaxy to derive the loss fn.
    # Data baked into this fitter is ignored, loss_fn takes data_args explicitly.
    template = Fitter(model, fluxes[0], noises[0])
    if template.data_type != "photometry":
        raise ValueError(
            f"fit_batch_map_vmap is photometry-only; got data_type={template.data_type!r}"
        )

    loss_fn = build_loss_fn(template)
    # template._initialize_unbounded gives one galaxy's init dict; replicate it.
    init_unbounded = template._initialize_unbounded(jax.random.PRNGKey(seed))

    def _replicate(x):
        """Replicate scalar across batch dimension."""
        return jnp.broadcast_to(x, (n_gal, *jnp.shape(x)))

    params_batch = jax.tree.map(_replicate, init_unbounded)
    data_args_batch = {"data": fluxes, "noise": noises}

    optimizer = optax.adam(learning_rate)

    def _single_step(carry, _):
        """Execute one Adam iteration across batch via vmap."""
        params, opt_state = carry

        # vmap loss over leading axis of params and data_args
        def _loss_one(p, da):
            """Compute loss for one galaxy."""
            return loss_fn(p, da)

        losses, grads = jax.vmap(jax.value_and_grad(_loss_one))(params, data_args_batch)
        updates, opt_state = jax.vmap(optimizer.update)(grads, opt_state, params)
        params = jax.vmap(optax.apply_updates)(params, updates)
        return (params, opt_state), losses

    opt_state_single = optimizer.init(init_unbounded)
    opt_state_batch = jax.tree.map(_replicate, opt_state_single)

    t0 = time.time()
    (params_final, _), loss_trace = jax.lax.scan(
        _single_step, (params_batch, opt_state_batch), None, length=n_steps
    )
    loss_trace.block_until_ready()
    t_run = time.time() - t0

    if verbose:
        last_loss = jnp.asarray(loss_trace[-1])
        print(
            f"fit_batch_map_vmap: N={n_gal}, n_steps={n_steps}, "
            f"wall={t_run:.2f}s, mean final loss={float(last_loss.mean()):.3f}"
        )

    # Transform to physical space
    bounds = template._bounds
    fixed_values = template._fixed_values
    physical: dict[str, jnp.ndarray] = {}
    for name in template._free_names:
        lo, hi = bounds[name]
        physical[name] = to_bounded(params_final[name], lo, hi)
    if template.spec.stochastic and "psd_xi" in params_final:
        physical["psd_xi"] = params_final["psd_xi"]
    for name, val in fixed_values.items():
        physical[name] = jnp.broadcast_to(jnp.asarray(val), (n_gal,))
    return physical


def fit_batch(
    model: SEDModel,
    catalog,
    flux_cols: list[str],
    err_cols: list[str],
    redshift_col: str | None = None,
    method: str = "vi",
    n_workers: int = 1,
    verbose: bool = True,
    output_dir: str | None = None,
    id_col: str | None = None,
    approx="auto",
    **kwargs,
) -> list:
    """Fit a batch of galaxies from a catalog, one row at a time.

    .. deprecated:: 2026-07
        Use :class:`Catalog` directly for new code. ``fit_batch`` will be
        removed in a future release.

    Accepts pandas.DataFrame, astropy.table.Table, or list of dicts.
    Supports checkpoint resume via ``output_dir``.

    Parameters
    ----------
    model : SEDModel
        Forward model instance.
    catalog : DataFrame, Table, or list of dict
        Input catalog with flux and error columns.
    flux_cols : list of str
        Column names for per-band flux [erg/s/cm²/Hz].
    err_cols : list of str
        Column names for per-band 1-sigma uncertainty [erg/s/cm²/Hz].
    redshift_col : str, optional
        Column name for per-row redshift. If None, uses model redshift.
    method : str, optional
        Inference method (e.g. "vi", "mcmc"). Default ``"vi"``.
    n_workers : int, optional
        Reserved for future multiprocessing. Default 1.
    verbose : bool, optional
        Print per-galaxy progress. Default True.
    output_dir : str, optional
        Save each Posterior to ``{output_dir}/{id}.h5``. Supports checkpoint resume.
    id_col : str, optional
        Column name for galaxy IDs in output filenames. Default uses row index.
    **kwargs
        Forwarded to ``Fitter.run()`` for each galaxy.

    Returns
    -------
    list of Posterior
        One inference result (Posterior) per input catalog row.

    Notes
    -----
    **JIT-compatible**: no, uses Python-level loop and file I/O.
    """
    import os
    import time
    import warnings

    # One-shot deprecation warning
    warnings.warn(
        "fit_batch is deprecated; use Catalog instead. "
        "Catalog provides vectorized fitting and requires explicit flux_unit "
        "(fit_batch assumed cgs). See #1317, #1316.",
        DeprecationWarning,
        stacklevel=2,
    )

    from tengri.forward.sed_model import SEDModel as ModelClass
    from tengri.inference.fitter import Fitter
    from tengri.inference.posterior import Posterior
    from tengri.parameters.priors import Fixed

    # Normalize catalog to list of dicts
    rows: list[dict] = []
    try:
        import pandas as pd

        if isinstance(catalog, pd.DataFrame):
            rows = catalog.to_dict(orient="records")
    except ImportError:
        pass

    if not rows:
        try:
            from astropy.table import Table

            if isinstance(catalog, Table):
                rows = [dict(zip(catalog.colnames, row)) for row in catalog]
        except ImportError:
            pass

    if not rows and isinstance(catalog, (list, tuple)):
        rows = list(catalog)

    if not rows:
        raise TypeError(
            f"catalog must be a pandas DataFrame, astropy Table, or list of dicts. "
            f"Got {type(catalog)}"
        )

    # ── Consolidation (#1336): unless per-galaxy checkpointing is requested,
    # route the whole batch through Catalog, the one fitting code path. Catalog
    # owns ingestion, unit conversion, and per-galaxy redshift; we unwrap its
    # CatalogPosterior back to the legacy list-of-Posterior. The output_dir
    # checkpoint/resume path below keeps the per-row loop (Catalog has no
    # persistence layer), so every fit_batch feature is preserved.
    if output_dir is None:
        from tengri.forward.forward_model import ForwardModel
        from tengri.inference.catalog import Catalog

        fwd = ForwardModel.build(sed=model)
        # Forward an explicit approx= (the legacy loop passed it to every Fitter).
        # "auto" is the default and Catalog/Fitter already resolve it, so only a
        # non-default value needs applying, dropping it would silently change the
        # fit's approximation policy.
        if approx != "auto":
            fwd = fwd.with_approx(approx)
        band_names = list(fwd.observation.photometry.names)
        if len(flux_cols) != len(band_names) or len(err_cols) != len(band_names):
            raise ValueError(
                f"flux_cols/err_cols must have one entry per observation band "
                f"({len(band_names)}); got {len(flux_cols)} flux, {len(err_cols)} err."
            )
        # fit_batch's flux_cols/err_cols are POSITIONAL (catalog column names,
        # mapped to the model's bands in order), the catalog columns need not be
        # named after the bands. Rename them to the observation band names so
        # Catalog can name-match, preserving fit_batch's positional contract.
        table: dict = {}
        for band, fcol, ecol in zip(band_names, flux_cols, err_cols):
            table[band] = np.asarray([float(row[fcol]) for row in rows])
            table[f"{band}_err"] = np.asarray([float(row[ecol]) for row in rows])
        if redshift_col is not None:
            table["_fit_batch_z"] = np.asarray([float(row[redshift_col]) for row in rows])
        cat = Catalog(
            fwd,
            table,
            flux_unit="cgs_fnu",  # fit_batch always assumed cgs f_nu
            redshift_col="_fit_batch_z" if redshift_col is not None else None,
        )
        fit_key = kwargs.pop("key", None)
        if fit_key is None:
            fit_key = jax.random.PRNGKey(0)
        # Forward verbose so fit_batch(verbose=False) still silences progress
        # (the engine's per-galaxy "N galaxies done" line honors it).
        post = cat.fit(method=method, key=fit_key, verbose=verbose, **kwargs)
        return list(post.posteriors)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    n_gal = len(rows)
    results: list = []
    n_skipped = 0
    t0 = time.time()

    # Warn loudly once if redshift_col is set but catalog_z_range is not
    if redshift_col is not None and getattr(model, "_catalog_z_range", None) is None:
        warnings.warn(
            "fit_batch(redshift_col=...) without WavePrecomp(catalog_z_range=...) "
            "recompiles the forward model for EVERY row (one compile per galaxy). "
            "Build the model with approx=WavePrecomp(catalog_z_range=(zmin, zmax)) "
            "to compile once. See #1316.",
            UserWarning,
            stacklevel=2,
        )

    for i, row in enumerate(rows):
        gal_id = str(row[id_col]) if id_col is not None else str(i)

        if output_dir is not None:
            result_path = os.path.join(output_dir, f"{gal_id}.h5")
            if os.path.exists(result_path):
                result_i = Posterior.load(result_path, model=model)
                results.append(result_i)
                n_skipped += 1
                if verbose and n_skipped <= 3:
                    print(f"  [{i + 1}/{n_gal}] {gal_id}, loaded from checkpoint")
                elif verbose and n_skipped == 4:
                    print("  ... skipping remaining cached results")
                continue

        t_row = time.time()

        flux_i = jnp.array([float(row[c]) for c in flux_cols])
        noise_i = jnp.array([float(row[c]) for c in err_cols])

        if redshift_col is not None:
            row_z = float(row[redshift_col])
            row_spec = model.spec.with_params(redshift=Fixed(row_z))
            row_model = ModelClass.__new__(ModelClass)
            row_model.__dict__.update(model.__dict__)
            row_model.spec = row_spec
            fitter_i = Fitter(row_model, flux_i, noise_i, approx=approx)
        else:
            fitter_i = Fitter(model, flux_i, noise_i, approx=approx)

        result_i = fitter_i.run(method, **kwargs)

        if output_dir is not None:
            result_path = os.path.join(output_dir, f"{gal_id}.h5")
            result_i.save(result_path)

        results.append(result_i)

        if verbose:
            dt = time.time() - t_row
            elapsed = time.time() - t0
            chi2 = result_i.diagnostics.get("chi2_dof", "?")
            chi2_str = f"{chi2:.2f}" if isinstance(chi2, float) else str(chi2)
            print(f"  [{i + 1}/{n_gal}] chi2/dof={chi2_str}, row={dt:.1f}s, total={elapsed:.0f}s")

    if verbose and n_skipped > 0:
        print(f"  Checkpoint: {n_skipped}/{n_gal} loaded from {output_dir}")

    return results


# ── Catalog summary ───────────────────────────────────────────────


def catalog_summary(
    results: list,
    percentiles: tuple[float, ...] = (16.0, 50.0, 84.0),
    include_derived: bool = True,
) -> dict[str, np.ndarray]:
    """Aggregate posteriors into a summary catalog with percentile columns.

    For each parameter (and optionally derived quantities),
    computes percentiles across posterior samples.
    MAP results contribute a single value repeated.

    Parameters
    ----------
    results : list of Posterior
        One per galaxy, from ``fit_batch()``.
    percentiles : tuple of float, optional
        Percentiles to compute. Default (16, 50, 84) gives median + 68% CI.
    include_derived : bool, optional
        Include derived quantities (stellar_mass, sfr_100myr, etc.). Default True.

    Returns
    -------
    dict[str, ndarray]
        Keys are ``"{param}_p{pct}"`` (e.g. ``"dust_av_p50"``).
        Each value is 1-D array of length ``len(results)`` [physical units].
        Also includes ``"chi2_dof"`` if available [dimensionless].

    Notes
    -----
    **JIT-compatible**: no, uses Python loops for aggregation.

    Examples
    --------
    .. code-block:: python

        from tengri import catalog_summary

        # results is a list of Posterior objects, one per galaxy
        catalog = catalog_summary(results, percentiles=(16, 50, 84))
        # catalog["dust_av_p50"]  → shape (n_galaxies,)
        # catalog["sfr_100myr_p16"]  → lower 68% CI bound
        import numpy as np
        import astropy.table

        t = astropy.table.Table(catalog)
        t.write("catalog.fits", overwrite=True)
    """
    if not results:
        return {}

    n_gal = len(results)
    pct_arr = np.array(percentiles)

    # Collect parameter names from first result with samples
    param_names: list[str] = []
    for r in results:
        source = r.samples if r.samples is not None else r.params
        for k in sorted(source.keys()):
            if k == "psd_xi":
                continue
            arr = source[k]
            if np.asarray(arr).ndim <= 1:
                param_names.append(k)
        break

    out: dict[str, np.ndarray] = {}
    for name in param_names:
        for pct in percentiles:
            col = f"{name}_p{int(pct)}"
            out[col] = np.full(n_gal, np.nan)

    for i, r in enumerate(results):
        if r.samples is not None:
            for name in param_names:
                if name not in r.samples:
                    continue
                arr = np.asarray(r.samples[name])
                if arr.ndim != 1:
                    continue
                pvals = np.percentile(arr, pct_arr)
                for pct, pv in zip(percentiles, pvals):
                    out[f"{name}_p{int(pct)}"][i] = pv
        else:
            for name in param_names:
                if name not in r.params:
                    continue
                val = float(np.mean(np.asarray(r.params[name])))
                for pct in percentiles:
                    out[f"{name}_p{int(pct)}"][i] = val

    if include_derived:
        derived_keys: list[str] = []
        for r in results:
            if r._model is None:
                break
            try:
                d = r.derived
                derived_keys = [k for k in sorted(d.keys()) if np.asarray(d[k]).ndim <= 1]
                break
            except (RuntimeError, AttributeError):
                break

        if derived_keys:
            for dk in derived_keys:
                for pct in percentiles:
                    out[f"{dk}_p{int(pct)}"] = np.full(n_gal, np.nan)

            for i, r in enumerate(results):
                if r._model is None:
                    continue
                try:
                    d = r.derived
                except (RuntimeError, AttributeError):
                    continue
                for dk in derived_keys:
                    if dk not in d:
                        continue
                    arr = np.asarray(d[dk])
                    if arr.ndim == 0:
                        val = float(arr)
                        for pct in percentiles:
                            out[f"{dk}_p{int(pct)}"][i] = val
                    elif arr.ndim == 1:
                        pvals = np.percentile(arr, pct_arr)
                        for pct, pv in zip(percentiles, pvals):
                            out[f"{dk}_p{int(pct)}"][i] = pv

    chi2_dof = np.full(n_gal, np.nan)
    for i, r in enumerate(results):
        val = r.diagnostics.get("chi2_dof")
        if isinstance(val, (int, float)):
            chi2_dof[i] = float(val)
    out["chi2_dof"] = chi2_dof

    return out


# ── Population fitting ────────────────────────────────────────────


def fit_population(
    model: SEDModel,
    observations_list: list,
    method: str = DEFAULT_METHOD,
    population_prior: dict | None = None,
    **kwargs,
):
    """Fit a population of galaxies with shared PSD hyperparameters.

    Thin wrapper around ``PopulationFitter`` for hierarchical inference.

    Parameters
    ----------
    model : SEDModel
        Forward model instance.
    observations_list : list
        Each element is either (flux, noise) tuple or dict with
        ``"flux_obs"`` [erg/s/cm²/Hz] and ``"noise"`` [erg/s/cm²/Hz] keys.
    method : str, optional
        Hierarchical inference method (e.g. "vi", "mcmc"). Default ``"vi"``.
    population_prior : dict, optional
        Hyperpriors on shared PSD parameters (e.g. ``psd_sigma``, ``psd_tau_myr``).
    **kwargs
        Forwarded to ``PopulationFitter.run()``.

    Returns
    -------
    PopulationPosterior
        Population-level posterior with shared PSD hyperparameters
        and per-galaxy individual parameters.

    Notes
    -----
    **JIT-compatible**: no, uses PopulationFitter wrapper with internal state.
    """
    from tengri.forward.sed_model import SEDModel as ModelClass
    from tengri.inference.hierarchical import PopulationFitter
    from tengri.parameters.priors import Fixed

    # Normalize input
    galaxies = []
    for obs in observations_list:
        if isinstance(obs, (list, tuple)) and len(obs) == 2:
            flux, noise = obs
            galaxies.append({"flux_obs": flux, "noise": noise})
        elif isinstance(obs, dict):
            galaxies.append(obs)
        else:
            raise TypeError(
                f"Each element of observations_list must be a (flux, noise) tuple "
                f"or a dict with 'flux_obs'/'noise' keys. Got {type(obs)}"
            )

    # Extract population prior bounds
    psd_sigma_prior = (0.1, 4.0)
    psd_tau_prior = (1.0, 300.0)
    if population_prior:
        if "psd_sigma" in population_prior:
            dist = population_prior["psd_sigma"]
            psd_sigma_prior = getattr(dist, "bounds", psd_sigma_prior)
        if "psd_tau_myr" in population_prior:
            dist = population_prior["psd_tau_myr"]
            psd_tau_prior = getattr(dist, "bounds", psd_tau_prior)

    # Translate canonical → PopulationFitter method names
    _hier_method_map = {
        "vi": "geovi",
        "vi_linear": "mgvi",
        "mcmc_raytrace": "raytrace",
        "mcmc": "raytrace",
    }
    hier_method = _hier_method_map.get(method, method)

    def _model_factory(psd_sigma, psd_tau_myr):
        """Build model with fixed PSD hyperparameters."""
        new_spec = model.spec.with_params(
            sfh_field_psd_sigma=Fixed(float(psd_sigma)),
            sfh_field_psd_tau_myr=Fixed(float(psd_tau_myr)),
        )
        m = ModelClass.__new__(ModelClass)
        m.__dict__.update(model.__dict__)
        m.spec = new_spec
        return m

    hfitter = PopulationFitter(
        _model_factory,
        galaxies,
        psd_sigma_prior=psd_sigma_prior,
        psd_tau_prior=psd_tau_prior,
    )
    return hfitter.run(hier_method, **kwargs)


# ── from_config factory ───────────────────────────────────────────


def build_model_from_config(
    model_cls,
    ssp,
    sfh=_UNSET,
    dust=_UNSET,
    nebular=_UNSET,
    agn=_UNSET,
    redshift=_UNSET,
    filters: list[str] | None = None,
    wave_obs=None,
    priors: dict | None = None,
    **model_kwargs,
):
    """Build SEDModel from configuration specification.

    Resolves SFH, dust, nebular, and AGN settings from config defaults,
    builds Parameters and Observation, and instantiates the model.

    Parameters
    ----------
    model_cls : type
        SEDModel class.
    ssp : str or SSPData
        Path to SSP data file or SSPData object.
    sfh : str, optional
        SFH model type (e.g. "dpl", "tsnorm"). Uses default if unset.
    dust : str, optional
        Dust law (e.g. "charlot_fall", "kl04"). Uses default if unset.
    nebular : str or None, optional
        Nebular backend (e.g. "cloudy_grid", "linratios"). Uses default if unset.
    agn : str or None, optional
        AGN model type. Uses default if unset.
    redshift : float or str, optional
        Redshift value (e.g. 0.5) [dimensionless] or "free" to enable free parameter.
    filters : list of str, optional
        Filter names for photometry. If None, no photometry observation.
    wave_obs : array_like, optional
        Wavelength grid for spectroscopy [Angstrom]. If None, no spectroscopy.
    priors : dict, optional
        Prior overrides for parameters.
    **model_kwargs
        Additional keyword arguments for model constructor.

    Returns
    -------
    SEDModel
        Configured forward model instance with specified physics modules.

    Notes
    -----
    **JIT-compatible**: no, uses Python-level model construction.
    """
    from tengri.parameters.defaults import get_from_config_defaults
    from tengri.parameters.translate import resolve_short_names

    # Resolve each argument: use caller value if supplied, else read from TOML.
    _defs = get_from_config_defaults()
    sfh = _defs["sfh"] if sfh is _UNSET else sfh
    dust = _defs["dust"] if dust is _UNSET else dust
    nebular = _defs["nebular"] if nebular is _UNSET else nebular
    agn = _defs["agn"] if agn is _UNSET else agn
    redshift = _defs["redshift"] if redshift is _UNSET else redshift
    from tengri.components.stellar.sps.dsps_wrapper import SSPData, load_ssp_data
    from tengri.observation.observation import Observation
    from tengri.parameters.parameters import Parameters
    from tengri.parameters.priors import Uniform

    # --- Load SSP data ---
    if isinstance(ssp, str):
        ssp_data = load_ssp_data(ssp)
    elif isinstance(ssp, SSPData):
        ssp_data = ssp
    else:
        raise TypeError(f"ssp must be a file path (str) or SSPData, got {type(ssp)}")

    # --- Expand short names in priors ---
    expanded = resolve_short_names(sfh, priors or {})

    # --- Inject redshift ---
    if redshift == "free":
        if "redshift" not in expanded:
            expanded["redshift"] = Uniform(0.001, 6.0)
    else:
        expanded.setdefault("redshift", float(redshift))

    # --- Inject AGN parametric mode if AGN enabled ---
    # Default to agn_log_lbol (parametric) instead of agn_lum_ratio.
    # Parametric mode is compatible with all kernel paths (hybrid,
    # compositional) because L_bol is specified directly, avoiding
    # the circular dependency L_AGN = f × (L_stellar + L_AGN).
    if agn is not None and "agn_lum_ratio" not in expanded and "agn_log_lbol" not in expanded:
        expanded["agn_log_lbol"] = Uniform(8.0, 12.0)

    # --- Build Parameters ---
    sfh_tokens = [t.strip() for t in sfh.replace("+", " ").split()]

    spec_kwargs: dict = dict(expanded)
    spec_kwargs["mean_sfh_type"] = sfh_tokens

    if dust != "charlot_fall":
        spec_kwargs["dust_law_bc"] = dust

    if nebular is not None:
        spec_kwargs["nebular"] = nebular

    # Pass component kwargs to Parameters (not to Model.__init__)
    # These control the param registry and forward model configuration
    _component_kwargs = [
        "cloudy_grid_path",
        "dust_emission",
        "dl07_grid_path",
        "radio",
        "xray",
        "igm_patchy",
        "shock",
    ]
    for kwarg in _component_kwargs:
        if kwarg in model_kwargs:
            spec_kwargs[kwarg] = model_kwargs.pop(kwarg)

    if agn is not None:
        spec_kwargs["agn_model"] = agn

    spec = Parameters(**spec_kwargs)

    # --- Build Observation ---
    obs_photometry = None
    obs_spectroscopy = None

    if filters is not None:
        try:
            from tengri.observation.photometry_config import Photometry

            obs_photometry = Photometry.from_names(filters)
        except (ImportError, AttributeError):
            pass

    if wave_obs is not None:
        try:
            from tengri.observation.spectroscopy import Spectroscopy

            obs_spectroscopy = Spectroscopy(wave_obs=wave_obs)
        except (ImportError, AttributeError):
            pass

    if obs_photometry is not None or obs_spectroscopy is not None:
        observation = Observation(
            photometry=obs_photometry,
            spectroscopy=obs_spectroscopy,
        )
    else:
        observation = None

    return model_cls(spec, ssp_data, observation=observation, **model_kwargs)


# ── fit convenience wrapper ───────────────────────────────────────


def fit_model(
    model,
    data=None,
    noise=None,
    method=_UNSET,
    data_type: str | None = None,
    *,
    photometry: tuple | None = None,
    spectrum: tuple | None = None,
    init: str | None = None,
    approx="auto",
    **kwargs,
):
    """Fit observed data using specified inference method.

    Convenience wrapper for single-galaxy inference via Fitter.

    Parameters
    ----------
    model : SEDModel
        Forward model instance.
    data : array_like, optional
        Observed flux [erg/s/cm²/Hz]. Required unless photometry or spectrum given.
    noise : array_like, optional
        1-sigma uncertainty [erg/s/cm²/Hz]. Required unless photometry or spectrum given.
    method : str, optional
        Inference method (e.g. "vi", "mcmc"). Uses default if unset.
    data_type : str, optional
        Data type ("photometry", "spectroscopy", "joint"). Auto-detected if None.
    photometry : tuple, optional
        Shorthand for (flux, noise) tuple from photometry [erg/s/cm²/Hz].
    spectrum : tuple, optional
        Shorthand for (flux, noise) tuple from spectroscopy [erg/s/cm²/Hz].
    init : str, optional
        Initialization method ("map" runs MAP first for warm start).
    **kwargs
        Forwarded to ``Fitter.run()``.

    Returns
    -------
    Posterior
        Fitted posterior with posterior samples or MAP point estimate.

    Notes
    -----
    **JIT-compatible**: no, uses Fitter with inference internals.
    """
    from tengri.inference.fitter import Fitter
    from tengri.parameters.defaults import get_inference_defaults

    if method is None:
        from tengri.config.exceptions import ParameterError

        raise ParameterError(
            "method=None is not allowed. Pass an explicit method string "
            "(e.g. 'vi_nifty', 'mcmc_nuts', 'auto') or omit the argument to use "
            "the default from defaults.toml."
        )
    if method is _UNSET:
        method = get_inference_defaults().get("method", "vi")

    # --- A Data record goes to the canonical surface, not through here (#1366) ---
    #
    # ``SEDModel.fit`` is documented as sugar for
    # ``ForwardModel.build(sed=self).fit(...)``, but only ``ForwardModel.fit``
    # knew how to unpack a ``Data``. A record therefore fell through to the
    # positional-argument check below and the user got a message about
    # ``(flux, noise)`` tuples that never mentioned ``Data`` -- implying the call
    # shape was wrong rather than that this surface did not support the type.
    # Everything ``Data`` carries (censoring, line fluxes, joint
    # photometry+spectrum) was unreachable from the one-liner we teach.
    #
    # Forwarding rather than re-implementing is deliberate: ``Data`` unpacking is
    # ~70 lines that ends by rebuilding the Observation for line fluxes, and a
    # second copy would drift. One record type, one validation seam
    # (``Data.validate_against``), reached from both verbs.
    from tengri.observation.data import Data as _Data

    if isinstance(data, _Data):
        from tengri.forward.forward_model import ForwardModel

        if photometry is not None or spectrum is not None:
            raise TypeError(
                "fit(Data, photometry=... / spectrum=...) is ambiguous: the Data "
                "record already carries every channel. Pass the record alone."
            )
        if data_type is not None:
            # ForwardModel.fit has no data_type parameter, so forwarding it would
            # drop it silently. It is redundant anyway -- which channels exist is
            # already determined by which fields the record carries.
            raise TypeError(
                "fit(Data, data_type=...) is redundant: the Data record already "
                "declares its channels (photometry=, spectrum=, lines=). Drop "
                "data_type."
            )
        forward = ForwardModel.build(sed=model)
        return forward.fit(data, noise, method=method, approx=approx, **kwargs)

    # --- Resolve data arrays ---
    if photometry is not None or spectrum is not None:
        if photometry is not None and spectrum is not None:
            flux_p, noise_p = photometry
            flux_s, noise_s = spectrum
            data = jnp.concatenate([jnp.asarray(flux_p), jnp.asarray(flux_s)])
            noise = jnp.concatenate([jnp.asarray(noise_p), jnp.asarray(noise_s)])
            data_type = data_type or "joint"
        elif photometry is not None:
            data, noise = photometry
            data_type = data_type or "photometry"
        else:
            data, noise = spectrum
            data_type = data_type or "spectroscopy"
    else:
        if data is None or noise is None:
            raise ValueError(
                "Provide either positional (data, noise) or keyword "
                "photometry=(flux, noise) / spectrum=(flux, noise)."
            )

    # --- Infer data_type if still None ---
    if data_type is None:
        obs = getattr(model, "observation", None)
        if obs is not None:
            data_type = obs.data_type
        else:
            data_type = "photometry"

    # --- Build fitter ---
    # When SEDModel.fit() delegates to fit_model(), silence the Fitter(sed_model)
    # deprecation warning since SEDModel.fit is now un-deprecated sugar (#1322).
    import warnings

    from tengri.inference.fitter import split_fitter_kwargs

    # ``params`` is the per-fit Fixed-value override (#1329); constructor-owned
    # kwargs (calibration_marginalize, likelihood, ...) go to Fitter(...) and
    # the rest to run(), spec §7's fit-time flags (#1378).
    params_override = kwargs.pop("params", None)
    ctor_kwargs, kwargs = split_fitter_kwargs(kwargs)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Fitter\\(sed_model.*",
            category=DeprecationWarning,
        )
        fitter = Fitter(
            model,
            data,
            noise,
            data_type=data_type,
            approx=approx,
            params_override=params_override,
            **ctor_kwargs,
        )
    model.fitter_ = fitter

    # --- Optional MAP warm start ---
    init_from = None
    if init == "map":
        init_from = fitter.run("map")

    return fitter.run(method, init_from=init_from, **kwargs)
