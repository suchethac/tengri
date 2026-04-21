"""Convenience methods delegated from SEDModel.

Extracted from core/model.py to keep model.py focused on the forward model.
Each function takes (model, ...) where model is an SEDModel instance.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from tengri.forward.sed_model import MockData, PriorPredictive, SEDModel


# ── Mock data generation ──────────────────────────────────────────


def mock(model: SEDModel, params, snr=20.0, key=None) -> MockData:
    """Generate mock photometric observation."""
    from tengri.forward.sed_model import MockData

    flux_true = model.predict_photometry(params)
    noise = flux_true / snr

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


def mock_spectrum(model: SEDModel, params, wave_obs, snr=30.0, key=None) -> MockData:
    """Generate mock spectroscopic observation.

    Parameters
    ----------
    params : dict
        Parameter values.
    wave_obs : array
        Observed wavelength grid (Angstrom).
    snr : float
        Signal-to-noise ratio per pixel.
    key : PRNGKey, optional
        Random key for noise. If None, returns noiseless.

    Returns
    -------
    MockData
        Mock spectroscopic observation.
    """
    from tengri.forward.sed_model import MockData

    flux_true = model.predict_spectrum(params, wave_obs)
    noise = jnp.abs(flux_true) / snr

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


def mock_batch(model: SEDModel, params_batch, snr=20.0, key=None) -> MockData:
    """Generate batch of mock observations."""
    from tengri.forward.sed_model import MockData

    first_key = next(iter(params_batch))
    n_batch = params_batch[first_key].shape[0]

    def _get_single(i):
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


def predict_photometry_batch(model: SEDModel, params_batch):
    """Compute photometry for a batch of galaxies via jax.vmap.

    Parameters
    ----------
    params_batch : dict of arrays
        Each value has a leading batch dimension: shape (N, ...).
        E.g. ``{"sfh_dpl_alpha": array([1.0, 1.5, 2.0]), ...}``

    Returns
    -------
    array, shape (N, n_filters)
        Photometric flux for each galaxy.
    """
    return jax.vmap(model.predict_photometry)(params_batch)


def predict_spectrum_batch(model: SEDModel, params_batch):
    """Compute spectra for a batch of galaxies via jax.vmap.

    Requires ``precompute_spectroscopy()`` to have been called.

    Parameters
    ----------
    params_batch : dict of arrays
        Each value has leading batch dimension.

    Returns
    -------
    array, shape (N, n_pix)
        Spectral flux for each galaxy.
    """
    return jax.vmap(model.predict_spectrum)(params_batch)


# ── Prior predictive check ────────────────────────────────────────


def prior_predictive(model: SEDModel, n: int = 500, seed: int = 42) -> PriorPredictive:
    """Sample from the prior and evaluate the forward model on each draw.

    Returns a ``PriorPredictive`` object with draw arrays and convenience
    methods for model checking before inference.

    Parameters
    ----------
    n : int
        Number of prior draws. Default 500.
    seed : int
        Random seed. Default 42.

    Returns
    -------
    PriorPredictive
        ``ppc.flux`` — shape (n, n_filters) or None.
        ``ppc.sfh``  — shape (n, n_grid).
        ``ppc.params`` — dict of (n,) arrays.

    Examples
    --------
    >>> ppc = model.prior_predictive(n=500)
    >>> ppc.check_finite()
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
    """Batched MAP (photometry) inference via ``jax.vmap`` across galaxies.

    One XLA program compiles once; adam runs in parallel over all galaxies.
    Typical speedup on catalogs is **10-50x** vs ``fit_batch(method="map", ...)``,
    which loops one galaxy at a time.

    **Scope.** Photometry-only. Same model (same fixed redshift, same filters,
    same free-parameter spec) for every galaxy. For per-galaxy redshift or
    heterogeneous priors, fall back to ``fit_batch``. VI catalog batching
    (geoVI/MGVI) is NOT supported here — the NIFTy and native VI paths carry
    internal state (CG history, KL caches) that do not vmap cleanly.

    Parameters
    ----------
    model : SEDModel
        Forward model. Must have been built with ``Observation(photometry=...)``.
    fluxes : array, shape (N, n_filters)
        Per-galaxy observed fluxes.
    noises : array, shape (N, n_filters)
        Per-galaxy 1-sigma noise.
    n_steps : int
        Adam iterations per galaxy (constant across the batch).
    learning_rate : float
        Adam learning rate. The default matches ``Fitter._run_map``.
    seed : int
        PRNG seed for adam init randomness.
    verbose : bool
        Print batch size and wall time.

    Returns
    -------
    dict[str, jax.Array]
        Physical-space MAP point estimates. Each value has shape ``(N,)``
        (or ``(N, n_grid)`` for stochastic xi). Keys match
        ``model.spec.free_params`` plus any Fixed values.

    Notes
    -----
    Memory: roughly ``N × (model RAM per galaxy)``. The N=64 → 28 GB figure
    some analyses cite is pessimistic — precompute tables are shared, not
    multiplied. For 1000+ catalogs, chunk with ``jax.tree_map`` over slices.
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
    # Data baked into this fitter is ignored — loss_fn takes data_args explicitly.
    template = Fitter(model, fluxes[0], noises[0])
    if template.data_type != "photometry":
        raise ValueError(
            f"fit_batch_map_vmap is photometry-only; got data_type={template.data_type!r}"
        )

    loss_fn = build_loss_fn(template)
    # template._initialize_unbounded gives one galaxy's init dict; replicate it.
    init_unbounded = template._initialize_unbounded(jax.random.PRNGKey(seed))

    def _replicate(x):
        return jnp.broadcast_to(x, (n_gal, *jnp.shape(x)))

    params_batch = jax.tree.map(_replicate, init_unbounded)
    data_args_batch = {"data": fluxes, "noise": noises}

    optimizer = optax.adam(learning_rate)

    def _single_step(carry, _):
        params, opt_state = carry

        # vmap loss over leading axis of params and data_args
        def _loss_one(p, da):
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
    **kwargs,
) -> list:
    """Fit a batch of galaxies, one row at a time.

    Accepts a ``pandas.DataFrame``, an ``astropy.table.Table``, or a
    list of dicts. Each row becomes one ``Posterior``.

    Parameters
    ----------
    catalog : DataFrame, Table, or list of dict
        Input catalog.
    flux_cols : list of str
        Column names for per-band flux values (must match model's filter order).
    err_cols : list of str
        Column names for per-band 1-sigma uncertainties.
    redshift_col : str or None
        If provided, use this column as per-row redshift via ``spec.with_params()``.
    method : str
        Inference method. Default ``"vi"``.
    n_workers : int
        Currently ignored (reserved for future multiprocessing). Default 1.
    verbose : bool
        Print per-galaxy progress. Default True.
    output_dir : str or None
        If provided, save each ``Posterior`` to ``{output_dir}/{id}.h5``
        after fitting. On re-run, galaxies with existing result files are
        skipped (checkpoint resume). The directory is created if needed.
    id_col : str or None
        Column name for galaxy identifiers used in checkpoint filenames.
        If None, uses the row index (``0.h5``, ``1.h5``, ...).
    **kwargs
        Forwarded to ``Fitter.run()`` for every galaxy.

    Returns
    -------
    list of Posterior
        Same length as input catalog.

    Examples
    --------
    >>> results = model.fit_batch(
    ...     catalog_df,
    ...     flux_cols=["flux_u", "flux_g", "flux_r"],
    ...     err_cols=["err_u", "err_g", "err_r"],
    ...     redshift_col="z_spec",
    ...     output_dir="results/sdss_run1",
    ...     id_col="objID",
    ... )
    """
    import os
    import time

    from tengri.forward.sed_model import Model as ModelClass
    from tengri.inference.fitter import Fitter
    from tengri.inference.posterior import Posterior
    from tengri.parameters.priors import Fixed

    # Normalise catalog to list of dicts
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

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    n_gal = len(rows)
    results: list = []
    n_skipped = 0
    t0 = time.time()

    for i, row in enumerate(rows):
        gal_id = str(row[id_col]) if id_col is not None else str(i)

        if output_dir is not None:
            result_path = os.path.join(output_dir, f"{gal_id}.h5")
            if os.path.exists(result_path):
                result_i = Posterior.load(result_path, model=model)
                results.append(result_i)
                n_skipped += 1
                if verbose and n_skipped <= 3:
                    print(f"  [{i + 1}/{n_gal}] {gal_id} — loaded from checkpoint")
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
            fitter_i = Fitter(row_model, flux_i, noise_i)
        else:
            fitter_i = Fitter(model, flux_i, noise_i)

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


def fit_catalog(
    model: SEDModel,
    catalog,
    flux_cols: list[str],
    err_cols: list[str],
    redshift_col: str | None = None,
    method: str = "vi",
    n_workers: int = 1,
    verbose: bool = True,
    **kwargs,
) -> list:
    """Deprecated alias for fit_batch.

    .. deprecated:: 0.5.0
        Use :func:`fit_batch` instead.
    """
    warnings.warn(
        "fit_catalog is deprecated. Use fit_batch instead. Will be removed in tengri v1.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fit_batch(
        model,
        catalog,
        flux_cols,
        err_cols,
        redshift_col=redshift_col,
        method=method,
        n_workers=n_workers,
        verbose=verbose,
        **kwargs,
    )


# ── Catalog summary ───────────────────────────────────────────────


def catalog_summary(
    results: list,
    percentiles: tuple[float, ...] = (16.0, 50.0, 84.0),
    include_derived: bool = True,
) -> dict[str, np.ndarray]:
    """Aggregate a list of Posteriors into a summary catalog.

    For each free parameter (and optionally derived quantities),
    computes percentiles across each galaxy's posterior samples.
    MAP results contribute a single value repeated across all
    percentile columns.

    Parameters
    ----------
    results : list of Posterior
        One per galaxy, from ``fit_batch`` or ``Fitter.fit_batch``.
    percentiles : tuple of float
        Percentiles to compute. Default ``(16, 50, 84)`` gives
        median with 68% credible interval.
    include_derived : bool
        If True and model reference is available, include derived
        quantities (stellar_mass, sfr_100myr, etc.).

    Returns
    -------
    dict[str, np.ndarray]
        Keys are ``"{param}_p{pct}"`` (e.g. ``"dust_av_p50"``).
        Each value is a 1-D array of length ``len(results)``.
        Also includes ``"chi2_dof"`` if available in diagnostics.

    Examples
    --------
    >>> results = fit_batch(model, catalog, flux_cols, err_cols)
    >>> summary = catalog_summary(results)
    >>> summary["dust_av_p50"]  # median dust Av for each galaxy
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
    method: str = "vi",
    population_prior: dict | None = None,
    **kwargs,
):
    """Fit a population of galaxies with shared PSD hyperparameters.

    Thin wrapper around ``PopulationFitter``.

    Parameters
    ----------
    observations_list : list
        Each element is either a ``(flux, noise)`` tuple or a dict
        with ``"flux_obs"`` and ``"noise"`` keys.
    method : str
        Hierarchical inference method. Default ``"vi"``.
    population_prior : dict or None
        Hyperpriors on shared PSD parameters.
    **kwargs
        Forwarded to ``PopulationFitter.run()``.

    Returns
    -------
    HierarchicalResult
    """
    from tengri.forward.sed_model import Model as ModelClass
    from tengri.inference.hierarchical import PopulationFitter
    from tengri.parameters.priors import Fixed

    # Normalise input
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

from tengri.parameters.defaults import UNSET as _UNSET  # re-export for back-compat


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
    """Build a SEDModel from a grouped configuration.  See ``SEDModel.from_config``."""
    from tengri.parameters.defaults import get_from_config_defaults
    from tengri.parameters.translate import resolve_short_names

    # Resolve each argument: use caller value if supplied, else read from TOML.
    _defs = get_from_config_defaults()
    sfh = _defs["sfh"] if sfh is _UNSET else sfh
    dust = _defs["dust"] if dust is _UNSET else dust
    nebular = _defs["nebular"] if nebular is _UNSET else nebular
    agn = _defs["agn"] if agn is _UNSET else agn
    redshift = _defs["redshift"] if redshift is _UNSET else redshift
    from tengri.components.sps.dsps_wrapper import SSPData, load_ssp_data
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
    # Default to agn_log_lbol (parametric) instead of agn_frac (legacy).
    # Parametric mode is compatible with all kernel paths (hybrid,
    # compositional) because L_bol is specified directly, avoiding
    # the circular dependency L_AGN = f × (L_stellar + L_AGN).
    if agn is not None and "agn_frac" not in expanded and "agn_log_lbol" not in expanded:
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
    **kwargs,
):
    """Fit observed data.  See ``SEDModel.fit``."""
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
    fitter = Fitter(model, data, noise, data_type=data_type)
    model.fitter_ = fitter

    # --- Optional MAP warm start ---
    init_from = None
    if init == "map":
        init_from = fitter.run("map")

    return fitter.run(method, init_from=init_from, **kwargs)
