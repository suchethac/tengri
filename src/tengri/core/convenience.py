"""Convenience methods delegated from Model.

Extracted from core/model.py to keep model.py focused on the forward model.
Each function takes (model, ...) where model is a Model instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from tengri.core.model import MockData, Model, PriorPredictive


# ---------------------------------------------------------------------------
# Mock data generation
# ---------------------------------------------------------------------------


def mock(model: Model, params, snr=20.0, key=None) -> MockData:
    """Generate mock photometric observation."""
    from tengri.core.model import MockData

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


def mock_spectrum(model: Model, params, wave_obs, snr=30.0, key=None) -> MockData:
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
    from tengri.core.model import MockData

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


def mock_batch(model: Model, params_batch, snr=20.0, key=None) -> MockData:
    """Generate batch of mock observations."""
    from tengri.core.model import MockData

    first_key = next(iter(params_batch))
    n_batch = params_batch[first_key].shape[0]

    def _get_single(i):
        return {k: v[i] for k, v in params_batch.items()}

    if key is not None:
        noise_keys = jax.random.split(key, n_batch)
    else:
        noise_keys = [None] * n_batch

    results = [mock(_get_single(i), snr=snr, key=noise_keys[i]) for i in range(n_batch)]

    return MockData(
        flux_true=jnp.stack([r.flux_true for r in results]),
        flux_obs=jnp.stack([r.flux_obs for r in results]),
        noise=jnp.stack([r.noise for r in results]),
        params=params_batch,
    )


# ---------------------------------------------------------------------------
# Batch predictions (vmap over galaxies)
# ---------------------------------------------------------------------------


def predict_photometry_batch(model: Model, params_batch):
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


def predict_spectrum_batch(model: Model, params_batch):
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


# ---------------------------------------------------------------------------
# Prior predictive check
# ---------------------------------------------------------------------------


def prior_predictive(model: Model, n: int = 500, seed: int = 42) -> PriorPredictive:
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
    from tengri.core.model import PriorPredictive

    key = jax.random.PRNGKey(seed)
    params_batch = model.spec.sample_batch(key, n)

    # SFH draws
    sfh_batch = jax.vmap(model.predict_sfh)(params_batch)

    # Photometry draws (if filters present)
    flux_batch = None
    if model.filter_waves is not None:
        try:
            flux_batch = jax.vmap(model.predict_photometry)(params_batch)
        except Exception:
            flux_batch = None

    return PriorPredictive(
        flux=flux_batch,
        sfh=sfh_batch,
        params=params_batch,
        _model=model,
    )


# ---------------------------------------------------------------------------
# Catalog fitting
# ---------------------------------------------------------------------------


def fit_catalog(
    model: Model,
    catalog,
    flux_cols: list[str],
    err_cols: list[str],
    redshift_col: str | None = None,
    method: str = "vi",
    n_workers: int = 1,
    verbose: bool = True,
    **kwargs,
) -> list:
    """Fit a catalog of galaxies, one row at a time.

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
    **kwargs
        Forwarded to ``Fitter.run()`` for every galaxy.

    Returns
    -------
    list of Posterior
        Same length as input catalog.

    Examples
    --------
    >>> results = model.fit_catalog(
    ...     catalog_df,
    ...     flux_cols=["flux_u", "flux_g", "flux_r"],
    ...     err_cols=["err_u", "err_g", "err_r"],
    ...     redshift_col="z_spec",
    ... )
    """
    import time

    from tengri.core.model import Model as ModelClass
    from tengri.distributions import Fixed
    from tengri.inference.fitter import Fitter

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

    n_gal = len(rows)
    results: list = []
    t0 = time.time()

    for i, row in enumerate(rows):
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
        results.append(result_i)

        if verbose:
            dt = time.time() - t_row
            elapsed = time.time() - t0
            chi2 = result_i.diagnostics.get("chi2_dof", "?")
            chi2_str = f"{chi2:.2f}" if isinstance(chi2, float) else str(chi2)
            print(f"  [{i + 1}/{n_gal}] chi2/dof={chi2_str}, row={dt:.1f}s, total={elapsed:.0f}s")

    return results


# ---------------------------------------------------------------------------
# Population fitting
# ---------------------------------------------------------------------------


def fit_population(
    model: Model,
    observations_list: list,
    method: str = "vi",
    population_prior: dict | None = None,
    **kwargs,
):
    """Fit a population of galaxies with shared PSD hyperparameters.

    Thin wrapper around ``HierarchicalFitter``.

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
        Forwarded to ``HierarchicalFitter.run()``.

    Returns
    -------
    HierarchicalResult
    """
    from tengri.core.model import Model as ModelClass
    from tengri.distributions import Fixed
    from tengri.inference.hierarchical import HierarchicalFitter

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

    # Translate canonical → HierarchicalFitter method names
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

    hfitter = HierarchicalFitter(
        _model_factory,
        galaxies,
        psd_sigma_prior=psd_sigma_prior,
        psd_tau_prior=psd_tau_prior,
    )
    return hfitter.run(hier_method, **kwargs)
