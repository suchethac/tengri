# SPDX-License-Identifier: BSD-3-Clause
"""The astronomer-facing Catalog class for homogeneous catalog fitting.

#1317: one noun, action verbs. Wraps the existing CatalogFitter engine;
ingestion and validation happen at construction (fail fast, before any compile).
"""

from __future__ import annotations

import jax
import numpy as np

from tengri.inference.catalog_fitter import (
    CatalogPosterior,
    _CatalogFitterOriginal as CatalogFitter,
)
from tengri.inference.catalog_ingest import ingest_catalog

__all__ = ["Catalog"]


class Catalog:
    """Astronomer-facing catalog inference interface.

    Wraps the existing per-galaxy :class:`CatalogFitter` engine with
    table-in/table-out access, ingestion and validation at construction time
    (fail fast, before any compile), and `.fit()` / `.predict()` verbs.

    Parameters
    ----------
    fwd : ForwardModel
        Shared forward model for all galaxies.
    table : dict-like or None
        Column mapping or object supporting `__getitem__[col] -> array`
        and `len()`. Required for `.fit()`; pass `None` for prediction-only.
    flux_unit : str
        Unit of flux columns. One of: ``"cgs_fnu"`` (default), ``"mJy"``,
        ``"uJy"``, ``"maggies"``, ``"ab_mag"``. [erg/s/cm²/Hz]
    redshift_col : str, optional
        Column name for per-galaxy redshifts. If given, the model must have
        a ``Fixed`` redshift and a ``catalog_z_range`` that covers the
        redshift span in the table. If not given, the model must have a
        free redshift parameter. Exactly one of the two conditions is required.
    flux_cols : list[str], optional
        Explicit flux column names. If None, use ``"{name}"`` for each band in
        the model's observation.
    err_cols : list[str], optional
        Explicit error column names. If None, use ``"{name}_err"`` for each band.
    censor_cols : dict[str, str], optional
        Mapping from band name to censoring flag column. Flag values: 0
        (detected), 1 (upper limit), -1 (lower limit).
    missing : {"error", "mask"}, default "error"
        Policy for NaN flux values. ``"error"`` raises with guidance on
        ``missing="mask"``; ``"mask"`` sets presence to False for that cell.

    Raises
    ------
    ValueError
        If a required column is missing, flux/error counts mismatch, redshift
        mechanism validation fails (spec §6.2), or a NaN is present with
        ``missing="error"``.
    TypeError
        If ``flux_unit`` is not provided.

    Examples
    --------
    >>> cat = Catalog(fwd, table, flux_unit="cgs_fnu", redshift_col="z")
    >>> post = cat.fit(key=jax.random.PRNGKey(0))
    >>> post.n_galaxies
    100
    >>> stellar_masses = post.properties["stellar_mass"]  # (100,)
    """

    def __init__(
        self,
        fwd,
        table,
        *,
        flux_unit,
        redshift_col=None,
        flux_cols=None,
        err_cols=None,
        censor_cols=None,
        missing="error",
    ):
        """Initialize a Catalog with eager ingestion and validation.

        Parameters are documented in the class docstring.
        """
        self.fwd = fwd
        self.table = table

        # Fail fast: ingest and validate at construction.
        if table is not None:
            ca = ingest_catalog(
                table,
                photometry=fwd.observation.photometry,
                flux_unit=flux_unit,
                flux_cols=flux_cols,
                err_cols=err_cols,
                redshift_col=redshift_col,
                censor_cols=censor_cols,
                missing=missing,
            )
            self._catalog_arrays = ca

            # Redshift mechanism validation (spec §6.2).
            # Exactly one of two conditions must hold:
            # 1. redshift_col given: model has Fixed redshift + catalog_z_range
            #    covers [z.min(), z.max()]
            # 2. free redshift with NO redshift_col
            if redshift_col is not None:
                # Condition 1: must have Fixed redshift and catalog_z_range
                if ca.redshift is None:
                    raise ValueError(
                        "redshift_col was provided but catalog has no redshift column."
                    )

                # Check model has Fixed redshift
                model_spec = fwd.spec
                try:
                    redshift_prior = model_spec.get_distribution("redshift")
                except (KeyError, AttributeError) as e:
                    raise ValueError(
                        "redshift_col was provided but the model has no redshift parameter."
                    ) from e

                from tengri.parameters.priors import Fixed

                if not isinstance(redshift_prior, Fixed):
                    raise ValueError(
                        "redshift_col was provided but the model has a free redshift parameter. "
                        "For per-galaxy redshifts, pass redshift_col but keep the model's "
                        "redshift=Fixed(...) with a catalog_z_range."
                    )

                # Check catalog_z_range covers the table's redshift span
                z_range = fwd.populations[0].sed._catalog_z_range
                if z_range is None:
                    raise ValueError(
                        "redshift_col was provided but the model has no catalog_z_range. "
                        "Set approx=WavePrecomp(catalog_z_range=...) at model build time."
                    )

                z_lo, z_hi = z_range
                z_min, z_max = ca.redshift.min(), ca.redshift.max()
                if z_min < z_lo or z_max > z_hi:
                    raise ValueError(
                        f"Redshift span [{z_min:.3f}, {z_max:.3f}] exceeds model's "
                        f"catalog_z_range=[{z_lo:.3f}, {z_hi:.3f}]. "
                        f"Widen catalog_z_range at model build time or filter the table."
                    )
            else:
                # Condition 2: no redshift_col, must have free redshift
                model_spec = fwd.spec
                try:
                    redshift_prior = model_spec.get_distribution("redshift")
                except (KeyError, AttributeError):
                    redshift_prior = None

                if redshift_prior is not None:
                    from tengri.parameters.priors import Fixed

                    if isinstance(redshift_prior, Fixed):
                        raise ValueError(
                            "Model has a Fixed redshift but no redshift_col was provided. "
                            "Pass redshift_col='...' to inject per-galaxy redshifts."
                        )
        else:
            self._catalog_arrays = None

    def fit(
        self,
        method="map",
        *,
        key,
        forward_chunk_size=1,
        n_pad=None,
        store=None,
        percentiles=None,
        reducers=None,
        **kwargs,
    ) -> CatalogPosterior:
        """Fit all galaxies independently.

        Parameters
        ----------
        method : str, default "map"
            Inference method. Recommended: ``"map"`` for quick point estimates,
            ``"mcmc_nuts"`` for posteriors. Never defaults to VI.
        key : jax.random.PRNGKey
            Base random key; per-galaxy keys are derived via ``jax.random.split``.
        forward_chunk_size : int, default 1
            K galaxies evaluated in parallel per ``lax.map`` step (native methods only).
        n_pad : int, "auto", or None
            Pad the catalog up to this many galaxies before running. Allows
            different catalog sizes to share XLA cache entries. ``None``
            (default) pads only to the next multiple of K.
        store : {"full", "summary"} or None
            Storage mode for posterior samples. ``None`` (default) auto-selects:
            ``"full"`` if N <= 1000, else ``"summary"`` with a warning.
            ``"full"`` retains all samples. ``"summary"`` computes percentiles
            and reducer statistics per property, then drops samples.
        percentiles : tuple, optional
            Percentiles to compute when store="summary". Default (16, 50, 84).
        reducers : dict, optional
            Additional reducer functions {name: callable} to apply per property
            (e.g., {"mean": np.mean, "std": np.std}). With store="full", these
            are ignored.
        **kwargs
            Forwarded to the inference method (e.g., ``n_warmup``, ``n_samples``
            for MCMC).

        Returns
        -------
        CatalogPosterior
            Container for N independent per-galaxy posteriors.

        Raises
        ------
        ValueError
            If no table was provided at construction (use `.predict()` instead).
        """
        if self._catalog_arrays is None:
            raise ValueError("No table provided at construction; use .predict() instead.")

        ca = self._catalog_arrays

        # Build per-galaxy dicts for the engine.
        # adapter: engine takes list-of-dicts; the CatalogArrays are the source of
        # truth (spec 9.1); engine-native arrays are T5's problem.
        # Per-galaxy redshift rides in the galaxy dict; the engine injects it into
        # each galaxy's fit as a fixed-value override (the #1329 mechanism) so it
        # actually reaches the forward pass — NOT just the reported params.
        galaxies = []
        for i in range(ca.n_galaxies):
            galaxy_dict = {
                "flux_obs": ca.flux[i],
                "noise": ca.noise[i],
            }
            if ca.redshift is not None:
                galaxy_dict["redshift"] = float(ca.redshift[i])
            galaxies.append(galaxy_dict)

        # Delegate to the existing engine.
        fitter = CatalogFitter(self.fwd, galaxies, data_type="photometry")
        result = fitter.run(
            method=method,
            key=key,
            forward_chunk_size=forward_chunk_size,
            n_pad=n_pad,
            store=store,
            percentiles=percentiles,
            reducers=reducers,
            **kwargs,
        )

        return result

    def predict(self, param_table, *, chunk_size=1024) -> np.ndarray:
        """Predict photometry for a catalog of parameters.

        Evaluates the forward model on a table of parameters, returning
        observed-frame spectral flux density (or flux in the chosen units).

        Parameters
        ----------
        param_table : dict-like
            Mapping of parameter name → array of values, shape ``(N,)`` or
            ``(N, ...)`` for each galaxy. Parameter names must match the model's
            free parameters.
        chunk_size : int, default 1024
            Batch size for vmap chunking (``jax.lax.map``). Larger batches are
            faster but use more memory.

        Returns
        -------
        ndarray, shape (N, n_filters)
            Predicted photometry [erg/s/cm²/Hz].

        Examples
        --------
        >>> params = {"stellar_mass": np.linspace(9, 12, 100), ...}
        >>> flux = cat.predict(params, chunk_size=64)
        >>> flux.shape
        (100, 3)
        """
        # Extract parameter values in the correct order.
        free_params = self.fwd.spec.free_params
        param_arrays = []
        for name in free_params:
            param_arrays.append(param_table[name])

        # Stack into a pytree (list of arrays).
        n_samples = len(param_arrays[0])

        # Vmap predict_photometry over the parameter table.
        def predict_one_row(params_tuple):
            # Reconstruct the params dict for this row.
            params = {name: val for name, val in zip(free_params, params_tuple)}
            return self.fwd.predict_photometry(params)

        # Use jax.lax.map for chunked vmap.
        stacked = np.stack(param_arrays, axis=1)  # (N, n_params)

        def compute_chunk(chunk):
            # chunk shape: (chunk_size, n_params)
            return jax.vmap(predict_one_row)(tuple(chunk[:, i] for i in range(chunk.shape[1])))

        # Use lax.map for chunking.
        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        results = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n_samples)
            chunk = stacked[start:end]
            chunk_result = compute_chunk(chunk)
            results.append(chunk_result)

        return np.concatenate(results, axis=0)
