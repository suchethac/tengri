# SPDX-License-Identifier: BSD-3-Clause
"""The astronomer-facing Catalog class for homogeneous catalog fitting.

#1317: one noun, action verbs. Wraps the existing CatalogFitter engine;
ingestion and validation happen at construction (fail fast, before any compile).
"""

from __future__ import annotations

import warnings

import jax
import numpy as np

from tengri.inference._batching import AUTO
from tengri.inference.catalog_fitter import (
    CatalogPosterior,
    _CatalogFitterOriginal as CatalogFitter,
)
from tengri.inference.catalog_ingest import ingest_catalog

__all__ = ["Catalog"]


def _stellar_config(fwd):
    """The StellarSEDComponent's config off a built ForwardModel, or None.

    Used to tell a tabulated-SFH model from a parametric one before accepting
    histories, so :meth:`Catalog.from_histories` can name the fix instead of
    letting the #996 runtime check fire deep in the forward pass.
    """
    from tengri.components.stellar.component import StellarSEDComponent

    try:
        chain = fwd.populations[0].sed._build_component_chain()
    except (AttributeError, IndexError):
        return None
    stellar = next((c for c in chain if isinstance(c, StellarSEDComponent)), None)
    return None if stellar is None else stellar.config


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
        Unit of the flux columns — **required**, with no default, so a table
        is never ingested under a guessed unit. One of: ``"cgs_fnu"``
        [erg/s/cm²/Hz], ``"mJy"``, ``"uJy"``, ``"maggies"``, ``"ab_mag"``.
    redshift_col : str, optional
        Column name for per-galaxy redshifts. If given, the model must have
        a ``Fixed`` redshift and a ``catalog_z_range`` that covers the
        redshift span in the table. If not given, the model must have a
        free redshift parameter. Exactly one of the two conditions is required.
    flux_cols : list[str], optional
        Flux column names **in your table** — they need not resemble the band
        names. If None, use ``"{name}"`` for each band in the model's
        observation.

        Binding is **positional**: ``flux_cols[i]`` supplies the flux for
        band ``i`` of the observation, so order them to match the observation
        rather than the table::

            # observation bands ("sdss_g", "sdss_r")
            Catalog(model, table, flux_cols=["FLUX_G", "FLUX_R"],
                    err_cols=["ERR_G", "ERR_R"], flux_unit="cgs_fnu")
    err_cols : list[str], optional
        Error column names in your table, bound positionally the same way. If
        None, use ``"{name}_err"`` for each band.
    censor_cols : dict[str, str], optional
        Mapping from band name to censoring flag column. Flag values: 0
        (detected), 1 (upper limit), -1 (lower limit).
    line_cols : list[str], optional
        Emission-line flux column names in your table, bound positionally to the
        observation's line order (from ``observation.line_fluxes.names``).
        If None and the model carries line fluxes, raises ValueError.
        If None and the model has no line fluxes, this parameter is ignored.
    line_err_cols : list[str], optional
        Emission-line error column names, bound positionally the same way.
        If None, use ``"{name}_err"`` for each line, matching the ``err_cols``
        convention.
    line_censor_cols : list[str], optional
        Emission-line censoring-flag columns, bound positionally the same
        way: 0 (detected), 1 (upper limit), -1 (lower limit). This is the
        catalog-scale form of ``Data(lines={'Halpha': (f, e, 'upper')})``.
        ``censor_cols`` is the *photometric* band axis and cannot express a
        line limit. Non-detection is a per-galaxy property, so declaring the
        flag on the Observation would apply it to the whole catalog.
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
        line_cols=None,
        line_err_cols=None,
        line_censor_cols=None,
        missing="error",
    ):
        """Initialize a Catalog with eager ingestion and validation.

        Parameters are documented in the class docstring.
        """
        self.fwd = fwd
        self.table = table
        # Set by from_histories; makes predict() argument-optional (#1396).
        self._history_columns = None
        # Memoized jit(vmap(...)) per channel, so the XLA cache survives across
        # calls instead of being rebuilt (and recompiled) each time.
        self._batched_cache = {}

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
                line_cols=line_cols,
                line_err_cols=line_err_cols,
                line_censor_cols=line_censor_cols,
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

                # A catalog_z_range lets per-galaxy redshift flow as a runtime
                # input so the program compiles ONCE. Without it, each distinct
                # redshift recompiles the fit (correct, just slow) — warn loudly
                # rather than refuse, so ``fit_batch``-style catalogs still run.
                z_range = fwd.populations[0].sed._catalog_z_range
                if z_range is None:
                    warnings.warn(
                        "redshift_col was provided but the model has no "
                        "catalog_z_range, so the forward model recompiles for EVERY "
                        "distinct redshift (one compile per galaxy). Build the model "
                        "with approx=WavePrecomp(catalog_z_range=(zmin, zmax)) to "
                        "compile once. See #1316.",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    z_lo, z_hi = z_range
                    z_min, z_max = ca.redshift.min(), ca.redshift.max()
                    if z_min < z_lo or z_max > z_hi:
                        raise ValueError(
                            f"Redshift span [{z_min:.3f}, {z_max:.3f}] exceeds model's "
                            f"catalog_z_range=[{z_lo:.3f}, {z_hi:.3f}]. "
                            f"Widen catalog_z_range at model build time or filter the table."
                        )
            else:
                # Condition 2: no redshift_col. A free redshift is fit per galaxy;
                # a Fixed redshift is legitimate too — every galaxy is fit at the
                # model's shared redshift (e.g. a cluster at known z, or the
                # fit_batch shared-redshift case). Nothing to validate here.
                pass

            # Fail loud on line flux mismatch (#1480): prevent silent substitution.
            if fwd.observation.has_line_fluxes and ca.line_flux_obs is None:
                raise ValueError(
                    "The model's Observation carries line_fluxes but this Catalog "
                    "has no line columns, so every galaxy would be scored against the "
                    "template galaxy's line fluxes. Pass line_cols=/line_err_cols= to "
                    "Catalog(...), or build the model without line_fluxes for a "
                    "photometry-only fit."
                )
        else:
            self._catalog_arrays = None

    def fit(
        self,
        method="map",
        *,
        key,
        forward_chunk_size=AUTO,
        n_pad=None,
        store=None,
        percentiles=None,
        reducers=None,
        properties=None,
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
        forward_chunk_size : int or "auto", default "auto"
            K galaxies evaluated in parallel per ``lax.map`` step (native methods only).
            ``"auto"`` derives K from the memory budget; an explicit int is honored.
        n_pad : int, "auto", or None
            Pad the catalog up to this many galaxies before running. Allows
            different catalog sizes to share XLA cache entries. ``None``
            (default) pads only to the next multiple of K.
        store : {"full", "summary"} or None
            Storage mode for posterior samples. ``None`` (default) auto-selects:
            ``"full"`` if N <= 1000, else ``"summary"`` with a warning.
            ``"full"`` retains all samples. ``"summary"`` computes percentiles
            and reducer statistics per name, then drops samples. A method that
            produces no samples (``"map"``, ``"laplace"``) has nothing to
            summarize: it warns and the result reports ``store="full"``.
        percentiles : tuple, optional
            Percentile levels to compute when ``store="summary"``, in the order
            they should appear as columns. Default ``(16, 50, 84)``. The levels
            are recorded on the result as ``percentile_levels`` and drive both
            ``post[name]`` (which reads the 50 column) and ``to_table()``
            labels — include 50 if you want a median.
        reducers : dict, optional
            Additional reducer functions {name: callable} to apply per name
            (e.g., {"mean": np.mean, "std": np.std}). With store="full", these
            are ignored.
        properties : tuple of str or None
            Derived properties to include in the summary block alongside the
            sampled parameters. ``None`` (default) includes every property the
            model provides, so ``post.percentiles["stellar_mass"]`` works;
            ``()`` includes none (parameters only — cheapest); an explicit
            tuple narrows it, which is the knob to reach for on a large
            catalog since each name costs one property evaluation per galaxy.
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
        # Per-galaxy presence mask (for heterogeneous photometry) rides in the dict too.
        galaxies = []
        for i in range(ca.n_galaxies):
            galaxy_dict = {
                "flux_obs": ca.flux[i],
                "noise": ca.noise[i],
            }
            if ca.redshift is not None:
                galaxy_dict["redshift"] = float(ca.redshift[i])
            # Only thread a presence mask for galaxies that ACTUALLY have an
            # absent band. An all-present galaxy needs no mask (the likelihood is
            # bit-identical without it), and threading an all-ones mask would trip
            # the batched-method guard for every ordinary catalog fit — silently
            # blocking MCMC/VI on catalogs with no masking at all.
            if ca.presence is not None and not bool(np.all(ca.presence[i])):
                galaxy_dict["presence"] = ca.presence[i].astype(np.float32)
            # Per-galaxy emission-line fluxes (#1480); thread if available.
            if ca.line_flux_obs is not None:
                galaxy_dict["line_flux_obs"] = ca.line_flux_obs[i]
                galaxy_dict["line_flux_err"] = ca.line_flux_err[i]
                # Per-galaxy line limits (#1469). Threaded for every galaxy
                # once any censor column is given -- including the all-zero
                # "detected" rows -- so the censored adapter is selected once
                # for the catalog. Threading it only for flagged galaxies
                # would split the compile key and recompile per pattern.
                if ca.line_censor is not None:
                    galaxy_dict["line_censor"] = ca.line_censor[i]
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
            properties=properties,
            **kwargs,
        )

        return result

    @classmethod
    def from_histories(
        cls,
        fwd,
        *,
        t_gyr,
        sfr,
        met=None,
        redshift=None,
        params=None,
        flux_unit="cgs_fnu",
    ):
        """Build a mock catalog from tabulated SFH / metallicity histories (#1396).

        The simulation-catalog entry point: a hydro sim, semi-analytic model or
        UniverseMachine-style run supplies SFH(t) and optionally Z(t) per galaxy,
        and the result is a :class:`Catalog` whose :meth:`predict` returns
        photometry for the whole table.

        Histories are **records, not parameters**. They enter at the action, the
        same way fluxes do, which is why this is a classmethod rather than a
        model-construction argument — the ``table`` SFH declares zero free
        parameters because the table *is* the SFH.

        Parameters
        ----------
        fwd : ForwardModel
            Model built with ``sfh={'type': 'table'}`` (and
            ``met={'type': 'table'}`` if ``met`` is given).
        t_gyr : array_like, shape (n_t,) or (N, n_t)
            Cosmic time [Gyr], strictly increasing. A 1-D grid is shared by
            every galaxy and broadcast.
        sfr : array_like, shape (N, n_t)
            Star formation rate [Msun/yr] at those times. Must be non-negative.
        met : array_like, shape (N, n_t), optional
            Metallicity history, log10(Z/Zsun), at the same nodes. Requires the
            model's ``metallicity_model='table'``.
        redshift : array_like, shape (N,), optional
            Per-galaxy redshift. Needs a ``catalog_z_range`` on the model for
            the catalog to stay one compile (#1316).
        params : dict, optional
            Per-galaxy scalars, ``{name: (N,)}`` — every free parameter of the
            model that is not supplied by the histories.
        flux_unit : str, default "cgs_fnu"
            Recorded for symmetry with the data-table constructor; predictions
            are returned in [erg/s/cm²/Hz] regardless.

        Returns
        -------
        Catalog
            Prediction-ready. ``.predict()`` needs no argument; ``.fit()`` is
            still meaningful (fit dust or redshift at a known, fixed SFH).

        Raises
        ------
        ValueError
            If the model does not declare a tabulated SFH (or a tabulated
            metallicity when ``met`` is given), if ``t_gyr`` is not strictly
            increasing, if any SFR is negative, or if the shapes disagree.

        Examples
        --------
        >>> cat = Catalog.from_histories(
        ...     fwd, t_gyr=t, sfr=sfr, redshift=z, params={"dust_tau_diff": tau}
        ... )
        >>> flux = cat.predict()
        """
        cfg = _stellar_config(fwd)
        if cfg is None or cfg.sfh_model != "table":
            got = "unknown" if cfg is None else repr(cfg.sfh_model)
            raise ValueError(
                f"from_histories needs a model built with sfh={{'type': 'table'}}, "
                f"whose SFH arrives at runtime; this model's sfh_model is {got}. "
                f"Rebuild with SEDModel.build(..., sfh={{'type': 'table'}})."
            )

        sfr = np.asarray(sfr)
        if sfr.ndim != 2:
            raise ValueError(f"sfr must be (N, n_t); got shape {sfr.shape}.")
        n_galaxies, n_t = sfr.shape

        t_gyr = np.asarray(t_gyr)
        if t_gyr.ndim == 1:
            t_gyr = np.broadcast_to(t_gyr, (n_galaxies, t_gyr.shape[0])).copy()
        if t_gyr.ndim != 2:
            raise ValueError(f"t_gyr must be (n_t,) or (N, n_t); got shape {t_gyr.shape}.")
        if t_gyr.shape[1] != n_t:
            raise ValueError(
                f"t_gyr and sfr disagree on n_t: t_gyr has {t_gyr.shape[1]}, "
                f"sfr has {n_t}. They index the same history nodes."
            )

        if not np.all(np.diff(t_gyr, axis=1) > 0.0):
            raise ValueError(
                "t_gyr must be strictly increasing along the time axis (cosmic "
                "time [Gyr], not lookback). A non-monotonic grid interpolates to "
                "garbage without raising downstream."
            )
        if np.any(sfr < 0.0):
            raise ValueError(
                "sfr has negative entries [Msun/yr]. A negative SFR subtracts "
                "stellar mass in the age-weight integral, silently."
            )

        columns = {"sfh_t_gyr": t_gyr, "sfh_sfr": sfr}

        if met is not None:
            if cfg.metallicity_model != "table":
                raise ValueError(
                    f"met= needs a model built with met={{'type': 'table'}}; this "
                    f"model's metallicity_model is {cfg.metallicity_model!r}. Either "
                    f"rebuild with a tabulated metallicity or drop met=."
                )
            met = np.asarray(met)
            if met.shape != sfr.shape:
                raise ValueError(
                    f"met must match sfr's shape (N, n_t); got {met.shape} vs {sfr.shape}."
                )
            columns["met_history"] = met

        if redshift is not None:
            columns["redshift"] = np.asarray(redshift)

        for name, value in (params or {}).items():
            columns[name] = np.asarray(value)

        catalog = cls(fwd, None, flux_unit=flux_unit)
        # Validate the assembled columns through the same gate predict() uses,
        # so from_histories cannot accept a table predict() would then reject.
        catalog._history_columns, _ = catalog._as_columns(columns)
        return catalog

    def _resolve_line_defs(self, lines):
        """Named lines -> LineDef objects, in the caller's order."""
        from tengri.observation.line_measurement import DESI_LINES

        by_name = {d.name: d for d in DESI_LINES}
        unknown = [n for n in lines if n not in by_name]
        if unknown:
            raise ValueError(
                f"unknown emission line(s) {unknown}. Available: "
                f"{sorted(by_name)}. Pass LineDef objects directly for lines "
                f"outside this set."
            )
        return tuple(by_name[n] for n in lines)

    def simulate(
        self,
        *,
        lines=None,
        properties=None,
        chunk_size=1024,
        n_pad=None,
        noise=None,
        key=None,
    ):
        """Simulate mock observables for the whole catalog (#1396 §8.1).

        One verb, several channels: photometry always, plus emission lines and
        derived properties on request. Intended for a catalog built by
        :meth:`from_histories`, where the SFH/Z tables come from a simulation.

        Parameters
        ----------
        lines : sequence of str, optional
            Emission line names to measure, from
            :data:`~tengri.observation.line_measurement.DESI_LINES` (e.g.
            ``("Halpha", "OIII_5007")``). Measured through the window-LUT fast
            path, which since #1396 serves tabulated histories.
        properties : sequence of str, optional
            Derived quantities to evaluate, e.g. ``("stellar_mass",)``.
        chunk_size : int, default 1024
            Galaxies per vmapped batch. Chunks are padded to a uniform width,
            so each channel costs one compile for the whole catalog.
        n_pad : int, optional
            Pad the catalog to at least this many galaxies, so catalogs of
            different sizes share one XLA cache entry per channel.
        noise : optional
            **Not implemented** — noise draws are #1312. Refused rather than
            ignored, so a caller asking for a noisy mock never silently receives
            a noiseless one.
        key : optional
            Reserved for the noise draw (#1312); refused for the same reason.

        Returns
        -------
        MockCatalog
            ``.photometry``, ``.lines``, ``.properties``, and ``.to_table()``.

        Raises
        ------
        NotImplementedError
            If ``noise`` or ``key`` is passed.
        ValueError
            If a line name is not recognized.

        Examples
        --------
        >>> mock = cat.simulate(lines=("Halpha",), properties=("stellar_mass",))
        >>> mock.lines["Halpha"].shape
        (100,)
        """
        if noise is not None or key is not None:
            raise NotImplementedError(
                "simulate(noise=..., key=...) is not implemented — the noise draw "
                "is tracked as #1312. simulate() currently returns noiseless "
                "predictions; drawing from a NoiseModel will compose here once "
                "#1312 lands."
            )

        columns, n_galaxies = self._prediction_columns(None)
        photometry = self._map_chunks(
            self.fwd.predict_photometry,
            columns,
            n_galaxies,
            chunk_size,
            tag="photometry",
            n_pad=n_pad,
        )

        line_values = {}
        if lines:
            line_defs = self._resolve_line_defs(tuple(lines))

            def _measure(params):
                return self.fwd.measure_line_fluxes(params, line_defs, fast=True)

            # The tag carries the line set: a different set is a different
            # program, and reusing one cache entry across them would be wrong.
            measured = self._map_chunks(
                _measure,
                columns,
                n_galaxies,
                chunk_size,
                tag=f"lines:{','.join(d.name for d in line_defs)}",
                n_pad=n_pad,
            )
            line_values = {d.name: np.asarray(measured)[:, i] for i, d in enumerate(line_defs)}

        property_values = {}
        if properties:
            names = tuple(properties)

            def _props(params):
                return self.fwd.predict_properties(params, names=names)

            # predict_properties returns dict[str, scalar]; vmapped and joined
            # per key that is already {name: (N,)}.
            property_values = self._map_chunks(
                _props,
                columns,
                n_galaxies,
                chunk_size,
                tag=f"properties:{','.join(names)}",
                n_pad=n_pad,
            )

        from tengri.inference.mock_catalog import MockCatalog

        return MockCatalog(
            photometry=np.asarray(photometry),
            filter_names=tuple(self.fwd.observation.photometry.names),
            lines=line_values,
            properties=property_values,
        )

    def _prediction_columns(self, param_table):
        """Resolve the columns to predict from — explicit table, or the stored one.

        Fixed parameter values are broadcast in as ``(N,)`` columns. Not every
        consumer merges them for itself: ``predict_photometry`` does, but the
        window-LUT line path reaches ``compute_joint_weights``, which reads
        ``params["met_logzsol"]`` directly and raises ``KeyError`` on a dict
        carrying only the free parameters. Merging once here keeps every channel
        — photometry, lines, properties — seeing the same complete dict. Caller
        columns win, so a per-galaxy ``redshift`` still overrides a fixed one.
        """
        if param_table is None:
            if self._history_columns is None:
                raise ValueError(
                    "predict() needs a param_table. Only a catalog built by "
                    "Catalog.from_histories(...) already carries its columns and "
                    "can be predicted with no argument."
                )
            columns = self._history_columns
            n_galaxies = int(next(iter(columns.values())).shape[0])
        else:
            columns, n_galaxies = self._as_columns(param_table)

        fixed = {
            name: np.broadcast_to(np.asarray(value), (n_galaxies,)).copy()
            for name, value in self.fwd.spec.get_fixed_values().items()
            if np.asarray(value).ndim == 0
        }
        return {**fixed, **columns}, n_galaxies

    def _batched(self, tag, fn):
        """A memoized ``jit(vmap(fn))``, so the XLA cache survives across calls.

        Memoization is the load-bearing part. ``jax.jit`` keys its cache on the
        wrapped callable, so building a fresh ``jax.jit(...)`` per call would
        give every call a fresh cache and recompile the catalog every time.

        Scope is **per Catalog**, deliberately. Two Catalog objects over the same
        model each compile their own program — one compile, not the ~236 an
        unjitted path costs, and the persistent on-disk JAX cache absorbs it on
        a later run. Sharing across instances would mean a module-level cache
        keyed on the ForwardModel, which is a frozen dataclass holding JAX
        arrays: the lifetime and hashing hazards are not worth one compile.
        """
        cached = self._batched_cache.get(tag)
        if cached is None:
            cached = jax.jit(jax.vmap(fn))
            self._batched_cache[tag] = cached
        return cached

    def _map_chunks(self, fn, columns, n_galaxies, chunk_size, *, tag, n_pad=None):
        """Evaluate ``fn`` over the galaxy axis in **uniformly shaped** chunks.

        Two things make the whole catalog cost **one** compile rather than
        hundreds, and both are necessary:

        1. **jit the vmapped call.** ``jax.vmap`` alone does not build a
           compiled program — it dispatches op by op, so each primitive is
           compiled separately and keyed on its own shapes. Measured on a
           3-band tabulated-history model: 236 compiles for a bare
           ``vmap(predict_photometry)``, versus 1 for ``jit(vmap(...))``.
        2. **Pad the galaxy axis to a uniform chunk width.** Those caches key on
           shape, so a ragged trailing chunk is a second shape and pays the whole
           cost again — ``chunk_size=3`` over 8 galaxies means widths 3 and 2,
           i.e. two compiles (and, unjitted, 472). The catalog is padded by
           repeating its last row so every chunk is exactly ``width`` wide, and
           the padding is discarded before returning.

        Parameters
        ----------
        fn : callable
            Single-galaxy callable, returning an array or a dict of scalars.
        columns : dict
            ``{name: (N, ...) ndarray}``.
        n_galaxies : int
            True galaxy count, before padding.
        chunk_size : int
            Maximum galaxies per batch.
        tag : str
            Cache key for the memoized ``jit(vmap(fn))``.
        n_pad : int, optional
            Pad the catalog to at least this many galaxies before chunking, so
            catalogs of *different* sizes share one XLA cache entry (the same
            meaning ``fit`` gives it). Must be >= ``n_galaxies``.

        Returns
        -------
        ndarray or dict
            Results for the real galaxies only, padding removed.
        """
        if n_pad is not None:
            if int(n_pad) < n_galaxies:
                raise ValueError(
                    f"n_pad={n_pad} is smaller than the catalog ({n_galaxies} "
                    f"galaxies); it pads up to a shared size, it does not truncate."
                )
            n_target = int(n_pad)
        else:
            n_target = n_galaxies

        width = min(int(chunk_size), n_target)
        n_chunks = (n_target + width - 1) // width
        n_padded = n_chunks * width

        if n_padded > n_galaxies:
            # Repeat the last galaxy: a duplicate row is always evaluable, where
            # zeros could be an unphysical SFH that trips a guard. Discarded below.
            pad = n_padded - n_galaxies
            columns = {
                name: np.concatenate([v, np.repeat(v[-1:], pad, axis=0)], axis=0)
                for name, v in columns.items()
            }

        batched = self._batched(tag, fn)
        results = [
            batched({name: v[i * width : (i + 1) * width] for name, v in columns.items()})
            for i in range(n_chunks)
        ]

        if isinstance(results[0], dict):
            return {
                key: np.concatenate([np.asarray(r[key]) for r in results], axis=0)[:n_galaxies]
                for key in results[0]
            }
        return np.concatenate(results, axis=0)[:n_galaxies]

    def _as_columns(self, param_table):
        """Validate a parameter table into uniform columns keyed by name (#1396).

        One channel carries everything: each column's **leading axis is the
        galaxy**, and the trailing shape is free. A per-galaxy scalar is simply
        the ``(N,)`` case; a tabulated history is ``(N, n_t)``. This is what the
        :meth:`predict` docstring has always promised, and what the previous
        ``np.stack(param_arrays, axis=1)`` could not express.

        Parameters
        ----------
        param_table : dict-like
            Mapping of name → array whose leading axis indexes galaxies.

        Returns
        -------
        columns : dict
            ``{name: ndarray}``, every value at least 1-D.
        n_galaxies : int
            The shared leading-axis length.

        Raises
        ------
        ValueError
            If a free parameter has no column, if a column is 0-D (no galaxy
            axis), or if the columns disagree on their leading-axis length.
        """
        missing = [name for name in self.fwd.spec.free_params if name not in param_table]
        if missing:
            raise ValueError(
                f"predict() has no column for free parameter(s) {missing}. "
                f"Every free parameter needs one value per galaxy; pass a "
                f"(N,) array for each, or make the parameter Fixed at build time."
            )

        columns = {name: np.asarray(value) for name, value in param_table.items()}

        scalars = [name for name, value in columns.items() if value.ndim == 0]
        if scalars:
            raise ValueError(
                f"predict() columns {scalars} are 0-D and carry no galaxy axis. "
                f"Every column's leading axis indexes galaxies — pass (N,) even "
                f"when the value is the same for all N."
            )

        lengths = {name: int(value.shape[0]) for name, value in columns.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"predict() columns must share the same leading axis length "
                f"(the galaxy count); got {lengths}."
            )

        return columns, next(iter(lengths.values()))

    def predict(self, param_table=None, *, chunk_size=1024, n_pad=None) -> np.ndarray:
        """Predict photometry for a catalog of parameters or histories.

        Evaluates the forward model on a table of per-galaxy columns, returning
        observed-frame spectral flux density (or flux in the chosen units).

        Parameters
        ----------
        param_table : dict-like, optional
            Mapping of name → array whose **leading axis indexes galaxies**;
            the trailing shape is free. A per-galaxy scalar is ``(N,)``; a
            tabulated history (``sfh_t_gyr``, ``sfh_sfr``, ``met_history``) is
            ``(N, n_t)``. Every free parameter needs a column; names the model
            does not recognize are reported by the forward model's own
            unknown-parameter check, so a typo cannot pass silently. Omit it
            entirely on a catalog built by :meth:`from_histories`, which
            already carries its columns.
        chunk_size : int, default 1024
            Galaxies evaluated per vmapped batch. Larger batches are faster
            but use more memory. Chunks are padded to a uniform width, so the
            whole catalog costs **one** compile regardless of how it divides.
        n_pad : int, optional
            Pad the catalog to at least this many galaxies before chunking, so
            catalogs of *different* sizes share one XLA cache entry (the same
            meaning :meth:`fit` gives it). Padding rows are discarded from the
            result.

        Returns
        -------
        ndarray, shape (N, n_filters)
            Predicted photometry [erg/s/cm²/Hz].

        Notes
        -----
        Columns are **not** restricted to ``spec.free_params``. A tabulated SFH
        declares zero free parameters — the table *is* the SFH — so extracting
        only free parameters dropped the history arrays entirely and the forward
        then refused with the #996 runtime check (#1396).

        Examples
        --------
        >>> params = {"stellar_mass": np.linspace(9, 12, 100), ...}
        >>> flux = cat.predict(params, chunk_size=64)
        >>> flux.shape
        (100, 3)

        Driving a catalog from tabulated histories:

        >>> flux = cat.predict(
        ...     {
        ...         "dust_tau_diff": tau,  # (N,)
        ...         "sfh_t_gyr": t_gyr,  # (N, n_t)
        ...         "sfh_sfr": sfr,  # (N, n_t)
        ...     }
        ... )
        """
        # vmap maps the leading axis of every leaf of the dict pytree, so
        # (N,) scalars and (N, n_t) histories batch through the same call —
        # no stacking, and no shape agreement required beyond the galaxy axis.
        columns, n_galaxies = self._prediction_columns(param_table)
        return self._map_chunks(
            self.fwd.predict_photometry,
            columns,
            n_galaxies,
            chunk_size,
            tag="photometry",
            n_pad=n_pad,
        )
