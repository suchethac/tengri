# SPDX-License-Identifier: BSD-3-Clause
"""Independent per-galaxy catalog inference with optional K-way parallelism.

Unlike :class:`PopulationFitter`, galaxies share no parameters.
:class:`CatalogFitter` supports every method :class:`Fitter` accepts; for
``mcmc_nuts``, ``mcmc_hmc`` and the two ``tier="broken"`` ``native_vi_*``
backends it vmaps K galaxies per ``lax.map`` step so the compiled XLA graph
stays O(1) in N while K galaxies execute simultaneously on-device.
"""

from __future__ import annotations

import functools
import math
import time
import warnings
from dataclasses import dataclass, field

__all__ = ["CatalogFitter", "CatalogPosterior"]

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from tengri._mapping import ReadOnlyPropertyMapping
from tengri.inference._batching import AUTO, chunking_was_requested, resolve_forward_chunk_size
from tengri.inference._dimension_guard import warn_if_nuts_high_dim as _warn_if_nuts_high_dim
from tengri.inference._sample_utils import _mean_params, _vmap_samples_to_physical
from tengri.inference.backends.mcmc._shared import DEFAULT_MAX_NUM_DOUBLINGS
from tengri.inference.backends.mcmc.catalog import DEFAULT_MAP_INIT_STEPS

DEFAULT_PERCENTILES: tuple[float, ...] = (16.0, 50.0, 84.0)
"""Percentile levels used when ``percentiles=`` is not given."""


def _percentile_label(level) -> str:
    """Column-name fragment for a percentile level: ``2.5`` -> ``"2p5"``.

    ``.`` is spelled ``p`` so the resulting column name stays a valid
    identifier and survives a parquet/FITS round trip unquoted.
    """
    text = f"{float(level):g}"
    return text.replace(".", "p").replace("-", "m")


def _sample_dict_for_summary(posterior, properties=None):
    """Everything worth summarizing for one galaxy: parameters + derived properties.

    Parameters
    ----------
    posterior : Posterior
        A single galaxy's fit result; must still hold its samples.
    properties : sequence of str, ``None``, or empty
        Derived-property names to fold in. ``None`` (default) takes every
        property the model provides; an empty sequence takes none.

    Returns
    -------
    dict
        ``name -> ndarray, shape (n_samples,)``.

    Notes
    -----
    Parameters alone are not enough. ``store="summary"`` exists so a catalog
    of N ~ 1e5 never materializes its sample cube, and the quantity such a
    catalog is *for* is a derived one, ``stellar_mass``, ``sfr``. Summarizing
    only the sampled parameters left the spec's own worked example
    (``post.percentiles["stellar_mass"]``) raising ``KeyError`` (#1313).

    Property evaluation is the same work a ``store="full"`` user pays on first
    access; it is moved earlier so it can be chunk-reduced and dropped. If it
    fails for any reason the parameter summaries are still returned, a
    catalog fit must not die because one derived quantity could not be
    computed.
    """
    combined = dict(posterior.samples)
    if properties is not None and len(properties) == 0:
        return combined
    try:
        names = list(properties) if properties is not None else None
        combined.update(posterior.properties.to_dict(names=names))
    except Exception as exc:  # a summary must never take down the fit itself
        warnings.warn(
            f"Could not summarize derived properties ({type(exc).__name__}: {exc}); "
            "the summary block holds sampled parameters only. Pass properties=() "
            "to skip this step, or properties=('stellar_mass', ...) to narrow it.",
            UserWarning,
            stacklevel=3,
        )
    return combined


def _attach_summaries(posterior, store, percentiles, reducers, properties):
    """Summarize one galaxy in place, then drop its samples.

    Parameters
    ----------
    posterior : Posterior
        A single galaxy's result, still holding its samples.
    store : str
        ``"summary"`` to summarize; anything else is a no-op.
    percentiles : tuple
        Levels to compute, in the order the caller asked for them.
    reducers : dict or None
        Extra reducers ``{name: callable}``.
    properties : sequence of str or None
        Derived properties to fold in (``None`` = all, ``()`` = none).

    Returns
    -------
    bool
        Whether a summary block was produced. ``False`` for a sample-free
        method (MAP), which the caller must surface rather than swallow.
    """
    if store != "summary" or posterior.samples is None:
        return False
    block, summary = _compute_summaries(
        _sample_dict_for_summary(posterior, properties), percentiles, reducers
    )
    posterior._percentiles_stats_ = block
    if summary:
        posterior._summary_stats_ = summary
    posterior.samples = None  # the point of store="summary"
    return True


def _stack_summaries(posteriors, store, method, percentiles):
    """Stack per-galaxy summary blocks over the galaxy axis.

    Returns
    -------
    tuple
        ``(percentiles_dict, summary_dict, percentile_levels, effective_store)``.

    Notes
    -----
    ``effective_store`` is the honest answer, not the request. Asking for
    ``store="summary"`` from a method that produces no samples (MAP,
    ``laplace``) used to leave ``.percentiles`` and ``.summary`` at ``None``
    while ``.store`` still read ``"summary"``, a silent no-op that looks like
    a memory-bounded result and is not one. Now it warns and reports
    ``"full"`` (#1313).
    """
    have_blocks = bool(posteriors) and hasattr(posteriors[0], "_percentiles_stats_")
    if store != "summary":
        return None, None, None, store
    if not have_blocks:
        warnings.warn(
            f"store='summary' was requested but method={method!r} produced no posterior "
            "samples to summarize, so no percentile or reducer block was built and the "
            "result reports store='full'. Per-galaxy point values are still available "
            "via post['<name>'] and post.to_table(). Use a sampling method "
            "(e.g. 'mcmc_nuts') for percentile summaries.",
            UserWarning,
            stacklevel=3,
        )
        return None, None, None, "full"

    first = posteriors[0]
    stacked = {
        name: np.stack([p._percentiles_stats_[name] for p in posteriors])
        for name in first._percentiles_stats_
    }
    summary = None
    if getattr(first, "_summary_stats_", None):
        summary = {
            reducer: {
                name: np.array([p._summary_stats_[reducer][name] for p in posteriors])
                for name in first._summary_stats_[reducer]
            }
            for reducer in first._summary_stats_
        }
    return stacked, summary, tuple(percentiles), store


def _compute_summaries(samples, percentiles=None, reducers=None):
    """Compute percentiles and reducer statistics for a sample dict.

    Parameters
    ----------
    samples : dict
        Samples keyed by name, values are (n_samples,) arrays. Both sampled
        parameters and derived properties belong here, see
        :func:`_sample_dict_for_summary`.
    percentiles : tuple, optional
        Percentile values to compute. Default (16, 50, 84).
    reducers : dict, optional
        Additional reducers {name: callable} (e.g., {"mean": jnp.mean}).

    Returns
    -------
    percentiles_dict : dict
        Keys are names, values are (n_pct,) arrays **in the order the levels
        were requested**, the caller must record those levels alongside.
    summary_dict : dict
        Nested dict: {reducer_name: {name: scalar value}}.
    """
    if percentiles is None:
        percentiles = DEFAULT_PERCENTILES
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


#: The three engines a catalog method can dispatch to. ``sequential`` builds
#: one :class:`~tengri.inference.fitter.Fitter` per galaxy; the other two
#: compile a single loss and vmap the catalog through it.
SEQUENTIAL = "sequential"
MCMC_VMAPPED = "mcmc_vmapped"
NATIVE_VI = "native_vi"

_ALL_ENGINES = frozenset({SEQUENTIAL, MCMC_VMAPPED, NATIVE_VI})


@dataclass(frozen=True)
class GalaxyChannel:
    """One per-galaxy data channel and the engines that can carry it.

    Attributes
    ----------
    name : str
        Short identifier, for messages and tests.
    keys : tuple of str
        The galaxy-dict keys this channel occupies. Several keys can belong
        to one channel -- line fluxes travel as a value/error pair.
    engines : frozenset of str
        Engine kinds that actually thread it into the objective. An engine
        absent here must refuse, not ignore.
    what : str
        Human-readable subject, opening the refusal message.
    remedy : str
        What the caller should do instead, as a full sentence.

    Notes
    -----
    This table is the single place that knows which per-galaxy channels
    exist. The knowledge used to be split between ``catalog.py``, which
    writes the keys, and three engine branches that each read a hand-picked
    subset -- so a channel added to one side and not the other was dropped
    without a word. #1460, #1480 and #1599 are three instances of that one
    shape.
    """

    name: str
    keys: tuple[str, ...]
    engines: frozenset[str]
    what: str
    remedy: str


#: Every per-galaxy channel, and which engines thread it.
#:
#: Adding a channel means adding a row here *and* teaching the engines named
#: in ``engines`` to read it. A key that appears on a galaxy dict without a
#: row is refused outright, so the two halves cannot drift apart in silence.
GALAXY_CHANNELS: tuple[GalaxyChannel, ...] = (
    GalaxyChannel(
        name="photometry",
        keys=("flux_obs", "noise"),
        engines=_ALL_ENGINES,
        what="Per-galaxy photometry",
        remedy="Every engine carries it, so this should be unreachable.",
    ),
    GalaxyChannel(
        name="presence",
        keys=("presence",),
        engines=frozenset({SEQUENTIAL, MCMC_VMAPPED}),
        what="Per-galaxy presence masks (heterogeneous photometry)",
        remedy=(
            "Use method='mcmc_nuts' (batched, presence-aware) or "
            "method='map' (sequential). Batched native VI does not thread "
            "them (experimental, off the critical path; see #1337)."
        ),
    ),
    GalaxyChannel(
        name="redshift",
        keys=("redshift",),
        engines=frozenset({SEQUENTIAL, MCMC_VMAPPED}),
        what="Per-galaxy redshift",
        remedy=(
            "Use method='mcmc_nuts' / 'mcmc_hmc' (batched, runtime redshift) "
            "or method='map' (sequential). Batched native VI does not thread "
            "it (experimental, off the critical path; see #1337)."
        ),
    ),
    GalaxyChannel(
        name="line_fluxes",
        keys=("line_flux_obs", "line_flux_err"),
        engines=frozenset({SEQUENTIAL, MCMC_VMAPPED}),
        what="Per-galaxy emission-line fluxes (line_cols=)",
        remedy=(
            "Use method='map' (sequential) or 'mcmc_nuts' / 'mcmc_hmc' "
            "(batched). Batched native VI stacks only flux and noise."
        ),
    ),
    GalaxyChannel(
        name="line_censor",
        keys=("line_censor",),
        engines=frozenset({SEQUENTIAL}),
        what="Per-galaxy emission-line limits (line_censor_cols=)",
        remedy=(
            "Use method='map' (or another sequential method), which honors "
            "them per galaxy. The batched engines compile one loss for the "
            "whole catalog and carry no per-galaxy limit mask, so dropping "
            "the flags would fit every non-detection as a measurement."
        ),
    ),
)


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
    percentile_levels : tuple of float or None
        The percentile levels the ``percentiles`` block holds, in column order
        (e.g. ``(16, 50, 84)``). ``None`` when no summary was computed.

    Raises
    ------
    IndexError
        If ``__getitem__`` is called with an out-of-range integer index.
    KeyError
        If ``__getitem__`` is called with an unknown property name, or with a
        summarized name whose block carries no 50th percentile.

    Notes
    -----
    ``percentile_levels`` is not bookkeeping; it is what makes the block
    self-describing. Without it, both the median accessor and ``to_table``
    guessed from the array's *width*: the median was hardcoded to column 1 and
    labels were re-derived as ``[16, 50, 84]`` / ``[16, 84]`` / evenly-spaced.
    A caller asking for the spec's ``(2.5, 16, 50, 84, 97.5)`` therefore got
    the 16th percentile back as the "median", and an exported table whose
    columns were named ``_p0/_p25/_p50/_p75/_p100`` over 2.5/16/50/84/97.5
    data. Look levels up by value; never by position (#1313).

    Examples
    --------
    >>> result = cat.run("mcmc_nuts", key=jax.random.PRNGKey(0))
    >>> result[0].params  # first galaxy
    >>> result["stellar_mass"]  # per-galaxy medians, shape (n_galaxies,)
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
    percentile_levels: tuple | None = None

    def _effective_levels(self, n_pct: int) -> tuple | None:
        """The levels a block of width ``n_pct`` holds, or ``None`` if unknowable.

        Every path that builds a summary now records ``percentile_levels``, so
        a block without them predates that (an old pickle, or an object built
        by hand). For those, the only block the code could have produced at
        width ``len(DEFAULT_PERCENTILES)`` is the documented default, and at
        width 1 it is the median alone. Any other width is genuinely unknown,
        and unknown must stay unknown, inferring levels from a width is the
        bug this attribute exists to kill (#1313).
        """
        if self.percentile_levels is not None:
            return tuple(float(level) for level in self.percentile_levels)
        if n_pct == len(DEFAULT_PERCENTILES):
            return DEFAULT_PERCENTILES
        if n_pct == 1:
            return (50.0,)
        return None

    def _median_column(self, n_pct: int) -> int | None:
        """Column index of the 50th percentile, or ``None`` if it is not in the block."""
        levels = self._effective_levels(n_pct)
        if levels is None:
            return None
        for index, level in enumerate(levels):
            if level == 50.0:
                return index
        return None

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._property_medians(key)
        return self.posteriors[key]

    def _property_medians(self, name):
        """Per-galaxy medians for one property, shape ``(n_galaxies,)``.

        Answers from the stored percentile block under ``store="summary"``
        (same median column :meth:`to_table` uses), else from the per-galaxy
        posteriors via :attr:`properties`.

        The block column is found by **level**, not position. Hardcoding
        column 1 silently returned p16 for the spec's five-percentile example
        and p84 for ``(16, 84)`` (#1313). When no 50th percentile was
        requested there is no median to return, and refusing beats handing
        back a neighboring quantile under the median's name.
        """
        if self.percentiles is not None and name in self.percentiles:
            block = np.asarray(self.percentiles[name])
            median_idx = self._median_column(block.shape[1])
            if median_idx is None:
                raise KeyError(
                    f"No median available for {name!r}: this summary block holds "
                    f"percentiles {self._effective_levels(block.shape[1])}, which do "
                    "not include 50. Re-fit with a percentiles= tuple containing 50, "
                    "or read the level you want from post.percentiles[name]."
                )
            return block[:, median_idx]
        if not self.posteriors:
            available = sorted(self.percentiles.keys()) if self.percentiles else []
            raise KeyError(
                f"Unknown property {name!r}: this CatalogPosterior stores no per-galaxy "
                f"posteriors (store={self.store!r}) and its percentile block has "
                f"{available or 'no keys'}."
            )
        if name not in self.properties:
            from tengri.forward.properties import missing_property_message

            raise KeyError(missing_property_message(name, available=self.properties))
        vals = self.properties[name]
        if isinstance(vals, list):  # ragged posteriors, per-galaxy medians
            return np.array(
                [np.median(np.asarray(v)) if np.ndim(v) > 0 else float(v) for v in vals]
            )
        vals = np.asarray(vals)
        return np.median(vals, axis=1) if vals.ndim > 1 else vals

    def __iter__(self):
        return iter(self.posteriors)

    def __len__(self):
        return len(self.posteriors)

    @functools.cached_property
    def properties(self):
        """The property catalog over the galaxy axis.

        Contract §1, **same names, more axes**. The keys are the ones a single
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
            Dict mapping property names to (N,) arrays over galaxies. Percentile
            columns are named from the levels actually computed, ``percentiles=
            (2.5, 16, 50, 84, 97.5)`` gives ``stellar_mass_p2p5``,
            ``stellar_mass_p16``, ``stellar_mass_p50``, ``stellar_mass_p84``,
            ``stellar_mass_p97p5`` (``.`` is spelled ``p``, so the names stay
            valid identifiers and survive a parquet/FITS round trip). A bare
            ``stellar_mass`` median column appears only when 50 was requested.

        Notes
        -----
        The returned dict is a duck-type match for the input to
        :func:`~tengri.inference.catalog_ingest.ingest_catalog`, enabling
        round-trip workflows: ``Catalog(..., table).fit() -> cat.to_table()``.

        Column labels come from :attr:`percentile_levels`. They used to be
        re-derived from the block's *width*, which mislabeled every non-default
        request: the spec's five-percentile example exported as
        ``_p0/_p25/_p50/_p75/_p100`` over 2.5/16/50/84/97.5 data. Mislabeled
        numbers in a file that leaves the process are worse than no file
        (#1313).
        """
        table = {}

        # If percentiles are stored (store='summary' case), use them preferentially
        if self.percentiles:
            first_name = next(iter(self.percentiles.keys()))
            n_pct = np.asarray(self.percentiles[first_name]).shape[1]
            levels = self._effective_levels(n_pct)
            if levels is None or len(levels) != n_pct:
                raise ValueError(
                    f"summary block has {n_pct} percentile column(s) but its levels "
                    f"are {self.percentile_levels!r}. Refusing to export columns whose "
                    "labels cannot be trusted, re-fit so percentile_levels is recorded."
                )
            pct_values = list(levels)

            median_idx = self._median_column(n_pct)
            for name, percentile_array in self.percentiles.items():
                block = np.asarray(percentile_array)
                # The bare column is the MEDIAN. With no 50 in the request there
                # is no median, and emitting a neighboring quantile under the
                # bare name is how a p16 ends up in a paper as a mass estimate.
                if median_idx is not None:
                    table[name] = block[:, median_idx]
                for i, pct in enumerate(pct_values):
                    table[f"{name}_p{_percentile_label(pct)}"] = block[:, i]

        else:
            # store='full' case: try to get properties, with graceful fallback
            try:
                props = self.properties
                for name in props:
                    # Get the per-galaxy values
                    vals = props[name]
                    if isinstance(vals, list):
                        # Ragged posteriors, convert to array
                        medians = [np.median(np.asarray(v)) if np.ndim(v) > 0 else v for v in vals]
                        vals = np.array(medians)
                    else:
                        # Stacked posteriors
                        if vals.ndim > 1:
                            # Has sample axis, take median along it
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


class CatalogProperties(ReadOnlyPropertyMapping):
    """The property catalog lifted over the galaxy axis of a :class:`CatalogPosterior`.

    A ``CatalogPosterior`` is a *list of independent* ``Posterior`` objects, not
    a batched array, each galaxy was fit separately and may carry a different
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

    def ci(self, name: str, level: float = 0.68) -> np.ndarray:
        """Per-galaxy credible interval.

        Returns
        -------
        ndarray, shape (n_galaxies, 3)
            ``(lo, median, hi)`` per galaxy.
        """
        return np.array([p.properties.ci(name, level=level) for p in self._posteriors()])

    def __repr__(self):
        return (
            f"<CatalogProperties: {len(self.keys())} properties over "
            f"{len(self._posteriors())} galaxies>"
        )


class _CatalogFitterOriginal:
    """Per-galaxy catalog inference with optional K-way on-device parallelism.

    Wraps all :class:`~tengri.inference.fitter.Fitter` inference methods with a
    single ``run(method, ...)`` entry point. For ``mcmc_nuts`` / ``mcmc_hmc``
    (and the two ``tier="broken"`` ``native_vi_*`` backends), setting
    ``forward_chunk_size=K`` vmaps K galaxies per ``lax.map`` iteration so K
    galaxies execute in parallel on the accelerator while the XLA graph remains
    O(1) in the catalog size N.

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
    and shared across all galaxies; it does not capture any galaxy-specific
    data. The per-galaxy ``data`` and ``noise`` vectors are runtime arguments
    to the catalog VI engines, enabling ``jax.vmap`` over the catalog batch.

    For non-native methods (e.g. ``vi_nonlinear``, ``map``, ``mcmc``),
    ``CatalogFitter`` delegates to sequential :class:`Fitter` instances.
    JAX's XLA persistent cache means only the first galaxy pays the compilation cost.

    **Not JIT-compatible at the Python level**, ``CatalogFitter`` is a Python
    orchestrator; the individual catalog VI engine callables it dispatches to are
    JIT-compiled and vmap-compatible.

    Examples
    --------
    >>> cat = CatalogFitter(model, galaxies)
    >>> result = cat.run("mcmc_nuts", key=jax.random.PRNGKey(0), forward_chunk_size=4)
    >>> result[0].params  # first galaxy posterior
    """

    _NATIVE_VMAPPABLE: frozenset = frozenset({"native_vi_linear", "native_vi_nonlinear"})
    #: Sampling methods that honor ``forward_chunk_size`` by vmapping K galaxies'
    #: NUTS/HMC chains per ``lax.map`` step (per-galaxy warmup, diagonal mass).
    _MCMC_VMAPPABLE: frozenset = frozenset({"mcmc_nuts", "mcmc_hmc"})

    def _engine_kind(self, resolved: str) -> str:
        """Which of the three engines ``resolved`` dispatches to."""
        if resolved in self._NATIVE_VMAPPABLE:
            return NATIVE_VI
        if resolved in self._MCMC_VMAPPABLE:
            return MCMC_VMAPPED
        return SEQUENTIAL

    def _refuse_unsupported_channels(self, resolved: str) -> None:
        """Refuse per-galaxy channels the chosen engine cannot carry.

        Parameters
        ----------
        resolved : str
            The resolved method name, as returned by ``resolve_method``.

        Raises
        ------
        NotImplementedError
            If any galaxy carries a channel this engine does not thread, or
            a key that :data:`GALAXY_CHANNELS` does not declare at all.

        Notes
        -----
        One rule over one table, replacing a hand-written branch per channel.
        The branches were the problem: a channel nobody wrote a branch for was
        not refused, it was *dropped in silence*, because the engine simply
        never read that key. Per-galaxy line fluxes reached the objective on
        ``mcmc_nuts`` and not on ``map`` for exactly that reason (#1599), and
        the same shape produced #1460 and #1480.

        The unknown-key arm is what makes this hold going forward: adding a
        key in ``catalog.py`` without declaring it here fails loudly on every
        engine rather than working on whichever one happens to read it.
        """
        kind = self._engine_kind(resolved)
        declared = {key: ch for ch in GALAXY_CHANNELS for key in ch.keys}
        present = {key for galaxy in self.galaxies for key in galaxy}

        for key in sorted(present):
            channel = declared.get(key)
            if channel is None:
                raise NotImplementedError(
                    f"Per-galaxy key {key!r} is not a declared catalog "
                    "channel, so no engine is known to carry it and it would "
                    "be silently dropped. Add it to GALAXY_CHANNELS in "
                    "catalog_fitter.py, naming the engines that thread it."
                )
            if kind not in channel.engines:
                carried = sorted(channel.engines)
                raise NotImplementedError(
                    f"{channel.what} is not threaded by method={resolved!r} "
                    f"({kind}); it would be silently dropped. {channel.remedy} "
                    f"Engines that carry it: {carried}."
                )

    def _galaxy_line_fluxes(self, galaxy):
        """This galaxy's emission-line values on the shared line schema.

        Parameters
        ----------
        galaxy : dict
            One entry of ``self.galaxies``; carries ``line_flux_obs`` /
            ``line_flux_err`` when the catalog was built with ``line_cols``.

        Returns
        -------
        LineFluxData or None
            ``None`` when this catalog threads no per-galaxy lines, in which
            case the Fitter falls back to the Observation's own values.

        Notes
        -----
        The vmapped branch stacks these values into the batched loss, but the
        sequential branch built its per-galaxy ``Fitter`` without them, so
        every galaxy was scored against the *template* Observation's line
        flux -- the substitution #1480's guard exists to prevent (#1599).

        Only the measured values are per-galaxy. Names, wavelengths and the
        limit flags are the instrument schema and stay shared, so replacing
        them here would change the compile key and force a recompile per
        galaxy.
        """
        if "line_flux_obs" not in galaxy:
            return None

        import dataclasses

        obs = getattr(self.model, "observation", None)
        template = getattr(obs, "line_fluxes", None) if obs is not None else None
        if template is None:
            return None

        replacements = {
            "fluxes": jnp.asarray(galaxy["line_flux_obs"]),
            "errors": jnp.asarray(galaxy["line_flux_err"]),
        }
        censor = galaxy.get("line_censor")
        if censor is not None:
            # Trinary flags -> the boolean pair LineFluxData stores. Rebuilt
            # per galaxy because a non-detection is a property of the galaxy,
            # not of the instrument (#1469).
            censor = np.asarray(censor)
            replacements["is_upper_limit"] = censor == 1
            replacements["is_lower_limit"] = censor == -1
        return dataclasses.replace(template, **replacements)

    def __init__(self, model, galaxies, data_type="photometry", *, approx="auto"):
        from tengri.inference.fitter import _resolve_batch_fit_approx
        from tengri.inference.jit_engine import CompileCache

        # Same default as Fitter and PopulationFitter: a catalog fit pays the
        # per-evaluation cost times the catalog size, so it routes through the
        # precompute LUT unless the caller says approx=None (exact) or hands an
        # explicit config.
        self.model = _resolve_batch_fit_approx(model, approx, data_type)
        self.approx = approx
        self.galaxies = list(galaxies)
        self.n_galaxies = len(self.galaxies)
        # For the #1671 bias advisory in run(): the exact reference when
        # resolution produced a LUT clone. Deferred to run() so construction
        # stays cheap; priced once for the whole catalog (the probe caches on
        # the clone, one exact forward, not one per galaxy).
        self._pre_approx_model = model if self.model is not model else None
        self._lut_bias_checked = False
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
        method="mcmc_nuts",
        *,
        key,
        forward_chunk_size=AUTO,
        n_pad: int | str | None = None,
        devices=None,
        store: str | None = None,
        percentiles: tuple | None = None,
        reducers: dict | None = None,
        properties: tuple | None = None,
        allow_unvalidated: bool = False,
        **kwargs,
    ):
        """Fit all galaxies independently.

        Parameters
        ----------
        method : str
            Any method accepted by :class:`~tengri.inference.fitter.Fitter`.
            ``mcmc_nuts`` (default), ``mcmc_hmc`` and the two ``native_vi_*``
            backends support ``forward_chunk_size``-based on-device
            parallelism; every other method runs sequentially per galaxy.

            The default was ``native_vi_linear`` until 2026-07. That backend is
            registered ``tier="broken"``, it segfaults on DPL/dense_basis
            photometry mocks (#231), so the documented default could not be
            run as written. It also raises ``NotImplementedError`` for
            per-galaxy redshift and for presence masks, which ``mcmc_nuts``
            supports. NUTS is ``tier="primary"``, keeps ``forward_chunk_size``
            and ``n_pad``, and is the only tier that honors ``devices``.

            Both ``native_vi_*`` backends now refuse to run at all without
            ``allow_unvalidated=True``, changing the default alone left them
            one keystroke away, since this path never consulted the tier.
        key : jax.random.PRNGKey
            Base random key; per-galaxy keys are derived via ``jax.random.split``.
        forward_chunk_size : int
            K galaxies evaluated in parallel per ``lax.map`` step. Applies to
            ``mcmc_nuts`` / ``mcmc_hmc`` and to ``native_vi_linear`` /
            ``native_vi_nonlinear``; ignored (with a warning) for every other
            method. ``K=1`` (default) = sequential; ``K=N`` = fully vmapped.
        n_pad : int, "auto", or None
            Pad the catalog up to this many galaxies before running. The
            extra slots are dummy galaxies whose results are discarded
            after the run; their only purpose is to make the XLA program
            shape match a previously-cached compile so different catalog
            sizes share an artifact.

            - ``None`` (default), pad only to the next multiple of K
              (existing behavior).
            - ``"auto"``: pad to the next power of 2.
            - ``int``: pad to exactly this many galaxies (must be
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
        allow_unvalidated : bool, optional
            Run a ``tier="broken"`` method anyway, for benchmarking or backend
            development, not for science. Default False.
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
        BackendError
            If ``method`` is registered ``tier="broken"`` and
            ``allow_unvalidated`` is False.
        UserWarning
            If ``forward_chunk_size > 1`` is passed for a method that does not
            batch (ignored).

        Notes
        -----
        Padding is safe in :class:`CatalogFitter` because the catalog VI
        engine is fully per-galaxy: each galaxy's ``run_fn`` operates on
        only its own ``(init_pos, key, data, noise)`` with no cross-galaxy
        reduction. Dummy padded galaxies converge to their own (irrelevant)
        posteriors and are trimmed off the result. The same trick is
        **not** safe for :class:`PopulationFitter`, where the hierarchical
        population field couples all galaxies, there, rely on
        :func:`tengri.enable_persistent_cache` instead.
        """
        from tengri.inference._backend_registry import refuse_if_broken
        from tengri.inference.fitter import resolve_method

        # #1671 made operational: this catalog fit runs on a resolved
        # precompute LUT, so price the LUT's forward bias against the whole
        # catalog's SNR once and warn with the number when the amplified
        # estimate is material. Once per catalog fitter instance.
        if not self._lut_bias_checked and self._pre_approx_model is not None:
            self._lut_bias_checked = True
            from tengri.inference.fitter import _warn_if_lut_bias_amplified

            try:
                all_flux = np.concatenate(
                    [np.asarray(g["flux_obs"]).reshape(-1) for g in self.galaxies]
                )
                all_noise = np.concatenate(
                    [np.asarray(g["noise"]).reshape(-1) for g in self.galaxies]
                )
            except Exception:
                all_flux = all_noise = None
            if all_flux is not None:
                _warn_if_lut_bias_amplified(
                    self._pre_approx_model,
                    self.model,
                    all_flux,
                    all_noise,
                    self.data_type,
                    surface="CatalogFitter",
                )

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
            percentiles = DEFAULT_PERCENTILES

        resolved = resolve_method(method)
        # `resolve_method` validates only that the NAME is canonical -- it never
        # consults the registry tier. Dispatch below goes straight into the
        # backend module, so without this call `check_usable` is not on the path
        # and a tier="broken" method runs silently (#1394).
        refuse_if_broken(resolved, allow_unvalidated=allow_unvalidated)
        # Refusal precedes advice: warning about NUTS warmup memory for a method
        # that is about to be refused would be noise.
        #
        # Pre-flight memory guard, shared with Fitter/PopulationFitter. D here is
        # the PER-GALAXY free-parameter count: the batched MCMC path vmaps N
        # independent chains of that size, so the per-chain mass matrix, the
        # term that goes O(D^2), is set by the single-galaxy spec, not by N.
        _spec = getattr(self.model, "spec", None)
        _warn_if_nuts_high_dim(
            resolved, getattr(_spec, "n_free", None), surface="Catalog.fit / CatalogFitter.run"
        )
        # Every per-galaxy channel is checked against the engine about to run,
        # from one table (GALAXY_CHANNELS). Each channel used to carry its own
        # hand-written branch here, and a channel whose branch nobody wrote was
        # dropped in silence -- that is how per-galaxy line fluxes reached the
        # objective on mcmc_nuts and not on map (#1599).
        self._refuse_unsupported_channels(resolved)
        _has_pg_z = any("redshift" in g for g in self.galaxies)
        if _has_pg_z and resolved in self._MCMC_VMAPPABLE and self._catalog_z_range() is None:
            raise ValueError(
                "Batched per-galaxy redshift requires a catalog_z_range so ONE "
                "compiled program serves all redshifts. Build the model with "
                "approx=WavePrecomp(catalog_z_range=(zmin, zmax)), or use "
                "method='map' (sequential, recompiles per redshift). See #1337."
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
                properties=properties,
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
                properties=properties,
                **kwargs,
            )
        else:
            # Both lists are DERIVED from the dispatch frozensets above, never
            # written out by hand: the previous hard-coded text still named only
            # the native pair long after `_MCMC_VMAPPABLE` was added, steering
            # callers off the working path onto the broken one (#1394).
            if devices is not None:
                warnings.warn(
                    f"devices={devices!r} is ignored for method={method!r}. "
                    "Multi-device sharding is currently supported for "
                    f"{' / '.join(sorted(self._MCMC_VMAPPABLE))} only.",
                    UserWarning,
                    stacklevel=2,
                )
            # Ask the predicate, not the literal: `!= 1` was a comparison
            # against the OLD default, so once #1189 made AUTO the default it
            # fired for every caller who passed nothing (#1189 follow-up).
            if chunking_was_requested(forward_chunk_size):
                # Name the supported methods FROM the dispatch sets, not from a
                # hand-written list. The literal this replaces said "only
                # native_vi_linear and native_vi_nonlinear" long after
                # _MCMC_VMAPPABLE gave mcmc_nuts/mcmc_hmc the same capability,
                # so the advice steered callers off the working path onto a
                # tier="broken" one.
                #
                # The two sets are named SEPARATELY rather than unioned: half of
                # a combined list cannot be reached without allow_unvalidated,
                # and a reader picking a name out of it would hit a refusal.
                usable = " / ".join(sorted(self._MCMC_VMAPPABLE))
                gated = " / ".join(sorted(self._NATIVE_VMAPPABLE))
                warnings.warn(
                    f"forward_chunk_size={forward_chunk_size} is ignored for "
                    f"method={method!r}. Chunked parallelism is supported by "
                    f"{usable}, and by {gated} under allow_unvalidated=True "
                    "(those two are tier='broken').",
                    UserWarning,
                    stacklevel=2,
                )
            if n_pad is not None:
                warnings.warn(
                    f"n_pad={n_pad!r} is ignored for method={method!r}. "
                    "Sequential per-galaxy fits don't benefit from "
                    "shape-bucketing, each galaxy is its own jit.",
                    UserWarning,
                    stacklevel=2,
                )
            return self._run_sequential(
                method,
                key=key,
                store=store,
                percentiles=percentiles,
                reducers=reducers,
                properties=properties,
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

    def _catalog_z_range(self):
        """The model's ``catalog_z_range`` (the runtime-redshift LUT span), or None.

        A ``catalog_z_range`` is what lets a per-galaxy redshift flow as a runtime
        input so ONE compiled program serves all z (#1337 phase 2). Reads it from a
        ``ForwardModel`` (via its population's SED) or a bare ``SEDModel``.
        """
        m = self.model
        if hasattr(m, "populations"):
            try:
                return m.populations[0].sed._catalog_z_range
            except (AttributeError, IndexError):
                return None
        return getattr(m, "_catalog_z_range", None)

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
        forward_chunk_size=AUTO,
        n_pad: int | str | None = None,
        store: str = "full",
        percentiles: tuple = DEFAULT_PERCENTILES,
        reducers: dict | None = None,
        properties: tuple | None = None,
        n_iterations=20,
        n_samples=3,
        n_posterior_samples=500,
        kl_rtol=1e-2,
        verbose=True,
    ):
        from tengri.inference.posterior import Posterior

        t0 = time.time()
        n_gal = self.n_galaxies
        # K galaxies per dispatch (#1189). AUTO by default: one galaxy per
        # dispatch is the anti-pattern. This path already requires uniform
        # n_data (``_validate_uniform_data`` below runs regardless of K), so
        # homogeneous=True holds here.
        K = resolve_forward_chunk_size(
            forward_chunk_size,
            n_gal=n_gal,
            n_data_per_gal=len(self.galaxies[0]["flux_obs"]) if n_gal else None,
        )
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

            # Build the Posterior FIRST: summarizing derived properties needs
            # the model, which only the Posterior carries. Samples are dropped
            # inside _attach_summaries once the block exists.
            post_i = Posterior(
                samples=samples_phys,
                params=best_params,
                method=f"CatalogFitter/{method_tag}",
                wall_time_s=time.time() - t0,
                diagnostics={"n_iterations": int(all_n_iters[i])},
                loss_history=None,
                _model=self.model,
            )
            _attach_summaries(post_i, store, percentiles, reducers, properties)
            posteriors.append(post_i)

        wall = time.time() - t0
        if verbose:
            print(f"  Done in {wall:.1f}s ({wall / n_gal:.2f}s/galaxy)")

        # Stack per-galaxy summary blocks; record WHICH levels they hold and
        # report the store that actually happened (see _stack_summaries).
        cat_percentiles, cat_summary, cat_levels, store = _stack_summaries(
            posteriors, store, method_tag, percentiles
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
            percentile_levels=cat_levels,
        )

    # ------------------------------------------------------------------
    # Internal: vectorized sampling path (mcmc_nuts / mcmc_hmc)
    # ------------------------------------------------------------------

    def _run_native_mcmc(
        self,
        method_tag,
        *,
        key,
        forward_chunk_size=AUTO,
        n_pad: int | str | None = None,
        devices=None,
        store: str = "full",
        percentiles: tuple = DEFAULT_PERCENTILES,
        reducers: dict | None = None,
        properties: tuple | None = None,
        n_warmup=300,
        n_burnin=100,
        n_samples=1000,
        max_num_doublings=DEFAULT_MAX_NUM_DOUBLINGS,
        n_leapfrog_steps=10,
        target_accept_rate=0.85,
        dense_mass_matrix=None,
        init_from=None,
        map_init_steps=DEFAULT_MAP_INIT_STEPS,
        verbose=True,
    ):
        """Vectorized per-galaxy NUTS/HMC sampling via ``lax.map(batch_size=K)``.

        Each galaxy runs its own BlackJAX window adaptation and chain inside a
        single JIT'd program; K galaxies execute per ``lax.map`` step so the
        compiled graph stays O(1) in the catalog size while K chains run in
        parallel on the accelerator. Returns one :class:`Posterior` per galaxy,
        each carrying posterior ``samples``, the same public contract as the
        sequential path, minus the N serial warmups.

        ``dense_mass_matrix=None`` (the default) resolves through the same
        auto-policy as the single-galaxy samplers: dense below D = 8, diagonal
        at or above it (#319). Until PR #2031 this path hardcoded ``False`` and
        read it as ``bool(dense_mass_matrix)``, so a D=7 catalog silently got a
        diagonal mass where a single fit of the same model got a dense one, and
        passing the documented ``None`` default selected diagonal rather than
        the policy. Pass ``True``/``False`` to override.

        ``init_from`` mirrors the single-galaxy contract in
        ``_maybe_map_init`` (``tengri.inference._sample_utils``):

        * ``None`` (default): each galaxy gets its own ADAM MAP warm start,
          which is what a single fit has always done. Before PR #2031 this path had
          no MAP step and every galaxy started at ``0.1 * N(0, 1)`` about the
          prior center.
        * ``"prior"``: that former behavior, kept for reproducing older runs.
        * an array of shape ``(n_gal, n_dim)``: starting points in the flat
          unconstrained space, used as given.
        * a list of ``n_gal`` parameter dicts, or one dict broadcast to every
          galaxy: physical values, converted for you. The medians of a previous
          :class:`CatalogPosterior` are the intended source.

        When ``devices`` is given (``"all"`` or a device list), the galaxy axis
        is sharded across those devices via ``jax.shard_map``, each device runs
        ``lax.map`` on its own slice of the catalog with no cross-device
        reduction (galaxies are independent). Bit-parity with the single-device
        path holds up to float round-off.
        """
        from tengri.inference.backends.mcmc.catalog import (
            build_catalog_mcmc_engine,
        )
        from tengri.inference.backends.mcmc.nuts import _resolve_dense_mass_matrix
        from tengri.inference.posterior import Posterior

        sampler = "nuts" if method_tag == "mcmc_nuts" else "hmc"
        t0 = time.time()
        n_gal = self.n_galaxies
        # K galaxies per dispatch (#1189). AUTO by default: one galaxy per
        # dispatch is the anti-pattern. This path already requires uniform
        # n_data (``_validate_uniform_data`` below runs regardless of K), so
        # homogeneous=True holds here.
        K = resolve_forward_chunk_size(
            forward_chunk_size,
            n_gal=n_gal,
            n_data_per_gal=len(self.galaxies[0]["flux_obs"]) if n_gal else None,
        )
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
        # Per-galaxy redshift override (#1337 phase 2): only when the catalog actually
        # carries a per-galaxy Fixed redshift. Free / shared-redshift catalogs leave
        # thread_redshift False, so the compiled program is unchanged.
        per_galaxy_z = any("redshift" in g for g in self.galaxies)
        # Per-galaxy emission-line fluxes (#1480): only when the catalog carries them
        per_galaxy_lines = any("line_flux_obs" in g for g in self.galaxies)

        # Resolve dense-vs-diagonal through the same policy the single-galaxy
        # samplers use, so a catalog fit of a model does not silently get a
        # different mass matrix from a single fit of it (PR #2031, #1999).
        _dummy_flat = ravel_pytree(fitter._initialize_unbounded(jax.random.PRNGKey(0)))[0]
        user_set_dense = dense_mass_matrix is not None
        use_dense = _resolve_dense_mass_matrix(dense_mass_matrix, int(_dummy_flat.shape[0]))
        if verbose and not user_set_dense:
            policy = "dense (D<8)" if use_dense else "diagonal (D>=8, #319)"
            print(f"CatalogFitter auto-mass-matrix: {policy}")

        run_one, unravel_fn = build_catalog_mcmc_engine(
            fitter,
            sampler,
            n_warmup=n_warmup,
            n_burnin=n_burnin,
            n_samples=n_samples,
            max_num_doublings=max_num_doublings,
            n_leapfrog=n_leapfrog_steps,
            target_accept_rate=target_accept_rate,
            use_dense=use_dense,
            thread_redshift=per_galaxy_z,
            thread_line_fluxes=per_galaxy_lines,
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
        # Per-galaxy presence masks (0/1) for heterogeneous catalogs; default to all-ones.
        # Use the same dtype as data for consistent numeric behavior under multiplication.
        all_presence_orig = jnp.stack(
            [
                jnp.asarray(g.get("presence", np.ones(n_data)), dtype=all_data_orig.dtype)
                for g in self.galaxies
            ]
        )
        # Per-galaxy redshift (#1337 phase 2); dummy zeros when not threaded.
        if per_galaxy_z:
            all_redshift_orig = jnp.stack(
                [jnp.asarray(g["redshift"], dtype=all_data_orig.dtype) for g in self.galaxies]
            )
        else:
            all_redshift_orig = jnp.zeros((n_gal,), dtype=all_data_orig.dtype)
        # Per-galaxy emission-line fluxes (#1480); dummy zeros when not threaded.
        # Stack using the same n_padded and K as data/noise to avoid recompilation.
        if per_galaxy_lines:
            # Stack per-galaxy line flux arrays using the same n_padded and K
            all_line_flux_orig = jnp.stack(
                [jnp.asarray(g["line_flux_obs"], dtype=all_data_orig.dtype) for g in self.galaxies]
            )
            all_line_err_orig = jnp.stack(
                [jnp.asarray(g["line_flux_err"], dtype=all_data_orig.dtype) for g in self.galaxies]
            )
        else:
            all_line_flux_orig = jnp.zeros((n_gal, 0), dtype=all_data_orig.dtype)
            all_line_err_orig = jnp.zeros((n_gal, 0), dtype=all_data_orig.dtype)
        if n_pad_extra > 0:
            all_data = jnp.concatenate([all_data_orig, jnp.zeros((n_pad_extra, n_data))], axis=0)
            all_noise = jnp.concatenate([all_noise_orig, jnp.ones((n_pad_extra, n_data))], axis=0)
            # Padded rows: all-ones presence (no masking, they're discarded anyway).
            pad_presence = jnp.ones((n_pad_extra, n_data), dtype=all_data_orig.dtype)
            all_presence = jnp.concatenate([all_presence_orig, pad_presence], axis=0)
            # Padded rows reuse galaxy 0's redshift (a valid LUT value; discarded).
            pad_z = jnp.full((n_pad_extra,), all_redshift_orig[0], dtype=all_data_orig.dtype)
            all_redshift = jnp.concatenate([all_redshift_orig, pad_z], axis=0)
            # Padded rows for line fluxes: zeros (discarded after).
            # Unconditionally pad to shape (n_padded, n_line_cols), even if n_line_cols==0.
            # All eight arrays (all_data, all_noise, all_presence, all_redshift,
            # all_line_flux, all_line_err, plus the rest) must share leading dim n_padded
            # for jax.lax.map to work. A (n_pad_extra, 0) block is well-formed.
            n_line_cols = all_line_flux_orig.shape[1] if all_line_flux_orig.ndim > 1 else 0
            pad_line = jnp.zeros((n_pad_extra, n_line_cols), dtype=all_data_orig.dtype)
            all_line_flux = jnp.concatenate([all_line_flux_orig, pad_line], axis=0)
            all_line_err = jnp.concatenate([all_line_err_orig, pad_line], axis=0)
        else:
            all_data, all_noise, all_presence, all_redshift = (
                all_data_orig,
                all_noise_orig,
                all_presence_orig,
                all_redshift_orig,
            )
            all_line_flux = all_line_flux_orig
            all_line_err = all_line_err_orig

        init_keys = jax.random.split(key, n_padded)
        all_init = jnp.stack(
            [ravel_pytree(fitter._initialize_unbounded(k))[0] for k in init_keys[:n_gal]]
        )
        if init_from is not None and not isinstance(init_from, str):
            all_init = self._init_from_user(fitter, init_from, n_gal, d_params)
        elif init_from is None:
            all_init = self._map_warm_start(
                fitter,
                all_init,
                (
                    all_data_orig,
                    all_noise_orig,
                    all_presence_orig,
                    all_redshift_orig,
                    all_line_flux_orig,
                    all_line_err_orig,
                ),
                n_steps=map_init_steps,
                chunk=K,
                thread_redshift=per_galaxy_z,
                thread_line_fluxes=per_galaxy_lines,
                verbose=verbose,
            )
        elif init_from != "prior":
            raise ValueError(
                f"init_from must be None (MAP warm start), 'prior', an array of "
                f"shape (n_gal, n_dim), or per-galaxy parameter dicts; got {init_from!r}"
            )
        if n_pad_extra > 0:
            all_init = jnp.concatenate([all_init, jnp.zeros((n_pad_extra, d_params))], axis=0)
        gal_keys = jax.random.split(jax.random.fold_in(key, 1), n_padded)

        def _run_one(args):
            ini, gk, d, n, p, z, lf, le = args
            return run_one(ini, gk, d, n, p, z, lf, le)

        xs = (
            all_init,
            gal_keys,
            all_data,
            all_noise,
            all_presence,
            all_redshift,
            all_line_flux,
            all_line_err,
        )
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

            # Build the Posterior FIRST, summarizing derived properties needs
            # the model, which only the Posterior carries.
            post_i = Posterior(
                samples=samples_phys,
                params=best_params,
                method=f"CatalogFitter/{method_tag}",
                wall_time_s=time.time() - t0,
                diagnostics={"n_divergent": n_div, "n_samples": n_samples, "n_chains": 1},
                loss_history=None,
                _model=self.model,
            )
            _attach_summaries(post_i, store, percentiles, reducers, properties)
            posteriors.append(post_i)

        wall = time.time() - t0
        if verbose:
            print(f"  Done in {wall:.1f}s ({wall / n_gal:.2f}s/galaxy)")

        # Stack per-galaxy summary blocks; record WHICH levels they hold and
        # report the store that actually happened (see _stack_summaries).
        cat_percentiles, cat_summary, cat_levels, store = _stack_summaries(
            posteriors, store, method_tag, percentiles
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
            percentile_levels=cat_levels,
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

    def _init_from_user(self, fitter, init_from, n_gal, d_params):
        """Convert a caller-supplied starting point into the flat init array.

        Parameters
        ----------
        fitter : Fitter
            Template fitter, for the physical-to-unconstrained transform.
        init_from : array_like or dict or sequence of dict
            An ``(n_gal, n_dim)`` array already in the flat unconstrained space,
            one parameter dict broadcast to every galaxy, or a sequence of
            ``n_gal`` dicts.
        n_gal : int
            Galaxies in the catalog.
        d_params : int
            Flat dimension.

        Returns
        -------
        ndarray, shape (n_gal, d_params)

        Raises
        ------
        ValueError
            If an array has the wrong shape or a sequence the wrong length.
            Broadcasting the wrong count would start galaxies from each other's
            initial conditions, which no diagnostic downstream would catch.
        """
        from tengri.inference._sample_utils import _as_posterior_like

        def _flat(entry):
            # _as_posterior_like accepts a plain dict as well as a posterior and
            # validates the free names (#1854); going straight to
            # _unbounded_from_posterior dies on AttributeError instead.
            return ravel_pytree(
                fitter._unbounded_from_posterior(_as_posterior_like(fitter, entry))
            )[0]

        if isinstance(init_from, dict):
            return jnp.broadcast_to(_flat(init_from), (n_gal, d_params))

        if isinstance(init_from, (list, tuple)):
            if len(init_from) != n_gal:
                raise ValueError(
                    f"init_from has {len(init_from)} entries for {n_gal} galaxies; "
                    "pass one dict per galaxy, or a single dict to share one start"
                )
            return jnp.stack([_flat(d) for d in init_from])

        arr = jnp.asarray(init_from)
        if arr.shape != (n_gal, d_params):
            raise ValueError(
                f"init_from array must have shape ({n_gal}, {d_params}), got {arr.shape}"
            )
        return arr

    def _map_warm_start(
        self,
        fitter,
        all_init,
        per_galaxy,
        *,
        n_steps,
        chunk,
        thread_redshift,
        thread_line_fluxes,
        verbose,
    ):
        """Replace the random per-galaxy starts with independent MAP estimates.

        Runs the same chunked ``lax.map`` shape as the sampler, so the warm start
        costs one more pass of the same width rather than a second memory regime.

        Parameters
        ----------
        fitter : Fitter
            Template fitter for the shared model.
        all_init : ndarray, shape (n_gal, n_dim)
            The random starting points to improve on. Returned unchanged for any
            galaxy whose descent leaves the finite domain.
        per_galaxy : tuple of ndarray
            ``(data, noise, presence, redshift, line_flux_obs, line_flux_err)``,
            each with a leading galaxy axis.
        n_steps : int
            ADAM steps per galaxy.
        chunk : int
            Galaxies per ``lax.map`` step, i.e. the sampler's ``K``.
        thread_redshift, thread_line_fluxes : bool
            Whether the catalog carries per-galaxy redshifts / line fluxes.
        verbose : bool
            Print the wall clock and how many galaxies moved.

        Returns
        -------
        ndarray, shape (n_gal, n_dim)
            The warm starts.

        Notes
        -----
        **JIT-compatible**; the optimizer runs inside ``lax.scan`` under
        ``lax.map``. Optional: falls back to ``all_init`` with a warning if
        ``optax`` is missing, since a warm start is an improvement rather than a
        requirement.
        """
        from tengri.inference.backends.mcmc.catalog import build_catalog_map_init

        try:
            map_init_one = build_catalog_map_init(
                fitter,
                n_steps=n_steps,
                thread_redshift=thread_redshift,
                thread_line_fluxes=thread_line_fluxes,
            )
        except ImportError:  # pragma: no cover - optax is a declared MAP extra
            warnings.warn(
                "init_from='map' needs optax; falling back to random starts. "
                "Install optax, or pass init_from=None to silence this.",
                RuntimeWarning,
                stacklevel=2,
            )
            return all_init

        t0 = time.time()
        warm = jax.lax.map(
            lambda args: map_init_one(*args), (all_init, *per_galaxy), batch_size=chunk
        )
        jax.block_until_ready(warm)
        if verbose:
            moved = int(jnp.sum(jnp.any(jnp.abs(warm - all_init) > 0, axis=1)))
            print(
                f"CatalogFitter MAP warm start: {n_steps} ADAM steps, "
                f"{moved}/{all_init.shape[0]} galaxies moved, {time.time() - t0:.1f}s"
            )
        return warm

    @staticmethod
    def _sharded_vmap(run_one, xs, dev_list):
        """``jax.vmap(run_one)`` over a galaxy axis sharded across ``dev_list``.

        Per-galaxy fits are independent, so this is pure data parallelism with
        no cross-device reduction: the leading (galaxy) axis of every input is
        sharded over the devices and GSPMD distributes the vmapped program,
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
        percentiles: tuple = DEFAULT_PERCENTILES,
        reducers: dict | None = None,
        properties: tuple | None = None,
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
        # needed, the wrap was strictly worse for mixed-shape loops.
        for i, galaxy in enumerate(self.galaxies):
            if verbose:
                print(f"  Galaxy {i + 1}/{self.n_galaxies}...", end="\r", flush=True)
            # Per-galaxy redshift (or any fixed-value override) reaches the forward
            # pass via the #1329 params-override seam, not just the reported params.
            override = {"redshift": galaxy["redshift"]} if "redshift" in galaxy else None
            # Per-galaxy presence mask for heterogeneous catalogs (missing="mask" in ingestion)
            presence = galaxy.get("presence", None)
            fitter_i = Fitter(
                self.model,
                galaxy["flux_obs"],
                galaxy["noise"],
                data_type=self.data_type,
                presence=presence,
                line_flux_data=self._galaxy_line_fluxes(galaxy),
                cache=self.cache,
                params_override=override,
            )
            post_i = fitter_i.run(method, key=keys[i], verbose=False, **kwargs)

            _attach_summaries(post_i, store, percentiles, reducers, properties)
            posteriors.append(post_i)

        wall = time.time() - t0
        if verbose:
            per = wall / self.n_galaxies
            print(f"  {self.n_galaxies} galaxies done in {wall:.1f}s ({per:.2f}s/galaxy)")

        # Stack per-galaxy summary blocks; record WHICH levels they hold and
        # report the store that actually happened (see _stack_summaries).
        cat_percentiles, cat_summary, cat_levels, store = _stack_summaries(
            posteriors, store, method, percentiles
        )

        return CatalogPosterior(
            posteriors=posteriors,
            method=method,
            wall_time_s=wall,
            n_galaxies=self.n_galaxies,
            percentiles=cat_percentiles,
            summary=cat_summary,
            store=store,
            percentile_levels=cat_levels,
        )


class CatalogFitter(_CatalogFitterOriginal):
    """Deprecated public alias of the catalog engine, use :class:`tengri.Catalog`.

    .. deprecated::
        ``Catalog`` is the one taught catalog noun (#1317, spec decision 6):
        ``Catalog(fwd, table, flux_unit="mJy", redshift_col="z").fit(method="map",
        key=key)``. This class keeps working per the no-removal policy (spec
        §13) but warns at construction.

    Notes
    -----
    ``Catalog`` itself constructs :class:`_CatalogFitterOriginal` directly, so
    the internal path stays warning-free, only user-typed ``CatalogFitter``
    warns. (A previous module ``__getattr__`` hook warned on direct module
    imports, but the ``tengri.CatalogFitter`` re-export bypassed it, so the
    taught name never warned, #1369.)
    """

    def __init__(self, model, galaxies, data_type="photometry", *, approx="auto"):
        warnings.warn(
            "CatalogFitter is deprecated: use tengri.Catalog, "
            "Catalog(fwd, table, flux_unit=..., redshift_col=...).fit(method=..., "
            "key=...). CatalogFitter keeps working but is no longer taught. "
            "See #1369.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(model, galaxies, data_type, approx=approx)
