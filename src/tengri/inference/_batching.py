# SPDX-License-Identifier: BSD-3-Clause
r"""How many galaxies a single dispatch carries (#1189).

**A code path that processes one galaxy per dispatch is a bug, not a slow
path.** ``lax.map(..., batch_size=K)`` with ``K = 1`` is the anti-pattern: it
asks the accelerator to do one galaxy's work per launch, so launch overhead and
sequential dependencies dominate and there is nothing worth dispatching.

Width was the only lever that moved anything in the Metal/CPU measurements
behind #1189, a 256-step sequential scan went 0.002x -> 0.339x of CPU between
width 1 and width 65536 (**170x relative**), and a gather/interp kernel
0.052x -> 1.125x (**21x**). Op count, fusion, async dispatch and
``JAX_MPS_ASYNC_DISPATCH`` moved nothing measurable. Because the mechanism is a
``vmap``, the same source runs on CPU, Metal, CUDA and ROCm with no custom
kernels and no per-backend branches.

Five entry points defaulted to ``forward_chunk_size=1``, both
``PopulationFitter`` VI paths, ``CatalogFitter.run``, ``catalog.py``, and
``n_chains`` on the MCMC side. This module is the single place that decides,
so a sixth cannot quietly re-introduce the default.

Choosing K
----------
K is derived from a **memory budget the caller can pin**, not from the device's
free memory and not by auto-tuning. That is deliberate, and it is a
reproducibility-over-throughput trade:

* querying device memory adapts, but makes a run's numerics depend on what else
  happened to be resident, and tengri has a documented NUTS warmup OOM history
  (20+ GB at D ~ 8 with a dense mass matrix) that a co-tenant process can
  re-trigger;
* auto-tuning on the first call is fastest in steady state but adds a warmup,
  is non-deterministic, and probing upward until something OOMs is hostile on a
  shared machine;
* a pinned budget gives the same K for the same ``(budget, n_data, n_chains)``
  on every machine, which is what CI needs.

**The budget is divided across ``K * n_chains``, not K alone.** Chains are a
second real ``vmap`` axis (``_vmap_chains``), they multiply the live activation
set exactly as K does, and a helper that ignored them would hand back a K that
OOMs the moment somebody asks for four chains.

The per-galaxy cost model here is deliberately coarse, activations are not
just the data array, which is precisely *why* the budget is a knob rather than
a computed truth. Pin it up when your model is small, down when it is not.
"""

from __future__ import annotations

import os

__all__ = [
    "AUTO",
    "DEFAULT_MEMORY_BUDGET_GB",
    "chunking_was_requested",
    "resolve_forward_chunk_size",
]

#: Sentinel: derive K from the memory budget rather than taking it literally.
AUTO = "auto"

#: Conservative default budget [GB] for the live per-dispatch activation set.
#: Small enough to be safe on a laptop, which is where a novice meets this.
DEFAULT_MEMORY_BUDGET_GB = 2.0

#: Pin the budget without touching call sites.
_ENV_BUDGET = "TENGRI_FORWARD_MEMORY_BUDGET_GB"

#: Live activation per galaxy, per dispatch, as ``_BASE + n_data * _PER_DATUM``.
#:
#: The base term dominates and the ``n_data`` term is a rounding error on it:
#: a galaxy's forward pass is carrying the SSP contraction (~15 metallicities x
#: ~93 ages x ~6000 wavelengths), dust and nebular intermediates, and the
#: autodiff tape when this runs under ``grad``, none of which scale with the
#: number of *bands*. Modelling the cost as proportional to ``n_data`` alone was
#: the first version of this file and it made the knob **inert**: it put K at
#: ~10,000 for a 50-band fit, so ``min(k, n_gal)`` always bound and neither the
#: budget nor ``n_chains`` could move the answer.
#:
#: These are order-of-magnitude allowances, not measurements, which is exactly
#: why the budget is a knob rather than a computed truth. They are the two
#: numbers to revisit if the default K turns out wrong on real models.
_BASE_BYTES_PER_GALAXY = 32 * 1024**2
_BYTES_PER_DATUM = 4096


def _budget_gb(explicit: float | None) -> float:
    """Resolve the memory budget: explicit argument, then env, then default."""
    if explicit is not None:
        return float(explicit)
    raw = os.environ.get(_ENV_BUDGET)
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return DEFAULT_MEMORY_BUDGET_GB


def resolve_forward_chunk_size(
    requested: int | str | None,
    *,
    n_gal: int,
    n_data_per_gal: int | None,
    n_chains: int = 1,
    homogeneous: bool = True,
    memory_budget_gb: float | None = None,
) -> int:
    """Return K, galaxies per ``lax.map`` dispatch.

    Parameters
    ----------
    requested : int, ``"auto"``, or None
        An explicit ``int >= 1`` is honored exactly, so a caller that has
        measured its own machine always wins and stays reproducible.
        ``"auto"`` / ``None`` derives K from the budget.
    n_gal : int
        Galaxies in the fit. K is never larger than this.
    n_data_per_gal : int or None
        Data points per galaxy. ``None`` (unknown) forces ``K = 1``: guessing a
        width from an unknown shape is how an OOM gets shipped.
    n_chains : int, default 1
        MCMC chains, the second ``vmap`` axis. The budget is divided across
        ``K * n_chains``.
    homogeneous : bool, default True
        Whether every galaxy has the same ``n_data``. ``K > 1`` requires it,
        the callers raise otherwise, so a heterogeneous catalog resolves to
        ``K = 1`` rather than turning a working fit into an error.
    memory_budget_gb : float, optional
        Overrides the ``TENGRI_FORWARD_MEMORY_BUDGET_GB`` env var and the
        default.

    Returns
    -------
    int
        ``K >= 1``, and ``<= n_gal``.

    Notes
    -----
    Pure Python, evaluated at build time, never inside a trace, so K is a
    static Python int and ``lax.map``'s batch size stays a compile-time
    constant.
    """
    if isinstance(requested, str):
        if requested != AUTO:
            raise ValueError(f"forward_chunk_size must be an int or {AUTO!r}, got {requested!r}")
    elif requested is not None:
        k = int(requested)
        if k < 1:
            raise ValueError(f"forward_chunk_size must be >= 1, got {k}")
        return min(k, max(1, int(n_gal)))

    n_gal = max(1, int(n_gal))
    if not homogeneous or not n_data_per_gal or n_data_per_gal <= 0:
        return 1

    chains = max(1, int(n_chains))
    budget_bytes = _budget_gb(memory_budget_gb) * 1024**3
    per_galaxy = (_BASE_BYTES_PER_GALAXY + float(n_data_per_gal) * _BYTES_PER_DATUM) * chains
    k = int(budget_bytes // per_galaxy) if per_galaxy > 0 else 1
    return max(1, min(k, n_gal))


def chunking_was_requested(forward_chunk_size) -> bool:
    """True only when the caller explicitly asked for more than one galaxy per dispatch.

    A method that cannot chunk needs to tell a caller their request is being
    ignored, but only if there *was* a request. That question must be asked of
    the **unresolved** argument, which is why it is a predicate here rather than
    a comparison at the call site.

    The obvious spelling, ``forward_chunk_size != 1``, was correct only while 1
    was the default. When #1189 changed the default to :data:`AUTO` every caller
    who passed nothing began satisfying it, so a plain ``run("map")`` warned that
    a setting the user never set was being ignored. A literal that encodes a
    default is a comparison against a value that can move.

    Parameters
    ----------
    forward_chunk_size : int, ``"auto"``, or None
        The argument exactly as the caller supplied it, before
        :func:`resolve_forward_chunk_size`.

    Returns
    -------
    bool
        ``True`` for an explicit ``int >= 2``. ``False`` for :data:`AUTO`,
        ``None``, and an explicit ``1``, none of which ask for chunking, so
        none of which have anything to ignore.

    Notes
    -----
    Pure Python, no JAX. Validation belongs to
    :func:`resolve_forward_chunk_size`: an unrecognized string returns ``False``
    here and raises there, with the better message.
    """
    if forward_chunk_size is None or isinstance(forward_chunk_size, str):
        return False
    return int(forward_chunk_size) > 1
