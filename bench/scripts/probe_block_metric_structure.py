#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Does a block-diagonal metric recover the conditioning a full one does? (#2166)

A block-structured mass matrix sits between the two options tengri ships: a
diagonal metric (cheap, blind to correlation) and a full dense one (represents
everything, ``O(D^2)``). The middle term gives each correlated sub-group its own
dense block and leaves everything else diagonal, costing ``O(sum_i d_i^2)``.

Whether that is worth building is a **structural** question about one matrix, so
it is answered here before any sampler runs. ``preconditioning.py`` already
builds the analytic metric ``G`` at the expansion point. For a candidate block
layout, zero every entry of ``G`` outside the blocks to get ``G_block`` -- a
principal-submatrix extraction, so ``G_block`` is positive definite whenever
``G`` is -- and ask how much of the whitening survives:

* ``cond(G)`` -- the raw problem;
* ``cond`` after whitening with ``G`` itself -- the ceiling, exactly 1;
* ``cond`` after whitening with ``diag(G)`` -- what a diagonal metric buys, the
  control the block layout has to beat;
* ``cond`` after whitening with ``G_block`` -- the number that decides the design;
* ``cond`` after whitening with a **rank-k correction to a diagonal** -- the
  other structured middle term, which tengri already has a sampler for
  (``_hmc_low_rank_full_scan``). A block metric that a low-rank one dominates on
  the same geometry is not worth building, so the two are scored side by side.

Storage is reported with each arm, in matrix entries: diagonal ``D``, block
``D + sum_i (d_i^2 - d_i)``, rank-``k`` ``D + kD + k``, dense ``D^2``. That is the
whole trade -- conditioning bought per entry stored.

**Two conventions, both reported, because they do not agree.** What window
adaptation actually estimates is the *inverse* mass matrix, i.e. a covariance
``Sigma``, not the metric. So the arm that matches an implementation applies the
structure to ``Sigma = G^-1`` and uses ``M = structure(Sigma)^-1``; the column
labelled ``prec=struct(G)`` applies the same structure to ``G`` directly. They
coincide for the diagonal-free layouts and diverge sharply for low-rank, because
the degenerate directions that dominate ``Sigma`` are the *small* eigenvalues of
``G``. The covariance column is the one that describes a shippable adaptation.

Whitening with ``M`` means the change of variables ``x = M^{-1/2} z``, under
which the curvature the sampler faces is ``M^{-1/2} G M^{-1/2}``. Reported as a
recovered fraction of the log-conditioning gap,
``1 - log10(cond_whitened) / log10(cond_raw)``: 1.0 is a perfect metric, 0.0 is
no metric at all.

The prefix layout (``sfh_*``, ``dust_*``, ``met_*``, ``agn_*``, ... --
NAMING_CONTRACT section 3.2's domain prefixes) is an **assumption** about where
correlations live. This probe does not stop at scoring it: it also reports where
the off-diagonal mass of ``G`` actually sits, and derives layouts by thresholding
the metric's own correlation form, so a layout that captures materially more than
the prefix guess shows up as a number rather than a hunch.

No sampling, no seeds, no wall clocks. Every quantity here is a deterministic
function of one Hessian, so it is contention-immune and cheap.

Usage::

    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/probe_block_metric_structure.py
    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/probe_block_metric_structure.py 05 ctl-jwst
"""

from __future__ import annotations

import os
import sys

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_notebook_sampler import NOTEBOOKS

import tengri
from tengri import Fitter
from tengri.analysis.mock import generate_mock
from tengri.inference._sample_utils import _maybe_map_init
from tengri.inference.backends.mcmc._shared import _get_flat_logdensity
from tengri.inference.preconditioning import negative_hessian_metric

#: Fixtures probed by default: a D=8 photometry model (the primary gate fixture,
#: where a dense metric is affordable and all three options can be compared
#: directly), a D=9 nonparametric-SFH control whose free parameters are almost
#: all one prefix group, and the D=74 field fixture -- near-low-rank, and so the
#: most favorable geometry a block metric could hope for.
DEFAULT_FIXTURES = ("05", "ctl-jwst", "stoch-field")

#: Correlation thresholds swept when deriving a layout from the metric itself.
#: Connected components of ``|R_ij| >= t`` on the correlation form of ``G``.
DERIVED_THRESHOLDS = (0.9, 0.7, 0.5, 0.3)

#: Off-diagonal pairs listed per fixture, largest ``|R_ij|`` first. Enough to
#: show whether the dominant degeneracies sit inside prefix groups or across
#: them; the cross-group case is the failure mode a block metric cannot see.
N_TOP_PAIRS = 8

#: Ranks swept for the low-rank arm. 10 is the rank tengri's low-rank sampler
#: runs at by default, so it is the number a block layout actually competes with;
#: 3 and 20 bracket it.
LOW_RANK_K = (3, 10, 20)

#: Prefix groups merged into single blocks, tested alongside the plain per-prefix
#: layout. The classic SED stiffness is attenuation-mass-age, which runs *across*
#: ``dust_*`` and ``sfh_*``, and a block metric cannot see a correlation that
#: crosses two of its blocks. Merging is the cheapest repair, and it has to be
#: tried before per-prefix blocks are called a failure.
MERGED_LAYOUTS = (("sfh", "dust"), ("sfh", "dust", "met"))


def coordinate_names(init_params) -> list[str]:
    """Per-coordinate names for the flat latent vector.

    ``_get_flat_logdensity`` flattens ``init_params`` with
    ``jax.flatten_util.ravel_pytree``, which walks leaves in ``tree_flatten``
    order. Walking the same tree with paths and repeating each leaf's key by its
    size therefore reproduces the flat layout exactly, including the unnamed
    field-latent vectors that make ``n_latent`` exceed ``n_free`` (#1408).

    Parameters
    ----------
    init_params : pytree
        Unbounded initial parameters, as handed to ``_get_flat_logdensity``.

    Returns
    -------
    names : list of str
        One name per flat coordinate. A leaf of size ``k > 1`` contributes
        ``k`` entries suffixed ``[0] .. [k-1]``.
    """
    leaves, _ = jax.tree_util.tree_flatten_with_path(init_params)
    names: list[str] = []
    for path, leaf in leaves:
        key = ".".join(str(getattr(p, "key", p)) for p in path)
        size = int(np.asarray(leaf).size)
        if size == 1:
            names.append(key)
        else:
            names.extend(f"{key}[{i}]" for i in range(size))
    return names


def domain_prefix(name: str) -> str | None:
    """Domain group of a coordinate name, or ``None`` when it has no group.

    NAMING_CONTRACT section 3.2 makes every free parameter carry a domain prefix
    (``sfh_``, ``met_``, ``dust_``, ``neb_``, ``agn_``, ``eline_``, ``noise_``,
    ``radio_``, ``xray_``, ``shock_``, ``chem_``, ``igm_``, ``dla_``,
    ``spatial_``) or be exactly ``redshift``. The multi-population namespace
    (``<population>.<param>``, ADR-0012) is kept in the group key, so two
    populations' ``sfh_*`` parameters are different groups -- they are different
    physical objects and there is no reason to expect their latents to correlate.

    Parameters
    ----------
    name : str
        Coordinate name from :func:`coordinate_names`.

    Returns
    -------
    prefix : str or None
        Group key, or ``None`` for a name that carries no domain prefix
        (``redshift`` among them, which correlates with everything and is
        therefore a job for a dense metric rather than a block).
    """
    bare = name.split(".", 1)[-1].split("[", 1)[0]
    namespace = name[: len(name) - len(name.split(".", 1)[-1])]
    head, sep, _rest = bare.partition("_")
    if not sep:
        return None
    return namespace + head


def prefix_blocks(names: list[str]) -> list[list[int]]:
    """Block layout implied by the domain prefixes, blocks of size >= 2 only.

    Parameters
    ----------
    names : list of str
        Per-coordinate names, in flat order.

    Returns
    -------
    blocks : list of list of int
        Index groups. A group of one is a diagonal entry, not a block, so it is
        dropped.
    """
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        prefix = domain_prefix(name)
        if prefix is not None:
            groups.setdefault(prefix, []).append(i)
    return [idx for idx in groups.values() if len(idx) >= 2]


def merged_prefix_blocks(names: list[str], merge: tuple[str, ...]) -> list[list[int]]:
    """Prefix layout with the named domain groups fused into one block.

    Parameters
    ----------
    names : list of str
        Per-coordinate names, in flat order.
    merge : tuple of str
        Domain prefixes to fuse (e.g. ``("sfh", "dust")``). Groups not named
        here keep their own block.

    Returns
    -------
    blocks : list of list of int
        Index groups of size >= 2, indices sorted.
    """
    groups: dict[str, list[int]] = {}
    fused = "+".join(merge)
    for i, name in enumerate(names):
        prefix = domain_prefix(name)
        if prefix is None:
            continue
        key = fused if prefix.split(".")[-1] in merge else prefix
        groups.setdefault(key, []).append(i)
    return [sorted(idx) for idx in groups.values() if len(idx) >= 2]


def low_rank_approximation(matrix: np.ndarray, rank: int) -> np.ndarray:
    """Best rank-``k`` correction to a diagonal of a symmetric positive matrix.

    The structured alternative to a block layout: whiten by a diagonal, then
    correct the ``rank`` directions in which the whitened curvature is furthest
    from isotropic. In the diagonally-scaled basis ``R = diag(G)^-1/2 G
    diag(G)^-1/2`` this is the truncated eigendecomposition of ``R`` keeping the
    ``rank`` eigenvalues furthest from 1 in log, with the rest set to 1 --
    the same ``I + U (L - I) U^T`` shape a low-rank adaptation fits, evaluated
    here at its best possible estimate rather than from noisy draws.

    Parameters
    ----------
    matrix : ndarray, shape (D, D)
        Symmetric positive-definite matrix to approximate.
    rank : int
        Number of corrected directions ``k``. ``k >= D`` reproduces ``matrix``.

    Returns
    -------
    approximation : ndarray, shape (D, D)
        Symmetric positive-definite rank-``k``-plus-diagonal approximation.
    """
    scales = np.sqrt(np.diag(matrix))
    correlation = matrix / np.outer(scales, scales)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (correlation + correlation.T))
    keep = np.argsort(-np.abs(np.log(np.maximum(eigenvalues, 1e-300))))[:rank]
    corrected = np.ones_like(eigenvalues)
    corrected[keep] = eigenvalues[keep]
    scaled = (eigenvectors * corrected) @ eigenvectors.T
    return scaled * np.outer(scales, scales)


def threshold_blocks(correlation: np.ndarray, threshold: float) -> list[list[int]]:
    """Blocks derived from the metric itself: components of ``|R| >= threshold``.

    Parameters
    ----------
    correlation : ndarray, shape (D, D)
        Correlation form of the metric, ``diag(G)^-1/2 G diag(G)^-1/2``.
    threshold : float
        Edge threshold on ``|R_ij|`` [dimensionless].

    Returns
    -------
    blocks : list of list of int
        Connected components of size >= 2, indices sorted.
    """
    dim = correlation.shape[0]
    adjacency = np.abs(correlation) >= threshold
    np.fill_diagonal(adjacency, False)
    seen = np.zeros(dim, dtype=bool)
    blocks: list[list[int]] = []
    for start in range(dim):
        if seen[start]:
            continue
        stack, component = [start], []
        seen[start] = True
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in np.flatnonzero(adjacency[node]):
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(int(neighbor))
        if len(component) >= 2:
            blocks.append(sorted(component))
    return blocks


#: A group earns a dense block only if its own internal off-diagonal mass is at
#: least this fraction of its submatrix's Frobenius norm. Below it, the group's
#: coordinates are effectively uncorrelated among themselves and a ``d_i x d_i``
#: block spends ``d_i^2`` entries representing structure that is not there.
EARNED_INTERNAL_FRACTION = 0.25

#: ...and its internal off-diagonal mass must be at least this multiple of its
#: coupling to everything outside it. A group that talks to the rest of the model
#: more than to itself is a cross-block correlation wearing a group's name, and a
#: block-diagonal matrix cannot represent it whichever side of the partition it
#: is put on.
EARNED_INTERNAL_OVER_EXTERNAL = 1.0


def group_statistics(correlation: np.ndarray, index: list[int]) -> tuple[float, float, float]:
    """Per-group verdict inputs: internal mass, external coupling, submatrix cond.

    Parameters
    ----------
    correlation : ndarray, shape (D, D)
        Correlation form of the matrix being structured.
    index : list of int
        Coordinates of one candidate group.

    Returns
    -------
    internal_fraction : float
        Off-diagonal Frobenius mass inside the group's own submatrix, as a
        fraction of that submatrix's total Frobenius norm [dimensionless]. This
        is what a dense block would buy.
    internal_over_external : float
        Ratio of that internal off-diagonal mass to the group's coupling to every
        coordinate outside it [dimensionless]. Below 1 the group's correlations
        mostly leave the group, which no block can capture.
    submatrix_condition : float
        Condition number of the group's own submatrix [dimensionless].
    """
    rows = np.asarray(index)
    block = correlation[np.ix_(rows, rows)]
    offdiag = block.copy()
    np.fill_diagonal(offdiag, 0.0)
    internal = float(np.linalg.norm(offdiag))
    total = float(np.linalg.norm(block))
    outside = np.setdiff1d(np.arange(correlation.shape[0]), rows)
    external = float(np.linalg.norm(correlation[np.ix_(rows, outside)])) if outside.size else 0.0
    return (
        internal / total if total else float("nan"),
        internal / external if external else float("inf"),
        float(np.linalg.cond(block)),
    )


def earned_blocks(
    correlation: np.ndarray, groups: dict[str, list[int]]
) -> tuple[list[list[int]], list[str]]:
    """Keep only the groups whose own numbers justify a dense block.

    Returns the surviving index groups and the names of the groups kept, so the
    layout can be reported as an explicit per-group decision rather than a rule.
    """
    kept_index: list[list[int]] = []
    kept_names: list[str] = []
    for name, index in groups.items():
        if len(index) < 2:
            continue
        internal, ratio, _cond = group_statistics(correlation, index)
        if internal >= EARNED_INTERNAL_FRACTION and ratio >= EARNED_INTERNAL_OVER_EXTERNAL:
            kept_index.append(sorted(index))
            kept_names.append(name)
    return kept_index, kept_names


def subsystem_key(name: str) -> str | None:
    """Group key for the physical *structure* a coordinate belongs to.

    The domain prefix alone groups by parameter **kind**: every ``sfh_*`` name
    lands in one group whether it describes the mean star-formation history or
    the hyperparameters of a stochastic field sitting on top of it. Keeping the
    first two tokens of the bare name instead groups by the structure being
    described (``sfh_dpl`` and ``sfh_field`` become separate), which is the
    distinction worth testing: if correlations follow structures rather than
    kinds, the second layout captures more off-diagonal mass at the same or
    lower cost.

    Where a model carries only one structure per domain the two layouts are
    identical by construction, and the comparison is uninformative rather than
    confirmatory -- the caller must say which case it is in.

    Parameters
    ----------
    name : str
        Coordinate name from :func:`coordinate_names`.

    Returns
    -------
    key : str or None
        Structure key, or ``None`` for a name with no domain prefix.
    """
    prefix = domain_prefix(name)
    if prefix is None:
        return None
    bare = name.split(".", 1)[-1].split("[", 1)[0]
    tokens = bare.split("_")
    return prefix if len(tokens) < 3 else f"{prefix}_{tokens[1]}"


def subsystem_named_groups(names: list[str]) -> dict[str, list[int]]:
    """Candidate groups under the by-structure key."""
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        key = subsystem_key(name)
        if key is not None:
            groups.setdefault(key, []).append(i)
    return groups


def cross_group_mass(correlation: np.ndarray, left: list[int], right: list[int]) -> float:
    """Frobenius mass of the correlation block between two disjoint groups."""
    return float(np.linalg.norm(correlation[np.ix_(np.asarray(left), np.asarray(right))]))


def coupling_profile(
    correlation: np.ndarray, index: int, groups: dict[str, list[int]]
) -> list[tuple[str, float]]:
    """Where one coordinate's off-diagonal mass goes, by group.

    A row-wise question about the metric, and the one that decides whether a
    per-subsystem layout has a home for a parameter that scales everything. If a
    coordinate's coupling concentrates in one group it belongs in that group's
    block; if it spreads across several, no partition holds it.

    Parameters
    ----------
    correlation : ndarray, shape (D, D)
        Correlation form of the metric.
    index : int
        The coordinate to profile.
    groups : dict of str to list of int
        Candidate groups, keyed by name. Coordinates in no group are pooled
        under ``"(ungrouped)"``.

    Returns
    -------
    shares : list of (str, float)
        Group name and the fraction of this coordinate's off-diagonal Frobenius
        mass that lands in it [dimensionless], largest first. The coordinate's
        own group is included; its own diagonal entry is excluded.
    """
    row = np.abs(correlation[index]).copy()
    row[index] = 0.0
    total = float(np.linalg.norm(row))
    if total == 0.0:
        return []
    assigned = np.zeros(correlation.shape[0], dtype=bool)
    shares: list[tuple[str, float]] = []
    for name, members in groups.items():
        rows = np.asarray(members)
        assigned[rows] = True
        shares.append((name, float(np.linalg.norm(row[rows])) / total))
    leftover = np.flatnonzero(~assigned)
    if leftover.size:
        shares.append(("(ungrouped)", float(np.linalg.norm(row[leftover])) / total))
    return sorted((s for s in shares if s[1] > 0.0), key=lambda kv: -kv[1])


def named_groups(names: list[str]) -> dict[str, list[int]]:
    """All candidate prefix groups, including the size-1 ones a block cannot use."""
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        prefix = domain_prefix(name)
        if prefix is not None:
            groups.setdefault(prefix, []).append(i)
    return groups


def block_entries(dim: int, blocks: list[list[int]]) -> int:
    """Matrix entries a block layout stores: ``D + sum_i (d_i^2 - d_i)``."""
    return dim + sum(len(b) * len(b) - len(b) for b in blocks)


def block_mask(dim: int, blocks: list[list[int]]) -> np.ndarray:
    """Boolean mask of the entries a block layout keeps (diagonal always kept)."""
    mask = np.eye(dim, dtype=bool)
    for idx in blocks:
        rows = np.asarray(idx)
        mask[np.ix_(rows, rows)] = True
    return mask


def whitened_condition(metric: np.ndarray, whitener: np.ndarray) -> float:
    """Condition number of ``metric`` after whitening with ``whitener``.

    Parameters
    ----------
    metric : ndarray, shape (D, D)
        The true metric ``G``, symmetric positive definite.
    whitener : ndarray, shape (D, D)
        The metric actually used to build the change of variables, ``M``.
        ``M = G`` gives exactly 1; ``M = diag(G)`` gives the diagonal control.

    Returns
    -------
    condition : float
        ``cond(M^-1/2 G M^-1/2)`` [dimensionless].
    """
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (whitener + whitener.T))
    if eigenvalues.min() <= 0.0:
        return float("nan")
    inverse_sqrt = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    whitened = inverse_sqrt @ metric @ inverse_sqrt
    spectrum = np.linalg.eigvalsh(0.5 * (whitened + whitened.T))
    return float(spectrum.max() / spectrum.min())


def recovered_fraction(cond_raw: float, cond_whitened: float) -> float:
    """Fraction of the log-conditioning gap a metric closes.

    ``1.0`` is the full-metric ceiling (``cond`` 1); ``0.0`` is no improvement.
    Negative means the metric made the geometry worse than it found it.
    """
    if cond_raw <= 1.0:
        return float("nan")
    return 1.0 - np.log10(max(cond_whitened, 1.0)) / np.log10(cond_raw)


def offdiagonal_mass(matrix: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of off-diagonal Frobenius mass the mask keeps [dimensionless]."""
    offdiag = matrix.copy()
    np.fill_diagonal(offdiag, 0.0)
    total = np.linalg.norm(offdiag)
    if total == 0.0:
        return float("nan")
    kept = offdiag.copy()
    kept[~mask] = 0.0
    return float(np.linalg.norm(kept) / total)


def describe(blocks: list[list[int]], names: list[str], dim: int) -> str:
    """One-line summary of a layout: block count, sizes, and stored entries."""
    if not blocks:
        return f"no blocks (pure diagonal); stored entries {dim} of {dim * dim}"
    sizes = sorted((len(b) for b in blocks), reverse=True)
    stored = dim + sum(d * d - d for d in sizes)
    groups = ", ".join(f"{_group_label(b, names)}:{len(b)}" for b in blocks)
    return f"{len(blocks)} blocks [{groups}]; stored entries {stored} of {dim * dim}"


def _group_label(block: list[int], names: list[str]) -> str:
    """Shortest label that identifies a block: its shared prefix, or its span."""
    prefixes = {domain_prefix(names[i]) for i in block}
    if len(prefixes) == 1:
        only = prefixes.pop()
        if only is not None:
            return only
    return f"{block[0]}-{block[-1]}"


def probe(fixture: str) -> None:
    """Build one fixture, form its metric, and score every candidate layout."""
    cfg = NOTEBOOKS[fixture]
    ssp = tengri.load_ssp(cfg.get("ssp", "fsps_prsc_miles_chabrier"), download=True)
    sed = cfg["build"](ssp)

    k_truth, k_mock, k_fit = jax.random.split(jax.random.PRNGKey(cfg["seed"]), 3)
    mock = generate_mock(sed, sed.spec.sample(k_truth), key=k_mock, snr=cfg["snr"])
    fitter = Fitter(
        sed,
        np.asarray(mock["flux_obs"]),
        np.asarray(mock["noise"]),
        data_type="photometry",
    )
    init_params, _ = _maybe_map_init(fitter, k_fit, None, False)
    log_p2, _unravel, init_flat, data_args = _get_flat_logdensity(fitter, init_params)
    names = coordinate_names(init_params)
    metric = np.asarray(negative_hessian_metric(log_p2, init_flat, data_args), dtype=np.float64)
    metric = 0.5 * (metric + metric.T)
    dim = metric.shape[0]
    if len(names) != dim:
        names = [f"x[{i}]" for i in range(dim)]

    scales = np.sqrt(np.diag(metric))
    correlation = metric / np.outer(scales, scales)

    covariance = np.linalg.inv(metric)
    covariance = 0.5 * (covariance + covariance.T)

    cond_raw = float(np.linalg.cond(metric))

    def row(label: str, structured_cov, structured_metric, entries: int, extra: str = "") -> None:
        """One arm: conditioning under both conventions, with what it stores."""
        cond_cov = (
            whitened_condition(metric, np.linalg.inv(structured_cov))
            if structured_cov is not None
            else float("nan")
        )
        cond_prec = (
            whitened_condition(metric, structured_metric)
            if structured_metric is not None
            else float("nan")
        )
        print(
            f"  {label:<26} {cond_cov:>11.4g} {recovered_fraction(cond_raw, cond_cov):>7.3f}"
            f" {cond_prec:>11.4g} {recovered_fraction(cond_raw, cond_prec):>7.3f}"
            f" {entries:>8}  {extra}"
        )

    print(f"\n=== {fixture}  D = {dim} ===")
    print(f"  cond(G) raw {cond_raw:.4g}")
    print(
        "  arm                          IMM=cov(struct)     prec=struct(G)"
        "   entries  off-diagonal mass kept"
    )
    print(f"  {'':<26} {'cond':>11} {'recov':>7} {'cond':>11} {'recov':>7} {'':>8}")
    row("full dense (ceiling)", covariance, metric, dim * dim)
    row("diagonal (control)", np.diag(np.diag(covariance)), np.diag(np.diag(metric)), dim)

    for rank in LOW_RANK_K:
        if rank >= dim:
            continue
        row(
            f"low-rank k={rank}",
            low_rank_approximation(covariance, rank),
            low_rank_approximation(metric, rank),
            dim + rank * dim + rank,
        )

    groups = named_groups(names)
    earned_index, earned_names = earned_blocks(correlation, groups)

    layouts: list[tuple[str, list[list[int]]]] = [("prefix", prefix_blocks(names))]
    layouts.append((f"earned [{', '.join(earned_names) or 'none'}]", earned_index))
    for merge in MERGED_LAYOUTS:
        layouts.append((f"prefix {'+'.join(merge)}", merged_prefix_blocks(names, merge)))
    for threshold in DERIVED_THRESHOLDS:
        layouts.append((f"derived |R|>={threshold}", threshold_blocks(correlation, threshold)))

    for label, blocks in layouts:
        mask = block_mask(dim, blocks)
        row(
            f"block {label}",
            np.where(mask, covariance, 0.0),
            np.where(mask, metric, 0.0),
            block_entries(dim, blocks),
            f"raw {offdiagonal_mass(metric, mask):.3f} scaled "
            f"{offdiagonal_mass(correlation, mask):.3f}",
        )
        print(f"      {describe(blocks, names, dim)}")

    def per_group_table(label: str, candidate: dict[str, list[int]]) -> None:
        """Print the earn-a-block numbers for one candidate grouping."""
        print(f"  per-group verdict, {label} (correlation form of G):")
        print(
            f"      {'group':<14} {'size':>4} {'internal':>9} {'int/ext':>9}"
            f" {'cond':>10}  {'entries':>7}  earns a block?"
        )
        for name, index in sorted(candidate.items(), key=lambda kv: -len(kv[1])):
            internal, ratio, cond_group = group_statistics(correlation, index)
            if len(index) < 2:
                verdict = "no (single parameter, diagonal by definition)"
            elif internal < EARNED_INTERNAL_FRACTION:
                verdict = f"no (internal mass < {EARNED_INTERNAL_FRACTION})"
            elif ratio < EARNED_INTERNAL_OVER_EXTERNAL:
                verdict = "no (couples outward more than inward)"
            else:
                verdict = "YES"
            print(
                f"      {name:<14} {len(index):>4} {internal:>9.3f} {ratio:>9.3f}"
                f" {cond_group:>10.4g}  {len(index) ** 2 - len(index):>7}  {verdict}"
            )

    per_group_table("by parameter kind (domain prefix)", groups)
    per_group_table("by physical structure", subsystem_named_groups(names))

    structure_groups = subsystem_named_groups(names)
    structure_index, structure_names = earned_blocks(correlation, structure_groups)
    same_layout = sorted(map(sorted, structure_index)) == sorted(map(sorted, earned_index))
    print(
        f"  by-structure grouping: {len(structure_groups)} candidate groups"
        f" -> earned [{', '.join(structure_names) or 'none'}]"
        f"{'  (identical to by-kind here)' if same_layout else '  (DIFFERS from by-kind)'}"
    )
    if not same_layout:
        mask = block_mask(dim, structure_index)
        cond_structure = whitened_condition(metric, np.linalg.inv(np.where(mask, covariance, 0.0)))
        print(
            f"      cond {cond_structure:.4g}"
            f"   recovered {recovered_fraction(cond_raw, cond_structure):.3f}"
            f"   entries {block_entries(dim, structure_index)}"
            f"   offdiag kept scaled {offdiagonal_mass(correlation, mask):.3f}"
        )

    print("  cross-group coupling mass (Frobenius, correlation form):")
    keys = sorted(groups, key=lambda k: -len(groups[k]))
    for a in keys:
        internal_a, _ratio, _cond = group_statistics(correlation, groups[a])
        block = correlation[np.ix_(np.asarray(groups[a]), np.asarray(groups[a]))].copy()
        np.fill_diagonal(block, 0.0)
        own = float(np.linalg.norm(block))
        others = "  ".join(
            f"{b} {cross_group_mass(correlation, groups[a], groups[b]):.3f}"
            for b in keys
            if b != a
        )
        print(f"      {a:<10} internal {own:.3f} (frac {internal_a:.3f})  |  vs  {others}")

    print("  where each cross-group coupler's off-diagonal mass goes:")
    member_of = {i: name for name, index in groups.items() for i in index}
    for i in range(dim):
        shares = coupling_profile(correlation, i, groups)
        own = member_of.get(i)
        own_share = next((v for k, v in shares if k == own), 0.0)
        is_coupler = own_share < 0.5 or "total_mass" in names[i] or "norm" in names[i]
        if not is_coupler or not shares:
            continue
        spread = "  ".join(f"{k} {v:.2f}" for k, v in shares[:4])
        print(f"      {names[i]:<32} own group {own or '-'} {own_share:.2f}  |  {spread}")

    prefix_mask = block_mask(dim, prefix_blocks(names))
    upper = np.triu_indices(dim, k=1)
    order = np.argsort(-np.abs(correlation[upper]))[:N_TOP_PAIRS]
    print(f"  strongest off-diagonal correlations of G (top {N_TOP_PAIRS}):")
    for rank_index in order:
        i, j = int(upper[0][rank_index]), int(upper[1][rank_index])
        where = "within prefix block" if prefix_mask[i, j] else "ACROSS prefix blocks"
        print(f"      |R| = {abs(correlation[i, j]):.3f}  {names[i]} x {names[j]}   {where}")


def main(fixtures: tuple[str, ...]) -> None:
    """Probe each fixture in turn."""
    for fixture in fixtures:
        probe(fixture)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or DEFAULT_FIXTURES)
