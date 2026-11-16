# SPDX-License-Identifier: BSD-3-Clause
"""Measure a nebular feature table against the exact path it stands in for.

One measuring tool for every nebular backend, used by the contract tests and by
``tools/measure_table_accuracy.py``. It is not a gate: the build-time gate is the
shared cell-centre check (``_agn_table_cell_center_error`` in
``components/agn/blocks/composable_precompute.py``), and this module uses that
check's definitions so the two never disagree:

* a **cell centre** is the midpoint of adjacent nodes on every gridded axis, taken
  as a full tensor product;
* the **error** of a channel is ``|got / want - 1|``, the worst over the channels
  that carry flux (above a floor), never a mean.

Two measurements:

``measure_table_vs_exact``
    The ``(WavePrecomp, FeaturePrecomp)`` model against the same model with
    ``WavePrecomp`` alone, at grid nodes and cell centres, for every star-formation
    history given. Lines compare ``predict_line_fluxes`` (grid-served against the
    exact backend call); bands compare ``predict_photometry`` (grid-served nebular
    photometry against the full SED on the same WavePrecomp path, so the grid's
    error is isolated from WavePrecomp's own). At a node the error is the
    dependence on the star-formation history the grid does not carry; at a centre it
    adds the interpolation error.

``measure_qh_linearity``
    Whether a backend's line luminosities are linear in the ionizing photon rate:
    the exact lines at ``log_total_mass + log10(k)`` against ``k`` times the lines at
    the base point.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

#: A line or band is compared only when it carries at least this fraction of the
#: brightest line (or band) at that point; fainter channels hold no measurable flux.
DEFAULT_FLOOR: float = 1e-3


@dataclass(frozen=True)
class WorstError:
    """The worst relative error of one channel, with where it occurred.

    Attributes
    ----------
    channel : str
        ``"lines"``, ``"bands"`` or ``"qh_linearity"``.
    error : float
        ``max |got / want - 1|`` over the compared channels and points.
    where : str
        Line wavelength or band index of the worst entry.
    point : Mapping[str, float]
        Grid-axis values of the worst point.
    draw : str
        Name of the star-formation history at the worst point.
    at : str
        ``"nodes"`` or ``"centres"``.
    n_compared : int
        Number of (point, channel entry) pairs compared.
    """

    channel: str
    error: float
    where: str
    point: Mapping[str, float]
    draw: str
    at: str
    n_compared: int


def _centre_params(model) -> dict[str, float]:
    """Every free parameter at its prior centre."""
    return {
        n: float(model.spec._distributions[n].unstandardize(jnp.asarray(0.0)))
        for n in model.spec.free_params
    }


def grid_points(axis_names, axes, at: str, *, max_points=None, seed=0):
    """Tensor-product grid points at the nodes or the cell centres.

    Parameters
    ----------
    axis_names : sequence of str
        Gridded parameter names.
    axes : sequence of ndarray
        Ascending node values per axis.
    at : {"nodes", "centres"}
        Which points.
    max_points : int, optional
        Keep a seeded random subset of at most this many points.
    seed : int
        Seed of that subset.

    Returns
    -------
    list of dict
        One ``{name: value}`` per point; ``[{}]`` for a 0-axis table.
    """
    if not axis_names:
        return [{}]
    if at == "nodes":
        values = [np.asarray(a, dtype=np.float64) for a in axes]
    elif at == "centres":
        values = [0.5 * (np.asarray(a)[1:] + np.asarray(a)[:-1]) for a in axes]
    else:
        raise ValueError(f"at must be 'nodes' or 'centres', got {at!r}")
    points = [dict(zip(axis_names, map(float, combo))) for combo in itertools.product(*values)]
    if max_points is not None and len(points) > max_points:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(points), size=max_points, replace=False))
        points = [points[i] for i in keep]
    return points


def _relative_error(got, want, floor):
    """Worst ``|got/want - 1|`` over entries above ``floor`` times the brightest, and its index."""
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    live = np.isfinite(want) & (np.abs(want) >= floor * np.nanmax(np.abs(want)))
    if not np.any(live):
        return 0.0, -1, 0
    rel = np.where(live, np.abs(got / np.where(live, want, 1.0) - 1.0), 0.0)
    i = int(np.argmax(rel))
    return float(rel[i]), i, int(live.sum())


def measure_table_vs_exact(
    build: Callable,
    *,
    sfh_draws: Mapping[str, Mapping[str, float]],
    line_waves: Sequence[float],
    at: Sequence[str] = ("nodes", "centres"),
    channels: Sequence[str] = ("lines", "bands"),
    max_points: int | None = None,
    floor: float = DEFAULT_FLOOR,
) -> dict[tuple[str, str], WorstError]:
    """Worst error of the nebular feature table against the exact path.

    Parameters
    ----------
    build : callable
        ``build(approx) -> SEDModel`` for an ``approx`` tuple; the caller fixes the
        backend, bands and everything else, so any nebular backend can be measured.
    sfh_draws : mapping
        ``{name: {param: value}}`` star-formation histories (and any other
        non-gridded parameters) to evaluate; unlisted free parameters sit at their
        prior centre.
    line_waves : sequence of float
        Rest-frame vacuum line wavelengths to compare [Angstrom].
    at : sequence of {"nodes", "centres"}
        Grid points to evaluate.
    channels : sequence of {"lines", "bands"}
        Channels to compare.
    max_points : int, optional
        Cap on grid points per (``at``) set, a seeded subset.
    floor : float
        Relative flux floor for a channel entry to be compared.

    Returns
    -------
    dict
        ``{(channel, at): WorstError}``.
    """
    from tengri.forward.sed_model import FeaturePrecomp, WavePrecomp

    pair = build((WavePrecomp(), FeaturePrecomp()))
    ref = build((WavePrecomp(),))
    waves = jnp.asarray(line_waves, dtype=jnp.float64)
    base = _centre_params(pair)
    pair.predict_line_fluxes(base, waves)  # builds the grid
    table = pair._nebular_grid_table
    names, axes = tuple(table.axis_names), tuple(table.axes)

    worst: dict[tuple[str, str], WorstError] = {}
    for where_at in at:
        points = grid_points(names, axes, where_at, max_points=max_points)
        for channel in channels:
            best = WorstError(channel, 0.0, "", {}, "", where_at, 0)
            count = 0
            for draw_name, draw in sfh_draws.items():
                for point in points:
                    params = {**base, **draw, **point}
                    if channel == "lines":
                        got = pair.predict_line_fluxes(params, waves)
                        want = ref.predict_line_fluxes(params, waves)
                        err, i, n = _relative_error(got, want, floor)
                        label = f"{float(waves[i]):.2f} A" if i >= 0 else ""
                    else:
                        got = pair.predict_photometry(params)
                        want = ref.predict_photometry(params)
                        err, i, n = _relative_error(got, want, floor)
                        label = f"band {i}"
                    count += n
                    if err > best.error:
                        best = WorstError(channel, err, label, point, draw_name, where_at, 0)
            worst[(channel, where_at)] = WorstError(
                best.channel, best.error, best.where, best.point, best.draw, best.at, count
            )
    return worst


def measure_qh_linearity(
    build: Callable,
    *,
    sfh_draws: Mapping[str, Mapping[str, float]],
    line_waves: Sequence[float],
    log10_scale: float = 1.0,
    floor: float = DEFAULT_FLOOR,
) -> WorstError:
    """Worst departure of the exact line luminosities from linearity in Q_H.

    Parameters
    ----------
    build : callable
        ``build(approx) -> SEDModel``; called once with ``approx=None``.
    sfh_draws : mapping
        ``{name: {param: value}}`` points to test.
    line_waves : sequence of float
        Rest-frame vacuum line wavelengths [Angstrom].
    log10_scale : float
        ``log10(k)``: the stellar mass, and so Q_H, is raised by this many dex.
    floor : float
        Relative flux floor for a line to be compared.

    Returns
    -------
    WorstError
        ``max |L(k Q_H) / (k L(Q_H)) - 1|``, channel ``"qh_linearity"``.
    """
    model = build(None)
    mass = [n for n in model.spec.free_params if n.endswith("log_total_mass")]
    if len(mass) != 1:
        raise ValueError(f"need exactly one free *log_total_mass parameter, found {mass}")
    waves = jnp.asarray(line_waves, dtype=jnp.float64)
    base = _centre_params(model)
    best = WorstError("qh_linearity", 0.0, "", {}, "", "nodes", 0)
    count = 0
    for draw_name, draw in sfh_draws.items():
        params = {**base, **draw}
        lo = np.asarray(model.predict_line_fluxes(params, waves), dtype=np.float64)
        hi_params = {**params, mass[0]: params[mass[0]] + log10_scale}
        hi = np.asarray(model.predict_line_fluxes(hi_params, waves), dtype=np.float64)
        err, i, n = _relative_error(hi / 10.0**log10_scale, lo, floor)
        count += n
        if err > best.error:
            best = WorstError(
                "qh_linearity", err, f"{float(waves[i]):.2f} A", {}, draw_name, "", 0
            )
    return WorstError(best.channel, best.error, best.where, best.point, best.draw, best.at, count)
