# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapters for analytic dust-emission models.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
four analytic dust-emission models:

1. **modified_blackbody**: Optically-thin modified blackbody SED.
   Two free axes: ``dust_T`` (temperature), ``dust_beta_ir`` (emissivity index).

2. **casey2012**: Casey (2012) modified blackbody + mid-IR power law.
   Four free axes: ``dust_T`` (temperature), ``dust_beta_ir`` (emissivity index),
   ``dust_alpha_mir`` (mid-IR power-law slope), ``dust_lambda_0_um`` (opacity
   pivot wavelength).

3. **graybody**: General-opacity graybody with free pivot wavelength.
   Three free axes: ``dust_T`` (temperature), ``dust_beta_ir`` (emissivity index),
   ``dust_lambda_0_um`` (opacity pivot wavelength).

4. **pah_drude**: Sum of 18 PAH Drude profiles (Smith et al. 2007).
   No free axes (pure template; runtime amplitudes scale the combined PAH profile).

Each model is preintegrated through filter curves at model-initialization time.
Auto-collapses axes whose corresponding parameters are ``Fixed`` in the user's
``Parameters``: e.g., a user who pins ``dust_T`` gets a 1D grid.

References
----------
.. [1] Casey, C. M., "Dusty star-forming galaxies at high redshift,"
       MNRAS, 425, 3094 (2012). arXiv:1206.1595.
       https://doi.org/10.1111/j.1365-2966.2012.21455.x
.. [2] Smith, J. D., et al., "The mid-infrared emission of ultraluminous
       infrared galaxies," ApJ, 656, 770 (2007). arXiv:astro-ph/0701042.
       https://doi.org/10.1086/510378
.. [3] Hildebrand, R. H., "The determination of cloud masses from dust continuum
       emission," QJRAS, 24, 267 (1983).

"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.dust._params import PARAMS as _DUST_PARAMS
from tengri.components.dust.drude_profiles import compute_pah_template as _compute_pah
from tengri.components.dust.emission import (
    casey2012 as _casey2012,
    graybody as _graybody,
    modified_blackbody as _modified_blackbody,
)
from tengri.components.dust.emission.analytic._closures import (
    _CASEY_LAMBDA_MIN_UM,
)
from tengri.config.exceptions import GridSupportWarning
from tengri.forward.precompute.templates import (
    collapse_fixed_axes,
    precompute_template_photometry,
)
from tengri.utils.grid_interp import PreintegratedGrid, interp_nd_pchip
from tengri.utils.interpolation import edges_for_grid
from tengri.utils.physics_constants import C_CGS as _C_CGS

# ── Helper: read grid bounds from declared priors ──────────────────────────────────


def _get_param_bounds(param_name: str) -> tuple[float, float]:
    """Extract lower and upper bounds from a parameter's free_prior in PARAMS.

    Parameters
    ----------
    param_name : str
        Name of the parameter, e.g. 'dust_T', 'dust_beta_ir'.

    Returns
    -------
    tuple[float, float]
        (lower, upper) bounds from the free_prior.

    Raises
    ------
    ValueError
        If the parameter has no bounded free prior.
    """
    for p in _DUST_PARAMS:
        if (
            p.name == param_name
            and hasattr(p, "free_prior")
            and p.free_prior is not None
            and hasattr(p.free_prior, "lo")
        ):
            return float(p.free_prior.lo), float(p.free_prior.hi)
    raise ValueError(f"Parameter {param_name} has no bounded free_prior in PARAMS")


# ── Axis definitions per model ──────────────────────────────────────

# modified_blackbody: parametrized by temperature and emissivity index
AXIS_PARAMS_MBB = ("dust_T", "dust_beta_ir")

# casey2012: parametrized by temperature, emissivity index, mid-IR slope, opacity pivot
AXIS_PARAMS_CASEY = ("dust_T", "dust_beta_ir", "dust_alpha_mir", "dust_lambda_0_um")

# graybody: parametrized by temperature, emissivity index, opacity pivot
AXIS_PARAMS_GRAYBODY = ("dust_T", "dust_beta_ir", "dust_lambda_0_um")

# pah_drude: pure template; no grid axes (runtime amplitude scaling)
AXIS_PARAMS_PAH = ()

# Protocol-required dict form for multi-model module
AXIS_PARAMS: dict[str, tuple[str, ...]] = {
    "modified_blackbody": AXIS_PARAMS_MBB,
    "casey2012": AXIS_PARAMS_CASEY,
    "graybody": AXIS_PARAMS_GRAYBODY,
    "pah_drude": AXIS_PARAMS_PAH,
}


# Rest-frame integration grid of the dust models: 0.01 um to 10 m, log-spaced at
# 250 points per decade. The thermal continuum closures normalize their SED to
# L_absorbed by integrating over the grid they are given, and the far-IR through
# radio bands they feed (Herschel, SCUBA-2, ALMA, VLA, MeerKAT) sit at 70 um - 1 m on the
# Rayleigh-Jeans tail, so the grid has to contain the whole thermal bump and the bands.
_CONTINUUM_LOG10_WAVE_AA_MIN = 2.0
_CONTINUUM_LOG10_WAVE_AA_MAX = 11.0
_CONTINUUM_N_WAVE = 2250


def _continuum_wave_rest() -> np.ndarray:
    """Rest-frame wavelength grid [Angstrom] for the thermal-continuum precompute builders."""
    return np.logspace(
        _CONTINUUM_LOG10_WAVE_AA_MIN,
        _CONTINUUM_LOG10_WAVE_AA_MAX,
        _CONTINUUM_N_WAVE,
        dtype=np.float64,
    )


def _build_union_grid_with_fine_filters(
    filter_waves: list, redshift: float, wave_rest_base: np.ndarray
) -> np.ndarray:
    """Build union of base wavelength grid and per-filter fine grids.

    Each filter adds a fine logarithmic grid across its rest-frame support,
    sufficient to integrate the model accurately without template interpolation.

    Parameters
    ----------
    filter_waves : list[ndarray]
        Wavelength grids per filter [Angstrom], observed frame.
    redshift : float
        Source redshift for observed-to-rest frame conversion.
    wave_rest_base : ndarray
        Base rest-frame wavelength grid [Angstrom].

    Returns
    -------
    ndarray
        Union grid, sorted, unique.
    """
    union = set(wave_rest_base)

    for fw in filter_waves:
        fw = np.asarray(fw, dtype=np.float64)
        lo_rest = max(fw.min() / (1 + redshift), wave_rest_base.min())
        hi_rest = fw.max() / (1 + redshift)
        # Add fine grid across filter's rest-frame support
        fine = np.geomspace(lo_rest, hi_rest, max(400, len(fw)))
        union.update(fine)

    return np.array(sorted(union), dtype=np.float64)


def _validate_filter_coverage(
    filter_waves: list, redshift: float, wave_rest_base: np.ndarray
) -> None:
    """Refuse a filter whose rest-frame RED edge lies beyond the rest grid.

    Only the red edge is checked. The blue side needs no check: the template is taken as
    zero below the grid minimum (100 Å), and the fine per-filter grid starts at that
    minimum at the earliest.

    Raises
    ------
    ValueError
        If any filter's RED edge (max wavelength) extends beyond the rest grid.
    """
    wave_rest_max = wave_rest_base.max()

    for i, fw in enumerate(filter_waves):
        fw = np.asarray(fw, dtype=np.float64)
        hi_rest = fw.max() / (1 + redshift)
        if hi_rest > wave_rest_max:
            raise ValueError(
                f"Filter {i} RED edge at {hi_rest:.2e} Angstrom "
                f"exceeds rest-frame grid maximum {wave_rest_max:.2e} Angstrom. "
                f"Observed-frame range: [{fw.min():.2e}, {fw.max():.2e}] Angstrom at z={redshift}."
            )


_NODE_CHUNK_BYTES = 16_000_000


def _band_integrals_over_nodes(
    closure,
    node_kwargs: dict[str, np.ndarray],
    closure_kwargs: dict[str, float],
    wave_rest: np.ndarray,
    filter_waves: list,
    filter_trans: list,
    redshift: float,
) -> np.ndarray:
    """Evaluate ``closure`` at every node and reduce each batch to band integrals.

    Parameters
    ----------
    closure : callable
        Analytic SED closure ``closure(wave, L_absorbed, redshift=..., **params)``.
    node_kwargs : dict[str, ndarray, shape (n_nodes,)]
        Per-node values of the varied parameters.
    closure_kwargs : dict[str, float]
        Parameters held at one value for all nodes.
    wave_rest : ndarray, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom].
    filter_waves, filter_trans : list[ndarray]
        Per-filter wavelength [Angstrom, observed frame] and transmission grids.
    redshift : float
        Source redshift [dimensionless].

    Returns
    -------
    ndarray, shape (n_nodes, n_filters)
        Band-averaged L_nu per unit absorbed luminosity [1/Hz].
    """
    names = tuple(node_kwargs)
    n_nodes = len(next(iter(node_kwargs.values())))
    chunk = max(1, _NODE_CHUNK_BYTES // (8 * wave_rest.size))
    wave_j = jnp.asarray(wave_rest)

    def one_node(*values):
        return closure(
            wave_j, 1.0, redshift=float(redshift), **closure_kwargs, **dict(zip(names, values))
        )

    batched = jax.jit(jax.vmap(one_node))
    rows = []
    for start in range(0, n_nodes, chunk):
        # Pad the last batch to the common size so the compiled kernel is reused.
        idx = np.minimum(np.arange(start, start + chunk), n_nodes - 1)
        templates = np.asarray(batched(*(jnp.asarray(node_kwargs[n][idx]) for n in names)))
        band = precompute_template_photometry(
            templates=templates,
            wave_rest=wave_rest,
            filter_waves=[np.asarray(fw, dtype=np.float64) for fw in filter_waves],
            filter_trans=[np.asarray(ft, dtype=np.float64) for ft in filter_trans],
            axes=(),
            redshift=redshift,
            dl_cm=1.0,
            energy_normalize=False,
            units="lnu",
        )
        rows.append(np.asarray(band.phot)[: min(chunk, n_nodes - start)])
    return np.concatenate(rows, axis=0)


def _build_grid(
    closure,
    param_names: tuple[str, ...],
    axes: tuple[np.ndarray, ...],
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    closure_kwargs: dict[str, float] | None = None,
    exact_nodes_aa: tuple[float, ...] = (),
) -> tuple[PreintegratedGrid, np.ndarray]:
    """Preintegrate a continuum closure over the Cartesian product of ``axes``.

    ``exact_nodes_aa`` are rest wavelengths [Angstrom] added as nodes of the rest
    grid, so that a hard bound of the closure falls on a node.

    The closure is evaluated on the union of the base rest grid and a fine grid
    across each filter, in batches, so no template interpolation enters the band
    integral and the node array never exceeds ``_NODE_CHUNK_BYTES`` per field.

    Returns
    -------
    PreintegratedGrid
        ``phot`` has shape ``(*[len(ax) for ax in axes], n_filters)``.
    ndarray, float64
        ln(band flux) at the nodes, same shape as ``phot``, taken in float64 and floored at
        the float64 smallest normal so a band that underflows float32 still has a finite log.
    """
    axes = tuple(np.asarray(ax, dtype=np.float64) for ax in axes)
    wave_rest_base = _continuum_wave_rest()
    _validate_filter_coverage(filter_waves, redshift, wave_rest_base)
    wave_rest = _build_union_grid_with_fine_filters(filter_waves, redshift, wave_rest_base)
    wave_rest = np.unique(
        np.concatenate([wave_rest, np.asarray(exact_nodes_aa, dtype=np.float64)])
    )

    mesh = np.meshgrid(*axes, indexing="ij")
    node_kwargs = {name: m.ravel() for name, m in zip(param_names, mesh)}
    flat = _band_integrals_over_nodes(
        closure,
        node_kwargs,
        closure_kwargs or {},
        wave_rest,
        filter_waves,
        filter_trans,
        redshift,
    )
    probe = precompute_template_photometry(
        templates=np.zeros((1, wave_rest.size)),
        wave_rest=wave_rest,
        filter_waves=[np.asarray(fw, dtype=np.float64) for fw in filter_waves],
        filter_trans=[np.asarray(ft, dtype=np.float64) for ft in filter_trans],
        axes=(),
        redshift=redshift,
        dl_cm=1.0,
        energy_normalize=False,
        units="lnu",
    )
    axes_jax = tuple(jnp.asarray(ax) for ax in axes)
    shaped = flat.reshape(*(len(ax) for ax in axes), len(filter_waves))
    grid = dataclasses.replace(
        probe,
        phot=jnp.asarray(shaped),
        axes=axes_jax,
        edges=tuple(edges_for_grid(ax) for ax in axes_jax),
    )
    return grid, np.log(np.maximum(shaped.astype(np.float64), _FLOAT64_TINY))


def _build_grid_modified_blackbody(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    T_grid: np.ndarray,
    beta_grid: np.ndarray,
) -> tuple[PreintegratedGrid, np.ndarray]:
    """Preintegrate the modified blackbody over (T, beta); shape (n_T, n_beta, n_filters)."""
    return _build_grid(
        _modified_blackbody,
        AXIS_PARAMS_MBB,
        (T_grid, beta_grid),
        filter_waves,
        filter_trans,
        redshift,
    )


def _build_grid_casey2012(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    T_grid: np.ndarray,
    beta_grid: np.ndarray,
    alpha_mir_grid: np.ndarray,
    lambda_0_um_grid: np.ndarray,
) -> tuple[PreintegratedGrid, np.ndarray]:
    """Preintegrate Casey (2012) over (T, beta, alpha_mir, lambda_0); 4 axes + filters."""
    return _build_grid(
        _casey2012,
        AXIS_PARAMS_CASEY,
        (T_grid, beta_grid, alpha_mir_grid, lambda_0_um_grid),
        filter_waves,
        filter_trans,
        redshift,
        exact_nodes_aa=(_CASEY_LAMBDA_MIN_UM * 1e4,),
    )


def _build_grid_graybody(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    T_grid: np.ndarray,
    beta_grid: np.ndarray,
    lambda_0_um_grid: np.ndarray,
) -> tuple[PreintegratedGrid, np.ndarray]:
    """Preintegrate the graybody over (T, beta, lambda_0); axes (n_T, n_beta, n_lambda_0)."""
    return _build_grid(
        _graybody,
        AXIS_PARAMS_GRAYBODY,
        (T_grid, beta_grid, lambda_0_um_grid),
        filter_waves,
        filter_trans,
        redshift,
    )


def _build_grid_pah_drude(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    L_absorbed_ref: float = 1.0,
) -> PreintegratedGrid:
    """Preintegrate PAH Drude template through filters.

    The PAH template is pure shape (no axes); an amplitude scales it. The adapter is
    registered in ``forward/precompute/registry.py``; no kernel consumes its lookups
    today.

    Parameters
    ----------
    filter_waves : list[ndarray]
        Per-filter wavelength arrays [Angstrom], observed frame.
    filter_trans : list[ndarray]
        Per-filter transmission curves.
    redshift : float
        Source redshift. The band integral reads the rest-frame template at
        ``lambda_obs / (1 + z)``; no ``(1 + z)`` or distance factor is applied.
    L_absorbed_ref : float
        Reference absorbed luminosity for normalization [L_sun]. Default 1.0.

    Returns
    -------
    PreintegratedGrid
        Preintegrated photometry with shape (1, n_filters) (scalar template).
    """
    # PAH grid uses the same rest range as the continuum builders
    wave_rest_base = _continuum_wave_rest()

    # Validate filters are within base grid bounds before building union grid
    _validate_filter_coverage(filter_waves, redshift, wave_rest_base)

    wave_rest = _build_union_grid_with_fine_filters(filter_waves, redshift, wave_rest_base)

    # Compute PAH template using Smith+2007 SINGS median strengths
    pah_llam = np.asarray(_compute_pah(jnp.asarray(wave_rest * 1e-4)))  # Å -> μm

    # L_nu = L_lambda * lambda^2 / c. PAH template is dimensionless (relative);
    # scale to L_absorbed_ref.
    wave_cm = wave_rest * 1e-8
    lnu = L_absorbed_ref * pah_llam * (wave_cm**2) / _C_CGS

    # Wrap in shape (1, n_wave) for template compatibility
    templates = np.array([lnu], dtype=np.float64)

    # No grid axes for PAH: preintegrate as a single template
    return precompute_template_photometry(
        templates=templates,
        wave_rest=wave_rest,
        filter_waves=[np.asarray(fw, dtype=np.float64) for fw in filter_waves],
        filter_trans=[np.asarray(ft, dtype=np.float64) for ft in filter_trans],
        axes=(),  # No axes: scalar template
        redshift=redshift,  # observed-frame filters: template read at lambda_obs/(1+z)
        dl_cm=1.0,
        energy_normalize=False,  # template already normalized to L_absorbed_ref
        units="lnu",
    )


# Default node counts per axis. Measured over 200 seeded random points inside the declared
# priors in the 60-90, 250-500 and 750-950 um bands, the log band flux interpolated with
# PCHIP agrees with the exact closure to <= 6e-4 at these counts (far-IR bands, z = 0).
# casey2012 mid-IR bands (8-24 um at z = 0; 60-90 um, 250-500 um at z = 3): beta_ir raised
# to 10 (from 8) so 13-point grid achieves <= 4e-4 error; per-band tested with 13 seeded
# random points (the issue query plus 12 RandomState(7) points in the declared priors).
_DEFAULT_NODES: dict[str, dict[str, int]] = {
    "modified_blackbody": {"dust_T": 49, "dust_beta_ir": 12},
    "casey2012": {"dust_T": 41, "dust_beta_ir": 10, "dust_alpha_mir": 21, "dust_lambda_0_um": 26},
    "graybody": {"dust_T": 41, "dust_beta_ir": 10, "dust_lambda_0_um": 30},
}

# The ln of a band flux is taken in float64 at build, so its floor is the float64 smallest normal.
_FLOAT64_TINY = np.finfo(np.float64).tiny

# Axes whose interpolation coordinate is the natural log of the parameter.
_LOG_AXIS_PARAMS = ("dust_T", "dust_lambda_0_um")


def _active_support(param_name: str, parameters: Any) -> tuple[float, float] | None:
    """Range of ``param_name`` the model can reach, or None when it is not bounded by the model.

    ``Fixed(v)`` gives ``(v, v)`` and a free parameter its prior's finite ``bounds``. ``None``
    means no model, a parameter the model does not declare, or a prior with an infinite bound; the
    last warns once, because the nodes then span the declared range and the lookup holds the edge
    value beyond it.
    """
    if parameters is None:
        return None
    fixed = parameters.get_fixed_values()
    if param_name in fixed:
        return (fixed[param_name], fixed[param_name])
    if param_name not in parameters.free_params:
        return None
    lo, hi = parameters.get_distribution(param_name).bounds
    if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
        return (float(lo), float(hi))
    declared = _get_param_bounds(param_name)
    warnings.warn(
        f"{param_name} has an unbounded prior; the nodes span its declared range "
        f"[{declared[0]:g}, {declared[1]:g}] and the lookup holds the edge value, with zero "
        f"gradient, beyond it. Give the prior finite bounds to widen the nodes.",
        GridSupportWarning,
        stacklevel=3,
    )
    return None


def _default_axis(
    param_name: str, n_nodes: int, support: tuple[float, float] | None = None
) -> np.ndarray:
    """Node grid over the declared prior extended to ``support``, at the declared node density.

    Geometric for the log axes, linear otherwise. The count scales with the span in the
    interpolation coordinate (``ln`` for :data:`_LOG_AXIS_PARAMS`),
    ``ceil(n_nodes * span_axis / span_declared)``, never below ``n_nodes``, so the node spacing
    that the #2676 accuracy figures were measured at is kept when the support is wider. With
    ``support`` None, or inside the declared prior, the axis is the declared one exactly.
    """
    declared_lo, declared_hi = _get_param_bounds(param_name)
    lo, hi = declared_lo, declared_hi
    if support is not None:
        lo, hi = min(lo, support[0]), max(hi, support[1])
    log_axis = param_name in _LOG_AXIS_PARAMS
    if log_axis and lo <= 0.0:
        raise ValueError(f"{param_name} reaches {lo:g}; a logarithmic node axis needs lo > 0.")
    coordinate = np.log if log_axis else np.asarray
    stretch = (coordinate(hi) - coordinate(lo)) / (
        coordinate(declared_hi) - coordinate(declared_lo)
    )
    n_axis = max(n_nodes, int(np.ceil(n_nodes * stretch)))
    if log_axis:
        return np.geomspace(lo, hi, n_axis, dtype=np.float64)
    return np.linspace(lo, hi, n_axis, dtype=np.float64)


def _check_user_axis(
    param_name: str, axis: np.ndarray, support: tuple[float, float] | None
) -> None:
    """Refuse a user-supplied axis that does not span the model's reach; warn below 4 nodes."""
    if support is not None and (axis.min() > support[0] or axis.max() < support[1]):
        raise ValueError(
            f"{param_name} nodes span [{axis.min():g}, {axis.max():g}] but the model reaches "
            f"[{support[0]:g}, {support[1]:g}]; the lookup would hold the edge value with zero "
            f"gradient beyond the nodes. Supply nodes covering the support."
        )
    if axis.size < 4:
        warnings.warn(
            f"{param_name} has {axis.size} nodes; the PCHIP lookup degrades to a parabola or a "
            f"chord below 4.",
            UserWarning,
            stacklevel=3,
        )


_CONTINUUM_BUILDERS = {
    "modified_blackbody": _build_grid_modified_blackbody,
    "casey2012": _build_grid_casey2012,
    "graybody": _build_grid_graybody,
}


# ── Protocol-shaped entry points ──────────────────────────────────


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any,
    *,
    model: str = "modified_blackbody",
    T_grid: np.ndarray | None = None,
    beta_grid: np.ndarray | None = None,
    alpha_mir_grid: np.ndarray | None = None,
    lambda_0_um_grid: np.ndarray | None = None,
) -> dict:
    """Build preintegrated analytic dust grid, auto-collapsing Fixed-parameter axes.

    Multi-model entry point. Dispatches to the appropriate builder based
    on ``model`` parameter. Each node is the closed-form model evaluated on the union of
    the rest grid (0.01 um to 10 m, 250 points per decade) and a fine grid across each
    filter, in batches, then reduced to band integrals; no template interpolation enters
    the band integral. A filter whose rest-frame red edge lies beyond 10 m raises
    ``ValueError`` naming the filter; below 100 A (0.01 um) the template is taken as
    zero: exact to double precision for the thermal models, and for ``pah_drude`` the Drude
    wings there are below 1.7e-14 of the peak (measured). Accuracy of :func:`build_lookup` at
    the default nodes against the exact closure, maximum over 200 seeded random points inside
    the declared priors, z = 0, bands 60-90 / 250-500 / 750-950 um: ``modified_blackbody``
    3.5e-4, ``graybody`` 2.9e-4, ``casey2012`` 8.0e-4 (60-90 um), 6.7e-4 (250-500 um), 7.0e-4
    (750-950 um). ``casey2012`` at 8-24 um and 24-40 um is 1.9e-4 at z = 0; at z = 3 (observed
    bands) it is 1.8e-4 in 60-90 um, 5.3e-4 in 250-500 um (13 seeded points in the declared
    priors). The 1 um lower bound of ``casey2012`` is a node of the rest grid, so its
    normalization has no cell straddling the bound.

    Parameters
    ----------
    filter_waves : list[ndarray]
        Wavelength grid per filter [Angstrom], observed frame.
    filter_trans : list[ndarray]
        Transmission per filter (0–1).
    redshift : float
        Source redshift. [dimensionless]
    parameters : Parameters | None
        Parameters spec, used to detect Fixed-axis parameters.
    model : str, keyword-only
        One of "modified_blackbody", "casey2012", "graybody", "pah_drude".
        Default: "modified_blackbody".
    T_grid : ndarray, optional
        Temperature nodes [K]. If None, geometric nodes over the declared free prior
        (dust_T: 20-80 K) extended to the model's reach (see Notes).
    beta_grid : ndarray, optional
        Emissivity-index nodes [dimensionless]. If None, linear nodes over the declared free
        prior (dust_beta_ir: 1.0-2.5), extended likewise.
    alpha_mir_grid : ndarray, optional
        Mid-IR power-law slope nodes for casey2012 [dimensionless]. If None, linear nodes over
        the declared free prior (dust_alpha_mir: 1.0-3.0), extended likewise.
    lambda_0_um_grid : ndarray, optional
        Opacity pivot wavelength nodes for graybody and casey2012 [micron]. If None, geometric
        nodes over the declared free prior (dust_lambda_0_um: 50-500 um), extended likewise.

    Returns
    -------
    dict
        Keys: "grid_phot" (photometry array), "axes" (free axes), "_preint"
        (PreintegratedGrid), optionally "_collapsed_axes" (if any axes fixed).

    References
    ----------
    .. [1] Casey, C. M., "Dusty star-forming galaxies at high redshift,"
           MNRAS, 425, 3094 (2012).
    .. [2] Smith, J. D., et al., "The mid-infrared emission of ultraluminous
           infrared galaxies," ApJ, 656, 770 (2007).

    Raises
    ------
    ValueError
        If a supplied node axis does not span what ``parameters`` lets the axis parameter reach
        (``Fixed(v)``: ``v``; a free prior: its finite bounds), or a default axis would need a
        non-positive node on a logarithmic axis.

    Warns
    -----
    GridSupportWarning
        Once per axis whose prior has an infinite bound: the nodes then span the declared range
        and the lookup holds the edge value, with zero gradient, beyond it.
    UserWarning
        If a supplied node axis has fewer than 4 nodes (the PCHIP lookup degrades to a parabola
        or a chord).

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function using NumPy.

    **Node axes.** The lookup holds the edge value beyond its nodes, so a default axis spans the
    declared free prior extended to the range ``parameters`` can reach: ``Fixed(v)`` extends it
    to include ``v``, a free prior with finite bounds to include them. The node count keeps the
    declared density in the interpolation coordinate (``ln`` for dust_T and dust_lambda_0_um),
    ``ceil(n_default * span_axis / span_declared)``, so the #2676 accuracy figures above carry
    over. With ``parameters=None``, or priors inside the declared ranges, the axes are the
    declared ones exactly. A supplied axis is used as given and is checked against the same
    reach. Nothing here changes the exact closures.
    """
    if model == "pah_drude":
        preint = _build_grid_pah_drude(filter_waves, filter_trans, redshift)
        ln_phot = np.log(np.maximum(np.asarray(preint.phot, dtype=np.float64), _FLOAT64_TINY))
        result = {"grid_phot": preint.phot, "axes": (), "_preint": preint, "_ln_phot": ln_phot}
        axis_params = AXIS_PARAMS_PAH
    elif model in _CONTINUUM_BUILDERS:
        axis_params = AXIS_PARAMS[model]
        supplied = {
            "dust_T": T_grid,
            "dust_beta_ir": beta_grid,
            "dust_alpha_mir": alpha_mir_grid,
            "dust_lambda_0_um": lambda_0_um_grid,
        }
        axes = []
        for name in axis_params:
            support = _active_support(name, parameters)
            if supplied[name] is None:
                axes.append(_default_axis(name, _DEFAULT_NODES[model][name], support))
            else:
                axis = np.asarray(supplied[name], dtype=np.float64)
                _check_user_axis(name, axis, support)
                axes.append(axis)
        axes = tuple(axes)
        preint, ln_phot = _CONTINUUM_BUILDERS[model](filter_waves, filter_trans, redshift, *axes)
        result = {
            "grid_phot": preint.phot,
            "axes": tuple(jnp.asarray(ax) for ax in axes),
            "_preint": preint,
            "_ln_phot": ln_phot,
        }
    else:
        raise ValueError(f"Unknown analytic dust model: {model}")

    # Auto-collapse any Fixed axes: collapse_fixed_axes names them (and warns on dead
    # labels); the pinned values are then interpolated out with the lookup's own interpolant.
    preint: PreintegratedGrid = result["_preint"]
    _, remaining_axes, fixed = collapse_fixed_axes(
        preint, axis_params, parameters, origin=f"dust_analytic_precompute[{model}]"
    )
    if not fixed:
        return result

    collapsed, ln_collapsed = _pin_axes(preint, result["_ln_phot"], axis_params, fixed)
    return {
        "grid_phot": collapsed.phot,
        "axes": remaining_axes,
        "_preint": collapsed,
        "_ln_phot": ln_collapsed,
        "_collapsed_axes": fixed,
    }


def _axis_coordinate(param_name: str, values):
    """Interpolation coordinate of an axis: ln of the value for log axes, else the value."""
    return jnp.log(values) if param_name in _LOG_AXIS_PARAMS else values


def _pin_axes(
    preint: PreintegratedGrid,
    ln_phot: np.ndarray,
    axis_params: tuple[str, ...],
    fixed: dict[int, float],
) -> tuple[PreintegratedGrid, jnp.ndarray]:
    """Remove the pinned axes by PCHIP interpolation of ln(band flux) at the pinned values."""
    keep = [i for i in range(len(axis_params)) if i not in fixed]
    pinned = sorted(fixed)
    ln_grid = jnp.asarray(ln_phot)
    moved = jnp.transpose(ln_grid, (*pinned, *keep, ln_grid.ndim - 1))
    pinned_axes = tuple(_axis_coordinate(axis_params[i], preint.axes[i]) for i in pinned)
    point = tuple(_axis_coordinate(axis_params[i], jnp.asarray(fixed[i])) for i in pinned)
    reduced = interp_nd_pchip(moved, pinned_axes, point)
    pinned_grid = dataclasses.replace(
        preint,
        phot=jnp.exp(reduced),
        axes=tuple(preint.axes[i] for i in keep),
        edges=tuple(preint.edges[i] for i in keep),
    )
    return pinned_grid, reduced


def build_lookup(
    preint: dict,
    *,
    model: str = "modified_blackbody",
    free_param_names: tuple[str, ...] | None = None,
):
    """Build the runtime analytic dust photometry lookup from a preintegrated dict.

    Interpolates ln(band flux) with monotone cubic Hermite (PCHIP) in the coordinates
    (ln T, beta, alpha_mir, ln lambda_0) of the axes the model has. The contract
    tests assert 1e-3 against the exact closure at random points inside the declared priors.
    The nodes are band integrals of the closed-form model on a rest grid of 0.01 um
    to 10 m, so no template interpolation enters the band integral. A band whose
    rest-frame red edge lies beyond 10 m is refused at build time with ``ValueError``;
    below 100 A the template is taken as zero. Accuracy figures are those of :func:`precompute`
    (far-IR 2.9e-4 to 8.0e-4; ``casey2012`` mid-IR 8-24 um and 24-40 um 1.9e-4 at z = 0,
    1.8e-4 in 60-90 um and 5.3e-4 in 250-500 um at z = 3). A query outside the node span is
    clamped to the edge node: the value is constant and the gradient zero beyond it.

    Parameters
    ----------
    preint : dict
        Preintegrated data dict with keys "grid_phot", "axes", optionally
        "_collapsed_axes".
    model : str, keyword-only
        Analytic dust model name (for documentation; not used in lookup logic).
    free_param_names : tuple of str, optional
        Names of remaining free axes in the collapsed case.
        Not used in the default (no-collapse) case.

    Returns
    -------
    callable
        JIT-compiled photometry lookup function with signature::

            fn(L_absorbed, *free_axis_values) -> ndarray, shape (n_filters,)

        Returns dust emission L_ν [erg/s/Hz]. Caller applies flux scaling.
        Off-node accuracy: monotone cubic Hermite interpolation on log-flux,
        Accuracy at the default node grids is stated in :func:`precompute`.

    References
    ----------
    .. [1] Casey, C. M., "Dusty star-forming galaxies at high redshift,"
           MNRAS, 425, 3094 (2012).

    Notes
    -----
    **JIT-compatible**: yes, the returned function uses ``jnp`` and PCHIP
    interpolation.

    **Gradient-safe**: yes, PCHIP kernel is fully differentiable.

    A query outside the node span returns the edge value with exactly zero gradient (the
    interpolation coordinate is clamped). :func:`precompute` spans the nodes over everything the
    model can reach, so that happens only under an unbounded prior, which warns at build.
    """
    axis_params_names = tuple(
        name
        for i, name in enumerate(AXIS_PARAMS[model])
        if i not in preint.get("_collapsed_axes", {})
    )
    axes = tuple(
        _axis_coordinate(name, ax) for name, ax in zip(axis_params_names, preint["_preint"].axes)
    )
    log_grid_phot = jnp.asarray(preint["_ln_phot"])

    @jax.jit
    def dust_phot(L_absorbed, *free_axis_values):
        """Band-averaged L_nu [erg/s/Hz]: ``L_absorbed`` times the PCHIP interpolant of ln flux."""
        query = tuple(_axis_coordinate(n, v) for n, v in zip(axis_params_names, free_axis_values))
        normed = (
            jnp.exp(interp_nd_pchip(log_grid_phot, axes, query))
            if axes
            else jnp.exp(log_grid_phot.ravel())
        )
        return L_absorbed * normed

    return dust_phot
