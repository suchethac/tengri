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
    """Validate RED-edge coverage; clip BLUE edge.

    Thermal dust emission models emit negligibly in the Wien tail (< 100 Å).
    Clips the BLUE side of each filter to the grid minimum (100 Å) and refuses
    only when the RED edge (max wavelength) exceeds the rest-frame grid.

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
) -> PreintegratedGrid:
    """Preintegrate a continuum closure over the Cartesian product of ``axes``.

    The closure is evaluated on the union of the base rest grid and a fine grid
    across each filter, in batches, so no template interpolation enters the band
    integral and the node array never exceeds ``_NODE_CHUNK_BYTES`` per field.

    Returns
    -------
    PreintegratedGrid
        ``phot`` has shape ``(*[len(ax) for ax in axes], n_filters)``.
    """
    axes = tuple(np.asarray(ax, dtype=np.float64) for ax in axes)
    wave_rest_base = _continuum_wave_rest()
    _validate_filter_coverage(filter_waves, redshift, wave_rest_base)
    wave_rest = _build_union_grid_with_fine_filters(filter_waves, redshift, wave_rest_base)

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
    return dataclasses.replace(
        probe,
        phot=jnp.asarray(flat.reshape(*(len(ax) for ax in axes), len(filter_waves))),
        axes=axes_jax,
        edges=tuple(edges_for_grid(ax) for ax in axes_jax),
    )


def _build_grid_modified_blackbody(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    T_grid: np.ndarray,
    beta_grid: np.ndarray,
) -> PreintegratedGrid:
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
) -> PreintegratedGrid:
    """Preintegrate Casey (2012) over (T, beta, alpha_mir, lambda_0); 4 axes + filters."""
    return _build_grid(
        _casey2012,
        AXIS_PARAMS_CASEY,
        (T_grid, beta_grid, alpha_mir_grid, lambda_0_um_grid),
        filter_waves,
        filter_trans,
        redshift,
    )


def _build_grid_graybody(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    T_grid: np.ndarray,
    beta_grid: np.ndarray,
    lambda_0_um_grid: np.ndarray,
) -> PreintegratedGrid:
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
# PCHIP agrees with the exact closure to <= 5e-4 at these counts.
_DEFAULT_NODES: dict[str, dict[str, int]] = {
    "modified_blackbody": {"dust_T": 49, "dust_beta_ir": 12},
    "casey2012": {"dust_T": 41, "dust_beta_ir": 8, "dust_alpha_mir": 21, "dust_lambda_0_um": 26},
    "graybody": {"dust_T": 41, "dust_beta_ir": 10, "dust_lambda_0_um": 30},
}

# Axes whose interpolation coordinate is the natural log of the parameter.
_LOG_AXIS_PARAMS = ("dust_T", "dust_lambda_0_um")

# Floor of a band flux before its log is taken [1/Hz].
_FLUX_FLOOR = 1e-300


def _default_axis(param_name: str, n_nodes: int) -> np.ndarray:
    """Node grid spanning the parameter's declared free prior: geometric for log axes."""
    lo, hi = _get_param_bounds(param_name)
    if param_name in _LOG_AXIS_PARAMS:
        return np.geomspace(lo, hi, n_nodes, dtype=np.float64)
    return np.linspace(lo, hi, n_nodes, dtype=np.float64)


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
    ``ValueError`` naming the filter; the blue side is clipped at 0.01 um, where thermal
    dust emission is negligible. At the default nodes the lookup of
    :func:`build_lookup` agrees with the exact closure to <= 5e-4 over the declared
    priors in the 60-90, 250-500 and 750-950 um bands.

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
        Temperature nodes [K]. If None, ``_DEFAULT_NODES`` geometric nodes spanning the
        declared free prior (dust_T: 20-80 K).
    beta_grid : ndarray, optional
        Emissivity-index nodes [dimensionless]. If None, linear nodes spanning the declared
        free prior (dust_beta_ir: 1.0-2.5).
    alpha_mir_grid : ndarray, optional
        Mid-IR power-law slope nodes for casey2012 [dimensionless]. If None, linear nodes
        spanning the declared free prior (dust_alpha_mir: 1.0-3.0).
    lambda_0_um_grid : ndarray, optional
        Opacity pivot wavelength nodes for graybody and casey2012 [micron]. If None,
        geometric nodes spanning the declared free prior (dust_lambda_0_um: 50-500 um).

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

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function using NumPy.
    """
    if model == "pah_drude":
        preint = _build_grid_pah_drude(filter_waves, filter_trans, redshift)
        result = {"grid_phot": preint.phot, "axes": (), "_preint": preint}
        axis_params = AXIS_PARAMS_PAH
    elif model in _CONTINUUM_BUILDERS:
        axis_params = AXIS_PARAMS[model]
        supplied = {
            "dust_T": T_grid,
            "dust_beta_ir": beta_grid,
            "dust_alpha_mir": alpha_mir_grid,
            "dust_lambda_0_um": lambda_0_um_grid,
        }
        axes = tuple(
            _default_axis(name, _DEFAULT_NODES[model][name])
            if supplied[name] is None
            else supplied[name]
            for name in axis_params
        )
        preint = _CONTINUUM_BUILDERS[model](filter_waves, filter_trans, redshift, *axes)
        result = {
            "grid_phot": preint.phot,
            "axes": tuple(jnp.asarray(ax) for ax in axes),
            "_preint": preint,
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

    collapsed = _pin_axes(preint, axis_params, fixed)
    return {
        "grid_phot": collapsed.phot,
        "axes": remaining_axes,
        "_preint": collapsed,
        "_collapsed_axes": fixed,
    }


def _axis_coordinate(param_name: str, values):
    """Interpolation coordinate of an axis: ln of the value for log axes, else the value."""
    return jnp.log(values) if param_name in _LOG_AXIS_PARAMS else values


def _pin_axes(
    preint: PreintegratedGrid, axis_params: tuple[str, ...], fixed: dict[int, float]
) -> PreintegratedGrid:
    """Remove the pinned axes by PCHIP interpolation of ln(band flux) at the pinned values."""
    keep = [i for i in range(len(axis_params)) if i not in fixed]
    pinned = sorted(fixed)
    log_phot = jnp.log(jnp.maximum(preint.phot, _FLUX_FLOOR))
    moved = jnp.transpose(log_phot, (*pinned, *keep, log_phot.ndim - 1))
    pinned_axes = tuple(_axis_coordinate(axis_params[i], preint.axes[i]) for i in pinned)
    point = tuple(_axis_coordinate(axis_params[i], jnp.asarray(fixed[i])) for i in pinned)
    reduced = interp_nd_pchip(moved, pinned_axes, point)
    kept_axes = tuple(preint.axes[i] for i in keep)
    return dataclasses.replace(
        preint,
        phot=jnp.exp(reduced),
        axes=kept_axes,
        edges=tuple(preint.edges[i] for i in keep),
    )


def build_lookup(
    preint: dict,
    *,
    model: str = "modified_blackbody",
    free_param_names: tuple[str, ...] | None = None,
):
    """Build the runtime analytic dust photometry lookup from a preintegrated dict.

    Interpolates ln(band flux) with monotone cubic Hermite (PCHIP) in the coordinates
    (ln T, beta, alpha_mir, ln lambda_0) of the axes the model has. Tested accuracy: at
    the default node grids, <= 5e-4 relative to the exact closure at random points inside
    the declared priors (60-90, 250-500 and 750-950 um bands); the contract tests assert
    1e-3. The nodes are band integrals of the closed-form model on a rest grid of 0.01 um
    to 10 m, so no template interpolation enters the band integral. A band whose
    rest-frame red edge lies beyond 10 m is refused at build time with ``ValueError``;
    the blue side is clipped at 0.01 um.

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
        achieving <= 5e-4 relative tolerance at default node grids.

    References
    ----------
    .. [1] Casey, C. M., "Dusty star-forming galaxies at high redshift,"
           MNRAS, 425, 3094 (2012).

    Notes
    -----
    **JIT-compatible**: yes, the returned function uses ``jnp`` and PCHIP
    interpolation.

    **Gradient-safe**: yes, PCHIP kernel is fully differentiable.
    """
    axis_params_names = tuple(
        name
        for i, name in enumerate(AXIS_PARAMS[model])
        if i not in preint.get("_collapsed_axes", {})
    )
    grid_phot = preint["_preint"].phot
    axes = tuple(
        _axis_coordinate(name, ax) for name, ax in zip(axis_params_names, preint["_preint"].axes)
    )
    log_grid_phot = jnp.log(jnp.maximum(grid_phot, _FLUX_FLOOR))

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
