# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapter for the composable AGN runner.

Implements :class:`tengri.forward.precompute.protocol.PrecomputeModule` for
any recipe of registered AGN blocks. The composable runner is exact + JIT-
composable today; this module adds the missing third mode (precomputed
photometry lookup) so single-galaxy interactive fits don't pay the full
block-compile cost when the recipe is template-heavy.

How it works
------------
For a fixed :class:`Recipe` (a 5-tuple of block selectors) and a set of
"fast axes" (params the user wants to vary in fits), this module:

1. Loads templates **once** outside the JIT trace, hoisted into a closure
   so blocks don't reload them inside the grid loop.
2. Evaluates :func:`composable_agn_l_nu` on the outer product of the axis
   grids, holding non-axis params at their defaults (or user-provided
   fixed values).
3. Pre-integrates the resulting spectra through the user's filter set via
   the shared :func:`precompute_template_photometry` helper (same path
   used by :mod:`qsogen_precompute`, :mod:`skirtor_precompute`, etc.).
4. Returns the standard ``{grid_phot, axes, _preint, _collapsed_axes?}``
   dict consumed by :func:`build_template_photometry_lookup` at runtime.

Auto-collapse
-------------
Any axis whose corresponding parameter is :class:`~tengri.parameters.priors.Fixed`
in the user's :class:`Parameters` is collapsed at construction time via
:func:`slice_fixed_axes`, mirroring how :mod:`qsogen_precompute` handles
``Fixed(agn_plslp1)``.

References
----------
Plan: ``~/.claude/plans/enumerated-watching-rainbow.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.agn._params import DEFAULT_AGN_LOG_LBOL
from tengri.components.agn.blocks.recipe import Recipe
from tengri.components.agn.blocks.runner import composable_agn_l_nu
from tengri.components.agn.grahsp.templates import load_grahsp_templates
from tengri.forward.precompute.templates import (
    build_template_photometry_lookup,
    collapse_fixed_axes,
    precompute_template_photometry,
)
from tengri.utils.grid_interp import (
    edges_for_grid,
    interp_nd_triweight,
)

__all__ = ["AXIS_PARAMS", "build_lookup", "precompute"]

#: Placeholder. Unlike per-model precompute modules, the composable
#: recipe determines its own axes via ``Recipe.axis_params``. The
#: ``PrecomputeModule`` Protocol requires this attribute, so we expose
#: an empty tuple here as the contract baseline; callers always pass
#: ``axis_grids`` explicitly to :func:`precompute`.
AXIS_PARAMS: tuple[str, ...] = ()


def _build_template_state(recipe: Recipe) -> dict[str, Any] | None:
    """Load any templates the recipe's blocks need; return runner-state dict.

    Only the GRAHSP family currently has block-loaded templates; other
    blocks either are analytic (powerlaw, SBPL, simple torus, SMC) or load
    grids via dedicated precompute paths (SKIRTOR, Silva04, CAT3D). When the
    recipe doesn't touch GRAHSP we return ``None`` (no hoist needed).
    """
    selectors = recipe.as_selector_dict()
    if any(
        name == "grahsp" or name == "grahsp_sbpl" or name == "grahsp_biatten"
        for name in selectors.values()
    ):
        return {"grahsp": load_grahsp_templates()}
    return None


def _evaluate_recipe_on_grid(
    wave_rest: np.ndarray,
    recipe: Recipe,
    axis_grids: Mapping[str, np.ndarray],
    fixed_values: Mapping[str, float] | None,
    agn_log_lbol_default: float,
) -> np.ndarray:
    r"""Evaluate ``composable_agn_l_nu`` on the outer product of axis grids.

    Returns a 4-D array shaped ``(*grid_dims, n_wave)`` ready for the
    template-photometry helper.
    """
    fixed_values = dict(fixed_values or {})
    template_state = _build_template_state(recipe)
    axis_names = recipe.axis_params or tuple(axis_grids.keys())
    grids = [np.asarray(axis_grids[name], dtype=np.float64) for name in axis_names]
    grid_shape = tuple(g.size for g in grids)

    wave_jax = jnp.asarray(wave_rest)

    @jax.jit
    def _one_point(**call_params):
        # Convert L_nu (composable_agn_l_nu output) back to L_lambda for
        # the standard precompute_template_photometry helper, which prefers
        # L_lambda for the L_nu integration convention.
        l_nu = composable_agn_l_nu(
            wave_jax,
            template_state=template_state,
            **recipe.as_selector_dict(),
            **call_params,
        )
        return l_nu

    # Iterate over the cartesian product of axis grids. The inner call is
    # JITed once and reused; the Python loop is over axis-grid indices,
    # which is fine because total grid size is small (typical <100 points).
    n_total = int(np.prod(grid_shape))
    indices = np.indices(grid_shape).reshape(len(grid_shape), n_total).T
    spectra = np.zeros((n_total, wave_rest.size), dtype=np.float64)

    base_call = dict(fixed_values)
    base_call.setdefault("agn_log_lbol", agn_log_lbol_default)
    base_call.setdefault("agn_lum_ratio", 1.0)

    for flat_idx, multi_idx in enumerate(indices):
        call = dict(base_call)
        for name, grid, idx in zip(axis_names, grids, multi_idx, strict=True):
            call[name] = float(grid[idx])
        spectra[flat_idx] = np.asarray(_one_point(**call))

    return spectra.reshape(*grid_shape, wave_rest.size)


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any | None = None,
    *,
    recipe: Recipe,
    axis_grids: Mapping[str, np.ndarray],
    fixed_values: Mapping[str, float] | None = None,
    wave_rest: np.ndarray | None = None,
    agn_log_lbol_default: float = DEFAULT_AGN_LOG_LBOL,
) -> dict:
    r"""Build a preintegrated photometry grid for a composable AGN recipe.

    Parameters
    ----------
    filter_waves: list[ndarray]
        Per-filter wavelength arrays [Å, observed frame].
    filter_trans: list[ndarray]
        Per-filter transmission curves.
    redshift: float
        Source redshift.
    parameters: Parameters or None, optional
        Used to detect ``Fixed`` axes for auto-collapse. ``None`` skips
        auto-collapse and returns the full grid.
    recipe: Recipe
        Block selectors + axis-params for this fit. Constructed via
        :meth:`Recipe.from_parameters` or :meth:`Recipe.from_selectors`.
    axis_grids: Mapping[str, ndarray]
        ``{param_name: grid_values}`` per fast axis. Keys must match
        ``recipe.axis_params`` (or, when that's empty, be supplied in order).
    fixed_values: Mapping[str, float], optional
        Static values for non-axis params (block kwargs). Overrides the
        registry defaults baked into each block.
    wave_rest: ndarray, optional
        Rest-frame wavelength grid the spectra are evaluated on. Default:
        ``np.logspace(2.0, 6.0, 1500)`` (100 Å to 1e6 Å, log-spaced).
    agn_log_lbol_default: float, optional
        Default ``agn_log_lbol`` when not in ``fixed_values`` or
        ``axis_grids``. Defaults to the declared
        ``agn_log_lbol`` default.

    Returns
    -------
    dict
        Keys: ``grid_phot`` (preintegrated photometry array),
        ``axes`` (tuple of axis arrays), ``_preint`` (:class:`PreintegratedGrid`),
        optionally ``_collapsed_axes`` when any axis was auto-collapsed.

    Notes
    -----
    Build time scales with ``prod(axis_grid_sizes)``; runtime lookup is
    independent of grid size.
    """
    axis_names = recipe.axis_params or tuple(axis_grids.keys())
    if not axis_names:
        raise ValueError(
            "composable_precompute requires at least one axis_param; pass "
            "axis_grids={'agn_log_lbol': np.linspace(43, 47, 5)} or similar."
        )

    if wave_rest is None:
        wave_rest = np.logspace(2.0, 6.0, 1500, dtype=np.float64)
    else:
        wave_rest = np.asarray(wave_rest, dtype=np.float64)

    spectra = _evaluate_recipe_on_grid(
        wave_rest, recipe, axis_grids, fixed_values, agn_log_lbol_default
    )

    preint = precompute_template_photometry(
        templates=spectra,
        wave_rest=wave_rest,
        filter_waves=[np.asarray(fw, dtype=np.float64) for fw in filter_waves],
        filter_trans=[np.asarray(ft, dtype=np.float64) for ft in filter_trans],
        axes=tuple(np.asarray(axis_grids[name], dtype=np.float64) for name in axis_names),
        redshift=redshift,
        dl_cm=1.0,
        # composable_agn_l_nu returns L_nu; tell the helper not to convert
        # via L_nu = L_lambda * lambda^2 / c again.
        energy_normalize=False,
        units="lnu",
    )

    result: dict[str, Any] = {
        "grid_phot": preint.phot,
        "axes": tuple(jnp.asarray(axis_grids[name]) for name in axis_names),
        "_preint": preint,
        "_axis_names": tuple(axis_names),
    }

    # Auto-collapse Fixed axes (mirror qsogen_precompute).
    collapsed, remaining_axes, fixed_indices = collapse_fixed_axes(
        preint,
        axis_names,
        parameters,
        defaults=fixed_values,
        origin="composable_precompute",
    )
    if not fixed_indices:
        return result

    return {
        "grid_phot": collapsed.phot,
        "axes": remaining_axes,
        "_preint": collapsed,
        "_axis_names": tuple(axis_names),
        "_collapsed_axes": fixed_indices,
    }


class ComposableLookup:
    """Callable wrapper exposing the precompute's axis names statically.

    The :func:`build_lookup` callable from this module is bundled with the
    list of axis parameter names so the JIT-compiled kernel can extract
    the right values from its local namespace at trace time. Existing
    callers continue to use it as a callable.

    Attributes
    ----------
    lookup: callable
        The JIT-compiled ``(scale, *free_axis_values) -> photometry``
        function from :func:`build_template_photometry_lookup`.
    axis_names: tuple of str
        Names of the remaining (non-collapsed) axis parameters, in the
        order ``lookup`` expects.
    """

    # Note: no __slots__. ``jax.jit(ComposableLookup_instance)`` weak-refs
    # the wrapper internally; __slots__ would block that.

    def __init__(self, lookup, axis_names: tuple[str, ...]) -> None:
        self.lookup = lookup
        self.axis_names = axis_names

    def __call__(self, *args, **kwargs):
        return self.lookup(*args, **kwargs)


def build_lookup(preint: dict, *, free_param_names: tuple[str, ...] | None = None):
    r"""Build the runtime photometry lookup from a preintegrated dict.

    Returns a :class:`ComposableLookup` whose ``__call__`` matches the
    shared :func:`build_template_photometry_lookup` signature::

        fn(scale, *free_axis_values) -> ndarray, shape (n_filters,)

    The ``axis_names`` attribute lists the remaining free-axis parameter
    names so downstream consumers (e.g. the SEDModel kernel) can pass the
    right values positionally at JIT trace time.

    Parameters
    ----------
    preint: dict
        Output of :func:`precompute`.
    free_param_names: tuple of str, optional
        Names of the remaining free axes (after auto-collapse). Currently
        unused but accepted for API symmetry with sibling modules.
    """
    del free_param_names

    # Reconstruct the surviving axis-name list from the preint dict.
    # ``precompute`` stored the full list under ``_axis_names`` and any
    # auto-collapsed indices under ``_collapsed_axes`` (keys = axis index).
    full_names: tuple[str, ...] = preint.get("_axis_names", ())
    collapsed = preint.get("_collapsed_axes") or {}
    surviving_names = tuple(name for i, name in enumerate(full_names) if i not in collapsed)

    if not collapsed:
        return ComposableLookup(
            build_template_photometry_lookup(preint["_preint"]),
            axis_names=surviving_names,
        )

    grid_phot = preint["grid_phot"]
    axes = preint["axes"]
    edges = tuple(edges_for_grid(ax) for ax in axes) if axes else ()

    @jax.jit
    def _lookup_collapsed(scale, *free_axis_values):
        """Triweight lookup on the remaining free axes."""
        if not axes:
            return scale * grid_phot
        normed = interp_nd_triweight(grid_phot, axes, edges, tuple(free_axis_values))
        return scale * normed

    return ComposableLookup(_lookup_collapsed, axis_names=surviving_names)
