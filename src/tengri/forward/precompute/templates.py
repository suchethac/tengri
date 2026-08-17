# SPDX-License-Identifier: BSD-3-Clause
"""Generic template-based precompute helpers.

Thin wrappers over :mod:`tengri.forward.precompute.grid`
(:func:`~tengri.forward.precompute.grid.preintegrate_grid` +
:func:`~tengri.forward.precompute.grid.interp_nd_triweight`) that handle the
three steps every template adapter needs: L_λ → L_ν unit conversion,
energy-normalization for templates scaled by L_absorbed / L_bol at runtime,
and a standard JIT lookup closure.

Component-specific adapters (``components/dust/dust_emission_precompute.py``,
``components/agn/skirtor_precompute.py``, etc.) should call these functions
rather than talk to ``preintegrate_grid`` directly, so their shapes remain
consistent under the Precompute Protocol.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from tengri.config.exceptions import DeadPrecomputeAxisWarning
from tengri.utils.grid_interp import (
    PreintegratedGrid,
    PreintegratedLines,
    interp_nd_triweight,
    preintegrate_grid,
    slice_fixed_axes,
)
from tengri.utils.physics_constants import AA_TO_CM as _AA_TO_CM, C_CGS as _C_CGS

if TYPE_CHECKING:  # pragma: no cover - annotations only, avoids a build-time cycle
    from tengri.parameters.parameters import Parameters


def collapse_fixed_axes(
    preint: PreintegratedGrid | PreintegratedLines,
    axis_params: Sequence[str],
    parameters: Parameters | None,
    *,
    defaults: Mapping[str, float] | None = None,
    internal_axes: frozenset[str] | None = None,
    origin: str = "precompute",
) -> tuple[PreintegratedGrid | PreintegratedLines, tuple[Any, ...], dict[int, float]]:
    """Collapse every grid axis whose governing parameter is Fixed.

    The auto-collapse step every ``*_precompute.py`` module performs after
    building its grid: read the model's Fixed parameter values, match them
    against the module's ``AXIS_PARAMS`` (one name per axis, in axis order),
    and triweight-interpolate those axes away so runtime interpolation is
    cheaper and the stored grid smaller.

    Parameters
    ----------
    preint : PreintegratedGrid or PreintegratedLines
        The freshly built grid, before collapse.
    axis_params : sequence of str
        Parameter name governing each grid axis, **in axis order**. Position
        ``i`` in this sequence must be axis ``i`` of ``preint``; the whole
        mechanism is positional, so a reordering silently collapses the wrong
        axis at another parameter's value.
    parameters : Parameters or None
        The model's parameter specification. ``None`` skips collapse entirely
        (the caller has no way to know which axes are Fixed).
    defaults : mapping of str to float, optional
        Fallback values for axis parameters that are neither declared ``Fixed``
        nor free. Used by the components that carry their own axis defaults
        (GRAHSP) or accept caller-supplied ones (the composable AGN block). A
        name absent from both the model and this mapping leaves its axis alone.
    internal_axes : frozenset of str, optional
        Axis labels that are deliberately internal grid-axis constructs, not
        user-facing parameters. Names in this set are skipped silently (no
        warning, never collapsed). Used for grid axes like ``log_age`` in
        CLOUDY adapters or ``HbFrac`` in CB19 that are internal bookkeeping
        and not meant to be user parameters. Default: None. See issue #1827.
    origin : str, optional
        Module or component name, used only in the ``DeadPrecomputeAxisWarning``
        message so the report names the declaration that needs fixing.

    Returns
    -------
    preint_out : PreintegratedGrid or PreintegratedLines
        The collapsed grid, or ``preint`` unchanged when nothing collapsed.
    remaining_axes : tuple
        Axes surviving the collapse, in order — the axes a runtime lookup must
        still be queried at. Equal to ``preint_out.axes``.
    collapsed : dict[int, float]
        Axis index to the value it was collapsed at, empty when nothing
        collapsed. Callers branch on this to decide whether to rebuild their
        result dict.

    Warns
    -----
    DeadPrecomputeAxisWarning
        Once per axis name that is neither a valid parameter nor in
        ``internal_axes`` nor in ``defaults``. Reports the origin/adapter,
        the axis name, and the two remedies (declare the parameter; or declare
        the axis internal). When nothing collapsed *and* every name is invalid,
        a module's ``defaults`` still collapse its axes and is working, so
        names also in defaults stay silent (issue #1827).

    Raises
    ------
    ValueError
        When ``axis_params`` and ``preint.axes`` disagree in length, or the
        grid's array rank disagrees with its axis count, and a collapse was
        about to happen. Both make the positional axis index meaningless, and
        contracting the wrong axis is a silently wrong SED rather than a
        crash — so this refuses instead of proceeding.

    Notes
    -----
    **JIT-compatible**: no — build-time orchestration over NumPy/host values.

    This replaced a byte-identical block copied into eleven precompute modules
    (issue #1738). The duplication is why the mismatch checks above did not
    exist: no single copy was the obvious place to put them, and six of the
    eleven declared axis names that no ``Parameters`` object can ever contain,
    so their advertised auto-collapse had never once fired. Issue #1827
    resolved the third cause: internal grid-axis labels via ``internal_axes``.
    """
    no_collapse: tuple[Any, ...] = tuple(preint.axes)
    if parameters is None or not axis_params:
        return preint, no_collapse, {}

    fixed_values = parameters.get_fixed_values()
    free_names = set(getattr(parameters, "free_params", ()) or ())
    defaults = defaults or {}
    internal_axes = internal_axes or frozenset()

    collapsed_at: dict[int, float] = {}
    for i, pname in enumerate(axis_params):
        if pname in fixed_values:
            collapsed_at[i] = float(fixed_values[pname])
        elif pname not in free_names and pname in defaults:
            collapsed_at[i] = float(defaults[pname])

    # Warn once per dead name (not a valid parameter, not internal, not in defaults).
    # This replaces the old logic that warned only when nothing collapsed AND every
    # name was invalid (issue #1827).
    valid = getattr(parameters, "valid_param_names", None)
    if isinstance(valid, (set, frozenset)) and valid:
        for pname in axis_params:
            if pname not in valid and pname not in internal_axes and pname not in defaults:
                warnings.warn(
                    f"{origin}: grid-axis label '{pname}' is not a valid "
                    f"parameter name and not declared internal. The axis can never "
                    f"be collapsed by declaration. Either: (1) declare '{pname}' "
                    f"as a parameter in the component's parameter spec, or "
                    f"(2) list it in internal_axes to skip it silently. "
                    f"See issue #1827.",
                    DeadPrecomputeAxisWarning,
                    stacklevel=2,
                )

    if not collapsed_at:
        return preint, no_collapse, {}

    _check_axis_alignment(preint, axis_params, origin)

    out = slice_fixed_axes(preint, collapsed_at)
    return out, tuple(out.axes), collapsed_at


def _check_axis_alignment(
    preint: PreintegratedGrid | PreintegratedLines,
    axis_params: Sequence[str],
    origin: str,
) -> None:
    """Refuse to collapse a grid whose axis count contradicts its declaration.

    ``collapse_fixed_axes`` maps a name's position in ``axis_params`` straight
    onto an axis index, and :func:`slice_fixed_axes` contracts that axis. If the
    two disagree the contraction still succeeds whenever the shapes happen to
    line up, and returns an SED built from the wrong axis — no exception, no NaN.
    """
    n_axes = len(preint.axes)
    if len(axis_params) != n_axes:
        raise ValueError(
            f"{origin}: AXIS_PARAMS declares {len(axis_params)} axes "
            f"{list(axis_params)} but the grid has {n_axes}. The axis index is "
            f"positional, so collapsing would contract an axis the name does not "
            f"govern and return a silently wrong SED (#1738)."
        )

    if isinstance(preint, PreintegratedGrid):
        # phot is (*grid_dims, n_filters): one trailing filter axis.
        n_grid_dims = np.ndim(preint.phot) - 1
        if n_grid_dims != n_axes:
            raise ValueError(
                f"{origin}: the preintegrated grid carries {n_axes} axes "
                f"{list(axis_params)} but its photometry array has "
                f"{n_grid_dims} grid dimensions. Collapsing axis i would "
                f"contract a different array dimension — for the last declared "
                f"axis, the filter dimension itself (#1738)."
            )


def precompute_template_photometry(
    templates: np.ndarray,
    wave_rest: np.ndarray,
    filter_waves: list,
    filter_trans: list,
    axes: tuple[np.ndarray, ...],
    redshift: float = 0.0,
    dl_cm: float = 1.0,
    energy_normalize: bool = True,
    units: str = "lnu",
) -> PreintegratedGrid:
    """Preintegrate any template grid through photometric filters.

    Generic entry point for template-based components. Handles unit
    conversion (L_λ → L_ν if needed) and delegates to
    :func:`~tengri.forward.precompute.grid.preintegrate_grid`.

    Parameters
    ----------
    templates : ndarray
        Shape ``(*grid_dims, n_wave)``.  Template spectra.
    wave_rest : ndarray
        Shape ``(n_wave,)``.  Rest-frame wavelengths [Ångström].
    filter_waves : list[ndarray]
        Per-filter wavelength arrays (observed frame).
    filter_trans : list[ndarray]
        Per-filter transmission curves.
    axes : tuple[ndarray, ...]
        One array per grid dimension (for triweight interpolation).
    redshift : float
        Source redshift (0 for rest-frame templates).
    dl_cm : float
        Luminosity distance [cm] (1 for normalized templates).
    energy_normalize : bool
        Normalize each template to unit bolometric luminosity before
        integration.  Required for templates scaled by L_absorbed or
        L_bol at runtime (DL07, Dale, SKIRTOR, etc.).  Default True.
    units : str
        ``"lnu"`` if templates are in L_ν [erg/s/Hz], or ``"llam"``
        if templates are in L_λ [erg/s/Å].  When ``"llam"``, converts
        to L_ν via L_ν = L_λ × λ²/c before integration.

    Returns
    -------
    PreintegratedGrid
        Precomputed filter-integrated photometry with triweight axes/edges.

    Notes
    -----
    Unit conversion from L_λ to L_ν (when units="llam") is exact: L_ν = L_λ × λ²/c.
    Energy normalization (when energy_normalize=True) ensures templates scaled by
    L_absorbed or L_bol at runtime are correctly normalized.
    """
    templates = np.asarray(templates)
    wave_rest = np.asarray(wave_rest)

    if units == "llam":
        wave_cm = wave_rest * _AA_TO_CM
        templates = templates * (wave_cm**2) / _C_CGS

    return preintegrate_grid(
        templates=templates,
        wave_rest=wave_rest,
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=redshift,
        dl_cm=dl_cm,
        axes=tuple(np.asarray(ax) for ax in axes),
        energy_normalize=energy_normalize,
    )


def build_template_photometry_lookup(preint: PreintegratedGrid):
    """Build a JIT-compiled lookup from a preintegrated template grid.

    Uses triweight interpolation for C²-continuous gradients.  The
    returned function takes the grid parameters + a scalar scaling
    factor and returns photometry in (n_filters,).

    Parameters
    ----------
    preint : PreintegratedGrid
        Output of :func:`precompute_template_photometry`.

    Returns
    -------
    callable
        ``(scale, *grid_params) -> array (n_filters,)`` where *grid_params*
        are scalar query points along each axis.

    Notes
    -----
    **JIT-compatible**: yes — returned function uses triweight interpolation via
    ``interp_nd_triweight``, providing C²-continuous gradients for inference.
    """
    phot = preint.phot
    axes = preint.axes
    edges = preint.edges

    @jax.jit
    def lookup(scale, *grid_params):
        """Interpolate template photometry at given grid parameters and scale by luminosity.

        Parameters
        ----------
        scale : float
            Luminosity scaling factor (L_absorbed, L_bol, etc.) [erg/s].
        *grid_params : float
            Per-axis query points for triweight interpolation (one per free grid axis).

        Returns
        -------
        array, shape (n_filters,)
            Photometry in filter bands [erg/s/Hz].

        Notes
        -----
        **JIT-compatible**: yes — uses ``jnp`` and ``interp_nd_triweight`` primitives.
        """
        normed = interp_nd_triweight(phot, axes, edges, grid_params)
        return scale * normed

    return lookup
