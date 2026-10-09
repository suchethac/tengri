# SPDX-License-Identifier: BSD-3-Clause
r"""Line-of-sight transmission of a library torus, read from the library's disc column.

On the untied path (no SKIRTOR-normalized disc) the central engine behind a
SKIRTOR or Fritz (2006) torus is dimmed by the library's own line-of-sight
disc: the ratio of the library disc at the viewing inclination to its
face-on value,

.. math::

    T_\nu(i) = \frac{{\rm disk}_{\rm lib}(\lambda; i)}{{\rm disk}_{\rm lib}(\lambda; 0)}.

This carries the library's scattered light by construction, unlike the
analytic screen of :mod:`~tengri.components.agn.blocks.torus_screen`, which
is kept only for a library without a disc column.

Two rules complete the ratio over the model's whole wavelength range.

* Below the library's shortest tabulated wavelength (10 A for both libraries,
  which covers the X-rays) the ratio is held at its value at that edge. No
  photoelectric absorption model is applied: the repository has none, so the
  X-ray transmission is the library's shortest-wavelength ratio, not a model
  of N_H.
* Type-1 sightlines (inside the dust-free polar cone) keep unit transmission.
  The Type-2 weight is the complement of the polar mask's Type-1 weight, at
  the same width. The ratio is blended on its logarithm,
  :math:`T = R\,\exp((w_2 - 1)\ln R)`, so a Type-2 sightline is exactly the
  library ratio and a Type-1 one is exactly unity.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tengri.components.agn.blocks.torus_screen import torus_screen_geometry
from tengri.components.agn.fritz import (
    FritzGrid,
    fritz_psy_from_cos_inc,
    load_fritz_default_grid,
)
from tengri.components.agn.polar_dust import _type1_mask
from tengri.components.agn.skirtor import (
    SKIRTORBundle,
    SKIRTORDiscAttenGrid,
    load_skirtor_disc_atten_grid,
    skirtor_disc_attenuation_from_grid,
)
from tengri.utils.grid_interp import interp_nd_pchip, resample_template
from tengri.utils.interpolation import edges_for_grid

#: Upper clip of a line-of-sight ratio; matches the SKIRTOR disc attenuation.
_RATIO_CLIP = 1.5

#: Floor on a ratio before its logarithm; a zero library ratio becomes this value.
_RATIO_FLOOR = 1.0e-30

#: Floor on a library disc value before its logarithm (a zero disc has no logarithm).
_DISK_FLOOR = 1.0e-35


def _hold_at_library_edges(wavelength, wave_grid):
    """Clamp the query wavelengths into the library's tabulated range.

    A ratio evaluated at the clamped wavelength is the edge value, so it is
    held constant beyond the library grid on both sides.
    """
    wg = jnp.asarray(wave_grid)
    return jnp.clip(jnp.asarray(wavelength), wg[0], wg[-1])


def _log_drift_carrier(value, log_drift):
    """``value`` carrying the derivative of ``log_drift``, and not its own derivative.

    The value is the library's own line-of-sight ratio. Its derivative is taken from the
    logarithm of the library disc instead. A quotient by a face-on disc of 1e-13 has a
    derivative of order 1/disk(0), and under a luminosity-scale cotangent that overflows
    float32 in the reverse pass; the logarithm has no such reciprocal.
    """
    held_value = jax.lax.stop_gradient(value)
    return held_value * jnp.exp(log_drift - jax.lax.stop_gradient(log_drift))


def skirtor_line_of_sight_ratio(
    grid: SKIRTORDiscAttenGrid,
    wavelength,
    *,
    tau: float,
    p: float,
    q: float,
    oa: float,
    radius: float,
    cos_inc: float,
) -> jnp.ndarray:
    r"""SKIRTOR ``disk(i)/disk(0)`` on the caller's grid, held at the library edges.

    Parameters
    ----------
    grid : SKIRTORDiscAttenGrid
        The SKIRTOR disc column.
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    tau, p, q, oa, radius, cos_inc : float
        SKIRTOR grid coordinates (edge-on 9.7 um depth, density indices,
        half-opening angle, radius ratio and cos of the inclination).

    Returns
    -------
    ndarray, shape (n_wave,)
        Line-of-sight ratio in [0, 1.5] [dimensionless].
    """
    held = _hold_at_library_edges(wavelength, grid.wave_grid)
    value = skirtor_disc_attenuation_from_grid(
        grid,
        held,
        agn_tau_skirtor=tau,
        agn_p_skirtor=p,
        agn_q_skirtor=q,
        agn_oa_skirtor=oa,
        agn_radius_ratio=radius,
        agn_cos_inc=cos_inc,
    )
    axes = tuple(jnp.asarray(a) for a in grid.axes)
    log_table = jnp.log(jnp.maximum(jnp.asarray(grid.disk), _DISK_FLOOR))

    def _log_disk(cos):
        return interp_nd_pchip(log_table, axes, (tau, p, q, oa, radius, cos))

    log_drift = resample_template(
        held,
        grid.wave_grid,
        _log_disk(cos_inc) - _log_disk(1.0),
        left=0.0,
        right=0.0,
    )
    return jnp.clip(_log_drift_carrier(value, log_drift), 0.0, _RATIO_CLIP)


def fritz_line_of_sight_ratio(
    grid: FritzGrid,
    wavelength,
    *,
    r_ratio: float,
    tau: float,
    beta: float,
    gamma: float,
    oa: float,
    cos_inc: float,
) -> jnp.ndarray:
    r"""Fritz ``disk(psi)/disk(89.99 deg)`` on the caller's grid, held at the library edges.

    Parameters
    ----------
    grid : FritzGrid
        The Fritz library, with its ``disk`` column.
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    r_ratio, tau, beta, gamma, oa : float
        Fritz grid coordinates.
    cos_inc : float
        Cosine of the inclination from the polar axis; the library's viewing
        elevation is :math:`\psi = 90^\circ - i`.

    Returns
    -------
    ndarray, shape (n_wave,)
        Line-of-sight ratio in [0, 1.5] [dimensionless].
    """
    axes = tuple(jnp.asarray(a) for a in grid.axes)
    table = jnp.asarray(grid.disk)
    log_table = jnp.log(jnp.maximum(table, _DISK_FLOOR))

    def _point(psy):
        return tuple(jnp.asarray(c) for c in (r_ratio, tau, beta, gamma, oa, psy))

    psy_i = fritz_psy_from_cos_inc(cos_inc)
    psy_0 = axes[5][-1]  # the face-on record, psi = 89.99 deg
    disk_i = interp_nd_pchip(table, axes, _point(psy_i))
    disk_0 = interp_nd_pchip(table, axes, _point(psy_0))
    live = disk_0 > 0.0
    ratio = jnp.where(live, disk_i / jnp.where(live, disk_0, 1.0), 0.0)
    wave_grid = jnp.asarray(grid.wave_grid)
    held = _hold_at_library_edges(wavelength, wave_grid)
    value = resample_template(held, wave_grid, ratio, left=0.0, right=0.0)
    log_drift_native = interp_nd_pchip(log_table, axes, _point(psy_i)) - interp_nd_pchip(
        log_table, axes, _point(psy_0)
    )
    log_drift = resample_template(held, wave_grid, log_drift_native, left=0.0, right=0.0)
    return jnp.clip(_log_drift_carrier(value, log_drift), 0.0, _RATIO_CLIP)


def _skirtor_disc_grid(library) -> SKIRTORDiscAttenGrid | None:
    """The SKIRTOR disc column: the threaded bundle's, else the packaged one.

    The bundle's ``disc_dust`` carries the same raw ``disk``, ``wave`` and ``axes``
    arrays as the packaged disc-attenuation grid, so it is repacked here without a
    second load. ``None`` for a grid with no separate disc column (v2).
    """
    if not isinstance(library, SKIRTORBundle):
        return load_skirtor_disc_atten_grid()
    dd = library.disc_dust
    if dd is None:
        return None
    axes = tuple(jnp.asarray(a) for a in dd.axes)
    return SKIRTORDiscAttenGrid(
        disk=jnp.asarray(dd.disk),
        wave_grid=jnp.asarray(dd.wave_grid),
        axes=axes,
        edges=tuple(edges_for_grid(a) for a in axes),
    )


def _fritz_disk_grid(library) -> FritzGrid | None:
    """The Fritz library with a disc column: the passed template, else the packaged one."""
    grid = library if isinstance(library, FritzGrid) else load_fritz_default_grid()
    return grid if grid.disk is not None else None


def library_torus_transmission(
    torus_block: str,
    wavelength,
    *,
    cos_inc: float,
    params: dict,
    library=None,
) -> jnp.ndarray | None:
    r"""Transmission of a library torus on the untied path, blended by sightline type.

    Parameters
    ----------
    torus_block : str
        ``"skirtor"`` or ``"fritz"``.
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    cos_inc : float
        Cosine of the inclination from the polar axis; 1 = face-on.
    params : dict
        The runner's parameter dict (torus geometry and grid coordinates).
    library : SKIRTORBundle or FritzGrid, optional
        The torus's pre-loaded template library, threaded through JIT as an
        argument so its disc column is not baked into the graph. When omitted,
        the packaged library is read from its cached loader.

    Returns
    -------
    ndarray, shape (n_wave,), or None
        ``T = R**w2`` with ``R = disk(i)/disk(0)`` from the library and ``w2`` the
        Type-2 weight at the torus's opening angle; linear, ``(1 - w2) + w2 R``,
        where ``R = 0``. ``None`` when the torus has no library disc column (SKIRTOR
        v2 grid), so the caller keeps the analytic screen.

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes in float32 as well. The value of
    each ratio is the library's own; its derivative is carried by the logarithm of
    the disc column (:func:`_log_drift_carrier`), so no quotient by a near-zero
    face-on disc enters the reverse pass.
    """
    oa_deg, _tau_v, _ = torus_screen_geometry(torus_block, params)
    if torus_block == "skirtor":
        grid = _skirtor_disc_grid(library)
        if grid is None:
            return None
        ratio = skirtor_line_of_sight_ratio(
            grid,
            wavelength,
            tau=params.get("agn_tau_skirtor", 7.0),
            p=params.get("agn_p_skirtor", 1.0),
            q=params.get("agn_q_skirtor", 1.0),
            oa=params.get("agn_oa_skirtor", 40.0),
            radius=params.get("agn_radius_ratio", 20.0),
            cos_inc=cos_inc,
        )
    elif torus_block == "fritz":
        fgrid = _fritz_disk_grid(library)
        if fgrid is None:
            return None
        ratio = fritz_line_of_sight_ratio(
            fgrid,
            wavelength,
            r_ratio=params.get("agn_fritz_r_ratio", 60.0),
            tau=params.get("agn_fritz_tau", 1.0),
            beta=params.get("agn_fritz_beta", -0.5),
            gamma=params.get("agn_fritz_gamma", 4.0),
            oa=params.get("agn_fritz_oa", 60.0),
            cos_inc=cos_inc,
        )
    else:
        return None
    type2 = 1.0 - _type1_mask(cos_inc, oa_deg)
    # T = R**w2 = R exp((w2 - 1) ln R): the Type-2 weight scales the library's line-of-sight
    # depth, so a Type-2 sightline (w2 = 1) is exactly R. A linear blend (1 - w2) + w2 R
    # would leave the Type-1 sigmoid tail on a Type-2 sightline (3e-3 absolute on a 0.036
    # ratio). It is written exp(w2 ln R), not R exp((w2 - 1) ln R), so the float32 gradient
    # has no R * (1/R) cancellation. R = 0 has no logarithm: there the blend is linear, so a
    # Type-1 sightline is within w2 of unity.
    live = ratio > 0.0
    safe = jnp.where(live, ratio, 1.0)
    log_blend = jnp.exp(type2 * jnp.log(safe))
    linear_blend = (1.0 - type2) + type2 * ratio
    return jnp.where(live, log_blend, linear_blend)
