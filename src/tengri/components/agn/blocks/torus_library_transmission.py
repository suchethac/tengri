# SPDX-License-Identifier: BSD-3-Clause
r"""Line-of-sight obscuration of a library torus, read from the library's disc column.

On the untied path (no SKIRTOR-normalized disc) the central engine behind a SKIRTOR or
Fritz (2006) torus is dimmed by the library's own line of sight. The library stores the
disc of each record divided by a per-record scale ``norm``, so the physical
inclination ratio carries it,

.. math::

    R_n(\lambda; i) = \frac{{\rm disk}_i(\lambda)\,{\rm norm}_i}
                          {{\rm disk}_0(\lambda)\,{\rm norm}_0},

with the subscript 0 the face-on record. Without ``norm`` the SKIRTOR ratio is 11 per cent
off the disc anisotropy it must reduce to, and the Fritz ratio exceeds unity.

:math:`R_n` contains the library's disc anisotropy :math:`\eta(i)` (the disc of the library
is not isotropic) as well as the obscuration. CIGALE's ``skirtor2016`` takes the viewing
angle from this library alone, with no separate Type-1/Type-2 switch; the obscuration
alone is therefore

.. math::

    T(\lambda; i) = \frac{R_n(\lambda; i)}{\eta(i)},
    \qquad
    \eta_{\rm SKIRTOR} = \frac{\cos i\,(1 + 2\cos i)}{3},
    \qquad
    \eta_{\rm Fritz} = 1,

evaluated at the actual inclination with node-exact PCHIP in :math:`\ln R_n`. The Fritz
library is isotropic in its disc (``docs/model_reference/agn.md``).

Two forms are returned, because :math:`\eta \to 0` at :math:`i = 90^\circ`.

* ``disc_factor`` is the factor for a disc that carries no inclination law of its own: the
  product with the repository's disc law :math:`2\cos i` (``agn_log_lbol`` is the accretion
  power, #2678),
  :math:`2\cos i\,R_n/\eta = 6 R_n/(1 + 2\cos i)` (SKIRTOR) or :math:`2\cos i\,R_n`
  (Fritz). It is written without the quotient, so it is finite and differentiable at
  :math:`i = 90^\circ`.
* ``transmission`` is :math:`T` for a source that is isotropic (broad lines, FeII, the
  corona). Its :math:`\eta` is taken at :math:`\cos i` smoothly floored at
  :data:`_COS_FLOOR` with a softplus of width 0.01, which equals :math:`T` to under
  :math:`10^{-3}` per cent for :math:`i \le 80^\circ` and keeps the value finite beyond.
  This floor is a numerical choice, not library physics: below it the library's edge-on
  disc is scattered light only, and :math:`R_n/\eta` has no meaning.

The analytic screen of :mod:`~tengri.components.agn.blocks.torus_screen` is kept only for a
torus without a disc column (the SKIRTOR v2 grid).

Where the library's face-on disc is zero (the long-wavelength tail) there is no library
sightline: the transmission is unity and the disc factor is :math:`2\cos i`.

Below the library's shortest tabulated wavelength (10 A for both libraries) the X-rays are
photoelectrically absorbed (see :func:`library_torus_sightline`).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.agn.fritz import (
    FritzGrid,
    fritz_psy_from_cos_inc,
    load_fritz_default_grid,
)
from tengri.components.agn.skirtor import (
    SKIRTORBundle,
    SKIRTORDiscAttenGrid,
    load_skirtor_disc_atten_grid,
)
from tengri.components.xray.xray import wabs_cross_section
from tengri.config.exceptions import TengriIOError
from tengri.utils.grid_interp import interp_nd_pchip, resample_template
from tengri.utils.interpolation import edges_for_grid
from tengri.utils.physics_constants import HC_KEV_ANGSTROM
from tengri.utils.scale import representable_floor

#: Hydrogen column per magnitude of V-band extinction [cm^-2 mag^-1] (Guver & Ozel 2009).
NH_PER_AV = 2.21e21

#: V-band wavelength [Angstrom] at which the sightline's own extinction is read.
_V_WAVE = 5500.0

#: Upper clip of a line-of-sight ratio.
_RATIO_CLIP = 1.5

#: Floor on a library table value before its logarithm.
_TABLE_FLOOR = 1.0e-35

#: A face-on disc below this many e-folds above the floor is treated as absent.
_LIVE_MARGIN = 1.0

#: Smooth floor on cos i [dimensionless] in the anisotropy that divides an isotropic source's
#: ratio: cos(85 deg). Numerical, see the module docstring.
_COS_FLOOR = 0.0871557427476582

#: Softening width of that floor [dimensionless].
_COS_FLOOR_WIDTH = 0.01


#: Per-axis extrapolation of the library lookups, by torus block. Only the inclination axis
#: has a numerical gap to its prior (SKIRTOR: stored cos 90 deg = 6.1e-17 against a prior
#: reaching 0; Fritz: stored psy 0.001 and 89.99 against the prior's 0 and 90 deg). Every
#: other axis is either fixed at a node or has its prior on the stored axis (census in
#: tests/contract/test_no_zero_gradient_outside_node_hull.py). Owner ruling 12.
_LOOKUP_EXTRAPOLATE = {
    "skirtor": (False, False, False, False, False, True),
    "fritz": (False, False, False, False, False, True),
}


class BelowLibraryIndex(NamedTuple):
    """The build-time X-ray absorption data of the grid samples shortward of a library's grid.

    Built once from the concrete rest-frame grid of a model. The MM83 cross-section depends
    only on wavelength, so it is evaluated here, on the concrete grid, and carried as a
    constant: inside a jitted kernel, where the grid is a traced value, the absorption is
    then ``exp(-sigma N_H)`` with no cross-section evaluation. ``n_wave`` is the length of the
    grid these arrays belong to; a caller whose grid has another length takes the dense path.

    Attributes
    ----------
    below : ndarray of bool, shape (n_wave,)
        ``True`` at the samples with ``grid < wave_min`` (the library's first node).
    sigma : ndarray, shape (n_wave,)
        The MM83 photoelectric cross-section [cm^2] at each sample (zero where ``below`` is
        ``False``).
    n_wave : int
        Length of the build grid.
    """

    below: np.ndarray
    sigma: np.ndarray
    n_wave: int


def below_library_index(wavelength, torus_block: str) -> BelowLibraryIndex | None:
    """:class:`BelowLibraryIndex` of a concrete grid, or ``None`` (traced grid or no library).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Concrete rest-frame wavelength [Angstrom], the grid a jitted kernel will use.
    torus_block : str
        ``"skirtor"`` or ``"fritz"``.

    Returns
    -------
    BelowLibraryIndex or None
        ``None`` when the wavelength is traced or the torus has no packaged library.

    Notes
    -----
    **JIT-compatible**: construction-time only; call it on the concrete grid, outside a trace.
    """
    idx = _below_library_index(wavelength, torus_block)
    if idx is None:
        return None
    n_wave = int(np.shape(wavelength)[0])
    below = np.zeros(n_wave, dtype=bool)
    below[idx] = True
    sigma = np.zeros(n_wave)
    if idx.size:
        energy = HC_KEV_ANGSTROM / jnp.asarray(wavelength)[idx]
        sigma[idx] = np.asarray(wabs_cross_section(energy))
    return BelowLibraryIndex(below=below, sigma=sigma, n_wave=n_wave)


class LibrarySightline(NamedTuple):
    """The library's line of sight on the caller's wavelength grid.

    Attributes
    ----------
    ratio : ndarray, shape (n_wave,)
        :math:`R_n(\\lambda; i)`, the library's normalized inclination ratio [dimensionless].
    transmission : ndarray, shape (n_wave,)
        :math:`T = R_n/\\eta`, for an isotropic source [dimensionless].
    disc_factor : ndarray, shape (n_wave,)
        :math:`2\\cos i\\,R_n/\\eta` for a disc without an inclination law
        [dimensionless].
    """

    ratio: jnp.ndarray
    transmission: jnp.ndarray
    disc_factor: jnp.ndarray


def _hold_at_library_edges(wavelength, wave_grid):
    """Clamp the query wavelengths into the library's tabulated range.

    A ratio evaluated at the clamped wavelength is the edge value, so it is
    held constant beyond the library grid on both sides.
    """
    wg = jnp.asarray(wave_grid)
    return jnp.clip(jnp.asarray(wavelength), wg[0], wg[-1])


def _normalized_ratio(
    log_disk, log_norm, axes, point, point_face, wave_grid, wavelength, extrapolate
):
    """:math:`R_n` on ``wavelength`` and the mask of wavelengths where the face-on disc exists.

    Node-exact PCHIP of the logarithms of the disc and of ``norm`` at the viewing record and
    at the face-on record; the ratio is the exponential of the difference, so no quotient by a
    near-zero face-on disc enters the reverse pass. ``extrapolate`` is the per-axis flag of
    :func:`interp_nd_pchip`: the numerical gaps of the prior (see :data:`_LOOKUP_EXTRAPOLATE`)
    continue the end cubic rather than clip to a zero gradient.
    """
    point = tuple(jnp.asarray(c) for c in point)
    point_face = tuple(jnp.asarray(c) for c in point_face)
    log_i = interp_nd_pchip(log_disk, axes, point, extrapolate=extrapolate)
    log_0 = interp_nd_pchip(log_disk, axes, point_face, extrapolate=extrapolate)
    log_n = interp_nd_pchip(log_norm, axes, point, extrapolate=extrapolate) - interp_nd_pchip(
        log_norm, axes, point_face, extrapolate=extrapolate
    )
    drift_native = log_i - log_0 + log_n
    live_native = (log_0 > jnp.log(_TABLE_FLOOR) + _LIVE_MARGIN).astype(drift_native.dtype)
    wave_grid = jnp.asarray(wave_grid)
    held = _hold_at_library_edges(wavelength, wave_grid)
    drift = resample_template(held, wave_grid, drift_native, left=0.0, right=0.0, log_flux=False)
    live = (
        resample_template(held, wave_grid, live_native, left=0.0, right=0.0, log_flux=False) > 0.5
    )
    return jnp.minimum(jnp.exp(drift), _RATIO_CLIP), live


def _log_table(table):
    """Logarithm of a library table, floored, taken at the process float precision.

    The files store float32; a logarithm of order -30 taken in float32 carries 2e-6 of
    absolute error into the ratio, so the table is promoted first when x64 is enabled.
    """
    return jnp.log(jnp.maximum(jnp.asarray(table, dtype=jnp.result_type(float)), _TABLE_FLOOR))


def _require_norm(norm, library: str):
    """The library's ``norm`` dataset, or an error naming the file."""
    if norm is None:
        raise TengriIOError(
            f"The {library} grid file carries no 'norm' dataset, which the line-of-sight "
            "ratio of its disc needs (the disc is stored divided by the per-record scale)."
        )
    return jnp.asarray(norm)


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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""SKIRTOR :math:`R_n` (disc times ``norm``, over its face-on value), held at the edges.

    Parameters
    ----------
    grid : SKIRTORDiscAttenGrid
        The SKIRTOR disc column and its ``norm``.
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    tau, p, q, oa, radius, cos_inc : float
        SKIRTOR grid coordinates (edge-on 9.7 um depth, density indices,
        half-opening angle, radius ratio and cos of the inclination).

    Returns
    -------
    ratio : ndarray, shape (n_wave,)
        :math:`R_n` in [0, 1.5] [dimensionless].
    live : ndarray, shape (n_wave,)
        True where the face-on library disc exists.

    Raises
    ------
    TengriIOError
        If the grid has no ``norm``.
    """
    axes = tuple(jnp.asarray(a) for a in grid.axes)
    log_disk = _log_table(grid.disk)
    log_norm = _log_table(_require_norm(grid.norm, "SKIRTOR"))
    geometry = (tau, p, q, oa, radius)
    return _normalized_ratio(
        log_disk,
        log_norm,
        axes,
        (*geometry, cos_inc),
        (*geometry, 1.0),
        grid.wave_grid,
        wavelength,
        _LOOKUP_EXTRAPOLATE["skirtor"],
    )


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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Fritz :math:`R_n` between the viewing record and the face-on record (89.99 deg).

    Parameters
    ----------
    grid : FritzGrid
        The Fritz library, with its ``disk`` and ``norm`` columns.
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    r_ratio, tau, beta, gamma, oa : float
        Fritz grid coordinates.
    cos_inc : float
        Cosine of the inclination from the polar axis; the library's viewing
        elevation is :math:`\psi = 90^\circ - i`.

    Returns
    -------
    ratio : ndarray, shape (n_wave,)
        :math:`R_n` in [0, 1.5] [dimensionless].
    live : ndarray, shape (n_wave,)
        True where the face-on library disc exists.

    Raises
    ------
    TengriIOError
        If the grid has no ``norm``.
    """
    axes = tuple(jnp.asarray(a) for a in grid.axes)
    log_disk = _log_table(grid.disk)
    log_norm = _log_table(_require_norm(grid.norm, "Fritz (2006)"))
    geometry = (r_ratio, tau, beta, gamma, oa)
    return _normalized_ratio(
        log_disk,
        log_norm,
        axes,
        (*geometry, fritz_psy_from_cos_inc(cos_inc)),
        (*geometry, axes[5][-1]),
        grid.wave_grid,
        wavelength,
        _LOOKUP_EXTRAPOLATE["fritz"],
    )


def _skirtor_anisotropy(cos_inc):
    r"""SKIRTOR disc anisotropy :math:`\eta = \cos i (1 + 2\cos i)/3` (skirtor2016)."""
    return cos_inc * (1.0 + 2.0 * cos_inc) / 3.0


def _floored_cos(cos_inc):
    """cos i smoothly floored at :data:`_COS_FLOOR` (softplus; exact to e^-8 above 0.17)."""
    gap = (jnp.asarray(cos_inc) - _COS_FLOOR) / _COS_FLOOR_WIDTH
    return _COS_FLOOR + _COS_FLOOR_WIDTH * jax.nn.softplus(gap)


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
        norm=None if dd.norm is None else jnp.asarray(dd.norm),
    )


def _fritz_disk_grid(library) -> FritzGrid | None:
    """The Fritz library with a disc column: the passed template, else the packaged one."""
    grid = library if isinstance(library, FritzGrid) else load_fritz_default_grid()
    return grid if grid.disk is not None else None


def _library_sightline(
    torus_block: str,
    wavelength,
    *,
    cos_inc: float,
    params: dict,
    library=None,
) -> tuple[LibrarySightline, float] | None:
    r"""The library's normalized line of sight, as the transmission and the disc factor.

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
        argument so its disc column and ``norm`` are not baked into the graph. When
        omitted, the packaged library is read from its cached loader.

    Returns
    -------
    LibrarySightline or None
        ``None`` when the torus has no library disc column (SKIRTOR v2 grid) or is not a
        library torus, so the caller keeps the analytic screen.

    Notes
    -----
    Implements :math:`T = R_n/\eta` and :math:`2\cos i\,R_n/\eta` of the module
    docstring; the library is read as in CIGALE's ``skirtor2016`` and ``fritz2006``
    (Boquien et al. 2019, A&A 622, A103), with the per-record ``norm`` of the library files.

    **JIT-compatible**: yes. **Gradient-safe**: yes in float32 as well; the ratio is the
    exponential of a difference of logarithms, and ``disc_factor`` has no quotient by
    :math:`\eta`.
    """
    c = jnp.asarray(cos_inc)
    if torus_block == "skirtor":
        grid = _skirtor_disc_grid(library)
        if grid is None:
            return None
        wave_min = jnp.asarray(grid.wave_grid)[0]
        ratio, live = skirtor_line_of_sight_ratio(
            grid,
            wavelength,
            tau=params.get("agn_tau_skirtor", 7.0),
            p=params.get("agn_p_skirtor", 1.0),
            q=params.get("agn_q_skirtor", 1.0),
            oa=params.get("agn_oa_skirtor", 40.0),
            radius=params.get("agn_radius_ratio", 20.0),
            cos_inc=cos_inc,
        )
        transmission = ratio / _skirtor_anisotropy(_floored_cos(c))
        disc_factor = 6.0 * ratio / (1.0 + 2.0 * c)
    elif torus_block == "fritz":
        fgrid = _fritz_disk_grid(library)
        if fgrid is None:
            return None
        wave_min = jnp.asarray(fgrid.wave_grid)[0]
        ratio, live = fritz_line_of_sight_ratio(
            fgrid,
            wavelength,
            r_ratio=params.get("agn_fritz_r_ratio", 60.0),
            tau=params.get("agn_fritz_tau", 1.0),
            beta=params.get("agn_fritz_beta", -0.5),
            gamma=params.get("agn_fritz_gamma", 4.0),
            oa=params.get("agn_fritz_oa", 60.0),
            cos_inc=cos_inc,
        )
        transmission = ratio
        disc_factor = 2.0 * c * ratio
    else:
        return None
    sight = LibrarySightline(
        ratio=ratio,
        transmission=jnp.where(live, transmission, 1.0),
        disc_factor=jnp.where(live, disc_factor, 2.0 * c),
    )
    return sight, wave_min


def library_torus_sightline(
    torus_block: str,
    wavelength,
    *,
    cos_inc: float,
    params: dict,
    library=None,
    below_index: BelowLibraryIndex | None = None,
) -> LibrarySightline | None:
    r"""The library's line of sight with X-rays photoelectrically absorbed below its grid.

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
        argument so its disc column and ``norm`` are not baked into the graph. When
        omitted, the packaged library is read from its cached loader.

    Returns
    -------
    LibrarySightline or None
        ``None`` when the torus has no library disc column (SKIRTOR v2 grid) or is not a
        library torus, so the caller keeps the analytic screen.

    Notes
    -----
    Implements :math:`T = R_n/\eta` and :math:`2\cos i\,R_n/\eta` of the module
    docstring; the library is read as in CIGALE's ``skirtor2016`` and ``fritz2006``
    (Boquien et al. 2019, A&A 622, A103), with the per-record ``norm`` of the library files.

    Below the library's shortest wavelength (10 A, 1.24 keV) the transmission is
    :math:`\exp(-\sigma(E)\,N_{\rm H})` with the Morrison & McCammon (1983) cross-section
    :math:`\sigma` (:func:`~tengri.components.xray.xray.wabs_cross_section`; above 10 keV the
    last segment is extrapolated), the column :math:`N_{\rm H} = 2.21\times10^{21}\,{\rm
    cm^{-2}}\,A_V` (Guver & Ozel 2009, MNRAS, 400, 2050) and
    :math:`A_V = -2.5\log_{10} T(5500\,\mathrm{\AA})` from this sightline's own :math:`T`,
    floored at zero (a Type-1 :math:`T > 1` has no column). The disc factor there is
    :math:`2\cos i` times that transmission.

    **JIT-compatible**: yes. **Gradient-safe**: yes in float32 as well; the ratio is the
    exponential of a difference of logarithms, and ``disc_factor`` has no quotient by
    :math:`\eta`.
    """
    wave = jnp.asarray(wavelength)
    extended = jnp.concatenate([wave, jnp.full((1,), _V_WAVE, dtype=wave.dtype)])
    found = _library_sightline(
        torus_block, extended, cos_inc=cos_inc, params=params, library=library
    )
    if found is None:
        return None
    sight, wave_min = found
    a_v_raw = -2.5 * jnp.log10(jnp.maximum(sight.transmission[-1], representable_floor(1.0e-30)))
    a_v = jnp.where(a_v_raw > 0.0, a_v_raw, 0.0)
    n_h = NH_PER_AV * a_v
    c = jnp.asarray(cos_inc)
    transmission = sight.transmission[:-1]
    disc_factor = sight.disc_factor[:-1]
    if below_index is not None:
        if below_index.n_wave != wave.shape[0]:
            raise ValueError(
                f"The build-time below-library index belongs to a grid of {below_index.n_wave} "
                f"samples, but the sightline was evaluated on {wave.shape[0]}. Pass the index "
                "only from the build master grid; any other grid takes below_index=None."
            )
        # Build-time absorption data (the caller's grid is the one it was built on, so its
        # length matches): the cross-section is a constant and the absorbed column is one
        # elementwise exponential. The 0/1 mask is applied as an arithmetic blend rather than
        # a select: both are exact for finite library values (the ratio is clipped and its
        # logarithms floored, so they are), and the blend keeps the compiled cost of the
        # kernel to the elementwise work the sightline needs. Values equal the dense form.
        t_x = jnp.exp(-jnp.asarray(below_index.sigma) * n_h)
        b = jnp.asarray(below_index.below, dtype=transmission.dtype)
        transmission = transmission * (1.0 - b) + b * t_x
        disc_factor = disc_factor * (1.0 - b) + b * (2.0 * c * t_x)
        return LibrarySightline(
            ratio=sight.ratio[:-1],
            transmission=transmission,
            disc_factor=disc_factor,
        )
    below_idx = _below_library_index(wavelength, torus_block)
    if below_idx is None:
        # Traced wavelength grid: no static index set, so the absorption is evaluated on
        # the whole grid and selected (the original dense form).
        below = wave < wave_min
        energy = HC_KEV_ANGSTROM / jnp.where(below, wave, 1.0)
        t_x = jnp.exp(-wabs_cross_section(energy) * n_h)
        transmission = jnp.where(below, t_x, transmission)
        disc_factor = jnp.where(below, 2.0 * c * t_x, disc_factor)
    else:
        # The absorption is evaluated only on the samples below the library's grid, which
        # are fixed by the caller's grid; every other sample keeps the library value.
        energy = HC_KEV_ANGSTROM / np.asarray(wavelength, dtype=np.float64)[below_idx]
        t_x = jnp.exp(-wabs_cross_section(energy) * n_h)
        transmission = transmission.at[below_idx].set(t_x)
        disc_factor = disc_factor.at[below_idx].set(2.0 * c * t_x)
    return LibrarySightline(
        ratio=sight.ratio[:-1],
        transmission=transmission,
        disc_factor=disc_factor,
    )


def _below_library_index(wavelength, torus_block: str) -> np.ndarray | None:
    """Indices of the grid samples shortward of the packaged library's grid, or ``None``.

    The library's shortest wavelength is read from the packaged grid, a host value; the
    threaded bundle is repacked from the same file, so it has the same first node. A
    traced ``wavelength`` has no static index set and returns ``None``.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    torus_block : str
        ``"skirtor"`` or ``"fritz"``.

    Returns
    -------
    ndarray of int, shape (n_below,), or None
        Positions ``i`` with ``wavelength[i] < wave_min``; ``None`` when the grid is
        traced or the torus has no packaged library.
    """
    if isinstance(wavelength, jax.core.Tracer):
        return None
    if torus_block == "skirtor":
        grid = load_skirtor_disc_atten_grid()
    elif torus_block == "fritz":
        grid = load_fritz_default_grid()
    else:
        return None
    wave_min = float(np.asarray(grid.wave_grid)[0])
    return np.flatnonzero(np.asarray(wavelength, dtype=np.float64) < wave_min)


def library_edge_wavelength(torus_block: str) -> float:
    """Shortest wavelength [Angstrom] of a packaged torus library's grid.

    Parameters
    ----------
    torus_block : str
        ``"skirtor"`` or ``"fritz"``.

    Returns
    -------
    float
        The library's first wavelength node: the edge below which the X-ray column applies.

    Notes
    -----
    **JIT-compatible**: construction-time only; a host value read from the packaged grid.
    """
    if torus_block == "skirtor":
        return float(np.asarray(load_skirtor_disc_atten_grid().wave_grid)[0])
    if torus_block == "fritz":
        return float(np.asarray(load_fritz_default_grid().wave_grid)[0])
    raise ValueError(f"library_edge_wavelength: no packaged library for torus {torus_block!r}")


def tied_incl_with_library_edge(incl_ratio, wavelength, sight: LibrarySightline, torus_block: str):
    r"""The tied disc's inclination ratio, with the X-ray column below the library's edge.

    The tied disc takes the library ratio :math:`{\rm disk}_i/{\rm disk}_0` on the caller's grid,
    which the library holds at zero below its first node, so the tied disc would emit nothing in
    the X-rays. Shortward of that node the untied path's factor :math:`2\cos i\,T_X` is taken
    instead (:attr:`LibrarySightline.disc_factor`, the Morrison & McCammon (1983) column of
    :func:`library_torus_sightline`); every other sample keeps ``incl_ratio``.

    Parameters
    ----------
    incl_ratio : array_like, shape (n_wave,)
        The tabulated inclination ratio on ``wavelength`` [dimensionless].
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom], the grid ``incl_ratio`` is evaluated on.
    sight : LibrarySightline
        The library's line of sight on the same grid, from :func:`library_torus_sightline`.
    torus_block : str
        ``"skirtor"`` or ``"fritz"``.

    Returns
    -------
    ndarray, shape (n_wave,)
        The inclination ratio with the absorbed library factor below the edge [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes; the edge is a host constant and the selection is a ``where``.
    **Gradient-safe**: yes.
    """
    edge = library_edge_wavelength(torus_block)
    wave = jnp.asarray(wavelength)
    return jnp.where(wave < edge, sight.disc_factor, jnp.asarray(incl_ratio))


def library_torus_transmission(
    torus_block: str,
    wavelength,
    *,
    cos_inc: float,
    params: dict,
    library=None,
) -> jnp.ndarray | None:
    r"""Transmission :math:`T = R_n/\eta` of a library torus for an isotropic source.

    Parameters
    ----------
    torus_block : str
        ``"skirtor"`` or ``"fritz"``.
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    cos_inc : float
        Cosine of the inclination from the polar axis; 1 = face-on.
    params : dict
        The runner's parameter dict.
    library : SKIRTORBundle or FritzGrid, optional
        The torus's pre-loaded template library.

    Returns
    -------
    ndarray, shape (n_wave,), or None
        ``LibrarySightline.transmission``; ``None`` when the torus has no library disc
        column, so the caller keeps the analytic screen.

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes.
    """
    sight = library_torus_sightline(
        torus_block, wavelength, cos_inc=cos_inc, params=params, library=library
    )
    return None if sight is None else sight.transmission
