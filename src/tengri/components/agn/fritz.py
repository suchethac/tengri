# SPDX-License-Identifier: BSD-3-Clause
r"""Fritz et al. (2006) smooth-dust AGN torus model.

Loads the full Fritz SED library (``create_fritz_from_grid``) and performs
6D triweight kernel interpolation in JAX. Provides C²-continuous gradients for
smooth inference (VI, MAP, NUTS).

The Fritz2006 model provides a semi-empirical radiative-transfer grid of
dust torus SEDs parameterized by six dimensions:

- r_ratio: maximum-to-minimum dust torus radius ratio
- tau: optical depth at 9.7 µm
- beta: radial dust density power-law index
- gamma: polar dust density gradient
- opening_angle: half-angle :math:`\theta_c` of the dust-free polar cone
  [degrees], 20, 40 or 60; the torus's full opening angle is
  :math:`\Theta = 180^\circ - 2\theta_c` (140, 100, 60 degrees)
- psy: viewing elevation :math:`\psi` above the equatorial plane [degrees],
  0.001 ... 89.99; the sightline reaches the nucleus directly (type 1) when
  :math:`\psi > 90^\circ - \theta_c`

In the composable AGN the library's ``psy`` is not a free parameter: the model
has one inclination, ``agn_cos_inc`` (:math:`\cos i`, :math:`i` from the polar
axis), and :func:`fritz_psy_from_cos_inc` derives :math:`\psi = 90^\circ - i`.

All functions are pure JAX and JIT-compilable.

References
----------
.. [1] J. Fritz, A. Franceschini and E. Hatziminaoglou, "Revisiting the
   infrared spectra of active galactic nuclei with a new torus emission
   model," MNRAS, 366, 767 (2006). arXiv:astro-ph/0511428.
   https://doi.org/10.1111/j.1365-2966.2006.09866.x
.. [2] M. Boquien et al., "CIGALE: a python Code Investigating GALaxy Emission,"
   A&A, 622, A103 (2019). arXiv:1811.03094.
   https://doi.org/10.1051/0004-6361/201834156
"""

import functools
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tengri._deprecated import deprecated_alias
from tengri.components.agn._params import DEFAULT_AGN_LOG_LBOL
from tengri.components.agn._phys import L_SUN as _L_SUN
from tengri.components.agn._template_grid import native_bolometric_nu
from tengri.components.agn.skirtor import SkirtorDiscTie
from tengri.config.exceptions import TengriIOError
from tengri.utils.grid_interp import interp_nd_pchip, interp_nd_triweight, resample_template
from tengri.utils.interpolation import edges_for_grid
from tengri.utils.physics_constants import C_AA as _C_AA_PER_S
from tengri.utils.scale import representable_denominator


class FritzComponents(NamedTuple):
    r"""Separate Fritz spectral components.

    Attributes
    ----------
    disk : jnp.ndarray, shape (n_wave,)
        Accretion disk emission (direct + scattered) [erg/s/Hz].
    dust : jnp.ndarray, shape (n_wave,)
        Dust thermal emission from the torus [erg/s/Hz].

    Notes
    -----
    The disk component is the accretion-disk SED, and dust is the thermal
    torus emission. Both are rest-frame spectral luminosity densities,
    obtained from the grid's :math:`L_\lambda` tables with
    :math:`L_\nu = L_\lambda \lambda^2 / c`.
    """

    disk: jnp.ndarray
    dust: jnp.ndarray


# ── Template grid interpolation ───────────────────────────────────


def _load_grid_arrays(grid_path: str):
    """Load raw numpy arrays from a Fritz grid file.

    Parameters
    ----------
    grid_path : str
        Path to ``.h5`` file.

    Returns
    -------
    dict
        Keys: ``wave``, ``dust``, ``disk``, ``axes`` (r_ratio, tau, beta, gamma, oa, psy).

    Notes
    -----
    **JIT-compatible**: no, performs file I/O at module load time.
    """
    import numpy as np

    result = {}

    import h5py as _h5py

    with _h5py.File(grid_path, "r") as f:
        result["wave"] = np.array(f["fritz2006/wavelength_aa"][:])
        result["dust"] = np.array(f["fritz2006/dust"][:])
        result["disk"] = np.array(f["fritz2006/disk"][:])
        # Per-record scale (pcigale ``model.norm``, 1e42-1e46: past float32), kept relative to
        # its largest value, which is all the face-on/viewing-angle ratio needs. Absent from
        # grids downloaded before it was added.
        if "norm" in f["fritz2006"]:
            norm = np.array(f["fritz2006/norm"][()], dtype=np.float64)
            result["norm"] = norm / norm.max()
        result["axes"] = (
            np.array(f["fritz2006/r_ratio_axis"][:]),
            np.array(f["fritz2006/tau_axis"][:]),
            np.array(f["fritz2006/beta_axis"][:]),
            np.array(f["fritz2006/gamma_axis"][:]),
            np.array(f["fritz2006/opening_angle_axis"][:]),
            np.array(f["fritz2006/psy_axis"][:]),
        )

    return result


#: The library's ``psy`` axis extent [deg]: the grid stops at these values, and
#: the exact endpoints 0 and 90 extrapolate.
FRITZ_PSY_MIN = 0.001
FRITZ_PSY_MAX = 89.99

#: Largest :math:`\cos i` handed to ``arccos``: :math:`1 - 10^{-6}`, 0.08 degrees
#: from the pole. At :math:`\cos i = 1` the slope of ``arccos`` is infinite and
#: the clip's zero turns it into NaN in the backward pass; this cap is well
#: inside the type-1 plateau of every library node.
_COS_INC_FACE_ON_CAP = 1.0 - 1.0e-6


def fritz_psy_from_cos_inc(cos_inc: float) -> jnp.ndarray:
    r"""The Fritz library's viewing elevation for the model's one inclination.

    Parameters
    ----------
    cos_inc : float
        :math:`\cos i`, with :math:`i` the inclination from the polar axis
        (1 = face-on). [dimensionless]

    Returns
    -------
    ndarray, scalar
        :math:`\psi = 90^\circ - i`, the elevation above the equatorial plane
        that keys the library, held to the grid's extent
        [:data:`FRITZ_PSY_MIN`, :data:`FRITZ_PSY_MAX`]. [deg]

    Notes
    -----
    :math:`\cos i = \sin\psi`. The library is type 1 at
    :math:`\psi > 90^\circ - \theta_c` (:math:`i < \theta_c`), where
    :math:`\theta_c` is ``agn_fritz_oa`` [1]_.

    **JIT-compatible**: yes. **Gradient-safe**: yes; the gradient is zero
    beyond the grid extent, where the library is constant.

    References
    ----------
    .. [1] J. Fritz, A. Franceschini and E. Hatziminaoglou, MNRAS, 366, 767
       (2006). arXiv:astro-ph/0511428.
    """
    c = jnp.minimum(jnp.asarray(cos_inc), _COS_INC_FACE_ON_CAP)
    psy = 90.0 - jnp.degrees(jnp.arccos(c))
    return jnp.clip(psy, FRITZ_PSY_MIN, FRITZ_PSY_MAX)


def _refuse_off_grid(name: str, value, axis) -> None:
    """Raise if a concrete ``value`` lies outside the grid ``axis``.

    Traced values, and a grid threaded through ``jit`` as an argument, cannot be
    checked here (the builder's declared bounds refuse them at build time);
    concrete ones would otherwise be clamped to the edge template without a word.
    """
    if isinstance(value, jax.core.Tracer):
        return
    try:
        axis_np = np.asarray(axis)
    except jax.errors.TracerArrayConversionError:
        return
    lo, hi = float(axis_np[0]), float(axis_np[-1])
    v = float(value)
    if not lo <= v <= hi:
        raise ValueError(
            f"{name}={v:g} is outside the Fritz2006 grid [{lo:g}, {hi:g}]; the "
            f"interpolator would return the edge template unchanged."
        )


def _interpolate_and_normalize(
    grid_jax: jnp.ndarray,
    wave_grid: jnp.ndarray,
    axes: tuple,
    edges: tuple,
    wavelength: jnp.ndarray,
    point: tuple,
    l_scale: float,
) -> jnp.ndarray:
    r"""Interpolate a template grid and normalize to physical :math:`L_\nu`.

    The shipped grid is the Fritz et al. (2006) [1]_ torus library as CIGALE's
    ``model.dust`` and ``model.disk`` arrays [2]_: luminosity per unit
    wavelength :math:`L_\lambda` [W/nm]. ``dust`` is normalized to unit
    :math:`\int L_\lambda \, d\lambda`; ``disk`` is in the same units,
    relative to that dust, and its integral differs from node to node. The
    template is put on the requested wavelength grid and converted with

    .. math::

        L_\nu = L_\lambda \, \frac{\lambda^2}{c},

    where :math:`\lambda` is wavelength [Angstrom], :math:`c` the speed of
    light [Angstrom/s], :math:`L_\lambda` the tabulated template and
    :math:`L_\nu` the specific luminosity [erg/s/Hz]. The result is scaled so
    that :math:`\int L_\nu \, d\nu = l_{\rm scale}` over the template's native grid.

    Parameters
    ----------
    grid_jax : ndarray, shape (n_r, n_tau, n_beta, n_gamma, n_oa, n_psy, n_wave)
        Template grid, :math:`L_\lambda` [W/nm]; only its shape matters, it is
        renormalized on use.
    wave_grid : ndarray, shape (n_wave_grid,)
        Grid wavelength array [Angstrom].
    axes : tuple of ndarray
        Grid axis values (r_ratio, tau, beta, gamma, oa, psy).
    edges : tuple of ndarray
        Precomputed bin edges for triweight interpolation.
    wavelength : ndarray, shape (n_wave,)
        Target wavelength array [Angstrom].
    point : tuple
        (r_ratio, tau, beta, gamma, oa, psy) query point.
    l_scale : float
        Luminosity scale factor [erg/s].

    Returns
    -------
    ndarray, shape (n_wave,)
        Specific luminosity L_ν [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp.interp`` and ``jax.vmap``.

    The normalization integral is taken over the template's own native
    wavelength grid, before resampling, so ``l_scale`` is the template's total
    power whatever the requested grid: a grid that stops short of the template
    (one ending at 30 micron holds as little as 15 % of its power) carries
    only the part of ``l_scale`` that falls inside it.

    Each call renormalizes its template to ``l_scale``, so
    :func:`fritz_components` carries the shape of the disc and of the dust but
    not the library's disc-to-dust ratio.

    References
    ----------
    .. [1] J. Fritz, A. Franceschini and E. Hatziminaoglou, "Revisiting the
       infrared spectra of active galactic nuclei with a new torus emission
       model," MNRAS, 366, 767 (2006). arXiv:astro-ph/0511428.
       https://doi.org/10.1111/j.1365-2966.2006.09866.x
    .. [2] M. Boquien et al., "CIGALE: a python Code Investigating GALaxy Emission,"
       A&A, 622, A103 (2019). arXiv:1811.03094.
       https://doi.org/10.1051/0004-6361/201834156
    """
    # Fritz tau and r_dust axes are non-uniform (I6 fix #1851).
    # Use index-space interpolation for correct gradients throughout the range.
    _refuse_off_grid("agn_fritz_oa", point[4], axes[4])
    _refuse_off_grid("agn_fritz_psy", point[5], axes[5])
    template = interp_nd_triweight(grid_jax, axes, edges, point, index_space_interp=True)
    # Normalize on the template's native grid, before resampling, so the
    # result does not depend on the caller's wavelength sampling or range.
    integral_safe = native_bolometric_nu(template * wave_grid**2 / _C_AA_PER_S, wave_grid)
    sed_lam = resample_template(wavelength, wave_grid, template, left=0.0, right=0.0)
    sed_nu = sed_lam * wavelength**2 / _C_AA_PER_S
    return l_scale * sed_nu / integral_safe


class FritzGrid(NamedTuple):
    """Fritz+2006 6-D torus template arrays, as a JAX pytree.

    Carried as a pytree (rather than closed over) so the forward model can
    pass the library into ``jax.jit`` as an argument. Closing over it instead
    bakes ~16 MB into the graph as ``Constant`` ops.

    Attributes
    ----------
    dust : ndarray, shape (n_r, n_tau, n_beta, n_gamma, n_oa, n_psy, n_wave)
        Tabulated torus SEDs [shape only; renormalized on use].
    wave_grid : ndarray, shape (n_wave,)
        Template rest-frame wavelength grid [Angstrom].
    axes : tuple of ndarray
        The six parameter axes, in interpolation order.
    edges : tuple of ndarray
        Triweight bin edges derived from ``axes``.
    disk : ndarray, shape (n_r, n_tau, n_beta, n_gamma, n_oa, n_psy, n_wave), optional
        Tabulated accretion-disc SEDs on the same axes, in the library's own units relative
        to the unit-integral ``dust``. Read by :func:`fritz_disc_dust_ratio` for the
        disc/dust tie; ``None`` for a grid built without it.
    norm : ndarray, shape (n_r, n_tau, n_beta, n_gamma, n_oa, n_psy), optional
        Each record's own scale (pcigale ``model.norm``) relative to the largest, with no
        wavelength axis; ``None`` for a grid without it.
    """

    dust: jnp.ndarray
    wave_grid: jnp.ndarray
    axes: tuple[jnp.ndarray, ...]
    edges: tuple[jnp.ndarray, ...]
    disk: jnp.ndarray | None = None
    norm: jnp.ndarray | None = None


@functools.cache
def load_fritz_grid(grid_path: str) -> FritzGrid:
    """Load a Fritz+2006 grid HDF5 into a :class:`FritzGrid` pytree.

    Parameters
    ----------
    grid_path : str
        Path to a Fritz2006 HDF5 grid file.

    Returns
    -------
    FritzGrid

    Notes
    -----
    **JIT-compatible**: no, performs HDF5 I/O. Call outside the trace.

    ``jax.ensure_compile_time_eval`` keeps the derived edge arrays concrete
    even when this first runs inside a trace; without it the
    ``functools.cache`` would immortalize ``DynamicJaxprTracer`` values that
    leak out of the trace scope.
    """
    raw = _load_grid_arrays(grid_path)
    with jax.ensure_compile_time_eval():
        axes = tuple(jnp.array(ax) for ax in raw["axes"])
        return FritzGrid(
            dust=jnp.array(raw["dust"]),
            wave_grid=jnp.array(raw["wave"]),
            axes=axes,
            edges=tuple(edges_for_grid(ax) for ax in axes),
            disk=jnp.array(raw["disk"]),
            norm=jnp.array(raw["norm"]) if "norm" in raw else None,
        )


def fritz_sed_from_grid(
    grid: FritzGrid,
    wavelength: jnp.ndarray,
    agn_log_lbol: float = 10.0,
    agn_torus_frac: float = 0.5,
    agn_fritz_r_ratio: float = 60.0,
    agn_fritz_tau: float = 1.0,
    agn_fritz_beta: float = -0.5,
    agn_fritz_gamma: float = 4.0,
    agn_fritz_oa: float = 60.0,
    agn_fritz_psy: float = 0.001,
    **_kwargs,
) -> jnp.ndarray:
    r"""Fritz+2006 torus :math:`L_\nu` by 6-D triweight grid interpolation.

    Parameters
    ----------
    grid : FritzGrid
        Template arrays, passed as an argument so they thread through JIT.
    wavelength : ndarray, shape (n_wave,)
        Wavelength grid. [Angstrom]
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`. [dimensionless]
    agn_torus_frac : float
        Fraction of L_bol reprocessed by the torus. [dimensionless]
    agn_fritz_r_ratio : float
        Dust torus radius ratio (r_max / r_min). Allowed: 10, 30, 60, 100, 150.
    agn_fritz_tau : float
        Optical depth at 9.7 um. Allowed: 0.1, 0.3, 0.6, 1, 2, 3, 6, 10.
    agn_fritz_beta : float
        Radial dust density power-law index. Allowed: -1, -0.75, -0.5, -0.25, 0.
    agn_fritz_gamma : float
        Polar dust density gradient. Allowed: 0, 2, 4, 6.
    agn_fritz_oa : float
        Half-angle of the dust-free polar cone [degrees]. Allowed: 20, 40, 60
        (full torus opening angle 180 - 2 x half = 140, 100, 60).
    agn_fritz_psy : float
        Library viewing elevation above the equatorial plane [degrees], 0.001
        to 89.99; type 1 when it exceeds 90 - ``agn_fritz_oa``. The composable
        torus block derives it from ``agn_cos_inc`` (:func:`fritz_psy_from_cos_inc`).

    Returns
    -------
    ndarray, shape (n_wave,)
        Dust torus specific luminosity :math:`L_\nu`. [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes, the triweight kernel
    is C2-continuous across all six axes.
    """
    l_scale = 10.0**agn_log_lbol * _L_SUN * agn_torus_frac
    point = (
        agn_fritz_r_ratio,
        agn_fritz_tau,
        agn_fritz_beta,
        agn_fritz_gamma,
        agn_fritz_oa,
        agn_fritz_psy,
    )
    return _interpolate_and_normalize(
        jnp.asarray(grid.dust),
        jnp.asarray(grid.wave_grid),
        tuple(jnp.asarray(a) for a in grid.axes),
        tuple(jnp.asarray(e) for e in grid.edges),
        wavelength,
        point,
        l_scale,
    )


def create_fritz_from_grid(grid_path: str) -> Callable:
    """Load Fritz2006 templates and return an interpolation function.

    The returned function interpolates the 6D Fritz grid using triweight
    kernel interpolation (C²-continuous, fully differentiable) and normalizes
    the output so the integrated luminosity equals ``agn_torus_frac × L_bol``.

    Parameters
    ----------
    grid_path : str
        Path to the Fritz grid file (``.h5``).

    Returns
    -------
    callable
        Function with signature::

            fn(wavelength, agn_log_lbol, agn_torus_frac,
               agn_fritz_r_ratio, agn_fritz_tau, agn_fritz_beta,
               agn_fritz_gamma, agn_fritz_oa, agn_fritz_psy,
               **kwargs) -> L_nu [erg s^-1 Hz^-1]

    Raises
    ------
    FileNotFoundError
        If ``grid_path`` does not exist.
    KeyError
        If the grid file is missing expected keys.

    Notes
    -----
    **JIT-compatible**: yes, the returned function is pure JAX.
    Grid loading is cached via ``@functools.cache``.

    **Gradient-safe**: yes, triweight interpolation is fully differentiable.

    References
    ----------
    .. [1] J. Fritz, A. Franceschini and E. Hatziminaoglou, "Revisiting the
       infrared spectra of active galactic nuclei with a new torus emission
       model," MNRAS, 366, 767 (2006). arXiv:astro-ph/0511428.
       https://doi.org/10.1111/j.1365-2966.2006.09866.x
    """
    grid = load_fritz_grid(grid_path)
    dust_jax, wave_grid, axes, edges = grid.dust, grid.wave_grid, grid.axes, grid.edges

    def fritz_grid(
        wavelength: jnp.ndarray,
        agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
        agn_torus_frac: float = 0.5,
        agn_fritz_r_ratio: float = 60.0,
        agn_fritz_tau: float = 1.0,
        agn_fritz_beta: float = -0.5,
        agn_fritz_gamma: float = 4.0,
        agn_fritz_oa: float = 60.0,
        agn_fritz_psy: float = 0.001,
        **_kwargs,
    ) -> jnp.ndarray:
        """Fritz2006 torus from template grid interpolation.

        Parameters
        ----------
        wavelength : ndarray, shape (n_wave,)
            Wavelength grid. [Å]
        agn_log_lbol : float
            log₁₀(L_bol / L_sun). [dimensionless]
        agn_torus_frac : float
            Fraction of L_bol reprocessed by the torus. [dimensionless]
        agn_fritz_r_ratio : float
            Dust torus radius ratio (r_max / r_min). [dimensionless]
            Allowed values: 10, 30, 60, 100, 150.
        agn_fritz_tau : float
            Optical depth at 9.7 µm. [dimensionless]
            Allowed values: 0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 6.0, 10.0.
        agn_fritz_beta : float
            Radial dust density power-law index. [dimensionless]
            Allowed values: -1.0, -0.75, -0.5, -0.25, 0.0.
        agn_fritz_gamma : float
            Polar dust density gradient. [dimensionless]
            Allowed values: 0, 2, 4, 6.
        agn_fritz_oa : float
            Half-angle of the dust-free polar cone. [degrees]
            Allowed values: 20, 40, 60 (full opening angle 180 - 2 x half).
            Outside [20, 60] raises ``ValueError``.
        agn_fritz_psy : float
            Library viewing elevation above the equatorial plane. [degrees]
            Grid nodes: 0.001, 10.1, 20.1, 30.1, 40.1, 50.1, 60.1, 70.1, 80.1,
            89.99; type 1 when it exceeds 90 - ``agn_fritz_oa``.

        Returns
        -------
        ndarray, shape (n_wave,)
            Dust torus specific luminosity L_ν. [erg s⁻¹ Hz⁻¹]
        """
        l_scale = 10.0**agn_log_lbol * _L_SUN * agn_torus_frac
        point = (
            agn_fritz_r_ratio,
            agn_fritz_tau,
            agn_fritz_beta,
            agn_fritz_gamma,
            agn_fritz_oa,
            agn_fritz_psy,
        )
        return _interpolate_and_normalize(
            dust_jax, wave_grid, axes, edges, wavelength, point, l_scale
        )

    return fritz_grid


def create_fritz_components_from_grid(grid_path: str) -> Callable:
    """Load Fritz2006 templates and return a function giving separate components.

    Parameters
    ----------
    grid_path : str
        Path to a Fritz2006 HDF5 grid file.

    Returns
    -------
    callable
        Function with signature::

            fn(wavelength, agn_log_lbol, agn_torus_frac,
               agn_fritz_r_ratio, agn_fritz_tau, agn_fritz_beta,
               agn_fritz_gamma, agn_fritz_oa, agn_fritz_psy, **kwargs)
                -> FritzComponents(disk, dust)

        Each component is in [erg s^-1 Hz^-1].

    Raises
    ------
    FileNotFoundError
        If the grid file does not exist.

    Notes
    -----
    **JIT-compatible**: yes, the returned function is pure JAX.
    Grid loading is cached via ``@functools.cache``.

    **Gradient-safe**: yes, triweight interpolation is fully differentiable.

    The separate components enable applying different extinction laws to
    disk vs. dust and computing anisotropy corrections independently.
    """
    raw = _load_grid_arrays(grid_path)

    with jax.ensure_compile_time_eval():
        disk_jax = jnp.array(raw["disk"])
        dust_jax = jnp.array(raw["dust"])
        wave_grid = jnp.array(raw["wave"])
        axes = tuple(jnp.array(ax) for ax in raw["axes"])
        edges = tuple(edges_for_grid(ax) for ax in axes)

    def fritz_components(
        wavelength: jnp.ndarray,
        agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
        agn_torus_frac: float = 0.5,
        agn_fritz_r_ratio: float = 60.0,
        agn_fritz_tau: float = 1.0,
        agn_fritz_beta: float = -0.5,
        agn_fritz_gamma: float = 4.0,
        agn_fritz_oa: float = 60.0,
        agn_fritz_psy: float = 0.001,
        **_kwargs,
    ) -> FritzComponents:
        """Fritz2006 torus with separate disk and dust components.

        Parameters
        ----------
        wavelength : ndarray, shape (n_wave,)
            Wavelength grid. [Å]
        agn_log_lbol : float
            log₁₀(L_bol / L_sun). [dimensionless]
        agn_torus_frac : float
            Fraction of bolometric luminosity. [dimensionless]
        agn_fritz_r_ratio, agn_fritz_tau, agn_fritz_beta, agn_fritz_gamma,
        agn_fritz_oa, agn_fritz_psy : float
            Grid parameters (see create_fritz_from_grid docstring).

        Returns
        -------
        FritzComponents
            Named tuple with ``disk`` and ``dust`` arrays,
            each shape (n_wave,) in [erg s⁻¹ Hz⁻¹].
        """
        l_scale = 10.0**agn_log_lbol * _L_SUN * agn_torus_frac
        point = (
            agn_fritz_r_ratio,
            agn_fritz_tau,
            agn_fritz_beta,
            agn_fritz_gamma,
            agn_fritz_oa,
            agn_fritz_psy,
        )
        disk = _interpolate_and_normalize(
            disk_jax, wave_grid, axes, edges, wavelength, point, l_scale
        )
        dust = _interpolate_and_normalize(
            dust_jax, wave_grid, axes, edges, wavelength, point, l_scale
        )
        return FritzComponents(disk=disk, dust=dust)

    return fritz_components


# ── Auto-load tabulated Fritz2006 as the default ────────────────────


_GRID_SEARCH_PATHS = [
    "data/fritz2006_torus_grid.h5",
]

_GRID_FILENAME = "fritz2006_torus_grid.h5"

_NOT_FOUND_MSG = (
    "Fritz2006 templates not found (fritz2006_torus_grid.h5) and the auto-"
    "download failed.\n"
    "Fetch the pre-converted grid (no CIGALE needed):\n"
    "    python scripts/download_fritz2006_templates.py\n"
    "or, if you have CIGALE installed, regenerate it:\n"
    "    python scripts/build_fritz2006_grid.py"
)


def _find_fritz_grid() -> str:
    """Locate the Fritz2006 grid file, auto-downloading it if missing.

    The grid is searched on disk first; if absent, it is fetched from the
    public template host (no CIGALE dependency). Only if both the local lookup
    and the download fail is :class:`FileNotFoundError` raised.
    """

    from tengri._data_setup import find_data

    # Must consult $TENGRI_DATA_DIR before falling through to the download
    # below (#1431): otherwise a user whose grids live off the source tree
    # re-fetches a file they already have.
    found = find_data(*_GRID_SEARCH_PATHS)
    if found is not None:
        return str(found)

    # Not on disk: try the public host (mirrors the SSP auto-fetch path).
    try:
        from tengri._data_setup import download_template

        # dest defaults to download_dir(), which is data_dirs()[0]: so the
        # loader above finds the file next time. The previous explicit
        # repo-root dest wrote where $TENGRI_DATA_DIR users never look.
        return str(download_template(_GRID_FILENAME))
    except Exception:
        raise FileNotFoundError(_NOT_FOUND_MSG) from None


def load_fritz_default_grid() -> FritzGrid:
    """Load the packaged Fritz+2006 grid pytree (discovery + cache).

    This is the ``template_loader`` the torus block registers, so the
    forward model can hoist the library out of the JIT trace.

    Returns
    -------
    FritzGrid

    Raises
    ------
    FileNotFoundError
        If the grid is neither on disk nor downloadable.
    """
    return load_fritz_grid(_find_fritz_grid())


def fritz_disc_dust_wave(template: FritzGrid | None = None):
    """Wavelength axis of the Fritz library [Å], the axis the disc/dust tie integrates on.

    Parameters
    ----------
    template : FritzGrid, optional
        Library to read the axis from. Defaults to the packaged grid.

    Returns
    -------
    ndarray, shape (n_wave,)
        The library's native wavelength axis [Å].
    """
    grid = load_fritz_default_grid() if template is None else template
    return jnp.asarray(grid.wave_grid)


def fritz_disc_dust_ratio(
    wave: jnp.ndarray,
    disc_lambda_unreddened: jnp.ndarray,
    disc_ext_fac: jnp.ndarray,
    *,
    agn_fritz_r_ratio: float = 60.0,
    agn_fritz_tau: float = 1.0,
    agn_fritz_beta: float = -0.5,
    agn_fritz_gamma: float = 4.0,
    agn_fritz_oa: float = 60.0,
    agn_fritz_psy: float = 0.001,
    incl_wave: jnp.ndarray | None = None,
    _template: FritzGrid | None = None,
) -> SkirtorDiscTie:
    r"""Disc-to-dust bolometric ratio of the Fritz library, the tie of CIGALE ``fritz2006``.

    CIGALE's ``fritz2006`` module ties the accretion disc to the dust power ``agn_power`` the
    way ``skirtor2016`` does: the analytic disc shape :math:`\hat s` (unit area on the library
    axis) is scaled to the face-on library disc integral :math:`I_0 = \int D_0\,d\lambda`
    (:math:`D_0` the :math:`\psi = 89.99^\circ` record), reweighted by the library ratio
    :math:`D_\psi/D_0`, reddened, and divided by the dust integral of the viewing-angle record
    :math:`U_\psi`:

    .. math::

        R = \frac{\int \hat s\, I_0\, (D_\psi/D_0)\, e\, d\lambda}{\int U_\psi\, d\lambda},
        \qquad R_{\rm face\text{-}on} = \frac{I_0\, n_0/n_\psi}{\int U_\psi\, d\lambda},

    so that the disc carries ``agn_power x R`` and the dust ``agn_power`` [1]_ [2]_. Here
    :math:`e` is the line-of-sight reddening (1 without it) and :math:`n_0/n_\psi` the ratio of the
    library records' own scales (``norm``; ``AGN1.disk *= AGN1.norm / fritz2006.norm`` in
    CIGALE), which reaches the face-on reference only (it cancels in :math:`R`). A grid file
    without ``norm`` raises.

    Parameters
    ----------
    wave : ndarray, shape (n_wave,)
        Wavelength grid of ``disc_lambda_unreddened`` [Å].
    disc_lambda_unreddened : ndarray, shape (n_wave,)
        Analytic disc spectrum before reddening; only its shape is used. [erg/s/Å]
    disc_ext_fac : ndarray, shape (n_wave,)
        Reddening factor :math:`10^{-0.4 k E(B-V)}` (1 = none). [dimensionless]
    agn_fritz_r_ratio, agn_fritz_tau, agn_fritz_beta, agn_fritz_gamma, agn_fritz_oa, agn_fritz_psy
        Library coordinates, as the torus block reads them.
    incl_wave : ndarray, shape (n_out,), optional
        Grid ``incl_ratio`` is returned on [Å]. Defaults to ``wave``.

    Returns
    -------
    tie : SkirtorDiscTie
        The same record as for SKIRTOR: ``R``, ``incl_ratio`` (:math:`D_\psi/D_0` on
        ``incl_wave``), ``R_faceon``, ``faceon_shape_native``, ``wave_native`` and
        ``incl_native``.

    Raises
    ------
    TengriIOError
        If the grid carries no ``norm`` dataset (an older ``fritz2006_torus_grid.h5``).

    Notes
    -----
    **JIT-compatible**: yes; the records are interpolated node-exactly (PCHIP), as the
    SKIRTOR tie does.

    References
    ----------
    .. [1] Fritz, J., Franceschini, A. & Hatziminaoglou, E. 2006, MNRAS, 366, 767,
       https://doi.org/10.1111/j.1365-2966.2006.09866.x
    .. [2] Boquien, M. et al. 2019, A&A, 622, A103 (CIGALE ``fritz2006``),
       https://doi.org/10.1051/0004-6361/201834156
    """
    grid = _template if _template is not None and _template.disk is not None else None
    if grid is None:
        grid = load_fritz_default_grid()
    axes = tuple(jnp.asarray(a) for a in grid.axes)
    wave_grid = jnp.asarray(grid.wave_grid)

    def _interp(table, psy):
        point = (
            agn_fritz_r_ratio,
            agn_fritz_tau,
            agn_fritz_beta,
            agn_fritz_gamma,
            agn_fritz_oa,
            psy,
        )
        # Node-exact PCHIP, not the triweight smoother the torus block uses: the smoother
        # does not pass through the grid nodes and moves the disk/dust integral ratio, which
        # is what this tie reads, by several per cent even at a node.
        return interp_nd_pchip(jnp.asarray(table), axes, tuple(jnp.asarray(c) for c in point))

    disk_i_n = _interp(grid.disk, agn_fritz_psy)
    dust_i_n = _interp(grid.dust, agn_fritz_psy)
    disk_0_n = _interp(grid.disk, axes[5][-1])  # the face-on record, psi = 89.99 deg

    disc_n = resample_template(wave_grid, wave, disc_lambda_unreddened, left=0.0, right=0.0)
    ext_n = resample_template(wave_grid, wave, disc_ext_fac, left=1.0, right=1.0)
    int_disk0 = jnp.trapezoid(disk_0_n, wave_grid)
    shape_n = disc_n / jnp.maximum(
        jnp.trapezoid(disc_n, wave_grid), representable_denominator(1e-30)
    )
    finite_mask = disk_0_n > 0
    last_finite_idx = jnp.max(jnp.where(finite_mask, jnp.arange(disk_0_n.shape[0]), -1))
    last_finite_ratio = jnp.where(
        disk_0_n[last_finite_idx] > 0, disk_i_n[last_finite_idx] / disk_0_n[last_finite_idx], 1.0
    )
    incl_n = jnp.where(
        finite_mask, disk_i_n / jnp.where(finite_mask, disk_0_n, 1.0), last_finite_ratio
    )
    int_dust = jnp.maximum(jnp.trapezoid(dust_i_n, wave_grid), representable_denominator(1e-30))
    # ``norm(face-on)/norm(psi)``: CIGALE's ``AGN1.disk *= AGN1.norm / fritz2006.norm``. It
    # reaches the face-on reference only; ``R`` divides it back out.
    if grid.norm is None:
        raise TengriIOError(
            "The Fritz (2006) grid file has no 'fritz2006/norm' dataset, which the tied disc "
            "needs for the face-on reference norm(0)/norm(psi). Add it to the existing file with "
            "`python scripts/build_fritz2006_grid.py --add-norm --dest <data directory>`."
        )
    norm_i = _interp(grid.norm, agn_fritz_psy)
    norm_0 = _interp(grid.norm, axes[5][-1])
    incl_norm_ratio = jnp.where(norm_i > 0.0, norm_0 / jnp.where(norm_i > 0.0, norm_i, 1.0), 1.0)
    R = jnp.trapezoid(shape_n * int_disk0 * incl_n * ext_n, wave_grid) / int_dust
    incl_ratio = resample_template(
        wave if incl_wave is None else jnp.asarray(incl_wave),
        wave_grid,
        incl_n,
        left=0.0,
        right=last_finite_ratio,
    )
    return SkirtorDiscTie(
        R=R,
        incl_ratio=incl_ratio,
        R_faceon=int_disk0 * incl_norm_ratio / int_dust,
        faceon_shape_native=shape_n,
        wave_native=wave_grid,
        incl_native=incl_n,
    )


@functools.cache
def _load_fritz_default():
    """Load Fritz2006 template grid from file (dust-only, the torus component)."""
    return create_fritz_from_grid(_find_fritz_grid())


@functools.cache
def _load_fritz_components():
    """Load Fritz2006 template grid with separate components."""
    path = _find_fritz_grid()
    return create_fritz_components_from_grid(path)


def fritz_sed(*args, **kwargs):
    """Fritz2006 torus SED (auto-loaded from tabulated templates).

    This function uses the tabulated Fritz et al. (2006) template grid
    with 6D triweight interpolation.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom].
    agn_log_lbol : float, optional
        AGN bolometric luminosity [log10(L_sun)]. Default: 10.0.
    agn_torus_frac : float, optional
        Fraction of bolometric luminosity in torus [dimensionless, 0–1].
        Default: 0.5.
    agn_fritz_r_ratio : float, optional
        Dust torus radius ratio (r_max / r_min) [dimensionless].
        Default: 60.0. Allowed: 10, 30, 60, 100, 150.
    agn_fritz_tau : float, optional
        Optical depth at 9.7 µm [dimensionless]. Default: 1.0.
        Allowed: 0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 6.0, 10.0.
    agn_fritz_beta : float, optional
        Radial dust density power-law index [dimensionless].
        Default: -0.5. Allowed: -1.0, -0.75, -0.5, -0.25, 0.0.
    agn_fritz_gamma : float, optional
        Polar dust density gradient [dimensionless]. Default: 4.0.
        Allowed: 0, 2, 4, 6.
    agn_fritz_oa : float, optional
        Half-angle of the dust-free polar cone [degrees]. Default: 60.0.
        Allowed: 20, 40, 60 (full opening angle 180 - 2 x half = 140, 100,
        60); outside [20, 60] raises ``ValueError``.
    agn_fritz_psy : float, optional
        Library viewing elevation above the equatorial plane [degrees].
        Default: 0.001 (type-2). Grid nodes: 0.001, 10.1, 20.1, 30.1, 40.1,
        50.1, 60.1, 70.1, 80.1, 89.99; type 1 when it exceeds
        90 - ``agn_fritz_oa``.
    _template : callable, optional
        Pre-loaded template function (for JIT threading). When provided,
        uses this instead of the module-level cached loader. Internal use.
    **kwargs
        Additional keyword arguments (ignored for compatibility).

    Returns
    -------
    ndarray, shape (n_wave,)
        Dust torus spectral luminosity density L_ν [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes, delegates to cached grid function or
    pre-loaded template (when _template is threaded).

    See ``create_fritz_from_grid`` for full parameter documentation and
    grid node locations.

    References
    ----------
    .. [1] J. Fritz, A. Franceschini and E. Hatziminaoglou, MNRAS, 366, 767 (2006).
    .. [2] M. Boquien et al., A&A, 622, A103 (2019).
    """
    # Allow the template to be threaded as a JIT runtime input
    _template = kwargs.pop("_template", None)
    if isinstance(_template, FritzGrid):
        # Threaded grid arrays: evaluate directly so they stay JIT arguments.
        return fritz_sed_from_grid(_template, *args, **kwargs)
    template_fn = _template if _template is not None else _load_fritz_default()
    return template_fn(*args, **kwargs)


def fritz_components(*args, **kwargs) -> FritzComponents:
    """Fritz2006 torus with separate disk/dust (auto-loaded).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom].
    agn_log_lbol : float, optional
        AGN bolometric luminosity [log10(L_sun)]. Default: 10.0.
    agn_torus_frac : float, optional
        Covering factor [0, 1]. Default: 0.5.
    agn_fritz_r_ratio, agn_fritz_tau, agn_fritz_beta, agn_fritz_gamma,
    agn_fritz_oa, agn_fritz_psy : float, optional
        Grid parameters (see fritz_analytic docstring).
    _template : callable, optional
        Pre-loaded template function (for JIT threading). When provided,
        uses this instead of the module-level cached loader. Internal use.
    **kwargs
        Additional keyword arguments (ignored for compatibility).

    Returns
    -------
    FritzComponents
        Named tuple with ``disk`` and ``dust`` arrays, each
        shape (n_wave,) with units [erg/s/Hz].

    Raises
    ------
    FileNotFoundError
        If grid file is not found.

    Notes
    -----
    **JIT-compatible**: yes, delegates to cached grid function or
    pre-loaded template (when _template is threaded).
    """
    _template = kwargs.pop("_template", None)

    if _template is not None:
        fn = _template
    else:
        fn = _load_fritz_components()
    return fn(*args, **kwargs)


# Deprecated: "_analytic" was a misnomer; Fritz+2006 is a 6D template-grid
# interpolation, not a closed-form model. Use fritz_sed. Removed in v1.0.
fritz_analytic = deprecated_alias(fritz_sed, old_name="fritz_analytic", new_name="fritz_sed")
