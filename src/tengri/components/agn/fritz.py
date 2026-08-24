# SPDX-License-Identifier: BSD-3-Clause
"""Fritz et al. (2006) smooth-dust AGN torus model.

Loads the full Fritz SED library (``create_fritz_from_grid``) and performs
6D triweight kernel interpolation in JAX. Provides C²-continuous gradients for
smooth inference (VI, MAP, NUTS).

The Fritz2006 model provides a semi-empirical radiative-transfer grid of
dust torus SEDs parameterized by six dimensions:

- r_ratio: maximum-to-minimum dust torus radius ratio
- tau: optical depth at 9.7 µm
- beta: radial dust density power-law index
- gamma: polar dust density gradient
- opening_angle: dust torus half-opening angle [degrees]
- psy: viewing angle from torus axis [degrees]; 0° = type-2, 90° = type-1 AGN

All functions are pure JAX and JIT-compilable.

References
----------
.. [1] O. Fritz et al., "Dust tori around Type II active nuclei. I.
   Observational constraints and allowed dust models," A&A, 470, 221 (2006).
   arXiv:0606147. https://doi.org/10.1051/0004-6361:20066130
.. [2] M. Boquien et al., "CIGALE: Code Investigating GALaxy Emission,"
   A&A, 622, A103 (2019). arXiv:1811.03094.
   https://doi.org/10.1051/0004-6361/201834156
"""

import functools
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from tengri._deprecated import deprecated_alias
from tengri.components.agn._params import DEFAULT_AGN_LOG_LBOL
from tengri.components.agn._phys import (
    L_SUN as _L_SUN,
    bolometric_integral_nu as _bolometric_integral_nu,
    wavelength_to_nu as _wavelength_to_nu,
)
from tengri.utils.grid_interp import interp_nd_triweight, resample_template
from tengri.utils.interpolation import edges_for_grid


class FritzComponents(NamedTuple):
    """Separate Fritz spectral components.

    Attributes
    ----------
    disk : jnp.ndarray, shape (n_wave,)
        Accretion disk emission (direct + scattered) [erg/s/Hz].
    dust : jnp.ndarray, shape (n_wave,)
        Dust thermal emission from the torus [erg/s/Hz].

    Notes
    -----
    The disk component is the accretion-disk SED, and dust is the thermal
    torus emission. Both are rest-frame spectral luminosity densities.
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
        result["axes"] = (
            np.array(f["fritz2006/r_ratio_axis"][:]),
            np.array(f["fritz2006/tau_axis"][:]),
            np.array(f["fritz2006/beta_axis"][:]),
            np.array(f["fritz2006/gamma_axis"][:]),
            np.array(f["fritz2006/opening_angle_axis"][:]),
            np.array(f["fritz2006/psy_axis"][:]),
        )

    return result


def _interpolate_and_normalize(
    grid_jax: jnp.ndarray,
    wave_grid: jnp.ndarray,
    axes: tuple,
    edges: tuple,
    wavelength: jnp.ndarray,
    point: tuple,
    l_scale: float,
) -> jnp.ndarray:
    """Interpolate a template grid and normalize to physical luminosity.

    Parameters
    ----------
    grid_jax : ndarray, shape (n_r, n_tau, n_beta, n_gamma, n_oa, n_psy, n_wave)
        Template grid [erg/s/Hz, per-L_sun normalized at runtime].
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
        Luminosity scale factor [erg s^-1].

    Returns
    -------
    ndarray, shape (n_wave,)
        Specific luminosity L_ν [erg s^-1 Hz^-1].

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp.interp`` and ``jax.vmap``.
    """
    # Fritz tau and r_dust axes are non-uniform (I6 fix #1851).
    # Use index-space interpolation for correct gradients throughout the range.
    template = interp_nd_triweight(grid_jax, axes, edges, point, index_space_interp=True)
    sed = resample_template(wavelength, wave_grid, template, left=0.0, right=0.0)
    nu = _wavelength_to_nu(wavelength)
    integral_safe = _bolometric_integral_nu(sed, nu, floor=1e-100)
    return l_scale * sed / integral_safe


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
    """

    dust: jnp.ndarray
    wave_grid: jnp.ndarray
    axes: tuple[jnp.ndarray, ...]
    edges: tuple[jnp.ndarray, ...]


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
        Dust torus half-opening angle [degrees]. Allowed: 60, 100, 140.
    agn_fritz_psy : float
        Viewing angle from torus axis [degrees]; 0 = type-2 (edge-on),
        90 = type-1 (face-on).

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
    .. [1] O. Fritz et al., "Dust tori around Type II active nuclei,"
       A&A, 470, 221 (2006). arXiv:0606147.
       https://doi.org/10.1051/0004-6361:20066130
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
            Dust torus half-opening angle. [degrees]
            Allowed values: 60, 100, 140.
        agn_fritz_psy : float
            Viewing angle from torus axis. [degrees]
            0° = type-2 AGN (edge-on), 90° = type-1 AGN (face-on).
            Allowed values: 0.001, 10.1, 20.1, 30.1, 40.1, 50.1, 60.1, 70.1, 80.1, 89.99.

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
        Dust torus half-opening angle [degrees]. Default: 60.0.
        Allowed: 60, 100, 140.
    agn_fritz_psy : float, optional
        Viewing angle from torus axis [degrees]. Default: 0.001 (type-2).
        Allowed: 0.001, 10.1, 20.1, 30.1, 40.1, 50.1, 60.1, 70.1, 80.1, 89.99.
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
    .. [1] O. Fritz et al., A&A, 470, 221 (2006).
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


# Deprecated: "_analytic" was a misnomer: Fritz+2006 is a 6D template-grid
# interpolation, not a closed-form model. Use fritz_sed. Removed in v1.0.
fritz_analytic = deprecated_alias(fritz_sed, old_name="fritz_analytic", new_name="fritz_sed")
