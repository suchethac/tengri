"""SKIRTOR clumpy two-phase torus model (Stalevski et al. 2012, 2016).

Loads the full SKIRTOR SED library (``create_skirtor_from_grid``) and performs
5D triweight kernel interpolation in JAX.  Provides C²-continuous gradients for
smooth inference (VI, MAP, NUTS).  Requires a prior download of the template
grid (~1 GB).

Supports two HDF5 layouts:

- **v2** (legacy): single ``spectra/torus_emission`` array (total only).
- **v3**: separate ``spectra/disk_emission``, ``spectra/dust_emission``, and
  ``spectra/torus_emission``.  Matches the CIGALE ``skirtor2016`` processing
  convention (disk = direct + scattered, both divided by wavelength, normalized
  so dust thermal integrates to 1 W).

All functions are pure JAX and JIT-compilable.

References
----------
.. [1] M. Stalevski et al., "3D radiative transfer modelling of the dusty
   torus around AGN — the influence of clumping," MNRAS, 420, 2756 (2012).
   arXiv:1109.1286. https://doi.org/10.1111/j.1365-2966.2011.19775.x
.. [2] M. Stalevski et al., "The dust covering factor in AGN — combining the
   IR torus emission with polar dust component," MNRAS, 458, 2288 (2016).
   arXiv:1602.01954. https://doi.org/10.1093/mnras/stw444
"""

import functools
from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp

from tengri.components.agn._phys import (
    LSUN_ERG as _LSUN_ERG,
    wavelength_to_nu as _wavelength_to_nu,
)
from tengri.forward.precompute.grid import interp_nd_triweight
from tengri.utils.interpolation import edges_for_grid


class SKIRTORComponents(NamedTuple):
    """Separate SKIRTOR spectral components.

    Attributes
    ----------
    disk : jnp.ndarray, shape (n_wave,)
        Accretion disk emission (direct + scattered). [erg s⁻¹ Hz⁻¹]
    dust : jnp.ndarray, shape (n_wave,)
        Dust thermal emission from the torus. [erg s⁻¹ Hz⁻¹]
    total : jnp.ndarray, shape (n_wave,)
        Total emission (disk + dust). [erg s⁻¹ Hz⁻¹]
    """

    disk: jnp.ndarray
    dust: jnp.ndarray
    total: jnp.ndarray


# ── Template grid interpolation ───────────────────────────────────


def _load_grid_arrays(grid_path: str):
    """Load raw numpy arrays from a SKIRTOR grid file.

    Parameters
    ----------
    grid_path : str
        Path to ``.npz`` or ``.h5`` file.

    Returns
    -------
    dict
        Keys: ``wave``, ``total``, ``axes`` (tau, p, q, oa, cos_inc),
        and optionally ``disk``, ``dust``.
    """
    import numpy as np

    result = {}

    if grid_path.endswith(".npz"):
        data = np.load(grid_path)
        required_keys = {"grid", "wavelength", "tau", "p", "q", "oa", "cos_inc"}
        missing = required_keys - set(data.keys())
        if missing:
            raise KeyError(
                f"SKIRTOR grid file missing keys: {missing}. Available: {list(data.keys())}"
            )
        result["total"] = np.array(data["grid"])
        result["wave"] = np.array(data["wavelength"])
        result["axes"] = (
            np.array(data["tau"]),
            np.array(data["p"]),
            np.array(data["q"]),
            np.array(data["oa"]),
            np.array(data["cos_inc"]),
        )
    else:
        import h5py as _h5py

        with _h5py.File(grid_path, "r") as f:
            result["wave"] = np.array(f["wavelength"][:])
            if "grid" in f and isinstance(f["grid"], _h5py.Group):
                result["total"] = np.array(f["spectra/torus_emission"][:])
                result["axes"] = (
                    np.array(f["grid/tau_97"][:]),
                    np.array(f["grid/p"][:]),
                    np.array(f["grid/q"][:]),
                    np.array(f["grid/opening_angle"][:]),
                    np.array(f["grid/cos_inclination"][:]),
                )
                if "spectra/disk_emission" in f:
                    result["disk"] = np.array(f["spectra/disk_emission"][:])
                if "spectra/dust_emission" in f:
                    result["dust"] = np.array(f["spectra/dust_emission"][:])
            else:
                result["total"] = np.array(f["grid"][:])
                result["axes"] = (
                    np.array(f["tau"][:]),
                    np.array(f["p"][:]),
                    np.array(f["q"][:]),
                    np.array(f["oa"][:]),
                    np.array(f["cos_inc"][:]),
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
    grid_jax : ndarray, shape (n_tau, n_p, n_q, n_oa, n_inc, n_wave)
        Template grid.
    wave_grid : ndarray, shape (n_wave_grid,)
        Grid wavelength array. [Å]
    axes : tuple of ndarray
        Grid axis values (tau, p, q, oa, cos_inc).
    edges : tuple of ndarray
        Precomputed bin edges for triweight interpolation.
    wavelength : ndarray, shape (n_wave,)
        Target wavelength array. [Å]
    point : tuple
        (tau, p, q, oa, cos_inc) query point.
    l_scale : float
        Luminosity scale factor. [erg s⁻¹]

    Returns
    -------
    ndarray, shape (n_wave,)
        Specific luminosity L_ν. [erg s⁻¹ Hz⁻¹]
    """
    template = interp_nd_triweight(grid_jax, axes, edges, point)
    sed = jnp.interp(wavelength, wave_grid, template, left=0.0, right=0.0)
    nu = _wavelength_to_nu(wavelength)
    idx_sort = jnp.argsort(nu)
    integral = jnp.trapezoid(sed[idx_sort], nu[idx_sort])
    integral_safe = jnp.maximum(jnp.abs(integral), 1e-100)
    return l_scale * sed / integral_safe


def create_skirtor_from_grid(grid_path: str) -> Callable:
    """Load SKIRTOR templates and return an interpolation function.

    The returned function has the same signature as ``skirtor_analytic``
    and can be used as a drop-in replacement.

    Grid dimensions: tau × p × q × oa × inc × wave.
    Interpolation: 5D triweight kernel in JAX (JIT-compatible, C²-continuous
    gradients).

    Parameters
    ----------
    grid_path : str
        Path to the SKIRTOR grid file (``.npz`` or ``.h5``).

    Returns
    -------
    callable
        Function with signature::

            fn(wavelength, agn_log_lbol, agn_tau_skirtor, agn_p_skirtor,
               agn_q_skirtor, agn_oa_skirtor, agn_cos_inc,
               agn_torus_frac, **kwargs) -> L_nu [erg s^-1 Hz^-1]

    Raises
    ------
    FileNotFoundError
        If ``grid_path`` does not exist.
    KeyError
        If the grid file is missing expected keys.

    Notes
    -----
    **JIT-compatible**: yes — the returned function is pure JAX.

    Supports v2 (total-only) and v3 (separate disk/dust) HDF5 layouts.
    When v3 is available, use ``create_skirtor_components_from_grid``
    to access individual disk and dust spectra.

    References
    ----------
    .. [1] M. Stalevski et al., "3D radiative transfer modelling of the dusty
       torus around AGN," MNRAS, 420, 2756 (2012). arXiv:1109.1286.
       https://doi.org/10.1111/j.1365-2966.2011.19775.x
    .. [2] M. Stalevski et al., "The dust covering factor in AGN," MNRAS, 458,
       2288 (2016). arXiv:1602.01954. https://doi.org/10.1093/mnras/stw444
    """
    raw = _load_grid_arrays(grid_path)

    grid_jax = jnp.array(raw["total"])
    wave_grid = jnp.array(raw["wave"])
    axes = tuple(jnp.array(ax) for ax in raw["axes"])
    edges = tuple(edges_for_grid(ax) for ax in axes)

    def skirtor_grid(
        wavelength: jnp.ndarray,
        agn_log_lbol: float = 44.0,
        agn_tau_skirtor: float = 7.0,
        agn_p_skirtor: float = 1.0,
        agn_q_skirtor: float = 1.0,
        agn_oa_skirtor: float = 40.0,
        agn_cos_inc: float = 0.5,
        agn_torus_frac: float = 0.5,
        **_kwargs,
    ) -> jnp.ndarray:
        """SKIRTOR torus from template grid interpolation.

        Parameters
        ----------
        wavelength : ndarray, shape (n_wave,)
            Wavelength grid. [Å]
        agn_log_lbol : float
            log₁₀(L_bol / L_sun). [dimensionless]
        agn_tau_skirtor : float
            Edge-on optical depth at 9.7 μm. [dimensionless]
        agn_p_skirtor : float
            Radial dust density power-law index. [dimensionless]
        agn_q_skirtor : float
            Polar dust density gradient index. [dimensionless]
        agn_oa_skirtor : float
            Torus half-opening angle. [degrees]
        agn_cos_inc : float
            Cosine of inclination (1 = face-on, 0 = edge-on). [dimensionless]
        agn_torus_frac : float
            Fraction of L_bol reprocessed by the torus. [dimensionless]

        Returns
        -------
        ndarray, shape (n_wave,)
            Specific luminosity L_ν. [erg s⁻¹ Hz⁻¹]
        """
        l_scale = 10.0**agn_log_lbol * _LSUN_ERG * agn_torus_frac
        point = (
            agn_tau_skirtor,
            agn_p_skirtor,
            agn_q_skirtor,
            agn_oa_skirtor,
            agn_cos_inc,
        )
        return _interpolate_and_normalize(
            grid_jax, wave_grid, axes, edges, wavelength, point, l_scale
        )

    return skirtor_grid


def create_skirtor_components_from_grid(grid_path: str) -> Callable:
    """Load SKIRTOR v3 templates and return a function giving separate components.

    Requires a v3 HDF5 grid with ``spectra/disk_emission`` and
    ``spectra/dust_emission`` datasets (produced by
    ``scripts/download_skirtor_templates.py``).

    Parameters
    ----------
    grid_path : str
        Path to a v3 SKIRTOR HDF5 file.

    Returns
    -------
    callable
        Function with signature::

            fn(wavelength, agn_log_lbol, ..., agn_torus_frac, **kwargs)
                -> SKIRTORComponents(disk, dust, total)

        Each component is in [erg s⁻¹ Hz⁻¹].

    Raises
    ------
    KeyError
        If the grid lacks ``spectra/disk_emission`` or
        ``spectra/dust_emission``.

    Notes
    -----
    **JIT-compatible**: yes — the returned function is pure JAX.

    The separate components enable:

    - Applying different extinction laws to disk vs. torus dust
    - Computing polar dust from the disk component alone
    - Anisotropy corrections on individual components

    References
    ----------
    .. [1] M. Stalevski et al., MNRAS, 420, 2756 (2012). arXiv:1109.1286.
    .. [2] M. Boquien et al., "CIGALE," A&A, 622, A103 (2019).
       arXiv:1811.03094. https://doi.org/10.1051/0004-6361/201834156
    """
    raw = _load_grid_arrays(grid_path)

    if "disk" not in raw or "dust" not in raw:
        raise KeyError(
            "SKIRTOR grid lacks separate disk/dust components. "
            "Use a v3 grid from scripts/download_skirtor_templates.py "
            "or use create_skirtor_from_grid() for total-only mode."
        )

    disk_jax = jnp.array(raw["disk"])
    dust_jax = jnp.array(raw["dust"])
    total_jax = jnp.array(raw["total"])
    wave_grid = jnp.array(raw["wave"])
    axes = tuple(jnp.array(ax) for ax in raw["axes"])
    edges = tuple(edges_for_grid(ax) for ax in axes)

    def skirtor_components(
        wavelength: jnp.ndarray,
        agn_log_lbol: float = 44.0,
        agn_tau_skirtor: float = 7.0,
        agn_p_skirtor: float = 1.0,
        agn_q_skirtor: float = 1.0,
        agn_oa_skirtor: float = 40.0,
        agn_cos_inc: float = 0.5,
        agn_torus_frac: float = 0.5,
        **_kwargs,
    ) -> SKIRTORComponents:
        """SKIRTOR torus with separate disk and dust components.

        Parameters
        ----------
        wavelength : ndarray, shape (n_wave,)
            Wavelength grid. [Å]
        agn_log_lbol : float
            log₁₀(L_bol / L_sun). [dimensionless]
        agn_tau_skirtor : float
            Edge-on optical depth at 9.7 μm. [dimensionless]
        agn_p_skirtor : float
            Radial dust density power-law index. [dimensionless]
        agn_q_skirtor : float
            Polar dust density gradient index. [dimensionless]
        agn_oa_skirtor : float
            Torus half-opening angle. [degrees]
        agn_cos_inc : float
            Cosine of inclination. [dimensionless]
        agn_torus_frac : float
            Fraction of L_bol reprocessed by torus. [dimensionless]

        Returns
        -------
        SKIRTORComponents
            Named tuple with ``disk``, ``dust``, ``total`` arrays,
            each shape (n_wave,) in [erg s⁻¹ Hz⁻¹].
        """
        l_scale = 10.0**agn_log_lbol * _LSUN_ERG * agn_torus_frac
        point = (
            agn_tau_skirtor,
            agn_p_skirtor,
            agn_q_skirtor,
            agn_oa_skirtor,
            agn_cos_inc,
        )
        disk = _interpolate_and_normalize(
            disk_jax, wave_grid, axes, edges, wavelength, point, l_scale
        )
        dust = _interpolate_and_normalize(
            dust_jax, wave_grid, axes, edges, wavelength, point, l_scale
        )
        total = _interpolate_and_normalize(
            total_jax, wave_grid, axes, edges, wavelength, point, l_scale
        )
        return SKIRTORComponents(disk=disk, dust=dust, total=total)

    return skirtor_components


# ── Auto-load tabulated SKIRTOR as the default ────────────────────


_GRID_SEARCH_PATHS = [
    "data/skirtor_templates_v3.h5",
    "data/skirtor_templates_v2.h5",
    "data/skirtor_templates.npz",
]

_NOT_FOUND_MSG = (
    "SKIRTOR templates not found (skirtor_templates_v3.h5, _v2.h5, or .npz). "
    "The analytic fallback has been removed because it produced scientifically "
    "incorrect results (3-temperature MBB, not radiative transfer). "
    "Download from: https://sites.google.com/site/skirtorus/sed-library "
    "or run: python scripts/download_skirtor_templates.py"
)


def _find_skirtor_grid() -> str:
    """Locate the best available SKIRTOR grid file on disk."""
    from pathlib import Path

    base = Path(__file__).resolve().parents[4]
    for rel in _GRID_SEARCH_PATHS:
        for candidate in [base / rel, Path(rel)]:
            if candidate.is_file():
                return str(candidate)
    raise FileNotFoundError(_NOT_FOUND_MSG)


@functools.cache
def _load_skirtor_default():
    """Load SKIRTOR template grid from file (total-only)."""
    return create_skirtor_from_grid(_find_skirtor_grid())


@functools.cache
def _load_skirtor_components():
    """Load SKIRTOR template grid with separate components.

    Falls back to total-only if v3 grid is not available.
    """
    path = _find_skirtor_grid()
    try:
        return create_skirtor_components_from_grid(path)
    except KeyError:
        return None


def skirtor_analytic(*args, **kwargs):
    """SKIRTOR torus SED (auto-loaded from tabulated templates).

    This function uses the tabulated Stalevski+2016 template grid
    with 5D triweight interpolation.

    See ``create_skirtor_from_grid`` for parameters.
    """
    return _load_skirtor_default()(*args, **kwargs)


def skirtor_components(*args, **kwargs) -> SKIRTORComponents:
    """SKIRTOR torus with separate disk/dust (auto-loaded).

    Requires a v3 grid file. See ``create_skirtor_components_from_grid``.

    Returns
    -------
    SKIRTORComponents
        Named tuple with ``disk``, ``dust``, ``total`` arrays.

    Raises
    ------
    RuntimeError
        If no v3 grid with separate components is available.
    """
    fn = _load_skirtor_components()
    if fn is None:
        raise RuntimeError(
            "Separate SKIRTOR components require a v3 grid file. "
            "Run: python scripts/download_skirtor_templates.py --input-dir <raw-files>"
        )
    return fn(*args, **kwargs)
