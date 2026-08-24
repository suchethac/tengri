# SPDX-License-Identifier: BSD-3-Clause
"""Toy torus models for AGN infrared emission.

.. warning::
    These are **toy models** using 1-2 temperature modified blackbodies.
    They are NOT radiative transfer results and should NOT be used for
    science.  For production work, use the SKIRTOR templates in
    ``tengri.components.agn.skirtor`` (tabulated from 3D Monte Carlo RT).

Two toy models are provided for testing and fast prototyping:

1. **simple_torus**: single-temperature modified blackbody
   with silicate opacity. 2 free parameters.
2. **two_temperature_torus**: hot + warm dust components. 4 free params.

Both return specific luminosity L_nu in erg/s/Hz. All functions are pure
JAX and JIT-compilable.

References
----------

- Nenkova et al. 2008, ApJ, 685, 147 (CLUMPY torus)
- Stalevski et al. 2012, MNRAS, 420, 2756 (SKIRTOR)
- Draine 2003, ARA&A, 41, 241 (silicate opacity)

"""

import functools
import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.agn._params import DEFAULT_AGN_LOG_LBOL
from tengri.components.agn._phys import (
    L_SUN as _L_SUN,
    bolometric_integral_nu as _bolometric_integral_nu,
    planck_lnu as _planck_lnu,
    wavelength_to_nu as _wavelength_to_nu,
)
from tengri.utils.grid_interp import interp_nd_triweight, resample_template
from tengri.utils.interpolation import edges_for_grid

# ── Physical constants (CGS) ──────────────────────────────────────

_MICRON_ANGSTROM = 1e4  # Micron -> Angstrom

# Silicate feature wavelength
_LAMBDA_SI = 9.7 * _MICRON_ANGSTROM  # 9.7 um in Angstrom

# ── Module-level warning guard ────────────────────────────────────
_WARNED: set[str] = set()


# ── Model 1: Simple hot blackbody torus ───────────────────────────


def simple_torus(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    agn_torus_frac: float = 0.5,
    agn_T_torus: float = 1000.0,
    agn_tau_torus: float = 5.0,
    agn_tau_beta: float = 1.5,
    **_kwargs,
) -> jnp.ndarray:
    """Simple single-temperature dust torus with silicate opacity.

    L_nu = L_bol * f_torus * B_nu(T_torus) / B_int * (1 - exp(-tau * (9.7um/lam)^beta))

    where B_int normalizes the modified blackbody to integrate to
    L_bol * f_torus.

    Parameters
    ----------
    wavelength: array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol: float
        log10(L_bol / Lsun). Total AGN bolometric luminosity.
        [log10(L_sun)]
    agn_torus_frac: float
        Fraction of L_bol re-emitted by torus (covering factor).
        Typical range: 0.1 to 0.9. Default 0.5. [dimensionless]
    agn_T_torus: float
        Torus dust temperature [K].
        Typical range: 500 to 1500. Default 1000.
    agn_tau_torus: float
        Optical depth at 9.7 um silicate feature [dimensionless].
        Typical range: 1 to 10. Default 5.
    agn_tau_beta: float
        Power-law index for opacity wavelength dependence [dimensionless].
        Typical range: 1.0 to 2.0. Default 1.5.

    Returns
    -------
    array, shape (n_wave,)
        Specific luminosity L_nu [erg s^-1 Hz^-1].

    Notes
    -----
    .. warning::
        This is a **toy model** using a single-temperature modified blackbody.
        It is NOT a radiative transfer result and should NOT be used for
        science. For production work, use the SKIRTOR templates in
        :mod:`tengri.components.agn.skirtor`.

    **JIT-compatible**: yes, uses ``jnp`` primitives only.
    """
    if "simple_torus" not in _WARNED:
        warnings.warn(
            "simple_torus is a toy AGN torus model not suitable for science fits; "
            "use silva04_sed (Silva+04 radiative-transfer torus) for production work. "
            "Will be removed in tengri v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        _WARNED.add("simple_torus")
    l_bol_erg = 10.0**agn_log_lbol * _L_SUN
    nu = _wavelength_to_nu(wavelength)

    # Blackbody emission
    b_nu = _planck_lnu(nu, agn_T_torus)

    # Silicate opacity: tau(lambda) = tau_torus * (9.7um / lambda)^beta
    opacity = 1.0 - jnp.exp(
        -agn_tau_torus * (_LAMBDA_SI / jnp.maximum(wavelength, 1.0)) ** agn_tau_beta
    )

    # Modified blackbody shape
    shape = b_nu * opacity

    # Normalize to L_bol * f_torus
    integral_safe = _bolometric_integral_nu(shape, nu, floor=1e-100)

    l_nu_erg = l_bol_erg * agn_torus_frac * shape / integral_safe
    return l_nu_erg


# ── Model 2: Two-temperature torus (SKIRTOR-inspired) ─────────────


def two_temperature_torus(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    agn_torus_frac: float = 0.5,
    agn_T_hot: float = 1200.0,
    agn_T_warm: float = 300.0,
    agn_frac_hot: float = 0.3,
    agn_tau_torus: float = 5.0,
    agn_tau_beta: float = 1.5,
    **_kwargs,
) -> jnp.ndarray:
    """Two-temperature dust torus (hot sublimation + warm outer torus).

    Inspired by SKIRTOR clumpy torus models. The emission is a mixture
    of two modified blackbodies:

        L_nu = f_hot * BB(T_hot) + (1 - f_hot) * BB(T_warm)

    both modified by the same silicate opacity profile.

    Parameters
    ----------
    wavelength: array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol: float
        log10(L_bol / Lsun). Total AGN bolometric luminosity.
        [log10(L_sun)]
    agn_torus_frac: float
        Fraction of L_bol re-emitted by torus. Default 0.5.
        [dimensionless]
    agn_T_hot: float
        Hot dust temperature [K], near sublimation.
        Typical range: 1000 to 1500. Default 1200.
    agn_T_warm: float
        Warm dust temperature [K], outer torus.
        Typical range: 200 to 800. Default 300.
    agn_frac_hot: float
        Luminosity fraction in hot component (0 to 1). Default 0.3.
        [dimensionless]
    agn_tau_torus: float
        Optical depth at 9.7 um [dimensionless]. Default 5.
    agn_tau_beta: float
        Opacity power-law index [dimensionless]. Default 1.5.

    Returns
    -------
    array, shape (n_wave,)
        Specific luminosity L_nu [erg s^-1 Hz^-1].

    Notes
    -----
    .. warning::
        This is a **toy model** using two modified blackbodies.
        It is NOT a radiative transfer result and should NOT be used for
        science. For production work, use the SKIRTOR templates in
        :mod:`tengri.components.agn.skirtor`.

    **JIT-compatible**: yes, uses ``jnp`` primitives only.
    """
    if "two_temperature_torus" not in _WARNED:
        warnings.warn(
            "two_temperature_torus is a toy AGN torus model not suitable for science fits; "
            "use silva04_sed (Silva+04 radiative-transfer torus) for production work. "
            "Will be removed in tengri v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        _WARNED.add("two_temperature_torus")
    l_bol_erg = 10.0**agn_log_lbol * _L_SUN
    nu = _wavelength_to_nu(wavelength)

    # Two blackbody components
    b_hot = _planck_lnu(nu, agn_T_hot)
    b_warm = _planck_lnu(nu, agn_T_warm)

    # Silicate opacity
    opacity = 1.0 - jnp.exp(
        -agn_tau_torus * (_LAMBDA_SI / jnp.maximum(wavelength, 1.0)) ** agn_tau_beta
    )

    # Weighted mixture with opacity
    shape = (agn_frac_hot * b_hot + (1.0 - agn_frac_hot) * b_warm) * opacity

    # Normalize
    integral_safe = _bolometric_integral_nu(shape, nu, floor=1e-100)

    l_nu_erg = l_bol_erg * agn_torus_frac * shape / integral_safe
    return l_nu_erg


# ── Nenkova+2008 CLUMPY torus ─────────────────────────────────────
#
# Prospector's only AGN SED component. tengri vendors the FSPS CLUMPY grid
# (``data/nenkova08_torus_grid.h5``, built by ``scripts/build_nenkova_grid.py``)
# and interpolates it with a pure-JAX triweight kernel in optical depth so
# that ``agn_tau`` is a fully differentiable, JIT/vmap-safe *fitted* parameter
#: matching how SKIRTOR / Silva+04 / CAT3D are handled.

_NENKOVA_GRID_SEARCH_PATHS: tuple[str, ...] = (
    "data/nenkova08_torus_grid.h5",
    "nenkova08_torus_grid.h5",
)

_NENKOVA_NOT_FOUND_MSG = (
    "Nenkova+2008 CLUMPY torus grid not found. "
    "Build it with: python scripts/build_nenkova_grid.py "
    '--input "$SPS_HOME/dust/Nenkova08_y010_torusg_n10_q2.0.dat"'
)


def _load_nenkova_arrays(grid_path: str) -> dict:
    """Load raw numpy arrays from a vendored Nenkova+2008 CLUMPY grid HDF5.

    Parameters
    ----------
    grid_path: str
        Path to ``nenkova08_torus_grid.h5`` produced by
        ``scripts/build_nenkova_grid.py``.

    Returns
    -------
    dict
        Keys ``tau_axis`` (n_tau,), ``wavelength`` (n_wave,), and ``template``
        (n_tau, n_wave).

    Notes
    -----
    **JIT-compatible**: no, performs HDF5 I/O at grid-load time.
    """
    import h5py

    with h5py.File(grid_path, "r") as f:
        g = f["nenkova"]
        return {
            "tau_axis": np.asarray(g["tau_axis"][:], dtype=np.float64),
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "template": np.asarray(g["template"][:], dtype=np.float64),
        }


def create_nenkova_from_grid(grid_path: str) -> Callable:
    """Load the Nenkova+2008 CLUMPY grid and return a JAX-native interpolator.

    Parameters
    ----------
    grid_path: str
        Path to ``nenkova08_torus_grid.h5``.

    Returns
    -------
    callable
        Function ``fn(wavelength, agn_log_lbol, agn_tau, agn_torus_frac, **_)
        -> L_nu [erg/s/Hz]``, a drop-in replacement for :func:`nenkova_torus`.

    Raises
    ------
    FileNotFoundError
        If ``grid_path`` does not exist.
    KeyError
        If the grid file is missing the expected ``/nenkova`` datasets.

    Notes
    -----
    **JIT-compatible**: yes, the returned closure uses only ``jnp`` and a
    triweight interpolation kernel.

    **Gradient-safe**: yes, the triweight kernel is C²-continuous in
    ``agn_tau``, so it survives ``jax.grad`` / ``jax.vmap``.

    References
    ----------
    .. [1] M. Nenkova et al., "AGN Dusty Tori. I. Handling of Clumpy Media,"
       ApJ, 685, 147 (2008). https://doi.org/10.1086/590482
    .. [2] B. D. Johnson et al., "Stellar Population Inference," ApJS, 254,
       22 (2021). arXiv:2012.01426. https://doi.org/10.3847/1538-4295/abef67
    """
    raw = _load_nenkova_arrays(grid_path)

    # ensure_compile_time_eval so the cached closure captures concrete arrays
    # even if the first call happens inside jax.jit (mirrors the SKIRTOR path).
    with jax.ensure_compile_time_eval():
        grid_jax = jnp.asarray(raw["template"])
        wave_grid = jnp.asarray(raw["wavelength"])
        tau_axis = jnp.asarray(raw["tau_axis"])
        edges = (edges_for_grid(tau_axis),)

    def nenkova_grid(
        wavelength: jnp.ndarray,
        agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
        agn_tau: float = 30.0,
        agn_torus_frac: float = 0.5,
        **_kwargs,
    ) -> jnp.ndarray:
        r"""Nenkova+2008 CLUMPY torus SED at a single equatorial optical depth.

        Parameters
        ----------
        wavelength: array_like, shape (n_wave,)
            Rest-frame wavelength grid. [Å]
        agn_log_lbol: float, optional
            Bolometric luminosity, ``log10(L_bol / L_sun)``. Default 10.0.
        agn_tau: float, optional
            Equatorial optical depth of the clumpy torus. Valid over the grid
            extent (5–150). Default 30.0. [dimensionless]
        agn_torus_frac: float, optional
            Fraction of L_bol re-emitted by the torus (covering factor).
            Default 0.5. [dimensionless]

        Returns
        -------
        ndarray, shape (n_wave,)
            Spectral luminosity density. [erg/s/Hz]

        Notes
        -----
        .. math::

            L_\nu(\lambda) = L_{\rm bol}\,f_{\rm torus}\,
                             \frac{T(\lambda,\,\tau)}
                                  {\int T(\nu,\,\tau)\,\mathrm{d}\nu}

        where :math:`T` is the tabulated CLUMPY template and the integral is
        evaluated on the (sorted) frequency grid of ``wavelength``.

        **JIT-compatible**: yes. **Gradient-safe**: yes, ``agn_tau`` is a
        differentiable, traceable parameter.
        """
        # Nenkova tau axis is non-uniform (I6 fix #1851).
        # Use index-space interpolation for correct gradients throughout the range.
        template = interp_nd_triweight(
            grid_jax, (tau_axis,), edges, (agn_tau,), index_space_interp=True
        )
        sed = resample_template(wavelength, wave_grid, template, left=0.0, right=0.0)
        nu = _wavelength_to_nu(wavelength)
        integral_safe = _bolometric_integral_nu(sed, nu, floor=1e-100)
        l_scale = 10.0**agn_log_lbol * _L_SUN * agn_torus_frac
        return l_scale * sed / integral_safe

    return nenkova_grid


def _find_nenkova_grid() -> str:
    from tengri._data_setup import find_data

    # Searches every directory the old parents[4] walk reached, plus
    # $TENGRI_DATA_DIR (#1431).
    found = find_data(*_NENKOVA_GRID_SEARCH_PATHS)
    if found is not None:
        return str(found)
    raise FileNotFoundError(_NENKOVA_NOT_FOUND_MSG)


@functools.cache
def _load_nenkova_default() -> Callable:
    return create_nenkova_from_grid(_find_nenkova_grid())


def nenkova_torus(*args, **kwargs) -> jnp.ndarray:
    """AGN torus emission from Nenkova et al. (2008) CLUMPY templates.

    Interpolates the CLUMPY radiative-transfer torus library in equatorial
    optical depth ``agn_tau`` with a pure-JAX triweight kernel, then normalizes
    to ``agn_torus_frac * L_bol``. This is the production-quality torus
    matching FSPS / Prospector (Johnson et al. 2021 [2]_); for silicate-feature
    accuracy the SKIRTOR templates in ``tengri.components.agn.skirtor`` may be
    preferred.

    Parameters
    ----------
    wavelength: array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol: float, optional
        ``log10(L_bol / L_sun)``. AGN bolometric luminosity. Default 10.0.
    agn_tau: float, optional
        Equatorial optical depth of the clumpy torus. Valid range 5–150.
        Default 30.0. [dimensionless]
    agn_torus_frac: float, optional
        Fraction of L_bol re-emitted by the torus (covering factor).
        Default 0.5.

    Returns
    -------
    L_nu: jnp.ndarray, shape (n_wave,)
        Specific luminosity [erg s^-1 Hz^-1].

    Raises
    ------
    FileNotFoundError
        If no vendored Nenkova grid HDF5 is present on disk. Build it with
        ``scripts/build_nenkova_grid.py``.

    Notes
    -----
    **JIT-compatible**: yes, loads the vendored grid
    (``data/nenkova08_torus_grid.h5``) once via a cached closure, then
    interpolates with pure JAX. **Gradient-safe**: yes, ``agn_tau`` is a
    differentiable, traceable parameter (it can be freely sampled/optimized by
    MAP, NUTS, and VI).

    Data source: the same CLUMPY templates shipped with FSPS
    (``$SPS_HOME/dust/Nenkova08_y010_torusg_n10_q2.0.dat``; Conroy & Gunn 2010)
    and used by Prospector (Johnson et al. 2021 [2]_), vendored into
    ``data/nenkova08_torus_grid.h5`` by ``scripts/build_nenkova_grid.py``.

    References
    ----------
    .. [1] M. Nenkova et al., "AGN Dusty Tori. I. Handling of Clumpy Media,"
       ApJ, 685, 147 (2008). https://doi.org/10.1086/590482
    .. [2] B. D. Johnson et al., "Stellar Population Inference," ApJS, 254,
       22 (2021). arXiv:2012.01426. https://doi.org/10.3847/1538-4295/abef67

    Examples
    --------
    >>> import jax, jax.numpy as jnp
    >>> wave = jnp.logspace(3, 6, 100)
    >>> sed = nenkova_torus(wave, agn_log_lbol=12.0, agn_tau=30.0)
    >>> sed.shape
    (100,)
    >>> # agn_tau is differentiable and JIT-safe:
    >>> _ = jax.jit(lambda t: nenkova_torus(wave, agn_log_lbol=12.0, agn_tau=t))(30.0)
    """
    return _load_nenkova_default()(*args, **kwargs)
