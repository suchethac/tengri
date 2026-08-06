# SPDX-License-Identifier: BSD-3-Clause
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
.. [1] M. Stalevski et al., "3D radiative transfer modeling of the dusty
   torus around AGN — the influence of clumping," MNRAS, 420, 2756 (2012).
   arXiv:1109.1286. https://doi.org/10.1111/j.1365-2966.2011.19775.x
.. [2] M. Stalevski et al., "The dust covering factor in AGN — combining the
   IR torus emission with polar dust component," MNRAS, 458, 2288 (2016).
   arXiv:1602.01954. https://doi.org/10.1093/mnras/stw444
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
)
from tengri.utils.grid_interp import interp_nd_pchip, interp_nd_triweight, resample_template
from tengri.utils.interpolation import edges_for_grid

#: Speed of light in Å/s. Used for L_λ ↔ L_ν conversions on SKIRTOR's
#: Angstrom-grid templates. Matches the value used in
#: ``tengri.components.agn.blocks.runner.C_AA_PER_S``.
from tengri.utils.physics_constants import C_AA as _C_AA_PER_S


class SKIRTORComponents(NamedTuple):
    """Separate SKIRTOR spectral components.

    Attributes
    ----------
    disk : jnp.ndarray, shape (n_wave,)
        Accretion disk emission (direct + scattered) [erg/s/Hz].
    dust : jnp.ndarray, shape (n_wave,)
        Dust thermal emission from the torus [erg/s/Hz].
    total : jnp.ndarray, shape (n_wave,)
        Total emission (disk + dust) [erg/s/Hz].

    Notes
    -----
    This is a lightweight container for decomposed SKIRTOR output.
    All arrays are rest-frame rest-frame spectral luminosity densities.
    The ``total`` field is the sum of ``disk`` and ``dust``.
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

    Notes
    -----
    **JIT-compatible**: no — performs file I/O at module load time.
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
                has_R = "radius_ratio" in f["grid"]
                axes = [
                    np.array(f["grid/tau_97"][:]),
                    np.array(f["grid/p"][:]),
                    np.array(f["grid/q"][:]),
                    np.array(f["grid/opening_angle"][:]),
                ]
                if has_R:
                    axes.append(np.array(f["grid/radius_ratio"][:]))
                axes.append(np.array(f["grid/cos_inclination"][:]))
                result["axes"] = tuple(axes)
                result["has_radius_ratio"] = has_R
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
                result["has_radius_ratio"] = False

    # ``has_radius_ratio`` tells callers whether the grid carries the R axis
    # (6-axis: tau, p, q, oa, radius_ratio, cos_inc) or is a legacy 5-axis grid
    # (tau, p, q, oa, cos_inc). The interpolation helpers drop the R coordinate
    # from the query point for legacy grids — see ``_match_point_to_axes`` (#772).
    return result


def _match_point_to_axes(point: tuple, axes: tuple) -> tuple:
    """Drop the R (radius_ratio) coordinate when the grid lacks that axis.

    Interpolation points are built with the R coordinate at index 4 (after
    ``oa``). Legacy 5-axis grids (no ``radius_ratio``) need it removed so the
    point length matches the grid (#772).
    """
    if len(point) == len(axes) + 1:
        return point[:4] + point[5:]
    return point


def _interpolate_and_normalize(
    grid_jax: jnp.ndarray,
    wave_grid: jnp.ndarray,
    axes: tuple,
    edges: tuple,
    wavelength: jnp.ndarray,
    point: tuple,
    l_scale: float,
) -> jnp.ndarray:
    r"""Interpolate a template grid and normalize to physical L_ν.

    SKIRTOR v3 templates are stored as L_λ-like (the download script does
    ``disk /= wl`` and normalizes by ``trapezoid(dust, wl)`` — issue #459).
    The original implementation integrated the L_λ array against the
    frequency grid and treated the output as L_ν, leaving the returned
    array with L_λ shape — so νL_ν (the visible torus IR bump) peaked at
    ~5 µm instead of the SKIRTOR 30–50 µm thermal bump.

    Fix: normalize the bolometric integral in the wavelength variable
    (matching the convention used at template-build time) and convert
    L_λ → L_ν at the end via L_ν = L_λ × λ²/c.

    Parameters
    ----------
    grid_jax : ndarray, shape (n_tau, n_p, n_q, n_oa, n_inc, n_wave)
        L_λ-like template grid [erg/s/Å, normalized per unit bolometric].
    wave_grid : ndarray, shape (n_wave_grid,)
        Grid wavelength array [Angstrom].
    axes : tuple of ndarray
        Grid axis values (tau, p, q, oa, cos_inc).
    edges : tuple of ndarray
        Precomputed bin edges for triweight interpolation.
    wavelength : ndarray, shape (n_wave,)
        Target wavelength array [Angstrom].
    point : tuple
        (tau, p, q, oa, cos_inc) query point.
    l_scale : float
        Bolometric luminosity scale factor [erg/s].

    Returns
    -------
    ndarray, shape (n_wave,)
        Specific luminosity L_ν [erg s^-1 Hz^-1].

    Notes
    -----
    **JIT-compatible**: yes — uses ``resample_template`` and ``jax.vmap``.

    The template is put on the user wavelength grid by
    :func:`~tengri.utils.grid_interp.resample_template`, which interpolates in
    log λ and log flux. The native SKIRTOR grid is coarse (136 points, R ~ 7),
    and its tails are power laws, which that form reproduces exactly.

    **Citation**: matches CIGALE skirtor2016 processing (see
    ``scripts/download_skirtor_templates.py``).
    """
    template = interp_nd_triweight(grid_jax, axes, edges, _match_point_to_axes(point, axes))
    # Bolometric integral on the *template* wavelength grid (full UV–FIR
    # coverage). Using the user wave grid would clip the FIR tail and
    # over-normalize on truncated grids; trapezoid in λ matches the
    # download script's normalization convention.
    #
    # ``wave_grid`` is monotonically ascending (set at load time in
    # ``_load_grid_arrays``), so trapezoid integrates correctly without
    # an explicit ``argsort`` — the prior ``trapezoid(sed[idx_sort],
    # nu[idx_sort])`` pattern existed because ``nu = c/λ`` was
    # *descending* and needed reordering. Don't re-add a sort here.
    integral_lam = jnp.trapezoid(template, wave_grid)
    integral_safe = jnp.maximum(jnp.abs(integral_lam), 1e-100)
    template_lam = l_scale * template / integral_safe  # erg/s/Å
    sed_lam = resample_template(wavelength, wave_grid, template_lam, left=0.0, right=0.0)
    # L_λ → L_ν: L_ν = L_λ × λ²/c (c in Å/s).
    return sed_lam * wavelength**2 / _C_AA_PER_S


class SKIRTORGrid(NamedTuple):
    """Threadable SKIRTOR template arrays — a JAX pytree.

    Holds the interpolation grid as pure array data so it threads through
    ``jax.jit`` as a runtime input (small compile), instead of baking into the
    trace as a constant or — worse — being threaded as a Python closure, which
    JAX cannot treat as a traced argument (issue #1198).

    Attributes
    ----------
    grid : ndarray
        Template luminosity cube (dust-only for v3 grids, total for v2).
    wave_grid : ndarray, shape (n_wave,)
        Template rest-frame wavelength grid. [Å]
    axes : tuple of ndarray
        Parameter-axis node values (tau, p, q, oa, radius, inc).
    edges : tuple of ndarray
        Per-axis triweight-kernel bin edges.
    """

    grid: jnp.ndarray
    wave_grid: jnp.ndarray
    axes: tuple
    edges: tuple


def _load_skirtor_grid_data(grid_path: str) -> SKIRTORGrid:
    """Load a SKIRTOR grid file into a threadable :class:`SKIRTORGrid` pytree.

    Prefers the v3 dust-only template when present (avoids disc/dust
    double-counting); falls back to the v2 ``total`` cube.
    """
    raw = _load_grid_arrays(grid_path)
    # ``ensure_compile_time_eval`` so the concrete arrays are captured even if
    # the first call happens inside a jit trace (mirrors the legacy closure).
    with jax.ensure_compile_time_eval():
        _grid_key = "dust" if "dust" in raw else "total"
        grid = jnp.array(raw[_grid_key])
        wave_grid = jnp.array(raw["wave"])
        axes = tuple(jnp.array(ax) for ax in raw["axes"])
        edges = tuple(edges_for_grid(ax) for ax in axes)
    return SKIRTORGrid(grid=grid, wave_grid=wave_grid, axes=axes, edges=edges)


def _skirtor_grid_sed(
    wavelength: jnp.ndarray,
    grid_data: SKIRTORGrid,
    agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
    agn_tau_skirtor: float = 7.0,
    agn_p_skirtor: float = 1.0,
    agn_q_skirtor: float = 1.0,
    agn_oa_skirtor: float = 40.0,
    agn_radius_ratio: float = 20.0,
    agn_cos_inc: float = 0.86602540378443864,
    agn_torus_frac: float = 0.5,
    **_kwargs,
) -> jnp.ndarray:
    r"""SKIRTOR torus :math:`L_\nu` interpolated from a threaded grid.

    Pure JAX: ``grid_data`` carries ordinary arrays, so this composes under
    ``jax.jit`` whether the grid is closed over (eager) or passed as a runtime
    argument (threaded). See :func:`create_skirtor_from_grid` for the physics
    of each parameter.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    grid_data : SKIRTORGrid
        Threadable template arrays.
    agn_log_lbol, agn_tau_skirtor, agn_p_skirtor, agn_q_skirtor, \
agn_oa_skirtor, agn_radius_ratio, agn_cos_inc, agn_torus_frac : float
        SKIRTOR grid coordinates and normalization (see
        :func:`create_skirtor_from_grid`).

    Returns
    -------
    ndarray, shape (n_wave,)
        Specific luminosity :math:`L_\nu`. [erg s⁻¹ Hz⁻¹]

    Notes
    -----
    **JIT/grad-safe**: yes — triweight interpolation over threaded arrays.
    """
    l_scale = 10.0**agn_log_lbol * _L_SUN * agn_torus_frac
    point = (
        agn_tau_skirtor,
        agn_p_skirtor,
        agn_q_skirtor,
        agn_oa_skirtor,
        agn_radius_ratio,
        agn_cos_inc,
    )
    return _interpolate_and_normalize(
        grid_data.grid,
        grid_data.wave_grid,
        grid_data.axes,
        grid_data.edges,
        wavelength,
        point,
        l_scale,
    )


@functools.cache
def _load_skirtor_default_grid() -> SKIRTORGrid:
    """Cached default SKIRTOR grid arrays (first grid file found on disk)."""
    return _load_skirtor_grid_data(_find_skirtor_grid())


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
    Grid loading is cached via ``@functools.cache``.

    **Gradient-safe**: yes — triweight interpolation is fully differentiable.

    Supports v2 (total-only) and v3 (separate disk/dust) HDF5 layouts.
    When v3 is available, use ``create_skirtor_components_from_grid``
    to access individual disk and dust spectra.

    References
    ----------
    .. [1] M. Stalevski et al., "3D radiative transfer modeling of the dusty
       torus around AGN," MNRAS, 420, 2756 (2012). arXiv:1109.1286.
       https://doi.org/10.1111/j.1365-2966.2011.19775.x
    .. [2] M. Stalevski et al., "The dust covering factor in AGN," MNRAS, 458,
       2288 (2016). arXiv:1602.01954. https://doi.org/10.1093/mnras/stw444
    """
    # Single-sourced array loading (dust-preferred for v3, total for v2) so the
    # closure below and the threaded :func:`_skirtor_grid_sed` share one grid.
    grid_data = _load_skirtor_grid_data(grid_path)

    def skirtor_grid(
        wavelength: jnp.ndarray,
        agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
        agn_tau_skirtor: float = 7.0,
        agn_p_skirtor: float = 1.0,
        agn_q_skirtor: float = 1.0,
        agn_oa_skirtor: float = 40.0,
        agn_radius_ratio: float = 20.0,
        agn_cos_inc: float = 0.86602540378443864,
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
            Fraction of L_bol reprocessed by the torus, integrated
            over the dust-emission spectrum. With the v3 grid (dust-
            only template), this is the true covering-factor × dust
            absorption efficiency. [dimensionless]

        Returns
        -------
        ndarray, shape (n_wave,)
            Specific luminosity L_ν. [erg s⁻¹ Hz⁻¹]
        """
        return _skirtor_grid_sed(
            wavelength,
            grid_data,
            agn_log_lbol=agn_log_lbol,
            agn_tau_skirtor=agn_tau_skirtor,
            agn_p_skirtor=agn_p_skirtor,
            agn_q_skirtor=agn_q_skirtor,
            agn_oa_skirtor=agn_oa_skirtor,
            agn_radius_ratio=agn_radius_ratio,
            agn_cos_inc=agn_cos_inc,
            agn_torus_frac=agn_torus_frac,
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

            fn(wavelength, agn_log_lbol, ..., frac_agn, **kwargs)
                -> SKIRTORComponents(disk, dust, total)

        Each component is in [erg s^-1 Hz^-1].

    Raises
    ------
    KeyError
        If the grid lacks ``spectra/disk_emission`` or
        ``spectra/dust_emission``.

    Notes
    -----
    **JIT-compatible**: yes — the returned function is pure JAX.
    Grid loading is cached via ``@functools.cache``.

    **Gradient-safe**: yes — triweight interpolation is fully differentiable.

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

    # See note in create_skirtor_from_grid — wrap conversion in
    # ensure_compile_time_eval to keep the cached closure trace-safe.
    with jax.ensure_compile_time_eval():
        disk_jax = jnp.array(raw["disk"])
        dust_jax = jnp.array(raw["dust"])
        total_jax = jnp.array(raw["total"])
        wave_grid = jnp.array(raw["wave"])
        axes = tuple(jnp.array(ax) for ax in raw["axes"])
        edges = tuple(edges_for_grid(ax) for ax in axes)

    def skirtor_components(
        wavelength: jnp.ndarray,
        agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
        agn_tau_skirtor: float = 7.0,
        agn_p_skirtor: float = 1.0,
        agn_q_skirtor: float = 1.0,
        agn_oa_skirtor: float = 40.0,
        agn_radius_ratio: float = 20.0,
        agn_cos_inc: float = 0.86602540378443864,
        frac_agn: float = 0.5,
        agn_torus_frac: float | None = None,  # deprecated; falls back to frac_agn
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
        frac_agn : float
            CIGALE-style AGN fraction (L_AGN / L_total) in a configurable band.
            [dimensionless, 0–1]
        agn_torus_frac : float, optional
            **Deprecated**: use frac_agn instead.

        Returns
        -------
        SKIRTORComponents
            Named tuple with ``disk``, ``dust``, ``total`` arrays,
            each shape (n_wave,) in [erg s⁻¹ Hz⁻¹].
        """
        # Handle deprecated parameter
        if agn_torus_frac is not None:
            luminosity_frac = agn_torus_frac
        else:
            luminosity_frac = frac_agn

        l_scale = 10.0**agn_log_lbol * _L_SUN * luminosity_frac
        point = (
            agn_tau_skirtor,
            agn_p_skirtor,
            agn_q_skirtor,
            agn_oa_skirtor,
            agn_radius_ratio,
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


def skirtor_disc_dust_ratio(
    wave: jnp.ndarray,
    disc_lambda_unreddened: jnp.ndarray,
    disc_ext_fac: jnp.ndarray,
    *,
    agn_tau_skirtor: float = 7.0,
    agn_p_skirtor: float = 1.0,
    agn_q_skirtor: float = 1.0,
    agn_oa_skirtor: float = 40.0,
    agn_radius_ratio: float = 20.0,
    agn_cos_inc: float = 0.86602540378443864,
) -> jnp.ndarray:
    r"""CIGALE disc/dust bolometric ratio ``R = lumin_disk / lumin_dust``.

    Replicates CIGALE ``skirtor2016.py`` so the composable AGN can tie the
    disc to the single ``agn_power`` reference (energy-conserving). The
    analytic disc shape is renormalized to the **face-on** SKIRTOR disc
    integral ``∫disk(i=0)``, reweighted by the inclination ratio
    ``disk(i)/disk(0)``, reddened, and the **anisotropy factor**
    ``η(i) = cos(i)(1+2cos(i))/3`` (= 0.789 at i=30°) applied — then divided
    by the SKIRTOR dust integral ``∫dust(i)``::

        R = η(i) · ∫[ŝ·∫disk(0) · disk(i)/disk(0) · ext_fac] dλ / ∫dust(i) dλ

    where ``ŝ`` is the unit-area analytic disc shape. At the §9 fiducial this
    yields R ≈ 2.23, matching CIGALE ``lumin_disk/total_dust = 2.22``. The η
    factor (the disc's anisotropic emission, Stalevski+2016 §2.2.1) is what
    sets the *observed* disc bolometric with viewing angle and couples it to
    the polar-dust energy balance.

    Parameters
    ----------
    wave : ndarray, shape (n_wave,)
        Rest-frame wavelength grid [Å].
    disc_lambda_unreddened : ndarray, shape (n_wave,)
        Analytic disc spectrum *before* reddening, any normalization
        (shape only is used). [erg/s/Å]
    disc_ext_fac : ndarray, shape (n_wave,)
        Line-of-sight reddening factor ``10^(-0.4·k·E(B-V))`` (1.0 = no
        reddening). [dimensionless]
    agn_tau_skirtor, agn_p_skirtor, agn_q_skirtor, agn_oa_skirtor, agn_cos_inc
        SKIRTOR grid parameters (edge-on τ_9.7, radial/polar density
        indices, half-opening angle, cos inclination).

    Returns
    -------
    R : ndarray, scalar
        Disc/dust bolometric luminosity ratio. Returns 1.0 if the v3 grid
        (separate disk/dust components) is unavailable.
    incl_ratio : ndarray, shape (n_wave,)
        Wavelength-dependent disc inclination attenuation ``disk(i)/disk(0)``
        for reweighting the disc output spectrum. Ones if the grid is
        unavailable.
    R_faceon : ndarray, scalar
        Face-on UN-reddened disc/dust ratio ``∫disk(i=0)/∫dust(i)`` — the
        ratio the polar ``l_ext`` proxy needs (CIGALE ``l_ext =
        geom·∫AGN1.disk·(1-ext_fac)``). 1.0 if the grid is unavailable.

    Notes
    -----
    **JIT-compatible**: yes — grid interpolation is pure JAX.

    **Reference**: Implements CIGALE ``skirtor2016.py`` (Boquien+2019).
    """
    raw = _load_raw_disk_dust_grid()
    if raw is None:
        return jnp.asarray(1.0), jnp.ones_like(wave), jnp.asarray(1.0)
    disk_jax, dust_jax, wave_grid, axes = raw

    def _interp_native(grid, cos_inc):
        point = (
            jnp.asarray(agn_tau_skirtor),
            jnp.asarray(agn_p_skirtor),
            jnp.asarray(agn_q_skirtor),
            jnp.asarray(agn_oa_skirtor),
            jnp.asarray(agn_radius_ratio),
            jnp.asarray(cos_inc),
        )
        # RAW interpolated L_λ template on the NATIVE grid — NO per-component
        # normalization (preserves the disk/dust ratio) and NO wave resampling
        # (CIGALE integrates lumin_disk/lumin_dust on the SKIRTOR template grid;
        # resampling onto the user grid distorts the integrals → R ~10% off).
        # PCHIP (node-EXACT) not triweight (a smoother): the triweight kernel
        # does not pass through grid nodes, distorting the disk/dust *integral
        # ratio* by ~10% even at exact-node params (R 2.45 vs CIGALE 2.22).
        return interp_nd_pchip(grid, axes, _match_point_to_axes(point, axes))

    disk_i_n = _interp_native(disk_jax, agn_cos_inc)
    dust_i_n = _interp_native(dust_jax, agn_cos_inc)
    disk_0_n = _interp_native(disk_jax, 1.0)  # face-on

    # --- R on the native template grid (CIGALE convention) ---
    # Bring the analytic disc shape + reddening onto the native grid.
    disc_n = resample_template(wave_grid, wave, disc_lambda_unreddened, left=0.0, right=0.0)
    ext_n = resample_template(wave_grid, wave, disc_ext_fac, left=1.0, right=1.0)
    int_disk0 = jnp.trapezoid(disk_0_n, wave_grid)
    shape_n = disc_n / jnp.maximum(jnp.trapezoid(disc_n, wave_grid), 1e-30)
    disk_analytic = shape_n * int_disk0
    # CIGALE nan_to_num: zero the disc where the face-on disc vanishes.
    incl_n = jnp.where(disk_0_n > 0, disk_i_n / jnp.where(disk_0_n > 0, disk_0_n, 1.0), 0.0)
    sk_disk_reddened = disk_analytic * incl_n * ext_n

    int_dust = jnp.maximum(jnp.trapezoid(dust_i_n, wave_grid), 1e-30)
    eta = agn_cos_inc * (1.0 + 2.0 * agn_cos_inc) / 3.0
    R = eta * jnp.trapezoid(sk_disk_reddened, wave_grid) / int_dust
    # ``R_faceon`` = ∫AGN1.disk(face-on, UN-reddened) / ∫dust — the ratio the
    # polar ``l_ext`` proxy needs (CIGALE l_ext = geom·∫AGN1.disk·(1-ext_fac)),
    # distinct from ``R`` (the reddened, inclination-weighted *observed* disc
    # used for the disc output bolometric).
    R_faceon = int_disk0 / int_dust
    # ``incl_ratio`` on the *user* grid for the disc-shape reweighting.
    incl_ratio = resample_template(wave, wave_grid, incl_n, left=0.0, right=0.0)
    # ``incl_ratio`` = disk(i)/disk(0) is the wavelength-dependent SKIRTOR
    # inclination attenuation of the disc continuum (CIGALE
    # ``SKIRTOR.disk(i)/AGN1.disk(0)``); the caller applies it to the disc
    # output *shape* so the disc spectrum (not just its bolometric R) is
    # inclination-correct.
    return R, incl_ratio, R_faceon


@functools.cache
def _load_raw_disk_dust_grid():
    """Load raw (un-normalized) SKIRTOR disk/dust template grids for R.

    Returns ``(disk_jax, dust_jax, wave_grid, axes)`` or ``None`` if the v3
    grid (separate disk/dust components) is unavailable. Any descending axis
    (the grid stores ``cos_inclination`` 1→0) is reversed so the node-exact
    PCHIP interpolant used for R sees strictly-ascending coordinates.
    """
    import numpy as _np

    raw = _load_grid_arrays(_find_skirtor_grid())
    if "disk" not in raw or "dust" not in raw:
        return None
    disk = _np.asarray(raw["disk"])
    dust = _np.asarray(raw["dust"])
    axes_list = [_np.asarray(ax) for ax in raw["axes"]]
    for i, ax in enumerate(axes_list):
        if ax.size > 1 and ax[0] > ax[-1]:  # descending → reverse axis i
            axes_list[i] = ax[::-1]
            disk = _np.flip(disk, axis=i)
            dust = _np.flip(dust, axis=i)
    with jax.ensure_compile_time_eval():
        disk_jax = jnp.array(disk)
        dust_jax = jnp.array(dust)
        wave_grid = jnp.array(raw["wave"])
        axes = tuple(jnp.array(ax) for ax in axes_list)
    return disk_jax, dust_jax, wave_grid, axes


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
    from tengri._data_setup import find_data

    # find_data searches every directory the old parents[4] walk reached, plus
    # $TENGRI_DATA_DIR (#1431), and keeps the v3 > v2 > npz preference order.
    found = find_data(*_GRID_SEARCH_PATHS)
    if found is not None:
        return str(found)
    raise FileNotFoundError(_NOT_FOUND_MSG)


def _find_skirtor_raw_grid() -> str | None:
    """Locate the faithful full-coverage SKIRTOR v4 grid, or None if absent.

    Built by ``scripts/build_skirtor_raw_grid.py``; carries the full
    ``ta,p,q,oa,R,i`` axes and the published radiative-transfer total.
    """
    from tengri._data_setup import find_data

    found = find_data("skirtor_raw_v4.h5")
    return str(found) if found is not None else None


def create_skirtor_raw_total_from_grid(grid_path: str) -> Callable:
    r"""Loader for the faithful v4 SKIRTOR grid: the published RT total SED.

    The v4 grid (``scripts/build_skirtor_raw_grid.py``) carries the full
    ``(tau_97, p, q, opening_angle, radius_ratio, inclination)`` axes and stores
    the **radiative-transfer total** (``.dat`` column 2) — the disc + torus as
    Stalevski (2016) computed them, with no analytic-disc substitution. The
    returned function interpolates that total at any point (so a fixed parameter
    is just a slice) and scales it to the requested bolometric luminosity.

    Parameters
    ----------
    grid_path : str
        Path to a v4 SKIRTOR HDF5 grid.

    Returns
    -------
    callable
        ``fn(wavelength, agn_log_lbol, agn_tau_skirtor, agn_p_skirtor,
        agn_q_skirtor, agn_oa_skirtor, agn_radius_ratio, agn_cos_inc,
        frac_agn) -> L_nu [erg/s/Hz]``.

    Notes
    -----
    **JIT-compatible**: yes — pure-JAX triweight interpolation; grid load cached.
    """
    import h5py

    with h5py.File(grid_path, "r") as f:
        ta = f["grid/tau_97"][:]
        p = f["grid/p"][:]
        q = f["grid/q"][:]
        oa = f["grid/opening_angle"][:]
        rr = f["grid/radius_ratio"][:]
        ideg = f["grid/inclination_deg"][:]
        total = f["spectra/total_emission"][:]
        wave = f["wavelength"][:]

    with jax.ensure_compile_time_eval():
        total_j = jnp.array(total)
        wave_g = jnp.array(wave)
        axes = tuple(jnp.array(a) for a in (ta, p, q, oa, rr, ideg))
        edges = tuple(edges_for_grid(a) for a in axes)
        nu_g = _C_AA_PER_S / wave_g
        order = jnp.argsort(nu_g)

    def fn(
        wavelength,
        agn_log_lbol,
        agn_tau_skirtor=7.0,
        agn_p_skirtor=1.0,
        agn_q_skirtor=1.0,
        agn_oa_skirtor=40.0,
        agn_radius_ratio=20.0,
        agn_cos_inc=0.86602540378443864,
        frac_agn=1.0,
        **_kwargs,
    ):
        i_deg = jnp.degrees(jnp.arccos(jnp.clip(agn_cos_inc, 0.0, 1.0)))
        point = (
            agn_tau_skirtor,
            agn_p_skirtor,
            agn_q_skirtor,
            agn_oa_skirtor,
            agn_radius_ratio,
            i_deg,
        )
        spec = interp_nd_triweight(total_j, axes, edges, point)  # L_nu shape on wave_g
        l_scale = 10.0**agn_log_lbol * _L_SUN * frac_agn
        bolo = jnp.trapezoid(spec[order], nu_g[order])
        spec_n = spec * (l_scale / jnp.maximum(jnp.abs(bolo), 1e-100))
        return resample_template(wavelength, wave_g, spec_n, left=0.0, right=0.0)

    return fn


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


def create_skirtor_disc_attenuation_from_grid(grid_path: str) -> Callable:
    r"""Build the SKIRTOR inclination-dependent disc-attenuation factor.

    Returns a callable that produces the wavelength-dependent ratio
    :math:`\rm SKIRTOR.disk(\lambda; i, \tau, p, q, oa) /
    SKIRTOR.disk(\lambda; i=0, \tau, p, q, oa)` (face-on baseline).
    Multiplied with the analytic Schartmann-2005 disc shape, this
    reproduces CIGALE ``skirtor2016.py:336`` (Boquien+2019)::

        SKIRTOR2016.disk = analytic × SKIRTOR.disk_at_i / AGN1.disk_at_face

    For face-on type-1 (i ≲ 90°−oa, e.g. i=30° with oa=40°), the
    ratio is near unity at most wavelengths; for type-2 sightlines it
    captures the clumpy self-attenuation through the equatorial torus
    that CIGALE's analytic-disc replacement is supposed to preserve.

    Parameters
    ----------
    grid_path : str
        Path to a v3 SKIRTOR HDF5 file with ``spectra/disk_emission``.

    Returns
    -------
    callable
        ``att_fn(wavelength, agn_tau_skirtor, agn_p_skirtor, agn_q_skirtor,
        agn_oa_skirtor, agn_cos_inc) -> ndarray`` of shape ``(n_wave,)``,
        dimensionless attenuation factor. Returns 1.0 everywhere when
        the disk grid is unavailable (v2 fallback).

    Notes
    -----
    **JIT-compatible**: yes — triweight interpolation in JAX.

    **Gradient-safe**: yes.
    """
    raw = _load_grid_arrays(grid_path)

    if "disk" not in raw:
        # v2 grid: no separate disc column, return identity attenuation
        def _identity_att(wavelength, *_, **__):
            return jnp.ones_like(jnp.asarray(wavelength))

        return _identity_att

    with jax.ensure_compile_time_eval():
        disk_grid = jnp.array(raw["disk"])
        wave_grid = jnp.array(raw["wave"])
        axes = tuple(jnp.array(ax) for ax in raw["axes"])
        edges = tuple(edges_for_grid(ax) for ax in axes)

    def disc_attenuation(
        wavelength: jnp.ndarray,
        agn_tau_skirtor: float = 7.0,
        agn_p_skirtor: float = 1.0,
        agn_q_skirtor: float = 1.0,
        agn_oa_skirtor: float = 40.0,
        agn_radius_ratio: float = 20.0,
        agn_cos_inc: float = 0.86602540378443864,  # cos(30°)
    ) -> jnp.ndarray:
        r"""Wavelength-dependent disc attenuation factor at chosen i.

        Returns ``SKIRTOR.disk(i) / SKIRTOR.disk(i=0)`` interpolated to
        the requested wavelength grid.
        """
        point_i = (
            agn_tau_skirtor,
            agn_p_skirtor,
            agn_q_skirtor,
            agn_oa_skirtor,
            agn_radius_ratio,
            agn_cos_inc,
        )
        point_face = (
            agn_tau_skirtor,
            agn_p_skirtor,
            agn_q_skirtor,
            agn_oa_skirtor,
            agn_radius_ratio,
            1.0,  # cos_inc = 1 = face-on
        )
        disk_at_i = interp_nd_triweight(
            disk_grid, axes, edges, _match_point_to_axes(point_i, axes)
        )
        disk_at_face = interp_nd_triweight(
            disk_grid, axes, edges, _match_point_to_axes(point_face, axes)
        )
        # Safe ratio: where face-on is zero (shouldn't be, but be safe),
        # return 0 attenuation (no contribution).
        ratio_template = jnp.where(
            disk_at_face > 1e-30,
            disk_at_i / jnp.maximum(disk_at_face, 1e-30),
            0.0,
        )
        # Interpolate to user wave grid; clip to [0, 1.5] so numerical
        # noise can't introduce un-physically large amplifications
        # (face-on baseline is the maximum disc visibility).
        ratio = resample_template(wavelength, wave_grid, ratio_template, left=1.0, right=1.0)
        return jnp.clip(ratio, 0.0, 1.5)

    return disc_attenuation


@functools.cache
def _load_skirtor_disc_attenuation():
    """Build SKIRTOR disc attenuation pattern, cached.

    Returns an identity function when the v2 grid is used (no disc grid).
    """
    path = _find_skirtor_grid()
    return create_skirtor_disc_attenuation_from_grid(path)


def skirtor_disc_attenuation(*args, **kwargs):
    r"""SKIRTOR inclination-dependent disc attenuation factor (auto-loaded).

    Wraps :func:`_load_skirtor_disc_attenuation`. Identity-1.0 when the
    v3 disc grid is unavailable (v2 fallback). See
    :func:`create_skirtor_disc_attenuation_from_grid` for the signature
    and physics.
    """
    return _load_skirtor_disc_attenuation()(*args, **kwargs)


def skirtor_sed(*args, **kwargs):
    """SKIRTOR torus SED (auto-loaded from tabulated templates).

    This function uses the tabulated Stalevski+2016 template grid
    with 5D triweight interpolation.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom].
    agn_log_lbol : float, optional
        AGN bolometric luminosity [log10(L_sun)]. Default: 10.0.
    agn_tau_skirtor : float, optional
        V-band optical depth of torus [dimensionless]. Default: 7.0.
        Range: ~1–15 (grid-dependent).
    agn_p_skirtor : float, optional
        Radial dust distribution power-law index [dimensionless].
        Default: 1.0. Range: ~0–2 (grid-dependent).
    agn_q_skirtor : float, optional
        Radial dust distribution power-law index (temperature profile)
        [dimensionless]. Default: 1.0. Range: ~0–2 (grid-dependent).
    agn_oa_skirtor : float, optional
        Half-opening angle of the torus [degrees]. Default: 40.0.
        Range: ~10–80° (grid-dependent).
    agn_cos_inc : float, optional
        Cosine of inclination angle [dimensionless, 0–1].
        Default: 0.5 (60°).
    agn_torus_frac : float, optional
        Fraction of bolometric luminosity from torus [dimensionless, 0–1].
        Default: 0.5.
    _template : callable, optional
        Pre-loaded template function (for JIT threading). When provided,
        uses this instead of the module-level cached loader. Internal use.
    **kwargs
        Additional keyword arguments (ignored for compatibility).

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density L_ν [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — delegates to cached grid function or
    pre-loaded template (when _template is threaded).

    See ``create_skirtor_from_grid`` for full parameter documentation and
    grid-dependent ranges.
    """
    # Allow the template to be threaded as a JIT runtime input. Preferred form
    # is a SKIRTORGrid pytree of arrays (threads as a runtime arg, small
    # compile); a legacy closure is still accepted (#1198).
    _template = kwargs.pop("_template", None)
    if isinstance(_template, SKIRTORGrid):
        return _skirtor_grid_sed(args[0], _template, *args[1:], **kwargs)
    if _template is not None:
        return _template(*args, **kwargs)
    return _skirtor_grid_sed(args[0], _load_skirtor_default_grid(), *args[1:], **kwargs)


def skirtor_components(*args, **kwargs) -> SKIRTORComponents:
    """SKIRTOR torus with separate disk/dust (auto-loaded).

    Requires a v3 grid file. See ``create_skirtor_components_from_grid``.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom].
    agn_log_lbol : float, optional
        AGN bolometric luminosity [log10(L_sun)]. Default: 10.0.
    agn_tau_skirtor : float, optional
        V-band optical depth of torus [dimensionless]. Default: 7.0.
    agn_p_skirtor : float, optional
        Radial dust distribution power-law index [dimensionless].
        Default: 1.0.
    agn_q_skirtor : float, optional
        Temperature profile power-law index [dimensionless]. Default: 1.0.
    agn_oa_skirtor : float, optional
        Half-opening angle of the torus [degrees]. Default: 40.0.
    agn_cos_inc : float, optional
        Cosine of inclination angle [dimensionless, 0–1].
        Default: 0.5 (60°).
    frac_agn : float, optional
        AGN fraction (CIGALE-style). Preferred parameter. Default: 0.5.
    agn_torus_frac : float, optional
        **Deprecated**: use frac_agn instead. Falls back to this if
        frac_agn is not provided.
    _template : callable, optional
        Pre-loaded template function (for JIT threading). When provided,
        uses this instead of the module-level cached loader. Internal use.
    **kwargs
        Additional keyword arguments (ignored for compatibility).

    Returns
    -------
    SKIRTORComponents
        Named tuple with ``disk``, ``dust``, ``total`` arrays, each
        shape (n_wave,) with units [erg/s/Hz].

    Raises
    ------
    RuntimeError
        If no v3 grid with separate components is available.

    Notes
    -----
    **JIT-compatible**: yes — delegates to cached grid function or
    pre-loaded template (when _template is threaded).
    """
    # Allow the template to be threaded as a JIT runtime input
    _template = kwargs.pop("_template", None)

    # Handle deprecated agn_torus_frac → frac_agn migration
    if "frac_agn" not in kwargs and "agn_torus_frac" in kwargs:
        kwargs["frac_agn"] = kwargs.pop("agn_torus_frac")

    if _template is not None:
        fn = _template
    else:
        fn = _load_skirtor_components()
    if fn is None:
        raise RuntimeError(
            "Separate SKIRTOR components require a v3 grid file. "
            "Run: python scripts/download_skirtor_templates.py --input-dir <raw-files>"
        )
    return fn(*args, **kwargs)


# Deprecated: "_analytic" was a misnomer — SKIRTOR is a template-grid
# interpolation, not a closed-form model. Use skirtor_sed. Removed in v1.0.
skirtor_analytic = deprecated_alias(
    skirtor_sed, old_name="skirtor_analytic", new_name="skirtor_sed"
)
