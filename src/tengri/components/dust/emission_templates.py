# SPDX-License-Identifier: BSD-3-Clause
"""Template loaders for dust IR emission models.

This module holds template-loading functions and file-discovery helpers for
dust IR emission models (DL07, DL14, Dale, Astrodust, BOSA, THEMIS).

These functions are called at SEDModel init time to load pre-computed templates
from HDF5/NPZ files. They are NOT on the hot JAX path and do not participate
in differentiation or JIT compilation.

Template-based models:

- ``load_draine_li_templates`` / ``create_dl07_from_grid``: Draine & Li 2007
- ``load_dl14_templates`` / ``create_dl14_from_grid``: Draine & Li 2014 update
- ``load_astrodust_templates`` / ``create_astrodust_from_grid``: Astrodust+PAH
- ``load_bosa_templates`` / ``create_bosa_from_grid``: BOSA (Boquien & Salim 2021)
- ``load_themis_templates`` / ``create_themis_from_grid``: THEMIS (Jones et al. 2017)

Lazy loaders (auto-load templates on first call):

- ``_make_lazy_loader``: factory for lazy-loading wrappers
- ``_find_dl07_templates``, ``_dl07_lazy_wrapper``, ``_dl14_lazy_wrapper``

Re-exported here from emission.py:

- Import from ``tengri.components.dust.emission`` (re-exports from this module)

"""

import functools
from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp

from tengri._data_setup import find_data_str
from tengri.utils.grid_interp import loglog_integral, resample_template
from tengri.utils.physics_constants import (
    AA_TO_CM as _AA_TO_CM,
    C_CGS as _C_CGS,
)

# Upper bound of the U^-2 power-law radiation-field distribution used to
# generate the DL07 power-law (PDR) templates (``scripts/convert_dl07_templates.py``
# writes ``umax_powerlaw = 1e6``). Used to restore the PDR component's
# relative luminosity via the DL07 Eq. 33 factor (see ``dl07_tabulated``).
_DL07_UMAX_POWERLAW = 1.0e6
# DL14 (Draine et al. 2014) extends the power-law upper bound to U_max = 1e7.
_DL14_UMAX_POWERLAW = 1.0e7


def _pdr_luminosity_weight(umin, umax, alpha):
    r"""Relative luminosity of the power-law (PDR) vs single-U dust component.

    For a dust-mass distribution ``dM/dU \propto U^{-alpha}`` over
    ``[U_min, U_max]``, equilibrium dust emits a luminosity ``\propto U``, so
    the power-law (PDR) component radiates ``R = <U>_pl / U_min`` times more
    per unit dust mass than the diffuse (``U = U_min``) component
    (Draine & Li 2007, Eq. 33; Draine et al. 2014). Multiplying the power-law
    template by ``R`` converts the dust-*mass* fraction ``gamma`` into the
    correct *luminosity* weighting; without it the warm PDR emission is
    under-represented (~14x at U_min=1) and the IR SED comes out spuriously
    cold.

    Closed form (``alpha != 1, 2``), with ``x = U_max/U_min``::

        R = (1 - alpha) / (2 - alpha) * (x ^ {2 - alpha} - 1) / (x ^ {1 - alpha} - 1)

    and the integrable-pole limits ``R = (x-1)/ln x`` at ``alpha=1`` and
    ``R = x ln x / (x-1)`` at ``alpha=2``.

    Notes
    -----
    **JIT/grad-safe**: the general branch evaluates a pole-shifted ``alpha`` so
    it stays finite, and ``jnp.where`` selects the exact limit forms at
    ``alpha = 1, 2``: no NaN leaks through the ``where`` VJP.
    """
    x = umax / umin
    lnx = jnp.log(x)
    eps = 1e-3
    near1 = jnp.abs(alpha - 1.0) < eps
    near2 = jnp.abs(alpha - 2.0) < eps
    # Shift alpha off the integrable poles in the general branch so it never
    # evaluates 0/0 (which would poison the gradient even when unselected).
    a_gen = jnp.where(near1, 1.0 + eps, jnp.where(near2, 2.0 + eps, alpha))
    a1 = 1.0 - a_gen
    a2 = 2.0 - a_gen
    general = (a1 / a2) * (x**a2 - 1.0) / (x**a1 - 1.0)
    return jnp.where(
        near1,
        (x - 1.0) / lnx,
        jnp.where(near2, x * lnx / (x - 1.0), general),
    )


# Import DUST_EMISSION_MODELS from emission.py after module initialization
# This will be populated by lazy loaders
DUST_EMISSION_MODELS: dict[str, Callable] = {}

# ── All template loading functions extracted from emission.py ─────


# Template-based DL07, Dale, and DL14 models
# Template-based DL07: create from grid file


def create_dl07_from_grid(grid_path: str | dict) -> Callable:
    r"""Create a DL07 emission model function backed by tabulated templates.

    Loads the HDF5 grid once and returns a function matching the emission
    model registry interface. Use this instead of the analytic approximation
    for production work.

    Parameters
    ----------
    grid_path : str
        Path to ``dl07_templates.h5`` (from ``scripts/convert_dl07_templates.py``).

    Returns
    -------
    Callable
        Model function with signature
        ``(wavelength_aa, L_absorbed, **params) -> L_nu``.

    Notes
    -----
    **JIT-compatible**: yes, all operations inside the returned function are ``jnp`` primitives.

    Example
    -------
    >>> dl07 = create_dl07_from_grid("data/dl07_templates.h5")
    >>> DUST_EMISSION_MODELS["dl07_tabulated"] = dl07  # optional: register
    >>> sed_ir = dl07(wavelength, L_absorbed, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5)
    """
    # A dict means the caller already loaded the grid: typically the component,
    # so the arrays can be threaded into jit rather than baked (#1649). Building
    # this closure over traced arrays is fine; only capture of *concrete* arrays
    # freezes into the graph.
    templates = grid_path if isinstance(grid_path, dict) else load_draine_li_templates(grid_path)

    # Pre-extract arrays for the closure
    single_u = templates["single_u"]  # (n_qpah, n_umin, n_wave)
    powerlaw = templates["powerlaw"]  # (n_qpah, n_umin, n_wave)
    tmpl_wave = templates["wavelength"]
    umin_grid = templates["umin_grid"]
    qpah_grid = templates["qpah_grid"]

    def dl07_tabulated(
        wavelength_aa: jnp.ndarray,
        L_absorbed: float,
        dust_umin: float = 1.0,
        dust_gamma_dl: float = 0.01,
        dust_qpah: float = 2.5,
        **_kwargs,
    ) -> jnp.ndarray:
        """DL07 emission from tabulated templates (Draine & Li 2007).

        j_nu = (1-gamma) * single_U(q_PAH, U_min)
             + gamma * R * powerlaw(q_PAH, U_min)

        where ``R = U_max ln(U_max/U_min) / (U_max - U_min)`` is the DL07
        Eq. 33 relative luminosity of the power-law (PDR) component (alpha=2,
        U_max=1e6). The single-U and power-law templates are each
        shape-normalized to unit wavelength integral; ``R`` restores the PDR
        component's higher luminosity per unit dust mass, converting the
        mass-fraction ``gamma`` into the correct luminosity weighting.

        Templates are in L_lambda convention (normalized to integrate to
        1 over wavelength).  This function converts to L_nu (Lsun/Hz)
        and scales by L_absorbed to enforce energy balance.

        Parameters
        ----------
        wavelength_aa : array_like, shape (n_wave,)
            Rest-frame wavelength grid [Å].
        L_absorbed : float
            Total absorbed luminosity [Lsun].
        dust_umin : float
            Minimum radiation field intensity [dimensionless]. Default: 1.0.
        dust_gamma_dl : float
            Mixing fraction for power-law component [dimensionless]. Default: 0.01.
        dust_qpah : float
            PAH mass fraction [dimensionless]. Default: 2.5.
        **_kwargs
            Extra keyword arguments (ignored).

        Returns
        -------
        ndarray, shape (n_wave,)
            Dust emission L_ν [Lsun/Hz].

        Notes
        -----
        **JIT-compatible**: yes, all operations are ``jnp`` primitives.

        **Gradient-safe**: yes, differentiable everywhere.
        """
        dust_umin_c = jnp.clip(dust_umin, umin_grid[0], umin_grid[-1])
        dust_qpah_c = jnp.clip(dust_qpah, qpah_grid[0], qpah_grid[-1])

        # Bilinear interpolation indices
        i_u = jnp.clip(jnp.searchsorted(umin_grid, dust_umin_c) - 1, 0, len(umin_grid) - 2)
        i_q = jnp.clip(jnp.searchsorted(qpah_grid, dust_qpah_c) - 1, 0, len(qpah_grid) - 2)

        fu = (dust_umin_c - umin_grid[i_u]) / (umin_grid[i_u + 1] - umin_grid[i_u])
        fq = (dust_qpah_c - qpah_grid[i_q]) / (qpah_grid[i_q + 1] - qpah_grid[i_q])

        def _bilinear(grid):
            """Perform 2D linear interpolation over qpah and gamma axes."""
            return (
                (1.0 - fq) * (1.0 - fu) * grid[i_q, i_u]
                + (1.0 - fq) * fu * grid[i_q, i_u + 1]
                + fq * (1.0 - fu) * grid[i_q + 1, i_u]
                + fq * fu * grid[i_q + 1, i_u + 1]
            )

        # Mix single-U (diffuse) and power-law (PDR) components.
        #
        # ``dust_gamma_dl`` is the *dust-mass* fraction in the power-law-heated
        # PDR, but the templates are each shape-normalized (unit integral), which
        # discards the PDR component's higher luminosity per unit mass. PDR dust
        # is heated by U from U_min to U_max, so it emits a factor
        #   R = <U>_pl / U_min = U_max * ln(U_max/U_min) / (U_max - U_min)
        # more than the diffuse component (Draine & Li 2007, Eq. 33, alpha=2).
        # Restoring R converts the mass fraction into the correct luminosity
        # weighting; without it the PDR (warm) emission is under-represented by
        # ~14x at U_min=1 and the IR SED comes out spuriously cold.
        r_power = _pdr_luminosity_weight(dust_umin_c, _DL07_UMAX_POWERLAW, 2.0)
        template = (1.0 - dust_gamma_dl) * _bilinear(single_u) + (
            dust_gamma_dl * r_power
        ) * _bilinear(powerlaw)

        # Interpolate template onto target wavelength grid
        # Template is in L_lambda space (integral over wavelength = 1)
        sed_llam = resample_template(wavelength_aa, tmpl_wave, template, left=0.0, right=0.0)

        # Convert L_lambda -> L_nu: L_nu = L_lambda * lambda^2 / c
        wavelength_cm = wavelength_aa * _AA_TO_CM
        nu = _C_CGS / wavelength_cm
        sed_lnu = sed_llam * (wavelength_cm**2) / _C_CGS

        # Renormalize so that integral(L_nu, d_nu) = L_absorbed
        # nu is descending (wavelength ascending), so negate
        integral = -jnp.trapezoid(sed_lnu, nu)
        norm = jnp.where(integral > 0.0, L_absorbed / integral, 0.0)

        return norm * sed_lnu

    return dl07_tabulated


def load_draine_li_templates(filepath: str) -> dict:
    r"""Load DL07 template grid from HDF5 or NPZ.

    Supports two formats:

    - HDF5 with keys: wavelength, umin_grid, qpah_grid, single_u, powerlaw
    - NPZ with keys: wavelength, umin_grid, qpah_grid,
      templates_umin_only, templates_umin_umax

    The templates must be pre-normalized so that each template integrates
    to 1 over wavelength (L_lambda convention). The model function handles
    the L_absorbed scaling.

    Parameters
    ----------
    filepath : str
        Path to template file (.h5).

    Returns
    -------
    dict with keys: wavelength, umin_grid, qpah_grid, single_u, powerlaw
        All arrays are JAX arrays. single_u and powerlaw have shape
        (n_qpah, n_umin, n_wave).

    Notes
    -----
    **JIT-compatible**: no, file I/O operations not supported in JIT.
    Call at factory/init time before JIT compilation.
    """
    import numpy as np

    if filepath.endswith(".npz"):
        data = np.load(filepath)
        wavs = data["wavelength"]
        single_u = data["templates_umin_only"]  # (n_qpah, n_umin, n_wave)
        powerlaw = data["templates_umin_umax"]

        # Normalize each template to integrate to 1 over wavelength
        for i in range(single_u.shape[0]):
            for j in range(single_u.shape[1]):
                norm = np.trapezoid(single_u[i, j], wavs)
                if norm > 0:
                    single_u[i, j] /= norm
                norm = np.trapezoid(powerlaw[i, j], wavs)
                if norm > 0:
                    powerlaw[i, j] /= norm

        return {
            "wavelength": jnp.array(wavs, dtype=jnp.float64),
            "umin_grid": jnp.array(data["umin_grid"], dtype=jnp.float64),
            "qpah_grid": jnp.array(data["qpah_grid"], dtype=jnp.float64),
            "single_u": jnp.array(single_u, dtype=jnp.float64),
            "powerlaw": jnp.array(powerlaw, dtype=jnp.float64),
        }

    # HDF5 format
    import h5py as _h5py

    with _h5py.File(filepath, "r") as f:
        # v2 standardized format: /grid/qpah, /grid/umin, /spectra/single_u, /spectra/pdr
        if "grid" in f and "spectra" in f:
            wavs = np.array(f["wavelength"][:])
            # Convert micron to Angstrom if needed
            wave_unit = f["wavelength"].attrs.get("unit", "Angstrom")
            if wave_unit == "micron":
                wavs = wavs * 1e4
            single_u = np.array(f["spectra"]["single_u"][:])
            powerlaw = np.array(f["spectra"]["pdr"][:])
            # Normalize
            for i in range(single_u.shape[0]):
                for j in range(single_u.shape[1]):
                    norm = np.trapezoid(single_u[i, j], wavs)
                    if norm > 0:
                        single_u[i, j] /= norm
                    norm = np.trapezoid(powerlaw[i, j], wavs)
                    if norm > 0:
                        powerlaw[i, j] /= norm
            return {
                "wavelength": jnp.array(wavs, dtype=jnp.float64),
                "umin_grid": jnp.array(f["grid"]["umin"][:], dtype=jnp.float64),
                "qpah_grid": jnp.array(f["grid"]["qpah"][:], dtype=jnp.float64),
                "single_u": jnp.array(single_u, dtype=jnp.float64),
                "powerlaw": jnp.array(powerlaw, dtype=jnp.float64),
            }
        # Legacy flat format
        return {
            "wavelength": jnp.array(f["wavelength"][:], dtype=jnp.float64),
            "umin_grid": jnp.array(f["umin_grid"][:], dtype=jnp.float64),
            "qpah_grid": jnp.array(f["qpah_grid"][:], dtype=jnp.float64),
            "single_u": jnp.array(f["single_u"][:], dtype=jnp.float64),
            "powerlaw": jnp.array(f["powerlaw"][:], dtype=jnp.float64),
        }


# Create DL14 and register functions


def dl14_sed_from_grid(
    templates: dict,
    wavelength_aa: jnp.ndarray,
    L_absorbed: float,
    dust_umin: float = 1.0,
    dust_gamma_dl: float = 0.01,
    dust_qpah: float = 2.5,
    dust_alpha_dl14: float = 2.0,
    **_kwargs,
) -> jnp.ndarray:
    """DL14 emission from tabulated templates.

    j_nu = (1-gamma) * single_U(q_PAH, U_min)
         + gamma * R * powerlaw(q_PAH, U_min, alpha)

    where ``R = R(U_min, U_max, alpha)`` is the DL14/DL07 Eq. 33 relative
    luminosity of the power-law (PDR) component (U_max=1e7). It converts the
    dust-mass fraction ``gamma`` into the correct luminosity weighting; see
    ``_pdr_luminosity_weight``.

    Normalized to L_absorbed via energy balance.

    Parameters
    ----------
    wavelength_aa : array_like, shape (n_wave,)
        Rest-frame wavelength grid [Å].
    L_absorbed : float
        Total absorbed luminosity [Lsun].
    dust_umin : float
        Minimum radiation field intensity [dimensionless]. Default: 1.0.
    dust_gamma_dl : float
        Mixing fraction for power-law component [dimensionless]. Default: 0.01.
    dust_qpah : float
        PAH mass fraction [dimensionless]. Default: 2.5.
    dust_alpha_dl14 : float
        Radiation field power-law slope [dimensionless]. Default: 2.0.
    **_kwargs
        Extra keyword arguments (ignored).

    Returns
    -------
    ndarray, shape (n_wave,)
        Dust emission L_ν [Lsun/Hz].

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    **Gradient-safe**: yes, differentiable everywhere.
    """
    single_u = jnp.asarray(templates["single_u"])  # (n_qpah, n_umin, n_wave)
    powerlaw = jnp.asarray(templates["powerlaw"])  # (n_qpah, n_umin, n_alpha, n_wave)
    tmpl_wave = jnp.asarray(templates["wavelength"])
    umin_grid = jnp.asarray(templates["umin_grid"])
    qpah_grid = jnp.asarray(templates["qpah_grid"])
    alpha_grid = jnp.asarray(templates["alpha_grid"])

    # Clip parameters to grid bounds
    dust_umin_c = jnp.clip(dust_umin, umin_grid[0], umin_grid[-1])
    dust_qpah_c = jnp.clip(dust_qpah, qpah_grid[0], qpah_grid[-1])
    dust_alpha_c = jnp.clip(dust_alpha_dl14, alpha_grid[0], alpha_grid[-1])

    # Interpolation indices and fractions
    n_u = umin_grid.shape[0]
    n_q = qpah_grid.shape[0]
    n_a = alpha_grid.shape[0]

    i_u = jnp.clip(jnp.searchsorted(umin_grid, dust_umin_c) - 1, 0, n_u - 2)
    i_q = jnp.clip(jnp.searchsorted(qpah_grid, dust_qpah_c) - 1, 0, n_q - 2)
    i_a = jnp.clip(jnp.searchsorted(alpha_grid, dust_alpha_c) - 1, 0, n_a - 2)

    fu = (dust_umin_c - umin_grid[i_u]) / (umin_grid[i_u + 1] - umin_grid[i_u])
    fq = (dust_qpah_c - qpah_grid[i_q]) / (qpah_grid[i_q + 1] - qpah_grid[i_q])
    fa = (dust_alpha_c - alpha_grid[i_a]) / (alpha_grid[i_a + 1] - alpha_grid[i_a])

    # Bilinear interpolation for single-U (q_PAH, U_min)
    def _bilinear(grid):
        """Perform 2D linear interpolation over qpah and Umin axes."""
        return (
            (1.0 - fq) * (1.0 - fu) * grid[i_q, i_u]
            + (1.0 - fq) * fu * grid[i_q, i_u + 1]
            + fq * (1.0 - fu) * grid[i_q + 1, i_u]
            + fq * fu * grid[i_q + 1, i_u + 1]
        )

    # Trilinear interpolation for powerlaw (q_PAH, U_min, alpha)
    def _trilinear(grid):
        """Perform 3D linear interpolation over qpah, Umin, and alpha axes."""

        # Interpolate at alpha[i_a] and alpha[i_a+1] via bilinear in (q, u)
        def _bilinear_at_alpha(ia_idx):
            """Interpolate bilinearly at fixed alpha index."""
            return (
                (1.0 - fq) * (1.0 - fu) * grid[i_q, i_u, ia_idx]
                + (1.0 - fq) * fu * grid[i_q, i_u + 1, ia_idx]
                + fq * (1.0 - fu) * grid[i_q + 1, i_u, ia_idx]
                + fq * fu * grid[i_q + 1, i_u + 1, ia_idx]
            )

        lo = _bilinear_at_alpha(i_a)
        hi = _bilinear_at_alpha(i_a + 1)
        return (1.0 - fa) * lo + fa * hi

    # Mix single-U (diffuse) and power-law (PDR) components. ``gamma`` is a
    # dust-mass fraction; weight the PDR template by its DL14 relative
    # luminosity R(U_min, U_max, alpha) so it is applied as a luminosity
    # fraction (see ``_pdr_luminosity_weight``; same fix as DL07).
    r_power = _pdr_luminosity_weight(dust_umin_c, _DL14_UMAX_POWERLAW, dust_alpha_c)
    template = (1.0 - dust_gamma_dl) * _bilinear(single_u) + (
        dust_gamma_dl * r_power
    ) * _trilinear(powerlaw)

    # Interpolate template onto target wavelength grid FIRST
    sed = resample_template(wavelength_aa, tmpl_wave, template, left=0.0, right=0.0)

    # Normalize template to enforce energy balance: ∫L_nu dnu = L_absorbed.
    # Must normalize AFTER resampling to ensure the delivered SED integrates
    # to L_absorbed regardless of the native template grid spacing.
    # (Same approach as DL07 loader; DL14 stores j_nu so no L_lambda→L_nu conversion.)
    nu = _C_CGS / (wavelength_aa * _AA_TO_CM)
    eval_integral = -jnp.trapezoid(sed, nu)
    norm = jnp.where(eval_integral > 0.0, L_absorbed / eval_integral, 0.0)

    return norm * sed


def create_dl14_from_grid(grid_path: str) -> Callable:
    r"""Create a DL14 emission model function backed by tabulated templates.

    Loads the HDF5 grid once and returns a function matching the emission
    model registry interface. The key difference from DL07: the powerlaw
    template now depends on alpha too, requiring trilinear interpolation
    in (q_PAH, U_min, alpha) space instead of bilinear.

    Parameters
    ----------
    grid_path : str
        Path to ``dl14_templates.h5`` (from ``scripts/convert_dl14_templates.py``).

    Returns
    -------
    Callable
        Model function with signature
        ``(wavelength_aa, L_absorbed, **params) -> L_nu``.

    Notes
    -----
    **JIT-compatible**: yes, all operations inside the returned function are ``jnp`` primitives.

    Example
    -------
    >>> dl14 = create_dl14_from_grid("data/dl14_templates.h5")
    >>> DUST_EMISSION_MODELS["dl14_tabulated"] = dl14
    >>> sed = dl14(
    ...     wav, L_abs, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5, dust_alpha_dl14=2.0
    ... )
    """
    return functools.partial(dl14_sed_from_grid, load_dl14_templates(grid_path))


def load_dl14_templates(filepath: str) -> dict:
    r"""Load DL14 template grid from HDF5 file.

    Must contain keys: wavelength, umin_grid, qpah_grid, alpha_grid, single_u, powerlaw.

    Parameters
    ----------
    filepath : str
        Path to HDF5 template file.

    Returns
    -------
    dict
        Loaded and JAX-wrapped template arrays.

    Notes
    -----
    **JIT-compatible**: no, file I/O operations not supported in JIT.
    """
    import h5py as _h5py
    import numpy as _np_dl14

    with _h5py.File(filepath, "r") as f:
        if "grid" in f and "spectra" in f:
            # v2 standardized format: /grid/*, /spectra/*
            wavelength = _np_dl14.asarray(f["wavelength"][:], dtype=_np_dl14.float64)
            umin_grid = _np_dl14.asarray(f["grid"]["umin"][:], dtype=_np_dl14.float64)
            qpah_grid = _np_dl14.asarray(f["grid"]["qpah"][:], dtype=_np_dl14.float64)
            alpha_grid = _np_dl14.asarray(f["grid"]["alpha"][:], dtype=_np_dl14.float64)
            raw_single = _np_dl14.asarray(f["spectra"]["single_u"][:], dtype=_np_dl14.float64)
            single_u = raw_single[0]  # alpha-independent
            raw_pdr = _np_dl14.asarray(f["spectra"]["pdr"][:], dtype=_np_dl14.float64)
            powerlaw = _np_dl14.transpose(raw_pdr, (1, 2, 0, 3))
        elif "single_u" in f:
            # Legacy flat format with correct key names
            wavelength = _np_dl14.asarray(f["wavelength"][:], dtype=_np_dl14.float64)
            umin_grid = _np_dl14.asarray(f["umin_grid"][:], dtype=_np_dl14.float64)
            qpah_grid = _np_dl14.asarray(f["qpah_grid"][:], dtype=_np_dl14.float64)
            alpha_grid = _np_dl14.asarray(f["alpha_grid"][:], dtype=_np_dl14.float64)
            single_u = _np_dl14.asarray(f["single_u"][:], dtype=_np_dl14.float64)
            powerlaw = _np_dl14.asarray(f["powerlaw"][:], dtype=_np_dl14.float64)
        elif "templates_single_u" in f:
            # Older format with templates_single_u/templates_pdr keys
            wavelength = _np_dl14.asarray(f["wavelength"][:], dtype=_np_dl14.float64)
            umin_grid = _np_dl14.asarray(f["umin_grid"][:], dtype=_np_dl14.float64)
            qpah_grid = _np_dl14.asarray(f["qpah_grid"][:], dtype=_np_dl14.float64)
            alpha_grid = _np_dl14.asarray(f["alpha_grid"][:], dtype=_np_dl14.float64)
            raw_single = _np_dl14.asarray(f["templates_single_u"][:], dtype=_np_dl14.float64)
            single_u = raw_single[0]
            raw_pdr = _np_dl14.asarray(f["templates_pdr"][:], dtype=_np_dl14.float64)
            powerlaw = _np_dl14.transpose(raw_pdr, (1, 2, 0, 3))
        else:
            raise KeyError(f"DL14 HDF5 missing expected keys. Found: {list(f.keys())}")

    # Use jnp.array so dynamic JAX indexing works inside JIT.
    # Call preload_emission_model() at factory time (outside JIT) to avoid tracer leaks.
    return {
        "wavelength": jnp.array(wavelength, dtype=jnp.float64),
        "umin_grid": jnp.array(umin_grid, dtype=jnp.float64),
        "qpah_grid": jnp.array(qpah_grid, dtype=jnp.float64),
        "alpha_grid": jnp.array(alpha_grid, dtype=jnp.float64),
        "single_u": jnp.array(single_u, dtype=jnp.float64),
        "powerlaw": jnp.array(powerlaw, dtype=jnp.float64),
    }


def register_dl14_tabulated(grid_path: str, name: str = "dl14_tabulated") -> None:
    r"""Load and register the tabulated DL14 model in the emission registry.

    After calling this, the model is available via
    ``DUST_EMISSION_MODELS["dl14_tabulated"]`` and can be used as the
    ``dust_emission_model`` in ``SEDModel()``.

    Parameters
    ----------
    grid_path : str
        Path to ``dl14_templates.h5``.
    name : str
        Registry name. Default: "dl14_tabulated".

    Returns
    -------
    None
        Model is registered in ``DUST_EMISSION_MODELS`` dict as a side effect.

    Notes
    -----
    **JIT-compatible**: no, registration happens at factory time before JIT.
    """
    from . import emission

    model_fn = create_dl14_from_grid(grid_path)
    emission.DUST_EMISSION_MODELS[name] = model_fn


# Register Dale and DL07


def dale2014_emission_lnu(
    wavelength_aa: jnp.ndarray,
    L_absorbed: jnp.ndarray | float,
    *,
    wavelength_grid: jnp.ndarray,
    alpha_grid: jnp.ndarray,
    templates_sf: jnp.ndarray,
    templates_qso: jnp.ndarray | None,
    has_qso: bool,
    dust_alpha_dale: float = 2.0,
    dust_frac_agn: float = 0.0,
) -> jnp.ndarray:
    r"""Dale+2014 star-forming + AGN dust emission, mixed and scaled to L_nu.

    Single source of truth for the Dale2014 ``fracAGN`` mixing, shared by the
    ``dale2014`` engine emission model (:func:`create_dale2014_from_grid` registry
    closure, resolved via
    the ``DUST_EMISSION_MODELS`` loader cache) so the
    template-loading and mixing paths cannot diverge (#717 consolidation).

    .. math::
        L_\nu(\lambda) = \frac{L_{\rm abs}}{1 - f_{\rm AGN}}
            \left[(1 - f_{\rm AGN})\,T_{\rm SF}(\alpha)
                  + f_{\rm AGN}\,T_{\rm QSO}\right]

    :math:`L_{\rm abs}` is the dust-absorbed luminosity [erg/s], :math:`f_{\rm
    AGN}` the AGN heating fraction, :math:`T_{\rm SF}(\alpha)` the SF template
    linearly interpolated in :math:`\alpha` and unit-normalized
    (:math:`\int T\,d\nu = 1`), and :math:`T_{\rm QSO}` the AGN template carrying
    CIGALE's full-grid normalization (its integral over the dust grid is ~0.54,
    ~0.42 redward of 1 um).

    Parameters
    ----------
    wavelength_aa : ndarray, shape (n_wave,)
        Output rest-frame wavelength grid [Å].
    L_absorbed : float
        Dust-absorbed luminosity [erg/s].
    wavelength_grid : ndarray, shape (n_tmpl,)
        Template wavelength grid [Å].
    alpha_grid : ndarray, shape (n_alpha,)
        Radiation-field slope grid [dimensionless].
    templates_sf : ndarray, shape (n_alpha, n_tmpl)
        Unit-normalized SF templates [L_nu].
    templates_qso : ndarray, shape (n_tmpl,) or None
        AGN template [L_nu], CIGALE full-grid normalization. ``None`` if absent.
    has_qso : bool
        Whether ``templates_qso`` is available (static branch selector).
    dust_alpha_dale : float
        Radiation-field power-law slope [dimensionless]. Default 2.0.
    dust_frac_agn : float
        AGN heating fraction [dimensionless, [0, 0.99)]. Default 0.0.

    Returns
    -------
    ndarray, shape (n_wave,)
        Dust emission :math:`L_\nu` [erg/s/Hz].

    Notes
    -----
    **JIT/grad/vmap-compatible**: yes, ``has_qso`` is a static Python bool and
    all array ops are ``jnp`` primitives (guarded division).
    """
    dust_alpha_c = jnp.clip(dust_alpha_dale, alpha_grid[0], alpha_grid[-1])
    i_a = jnp.clip(
        jnp.searchsorted(alpha_grid, dust_alpha_c) - 1,
        0,
        alpha_grid.shape[0] - 2,
    )
    fa = (dust_alpha_c - alpha_grid[i_a]) / (alpha_grid[i_a + 1] - alpha_grid[i_a])
    template_sf = (1.0 - fa) * templates_sf[i_a] + fa * templates_sf[i_a + 1]

    if has_qso:
        f_agn = jnp.clip(dust_frac_agn, 0.0, 0.99)
        template_mixed = (1.0 - f_agn) * template_sf + f_agn * templates_qso
        # AGN is an independent power source: total IR = L_abs / (1 - f_agn).
        scale_factor = L_absorbed / jnp.maximum(1.0 - f_agn, 1e-10)
    else:
        template_mixed = template_sf
        scale_factor = L_absorbed

    # Resample onto target wavelength grid FIRST
    sed = resample_template(wavelength_aa, wavelength_grid, template_mixed, left=0.0, right=0.0)

    # Normalize to enforce energy balance on the evaluation grid.
    # Must normalize AFTER resampling to ensure the delivered SED integrates
    # correctly regardless of the native template grid spacing.
    nu = _C_CGS / (wavelength_aa * _AA_TO_CM)
    t_integral = -jnp.trapezoid(sed, nu)
    norm = jnp.where(t_integral > 0.0, 1.0 / t_integral, 0.0)

    return scale_factor * norm * sed


def load_dale2014_lnu_grid(grid_path: str) -> dict:
    r"""Load + normalize a Dale+2014 template grid into L_nu jnp arrays.

    Single source of truth for Dale2014 template loading + normalization, used by
    the ``dale2014`` engine model via :func:`create_dale2014_from_grid`
    (#717 consolidation).

    SF templates are unit-normalized (:math:`\int L_\nu\,d\nu = 1`). The QSO
    template is unit-normalized then rescaled by the fraction of a unit quasar
    that lives on the (truncated) dust grid (~0.54), preserving CIGALE's energy
    partition so the ``dust_frac_agn`` mixing matches CIGALE (#717).

    Parameters
    ----------
    grid_path : str
        Path to a ``.npz`` or ``.h5`` Dale2014 template file. Must contain
        ``wavelength_aa`` (or ``wavelength``), ``alpha_grid`` (or ``grid/alpha``),
        ``templates_sf`` (or ``spectra/templates``), and optionally
        ``templates_qso``.

    Returns
    -------
    dict
        ``wavelength_aa`` (n_wave,), ``alpha_grid`` (n_alpha,), ``templates_sf``
        (n_alpha, n_wave) [L_nu], ``templates_qso`` (n_wave,) [L_nu] or ``None``,
        ``has_qso`` (bool): arrays are jnp.

    Notes
    -----
    **JIT-compatible**: no, file I/O. Call at factory time (before JIT).
    """
    import numpy as np

    if grid_path.endswith(".npz"):
        data = np.load(grid_path)
        tmpl_wave_raw = np.array(data["wavelength_aa"])
        alpha_grid_raw = np.array(data["alpha_grid"])
        templates_raw = np.array(data["templates_sf"])
        templates_qso_raw = data.get("templates_qso", None)
        if templates_qso_raw is not None:
            templates_qso_raw = np.array(templates_qso_raw)
        already_lnu = False
    else:
        import h5py as _h5py

        with _h5py.File(grid_path, "r") as f:
            if "grid" in f:
                # v2 layout
                tmpl_wave_raw = np.array(f["wavelength"][:])
                alpha_grid_raw = np.array(f["grid/alpha"][:])
                templates_raw = np.array(f["spectra/templates"][:])
                templates_qso_raw = None
            else:
                tmpl_wave_raw = np.array(f["wavelength_aa"][:])
                alpha_grid_raw = np.array(f["alpha_grid"][:])
                templates_raw = np.array(f["templates_sf"][:])
                templates_qso_raw = None
            # Check if already in L_nu normalized form
            already_lnu = (
                f.attrs.get("spectra_unit", "") == "L_nu normalized (integral over nu = 1)"
            )
            # Optional: load pure-AGN QSO template
            if "templates_qso" in f:
                templates_qso_raw = np.array(f["templates_qso"][:])

    tmpl_wave_np = np.asarray(tmpl_wave_raw, dtype=np.float64)
    alpha_grid_np = np.asarray(alpha_grid_raw, dtype=np.float64)
    # Handle both (n_alpha, n_wave) and (n_wave, n_alpha) layouts
    if templates_raw.shape[0] == len(tmpl_wave_np) and templates_raw.shape[1] == len(
        alpha_grid_np
    ):
        templates_raw = templates_raw.T  # -> (n_alpha, n_wave)

    if already_lnu:
        # Templates are pre-normalized in L_nu convention: use directly
        templates_np = np.asarray(templates_raw, dtype=np.float64)
        if templates_qso_raw is not None:
            templates_qso_np = np.asarray(templates_qso_raw, dtype=np.float64)
        else:
            templates_qso_np = None
    else:
        # Convert from L_lambda to L_nu: L_nu = L_lambda * lambda^2 / c
        wave_cm = tmpl_wave_np * _AA_TO_CM
        nu = _C_CGS / wave_cm  # descending for ascending wavelengths

        templates_lnu = templates_raw * (wave_cm**2)[None, :] / _C_CGS

        # Unit-normalize each SF template so integral(L_nu, dnu) = 1.
        # nu is descending, so negate for positive integral.
        #
        # Integrated with ``loglog_integral``, not ``trapezoid``, because the
        # template is put on the model grid by ``resample_template``, which
        # interpolates each segment as a power law. Normalizing with a
        # different interpolant than the one used to resample leaves the
        # delivered SED integrating to something other than 1 -- energy balance
        # then drifts with the native grid spacing, which for Dale2014 reaches
        # dlog10(lambda) = 0.125.
        for i in range(templates_lnu.shape[0]):
            integral = -float(loglog_integral(jnp.asarray(nu), jnp.asarray(templates_lnu[i])))
            if integral > 0:
                templates_lnu[i] /= integral

        templates_np = np.asarray(templates_lnu, dtype=np.float64)  # (n_alpha, n_wave)

        # QSO template: unit-normalize in L_nu (matching the SF convention so
        # the L_lambda->L_nu unit scale cancels), THEN rescale to the fraction
        # of a unit quasar that lives on the dust grid.
        #
        # CIGALE normalizes ``model_quasar`` to unit total luminosity over its
        # *full* native grid (~60 nm onward). ~46% of that energy is the
        # UV/optical accretion-disc continuum below the dust-grid blue edge
        # (~360 nm), so only ~0.54 lands on the dust grid and ~0.42 redward of
        # 1 um. The h5 stores the QSO already carrying CIGALE's full-grid
        # normalization, so ``int templates_qso_raw dlambda`` over the stored
        # grid is ~0.54 (the SF templates integrate to 1). We preserve that
        # fraction: dividing by the L_nu integral fixes the unit scale,
        # multiplying by ``qso_frac`` restores the 1 : ~0.54 SF:QSO ratio that
        # matches CIGALE's energy partition. Forcing the QSO to unit (the old
        # behavior) over-weighted its IR share to ~0.78 and made the
        # ``dust_frac_agn`` mid-IR mixing grow too bright with fracAGN
        # (#717: ran 1.43x at f=0.6 vs CIGALE).
        templates_qso_np = None
        if templates_qso_raw is not None:
            templates_qso_lnu = templates_qso_raw * (wave_cm**2) / _C_CGS
            integral = -np.trapezoid(templates_qso_lnu, nu)
            qso_frac = float(np.trapezoid(templates_qso_raw, tmpl_wave_np))
            if integral > 0:
                templates_qso_lnu = templates_qso_lnu / integral * qso_frac
            templates_qso_np = np.asarray(templates_qso_lnu, dtype=np.float64)

    # Re-express each template's normalization in the interpolant it is
    # actually delivered with. The stored templates integrate to their intended
    # value under the TRAPEZOID rule (linear in nu), which is how Dale2014 and
    # CIGALE normalize them; ``resample_template`` puts them on the model grid
    # as power-law segments, whose integral differs by 1.27% on this grid
    # (dlog10(lambda) reaches 0.125). Left uncorrected the delivered SED carries
    # 0.9880 of L_absorbed -- a resolution-INDEPENDENT deficit, i.e. a
    # normalization mismatch, not an integration error.
    #
    # Scaling by trapezoid/loglog preserves each template's own normalization
    # value, so the SF templates still integrate to 1 and the QSO template still
    # integrates to CIGALE's ~0.54 dust-grid partition (#717) -- the SF:QSO
    # energy ratio is untouched.
    _nu_native = _C_CGS / (tmpl_wave_np * _AA_TO_CM)

    def _to_loglog_norm(row: np.ndarray) -> np.ndarray:
        trap = -float(np.trapezoid(row, _nu_native))
        ll = -float(loglog_integral(jnp.asarray(_nu_native), jnp.asarray(row)))
        return row * (trap / ll) if (ll > 0.0 and trap > 0.0) else row

    templates_np = np.stack([_to_loglog_norm(r) for r in templates_np])
    if templates_qso_np is not None:
        templates_qso_np = _to_loglog_norm(templates_qso_np)

    has_qso = templates_qso_np is not None
    return {
        "wavelength_aa": jnp.array(tmpl_wave_np, dtype=jnp.float64),
        "alpha_grid": jnp.array(alpha_grid_np, dtype=jnp.float64),
        "templates_sf": jnp.array(templates_np, dtype=jnp.float64),
        "templates_qso": (jnp.array(templates_qso_np, dtype=jnp.float64) if has_qso else None),
        "has_qso": has_qso,
    }


def create_dale2014_from_grid(grid_path: str) -> Callable:
    r"""Create a Dale+2014 emission model backed by tabulated templates.

    Thin registry-closure wrapper: loads + normalizes the grid via
    :func:`load_dale2014_lnu_grid` and returns a function that delegates to the
    shared :func:`dale2014_emission_lnu` mixing: identical physics and
    normalization across every consumer of the ``dale2014`` engine model
    (#717 consolidation).

    Parameters
    ----------
    grid_path : str
        Path to ``dale2014_templates_v2.h5`` or ``dale2014_templates.h5``.

    Returns
    -------
    Callable
        Model function with signature
        ``(wavelength_aa, L_absorbed, dust_alpha_dale=2.0, dust_frac_agn=0.0, **kw) -> L_nu``.

    Notes
    -----
    **JIT-compatible**: yes, the returned function is pure ``jnp``.

    Example
    -------
    >>> dale = create_dale2014_from_grid("data/dale2014_templates_v2.h5")
    >>> DUST_EMISSION_MODELS["dale2014_tabulated"] = dale
    >>> sed = dale(wav, L_abs, dust_alpha_dale=1.5, dust_frac_agn=0.1)
    """
    t = load_dale2014_lnu_grid(grid_path)

    def dale2014_tabulated(
        wavelength_aa: jnp.ndarray,
        L_absorbed: float,
        dust_alpha_dale: float = 2.0,
        dust_frac_agn: float = 0.0,
        **_kwargs,
    ) -> jnp.ndarray:
        return dale2014_emission_lnu(
            wavelength_aa,
            L_absorbed,
            wavelength_grid=t["wavelength_aa"],
            alpha_grid=t["alpha_grid"],
            templates_sf=t["templates_sf"],
            templates_qso=t["templates_qso"],
            has_qso=t["has_qso"],
            dust_alpha_dale=dust_alpha_dale,
            dust_frac_agn=dust_frac_agn,
        )

    return dale2014_tabulated


def load_dale2014_templates(filepath: str) -> dict:
    r"""Load Dale+2014 template grid from HDF5.

    Parameters
    ----------
    filepath : str
        Path to HDF5 template file.

    Returns
    -------
    dict
        Loaded and JAX-wrapped template arrays.
        Keys: wavelength_aa, alpha_grid, templates_sf.
        Optional: templates_qso (pure AGN template for fracAGN mixing).

    Notes
    -----
    **JIT-compatible**: no, file I/O operations not supported in JIT.
    """
    import h5py as _h5py
    import numpy as np

    with _h5py.File(filepath, "r") as f:
        if "grid" in f and "spectra" in f:
            wavs_aa = np.array(f["wavelength"][:]) * 1.0e4
            spectra = np.array(f["spectra"]["templates"][:])
            alpha_grid = np.array(f["grid"]["alpha"][:])
            templates_qso = None
        else:
            wavs_aa = np.array(f["wavelength_aa"][:])
            # Schema variant: ``templates_sf`` (current) vs ``spectra`` (legacy).
            spectra_key = "templates_sf" if "templates_sf" in f else "spectra"
            spectra = np.array(f[spectra_key][:])
            alpha_grid = np.array(f["alpha_grid"][:])
            # Optional: pure-AGN QSO template for fracAGN mixing
            templates_qso = None
            if "templates_qso" in f:
                templates_qso = np.array(f["templates_qso"][:])

    _sf = jnp.array(spectra, dtype=jnp.float64)
    result = {
        "wavelength_aa": jnp.array(wavs_aa, dtype=jnp.float64),
        "alpha_grid": jnp.array(alpha_grid, dtype=jnp.float64),
        "templates_sf": _sf,
        # Back-compat alias: ``precompute_dale2014_photometry`` (and other
        # legacy consumers) read the SF templates under the original "spectra"
        # key. ``templates_sf`` is the newer name introduced alongside
        # ``templates_qso`` for the fracAGN mixing.
        "spectra": _sf,
    }
    if templates_qso is not None:
        result["templates_qso"] = jnp.array(templates_qso, dtype=jnp.float64)
    return result


def create_schreiber2018_from_grid(grid_path: str | dict) -> Callable:
    r"""Create a Schreiber+2018 (S17) cold-dust model backed by tabulated templates.

    This is the tabulated counterpart of the analytic ``schreiber2016`` model:
    it shares the two-parameter ``(dust_T, dust_f_pah)`` interface but draws the
    dust-continuum and PAH shapes from the published Schreiber et al. (2018)
    library (the ``S17`` cold-dust templates packaged with AGNfitter-rX) rather
    than a modified-blackbody + Drude-profile approximation. The faithful PAH
    forest at 6--13 μm is the reason to prefer it over ``schreiber2016`` when
    reproducing AGNfitter-rX's cold-dust component.

    The grid (``data/schreiber2018_templates.h5``, built by
    ``scripts/build_schreiber2018_grid.py``) stores dust and PAH templates as
    *native* relative ``L_nu`` over a shared dust-temperature axis. At runtime
    the model linearly interpolates both in ``dust_T``, forms AGNfitter-rX's
    native mixture ``(1 - f_PAH)·dust + f_PAH·PAH``, and renormalizes the
    frequency integral to ``L_absorbed``.

    Parameters
    ----------
    grid_path : str
        Path to ``schreiber2018_templates.h5``.

    Returns
    -------
    Callable
        Model function with signature
        ``(wavelength_aa, L_absorbed, dust_T=30.0, dust_f_pah=0.05, **kw) -> L_nu``.

    Notes
    -----
    **JIT-compatible**: yes, all operations inside the returned function are
    ``jnp`` primitives.

    The temperature interpolation is node-exact piecewise-linear (matching
    AGNfitter-rX's nearest/linear template selection), not the smooth-kernel
    interpolation that smears tabulated peaks.

    References
    ----------
    .. [1] Schreiber, C., et al., 2018, A&A, 609, A30
           (https://doi.org/10.1051/0004-6361/201731506).
    .. [2] Martinez-Ramirez et al. 2024, A&A, 688, A46 (AGNfitter-rX packaging).
    """
    # A dict means the caller already loaded the grid: typically the component,
    # so the arrays can be threaded into jit rather than baked (#1649).
    grid = grid_path if isinstance(grid_path, dict) else load_schreiber2018_templates(grid_path)

    tdust = jnp.asarray(grid["tdust"])
    tmpl_wave = jnp.asarray(grid["wavelength"])
    dust_templates = jnp.asarray(grid["dust"])
    pah_templates = jnp.asarray(grid["pah"])

    def schreiber2018_tabulated(
        wavelength_aa: jnp.ndarray,
        L_absorbed: float,
        dust_T: float = 30.0,
        dust_f_pah: float = 0.05,
        **_kwargs,
    ) -> jnp.ndarray:
        """Schreiber+2018 (S17) cold-dust emission from tabulated templates.

        Parameters
        ----------
        wavelength_aa : array_like, shape (n_wave,)
            Rest-frame wavelength grid [Å].
        L_absorbed : float
            Total absorbed luminosity. The output L_nu is in the same units
            per Hz.
        dust_T : float
            Dust temperature [K]. Clipped to the grid range. Default: 30.0.
        dust_f_pah : float
            Fractional PAH contribution in [0, 1]. Default: 0.05.
        **_kwargs
            Extra keyword arguments (ignored, e.g. ``redshift``).

        Returns
        -------
        ndarray, shape (n_wave,)
            Cold-dust emission L_ν in ``[L_absorbed units] / Hz``.

        Notes
        -----
        **JIT-compatible**: yes, all operations are ``jnp`` primitives.
        """
        # Node-exact linear interpolation in dust temperature.
        t = jnp.clip(dust_T, tdust[0], tdust[-1])
        i = jnp.clip(jnp.searchsorted(tdust, t) - 1, 0, tdust.shape[0] - 2)
        ft = (t - tdust[i]) / (tdust[i + 1] - tdust[i])
        dust_T_template = (1.0 - ft) * dust_templates[i] + ft * dust_templates[i + 1]
        pah_T_template = (1.0 - ft) * pah_templates[i] + ft * pah_templates[i + 1]

        # Resample both onto the requested grid, then mix natively (AGNfitter-rX
        # mixes the unnormalized dust/PAH L_nu, so the relative amplitude: and
        # hence the physical meaning of f_PAH: is preserved).
        dust_on_grid = resample_template(
            wavelength_aa, tmpl_wave, dust_T_template, left=0.0, right=0.0
        )
        pah_on_grid = resample_template(
            wavelength_aa, tmpl_wave, pah_T_template, left=0.0, right=0.0
        )
        f_pah = jnp.clip(dust_f_pah, 0.0, 1.0)
        mixed = (1.0 - f_pah) * dust_on_grid + f_pah * pah_on_grid

        # Renormalize the frequency integral to L_absorbed (nu descending for
        # ascending wavelength, so negate for a positive integral).
        wave_cm = wavelength_aa * _AA_TO_CM
        nu = _C_CGS / wave_cm
        integral = -jnp.trapezoid(mixed, nu)
        norm = jnp.where(integral > 0.0, L_absorbed / integral, 0.0)
        return norm * mixed

    return schreiber2018_tabulated


def load_schreiber2018_templates(filepath: str) -> dict:
    r"""Load the Schreiber+2018 (S17) cold-dust template grid from HDF5.

    Parameters
    ----------
    filepath : str
        Path to ``schreiber2018_templates.h5``.

    Returns
    -------
    dict
        JAX-wrapped ``tdust`` [K], ``wavelength`` [Å], ``dust`` and ``pah``
        ``(n_T, n_wave)`` template arrays (native relative L_nu).

    Notes
    -----
    **JIT-compatible**: no, file I/O is not supported in JIT.
    """
    import h5py as _h5py
    import numpy as np

    with _h5py.File(filepath, "r") as f:
        g = f["schreiber2018"]
        return {
            "tdust": jnp.array(np.array(g["tdust"][:]), dtype=jnp.float64),
            "wavelength": jnp.array(np.array(g["wavelength"][:]), dtype=jnp.float64),
            "dust": jnp.array(np.array(g["dust"][:]), dtype=jnp.float64),
            "pah": jnp.array(np.array(g["pah"][:]), dtype=jnp.float64),
        }


def load_schreiber2016_templates(filepath: str) -> dict:
    r"""Load Schreiber+2016 template grid from HDF5.

    The template file must contain:

    - ``wavelength_aa``: wavelength grid in Angstrom (n_wave,)
    - ``tdust_grid``: dust temperature grid (n_tdust,) in Kelvin
    - ``continuum``: continuum templates (n_tdust, n_wave) in W/nm/kg
    - ``pah``: PAH templates (n_tdust, n_wave) in W/nm/kg

    Parameters
    ----------
    filepath : str
        Path to ``schreiber2016_templates.h5``.

    Returns
    -------
    dict
        Keys: wavelength_aa, tdust_grid, continuum, pah.
        All arrays are JAX arrays. wavelength_aa is in Angstrom.
        continuum and pah have shape (n_tdust, n_wave) and are
        normalized in L_nu convention (integral over nu = 1).

    Notes
    -----
    **JIT-compatible**: no, file I/O operations not supported in JIT.
    Call at factory/init time before JIT compilation.
    """
    import h5py as _h5py
    import numpy as np

    already_lnu = False

    with _h5py.File(filepath, "r") as f:
        already_lnu = f.attrs.get("spectra_unit", "") == "L_nu normalized (integral over nu = 1)"
        wavs_aa = np.array(f["wavelength_aa"][:])
        tdust_grid = np.array(f["tdust_grid"][:])
        continuum = np.array(f["continuum"][:])
        pah = np.array(f["pah"][:])

    if not already_lnu:
        # Convert to L_nu and normalize
        wave_cm = wavs_aa * _AA_TO_CM
        nu = _C_CGS / wave_cm

        for i in range(continuum.shape[0]):
            lnu = continuum[i] * (wave_cm**2) / _C_CGS
            integral = -np.trapezoid(lnu, nu)
            if integral > 0:
                continuum[i] = lnu / integral
            else:
                continuum[i] = lnu

        for i in range(pah.shape[0]):
            lnu = pah[i] * (wave_cm**2) / _C_CGS
            integral = -np.trapezoid(lnu, nu)
            if integral > 0:
                pah[i] = lnu / integral
            else:
                pah[i] = lnu

    # Use jnp.array so dynamic JAX indexing works inside JIT.
    # Call preload_emission_model() at factory time (outside JIT) to avoid tracer leaks.
    return {
        "wavelength_aa": jnp.array(wavs_aa, dtype=jnp.float64),
        "tdust_grid": jnp.array(tdust_grid, dtype=jnp.float64),
        "continuum": jnp.array(continuum, dtype=jnp.float64),
        "pah": jnp.array(pah, dtype=jnp.float64),
    }


def register_dale2014_tabulated(grid_path: str, name: str = "dale2014_tabulated") -> None:
    r"""Load and register the tabulated Dale+2014 model in the emission registry.

    After calling this, the model is available via
    ``DUST_EMISSION_MODELS["dale2014_tabulated"]`` and can be used as the
    ``dust_emission_model`` in ``SEDModel()``.

    Parameters
    ----------
    grid_path : str
        Path to ``dale2014_templates_v2.h5`` or ``dale2014_templates.h5``.
    name : str
        Registry name. Default: "dale2014_tabulated".

    Returns
    -------
    None
        Model is registered in ``DUST_EMISSION_MODELS`` dict as a side effect.

    Notes
    -----
    **JIT-compatible**: no, registration happens at factory time before JIT.
    """
    from . import emission

    model_fn = create_dale2014_from_grid(grid_path)
    emission.DUST_EMISSION_MODELS[name] = model_fn


def register_dl07_tabulated(grid_path: str, name: str = "dl07_tabulated") -> None:
    r"""Load and register the tabulated DL07 model in the emission registry.

    After calling this, the model is available via
    ``DUST_EMISSION_MODELS["dl07_tabulated"]`` and can be used as the
    ``dust_emission_model`` in ``SEDModel()``.

    Parameters
    ----------
    grid_path : str
        Path to ``dl07_templates_v2.h5`` or ``dl07_templates.h5``.
    name : str
        Registry name. Default: "dl07_tabulated".

    Returns
    -------
    None
        Model is registered in ``DUST_EMISSION_MODELS`` dict as a side effect.

    Notes
    -----
    **JIT-compatible**: no, registration happens at factory time before JIT.
    """
    from . import emission

    model_fn = create_dl07_from_grid(grid_path)
    emission.DUST_EMISSION_MODELS[name] = model_fn


# Load Astrodust templates


def load_astrodust_templates(filepath: str) -> dict:
    r"""Load Astrodust+PAH template grid from NPZ or HDF5.

    The template file must contain:

    - ``wavelength_um``: wavelength grid in microns (n_wave,)
    - ``qpah_grid``: PAH mass fractions (n_qpah,)
    - ``umin_grid``: minimum radiation field intensities (n_umin,)
    - ``spectra_single``: single-U templates (n_qpah, n_umin, n_wave)
    - ``spectra_pdr``: power-law U (PDR) templates (n_qpah, n_umin, n_wave)

    Parameters
    ----------
    filepath : str
        Path to ``astrodust_templates_v2.h5`` or ``astrodust_templates.h5``.

    Returns
    -------
    dict
        Keys: wavelength_aa, umin_grid, qpah_grid, single_u, powerlaw.
        All arrays are JAX arrays.  wavelength_aa is in Angstrom (converted
        from microns).  single_u and powerlaw have shape
        (n_qpah, n_umin, n_wave) and are normalized so each template
        integrates to 1 over frequency in L_nu convention.

    Notes
    -----
    **JIT-compatible**: no, file I/O operations not supported in JIT.
    """
    import numpy as np

    already_lnu = False

    if filepath.endswith(".npz"):
        data = np.load(filepath)
        wavs_um = np.array(data["wavelength_um"])
        single_u = np.array(data["spectra_single"])
        powerlaw = np.array(data["spectra_pdr"])
        umin_grid = np.array(data["umin_grid"])
        qpah_grid = np.array(data["qpah_grid"])
        wavs_aa = wavs_um * 1.0e4
    else:
        import h5py as _h5py

        with _h5py.File(filepath, "r") as f:
            already_lnu = (
                f.attrs.get("spectra_unit", "") == "L_nu normalized (integral over nu = 1)"
            )
            # New canonical Hensley & Draine 2023 schema (lgU axis only,
            # single fiducial size distribution / qpah).  Translate to the
            # legacy (qpah, umin) interface by treating umin = 10**lgU and
            # placing all spectra at a single qpah row matching the H&D
            # 2022 fiducial value (~3.79%).  ``dust_qpah`` then becomes a
            # no-op for the legacy registry path; users wanting real qpah-
            # axis variation should use the modern SEDComponent dispatch.
            if "L_nu_total" in f and "lgU" in f and "umin_grid" not in f:
                wave_um = np.array(f["wavelength_um"][:])
                wavs_aa = wave_um * 1.0e4
                lgU = np.array(f["lgU"][:])  # (n_lgU,)
                L_nu_total = np.array(f["L_nu_total"][:])  # (n_lgU, n_wave)
                # H&D 2022 fiducial size distribution → q_PAH ≈ 3.79%
                # (matches Draine+2021 PAHspec "standard" reference).
                qpah_fiducial = 3.79
                # Legacy interp needs >= 2 qpah points; duplicate the
                # single-fiducial spectrum so ``dust_qpah`` becomes a
                # no-op in the H&D 2023 model (real qpah variation
                # requires the per-grain cross-section dataset PEXRD0).
                qpah_grid = np.array([qpah_fiducial, qpah_fiducial + 1.0])
                umin_grid = 10.0 ** np.asarray(lgU, dtype=np.float64)
                # Per-H L_nu values are O(1e-30); the legacy registry
                # expects per-template ∫L_ν dν = 1.  Normalize each
                # lgU slice; the energy-balance rescale to L_absorbed
                # happens downstream.
                wave_cm_aa = wavs_aa * _AA_TO_CM
                nu_aa = _C_CGS / wave_cm_aa  # descending
                norms = np.zeros(L_nu_total.shape[0])
                for i in range(L_nu_total.shape[0]):
                    integral = -np.trapezoid(L_nu_total[i], nu_aa)
                    norms[i] = integral if integral > 0 else 1.0
                L_nu_normed = L_nu_total / norms[:, None]
                # Single-U component: the per-U spectrum at U = U_min, shape-
                # normalized. Power-law (PDR) component: dust mass distributed
                # as dM/dU ∝ U^-alpha from U_min to U_max (= max grid U). Each
                # mass element at field U' emits the per-U spectrum
                # L_nu_total[U'] (per H ∝ per mass), so integrate the *raw*
                # per-U spectra over the lgU grid weighted by U'^(1-alpha)
                # (dU' = U' ln10 dlgU on a uniform-lgU grid), then shape-
                # normalize. The forward applies the DL07 Eq. 33 relative-power
                # weight R. Without this the PDR was a copy of single_u and
                # ``gamma`` was a no-op (see #571).
                alpha_pdr = 2.0  # DL07-standard slope for the H&D 2023 grid
                w_pdr = umin_grid ** (1.0 - alpha_pdr)  # mass weight per lgU bin
                powerlaw_1d = np.zeros_like(L_nu_normed)
                n_u = umin_grid.shape[0]
                for iu in range(n_u):
                    pdr = (L_nu_total[iu:] * w_pdr[iu:, None]).sum(axis=0)
                    integ = -np.trapezoid(pdr, nu_aa)
                    powerlaw_1d[iu] = pdr / integ if integ > 0 else pdr
                single_u = np.broadcast_to(L_nu_normed[None, :, :], (2, *L_nu_normed.shape)).copy()
                powerlaw = np.broadcast_to(powerlaw_1d[None, :, :], (2, *powerlaw_1d.shape)).copy()
                already_lnu = True  # we normalized explicitly above
            elif "wavelength_aa" in f:
                # Standardized HDF5 (already Angstrom + L_nu normalized)
                wavs_aa = np.array(f["wavelength_aa"][:])
                single_u = np.array(f["single_u"][:])
                powerlaw = np.array(f["powerlaw"][:])
                umin_grid = np.array(f["umin_grid"][:])
                qpah_grid = np.array(f["qpah_grid"][:])
            elif "grid" in f:
                # v2 layout
                wavs_aa = np.array(f["wavelength"][:]) * 1.0e4
                single_u = np.array(f["spectra/single_u"][:])
                powerlaw = np.array(f["spectra/pdr"][:])
                umin_grid = np.array(f["grid/umin"][:])
                qpah_grid = np.array(f["grid/qpah"][:])
            else:
                wavs_aa = np.array(f["wavelength_um"][:]) * 1.0e4
                single_u = np.array(f["spectra_single"][:])
                powerlaw = np.array(f["spectra_pdr"][:])
                umin_grid = np.array(f["umin_grid"][:])
                qpah_grid = np.array(f["qpah_grid"][:])

    if not already_lnu:
        # Convert to L_nu: L_nu = L_lambda * lambda^2 / c
        wave_cm = wavs_aa * _AA_TO_CM
        nu = _C_CGS / wave_cm  # descending

        for arr in (single_u, powerlaw):
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    lnu = arr[i, j] * (wave_cm**2) / _C_CGS
                    integral = -np.trapezoid(lnu, nu)
                    if integral > 0:
                        arr[i, j] = lnu / integral
                    else:
                        arr[i, j] = lnu

    # Use jnp.array so dynamic JAX indexing works inside JIT.
    # Call preload_emission_model() at factory time (outside JIT) to avoid tracer leaks.
    return {
        "wavelength_aa": jnp.array(wavs_aa, dtype=jnp.float64),
        "umin_grid": jnp.array(umin_grid, dtype=jnp.float64),
        "qpah_grid": jnp.array(qpah_grid, dtype=jnp.float64),
        "single_u": jnp.array(single_u, dtype=jnp.float64),
        "powerlaw": jnp.array(powerlaw, dtype=jnp.float64),
    }


def _normalize_dl07_like_grid(raw: dict, q_key: str = "qpah_grid") -> dict:
    r"""Convert a raw DL07-like grid dict to the processed format.

    Raw grids have keys ``spectra_single``, ``spectra_pdr``, and
    ``wavelength_um``; the processed format uses ``single_u``,
    ``powerlaw``, and ``wavelength_aa`` (in Angstrom, L_nu-normalized).

    Parameters
    ----------
    raw : dict
        Raw template grid with wavelength_um, spectra_single, spectra_pdr,
        umin_grid, and either qpah_grid or qhac_grid.
    q_key : str
        Key for the grain composition parameter grid (``"qpah_grid"`` for
        Astrodust/DL07, ``"qhac_grid"`` for THEMIS).

    Returns
    -------
    dict
        Processed grid with wavelength_aa, single_u, powerlaw, umin_grid,
        and the composition grid key.

    Notes
    -----
    **JIT-compatible**: no, this is a preprocessing step before JIT.
    """
    import numpy as np

    wavs_um = np.asarray(raw["wavelength_um"])
    wavs_aa = wavs_um * 1.0e4

    single_u = np.array(raw["spectra_single"])
    powerlaw = np.array(raw["spectra_pdr"])

    wave_cm = wavs_aa * _AA_TO_CM
    nu = _C_CGS / wave_cm

    for arr in (single_u, powerlaw):
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                lnu = arr[i, j] * (wave_cm**2) / _C_CGS
                integral = -np.trapezoid(lnu, nu)
                if integral > 0:
                    arr[i, j] = lnu / integral
                else:
                    arr[i, j] = lnu

    # Use jnp.array so dynamic JAX indexing works inside JIT.
    result = {
        "wavelength_aa": jnp.array(wavs_aa, dtype=jnp.float64),
        "umin_grid": jnp.array(np.asarray(raw["umin_grid"]), dtype=jnp.float64),
        "single_u": jnp.array(single_u, dtype=jnp.float64),
        "powerlaw": jnp.array(powerlaw, dtype=jnp.float64),
    }
    result[q_key] = jnp.array(np.asarray(raw[q_key]), dtype=jnp.float64)
    return result


# DH02_CE01 (Dale & Helou 2002 + Chary & Elbaz 2001) cold-dust model


def create_dh02_ce01_from_grid(grid_path: str | dict) -> Callable:
    r"""Create a DH02_CE01 cold-dust emission model from tabulated templates.

    Evaluates the DH02_CE01 template library published with AGNfitter-rX
    (Dale & Helou 2002 + Chary & Elbaz 2001). The single-axis grid is
    parameterized by infrared luminosity :math:`L_{\rm IR}`. Templates
    are linearly interpolated in :math:`\log_{10}(L_{\rm IR} / L_\odot)`,
    and the output SED is normalized via energy balance to match the
    absorbed UV/optical luminosity.

    Parameters
    ----------
    grid_path : str or dict
        Path to ``dh02_ce01_grid.h5`` (built by
        ``scripts/build_dh02_ce01_grid.py``), or an already-loaded grid dict
        from :func:`load_dh02_ce01_lnu_grid`. The dict form is what lets the
        component thread the library in as a traced argument instead of
        capturing it as a compile-time constant (#1649); passing a path here
        inside a trace bakes 1.32 MB into the graph. Mirrors
        :func:`create_bosa_from_grid`.

    Returns
    -------
    Callable
        Model function with signature
        ``(wavelength_aa, L_absorbed, dust_log_lir=10.0, **kw) -> L_nu``.

    Notes
    -----
    **JIT-compatible**: yes, all operations inside the returned function are
    ``jnp`` primitives.

    References
    ----------
    .. [1] Dale, D. A. & Helou, G. 2002, ApJ, 576, 159.
           (https://doi.org/10.1086/341632)
    .. [2] Chary, R. & Elbaz, D. 2001, ApJ, 556, 562.
           (https://doi.org/10.1086/321609)
    .. [3] Martínez-Ramírez, L. N. et al., 2024, A&A, 688, A46.
           (https://doi.org/10.1051/0004-6361/202449329), AGNfitter-rX packaging.
    """
    templates = grid_path if isinstance(grid_path, dict) else load_dh02_ce01_lnu_grid(grid_path)

    irlum_axis = templates["irlum_axis"]
    wavelength_grid = templates["wavelength"]
    template_grid = templates["templates"]

    def dh02_ce01_tabulated(
        wavelength_aa: jnp.ndarray,
        L_absorbed: float,
        dust_log_lir: float = 10.0,
        **_kwargs,
    ) -> jnp.ndarray:
        """DH02_CE01 dust emission from tabulated templates (Dale & Helou 2002).

        Parameters
        ----------
        wavelength_aa : array_like, shape (n_wave,)
            Rest-frame wavelength grid [Å].
        L_absorbed : float
            Total absorbed luminosity [Lsun].
        dust_log_lir : float
            Log₁₀ of the infrared luminosity [log₁₀(L_IR/L_sun)].
            Clipped to the grid range [8.3, 14.3]. Default: 10.0.
        **_kwargs
            Extra keyword arguments (ignored).

        Returns
        -------
        ndarray, shape (n_wave,)
            Dust emission L_ν [Lsun/Hz].

        Notes
        -----
        **JIT-compatible**: yes, all operations are ``jnp`` primitives.

        **Gradient-safe**: yes, differentiable everywhere via linear interpolation.
        """
        # Clip input to grid bounds
        lir_c = jnp.clip(dust_log_lir, irlum_axis[0], irlum_axis[-1])

        # Linear interpolation index and fraction
        i = jnp.clip(
            jnp.searchsorted(irlum_axis, lir_c) - 1,
            0,
            irlum_axis.shape[0] - 2,
        )
        f = (lir_c - irlum_axis[i]) / (irlum_axis[i + 1] - irlum_axis[i])

        # Linearly interpolate template in log(L_IR) space
        template_interp = (1.0 - f) * template_grid[i] + f * template_grid[i + 1]

        # Resample to output wavelength grid
        sed = resample_template(
            wavelength_aa, wavelength_grid, template_interp, left=0.0, right=0.0
        )

        # Normalize via frequency integral (energy balance)
        wave_cm = wavelength_aa * _AA_TO_CM
        nu = _C_CGS / wave_cm
        integral = -jnp.trapezoid(sed, nu)
        norm = jnp.where(integral > 0.0, L_absorbed / integral, 0.0)

        return norm * sed

    return dh02_ce01_tabulated


def load_dh02_ce01_lnu_grid(grid_path: str) -> dict:
    r"""Load DH02_CE01 template grid from HDF5 into normalized L_nu jnp arrays.

    The templates are pre-normalized so that each integrates to 1 over
    frequency (:math:`\int L_\nu\,d\nu = 1`).

    Parameters
    ----------
    grid_path : str
        Path to ``dh02_ce01_grid.h5``.

    Returns
    -------
    dict
        Keys: ``wavelength`` (n_wave,) [Å], ``irlum_axis`` (n_irlum,)
        [log₁₀(L_IR/L_sun)], ``templates`` (n_irlum, n_wave) [L_nu normalized].
        All arrays are jnp.

    Notes
    -----
    **JIT-compatible**: no, file I/O operations not supported in JIT.
    Call at factory time (before JIT compilation).
    """
    import h5py as _h5py
    import numpy as _np_dh02

    with _h5py.File(grid_path, "r") as f:
        grp = f["dh02_ce01"]
        wavelength = _np_dh02.asarray(grp["wavelength"][:], dtype=_np_dh02.float64)
        irlum_axis = _np_dh02.asarray(grp["irlum_axis"][:], dtype=_np_dh02.float64)
        templates = _np_dh02.asarray(grp["template"][:], dtype=_np_dh02.float64)

    return {
        "wavelength": jnp.array(wavelength, dtype=jnp.float64),
        "irlum_axis": jnp.array(irlum_axis, dtype=jnp.float64),
        "templates": jnp.array(templates, dtype=jnp.float64),
    }


# Normalize BOSA grid helper
def _normalize_bosa_grid(raw: dict) -> dict:
    r"""Convert a raw BOSA grid dict to the processed format.

    Raw grids have ``wavelength_um`` and ``spectra``; the processed
    format uses ``wavelength_aa`` (Angstrom) with L_nu-normalized spectra.

    Parameters
    ----------
    raw : dict
        Raw BOSA grid with wavelength_um, log_ltir_grid, log_ssfr_grid,
        and spectra.

    Returns
    -------
    dict
        Processed grid with wavelength_aa and L_nu-normalized spectra.

    Notes
    -----
    **JIT-compatible**: no, this is a preprocessing step before JIT.
    """
    import numpy as np

    wavs_um = np.asarray(raw["wavelength_um"])
    wavs_aa = wavs_um * 1.0e4

    spectra = np.array(raw["spectra"])

    wave_cm = wavs_aa * _AA_TO_CM
    nu = _C_CGS / wave_cm

    for i in range(spectra.shape[0]):
        for j in range(spectra.shape[1]):
            lnu = spectra[i, j] * (wave_cm**2) / _C_CGS
            integral = -np.trapezoid(lnu, nu)
            if integral > 0:
                spectra[i, j] = lnu / integral
            else:
                spectra[i, j] = lnu

    # Use jnp.array so dynamic JAX indexing works inside JIT.
    return {
        "wavelength_aa": jnp.array(wavs_aa, dtype=jnp.float64),
        "log_ltir_grid": jnp.array(np.asarray(raw["log_ltir_grid"]), dtype=jnp.float64),
        "log_ssfr_grid": jnp.array(np.asarray(raw["log_ssfr_grid"]), dtype=jnp.float64),
        "spectra": jnp.array(spectra, dtype=jnp.float64),
    }


# Create Astrodust and register
def create_astrodust_from_grid(
    template_data: dict | str,
) -> Callable:
    r"""Create Astrodust+PAH emission function from pre-loaded template grid.

    The mixing formula is identical to DL07::

        j_nu = (1 - gamma) * j_single(qPAH, Umin)
             + gamma * j_PDR(qPAH, Umin)

    Parameters
    ----------
    template_data : dict or str
        Either a dict (from ``load_astrodust_templates``) or a file path.
        If a string, ``load_astrodust_templates`` is called automatically.

    Returns
    -------
    Callable
        Model function with signature
        ``(wavelength_aa, L_absorbed, **params) -> L_nu``.

    Notes
    -----
    **JIT-compatible**: yes, all operations inside the returned function are ``jnp`` primitives.

    References
    ----------
    Hensley, B. S. & Draine, B. T. 2023, ApJ, 948, 55.
    """
    if isinstance(template_data, str):
        template_data = load_astrodust_templates(template_data)

    # Accept both raw grid format (spectra_single/spectra_pdr/wavelength_um)
    # and processed format (single_u/powerlaw/wavelength_aa) from load_*
    if "spectra_single" in template_data and "single_u" not in template_data:
        template_data = _normalize_dl07_like_grid(template_data, q_key="qpah_grid")

    single_u = template_data["single_u"]  # (n_qpah, n_umin, n_wave)
    powerlaw = template_data["powerlaw"]  # (n_qpah, n_umin, n_wave)
    tmpl_wave = template_data["wavelength_aa"]
    umin_grid = template_data["umin_grid"]
    qpah_grid = template_data["qpah_grid"]

    def astrodust_emission(
        wavelength_aa: jnp.ndarray,
        L_absorbed: float,
        dust_umin: float = 1.0,
        dust_gamma_dl: float = 0.01,
        dust_qpah: float = 3.0,
        redshift: float = 0.0,
        **_kwargs,
    ) -> jnp.ndarray:
        """Astrodust+PAH emission from tabulated templates (Hensley & Draine 2023).

        j_nu = (1-gamma) * single_U(qPAH, Umin) + gamma * PDR(qPAH, Umin)

        Templates are pre-normalized in L_nu convention.  The function
        interpolates bilinearly in (qPAH, Umin) space, mixes via gamma,
        and scales by L_absorbed to enforce energy balance.

        Parameters
        ----------
        wavelength_aa : array, shape (n_wave,)
            Wavelength grid in Angstrom (sorted ascending).
        L_absorbed : float
            Total absorbed luminosity in Lsun.
        dust_umin : float
            Minimum radiation field intensity (Mathis ISRF units).
        dust_gamma_dl : float
            Fraction of dust mass in PDR (high-U) component.
        dust_qpah : float
            PAH mass fraction (%).
        redshift : float
            Source redshift (for CMB contrast correction).

        Returns
        -------
        array, shape (n_wave,)
            Dust emission L_nu in Lsun/Hz.

        Notes
        -----
        **JIT-compatible**: yes, all operations are ``jnp`` primitives.

        **Gradient-safe**: yes, differentiable everywhere.
        """
        dust_umin_c = jnp.clip(dust_umin, umin_grid[0], umin_grid[-1])
        dust_qpah_c = jnp.clip(dust_qpah, qpah_grid[0], qpah_grid[-1])

        # Bilinear interpolation indices
        i_u = jnp.clip(
            jnp.searchsorted(umin_grid, dust_umin_c) - 1,
            0,
            len(umin_grid) - 2,
        )
        i_q = jnp.clip(
            jnp.searchsorted(qpah_grid, dust_qpah_c) - 1,
            0,
            len(qpah_grid) - 2,
        )

        fu = (dust_umin_c - umin_grid[i_u]) / (umin_grid[i_u + 1] - umin_grid[i_u])
        fq = (dust_qpah_c - qpah_grid[i_q]) / (qpah_grid[i_q + 1] - qpah_grid[i_q])

        def _bilinear(grid: jnp.ndarray) -> jnp.ndarray:
            """Perform 2D linear interpolation over qpah and Umin axes."""
            return (
                (1.0 - fq) * (1.0 - fu) * grid[i_q, i_u]
                + (1.0 - fq) * fu * grid[i_q, i_u + 1]
                + fq * (1.0 - fu) * grid[i_q + 1, i_u]
                + fq * fu * grid[i_q + 1, i_u + 1]
            )

        # Mix single-U (diffuse) and power-law (PDR) components. ``gamma`` is a
        # dust-mass fraction; weight the PDR template (built in the loader by
        # integrating the H&D per-U spectra over dM/dU ∝ U^-2) by its DL07
        # Eq. 33 relative luminosity R so gamma acts as a luminosity fraction
        # (U_max = max grid U; same fix as DL07/DL14; see #571).
        r_power = _pdr_luminosity_weight(dust_umin_c, umin_grid[-1], 2.0)
        template = (1.0 - dust_gamma_dl) * _bilinear(single_u) + (
            dust_gamma_dl * r_power
        ) * _bilinear(powerlaw)

        # Interpolate onto target wavelength grid FIRST
        sed = resample_template(wavelength_aa, tmpl_wave, template, left=0.0, right=0.0)

        # Energy balance: renormalize to unit frequency integral on the
        # evaluation grid AFTER resampling to ensure the delivered SED
        # integrates to L_absorbed regardless of the native template grid spacing.
        # Use the same log-wavelength integration as the component-level normalization
        # for consistency across different astrodust calling paths.
        nu_lnu = (_C_CGS / (wavelength_aa * _AA_TO_CM)) * sed
        t_integral = jnp.trapezoid(nu_lnu, jnp.log(wavelength_aa))
        norm = jnp.where(t_integral > 0.0, L_absorbed / t_integral, 0.0)

        # No CMB contrast factor here; see the note in ``create_themis_from_grid``.
        # It is an *observational* suppression, so applying it to the emitted SED
        # breaks the energy-balance invariant (int L_nu dnu == L_absorbed) that
        # this component exists to satisfy. ``redshift`` is kept in the signature
        # for call-site compatibility.
        return norm * sed

    return astrodust_emission


def register_astrodust_tabulated(grid_path: str, name: str = "astrodust_tabulated") -> None:
    r"""Load and register the tabulated Astrodust model.

    Parameters
    ----------
    grid_path : str
        Path to ``astrodust_templates_v2.h5`` or ``astrodust_templates.h5``.
    name : str
        Registry name. Default: "astrodust_tabulated".

    Returns
    -------
    None
        Model is registered in ``DUST_EMISSION_MODELS`` dict as a side effect.

    Notes
    -----
    **JIT-compatible**: no, registration happens at factory time before JIT.
    """
    from . import emission

    model_fn = create_astrodust_from_grid(grid_path)
    emission.DUST_EMISSION_MODELS[name] = model_fn


# Load BOSA templates


def load_bosa_templates(filepath: str) -> dict:
    r"""Load BOSA template grid from HDF5.

    The template file must contain:

    - ``wavelength_um``: wavelength grid in microns (n_wave,)
    - ``log_ltir_grid``: log10(L_TIR/Lsun) grid (n_ltir,)
    - ``log_ssfr_grid``: log10(sSFR/yr^-1) grid (n_ssfr,)
    - ``spectra``: normalized SED templates (n_ltir, n_ssfr, n_wave)

    Parameters
    ----------
    filepath : str
        Path to ``bosa_templates_v2.h5`` or ``bosa_templates.h5``.

    Returns
    -------
    dict
        Keys: wavelength_aa, log_ltir_grid, log_ssfr_grid, spectra.
        All arrays are JAX arrays.  wavelength_aa is in Angstrom.
        spectra have shape (n_ltir, n_ssfr, n_wave) and are normalized
        so each template integrates to 1 over frequency in L_nu convention.

    Notes
    -----
    **JIT-compatible**: no, file I/O operations not supported in JIT.
    """
    import numpy as np

    already_lnu = False

    if filepath.endswith(".npz"):
        data = np.load(filepath)
        wavs_um = np.array(data["wavelength_um"])
        spectra = np.array(data["spectra"])
        log_ltir_grid = np.array(data["log_ltir_grid"])
        log_ssfr_grid = np.array(data["log_ssfr_grid"])
        wavs_aa = wavs_um * 1.0e4
    else:
        import h5py as _h5py

        with _h5py.File(filepath, "r") as f:
            already_lnu = (
                f.attrs.get("spectra_unit", "") == "L_nu normalized (integral over nu = 1)"
            )
            if "wavelength_aa" in f:
                # Standardized HDF5
                wavs_aa = np.array(f["wavelength_aa"][:])
                spectra = np.array(f["spectra"][:])
                log_ltir_grid = np.array(f["log_ltir_grid"][:])
                log_ssfr_grid = np.array(f["log_ssfr_grid"][:])
            elif "grid" in f:
                wavs_aa = np.array(f["wavelength"][:]) * 1.0e4
                spectra = np.array(f["spectra"]["templates"][:])
                log_ltir_grid = np.array(f["grid"]["log_ltir"][:])
                log_ssfr_grid = np.array(f["grid"]["log_ssfr"][:])
            else:
                wavs_aa = np.array(f["wavelength_um"][:]) * 1.0e4
                spectra = np.array(f["spectra"][:])
                log_ltir_grid = np.array(f["log_ltir_grid"][:])
                log_ssfr_grid = np.array(f["log_ssfr_grid"][:])

    if not already_lnu:
        # Convert to L_nu and normalize
        wave_cm = wavs_aa * _AA_TO_CM
        nu = _C_CGS / wave_cm

        for i in range(spectra.shape[0]):
            for j in range(spectra.shape[1]):
                lnu = spectra[i, j] * (wave_cm**2) / _C_CGS
                integral = -np.trapezoid(lnu, nu)
                if integral > 0:
                    spectra[i, j] = lnu / integral
                else:
                    spectra[i, j] = lnu

    # Use jnp.array so dynamic JAX indexing works inside JIT.
    return {
        "wavelength_aa": jnp.array(wavs_aa, dtype=jnp.float64),
        "log_ltir_grid": jnp.array(log_ltir_grid, dtype=jnp.float64),
        "log_ssfr_grid": jnp.array(log_ssfr_grid, dtype=jnp.float64),
        "spectra": jnp.array(spectra, dtype=jnp.float64),
    }


# Create BOSA and register
def create_bosa_from_grid(template_data: dict | str) -> Callable:
    r"""Create BOSA emission function from pre-loaded template grid.

    The BOSA model (Boquien & Salim 2021) parameterizes dust emission
    templates by (L_TIR, sSFR) instead of radiation field parameters.
    This provides a direct link between star formation activity and
    dust temperature.

    For fitting, L_TIR is derived from L_absorbed (energy balance),
    so the free parameter is just ``dust_log_ssfr``.  The template
    is selected by interpolating in (log L_TIR, log sSFR) space.

    Parameters
    ----------
    template_data : dict or str
        Either a dict (from ``load_bosa_templates``) or a file path.

    Returns
    -------
    Callable
        Model function with signature
        ``(wavelength_aa, L_absorbed, **params) -> L_nu``.

    Notes
    -----
    **JIT-compatible**: yes, all operations inside the returned function are ``jnp`` primitives.

    References
    ----------
    Boquien, M. & Salim, S. 2021, A&A, 653, A149.
    """
    if isinstance(template_data, str):
        template_data = load_bosa_templates(template_data)

    # Accept both raw grid format (wavelength_um) and processed (wavelength_aa)
    if "wavelength_um" in template_data and "wavelength_aa" not in template_data:
        template_data = _normalize_bosa_grid(template_data)

    spectra = template_data["spectra"]  # (n_ltir, n_ssfr, n_wave)
    tmpl_wave = template_data["wavelength_aa"]
    log_ltir_grid = template_data["log_ltir_grid"]
    log_ssfr_grid = template_data["log_ssfr_grid"]

    def bosa_emission(
        wavelength_aa: jnp.ndarray,
        L_absorbed: float,
        dust_log_ssfr: float = -10.0,
        redshift: float = 0.0,
        **_kwargs,
    ) -> jnp.ndarray:
        """BOSA emission from tabulated templates (Boquien & Salim 2021).

        Interpolates in (log L_TIR, log sSFR) space.  L_TIR is derived
        from L_absorbed via energy balance.

        Parameters
        ----------
        wavelength_aa : array, shape (n_wave,)
            Wavelength grid in Angstrom (sorted ascending).
        L_absorbed : float
            Total absorbed luminosity in Lsun (= L_TIR).
        dust_log_ssfr : float
            log10(sSFR / yr^-1).  Typical range: -12 to -8.
        redshift : float
            Source redshift (for CMB contrast correction).

        Returns
        -------
        array, shape (n_wave,)
            Dust emission L_nu in Lsun/Hz.

        Notes
        -----
        **JIT-compatible**: yes, all operations are ``jnp`` primitives.

        **Gradient-safe**: yes, differentiable everywhere.
        """
        # L_TIR ~ L_absorbed (energy balance)
        log_ltir = jnp.log10(jnp.clip(L_absorbed, 1.0e-30, None))

        log_ltir_c = jnp.clip(log_ltir, log_ltir_grid[0], log_ltir_grid[-1])
        log_ssfr_c = jnp.clip(dust_log_ssfr, log_ssfr_grid[0], log_ssfr_grid[-1])

        n_l = len(log_ltir_grid)
        n_s = len(log_ssfr_grid)

        # Bilinear interpolation
        i_l = jnp.clip(
            jnp.searchsorted(log_ltir_grid, log_ltir_c) - 1,
            0,
            n_l - 2,
        )
        i_s = jnp.clip(
            jnp.searchsorted(log_ssfr_grid, log_ssfr_c) - 1,
            0,
            n_s - 2,
        )

        fl = (log_ltir_c - log_ltir_grid[i_l]) / (log_ltir_grid[i_l + 1] - log_ltir_grid[i_l])
        fs = (log_ssfr_c - log_ssfr_grid[i_s]) / (log_ssfr_grid[i_s + 1] - log_ssfr_grid[i_s])

        template = (
            (1.0 - fl) * (1.0 - fs) * spectra[i_l, i_s]
            + (1.0 - fl) * fs * spectra[i_l, i_s + 1]
            + fl * (1.0 - fs) * spectra[i_l + 1, i_s]
            + fl * fs * spectra[i_l + 1, i_s + 1]
        )

        # Interpolate onto target wavelength grid FIRST
        sed = resample_template(wavelength_aa, tmpl_wave, template, left=0.0, right=0.0)

        # Normalize to enforce energy balance: ∫L_nu dnu = L_absorbed.
        # Must normalize AFTER resampling to ensure the delivered SED integrates
        # to L_absorbed regardless of the native template grid spacing.
        nu = _C_CGS / (wavelength_aa * _AA_TO_CM)
        t_integral = -jnp.trapezoid(sed, nu)
        norm = jnp.where(t_integral > 0.0, L_absorbed / t_integral, 0.0)

        # No CMB contrast factor here; see the note in ``create_themis_from_grid``.
        # It is an *observational* suppression, so applying it to the emitted SED
        # breaks the energy-balance invariant (int L_nu dnu == L_absorbed) that
        # this component exists to satisfy. ``redshift`` is kept in the signature
        # for call-site compatibility.
        return norm * sed

    return bosa_emission


def register_bosa_tabulated(grid_path: str, name: str = "bosa_tabulated") -> None:
    r"""Load and register the tabulated BOSA model.

    Parameters
    ----------
    grid_path : str
        Path to ``bosa_templates_v2.h5`` or ``bosa_templates.h5``.
    name : str
        Registry name. Default: "bosa_tabulated".

    Returns
    -------
    None
        Model is registered in ``DUST_EMISSION_MODELS`` dict as a side effect.

    Notes
    -----
    **JIT-compatible**: no, registration happens at factory time before JIT.
    """
    from . import emission

    model_fn = create_bosa_from_grid(grid_path)
    emission.DUST_EMISSION_MODELS[name] = model_fn


# Load and create THEMIS
# ── Model 8: THEMIS (Jones et al. 2017), template-based ──────────


def load_themis_templates(filepath: str) -> dict:
    """Load THEMIS template grid from HDF5.

    The template file must contain:

    - ``wavelength_aa``: wavelength grid in Angstrom (n_wave,)
    - ``qhac_grid``: a-C(:H) aromatic fraction (n_qhac,)
    - ``umin_grid``: minimum radiation field intensities (n_umin,)
    - ``single_u``: single-U templates (n_qhac, n_umin, n_wave)
    - ``powerlaw``: 2D PDR templates for back-compat (n_qhac, n_umin, n_wave)
    - ``alpha_grid``: (optional) power-law slope values (n_alpha,)
    - ``powerlaw_alpha``: (optional) 3D PDR templates (n_qhac, n_umin, n_alpha, n_wave)

    Parameters
    ----------
    filepath : str
        Path to ``themis_templates_v2.h5`` or ``themis_templates.h5``.

    Returns
    -------
    dict
        Keys: wavelength_aa, umin_grid, qhac_grid, single_u, powerlaw.
        For extended grids: also alpha_grid, powerlaw_alpha.
        All arrays are JAX arrays.  wavelength_aa is in Angstrom.
        single_u and powerlaw have shape (n_qhac, n_umin, n_wave) and are
        normalized in L_nu convention. If present, powerlaw_alpha has shape
        (n_qhac, n_umin, n_alpha, n_wave).

    Notes
    -----
    **JIT-compatible**: no, file I/O operations not supported in JIT.
    Call at factory/init time before JIT compilation.
    """
    import numpy as np

    already_lnu = False
    alpha_grid = None
    powerlaw_alpha = None

    if filepath.endswith(".npz"):
        data = np.load(filepath)
        wavs_um = np.array(data["wavelength_um"])
        single_u = np.array(data["spectra_single"])
        powerlaw = np.array(data["spectra_pdr"])
        umin_grid = np.array(data["umin_grid"])
        qhac_grid = np.array(data["qhac_grid"])
        wavs_aa = wavs_um * 1.0e4
    else:
        import h5py as _h5py

        with _h5py.File(filepath, "r") as f:
            already_lnu = (
                f.attrs.get("spectra_unit", "") == "L_nu normalized (integral over nu = 1)"
            )
            if "wavelength_aa" in f:
                # Standardized HDF5
                wavs_aa = np.array(f["wavelength_aa"][:])
                single_u = np.array(f["single_u"][:])
                powerlaw = np.array(f["powerlaw"][:])
                umin_grid = np.array(f["umin_grid"][:])
                qhac_grid = np.array(f["qhac_grid"][:])
                # Try to load alpha-dependent PDR component if present
                if "alpha_grid" in f:
                    alpha_grid = np.array(f["alpha_grid"][:])
                if "powerlaw_alpha" in f:
                    powerlaw_alpha = np.array(f["powerlaw_alpha"][:])
                elif "powerlaw_alpha_ratio" in f:
                    # Compact storage: reconstruct the 4-D PDR grid from the
                    # FSPS power-law and a (n_umin, n_alpha, n_wave) reshaping
                    # ratio (scripts/build_themis_alpha_axis.py). Uses the RAW
                    # power-law (before the L_nu normalization below).
                    _ratio = np.array(f["powerlaw_alpha_ratio"][:], dtype=np.float64)
                    powerlaw_alpha = powerlaw[:, :, None, :] * _ratio[None, :, :, :]
            elif "grid" in f:
                wavs_aa = np.array(f["wavelength"][:]) * 1.0e4
                single_u = np.array(f["spectra/single_u"][:])
                powerlaw = np.array(f["spectra/pdr"][:])
                umin_grid = np.array(f["grid/umin"][:])
                qhac_grid = np.array(f["grid/qhac"][:])
                if "grid/alpha" in f:
                    alpha_grid = np.array(f["grid/alpha"][:])
                if "spectra/pdr_alpha" in f:
                    powerlaw_alpha = np.array(f["spectra/pdr_alpha"][:])
            else:
                wavs_aa = np.array(f["wavelength_um"][:]) * 1.0e4
                single_u = np.array(f["spectra_single"][:])
                powerlaw = np.array(f["spectra_pdr"][:])
                umin_grid = np.array(f["umin_grid"][:])
                qhac_grid = np.array(f["qhac_grid"][:])
                if "alpha_grid" in f:
                    alpha_grid = np.array(f["alpha_grid"][:])
                if "powerlaw_alpha" in f:
                    powerlaw_alpha = np.array(f["powerlaw_alpha"][:])

    if not already_lnu:
        # Convert to L_nu and normalize
        wave_cm = wavs_aa * _AA_TO_CM
        nu = _C_CGS / wave_cm

        for arr in (single_u, powerlaw):
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    lnu = arr[i, j] * (wave_cm**2) / _C_CGS
                    integral = -np.trapezoid(lnu, nu)
                    if integral > 0:
                        arr[i, j] = lnu / integral
                    else:
                        arr[i, j] = lnu

        # Normalize powerlaw_alpha if present (4D array)
        if powerlaw_alpha is not None:
            for i in range(powerlaw_alpha.shape[0]):
                for j in range(powerlaw_alpha.shape[1]):
                    for k in range(powerlaw_alpha.shape[2]):
                        lnu = powerlaw_alpha[i, j, k] * (wave_cm**2) / _C_CGS
                        integral = -np.trapezoid(lnu, nu)
                        if integral > 0:
                            powerlaw_alpha[i, j, k] = lnu / integral
                        else:
                            powerlaw_alpha[i, j, k] = lnu

    # Use jnp.array so dynamic JAX indexing works inside JIT.
    # Call preload_emission_model() at factory time (outside JIT) to avoid tracer leaks.
    result = {
        "wavelength_aa": jnp.array(wavs_aa, dtype=jnp.float64),
        "umin_grid": jnp.array(umin_grid, dtype=jnp.float64),
        "qhac_grid": jnp.array(qhac_grid, dtype=jnp.float64),
        "single_u": jnp.array(single_u, dtype=jnp.float64),
        "powerlaw": jnp.array(powerlaw, dtype=jnp.float64),
    }
    if alpha_grid is not None:
        result["alpha_grid"] = jnp.array(alpha_grid, dtype=jnp.float64)
    if powerlaw_alpha is not None:
        result["powerlaw_alpha"] = jnp.array(powerlaw_alpha, dtype=jnp.float64)
    return result


#: Template-backed dust emission models -> ``{grid file: {axis: parameter}}``.
#:
#: Only axes that correspond to a **declared, user-settable parameter** appear.
#: The L_TIR axes (``bosa``'s ``log_ltir_grid``, ``dh02_ce01``'s
#: ``irlum_axis``) are deliberately absent: both are derived from
#: ``L_absorbed`` by energy balance, not set by the user, so a prior can never
#: overhang them and reporting one would be a false positive.
_DUST_EMISSION_GRID_AXES: dict[str, tuple[str, dict[str, str]]] = {
    "dl07": ("dl07_templates.h5", {"umin_grid": "dust_umin", "qpah_grid": "dust_qpah"}),
    "dl14": (
        "dl14_templates.h5",
        {
            "umin_grid": "dust_umin",
            "qpah_grid": "dust_qpah",
            "alpha_grid": "dust_alpha_dl14",
        },
    ),
    "themis": (
        "themis_templates.h5",
        {
            "umin_grid": "dust_umin",
            "qhac_grid": "dust_qhac",
            "alpha_grid": "dust_alpha",
        },
    ),
    "dale2014": ("dale2014_templates.h5", {"alpha_grid": "dust_alpha_dale"}),
    "schreiber2016": ("schreiber2016_templates.h5", {"tdust_grid": "dust_T"}),
    "schreiber2018": ("schreiber2018_templates.h5", {"schreiber2018/tdust": "dust_T"}),
    "bosa": ("bosa_templates.h5", {"log_ssfr_grid": "dust_log_ssfr"}),
    "astrodust": ("astrodust_templates.h5", {"lgU": "dust_lgU"}),
    "dale2014_cigale": ("dale2014_templates_cigale.h5", {"alpha_grid": "dust_alpha_dale"}),
}

#: Selectable menu names that are the same model under another name.
#:
#: The builder menu exposes several spellings per model (``draine_li2007`` and
#: ``dl07_tabulated`` are both ``dl07``). Resolving them here rather than by
#: stripping a suffix is what keeps the census complete: a suffix rule silently
#: missed ``draine_li2007``, leaving that spelling unchecked -- the same
#: census hole this whole registry exists to close.
_DUST_EMISSION_ALIASES: dict[str, str] = {
    "dl07_tabulated": "dl07",
    "draine_li2007": "dl07",
    "draine_li2014": "dl14",
}


def dust_emission_grid_support(name: str) -> dict[str, tuple[float, float]]:
    """Interpolation support of a template-backed dust emission model.

    Reports the extent of each grid axis that a **user-settable** parameter is
    clipped onto, so :mod:`tengri.components.grid_support` can warn when a
    prior overhangs it (#1586).

    Parameters
    ----------
    name : str
        Registry name of the emission model, e.g. ``'themis'``.

    Returns
    -------
    support : dict[str, tuple[float, float]]
        ``{parameter_name: (lo, hi)}``, empty when the model is not
        template-backed or its grid file is not installed.

    Raises
    ------
    FileNotFoundError
        Never -- an absent grid yields ``{}`` so model construction still
        works; the model's own loader raises if it is actually called.

    Notes
    -----
    **JIT-compatible**: not applicable -- composition-time only.

    Axes are read from the grid file, so they cannot go stale if a packaged
    grid is rebuilt. ``themis``'s ``qhac`` axis is routed through
    :func:`_qhac_axis_to_cigale` because the model interpolates on the
    relabeled axis, not the raw dataset; every other axis here was verified
    by measurement to be used as stored.
    """
    entry = _DUST_EMISSION_GRID_AXES.get(_DUST_EMISSION_ALIASES.get(name, name))
    if entry is None:
        return {}
    filename, axis_map = entry

    import h5py
    import numpy as np

    from tengri._data_setup import find_data

    path = find_data(filename)
    if path is None:
        return {}

    support: dict[str, tuple[float, float]] = {}
    with h5py.File(str(path), "r") as f:
        for axis_key, param in axis_map.items():
            if axis_key not in f:
                continue
            axis = np.asarray(f[axis_key][()], dtype=float).ravel()
            if axis.size < 2:
                continue
            if param == "dust_qhac":
                axis = np.asarray(_qhac_axis_to_cigale(jnp.asarray(axis)), dtype=float)
            support[param] = (float(axis.min()), float(axis.max()))
    return support


#: FSPS-to-CIGALE rescaling of the THEMIS a-C(:H) mass-fraction axis.
_QHAC_FSPS_TO_CIGALE = 2.2 / 100.0


def _qhac_axis_to_cigale(qhac_grid: jnp.ndarray) -> jnp.ndarray:
    """Relabel an FSPS-scaled THEMIS ``qhac`` axis to the CIGALE convention.

    The shipped THEMIS grid (``data/themis_templates.h5``) stores its qhac axis
    in FSPS scaling (CIGALE value x 100/2.2, i.e. ``[0.909 .. 18.18]``). The
    user-facing parameter follows CIGALE (``[0.02, 0.40]``, default 0.17 -- see
    ``ThemisIRSEDComponent.qhac`` and CIGALE's ``themis.qhac``). Relabeling
    makes the interpolation happen in the input's units; without it every
    physical qhac < 0.909 -- the entire CIGALE range, including the 0.17
    default -- silently clipped to the grid minimum and selected the wrong
    grain composition, shifting the mid-IR PAH strength and FIR peak by tens of
    percent. That is the #1586 failure mode, and this is why a grid-support
    accessor must report the axis *after* this call, never the raw dataset.

    The a-C(:H) mass fraction never exceeds ~0.5 in the CIGALE convention, so a
    grid whose max exceeds that is unambiguously FSPS-scaled -- which keeps
    CIGALE-unit grids (e.g. synthetic test grids) untouched and makes this
    idempotent.

    Parameters
    ----------
    qhac_grid : ndarray, shape (n_qhac,)
        Raw a-C(:H) mass-fraction axis, either convention [dimensionless].

    Returns
    -------
    ndarray, shape (n_qhac,)
        The axis in the CIGALE convention.

    Notes
    -----
    **JIT-compatible**: no -- reads a concrete max to pick a convention.

    Regression: ``tests/components/dust/test_themis_qhac_convention.py``.
    """
    if float(jnp.max(qhac_grid)) > 0.5:
        return qhac_grid * _QHAC_FSPS_TO_CIGALE
    return qhac_grid


def create_themis_from_grid(template_data: dict | str) -> Callable:
    """Create THEMIS emission function from pre-loaded DustEM template grid.

    The THEMIS model (Jones et al. 2017) uses the same mixing formula
    as DL07 but with different grain compositions.  The aromatic fraction
    parameter ``qhac`` (a-C(:H) aromatic carbon mass fraction) replaces
    ``qpah`` from DL07.

    If the template data includes alpha-dependent PDR grids (``alpha_grid``
    and ``powerlaw_alpha``), the PDR component supports trilinear interpolation
    in (qhac, umin, alpha) space. The single-U component remains bilinear
    (alpha-independent). Back-compat: if alpha grids are absent, the 2D
    ``powerlaw`` slice (fixed alpha=2.0) is used.

    Parameters
    ----------
    template_data : dict or str
        Either a dict (from ``load_themis_templates``) or a file path.

    Returns
    -------
    Callable
        Model function with signature
        ``(wavelength_aa, L_absorbed, dust_alpha=2.0, **params) -> L_nu``.

    Notes
    -----
    **JIT-compatible**: yes, all operations inside the returned function are ``jnp`` primitives.

    References
    ----------
    Jones, A. P. et al. 2017, A&A, 602, A46.
    """
    if isinstance(template_data, str):
        template_data = load_themis_templates(template_data)

    # Accept both raw grid format (spectra_single/spectra_pdr/wavelength_um)
    # and processed format (single_u/powerlaw/wavelength_aa) from load_*
    if "spectra_single" in template_data and "single_u" not in template_data:
        template_data = _normalize_dl07_like_grid(template_data, q_key="qhac_grid")

    single_u = template_data["single_u"]  # (n_qhac, n_umin, n_wave)
    powerlaw = template_data["powerlaw"]  # (n_qhac, n_umin, n_wave)
    tmpl_wave = template_data["wavelength_aa"]
    umin_grid = template_data["umin_grid"]
    # ``_qhac_axis_to_cigale`` reads a concrete max to pick a unit convention,
    # so it cannot run on traced arrays. When the caller threaded this grid in
    # (see EmissionComponent.threaded_templates) it supplies the axis already
    # converted, under a distinct key. The membership test is Python-level and
    # therefore safe under trace; the conversion itself stays eager (#1649).
    if "qhac_grid_cigale" in template_data:
        qhac_grid = template_data["qhac_grid_cigale"]
    else:
        qhac_grid = _qhac_axis_to_cigale(template_data["qhac_grid"])

    # Optional: alpha-dependent PDR component
    alpha_grid = template_data.get("alpha_grid", None)
    powerlaw_alpha = template_data.get("powerlaw_alpha", None)
    has_alpha = alpha_grid is not None and powerlaw_alpha is not None

    def themis_emission(
        wavelength_aa: jnp.ndarray,
        L_absorbed: float,
        dust_umin: float = 1.0,
        dust_gamma_dl: float = 0.01,
        dust_qhac: float = 0.17,
        dust_alpha: float = 2.0,
        redshift: float = 0.0,
        **_kwargs,
    ) -> jnp.ndarray:
        """THEMIS emission from tabulated DustEM templates (Jones+2017).

        j_nu = (1-gamma) * single_U(qhac, Umin) + gamma * PDR(qhac, Umin, alpha)

        The single-U component is bilinear in (qhac, Umin). The PDR component
        is bilinear in (qhac, Umin) if alpha grids are absent (back-compat),
        or trilinear in (qhac, Umin, alpha) if present.

        Parameters
        ----------
        wavelength_aa : array, shape (n_wave,)
            Wavelength grid in Angstrom (sorted ascending).
        L_absorbed : float
            Total absorbed luminosity in Lsun.
        dust_umin : float
            Minimum radiation field intensity (Mathis ISRF units).
        dust_gamma_dl : float
            Fraction of dust mass in PDR (high-U) component.
        dust_qhac : float
            a-C(:H) aromatic carbon mass fraction.
            Typical range: 0.02--0.30.
        dust_alpha : float
            Radiation field power-law slope [dimensionless].
            Default: 2.0 (back-compat with 2D powerlaw slice).
        redshift : float
            Source redshift (for CMB contrast correction).

        Returns
        -------
        array, shape (n_wave,)
            Dust emission L_nu in Lsun/Hz.

        Notes
        -----
        **JIT-compatible**: yes, all operations are ``jnp`` primitives.

        **Gradient-safe**: yes, differentiable everywhere.
        """
        dust_umin_c = jnp.clip(dust_umin, umin_grid[0], umin_grid[-1])
        dust_qhac_c = jnp.clip(dust_qhac, qhac_grid[0], qhac_grid[-1])

        # Bilinear interpolation indices for (qhac, umin)
        i_u = jnp.clip(
            jnp.searchsorted(umin_grid, dust_umin_c) - 1,
            0,
            len(umin_grid) - 2,
        )
        i_q = jnp.clip(
            jnp.searchsorted(qhac_grid, dust_qhac_c) - 1,
            0,
            len(qhac_grid) - 2,
        )

        fu = (dust_umin_c - umin_grid[i_u]) / (umin_grid[i_u + 1] - umin_grid[i_u])
        fq = (dust_qhac_c - qhac_grid[i_q]) / (qhac_grid[i_q + 1] - qhac_grid[i_q])

        def _bilinear(grid: jnp.ndarray) -> jnp.ndarray:
            """Perform 2D linear interpolation over qhac and Umin axes."""
            return (
                (1.0 - fq) * (1.0 - fu) * grid[i_q, i_u]
                + (1.0 - fq) * fu * grid[i_q, i_u + 1]
                + fq * (1.0 - fu) * grid[i_q + 1, i_u]
                + fq * fu * grid[i_q + 1, i_u + 1]
            )

        # Compute PDR component: bilinear if no alpha axis, else trilinear
        if has_alpha:
            dust_alpha_c = jnp.clip(dust_alpha, alpha_grid[0], alpha_grid[-1])
            n_a = len(alpha_grid)
            i_a = jnp.clip(
                jnp.searchsorted(alpha_grid, dust_alpha_c) - 1,
                0,
                n_a - 2,
            )
            fa = (dust_alpha_c - alpha_grid[i_a]) / (alpha_grid[i_a + 1] - alpha_grid[i_a])

            def _trilinear_pdr(grid: jnp.ndarray) -> jnp.ndarray:
                """Perform 3D linear interpolation over qhac, Umin, and alpha axes."""

                # Interpolate at alpha[i_a] and alpha[i_a+1] via bilinear in (q, u)
                def _bilinear_at_alpha(ia_idx):
                    """Interpolate bilinearly at fixed alpha index."""
                    return (
                        (1.0 - fq) * (1.0 - fu) * grid[i_q, i_u, ia_idx]
                        + (1.0 - fq) * fu * grid[i_q, i_u + 1, ia_idx]
                        + fq * (1.0 - fu) * grid[i_q + 1, i_u, ia_idx]
                        + fq * fu * grid[i_q + 1, i_u + 1, ia_idx]
                    )

                lo = _bilinear_at_alpha(i_a)
                hi = _bilinear_at_alpha(i_a + 1)
                return (1.0 - fa) * lo + fa * hi

            pdr_template = _trilinear_pdr(powerlaw_alpha)
        else:
            # Back-compat: use 2D powerlaw at fixed alpha=2.0
            pdr_template = _bilinear(powerlaw)

        # Mix single-U (diffuse) and power-law (PDR) components. ``gamma`` is a
        # dust-mass fraction; the FSPS/DustEM ``powerlaw`` template carries its
        # real relative luminosity (∫powerlaw/∫single_u ≈ 14-19), so gamma acts
        # as a luminosity fraction directly: no analytic R needed (cf. DL07).
        template = (1.0 - dust_gamma_dl) * _bilinear(single_u) + dust_gamma_dl * pdr_template

        # Interpolate onto target wavelength grid FIRST
        sed = resample_template(wavelength_aa, tmpl_wave, template, left=0.0, right=0.0)

        # Energy balance: renormalize to unit frequency integral on the
        # evaluation grid AFTER resampling to ensure the delivered SED
        # integrates to L_absorbed regardless of the native template grid spacing.
        nu = _C_CGS / (wavelength_aa * _AA_TO_CM)
        t_integral = -jnp.trapezoid(sed, nu)
        norm = jnp.where(t_integral > 0.0, L_absorbed / t_integral, 0.0)

        # The da Cunha et al. (2013) CMB contrast factor is deliberately NOT
        # applied to the emitted SED.
        #
        # It is an *observational* suppression: the flux you measure above the
        # CMB background: so multiplying the emitted spectrum by it breaks the
        # invariant this component exists to satisfy: the dust must re-emit the
        # starlight it absorbed, int L_nu dnu == L_absorbed. Applying it after the
        # unit-integral renormalization silently destroyed up to 1.6% of the
        # absorbed energy (worst for the aromatic-rich, mm-bright templates), and
        # it never even received a real redshift: no component plumbs one through
        #: so it only ever imposed the z=0 suppression. DL07/DL14 never applied
        # it, and CIGALE conserves to 0.01%, so THEMIS/Astrodust/BOSA were the
        # odd ones out. If a CMB contrast is wanted it belongs at the observation
        # layer, applied uniformly, not inside the energy-balanced emitter.
        # ``redshift`` is kept in the signature for call-site compatibility.
        return norm * sed

    return themis_emission


# Register THEMIS
def register_themis_tabulated(grid_path: str, name: str = "themis_tabulated") -> None:
    """Load and register the tabulated THEMIS model.

    Parameters
    ----------
    grid_path : str
        Path to ``themis_templates_v2.h5`` or ``themis_templates.h5``.
    name : str
        Registry name.  Default ``"themis_tabulated"``.

    Returns
    -------
    None
        Model is registered in ``DUST_EMISSION_MODELS`` dict as a side effect.

    Notes
    -----
    **JIT-compatible**: no, registration happens at factory time before JIT.
    """
    from . import emission

    model_fn = create_themis_from_grid(grid_path)
    emission.DUST_EMISSION_MODELS[name] = model_fn


# Lazy loaders and initialization
_resolved: set[str] = set()


def _make_lazy_loader(
    name: str,
    template_filename: str,
    loader_fn_name: str,
) -> Callable:
    """Create a lazy-loading wrapper that auto-loads templates on first call.

    Parameters
    ----------
    name : str
        Registry name (e.g. ``"dale2014"``).
    template_filename : str
        Canonical HDF5 filename to search for in data/ (e.g. ``"dale2014_templates.h5"``).
        The v2 variant (``"*_v2.h5"``) is tried first if present.
    loader_fn_name : str
        Name of the ``create_*_from_grid`` function in this module.
    """

    def _lazy_wrapper(*args, **kwargs):
        """Lazy load template on first call, then dispatch to runtime function."""
        if name not in _resolved:
            _resolved.add(name)
            # Try v2 HDF5 first (improved grid), then canonical HDF5
            stem = template_filename.rsplit(".", 1)[0]
            v2_name = stem + "_v2.h5"
            path = find_data_str(v2_name, template_filename)
            if path is not None:
                from . import emission

                loader = globals()[loader_fn_name]
                tabulated = loader(path)
                emission.DUST_EMISSION_MODELS[name] = tabulated
                return tabulated(*args, **kwargs)
            else:
                raise FileNotFoundError(
                    f"Template file '{template_filename}' not found in data/. "
                    f"The analytic fallback for {name} has been removed because it "
                    f"produced scientifically incorrect results. Download templates "
                    f"or register manually via register_*_tabulated()."
                )
        from . import emission

        return emission.DUST_EMISSION_MODELS[name](*args, **kwargs)

    _lazy_wrapper.__name__ = name
    _lazy_wrapper.__doc__ = (
        f"Lazy-loading wrapper for {name}. Auto-loads tabulated templates "
        f"from data/{template_filename} on first call (v2 grid preferred if present)."
    )
    return _lazy_wrapper


# --- DL07: tries v2 HDF5 first, then legacy .h5 ---
def _dl07_lazy_wrapper(*args, **kwargs):
    """Draine & Li (2007): auto-loads tabulated templates on first call."""
    from . import emission

    if "draine_li2007" not in _resolved:
        _resolved.add("draine_li2007")
        path = find_data_str("dl07_templates_v2.h5", "dl07_templates.h5")
        if path is not None:
            tabulated = create_dl07_from_grid(path)
            emission.DUST_EMISSION_MODELS["draine_li2007"] = tabulated
            emission.DUST_EMISSION_MODELS["dl07_tabulated"] = tabulated
            return tabulated(*args, **kwargs)
        else:
            raise FileNotFoundError(
                "DL07 template files (dl07_templates_v2.h5 / dl07_templates.h5) "
                "not found in data/. "
                "The analytic fallback has been removed because it produced "
                "scientifically incorrect results (single-Gaussian PAH approximation). "
                "Run: python scripts/convert_dl07_templates.py"
            )
    return emission.DUST_EMISSION_MODELS["draine_li2007"](*args, **kwargs)


# ===========================================================================
# Draine, Li, Hensley et al. 2021 PAHspec template loader
# ===========================================================================


@dataclass(frozen=True)
class Draine2021PAHTemplates:
    """Frozen container for the Draine+2021 PAHspec template grid.

    All spectrum arrays are :math:`\nu P_\nu` in erg/s/H (per H atom),
    shaped ``(n_starlight, n_slab, n_lgU, n_ion, n_size, n_wave)``.
    The spectra are emitted-power per H atom: galaxy-scale rescaling is
    handled downstream by the SEDComponent via energy balance with the
    absorbed luminosity ``L_ir``.
    """

    wavelength_um: jnp.ndarray
    lgU: jnp.ndarray
    nu_pnu_total: jnp.ndarray
    nu_pnu_astrodust: jnp.ndarray
    nu_pnu_pah_plus: jnp.ndarray
    nu_pnu_pah_neutral: jnp.ndarray
    tir_total: jnp.ndarray
    present: jnp.ndarray
    starlight_names: tuple[str, ...]
    ion_names: tuple[str, ...]
    size_names: tuple[str, ...]
    slab: jnp.ndarray
    paper: str = "Draine, Li, Hensley, Hunt, Sandstrom, Smith 2021, ApJ 917, 3"
    arxiv: str = "2011.07046"


def load_draine2021_pahspec_templates(filepath: str) -> Draine2021PAHTemplates:
    """Load a PAHspec HDF5 grid built by ``scripts/build_pahspec_hdf5.py``.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.  See ``scripts/build_pahspec_hdf5.py``
        for the layout.

    Returns
    -------
    Draine2021PAHTemplates
        Frozen dataclass with JAX arrays for all spectrum cubes.

    Notes
    -----
    **JIT-compatible**: no, file I/O.  Call once outside the JIT
    boundary (typically in ``SEDComponent.precompute()``).

    References
    ----------
    .. [1] Draine, B.T., Li, A., Hensley, B.S., Hunt, L.K.,
       Sandstrom, K., Smith, J.-D.T., 2021, "Excitation of PAH
       Emission: Dependence on Size Distribution, Ionization, and
       Starlight Spectrum and Intensity", ApJ, 917, 3,
       arXiv:2011.07046.
    """
    import h5py

    with h5py.File(filepath, "r") as f:
        wave_um = jnp.asarray(f["wavelength_um"][:])
        lgU = jnp.asarray(f["lgU"][:])
        nu_pnu_total = jnp.asarray(f["nu_pnu_total"][...])
        nu_pnu_astrodust = jnp.asarray(f["nu_pnu_astrodust"][...])
        nu_pnu_pah_plus = jnp.asarray(f["nu_pnu_pah_plus"][...])
        nu_pnu_pah_neutral = jnp.asarray(f["nu_pnu_pah_neutral"][...])
        tir_total = jnp.asarray(f["tir_total"][...])
        present = jnp.asarray(f["present"][...])
        starlight_names = tuple(
            n.decode() if isinstance(n, bytes) else n for n in f["starlight_names"][:]
        )
        ion_names = tuple(n.decode() if isinstance(n, bytes) else n for n in f["ion_names"][:])
        size_names = tuple(n.decode() if isinstance(n, bytes) else n for n in f["size_names"][:])
        slab = jnp.asarray(f["slab"][:])

    return Draine2021PAHTemplates(
        wavelength_um=wave_um,
        lgU=lgU,
        nu_pnu_total=nu_pnu_total,
        nu_pnu_astrodust=nu_pnu_astrodust,
        nu_pnu_pah_plus=nu_pnu_pah_plus,
        nu_pnu_pah_neutral=nu_pnu_pah_neutral,
        tir_total=tir_total,
        present=present,
        starlight_names=starlight_names,
        ion_names=ion_names,
        size_names=size_names,
        slab=slab,
    )
