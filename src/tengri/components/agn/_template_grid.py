# SPDX-License-Identifier: BSD-3-Clause
"""Shared carrier and evaluator for tabulated AGN torus template libraries.

Several torus families (CAT3D-Wind, Nenkova/CLUMPY, SKIRTOR-AGNfitter) are the
same computation over different data: PCHIP-interpolate an N-D template grid at
the model's coordinates, resample onto the requested wavelengths, then
renormalize by the frequency integral and scale to
``L_bol × f_torus``. This module holds that computation once.

The reason the grid is a :class:`TorusTemplateGrid` **argument** rather than a
closed-over array is threading. A closure's captured arrays are concrete at
trace time, so JAX freezes them into the graph as ``Constant`` ops, the whole
library, inlined, every time. A pytree passed as an argument becomes a
``Parameter`` instead. See ``tengri.components.agn.blocks._protocol.collect_block_templates``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.agn._phys import (
    bolometric_integral_nu as _bolometric_integral_nu,
    wavelength_to_nu as _wavelength_to_nu,
)
from tengri.utils.grid_interp import interp_nd_pchip, loglog_integral, resample_template
from tengri.utils.host_array import device_table
from tengri.utils.physics_constants import C_AA as _C_AA, L_SUN as _LSUN_ERG
from tengri.utils.scale import apply_log10_scale

__all__ = [
    "BUDGET_WAVE",
    "TorusTemplateGrid",
    "analytic_bolometric_nu",
    "cache_per_float_dtype",
    "float32_safe_template",
    "loaded_template",
    "native_bolometric_nu",
    "native_bolometric_nu_np",
    "scale_by_torus_power",
    "scale_to_lbol_native",
    "scale_to_torus_power_native",
    "torus_lnu_from_grid",
]

#: Fixed wavelength grid [Angstrom] on which a published bolometric ``L_*`` is integrated.
#: It spans every AGN disc, torus and graybody support (1000 log points per decade, the
#: resolution the budgets are defined at) and never depends on the caller's wavelength array.
BUDGET_WAVE = np.geomspace(1.0e-3, 1.0e10, 13001)


class TorusTemplateGrid(NamedTuple):
    """A tabulated torus library, as a JAX pytree.

    Attributes
    ----------
    template : ndarray, shape (n_ax1, ..., n_axk, n_wave)
        Tabulated SEDs [arbitrary units: shape only; renormalized on use].
    axes : tuple of ndarray
        One 1-D coordinate array per leading template axis, ascending.
    wave_grid : ndarray, shape (n_wave,)
        Template rest-frame wavelength grid [Angstrom].
    anisotropy : ndarray, shape (n_ax1, ..., n_axk), optional
        For a library whose records share one scale per model across inclination: each
        record's power over its model's mean (``_torus_inclination.anisotropy_values``),
        computed once at load. ``None`` for a library normalized per sightline.

    Notes
    -----
    Leaves are normally ``np.ndarray`` when loaded from disk and JAX tracers
    when threaded through ``jax.jit``; both work, because every use site
    calls ``jnp.asarray`` (identity on a tracer).
    """

    template: jnp.ndarray
    axes: tuple[jnp.ndarray, ...]
    wave_grid: jnp.ndarray
    anisotropy: jnp.ndarray | None = None


def float32_safe_template(template, dtype):
    """A tabulated template as a float32-representable array when ``dtype`` is float32.

    Several libraries are stored as shape only on an arbitrary scale (the Silva+04 and CAT3D
    tables peak at ~1e-68, the SKIRTOR AGNfitter 1- and 2-parameter tables at ~1e-71), so
    casting them to float32 flushes every entry to zero. In float32 a host (numpy) table is
    divided by its peak, in float64 on the host, before it is cast; the scale is divided out
    again by :func:`scale_to_lbol_native`, so the SED is unchanged. In float64, and for a table
    already on the device (a tracer threaded through ``jit``), the table is returned as is; a
    threaded table is float32-safe only if its loader stored it so (:func:`loaded_template`).

    Parameters
    ----------
    template : array_like
        Tabulated SEDs [arbitrary units: shape only].
    dtype : dtype
        Floating dtype of the evaluation (that of the caller's wavelength array).

    Returns
    -------
    array_like
        ``template`` or ``template / max(template)``.

    Notes
    -----
    **JIT-compatible**: yes, for a host table the peak is a trace-time constant.
    """
    if dtype == jnp.float32 and isinstance(template, np.ndarray):
        peak = float(np.max(template))
        if peak > 0.0:
            return template / peak
    return template


def loaded_template(template) -> np.ndarray:
    """A library's template as a loader stores it: float32-safe in a float32 process.

    The kernels make a host table float32-safe at the point of use
    (:func:`float32_safe_template`), but a table threaded through ``jit`` as an argument
    (``blocks._protocol.collect_block_templates``) is cast to float32 when it enters the
    trace, before any kernel sees it, and a ~1e-68 table flushes to zero: the torus
    is then silently 0. So a loader running in float32 stores the table divided by its
    peak (the shape is all a template carries; every kernel divides by its integral).
    In float64 the table is stored as read, so float64 results are unchanged bit for bit.
    Pair with :func:`cache_per_float_dtype`.

    Parameters
    ----------
    template : array_like
        Tabulated SEDs as read [arbitrary units: shape only].

    Returns
    -------
    ndarray
        ``template`` as a NumPy array, or ``template / max(template)`` when JAX runs in
        float32.
    """
    return float32_safe_template(np.asarray(template), jax.dtypes.canonicalize_dtype(float))


def cache_per_float_dtype(loader: Callable) -> Callable:
    """Cache a grid loader once per process float dtype (float64 under x64, else float32).

    A loader that stores its template float32-safe (:func:`float32_safe_template` at the
    process dtype) builds a different table in each precision mode, so a cache keyed on the
    path alone hands one mode's table to the other: the raw table to a float32 forward
    that threads it through ``jit`` (where the cast flushes it to zero), or the rescaled
    table to a float64 forward. The key carries the dtype, as the SKIRTOR loaders do
    (#2275).

    Parameters
    ----------
    loader : callable
        The loader, with hashable positional arguments.

    Returns
    -------
    callable
        The cached loader, with ``cache_clear``, ``cache_info`` and ``__wrapped__``.
    """

    @functools.cache
    def _for_dtype(float_dtype_name: str, *args):
        return loader(*args)

    @functools.wraps(loader)
    def cached(*args):
        return _for_dtype(jax.dtypes.canonicalize_dtype(float).name, *args)

    cached.cache_clear = _for_dtype.cache_clear
    cached.cache_info = _for_dtype.cache_info
    return cached


def native_bolometric_nu(
    lnu_native: jnp.ndarray, wave_native: jnp.ndarray, *, floor: float = 1e-100
) -> jnp.ndarray:
    r"""Frequency integral of a template on its OWN wavelength grid.

    The normalization every tabulated component divides by. Taken on the
    template's native grid, before resampling, it is a property of the template
    and the model coordinates alone: the caller's wavelength sampling and range
    never enter. (A trapezoid over the caller's grid would clip the template
    wherever the grid stops short of its support and would shift with the node
    density.)

    The integral is that of the interpolant
    :func:`~tengri.utils.grid_interp.resample_template` builds (a power law
    between adjacent nodes in :math:`\lambda`), so the resampled template
    integrates to the same value on any grid fine enough to resolve it. A
    trapezoid in :math:`\nu` over the native nodes would instead lay a chord
    across a convex segment and differ by up to ~1e-2 on the coarse (R ~ 7)
    libraries.

    Parameters
    ----------
    lnu_native : array_like, shape (n_native,)
        Template :math:`L_\nu` on ``wave_native`` [arbitrary units per Hz;
        shape only].
    wave_native : array_like, shape (n_native,)
        Template wavelength grid [Angstrom], ascending.
    floor : float, optional
        Lower bound on the returned magnitude, so the division cannot hit zero.

    Returns
    -------
    ndarray, shape ()
        :math:`\max(|\int L_\nu\,d\nu|, \text{floor})` in the template's units.

    Notes
    -----
    .. math::

        \int L_\nu\,\mathrm{d}\nu
        = \int L_\nu(\lambda)\,\frac{c}{\lambda^2}\,\mathrm{d}\lambda,

    where the integrand is a power law in :math:`\lambda` on every segment
    (:math:`L_\nu\propto\lambda^{s}` gives :math:`\lambda^{s-2}`), integrated
    in closed form by :func:`~tengri.utils.grid_interp.loglog_integral`.

    **JIT-compatible**: yes. **Gradient-safe**: yes. The resampling kernels used
    after this are homogeneous of degree one in the template values (log-flux
    interpolation is not linear, but scaling the template scales the result by the
    same factor), so dividing the resampled template by this integral equals
    resampling the normalized template.
    """
    wave = jnp.asarray(wave_native)
    integrand = jnp.asarray(lnu_native) * (_C_AA / wave**2)
    return jnp.maximum(jnp.abs(loglog_integral(wave, integrand)), floor)


def native_bolometric_nu_np(
    lnu_native: np.ndarray, wave_native: np.ndarray, *, floor: float = 1e-100
) -> float:
    r"""NumPy twin of :func:`native_bolometric_nu`, for build-time precompute tables.

    Same integral (a power law in wavelength between nodes, closed form), in
    float64 whatever the JAX precision mode, so a precomputed table and the
    runtime path normalize a template identically.

    Parameters
    ----------
    lnu_native : array_like, shape (n_native,)
        Template :math:`L_\nu` on ``wave_native`` [arbitrary units per Hz].
    wave_native : array_like, shape (n_native,)
        Template wavelength grid [Angstrom], ascending.
    floor : float, optional
        Lower bound on the returned magnitude.

    Returns
    -------
    float
        :math:`\max(|\int L_\nu\,d\nu|, \text{floor})`.
    """
    x = np.asarray(wave_native, dtype=np.float64)
    y = np.asarray(lnu_native, dtype=np.float64) * (_C_AA / x**2)
    x0, x1, y0, y1 = x[:-1], x[1:], y[:-1], y[1:]
    positive = (y0 > 0.0) & (y1 > 0.0)
    y0_safe = np.where(positive, y0, 1.0)
    y1_safe = np.where(positive, y1, 1.0)
    a = np.log(x1 / x0)
    u = a + np.log(y1_safe) - np.log(y0_safe)
    small = np.abs(u) < 1e-7
    ratio = np.where(small, 1.0 + 0.5 * u, np.expm1(u) / np.where(small, 1.0, u))
    power_law = x0 * y0_safe * a * ratio
    chord = 0.5 * (y0 + y1) * (x1 - x0)
    return max(abs(float(np.sum(np.where(positive, power_law, chord)))), floor)


def scale_to_lbol_native(
    template_native: jnp.ndarray,
    wave_native: jnp.ndarray,
    sed: jnp.ndarray,
    l_scale: float,
) -> jnp.ndarray:
    r"""Scale a resampled shape-only template to ``l_scale`` of bolometric power.

    .. math::

        L_\nu(\lambda) = L_{\rm scale}\,
            \frac{T(\lambda)}{\int T(\nu)\,\mathrm{d}\nu},

    with the integral taken over the template's NATIVE grid (``wave_native``),
    never over the caller's wavelength array, so the result does not depend on
    how the caller samples or truncates wavelength.

    Parameters
    ----------
    template_native : array_like, shape (n_native,)
        Interpolated template on ``wave_native`` [shape only; any scale].
    wave_native : array_like, shape (n_native,)
        Template wavelength grid [Angstrom].
    sed : array_like, shape (n_wave,)
        The same template resampled onto the caller's grid [same units as
        ``template_native``].
    l_scale : float
        Bolometric power the full template integrates to [erg/s].

    Returns
    -------
    ndarray, shape (n_wave,)
        :math:`L_\nu` [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes.

    Float32 (#1206): the template's bolometric integral can reach ~1e45 erg/s,
    which overflows float32 and would flush the disc to zero, and the
    ``1e-100`` floor is itself below the float32 minimum. In float32 the
    integrand is therefore divided by its (stop-gradient) peak before it is
    integrated and the factors regrouped as
    ``(l_scale / hat_int) * (sed / peak)``, algebraically identical.
    """
    template_native = jnp.asarray(template_native)
    if sed.dtype == jnp.float32:
        peak = jax.lax.stop_gradient(jnp.max(jnp.abs(template_native)))
        peak = jnp.where(peak > 0.0, peak, 1.0)
        hat_int = native_bolometric_nu(template_native / peak, wave_native, floor=1e-30)
        return (l_scale / hat_int) * (sed / peak)
    return l_scale * sed / native_bolometric_nu(template_native, wave_native)


def _float32_torus_power(linear, sed, integral, agn_log_lbol, agn_torus_frac):
    r"""Float32 :math:`f\,10^{\rm agn\_log\_lbol} L_\odot\,T_\nu / \int T_\nu d\nu`, range-safe.

    Below ``agn_log_lbol = 0`` the power is at most :math:`L_\odot`, so the linear product
    ``linear(l_scale)`` is finite and is used as is: that is the ~1e10 erg/s reference
    luminosity ``_lbol_reference.reference_evaluation`` evaluates at, which keeps its value
    bit for bit. Above it the power and the integral are one ``log10`` offset, applied by
    ``utils.scale.apply_log10_scale`` (NAMING_CONTRACT 4c), and :math:`L_\odot` (in range)
    multiplies the result, so no erg/s-scale linear value is formed and the output is
    finite whenever the emitted :math:`L_\nu` is. The solar decades stay out of the
    exponent on purpose: a float32 exponent of ~34 dex resolves only ~4e-6 of a decade,
    an error of ~1e-5 in :math:`L_\nu`; the offset left is ``agn_log_lbol - log10(integral)``.
    Each branch reads ``agn_log_lbol`` clamped to its own side of 0, so the branch not taken
    stays finite and contributes a zero, not a NaN, to the gradient.
    """
    log_lbol = jnp.asarray(agn_log_lbol)
    above_zero = log_lbol > 0.0
    low = jnp.where(above_zero, 0.0, log_lbol)
    high = jnp.where(above_zero, log_lbol, 0.0)
    below = linear(10.0**low * _LSUN_ERG * agn_torus_frac)
    above = apply_log10_scale(agn_torus_frac * sed, high - jnp.log10(integral)) * _LSUN_ERG
    return jnp.where(above_zero, above, below)


def scale_by_torus_power(sed, integral, agn_log_lbol, agn_torus_frac):
    r"""Scale a template by :math:`f L_{\rm bol}` over its frequency integral, range-safe.

    .. math::

        L_\nu(\lambda) = f\,10^{\rm agn\_log\_lbol}\,L_\odot\,
            \frac{T_\nu(\lambda)}{\int T_\nu\,\mathrm{d}\nu},

    where :math:`T_\nu` is the template [any units per Hz], :math:`f` the torus fraction
    [dimensionless] and :math:`L_\odot` [erg/s].

    Parameters
    ----------
    sed : array_like, shape (n_wave,)
        The template on the caller's grid [template units].
    integral : array_like, shape ()
        Its frequency integral [template units x Hz], positive.
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`, the accretion power [dex].
    agn_torus_frac : float
        Fraction of :math:`L_{\rm bol}` the template carries [dimensionless].

    Returns
    -------
    ndarray, shape (n_wave,)
        :math:`L_\nu` [erg/s/Hz].

    Notes
    -----
    **JIT/grad/vmap-safe**: yes.

    In float64 this is the linear product ``10**agn_log_lbol * L_sun * f * sed / integral``,
    evaluated in that order. In float32 the power ``10**agn_log_lbol * L_sun`` passes
    float32's 3.4e38 at ``agn_log_lbol`` of about 4.9 and the product is ``inf`` (#3084);
    see :func:`_float32_torus_power` for how it is carried there.
    """
    if jnp.asarray(sed).dtype == jnp.float32:
        return _float32_torus_power(
            lambda l_scale: l_scale * sed / integral, sed, integral, agn_log_lbol, agn_torus_frac
        )
    l_scale = 10.0**agn_log_lbol * _LSUN_ERG * agn_torus_frac
    return l_scale * sed / integral


def scale_to_torus_power_native(
    template_native: jnp.ndarray,
    wave_native: jnp.ndarray,
    sed: jnp.ndarray,
    agn_log_lbol: float,
    agn_torus_frac: float,
) -> jnp.ndarray:
    r""":func:`scale_to_lbol_native` with the power given as ``agn_log_lbol`` and a fraction.

    Parameters
    ----------
    template_native : array_like, shape (n_native,)
        Interpolated template on ``wave_native`` [shape only; any scale].
    wave_native : array_like, shape (n_native,)
        Template wavelength grid [Angstrom].
    sed : array_like, shape (n_wave,)
        The same template resampled onto the caller's grid [same units as
        ``template_native``].
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`, the accretion power [dex].
    agn_torus_frac : float
        Fraction of :math:`L_{\rm bol}` the template carries [dimensionless].

    Returns
    -------
    ndarray, shape (n_wave,)
        :math:`L_\nu` [erg/s/Hz], integrating to :math:`f L_{\rm bol}` over the template's
        native grid.

    Notes
    -----
    **JIT/grad/vmap-safe**: yes.

    Float64 is :func:`scale_to_lbol_native` with ``l_scale = 10**agn_log_lbol * L_sun * f``,
    operation for operation. In float32 the template is divided by its (stop-gradient) peak
    before it is integrated, as there; at ``agn_log_lbol <= 0`` the result is then that
    function's float32 value, and above it the power is applied as a ``log10`` offset
    (:func:`_float32_torus_power`), so no erg/s-scale linear value is formed (#3084).
    """
    if sed.dtype != jnp.float32:
        l_scale = 10.0**agn_log_lbol * _LSUN_ERG * agn_torus_frac
        return scale_to_lbol_native(template_native, wave_native, sed, l_scale)
    template_native = jnp.asarray(template_native)
    peak = jax.lax.stop_gradient(jnp.max(jnp.abs(template_native)))
    peak = jnp.where(peak > 0.0, peak, 1.0)
    hat_int = native_bolometric_nu(template_native / peak, wave_native, floor=1e-30)
    unit = sed / peak
    return _float32_torus_power(
        lambda l_scale: (l_scale / hat_int) * unit, unit, hat_int, agn_log_lbol, agn_torus_frac
    )


def analytic_bolometric_nu(
    shape_fn: Callable[[jnp.ndarray], jnp.ndarray],
    wave_lo: float,
    wave_hi: float,
    n_nodes: int = 4097,
    *,
    floor: float | None = None,
) -> jnp.ndarray:
    r"""Frequency integral of an analytic shape on a FIXED internal grid.

    For shapes with no tabulated native grid and no closed form (a Planck
    function times an opacity, a blend of sigmoids). The integration grid is
    ``n_nodes`` log-spaced wavelengths over ``[wave_lo, wave_hi]``, chosen by
    the caller to span the shape's support; the caller's own wavelength
    array never enters, so the normalization cannot depend on it.

    Parameters
    ----------
    shape_fn : callable
        Maps wavelength ``array_like, shape (n,)`` [Angstrom] to
        :math:`L_\nu` ``ndarray, shape (n,)`` [arbitrary units per Hz].
    wave_lo, wave_hi : float
        Wavelength span of the internal grid [Angstrom]; the shape must be
        negligible outside it.
    n_nodes : int, optional
        Grid nodes. The trapezoid error on a smooth shape is
        :math:`O((\Delta\ln\lambda)^2)`, ~1e-6 at the default.
    floor : float, optional
        When given, return ``max(|integral|, floor)`` (safe denominator).

    Returns
    -------
    ndarray, shape ()
        :math:`\int L_\nu\,d\nu` in the shape's units (floored magnitude if ``floor``).

    Notes
    -----
    **JIT-compatible**: yes (the grid is a trace-time constant).
    **Gradient-safe**: yes.
    """
    wave = jnp.asarray(np.geomspace(wave_lo, wave_hi, n_nodes))
    return _bolometric_integral_nu(shape_fn(wave), _wavelength_to_nu(wave), floor=floor)


def torus_lnu_from_grid(
    grid: TorusTemplateGrid,
    wavelength: jnp.ndarray,
    coords: tuple,
    *,
    agn_log_lbol: float,
    agn_torus_frac: float,
) -> jnp.ndarray:
    r"""Interpolate a torus template library and scale it to ``L_bol``.

    Parameters
    ----------
    grid : TorusTemplateGrid
        Template arrays, passed in so they can thread through ``jax.jit``.
    wavelength : array_like, shape (n_wave,)
        Rest-frame output wavelength grid [Angstrom].
    coords : tuple of float
        Interpolation coordinate per leading axis of ``grid.template``,
        in the same order as ``grid.axes``.
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`.
    agn_torus_frac : float
        Fraction of :math:`L_{\rm bol}` reprocessed by the torus [0, 1].

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density [erg/s/Hz].

    Notes
    -----
    .. math::

        L_\nu(\lambda) = L_{\rm bol}\, f_{\rm torus}\,
                         \frac{T(\lambda;\, \mathbf{c})}
                              {\int T(\nu;\, \mathbf{c})\,\mathrm{d}\nu}

    where :math:`T` is the interpolated template, :math:`\mathbf{c}` the
    coordinates, :math:`L_{\rm bol} = 10^{\rm agn\_log\_lbol} L_\odot`, and
    the integral runs over the template's own native frequency grid (before
    resampling), so the result does not depend on how ``wavelength`` is
    sampled or where it starts and stops. The template carries shape only; its
    absolute scale is divided out.

    **JIT-compatible**: yes. **Gradient-safe**: yes, node-exact PCHIP is
    C¹-continuous across every axis.
    """
    template = interp_nd_pchip(
        device_table(float32_safe_template(grid.template, wavelength.dtype)),
        tuple(device_table(axis) for axis in grid.axes),
        coords,
    )
    wave_native = device_table(grid.wave_grid)
    sed = resample_template(wavelength, wave_native, template, left=0.0, right=0.0)
    return scale_to_torus_power_native(template, wave_native, sed, agn_log_lbol, agn_torus_frac)
