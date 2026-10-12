# SPDX-License-Identifier: BSD-3-Clause
"""Publication grid for AGN published luminosities (#2745).

A published bolometric luminosity is the integral of the SED the component emits.
Integrating on the caller's ``wave`` makes the value depend on how the caller samples
wavelength, so the AGN classes evaluate the emitted SED on this fixed grid instead and
integrate there. The grid is log-uniform in wavelength over the whole emission range of
the AGN templates and disc models, and contains any diagnostic wavelength passed in
``extra_aa`` exactly, so a node value is the emitted value at that wavelength.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.agn._lbol_reference import reference_evaluation, rescale
from tengri.components.agn._phys import wavelength_to_nu
from tengri.utils.grid_interp import loglog_integral
from tengri.utils.physics_constants import C_AA, L_SUN
from tengri.utils.scale import log10_magnitude, pow10

__all__ = [
    "INCLINATION_GL_ORDER",
    "PUBLICATION_HI_AA",
    "PUBLICATION_LO_AA",
    "PUBLICATION_NODES",
    "emitted_bolometric",
    "inclination_edges",
    "inclination_quadrature",
    "log10_emitted_bolometric",
    "log10_native_bolometric",
    "publication_wave",
    "template_sed_and_log_power",
    "with_lsun_outputs",
    "with_lsun_siblings",
]

#: Lower edge of the publication grid [Angstrom] (X-ray corona of KD18 reaches below 10 A).
PUBLICATION_LO_AA: float = 1.0e-3

#: Upper edge of the publication grid [Angstrom] (far infrared, 1e10 A = 1 m).
PUBLICATION_HI_AA: float = 1.0e10

#: Number of log-uniform nodes. The trapezoid in ln(nu) of the emitted SED agrees with a
#: 200000-node reference to better than 3e-7 for every AGN template class at this count;
#: at 4000 nodes the template classes miss 1e-6.
PUBLICATION_NODES: int = 8000


#: ``log10(L_sun / (erg/s))`` [dex].
_LOG10_L_SUN: float = float(np.log10(L_SUN))


def lsun_sibling_key(log_key: str) -> str:
    """Name of the ``Lsun`` sibling of a ``log_L_agn_<piece>`` key: ``L_agn_<piece>_lsun``.

    Parameters
    ----------
    log_key : str
        A published key of the form ``log_L_agn_<piece>``.

    Returns
    -------
    str
        ``L_agn_<piece>_lsun``.
    """
    if not log_key.startswith("log_L_agn_"):
        raise ValueError(f"not a log_L_agn_<piece> key: {log_key!r}")
    return f"{log_key[len('log_') :]}_lsun"


def with_lsun_siblings(published: dict) -> dict:
    """``published`` plus an ``L_agn_<piece>_lsun`` [Lsun] sibling of every ``log_L_agn_<piece>``.

    An AGN power of 1e8 to 1e14 Lsun is representable in float32 (NAMING_CONTRACT 4c), so the
    solar-luminosity value is published beside the exponent. It is formed from the log as
    ``pow10(log - log10(L_sun))``: the erg/s value (~1e44, ``inf`` in float32) is never formed,
    and an absent component (``-inf`` dex) is exactly 0.

    Parameters
    ----------
    published : dict
        A component's published values.

    Returns
    -------
    dict
        A new dict with the siblings added; the input is unchanged.

    Notes
    -----
    **JIT/grad/vmap-safe**: yes.
    """
    siblings = {
        lsun_sibling_key(k): pow10(jnp.asarray(v) - _LOG10_L_SUN)
        for k, v in published.items()
        if k.startswith("log_L_agn_")
    }
    return {**published, **siblings}


def with_lsun_outputs(outputs: dict) -> dict:
    """``outputs`` plus the ``Lsun`` declaration of every ``log_L_agn_<piece>`` sibling.

    Parameters
    ----------
    outputs : dict
        Declared outputs, ``{key: units}``.

    Returns
    -------
    dict
        A new dict with ``{L_agn_<piece>_lsun: "Lsun"}`` added.
    """
    siblings = {lsun_sibling_key(k): "Lsun" for k in outputs if k.startswith("log_L_agn_")}
    return {**outputs, **siblings}


def publication_wave(dtype, extra_aa=(), n_nodes: int = PUBLICATION_NODES) -> jnp.ndarray:
    """Fixed publication wavelength grid, ascending, with diagnostic nodes included.

    Parameters
    ----------
    dtype : dtype
        Floating dtype of the returned array (the caller's wavelength dtype).
    extra_aa : sequence of float, optional
        Diagnostic wavelengths [Angstrom] that must be grid nodes exactly.
    n_nodes : int, optional
        Number of log-uniform nodes; the coarsest count that meets the identity tolerance.

    Returns
    -------
    ndarray, shape (n_pub,)
        Ascending wavelengths [Angstrom], independent of any caller grid.
    """
    nodes = np.geomspace(PUBLICATION_LO_AA, PUBLICATION_HI_AA, n_nodes)
    nodes = np.unique(np.concatenate([nodes, np.asarray(extra_aa, dtype=np.float64)]))
    return jnp.asarray(nodes, dtype=dtype)


def emitted_bolometric(sed_pub, wave_pub) -> jnp.ndarray:
    r"""Bolometric luminosity of an emitted SED on the publication grid.

    .. math::

        L = \int L_\nu \, \mathrm{d}\nu = \int L_\nu(\nu)\,\nu \,\mathrm{d}\ln\nu

    evaluated by the trapezoid rule in :math:`\ln\nu` on the nodes of ``wave_pub``.

    Parameters
    ----------
    sed_pub : array_like, shape (n_pub,)
        Emitted spectral luminosity density on ``wave_pub`` [erg/s/Hz].
    wave_pub : array_like, shape (n_pub,)
        Ascending wavelengths of the publication grid [Angstrom].

    Returns
    -------
    ndarray, shape ()
        Integrated luminosity over the whole grid [erg/s].

    Notes
    -----
    **JIT/grad-safe**: yes. ``nu`` is descending on an ascending wavelength grid, so the
    trapezoid is negated, following the repository's convention.
    """
    nu = wavelength_to_nu(wave_pub)
    return -jnp.trapezoid(jnp.asarray(sed_pub) * nu, jnp.log(nu))


def log10_emitted_bolometric(sed_pub, wave_pub) -> jnp.ndarray:
    r"""``log10`` of :func:`emitted_bolometric`, formed without an erg/s-scale product.

    Parameters
    ----------
    sed_pub : array_like, shape (n_pub,)
        Emitted spectral luminosity density on ``wave_pub`` [erg/s/Hz].
    wave_pub : array_like, shape (n_pub,)
        Ascending wavelengths of the publication grid [Angstrom].

    Returns
    -------
    ndarray, shape ()
        :math:`\log_{10}(L/\mathrm{erg\,s^{-1}})` [dex]; ``-inf`` for an exactly zero SED
        (an absent component), ``+inf`` where the integral is not finite.

    Notes
    -----
    **JIT/grad/vmap-safe**: yes. The SED is divided by its peak (held under
    ``stop_gradient``: the value does not depend on it) before the trapezoid in
    :math:`\ln\nu`, so the integrand is bounded by :math:`\nu` (at most ~1e22 Hz) and the
    ~1e44 erg/s total is never formed; the peak's decades are added back as a logarithm. In
    float64 this equals ``log10(emitted_bolometric(...))`` to rounding. Written for pure
    float32, where a luminosity above 3.4e38 erg/s is ``inf`` (NAMING_CONTRACT section 4c).
    """
    nu = wavelength_to_nu(wave_pub)
    sed = jnp.asarray(sed_pub)
    peak = jax.lax.stop_gradient(jnp.max(jnp.abs(sed), initial=0.0))
    safe_peak = jnp.where(peak > 0, peak, jnp.ones_like(peak))
    total = -jnp.trapezoid(sed / safe_peak * nu, jnp.log(nu))
    return log10_magnitude(total) + jnp.log10(safe_peak)


def log10_native_bolometric(lnu_native, wave_native) -> jnp.ndarray:
    r"""``log10`` of the frequency integral of a template on its own wavelength grid.

    The log-space twin of ``native_bolometric_nu`` of ``_template_grid``:
    the same integral of the interpolant (a power law in :math:`\lambda` between nodes), with
    the SED peak factored out so no erg/s-scale value forms.

    Parameters
    ----------
    lnu_native : array_like, shape (n_native,)
        Emitted :math:`L_\nu` on ``wave_native`` [erg/s/Hz].
    wave_native : array_like, shape (n_native,)
        The template's wavelength nodes [Angstrom], ascending.

    Returns
    -------
    ndarray, shape ()
        :math:`\log_{10}(L/\mathrm{erg\,s^{-1}})` [dex]; ``-inf`` for an exactly zero SED.

    Notes
    -----
    **JIT/grad/vmap-safe**: yes; see :func:`log10_emitted_bolometric` for the peak factoring.
    """
    wave = jnp.asarray(wave_native)
    lnu = jnp.asarray(lnu_native)
    peak = jax.lax.stop_gradient(jnp.max(jnp.abs(lnu), initial=0.0))
    safe_peak = jnp.where(peak > 0, peak, jnp.ones_like(peak))
    total = jnp.abs(loglog_integral(wave, lnu / safe_peak * (C_AA / wave**2)))
    return log10_magnitude(total) + jnp.log10(safe_peak)


#: Gauss-Legendre points per interval between the template's inclination nodes. Measured on
#: the SKIRTOR v3 grid against an 801-point Simpson reference of the same interpolant (the
#: reference agrees with its own trapezoid to 1e-7): worst relative error over five parameter
#: sets (oa 10 to 80 deg, tau 3 to 11, delta -0.8 to 0.5, polar E(B-V) 0 to 0.3) is 7e-5 for
#: the disc, 2e-8 for the torus and 1e-3 for the polar-dust power; the trapezoid on the nodes
#: gives 1e-3, 2e-5 and 3e-2, two points per interval 2e-4, 4e-7 and 2e-3.
INCLINATION_GL_ORDER: int = 12


def inclination_edges(cos_inc_nodes) -> np.ndarray:
    r"""Interval edges in :math:`\cos i` that the inclination quadrature panels span.

    Parameters
    ----------
    cos_inc_nodes : array_like, shape (n_inc,)
        The template's :math:`\cos i` nodes [dimensionless], ascending.

    Returns
    -------
    ndarray, shape (n_edge,)
        ``[0, nodes[1:]]`` when the first node sits at ~0 (the edge-on end is then extended
        to exactly 0), else the nodes themselves.
    """
    nodes = np.asarray(cos_inc_nodes, dtype=np.float64)
    return np.concatenate([[0.0], nodes[1:]]) if nodes[0] < 1e-9 else nodes


def inclination_quadrature(cos_inc_nodes, order: int = INCLINATION_GL_ORDER):
    r"""Quadrature points and weights for :math:`\int_0^1 f(\mu)\,d\mu` on a template's nodes.

    Gauss-Legendre of ``order`` points on each interval between consecutive inclination nodes,
    so every panel is a smooth stretch of the template's interpolant.

    Parameters
    ----------
    cos_inc_nodes : array_like, shape (n_inc,)
        The template's :math:`\cos i` nodes [dimensionless], ascending, the first at ~0 and
        the last at 1.
    order : int, optional
        Points per interval.

    Returns
    -------
    mu : ndarray, shape (n_points,)
        :math:`\cos i` at the quadrature points [dimensionless], within ``[0, 1]``.
    weight : ndarray, shape (n_points,)
        Weights [dimensionless], summing to 1.

    Notes
    -----
    A host (numpy) constant: the nodes are static, so this is safe under ``jit``.

    For an emitter with luminosity :math:`L(\theta)` per solid angle, the angle-integrated
    power divided by :math:`4\pi` is :math:`\tfrac12\int_0^\pi L(\theta)\sin\theta\,d\theta
    = \int_0^1 L(\mu)\,d\mu` by symmetry about the equator, the quantity this weights.
    """
    edges = inclination_edges(cos_inc_nodes)
    x, w = np.polynomial.legendre.leggauss(order)
    lo, hi = edges[:-1, None], edges[1:, None]
    mu = (0.5 * (hi - lo) * x + 0.5 * (hi + lo)).ravel()
    weight = (0.5 * (hi - lo) * w).ravel()
    return mu, weight / weight.sum()


def template_sed_and_log_power(shape_fn, native_wave, wave, agn_log_lbol):
    r"""Emitted SED of a template component and its published ``log10`` bolometric power.

    The template classes scale a unit-shape template to ``10**agn_log_lbol L_sun`` of power. In
    pure float32 that product (~1e44 erg/s) is ``inf``, so the template is evaluated at the
    reference luminosity of ``reference_evaluation`` of ``_lbol_reference``
    and the true magnitude is restored in log space. In float64 the reference is the true
    luminosity and the SED is returned untouched.

    Parameters
    ----------
    shape_fn : callable
        ``shape_fn(wave_eval, agn_log_lbol_eval) -> L_nu`` [erg/s/Hz], linear in
        ``10**agn_log_lbol_eval``.
    native_wave : array_like, shape (n_native,)
        The template's own wavelength nodes [Angstrom], where its interpolant is a power law
        between nodes, so the integral there is closed form.
    wave : array_like, shape (n_wave,)
        The caller's rest-frame grid [Angstrom]; only its dtype and shape enter the SED.
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)` [dex].

    Returns
    -------
    sed : ndarray, shape (n_wave,)
        Emitted :math:`L_\nu` on ``wave`` [erg/s/Hz].
    log_power : ndarray, shape ()
        :math:`\log_{10}` of its frequency integral [dex re erg/s].

    Notes
    -----
    **JIT/grad/vmap-safe**: yes.
    """
    lbol_eval, use_ref, offset = reference_evaluation(agn_log_lbol, wave)
    native = jnp.asarray(native_wave, dtype=wave.dtype)
    sed = shape_fn(wave, lbol_eval)
    log_power = log10_native_bolometric(shape_fn(native, lbol_eval), native) + offset
    return (rescale(sed, offset) if use_ref else sed), log_power
