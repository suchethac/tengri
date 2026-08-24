# SPDX-License-Identifier: BSD-3-Clause
"""Pure JAX magnitude system utilities (AB, Vega, absolute, apparent, surface brightness).

Provides JIT-compatible functions for common magnitude conversions in observational
astronomy. All functions are pure, immutable, and GPU-compatible.

Conventions
-----------

- Magnitude system: AB (Oke & Gunn 1983) by default
- Flux unit: erg s⁻¹ cm⁻² Hz⁻¹ (CGS)
- Luminosity unit: erg s⁻¹ Hz⁻¹ (CGS)
- Distance: cm (CGS) unless otherwise specified
- Wavelength: Ångström

References
----------
Oke & Gunn 1983, ApJ, 266, 713: AB magnitude zeropoint
Blanton & Roweis 2007: Vega offsets (Table 3, 5)
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from tengri.utils.physics_constants import (
    MAGGIES_ZP_CGS,
    MPC_CM,
    TEN_PC_CM,
)
from tengri.utils.scale import pow10, representable_floor

#: ``log10`` of the AB zero-point flux [dex re erg/s/cm2/Hz] and of the 10 pc
#: geometric factor [dex re cm2]. Both are folded into the exponent rather than
#: applied as multiplicative constants: ``4π (10 pc)^2 = 1.196e40`` and
#: ``f_ν / 3.631e-20`` for a UV-bright galaxy (~2e47) each exceed float32's
#: 3.4e38 ceiling, even though the magnitudes on both sides of the conversion
#: are of order 1-100 and perfectly representable (#1837).
#:
#: ``TEN_PC_CM**2 = 9.52e38`` overflows float32 *at the square*, one step before
#: the expression it appears in: so the linear form cannot be rescued by
#: reordering the multiplication.
LOG10_MAGGIES_ZP_CGS: float = math.log10(MAGGIES_ZP_CGS)
LOG10_FOUR_PI_TEN_PC_CM2: float = math.log10(4.0 * math.pi * TEN_PC_CM**2)

__all__ = [
    "AB_VEGA_OFFSETS",
    "ab_mag_to_fnu",
    "ab_to_vega",
    "absolute_ab_mag_to_lnu",
    "absolute_to_apparent",
    "apparent_to_absolute",
    "cosmological_dimming",
    "distance_modulus_from_dl",
    "distance_modulus_from_dl_mpc",
    "fnu_to_ab_mag",
    "lnu_to_absolute_ab_mag",
    "mag_to_surface_brightness",
    "surface_brightness_to_mag",
    "vega_to_ab",
]


# ── AB Magnitude System ───────────────────────────────────────────


@jax.jit
def fnu_to_ab_mag(fnu_cgs: jnp.ndarray) -> jnp.ndarray:
    r"""Convert flux density to AB magnitude.

    Computes the AB magnitude from a flux density using the relation:

    .. math::

        m_{\mathrm{AB}} = -2.5 \log_{10}(f_\nu / f_0)

    where f_ν is in erg s⁻¹ cm⁻² Hz⁻¹ and f_0 = 3.631e-20 erg s⁻¹ cm⁻² Hz⁻¹
    is the AB zeropoint flux.

    The AB system is defined so that m_AB = 0 corresponds to f_ν = 3631 Jy.

    Parameters
    ----------
    fnu_cgs: jnp.ndarray
        Flux density in erg s⁻¹ cm⁻² Hz⁻¹. Shape: arbitrary.

    Returns
    -------
    jnp.ndarray
        AB magnitude. Same shape as input. For fnu → 0, result → ∞.

    Notes
    -----
    **JIT/grad/vmap-compatible**: yes.

    Guards the logarithm against negative or zero values (which would produce
    NaN or -inf) with :func:`~tengri.utils.scale.representable_floor`. The
    literal floor was ``1e-300``, which is below float32's smallest subnormal
    (1.4e-45) and therefore an exact no-op in the precision the guard exists to
    protect (#1492).

    The zero-point is subtracted in the log domain rather than divided out.
    ``f_ν / 3.631e-20`` reaches ~2e47 for the L_ν values
    :func:`~tengri.utils.sed_quantities.compute_rest_uv_color` passes in, which
    overflows float32 even though the resulting magnitude is ~-118 (#1837).
    float64 is unchanged: ``log10(a/b)`` and ``log10(a) - log10(b)`` agree to
    ~1e-16 relative.

    References
    ----------
    Oke & Gunn 1983, ApJ, 266, 713: definition of AB system
    """
    fnu_safe = jnp.maximum(fnu_cgs, representable_floor(1e-300))
    return -2.5 * (jnp.log10(fnu_safe) - LOG10_MAGGIES_ZP_CGS)


@jax.jit
def ab_mag_to_fnu(mag_ab: jnp.ndarray) -> jnp.ndarray:
    r"""Convert AB magnitude to flux density.

    Inverts :func:`fnu_to_ab_mag`:

    .. math::

        f_\nu = f_0 \times 10^{-0.4 m_{\mathrm{AB}}}

    where f_0 = 3.631e-20 erg s⁻¹ cm⁻² Hz⁻¹ is the AB zeropoint flux.

    Parameters
    ----------
    mag_ab: jnp.ndarray
        AB magnitude. Shape: arbitrary.

    Returns
    -------
    jnp.ndarray
        Flux density in erg s⁻¹ cm⁻² Hz⁻¹. Same shape as input.

    Examples
    --------
    A zero-magnitude AB source:

    >>> fnu = ab_mag_to_fnu(jnp.array(0.0))
    >>> jnp.allclose(fnu, 3.631e-20)
    True

    Notes
    -----
    **JIT/grad/vmap-compatible**: yes. The zero-point is added in the exponent
    rather than multiplied in: for the very negative magnitudes that arise when
    a luminosity is passed through the flux surface, ``10**(-0.4 m)`` alone
    overflows float32 before the ~1e-20 zero-point can bring it back into
    range (#1837).

    References
    ----------
    Oke & Gunn 1983, ApJ, 266, 713
    """
    return pow10(-0.4 * mag_ab + LOG10_MAGGIES_ZP_CGS)


@jax.jit
def lnu_to_absolute_ab_mag(lnu: jnp.ndarray) -> jnp.ndarray:
    r"""Convert monochromatic luminosity to absolute AB magnitude.

    Computes the absolute magnitude at 10 pc by calculating the flux density
    at that distance and converting to AB magnitude.

    .. math::

        m_{\mathrm{AB}}(10 \text{ pc}) = -2.5 \log_{10}
        \left( \frac{L_\nu}{4\pi (10 \text{ pc})^2} \right) - 48.6

    Parameters
    ----------
    lnu: jnp.ndarray
        Monochromatic luminosity in erg s⁻¹ Hz⁻¹. Shape: arbitrary.

    Returns
    -------
    jnp.ndarray
        Absolute AB magnitude. Same shape as input.

    Notes
    -----
    **JIT/grad/vmap-compatible**: yes.

    The distance 10 pc is defined in `tengri.utils.physics_constants.TEN_PC_CM`.
    The geometric factor is subtracted in the log domain: ``4π (10 pc)^2`` is
    1.196e40 and its float32 evaluation is ``inf``; already at ``TEN_PC_CM**2``: so the linear
    form returned ``inf`` for every input, despite M_UV being
    ~-17 (#1837). Clamping in the log domain is exactly equivalent to the
    previous linear clamp because ``log10`` is monotone:
    ``log10(max(x, f)) == max(log10(x), log10(f))``.

    References
    ----------
    Oke & Gunn 1983, ApJ, 266, 713
    """
    # f_ν = L_ν / (4π d²), evaluated as log10 f_ν = log10 L_ν - log10(4π d²).
    log_fnu_at_10pc = jnp.maximum(
        jnp.log10(jnp.maximum(lnu, 0.0)) - LOG10_FOUR_PI_TEN_PC_CM2,
        math.log10(representable_floor(1e-300)),
    )
    return -2.5 * (log_fnu_at_10pc - LOG10_MAGGIES_ZP_CGS)


@jax.jit
def absolute_ab_mag_to_lnu(mag_abs: jnp.ndarray) -> jnp.ndarray:
    r"""Convert absolute AB magnitude to monochromatic luminosity.

    Inverts :func:`lnu_to_absolute_ab_mag`.

    Parameters
    ----------
    mag_abs: jnp.ndarray
        Absolute AB magnitude. Shape: arbitrary.

    Returns
    -------
    jnp.ndarray
        Monochromatic luminosity in erg s⁻¹ Hz⁻¹. Same shape as input.

    Notes
    -----
    **JIT/grad/vmap-compatible**: yes.

    The distance 10 pc is defined in `tengri.utils.physics_constants.TEN_PC_CM`.
    The geometric factor is folded into the exponent for the reason given in
    :func:`lnu_to_absolute_ab_mag`: ``4π (10 pc)^2`` is not representable in
    float32, so the multiplicative form returned ``inf`` for every input (#1837).

    References
    ----------
    Oke & Gunn 1983, ApJ, 266, 713
    """
    # L_ν = f_ν × 4π d², with f_ν = 10^(-0.4(M + 48.6)): one exponent, so no
    # intermediate leaves float32 range.
    return pow10(-0.4 * mag_abs + LOG10_MAGGIES_ZP_CGS + LOG10_FOUR_PI_TEN_PC_CM2)


# ── Apparent ↔ Absolute Magnitude ─────────────────────────────────


@jax.jit
def apparent_to_absolute(m_app: jnp.ndarray, dist_modulus: jnp.ndarray) -> jnp.ndarray:
    r"""Convert apparent to absolute magnitude via distance modulus.

    .. math::

        M = m - \mu

    Parameters
    ----------
    m_app: jnp.ndarray
        Apparent magnitude. Shape: arbitrary.
    dist_modulus: jnp.ndarray
        Distance modulus μ. Same shape as m_app, or broadcastable.

    Returns
    -------
    jnp.ndarray
        Absolute magnitude. Shape: broadcast of m_app and dist_modulus.

    Notes
    -----
    The distance modulus encodes both luminosity distance and any
    cosmological dimming effects.
    """
    return m_app - dist_modulus


@jax.jit
def absolute_to_apparent(m_abs: jnp.ndarray, dist_modulus: jnp.ndarray) -> jnp.ndarray:
    r"""Convert absolute to apparent magnitude via distance modulus.

    .. math::

        m = M + \mu

    Parameters
    ----------
    m_abs: jnp.ndarray
        Absolute magnitude. Shape: arbitrary.
    dist_modulus: jnp.ndarray
        Distance modulus μ. Same shape as m_abs, or broadcastable.

    Returns
    -------
    jnp.ndarray
        Apparent magnitude. Shape: broadcast of m_abs and dist_modulus.
    """
    return m_abs + dist_modulus


@jax.jit
def distance_modulus_from_dl(dl_cm: jnp.ndarray) -> jnp.ndarray:
    r"""Compute distance modulus from luminosity distance.

    .. math::

        \mu = 5 \log_{10}(d_L / 10 \text{ pc})

    Parameters
    ----------
    dl_cm: jnp.ndarray
        Luminosity distance in cm. Shape: arbitrary.

    Returns
    -------
    jnp.ndarray
        Distance modulus in magnitudes. Same shape as input.

    Notes
    -----
    In a Euclidean geometry (z → 0), μ = 5 log10(d/10 pc).
    At nonzero redshift, d_L is the cosmological luminosity distance
    (as provided by DSPS or similar).

    The reference distance 10 pc is defined in
    `tengri.utils.physics_constants.TEN_PC_CM`.

    For distance → 0, returns -∞ (as expected mathematically).
    """
    # Don't guard against zero; let it naturally produce -inf at zero distance
    return 5.0 * jnp.log10(dl_cm / TEN_PC_CM)


@jax.jit
def distance_modulus_from_dl_mpc(dl_mpc: jnp.ndarray) -> jnp.ndarray:
    r"""Compute distance modulus from luminosity distance in Mpc.

    Convenience wrapper that converts input from Mpc to cm, then calls
    :func:`distance_modulus_from_dl`.

    Parameters
    ----------
    dl_mpc: jnp.ndarray
        Luminosity distance in Mpc. Shape: arbitrary.

    Returns
    -------
    jnp.ndarray
        Distance modulus in magnitudes. Same shape as input.

    Notes
    -----
    DSPS and most cosmological codes return distances in Mpc.
    The conversion factor (1 Mpc = 3.0856...e24 cm) is defined in
    `tengri.utils.physics_constants.MPC_CM`.
    """
    dl_cm = dl_mpc * MPC_CM
    return distance_modulus_from_dl(dl_cm)


# ── Cosmological Dimming ──────────────────────────────────────────


@jax.jit
def cosmological_dimming(dist_modulus: jnp.ndarray, redshift: jnp.ndarray) -> jnp.ndarray:
    r"""Account for cosmological dimming (bandwidth compression) in high-z observations.

    The observed-frame magnitude of a high-redshift source differs from the
    rest-frame absolute magnitude by both the luminosity distance (in dist_modulus)
    and the (1+z) dimming factor:

    .. math::

        m_{\mathrm{obs}} = M_{\mathrm{rest}} + \mu - 2.5 \log_{10}(1 + z)

    This function computes the effective distance modulus after accounting for
    bandwidth compression.

    Parameters
    ----------
    dist_modulus: jnp.ndarray
        Distance modulus μ. Shape: arbitrary.
    redshift: jnp.ndarray
        Redshift z. Same shape as dist_modulus, or broadcastable.

    Returns
    -------
    jnp.ndarray
        Effective distance modulus including (1+z) dimming. Shape: broadcast.

    Notes
    -----
    This is the DSPS convention for K-correction. See Hogg et al. 1998,
    ApJ, 504, 788 for discussion of the (1+z) factor.

    The (1+z) term accounts for the fact that the observed filter integrates
    over frequencies (not wavelengths), and observed frequencies are redshifted
    by (1+z) compared to rest-frame frequencies.
    """
    dimming = 2.5 * jnp.log10(1.0 + redshift)
    return dist_modulus - dimming


# ── Vega Magnitude System ─────────────────────────────────────────

AB_VEGA_OFFSETS: dict[str, float] = {
    "U": 0.79,
    "B": -0.09,
    "V": 0.02,
    "R": 0.21,
    "I": 0.45,
    "J": 0.91,
    "H": 1.39,
    "K": 1.85,
    "u": 0.91,
    "g": -0.08,
    "r": 0.16,
    "i": 0.37,
    "z": 0.54,
}
"""Vega-to-AB magnitude offsets for common filters.

These are the zero-magnitude offsets: mag_Vega = mag_AB - offset.

Keys are Johnson/Bessel and SDSS **short** band names (``"V"``, ``"r"``),
not filter-registry ids (``"sdss_r"``). Passing a registry id raises
``KeyError``; a filter-registry lookup for the offset does not exist.

Available as ``tengri.units.AB_VEGA_OFFSETS`` as well as from
``tengri.utils``: it is the only argument ``ab_to_vega`` and ``vega_to_ab``
take besides the magnitude, so it lives wherever they do (#1613).

Sources
-------
Blanton & Roweis 2007, AJ, 133, 734: Table 3, 5 (SDSS and Johnson/Bessel).
"""


@jax.jit
def ab_to_vega(mag_ab: jnp.ndarray, ab_vega_offset: float) -> jnp.ndarray:
    r"""Convert AB magnitude to Vega magnitude.

    .. math::

        m_{\mathrm{Vega}} = m_{\mathrm{AB}} - \text{offset}

    Parameters
    ----------
    mag_ab: jnp.ndarray
        AB magnitude. Shape: arbitrary.
    ab_vega_offset: float
        Zero-magnitude offset for this band, e.g.
        ``tengri.units.AB_VEGA_OFFSETS["V"]``. A float, not a band name.

    Returns
    -------
    jnp.ndarray
        Vega magnitude. Same shape as input.

    Examples
    --------
    Convert a V-band AB magnitude to Vega:

    >>> mag_ab = jnp.array(20.0)
    >>> mag_vega = ab_to_vega(mag_ab, AB_VEGA_OFFSETS["V"])
    >>> mag_vega
    Array(19.98, dtype=float32)

    References
    ----------
    Blanton & Roweis 2007, AJ, 133, 734
    """
    return mag_ab - ab_vega_offset


@jax.jit
def vega_to_ab(mag_vega: jnp.ndarray, ab_vega_offset: float) -> jnp.ndarray:
    r"""Convert Vega magnitude to AB magnitude.

    .. math::

        m_{\mathrm{AB}} = m_{\mathrm{Vega}} + \text{offset}

    Parameters
    ----------
    mag_vega: jnp.ndarray
        Vega magnitude. Shape: arbitrary.
    ab_vega_offset: float
        Zero-magnitude offset for this band, e.g.
        ``tengri.units.AB_VEGA_OFFSETS["V"]``. A float, not a band name.

    Returns
    -------
    jnp.ndarray
        AB magnitude. Same shape as input.

    References
    ----------
    Blanton & Roweis 2007, AJ, 133, 734
    """
    return mag_vega + ab_vega_offset


# ── Surface Brightness ────────────────────────────────────────────


@jax.jit
def mag_to_surface_brightness(mag: jnp.ndarray, area_arcsec2: jnp.ndarray) -> jnp.ndarray:
    r"""Convert total magnitude to surface brightness.

    Surface brightness (magnitude per unit area) is derived from the total
    magnitude by diluting the flux over the object's area:

    .. math::

        \mu = m + 2.5 \log_{10}(A)

    where A is the area in arcsec².

    Parameters
    ----------
    mag: jnp.ndarray
        Total magnitude. Shape: arbitrary.
    area_arcsec2: jnp.ndarray
        Area in arcsec². Same shape as mag, or broadcastable.

    Returns
    -------
    jnp.ndarray
        Surface brightness in mag/arcsec². Shape: broadcast.

    Notes
    -----
    A 1 arcsec² source has μ = mag (no dilution).
    A larger source (e.g., 10 arcsec²) is dimmer per arcsec².

    References
    ----------
    Binney & Merrifield 1998, Galactic Astronomy (Section 2.1)
    """
    area_safe = jnp.maximum(area_arcsec2, 1e-300)
    return mag + 2.5 * jnp.log10(area_safe)


@jax.jit
def surface_brightness_to_mag(mu: jnp.ndarray, area_arcsec2: jnp.ndarray) -> jnp.ndarray:
    r"""Convert surface brightness to total magnitude.

    Inverts :func:`mag_to_surface_brightness`.

    Parameters
    ----------
    mu: jnp.ndarray
        Surface brightness in mag/arcsec². Shape: arbitrary.
    area_arcsec2: jnp.ndarray
        Area in arcsec². Same shape as mu, or broadcastable.

    Returns
    -------
    jnp.ndarray
        Total magnitude. Shape: broadcast.

    References
    ----------
    Binney & Merrifield 1998, Galactic Astronomy (Section 2.1)
    """
    area_safe = jnp.maximum(area_arcsec2, 1e-300)
    return mu - 2.5 * jnp.log10(area_safe)
