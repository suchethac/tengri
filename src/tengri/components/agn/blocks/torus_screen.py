# SPDX-License-Identifier: BSD-3-Clause
r"""Inclination-dependent torus screen on the AGN central engine.

The composable AGN runner sums disc + lines + FeII + torus. Physically the
dusty torus also *obscures* the central engine (disc + broad/narrow lines) along
edge-on (Type-2) sightlines, while its own IR emission is not re-extinguished by
that same screen. This module supplies the screen so the runner can apply it to
the central-engine components only: closing the "disc + torus composed
additively, no torus screen on disc" gap (#294).

Geometry (Stalevski+2016 / CIGALE ``skirtor2016`` convention): the torus has a
half-opening angle ``oa`` measured from the equatorial plane and the inclination
``i`` is measured from the polar axis. The sightline grazes/enters the torus
(Type 2) when :math:`i > 90^\circ - {\rm oa}`, i.e.
:math:`\cos i < \sin({\rm oa})`. The transition is smoothed with a sigmoid so the
screen is C¹ in ``cos_inc`` (gradient-safe for inference); face-on (Type 1)
sightlines get unit transmission, so a default-inclination model is unchanged.

The wavelength dependence uses the torus equatorial V-band optical depth
``tau_v`` and an SMC (default) or Calzetti reddening curve, normalized at V
(5500 Å): :math:`\tau(\lambda) = \tau_V\,k(\lambda)/k(V)`.
"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.components.agn.polar_dust import (
    calzetti2000_extinction_curve,
    smc_extinction_curve,
)

# Torus blocks that represent a genuine dusty torus with an equatorial optical
# depth + opening angle, mapped to the (opening-angle, V-band-tau) param names
# the screen reads from the runner's param dict. Blocks not listed here (toy
# two-temperature, GRAHSP) get no torus screen.
TORUS_SCREEN_PARAMS: dict[str, tuple[str, str]] = {
    "skirtor": ("agn_oa_skirtor", "agn_tau_skirtor"),
    "fritz": ("agn_fritz_oa", "agn_fritz_tau"),
}

# Smoothing width in cos(i) units for the Type-1/2 edge. Kept small so a
# clearly face-on sightline (cos_inc well above sin(oa)) has transmission ~1 to
# <1e-3 even at large tau_v: i.e. default-inclination models are unchanged :
# while the screen still has a finite, differentiable slope across the edge.
_TRANSITION_WIDTH = 0.025


def torus_screen_transmission(
    wavelength: jnp.ndarray,
    cos_inc: float,
    oa_deg: float,
    tau_v: float,
    law: str = "smc",
) -> jnp.ndarray:
    r"""Transmission of the torus screen seen by the central engine.

    Parameters
    ----------
    wavelength: array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    cos_inc: float
        Cosine of the inclination (angle from the polar axis); 1 = face-on.
        [dimensionless]
    oa_deg: float
        Torus half-opening angle from the equatorial plane. [deg]
    tau_v: float
        Torus equatorial V-band optical depth (the disc seen through the torus
        rim is reddened by ~this depth). [dimensionless]
    law: str
        Reddening curve: ``"smc"`` (default) or ``"calzetti"``.

    Returns
    -------
    ndarray, shape (n_wave,)
        Multiplicative transmission in [0, 1]; identically ~1 for face-on
        (Type-1) sightlines, dropping toward edge-on (Type-2).

    Notes
    -----
    **JIT-compatible**: yes, pure ``jnp`` primitives.

    **Gradient-safe**: yes, the Type-1/Type-2 edge is a sigmoid in ``cos_inc``,
    so the screen is differentiable everywhere (no hard ``where`` step).

    The screen multiplies only the central-engine components (disc + lines +
    FeII); the torus IR emission is *not* screened by it.
    """
    wave = jnp.asarray(wavelength)
    if law == "calzetti":
        k_lambda = calzetti2000_extinction_curve(wave)
        k_v = calzetti2000_extinction_curve(jnp.array([5500.0]))[0]
    else:
        k_lambda = smc_extinction_curve(wave)
        k_v = smc_extinction_curve(jnp.array([5500.0]))[0]

    # Type-2 weight: 1 when edge-on (cos_inc < sin(oa)), 0 when face-on.
    sin_oa = jnp.sin(jnp.deg2rad(oa_deg))
    type2 = jax_sigmoid((sin_oa - cos_inc) / _TRANSITION_WIDTH)

    tau_lambda = jnp.maximum(tau_v, 0.0) * (k_lambda / jnp.maximum(k_v, 1e-30)) * type2
    return jnp.exp(-jnp.clip(tau_lambda, 0.0, 50.0))


def jax_sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """Numerically-stable logistic sigmoid."""
    return 0.5 * (1.0 + jnp.tanh(0.5 * x))
