# SPDX-License-Identifier: BSD-3-Clause
r"""Inclination-dependent torus screen on the AGN central engine.

The composable AGN runner sums disc + lines + FeII + torus. Physically the
dusty torus also *obscures* the central engine (disc + broad/narrow lines) along
edge-on (Type-2) sightlines, while its own IR emission is not re-extinguished by
that same screen. This module supplies the screen so the runner can apply it to
the central-engine components only: closing the "disc + torus composed
additively, no torus screen on disc" gap (#294).

Geometry. ``agn_cos_inc`` is the one inclination of every torus, measured from
the polar axis. The torus has a half-opening angle ``oa`` measured from the
equatorial plane (Stalevski+2016 / CIGALE ``skirtor2016``), so the dust-free
polar cone has half-angle :math:`90^\circ - {\rm oa}`; the sightline reaches
the nucleus directly (Type 1) when it lies inside it,
:math:`i < 90^\circ - {\rm oa}`, i.e. :math:`\cos i > \sin({\rm oa})`. The
Fritz et al. (2006) library keys its torus by that cone's half-angle
:math:`\theta_c` (``agn_fritz_oa``, full opening :math:`\Theta = 180^\circ -
2\theta_c`), so for it :math:`{\rm oa} = 90^\circ - \theta_c` and Type 1 is
:math:`i < \theta_c`; the library's own viewing elevation
:math:`\psi = 90^\circ - i` is derived from the same inclination. The
transition is the sigmoid of the polar-dust Type-1 mask (``polar_dust._type1_mask``),
at the one width :data:`~tengri.components.agn.polar_dust.TYPE_TRANSITION_WIDTH`
the polar mask uses, so the screen is C¹ in ``cos_inc`` (gradient-safe for
inference); face-on (Type 1) sightlines get unit transmission, so a
default-inclination model is unchanged.

The wavelength dependence uses the torus equatorial V-band optical depth
``tau_v`` and an SMC (default) or Calzetti reddening curve, normalized at V
(5500 Å): :math:`\tau(\lambda) = \tau_V\,k(\lambda)/k(V)`. The Fritz library
axis is the optical depth at 9.7 µm, not at V; it is converted with
:data:`FRITZ_TAU_V_PER_TAU_97`.

References
----------
.. [1] J. Fritz, A. Franceschini and E. Hatziminaoglou, "Revisiting the
   infrared spectra of active galactic nuclei with a new torus emission
   model," MNRAS, 366, 767 (2006). arXiv:astro-ph/0511428.
   https://doi.org/10.1111/j.1365-2966.2006.09866.x
.. [2] M. Stalevski, C. Ricci, Y. Ueda, P. Lira, J. Fritz, and M. Baes,
   "The dust covering factor in active galactic nuclei," MNRAS, 458,
   2288 (2016). arXiv:1602.06954. https://doi.org/10.1093/mnras/stw444
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from tengri.components.agn.polar_dust import (
    _type1_mask,
    calzetti2000_extinction_curve,
    smc_extinction_curve,
)
from tengri.utils.scale import representable_denominator

#: Torus blocks that represent a genuine dusty torus with an equatorial optical
#: depth and an opening angle, mapped to the (opening-angle, equatorial
#: optical-depth) parameter names the screen reads. Blocks not listed here (toy
#: two-temperature, GRAHSP) get no torus screen. The angle's convention differs
#: by torus; read both through :func:`torus_screen_geometry`.
TORUS_SCREEN_PARAMS: dict[str, tuple[str, str]] = {
    "skirtor": ("agn_oa_skirtor", "agn_tau_skirtor"),
    "fritz": ("agn_fritz_oa", "agn_fritz_tau"),
}
TORUS_SCREEN_BLOCKS: tuple[str, ...] = tuple(TORUS_SCREEN_PARAMS)

#: Fritz et al. (2006) dust model: the visual extinction per unit equatorial
#: optical depth at 9.7 um, :math:`A_V/\tau_{9.7} = 23`. Section 3 of the paper
#: gives :math:`\tau(9.7) = 0.1` for a column :math:`N_{\rm H} = 9.0 \times
#: 10^{21}` cm^-2, "an optical extinction of :math:`A_V = 2.3`"; Section 4
#: gives :math:`\tau(9.7) = 8` as :math:`A_V \sim 170` (21 per unit).
_FRITZ_AV_PER_TAU_97 = 23.0

#: :math:`\tau_V = A_V / (2.5 \log_{10} e)`: magnitudes to optical depth.
_TAU_PER_MAG = 0.4 * math.log(10.0)

#: Fritz library: equatorial :math:`\tau_V` per unit :math:`\tau_{9.7}`
#: (:math:`23 \times 0.4 \ln 10 = 21.2`).
FRITZ_TAU_V_PER_TAU_97 = _FRITZ_AV_PER_TAU_97 * _TAU_PER_MAG


#: The parameter naming the half-opening angle (from the equatorial plane) of the
#: tori that read one but are not screened; every other torus is governed by the
#: generic ``agn_theta_torus``.
_TORUS_OPENING_ANGLE_PARAM: dict[str, str] = {
    "skirtor": "agn_oa_skirtor",
    "skirtor_agnfitter": "agn_oa_skirtor",
    "skirtor_agnfitter_1p": "agn_oa_skirtor",
    "skirtor_agnfitter_2p": "agn_oa_skirtor",
    "nenkova_agnfitter_2p": "agn_oa_nenkova",
    "nenkova_agnfitter_3p": "agn_oa_nenkova",
}


def polar_follow_opening_angle(torus_block: str, params: dict) -> float:
    """The selected torus's own opening angle, which the polar cone follows.

    Parameters
    ----------
    torus_block : str
        The torus block name.
    params : dict
        The runner's parameter dict.

    Returns
    -------
    float
        Half-opening angle from the equatorial plane [deg]: ``90 - agn_fritz_oa``
        for the Fritz library, ``agn_oa_skirtor`` for the SKIRTOR family,
        ``agn_oa_nenkova`` for the AGNfitter CLUMPY grids and ``agn_theta_torus``
        for every torus governed by the generic Type-1/2 mask.

    Notes
    -----
    **JIT-compatible**: yes. The torus block name is a static Python string.
    """
    if torus_block == "fritz":
        return torus_screen_geometry("fritz", params)[0]
    key = _TORUS_OPENING_ANGLE_PARAM.get(torus_block)
    if key is None:
        return params.get("agn_theta_torus", 30.0)
    return params.get(key, 40.0)


def torus_screen_geometry(torus_block: str, params: dict) -> tuple[float, float, str]:
    r"""The half-opening angle, V-band optical depth and cone geometry of a torus.

    Parameters
    ----------
    torus_block : str
        ``"skirtor"`` or ``"fritz"``.
    params : dict
        The runner's parameter dict.

    Returns
    -------
    oa_deg : float
        Half-opening angle from the equatorial plane [deg], the convention of
        :func:`torus_screen_transmission` and ``agn_polar_oa``. For the Fritz
        library, whose ``agn_fritz_oa`` is the dust-free cone's half-angle
        :math:`\theta_c`, it is :math:`90^\circ - \theta_c`.
    tau_v : float
        Equatorial V-band optical depth. [dimensionless]
    geometry : str
        The polar-cone form this torus uses: ``"skirtor"`` or ``"fritz"``.

    Notes
    -----
    **JIT-compatible**: yes. The torus block name is a static Python string.
    """
    if torus_block == "fritz":
        cone_half = params.get("agn_fritz_oa", 60.0)
        tau_97 = params.get("agn_fritz_tau", 1.0)
        return 90.0 - cone_half, FRITZ_TAU_V_PER_TAU_97 * tau_97, "fritz"
    if torus_block == "skirtor":
        return params.get("agn_oa_skirtor", 40.0), params.get("agn_tau_skirtor", 7.0), "skirtor"
    raise ValueError(f"torus_screen_geometry: no screen for torus block {torus_block!r}")


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
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    cos_inc : float
        Cosine of the inclination (angle from the polar axis); 1 = face-on.
        [dimensionless]
    oa_deg : float
        Torus half-opening angle from the equatorial plane. [deg] The Type-1
        limit is :math:`i = 90^\circ - {\rm oa}`; see :func:`torus_screen_geometry`
        for the Fritz library's angle.
    tau_v : float
        Torus equatorial V-band optical depth (the disc seen through the torus
        rim is reddened by ~this depth). [dimensionless]
    law : str
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

    # Type-2 weight: 1 when edge-on (cos_inc < sin(oa)), 0 when face-on; the
    # complement of the polar mask's Type-1 weight, at the same width.
    type2 = 1.0 - _type1_mask(cos_inc, oa_deg)

    tau_lambda = (
        jnp.maximum(tau_v, 0.0)
        * (k_lambda / jnp.maximum(k_v, representable_denominator(1e-30)))
        * type2
    )
    return jnp.exp(-jnp.clip(tau_lambda, 0.0, 50.0))


def jax_sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """Numerically-stable logistic sigmoid."""
    return 0.5 * (1.0 + jnp.tanh(0.5 * x))
