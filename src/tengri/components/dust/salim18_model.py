# SPDX-License-Identifier: BSD-3-Clause
"""Salim et al. (2018) dust attenuation — SEDModelComponent port.

The single-file `SEDModelComponent`-style port of the Salim, Boquien & Lee (2018)
modified Calzetti + Leitherer with UV bump and slope tilt. Coexists with the
existing `DustAttenuationSEDComponent` adapter; the new class is opt-in.

The physics calls into :func:`tengri.components.dust.attenuation.salim_sbl18`.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from tengri.components.dust.attenuation import salim_sbl18 as _salim18_law
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Uniform

__all__ = ["Salim18"]


def _trapz_freq(L_lambda: jnp.ndarray, wave: jnp.ndarray) -> jnp.ndarray:
    """Frequency-space trapezoid integral of an L_nu spectrum.

    L_nu values arrive on a wavelength grid `wave` (Å). Integrating
    in frequency requires reversing into frequency-ascending order.
    """
    c = 2.99792458e18  # Å / s
    nu = c / wave
    return jnp.trapezoid(L_lambda[::-1], nu[::-1])


class Salim18(SEDModelComponent):
    r"""Salim, Boquien & Lee (2018) modified Calzetti + Leitherer attenuation.

    Multiplies the incoming SED by :math:`A(\lambda) = e^{-\tau_V k(\lambda)}`
    where :math:`k(\lambda)` combines Leitherer (2002) UV extension of Calzetti
    (2000) with a variable 2175 Å UV bump and power-law slope tilt.
    The modification order is: **(base × slope_mod) + bump**.

    Parameters
    ----------
    tau_v : float
        V-band optical depth :math:`\tau_V`. Free; range 0–4 by default.
    dust_bump_strength : float
        Amplitude of 2175 Å UV bump. Free; range 0–2 by default.
    dust_delta : float
        Power-law slope modification. Free; range -0.5 to +0.5 by default.

    Cross-component contract
    ------------------------
    Reads: nothing.
    Publishes: ``L_absorbed`` — total absorbed luminosity [erg/s].

    Notes
    -----
    **JIT-compatible**: yes. ``predict`` is pure JAX.

    **Physics**: attenuated SED is
    :math:`L_\nu^{\rm out}(\lambda) = L_\nu^{\rm in}(\lambda)\,e^{-\tau_V k(\lambda)}`.
    The attenuation curve combines Leitherer+2002 (far-UV below 1800 Å) and
    Calzetti+2000 (optical/NIR) with optional bump and slope modifications.
    Absorbed luminosity computed in frequency space.

    References
    ----------
    .. [1] S. Salim, M. Boquien, and J. C. Lee, "CANDELS: Constraining the AGN
       Contribution to the Star Formation Rate Density at z > 1,"
       ApJ, 859, 11 (2018).
       https://doi.org/10.3847/1538-4357/aabf3c
    .. [2] C. Leitherer et al., "Global Far-Ultraviolet (912–1800 Å) Properties of
       Star-forming Galaxies," ApJS, 140, 303 (2002).
       https://doi.org/10.1086/342486
    .. [3] S. Calzetti et al., "The Dust Content and Opacity of Star-Forming
       Galaxies," ApJ, 533, 682 (2000).
       https://doi.org/10.1086/308692
    """

    name = "salim18"
    parameter_prefix = "dust_"

    tau_v = Uniform(0.0, 4.0, description="V-band optical depth", units="")
    dust_bump_strength = Uniform(0.0, 2.0, description="Amplitude of 2175 Å UV bump", units="")
    dust_delta = Uniform(
        -0.5, 0.5, description="Power-law slope modification (Noll+2009)", units=""
    )

    inputs: dict[str, str] = {}  # noqa: RUF012
    outputs: dict[str, str] = {"L_absorbed": "erg/s"}  # noqa: RUF012

    def predict(
        self,
        p: dict[str, Any],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Apply Salim+18 attenuation to the incoming SED.

        Parameters
        ----------
        p : dict
            Parameter dict with prefix stripped: ``p["tau_v"]``,
            ``p["dust_bump_strength"]``, ``p["dust_delta"]``.
        sed_in : ndarray, shape (n_wave,)
            Incoming rest-frame L_ν from upstream components [erg/s/Hz].
        wave : ndarray, shape (n_wave,)
            Rest-frame wavelength grid [Å].

        Returns
        -------
        sed_out : ndarray, shape (n_wave,)
            Attenuated rest-frame L_ν [erg/s/Hz].
        published : dict
            ``{"L_absorbed": L_absorbed_erg_s}``.
        """
        k_salim18 = _salim18_law(
            wave,
            dust_bump_strength=p["dust_bump_strength"],
            dust_delta=p["dust_delta"],
        )
        atten = jnp.exp(-p["tau_v"] * k_salim18)
        sed_out = sed_in * atten
        L_absorbed = _trapz_freq(sed_in - sed_out, wave)
        return sed_out, {"L_absorbed": L_absorbed}
