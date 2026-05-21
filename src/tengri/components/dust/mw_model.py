"""Cardelli et al. (1989) Milky Way dust attenuation — SEDModelComponent port.

The single-file `SEDModelComponent`-style port of the Cardelli, Clayton & Mathis
(1989) attenuation curve with free R_V. Coexists with the existing
`DustAttenuationSEDComponent` adapter; the new class is opt-in.

The physics calls into :func:`tengri.components.dust.attenuation.cardelli`.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from tengri.components.dust.attenuation import cardelli as _cardelli_law
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Uniform

__all__ = ["MilkyWay"]


def _trapz_freq(L_lambda: jnp.ndarray, wave: jnp.ndarray) -> jnp.ndarray:
    """Frequency-space trapezoid integral of an L_nu spectrum.

    L_nu values arrive on a wavelength grid `wave` (Å). Integrating
    in frequency requires reversing into frequency-ascending order.
    """
    c = 2.99792458e18  # Å / s
    nu = c / wave
    return jnp.trapezoid(L_lambda[::-1], nu[::-1])


class MilkyWay(SEDModelComponent):
    r"""Cardelli et al. (1989) Milky Way attenuation with free R_V.

    Multiplies the incoming SED by :math:`A(\lambda) = e^{-\tau_V k(\lambda)}`
    where :math:`k(\lambda)` is the MW extinction curve from
    :func:`tengri.components.dust.attenuation.cardelli`. Detailed piecewise
    fit spanning UV to infrared with parameterized R_V for flexibility.

    Parameters
    ----------
    tau_v : float
        V-band optical depth :math:`\tau_V`. Free; range 0–4 by default.
    dust_Rv : float
        Total-to-selective extinction ratio R_V. Free; range 2.5–5.5 by default.

    Cross-component contract
    ------------------------
    Reads: nothing.
    Publishes: ``L_absorbed`` — total absorbed luminosity [erg/s].

    Notes
    -----
    **JIT-compatible**: yes. ``predict`` is pure JAX.

    **Physics**: attenuated SED is
    :math:`L_\nu^{\rm out}(\lambda) = L_\nu^{\rm in}(\lambda)\,e^{-\tau_V k(\lambda)}`.
    Absorbed luminosity computed in frequency space.

    References
    ----------
    .. [1] D. E. Cardelli, G. C. Clayton, and J. S. Mathis, "The Relationship
       between Infrared, Optical, and Ultraviolet Extinction," ApJ, 345, 245 (1989).
       https://doi.org/10.1086/167900
    """

    name = "mw"
    parameter_prefix = "dust_"

    tau_v = Uniform(0.0, 4.0, description="V-band optical depth", units="")
    dust_Rv = Uniform(2.5, 5.5, description="Total-to-selective extinction R_V", units="")

    inputs: dict[str, str] = {}  # noqa: RUF012
    outputs: dict[str, str] = {"L_absorbed": "erg/s"}  # noqa: RUF012

    def predict(
        self,
        p: dict[str, Any],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Apply Cardelli attenuation to the incoming SED.

        Parameters
        ----------
        p : dict
            Parameter dict with prefix stripped: ``p["tau_v"]``, ``p["dust_Rv"]``.
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
        k_cardelli = _cardelli_law(wave, dust_Rv=p["dust_Rv"])
        atten = jnp.exp(-p["tau_v"] * k_cardelli)
        sed_out = sed_in * atten
        L_absorbed = _trapz_freq(sed_in - sed_out, wave)
        return sed_out, {"L_absorbed": L_absorbed}
