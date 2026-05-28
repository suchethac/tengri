# SPDX-License-Identifier: BSD-3-Clause
"""SMC dust attenuation model — SEDModelComponent port.

The single-file `SEDModelComponent`-style port of the SMC Bar
(Pei 1992) attenuation curve with no UV bump. Coexists with the existing
`DustAttenuationSEDComponent` adapter; the new class is opt-in.

The physics calls into :func:`tengri.components.dust.attenuation.smc`.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from tengri.components.dust.attenuation import smc as _smc_law
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Uniform

__all__ = ["SMC"]


def _trapz_freq(L_lambda: jnp.ndarray, wave: jnp.ndarray) -> jnp.ndarray:
    """Frequency-space trapezoid integral of an L_nu spectrum.

    L_nu values arrive on a wavelength grid `wave` (Å). Integrating
    in frequency requires reversing into frequency-ascending order.
    """
    c = 2.99792458e18  # Å / s
    nu = c / wave
    return jnp.trapezoid(L_lambda[::-1], nu[::-1])


class SMC(SEDModelComponent):
    r"""SMC Bar dust attenuation curve (Pei 1992).

    Multiplies the incoming SED by :math:`A(\lambda) = e^{-\tau_V k(\lambda)}`
    where :math:`k(\lambda)` is the SMC extinction curve from
    :func:`tengri.components.dust.attenuation.smc`. Steep UV, no 2175 Å bump;
    common at high redshift. R_V = 2.93.

    Parameters
    ----------
    tau_v : float
        V-band optical depth :math:`\tau_V`. Free; range 0–4 by default.

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
    .. [1] P. G. Pei, "Interstellar Dust from the Ultraviolet to the
       Infrared," ApJ, 395, 130 (1992).
       https://doi.org/10.1086/171665
    """

    name = "smc"
    parameter_prefix = "dust_"

    tau_v = Uniform(
        0.0, 4.0, default=1.0, description="V-band optical depth", units="dimensionless"
    )

    inputs: dict[str, str] = {}  # noqa: RUF012
    outputs: dict[str, str] = {"L_absorbed": "erg/s"}  # noqa: RUF012

    def predict(
        self,
        p: dict[str, Any],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Apply SMC attenuation to the incoming SED.

        Parameters
        ----------
        p : dict
            Parameter dict with prefix stripped: ``p["tau_v"]``.
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
        k_smc = _smc_law(wave)
        atten = jnp.exp(-p["tau_v"] * k_smc)
        sed_out = sed_in * atten
        L_absorbed = _trapz_freq(sed_in - sed_out, wave)
        return sed_out, {"L_absorbed": L_absorbed}
