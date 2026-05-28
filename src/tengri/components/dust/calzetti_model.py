# SPDX-License-Identifier: BSD-3-Clause
"""Calzetti dust attenuation model — SEDModelComponent port.

The single-file `SEDModelComponent`-style port of the Calzetti+2000
starburst attenuation curve. Coexists with the existing
`DustAttenuationSEDComponent` adapter at
`src/tengri/components/dust/component.py`; that adapter remains the
default for current pipelines. The new class is opt-in.

The math is unchanged — physics calls into the existing
`calzetti(wave)` primitive in
:mod:`tengri.components.dust.attenuation`. This module is only the
authoring-shape adapter.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from tengri.components.dust.attenuation import calzetti as _calzetti_law
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Uniform

__all__ = ["Calzetti"]


def _trapz_freq(L_lambda: jnp.ndarray, wave: jnp.ndarray) -> jnp.ndarray:
    """Frequency-space trapezoid integral of an L_nu spectrum.

    L_nu values arrive on a wavelength grid `wave` (Å). Integrating
    in frequency requires reversing into frequency-ascending order.
    """
    # nu = c / lambda; ascending wavelength → descending frequency
    c = 2.99792458e18  # Å / s
    nu = c / wave
    # jnp.trapezoid integrates assuming ascending x; flip both arrays.
    return jnp.trapezoid(L_lambda[::-1], nu[::-1])


class Calzetti(SEDModelComponent):
    r"""Calzetti+2000 starburst attenuation curve.

    Multiplies the incoming SED by :math:`A(\lambda) = e^{-\tau_V k(\lambda)}`
    where :math:`k(\lambda)` is the wavelength-dependent attenuation
    shape from :func:`tengri.components.dust.attenuation.calzetti`.
    The absorbed luminosity is published into ``state.derived["L_absorbed"]``
    for any downstream dust IR component to consume.

    Parameters
    ----------
    tau_v : float
        V-band optical depth :math:`\tau_V`. Free; range 0–4 by
        default.
    delta : float
        UV slope deviation (Noll+2009). Adds a wavelength tilt to the
        Calzetti shape. Free; range −0.5 to +0.5 by default.

    Cross-component contract
    ------------------------
    Reads: nothing (this is a pure SED transformation).
    Publishes: ``L_absorbed`` — total absorbed luminosity [erg/s].

    Notes
    -----
    **JIT-compatible**: yes. ``predict`` is pure JAX.

    **Physics**: the attenuated SED is
    :math:`L_\nu^{\rm out}(\lambda) = L_\nu^{\rm in}(\lambda)\,e^{-\tau_V k(\lambda)}`.
    Absorbed luminosity is computed in frequency space:
    :math:`L_{\rm absorbed} = \int (L_\nu^{\rm in} - L_\nu^{\rm out})\,d\nu`.

    The :math:`\delta` parameter modulates the law shape per
    Noll+2009: :math:`k_{\rm eff}(\lambda) = k(\lambda)(\lambda/5500)^\delta`.
    Setting :math:`\delta = 0` recovers the bare Calzetti law.

    References
    ----------
    .. [1] D. Calzetti et al., "The Dust Content and Opacity of
       Star-Forming Galaxies," ApJ, 533, 682 (2000).
       https://doi.org/10.1086/308692
    .. [2] S. Noll et al., "Analysis of galaxy spectral energy
       distributions from far-UV to far-IR with CIGALE," A&A, 507,
       1793 (2009). https://doi.org/10.1051/0004-6361/200912497
    """

    name = "calzetti"
    parameter_prefix = "dust_"

    tau_v = Uniform(
        0.0, 4.0, default=1.0, description="V-band optical depth", units="dimensionless"
    )
    delta = Uniform(
        -0.5, 0.5, default=0.0, description="UV slope deviation (Noll+2009)", units="dimensionless"
    )

    inputs: dict[str, str] = {}  # noqa: RUF012
    outputs: dict[str, str] = {"L_absorbed": "erg/s"}  # noqa: RUF012

    def predict(
        self,
        p: dict[str, Any],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Apply Calzetti attenuation to the incoming SED.

        Parameters
        ----------
        p : dict
            Parameter dict with prefix stripped: ``p["tau_v"]``, ``p["delta"]``.
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
        # k(lambda) — wavelength-dependent shape of the Calzetti law
        k_calz = _calzetti_law(wave)
        # Noll+2009 modulation: k_eff = k * (lambda / 5500 A)**delta
        tilt = (wave / 5500.0) ** p["delta"]
        k_eff = k_calz * tilt
        # Attenuation factor and the new SED
        atten = jnp.exp(-p["tau_v"] * k_eff)
        sed_out = sed_in * atten
        # Absorbed luminosity in frequency space
        L_absorbed = _trapz_freq(sed_in - sed_out, wave)
        return sed_out, {"L_absorbed": L_absorbed}
