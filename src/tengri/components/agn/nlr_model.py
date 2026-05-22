"""AGN narrow-line region (NLR) emission — SEDModelComponent port.

The NLR is photoionised gas illuminated by the AGN accretion disc.
It produces nebular-like emission: a power-law continuum plus
forbidden-line emission at key wavelengths. Distinct from the
star-formation-driven nebular component (Cue, CloudyGrid, …); the
energy source is the AGN.

Calls the existing :func:`compute_nlr_sed` primitive in
:mod:`tengri.components.agn.nlr`. No physics reimplementation.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from tengri.components.agn.nlr import compute_nlr_sed
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Fixed, Uniform

__all__ = ["AGNNebular"]


_C_AA_PER_S = 2.99792458e18


def _trapz_freq(L_lambda: jnp.ndarray, wave: jnp.ndarray) -> jnp.ndarray:
    nu = _C_AA_PER_S / wave
    return jnp.abs(jnp.trapezoid(L_lambda, nu))


class AGNNebular(SEDModelComponent):
    r"""AGN narrow-line region — line + power-law continuum emission.

    Photoionised by the AGN disc; the covering fraction sets what
    fraction of the disc luminosity reaches the NLR, and small line +
    continuum re-emission efficiencies determine what fraction comes
    out at each wavelength.

    Cross-component contract
    ------------------------
    Reads:
      * ``L_agn_bol`` — AGN bolometric luminosity (erg/s) from the
        upstream AGN component.
    Publishes:
      * ``L_nlr`` — total NLR luminosity (erg/s).

    Notes
    -----
    **JIT-compatible**: yes. ``predict`` is pure JAX and the line
    profile is a sum of Gaussians via ``jax.vmap``.

    **Default emission set**: 11 key forbidden + Balmer lines from
    [OII] 3727 through [SII] 6731, calibrated against typical Seyfert 2
    relative strengths.

    References
    ----------
    .. [1] B. A. Groves, M. A. Dopita & R. S. Sutherland, "Dusty,
       Radiation Pressure-dominated Photoionization. I.," ApJS, 153,
       9 (2004). https://doi.org/10.1086/421113
    .. [2] A. Feltre, S. Charlot & J. Gutkin, "Nuclear activity versus
       star formation: emission-line diagnostics at ultraviolet and
       optical wavelengths," MNRAS, 456, 3354 (2016).
       https://doi.org/10.1093/mnras/stv2794
    """

    name = "agn_nlr"
    parameter_prefix = "agn_nlr_"

    cov_frac = Uniform(0.0, 0.5, description="NLR covering fraction", units="")
    fwhm_kms = Fixed(500.0, description="NLR line FWHM", units="km/s")
    line_eff = Fixed(0.10, description="line-emission efficiency", units="")

    inputs: dict[str, str] = {"L_agn_bol": "erg/s"}  # noqa: RUF012
    outputs: dict[str, str] = {"L_nlr": "erg/s"}  # noqa: RUF012

    def predict(
        self,
        p: dict[str, Any],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        *,
        L_agn_bol: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        addition = compute_nlr_sed(
            wavelength=wave,
            l_disc_bol_erg=L_agn_bol,
            covering_fraction=p["cov_frac"],
            fwhm_kms=p["fwhm_kms"],
            line_efficiency=p["line_eff"],
        )
        return sed_in + addition, {"L_nlr": _trapz_freq(addition, wave)}
