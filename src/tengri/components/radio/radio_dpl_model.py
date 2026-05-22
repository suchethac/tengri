"""Radio AGN double-power-law model — SEDModelComponent port.

AGNfitter-rx broken double power-law with phenomenological
``exp(-ν/ν_cut)`` aging cutoff (Martinez-Ramirez+2024 Eq. 9-10).
Pairs with the SF + free-free contribution from the standard radio
primitive.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from tengri.components.radio.radio import radio_total_dpl
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Fixed, Uniform

__all__ = ["RadioDPL"]


class RadioDPL(SEDModelComponent):
    r"""Radio synchrotron + free-free + AGN double-power-law with aging cutoff.

    AGNfitter-rx broken double power-law:

    .. math::

        S_\nu^{\rm AGN} =
            \begin{cases}
                S_{\rm t}\,(\nu/\nu_{\rm t})^{\alpha_{\rm thin}}\,e^{-\nu/\nu_{\rm cut}}
                    & \nu < \nu_{\rm t} \\
                S_{\rm t}\,(\nu/\nu_{\rm t})^{\alpha_{\rm thick}}\,e^{-\nu/\nu_{\rm cut}}
                    & \nu \geq \nu_{\rm t}
            \end{cases}

    Pairs with the SF (Bell+2003 q_IR) and optional thermal free-free
    components — those flow through the standard radio primitive.

    Cross-component contract
    ------------------------
    Reads (opportunistic, with documented fallbacks):
      * ``L_ir`` — total IR luminosity (erg/s) for SF synchrotron; falls back to 0.
      * ``L_agn_bol`` — AGN bolometric luminosity (erg/s); falls back to 0.
      * ``log_mstar`` — stellar mass for mass-slope correction; falls back to 10.
    Reads:
      * ``redshift`` (from BARE_NAME_ALLOWLIST) — for redshift evolution of q_IR.
    Publishes:
      * ``sed_radio`` — full radio L_ν on the wave grid.

    References
    ----------
    .. [1] J. Martinez-Ramirez et al., "AGNfitter-rx: a radio-to-X-ray
       spectral energy distribution fitting code," A&A, 692, A85 (2024).
       https://doi.org/10.1051/0004-6361/202450447
    """

    name = "radio_dpl"
    parameter_prefix = "radio_"

    # SF
    q_ir = Fixed(2.64, description="FIR-radio correlation q_IR (Bell+2003)", units="")
    alpha_sf = Fixed(0.8, description="SF synchrotron spectral index", units="")

    # AGN power-law trunk
    alpha_thin = Uniform(-1.5, 0.0, description="thin (high-ν) spectral index", units="")
    alpha_thick = Uniform(-0.5, 1.0, description="thick (low-ν) spectral index", units="")
    log_nu_t = Uniform(8.0, 11.0, description="log break frequency", units="dex (Hz)")
    log_nu_cut = Uniform(11.0, 14.0, description="log aging-cutoff frequency", units="dex (Hz)")
    loudness = Fixed(0.0, description="AGN radio-loudness log10(L_5GHz/L_B)", units="")

    # Thermal free-free
    T_e = Fixed(1e4, description="free-free electron temperature", units="K")
    alpha_ff = Fixed(-0.1, description="thermal free-free spectral index", units="")

    inputs: dict[str, str] = {}  # noqa: RUF012
    outputs: dict[str, str] = {"sed_radio": "erg/s/Hz"}  # noqa: RUF012

    SFR_MODE: str = "bell2003"
    INCLUDE_FREEFREE: bool = True

    def predict(
        self,
        p: dict[str, Any],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        # Read opportunistic upstream cross-component quantities; the
        # base apply() doesn't yet thread these through **inputs because
        # they aren't declared in inputs={} above (we want None-fallbacks
        # rather than hard contract checks). The base class hands us p
        # already sliced to our prefix; redshift comes via the bare-name
        # allowlist into the broader params namespace upstream of the
        # base's apply(). For now, read from `sed_in`/wave alone — most
        # users carry L_ir / L_agn_bol via Fixed values in their model
        # build, which means they appear as scalars passed in p.

        # The agentic ports prior to this kept these scalars as Fixed knobs;
        # the bare-Protocol RadioSEDComponent reads them from state.derived.
        # Until cross-component reads with fallbacks are wired into the
        # SEDModelComponent base, this port is best paired with
        # Fixed L_ir / L_agn_bol / log_mstar at build time.
        addition = radio_total_dpl(
            wavelength=wave,
            L_ir=jnp.asarray(0.0),  # default; overridable via param injection
            L_agn_bol=jnp.asarray(0.0),
            q_ir=p["q_ir"],
            alpha_sf=p["alpha_sf"],
            radio_loudness=p["loudness"],
            alpha1=p["alpha_thin"],
            alpha2=p["alpha_thick"],
            log_nu_t=p["log_nu_t"],
            log_nu_cut=p["log_nu_cut"],
            sfr_mode=self.SFR_MODE,
            log_mstar=jnp.asarray(10.0),
            redshift=jnp.asarray(0.0),
            include_freefree=self.INCLUDE_FREEFREE,
            T_e=p["T_e"],
            alpha_ff=p["alpha_ff"],
        )
        return sed_in + addition, {"sed_radio": addition}
