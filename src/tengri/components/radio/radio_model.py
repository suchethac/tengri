# SPDX-License-Identifier: BSD-3-Clause
"""Radio power-law model (single-power-law AGN + star-formation synchrotron)
SEDModelComponent.

Implements the radio power-law model on the SEDModelComponent contract.
Provides differentiable radio SED prediction.

Physical pipeline
-----------------
1. L_IR (dust emission) → SFR via FIR-radio correlation
2. SFR + L_AGN → radio spectral energy distribution
3. Power-law synchrotron + free-free thermal emission
4. Add to full SED

Cross-component contract
------------------------
Inputs: L_ir (dust IR luminosity), L_agn_bol (AGN bolometric), log_mstar
(stellar mass) from upstream components with fallbacks.
Outputs: sed_radio (radio continuum on pipeline wavelength grid).

Notes
-----
**JIT-compatible**: yes.

**Model**: single power-law (AGN + star-formation driven). Double
power-law with aging cutoff (DPL mode) is a follow-up PR.

**Fallbacks**: radio is fully functional even without dust/AGN/stellar
components (uses sensible defaults).

References
----------
.. [1] Bell 2003, ApJ, 586, 794 (FIR-radio correlation)
.. [2] Condon 1992, ARA&A, 30, 575 (Radio SED theory)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import jax.numpy as jnp

from tengri.components.radio.radio import radio_total
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Fixed, Uniform
from tengri.parameters.resolve import require_redshift
from tengri.protocols.component import (
    ParamDeclaration,
    SEDComponentConfig,
)

__all__ = ["RadioPowerLawSEDComponent", "RadioPowerLawSEDComponentConfig"]


@dataclass(frozen=True)
class RadioPowerLawSEDComponentConfig(SEDComponentConfig):
    """Frozen knobs for RadioPowerLawSEDComponent.

    Attributes
    ----------
    name: str
        Diagnostic identifier. Default ``"radio_powerlaw"``.
    sfr_mode: str
        FIR-radio correlation mode: ``"bell2003"``, ``"delvecchio2021"``,
        or ``"mccheyne2022"``. Default ``"bell2003"``.
    include_freefree: bool
        Include Murphy+2011 thermal free-free component. Default ``True``.
    """

    name: str = "radio_powerlaw"
    sfr_mode: str = "bell2003"
    include_freefree: bool = True


class RadioPowerLawSEDComponent(SEDModelComponent):
    """SEDComponent for radio synchrotron (power-law) + thermal emission.

    Computes radio continuum from star-formation rate (via FIR-radio
    correlation) and AGN luminosity. Single power-law model (double
    power-law with aging is a follow-up).

    Free parameters (7):

    - radio_q_ir: FIR-radio correlation parameter
    - radio_alpha_sf: SFR-driven radio slope
    - radio_loudness: AGN radio loudness parameter
    - radio_alpha_agn: AGN radio spectral index
    - radio_T_e: electron temperature [K]
    - radio_alpha_ff: free-free spectral index

    Notes
    -----
    **JIT-compatible**: yes.
    **Optional inputs**: reads L_ir, L_agn_bol, log_mstar with fallbacks.
    Component is fully functional without dust/AGN/stellar upstream.
    """

    def __init__(self) -> None:
        """Initialize component with radio_powerlaw config."""
        self.config = RadioPowerLawSEDComponentConfig()

    name: str = "radio_powerlaw"
    parameter_prefix: str = "radio_"
    requires_template_data: ClassVar[bool] = False

    # Free parameters
    q_ir = Uniform(
        1.0,
        3.0,
        description="FIR-radio correlation parameter",
        units="dimensionless",
        default=2.4,
    )
    alpha_sf = Uniform(
        -1.0,
        1.0,
        description="SFR-driven radio spectral index",
        units="dimensionless",
        default=-0.7,
    )
    loudness = Fixed(0.0, description="AGN radio loudness", units="dex")
    alpha_agn = Uniform(
        -1.0,
        0.5,
        description="AGN radio spectral index",
        units="dimensionless",
        default=-0.7,
    )
    T_e = Fixed(8000.0, description="Electron temperature (free-free)", units="K")
    alpha_ff = Fixed(-0.1, description="Free-free spectral index", units="dimensionless")

    # No required cross-component inputs: radio reads opportunistically
    # with documented fallbacks (zero for missing).
    inputs: ClassVar[dict[str, str]] = {}
    optional_inputs: ClassVar[dict[str, str]] = {
        "L_ir": "erg/s",
        "L_agn_bol": "erg/s",
        "log_mstar": "dex",
    }
    outputs: ClassVar[dict[str, str]] = {
        "sed_radio": "erg/s/Hz",
    }

    def load(self, wave: jnp.ndarray | None = None) -> None:
        """No-op precomputation for power-law radio."""
        return None

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Declare the 6 free parameters owned by radio power-law."""
        return super().declared_parameters()

    def predict(
        self,
        p: Mapping[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        **inputs: Any,
    ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        """Predict radio SED via power-law model.

        Parameters
        ----------
        p: mapping[str, ndarray]
            Parameters with prefix stripped: q_ir, alpha_sf, loudness, alpha_agn,
            T_e, alpha_ff.
        sed_in: ndarray
            Input SED (stellar + nebular + AGN continuum).
        wave: ndarray
            Rest-frame wavelength grid in Angstrom.
        **inputs: ndarray
            Opportunistic cross-component reads: L_ir, L_agn_bol, log_mstar
            (with defaults if not present).

        Returns
        -------
        tuple[ndarray, mapping]

            - sed_out: sed_in + radio continuum.
            - published: Dict with "sed_radio".

        """
        # Read cross-component inputs with fallbacks
        L_ir = jnp.asarray(inputs.get("L_ir", 0.0))
        L_agn_bol = jnp.asarray(inputs.get("L_agn_bol", 0.0))
        log_mstar = jnp.asarray(inputs.get("log_mstar", 10.0))
        redshift = jnp.asarray(require_redshift(p, "components.radio.radio_model.predict"))

        # Call radio_total (power-law path)
        L_radio = radio_total(
            wave,
            L_ir=L_ir,
            L_agn_bol=L_agn_bol,
            q_ir=jnp.asarray(p["q_ir"]),
            alpha_sf=jnp.asarray(p["alpha_sf"]),
            radio_loudness=jnp.asarray(p["loudness"]),
            alpha_agn=jnp.asarray(p["alpha_agn"]),
            log_mstar=log_mstar,
            redshift=redshift,
            sfr_mode=self.config.sfr_mode,
            include_freefree=self.config.include_freefree,
            T_e=jnp.asarray(p["T_e"]),
            alpha_ff=jnp.asarray(p["alpha_ff"]),
        )

        return sed_in + L_radio, {
            "sed_radio": L_radio,
        }
