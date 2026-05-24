# SPDX-License-Identifier: BSD-3-Clause
"""XRayAirdSEDComponent: X-ray emission (Aird+2017) as a SEDModelComponent.

Ports the X-ray emission model (Aird et al. 2017 SFR-X-ray scaling +
Lusso & Risaliti 2016 AGN X-ray) to the SEDModelComponent architecture.
Provides differentiable X-ray SED prediction.

Physical pipeline
-----------------
1. SFR → X-ray binaries (HMXB + LMXB)
2. M_* → additional XRB scaling
3. L_AGN → AGN X-ray corona
4. Combine into X-ray SED
5. Add to full SED

Cross-component contract
------------------------
Inputs: sfr (star-formation rate), log_mstar (stellar mass), L_agn_bol
(AGN bolometric luminosity) from upstream with fallbacks.
Outputs: sed_xray (X-ray continuum on pipeline wavelength grid).

Notes
-----
**JIT-compatible**: yes.

**Models**: Lehmer et al. (2010, 2016) for X-ray binaries + Lusso &
Risaliti (2016) for AGN X-ray corona. Both are analytic scalings with
no grid dependence.

**Fallbacks**: X-ray is fully functional without AGN component; it
gracefully defaults to XRB-only emission.

References
----------
.. [1] Lehmer et al. 2010, ApJ, 724, 559
.. [2] Lehmer et al. 2016, ApJ, 825, 7
.. [3] Lusso & Risaliti 2016, ApJ, 819, 154
.. [4] Aird et al. 2017, MNRAS, 465, 3390
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import jax.numpy as jnp

from tengri.components.sed_model_component import SEDModelComponent
from tengri.components.xray.xray import xray_total
from tengri.parameters.priors import Fixed, Uniform
from tengri.protocols.component import (
    ParamDeclaration,
    SEDComponentConfig,
)

__all__ = ["XRayAirdSEDComponent", "XRayAirdSEDComponentConfig"]


@dataclass(frozen=True)
class XRayAirdSEDComponentConfig(SEDComponentConfig):
    """Frozen knobs for XRayAirdSEDComponent.

    Attributes
    ----------
    name : str
        Diagnostic identifier. Default ``"xray_aird"``.
    """

    name: str = "xray_aird"


class XRayAirdSEDComponent(SEDModelComponent):
    """SEDComponent for X-ray emission (XRB + AGN corona).

    Computes X-ray continuum from star-formation rate (X-ray binaries)
    and AGN luminosity (corona). Implements Lehmer+2010/2016 (XRB scaling)
    and Lusso & Risaliti 2016 (AGN X-ray).

    Free parameters (5):
    - xray_gamma_hmxb: HMXB spectral index
    - xray_gamma_lmxb: LMXB spectral index
    - xray_gamma_agn: AGN spectral index
    - xray_E_cut: high-energy cutoff [keV]
    - xray_alpha_ox: alpha_ox AGN parameter

    Notes
    -----
    **JIT-compatible**: yes.
    **Optional inputs**: reads sfr, log_mstar, L_agn_bol with sensible defaults.
    Component works without stellar/AGN components (defaults to XRB-only).
    """

    def __init__(self) -> None:
        """Initialize component with xray_aird config."""
        self.config = XRayAirdSEDComponentConfig()

    name: str = "xray_aird"
    parameter_prefix: str = "xray_"

    # Free parameters
    gamma_hmxb = Uniform(1.0, 3.0, description="HMXB spectral index", units="dimensionless")
    gamma_lmxb = Uniform(1.0, 3.0, description="LMXB spectral index", units="dimensionless")
    gamma_agn = Uniform(1.0, 3.0, description="AGN X-ray spectral index", units="dimensionless")
    E_cut = Fixed(300.0, description="High-energy cutoff", units="keV")
    alpha_ox = Fixed(-0.5, description="Lusso & Risaliti alpha_ox", units="dimensionless")

    # No required cross-component inputs (all have fallbacks)
    inputs: ClassVar[dict[str, str]] = {}
    outputs: ClassVar[dict[str, str]] = {
        "sed_xray": "erg/s/Hz",
    }

    def load(self, wave: jnp.ndarray | None = None) -> None:
        """No-op precomputation for analytic X-ray model."""
        return None

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Declare the 5 free parameters owned by X-ray."""
        return super().declared_parameters()

    def predict(
        self,
        p: Mapping[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        **inputs: Any,
    ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        """Predict X-ray SED via XRB + AGN corona model.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Parameters with prefix stripped: gamma_hmxb, gamma_lmxb,
            gamma_agn, E_cut, alpha_ox.
        sed_in : ndarray
            Input SED (stellar + nebular + AGN + radio).
        wave : ndarray
            Rest-frame wavelength grid in Angstrom.
        **inputs : ndarray
            Opportunistic cross-component reads: sfr, log_mstar, L_agn_bol
            (with defaults if not present).

        Returns
        -------
        tuple[ndarray, mapping]
            - sed_out: sed_in + X-ray continuum.
            - published: Dict with "sed_xray".
        """
        # Read cross-component inputs with fallbacks
        sfr = jnp.asarray(inputs.get("sfr", 1.0))
        log_mstar = jnp.asarray(inputs.get("log_mstar", 10.0))
        stellar_mass = 10.0**log_mstar
        L_agn_bol = jnp.asarray(inputs.get("L_agn_bol", 0.0))

        # Call xray_total
        L_xray = xray_total(
            wave,
            sfr=sfr,
            stellar_mass=stellar_mass,
            L_agn_bol=L_agn_bol,
            gamma_hmxb=jnp.asarray(p["gamma_hmxb"]),
            gamma_lmxb=jnp.asarray(p["gamma_lmxb"]),
            gamma_agn=jnp.asarray(p["gamma_agn"]),
            E_cut=jnp.asarray(p["E_cut"]),
            alpha_ox=jnp.asarray(p["alpha_ox"]),
        )

        return sed_in + L_xray, {
            "sed_xray": L_xray,
        }
