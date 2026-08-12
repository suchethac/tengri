# SPDX-License-Identifier: BSD-3-Clause
"""AGN X-ray corona model (Lusso & Risaliti 2016 alpha_ox relation)
SEDModelComponent.

Implements the AGN X-ray corona model on the SEDModelComponent contract.
Provides differentiable AGN X-ray SED prediction driven by AGN bolometric
luminosity.

Physical pipeline
-----------------
1. AGN bolometric luminosity → UV luminosity at 2500 A (bolometric correction)
2. UV luminosity → 2 keV luminosity (alpha_ox relation)
3. 2 keV luminosity → X-ray SED (power-law + exponential cutoff)
4. Add to full SED

Cross-component contract
------------------------
Inputs: L_agn_bol (AGN bolometric luminosity) from upstream AGN component
        (with fallback to 0 if not present).
Outputs: L_xray_agn (X-ray luminosity on pipeline wavelength grid).

Notes
-----
**JIT-compatible**: yes.

**Model**: Lusso & Risaliti (2016) for AGN X-ray corona via alpha_ox
scaling. Inverse-Compton scattering with power-law spectrum and
exponential high-energy cutoff.

**Fallbacks**: X-ray is fully functional without AGN component; it
gracefully defaults to zero AGN contribution.

References
----------
.. [1] Lusso & Risaliti 2016, ApJ, 819, 154
.. [2] Hopkins et al. 2007, ApJ, 654, 731
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import jax.numpy as jnp

from tengri.components.sed_model_component import SEDModelComponent
from tengri.components.xray.xray import xray_agn_corona
from tengri.parameters.priors import Fixed, Uniform
from tengri.protocols.component import (
    ParamDeclaration,
    SEDComponentConfig,
)

__all__ = ["AGNXRayCoronaSEDComponent", "AGNXRayCoronaSEDComponentConfig"]


@dataclass(frozen=True)
class AGNXRayCoronaSEDComponentConfig(SEDComponentConfig):
    """Frozen knobs for AGNXRayCoronaSEDComponent.

    Attributes
    ----------
    name : str
        Diagnostic identifier. Default ``"agn_xray_corona"``.
    """

    name: str = "agn_xray_corona"


class AGNXRayCoronaSEDComponent(SEDModelComponent):
    """SEDComponent for AGN X-ray corona emission.

    Computes X-ray continuum from the AGN disc UV luminosity via the
    empirical alpha_ox(L_2500) relation (Just+2007 by default), following
    the X-CIGALE corona treatment.

    Free parameters (3):

    - agn_xray_gamma: X-ray photon index (Gamma)
    - agn_xray_delta_alpha_ox: offset [dex] on the Just+2007 alpha_ox(L_2500)
    - agn_xray_e_cut: high-energy cutoff [keV]

    Notes
    -----
    **JIT-compatible**: yes.
    **Optional inputs**: anchors the α_ox corona to L_2500 via the chain
    ``L_2500_intrinsic`` (actual disc L_ν(2500 Å), any disc) → ``L_2500_30deg``
    (SKIRTOR) → ``L_agn_bol`` with a Hopkins+2007 BC fallback. Returns zero
    X-ray if no AGN luminosity is published.

    The corona is evaluated at the Yang+2020 30° reference inclination
    (anisotropy factor exactly 1): this component has no inclination input,
    so it stays at the anchor where the alpha_ox relation is defined —
    same policy as ``XRayAirdSEDComponent`` (#980).
    """

    def __init__(self) -> None:
        """Initialize component with agn_xray_corona config."""
        self.config = AGNXRayCoronaSEDComponentConfig()

    name: str = "agn_xray_corona"
    parameter_prefix: str = "agn_xray_"
    requires_template_data: ClassVar[bool] = False

    # Free parameters
    gamma = Uniform(
        1.4,
        2.4,
        description="X-ray photon index",
        units="dimensionless",
        default=1.9,
    )
    # Offset on the empirical alpha_ox(L_2500), NOT an absolute alpha_ox.
    # The absolute default -1.4 was silently re-applied on top of Just+2007
    # (alpha_ox ~ -2.8, a 3.6 dex under-prediction) until #981.
    delta_alpha_ox = Fixed(0.0, description="Offset on Just+2007 alpha_ox(L_2500)", units="dex")
    e_cut = Fixed(300.0, description="High-energy cutoff", units="keV")

    # Reads AGN bolometric luminosity with fallback
    inputs: ClassVar[dict[str, str]] = {}
    outputs: ClassVar[dict[str, str]] = {
        "L_xray_agn": "erg/s",
    }

    def load(self, wave: jnp.ndarray | None = None) -> None:
        """No-op precomputation for analytic X-ray model."""
        return None

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Declare the 3 free parameters owned by AGN X-ray."""
        return super().declared_parameters()

    def predict(
        self,
        p: Mapping[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        **inputs: Any,
    ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        """Predict AGN X-ray SED via corona model.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Parameters with prefix stripped: gamma, delta_alpha_ox, e_cut.
        sed_in : ndarray
            Input SED (stellar + nebular + radio).
        wave : ndarray
            Rest-frame wavelength grid in Angstrom.
        **inputs : ndarray
            Opportunistic cross-component reads: L_agn_bol (with fallback
            to 0.0 if not present from AGN component).

        Returns
        -------
        tuple[ndarray, mapping]

            - sed_out: sed_in + X-ray continuum.
            - published: Dict with "L_xray_agn" (integrated X-ray luminosity).

        """
        # L_2500 anchor chain (matches the live ``xray/component.py``):
        # 1. ``L_2500_intrinsic`` — the composable AGN runner's actual disc
        #    L_ν(2500 Å), published for *every* disc type (qsogen, richards2006,
        #    …), so the α_ox corona is anchored to the real disc luminosity.
        # 2. ``L_2500_30deg`` — SKIRTOR's 30° reference value.
        # 3. ``L_agn_bol`` → L_2500 via the Hopkins+2007 BC_2500 as a last
        #    resort (only when no disc L_2500 is published).
        # Reading only ``L_2500_30deg`` (SKIRTOR-only) made this ~1.6× too bright
        # for non-SKIRTOR discs, since the BC fallback over-estimates L_2500.
        L_2500_intrinsic = jnp.asarray(inputs.get("L_2500_intrinsic", 0.0))
        L_2500_30deg = jnp.asarray(inputs.get("L_2500_30deg", 0.0))
        L_agn_bol = jnp.asarray(inputs.get("L_agn_bol", 0.0))
        L_2500_fallback = L_agn_bol / (5.15 * 1.199e15)  # erg/s/Hz
        L_2500 = jnp.where(
            L_2500_intrinsic > 0.0,
            L_2500_intrinsic,
            jnp.where(L_2500_30deg > 0.0, L_2500_30deg, L_2500_fallback),
        )

        # alpha_ox is derived from L_2500 via Just+2007 inside the corona;
        # the component knob is the delta offset around that empirical prior
        # (default 0.0 — #981 fixed the absolute -1.4 being fed here).
        L_xray = xray_agn_corona(
            wave,
            l_2500_30deg_erg_hz=L_2500,
            gamma=jnp.asarray(p["gamma"]),
            E_cut=jnp.asarray(p["e_cut"]),
            delta_alpha_ox=jnp.asarray(p["delta_alpha_ox"]),
        )

        # Integrate X-ray luminosity over spectrum
        L_xray_total = jnp.trapezoid(L_xray, wave)

        return sed_in + L_xray, {
            "L_xray_agn": L_xray_total,
        }
