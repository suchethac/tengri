# SPDX-License-Identifier: BSD-3-Clause
"""Phenomenological power-law accretion disc model SEDModelComponent.

Implements the power-law accretion disc model on the SEDModelComponent
contract for fast prototyping and simple AGN fits.

This is an opt-in adapter, the existing AGNSEDComponent continues to
support power-law discs through the unified AGN registry.

Notes
-----
This is a simplified model suitable for fast fitting and exploratory work.
For production science, use Kubota & Done (2018) or SKIRTOR torus models.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import jax.numpy as jnp

from tengri.components.agn.disc import powerlaw_disc as _powerlaw_disc_fn
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Uniform
from tengri.protocols.component import SEDComponentConfig

__all__ = ["PowerLawDisc"]


@dataclass(frozen=True)
class PowerLawDiscConfig(SEDComponentConfig):
    """Configuration for power-law AGN disc (minimal; no state)."""

    pass


@dataclass(frozen=True)
class PowerLawDisc(SEDModelComponent):
    """Simple power-law accretion disc with exponential UV cutoff.

    A fast phenomenological disc model: L_ν ∝ ν^α exp(-hν / k_B T_max).
    Suitable for rapid fitting and when fine spectral details are not required.

    Attributes
    ----------
    name: str
        Component registry key: ``"powerlaw_disc"``.
    parameter_prefix: str
        Parameter namespace: ``"agn_"``.
    config: PowerLawDiscConfig
        Frozen configuration (unused; here for consistency).

    Free parameters (class-level declarations, auto-discovered)
    -----------------------------------------------------------
    log_lbol: Uniform
        log₁₀(L_bol / L_sun). [dex, 8–14]
    alpha: Uniform
        Power-law spectral index. [dimensionless, -1.5–-0.5]
    T_max: Uniform
        UV cutoff temperature. [K, 10^4–10^6]
    frac: Uniform
        Fraction of bolometric luminosity from disc. [dimensionless, 0–1]

    Cross-component outputs
    -----------------------
    L_agn_disc: erg/s
        Bolometric luminosity contribution from disc.

    Notes
    -----
    **JIT-compatible**: yes, predict() is pure JAX.

    **Approximation**: This model is a simplified phenomenological
    representation. It does not capture multi-zone temperature structure,
    soft X-ray excess, or hard X-ray corona. Use for fast prototyping only.

    Examples
    --------
    Minimal model with power-law disc::

        from tengri import SEDModel, Fixed, Uniform, builders
        from tengri.components.agn.powerlaw_disc_model import PowerLawDisc

        model = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh=builders.sfh.dpl(_=Fixed(1.5), beta=Fixed(1.0)),
            dust_attenuation={"type": "two_component", "all_params": Fixed},
            agn=PowerLawDisc(),
        )
    """

    name = "powerlaw_disc"
    parameter_prefix = "agn_"
    config: PowerLawDiscConfig = PowerLawDiscConfig()

    # Free parameters: auto-discovered
    log_lbol = Uniform(
        8.0,
        14.0,
        description="AGN bolometric luminosity",
        units="dex (L_sun)",
        default=11.0,
    )
    alpha = Uniform(
        -1.5,
        -0.5,
        description="Power-law spectral index",
        units="dimensionless",
        default=-0.5,
    )
    T_max = Uniform(
        1e4,
        1e6,
        description="UV cutoff temperature",
        units="K",
        default=1e5,
    )
    lum_ratio = Uniform(
        0.0,
        1.0,
        description="Disc luminosity fraction of L_bol",
        units="dimensionless",
        default=0.5,
    )

    # Cross-component output
    outputs: ClassVar[dict[str, str]] = {"L_agn_disc": "erg/s"}

    def predict(
        self,
        p: Mapping[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        **inputs: Any,
    ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        """Pure JAX power-law disc prediction.

        Parameters
        ----------
        p: mapping[str, ndarray]
            Parameters with prefix already stripped:

            - log_lbol: log₁₀(L_bol / L_sun)
            - alpha: power-law index
            - T_max: UV cutoff temperature (K)
            - frac: disc luminosity fraction

        sed_in: ndarray, shape (n_wave,)
            Input SED in erg/s/Hz.
        wave: ndarray, shape (n_wave,)
            Rest-frame wavelength grid in Angstrom.
        **inputs: ndarray
            Unused (AGN disc is self-contained).

        Returns
        -------
        tuple[ndarray, dict]
            (sed_out, published) where:

            - sed_out: Updated SED (sed_in + disc contribution).
            - published: {"L_agn_disc": bolometric disc luminosity [erg/s]}.

        """
        # Call power-law disc model
        sed_disc = _powerlaw_disc_fn(
            wavelength=wave,
            agn_log_lbol=p["log_lbol"],
            agn_lum_ratio=p["frac"],
            agn_alpha=p["alpha"],
            agn_T_max=p["T_max"],
        )

        # Integrate to bolometric luminosity
        from tengri.components.agn._phys import bolometric_integral_nu, wavelength_to_nu

        nu = wavelength_to_nu(wave)
        L_disc = bolometric_integral_nu(sed_disc, nu)

        # Add to intrinsic SED
        sed_out = sed_in + sed_disc

        return sed_out, {"L_agn_disc": L_disc}
