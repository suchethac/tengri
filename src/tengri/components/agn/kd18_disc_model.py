# SPDX-License-Identifier: BSD-3-Clause
"""Three-zone accretion disc model (Kubota & Done 2018) SEDModelComponent.

Implements the Kubota & Done (2018) accretion disc model on the
SEDModelComponent contract, enabling use in the model-building API.

This is an opt-in adapter, the existing AGNSEDComponent continues to
support K&D18 through the unified AGN registry.

References
----------
.. [1] A. Kubota & C. Done, "A physical interpretation of the hard
   X-ray excess in low-luminosity AGN," MNRAS 480, 1247 (2018).
   arXiv:1804.02334. https://doi.org/10.1093/mnras/sty1890
.. [2] A. A. Beloborodov, "Energetic Radiation from Accretion Tori,"
   ApJ 510, L123 (1999). arXiv:astro-ph/9810145.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import jax.numpy as jnp

from tengri.components.agn.disc import kubota_done_disc as _kubota_done_disc_fn
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Uniform
from tengri.protocols.component import SEDComponentConfig

__all__ = ["KD18Disc"]


@dataclass(frozen=True)
class KD18DiscConfig(SEDComponentConfig):
    """Configuration for K&D18 three-zone accretion disc.

    Parameters
    ----------
    self_consistent_gamma : bool, optional
        If True, derive the hard X-ray photon index Γ_hot self-consistently
        from the Beloborodov (1999) energy-balance relation. If False,
        use the agn_gamma_hard parameter directly. Default: False.
    n_radii : int, optional
        Number of radial grid points for zone integral approximations.
        Default: 50. Higher values increase accuracy but cost.
    """

    self_consistent_gamma: bool = False
    n_radii: int = 50


@dataclass(frozen=True)
class KD18Disc(SEDModelComponent):
    """Kubota & Done (2018) three-zone accretion disc.

    A physically stratified model with outer standard disc, warm Comptonization
    zone, and hot corona. Self-consistent zone radii and temperature profiles
    enable smooth inference gradients across the full parameter space.

    Attributes
    ----------
    name : str
        Component registry key: ``"kd18_disc"``.
    parameter_prefix : str
        Parameter namespace: ``"agn_"``.
    config : KD18DiscConfig
        Frozen configuration (self-consistent gamma, radial grid).

    Free parameters (class-level declarations, auto-discovered)
    -----------------------------------------------------------
    log_lbol : Uniform
        log₁₀(L_bol / L_sun). [dex, 8–14]
    log_mbh : Uniform
        log₁₀(M_BH / M_sun). [dex, 6–10]
    log_ledd : Uniform
        Eddington ratio log₁₀(L / L_Edd). [dex, -3–0]
    a_spin : Uniform
        Dimensionless black hole spin (prograde). [dimensionless, 0–0.998]
    cos_inc : Uniform
        Cosine of inclination (1 = face-on, 0 = edge-on). [dimensionless, 0.01–1]
    f_hard : Uniform
        Fraction of Eddington luminosity in hot corona. [dimensionless, 0.01–0.5]
    gamma_warm : Uniform
        Photon index of warm Comptonization zone. [dimensionless, 1.5–3.5]
    kt_warm : Uniform
        Warm zone electron temperature. [keV, 0.1–0.5]
    gamma_hard : Uniform
        Hard X-ray photon index (used if self_consistent_gamma=False).
        [dimensionless, 1.5–2.5]
    kt_hot : Uniform
        Hot corona electron temperature. [keV, 50–200]
    r_warm_ratio : Uniform
        Radius ratio R_warm / R_hot. [dimensionless, 1.1–5]
    frac : Uniform
        Fraction of bolometric luminosity from disc. [dimensionless, 0–1]

    Cross-component outputs
    -----------------------
    L_agn_disc : erg/s
        Bolometric luminosity contribution from all three zones.

    Notes
    -----
    **JIT-compatible**: yes, predict() is pure JAX.

    **Gradient-safe**: yes, self-consistent zone radii via bisection
    are smooth over the parameter space.

    **Cross-component note**: AGN inclination (cos_inc) is shared with
    torus/corona models and is **never** auto-derived from geometry.
    Declare it as a free parameter independently.

    Examples
    --------
    Minimal model with K&D18 disc::

        from tengri import SEDModel, Fixed, FIXED, Uniform, builders
        from tengri.components.agn.kd18_disc_model import KD18Disc, KD18DiscConfig

        model = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh=builders.sfh.dpl(alpha=Fixed(1.5), beta=Fixed(1.0)),
            dust_attenuation={"type": "two_component", "all_params": FIXED},
            agn=KD18Disc(config=KD18DiscConfig(self_consistent_gamma=True)),
        )

    See Also
    --------
    tengri.components.agn.disc : Kubota & Done (2018) implementation.
    """

    name = "kd18_disc"
    parameter_prefix = "agn_"
    config: KD18DiscConfig = field(default_factory=KD18DiscConfig)

    # Free parameters: auto-discovered
    log_lbol = Uniform(
        8.0,
        14.0,
        description="AGN bolometric luminosity",
        units="dex (L_sun)",
        default=11.0,
    )
    log_mbh = Uniform(
        6.0,
        10.0,
        description="Black hole mass",
        units="dex (M_sun)",
        default=8.0,
    )
    log_ledd = Uniform(
        -3.0,
        0.0,
        description="Eddington ratio",
        units="dex",
        default=-1.5,
    )
    a_spin = Uniform(
        0.0,
        0.998,
        description="Black hole spin parameter",
        units="dimensionless",
        default=0.5,
    )
    cos_inc = Uniform(
        0.01,
        1.0,
        description="Cosine of inclination",
        units="dimensionless",
        default=0.8,
    )
    f_hard = Uniform(
        0.01,
        0.5,
        description="Corona luminosity fraction",
        units="dimensionless",
        default=0.1,
    )
    gamma_warm = Uniform(
        1.5,
        3.5,
        description="Warm Comptonization photon index",
        units="dimensionless",
        default=2.5,
    )
    kt_warm = Uniform(
        0.1,
        0.5,
        description="Warm zone electron temperature",
        units="keV",
        default=0.2,
    )
    gamma_hard = Uniform(
        1.5,
        2.5,
        description="Hard X-ray photon index",
        units="dimensionless",
        default=1.9,
    )
    kt_hot = Uniform(
        50.0,
        200.0,
        description="Hot corona electron temperature",
        units="keV",
        default=100.0,
    )
    r_warm_ratio = Uniform(
        1.1,
        5.0,
        description="Radius ratio R_warm / R_hot",
        units="dimensionless",
        default=3.0,
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
        """Pure JAX K&D18 three-zone disc prediction.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Parameters with prefix already stripped:

            - log_lbol: log₁₀(L_bol / L_sun)
            - log_mbh: log₁₀(M_BH / M_sun)
            - log_ledd: Eddington ratio
            - a_spin: black hole spin
            - cos_inc: cosine of inclination
            - f_hard: hot corona fraction
            - gamma_warm: warm zone photon index
            - kt_warm: warm zone temperature (keV)
            - gamma_hard: hard X-ray index
            - kt_hot: hot corona temperature (keV)
            - r_warm_ratio: R_warm / R_hot
            - frac: disc luminosity fraction

        sed_in : ndarray, shape (n_wave,)
            Input SED in erg/s/Hz.
        wave : ndarray, shape (n_wave,)
            Rest-frame wavelength grid in Angstrom.
        **inputs : ndarray
            Unused (AGN disc is self-contained).

        Returns
        -------
        tuple[ndarray, dict]
            (sed_out, published) where:

            - sed_out: Updated SED (sed_in + disc contribution).
            - published: {"L_agn_disc": bolometric disc luminosity [erg/s]}.

        """
        # Call K&D18 disc model
        sed_disc = _kubota_done_disc_fn(
            wavelength=wave,
            agn_log_lbol=p["log_lbol"],
            agn_lum_ratio=p["frac"],
            agn_log_mbh=p["log_mbh"],
            agn_log_ledd=p["log_ledd"],
            agn_a_spin=p["a_spin"],
            agn_cos_inc=p["cos_inc"],
            agn_f_hard=p["f_hard"],
            agn_gamma_warm=p["gamma_warm"],
            agn_kt_warm=p["kt_warm"],
            agn_gamma_hard=p["gamma_hard"],
            agn_kt_hot=p["kt_hot"],
            agn_r_warm_ratio=p["r_warm_ratio"],
            n_radii=self.config.n_radii,
            agn_self_consistent_gamma=self.config.self_consistent_gamma,
        )

        # Integrate to bolometric luminosity
        from tengri.components.agn._phys import bolometric_integral_nu, wavelength_to_nu

        nu = wavelength_to_nu(wave)
        L_disc = bolometric_integral_nu(sed_disc, nu)

        # Add to intrinsic SED
        sed_out = sed_in + sed_disc

        return sed_out, {"L_agn_disc": L_disc}
