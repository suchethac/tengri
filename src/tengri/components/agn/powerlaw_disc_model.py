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

from tengri.components.agn._params import PARAMS as _AGN_PARAMS
from tengri.components.agn.disc import powerlaw_disc as _powerlaw_disc_fn
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Uniform
from tengri.protocols.component import SEDComponentConfig, declared_prior

__all__ = ["PowerLawDisc"]

#: Single source of truth for every class-level prior below: the shared
#: ``agn_*`` declarations in ``_params.py``. Read once at class-definition
#: time so these class attributes cannot drift from the canonical
#: declaration the way they did before this fix (found by
#: ``tools/check_param_restatements.py``, Task 11 item 5 / fix round 1,
#: R25): ``log_lbol`` default 11.0 vs canonical 10.0, ``alpha`` bounds
#: (-1.5, -0.5) vs canonical (-2, 0), ``lum_ratio`` bounds (0, 1) vs
#: canonical (0, 5).
_LOG_LBOL_PRIOR = declared_prior(_AGN_PARAMS, "agn_log_lbol")
_ALPHA_PRIOR = declared_prior(_AGN_PARAMS, "agn_alpha")
_T_MAX_PRIOR = declared_prior(_AGN_PARAMS, "agn_T_max")
_LUM_RATIO_PRIOR = declared_prior(_AGN_PARAMS, "agn_lum_ratio")


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
    name : str
        Component registry key: ``"powerlaw_disc"``.
    parameter_prefix : str
        Parameter namespace: ``"agn_"``.
    config : PowerLawDiscConfig
        Frozen configuration (unused; here for consistency).

    Free parameters (class-level declarations, auto-discovered)
    -----------------------------------------------------------
    log_lbol : Uniform
        log₁₀(L_bol / L_sun). [dex, 8–14]
    alpha : Uniform
        Power-law spectral index. [dimensionless, -2–0]
    T_max : Uniform
        UV cutoff temperature. [K, 10^4–10^6]
    lum_ratio : Uniform
        Fraction of bolometric luminosity from disc. [dimensionless, 0–5]

    Cross-component outputs
    -----------------------
    L_agn_disc : erg/s
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

        from tengri import SEDModel, Fixed, DEFAULT, Uniform, builders
        from tengri.components.agn.powerlaw_disc_model import PowerLawDisc

        model = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh=builders.sfh.dpl(alpha=Fixed(1.5), beta=Fixed(1.0)),
            dust_attenuation={"type": "two_component", "all_params": Fixed(DEFAULT)},
            agn=PowerLawDisc(),
        )
    """

    name = "powerlaw_disc"
    parameter_prefix = "agn_"
    config: PowerLawDiscConfig = PowerLawDiscConfig()

    # Free parameters: auto-discovered
    log_lbol = Uniform(
        _LOG_LBOL_PRIOR.lo,
        _LOG_LBOL_PRIOR.hi,
        description="AGN bolometric luminosity",
        units="dex (L_sun)",
        default=_LOG_LBOL_PRIOR.default,
    )
    alpha = Uniform(
        _ALPHA_PRIOR.lo,
        _ALPHA_PRIOR.hi,
        description="Power-law spectral index",
        units="dimensionless",
        default=_ALPHA_PRIOR.default,
    )
    T_max = Uniform(
        _T_MAX_PRIOR.lo,
        _T_MAX_PRIOR.hi,
        description="UV cutoff temperature",
        units="K",
        default=_T_MAX_PRIOR.default,
    )
    lum_ratio = Uniform(
        _LUM_RATIO_PRIOR.lo,
        _LUM_RATIO_PRIOR.hi,
        description="Disc luminosity fraction of L_bol",
        units="dimensionless",
        default=_LUM_RATIO_PRIOR.default,
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
        p : mapping[str, ndarray]
            Parameters with prefix already stripped:

            - log_lbol: log₁₀(L_bol / L_sun)
            - alpha: power-law index
            - T_max: UV cutoff temperature (K)
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
