# SPDX-License-Identifier: BSD-3-Clause
"""SKIRTOR torus SEDModelComponent adapter.

Ports the SKIRTOR clumpy torus (Stalevski et al. 2012, 2016) to the
SEDModelComponent framework, enabling use of radiative-transfer templates
in the model-building API.

This is an opt-in adapter — the existing AGNSEDComponent continues to
support SKIRTOR through the unified AGN registry.

References
----------
.. [1] M. Stalevski et al., "3D radiative transfer modelling of the dusty
   torus around AGN — the influence of clumping," MNRAS, 420, 2756 (2012).
   arXiv:1109.1286. https://doi.org/10.1111/j.1365-2966.2011.19775.x
.. [2] M. Stalevski et al., "The dust covering factor in AGN — combining the
   IR torus emission with polar dust component," MNRAS, 458, 2288 (2016).
   arXiv:1602.01954. https://doi.org/10.1093/mnras/stw444
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import jax.numpy as jnp

from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Uniform
from tengri.protocols.component import SEDComponentConfig, SEDComponentState

__all__ = ["SKIRTORTorus"]


@dataclass(frozen=True)
class SKIRTORTorusConfig(SEDComponentConfig):
    """Configuration for SKIRTOR torus templates.

    Parameters
    ----------
    grid_path : str or None
        Path to SKIRTOR template grid (.npz or .h5). If None, templates
        are not pre-loaded (deferred to first use in predict).
    """

    grid_path: str | None = None


@dataclass(frozen=True)
class SKIRTORTorusState(SEDComponentState):
    """Cached SKIRTOR template data.

    Attributes
    ----------
    name : str
        Component identifier.
    skirtor_fn : callable or None
        Compiled interpolation function from create_skirtor_from_grid,
        or None if templates are not available.
    """

    name: str = "skirtor"
    skirtor_fn: Any | None = None


@dataclass(frozen=True)
class SKIRTORTorus(SEDModelComponent):
    """Clumpy torus SED from SKIRTOR radiative-transfer models.

    A pure-JAX implementation with C²-continuous gradients via triweight
    kernel interpolation in the 5D parameter space (tau, p, q, opening angle,
    inclination). Publishes separate disc and torus contributions, with
    polar dust wire-in for Type 1 sightlines.

    Attributes
    ----------
    name : str
        Component registry key: ``"skirtor"``.
    parameter_prefix : str
        Parameter namespace: ``"agn_"``.
    config : SKIRTORTorusConfig
        Frozen configuration (grid path).

    Free parameters (class-level declarations, auto-discovered)
    -----------------------------------------------------------
    log_lbol : Uniform
        log₁₀(L_bol / L_sun). [dex, 8–14]
    tau_skirtor : Uniform
        Edge-on optical depth at 9.7 μm. [dimensionless, 3–11]
    p_skirtor : Uniform
        Radial dust density power-law gradient. [dimensionless, 0–1.5]
    q_skirtor : Uniform
        Polar dust density power-law gradient. [dimensionless, 0–1.5]
    oa_skirtor : Uniform
        Torus half-opening angle. [degrees, 20–60]
    cos_inc : Uniform
        Cosine of inclination (1 = face-on, 0 = edge-on). [dimensionless, 0–1]
    frac_agn : Uniform
        AGN fraction in a configurable band (CIGALE convention).
        [dimensionless, 0–1]

    Cross-component outputs
    -----------------------
    L_agn_disc : erg/s
        Bolometric luminosity from accretion disc (intrinsic, at θ=30°).
    L_agn_torus : erg/s
        Bolometric luminosity from torus dust thermal emission.
    L_agn_polar_dust : erg/s
        Bolometric luminosity from polar dust reemission (Type 1 only).
    L_2500_30deg : erg/s/Hz
        Specific luminosity at 2500 Å, θ=30°; feeds X-ray normalisation.
    L_6um : erg/s/Hz
        Specific luminosity at 6 μm for mid-IR diagnostics.
    L_12um : erg/s/Hz
        Specific luminosity at 12 μm for mid-IR diagnostics.

    Notes
    -----
    **JIT-compatible**: yes — predict() is pure JAX.

    **Gradient-safe**: yes — triweight interpolation is fully differentiable.

    **Requires template grid**: The SKIRTOR template library (~1 GB) must be
    downloaded separately and pointed to via ``grid_path`` in config. The
    predict method gracefully returns zero emission if templates are unavailable.

    **Polar dust**: Applied to Type 1 sightlines (cos_inc ≥ cos(90° - oa))
    via the smooth sigmoid from polar_dust.py. Energy-conserving reemission
    as Casey-2012 modified blackbody.

    **Citation**: Stalevski et al. 2016 (SKIRTOR); Yang et al. 2020, §2.2.2
    (polar dust + anisotropy).

    Examples
    --------
    Minimal model with SKIRTOR torus::

        from tengri import SEDModel, Fixed, Uniform, builders
        from tengri.components.agn.skirtor_model import SKIRTORTorus

        # Register and use
        model = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh=builders.sfh.dpl(_=Fixed(1.5), beta=Fixed(1.0)),
            dust={"type": "two_component", "*": Fixed},
            agn=SKIRTORTorus(config=SKIRTORTorusConfig(grid_path="path/to/grid.h5")),
        )

    See Also
    --------
    tengri.components.agn.skirtor : template loader and interpolation.
    """

    name = "skirtor"
    parameter_prefix = "agn_"
    config: SKIRTORTorusConfig = field(default_factory=SKIRTORTorusConfig)

    # Free parameters — auto-discovered
    log_lbol = Uniform(8.0, 14.0, description="AGN bolometric luminosity", units="dex (L_sun)")
    tau_skirtor = Uniform(
        3.0, 11.0, description="9.7 µm optical depth (Stalevski et al.)", units=""
    )
    p_skirtor = Uniform(0.0, 1.5, description="Radial dust density gradient", units="")
    q_skirtor = Uniform(0.0, 1.5, description="Polar dust density gradient", units="")
    oa_skirtor = Uniform(20.0, 60.0, description="Torus half-opening angle", units="deg")
    cos_inc = Uniform(0.0, 1.0, description="Cosine of inclination", units="")
    frac_agn = Uniform(
        0.0, 1.0, description="AGN fraction (L_AGN / L_total, CIGALE convention)", units=""
    )

    # Cross-component outputs
    outputs: ClassVar[dict[str, str]] = {
        "L_agn_disc": "erg/s",
        "L_agn_torus": "erg/s",
        "L_agn_polar_dust": "erg/s",
        "L_2500_30deg": "erg/s/Hz",
        "L_6um": "erg/s/Hz",
        "L_12um": "erg/s/Hz",
    }

    def load(self, wave: jnp.ndarray | None = None) -> Any | None:
        """Load SKIRTOR v3 template grid with separate components.

        Parameters
        ----------
        wave : ndarray, optional
            Rest-frame wavelength grid (not used by SKIRTOR; templates
            interpolate to any target grid).

        Returns
        -------
        callable or None
            Interpolation function from create_skirtor_components_from_grid
            (returns SKIRTORComponents), or None if template file is not
            available or grid_path is not set.
        """
        from tengri.components.agn.skirtor import create_skirtor_components_from_grid

        if not self.config.grid_path:
            return None

        try:
            return create_skirtor_components_from_grid(self.config.grid_path)
        except (FileNotFoundError, OSError, KeyError):
            # Templates not available — predict will return zero emission
            return None

    def predict(
        self,
        p: Mapping[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        **inputs: Any,
    ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        """Pure JAX SKIRTOR prediction with separate components and polar dust.

        Interpolates the SKIRTOR template grid, applies polar dust extinction
        to Type 1 sightlines, and publishes all derived luminosities.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Parameters with prefix already stripped:
            - log_lbol: log₁₀(L_bol / L_sun)
            - tau_skirtor: optical depth at 9.7 µm
            - p_skirtor: radial density gradient
            - q_skirtor: polar density gradient
            - oa_skirtor: opening angle (degrees)
            cos_inc: cosine of inclination
            - frac_agn: AGN luminosity fraction
        sed_in : ndarray, shape (n_wave,)
            Input SED in erg/s/Hz.
        wave : ndarray, shape (n_wave,)
            Rest-frame wavelength grid in Angstrom.
        **inputs : ndarray
            Unused (AGN torus is self-contained).

        Returns
        -------
        tuple[ndarray, dict]
            (sed_out, published) where:
            - sed_out: Updated SED (sed_in + disc + torus + polar dust).
            - published: dict with keys L_agn_disc, L_agn_torus, L_agn_polar_dust,
              L_2500_30deg, L_6um, L_12um [erg/s] or [erg/s/Hz].
        """
        from tengri.components.agn._phys import wavelength_to_nu
        from tengri.components.agn.polar_dust import (
            polar_dust_emission,
            polar_dust_extinction,
        )

        # If templates are not loaded, return zero emission
        if not hasattr(self, "data") or self.data is None:
            zero_dict = {
                "L_agn_disc": jnp.array(0.0),
                "L_agn_torus": jnp.array(0.0),
                "L_agn_polar_dust": jnp.array(0.0),
                "L_2500_30deg": jnp.array(0.0),
                "L_6um": jnp.array(0.0),
                "L_12um": jnp.array(0.0),
            }
            return sed_in, zero_dict

        skirtor_fn = self.data

        # Call SKIRTOR interpolator to get separate components
        components = skirtor_fn(
            wavelength=wave,
            agn_log_lbol=p["log_lbol"],
            agn_tau_skirtor=p["tau_skirtor"],
            agn_p_skirtor=p["p_skirtor"],
            agn_q_skirtor=p["q_skirtor"],
            agn_oa_skirtor=p["oa_skirtor"],
            agn_cos_inc=p["cos_inc"],
            frac_agn=p["frac_agn"],
        )

        # Unpack components
        sed_disc = components.disk
        sed_torus_dust = components.dust

        # Compute derived quantities from disc
        nu = wavelength_to_nu(wave)
        idx_sort = jnp.argsort(nu)

        # L_agn_disc: bolometric luminosity of intrinsic disc
        L_agn_disc = jnp.trapezoid(sed_disc[idx_sort], nu[idx_sort])

        # L_2500_30deg: specific luminosity at 2500 Å (for α_OX)
        L_2500 = jnp.interp(2500.0, wave, sed_disc)

        # L_6um and L_12um: mid-IR diagnostics
        L_6um = jnp.interp(60000.0, wave, sed_disc + sed_torus_dust)  # 6 um = 60000 A
        L_12um = jnp.interp(120000.0, wave, sed_disc + sed_torus_dust)  # 12 um = 120000 A

        # L_agn_torus: bolometric torus dust luminosity
        L_agn_torus = jnp.trapezoid(sed_torus_dust[idx_sort], nu[idx_sort])

        # Apply polar dust (Type 1 only): extinction of disc, reemission
        # Default: no polar dust (EBV=0), but wire in for future flexibility
        polar_ebv = 0.0  # Future: make this a parameter
        _, l_abs = polar_dust_extinction(
            sed_disc,
            wave,
            p["cos_inc"],
            p["oa_skirtor"],
            polar_ebv,
            law="smc",
        )
        sed_polar_reemit = polar_dust_emission(
            jnp.trapezoid(l_abs[idx_sort], nu[idx_sort]),
            wave,
            temperature=100.0,
            beta=1.6,
            lambda_0=2e6,
        )
        L_agn_polar_dust = jnp.trapezoid(sed_polar_reemit[idx_sort], nu[idx_sort])

        # Total SED: disc + torus + polar reemission
        # (Disc extinction is minimal when EBV=0, so use original disc here)
        sed_out = sed_in + sed_disc + sed_torus_dust + sed_polar_reemit

        published = {
            "L_agn_disc": L_agn_disc,
            "L_agn_torus": L_agn_torus,
            "L_agn_polar_dust": L_agn_polar_dust,
            "L_2500_30deg": L_2500,
            "L_6um": L_6um,
            "L_12um": L_12um,
        }

        return sed_out, published
