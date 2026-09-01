# SPDX-License-Identifier: BSD-3-Clause
"""SKIRTOR_mean_3p three-parameter clumpy torus (Stalevski et al. 2016)
SEDModelComponent.

Implements the SKIRTOR_mean_3p torus library as averaged by AGNfitter-rX
on the SEDModelComponent contract, enabling use of AGNfitter-faithful torus
templates in the model-building API.

This is an opt-in adapter distinct from the default ``skirtor`` component,
which uses the full-grid SKIRTOR implementation (X-CIGALE faithful).
The two differ in peak wavelength and other properties; choose based on your
fidelity target (see issue #614, #592, and #633 for motivation).

References
----------
.. [1] M. Stalevski et al., "3D radiative transfer modeling of the dusty
   torus around AGN, the influence of clumping," MNRAS, 420, 2756 (2012).
   arXiv:1109.1286. https://doi.org/10.1111/j.1365-2966.2011.19775.x
.. [2] M. Stalevski et al., "The dust covering factor in AGN: combining the
   IR torus emission with polar dust component," MNRAS, 458, 2288 (2016).
   arXiv:1602.01954. https://doi.org/10.1093/mnras/stw444
.. [3] L. N. Martinez-Ramirez, et al., "AGNFITTER-RX: Modeling the
   radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
   (2024). arXiv:2405.12111.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import jax.numpy as jnp

from tengri.components.agn.skirtor_agnfitter import create_skirtor_agnfitter_from_grid
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Uniform
from tengri.protocols.component import SEDComponentConfig, SEDComponentState

__all__ = ["SKIRTORAgnfitterTorus"]


@dataclass(frozen=True)
class SKIRTORAgnfitterTorusConfig(SEDComponentConfig):
    """Configuration for SKIRTOR_mean_3p torus templates.

    Parameters
    ----------
    grid_path : str or None
        Path to SKIRTOR_mean_3p template grid (HDF5). If None, templates
        are not pre-loaded (deferred to first use in predict).
    """

    grid_path: str | None = None


@dataclass(frozen=True)
class SKIRTORAgnfitterTorusState(SEDComponentState):
    """Cached SKIRTOR_mean_3p template data.

    Attributes
    ----------
    name : str
        Component identifier.
    skirtor_agnfitter_fn : callable or None
        Compiled interpolation function from create_skirtor_agnfitter_from_grid,
        or None if templates are not available.
    """

    name: str = "skirtor_agnfitter"
    skirtor_agnfitter_fn: Any | None = None


@dataclass(frozen=True)
class SKIRTORAgnfitterTorus(SEDModelComponent):
    """SKIRTOR_mean_3p clumpy AGN torus (AGNfitter-faithful averaging).

    Three-parameter torus model with half-opening angle, inclination, and
    equatorial optical depth. Provides C²-continuous gradients via triweight
    kernel interpolation over the 3D parameter space.  This is the
    **AGNfitter-faithful** variant, differing from the default full-grid
    SKIRTOR in peak wavelength and other properties.

    Attributes
    ----------
    name : str
        Component registry key: ``"skirtor_agnfitter"``.
    parameter_prefix : str
        Parameter namespace: ``"agn_"``.
    config : SKIRTORAgnfitterTorusConfig
        Frozen configuration (grid path).

    Free parameters (class-level declarations, auto-discovered)
    -----------------------------------------------------------
    log_lbol : Uniform
        log₁₀(L_bol / L_sun). [dex, 8–14]
    oa_skirtor : Uniform
        Half-opening angle [deg]. [deg, 10–80]
    incl_skirtor : Uniform
        Inclination angle measured from pole [deg]. [deg, 0–90]
    tv_skirtor : Uniform
        Equatorial optical depth τ_9.7. [dimensionless, 3–11]
    torus_frac : Uniform
        Fraction of L_bol reprocessed by torus. [dimensionless, 0–1]

    Cross-component outputs
    -----------------------
    L_agn_torus : erg/s
        Bolometric luminosity contribution from torus emission.

    Notes
    -----
    **JIT-compatible**: yes, predict() is pure JAX.

    **Gradient-safe**: yes, triweight interpolation is fully differentiable
    across the three parameter axes.

    **Requires template grid**: The SKIRTOR_mean_3p template library must be
    built separately and pointed to via ``grid_path`` in config.
    The predict method gracefully returns zero emission if templates
    are unavailable.

    **AGNfitter parity**: This component reproduces AGNfitter-rX's
    SKIRTOR_mean_3p template selection exactly (nearest-neighbor in
    3D parameter space, then per-L_sun normalization).  The upstream
    full-grid SKIRTOR differs in parameter space (5D vs 3D), peak
    wavelength (Stalevski +40µm vs AGNfitter +25µm at matched geometry),
    and should not be conflated (see issue #614).

    Examples
    --------
    Minimal model with SKIRTOR_mean_3p torus::

        from tengri import SEDModel, Fixed, DEFAULT, Uniform, builders
        from tengri.components.agn.skirtor_agnfitter_model import (
            SKIRTORAgnfitterTorus,
            SKIRTORAgnfitterTorusConfig,
        )

        model = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh=builders.sfh.dpl(alpha=Fixed(1.5), beta=Fixed(1.0)),
            dust_attenuation={"type": "two_component", "all_params": Fixed(DEFAULT)},
            agn=SKIRTORAgnfitterTorus(
                config=SKIRTORAgnfitterTorusConfig(grid_path="path/to/grid.h5")
            ),
        )

    See Also
    --------
    tengri.components.agn.skirtor_agnfitter : template loader and interpolation.
    tengri.components.agn.skirtor_model : default (X-CIGALE-faithful) SKIRTOR.
    """

    name = "skirtor_agnfitter"
    parameter_prefix = "agn_"
    config: SKIRTORAgnfitterTorusConfig = field(default_factory=SKIRTORAgnfitterTorusConfig)

    # Free parameters: auto-discovered
    log_lbol = Uniform(
        8.0,
        14.0,
        description="AGN bolometric luminosity",
        units="dex (L_sun)",
        default=11.0,
    )
    oa_skirtor = Uniform(
        10.0,
        80.0,
        description="Half-opening angle (Stalevski et al.)",
        units="deg",
        default=40.0,
    )
    incl_skirtor = Uniform(
        0.0,
        90.0,
        description="Inclination angle (Stalevski et al.)",
        units="deg",
        default=30.0,
    )
    tv_skirtor = Uniform(
        3.0,
        11.0,
        description="Equatorial optical depth τ_9.7",
        units="dimensionless",
        default=7.0,
    )
    torus_frac = Uniform(
        0.0,
        1.0,
        description="Torus luminosity fraction of L_bol",
        units="dimensionless",
        default=0.5,
    )

    # Cross-component output
    outputs: ClassVar[dict[str, str]] = {"L_agn_torus": "erg/s"}

    def load(self, wave: jnp.ndarray | None = None) -> Any | None:
        """Load SKIRTOR_mean_3p template grid if available.

        Parameters
        ----------
        wave : ndarray, optional
            Rest-frame wavelength grid (not used by SKIRTOR; templates
            interpolate to any target grid).

        Returns
        -------
        callable or None
            Interpolation function from create_skirtor_agnfitter_from_grid, or None
            if template file is not available or grid_path is not set.
        """
        if not self.config.grid_path:
            return None

        try:
            return create_skirtor_agnfitter_from_grid(self.config.grid_path)
        except (FileNotFoundError, OSError, KeyError):
            # Templates not available: predict will return zero emission
            return None

    def predict(
        self,
        p: Mapping[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        **inputs: Any,
    ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        """Pure JAX SKIRTOR_mean_3p torus prediction.

        Interpolates the SKIRTOR_mean_3p template grid to the requested
        opening angle, inclination, and optical depth; normalizes to the
        user's luminosity scale; and returns the torus SED contribution.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Parameters with prefix already stripped:

            - log_lbol: log₁₀(L_bol / L_sun)
            - oa_skirtor: half-opening angle [deg]
            - incl_skirtor: inclination [deg]
            - tv_skirtor: equatorial optical depth
            - torus_frac: torus luminosity fraction

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

            - sed_out: Updated SED (sed_in + torus contribution).
            - published: {"L_agn_torus": bolometric torus luminosity [erg/s]}.

        """
        # If templates are not loaded, return zero emission
        if not hasattr(self, "data") or self.data is None:
            return sed_in, {"L_agn_torus": jnp.array(0.0)}

        skirtor_agnfitter_fn = self.data

        # Call SKIRTOR_mean_3p interpolator
        sed_torus = skirtor_agnfitter_fn(
            wavelength=wave,
            agn_log_lbol=p["log_lbol"],
            agn_oa_skirtor=p["oa_skirtor"],
            agn_incl_skirtor=p["incl_skirtor"],
            agn_tv_skirtor=p["tv_skirtor"],
            agn_torus_frac=p["torus_frac"],
        )

        # Integrate to bolometric luminosity
        from tengri.components.agn._phys import bolometric_integral_nu, wavelength_to_nu

        nu = wavelength_to_nu(wave)
        L_torus = bolometric_integral_nu(sed_torus, nu)

        # Add to intrinsic SED
        sed_out = sed_in + sed_torus

        return sed_out, {"L_agn_torus": L_torus}
