# SPDX-License-Identifier: BSD-3-Clause
"""CAT3D-Wind clumpy-disc-plus-polar-wind torus (Hönig & Kishimoto 2017)
SEDModelComponent.

Implements the CAT3D-Wind torus model on the SEDModelComponent contract,
enabling use of three-parameter clumpy torus templates in the model-building
API.

This is an opt-in adapter, the existing AGNSEDComponent continues to
support CAT3D through the unified AGN registry.

References
----------
.. [1] S. F. Hönig & M. Kishimoto, "Dusty winds in active galactic nuclei: reconciling
   observations with models," ApJL 838,
   L20 (2017). arXiv:1702.08691.
.. [2] L. N. Martínez-Ramírez, G. Calistro Rivera, E. Lusso, et al.,
   "AGNfitter-rx: Modeling the radio-to-X-ray spectral energy
   distributions of AGNs," A&A 688, A46 (2024). arXiv:2405.12111.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import jax.numpy as jnp

from tengri.components.agn.cat3d_wind import create_cat3d_wind_from_grid
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Uniform
from tengri.protocols.component import SEDComponentConfig, SEDComponentState

__all__ = ["CAT3DTorus"]


@dataclass(frozen=True)
class CAT3DTorusConfig(SEDComponentConfig):
    """Configuration for CAT3D-Wind torus templates.

    Parameters
    ----------
    grid_path : str or None
        Path to CAT3D-Wind template grid (HDF5). If None, templates
        are not pre-loaded (deferred to first use in predict).
    """

    grid_path: str | None = None


@dataclass(frozen=True)
class CAT3DTorusState(SEDComponentState):
    """Cached CAT3D template data.

    Attributes
    ----------
    name : str
        Component identifier.
    cat3d_fn : callable or None
        Compiled interpolation function from create_cat3d_wind_from_grid,
        or None if templates are not available.
    """

    name: str = "cat3d_wind"
    cat3d_fn: Any | None = None


@dataclass(frozen=True)
class CAT3DTorus(SEDModelComponent):
    """CAT3D-Wind clumpy-disc-plus-polar-wind AGN torus.

    Three-parameter torus model with inclination, clump distribution,
    and polar-wind mass fraction. Provides C²-continuous gradients via
    triweight kernel interpolation over the 3D parameter space.

    Attributes
    ----------
    name : str
        Component registry key: ``"cat3d_wind"``.
    parameter_prefix : str
        Parameter namespace: ``"agn_"``.
    config : CAT3DTorusConfig
        Frozen configuration (grid path).

    Free parameters (class-level declarations, auto-discovered)
    -----------------------------------------------------------
    log_lbol : Uniform
        log₁₀(L_bol / L_sun). [dex, 8–14]
    cos_inc : Uniform
        Cosine of inclination (1 = face-on, 0 = edge-on). [dimensionless, 0–1]
    a_cat3d : Uniform
        Radial power-law index of clump distribution. [dimensionless, -2.5–-0.5]
    fwd_cat3d : Uniform
        Polar-wind mass fraction. [dimensionless, 0–1]
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

    **Requires template grid**: The CAT3D-Wind template library must be
    downloaded separately and pointed to via ``grid_path`` in config.
    The predict method gracefully returns zero emission if templates
    are unavailable.

    **Inclination parameterization**: Uses cosine of inclination (not degrees)
    to align with SKIRTOR and other components.

    Examples
    --------
    Minimal model with CAT3D-Wind torus::

        from tengri import SEDModel, Fixed, Uniform, builders
        from tengri.components.agn.cat3d_torus_model import CAT3DTorus, CAT3DTorusConfig

        model = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh=builders.sfh.dpl(_=Fixed(1.5), beta=Fixed(1.0)),
            dust_attenuation={"type": "two_component", "all_params": Fixed},
            agn=CAT3DTorus(config=CAT3DTorusConfig(grid_path="path/to/grid.h5")),
        )

    See Also
    --------
    tengri.components.agn.cat3d_wind : template loader and interpolation.
    """

    name = "cat3d_wind"
    parameter_prefix = "agn_"
    config: CAT3DTorusConfig = field(default_factory=CAT3DTorusConfig)

    # Free parameters: auto-discovered
    log_lbol = Uniform(
        8.0,
        14.0,
        description="AGN bolometric luminosity",
        units="dex (L_sun)",
        default=11.0,
    )
    cos_inc = Uniform(
        0.0,
        1.0,
        description="Cosine of inclination",
        units="dimensionless",
        default=0.45,
    )
    a_cat3d = Uniform(
        -2.5,
        -0.5,
        description="Radial power-law index of clump distribution",
        units="dimensionless",
        default=-0.5,
    )
    fwd_cat3d = Uniform(
        0.0,
        1.0,
        description="Polar-wind mass fraction",
        units="dimensionless",
        default=0.4,
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
        """Load CAT3D-Wind template grid if available.

        Parameters
        ----------
        wave : ndarray, optional
            Rest-frame wavelength grid (not used by CAT3D; templates
            interpolate to any target grid).

        Returns
        -------
        callable or None
            Interpolation function from create_cat3d_wind_from_grid, or None
            if template file is not available or grid_path is not set.
        """
        if not self.config.grid_path:
            return None

        try:
            return create_cat3d_wind_from_grid(self.config.grid_path)
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
        """Pure JAX CAT3D-Wind torus prediction.

        Interpolates the CAT3D-Wind template grid to the requested inclination,
        clump distribution, and wind fraction; normalizes to the user's
        luminosity scale; and returns the torus SED contribution.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Parameters with prefix already stripped:

            - log_lbol: log₁₀(L_bol / L_sun)
            - cos_inc: cosine of inclination
            - a_cat3d: radial power-law index
            - fwd_cat3d: polar-wind mass fraction
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

        cat3d_fn = self.data

        # Call CAT3D-Wind interpolator
        sed_torus = cat3d_fn(
            wavelength=wave,
            agn_log_lbol=p["log_lbol"],
            agn_cos_inc=p["cos_inc"],
            agn_a_cat3d=p["a_cat3d"],
            agn_fwd_cat3d=p["fwd_cat3d"],
            agn_torus_frac=p["torus_frac"],
        )

        # Integrate to bolometric luminosity
        from tengri.components.agn._phys import bolometric_integral_nu, wavelength_to_nu

        nu = wavelength_to_nu(wave)
        L_torus = bolometric_integral_nu(sed_torus, nu)

        # Add to intrinsic SED
        sed_out = sed_in + sed_torus

        return sed_out, {"L_agn_torus": L_torus}
