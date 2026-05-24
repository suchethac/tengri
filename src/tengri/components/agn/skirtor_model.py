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

from tengri.components.agn.skirtor import create_skirtor_from_grid
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
    inclination). Publishes the torus contribution to sed_intrinsic and
    bolometric luminosity to cross-component readers.

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
    torus_frac : Uniform
        Fraction of L_bol reprocessed by torus. [dimensionless, 0–1]

    Cross-component outputs
    -----------------------
    L_agn_torus : erg/s
        Bolometric luminosity contribution from torus emission.

    Notes
    -----
    **JIT-compatible**: yes — predict() is pure JAX.

    **Gradient-safe**: yes — triweight interpolation is fully differentiable.

    **Requires template grid**: The SKIRTOR template library (~1 GB) must be
    downloaded separately and pointed to via ``grid_path`` in config. The
    predict method gracefully returns zero emission if templates are unavailable.

    **Cross-component note**: AGN inclination (cos_inc) is shared with
    disc/corona models and is **never** auto-derived from geometry
    (discontinuous gradient). Declare it as a free parameter independently.

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
        3.0,
        11.0,
        description="9.7 µm optical depth (Stalevski et al.)",
        units="dimensionless",
    )
    p_skirtor = Uniform(
        0.0, 1.5, description="Radial dust density gradient", units="dimensionless"
    )
    q_skirtor = Uniform(0.0, 1.5, description="Polar dust density gradient", units="dimensionless")
    oa_skirtor = Uniform(20.0, 60.0, description="Torus half-opening angle", units="deg")
    cos_inc = Uniform(0.0, 1.0, description="Cosine of inclination", units="dimensionless")
    torus_frac = Uniform(
        0.0,
        1.0,
        description="Torus luminosity fraction of L_bol",
        units="dimensionless",
    )

    # Cross-component output
    outputs: ClassVar[dict[str, str]] = {"L_agn_torus": "erg/s"}

    def load(self, wave: jnp.ndarray | None = None) -> Any | None:
        """Load SKIRTOR template grid if available.

        Parameters
        ----------
        wave : ndarray, optional
            Rest-frame wavelength grid (not used by SKIRTOR; templates
            interpolate to any target grid).

        Returns
        -------
        callable or None
            Interpolation function from create_skirtor_from_grid, or None
            if template file is not available or grid_path is not set.
        """
        if not self.config.grid_path:
            return None

        try:
            return create_skirtor_from_grid(self.config.grid_path)
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
        """Pure JAX SKIRTOR torus prediction.

        Interpolates the SKIRTOR template grid to the requested parameters,
        normalizes to the user's luminosity scale, and returns the torus SED
        contribution.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Parameters with prefix already stripped:
            - log_lbol: log₁₀(L_bol / L_sun)
            - tau_skirtor: optical depth at 9.7 µm
            - p_skirtor: radial density gradient
            - q_skirtor: polar density gradient
            - oa_skirtor: opening angle (degrees)
            - cos_inc: cosine of inclination
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

        skirtor_fn = self.data

        # Call SKIRTOR interpolator
        sed_torus = skirtor_fn(
            wavelength=wave,
            agn_log_lbol=p["log_lbol"],
            agn_tau_skirtor=p["tau_skirtor"],
            agn_p_skirtor=p["p_skirtor"],
            agn_q_skirtor=p["q_skirtor"],
            agn_oa_skirtor=p["oa_skirtor"],
            agn_cos_inc=p["cos_inc"],
            agn_torus_frac=p["torus_frac"],
        )

        # Integrate to bolometric luminosity
        from tengri.components.agn._phys import wavelength_to_nu

        nu = wavelength_to_nu(wave)
        idx_sort = jnp.argsort(nu)
        L_torus = jnp.trapezoid(sed_torus[idx_sort], nu[idx_sort])

        # Add to intrinsic SED
        sed_out = sed_in + sed_torus

        return sed_out, {"L_agn_torus": L_torus}
