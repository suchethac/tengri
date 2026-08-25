# SPDX-License-Identifier: BSD-3-Clause
"""Silva+04 semi-empirical smooth torus library (Silva et al. 2004)
SEDModelComponent.

Implements the Silva+04 torus model on the SEDModelComponent contract,
enabling use of column-density-parameterized templates in the model-building
API.

This is an opt-in adapter, the existing AGNSEDComponent continues to
support Silva+04 through the unified AGN registry.

References
----------
.. [1] L. Silva, R. Maiolino & G. L. Granato, "The nature of the
   Compton-thick AGN in NGC 1068 and implications for the cosmic
   X-ray background," MNRAS 355, 973 (2004). arXiv:astro-ph/0403425.
.. [2] G. Calistro Rivera et al., "AGNfitter: a Bayesian MCMC approach to
   fitting spectral energy distributions of AGNs," ApJ 833, 98 (2016).
   arXiv:1606.05648.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import jax.numpy as jnp

from tengri.components.agn.silva04 import create_silva04_from_grid
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Uniform
from tengri.protocols.component import SEDComponentConfig, SEDComponentState

__all__ = ["Silva04Torus"]


@dataclass(frozen=True)
class Silva04TorusConfig(SEDComponentConfig):
    """Configuration for Silva+04 torus templates.

    Parameters
    ----------
    grid_path : str or None
        Path to Silva+04 template grid (HDF5). If None, templates
        are not pre-loaded (deferred to first use in predict).
    """

    grid_path: str | None = None


@dataclass(frozen=True)
class Silva04TorusState(SEDComponentState):
    """Cached Silva+04 template data.

    Attributes
    ----------
    name : str
        Component identifier.
    silva04_fn : callable or None
        Compiled interpolation function from create_silva04_from_grid,
        or None if templates are not available.
    """

    name: str = "silva04"
    silva04_fn: Any | None = None


@dataclass(frozen=True)
class Silva04Torus(SEDModelComponent):
    """Silva, Maiolino & Granato (2004) smooth AGN torus.

    One-parameter semi-empirical torus library keyed on hydrogen column
    density. Provides C²-continuous gradients via triweight kernel
    interpolation. Requires a prior download of the template grid.

    Attributes
    ----------
    name : str
        Component registry key: ``"silva04"``.
    parameter_prefix : str
        Parameter namespace: ``"agn_"``.
    config : Silva04TorusConfig
        Frozen configuration (grid path).

    Free parameters (class-level declarations, auto-discovered)
    -----------------------------------------------------------
    log_lbol : Uniform
        log₁₀(L_bol / L_sun). [dex, 8–14]
    log_nh_silva : Uniform
        log₁₀(N_H / cm^-2), hydrogen column density. [dex, 22–25]
    torus_frac : Uniform
        Fraction of L_bol reprocessed by torus. [dimensionless, 0–1]

    Cross-component outputs
    -----------------------
    L_agn_torus : erg/s
        Bolometric luminosity contribution from torus emission.

    Notes
    -----
    **JIT-compatible**: yes, predict() is pure JAX.

    **Gradient-safe**: yes, triweight interpolation is fully differentiable.

    **Requires template grid**: The Silva+04 template library must be
    downloaded separately and pointed to via ``grid_path`` in config.
    The predict method gracefully returns zero emission if templates
    are unavailable.

    **References**: Uses template data from AGNfitter (Calistro Rivera et al. 2016)
    via scripts/build_silva04_grid.py. Original templates from
    Silva, Maiolino & Granato (2004).

    Examples
    --------
    Minimal model with Silva+04 torus::

        from tengri import SEDModel, Fixed, Uniform, builders
        from tengri.components.agn.silva04_model import Silva04Torus, Silva04TorusConfig

        model = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh=builders.sfh.dpl(_=Fixed(1.5), beta=Fixed(1.0)),
            dust_attenuation={"type": "two_component", "all_params": Fixed},
            agn=Silva04Torus(config=Silva04TorusConfig(grid_path="path/to/grid.h5")),
        )

    See Also
    --------
    tengri.components.agn.silva04 : template loader and interpolation.
    """

    name = "silva04"
    parameter_prefix = "agn_"
    config: Silva04TorusConfig = field(default_factory=Silva04TorusConfig)

    # Free parameters: auto-discovered
    log_lbol = Uniform(
        8.0,
        14.0,
        description="AGN bolometric luminosity",
        units="dex (L_sun)",
        default=11.0,
    )
    log_nh_silva = Uniform(
        22.0,
        25.0,
        description="Hydrogen column density (Silva et al.)",
        units="dex (cm^-2)",
        default=23.5,
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
        """Load Silva+04 template grid if available.

        Parameters
        ----------
        wave : ndarray, optional
            Rest-frame wavelength grid (not used by Silva+04; templates
            interpolate to any target grid).

        Returns
        -------
        callable or None
            Interpolation function from create_silva04_from_grid, or None
            if template file is not available or grid_path is not set.
        """
        if not self.config.grid_path:
            return None

        try:
            return create_silva04_from_grid(self.config.grid_path)
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
        """Pure JAX Silva+04 torus prediction.

        Interpolates the Silva+04 template grid to the requested column
        density, normalizes to the user's luminosity scale, and returns
        the torus SED contribution.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Parameters with prefix already stripped:

            - log_lbol: log₁₀(L_bol / L_sun)
            - log_nh_silva: log₁₀(N_H / cm^-2)
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

        silva04_fn = self.data

        # Call Silva+04 interpolator
        sed_torus = silva04_fn(
            wavelength=wave,
            agn_log_lbol=p["log_lbol"],
            agn_log_nh_silva=p["log_nh_silva"],
            agn_torus_frac=p["torus_frac"],
        )

        # Integrate to bolometric luminosity
        from tengri.components.agn._phys import bolometric_integral_nu, wavelength_to_nu

        nu = wavelength_to_nu(wave)
        L_torus = bolometric_integral_nu(sed_torus, nu)

        # Add to intrinsic SED
        sed_out = sed_in + sed_torus

        return sed_out, {"L_agn_torus": L_torus}
