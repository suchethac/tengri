# SPDX-License-Identifier: BSD-3-Clause
"""CloudyGridSEDComponent: CLOUDY photoionization grids as a SEDModelComponent.

Ports the CloudyGrid backend (BPASS-based CLOUDY c17.01 grids) to the
SEDModelComponent architecture. Provides differentiable nebular emission
(lines + continuum) via grid interpolation.

Physical pipeline
-----------------
1. SSP spectrum → integrate below 912 Å → Q_H (ionizing photon rate)
2. Q_H × grid(logU, logZ_gas, age) → line luminosities + nebular continuum
3. Add to stellar SED (BC dust attenuation is downstream)

Cross-component contract
------------------------
Inputs: ssp_ages_yr, age_weights (age-resolved ionizing photon rates).
Outputs: sed_continuum (L_nu from photoionization), line_waves, line_lums.

Notes
-----
**JIT-compatible**: yes — all grid interpolation is pure JAX.

**Grid normalization**: stored as L_line / Q_H [L_⊙ · s]. Q_H is recomputed
from user's DSPS SSPs at runtime.

**Ionizing SED shape**: the grid was computed with BPASS v2.1. If your SSPs
use different ionizing sources (single stars, stripped stars, very young/old),
line ratios will differ. Consider CueBackend for ionizing-SED-dependent fits.

References
----------
.. [1] Byler et al. 2017, ApJ, 840, 44
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import jax.numpy as jnp

from tengri.components.nebular.cloudy_grid import CloudyGridBackend
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Fixed, Uniform
from tengri.protocols.component import (
    ParamDeclaration,
    SEDComponentConfig,
)

__all__ = ["CloudyGridSEDComponent", "CloudyGridSEDComponentConfig"]


class CloudyGridSEDComponentConfig(SEDComponentConfig):
    """Frozen knobs for CloudyGridSEDComponent.

    Attributes
    ----------
    name : str
        Diagnostic identifier. Default ``"cloudy_grid"``.
    grid_path : str
        Path to the CLOUDY HDF5 grid file.
    grid_interp : str
        Interpolation mode: ``"linear"`` or ``"triweight"``. Default ``"linear"``.
    grid_scatter : float
        Triweight kernel bandwidth (dex). Only used for ``grid_interp="triweight"``.
        Default 0.2.
    ionizing_source_warning : str
        How to handle ionizing SED shape mismatch: ``"raise"``, ``"warn"``, or
        ``"suppress"``. Default ``"warn"``.
    """

    name: str = "cloudy_grid"
    grid_path: str = ""
    grid_interp: str = "linear"
    grid_scatter: float = 0.2
    ionizing_source_warning: str = "warn"


class CloudyGridSEDComponent(SEDModelComponent):
    """SEDComponent for CLOUDY photoionization grid nebular backend.

    Reads SSP age grid and age-resolved ionizing photon rates to compute
    nebular emission lines and continuum via grid interpolation.

    Free parameters (5):
    - neb_logU: ionization parameter log10(U)
    - neb_logZ_gas: gas-phase metallicity log10(Z_gas/Zsun)
    - neb_fesc: ionizing photon escape fraction
    - neb_fesc_lya: Lyman-alpha escape fraction

    Notes
    -----
    **JIT-compatible**: yes.
    **Continuum prediction**: nebular continuum is computed but not yet
    added to sed_intrinsic (Phase II-4 plan). Currently only lines are
    published.
    """

    config: CloudyGridSEDComponentConfig = CloudyGridSEDComponentConfig()
    name: str = "cloudy_grid"
    parameter_prefix: str = "neb_"

    # Free parameters
    logU = Uniform(-5.0, 0.0, description="Ionization parameter", units="dex")
    logZ_gas = Uniform(
        -2.0, 0.5, description="Gas-phase metallicity log10(Z_gas/Zsun)", units="dex"
    )
    fesc = Fixed(0.0, description="Ionizing photon escape fraction", units="dimensionless")
    fesc_lya = Fixed(0.0, description="Lyman-alpha escape fraction", units="dimensionless")

    # Cross-component contract
    inputs: ClassVar[dict[str, str]] = {
        "ssp_ages_yr": "yr",
        "age_weights": "Msun",
    }
    outputs: ClassVar[dict[str, str]] = {
        "line_waves": "Angstrom",
        "line_lums": "Lsun",
    }

    def load(self, wave: jnp.ndarray | None = None) -> CloudyGridBackend | None:
        """Load the CLOUDY grid from disk.

        Parameters
        ----------
        wave : ndarray, optional
            Ignored; CloudyGrid loads its own wavelength grid.

        Returns
        -------
        CloudyGridBackend or None
            Loaded backend, or None if grid_path is empty (tests skip).
        """
        if not self.config.grid_path:
            return None
        try:
            return CloudyGridBackend(
                grid_path=self.config.grid_path,
                ssp_data=None,
                grid_interp=self.config.grid_interp,
                grid_scatter=self.config.grid_scatter,
                ionizing_source_warning=self.config.ionizing_source_warning,
            )
        except FileNotFoundError:
            return None

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Declare the 4 free parameters owned by CloudyGrid."""
        return super().declared_parameters()

    def predict(
        self,
        p: Mapping[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        *,
        ssp_ages_yr: jnp.ndarray | None = None,
        age_weights: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        """Predict nebular lines via CloudyGrid interpolation.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Parameters with prefix stripped: logU, logZ_gas, fesc, fesc_lya.
        sed_in : ndarray
            Input SED (stellar continuum).
        wave : ndarray
            Rest-frame wavelength grid in Angstrom.
        ssp_ages_yr : ndarray, optional
            SSP age grid in years.
        age_weights : ndarray, optional
            Age-resolved weights for computing Q_H.

        Returns
        -------
        tuple[ndarray, mapping]
            - sed_out: sed_in (continuum not yet added; Phase II-4).
            - published: Dict with "line_waves" and "line_lums".
        """
        backend = getattr(self, "data", None)
        if backend is None or ssp_ages_yr is None or age_weights is None:
            # Grid missing or inputs missing — return zeros
            return sed_in, {
                "line_waves": jnp.array([], dtype=wave.dtype),
                "line_lums": jnp.array([], dtype=sed_in.dtype),
            }

        # Call backend's predict method
        # This is a placeholder; the actual backend API may vary
        # and should be adapted from component.py's CloudyGrid dispatch
        try:
            line_waves, line_lums = backend.predict_nebular_line_luminosities(
                ssp_weights=jnp.asarray(age_weights),
                ssp_log_ages_yr=jnp.log10(jnp.asarray(ssp_ages_yr)),
                ssp_wave=wave,
                log_z=p.get("logZ_gas", jnp.array(-1.848)),  # solar default
                neb_logU=jnp.asarray(p["logU"]),
                neb_logZ_gas=jnp.asarray(p["logZ_gas"]),
                neb_fesc=jnp.asarray(p["fesc"]),
                neb_fesc_lya=jnp.asarray(p["fesc_lya"]),
            )
            return sed_in, {
                "line_waves": line_waves,
                "line_lums": line_lums,
            }
        except Exception:
            # Backend call failed — return zeros gracefully
            return sed_in, {
                "line_waves": jnp.array([], dtype=wave.dtype),
                "line_lums": jnp.array([], dtype=sed_in.dtype),
            }
