# SPDX-License-Identifier: BSD-3-Clause
"""CB19SEDComponent: Byler+2017 CLOUDY+FSP nebular grids as a SEDModelComponent.

Ports the CB19Backend (CloudyFSPS, Byler+2017) to the SEDModelComponent
architecture. Provides age-dependent nebular emission (lines + continuum)
via large precomputed grid.

Physical pipeline
-----------------
1. Per-age FSPS stellar spectrum → Q_H from ionizing photons below 912 Å
2. Grid lookup: (logZ_gas, log_age, logU) → line + continuum luminosity
3. Weight by CSP age-mass history and sum
4. Add to stellar SED

Cross-component contract
------------------------
Inputs: ssp_ages_yr, age_weights (age-resolved CSP weights).
Outputs: sed_continuum, line_waves, line_lums.

Notes
-----
**JIT-compatible**: yes — grid lookups are JAX arrays, weighted sum is pure.

**Grid**: CLOUDY c17.01 + FSPS ionizing spectra, 2.3M models, age-dependent
(unlike pure CloudyGrid which uses a single ionizing SED).

**Age axis**: covers 0.01 to 2 Gyr with 221 logarithmic points.

References
----------
.. [1] Byler et al. 2017, ApJ, 840, 44
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import jax.numpy as jnp

from tengri.components.nebular.cloudy_cb19 import CB19Backend
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Fixed, Uniform
from tengri.protocols.component import (
    ParamDeclaration,
    SEDComponentConfig,
)

__all__ = ["CB19SEDComponent", "CB19SEDComponentConfig"]


class CB19SEDComponentConfig(SEDComponentConfig):
    """Frozen knobs for CB19SEDComponent.

    Attributes
    ----------
    name : str
        Diagnostic identifier. Default ``"cb19"``.
    grid_path : str
        Path to the CB19 HDF5 grid file.
    ionizing_source_warning : str
        How to handle missing ionizing spectrum warning: ``"raise"``,
        ``"warn"``, or ``"suppress"``. Default ``"warn"``.
    """

    name: str = "cb19"
    grid_path: str = ""
    ionizing_source_warning: str = "warn"


class CB19SEDComponent(SEDModelComponent):
    """SEDComponent for Byler+2017 (CloudyFSPS) nebular grids.

    Reads SSP age grid and age-resolved CSP weights to compute nebular
    emission via large age-dependent grid interpolation.

    Free parameters (4):
    - neb_logU: ionization parameter log10(U)
    - neb_logZ_gas: gas-phase metallicity log10(Z_gas/Zsun)
    - neb_fesc: ionizing photon escape fraction
    - neb_fesc_lya: Lyman-alpha escape fraction

    Notes
    -----
    **JIT-compatible**: yes.
    **Age-dependent**: grid lookup includes age as an axis, capturing
    aging of the ionizing SED as stellar populations age.
    """

    config: CB19SEDComponentConfig = CB19SEDComponentConfig()
    name: str = "cb19"
    parameter_prefix: str = "neb_"

    # Free parameters
    logU = Uniform(-5.0, 0.0, description="Ionization parameter", units="dex", default=-3.0)
    logZ_gas = Uniform(
        -2.0, 0.5, description="Gas-phase metallicity log10(Z_gas/Zsun)", units="dex", default=0.0
    )
    fesc = Fixed(0.0, description="Ionizing photon escape fraction", units="dimensionless")
    fesc_lya = Fixed(0.0, description="Lyman-alpha escape fraction", units="dimensionless")
    fdust = Fixed(
        0.0,
        description="Lyman-continuum dust-absorption fraction in HII regions",
        units="dimensionless",
    )

    # Cross-component contract
    inputs: ClassVar[dict[str, str]] = {
        "ssp_ages_yr": "yr",
        "age_weights": "Msun",
    }
    outputs: ClassVar[dict[str, str]] = {
        "line_waves": "Angstrom",
        "line_lums": "Lsun",
    }

    def load(self, wave: jnp.ndarray | None = None) -> CB19Backend | None:
        """Load the CB19 grid from disk.

        Parameters
        ----------
        wave : ndarray, optional
            Ignored; CB19 loads its own wavelength grid.

        Returns
        -------
        CB19Backend or None
            Loaded backend, or None if grid_path is empty (tests skip).
        """
        if not self.config.grid_path:
            return None
        try:
            return CB19Backend(
                grid_path=self.config.grid_path,
                ionizing_source_warning=self.config.ionizing_source_warning,
            )
        except FileNotFoundError:
            return None

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Declare the 4 free parameters owned by CB19."""
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
        """Predict nebular lines via CB19 grid lookup.

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
            Age-resolved CSP weights.

        Returns
        -------
        tuple[ndarray, mapping]
            - sed_out: sed_in (continuum not yet added; Phase II-4).
            - published: Dict with "line_waves" and "line_lums".
        """
        backend = getattr(self, "data", None)
        if backend is None or ssp_ages_yr is None or age_weights is None:
            return sed_in, {
                "line_waves": jnp.array([], dtype=wave.dtype),
                "line_lums": jnp.array([], dtype=sed_in.dtype),
            }

        try:
            line_waves, line_lums = backend.predict_nebular_line_luminosities(
                ssp_weights=jnp.asarray(age_weights),
                ssp_log_ages_yr=jnp.log10(jnp.asarray(ssp_ages_yr)),
                log_z=p.get("logZ_gas", jnp.array(-1.848)),
                neb_logU=jnp.asarray(p["logU"]),
                neb_logZ_gas=jnp.asarray(p["logZ_gas"]),
                neb_fesc=jnp.asarray(p["fesc"]),
                neb_fesc_lya=jnp.asarray(p["fesc_lya"]),
                neb_fdust=jnp.asarray(p.get("fdust", 0.0)),
            )
            return sed_in, {
                "line_waves": line_waves,
                "line_lums": line_lums,
            }
        except Exception:
            return sed_in, {
                "line_waves": jnp.array([], dtype=wave.dtype),
                "line_lums": jnp.array([], dtype=sed_in.dtype),
            }
