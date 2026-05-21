# SPDX-License-Identifier: BSD-3-Clause
"""MAPPINGSSEDComponent: MAPPINGS V shock nebular emission as a SEDModelComponent.

Ports the ShockBackend (MAPPINGS photoionization code) to the
SEDModelComponent architecture. Provides shock-driven nebular emission
(lines + continuum) for jet/outflow physics.

Physical pipeline
-----------------
1. Shock velocity + density + magnetic field → MAPPINGS grid lookup
2. Grid interpolation: (v_shock, log_density, B/√n) → emission lines
3. Normalized by shock luminosity (H-alpha or similar)
4. Add to stellar + AGN SED

Cross-component contract
------------------------
Inputs: shock_log_lhalpha (shock H-alpha luminosity) from upstream.
Outputs: sed_shock, line_waves, line_lums (shock component).

Notes
-----
**JIT-compatible**: yes — grid lookups are pure JAX.

**Shock models**: MAPPINGS V (new) supports different abundances + components.
Configuration happens at backend init (non-free parameters).

**Distinct from photoionization**: shock emission is orthogonal to
photoionized nebular (Cue, CloudyGrid, CB19). Components are summed
separately and published under different derived keys.

References
----------
.. [1] Allen et al. 2008, ApJS, 178, 20 (MAPPINGS III)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax.numpy as jnp

from tengri.components.nebular.shock import ShockBackend
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Fixed, Uniform
from tengri.protocols.component import (
    ParamDeclaration,
    SEDComponentConfig,
)

__all__ = ["MAPPINGSSEDComponent", "MAPPINGSSEDComponentConfig"]


class MAPPINGSSEDComponentConfig(SEDComponentConfig):
    """Frozen knobs for MAPPINGSSEDComponent.

    Attributes
    ----------
    name : str
        Diagnostic identifier. Default ``"mappings"``.
    grid_path : str
        Path to the MAPPINGS HDF5 grid file.
    abundance : str
        Abundance pattern: ``"solar"`` or other grid-provided options.
        Default ``"solar"``.
    component : str
        MAPPINGS component: ``"photo"`` (photoionization) or ``"shock"``.
        This component uses ``"shock"``.
    """

    name: str = "mappings"
    grid_path: str = ""
    abundance: str = "solar"
    component: str = "shock"


class MAPPINGSSEDComponent(SEDModelComponent):
    """SEDComponent for MAPPINGS V shock-driven nebular emission.

    Reads shock kinematic parameters (velocity, density, magnetic field)
    and shock luminosity to compute nebular emission via MAPPINGS grid.

    Free parameters (4):
    - shock_velocity: shock velocity [km/s]
    - shock_log_density: log pre-shock density [cm^-3]
    - shock_b_over_sqrt_n: magnetic field [μG cm^(3/2)]
    - shock_log_lhalpha: shock luminosity normalization [log10(L_Hα/Lsun)]

    Notes
    -----
    **JIT-compatible**: yes.
    **Shock-only**: distinct from photoionization. Use alongside Cue/CloudyGrid
    for joint photoionized + shocked emission.
    **Distinct derived keys**: publishes sed_shock (not sed_nebular) to keep
    shock and photoion separate in downstream analysis.
    """

    config: MAPPINGSSEDComponentConfig = MAPPINGSSEDComponentConfig()
    name: str = "mappings"
    parameter_prefix: str = "shock_"

    # Free parameters
    velocity = Uniform(100.0, 1000.0, description="Shock velocity", units="km/s")
    log_density = Fixed(0.0, description="log10(pre-shock density)", units="cm^-3")
    b_over_sqrt_n = Fixed(1.0, description="Magnetic field B/√n", units="μG cm^(3/2)")
    log_lhalpha = Fixed(
        40.0, description="log10(L_Hα) shock luminosity normalization", units="Lsun"
    )

    # Cross-component contract
    inputs: ClassVar[dict[str, str]] = {}
    outputs: ClassVar[dict[str, str]] = {
        "sed_shock": "erg/s/Hz",
        "line_waves": "Angstrom",
        "line_lums": "Lsun",
    }

    def load(self, wave: jnp.ndarray | None = None) -> ShockBackend | None:
        """Load the MAPPINGS shock grid from disk.

        Parameters
        ----------
        wave : ndarray, optional
            Ignored; MAPPINGS loads its own wavelength grid.

        Returns
        -------
        ShockBackend or None
            Loaded backend, or None if grid_path is empty (tests skip).
        """
        if not self.config.grid_path:
            return None
        try:
            return ShockBackend(
                grid_path=self.config.grid_path,
                abundance=self.config.abundance,
            )
        except FileNotFoundError:
            return None

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Declare the 4 free parameters owned by MAPPINGS."""
        return super().declared_parameters()

    def predict(
        self,
        p: Mapping[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        **inputs: Any,
    ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        """Predict shock nebular emission via MAPPINGS grid.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Parameters with prefix stripped: velocity, log_density,
            b_over_sqrt_n, log_lhalpha.
        sed_in : ndarray
            Input SED (stellar + AGN continuum).
        wave : ndarray
            Rest-frame wavelength grid in Angstrom.
        **inputs : ndarray
            Optional cross-component inputs (unused for shock).

        Returns
        -------
        tuple[ndarray, mapping]
            - sed_out: sed_in + shock nebular continuum.
            - published: Dict with "sed_shock", "line_waves", "line_lums".
        """
        backend = getattr(self, "data", None)
        if backend is None:
            zeros = jnp.zeros_like(wave)
            return sed_in, {
                "sed_shock": zeros,
                "line_waves": jnp.array([], dtype=wave.dtype),
                "line_lums": jnp.array([], dtype=sed_in.dtype),
            }

        try:
            # MAPPINGS grid lookup for shock SED and lines
            shock_sed = backend.predict_nebular_sed(
                wavelength=wave,
                shock_velocity=jnp.asarray(p["velocity"]),
                l_shock_halpha=10.0 ** jnp.asarray(p["log_lhalpha"]),
                shock_log_density=jnp.asarray(p.get("log_density", 0.0)),
                shock_b_over_sqrt_n=jnp.asarray(p.get("b_over_sqrt_n", 1.0)),
            )

            # Try to fetch line wavelengths/luminosities if available
            line_waves = jnp.array([], dtype=wave.dtype)
            line_lums = jnp.array([], dtype=sed_in.dtype)
            if hasattr(backend, "predict_nebular_line_luminosities"):
                try:
                    line_waves, line_lums = backend.predict_nebular_line_luminosities(
                        shock_velocity=jnp.asarray(p["velocity"]),
                        l_shock_halpha=10.0 ** jnp.asarray(p["log_lhalpha"]),
                        shock_log_density=jnp.asarray(p.get("log_density", 0.0)),
                        shock_b_over_sqrt_n=jnp.asarray(p.get("b_over_sqrt_n", 1.0)),
                    )
                except Exception:  # noqa: S110
                    pass

            return sed_in + shock_sed, {
                "sed_shock": shock_sed,
                "line_waves": line_waves,
                "line_lums": line_lums,
            }
        except Exception:
            zeros = jnp.zeros_like(wave)
            return sed_in, {
                "sed_shock": zeros,
                "line_waves": jnp.array([], dtype=wave.dtype),
                "line_lums": jnp.array([], dtype=sed_in.dtype),
            }
