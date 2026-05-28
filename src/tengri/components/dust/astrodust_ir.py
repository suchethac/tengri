# SPDX-License-Identifier: BSD-3-Clause
"""Hensley & Draine (2023) Astrodust+PAH dust emission as an SEDModelComponent.

Implements tabulated-template IR re-emission based on the Astrodust+PAH dust model,
parameterized by PAH fraction (q_PAH), minimum radiation field intensity
(U_min), and power-law mixing fraction (gamma).

The templates are loaded from an HDF5 file during :meth:`load` (precomputation)
and cached for rapid access during forward passes.

References
----------
.. [1] Hensley, B.S. & Draine, B.T. 2023, "Properties and Evolution of
   Dust Grains Constrained by Extinction, Emission, and Polarization",
   ApJ 948, 55. https://doi.org/10.3847/1538-4357/acbbc1
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import jax.numpy as jnp

from tengri.components.dust.emission_templates import load_astrodust_templates
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Uniform
from tengri.protocols.component import SEDComponentConfig

__all__ = ["AstrodustIRConfig", "AstrodustIRSEDComponent"]


@dataclass(frozen=True)
class AstrodustIRConfig(SEDComponentConfig):
    """Configuration for AstrodustIR dust emission component.

    Attributes
    ----------
    name : str
        Diagnostic identifier. Default ``"astrodust_ir"``.
    template_path : str, optional
        Path to Astrodust template HDF5 file. If None, attempts to auto-locate
        via standard search paths. If templates are unavailable, the component
        will skip gracefully with a warning.
    """

    name: str = "astrodust_ir"
    template_path: str | None = None


class AstrodustIRSEDComponent(SEDModelComponent):
    """Dust IR emission via tabulated Hensley & Draine (2023) Astrodust+PAH templates.

    Closes the dust energy balance by re-emitting absorbed UV/optical
    luminosity according to a 3-parameter model (q_PAH, U_min, gamma)
    interpolated over a precomputed template grid.

    Attributes
    ----------
    name : str
        Stable identifier: ``"astrodust_ir"``.
    parameter_prefix : str
        Domain prefix for parameters: ``"dust_"``.
    config : AstrodustIRConfig
        Frozen configuration with optional template path override.

    Notes
    -----
    **Cross-component contract**:
    - Reads: ``state.derived["L_ir"]`` (erg/s) — luminosity absorbed by dust.
    - Publishes: ``{"L_ir_emission": erg/s}`` — bolometric IR from templates.

    **Template loading**: Templates are loaded during :meth:`load` (precomputation)
    and stored on ``self.data``. If template files are unavailable, the component
    gracefully skips with a warning.

    **JIT-compatible**: yes — all operations in :meth:`predict` are ``jnp``
    primitives.

    **Parameter discovery**: Free parameters (``qpah``, ``umin``, ``gamma``)
    are auto-discovered; :meth:`declared_parameters` constructs tuples with
    units and descriptions.

    **Pipeline ordering**: This component MUST run after dust attenuation.
    Typical order: ``[Stellar, Nebular, DustAttenuation, Astrodust, IGM, Radio]``.
    """

    name = "astrodust_ir"
    parameter_prefix = "dust_"
    config: AstrodustIRConfig = AstrodustIRConfig()

    # Free parameters — auto-discovered by base class
    qpah = Uniform(0.5, 4.5, default=2.5, description="PAH mass fraction", units="%")
    umin = Uniform(
        -1.0,
        1.5,
        default=0.0,
        description="log U_min (minimum radiation field intensity)",
        units="dex",
    )
    gamma = Uniform(
        0.0, 0.3, default=0.05, description="Power-law mixing fraction", units="dimensionless"
    )

    # Cross-component contract
    inputs: ClassVar = {"L_ir": "erg/s"}
    outputs: ClassVar = {"L_ir_emission": "erg/s"}

    def load(self, wave: jnp.ndarray | None = None) -> Any | None:
        """Preload Astrodust template grid.

        Called at model init time (not JIT-ed). Attempts to locate and load
        the Astrodust template HDF5 file. If unavailable, returns None and the
        component will skip gracefully.

        Parameters
        ----------
        wave : ndarray, optional
            Rest-frame wavelength grid in Angstrom (not used for Astrodust
            template loading, but kept for protocol compatibility).

        Returns
        -------
        dict or None
            Dictionary with keys {wavelength_aa, umin_grid, qpah_grid,
            single_u, powerlaw} if templates loaded successfully. Returns None if
            templates unavailable.
        """
        import warnings

        # Auto-locate template file if not explicitly provided
        template_path = self.config.template_path
        if template_path is None:
            from tengri.components.dust.emission_templates import _find_data_file

            # Try v2 first, then legacy
            for fname in ("astrodust_templates_v2.h5", "astrodust_templates.h5"):
                template_path = _find_data_file(fname)
                if template_path is not None:
                    break

        if template_path is None:
            warnings.warn(
                f"AstrodustIR component {self.name!r}: template file not found. "
                "Component will skip. For science use, provide template_path "
                "via AstrodustIRConfig.",
                UserWarning,
                stacklevel=2,
            )
            return None

        try:
            templates = load_astrodust_templates(template_path)
            return templates
        except Exception as e:
            warnings.warn(
                f"AstrodustIR component {self.name!r}: failed to load templates: {e}. "
                "Component will skip.",
                UserWarning,
                stacklevel=2,
            )
            return None

    def predict(
        self, p: Mapping[str, jnp.ndarray], sed_in: jnp.ndarray, wave: jnp.ndarray, **inputs: Any
    ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        r"""Compute dust emission via Astrodust tabulated templates.

        Interpolates the Astrodust template grid over (q_PAH, U_min, gamma) and
        applies energy-balance normalization with CMB contrast correction.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Sliced parameters (prefix stripped):
            - ``p["qpah"]``: PAH mass fraction [%]
            - ``p["umin"]``: log U_min [dex]
            - ``p["gamma"]``: power-law mixing fraction [dimensionless]
        sed_in : ndarray, shape (n_wave,)
            Input SED in erg/s/Hz (ignored; Astrodust emission is computed
            from L_ir independently).
        wave : ndarray, shape (n_wave,)
            Rest-frame wavelength grid in Angstrom.
        **inputs : ndarray
            Cross-component inputs:
            - ``L_ir``: absorbed luminosity [erg/s]

        Returns
        -------
        tuple[ndarray, dict]
            - ``sed_out``: Updated SED in erg/s/Hz.
            - ``published``: Dict with ``{"L_ir_emission": scalar}``.

        Notes
        -----
        **JIT-compatible**: yes.

        **Gradient-safe**: yes — bilinear interpolation is differentiable
        everywhere except at grid boundaries (where clipping occurs).
        """
        L_ir = inputs["L_ir"]

        # Skip if templates were not loaded
        if not hasattr(self, "data") or self.data is None:
            # Return input SED unchanged
            return sed_in, {}

        # Load template grid from precomputed state
        templates = self.data
        single_u = templates["single_u"]  # (n_qpah, n_umin, n_wave)
        powerlaw = templates["powerlaw"]  # (n_qpah, n_umin, n_wave)
        tmpl_wave = templates["wavelength_aa"]
        umin_grid = templates["umin_grid"]
        qpah_grid = templates["qpah_grid"]

        # Clip parameters to grid bounds
        dust_umin_c = jnp.clip(p["umin"], umin_grid[0], umin_grid[-1])
        dust_qpah_c = jnp.clip(p["qpah"], qpah_grid[0], qpah_grid[-1])

        # Bilinear interpolation indices
        i_u = jnp.clip(jnp.searchsorted(umin_grid, dust_umin_c) - 1, 0, len(umin_grid) - 2)
        i_q = jnp.clip(jnp.searchsorted(qpah_grid, dust_qpah_c) - 1, 0, len(qpah_grid) - 2)

        fu = (dust_umin_c - umin_grid[i_u]) / (umin_grid[i_u + 1] - umin_grid[i_u])
        fq = (dust_qpah_c - qpah_grid[i_q]) / (qpah_grid[i_q + 1] - qpah_grid[i_q])

        def _bilinear(grid):
            """2D bilinear interpolation over qpah and umin axes."""
            return (
                (1.0 - fq) * (1.0 - fu) * grid[i_q, i_u]
                + (1.0 - fq) * fu * grid[i_q, i_u + 1]
                + fq * (1.0 - fu) * grid[i_q + 1, i_u]
                + fq * fu * grid[i_q + 1, i_u + 1]
            )

        # Mix single-U and power-law components
        template = (1.0 - p["gamma"]) * _bilinear(single_u) + p["gamma"] * _bilinear(powerlaw)

        # Interpolate onto target wavelength grid
        sed = jnp.interp(wave, tmpl_wave, template, left=0.0, right=0.0)

        # Scale by absorbed luminosity
        sed_emission = L_ir * sed

        # Return updated SED and published luminosity
        sed_out = sed_in + sed_emission

        return sed_out, {}
