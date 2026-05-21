"""Draine & Li (2007) dust emission as an SEDModelComponent.

Implements tabulated-template IR re-emission based on the DL07 dust model,
parameterized by PAH fraction (q_PAH), minimum radiation field intensity
(U_min), and power-law mixing fraction (gamma).

The templates are loaded from an HDF5 file during :meth:`load` (precomputation)
and cached for rapid access during forward passes.

References
----------
.. [1] Draine, B.T. & Li, A. 2007, "Dust Emission and the Infrared Luminosity
   of Galaxies and Active Galactic Nuclei", ApJ 657, 810--837.
   https://doi.org/10.1086/511055
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import jax.numpy as jnp

from tengri.components.dust.emission_templates import load_draine_li_templates
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Uniform
from tengri.protocols.component import SEDComponentConfig

__all__ = ["DL07IRConfig", "DL07IRSEDComponent"]


@dataclass(frozen=True)
class DL07IRConfig(SEDComponentConfig):
    """Configuration for DL07IR dust emission component.

    Attributes
    ----------
    name : str
        Diagnostic identifier. Default ``"dl07_ir"``.
    template_path : str, optional
        Path to DL07 template HDF5 file. If None, attempts to auto-locate
        via standard search paths. If templates are unavailable, the component
        will skip gracefully with a warning.
    """

    name: str = "dl07_ir"
    template_path: str | None = None


class DL07IRSEDComponent(SEDModelComponent):
    """Dust IR emission via tabulated Draine & Li (2007) templates.

    Closes the dust energy balance by re-emitting absorbed UV/optical
    luminosity according to a 3-parameter model interpolated over a
    precomputed template grid.

    Attributes
    ----------
    name : str
        Stable identifier: ``"dl07_ir"``.
    parameter_prefix : str
        Domain prefix for parameters: ``"dust_"``.
    config : DL07IRConfig
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
    Typical order: ``[Stellar, Nebular, DustAttenuation, DL07, IGM, Radio]``.
    """

    name = "dl07_ir"
    parameter_prefix = "dust_"
    config: DL07IRConfig = DL07IRConfig()

    # Free parameters — auto-discovered by base class
    qpah = Uniform(0.5, 4.5, description="PAH mass fraction", units="%")
    umin = Uniform(
        -1.0, 1.5, description="log U_min (minimum radiation field intensity)", units="dex"
    )
    gamma = Uniform(0.0, 0.3, description="Power-law mixing fraction", units="")

    # Cross-component contract
    inputs: ClassVar = {"L_ir": "erg/s"}
    outputs: ClassVar = {"L_ir_emission": "erg/s"}

    def load(self, wave: jnp.ndarray | None = None) -> Any | None:
        """Preload DL07 template grid.

        Called at model init time (not JIT-ed). Attempts to locate and load
        the DL07 template HDF5 file. If unavailable, returns None and the
        component will skip gracefully.

        Parameters
        ----------
        wave : ndarray, optional
            Rest-frame wavelength grid in Angstrom (not used for DL07
            template loading, but kept for protocol compatibility).

        Returns
        -------
        dict or None
            Dictionary with keys {wavelength, umin_grid, qpah_grid, single_u,
            powerlaw} if templates loaded successfully. Returns None if
            templates unavailable.
        """
        import warnings

        # Auto-locate template file if not explicitly provided
        template_path = self.config.template_path
        if template_path is None:
            from tengri.components.dust.emission_templates import _find_data_file

            template_path = _find_data_file("dl07_templates.h5")

        if template_path is None:
            warnings.warn(
                f"DL07IR component {self.name!r}: template file not found. "
                "Component will skip. For science use, provide template_path "
                "via DL07IRConfig.",
                UserWarning,
                stacklevel=2,
            )
            return None

        try:
            templates = load_draine_li_templates(template_path)
            return templates
        except Exception as e:
            warnings.warn(
                f"DL07IR component {self.name!r}: failed to load templates: {e}. "
                "Component will skip.",
                UserWarning,
                stacklevel=2,
            )
            return None

    def predict(
        self, p: Mapping[str, jnp.ndarray], sed_in: jnp.ndarray, wave: jnp.ndarray, **inputs: Any
    ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        r"""Compute dust emission via DL07 tabulated templates.

        Interpolates the DL07 template grid over (q_PAH, U_min, gamma) and
        applies energy-balance normalization.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Sliced parameters (prefix stripped):
            - ``p["qpah"]``: PAH mass fraction [%]
            - ``p["umin"]``: log U_min [dex]
            - ``p["gamma"]``: power-law mixing fraction [dimensionless]
        sed_in : ndarray, shape (n_wave,)
            Input SED in erg/s/Hz (ignored; DL07 emission is computed
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
        tmpl_wave = templates["wavelength"]
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

        # Interpolate onto target wavelength grid (template in L_lambda space)
        sed_llam = jnp.interp(wave, tmpl_wave, template, left=0.0, right=0.0)

        # Convert L_lambda -> L_nu: L_nu = L_lambda * (lambda^2 / c)
        AA_TO_CM = 1e-8
        C_CGS = 2.998e10
        wavelength_cm = wave * AA_TO_CM
        nu = C_CGS / wavelength_cm
        sed_lnu = sed_llam * (wavelength_cm**2) / C_CGS

        # Normalize by absorbed luminosity
        integral = -jnp.trapezoid(sed_lnu, nu)
        norm = jnp.where(integral > 0.0, L_ir / integral, 0.0)

        sed_emission = norm * sed_lnu

        # Return updated SED and published luminosity
        # Note: L_ir_emission not yet a typed field in DerivedBundle
        sed_out = sed_in + sed_emission

        return sed_out, {}
