# SPDX-License-Identifier: BSD-3-Clause
"""Dale et al. (2014) dust emission as an SEDModelComponent.

Implements tabulated-template IR re-emission based on the Dale+2014 dust model,
parameterized by a single radiation field power-law index (alpha).

The templates are loaded from an HDF5 file during :meth:`load` (precomputation)
and cached for rapid access during forward passes.

References
----------
.. [1] Dale, D.A. et al. 2014, "The Infrared Emission of Star-forming
   Galaxies in the Herschel Reference Survey", ApJ 784, 83.
   https://doi.org/10.1088/0004-637X/784/1/83
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import jax.numpy as jnp

from tengri.components.dust.emission_templates import load_dale2014_templates
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Uniform
from tengri.protocols.component import SEDComponentConfig

__all__ = ["Dale2014IRConfig", "Dale2014IRSEDComponent"]


@dataclass(frozen=True)
class Dale2014IRConfig(SEDComponentConfig):
    """Configuration for Dale2014IR dust emission component.

    Attributes
    ----------
    name : str
        Diagnostic identifier. Default ``"dale2014_ir"``.
    template_path : str, optional
        Path to Dale2014 template HDF5 file. If None, attempts to auto-locate
        via standard search paths. If templates are unavailable, the component
        will skip gracefully with a warning.
    """

    name: str = "dale2014_ir"
    template_path: str | None = None


class Dale2014IRSEDComponent(SEDModelComponent):
    """Dust IR emission via tabulated Dale+2014 templates.

    Closes the dust energy balance by re-emitting absorbed UV/optical
    luminosity according to a 1-parameter model (alpha) interpolated over
    a precomputed template grid.

    Attributes
    ----------
    name : str
        Stable identifier: ``"dale2014_ir"``.
    parameter_prefix : str
        Domain prefix for parameters: ``"dust_"``.
    config : Dale2014IRConfig
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

    **Parameter discovery**: Free parameters (``alpha_dale``)
    are auto-discovered; :meth:`declared_parameters` constructs tuples with
    units and descriptions.

    **Pipeline ordering**: This component MUST run after dust attenuation.
    Typical order: ``[Stellar, Nebular, DustAttenuation, Dale2014, IGM, Radio]``.
    """

    name = "dale2014_ir"
    parameter_prefix = "dust_"
    config: Dale2014IRConfig = Dale2014IRConfig()

    # Free parameters — auto-discovered by base class
    alpha_dale = Uniform(
        0.5,
        3.0,
        default=2.0,
        description="Radiation field power-law index",
        units="dimensionless",
    )
    frac_agn = Uniform(
        0.0,
        0.99,
        default=0.0,
        description="AGN heating fraction (additive)",
        units="dimensionless",
    )

    # Cross-component contract
    inputs: ClassVar = {"L_ir": "erg/s"}
    outputs: ClassVar = {"L_ir_emission": "erg/s"}

    def load(self, wave: jnp.ndarray | None = None) -> Any | None:
        """Preload Dale2014 template grid.

        Called at model init time (not JIT-ed). Attempts to locate and load
        the Dale2014 template HDF5 file. If unavailable, returns None and the
        component will skip gracefully.

        Parameters
        ----------
        wave : ndarray, optional
            Rest-frame wavelength grid in Angstrom (not used for Dale2014
            template loading, but kept for protocol compatibility).

        Returns
        -------
        dict or None
            Dictionary with keys {wavelength_aa, alpha_grid, spectra}
            if templates loaded successfully. Returns None if
            templates unavailable.
        """
        import warnings

        # Auto-locate template file if not explicitly provided
        template_path = self.config.template_path
        if template_path is None:
            from tengri.components.dust.emission_templates import _find_data_file

            # Try v2 first, then legacy
            for fname in ("dale2014_templates_v2.h5", "dale2014_templates.h5"):
                template_path = _find_data_file(fname)
                if template_path is not None:
                    break

        if template_path is None:
            warnings.warn(
                f"Dale2014IR component {self.name!r}: template file not found. "
                "Component will skip. For science use, provide template_path "
                "via Dale2014IRConfig.",
                UserWarning,
                stacklevel=2,
            )
            return None

        try:
            templates = load_dale2014_templates(template_path)
            return templates
        except Exception as e:
            warnings.warn(
                f"Dale2014IR component {self.name!r}: failed to load templates: {e}. "
                "Component will skip.",
                UserWarning,
                stacklevel=2,
            )
            return None

    def predict(
        self, p: Mapping[str, jnp.ndarray], sed_in: jnp.ndarray, wave: jnp.ndarray, **inputs: Any
    ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        r"""Compute dust emission via Dale+2014 tabulated templates.

        Interpolates the Dale2014 template grid over alpha. If an AGN template
        is available, mixes SF and AGN components via additive composition:
        SED = (1 - fracAGN) * SF + fracAGN * QSO, with total IR scaled by
        L_absorbed / (1 - fracAGN) to account for the AGN as an independent
        heating source.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Sliced parameters (prefix stripped):
            - ``p["alpha_dale"]``: radiation field power-law index [dimensionless]
            - ``p["frac_agn"]``: AGN heating fraction [dimensionless, default 0.0]
        sed_in : ndarray, shape (n_wave,)
            Input SED in erg/s/Hz (ignored; Dale2014 emission is computed
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

        **Gradient-safe**: yes — linear interpolation is differentiable
        everywhere except at grid boundaries (where clipping occurs).
        """
        L_ir = inputs["L_ir"]

        # Skip if templates were not loaded
        if not hasattr(self, "data") or self.data is None:
            # Return input SED unchanged
            return sed_in, {}

        # Load template grid from precomputed state
        templates = self.data
        tmpl_wave = templates["wavelength_aa"]
        alpha_grid = templates["alpha_grid"]
        templates_sf = templates["templates_sf"]  # (n_alpha, n_wave)
        has_qso = "templates_qso" in templates
        if has_qso:
            templates_qso = templates["templates_qso"]  # (n_wave,)

        # Clip parameters to grid bounds
        dust_alpha_c = jnp.clip(p["alpha_dale"], alpha_grid[0], alpha_grid[-1])

        # Linear interpolation index for SF component
        i_a = jnp.clip(
            jnp.searchsorted(alpha_grid, dust_alpha_c) - 1,
            0,
            len(alpha_grid) - 2,
        )
        fa = (dust_alpha_c - alpha_grid[i_a]) / (alpha_grid[i_a + 1] - alpha_grid[i_a])

        # Interpolate SF template spectrum
        template_sf = (1.0 - fa) * templates_sf[i_a] + fa * templates_sf[i_a + 1]

        # AGN mixing (only if QSO template is available)
        if has_qso:
            f_agn = jnp.clip(p.get("frac_agn", 0.0), 0.0, 0.99)
            # Additive mixing: (1 - f) * SF + f * QSO
            template_mixed = (1.0 - f_agn) * template_sf + f_agn * templates_qso
            # Scale by L_absorbed / (1 - f_agn) to account for the AGN power source
            scale_factor = L_ir / jnp.maximum(1.0 - f_agn, 1e-10)
        else:
            # Back-compat: SF-only if QSO template absent
            template_mixed = template_sf
            scale_factor = L_ir

        # Interpolate onto target wavelength grid
        sed = jnp.interp(wave, tmpl_wave, template_mixed, left=0.0, right=0.0)

        # Scale by absorbed luminosity (adjusted for AGN if present)
        sed_emission = scale_factor * sed

        # Return updated SED and published luminosity
        sed_out = sed_in + sed_emission

        return sed_out, {}
