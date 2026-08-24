# SPDX-License-Identifier: BSD-3-Clause
"""Schreiber et al. (2016) dust emission as an SEDModelComponent.

Implements tabulated-template IR re-emission based on the Schreiber+2016 dust model,
parameterized by dust temperature (T_dust) and PAH fraction (f_pah).

The templates are loaded from an HDF5 file during :meth:`load` (precomputation)
and cached for rapid access during forward passes.

References
----------
.. [1] Schreiber, C. et al. 2016, "Cold dust properties from dust SED fitting
   and its correlation with the far-infrared properties of nearby galaxies",
   A&A, 602, A96. https://doi.org/10.1051/0004-6361/201629925
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import jax.numpy as jnp

from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Uniform
from tengri.protocols.component import SEDComponentConfig
from tengri.utils.grid_interp import resample_template

__all__ = ["Schreiber2016IRConfig", "Schreiber2016IRSEDComponent"]


@dataclass(frozen=True)
class Schreiber2016IRConfig(SEDComponentConfig):
    """Configuration for Schreiber2016IR dust emission component.

    Attributes
    ----------
    name: str
        Diagnostic identifier. Default ``"schreiber2016_ir"``.
    template_path: str, optional
        Path to Schreiber2016 template HDF5 file. If None, attempts to auto-locate
        via standard search paths. If templates are unavailable, the component
        will skip gracefully with a warning.
    """

    name: str = "schreiber2016_ir"
    template_path: str | None = None


class Schreiber2016IRSEDComponent(SEDModelComponent):
    """Dust IR emission via tabulated Schreiber+2016 templates.

    Closes the dust energy balance by re-emitting absorbed UV/optical
    luminosity according to a 2-parameter model (T_dust, f_pah)
    interpolated over a precomputed template grid.

    Attributes
    ----------
    name: str
        Stable identifier: ``"schreiber2016_ir"``.
    parameter_prefix: str
        Domain prefix for parameters: ``"dust_"``.
    config: Schreiber2016IRConfig
        Frozen configuration with optional template path override.

    Notes
    -----
    **Cross-component contract**:

    - Reads: ``state.derived["L_ir"]`` (erg/s): luminosity absorbed by dust.
    - Publishes: ``{"L_ir_emission": erg/s}``: bolometric IR from templates.

    **Template loading**: Templates are loaded during :meth:`load` (precomputation)
    and stored on ``self.data``. If template files are unavailable, the component
    gracefully skips with a warning.

    **JIT-compatible**: yes, all operations in :meth:`predict` are ``jnp``
    primitives.

    **Parameter discovery**: Free parameters (``T``, ``f_pah``) are
    auto-discovered; :meth:`declared_parameters` constructs tuples with
    units and descriptions. Canonical names ``dust_T`` / ``dust_f_pah`` (#849);
    the old ``dust_tdust`` / ``dust_fpah`` spellings resolve via
    ``_LEGACY_PARAM_ALIASES`` with a deprecation warning.

    **Pipeline ordering**: This component MUST run after dust attenuation.
    Typical order: ``[Stellar, Nebular, DustAttenuation, Schreiber2016, IGM, Radio]``.
    """

    name = "schreiber2016_ir"
    parameter_prefix = "dust_"
    config: Schreiber2016IRConfig = Schreiber2016IRConfig()

    # Free parameters: auto-discovered by base class. Canonical names (#849):
    # ``dust_T`` (was ``dust_tdust``) + ``dust_f_pah`` (was ``dust_fpah``);
    # the old spellings resolve via _LEGACY_PARAM_ALIASES.
    T = Uniform(15.0, 99.0, default=25.0, description="Dust temperature", units="K")
    f_pah = Uniform(0.0, 1.0, default=0.05, description="PAH fraction", units="dimensionless")

    # Cross-component contract
    inputs: ClassVar = {"L_ir": "erg/s"}
    outputs: ClassVar = {"L_ir_emission": "erg/s"}

    def load(self, wave: jnp.ndarray | None = None) -> Any | None:
        """Preload Schreiber2016 template grid.

        Called at model init time (not JIT-ed). Attempts to locate and load
        the Schreiber2016 template HDF5 file. If unavailable, returns None and the
        component will skip gracefully.

        Parameters
        ----------
        wave: ndarray, optional
            Rest-frame wavelength grid in Angstrom (not used for Schreiber2016
            template loading, but kept for protocol compatibility).

        Returns
        -------
        dict or None
            Dictionary with keys {wavelength_aa, tdust_grid, continuum, pah}
            if templates loaded successfully. Returns None if templates unavailable.
        """
        import warnings

        # Auto-locate template file if not explicitly provided
        template_path = self.config.template_path
        if template_path is None:
            from tengri._data_setup import find_data_str

            # Try standard names
            for fname in ("schreiber2016_templates.h5",):
                template_path = find_data_str(fname)
                if template_path is not None:
                    break

        if template_path is None:
            warnings.warn(
                f"Schreiber2016IR component {self.name!r}: template file not found. "
                "Component will skip. For science use, provide template_path "
                "via Schreiber2016IRConfig.",
                UserWarning,
                stacklevel=2,
            )
            return None

        try:
            from tengri.components.dust.emission_templates import load_schreiber2016_templates

            templates = load_schreiber2016_templates(template_path)
            return templates
        except Exception as e:
            warnings.warn(
                f"Schreiber2016IR component {self.name!r}: failed to load templates: {e}. "
                "Component will skip.",
                UserWarning,
                stacklevel=2,
            )
            return None

    def predict(
        self, p: Mapping[str, jnp.ndarray], sed_in: jnp.ndarray, wave: jnp.ndarray, **inputs: Any
    ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        r"""Compute dust emission via Schreiber2016 tabulated templates.

        Interpolates the Schreiber2016 template grid over (tdust, fpah) via
        1D linear interpolation in tdust and linear mixing in fpah.

        Parameters
        ----------
        p: mapping[str, ndarray]
            Sliced parameters (prefix stripped):

            - ``p["T"]``: dust temperature [K]
            - ``p["f_pah"]``: PAH fraction [dimensionless]

        sed_in: ndarray, shape (n_wave,)
            Input SED in erg/s/Hz (ignored; Schreiber2016 emission is computed
            from L_ir independently).
        wave: ndarray, shape (n_wave,)
            Rest-frame wavelength grid in Angstrom.
        **inputs: ndarray
            Cross-component inputs:

            - ``L_ir``: absorbed luminosity [erg/s]

        Returns
        -------
        tuple[ndarray, dict]

            - ``sed_out``: Updated SED in erg/s/Hz.
            - ``published``: Dict with ``{"L_ir_emission": erg/s,
              "sed_dust_ir": erg/s/Hz}``.

        Notes
        -----
        **JIT-compatible**: yes.

        **Gradient-safe**: yes, 1D linear interpolation is differentiable
        everywhere except at grid boundaries (where clipping occurs).
        """
        L_ir = inputs["L_ir"]

        # Try precomputed data first; if unavailable, lazy-load at trace time
        if hasattr(self, "data") and self.data is not None:
            templates = self.data
        else:
            # Lazy-load templates like architecture A (module-level closure).
            # Ensures component works both with precomputation (self.data set)
            # and without (approx=None, default builds). Loads inside a trace
            # via jax.ensure_compile_time_eval() to avoid tracer leakage.
            import warnings

            from tengri._data_setup import find_data_str
            from tengri.components.dust.emission_templates import load_schreiber2016_templates

            template_path = self.config.template_path
            if template_path is None:
                for fname in ("schreiber2016_templates.h5",):
                    template_path = find_data_str(fname)
                    if template_path is not None:
                        break

            if template_path is None:
                warnings.warn(
                    f"Schreiber2016IR component {self.name!r}: template file not found. "
                    "Component will skip. For science use, provide template_path "
                    "via Schreiber2016IRConfig.",
                    UserWarning,
                    stacklevel=2,
                )
                return sed_in, {}

            try:
                templates = load_schreiber2016_templates(template_path)
            except Exception as e:
                warnings.warn(
                    f"Schreiber2016IR component {self.name!r}: failed to load templates: {e}. "
                    "Component will skip.",
                    UserWarning,
                    stacklevel=2,
                )
                return sed_in, {}

        # Load template grid from precomputed state or just-loaded data
        continuum = templates["continuum"]  # (n_tdust, n_wave)
        pah = templates["pah"]  # (n_tdust, n_wave)
        tmpl_wave = templates["wavelength_aa"]
        tdust_grid = templates["tdust_grid"]

        # Clip parameters to grid bounds
        T_c = jnp.clip(p["T"], tdust_grid[0], tdust_grid[-1])
        f_pah_c = jnp.clip(p["f_pah"], 0.0, 1.0)

        # Linear interpolation index in temperature
        i_t = jnp.clip(
            jnp.searchsorted(tdust_grid, T_c) - 1,
            0,
            len(tdust_grid) - 2,
        )
        ft = (T_c - tdust_grid[i_t]) / (tdust_grid[i_t + 1] - tdust_grid[i_t])

        # Interpolate both continuum and PAH at the requested temperature
        continuum_interp = (1.0 - ft) * continuum[i_t] + ft * continuum[i_t + 1]
        pah_interp = (1.0 - ft) * pah[i_t] + ft * pah[i_t + 1]

        # Mix continuum and PAH
        template = (1.0 - f_pah_c) * continuum_interp + f_pah_c * pah_interp

        # Interpolate onto target wavelength grid
        sed = resample_template(wave, tmpl_wave, template, left=0.0, right=0.0)

        # Scale by absorbed luminosity
        sed_emission = L_ir * sed

        # Return updated SED and published outputs
        sed_out = sed_in + sed_emission
        published = {
            "L_ir_emission": L_ir,
            "sed_dust_ir": sed_emission,
        }

        return sed_out, published
