# SPDX-License-Identifier: BSD-3-Clause
"""Draine, Li, Hensley et al. (2021) PAHspec dust emission as an SEDModelComponent.

Implements tabulated-template IR re-emission based on the PAHspec dust model,
parameterized by starlight intensity U (via log10(U)) and categorical choices
for starlight spectrum, ionization state, size distribution, and slab thickness.

The templates are loaded from an HDF5 file during :meth:`load` (precomputation)
and cached for rapid access during forward passes.

References
----------
.. [1] Draine, B.T., Li, A., Hensley, B.S., Hunt, L.K., Sandstrom, K.,
   Smith, J.-D.T., 2021, "Excitation of Polycyclic Aromatic Hydrocarbon
   Emission: Dependence on Size Distribution, Ionization, and Starlight
   Spectrum and Intensity", ApJ, 917, 3.  arXiv:2011.07046.
   DOI: 10.3847/1538-4357/abff51.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from tengri.components.dust.draine2021_pah import (
    load_pahspec_or_raise,
    resample_lnu_on_aa_grid,
    select_pahspec_axes,
    select_pahspec_starlight_auto,
)
from tengri.components.dust.emission._physics import integrate_lnu_over_nu
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Uniform
from tengri.protocols.component import SEDComponentConfig

__all__ = ["Draine2021PAHIRConfig", "Draine2021PAHIRSEDComponent"]


@dataclass(frozen=True)
class Draine2021PAHIRConfig(SEDComponentConfig):
    """Configuration for Draine2021PAHIR dust emission component.

    Attributes
    ----------
    name : str
        Diagnostic identifier. Default ``"draine2021_pah_ir"``.
    template_path : str, optional
        Path to PAHspec template HDF5 file. If None, attempts to auto-locate
        via TENGRI_PAHSPEC_PATH environment variable or standard search paths.
        If templates are unavailable, the component will skip gracefully.
    starlight : str
        Starlight spectrum selector: one of the 13 PAHspec choices
        (e.g. ``"mMMP"``, ``"BPASS_Z0.02_3Myr"``, ``"m31bulge"``),
        or ``"auto"`` to auto-select based on upstream stellar parameters.
        Default ``"mMMP"``.
    ionization : str
        PAH ionization state: ``"lo"`` (low), ``"st"`` (standard), ``"hi"`` (high).
        Default ``"st"``.
    size_distribution : str
        PAH size distribution: ``"sma"`` (small), ``"std"`` (standard), ``"lrg"`` (large).
        Default ``"std"``.
    slab : bool
        Whether to use the :math:`A_V=2` slab variant. Default ``False``.
    auto_age_myr : float, optional
        (``starlight="auto"`` only.) Characteristic age of the FUV-emitting
        young population in Myr. Required when ``starlight="auto"``.
    auto_log_z_solar : float, optional
        (``starlight="auto"`` only.) :math:`\\log_{10}(Z/Z_\\odot)` for the
        ionizing stellar population. Required when ``starlight="auto"``.
    auto_sps_family : str, optional
        (``starlight="auto"`` only.) SPS family (``"BC03"``, ``"BPASS"``, or ``None``
        for non-SSP ambient spectra). Default ``None``.
    """

    name: str = "draine2021_pah_ir"
    template_path: str | None = None
    starlight: str = "mMMP"
    ionization: str = "st"
    size_distribution: str = "std"
    slab: bool = False
    auto_age_myr: float | None = None
    auto_log_z_solar: float | None = None
    auto_sps_family: str | None = None


class Draine2021PAHIRSEDComponent(SEDModelComponent):
    """Dust IR emission via tabulated Draine+2021 PAHspec templates.

    Closes the dust energy balance by re-emitting absorbed UV/optical
    luminosity according to a 1-parameter model (log10 U) with categorical
    configuration choices for starlight, ionization, size, and slab.

    Attributes
    ----------
    name : str
        Stable identifier: ``"draine2021_pah_ir"``.
    parameter_prefix : str
        Domain prefix for parameters: ``"dust_"``.
    config : Draine2021PAHIRConfig
        Frozen configuration with template path and categorical choices.

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

    **Parameter discovery**: Free parameters (``lgU``)
    are auto-discovered; :meth:`declared_parameters` constructs tuples with
    units and descriptions.

    **Pipeline ordering**: This component MUST run after dust attenuation.
    Typical order: ``[Stellar, Nebular, DustAttenuation, Draine2021PAH, IGM, Radio]``.
    """

    name = "draine2021_pah_ir"
    citations = ("draine2021_pah",)
    parameter_prefix = "dust_"
    config: Draine2021PAHIRConfig = Draine2021PAHIRConfig()

    # Free parameters — auto-discovered by base class
    lgU = Uniform(0.0, 7.0, default=1.0, description="log10(U), starlight intensity", units="dex")

    # Cross-component contract
    inputs: ClassVar = {"L_ir": "erg/s"}
    outputs: ClassVar = {"L_ir_emission": "erg/s"}

    def __init__(self, config: Draine2021PAHIRConfig | None = None) -> None:
        # Override the class-level default config if an instance is provided.
        if config is not None:
            self.config = config

    def _resolve_starlight(self) -> str:
        """Resolve starlight selector, expanding 'auto' if needed."""
        if self.config.starlight != "auto":
            return self.config.starlight
        if self.config.auto_age_myr is None:
            raise ValueError(
                "starlight='auto' requires auto_age_myr (Myr) to be set on Draine2021PAHIRConfig."
            )
        if self.config.auto_log_z_solar is None:
            raise ValueError(
                "starlight='auto' requires auto_log_z_solar (log10(Z/Zsun)) "
                "to be set on Draine2021PAHIRConfig."
            )
        return select_pahspec_starlight_auto(
            sps_family=self.config.auto_sps_family,
            age_myr=self.config.auto_age_myr,
            log_z_solar=self.config.auto_log_z_solar,
        )

    def load(self, wave: jnp.ndarray | None = None) -> Any | None:
        """Preload PAHspec template grid.

        Called at model init time (not JIT-ed). Attempts to locate and load
        the PAHspec template HDF5 file. If unavailable, returns None and the
        component will skip gracefully.

        Parameters
        ----------
        wave : ndarray, optional
            Rest-frame wavelength grid in Angstrom. Required for slicing
            templates onto the pipeline wavelength grid.

        Returns
        -------
        dict or None
            Dictionary with keys {lgU_grid, lnu_template, norm_per_lgU}
            if templates loaded successfully. Returns None if templates unavailable.
        """
        import warnings

        if wave is None:
            warnings.warn(
                f"Draine2021PAHIR component {self.name!r}: wave_grid is None. "
                "Component will skip.",
                UserWarning,
                stacklevel=2,
            )
            return None

        # Auto-locate template file if not explicitly provided
        template_path = self.config.template_path
        if template_path is None:
            import os

            from tengri.components.dust.draine2021_pah import (
                DRAINE2021_PAH_DEFAULT_PATH,
                PAHSPEC_PATH_ENV,
            )
            from tengri.components.dust.emission_templates import _find_data_file

            template_path = os.environ.get(PAHSPEC_PATH_ENV)
            if template_path is None:
                # ``_find_data_file`` searches the ``_DATA_CANDIDATES`` dirs
                # (each already ending in ``/data``) for a BARE filename, so
                # pass the basename — the ``data/`` prefix in
                # ``DRAINE2021_PAH_DEFAULT_PATH`` (kept for the direct
                # ``Path(...)`` fallback in ``draine2021_pah.py``) would
                # otherwise produce a doubled ``data/data/…`` miss and the
                # component would silently emit zeros (#852).
                template_path = _find_data_file(os.path.basename(DRAINE2021_PAH_DEFAULT_PATH))

        if template_path is None:
            warnings.warn(
                f"Draine2021PAHIR component {self.name!r}: template file not found. "
                "Set TENGRI_PAHSPEC_PATH or provide template_path via config. "
                "Component will skip.",
                UserWarning,
                stacklevel=2,
            )
            return None

        try:
            templates = load_pahspec_or_raise(template_path)
            starlight = self._resolve_starlight()
            nu_pnu_um = select_pahspec_axes(
                templates,
                starlight=starlight,
                ionization=self.config.ionization,
                size_distribution=self.config.size_distribution,
                slab=self.config.slab,
            )
            wave_aa = jnp.asarray(wave)
            lnu_template = resample_lnu_on_aa_grid(
                nu_pnu_um=nu_pnu_um,
                wave_um=templates.wavelength_um,
                wave_aa=wave_aa,
            )
            norms = integrate_lnu_over_nu(lnu_template, wave_aa)
            return {
                "lgU_grid": jnp.asarray(templates.lgU),
                "lnu_template": lnu_template,
                "norm_per_lgU": norms,
            }
        except Exception as e:
            warnings.warn(
                f"Draine2021PAHIR component {self.name!r}: failed to load templates: {e}. "
                "Component will skip.",
                UserWarning,
                stacklevel=2,
            )
            return None

    def predict(
        self, p: Mapping[str, jnp.ndarray], sed_in: jnp.ndarray, wave: jnp.ndarray, **inputs: Any
    ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        r"""Compute dust emission via PAHspec tabulated templates.

        Linearly interpolates the precomputed :math:`L_\nu(\lg U)` cube
        at the requested ``dust_lgU`` and rescales so its frequency integral
        equals the absorbed luminosity ``L_ir``.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Sliced parameters (prefix stripped):

            - ``p["lgU"]``: log10(U) [dex]

        sed_in : ndarray, shape (n_wave,)
            Input SED in erg/s/Hz (ignored; PAHspec emission is computed
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
        precomp = self.data
        lgU_grid = precomp["lgU_grid"]
        lnu_template = precomp["lnu_template"]  # (n_lgU, n_wave)
        norm_per_lgU = precomp["norm_per_lgU"]

        # Clip lgU to grid bounds
        dust_lgU_c = jnp.clip(p["lgU"], lgU_grid[0], lgU_grid[-1])

        # Linear interpolation of spectrum shape across lgU axis
        L_nu_shape = jax.vmap(
            lambda col: jnp.interp(dust_lgU_c, lgU_grid, col),
            in_axes=1,
        )(lnu_template)

        # Interpolate normalization
        norm_at_lgU = jnp.interp(dust_lgU_c, lgU_grid, norm_per_lgU)

        # Scale to match absorbed luminosity
        scale = jnp.where(norm_at_lgU > 0, L_ir / norm_at_lgU, 0.0)
        sed_emission = L_nu_shape * scale

        # Return updated SED and published luminosity
        sed_out = sed_in + sed_emission

        return sed_out, {}
