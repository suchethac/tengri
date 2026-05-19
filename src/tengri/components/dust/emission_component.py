# SPDX-License-Identifier: BSD-3-Clause
r"""DustEmissionSEDComponent: SEDComponent adapter for IR re-emission.

Phase II-1 sixth adapter and the **first cross-component closed-loop**:
:class:`DustAttenuationSEDComponent` publishes ``state.derived["L_ir"]``
(the integral of absorbed UV/optical/NIR luminosity); this adapter
re-emits it as one of several IR templates, closing the dust energy
balance inside the orchestrator pipeline.

Currently dispatched templates
------------------------------
- ``"modified_blackbody"`` — closed-form analytic 2-parameter MBB
  (``dust_T``, ``dust_beta_ir``) with optional CMB-heating / CMB-
  contrast correction (da Cunha et al. 2013).
- ``"draine2021_pah"`` — Draine, Li, Hensley et al. 2021 PAHspec
  template grid (arXiv:2011.07046), parameterised by continuous
  ``dust_lgU`` and three categorical config knobs (``starlight``,
  ``ionization``, ``size_distribution``, ``slab``).

Casey/Dale/DL07/DL14/Astrodust/BOSA/THEMIS will be added as additional
``template`` cases as their precompute paths are migrated to the
:class:`tengri.protocols.PipelineState` Protocol.

Cross-component reads
---------------------
- ``state.derived["L_ir"]`` (erg/s) — published by
  :class:`DustAttenuationSEDComponent` via the energy-balance integral
  ``∫ (L_ν_intrinsic − L_ν_attenuated) dν``. Falls back to ``0.0`` if no
  upstream attenuator has run, in which case this adapter contributes
  nothing.

Pipeline ordering
-----------------
This adapter MUST run **after** the dust attenuator so ``L_ir`` is
present in ``state.derived``. The recommended natural order is
``[Stellar, AGN, Nebular, DustAttenuation, DustEmission, IGM, Radio, XRay]``.

References
----------
.. [1] da Cunha, E. et al. 2013, "On the effect of the cosmic
   microwave background in high-redshift (sub-)millimeter
   observations", ApJ 766, 13.  arXiv:1302.0844.
.. [2] Draine, B.T., Li, A., Hensley, B.S., Hunt, L.K., Sandstrom, K.,
   Smith, J.-D.T. 2021, "Excitation of Polycyclic Aromatic Hydrocarbon
   Emission: Dependence on Size Distribution, Ionization, and Starlight
   Spectrum and Intensity", ApJ 917, 3.  arXiv:2011.07046.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp

from tengri.components.dust import astrodust_hd23 as _ad
from tengri.components.dust.draine2021_pah import (
    integrate_lnu_over_nu,
    load_pahspec_or_raise,
    resample_lnu_on_aa_grid,
    select_pahspec_axes,
    select_pahspec_starlight_auto,
)
from tengri.components.dust.emission import modified_blackbody
from tengri.parameters.priors import Fixed, Uniform
from tengri.protocols.component import (
    ParamDeclaration,
    PipelineState,
    SEDComponentConfig,
    SEDComponentState,
)

__all__ = [
    "DustEmissionSEDComponent",
    "DustEmissionSEDComponentConfig",
    "DustEmissionSEDComponentState",
]


# Currently dispatched template values for ``DustEmissionSEDComponentConfig.template``.
_SUPPORTED_TEMPLATES: tuple[str, ...] = (
    "modified_blackbody",
    "draine2021_pah",
    "astrodust",
)


# Per-component selectors for the ``astrodust`` template emission output.
_ASTRODUST_COMPONENTS: tuple[str, ...] = ("total", "astrodust", "pah")


@dataclass(frozen=True)
class DustEmissionSEDComponentConfig(SEDComponentConfig):
    r"""Frozen knobs for :class:`DustEmissionSEDComponent`.

    Attributes
    ----------
    name : str
        Diagnostic identifier. Default ``"dust_emission"``.
    template : str
        IR template name.  One of :data:`_SUPPORTED_TEMPLATES`.
        Currently ``"modified_blackbody"`` (analytic) and
        ``"draine2021_pah"`` (Draine+2021 PAHspec).  Casey/Dale/DL07/
        DL14/Astrodust/BOSA/THEMIS will be added in Phase II-3.
    pahspec_starlight : str
        (``"draine2021_pah"`` only.) Starlight spectrum selector.
        One of the 13 PAHspec choices (e.g. ``"mMMP"``,
        ``"BPASS_Z0.02_3Myr"``, ``"m31bulge"``), or the special
        token ``"auto"`` to nearest-neighbour-select from the
        upstream stellar parameters (``pahspec_auto_*`` fields).
        Default ``"mMMP"``.
    pahspec_auto_age_myr : float or None
        (``pahspec_starlight="auto"`` only.)  Characteristic age of
        the FUV-emitting young population, in Myr.  Required when
        ``pahspec_starlight="auto"``.
    pahspec_auto_log_z_solar : float or None
        (``pahspec_starlight="auto"`` only.)
        :math:`\log_{10}(Z/Z_\odot)` for the ionising stellar
        population.  Required when ``pahspec_starlight="auto"``.
    pahspec_auto_sps_family : {"BC03", "BPASS", None} or other str
        (``pahspec_starlight="auto"`` only.)  SPS family used by the
        upstream stellar component.  ``None`` allows any SSP family;
        non-supported strings (FSPS / MIST / PrSc) fall back to the
        non-SSP ambient choices (``mMMP`` / ``m31bulge``).
    pahspec_ionization : str
        (``"draine2021_pah"`` only.) PAH ionization selector,
        ``"lo"`` / ``"st"`` / ``"hi"``.  Default ``"st"``.
    pahspec_size_distribution : str
        (``"draine2021_pah"`` only.) PAH size-distribution selector,
        ``"sma"`` / ``"std"`` / ``"lrg"``.  Default ``"std"``.
    pahspec_slab : bool
        (``"draine2021_pah"`` only.)  ``True`` to use the
        :math:`A_V=2` slab variant.  Default ``False``.
    pahspec_template_path : str or None
        (``"draine2021_pah"`` only.)  Override path to the PAHspec
        HDF5 grid.  When ``None``, the adapter looks at the
        ``TENGRI_PAHSPEC_PATH`` environment variable, then falls back
        to ``data/pahspec_draine2021.h5``.  Default ``None``.
    """

    name: str = "dust_emission"
    template: str = "modified_blackbody"
    pahspec_starlight: str = "mMMP"
    pahspec_ionization: str = "st"
    pahspec_size_distribution: str = "std"
    pahspec_slab: bool = False
    pahspec_template_path: str | None = None
    pahspec_auto_age_myr: float | None = None
    pahspec_auto_log_z_solar: float | None = None
    pahspec_auto_sps_family: str | None = None
    # ─── Hensley & Draine 2023 Astrodust+PAH knobs ─────────────────
    astrodust_component: str = "total"  # "total" | "astrodust" | "pah"
    astrodust_include_spinning_dust: bool = False
    astrodust_f_cnm: float = 0.28  # CNM filling for spinning-dust mixing
    astrodust_template_path: str | None = None


@dataclass(frozen=True)
class DustEmissionSEDComponentState(SEDComponentState):
    r"""Cached tensors for :class:`DustEmissionSEDComponent`.

    Empty for analytic templates (``modified_blackbody``); populated
    for template-grid models (``draine2021_pah``).

    Attributes
    ----------
    name : str
        Diagnostic identifier (matches the component name).
    pahspec_lgU_grid : jnp.ndarray, shape ``(15,)``
        :math:`\log_{10} U` grid points 0.0..7.0 step 0.5.  Empty for
        non-PAHspec templates.
    pahspec_lnu_template : jnp.ndarray, shape ``(15, n_wave)``
        :math:`L_\nu` per H atom on the pipeline rest-frame Å grid,
        sliced to the chosen ``(starlight, ion, size, slab)``.  Empty
        for non-PAHspec templates.  Units [erg/s/Hz/H]; absolute
        normalisation is irrelevant — only the spectrum *shape*
        survives the energy-balance rescale in :meth:`apply`.
    pahspec_norm_per_lgU : jnp.ndarray, shape ``(15,)``
        :math:`\int L_\nu \, d\nu` per ``lgU`` slice.  Used by
        :meth:`apply` to renormalise against ``state.derived["L_ir"]``
        without re-integrating in the JIT hot path.
    """

    name: str = "dust_emission"
    pahspec_lgU_grid: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    pahspec_lnu_template: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    pahspec_norm_per_lgU: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    astrodust_lgU_grid: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    astrodust_lnu_template: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    astrodust_norm_per_lgU: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    astrodust_lnu_spinning: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))


@dataclass(frozen=True)
class DustEmissionSEDComponent:
    r"""SEDComponent adapter dispatching IR re-emission templates.

    Selects the IR template from :attr:`config.template`
    (currently ``"modified_blackbody"`` or ``"draine2021_pah"``) and
    re-emits the absorbed luminosity ``state.derived["L_ir"]`` with
    the matching spectral shape.

    Notes
    -----
    **JIT-compatible**: yes — :meth:`apply` is pure JAX for both
    supported templates.

    **Additive**: writes ``sed_intrinsic = sed_intrinsic + L_dust(λ)``.

    **Closed-loop**: reads ``L_ir`` from ``state.derived`` (published
    by :class:`DustAttenuationSEDComponent`) — the first adapter that
    consumes a derived quantity from another adapter rather than from
    a parameter.  No-op when ``L_ir == 0``.

    Examples
    --------
    Modified-blackbody (default):

    >>> from tengri.components.dust.emission_component import (
    ...     DustEmissionSEDComponent,
    ...     DustEmissionSEDComponentConfig,
    ... )
    >>> comp = DustEmissionSEDComponent()  # MBB default

    Draine+2021 PAHspec, M31-bulge starlight, large size distribution:

    >>> cfg = DustEmissionSEDComponentConfig(
    ...     template="draine2021_pah",
    ...     pahspec_starlight="m31bulge",
    ...     pahspec_size_distribution="lrg",
    ... )
    >>> comp = DustEmissionSEDComponent(config=cfg)
    """

    config: DustEmissionSEDComponentConfig = field(default_factory=DustEmissionSEDComponentConfig)
    name: str = "dust_emission"
    parameter_prefix: str = "dust_"

    def _resolve_pahspec_starlight(self) -> str:
        r"""Resolve ``config.pahspec_starlight``, expanding ``"auto"``.

        When ``pahspec_starlight == "auto"``, calls
        :func:`select_pahspec_starlight_auto` with the
        ``pahspec_auto_*`` config fields and returns the chosen
        starlight name.  Otherwise returns the literal value.

        Returns
        -------
        str
            One of the keys of
            :data:`tengri.components.dust.draine2021_pah.STARLIGHT_PROPERTIES`.

        Raises
        ------
        ValueError
            If ``pahspec_starlight == "auto"`` but
            ``pahspec_auto_age_myr`` or ``pahspec_auto_log_z_solar``
            is ``None``.
        """
        if self.config.pahspec_starlight != "auto":
            return self.config.pahspec_starlight
        if self.config.pahspec_auto_age_myr is None:
            raise ValueError(
                "pahspec_starlight='auto' requires pahspec_auto_age_myr "
                "to be set on DustEmissionSEDComponentConfig (Myr)."
            )
        if self.config.pahspec_auto_log_z_solar is None:
            raise ValueError(
                "pahspec_starlight='auto' requires pahspec_auto_log_z_solar "
                "to be set on DustEmissionSEDComponentConfig "
                "(log10(Z / Z_solar))."
            )
        return select_pahspec_starlight_auto(
            sps_family=self.config.pahspec_auto_sps_family,
            age_myr=self.config.pahspec_auto_age_myr,
            log_z_solar=self.config.pahspec_auto_log_z_solar,
        )

    def declared_parameters(self) -> list[ParamDeclaration]:
        r"""Free parameters this component owns, by template.

        Returns the **superset** of parameters across all dispatched
        templates is *not* the convention here — instead each
        template returns only its own parameters.  Users compose by
        instantiating one component per template; the orchestrator
        merges the declarations.

        Returns
        -------
        list[ParamDeclaration]
            Parameter declarations specific to ``self.config.template``.

        Raises
        ------
        ValueError
            If ``self.config.template`` is not in
            :data:`_SUPPORTED_TEMPLATES`.
        """
        if self.config.template == "modified_blackbody":
            return [
                ParamDeclaration(
                    "dust_T",
                    Fixed(30.0),
                    "Modified blackbody dust temperature [K]",
                ),
                ParamDeclaration(
                    "dust_beta_ir",
                    Fixed(1.8),
                    "Modified blackbody emissivity index β [dimensionless]",
                ),
            ]
        if self.config.template == "draine2021_pah":
            return [
                ParamDeclaration(
                    "dust_lgU",
                    Uniform(0.0, 7.0),
                    "log10(U), starlight intensity in mMMP units "
                    "(Draine+2021 PAHspec) [dimensionless]",
                ),
            ]
        if self.config.template == "astrodust":
            return [
                ParamDeclaration(
                    "dust_lgU",
                    Uniform(-3.0, 6.0),
                    "log10(U), starlight intensity in mMMP units "
                    "(Hensley & Draine 2023 Astrodust+PAH) [dimensionless]",
                ),
            ]
        raise ValueError(
            f"DustEmissionSEDComponent: unsupported template "
            f"{self.config.template!r}; valid: {_SUPPORTED_TEMPLATES}"
        )

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
    ) -> DustEmissionSEDComponentState:
        r"""Build cached tensors per template.

        Parameters
        ----------
        ssp_data : ignored
            Present for Protocol conformance.
        wave_grid : jnp.ndarray, shape ``(n_wave_aa,)``, optional
            Pipeline rest-frame wavelength grid in Å.  Required for
            ``template="draine2021_pah"``; ignored for analytic
            templates.

        Returns
        -------
        DustEmissionSEDComponentState
            Empty for analytic templates; populated with sliced
            template tensors for PAHspec.

        Raises
        ------
        FileNotFoundError
            (``"draine2021_pah"`` only.)  When the PAHspec HDF5 grid
            is not on disk, with a message naming the
            ``scripts/build_pahspec_hdf5.py`` invocation that builds
            it.

        Notes
        -----
        **JIT-compatible**: no — file I/O happens here at compile
        time, then the cached tensors enter the JIT graph as static
        constants.
        """
        if self.config.template == "modified_blackbody":
            return DustEmissionSEDComponentState(name=self.name)

        if self.config.template == "draine2021_pah":
            if wave_grid is None:
                raise ValueError(
                    "DustEmissionSEDComponent.precompute requires wave_grid "
                    "(Angstrom) for template='draine2021_pah'"
                )
            templates = load_pahspec_or_raise(self.config.pahspec_template_path)
            starlight = self._resolve_pahspec_starlight()
            nu_pnu_um = select_pahspec_axes(
                templates,
                starlight=starlight,
                ionization=self.config.pahspec_ionization,
                size_distribution=self.config.pahspec_size_distribution,
                slab=self.config.pahspec_slab,
            )
            wave_aa = jnp.asarray(wave_grid)
            lnu_template = resample_lnu_on_aa_grid(
                nu_pnu_um=nu_pnu_um,
                wave_um=templates.wavelength_um,
                wave_aa=wave_aa,
            )
            norms = integrate_lnu_over_nu(lnu_template, wave_aa)
            return DustEmissionSEDComponentState(
                name=self.name,
                pahspec_lgU_grid=jnp.asarray(templates.lgU),
                pahspec_lnu_template=lnu_template,
                pahspec_norm_per_lgU=norms,
            )

        if self.config.template == "astrodust":
            if wave_grid is None:
                raise ValueError(
                    "DustEmissionSEDComponent.precompute requires wave_grid "
                    "(Angstrom) for template='astrodust'"
                )
            wave_aa = jnp.asarray(wave_grid)
            templates = _ad.load_astrodust_hd23_or_raise(self.config.astrodust_template_path)

            comp = self.config.astrodust_component
            if comp not in _ASTRODUST_COMPONENTS:
                raise ValueError(f"astrodust_component={comp!r} not in {_ASTRODUST_COMPONENTS}")
            if comp == "total":
                lnu_um = templates.L_nu_total
            elif comp == "astrodust":
                lnu_um = templates.L_nu_astrodust
            else:  # "pah"
                lnu_um = templates.L_nu_pah

            lnu_template = _ad.resample_lnu_on_aa_grid(
                L_nu_um=lnu_um,
                wave_um=templates.wavelength_um,
                wave_aa=wave_aa,
            )
            norms = _ad.integrate_lnu_over_nu_aa(lnu_template, wave_aa)

            # Spinning dust: precompose the f_CNM mixture once at
            # config time so apply() stays a pure JIT kernel.
            if self.config.astrodust_include_spinning_dust:
                f_cnm = float(self.config.astrodust_f_cnm)
                if not (0.0 <= f_cnm <= 1.0):
                    raise ValueError(f"astrodust_f_cnm must be in [0, 1]; got {f_cnm!r}")
                if comp == "astrodust":
                    spd_um = (
                        f_cnm * templates.L_nu_spdust_Ad_CNM
                        + (1.0 - f_cnm) * templates.L_nu_spdust_Ad_WNM
                    )
                elif comp == "pah":
                    spd_um = (
                        f_cnm * templates.L_nu_spdust_PAH_CNM
                        + (1.0 - f_cnm) * templates.L_nu_spdust_PAH_WNM
                    )
                else:  # "total" -> Ad + PAH at the chosen f_cnm
                    spd_um = f_cnm * (
                        templates.L_nu_spdust_Ad_CNM + templates.L_nu_spdust_PAH_CNM
                    ) + (1.0 - f_cnm) * (
                        templates.L_nu_spdust_Ad_WNM + templates.L_nu_spdust_PAH_WNM
                    )
                spd_aa = _ad.resample_lnu_on_aa_grid(
                    L_nu_um=spd_um[None, :],
                    wave_um=templates.wavelength_um,
                    wave_aa=wave_aa,
                )[0]
            else:
                spd_aa = jnp.zeros_like(wave_aa)

            return DustEmissionSEDComponentState(
                name=self.name,
                astrodust_lgU_grid=jnp.asarray(templates.lgU),
                astrodust_lnu_template=lnu_template,
                astrodust_norm_per_lgU=norms,
                astrodust_lnu_spinning=spd_aa,
            )

        raise ValueError(
            f"DustEmissionSEDComponent: unsupported template "
            f"{self.config.template!r}; valid: {_SUPPORTED_TEMPLATES}"
        )

    def apply(
        self,
        state: PipelineState,
        params: Mapping[str, jnp.ndarray],
        precomputed: DustEmissionSEDComponentState | None = None,
    ) -> PipelineState:
        r"""Add IR re-emission to ``state.sed_intrinsic``.

        Parameters
        ----------
        state : PipelineState
            Carries rest-frame ``wave`` [Å].  Reads
            ``state.derived["L_ir"]`` [erg/s], the energy-balance
            absorbed luminosity; falls back to 0 if absent (no-op).
        params : mapping
            Receives ``dust_*`` keys plus ``redshift`` from the bare-
            name allowlist.
        precomputed : DustEmissionSEDComponentState, optional
            Cached tensors from :meth:`precompute`.  Required for
            ``template="draine2021_pah"``; ignored for analytic
            templates.

        Returns
        -------
        PipelineState
            New state with ``sed_intrinsic`` updated additively and
            ``derived["sed_dust_ir"]`` published (the IR component
            :math:`L_\nu` profile, useful for diagnostics and
            consumed by :meth:`tengri.Posterior.sed_components`).

        Notes
        -----
        **JIT-compatible**: yes for both supported templates.

        **Energy balance**: for ``draine2021_pah`` the template
        spectrum is rescaled so that
        :math:`\int L_\nu \, d\nu = L_{\rm ir}` exactly.
        """
        L_ir = jnp.asarray(state.derived.get("L_ir", 0.0))

        if self.config.template == "modified_blackbody":
            z = jnp.asarray(params.get("redshift", 0.0))
            L_dust_lnu = modified_blackbody(
                state.wave,
                L_absorbed=L_ir,
                dust_T=jnp.asarray(params["dust_T"]),
                dust_beta_ir=jnp.asarray(params["dust_beta_ir"]),
                redshift=z,
            )

        elif self.config.template == "draine2021_pah":
            if precomputed is None:
                precomputed = self.precompute(wave_grid=state.wave)
            L_dust_lnu = _apply_pahspec(
                precomputed=precomputed,
                wave_aa=state.wave,
                L_ir=L_ir,
                dust_lgU=jnp.asarray(params["dust_lgU"]),
            )

        elif self.config.template == "astrodust":
            if precomputed is None:
                precomputed = self.precompute(wave_grid=state.wave)
            L_dust_lnu = _apply_astrodust(
                precomputed=precomputed,
                L_ir=L_ir,
                dust_lgU=jnp.asarray(params["dust_lgU"]),
            )

        else:
            raise ValueError(
                f"DustEmissionSEDComponent: unsupported template "
                f"{self.config.template!r}; valid: {_SUPPORTED_TEMPLATES}"
            )

        if state.sed_intrinsic is None:
            new_sed = jnp.zeros_like(state.wave) + L_dust_lnu
        else:
            new_sed = state.sed_intrinsic + L_dust_lnu

        return state.with_(
            sed_intrinsic=new_sed,
            derived=state.derived.with_(sed_dust_ir=L_dust_lnu),
        )


# ─────────────────────────────────────────────────────────────────────
# PAHspec apply kernel (private)
# ─────────────────────────────────────────────────────────────────────


def _apply_pahspec(
    precomputed: DustEmissionSEDComponentState,
    wave_aa: jnp.ndarray,
    L_ir: jnp.ndarray,
    dust_lgU: jnp.ndarray,
) -> jnp.ndarray:
    r"""JIT-compatible PAHspec evaluation kernel.

    Linearly interpolates the precomputed
    :math:`L_\nu(\lg U)` cube at the requested ``dust_lgU`` (clipped
    to the template support :math:`[0, 7]`) and rescales the result so
    its frequency integral equals the absorbed luminosity ``L_ir``.

    Parameters
    ----------
    precomputed : DustEmissionSEDComponentState
        State carrying ``pahspec_lgU_grid``, ``pahspec_lnu_template``,
        and ``pahspec_norm_per_lgU``.
    wave_aa : jnp.ndarray, shape ``(n_wave_aa,)``
        Pipeline rest-frame wavelength grid [Å].
    L_ir : jnp.ndarray, scalar
        Absorbed luminosity from the upstream attenuator [erg/s].
    dust_lgU : jnp.ndarray, scalar
        :math:`\log_{10} U` query [dimensionless].

    Returns
    -------
    jnp.ndarray, shape ``(n_wave_aa,)``
        IR :math:`L_\nu` [erg/s/Hz] with
        :math:`\int L_\nu \, d\nu = L_{\rm ir}`.
    """
    del wave_aa  # unused — wavelength axis lives inside precomputed.lnu_template

    lgU_grid = precomputed.pahspec_lgU_grid
    lnu_template = precomputed.pahspec_lnu_template
    norm_per_lgU = precomputed.pahspec_norm_per_lgU

    lgU_clipped = jnp.clip(dust_lgU, lgU_grid[0], lgU_grid[-1])
    L_nu_shape = jax.vmap(
        lambda col: jnp.interp(lgU_clipped, lgU_grid, col),
        in_axes=1,
    )(lnu_template)
    norm_at_lgU = jnp.interp(lgU_clipped, lgU_grid, norm_per_lgU)

    scale = jnp.where(norm_at_lgU > 0, L_ir / norm_at_lgU, 0.0)
    return L_nu_shape * scale


# ─────────────────────────────────────────────────────────────────────
# Astrodust+PAH apply kernel (private)
# ─────────────────────────────────────────────────────────────────────


def _apply_astrodust(
    precomputed: DustEmissionSEDComponentState,
    L_ir: jnp.ndarray,
    dust_lgU: jnp.ndarray,
) -> jnp.ndarray:
    r"""JIT-compatible Hensley & Draine 2023 Astrodust+PAH kernel.

    Linearly interpolates the precomputed
    :math:`L_\nu(\lg U)` cube at ``dust_lgU`` (clipped to the
    template support :math:`[-3, 6]`), rescales to absorbed
    luminosity ``L_ir``, and adds the (lgU-independent) precomposed
    spinning-dust spectrum if it was included at ``precompute()``
    time.

    Parameters
    ----------
    precomputed : DustEmissionSEDComponentState
        Carries ``astrodust_lgU_grid``, ``astrodust_lnu_template``,
        ``astrodust_norm_per_lgU``, and ``astrodust_lnu_spinning``.
    L_ir : jnp.ndarray, scalar
        Upstream absorbed luminosity [erg/s].
    dust_lgU : jnp.ndarray, scalar
        :math:`\log_{10} U` query [dimensionless].

    Returns
    -------
    jnp.ndarray, shape ``(n_wave_aa,)``
        :math:`L_\nu` [erg/s/Hz] with
        :math:`\int L_\nu \, d\nu = L_{\rm ir}` for the thermal
        component.  Spinning dust, when included, adds on top
        without renormalisation (its bolometric power is small).
    """
    lgU_grid = precomputed.astrodust_lgU_grid
    lnu_template = precomputed.astrodust_lnu_template
    norm_per_lgU = precomputed.astrodust_norm_per_lgU

    lgU_clipped = jnp.clip(dust_lgU, lgU_grid[0], lgU_grid[-1])
    L_nu_shape = jax.vmap(
        lambda col: jnp.interp(lgU_clipped, lgU_grid, col),
        in_axes=1,
    )(lnu_template)
    norm_at_lgU = jnp.interp(lgU_clipped, lgU_grid, norm_per_lgU)
    scale = jnp.where(norm_at_lgU > 0, L_ir / norm_at_lgU, 0.0)
    L_nu_thermal = L_nu_shape * scale

    # Spinning-dust contribution is precomposed at config time at the
    # chosen f_CNM and stored per H atom.  Rescale it by the same
    # thermal-budget factor so the spinning/thermal *ratio* matches
    # the published per-H value at this lgU; this keeps the absolute
    # spinning-dust amplitude tied to the dust mass implied by L_ir.
    L_nu_spinning = precomputed.astrodust_lnu_spinning * scale
    return L_nu_thermal + L_nu_spinning
