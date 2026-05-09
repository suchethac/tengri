# SPDX-License-Identifier: BSD-3-Clause
"""GRAHSPSEDComponent: composable SEDComponent adapter for the GRAHSP model.

Mirrors the canonical adapter pattern of :mod:`tengri.components.radio.component`.
The component publishes (all under ``state.derived``):

- ``sed_grahsp`` — :math:`L_\\nu` [erg/s/Hz] of the full GRAHSP AGN-side SED.
- ``L_agn_bol`` — bolometric BBB luminosity (paper §2.1.4 ``lumBolBBB``).
- ``L_agn_torus`` — torus bolometric luminosity (``lumBolTOR``).
- ``L_agn_absorbed`` — energy absorbed by the AGN bi-attenuation (the
  intrinsic-minus-attenuated integral over the AGN-side spectrum). Useful
  for energy-budget diagnostics; see "Energy balance" below.

Composition
-----------
Users compose via :class:`GRAHSPSEDComponentConfig`:

- ``include_bbb``: include the smooth bending power-law BBB.
- ``include_lines``: include broad/narrow line Gaussians.
- ``include_feii``: include the Bruhweiler+Verner 2008 FeII forest.
- ``include_torus``: include the cool+hot log-Gaussian dust continuum.
- ``include_si``: include the Mullaney 2011 Si difference template.
- ``apply_attenuation``: attenuate the AGN side with E(B-V) + E(B-V)-AGN.

This makes it possible to mix GRAHSP pieces with **other** tengri AGN
models (e.g. SKIRTOR torus + GRAHSP BBB; or QSOgen + GRAHSP attenuation).

Energy balance
--------------
Two notes that matter for getting realistic IR SEDs:

1. **Galaxy attenuation is *not* this component's job.** In the upstream
   CIGALE ``biattenuation`` module, ``E(B-V)`` attenuates *both* galaxy
   and AGN. In tengri's component model the galaxy SED is owned by the
   stellar+dust pipeline and is attenuated by :mod:`tengri.components.dust`.
   This component's ``agn_grahsp_ebv`` is therefore the **AGN baseline
   line-of-sight extinction**, not the galaxy's. To reproduce upstream
   behaviour, configure ``dust_*`` with the same E(B-V) and an SMC-like
   law (see :func:`tengri.components.dust.attenuation.prevot_smc`).
2. **The AGN side does not need an explicit re-emission loop.** GRAHSP's
   torus parameters (``fcov``, cool/hot log-Gaussians) already empirically
   model the dust re-radiation of absorbed UV/optical AGN light (paper
   §2.1.5). We publish ``L_agn_absorbed`` for diagnostics only — it is
   *not* fed into the Dale 2014 dust-emission component, since doing so
   would double-count the torus contribution.

Parameter contract
------------------
All free parameters carry the ``agn_grahsp_`` prefix
(NAMING_CONTRACT §3.2; CI-enforced).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.grahsp.attenuation import attenuation_factors
from tengri.components.agn.grahsp.bbb import sbpl_bbb
from tengri.components.agn.grahsp.bolometric import (
    bolometric_luminosity_bbb,
    bolometric_luminosity_torus,
)
from tengri.components.agn.grahsp.lines import feii_forest, gaussian_lines
from tengri.components.agn.grahsp.templates import (
    GRAHSPTemplates,
    load_grahsp_templates,
)
from tengri.components.agn.grahsp.torus import si_feature, torus_dust_continuum
from tengri.core.component import (
    ParamDeclaration,
    PipelineState,
    SEDComponentConfig,
    SEDComponentState,
)
from tengri.parameters.priors import Fixed, LogUniform, Uniform

__all__ = [
    "GRAHSPSEDComponent",
    "GRAHSPSEDComponentConfig",
    "GRAHSPSEDComponentState",
]


# Speed of light in nm * Hz (== c in nm / s) for L_lambda <-> L_nu conversion.
_C_NM_PER_S: float = 2.99792458e17  # 1e9 * 2.998e8


@dataclass(frozen=True)
class GRAHSPSEDComponentConfig(SEDComponentConfig):
    """Composable knobs for :class:`GRAHSPSEDComponent`.

    Attributes
    ----------
    name : str
        Diagnostic identifier.
    agn_type : int
        ``1`` Sy1 / QSO (broad+narrow_sy2+FeII), ``2`` Sy2 (narrow_sy2 only),
        ``3`` LINER (narrow_liner only).
    include_bbb, include_lines, include_feii : bool
        Which AGN-side components to include in the BBB-side SED.
    include_torus, include_si : bool
        Whether to include the IR torus continuum and the Si difference
        feature.
    apply_attenuation : bool
        Apply the GRAHSP bi-attenuation curve to the AGN spectrum. When
        ``False`` the component emits the intrinsic SED only.
    template_path : str | None
        Override path to ``grahsp_templates.h5``. ``None`` -> default.
    """

    name: str = "agn_grahsp"
    agn_type: int = 1
    include_bbb: bool = True
    include_lines: bool = True
    include_feii: bool = True
    include_torus: bool = True
    include_si: bool = True
    apply_attenuation: bool = True
    template_path: str | None = None


@dataclass(frozen=True)
class GRAHSPSEDComponentState(SEDComponentState):
    r"""Cached templates for :class:`GRAHSPSEDComponent`."""

    name: str = "agn_grahsp"
    feii_wave_nm: Array | None = None
    feii_lumin: Array | None = None
    line_wave_nm: Array | None = None
    line_broad: Array | None = None
    line_narrow_sy2: Array | None = None
    line_narrow_liner: Array | None = None


@dataclass(frozen=True)
class GRAHSPSEDComponent:
    r"""SEDComponent adapter wrapping the GRAHSP AGN model.

    Notes
    -----
    **JIT-compatible**: yes (``agn_type`` and component toggles are static).
    **Additive**: writes ``sed_intrinsic = sed_intrinsic + L_nu_grahsp``,
    initialising from zeros if upstream did not.
    **Wavelength convention**: tengri uses Å rest-frame; GRAHSP uses nm. The
    adapter performs the unit conversion internally.
    **Output convention**: tengri's ``sed_intrinsic`` is :math:`L_\nu`
    [erg/s/Hz]; GRAHSP returns :math:`L_\lambda` [erg/s/nm]. Conversion:
    :math:`L_\nu = L_\lambda \, \lambda^2 / c`.
    """

    config: GRAHSPSEDComponentConfig = field(default_factory=GRAHSPSEDComponentConfig)
    name: str = "agn_grahsp"
    parameter_prefix: str = "agn_grahsp_"

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Free parameters this component owns.

        All names start with ``agn_grahsp_`` per NAMING_CONTRACT §3.2.
        Defaults track Buchner+ 2024 Table parameters in §2.1.6.
        """
        return [
            ParamDeclaration(
                "agn_grahsp_l5100",
                LogUniform(1.0e38, 1.0e50),
                "lambda*L_lambda(5100Å) in erg/s",
            ),
            ParamDeclaration(
                "agn_grahsp_uvslope",
                Fixed(0.0),
                "BBB UV slope alpha_1 [dimensionless]",
            ),
            ParamDeclaration(
                "agn_grahsp_plslope",
                Uniform(-2.7, -1.0),
                "BBB optical slope alpha_2 [dimensionless]",
            ),
            ParamDeclaration(
                "agn_grahsp_plbendloc_nm",
                Uniform(50.0, 150.0),
                "BBB bend wavelength [nm]",
            ),
            ParamDeclaration(
                "agn_grahsp_plbendwidth",
                LogUniform(0.1, 10.0),
                "BBB bend width Lambda [dex]",
            ),
            ParamDeclaration(
                "agn_grahsp_cutoff_nm",
                Fixed(10000.0),
                "BBB IR cutoff [nm]; <0 disables",
            ),
            ParamDeclaration(
                "agn_grahsp_a_lines",
                LogUniform(0.3, 20.0),
                "Line strength scale Alines [dimensionless]",
            ),
            ParamDeclaration(
                "agn_grahsp_a_feii",
                LogUniform(0.6, 32.0),
                "FeII forest strength relative to broad H-beta",
            ),
            ParamDeclaration(
                "agn_grahsp_linewidth_kms",
                Fixed(10000.0),
                "Line FWHM [km/s]",
            ),
            ParamDeclaration(
                "agn_grahsp_fcov",
                Uniform(0.05, 0.95),
                "Torus covering factor at 12 um",
            ),
            ParamDeclaration(
                "agn_grahsp_si",
                Uniform(-4.0, 4.0),
                "Si feature strength [dimensionless]",
            ),
            ParamDeclaration(
                "agn_grahsp_cool_lam_um",
                Uniform(10.0, 30.0),
                "Cool dust peak wavelength [um]",
            ),
            ParamDeclaration(
                "agn_grahsp_cool_width",
                Uniform(0.2, 0.65),
                "Cool dust log-width [dex]",
            ),
            ParamDeclaration(
                "agn_grahsp_hot_lam_um",
                Uniform(1.0, 5.5),
                "Hot dust peak wavelength [um]",
            ),
            ParamDeclaration(
                "agn_grahsp_hot_width",
                Uniform(0.2, 0.65),
                "Hot dust log-width [dex]",
            ),
            ParamDeclaration(
                "agn_grahsp_hot_fcov",
                LogUniform(0.04, 10.0),
                "Hot/cool peak ratio in lambda*L_lambda",
            ),
            ParamDeclaration(
                "agn_grahsp_ebv",
                LogUniform(0.01, 10.0),
                "Galaxy E(B-V) [mag] (also seen by AGN)",
            ),
            ParamDeclaration(
                "agn_grahsp_ebv_agn",
                LogUniform(0.01, 0.1),
                "AGN-only additional E(B-V) [mag]",
            ),
        ]

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: Array | None = None,
    ) -> GRAHSPSEDComponentState:
        """Load the HDF5 template bundle once and cache it as JAX arrays.

        Parameters
        ----------
        ssp_data : ignored
        wave_grid : ignored — templates are independent of the user grid.
        """
        del ssp_data, wave_grid
        if self.config.template_path is not None:
            templates: GRAHSPTemplates = load_grahsp_templates(self.config.template_path)
        else:
            templates = load_grahsp_templates()
        return GRAHSPSEDComponentState(
            name=self.name,
            feii_wave_nm=templates.feii_wave_nm,
            feii_lumin=templates.feii_lumin,
            line_wave_nm=templates.line_wave_nm,
            line_broad=templates.line_broad,
            line_narrow_sy2=templates.line_narrow_sy2,
            line_narrow_liner=templates.line_narrow_liner,
        )

    def apply(
        self,
        state: PipelineState,
        params: Mapping[str, Array],
        templates_state: GRAHSPSEDComponentState | None = None,
    ) -> PipelineState:
        r"""Add GRAHSP AGN emission to ``state.sed_intrinsic``.

        Parameters
        ----------
        state : PipelineState
            ``state.wave`` is rest-frame Å.
        params : mapping
            ``agn_grahsp_*`` keys.
        templates_state : GRAHSPSEDComponentState, optional
            Pre-loaded template tensors. If ``None``, a fresh bundle is
            loaded (eager file I/O — avoid in JITed paths).

        Returns
        -------
        PipelineState
            Updated with ``sed_intrinsic`` augmented and ``derived``
            keys ``sed_grahsp``, ``L_agn_bol``, ``L_agn_torus``.
        """
        if templates_state is None:
            templates_state = self.precompute()

        wave_angstrom = state.wave
        wave_nm = wave_angstrom * 0.1

        cfg = self.config
        # Build component-by-component, gated by the config toggles.
        bbb = (
            sbpl_bbb(
                wave_nm=wave_nm,
                l5100=jnp.asarray(params["agn_grahsp_l5100"]),
                uvslope=jnp.asarray(params["agn_grahsp_uvslope"]),
                plslope=jnp.asarray(params["agn_grahsp_plslope"]),
                plbendloc_nm=jnp.asarray(params["agn_grahsp_plbendloc_nm"]),
                plbendwidth=jnp.asarray(params["agn_grahsp_plbendwidth"]),
                cutoff_nm=jnp.asarray(params.get("agn_grahsp_cutoff_nm", 10000.0)),
            )
            if cfg.include_bbb
            else jnp.zeros_like(wave_nm)
        )
        if cfg.include_lines:
            broad, narrow = gaussian_lines(
                wave_nm=wave_nm,
                line_wave_nm=templates_state.line_wave_nm,
                line_broad=templates_state.line_broad,
                line_narrow_sy2=templates_state.line_narrow_sy2,
                line_narrow_liner=templates_state.line_narrow_liner,
                l5100=jnp.asarray(params["agn_grahsp_l5100"]),
                a_lines=jnp.asarray(params["agn_grahsp_a_lines"]),
                linewidth_kms=jnp.asarray(params["agn_grahsp_linewidth_kms"]),
                agn_type=cfg.agn_type,
            )
        else:
            broad = jnp.zeros_like(wave_nm)
            narrow = jnp.zeros_like(wave_nm)
        feii = (
            feii_forest(
                wave_nm=wave_nm,
                template_wave_nm=templates_state.feii_wave_nm,
                template_lumin=templates_state.feii_lumin,
                l5100=jnp.asarray(params["agn_grahsp_l5100"]),
                a_lines=jnp.asarray(params["agn_grahsp_a_lines"]),
                a_feii=jnp.asarray(params["agn_grahsp_a_feii"]),
            )
            if cfg.include_feii and cfg.agn_type == 1
            else jnp.zeros_like(wave_nm)
        )
        torus = (
            torus_dust_continuum(
                wave_nm=wave_nm,
                l5100=jnp.asarray(params["agn_grahsp_l5100"]),
                fcov=jnp.asarray(params["agn_grahsp_fcov"]),
                cool_lam_um=jnp.asarray(params["agn_grahsp_cool_lam_um"]),
                cool_width=jnp.asarray(params["agn_grahsp_cool_width"]),
                hot_lam_um=jnp.asarray(params["agn_grahsp_hot_lam_um"]),
                hot_width=jnp.asarray(params["agn_grahsp_hot_width"]),
                hot_fcov=jnp.asarray(params["agn_grahsp_hot_fcov"]),
            )
            if cfg.include_torus
            else jnp.zeros_like(wave_nm)
        )
        si = (
            si_feature(
                wave_nm=wave_nm,
                l5100=jnp.asarray(params["agn_grahsp_l5100"]),
                fcov=jnp.asarray(params["agn_grahsp_fcov"]),
                si=jnp.asarray(params["agn_grahsp_si"]),
            )
            if cfg.include_si and cfg.include_torus
            else jnp.zeros_like(wave_nm)
        )
        # Mirror upstream activategtorus: clip si so total torus stays >= 0.
        si = jnp.maximum(si, -torus)

        bbb_intrinsic = bbb + broad + narrow + feii
        torus_intrinsic = torus + si

        if cfg.apply_attenuation:
            _, factor_agn = attenuation_factors(
                wave_nm=wave_nm,
                ebv=jnp.asarray(params["agn_grahsp_ebv"]),
                ebv_agn=jnp.asarray(params["agn_grahsp_ebv_agn"]),
            )
            bbb_total = bbb_intrinsic * factor_agn
            torus_total = torus_intrinsic * factor_agn
        else:
            bbb_total = bbb_intrinsic
            torus_total = torus_intrinsic

        # L_lambda [erg/s/nm] -> L_nu [erg/s/Hz]: L_nu = L_lambda * lambda^2 / c
        L_lambda_total = bbb_total + torus_total
        L_nu = L_lambda_total * wave_nm**2 / _C_NM_PER_S

        if state.sed_intrinsic is None:
            new_sed = L_nu
        else:
            new_sed = state.sed_intrinsic + L_nu

        # Bolometric quantities (computed from L_lambda on the nm grid).
        L_bol_BBB = bolometric_luminosity_bbb(wave_nm, bbb_intrinsic)
        L_bol_torus = bolometric_luminosity_torus(wave_nm, torus_intrinsic)
        # Diagnostic: integrated AGN-side absorbed luminosity (intrinsic - attenuated).
        # Reported but NOT injected into the Dale 2014 dust-emission loop —
        # GRAHSP's torus already empirically captures dust re-radiation.
        L_agn_absorbed = jnp.trapezoid(
            (bbb_intrinsic + torus_intrinsic) - L_lambda_total, wave_nm
        )

        new_derived = dict(state.derived)
        new_derived["sed_grahsp"] = L_nu
        new_derived["L_agn_bol"] = L_bol_BBB
        new_derived["L_agn_torus"] = L_bol_torus
        new_derived["L_agn_absorbed"] = L_agn_absorbed
        return state.with_(sed_intrinsic=new_sed, derived=new_derived)
