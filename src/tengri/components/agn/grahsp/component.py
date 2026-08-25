# SPDX-License-Identifier: BSD-3-Clause
"""GRAHSPSEDComponent: composable SEDComponent adapter for the GRAHSP model.

Mirrors the canonical adapter pattern of :mod:`tengri.components.radio.component`.
The component publishes (all under ``state.derived``):

- ``sed_grahsp``: :math:`L_\\nu` [erg/s/Hz] of the full GRAHSP AGN-side SED.
- ``L_agn_bol``: bolometric BBB luminosity (paper §2.1.4 ``lumBolBBB``).
- ``L_agn_torus``: torus bolometric luminosity (``lumBolTOR``).
- ``L_agn_absorbed``: energy absorbed by the AGN bi-attenuation (the
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
   behavior, configure ``dust_*`` with the same E(B-V) and an SMC-like
   law (see :func:`tengri.components.dust.attenuation.prevot_smc`).
2. **The AGN side does not need an explicit re-emission loop.** GRAHSP's
   torus parameters (``fcov``, cool/hot log-Gaussians) already empirically
   model the dust re-radiation of absorbed UV/optical AGN light (paper
   §2.1.5). We publish ``L_agn_absorbed`` for diagnostics only: it is
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
from tengri.components.agn.grahsp.balmer import balmer_continuum
from tengri.components.agn.grahsp.bbb import floor_disc_xray, sbpl_bbb
from tengri.components.agn.grahsp.bolometric import (
    bolometric_luminosity_bbb,
    bolometric_luminosity_torus,
)
from tengri.components.agn.grahsp.disc import netzer_disc, select_disc_model
from tengri.components.agn.grahsp.lines import feii_forest, gaussian_lines
from tengri.components.agn.grahsp.templates import (
    GRAHSPTemplates,
    load_grahsp_templates,
)
from tengri.components.agn.grahsp.torus import (
    si_feature,
    torus_dust_continuum,
    torus_mn12_continuum,
    torus_mn12_si,
)
from tengri.parameters.priors import Fixed, LogUniform, Uniform
from tengri.protocols.component import (
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
    SEDComponentState,
)

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
    include_balmer : bool
        Include the Grandi 1982 Balmer continuum (only added for
        ``agn_type == 1``; controlled in strength by ``agn_grahsp_a_bc``).
    apply_attenuation : bool
        Apply the GRAHSP bi-attenuation curve to the AGN spectrum. When
        ``False`` the component emits the intrinsic SED only.
    torus_model : {"gaussian", "mn12"}
        ``"gaussian"`` -> empirical log-Gaussian torus (``activategtorus``);
        ``"mn12"`` -> Mor & Netzer 2012 template torus (``activatetorus``).
        **Static** (structural choice).
    feii_template : {"bruhweiler2008", "veroncetty2004"}
        FeII forest template. **Static**.
    disc_model : {None, "netzer"}
        ``None`` -> smooth bending power-law BBB; ``"netzer"`` -> Netzer
        accretion-disc grid (replaces the BBB). **Static**.
    disc_m, disc_a, disc_mdot : str
        Netzer disc grid selection (log10 M_BH/Msun, spin, Eddington ratio).
        Only used when ``disc_model == "netzer"``. **Static**.
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
    include_balmer: bool = True
    apply_attenuation: bool = True
    torus_model: str = "gaussian"
    feii_template: str = "bruhweiler2008"
    disc_model: str | None = None
    disc_m: str = "8.0"
    disc_a: str = "0"
    disc_mdot: str = "0.3"
    template_path: str | None = None


@dataclass(frozen=True)
class GRAHSPSEDComponentState(SEDComponentState):
    r"""Cached templates for :class:`GRAHSPSEDComponent`.

    Holds every GRAHSP template tensor so :meth:`GRAHSPSEDComponent.apply`
    reads pre-loaded JAX arrays under JIT (no eager file I/O in the traced
    path). The Mor & Netzer 2012 template torus, Veron-Cetty 2004 FeII and
    Netzer disc tensors are ``None`` when an older bundle lacks them.
    """

    name: str = "agn_grahsp"
    feii_wave_nm: Array | None = None
    feii_lumin: Array | None = None
    line_wave_nm: Array | None = None
    line_broad: Array | None = None
    line_narrow_sy2: Array | None = None
    line_narrow_liner: Array | None = None
    feii_vc04_wave_nm: Array | None = None
    feii_vc04_lumin: Array | None = None
    torus_mn12_wave_nm: Array | None = None
    torus_mn12_avg: Array | None = None
    torus_mn12_lo: Array | None = None
    torus_mn12_hi: Array | None = None
    torus_mn12_si_wave_nm: Array | None = None
    torus_mn12_si_lumin: Array | None = None
    disc_wave_nm: Array | None = None
    disc_lumin: Array | None = None
    disc_m: tuple[str, ...] | None = None
    disc_a: tuple[str, ...] | None = None
    disc_mdot: tuple[str, ...] | None = None


@dataclass(frozen=True)
class GRAHSPSEDComponent:
    r"""SEDComponent adapter wrapping the GRAHSP AGN model.

    Notes
    -----
    **JIT-compatible**: yes (``agn_type`` and component toggles are static).
    **Additive**: writes ``sed_intrinsic = sed_intrinsic + L_nu_grahsp``,
    initializing from zeros if upstream did not.
    **Wavelength convention**: tengri uses Å rest-frame; GRAHSP uses nm. The
    adapter performs the unit conversion internally.
    **Output convention**: tengri's ``sed_intrinsic`` is :math:`L_\nu`
    [erg/s/Hz]; GRAHSP returns :math:`L_\lambda` [erg/s/nm]. Conversion:
    :math:`L_\nu = L_\lambda \, \lambda^2 / c`.
    """

    config: GRAHSPSEDComponentConfig = field(default_factory=GRAHSPSEDComponentConfig)
    name: str = "agn_grahsp"
    parameter_prefix: str = "agn_grahsp_"

    def citations(self) -> tuple[str, ...]:
        """Registry keys for GRAHSP and the active empirical sub-models.

        Returns the GRAHSP paper (Buchner+2024) plus one key per
        *enabled* sub-model, so a run only advertises what it actually
        uses (the "cite what you use" policy of
        :func:`tengri.citations.collect_citations`).

        Returns
        -------
        tuple of str
            Registry keys present in ``references.bib``:

            - ``buchner2024``: always (the GRAHSP model itself).
            - ``grandi1982``: if ``include_balmer``.
            - ``bruhweiler_verner2008`` / ``veron_cetty2004``: if
              ``include_feii``, selected by ``feii_template``.
            - ``mor_netzer2012``: if ``include_torus`` and
              ``torus_model == "mn12"`` (template torus).
            - ``netzer_trakhtenbrot2014``: if ``disc_model == "netzer"``.

        Notes
        -----
        **JIT-compatible**: no, reads ``self.config`` (static metadata).
        """
        cfg = self.config
        keys: list[str] = ["buchner2024"]
        if cfg.include_balmer:
            keys.append("grandi1982")
        if cfg.include_feii:
            if cfg.feii_template == "veroncetty2004":
                keys.append("veron_cetty2004")
            else:
                keys.append("bruhweiler_verner2008")
        if cfg.include_torus and cfg.torus_model == "mn12":
            keys.append("mor_netzer2012")
        if cfg.disc_model == "netzer":
            keys.append("netzer_trakhtenbrot2014")
        return tuple(keys)

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
            ParamDeclaration(
                "agn_grahsp_a_bc",
                Fixed(0.0),
                "Balmer continuum strength relative to powerlaw (Grandi 1982); 0 disables",
            ),
            ParamDeclaration(
                "agn_grahsp_tor_temp",
                Uniform(-1.0, 1.0),
                "MN12 template-torus temperature blend [-1, +1] (torus_model='mn12')",
            ),
            ParamDeclaration(
                "agn_grahsp_tor_cutoff_um",
                Fixed(1.2),
                "MN12 template-torus short-lambda cutoff [um] (torus_model='mn12')",
            ),
        ]

    def outputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component derived keys this GRAHSP AGN component publishes.

        See :func:`tengri.forward.orchestrator.validate_pipeline`.

        Notes
        -----
        Listed as an alternate publisher of ``L_agn_bol`` in
        ``tengri.forward.orchestrator._ALTERNATE_PUBLISHERS`` alongside
        :class:`AGNSEDComponent`. The pipeline factory chooses one variant
        by configuration; only one ever runs at a time.
        """
        return (
            DerivedKey("L_agn_bol", "erg/s", "AGN bolometric luminosity (GRAHSP variant)"),
            DerivedKey("L_agn_torus", "erg/s", "Torus thermal luminosity component"),
            DerivedKey("L_agn_absorbed", "erg/s", "Energy absorbed by torus"),
            DerivedKey("sed_grahsp", "erg/s/Hz", "Full GRAHSP AGN SED on pipeline wave grid"),
        )

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: Array | None = None,
    ) -> GRAHSPSEDComponentState:
        """Load the HDF5 template bundle once and cache it as JAX arrays.

        Parameters
        ----------
        ssp_data : ignored
        wave_grid : ignored; templates are independent of the user grid.
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
            feii_vc04_wave_nm=templates.feii_vc04_wave_nm,
            feii_vc04_lumin=templates.feii_vc04_lumin,
            torus_mn12_wave_nm=templates.torus_mn12_wave_nm,
            torus_mn12_avg=templates.torus_mn12_avg,
            torus_mn12_lo=templates.torus_mn12_lo,
            torus_mn12_hi=templates.torus_mn12_hi,
            torus_mn12_si_wave_nm=templates.torus_mn12_si_wave_nm,
            torus_mn12_si_lumin=templates.torus_mn12_si_lumin,
            disc_wave_nm=templates.disc_wave_nm,
            disc_lumin=templates.disc_lumin,
            disc_m=templates.disc_m,
            disc_a=templates.disc_a,
            disc_mdot=templates.disc_mdot,
        )

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, Array],
        templates_state: GRAHSPSEDComponentState | None = None,
        ssp_data: Any | None = None,
        template_data: Any | None = None,
    ) -> ForwardState:
        r"""Add GRAHSP AGN emission to ``state.sed_intrinsic``.

        ``ssp_data`` is accepted for Protocol uniformity but unused: this
        component reads only from ``state`` and ``params``.

        Parameters
        ----------
        state : ForwardState
            ``state.wave`` is rest-frame Å.
        params : mapping
            ``agn_grahsp_*`` keys.
        templates_state : GRAHSPSEDComponentState, optional
            Pre-loaded template tensors. If ``None``, a fresh bundle is
            loaded (eager file I/O: avoid in JITed paths).

        Returns
        -------
        ForwardState
            Updated with ``sed_intrinsic`` augmented and ``derived``
            keys ``sed_grahsp``, ``L_agn_bol``, ``L_agn_torus``.
        """
        if templates_state is None:
            templates_state = self.precompute()

        wave_angstrom = state.wave
        wave_nm = wave_angstrom * 0.1

        cfg = self.config
        l5100 = jnp.asarray(params["agn_grahsp_l5100"])
        zeros = jnp.zeros_like(wave_nm)
        # --- Big blue bump: SBPL power-law (default) or Netzer disc grid. ---
        if not cfg.include_bbb:
            bbb = zeros
        elif cfg.disc_model == "netzer":
            idx = select_disc_model(
                templates_state.disc_m,
                templates_state.disc_a,
                templates_state.disc_mdot,
                m=cfg.disc_m,
                a=cfg.disc_a,
                mdot=cfg.disc_mdot,
            )
            bbb = netzer_disc(
                wave_nm=wave_nm,
                l5100=l5100,
                disc_wave_nm=templates_state.disc_wave_nm,
                disc_lumin_model=templates_state.disc_lumin[idx],
            )
        else:
            bbb = sbpl_bbb(
                wave_nm=wave_nm,
                l5100=l5100,
                uvslope=jnp.asarray(params["agn_grahsp_uvslope"]),
                plslope=jnp.asarray(params["agn_grahsp_plslope"]),
                plbendloc_nm=jnp.asarray(params["agn_grahsp_plbendloc_nm"]),
                plbendwidth=jnp.asarray(params["agn_grahsp_plbendwidth"]),
                cutoff_nm=jnp.asarray(params.get("agn_grahsp_cutoff_nm", 10000.0)),
            )
        # GRAHSP has no X-ray physics: floor the disc below the alpha_ox
        # corona's blue edge so it does not double-count with the corona
        # (#1168). No-op for the netzer branch (already bounded via interp).
        bbb = floor_disc_xray(wave_nm, bbb)
        if cfg.include_lines:
            broad, narrow = gaussian_lines(
                wave_nm=wave_nm,
                line_wave_nm=templates_state.line_wave_nm,
                line_broad=templates_state.line_broad,
                line_narrow_sy2=templates_state.line_narrow_sy2,
                line_narrow_liner=templates_state.line_narrow_liner,
                l5100=l5100,
                a_lines=jnp.asarray(params["agn_grahsp_a_lines"]),
                linewidth_kms=jnp.asarray(params["agn_grahsp_linewidth_kms"]),
                agn_type=cfg.agn_type,
            )
        else:
            broad = zeros
            narrow = zeros
        # --- FeII forest: Bruhweiler+Verner 2008 (default) or Veron-Cetty 2004. ---
        if cfg.feii_template == "veroncetty2004":
            feii_wave, feii_lumin = (
                templates_state.feii_vc04_wave_nm,
                templates_state.feii_vc04_lumin,
            )
        else:
            feii_wave, feii_lumin = templates_state.feii_wave_nm, templates_state.feii_lumin
        feii = (
            feii_forest(
                wave_nm=wave_nm,
                template_wave_nm=feii_wave,
                template_lumin=feii_lumin,
                l5100=l5100,
                a_lines=jnp.asarray(params["agn_grahsp_a_lines"]),
                a_feii=jnp.asarray(params["agn_grahsp_a_feii"]),
            )
            if cfg.include_feii and cfg.agn_type == 1
            else zeros
        )
        # --- Balmer continuum (Grandi 1982); only for broad-line AGN (type 1). ---
        balmer = (
            balmer_continuum(
                wave_nm=wave_nm,
                l5100=l5100,
                a_bc=jnp.asarray(params.get("agn_grahsp_a_bc", 0.0)),
                linewidth_kms=jnp.asarray(params["agn_grahsp_linewidth_kms"]),
            )
            if cfg.include_balmer and cfg.agn_type == 1
            else zeros
        )
        # --- Torus: log-Gaussian (default) or Mor & Netzer 2012 template. ---
        if not cfg.include_torus:
            torus = zeros
            si = zeros
        elif cfg.torus_model == "mn12":
            torus = torus_mn12_continuum(
                wave_nm=wave_nm,
                l5100=l5100,
                fcov=jnp.asarray(params["agn_grahsp_fcov"]),
                tor_temp=jnp.asarray(params.get("agn_grahsp_tor_temp", 0.0)),
                tor_cutoff_um=jnp.asarray(params.get("agn_grahsp_tor_cutoff_um", 1.2)),
                mn12_wave_nm=templates_state.torus_mn12_wave_nm,
                mn12_avg=templates_state.torus_mn12_avg,
                mn12_lo=templates_state.torus_mn12_lo,
                mn12_hi=templates_state.torus_mn12_hi,
            )
            si = (
                torus_mn12_si(
                    wave_nm=wave_nm,
                    l5100=l5100,
                    fcov=jnp.asarray(params["agn_grahsp_fcov"]),
                    si=jnp.asarray(params["agn_grahsp_si"]),
                    si_wave_nm=templates_state.torus_mn12_si_wave_nm,
                    si_lumin=templates_state.torus_mn12_si_lumin,
                )
                if cfg.include_si
                else zeros
            )
        else:
            torus = torus_dust_continuum(
                wave_nm=wave_nm,
                l5100=l5100,
                fcov=jnp.asarray(params["agn_grahsp_fcov"]),
                cool_lam_um=jnp.asarray(params["agn_grahsp_cool_lam_um"]),
                cool_width=jnp.asarray(params["agn_grahsp_cool_width"]),
                hot_lam_um=jnp.asarray(params["agn_grahsp_hot_lam_um"]),
                hot_width=jnp.asarray(params["agn_grahsp_hot_width"]),
                hot_fcov=jnp.asarray(params["agn_grahsp_hot_fcov"]),
            )
            si = (
                si_feature(
                    wave_nm=wave_nm,
                    l5100=l5100,
                    fcov=jnp.asarray(params["agn_grahsp_fcov"]),
                    si=jnp.asarray(params["agn_grahsp_si"]),
                )
                if cfg.include_si
                else zeros
            )
        # Mirror upstream ``mask_negative``: clip si so total torus stays >= 0.
        si = jnp.maximum(si, -torus)

        bbb_intrinsic = bbb + broad + narrow + feii + balmer
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

        # Bolometric quantities (computed from L_lambda on the nm grid).
        L_bol_BBB = bolometric_luminosity_bbb(wave_nm, bbb_intrinsic)
        L_bol_torus = bolometric_luminosity_torus(wave_nm, torus_intrinsic)
        # Diagnostic: integrated AGN-side absorbed luminosity (intrinsic - attenuated).
        # Reported but NOT injected into the Dale 2014 dust-emission loop;
        # GRAHSP's torus already empirically captures dust re-radiation.
        L_agn_absorbed = jnp.trapezoid((bbb_intrinsic + torus_intrinsic) - L_lambda_total, wave_nm)

        return state.add_intrinsic(L_nu).with_(
            derived=state.derived.with_(
                sed_grahsp=L_nu,
                L_agn_bol=L_bol_BBB,
                L_agn_torus=L_bol_torus,
                L_agn_absorbed=L_agn_absorbed,
            ),
        )
