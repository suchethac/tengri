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

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import erfc

from tengri.components.agn.grahsp.attenuation import PREVOT_DEFAULTS, attenuation_factors
from tengri.components.agn.grahsp.balmer import (
    _BALMER_EDGE_NM,
    _BC_CONVOLUTION_THRESHOLD_NM,
    balmer_continuum,
)
from tengri.components.agn.grahsp.bbb import XRAY_FLOOR_NM, floor_disc_xray, sbpl_bbb
from tengri.components.agn.grahsp.bolometric import LYMAN_LIMIT_NM
from tengri.components.agn.grahsp.disc import netzer_disc, select_disc_model
from tengri.components.agn.grahsp.lines import (
    _HBETA_BROAD_RATIO,
    _HBETA_NARROW_RATIO,
    _LAMBDA_5100_NM,
    feii_forest,
    gaussian_lines,
)
from tengri.components.agn.grahsp.publication import (
    PUBLICATION_HI_NM,
    WINDOW_SIGMA,
    gauss_legendre_segments,
    linewidth_sigma_nm,
    publication_breakpoints,
    window_edges,
)
from tengri.components.agn.grahsp.published import (
    feii_segment_product,
    line_expectation,
    log_breaks,
    log_gaussian_expectation,
    lognormal_c0,
)
from tengri.components.agn.grahsp.templates import (
    GRAHSPTemplates,
    load_grahsp_templates,
)
from tengri.components.agn.grahsp.torus import (
    SI_DEFAULTS,
    _log_gaussian_pair,
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

#: Gauss-Legendre segments per decade for the smooth BBB quadrature (published L_* only).
SBPL_PER_DECADE: int = 20
#: Breakpoints per decade for the Balmer continuum quadrature (published L_* only).
BALMER_PER_DECADE: int = 20
#: Gauss-Legendre points per side of the log-normal torus expectation (published L_* only).
LOG_GL_ORDER: int = 16
#: Gauss-Hermite points per line and per Si feature (published L_* only).
LINE_GH_ORDER: int = 13
#: Upper edge of the Balmer quadrature [nm]; its edge window ends below ~400 nm for any width.
_BALMER_HI_NM: float = 600.0


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


def _sign_change_roots(fn, lo: Array, hi: Array, iterations: int = 4) -> Array:
    r"""Regula-falsi roots of ``fn`` on the intervals ``[lo, hi]`` with a sign change.

    Intervals without a sign change return their midpoint, so the split is harmless there.
    Used to place the kinks of :math:`\max(0,-f)` at the crossings of :math:`f`; the roots are
    stopped from the gradient, because :math:`\max(0,-f)` is continuous there.

    Parameters
    ----------
    fn : callable
        ``fn(wave_nm) -> ndarray`` (vectorized).
    lo, hi : array_like, shape (n,)
        Interval ends [nm], ascending.
    iterations : int, optional
        Regula-falsi steps.

    Returns
    -------
    ndarray, shape (n,)
        The crossing estimates [nm], inside ``[lo, hi]``.
    """
    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)
    f_lo = fn(lo)
    f_hi = fn(hi)
    crosses = f_lo * f_hi < 0.0
    safe = jnp.where(crosses, f_hi - f_lo, 1.0)
    root = jnp.where(crosses, lo - f_lo * (hi - lo) / safe, 0.5 * (lo + hi))
    a, b, fa, fb = lo, hi, f_lo, f_hi
    for _ in range(iterations):
        f_root = fn(root)
        left = fa * f_root < 0.0
        b = jnp.where(left, root, b)
        fb = jnp.where(left, f_root, fb)
        a = jnp.where(left, a, root)
        fa = jnp.where(left, fa, f_root)
        denom = jnp.where(crosses, fb - fa, 1.0)
        root = jnp.where(crosses, a - fa * (b - a) / denom, 0.5 * (lo + hi))
    return root


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
                "Cool dust log-Gaussian standard deviation [dex]",
            ),
            ParamDeclaration(
                "agn_grahsp_hot_lam_um",
                Uniform(1.0, 5.5),
                "Hot dust peak wavelength [um]",
            ),
            ParamDeclaration(
                "agn_grahsp_hot_width",
                Uniform(0.2, 0.65),
                "Hot dust log-Gaussian standard deviation [dex]",
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
        ztable_data: Any | None = None,
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

        wave_nm = state.wave * 0.1
        bbb_intrinsic, torus_intrinsic = self._pieces(wave_nm, params, templates_state)
        factor_agn = self._attenuation(wave_nm, params)
        if factor_agn is None:
            bbb_total, torus_total = bbb_intrinsic, torus_intrinsic
        else:
            bbb_total = bbb_intrinsic * factor_agn
            torus_total = torus_intrinsic * factor_agn

        # L_lambda [erg/s/nm] -> L_nu [erg/s/Hz]: L_nu = L_lambda * lambda^2 / c
        L_nu = (bbb_total + torus_total) * wave_nm**2 / _C_NM_PER_S
        published = self._published_luminosities(params, templates_state)
        return state.add_intrinsic(L_nu).with_(
            derived=state.derived.with_(sed_grahsp=L_nu, **published),
        )

    def _pieces(
        self,
        wave_nm: Array,
        params: Mapping[str, Array],
        templates_state: GRAHSPSEDComponentState,
    ) -> tuple[Array, Array]:
        """Intrinsic AGN-side L_lambda [erg/s/nm] of the BBB side and of the torus side.

        Parameters
        ----------
        wave_nm : array_like, shape (n_wave,)
            Rest-frame wavelength [nm]; the caller's grid for the emitted SED, the
            publication grid for the published luminosities.
        params : mapping
            ``agn_grahsp_*`` keys.
        templates_state : GRAHSPSEDComponentState
            Pre-loaded template tensors.

        Returns
        -------
        bbb_intrinsic : ndarray, shape (n_wave,)
            BBB + lines + FeII + Balmer continuum [erg/s/nm], before attenuation.
        torus_intrinsic : ndarray, shape (n_wave,)
            Torus continuum + Si feature [erg/s/nm], before attenuation.

        Notes
        -----
        **JIT-compatible**: yes. Pointwise in ``wave_nm``.
        """
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

        return bbb + broad + narrow + feii + balmer, torus + si

    def _attenuation(self, wave_nm: Array, params: Mapping[str, Array]) -> Array | None:
        """AGN-side bi-attenuation factor 10**(-(E(B-V)+E(B-V)_AGN) A_SMC / 2.5), or None.

        Parameters
        ----------
        wave_nm : array_like, shape (n_wave,)
            Rest-frame wavelength [nm].
        params : mapping
            ``agn_grahsp_ebv`` and ``agn_grahsp_ebv_agn``.

        Returns
        -------
        ndarray, shape (n_wave,) or None
            The factor [dimensionless]; None when ``apply_attenuation`` is off.
        """
        if not self.config.apply_attenuation:
            return None
        _, factor_agn = attenuation_factors(
            wave_nm=wave_nm,
            ebv=jnp.asarray(params["agn_grahsp_ebv"]),
            ebv_agn=jnp.asarray(params["agn_grahsp_ebv_agn"]),
        )
        return factor_agn

    def _published_luminosities(
        self,
        params: Mapping[str, Array],
        templates_state: GRAHSPSEDComponentState,
    ) -> dict[str, Array]:
        r"""Published ``L_agn_bol``, ``L_agn_torus``, ``L_agn_absorbed`` [erg/s].

        Each is the integral of the emitted AGN-side SED under its documented definition, on
        the component's own publication grid (:mod:`~tengri.components.agn.grahsp.publication`),
        so it does not depend on the caller's wavelength array:

        .. math::

           L_{\rm bol}^{\rm BBB} = \int_{91.2\,\mathrm{nm}}^{\infty}
               L_\lambda^{\rm BBB+lines+FeII+Balmer}\,\mathrm{d}\lambda, \qquad
           L_{\rm bol}^{\rm TOR} = \int_0^\infty L_\lambda^{\rm TOR}\,\mathrm{d}\lambda, \qquad
           L_{\rm abs} = \int \left(L_\lambda^{\rm int} - L_\lambda^{\rm att}\right)
               \mathrm{d}\lambda,

        where the BBB and torus terms are intrinsic (before the bi-attenuation) and the
        absorbed power is the intrinsic minus the attenuated AGN-side power (Buchner+ 2024
        section 2.1.4-2.1.5, arXiv:2405.19297).

        Parameters
        ----------
        params : mapping
            ``agn_grahsp_*`` keys.
        templates_state : GRAHSPSEDComponentState
            Pre-loaded template tensors.

        Returns
        -------
        dict
            ``L_agn_bol``, ``L_agn_torus``, ``L_agn_absorbed`` [erg/s], scalars.

        Notes
        -----
        **JIT-compatible**: yes. The quadrature is segmented Gauss-Legendre in ln(lambda)
        between the breakpoints from ``_publication_breaks``; their count is fixed by the
        configuration. **Gradient-safe**: yes. ``L_agn_absorbed`` is exactly zero when
        ``apply_attenuation`` is off.
        """
        if self._default_published_path():
            return self._published_default(params, templates_state)
        breaks = self._publication_breaks(params, templates_state)
        lam_nm, weight_nm = gauss_legendre_segments(breaks)
        bbb_i, torus_i = self._pieces(lam_nm, params, templates_state)
        factor_agn = self._attenuation(lam_nm, params)
        in_bol_band = lam_nm >= LYMAN_LIMIT_NM
        L_bol = jnp.sum(weight_nm * jnp.where(in_bol_band, bbb_i, 0.0))
        L_torus = jnp.sum(weight_nm * torus_i)
        if factor_agn is None:
            L_absorbed = jnp.zeros((), dtype=L_bol.dtype)
        else:
            L_absorbed = jnp.sum(weight_nm * (bbb_i + torus_i) * (1.0 - factor_agn))
        return {"L_agn_bol": L_bol, "L_agn_torus": L_torus, "L_agn_absorbed": L_absorbed}

    def _published_default(
        self,
        params: Mapping[str, Array],
        templates_state: GRAHSPSEDComponentState,
    ) -> dict[str, Array]:
        """Published L_* on the default pieces: closed forms and fixed-node quadrature."""
        absorbed = (
            self._published_absorbed(params, templates_state)
            if self.config.apply_attenuation
            else jnp.zeros(())
        )
        return {
            "L_agn_bol": self._published_bol(params, templates_state),
            "L_agn_torus": self._published_torus(params),
            "L_agn_absorbed": absorbed,
        }

    def _default_published_path(self) -> bool:
        """True when the closed-form path covers the configuration (default torus, BBB, FeII)."""
        cfg = self.config
        return (
            cfg.torus_model == "gaussian"
            and cfg.disc_model is None
            and cfg.feii_template == "bruhweiler2008"
        )

    def _sbpl_at(self, wave_nm: Array, params: Mapping[str, Array]) -> Array:
        """Intrinsic smooth-bending power-law BBB [erg/s/nm], floored at the X-ray edge."""
        spec = sbpl_bbb(
            wave_nm=wave_nm,
            l5100=jnp.asarray(params["agn_grahsp_l5100"]),
            uvslope=jnp.asarray(params["agn_grahsp_uvslope"]),
            plslope=jnp.asarray(params["agn_grahsp_plslope"]),
            plbendloc_nm=jnp.asarray(params["agn_grahsp_plbendloc_nm"]),
            plbendwidth=jnp.asarray(params["agn_grahsp_plbendwidth"]),
            cutoff_nm=jnp.asarray(params.get("agn_grahsp_cutoff_nm", 10000.0)),
        )
        return floor_disc_xray(wave_nm, spec)

    def _balmer_breaks(self, lo_nm: float, params: Mapping[str, Array]) -> Array:
        """Breakpoints [nm] of the Balmer continuum: log grid, the 250 nm switch, the edge."""
        linewidth = jnp.asarray(params["agn_grahsp_linewidth_kms"])
        edge = jnp.asarray([_BALMER_EDGE_NM])
        windows = window_edges(edge, linewidth_sigma_nm(edge, linewidth))
        fixed = jnp.asarray(log_breaks(lo_nm, _BALMER_HI_NM, BALMER_PER_DECADE))
        breaks = jnp.sort(
            jnp.concatenate([fixed, jnp.asarray([_BC_CONVOLUTION_THRESHOLD_NM]), windows])
        )
        return jnp.clip(breaks, lo_nm, _BALMER_HI_NM)

    def _balmer_at(self, wave_nm: Array, params: Mapping[str, Array]) -> Array:
        """Intrinsic Balmer continuum [erg/s/nm] (type 1 only; the caller gates it)."""
        return balmer_continuum(
            wave_nm=wave_nm,
            l5100=jnp.asarray(params["agn_grahsp_l5100"]),
            a_bc=jnp.asarray(params.get("agn_grahsp_a_bc", 0.0)),
            linewidth_kms=jnp.asarray(params["agn_grahsp_linewidth_kms"]),
        )

    def _line_terms(
        self,
        params: Mapping[str, Array],
        templates_state: GRAHSPSEDComponentState,
    ) -> tuple[Array, Array, Array]:
        """Line centers [nm], sigmas [nm] and Lambda_5100-scaled amplitudes [erg/s].

        Each line contributes ``amplitude * G(lambda; center, sigma)`` with G of unit area, as
        :func:`gaussian_lines` evaluates it: the amplitude is ``510 nm * line_lumin``.
        """
        cfg = self.config
        l_agn = jnp.asarray(params["agn_grahsp_l5100"]) / _LAMBDA_5100_NM
        a_lines = jnp.asarray(params["agn_grahsp_a_lines"])
        l_broad = _HBETA_BROAD_RATIO * l_agn * a_lines
        l_narrow = _HBETA_NARROW_RATIO * l_agn * a_lines
        if cfg.agn_type == 1:
            lumin = (
                l_broad * templates_state.line_broad + l_narrow * templates_state.line_narrow_sy2
            )
        elif cfg.agn_type == 2:
            lumin = l_narrow * templates_state.line_narrow_sy2
        else:
            lumin = l_narrow * templates_state.line_narrow_liner
        centers = jnp.asarray(templates_state.line_wave_nm)
        sigmas = linewidth_sigma_nm(centers, jnp.asarray(params["agn_grahsp_linewidth_kms"]))
        return centers, sigmas, _LAMBDA_5100_NM * lumin

    def _feii_amplitude(self, params: Mapping[str, Array]) -> Array:
        """FeII forest amplitude a_FeII * L_Hbeta,broad [erg/s/nm per unit template]."""
        l_agn = jnp.asarray(params["agn_grahsp_l5100"]) / _LAMBDA_5100_NM
        l_broad = _HBETA_BROAD_RATIO * l_agn * jnp.asarray(params["agn_grahsp_a_lines"])
        return jnp.asarray(params["agn_grahsp_a_feii"]) * l_broad

    def _torus_terms(self, params: Mapping[str, Array]) -> tuple[Array, Array, Array, Array]:
        r"""Closed-form amplitudes and log-centers of the two log-normal torus components.

        With :math:`K = (\lambda L_{12\,\mu m})/(12000\,\mathrm{nm})/T(12\,\mu\mathrm{m})` the
        torus normalization, the cool component integrates to :math:`K\,\lambda_{\rm COOL}\,
        C_0(W_{\rm COOL})` and the hot to :math:`K\,f_{\rm hot}\,\lambda_{\rm COOL}\,
        C_0(W_{\rm HOT})` (see :mod:`~tengri.components.agn.grahsp.published`).

        Returns
        -------
        amp_cool, amp_hot : ndarray
            Closed-form integrals of the components [erg/s].
        mu_cool, mu_hot : ndarray
            :math:`\log_{10}` of the component peak wavelengths [um].
        """
        cool_lam = jnp.asarray(params["agn_grahsp_cool_lam_um"])
        hot_lam = jnp.asarray(params["agn_grahsp_hot_lam_um"])
        w_cool = jnp.asarray(params["agn_grahsp_cool_width"])
        w_hot = jnp.asarray(params["agn_grahsp_hot_width"])
        fhot = jnp.asarray(params["agn_grahsp_hot_fcov"])
        l_torus = (
            2.5 * jnp.asarray(params["agn_grahsp_l5100"]) * jnp.asarray(params["agn_grahsp_fcov"])
        )
        mu_cool = jnp.log10(cool_lam)
        mu_hot = jnp.log10(hot_lam)
        spec12 = _log_gaussian_pair(jnp.log10(12.0), mu_cool, w_cool, mu_hot, w_hot, fhot)
        scale = (l_torus / 12000.0) / spec12
        amp_cool = scale * cool_lam * lognormal_c0(w_cool)
        amp_hot = scale * fhot * cool_lam * lognormal_c0(w_hot)
        return amp_cool, amp_hot, mu_cool, mu_hot

    def _si_terms(self, params: Mapping[str, Array]) -> tuple[Array, Array, Array, Array]:
        """Si feature: prefactor, closed-form integral, feature centers [nm] and sigmas [nm].

        The feature is ``l_torus/12000 * si * (A_em G_em - A_ab G_ab)`` with unit-peak Gaussians
        G and amplitudes from :data:`SI_DEFAULTS`, so with unit-area G the closed integral is
        ``l_torus/12000 * si * sqrt(2 pi) * (A_em sigma_em - A_ab sigma_ab)``.
        """
        l_torus = (
            2.5 * jnp.asarray(params["agn_grahsp_l5100"]) * jnp.asarray(params["agn_grahsp_fcov"])
        )
        pref = (l_torus / 12000.0) * jnp.asarray(params["agn_grahsp_si"])
        em_amp = SI_DEFAULTS["si_em_ampl"]
        ab_amp = em_amp * SI_DEFAULTS["si_ratio"]
        centers = jnp.asarray([SI_DEFAULTS["si_em_lam_nm"], SI_DEFAULTS["si_abs_lam_nm"]])
        sigmas = jnp.asarray([SI_DEFAULTS["si_em_width_nm"], SI_DEFAULTS["si_abs_width_nm"]])
        # Amplitude vector: (+A_em, -A_ab) times the unit-area Gaussian integral sqrt(2 pi) s.
        signed = jnp.asarray([em_amp, -ab_amp]) * math.sqrt(2.0 * math.pi) * sigmas
        return pref, jnp.sum(signed) * pref, centers, sigmas

    def _published_bol(self, params, templates_state) -> Array:
        """Intrinsic BBB + lines + FeII + Balmer above the Lyman limit [erg/s], exact pieces."""
        cfg = self.config
        type1 = cfg.agn_type == 1
        total = jnp.zeros(())
        if cfg.include_bbb:
            lam, w = gauss_legendre_segments(
                jnp.asarray(log_breaks(LYMAN_LIMIT_NM, PUBLICATION_HI_NM, SBPL_PER_DECADE))
            )
            total = total + jnp.sum(w * self._sbpl_at(lam, params))
        if cfg.include_balmer and type1:
            lam, w = gauss_legendre_segments(self._balmer_breaks(LYMAN_LIMIT_NM, params))
            total = total + jnp.sum(w * self._balmer_at(lam, params))
        if cfg.include_feii and type1:
            wave = jnp.asarray(templates_state.feii_wave_nm)
            total = total + self._feii_amplitude(params) * jnp.trapezoid(
                templates_state.feii_lumin, wave
            )
        if cfg.include_lines:
            centers, sigmas, amps = self._line_terms(params, templates_state)
            cut = 0.5 * erfc((LYMAN_LIMIT_NM - centers) / (math.sqrt(2.0) * sigmas))
            total = total + jnp.sum(amps * cut)
        return total

    def _published_torus(self, params) -> Array:
        """Intrinsic torus continuum + Si feature over (0, inf) [erg/s].

        The torus and Si terms are closed-form integrals; the emitted ``max(si, -torus)`` clip
        adds its correction :meth:`_si_clip_correction`, which binds inside the declared prior
        box (corner si = -4, cool peak 30 um, width 0.2 dex).
        """
        cfg = self.config
        if not cfg.include_torus:
            return jnp.zeros(())
        amp_cool, amp_hot, _, _ = self._torus_terms(params)
        total = amp_cool + amp_hot
        if cfg.include_si:
            total = total + self._si_terms(params)[1]
            total = total + self._si_clip_correction(params)
        return total

    def _si_clip_correction(
        self,
        params: Mapping[str, Array],
        weight_fn=None,
    ) -> Array:
        r"""Integral of the Si clip correction :math:`\int\max(0,-(T+S))\,h\,\mathrm{d}\lambda`.

        The emitted torus term uses :math:`\max(\mathrm{si}, -\mathrm{torus})`, which equals
        :math:`\mathrm{si}+\max(0,-\mathrm{torus}-\mathrm{si})`. Both terms carry the factor
        :math:`l_{\rm torus}/12000`, so the correction is :math:`(l_{\rm torus}/12000)` times the
        dimensionless integral of :math:`\max(0,-(T+S))`, where :math:`T` is the torus shape
        normalized to unity at 12 um and :math:`S` the Si shape per unit ``si``. It is nonzero
        only inside the Si feature windows, and is integrated by Gauss-Legendre on each window
        split into segments of one sigma (the clip is a kink of the integrand, so each segment
        must hold at most one crossing).

        Parameters
        ----------
        params : mapping
            ``agn_grahsp_*`` keys.
        weight_fn : callable, optional
            ``weight_fn(wave_nm) -> ndarray`` multiplying the integrand (e.g. 1 - attenuation).

        Returns
        -------
        ndarray, shape ()
            The correction [erg/s].
        """
        l_torus = (
            2.5 * jnp.asarray(params["agn_grahsp_l5100"]) * jnp.asarray(params["agn_grahsp_fcov"])
        )
        si = jnp.asarray(params["agn_grahsp_si"])
        centers = np.asarray([SI_DEFAULTS["si_em_lam_nm"], SI_DEFAULTS["si_abs_lam_nm"]])
        sigmas = np.asarray([SI_DEFAULTS["si_em_width_nm"], SI_DEFAULTS["si_abs_width_nm"]])
        offsets = np.arange(-WINDOW_SIGMA, WINDOW_SIGMA + 1.0)
        breaks = jnp.sort(jnp.asarray((centers[:, None] + sigmas[:, None] * offsets).ravel()))
        cool_lam = jnp.asarray(params["agn_grahsp_cool_lam_um"])
        hot_lam = jnp.asarray(params["agn_grahsp_hot_lam_um"])
        w_cool = jnp.asarray(params["agn_grahsp_cool_width"])
        w_hot = jnp.asarray(params["agn_grahsp_hot_width"])
        fhot = jnp.asarray(params["agn_grahsp_hot_fcov"])
        mu_cool, mu_hot = jnp.log10(cool_lam), jnp.log10(hot_lam)
        spec12 = _log_gaussian_pair(jnp.log10(12.0), mu_cool, w_cool, mu_hot, w_hot, fhot)
        em_amp = SI_DEFAULTS["si_em_ampl"]
        ab_amp = em_amp * SI_DEFAULTS["si_ratio"]

        def net_shape(lam):
            """Torus shape plus Si shape (dimensionless); the clip acts where it is negative."""
            torus_shape = (
                _log_gaussian_pair(jnp.log10(lam / 1000.0), mu_cool, w_cool, mu_hot, w_hot, fhot)
                / spec12
            )
            g_em = jnp.exp(-0.5 * ((lam - centers[0]) / sigmas[0]) ** 2)
            g_ab = jnp.exp(-0.5 * ((lam - centers[1]) / sigmas[1]) ** 2)
            return torus_shape + si * (em_amp * g_em - ab_amp * g_ab)

        lo, hi = breaks[:-1], breaks[1:]
        split = _sign_change_roots(net_shape, lo, hi)
        # lo, split, hi per interval: ascending, each crossing a segment boundary.
        lam, weight = gauss_legendre_segments(jnp.stack([lo, split, hi], axis=1).ravel())
        crossing = jnp.maximum(0.0, -net_shape(lam))
        if weight_fn is not None:
            crossing = crossing * weight_fn(lam)
        return (l_torus / 12000.0) * jnp.sum(weight * crossing)

    def _published_absorbed(self, params, templates_state) -> Array:
        """Intrinsic-minus-attenuated AGN-side power, fixed-node quadrature [erg/s]."""
        cfg = self.config
        type1 = cfg.agn_type == 1

        def h_fn(wave_nm):
            return 1.0 - self._attenuation(wave_nm, params)

        total = jnp.zeros(())
        if cfg.include_bbb:
            lam, w = gauss_legendre_segments(
                jnp.asarray(
                    log_breaks(
                        XRAY_FLOOR_NM,
                        PUBLICATION_HI_NM,
                        SBPL_PER_DECADE,
                        extra=(PREVOT_DEFAULTS["lam_break_nm"],),
                    )
                )
            )
            total = total + jnp.sum(w * self._sbpl_at(lam, params) * h_fn(lam))
        if cfg.include_balmer and type1:
            lam, w = gauss_legendre_segments(self._balmer_breaks(XRAY_FLOOR_NM, params))
            total = total + jnp.sum(w * self._balmer_at(lam, params) * h_fn(lam))
        if cfg.include_feii and type1:
            total = total + self._feii_amplitude(params) * feii_segment_product(
                h_fn,
                templates_state.feii_wave_nm,
                templates_state.feii_lumin,
            )
        if cfg.include_lines:
            centers, sigmas, amps = self._line_terms(params, templates_state)
            total = total + jnp.sum(amps * line_expectation(h_fn, centers, sigmas, LINE_GH_ORDER))
        if cfg.include_torus:
            amp_cool, amp_hot, mu_cool, mu_hot = self._torus_terms(params)
            split = PREVOT_DEFAULTS["lam_break_nm"]
            total = total + amp_cool * log_gaussian_expectation(
                h_fn, mu_cool, params["agn_grahsp_cool_width"], split, LOG_GL_ORDER
            )
            total = total + amp_hot * log_gaussian_expectation(
                h_fn, mu_hot, params["agn_grahsp_hot_width"], split, LOG_GL_ORDER
            )
            if cfg.include_si:
                pref, _, centers, sigmas = self._si_terms(params)
                total = total + self._si_clip_correction(params, weight_fn=h_fn)
                em_ab = jnp.asarray(
                    [
                        SI_DEFAULTS["si_em_ampl"],
                        -SI_DEFAULTS["si_em_ampl"] * SI_DEFAULTS["si_ratio"],
                    ]
                )
                weighted = em_ab * math.sqrt(2.0 * math.pi) * sigmas
                expect = line_expectation(h_fn, centers, sigmas, LINE_GH_ORDER)
                total = total + pref * jnp.sum(weighted * expect)
        return total

    def _publication_breaks(
        self,
        params: Mapping[str, Array],
        templates_state: GRAHSPSEDComponentState,
    ) -> Array:
        """Breakpoints [nm] of the publication grid for the selected pieces.

        Parameters
        ----------
        params : mapping
            ``agn_grahsp_linewidth_kms`` moves the feature windows.
        templates_state : GRAHSPSEDComponentState
            Template nodes of the selected pieces become breakpoints.

        Returns
        -------
        ndarray, shape (n_breaks,)
            Ascending breakpoints [nm] from the X-ray floor to ``_PUBLICATION_HI_NM``.
        """
        cfg = self.config
        linewidth = jnp.asarray(params["agn_grahsp_linewidth_kms"])
        static = (
            LYMAN_LIMIT_NM,
            _BC_CONVOLUTION_THRESHOLD_NM,
            PREVOT_DEFAULTS["lam_break_nm"],
        )
        arrays: list[Array] = []
        if cfg.include_lines:
            centers = templates_state.line_wave_nm
            arrays.append(window_edges(centers, linewidth_sigma_nm(centers, linewidth)))
        if cfg.include_balmer and cfg.agn_type == 1:
            edge = jnp.asarray([_BALMER_EDGE_NM])
            arrays.append(window_edges(edge, linewidth_sigma_nm(edge, linewidth)))
        if cfg.include_feii and cfg.agn_type == 1:
            if cfg.feii_template == "veroncetty2004":
                arrays.append(templates_state.feii_vc04_wave_nm)
            else:
                arrays.append(templates_state.feii_wave_nm)
        if cfg.include_bbb and cfg.disc_model == "netzer":
            arrays.append(templates_state.disc_wave_nm)
        if cfg.include_torus and cfg.torus_model == "mn12":
            arrays.append(templates_state.torus_mn12_wave_nm)
            if cfg.include_si:
                arrays.append(templates_state.torus_mn12_si_wave_nm)
        elif cfg.include_torus and cfg.include_si:
            si_centers = jnp.asarray([SI_DEFAULTS["si_em_lam_nm"], SI_DEFAULTS["si_abs_lam_nm"]])
            si_sigma = jnp.asarray([SI_DEFAULTS["si_em_width_nm"], SI_DEFAULTS["si_abs_width_nm"]])
            arrays.append(window_edges(si_centers, si_sigma))
        return publication_breakpoints(XRAY_FLOOR_NM, PUBLICATION_HI_NM, static, arrays)
