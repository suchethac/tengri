# SPDX-License-Identifier: BSD-3-Clause
"""AGNSEDComponent: unified AGN emission (disc + torus + lines + jets/corona).

Dispatches to any registered AGN model (``"multicolor_agn"``,
``"kubota_done_full"``, ``"adaf"``, ``"unified_nlr_blr"``, ``"skirtor"``,
``"silva04"``, ``"cat3d_wind"``, ``"relagn"``, ``"qsogen"``) via
:func:`tengri.components.agn.unified.resolve_agn_model`.

Cross-component reads
---------------------
Nothing required; AGN is a self-contained additive emitter. ``redshift``
is read from :data:`BARE_NAME_ALLOWLIST`.

Cross-component publications
----------------------------

- ``state.derived["L_agn_bol"]`` (scalar, erg/s): the AGN accretion power,
  :math:`10^{\\mathtt{agn\\_log\\_lbol}} L_\\odot`, integrated over all directions
  (the same at every inclination). Consumed by
  :class:`tengri.components.xray.component.XRaySEDComponent` and
  :class:`tengri.components.radio.component.RadioSEDComponent` via their
  documented fallback (``state.derived.get("L_agn_bol", 0.0)``).
- ``state.derived["log_L_agn_direct_los"]`` (dex re erg/s, composable model only):
  :math:`\\log_{10}` of the line-of-sight isotropic-equivalent bolometric luminosity
  of the AGN's direct emission, :math:`\\int L_\\nu(i)\\,d\\nu` with
  :math:`L_\\nu(i) = 2\\cos i\\,D_\\nu + H_\\nu` (disc and warm zones scale with
  :math:`\\cos i`, the corona does not), before the torus screen and the polar dust.
  Compare this key, not ``L_agn_bol``, with a catalog :math:`L_{\\rm bol}` from a
  bolometric correction. Until #3076 it was ``log_L_agn_los``.
- The composable energy ledger (#3076, NAMING_CONTRACT 4c "Composable runner"), computed
  on request as properties (``pred.properties[...]``, ``model.predict_properties``), not
  in the forward pass: ``log_L_agn_los`` (the integral of the emitted ``sed_agn``), the
  angle-integrated ``log_L_agn_<piece>`` and line-of-sight ``log_L_agn_<piece>_los`` of the
  disc, torus, lines and polar dust, ``log_L_agn_{disc,torus,lines}_unemitted``,
  ``log_L_agn_power``, every one with an ``L_agn_*_lsun`` sibling, and
  ``agn_ledger_closed`` (1 when the pieces add up to the accretion power).
- ``state.derived["log_L_12um"]``, ``["log_L_6um"]`` (dex re erg/s):
  :math:`\\log_{10}\\nu L_\\nu` at 12 and 6 µm of the AGN's own emission (disc +
  torus + polar dust for the composable model, the whole SED for a monolithic
  one). The ``lopez24`` X-ray corona is anchored to ``log_L_12um``.
- ``state.derived["sed_agn"]``: the AGN SED contribution
  (erg/s/Hz, shape n_wave) for diagnostics.
- ``state.derived["sed_agn_disc"]``, ``["sed_agn_torus"]``,
  ``["sed_agn_lines"]`` (nlr + blr + feii), ``["sed_agn_polar"]``
  (erg/s/Hz, shape n_wave): the composable-runner's own per-sub-block
  rest-frame SEDs (NAMING_CONTRACT §4b.5), summing exactly to
  ``sed_agn``. Published only when ``model == "composable"`` (absent for
  monolithic models, which have no separate sub-blocks to decompose);
  ``sed_agn_polar`` is a zeros-shaped array (not absent) whenever the
  composable model is configured with an ``agn_attenuation_block`` other
  than ``"polar_dust"``, since polar-dust re-emission is inactive rather
  than unconfigured for that recipe.

Architectural notes
-------------------
``agn_torus_frac`` is **never** auto-derived from ``agn_cos_inc`` /
``theta_torus`` (CLAUDE.md gotcha: gradient discontinuity). It is
read directly from ``params`` as an independent free parameter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components._term_response import term_band_response
from tengri.components.agn._ir_frac_window import (
    PROVISIONAL_TORUS_FRAC,
    log10_torus_frac_from_window,
)
from tengri.components.agn._lbol_reference import (
    _LOG10_L_SUN,
    reference_evaluation,
    rescale,
)
from tengri.components.agn._params import PARAMS as _AGN_PARAMS
from tengri.components.agn._phys import log10_nu_lnu_at
from tengri.components.agn._publication import with_lsun_siblings
from tengri.components.agn.adaf import adaf_accretion_diagnostics
from tengri.components.agn.blocks._ledger import ledger_property_specs
from tengri.components.agn.blocks._protocol import collect_block_templates
from tengri.components.agn.blocks.runner import (
    DIRECT_LOS_LOG_LUMINOSITY_KEY,
    composable_agn_l_nu,
)
from tengri.components.agn.blocks.torus_library_transmission import below_library_index
from tengri.components.agn.unified import (
    monolithic_models_with_line_components,
    preset_overrides,
    resolve_agn_model,
)
from tengri.components.template_threading import TemplateThreading
from tengri.components.xray.xray import COS_INC_REF_30DEG as _XRAY_COS_INC_REF_30DEG
from tengri.parameters.resolve import require_redshift
from tengri.protocols.component import (
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
    SEDComponentState,
    declared_default,
)

__all__ = [
    "AGNSEDComponent",
    "AGNSEDComponentConfig",
]

#: ``(alpha, beta, delta)`` of the ADAF when the parameters are not supplied: the defaults of
#: :func:`~tengri.components.agn.adaf.adaf_spectrum`.
_ADAF_DEFAULTS: tuple[float, float, float] = (0.3, 0.5, 0.1)


#: Wavelengths [Angstrom] at which the composable AGN publishes its 6 and 12 um diagnostics.
_MIR_DIAGNOSTIC_AA = (6.0e4, 1.2e5)


def _amplitude_probe_draws() -> tuple[dict[str, float], ...]:
    """Probe draws for the rank-1 check, spread over the declared amplitude priors (#1118).

    ``agn_log_lbol`` is probed at the low bound, midpoint and high bound of its prior, with
    ``agn_lum_ratio`` at its default; ``agn_lum_ratio`` is probed at its midpoint and high
    bound, with ``agn_log_lbol`` at its default. The ratio's low bound is zero, which
    switches the whole AGN off, so it is not a probe point.
    """
    priors = {decl.name: decl.prior for decl in _AGN_PARAMS}
    lbol, ratio = priors["agn_log_lbol"], priors["agn_lum_ratio"]
    lbol_default = declared_default(_AGN_PARAMS, "agn_log_lbol")
    ratio_default = declared_default(_AGN_PARAMS, "agn_lum_ratio")
    lbol_mid = 0.5 * (lbol.lo + lbol.hi)
    ratio_mid = 0.5 * (ratio.lo + ratio.hi)
    lbol_draws = [
        {"agn_log_lbol": value, "agn_lum_ratio": ratio_default}
        for value in (lbol.lo, lbol_mid, lbol.hi)
    ]
    ratio_draws = [
        {"agn_log_lbol": lbol_default, "agn_lum_ratio": value} for value in (ratio_mid, ratio.hi)
    ]
    return tuple(lbol_draws + ratio_draws)


@dataclass(frozen=True)
class AGNSEDComponentConfig(SEDComponentConfig):
    """Frozen knobs for :class:`AGNSEDComponent`.

    Parameters
    ----------
    name : str
        Diagnostic identifier. Default ``"agn"``.
    model : str
        AGN model registry key. One of ``"multicolor_agn"`` (Kubota & Done
        outer-zone disc + 2-T torus), ``"kubota_done_full"`` (full 3-zone
        disc), ``"adaf"``, ``"unified_nlr_blr"``, ``"skirtor"``,
        ``"silva04"``, ``"cat3d_wind"``, ``"relagn"``, ``"qsogen"``, or
        ``"composable"`` (block-composed via the selectors below).
        Default ``"multicolor_agn"``.
    agn_disc_block, agn_nlr_block, agn_blr_block, agn_feii_block,
    agn_torus_block, agn_attenuation_block : str
        Composable-AGN block selectors. Only consulted when
        ``model == "composable"``, the runner reads them from this config
        (they are static strings, not traced JAX values, so they cannot
        ride in ``params``). Each defaults to ``"none"`` so non-composable
        AGN models receive harmless no-op selectors that the underlying
        registry function absorbs via ``**kwargs``.
    agn_ir_frac_window : tuple of float
        CIGALE ``lambda_fracAGN`` window (#2662), ``(lambda_min_um, lambda_max_um)``. ``(0, 0)``
        (default) is the total-dust case: ``agn_ir_frac`` is the fraction of the total dust
        luminosity. A window makes ``agn_ir_frac`` the fraction of the light in
        ``[lambda_min, lambda_max]``; ``lambda_min == lambda_max != 0`` is monochromatic.
        A static build-time value, not a parameter.
    agn_polar_law : str
        Polar-dust extinction curve: ``"smc"`` (default), ``"calzetti"``,
        ``"gaskell"`` or ``"bongiorno"``. A static string read by the runner.
    agn_norm : str
        Cross-block normalization policy (#556). ``"cigale_joint"``
        (default) ties the disc, torus and polar to CIGALE's single
        ``agn_power`` reference via the fixed SKIRTOR template ratios
        (energy-conserving; only active for ``agn_torus_block="skirtor"`` +
        ``agn_ir_frac>0``). ``"independent"`` keeps each component on its own
        luminosity scale (disc on ``agn_log_lbol``, torus on ``agn_power``,
        polar via the legacy face-on proxy), the GRAHSP/AGNfitter-style
        bookkeeping. A static string, read by the runner like the block
        selectors.
    """

    name: str = "agn"
    model: str = "multicolor_agn"
    agn_disc_block: str = "none"
    agn_nlr_block: str = "none"
    agn_blr_block: str = "none"
    agn_feii_block: str = "none"
    agn_torus_block: str = "none"
    agn_attenuation_block: str = "none"
    agn_norm: str = "cigale_joint"
    agn_polar_law: str = "smc"
    agn_ir_frac_window: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class AGNSEDComponentState(SEDComponentState):
    """State for AGN component.

    Holds optional filter passbands when ``approx=WavePrecomp()``
    is set on the parent SEDModel. The component uses them at
    :meth:`AGNSEDComponent.apply` time to filter-integrate the
    analytically computed AGN SED and publish ``agn_phot_lnu_precomp``.

    Also caches the template libraries of whichever composable blocks the
    recipe selected, so they become Parameter ops rather than baked
    Constants.
    """

    name: str = "agn"
    filter_waves: Any | None = None
    filter_trans: Any | None = None
    skirtor_templates: Any | None = None
    #: ``{"<category>/<name>": pytree}`` for every selected block that
    #: declared a ``template_loader``. See ``collect_block_templates``.
    block_templates: Any | None = None
    #: Build-time :class:`BelowLibraryIndex` (X-ray absorption data below the library edge) of
    #: the composable torus's library sightline, on the grid the runner evaluates (the master
    #: grid plus the diagnostic nodes). A static constant, so the traced photometry grid needs
    #: no cross-section evaluation of its own.
    torus_below_index: Any | None = None
    #: The master grid object the index was built on. The index is used only when the apply
    #: state carries this very object, so a grid that merely has the same length (an explicit
    #: or resampled grid) takes the dense path.
    torus_master_wave: Any | None = None


@dataclass(frozen=True)
class AGNSEDComponent(TemplateThreading):
    """SEDComponent adapter for the unified AGN model registry.

    Notes
    -----
    **JIT-compatible**: yes, :meth:`apply` delegates to a registered
    AGN function which is pure JAX.
    **Additive**: writes ``sed_intrinsic = sed_intrinsic + L_AGN(λ)``,
    matching the convention used by :class:`RadioSEDComponent` and
    :class:`XRaySEDComponent`.
    """

    config: AGNSEDComponentConfig = field(default_factory=AGNSEDComponentConfig)
    name: str = "agn"
    parameter_prefix: str = "agn_"
    _state: AGNSEDComponentState | None = None

    def citations(self) -> tuple[str, ...]:
        """AGN sub-block citations (disc, torus, BLR) flow from
        :mod:`tengri.citations.associations`; the wrapper has no
        always-required paper."""
        return ()

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Free parameters this component owns.

        Returns the canonical :data:`PARAMS` tuple from
        ``tengri.components.agn._params``. The legacy ``_AGN_PARAMS``
        bucket in ``tengri.parameters._builders`` is a derived view
        of the same tuple, exactly (R41, #2214, retired the one orphan
        the bucket used to add on top).

        The full ``agn_*`` parameter superset is declared so that any
        registered model can run without missing keys. Users freely
        ``Fixed`` whatever a particular model does not consume; the
        AGN registry functions accept ``**kwargs`` and ignore unused
        names.
        """
        return list(_AGN_PARAMS)

    #: Parameters that set a sub-block's amplitude and never its spectral shape. Freeing
    #: one scales every term together, so the build-time band response stays valid (#1118).
    AMPLITUDE_PARAMS: ClassVar[frozenset[str]] = frozenset({"agn_log_lbol", "agn_lum_ratio"})

    #: Probe draws for the rank-1 check (#1107, #1118), read from the declared priors. They
    #: differ only in the amplitude, so every sub-block must come back proportional to the
    #: first draw. Each draw sets both amplitudes, so the probe never reads a free value.
    EMITTER_PROBE_INPUTS: ClassVar[tuple[dict[str, float], ...]] = _amplitude_probe_draws()

    #: Wavelengths [Angstrom] at which the AGN publishes point diagnostics (6 and 12 um); the
    #: term response carries each term's shape there (``S_diag``).
    EMITTER_DIAGNOSTIC_NODES: ClassVar[tuple[float, ...]] = _MIR_DIAGNOSTIC_AA

    #: The composable sub-blocks whose sum is the SED, in ``emission_terms`` order.
    _TERM_KEYS: ClassVar[tuple[str, ...]] = ("disc", "torus", "lines", "polar")

    #: The term band response carries each term's amplitude at its reference wavelength when
    #: no amplitude parameter is free (#2987), so the photometry does not re-evaluate the AGN
    #: on every call. Read by the term-response builder in ``tengri.SEDModel``.
    FIXED_TERM_AMPLITUDES: ClassVar[bool] = True

    def term_response_refusal(self, free: set[str], fixed: Mapping[str, Any]) -> str | None:
        """Why the rank-1 band response must not be built for this AGN, or ``None``.

        Parameters
        ----------
        free : set of str
            Free parameter names of the whole model.
        fixed : mapping
            Build-time fixed values, keyed by parameter name.

        Returns
        -------
        str or None
            A reason the dense per-call filter integral must stay, or ``None`` when every
            sub-block is a fixed spectral shape times an amplitude.

        Notes
        -----
        A monolithic AGN has no sub-blocks to decompose. An attenuation block reddens the
        disc, and the galaxy dust screen on the AGN (checked by the caller) puts a runtime
        transmission in front of it; both are declined rather than approximated. The
        dust-coupled torus fraction (``agn_ir_frac > 0``) reads the absorbed power at run
        time, which the term amplitudes do not carry.
        """
        if self.config.model != "composable":
            return (
                f"AGN model {self.config.model!r} is monolithic: it has no per-sub-block "
                "decomposition to take a band response over"
            )
        shape = sorted(p for p in free if p.startswith("agn_") and p not in self.AMPLITUDE_PARAMS)
        if shape:
            return f"free agn parameters other than the luminosity and ratio: {shape}"
        if self.config.agn_attenuation_block != "none":
            return (
                f"attenuation block {self.config.agn_attenuation_block!r} reddens the disc "
                "and lines, a shape outside the band-response contract"
            )
        ir_frac = fixed.get("agn_ir_frac")
        if ir_frac is not None and float(jnp.asarray(ir_frac)) != 0.0:
            return (
                "agn_ir_frac > 0 ties the torus amplitude to the dust-absorbed power at run "
                "time, which the band response does not carry"
            )
        return None

    def emission_terms(
        self,
        params: Mapping[str, Any],
        wave: jnp.ndarray,
        *,
        agn_log_lbol: jnp.ndarray | None = None,
        agn_lum_ratio: jnp.ndarray | None = None,
        template_state: Mapping[str, Any] | None = None,
    ) -> dict[str, jnp.ndarray]:
        r"""The composable AGN sub-block SEDs, unsummed (#1118).

        Each term is one sub-block of the composable runner: the disc, the torus, the
        lines (NLR, BLR and FeII, masked as the runner masks them) and the polar dust
        re-emission. Their sum is the AGN SED that :meth:`apply` integrates.

        Parameters
        ----------
        params : mapping
            Full parameter dict; reads the ``agn_*`` keys.
        wave : array_like, shape (n_wave,)
            Rest-frame wavelength grid [Angstrom].
        agn_log_lbol : array_like, scalar, optional
            :math:`\log_{10}(L_{\rm bol}/L_\odot)`. Defaults to ``params["agn_log_lbol"]``.
        agn_lum_ratio : array_like, scalar, optional
            Overrides ``params["agn_lum_ratio"]`` (default 1) for the probe draws.
        template_state : mapping, optional
            Per-block template libraries, as :meth:`apply` threads them.

        Returns
        -------
        dict of ndarray, each shape (n_wave,)
            ``{"disc", "torus", "lines", "polar"}`` [erg/s/Hz], scaled by ``agn_lum_ratio``.

        Notes
        -----
        **JIT/grad/vmap-safe.** Pure JAX; the block selectors are static config. A term is
        rank-1 in its amplitude only when its shape is set by fixed parameters, which the
        build-time probe verifies (see :meth:`term_response_refusal`).
        """
        from tengri.components.agn.blocks.runner import composable_agn_l_nu

        lbol = params["agn_log_lbol"] if agn_log_lbol is None else agn_log_lbol
        kwargs = self._composable_kwargs(params)
        if agn_lum_ratio is not None:
            kwargs["agn_lum_ratio"] = jnp.asarray(agn_lum_ratio)
        _, components = composable_agn_l_nu(
            wave,
            agn_log_lbol=lbol,
            return_components=True,
            template_state=template_state,
            strict_keys=False,
            **kwargs,
        )
        return {key: components[key] for key in self._TERM_KEYS}

    def _composable_kwargs(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Keyword arguments the composable runner receives from :meth:`apply`.

        Mirrors the ``agn_kwargs`` assembled in :meth:`apply` for the composable model, so
        :meth:`emission_terms` and the dense path evaluate the same function.
        """
        kwargs = numeric_agn_kwargs(params, params.get("agn_torus_frac", 0.5))
        kwargs["agn_disc_block"] = self.config.agn_disc_block
        kwargs["agn_nlr_block"] = self.config.agn_nlr_block
        kwargs["agn_blr_block"] = self.config.agn_blr_block
        kwargs["agn_feii_block"] = self.config.agn_feii_block
        kwargs["agn_torus_block"] = self.config.agn_torus_block
        kwargs["agn_attenuation_block"] = self.config.agn_attenuation_block
        kwargs["agn_norm"] = self.config.agn_norm
        kwargs["agn_polar_law"] = self.config.agn_polar_law
        return kwargs

    def _term_band_photometry(
        self,
        band: Mapping[str, Any],
        params: Mapping[str, Any],
        template_state: Mapping[str, Any] | None,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        """Filter-integrated AGN :math:`L_\\nu` and the 6 and 12 um logs from the term response.

        The amplitude of each term is its value at the reference wavelength divided by the
        shape there, so the photometry is ``sum_k A_k R_kf`` with no dense grid. The same
        amplitudes times the build-time term shapes at the two mid-infrared diagnostic nodes
        (``S_diag``) give the 6 and 12 um luminosities, so a model that reads ``log_L_12um``
        (the ``lopez24`` corona) does not keep a dense AGN evaluation alive for it.

        Returns
        -------
        phot : ndarray, shape (n_filters,)
            Filter-integrated rest-frame :math:`L_\\nu` [erg/s/Hz].
        mir_logs : tuple of ndarray, scalar
            ``(log_L_6um, log_L_12um)`` [dex, log10(erg/s/Hz) + log10(Hz)].
        """
        if "amp" in band:
            # Every amplitude parameter is fixed, so the build evaluated these at the fixed
            # values: the same numbers this branch would compute (#2987).
            amps = band["amp"]
        else:
            ref = self.emission_terms(params, band["lam_ref"], template_state=template_state)
            amps = jnp.stack([term[i] for i, term in enumerate(ref.values())]) / band["S_ref"]
        keys = list(self._TERM_KEYS)
        rows = jnp.asarray([keys.index(k) for k in ("disc", "torus", "polar")])
        mir_lnu = jnp.sum(amps[rows][:, None] * band["S_diag"][rows], axis=0)
        diag = jnp.asarray(_MIR_DIAGNOSTIC_AA, dtype=mir_lnu.dtype)
        return amps @ band["R"], _mir_logs(diag, mir_lnu, 0.0)

    def outputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component derived keys this AGN component publishes.

        See :func:`tengri.forward.orchestrator.validate_pipeline`.
        """
        return (
            DerivedKey(
                "L_agn_bol",
                "erg/s",
                "AGN accretion power 10**agn_log_lbol L_sun, integrated over all directions "
                "(independent of the inclination)",
            ),
            DerivedKey(
                "log_L_agn_bol",
                "dex",
                "log10(L_agn_bol / (erg/s)); float32-safe form "
                "(L_agn_bol ~1e46 overflows float32)",
            ),
            DerivedKey(
                "log_L_agn_direct_los",
                "dex",
                "log10 of the line-of-sight isotropic-equivalent bolometric luminosity "
                "[dex re erg/s] of the AGN direct emission, int L_nu(i) dnu with "
                "L_nu(i) = 2 cos i D_nu + H_nu, before the torus screen and the polar dust; "
                "composable model only. Compare with a catalog L_bol",
            ),
            DerivedKey(
                "agn_torus_frac_effective",
                "",
                "torus fraction the composable AGN was evaluated at: agn_torus_frac, or the "
                "value the agn_ir_frac coupling derives from the absorbed stellar power; "
                "composable model only (the energy ledger re-evaluates the chain at it)",
            ),
            DerivedKey(
                "agn_adaf_mdot",
                "",
                "ADAF accretion rate in Eddington units, mdot = Mdot / Mdot_Edd with "
                "Mdot_Edd = L_Edd / (0.1 c^2) (Mahadevan 1997, Eq. 4), derived from the "
                "radiated agn_log_lbol; composable model with the 'adaf' disc only",
            ),
            DerivedKey(
                "log_L_agn_adaf_accretion",
                "dex",
                "log10 of the ADAF accretion power mdot L_Edd = eta_eff Mdot c^2 [dex re "
                "erg/s] with eta_eff = 0.1 (Mahadevan 1997, Eq. 4), the power the paper "
                "compares L_ADAF against; agn_log_lbol stays the radiated luminosity "
                "Q_ie + delta Q+; composable model with the 'adaf' disc only",
            ),
            DerivedKey(
                "L_agn_bol_lsun",
                "Lsun",
                "10**agn_log_lbol, the AGN accretion power in solar luminosities",
            ),
            DerivedKey(
                "L_agn_direct_los_lsun",
                "Lsun",
                "log_L_agn_direct_los in solar luminosities; composable model only",
            ),
            DerivedKey(
                "L_agn_adaf_accretion_lsun",
                "Lsun",
                "log_L_agn_adaf_accretion in solar luminosities; composable model with the "
                "'adaf' disc only",
            ),
            DerivedKey("sed_agn", "erg/s/Hz", "AGN SED contribution on pipeline wave grid"),
            DerivedKey(
                "sed_agn_disc",
                "erg/s/Hz",
                "AGN disc-only rest-frame SED, composable model only "
                "(absent for monolithic AGN models)",
            ),
            DerivedKey(
                "sed_agn_torus",
                "erg/s/Hz",
                "AGN torus-only rest-frame SED, composable model only "
                "(absent for monolithic AGN models)",
            ),
            DerivedKey(
                "sed_agn_lines",
                "erg/s/Hz",
                "AGN NLR+BLR+FeII rest-frame SED, composable model only "
                "(absent for monolithic AGN models)",
            ),
            DerivedKey(
                "sed_agn_lines_attenuated",
                "erg/s/Hz",
                "AGN line-only rest-frame SED (NLR+BLR+FeII; GRAHSP lines+FeII) as it "
                "enters the pipeline SED, after the AGN's own screen and the host "
                "agn_screen; zeros for models without lines. Routed to the "
                "instrument-only kernel in the spectrum projection (#2565)",
            ),
            DerivedKey(
                "sed_agn_polar",
                "erg/s/Hz",
                "AGN polar-dust re-emission rest-frame SED (CIGALE "
                "skirtor2016 convention, Yang et al. 2020), composable "
                "model only; zeros unless agn_attenuation_block="
                "'polar_dust' (absent for monolithic AGN models)",
            ),
            DerivedKey(
                "L_2500_intrinsic",
                "erg/s/Hz",
                "AGN intrinsic disc L_nu at 2500 A (un-reddened); drives X-ray alpha_ox",
            ),
            DerivedKey(
                "L_4400_intrinsic",
                "erg/s/Hz",
                "AGN intrinsic disc L_nu at 4400 A (un-reddened); drives radio loudness",
            ),
            DerivedKey(
                "log_L_12um",
                "dex",
                "log10 nu L_nu at 12 um [dex re erg/s] of disc + torus + polar dust "
                "(the whole SED for monolithic models); drives the lopez24 alpha_IRX corona",
            ),
            DerivedKey(
                "log_L_6um",
                "dex",
                "log10 nu L_nu at 6 um [dex re erg/s] of disc + torus + polar dust "
                "(the whole SED for monolithic models)",
            ),
            DerivedKey(
                "agn_cos_inc",
                "dimensionless",
                "AGN cos(i), the X-ray corona tilts its Yang+2022 anisotropy "
                "to the same sightline (X-CIGALE yang20.py cosi; #980)",
            ),
        )

    def optional_inputs(self) -> tuple[DerivedKey, ...]:
        """Optional inputs: the dust publications and, under a CIGALE window, the host continuum.

        A window reads the host SED (``sed_intrinsic``), which holds the nebular continuum.
        Declaring ``sed_nebular`` keeps the fast nebular grid, which zeroes that continuum,
        from being armed for this chain (#2662).
        """
        if self.config.agn_ir_frac_window != (0.0, 0.0):
            nebular = DerivedKey(
                "sed_nebular",
                "erg/s/Hz",
                "Nebular continuum; read by the CIGALE lambda_fracAGN window (#2662)",
            )
            return (nebular, *self._cigale_joint_inputs())
        return self._cigale_joint_inputs()

    def _cigale_joint_inputs(self) -> tuple[DerivedKey, ...]:
        """Dust publications, when using CIGALE-joint cross-component norm.

        When ``agn_norm == "cigale_joint"``, the AGN component reads
        ``log_L_ir`` and ``L_absorbed`` from the dust component to compute
        the CIGALE-style disc/torus/polar normalization. Declaring these as
        optional inputs ensures the dust component runs before AGN in this
        configuration (#556). When ``agn_norm`` is ``"independent"`` or
        ``"conserving"``, AGN has no dust dependency and these are omitted.
        """
        if self.config.agn_norm == "cigale_joint":
            return (
                DerivedKey(
                    "log_L_ir",
                    "dex",
                    "log10(L_ir / (erg/s)); read by CIGALE-joint norm to scale torus",
                ),
                DerivedKey(
                    "L_absorbed",
                    "erg/s",
                    "Total absorbed luminosity; read by CIGALE-joint norm "
                    "(fallback if log_L_ir absent)",
                ),
            )
        else:
            return ()

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
        approx: Mapping[str, bool] | None = None,
        filters: tuple[tuple[jnp.ndarray, jnp.ndarray], ...] | None = None,
    ) -> AGNSEDComponentState:
        r"""Cache filter passbands and SKIRTOR templates for precomputation.

        When ``approx=WavePrecomp()`` is set:

        - Stores filter passbands so :meth:`apply` can publish
          ``agn_phot_lnu_precomp`` (filter-integrated AGN photometry).
        - If the AGN model is "skirtor", pre-loads the template grids
          so they thread through JIT as Parameter ops.

        Parameters
        ----------
        ssp_data : Any | None
            Unused; accepted for Protocol uniformity.
        wave_grid : ndarray | None
            Unused; accepted for Protocol uniformity.
        approx : Mapping[str, bool] | None
            Approximation flags. When ``approx.get('wave_precomp')``
            is ``True``, cache templates and filters.
        filters : tuple of (wave, trans) pairs | None
            Filter wavelengths and transmissions to cache.

        Returns
        -------
        AGNSEDComponentState
            State with optionally-populated filter_waves, filter_trans,
            and skirtor_templates.
        """
        del ssp_data
        approx = approx or {}
        torus_below_index = self._torus_below_index(wave_grid)

        filter_waves = None
        filter_trans = None
        skirtor_templates = None

        if approx.get("wave_precomp") and filters is not None:
            filter_waves = tuple(jnp.asarray(fw) for fw, _ in filters)
            filter_trans = tuple(jnp.asarray(ft) for _, ft in filters)

        # SKIRTOR torus models require template grids. Load them here so
        # they're available for JIT threading.
        if self.config.model == "skirtor":
            try:
                from tengri.components.agn.skirtor import _load_skirtor_default_grid

                # Store the template ARRAYS (a SKIRTORGrid pytree), not the
                # interpolation closure: arrays thread through jax.jit as a
                # runtime input; a closure cannot (#1198).
                skirtor_templates = _load_skirtor_default_grid()
            except FileNotFoundError:
                # No SKIRTOR grid on disk: continue without threading. The
                # apply() path lazy-loads and raises the clear not-found error
                # itself. Any other failure (a corrupt grid) propagates.
                pass

        # Composable-block template libraries. Driven by the block recipe, NOT
        # by ``config.model``: ``composable`` is the default in the build
        # grammar, so a gate on ``model == "skirtor"`` publishes nothing for
        # the recommended surface and every torus library bakes (#1383).
        block_templates = collect_block_templates(self.block_recipe()) or None

        return AGNSEDComponentState(
            name=self.name,
            filter_waves=filter_waves,
            filter_trans=filter_trans,
            skirtor_templates=skirtor_templates,
            block_templates=block_templates,
            torus_below_index=torus_below_index,
            torus_master_wave=wave_grid if torus_below_index is not None else None,
        )

    def _torus_below_index(self, wave_grid):
        """The composable torus's build-time below-library absorption data, or None.

        ``wave_grid`` is the concrete master grid the chain was built on. The runner evaluates
        on that grid with the 6 and 12 um diagnostic nodes merged in ascending order, so the
        index set is taken on the same merged grid, in the grid's own dtype.
        """
        if wave_grid is None or self.config.model != "composable":
            return None
        grid = np.asarray(wave_grid)
        merged = np.sort(np.concatenate([grid, np.asarray(_MIR_DIAGNOSTIC_AA, dtype=grid.dtype)]))
        return below_library_index(merged, self.config.agn_torus_block)

    def block_recipe(self) -> dict[str, str]:
        """Map each pipeline stage to the block name this config selected.

        Returns
        -------
        dict
            ``{"disc": ..., "nlr": ..., "blr": ..., "feii": ..., "torus": ...,
            "attenuation": ...}``.
        """
        return {
            "disc": self.config.agn_disc_block,
            "nlr": self.config.agn_nlr_block,
            "blr": self.config.agn_blr_block,
            "feii": self.config.agn_feii_block,
            "torus": self.config.agn_torus_block,
            "attenuation": self.config.agn_attenuation_block,
        }

    def resolved_recipe(self) -> dict[str, str]:
        """The block selectors and normalization the AGN model actually evaluates with.

        For a deprecated preset (``agn={'type': 'skirtor'}``) the preset's selectors and norm
        override the config, as they do in :func:`resolve_agn_model`; :meth:`block_recipe`
        returns the config, which holds the grammar's ``"none"`` selectors for a preset. For a
        ``composable`` model the two agree.

        Returns
        -------
        dict of str to str
            The six stages of :meth:`block_recipe`, plus ``"norm"`` (``agn_norm``) and
            ``"polar_law"`` (``agn_polar_law``).
        """
        overrides = preset_overrides(self.config.model)
        resolved = {
            stage: overrides.get(f"agn_{stage}_block", block)
            for stage, block in self.block_recipe().items()
        }
        resolved["norm"] = overrides.get("agn_norm", self.config.agn_norm)
        resolved["polar_law"] = overrides.get("agn_polar_law", self.config.agn_polar_law)
        return resolved

    def _agn_call_kwargs(self, params, template_data, wave) -> dict:
        """The keyword arguments the AGN model function is called with, before the coupling.

        Parameters
        ----------
        params : mapping
            The component's parameter slice.
        template_data : dict or None
            JIT-threaded templates (``template_data["agn"]``).
        wave : ndarray, shape (n_wave,)
            The grid the model is evaluated on [Angstrom]; the build-time library index is
            passed only for the chain's own grid.

        Returns
        -------
        dict
            Numeric parameters at the user ``agn_torus_frac``, the composable selectors, and
            the threaded templates.
        """
        # The numeric parameters and the static block selectors (a non-composable model
        # absorbs the selectors via ``**kwargs``).
        agn_kwargs = self._composable_kwargs(params)
        agn_data = _agn_template_data(template_data)
        if agn_data.get("skirtor") is not None:
            agn_kwargs["_template"] = agn_data["skirtor"]
        # Per-block template libraries for the composable runner, which hands each stage its
        # own family's bundle; without it every template-backed block loads its grid at trace
        # time and bakes it in.
        if agn_data.get("blocks"):
            agn_kwargs["template_state"] = agn_data["blocks"]
        # The composable runner's build-time library index set, when the chain built one for
        # this grid; without it the runner takes the dense traced path.
        below_index = getattr(self._state, "torus_below_index", None)
        master_wave = getattr(self._state, "torus_master_wave", None)
        if self.config.model == "composable" and below_index is not None and wave is master_wave:
            agn_kwargs["torus_below_index"] = below_index
        return agn_kwargs

    def energy_ledger(self, state, params, template_data=None) -> dict:
        """The composable runner's energy ledger, evaluated on request.

        Re-evaluates the chain at the ledger's sightlines and grid (NAMING_CONTRACT 4c,
        "Composable runner"), at the torus fraction :meth:`apply` used, so it costs nothing
        in a forward pass that does not ask for it.

        Parameters
        ----------
        state : ForwardState
            The state :meth:`apply` produced (its ``agn_torus_frac_effective``).
        params : mapping
            The component's parameter slice.
        template_data : dict or None
            JIT-threaded templates, as given to :meth:`apply`.

        Returns
        -------
        dict
            ``{key: scalar}`` over :func:`ledger_property_specs`.
        """
        probe = jnp.asarray(_LEDGER_PROBE_AA, dtype=state.wave.dtype)
        agn_kwargs = {
            **self._agn_call_kwargs(params, template_data, probe),
            "agn_torus_frac": state.derived["agn_torus_frac_effective"],
        }
        static = tuple(sorted((k, v) for k, v in agn_kwargs.items() if isinstance(v, str)))
        traced = {k: v for k, v in agn_kwargs.items() if not isinstance(v, str)}
        return _ledger_evaluation(probe, jnp.asarray(params["agn_log_lbol"]), traced, static)

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
        ssp_data: Any | None = None,
        template_data: Any | None = None,
        ztable_data: Any | None = None,
    ) -> ForwardState:
        """Add AGN emission to ``state.sed_intrinsic`` and publish ``L_agn_bol``.

        Parameters
        ----------
        state : ForwardState
            Must carry rest-frame ``wave`` (Å). If ``sed_intrinsic`` is
            ``None`` it is initialized to zeros of the same shape.
        params : mapping
            Receives ``agn_*`` keys plus the bare ``redshift``.
        template_data : dict | None
            Nested dict with component namespaces ("nebular", "agn", etc)
            carrying template grids/weights for JIT threading. When present,
            SKIRTOR templates are read from ``template_data["agn"]["skirtor"]``
            as a JIT runtime input instead of module-level lazy-loaded cache.

        Returns
        -------
        ForwardState
            New state with ``sed_intrinsic`` updated and ``L_agn_bol``
            published to ``derived``.
        """
        wave = state.wave

        import jax as _jax

        from tengri.utils.scale import pow10 as _pow10

        agn_log_lbol = jnp.asarray(params["agn_log_lbol"])

        # Resolve the AGN model from the registry. resolve_agn_model is
        # a factory-time lookup but the returned callable is pure JAX
        # so it folds into a JIT trace cleanly. If template_data is
        # provided, extract AGN SKIRTOR template and thread it.
        agn_fn = resolve_agn_model(self.config.model)

        agn_kwargs = self._agn_call_kwargs(params, template_data, wave)
        block_templates = agn_kwargs.get("template_state")
        band_table = _agn_template_data(template_data).get("band_table")

        # CIGALE-faithful cross-component coupling
        # ────────────────────────────────────────────────────────────
        # When ``agn_ir_frac > 0`` we follow CIGALE skirtor2016's
        # bookkeeping: the AGN dust-IR power is derived from the
        # stellar dust-absorbed luminosity via
        # ``agn_power = L_absorbed × fracAGN / (1 − fracAGN)``
        # (skirtor2016.py:498). Below we OVERRIDE ``agn_torus_frac`` so
        # the existing block normalization ``l_scale = L_bol × frac``
        # produces ``l_scale = agn_power``. This way the torus block
        # API stays unchanged while the higher-level driver matches
        # CIGALE bit-for-bit. ``lambda_fracAGN="0/0"`` (CIGALE's whole-IR default) is the
        # default; ``agn={'ir_frac_window': ...}`` selects a wavelength window instead (#2662),
        # which changes only the scale factor of ``agn_power`` (see ``_ir_frac_window``).
        #
        # Every luminosity here (``L_absorbed`` ~1e43, ``L_agn_bol`` ~1e46) is
        # ``inf`` in pure float32, so the ratio ``agn_power / L_agn_bol`` is
        # formed in LOG space (from the float32-safe ``log_L_absorbed``), never
        # as inf/inf. ``agn_ir_frac == 0`` leaves the coupling inert (the user
        # torus_frac), with the log branch still finite so its unused-branch
        # gradient cannot leak (#1206).
        agn_ir_frac = jnp.asarray(params.get("agn_ir_frac", 0.0))
        # Avoid divide-by-zero / negative leak when fracAGN ≥ 1.
        _one_minus_frac = jnp.maximum(1.0 - agn_ir_frac, 1e-6)
        agn_torus_frac_user = jnp.asarray(params.get("agn_torus_frac", 0.5))
        # ``log_L_absorbed``, not ``log_L_ir``: this coupling wants the ABSORBED
        # stellar+nebular energy the dust screen actually intercepted, not the
        # re-emitted IR budget, which ``dust_eta_balance`` (and a declared
        # ``dust_log_L_ir`` override) can scale away from it (#1837/
        # #2187-series split). The ``log_L_ir`` fallback below covers a chain
        # built before the split published the new key.
        ir_frac_window = tuple(self.config.agn_ir_frac_window)
        _log_L_absorbed = state.derived.get("log_L_absorbed")
        if _log_L_absorbed is None:
            _log_L_absorbed = state.derived.get("log_L_ir")
        if _log_L_absorbed is None and "L_absorbed" in state.derived:
            _log_L_absorbed = jnp.log10(
                jnp.maximum(jnp.asarray(state.derived["L_absorbed"]), 1e-30)
            )
        if _log_L_absorbed is None:
            # No dust screen published an absorbed luminosity. Construction
            # refuses an active fracAGN in that case (sed_model._validate_
            # fracagn_requires_dust), so only the inert Fixed(0) coupling
            # reaches here, and the user torus fraction is used directly.
            agn_torus_frac_effective = agn_torus_frac_user
        elif ir_frac_window != (0.0, 0.0):
            # CIGALE lambda_fracAGN window (#2662): agn_power = f/(1-f) * H * D / W, with H and
            # W the host and AGN window luminosities and D the AGN dust luminosity. The
            # AGN shape is set by the cigale_joint tie, so D/W is scale-free: one provisional
            # evaluation at the user torus fraction gives it, and the final evaluation runs
            # at the window-derived fraction. Only the scale factor changes; agn_power keeps
            # its meaning and its place in the ledger.
            from tengri.config.exceptions import ParameterError

            if self.config.model != "composable":
                raise ParameterError(
                    f"agn['ir_frac_window']={ir_frac_window} needs the composable AGN "
                    f"(model {self.config.model!r} has no disc/torus/polar split to take the "
                    "dust luminosity from)."
                )
            _host = (
                state.sed_intrinsic if state.sed_intrinsic is not None else jnp.zeros_like(wave)
            )
            # Same grid and kwargs as the final evaluation (extended master grid, build-time
            # below-library index), so the provisional D/W carries the final pass's numerics.
            _wave_ext, _to_caller, _ = _with_diagnostic_nodes(wave)
            _prov_kwargs = {**agn_kwargs, "agn_torus_frac": PROVISIONAL_TORUS_FRAC}
            _L_ext, _, _, _ = agn_fn(
                _wave_ext,
                agn_log_lbol=agn_log_lbol,
                return_l2500=True,
                return_components=True,
                **_prov_kwargs,
            )
            _L_prov = jnp.take(_L_ext, _to_caller, axis=-1)
            _log_torus_frac = log10_torus_frac_from_window(
                wave,
                _host,
                _L_prov,
                agn_ir_frac,
                ir_frac_window[0] * 1.0e4,
                ir_frac_window[1] * 1.0e4,
            )
            agn_torus_frac_effective = jnp.where(
                agn_ir_frac > 0.0, _pow10(_log_torus_frac), agn_torus_frac_user
            )
        else:
            # log10(agn_power / L_agn_bol)
            #   = log10(L_absorbed) + log10(frac/(1−frac)) − agn_log_lbol − log10(L_sun)
            _log_torus_frac = (
                _log_L_absorbed
                + jnp.log10(jnp.maximum(agn_ir_frac, 1e-30))
                - jnp.log10(_one_minus_frac)
                - agn_log_lbol
                - _LOG10_L_SUN
            )
            agn_torus_frac_effective = jnp.where(
                agn_ir_frac > 0.0, _pow10(_log_torus_frac), agn_torus_frac_user
            )
        agn_kwargs = {**agn_kwargs, "agn_torus_frac": agn_torus_frac_effective}
        # Published byproduct only [erg/s]; ~1e46 and ``inf`` in float32.
        # stop_gradient so that inf cannot poison the reverse pass via 0*inf;
        # consumers needing it linearly run in float64.
        L_agn_bol = _jax.lax.stop_gradient(_pow10(agn_log_lbol + _LOG10_L_SUN))

        # Call AGN function. For composable models, request L_2500_intrinsic
        # and L_4400_intrinsic. For monolithic models, both default to 0.0
        # (X-ray falls back to L_bol BC or SKIRTOR's published L_2500_30deg;
        # radio falls back to L_bol bolometric correction).
        #
        # Float32 reference evaluation (#1206, #2321): when operating in pure
        # float32, evaluating every AGN block at the true ``agn_log_lbol``
        # overflows. Both paths evaluate at a reference L_bol in float32 and
        # rescale in log space to recover the true L_bol. The factoring lives
        # in components/agn/_lbol_reference.py and is called by the composable
        # runner and the monolithic branch here.
        if self.config.model == "composable":
            # The composable runner (compose_l_nu) performs the reference
            # factoring AND the ``agn_log_lbol_shape`` hand-off itself;
            # injecting the kwarg here as well would duplicate its hand-off.
            # The two diagnostic wavelengths ride along on the same single evaluation (a second
            # evaluation of the chain would embed every block's constants in the graph twice).
            # Every block is pointwise in wavelength, so the values at the caller's nodes are
            # those of a caller-only evaluation.
            wave_ext, to_caller, to_diag = _with_diagnostic_nodes(wave)
            L_agn_ext, L_2500_intrinsic, L_4400_intrinsic, comps_ext = agn_fn(
                wave_ext,
                agn_log_lbol=agn_log_lbol,
                return_l2500=True,
                return_components=True,
                **agn_kwargs,
            )
            n_ext = wave_ext.shape[0]
            L_agn = jnp.take(L_agn_ext, to_caller, axis=-1)
            agn_components = {
                key: (
                    jnp.take(val, to_caller, axis=-1)
                    if getattr(val, "ndim", 0) >= 1 and val.shape[-1] == n_ext
                    else val
                )
                for key, val in comps_ext.items()
            }
            mir_lnu = jnp.take(
                comps_ext["disc"] + comps_ext["torus"] + comps_ext["polar"], to_diag, axis=-1
            )
            mir_logs = _mir_logs(jnp.asarray(_MIR_DIAGNOSTIC_AA, dtype=wave.dtype), mir_lnu, 0.0)
            agn_lines_for_kernel = agn_components["lines"]
            # nu L_nu at 12 and 6 um of the AGN's own emission: disc + torus +
            # polar dust of the runner's rescaled components (erg/s/Hz ~1e30,
            # representable in float32); the nu multiply is added in log space.
        else:
            lbol_eval, use_ref, offset = reference_evaluation(agn_log_lbol, wave)
            if use_ref:
                # Disc shape can depend on L_bol (temperature), so evaluating a
                # monolithic model wholly at the reference L_bol would give the
                # WRONG shape. Hand the block its TRUE L_bol for shape
                # (``agn_log_lbol_shape``) while the output MAGNITUDE stays on
                # the reference so the L_lambda arithmetic stays in float32
                # range; shape-invariant blocks ignore the kwarg. (#1206)
                agn_kwargs = {**agn_kwargs, "agn_log_lbol_shape": jnp.asarray(agn_log_lbol)}
            if self.config.model in monolithic_models_with_line_components():
                # Line-only light for the instrument-only kernel (#2565); the
                # total ``L_agn`` is unchanged.
                L_agn_unit, line_components = agn_fn(
                    wave, agn_log_lbol=lbol_eval, return_components=True, **agn_kwargs
                )
                lines_unit = line_components["lines"]
            else:
                L_agn_unit = agn_fn(wave, agn_log_lbol=lbol_eval, **agn_kwargs)
                lines_unit = jnp.zeros_like(L_agn_unit)
            L_agn = rescale(L_agn_unit, offset) if use_ref else L_agn_unit
            agn_lines_monolithic = rescale(lines_unit, offset) if use_ref else lines_unit
            L_2500_intrinsic = jnp.asarray(0.0)
            L_4400_intrinsic = jnp.asarray(0.0)
            agn_components = None
            agn_lines_for_kernel = agn_lines_monolithic
            # A monolithic model has no sub-blocks: nu L_nu comes from the whole
            # SED, measured on the reference-scale spectrum with the true scale
            # added in log space.

        # The rank-1 band response, when the build made one: its evaluation at the reference
        # wavelengths also yields the 6 and 12 um diagnostics (see ``_term_band_photometry``).
        term_phot = None
        if (
            self.config.model == "composable"
            and self._state is not None
            and self._state.filter_waves is not None
            and self._state.filter_trans is not None
        ):
            band = term_band_response(template_data, "agn", params)
            if band is not None:
                term_phot = self._term_band_photometry(band, params, block_templates)

        # Point diagnostics: the emitted 6 and 12 um values, from a second evaluation of the
        # same chain at exactly those wavelengths. The chain's normalization does not depend
        # on the wave, so these are the emitted values, not an interpolation of the caller's grid.
        if self.config.model == "composable":
            if term_phot is not None:
                mir_logs = term_phot[1]
            log_L_6um, log_L_12um = mir_logs
        else:
            log_L_6um, log_L_12um = _mir_point_logs(
                self.config.model, agn_fn, agn_kwargs, agn_log_lbol, wave.dtype
            )

        # Filter-integrate L_agn through the cached filter
        # passbands and publish ``agn_phot_lnu_precomp`` so predict_via_precomp
        # can include the AGN contribution in the LUT sum.
        derived_overrides = dict(
            L_agn_bol=L_agn_bol,
            log_L_agn_bol=(agn_log_lbol + _LOG10_L_SUN),
            sed_agn=L_agn,
            L_2500_intrinsic=L_2500_intrinsic,
            L_4400_intrinsic=L_4400_intrinsic,
            log_L_12um=log_L_12um,
            log_L_6um=log_L_6um,
            # X-CIGALE tilts the corona with the AGN viewing angle
            # (yang20.py: cosi = cos(agn.i) for SKIRTOR, sin(psy) for
            # Fritz); publish cos(i) so the X-ray block shares this
            # sightline at every inclination, not only the 30° anchor
            # (#980). Every torus reads the one ``agn_cos_inc`` (#2605,
            # #3085); models without it publish the Yang+2020 anchor
            # (anisotropy factor exactly 1).
            agn_cos_inc=(
                jnp.asarray(params["agn_cos_inc"])
                if "agn_cos_inc" in params
                else jnp.asarray(_XRAY_COS_INC_REF_30DEG)
            ),
        )
        if agn_components is not None:
            # Public per-sub-block rest-frame SEDs (NAMING_CONTRACT §4b.5),
            # reachable via ``model.predict(params).sed.components[...]``.
            # Composable-runner-only: absent for monolithic AGN models,
            # which have no separate sub-blocks to decompose.
            derived_overrides["sed_agn_disc"] = agn_components["disc"]
            derived_overrides["sed_agn_torus"] = agn_components["torus"]
            derived_overrides["sed_agn_lines"] = agn_components["lines"]
            derived_overrides["sed_agn_polar"] = agn_components["polar"]
            derived_overrides["log_L_agn_direct_los"] = agn_components[
                DIRECT_LOS_LOG_LUMINOSITY_KEY
            ]
            # The torus fraction of this evaluation, for the energy ledger computed on request.
            derived_overrides["agn_torus_frac_effective"] = jnp.asarray(agn_torus_frac_effective)
        if self.config.model == "composable" and self.config.agn_disc_block == "adaf":
            # The ADAF's accretion side (Mahadevan 1997, Eq. 4): ``agn_log_lbol`` is the
            # radiated luminosity Q_ie + delta Q+, so the accretion power is not it.
            derived_overrides["agn_adaf_mdot"], derived_overrides["log_L_agn_adaf_accretion"] = (
                adaf_accretion_diagnostics(
                    agn_log_lbol,
                    agn_kwargs["agn_log_mbh"],
                    agn_kwargs.get("agn_adaf_alpha", _ADAF_DEFAULTS[0]),
                    agn_kwargs.get("agn_adaf_beta", _ADAF_DEFAULTS[1]),
                    agn_kwargs.get("agn_adaf_delta", _ADAF_DEFAULTS[2]),
                    dtype=wave.dtype,
                )
            )
        # Solar-luminosity siblings of the log10 powers (NAMING_CONTRACT 4c).
        derived_overrides.update(
            {
                k: v
                for k, v in with_lsun_siblings(
                    {
                        key: derived_overrides[key]
                        for key in (
                            "log_L_agn_bol",
                            "log_L_agn_direct_los",
                            "log_L_agn_adaf_accretion",
                        )
                        if key in derived_overrides
                    }
                ).items()
                if k.endswith("_lsun")
            }
        )
        # The line-only light exactly as it enters the pipeline SED (#2565): the
        # AGN's own screen is already applied (``agn_components["lines"]``, the
        # monolithic ``return_components`` lines), the host dust screen is not.
        # The two-component / single-screen dust adapters multiply this by the
        # transmission they apply to ``sed_agn`` when the AGN runs before them
        # (``agn_screen != "none"``); when it runs after them nothing else
        # touches it. The spectrum projection broadens this array with the
        # instrument kernel alone: AGN lines are painted at their own width and
        # never pass through the stellar library.
        derived_overrides["sed_agn_lines_attenuated"] = agn_lines_for_kernel
        if (
            self._state is not None
            and self._state.filter_waves is not None
            and self._state.filter_trans is not None
        ):
            from tengri.observation.photometry import lnu_filter_integral

            z = jnp.asarray(require_redshift(params, "components.agn.component.apply"))
            # Filter-integrate L_agn directly via ``lnu_filter_integral``
            # (ADR-0016, #398.e). Replaces the previous
            # ``compute_flux_density(..., dl_cm=1) × inv_cosmology`` dance
            # that applied and immediately undid the (1+z)/(4π d_L²)
            # dimming. The new helper returns the bare filter-integrated
            # rest-frame L_ν: matching the publish convention of
            # ``stellar_phot_lnu_precomp``.
            if band_table is not None:
                # The registry table holds this recipe's band fluxes over its free axes at the
                # model redshift and at z = 0. Reading it replaces the per-filter integrals; the
                # SED above feeds only the outputs that need it, so a photometry-only graph drops
                # the SED build as dead code.
                agn_phot_lnu_precomp, agn_restband_lnu_precomp = self._band_table_phot(
                    params, band_table
                )
            else:
                if term_phot is not None:
                    # Rank-1 fast path (#1118): every sub-block is a fixed shape times its
                    # amplitude, so the filter integral is sum_k A_k R_kf with R_kf built once.
                    agn_phot_lnu_precomp = term_phot[0]
                else:
                    agn_phot_lnu_precomp = jnp.asarray(
                        [
                            lnu_filter_integral(L_agn, state.wave, fw, ft, redshift=z)
                            for fw, ft in zip(
                                self._state.filter_waves,
                                self._state.filter_trans,
                                strict=False,
                            )
                        ]
                    )
                # The REST band (#1148). ``phot_rest_fnu`` is the SED reprojected at
                # z=0, so the filter sits in the REST frame and samples the rest SED at
                # its own pivot, the SAME integral with redshift=0, not the observed-band
                # value reused. Reusing it is what made the LUT report a different
                # physical quantity from the exact path (769 % in des_g at z=0.5).
                agn_restband_lnu_precomp = jnp.asarray(
                    [
                        lnu_filter_integral(L_agn, state.wave, fw, ft, redshift=0.0)
                        for fw, ft in zip(
                            self._state.filter_waves,
                            self._state.filter_trans,
                            strict=False,
                        )
                    ]
                )
            derived_overrides["agn_phot_lnu_precomp"] = agn_phot_lnu_precomp
            derived_overrides["agn_restband_lnu_precomp"] = agn_restband_lnu_precomp

        # Spectrum LUT family (SpectrumPrecomp): a spectrum pixel is a single
        # wavelength, so point-sampling the rest-frame AGN SED at the pixel
        # wavelengths is exact (mirrors radio/X-ray ``_emit(spec_eff)`` and the
        # dust-IR ``jnp.interp`` projection). Without this, ``predict_spectrum``
        # under ``approx=SpectrumPrecomp()`` silently dropped the AGN
        # contribution: ``predict_spectrum_via_precomp`` only sums the
        # ``*_spec_lnu_precomp`` families that components actually publish, and
        # AGN previously published only the photometry family.
        spec_eff = state.derived.get("spec_eff_waves")
        if spec_eff is not None:
            derived_overrides["agn_spec_lnu_precomp"] = jnp.interp(spec_eff, state.wave, L_agn)

        return state.add_intrinsic(L_agn).with_(
            derived=state.derived.with_(**derived_overrides),
        )

    @staticmethod
    def _band_table_phot(
        params: Mapping[str, Any], table: Mapping[str, Any]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Band fluxes from the composable registry table at the current parameters.

        Parameters
        ----------
        params : mapping
            Parameters of the model; the table's axes and ``agn_lum_ratio`` are read from it.
        table : mapping
            The adapter's table arrays (``template_data["agn"]["band_table"]``).

        Returns
        -------
        tuple of ndarray
            ``(observed-band, rest-band)`` L_nu [erg/s/Hz], each of shape ``(n_filters,)``.
        """
        from tengri.forward.precompute import registry

        module = registry.resolve("composable_agn")
        values = [jnp.asarray(params[name]) for name in sorted(table["axes"])]
        both = module.lookup_from_table(
            table, "composable_agn", jnp.asarray(params.get("agn_lum_ratio", 1.0)), *values
        )
        n_filters = both.shape[-1] // 2
        return both[:n_filters], both[n_filters:]


#: The wavelength the on-request ledger evaluation runs its (discarded) caller grid on [Angstrom];
#: the ledger integrates on its own grid.
_LEDGER_PROBE_AA = (5100.0,)


@partial(jax.jit, static_argnames=("static",))
def _ledger_evaluation(probe, agn_log_lbol, traced: dict, static: tuple) -> dict:
    """The runner's ledger, compiled once per recipe (``static``: the string selectors)."""
    _, comps = composable_agn_l_nu(
        probe,
        agn_log_lbol=agn_log_lbol,
        return_components=True,
        return_ledger=True,
        strict_keys=False,
        **dict(static),
        **traced,
    )
    return {key: comps[key] for key in ledger_property_specs()}


def _agn_template_data(template_data) -> dict:
    """``template_data["agn"]`` when present, else an empty dict."""
    if isinstance(template_data, dict) and isinstance(template_data.get("agn"), dict):
        return template_data["agn"]
    return {}


def _composable_ledger_bundle(state, params, model, template_data) -> dict:
    """The composable AGN's energy ledger for a property request (see ``Property.bundle``)."""
    from tengri.forward.orchestrator import slice_params_for_component

    agn = next(c for c in model._full_state_chain() if c.name == "agn")
    full = {**dict(model.spec.get_fixed_values()), **dict(params)}
    return agn.energy_ledger(state, slice_params_for_component(agn, full), template_data)


def _is_composable(component) -> bool:
    """Whether ``component`` is the composable AGN (the one with an energy ledger)."""
    return getattr(getattr(component, "config", None), "model", None) == "composable"


def _register_ledger_properties() -> None:
    """Register the composable energy ledger as on-request ``agn`` properties."""
    from tengri.forward.properties import Property, register_properties

    register_properties(
        "agn",
        {
            key: Property(
                units=units,
                group="agn",
                doc=doc,
                bundle=_composable_ledger_bundle,
                available=_is_composable,
            )
            for key, (units, doc) in ledger_property_specs().items()
        },
    )


_register_ledger_properties()


def numeric_agn_kwargs(params: Mapping[str, Any], agn_torus_frac) -> dict[str, jnp.ndarray]:
    """Numeric keyword arguments the registered AGN model is called with.

    Two passes: explicit defaults for the parameters the registered models most commonly read
    (so a missing key does not break multicolor / skirtor / kubota_done at trace time), then
    every other ``agn_*`` key in ``params`` forwarded so block-specific parameters (skirtor,
    grahsp, cat3d_wind, ...) reach the model function through its ``**kwargs`` tail.
    ``agn_log_lbol`` is passed separately and so excluded.

    Parameters
    ----------
    params : mapping
        Parameters of the model, ``agn_*`` keys among them.
    agn_torus_frac : float or ndarray
        The torus fraction the model is called with [dimensionless].

    Returns
    -------
    dict
        ``{name: ndarray}``, shared by the per-call path and the build-time band table so both
        call the model with the same defaults.
    """
    kwargs = {
        "agn_lum_ratio": jnp.asarray(params.get("agn_lum_ratio", 1.0)),
        "agn_alpha": jnp.asarray(params.get("agn_alpha", -1.0)),
        "agn_log_mbh": jnp.asarray(params.get("agn_log_mbh", 8.0)),
        "agn_log_ledd": jnp.asarray(params.get("agn_log_ledd", -1.0)),
        "agn_a_spin": jnp.asarray(params.get("agn_a_spin", 0.0)),
        "agn_torus_frac": jnp.asarray(agn_torus_frac),
        "agn_T_torus": jnp.asarray(params.get("agn_T_torus", 1000.0)),
        "agn_tau_torus": jnp.asarray(params.get("agn_tau_torus", 3.0)),
        "agn_T_hot": jnp.asarray(params.get("agn_T_hot", 1500.0)),
        "agn_T_warm": jnp.asarray(params.get("agn_T_warm", 300.0)),
        "agn_frac_hot": jnp.asarray(params.get("agn_frac_hot", 0.5)),
        "agn_cos_inc": jnp.asarray(params.get("agn_cos_inc", 0.5)),
        "agn_polar_ebv": jnp.asarray(params.get("agn_polar_ebv", 0.0)),
        "agn_polar_oa": jnp.asarray(params.get("agn_polar_oa", 40.0)),
        "agn_ebv_disc": jnp.asarray(params.get("agn_ebv_disc", 0.0)),
    }
    for key, val in params.items():
        if key.startswith("agn_") and key != "agn_log_lbol" and key not in kwargs:
            kwargs[key] = jnp.asarray(val)
    return kwargs


# Register in the unified component dispatch table so build_components resolves
# the AGN component via _resolve_registry_component (single dispatch, #846)
# instead of importing the class directly. AGN is a composite: its config
# selects the disc/torus/nlr/blr/feii/atten sub-blocks (ADR-0018), but its
# top-level dispatch is a single registered component, exactly like the other
# composite domain (nebular).
from tengri.components.sed_model_component import _REGISTRY


def _with_diagnostic_nodes(wave):
    """``wave`` with the diagnostic wavelengths added, ascending, and the index maps back.

    Returns
    -------
    wave_ext : ndarray, shape (n_wave + 2,)
        Ascending merge of the caller's grid and the 6 and 12 um diagnostic nodes [Angstrom].
    to_caller : ndarray of int, shape (n_wave,)
        Positions in ``wave_ext`` of the caller's nodes, in the caller's order.
    to_diag : ndarray of int, shape (2,)
        Positions in ``wave_ext`` of the diagnostic nodes.
    """
    n = wave.shape[0]
    stacked = jnp.concatenate([wave, jnp.asarray(_MIR_DIAGNOSTIC_AA, dtype=wave.dtype)])
    order = jnp.argsort(stacked)
    position = jnp.argsort(order)
    return stacked[order], position[:n], position[n:]


def _mir_logs(diag_wave, lnu, scale):
    """``log10(nu L_nu)`` at the 6 and 12 um diagnostic nodes of ``lnu`` on ``diag_wave``."""
    return (
        log10_nu_lnu_at(diag_wave, lnu, _MIR_DIAGNOSTIC_AA[0], scale),
        log10_nu_lnu_at(diag_wave, lnu, _MIR_DIAGNOSTIC_AA[1], scale),
    )


def _mir_point_logs(model, agn_fn, agn_kwargs, agn_log_lbol, dtype):
    """``log10(nu L_nu)`` of a monolithic AGN's emitted SED at 6 and 12 um.

    The model is evaluated exactly at the two diagnostic wavelengths, so each value is the
    emitted SED at that wavelength and does not depend on the caller's wavelength grid. (The
    composable runner gets the same values from its single evaluation.)

    Parameters
    ----------
    model : str
        AGN model name (monolithic).
    agn_fn : callable
        The registered AGN function.
    agn_kwargs : Mapping
        Keyword arguments the AGN function was called with on the caller's grid.
    agn_log_lbol : float or array
        Input AGN ``log10(L_bol / L_sun)``.
    dtype : dtype
        Floating dtype of the wavelength grid.

    Returns
    -------
    tuple of ndarray, scalar
        ``(log_L_6um, log_L_12um)`` [dex, log10(erg/s/Hz) + log10(Hz)].
    """
    diag_wave = jnp.asarray(_MIR_DIAGNOSTIC_AA, dtype=dtype)
    lbol_eval, use_ref, offset = reference_evaluation(agn_log_lbol, diag_wave)
    kwargs = dict(agn_kwargs)
    if use_ref:
        kwargs["agn_log_lbol_shape"] = jnp.asarray(agn_log_lbol)
    if model in monolithic_models_with_line_components():
        lnu = agn_fn(diag_wave, agn_log_lbol=lbol_eval, return_components=True, **kwargs)[0]
    else:
        lnu = agn_fn(diag_wave, agn_log_lbol=lbol_eval, **kwargs)
    return _mir_logs(diag_wave, lnu, offset)


_REGISTRY["agn"] = AGNSEDComponent
