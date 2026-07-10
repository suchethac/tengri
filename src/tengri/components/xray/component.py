# SPDX-License-Identifier: BSD-3-Clause
"""XRaySEDComponent: SEDComponent adapter around :func:`xray_total`.

X-ray coronal and AGN emission. Implements the SEDComponent protocol
over the X-ray bands (0.1 keV to 100 keV). Reads AGN bolometric
luminosity, stellar mass, and SFR from upstream components to compute
accretion-driven and star-formation-driven X-ray fluxes.

Cross-component reads
---------------------
X-ray depends on quantities owned by other components:

- ``sfr`` (M_⊙/yr) — produced by the stellar component as the
  current-time SFR. Read from ``state.derived["sfr"]`` with a
  fallback to 1.0.
- ``log_mstar`` (log10 M_⊙) — produced by the stellar component.
  Read from ``state.derived["log_mstar"]`` with a fallback to 10.0
  (i.e. 10¹⁰ M_⊙). X-ray's ``xray_total`` consumes the linear stellar
  mass in M_⊙, so the adapter exponentiates: ``M_* = 10**log_mstar``.
- ``L_agn_bol`` (erg/s) — produced by the AGN component. Read from
  ``state.derived["L_agn_bol"]`` with a fallback to 0.0 (no AGN).
- ``redshift`` — bare parameter from :data:`BARE_NAME_ALLOWLIST`,
  passed through but consumed by the observation model rather than
  by ``xray_total`` itself.

The contract publishes ``log_mstar`` (not ``stellar_mass``) to match
:class:`tengri.components.radio.component.RadioSEDComponent` and the
:class:`StellarSEDComponent` planned in
:doc:`/dev/phase_ii_2_stellar_migration`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from tengri.components.xray._params import PARAMS as _XRAY_PARAMS
from tengri.components.xray.xray import COS_INC_REF_30DEG, xray_total, xray_total_lopez24
from tengri.protocols.component import (
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
    SEDComponentState,
)

__all__ = ["XRaySEDComponent", "XRaySEDComponentConfig"]


@dataclass(frozen=True)
class XRaySEDComponentConfig(SEDComponentConfig):
    r"""Frozen knobs for :class:`XRaySEDComponent`.

    Attributes
    ----------
    name : str
        Diagnostic identifier. Default ``"xray"``.
    model : str
        X-ray corona prescription. ``"yang20"``/``"simple"`` (default) ties the
        corona to the disc ``L_2500`` via the α_ox relation; ``"lopez24"`` ties
        it to the AGN 12 µm luminosity via the α_IRX relation (Lopez+2024).
    """

    name: str = "xray"
    model: str = "yang20"


@dataclass(frozen=True)
class XRaySEDComponentState(SEDComponentState):
    r"""X-ray has no precomputed tensors — typed marker only."""

    name: str = "xray"


@dataclass(frozen=True)
class XRaySEDComponent:
    r"""SEDComponent adapter around :func:`xray_total`.

    Notes
    -----
    **JIT-compatible**: yes — :meth:`apply` is pure JAX.
    **Additive**: writes ``sed_intrinsic = sed_intrinsic + L_xray(λ)``.
    Initializes ``sed_intrinsic`` from zeros if upstream did not.

    The physics covers two channels combined inside :func:`xray_total`:

    - X-ray binaries (HMXB + LMXB) scaling with SFR and M_*
      (Lehmer et al. 2010, 2016).
    - AGN corona via the alpha_ox–L_2500 relation (Lusso & Risaliti 2016).

    Both default to small (or zero) contributions when the cross-component
    reads fall back to defaults — i.e. a galaxy without an AGN gets
    XRB-only X-rays automatically.
    """

    config: XRaySEDComponentConfig = field(default_factory=XRaySEDComponentConfig)
    name: str = "xray"
    parameter_prefix: str = "xray_"

    def citations(self) -> tuple[str, ...]:
        """X-ray sub-blocks (XRB scaling, AGN corona) carry their citations
        via :mod:`tengri.citations.associations`; no always-required
        wrapper paper."""
        return ()

    def declared_parameters(self) -> list[ParamDeclaration]:
        r"""Free parameters this component owns.

        Returns the canonical :data:`PARAMS` tuple from
        :mod:`tengri.components.xray._params`. The legacy ``_XRAY_PARAMS``
        bucket in :mod:`tengri.parameters._param_defs` is a derived view
        of the same tuple, so the two registration paths agree by
        construction.
        """
        return list(_XRAY_PARAMS)

    def outputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component derived keys this X-ray component publishes.

        See :func:`tengri.forward.orchestrator.validate_pipeline`.

        """
        return (
            DerivedKey(
                "sed_xray",
                "erg/s/Hz",
                "X-ray luminosity contribution on pipeline wave grid",
            ),
        )

    def optional_inputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component derived keys X-ray reads *opportunistically*.

        Read from ``state.derived`` with documented fallbacks. The
        validator does NOT require an upstream publisher, but it WILL
        check that if one is present, its units match. Catches a future
        publisher rename or unit drift without forcing every pipeline
        to instantiate stellar + AGN. Phase B of #21 — see ADR-0004.
        """
        return (
            DerivedKey("sfr", "Msun/yr", "Read from stellar if present; falls back to 1.0"),
            DerivedKey(
                "log_mstar",
                "dex",
                "Read from stellar if present; falls back to 10.0",
            ),
            DerivedKey(
                "L_agn_bol",
                "erg/s",
                "Read from AGN if present; falls back to 0.0",
            ),
            DerivedKey(
                "L_2500_intrinsic",
                "erg/s/Hz",
                "Read from AGN if present; else fall back to L_2500_30deg, else L_bol->L_2500 BC",
            ),
            DerivedKey(
                "L_2500_30deg",
                "erg/s/Hz",
                "Fallback when L_2500_intrinsic unavailable (e.g. SKIRTOR monolithic path)",
            ),
            DerivedKey(
                "L_12um",
                "erg/s/Hz",
                "AGN 12 µm luminosity; drives the lopez24 alpha_IRX corona",
            ),
            DerivedKey(
                "agn_cos_inc",
                "dimensionless",
                "AGN cos(i) (composable models); corona anisotropy tilt — "
                "falls back to the Yang+2020 30-degree anchor (#980)",
            ),
            DerivedKey(
                "age_weights",
                "Msun",
                "SSP mass weights (stellar) — mass-weighted age drives the LMXB scaling",
            ),
            DerivedKey(
                "ssp_ages_yr",
                "yr",
                "SSP age grid (stellar); with age_weights gives the LMXB stellar age",
            ),
        )

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
        approx: Mapping[str, bool] | None = None,
        filters: tuple[tuple[jnp.ndarray, jnp.ndarray], ...] | None = None,
    ) -> XRaySEDComponentState:
        r"""No-op precompute. X-ray is a closed-form function of (λ, params)."""
        del ssp_data, wave_grid, filters
        return XRaySEDComponentState(name=self.name)

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
        ssp_data: Any | None = None,
        template_data: Any | None = None,
    ) -> ForwardState:
        r"""Add X-ray emission to ``state.sed_intrinsic``.

        Parameters
        ----------
        state : ForwardState
            Must carry rest-frame ``wave`` (Å). If ``sed_intrinsic`` is
            ``None`` it is initialized to zeros of the same shape.
        params : mapping
            Receives ``xray_*`` keys plus the bare ``redshift`` from
            the allowlist. Cross-component scalars (``sfr``,
            ``log_mstar``, ``L_agn_bol``) are read from
            ``state.derived`` with documented fallbacks.

        Returns
        -------
        ForwardState
            New state with ``sed_intrinsic`` updated and
            ``derived["sed_xray"]`` published for downstream readers.
        """
        wave = state.wave

        sfr = jnp.asarray(state.derived.get("sfr", 1.0))
        # Contract: stellar publishes log_mstar (log10 M_⊙). xray_total
        # takes M_* in M_⊙; exponentiate at the boundary.
        log_mstar = jnp.asarray(state.derived.get("log_mstar", 10.0))
        stellar_mass = 10.0**log_mstar
        L_agn_bol = jnp.asarray(state.derived.get("L_agn_bol", 0.0))

        # LMXB scaling (Lehmer+2016) is a steep polynomial in the stellar-
        # population age, so it must see the galaxy's actual age — not the
        # 1 Gyr default. Compute the SSP mass-weighted age from the stellar
        # component's published age weights (matches CIGALE's
        # ``stellar.age_m_star``). Without this the LMXB — which dominates the
        # galaxy X-ray — over-predicts by ~3x for an evolved (~3 Gyr)
        # population. Falls back to 1 Gyr only if the weights are absent.
        age_weights = jnp.asarray(state.derived.get("age_weights", 0.0))
        ssp_ages_yr = jnp.asarray(state.derived.get("ssp_ages_yr", 0.0))
        _w_sum = jnp.sum(age_weights)
        stellar_age_gyr = jnp.where(
            _w_sum > 0.0,
            jnp.sum(age_weights * ssp_ages_yr) / jnp.maximum(_w_sum, 1e-30) / 1.0e9,
            1.0,
        )

        # Compute l_2500_30deg with fallback chain:
        # 1. L_2500_intrinsic from composable AGN (un-reddened disc shape)
        # 2. L_2500_30deg from SKIRTOR or other torus models
        # 3. L_bol -> L_2500 BC (Hopkins+2007) as last resort
        L_2500 = jnp.asarray(state.derived.get("L_2500_intrinsic", 0.0))
        L_2500_skirtor = jnp.asarray(state.derived.get("L_2500_30deg", 0.0))
        # BC_2500 from Hopkins+2007: nu_2500 = 1.199e15 Hz, BC = 5.15
        L_2500_fallback = L_agn_bol / (5.15 * 1.199e15)
        l_2500 = jnp.where(
            L_2500 > 0.0,
            L_2500,
            jnp.where(L_2500_skirtor > 0.0, L_2500_skirtor, L_2500_fallback),
        )
        # The α_ox relations predict the corona seen at the Yang+2020 30°
        # reference; ``xray_anisotropy`` then tilts it to the model's own AGN
        # sightline, exactly as X-CIGALE reads cos i from its AGN module
        # (yang20.py). Components only see their prefix-matched param slice,
        # so the inclination arrives on the derived channel like L_2500.
        # Without this the corona was stuck face-on — a flat ×1.072 — and
        # ``agn_cos_inc`` was a silent no-op for the X-ray block (#980).
        # No published AGN inclination → stay at the anchor (factor 1).
        cos_inc = jnp.asarray(state.derived.get("agn_cos_inc", COS_INC_REF_30DEG))
        # Lopez+2024 (lopez24) ties the corona to the AGN 12 µm luminosity
        # instead of L_2500. Prefer the AGN-published ``L_12um`` [erg/s/Hz]; fall
        # back to a bolometric correction from L_agn_bol when the composable AGN
        # does not publish a monochromatic 12 µm luminosity: νLν(12µm) = f_12·L_bol
        # (Gandhi+2009 f_12 ≈ 0.07), so Lν(12µm) = f_12·L_bol / ν_12µm.
        _nu_12um = 2.998e18 / 1.2e5  # 12 µm = 120000 Å
        _l_12um_pub = jnp.asarray(state.derived.get("L_12um", 0.0))
        _l_12um_bc = 0.07 * L_agn_bol / _nu_12um
        l_12um = jnp.where(_l_12um_pub > 0.0, _l_12um_pub, _l_12um_bc)
        use_lopez24 = self.config.model == "lopez24"

        def _emit(w):
            if use_lopez24:
                # α_IRX corona (Lopez+2024): L_X(2-10 keV) = νLν(12µm) / 10^α_IRX,
                # shared Lehmer+2016 XRBs (age-aware LMXB, #854) + hot gas.
                return xray_total_lopez24(
                    w,
                    sfr=sfr,
                    stellar_mass=stellar_mass,
                    stellar_age_gyr=stellar_age_gyr,
                    l_12um_erg_hz=l_12um,
                    alpha_irx=jnp.asarray(params["xray_alpha_irx"]),
                    gamma_hmxb=jnp.asarray(params["xray_gamma_hmxb"]),
                    gamma_lmxb=jnp.asarray(params["xray_gamma_lmxb"]),
                    gamma_agn=jnp.asarray(params["xray_gamma_agn"]),
                    E_cut=jnp.asarray(params["xray_E_cut"]),
                    log_nh=jnp.asarray(params["xray_log_nh"]),
                )
            # ``alpha_ox`` is derived from ``l_2500_30deg`` via the Just+2007
            # relation inside ``xray_total`` (#722 — the disc 2500 A now drives
            # the X-ray corona). ``xray_delta_alpha_ox`` is now a live *offset* knob
            # (default 0.0 = pure empirical alpha_ox(L_2500); negative hardens
            # the corona, positive softens it). See ADR-0009 / xray_precompute.py
            # line 149 for the delta semantics.
            return xray_total(
                w,
                sfr=sfr,
                stellar_mass=stellar_mass,
                stellar_age_gyr=stellar_age_gyr,
                l_2500_30deg=l_2500,
                gamma_hmxb=jnp.asarray(params["xray_gamma_hmxb"]),
                gamma_lmxb=jnp.asarray(params["xray_gamma_lmxb"]),
                gamma_agn=jnp.asarray(params["xray_gamma_agn"]),
                E_cut=jnp.asarray(params["xray_E_cut"]),
                delta_alpha_ox=jnp.asarray(params["xray_delta_alpha_ox"]),
                cos_inc=cos_inc,
                log_nh=jnp.asarray(params["xray_log_nh"]),
            )

        L_xray = _emit(wave)
        derived_overrides = {"sed_xray": L_xray}

        # Precompute LUT families (#624): X-ray is additive and unattenuated.
        # Photometry: integrate the dense X-ray SED through the true filter
        # transmission (same integral as the exact path) — exact over the
        # bandpass, not sampled at one effective wavelength. Spectroscopy: a
        # pixel is a point-sample, so evaluating at the pixel wavelength is exact.
        filter_eff = state.derived.get("filter_eff_waves")
        if filter_eff is not None:
            fw_pad = state.derived.get("phot_filter_waves_padded")
            ft_pad = state.derived.get("phot_filter_trans_padded")
            if fw_pad is not None:
                from tengri.observation.photometry import lnu_filter_integral_batch

                derived_overrides["xray_phot_lnu_precomp"] = lnu_filter_integral_batch(
                    L_xray, wave, fw_pad, ft_pad, jnp.asarray(params.get("redshift", 0.0))
                )
            else:
                derived_overrides["xray_phot_lnu_precomp"] = _emit(filter_eff)
        spec_eff = state.derived.get("spec_eff_waves")
        if spec_eff is not None:
            derived_overrides["xray_spec_lnu_precomp"] = _emit(spec_eff)

        return state.add_intrinsic(L_xray).with_(
            derived=state.derived.with_(**derived_overrides),
        )


# Register in the unified component dispatch table so build_components resolves
# the X-ray component via _resolve_registry_component (single dispatch, #845)
# instead of importing the class directly.
from tengri.components.sed_model_component import _REGISTRY

_REGISTRY["xray"] = XRaySEDComponent
