# SPDX-License-Identifier: BSD-3-Clause
"""XRaySEDComponent: SEDComponent adapter around :func:`xray_total`.

X-ray coronal and AGN emission. Implements the SEDComponent protocol
over the X-ray bands (0.1 keV to 100 keV). Reads AGN bolometric
luminosity, stellar mass, and SFR from upstream components to compute
accretion-driven and star-formation-driven X-ray fluxes.

Cross-component reads
---------------------
X-ray depends on quantities owned by other components:

- ``sfr`` (M_⊙/yr): produced by the stellar component as the
  current-time SFR. Read from ``state.derived["sfr"]`` with a
  fallback to 1.0.
- ``log_mstar`` (log10 M_⊙): produced by the stellar component.
  Read from ``state.derived["log_mstar"]`` with a fallback to 10.0
  (i.e. 10¹⁰ M_⊙). X-ray's ``xray_total`` consumes the linear stellar
  mass in M_⊙, so the adapter exponentiates: ``M_* = 10**log_mstar``.
- ``L_agn_bol`` (erg/s): produced by the AGN component. Read from
  ``state.derived["L_agn_bol"]`` with a fallback to 0.0 (no AGN).
- ``log_metallicity_history`` (dex, absolute log10(Z)): produced by the
  stellar component. Its present-day bin drives the Lehmer+2016 HMXB
  metallicity term; falls back to :data:`~tengri.utils.physics_constants.Z_SUN`
  when no stellar component is present (#1755).
- ``redshift``: bare parameter from :data:`BARE_NAME_ALLOWLIST`,
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
from typing import Any, ClassVar

import jax.numpy as jnp

from tengri.components._term_response import term_band_response as _term_band_response
from tengri.components.template_threading import TemplateThreading
from tengri.components.xray._params import PARAMS as _XRAY_PARAMS
from tengri.components.xray.xray import (
    COS_INC_REF_30DEG,
    metallicity_from_history,
    xray_total_lopez24_terms,
    xray_total_terms,
)
from tengri.parameters.resolve import require_redshift
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
    name: str
        Diagnostic identifier. Default ``"xray"``.
    model: str
        X-ray corona prescription. ``"yang20"``/``"simple"`` (default) ties the
        corona to the disc ``L_2500`` via the α_ox relation; ``"lopez24"`` ties
        it to the AGN 12 µm luminosity via the α_IRX relation (Lopez+2024).
    """

    name: str = "xray"
    model: str = "yang20"


@dataclass(frozen=True)
class XRaySEDComponentState(SEDComponentState):
    r"""X-ray has no precomputed tensors: typed marker only."""

    name: str = "xray"


@dataclass(frozen=True)
class XRaySEDComponent(TemplateThreading):
    r"""SEDComponent adapter around :func:`xray_total`.

    Notes
    -----
    **JIT-compatible**: yes, :meth:`apply` is pure JAX.
    **Additive**: writes ``sed_intrinsic = sed_intrinsic + L_xray(λ)``.
    Initializes ``sed_intrinsic`` from zeros if upstream did not.

    The physics covers two channels combined inside :func:`xray_total`:

    - X-ray binaries (HMXB + LMXB) scaling with SFR and M_*
      (Lehmer et al. 2010, 2016).
    - AGN corona via the alpha_ox–L_2500 relation (Lusso & Risaliti 2016).

    Both default to small (or zero) contributions when the cross-component
    reads fall back to defaults: i.e. a galaxy without an AGN gets
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
        ``tengri.components.xray._params``. The legacy ``_XRAY_PARAMS``
        bucket in ``tengri.parameters._builders`` is a derived view
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
        to instantiate stellar + AGN. Phase B of #21: see ADR-0004.
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
                "AGN cos(i) (composable models); corona anisotropy tilt: "
                "falls back to the Yang+2020 30-degree anchor (#980)",
            ),
            DerivedKey(
                "age_weights",
                "Msun",
                "SSP mass weights (stellar): mass-weighted age drives the LMXB scaling",
            ),
            DerivedKey(
                "ssp_ages_yr",
                "yr",
                "SSP age grid (stellar); with age_weights gives the LMXB stellar age",
            ),
            DerivedKey(
                "log_metallicity_history",
                "dex",
                "log10(Z) per SFH bin (stellar); its present-day bin drives the "
                "Lehmer+2016 HMXB metallicity term",
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
        state: ForwardState
            Must carry rest-frame ``wave`` (Å). If ``sed_intrinsic`` is
            ``None`` it is initialized to zeros of the same shape.
        params: mapping
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
        inputs = self.emitter_inputs(state.derived)

        def _emit(w):
            t = self.emission_terms(params, w, **inputs)
            return t["hmxb"] + t["lmxb"] + t["hotgas"] + t["agn"]

        L_xray = _emit(wave)
        derived_overrides = {"sed_xray": L_xray}

        # Precompute LUT families (#624): X-ray is additive and unattenuated.
        # Spectroscopy: a pixel is a point-sample, so evaluating at the pixel
        # wavelength is exact.
        from tengri.components._band_projection import project_additive_onto_photometry

        filter_eff = state.derived.get("filter_eff_waves")
        if filter_eff is not None:
            band = _term_band_response(template_data, "xray")
            fw_pad = state.derived.get("phot_filter_waves_padded")
            ft_pad = state.derived.get("phot_filter_trans_padded")

            # Compute precomputed photometry if band response is available.
            precomputed = None
            if band is not None:
                # Exact fast path. X-ray is a sum of rank-1 terms; HMXB, LMXB, hot
                # gas, corona: each a scalar amplitude times a spectral shape fixed
                # by the (fixed) shape parameters. Their *sum* is not rank-1: HMXB
                # (Gamma=2.0) and LMXB (Gamma=1.6) carry different photon indices, so
                # the mix shifts with SFR and stellar mass. Integrating each term
                # separately at build time is therefore exact where integrating the
                # total would not be. See SEDModel._additive_term_band_response.
                ref = self.emission_terms(params, band["lam_ref"], **inputs)
                amps = jnp.stack([t[i] for i, t in enumerate(ref.values())]) / band["S_ref"]
                precomputed = amps @ band["R"]

            # Resolved only for the dense branch, which is the only one that reads
            # it. ``require_redshift`` RAISES on a params dict without 'redshift',
            # so hoisting it above this condition would make a band-response model
            # fail where it used to work.
            z_xray = (
                jnp.asarray(require_redshift(params, "components.xray.component.apply"))
                if precomputed is None and fw_pad is not None
                else None
            )

            # Project onto observed-frame photometric filters.
            derived_overrides["xray_phot_lnu_precomp"] = project_additive_onto_photometry(
                precomputed,
                L_xray,
                wave,
                filter_eff,
                fw_pad,
                ft_pad,
                z_xray,
                fallback_fn=_emit,
            )

            # The REST band (#1148): ``phot_rest_fnu`` projects at z=0, so its
            # filter samples the pivot itself, not pivot/(1+z). Same emission,
            # different wavelengths: reusing the observed-band value here is
            # what made the LUT report a different quantity from the exact path.
            _rb_eff = state.derived.get("filter_restband_eff_waves")
            if _rb_eff is not None:
                derived_overrides["xray_restband_lnu_precomp"] = _emit(_rb_eff)
        spec_eff = state.derived.get("spec_eff_waves")
        if spec_eff is not None:
            derived_overrides["xray_spec_lnu_precomp"] = _emit(spec_eff)

        return state.add_intrinsic(L_xray).with_(
            derived=state.derived.with_(**derived_overrides),
        )

    #: Cross-component scalars this emitter reads off ``state.derived``, with two
    #: deliberately distant probe draws: see
    #: ``tengri.SEDModel._additive_term_band_response``. Every term must come back
    #: *proportional* between the two, or it is not rank-1 and earns no constant band
    #: response. Every value is nonzero on purpose, so a term that stays identically
    #: zero across both draws is zero for a structural reason, not a probe artifact.
    EMITTER_PROBE_INPUTS: ClassVar[tuple[dict[str, float], dict[str, float]]] = (
        {
            "sfr": 1.0,
            "stellar_mass": 1.0e10,
            "stellar_age_gyr": 1.0,
            # Not Z_SUN, deliberately: these are probe draws, not a solar
            # anchor, and writing the constant here would invite someone to
            # import it and couple the probe to a value it has no stake in.
            "metallicity_z": 0.0150,
            "l_2500": 1.0e29,
            "cos_inc": 0.9,
            "l_12um": 1.0e28,
        },
        {
            "sfr": 50.0,
            "stellar_mass": 3.0e11,
            "stellar_age_gyr": 8.0,
            # Deliberately not the other draw's Z: metallicity scales the HMXB
            # amplitude only (Gamma_HMXB is a shape parameter and is untouched),
            # so the term stays rank-1 across the pair while the probe actually
            # exercises the axis. Equal values would let a metallicity-blind
            # HMXB pass the proportionality check unnoticed.
            "metallicity_z": 0.0040,
            "l_2500": 7.0e31,
            "cos_inc": 0.3,
            "l_12um": 5.0e30,
        },
    )

    def emitter_inputs(self, derived: Mapping[str, Any]) -> dict[str, jnp.ndarray]:
        r"""Reduce ``state.derived`` to the scalars :meth:`emission_terms` needs.

        Owns the two fallback chains, so they exist in exactly one place.

        Parameters
        ----------
        derived: mapping
            ``state.derived``.

        Returns
        -------
        dict
            ``sfr`` [Msun/yr], ``stellar_mass`` [Msun], ``stellar_age_gyr`` [Gyr],
            ``metallicity_z`` [mass fraction], ``l_2500`` [erg/s/Hz],
            ``cos_inc`` [dimensionless], ``l_12um`` [erg/s/Hz].
        """
        sfr = jnp.asarray(derived.get("sfr", 1.0))
        # Contract: stellar publishes log_mstar (log10 M_⊙). xray_total
        # takes M_* in M_⊙; exponentiate at the boundary.
        log_mstar = jnp.asarray(derived.get("log_mstar", 10.0))
        stellar_mass = 10.0**log_mstar
        L_agn_bol = jnp.asarray(derived.get("L_agn_bol", 0.0))

        # LMXB scaling (Lehmer+2016) is a steep polynomial in the stellar-
        # population age, so it must see the galaxy's actual age: not the
        # 1 Gyr default. Compute the SSP mass-weighted age from the stellar
        # component's published age weights (matches CIGALE's
        # ``stellar.age_m_star``). Without this the LMXB, which dominates the
        # galaxy X-ray: over-predicts by ~3x for an evolved (~3 Gyr)
        # population. Falls back to 1 Gyr only if the weights are absent.
        age_weights = jnp.asarray(derived.get("age_weights", 0.0))
        ssp_ages_yr = jnp.asarray(derived.get("ssp_ages_yr", 0.0))
        _w_sum = jnp.sum(age_weights)
        stellar_age_gyr = jnp.where(
            _w_sum > 0.0,
            jnp.sum(age_weights * ssp_ages_yr) / jnp.maximum(_w_sum, 1e-30) / 1.0e9,
            1.0,
        )

        # The HMXB half of Lehmer+2016 is a quartic in Z, and steeper than the
        # LMXB age term above: an 18x spread across met_logzsol in [-1, +0.3].
        # HMXBs trace the instantaneous SFR, so the metallicity that matters is
        # the one the *young* population was born with, the present-day bin of
        # the history, index 0, the same reduction the nebular component uses.
        # Published in absolute log10(Z), which is what xray_xrb wants and what
        # makes this correct for every SSP library: BASTI calls 0.0200 solar
        # and MIST 0.0142, but both tabulate absolute Z, so nothing here has to
        # know which grid is loaded.
        #
        # Until #1755 this read the key "metallicity_z", which no component has
        # ever published, so the fallback was the only reachable value and the
        # fitted metallicity moved the HMXB term not at all. Same failure as the
        # det_hmxb/det_lmxb offsets in #1706, one argument over.
        metallicity_z = metallicity_from_history(derived.get("log_metallicity_history"))

        # Compute l_2500_30deg with fallback chain:
        # 1. L_2500_intrinsic from composable AGN (un-reddened disc shape)
        # 2. L_2500_30deg from SKIRTOR or other torus models
        # 3. L_bol -> L_2500 BC (Hopkins+2007) as last resort
        L_2500 = jnp.asarray(derived.get("L_2500_intrinsic", 0.0))
        L_2500_skirtor = jnp.asarray(derived.get("L_2500_30deg", 0.0))
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
        # Without this the corona was stuck face-on, a flat ×1.072: and
        # ``agn_cos_inc`` was a silent no-op for the X-ray block (#980).
        # No published AGN inclination → stay at the anchor (factor 1).
        cos_inc = jnp.asarray(derived.get("agn_cos_inc", COS_INC_REF_30DEG))
        # Lopez+2024 (lopez24) ties the corona to the AGN 12 µm luminosity
        # instead of L_2500. Prefer the AGN-published ``L_12um`` [erg/s/Hz]; fall
        # back to a bolometric correction from L_agn_bol when the composable AGN
        # does not publish a monochromatic 12 µm luminosity: νLν(12µm) = f_12·L_bol
        # (Gandhi+2009 f_12 ≈ 0.07), so Lν(12µm) = f_12·L_bol / ν_12µm.
        _nu_12um = 2.998e18 / 1.2e5  # 12 µm = 120000 Å
        _l_12um_pub = jnp.asarray(derived.get("L_12um", 0.0))
        _l_12um_bc = 0.07 * L_agn_bol / _nu_12um
        l_12um = jnp.where(_l_12um_pub > 0.0, _l_12um_pub, _l_12um_bc)

        return {
            "sfr": sfr,
            "stellar_mass": stellar_mass,
            "stellar_age_gyr": stellar_age_gyr,
            "metallicity_z": metallicity_z,
            "l_2500": l_2500,
            "cos_inc": cos_inc,
            "l_12um": l_12um,
        }

    def emission_terms(
        self,
        params: Mapping[str, jnp.ndarray],
        wave: jnp.ndarray,
        *,
        sfr: jnp.ndarray,
        stellar_mass: jnp.ndarray,
        stellar_age_gyr: jnp.ndarray,
        metallicity_z: jnp.ndarray,
        l_2500: jnp.ndarray,
        cos_inc: jnp.ndarray,
        l_12um: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        r"""The additive terms of the X-ray SED, unsummed.

        Single source of truth: :meth:`apply` sums these to build the SED, and the
        build-time band-response precompute integrates *these same terms* through the
        filters. Two paths, one definition: so they cannot drift.

        Parameters
        ----------
        params: mapping
            Full (un-sliced) parameter dict; reads the ``xray_*`` keys.
        wave: array_like, shape (n_wave,)
            Rest-frame wavelength grid [Angstrom].
        sfr: array_like, scalar
            Star-formation rate [Msun/yr]: sets the HMXB and hot-gas amplitudes.
        stellar_mass: array_like, scalar
            Stellar mass [Msun]: sets the LMXB amplitude.
        stellar_age_gyr: array_like, scalar
            Mass-weighted stellar age [Gyr]: Lehmer+2016 LMXB age term.
        metallicity_z: array_like, scalar
            Present-day metallicity [mass fraction, absolute Z]: Lehmer+2016
            HMXB metallicity term. []
        l_2500: array_like, scalar
            Disc L_nu at rest-frame 2500 A seen at 30 deg [erg/s/Hz]; drives the
            corona through the alpha_ox relation (``yang20``).
        cos_inc: array_like, scalar
            Cosine of the AGN inclination [dimensionless]; a Yang+2022 anisotropy
            factor, hence a pure *amplitude* term, not a spectral shape.
        l_12um: array_like, scalar
            AGN L_nu at 12 micron [erg/s/Hz]; drives the corona through alpha_IRX
            (``lopez24``).

        Returns
        -------
        dict of ndarray, each shape (n_wave,)
            ``{"hmxb", "lmxb", "hotgas", "agn"}`` [erg/s/Hz].

        Notes
        -----
        **JIT/grad/vmap-safe.** Pure ``jnp``; the model branch is a static config read.

        Each term is *rank-1* in wavelength, a scalar amplitude times a spectral shape
        set only by the shape parameters (photon indices, ``E_cut``, ``log_nh``). The
        amplitudes need not be *linear* in their inputs, and are not: ``alpha_ox``
        depends on ``log10(l_2500)``, so the corona amplitude is a power law in it. That
        is irrelevant: it is still a scalar, and a scalar is all the factorization needs.

        The *total* is not rank-1: HMXB (Gamma=2.0) and LMXB (Gamma=1.6) carry different
        photon indices, so the HMXB/LMXB mix, and hence the summed spectral shape :
        shifts with SFR and stellar mass. That is why the terms are exposed separately.
        """
        if self.config.model == "lopez24":
            # α_IRX corona (Lopez+2024): L_X(2-10 keV) = νLν(12µm) / 10^α_IRX,
            # shared Lehmer+2016 XRBs (age-aware LMXB, #854) + hot gas.
            return xray_total_lopez24_terms(
                wave,
                sfr=sfr,
                stellar_mass=stellar_mass,
                stellar_age_gyr=stellar_age_gyr,
                metallicity_z=metallicity_z,
                l_12um_erg_hz=l_12um,
                alpha_irx=jnp.asarray(params["xray_alpha_irx"]),
                gamma_hmxb=jnp.asarray(params["xray_gamma_hmxb"]),
                gamma_lmxb=jnp.asarray(params["xray_gamma_lmxb"]),
                gamma_agn=jnp.asarray(params["xray_gamma_agn"]),
                E_cut=jnp.asarray(params["xray_E_cut"]),
                log_nh=jnp.asarray(params["xray_log_nh"]),
                log_L_hmxb_offset=jnp.asarray(params["xray_det_hmxb"]),
                log_L_lmxb_offset=jnp.asarray(params["xray_det_lmxb"]),
            )
        # ``alpha_ox`` is derived from ``l_2500_30deg`` via the Just+2007
        # relation inside ``xray_total`` (#722, the disc 2500 A now drives
        # the X-ray corona). ``xray_delta_alpha_ox`` is now a live *offset* knob
        # (default 0.0 = pure empirical alpha_ox(L_2500); negative hardens
        # the corona, positive softens it). See ADR-0009 / xray_precompute.py
        # line 149 for the delta semantics.
        return xray_total_terms(
            wave,
            sfr=sfr,
            stellar_mass=stellar_mass,
            stellar_age_gyr=stellar_age_gyr,
            metallicity_z=metallicity_z,
            l_2500_30deg=l_2500,
            gamma_hmxb=jnp.asarray(params["xray_gamma_hmxb"]),
            gamma_lmxb=jnp.asarray(params["xray_gamma_lmxb"]),
            gamma_agn=jnp.asarray(params["xray_gamma_agn"]),
            E_cut=jnp.asarray(params["xray_E_cut"]),
            delta_alpha_ox=jnp.asarray(params["xray_delta_alpha_ox"]),
            cos_inc=cos_inc,
            log_nh=jnp.asarray(params["xray_log_nh"]),
            # Lehmer+2016 XRB luminosity offsets (#1706). ``xray_xrb_terms`` has
            # accepted these all along; the component simply never passed them,
            # so both were free parameters that nothing read.
            log_L_hmxb_offset=jnp.asarray(params["xray_det_hmxb"]),
            log_L_lmxb_offset=jnp.asarray(params["xray_det_lmxb"]),
        )


# ─────────────────────────────────────────────────────────────────────
# Xray group property registration (Phase 1B)
# ─────────────────────────────────────────────────────────────────────

_TINY = 1e-30  # Floor for safe division


def _l_x_xrb_fn(state, params):
    """X-ray luminosity from X-ray binaries [erg/s]."""
    from tengri.utils.sed_quantities import compute_l_x_xrb

    derived = state.derived
    sfr = jnp.asarray(derived.get("sfr_100myr", derived.get("sfr", 0.0)))
    log_mstar = jnp.asarray(derived.get("log_mstar", 0.0))
    mstar = jnp.power(10.0, log_mstar)
    return compute_l_x_xrb(sfr, mstar)


def _l_x_agn_fn(state, params):
    """X-ray luminosity from AGN [erg/s]."""
    from tengri.utils.sed_quantities import compute_l_x_agn

    derived = state.derived
    L_agn_bol = jnp.asarray(derived.get("L_agn_bol", 0.0))
    # Preserve 0 for inactive AGN, not NaN (matches legacy behavior)
    return jnp.where(L_agn_bol > 0.0, compute_l_x_agn(jnp.maximum(L_agn_bol, _TINY)), 0.0)


def _l_x_total_fn(state, params):
    """Total X-ray luminosity (XRB + AGN) [erg/s]."""
    from tengri.utils.sed_quantities import compute_l_x_agn, compute_l_x_xrb

    derived = state.derived
    sfr = jnp.asarray(derived.get("sfr_100myr", derived.get("sfr", 0.0)))
    log_mstar = jnp.asarray(derived.get("log_mstar", 0.0))
    mstar = jnp.power(10.0, log_mstar)
    l_x_xrb = compute_l_x_xrb(sfr, mstar)

    L_agn_bol = jnp.asarray(derived.get("L_agn_bol", 0.0))
    l_x_agn = jnp.where(L_agn_bol > 0.0, compute_l_x_agn(jnp.maximum(L_agn_bol, _TINY)), 0.0)
    return l_x_xrb + l_x_agn


def _log_l_x_xrb_fn(state, params):
    """log10 X-ray luminosity from X-ray binaries [dex re erg/s]."""
    from tengri.utils.sed_quantities import compute_log_l_x_xrb

    derived = state.derived
    sfr = jnp.asarray(derived.get("sfr_100myr", derived.get("sfr", 0.0)))
    log_mstar = jnp.asarray(derived.get("log_mstar", 0.0))
    return compute_log_l_x_xrb(sfr, log_mstar)


def _log_l_x_agn_fn(state, params):
    """log10 X-ray luminosity from AGN [dex re erg/s]; -inf when the AGN is off."""
    from tengri.utils.scale import log10_magnitude
    from tengri.utils.sed_quantities import compute_log_l_x_agn

    derived = state.derived
    L_agn_bol = jnp.asarray(derived.get("L_agn_bol", 0.0))
    # -inf, not 0.0: in log space "no AGN" is an exactly-zero luminosity, which is
    # the -inf sentinel of log10_magnitude (#1527). Returning 0.0 here would claim
    # 1 erg/s. The linear sibling returns 0.0 for the same state, correctly.
    active = L_agn_bol > 0.0
    log_l_bol = log10_magnitude(jnp.where(active, L_agn_bol, 1.0))
    return jnp.where(active, compute_log_l_x_agn(log_l_bol), -jnp.inf)


def _log_l_x_total_fn(state, params):
    """log10 total X-ray luminosity (XRB + AGN) [dex re erg/s]."""
    from jax.scipy.special import logsumexp

    from tengri.utils.scale import LN10

    log_xrb = _log_l_x_xrb_fn(state, params)
    log_agn = _log_l_x_agn_fn(state, params)
    # The sum of two luminosities is a logsumexp of their logs, and an inactive AGN
    # at -inf drops out of it exactly, which is why the branch above returns -inf
    # rather than 0.0.
    stacked = jnp.stack(jnp.broadcast_arrays(log_xrb, log_agn))
    return logsumexp(LN10 * stacked, axis=0) / LN10


from tengri.forward.properties import Property, register_properties

_XRAY_PROPERTIES = {
    "l_x_xrb": Property(
        units="erg/s",
        group="xray",
        doc="X-ray luminosity from X-ray binaries",
        fn=_l_x_xrb_fn,
    ),
    "l_x_agn": Property(
        units="erg/s",
        group="xray",
        doc="X-ray luminosity from AGN",
        fn=_l_x_agn_fn,
    ),
    "l_x_total": Property(
        units="erg/s",
        group="xray",
        doc="Total X-ray luminosity (XRB + AGN)",
        fn=_l_x_total_fn,
    ),
    # Float32-safe companions (#1534). The HMXB coefficient alone is 2.6e39, past
    # float32's 3.4e38 ceiling, so the linear forms above are `inf` there at ANY
    # star formation rate: including zero. These are computed in log throughout,
    # not by taking a log of the linear value, which would inherit the overflow.
    "log_l_x_xrb": Property(
        units="dex",
        group="xray",
        doc="log10 X-ray luminosity from X-ray binaries [dex re erg/s]; "
        "float32-safe form of `l_x_xrb`",
        fn=_log_l_x_xrb_fn,
    ),
    "log_l_x_agn": Property(
        units="dex",
        group="xray",
        doc="log10 X-ray luminosity from AGN [dex re erg/s]; float32-safe form of "
        "`l_x_agn`. -inf when no AGN is present, where the linear form is 0.0",
        fn=_log_l_x_agn_fn,
    ),
    "log_l_x_total": Property(
        units="dex",
        group="xray",
        doc="log10 total X-ray luminosity (XRB + AGN) [dex re erg/s]; "
        "float32-safe form of `l_x_total`",
        fn=_log_l_x_total_fn,
    ),
}

register_properties("xray", _XRAY_PROPERTIES)

del Property, register_properties, _XRAY_PROPERTIES


# Register in the unified component dispatch table so build_components resolves
# the X-ray component via _resolve_registry_component (single dispatch, #845)
# instead of importing the class directly.
from tengri.components.sed_model_component import _REGISTRY

_REGISTRY["xray"] = XRaySEDComponent
