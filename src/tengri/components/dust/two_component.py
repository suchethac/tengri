# SPDX-License-Identifier: BSD-3-Clause
"""DustSEDComponent: two-component attenuation + energy-balanced IR re-emission.

The Charlot & Fall (2000) two-component attenuation model with
energy-balanced IR re-emission. Reads the per-age stellar L_ν cube
published by :class:`tengri.components.stellar.StellarSEDComponent`,
applies an age-dependent extinction (birth cloud + diffuse ISM), and
produces a full attenuated + IR-re-emitted SED.

Sibling to :class:`tengri.components.dust.DustAttenuationSEDComponent`
(single-screen, no per-age stellar input required).

Cross-component reads
---------------------

- ``state.derived["lnu_age"]``  — (n_age, n_wave) per-age L_ν cube
  (erg/s/Hz) from :class:`StellarSEDComponent`.
- ``state.derived["ssp_ages_yr"]`` — (n_age,) age axis (yr) from
  :class:`StellarSEDComponent`.

Cross-component publications
----------------------------

- ``state.derived["L_ir"]`` (scalar, erg/s) — total IR luminosity
  re-radiated by dust, consumed by
  :class:`tengri.components.radio.RadioSEDComponent` (FIR-radio
  correlation) and :class:`tengri.components.xray.XRaySEDComponent`.
- ``state.sed_intrinsic`` is overwritten with the attenuated stellar
  SED + IR re-emission added.

Architectural notes
-------------------
``ssp_data`` is **not** held on this component — it reads what it
needs from ``state.derived``. ``precompute()`` returns an empty
marker, consistent with every adapter except the stellar one.
The IR-emission backend is resolved lazily inside ``apply()`` so
template-grid HDF5 loading happens on first invocation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp

from tengri.components.dust.attenuation import (
    resolve_bc_diff_law_params,
    two_component_dust,
)
from tengri.parameters.priors import Fixed, Uniform
from tengri.protocols.component import (
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
    SEDComponentState,
)
from tengri.utils.physics_constants import C_AA

__all__ = [
    "DustSEDComponent",
    "DustSEDComponentConfig",
    "DustSEDComponentState",
]


def _young_indicator(
    ssp_ages_yr: jnp.ndarray, t_birth_yr: float, transition_width_dex: float
) -> jnp.ndarray:
    r"""Fraction of stars of each age still inside their birth cloud.

    .. math::

        y(t) = \sigma\!\left(-\frac{\log_{10} t - \log_{10} t_{\rm birth}}
                                  {\Delta_{\rm trans}}\right)

    with :math:`\sigma` the logistic sigmoid. 1 for the youngest bins, 0 for the
    oldest.

    This is the single definition. It is the same function
    :func:`~tengri.components.dust._apply.two_component_dust` uses for the screen
    itself, so the stars that sit behind the birth cloud, the stars whose Lyman
    continuum is reprocessed, and the stars the photometry LUT reddens are all the
    same stars. Two other spellings — ``1 / (1 + 10**u)`` — existed here and in the
    LUT's ``dust_young_indicator``; because :math:`10^u = e^{u\ln 10}`, they were
    2.3x sharper than the screen they claimed to match (#1122).

    Parameters
    ----------
    ssp_ages_yr : ndarray, shape (n_age,)
        SSP lookback ages [yr].
    t_birth_yr : float
        Birth-cloud dispersal age — the sigmoid center [yr].
    transition_width_dex : float
        Sigmoid width [dex].

    Returns
    -------
    ndarray, shape (n_age,)
        Young-star indicator in [0, 1] [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes.
    """
    log_t = jnp.log10(jnp.maximum(jnp.asarray(ssp_ages_yr), 1.0))
    return jax.nn.sigmoid(-(log_t - jnp.log10(t_birth_yr)) / transition_width_dex)


@dataclass(frozen=True)
class DustSEDComponentConfig(SEDComponentConfig):
    """Frozen knobs for :class:`DustSEDComponent`.

    Parameters
    ----------
    name : str
        Diagnostic identifier. Default ``"dust"``.
    law_bc : str
        Attenuation-law registry key for the birth-cloud (young star)
        component. Default ``"power_law"``.
    law_diff : str
        Attenuation-law registry key for the diffuse ISM (old star)
        component. Default ``"power_law"``.
    law_neb : str or None
        Attenuation-law registry key for the **nebular** birth-cloud
        screen. ``None`` (default) inherits ``law_bc`` — the nebular
        continuum is then reddened by exactly the same young-limit screen
        as the youngest stars (Charlot & Fall 2000; bagpipes/FSPS/CIGALE
        behavior). Set it to give HII-region emission a *different*
        birth-cloud law from the stars while still sharing the diffuse ISM
        screen (``law_diff``). See ``neb_law_overrides`` for the matching
        per-parameter knob.
    t_birth_yr : float
        Birth-cloud dispersal age (sigmoid center, yr).
        Default 1e7 (10 Myr) per Charlot & Fall (2000).
    transition_width_dex : float
        Sigmoid width (dex) for the BC→diffuse age transition.
    """

    name: str = "dust"
    law_bc: str = "power_law"
    law_diff: str = "power_law"
    law_neb: str | None = None
    t_birth_yr: float = 1e7
    transition_width_dex: float = 0.3
    #: Per-component law-parameter overrides (birth cloud / diffuse ISM), as a
    #: hashable tuple of ``(law_kwarg, value)`` pairs so the frozen config stays
    #: usable as a static JIT key. Empty -> both components share the global
    #: ``dust_slope`` / ``dust_bump_strength`` / ``dust_delta`` / ``dust_Rv``.
    #: e.g. ``bc_law_overrides=(("n_slope", -1.0),)`` for the FSPS birth cloud.
    bc_law_overrides: tuple[tuple[str, float], ...] = ()
    diff_law_overrides: tuple[tuple[str, float], ...] = ()
    #: Per-parameter overrides for the **nebular** birth-cloud screen, same
    #: ``(law_kwarg, value)`` tuple form. Empty -> the nebular birth cloud
    #: inherits the *stellar* birth-cloud parameters (default = identical to the
    #: youngest stars). Layered on top of the resolved stellar ``bc`` params, so
    #: an override changes only the nebular screen. e.g.
    #: ``neb_law_overrides=(("dust_delta", 0.2),)``.
    neb_law_overrides: tuple[tuple[str, float], ...] = ()
    #: Zero the attenuation curve (stellar BC + diffuse, and the nebular screen)
    #: below this wavelength [Å]. ``0.0`` -> disabled: the ``calzetti`` /
    #: ``leitherer02`` polynomials extrapolate through the FUV. Set to ``912.0``
    #: (the H Lyman limit) to reproduce CIGALE's ``dustatt_modified_starburst``,
    #: which clips its curve there on the assumption that LyC photons ionize H
    #: rather than heat dust. Static, non-fittable; enters ``compile_signature``.
    lyman_cutoff_aa: float = 0.0
    #: Which stellar populations have their Lyman continuum (λ < 912 Å) absorbed
    #: by ``neb_fesc``. ``False`` (default) — **young/birth-cloud only**: only
    #: stars inside birth clouds (weighted by the young indicator) have their LyC
    #: reprocessed, so the old/diffuse stellar LyC passes through (matches
    #: bagpipes ``model_galaxy``, which zeros only ``spectrum_bc[<912]``, and is
    #: consistent with ``neb_fesc`` being a birth-cloud escape fraction).
    #: ``True`` — **all** stellar LyC absorbed (old + young), matching FSPS
    #: (``frac_obrun`` on the whole spectrum) and CIGALE (absorbed_old +
    #: absorbed_young). Static, non-fittable; enters ``compile_signature``.
    lyc_absorb_all: bool = False
    #: Include the Lyman continuum (λ < 912 Å) in the dust energy-balance
    #: integral. ``False`` (default) — canonical LyC-masked ``L_absorbed``
    #: (#922): LyC photons ionize H and re-emerge as nebular emission, not
    #: dust heating (CIGALE convention). ``True`` — all absorbed energy heats
    #: dust, matching FSPS/Prospector, whose ``add_dust_emission`` re-emits
    #: the full absorbed luminosity (measured ~10 % higher L_IR at the
    #: star-forming reproduction fiducial, #961). Grammar key
    #: ``dust={'eb_include_lyc': True}``. Static, non-fittable; enters
    #: ``compile_signature``.
    eb_include_lyc: bool = False


@dataclass(frozen=True)
class DustSEDComponentState(SEDComponentState):
    """State for the dust component — the component name, and nothing else.

    This class carried a ``dust_emission_templates`` field until 2026-08,
    documented as holding pre-loaded IR templates so they would thread through
    JIT as Parameter ops rather than bake into HLO Constants. Nothing ever set
    it: the only construction site passes ``name`` alone, and no reader existed
    anywhere in the package. The docstring described a mechanism that never ran,
    which is worse than no docstring — it answers "are the dust templates
    threaded?" with a confident yes.

    They are not. Threading dust IR templates is still open work, tracked
    alongside the AGN torus case (#1383, fixed for AGN by the declarative
    ``template_loader`` in #1595). Removing the inert field does not change
    behavior; it stops the field from standing in as evidence that the job
    is done.
    """

    name: str = "dust"


@dataclass(frozen=True)
class DustSEDComponent:
    """SEDComponent adapter for two-component dust + energy-balanced IR.

    Notes
    -----
    **JIT-compatible**: yes — :meth:`apply` is pure JAX once the
    attenuation-law registry lookup completes (these registries return
    plain JAX functions that fold cleanly into the JIT trace).
    **Pipeline ordering**: dust runs after stellar (and after nebular
    if a nebular component publishes its emission with BC dust applied
    in-place). Reads ``state.derived["lnu_age"]`` and
    ``["ssp_ages_yr"]``; overwrites ``state.sed_intrinsic`` with the
    attenuated stellar SED plus the IR re-emission.
    """

    config: DustSEDComponentConfig = field(default_factory=DustSEDComponentConfig)
    name: str = "dust"
    parameter_prefix: str = "dust_"
    #: When set (via ``approx=WavePrecomp(fast_dust_emission=True)``), project IR
    #: re-emission at the filter effective wavelength rather than integrating the
    #: dense template through each bandpass — much cheaper, slightly approximate.
    fast_emission: bool = False

    def citations(self) -> tuple[str, ...]:
        """Structurally implements Charlot & Fall (2000) two-component dust;
        per-leaf attenuation laws are config-driven via
        :data:`tengri.citations.associations.DUST_LAW_CITATIONS`."""
        return ("charlot_fall2000",)

    def outputs(self) -> tuple[DerivedKey, ...]:
        """Dust attenuation-derived quantities: absorbed luminosity and spectra.

        Publishes:

        - L_ir: total absorbed UV/optical/NIR luminosity (erg/s), enabling
          downstream dust emission components to re-radiate.
        - L_absorbed: alias for L_ir (deprecated, use L_ir).
        - sed_dust_attenuated: stellar SED after two-component attenuation.
        - sed_nebular: re-published after dust reddening (same name as nebular,
          not declared to avoid duplicate-publisher conflict; consumed by
          implementations that apply dust-reddened nebular).

        """
        return (
            DerivedKey("L_ir", "erg/s", "Total absorbed UV/optical/NIR luminosity"),
            DerivedKey("L_absorbed", "erg/s", "Alias for L_ir (deprecated)"),
            DerivedKey("sed_dust_attenuated", "erg/s/Hz", "Attenuated stellar SED"),
        )

    def optional_inputs(self) -> tuple[DerivedKey, ...]:
        """Nebular continuum read, if a photoionized backend published one.

        Declaring ``sed_nebular`` as an optional input makes the pipeline
        topological sort (ADR-0006) place the nebular component *before* dust,
        so dust can redden the nebular continuum with the same birth-cloud +
        diffuse screen as the youngest stars (Charlot & Fall 2000; matches the
        emission-line treatment). Backends that bake nebular into the SSP
        (BakedIn) publish ``sed_nebular`` as zeros, so this is a no-op there.

        Also reads ``lyc_transmission`` — the stellar Lyman-continuum survival
        fraction ``where(λ<912, neb_fesc, 1)`` published by a photoionized
        backend. Applied to the per-age stellar reconstruction so the fesc
        absorption is honored on the ``lnu_age`` path (see :meth:`apply` §2a
        and #824). Absent for BakedIn -> LyC passes through unchanged.
        """
        return (
            DerivedKey(
                "sed_nebular",
                "erg/s/Hz",
                "Nebular continuum to attenuate (Cue/CloudyGrid); zeros for BakedIn",
            ),
            DerivedKey(
                "lyc_transmission",
                "",
                "Stellar LyC survival fraction where(λ<912, neb_fesc, 1); absent for BakedIn",
            ),
        )

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Free parameters this component owns (attenuation-only).

        Mirrors the canonical ``dust_*`` attenuation priors in
        :mod:`tengri.parameters._param_defs`. Users may override any
        entry as :class:`Fixed` to drop it from the prior.

        Emission-specific parameters (dust_T, dust_beta_ir, dust_alpha_dale,
        dust_umin, etc.) are now owned by the dust emission components
        (modified_blackbody, dale2014, etc.) and are no longer declared here.
        """
        return [
            ParamDeclaration(
                "dust_tau_bc",
                # default 1.0: Charlot & Fall (2000) canonical birth-cloud tau_V
                # (Prospector dust1 ~ 1). Explicit default so '*': FIXED does not
                # silently fall back to the prior midpoint (#478 / surfaced by #844).
                Uniform(0.0, 4.0, default=1.0),
                "Birth-cloud V-band optical depth [dimensionless]",
            ),
            ParamDeclaration(
                "dust_tau_diff",
                # default 0.3: Charlot & Fall (2000) canonical diffuse-ISM tau_V
                # (Prospector dust2 ~ 0.3).
                Uniform(0.0, 3.0, default=0.3),
                "Diffuse ISM V-band optical depth [dimensionless]",
            ),
            ParamDeclaration(
                "dust_slope",
                Fixed(-0.7),
                "Power-law attenuation slope (Charlot-Fall convention) [dimensionless]",
            ),
            ParamDeclaration(
                "dust_eta_balance",
                Fixed(1.0),
                "Energy-balance relaxation factor (1.0 = strict balance) [dimensionless]",
            ),
            ParamDeclaration(
                "dust_f_obscuration",
                Fixed(0.0),
                "Clumpy-geometry obscuration fraction [dimensionless, in [0, 1]]",
            ),
            ParamDeclaration(
                "dust_bump_strength",
                Fixed(0.0),
                "2175 Å UV bump strength (Kriek-Conroy) [dimensionless]",
            ),
            ParamDeclaration(
                "dust_delta",
                Fixed(0.0),
                "Kriek-Conroy attenuation slope deviation [dimensionless]",
            ),
            ParamDeclaration(
                "dust_Rv",
                Fixed(3.1),
                "Cardelli total-to-selective extinction R_V [dimensionless]",
            ),
        ]

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
        approx: dict[str, bool] | None = None,
        filters: tuple[tuple[jnp.ndarray, jnp.ndarray], ...] | None = None,
    ) -> DustSEDComponentState:
        r"""Return an empty state (emission is now handled by separate components).

        This component is attenuation-only; IR emission is now handled by
        dedicated dust emission components in the pipeline. This method
        accepts arguments for Protocol uniformity but does not load any state.

        Parameters
        ----------
        ssp_data : Any | None
            Unused; accepted for Protocol uniformity.
        wave_grid : ndarray | None
            Unused; accepted for Protocol uniformity.
        approx : dict[str, bool] | None
            Unused; accepted for Protocol uniformity.
        filters : tuple of (wave, trans) pairs | None
            Unused; accepted for Protocol uniformity.

        Returns
        -------
        DustSEDComponentState
            Empty state marker.
        """
        del ssp_data, wave_grid, approx, filters

        return DustSEDComponentState(name=self.name)

    def compute_transmission(
        self,
        params: Mapping[str, jnp.ndarray],
        wavelength: jnp.ndarray,
        ssp_ages_yr: jnp.ndarray,
    ) -> jnp.ndarray:
        r"""Age-resolved two-component transmission :math:`T(\lambda, \mathrm{age})`.

        The single source of this component's dust screen. :meth:`apply` calls it
        on the full SSP wave grid; the FeaturePrecomp fast path
        (:meth:`SEDModel.predict_spectral_indices` with ``fast=True``) calls it at
        the index window centers — so the fast path applies **exactly** the dust
        the forward applies, with no second implementation to keep in sync.

        Resolves per-component (birth-cloud vs diffuse) law parameters — the
        shared ``dust_<x>`` params with any config overrides layered on top; empty
        overrides reproduce the original single-slope Charlot & Fall (2000)
        behavior exactly — then evaluates :func:`two_component_dust`.

        Parameters
        ----------
        params : mapping
            Receives ``dust_tau_bc`` / ``dust_tau_diff`` (+ optional
            ``dust_f_obscuration`` and per-law override keys).
        wavelength : ndarray, shape (n_wave,)
            Rest-frame wavelengths [Å] at which to evaluate the screen.
        ssp_ages_yr : ndarray, shape (n_age,)
            SSP lookback ages [yr] — the birth-cloud axis.

        Returns
        -------
        ndarray, shape (n_age, n_wave)
            Transmission in ``[0, 1]``: the youngest bins (age < ``t_birth``)
            carry birth-cloud + diffuse, older bins the diffuse screen only.

        Notes
        -----
        **JIT-compatible**: yes — pure ``jnp`` + registry law resolution.
        """
        bc_law_params, diff_law_params = resolve_bc_diff_law_params(
            params,
            dict(self.config.bc_law_overrides),
            dict(self.config.diff_law_overrides),
        )
        return self._transmission_from_law_params(
            params, wavelength, ssp_ages_yr, bc_law_params, diff_law_params
        )

    def _transmission_from_law_params(
        self, params, wavelength, ssp_ages_yr, bc_law_params, diff_law_params
    ) -> jnp.ndarray:
        """:func:`two_component_dust` evaluated with pre-resolved law params.

        The single call site of the two-component screen. :meth:`apply` resolves
        ``bc_law_params`` / ``diff_law_params`` once (it reuses them for the
        nebular-continuum block) and passes them here; :meth:`compute_transmission`
        resolves and delegates for the fast path.
        """
        return two_component_dust(
            wavelength=jnp.asarray(wavelength),
            age_grid=jnp.asarray(ssp_ages_yr),
            tau_v1=jnp.asarray(params["dust_tau_bc"]),
            tau_v2=jnp.asarray(params["dust_tau_diff"]),
            law_bc=self.config.law_bc,
            law_diff=self.config.law_diff,
            f_obscuration=jnp.asarray(params.get("dust_f_obscuration", 0.0)),
            t_birth=self.config.t_birth_yr,
            transition_width=self.config.transition_width_dex,
            bc_params={k: jnp.asarray(v) for k, v in bc_law_params.items()},
            diff_params={k: jnp.asarray(v) for k, v in diff_law_params.items()},
            lyman_cutoff_aa=self.config.lyman_cutoff_aa,
        )  # (n_age, n_wave), in [0, 1]

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
        ssp_data: Any | None = None,
        template_data: Any | None = None,
    ) -> ForwardState:
        """Apply two-component attenuation + IR re-emission.

        ``ssp_data`` is accepted for Protocol uniformity but unused — this
        component reads only from ``state`` and ``params``.

        Parameters
        ----------
        state : ForwardState
            Must carry ``wave`` and have stellar published
            ``lnu_age`` (n_age, n_wave) and ``ssp_ages_yr`` (n_age,)
            in its ``derived`` dict.
        params : mapping
            Receives ``dust_*`` keys plus the bare ``redshift``.

        Returns
        -------
        ForwardState
            New state with ``sed_intrinsic`` set to the attenuated
            stellar SED plus IR re-emission, and ``L_ir`` published
            to ``derived``.
        """
        if "lnu_age" not in state.derived or "ssp_ages_yr" not in state.derived:
            raise ValueError(
                "DustSEDComponent.apply requires upstream stellar publications "
                "(state.derived['lnu_age'] and ['ssp_ages_yr']). Place "
                "StellarSEDComponent before DustSEDComponent in the chain."
            )

        wave = state.wave
        lnu_age = jnp.asarray(state.derived["lnu_age"])  # (n_age, n_wave)
        ssp_ages_yr = jnp.asarray(state.derived["ssp_ages_yr"])

        # ── 1. Two-component transmission T(λ, age) ─────────────────────
        # Resolve per-component (birth-cloud vs diffuse) law parameters once —
        # reused below for the nebular-continuum screen. The screen itself is
        # single-sourced with the FeaturePrecomp fast path via
        # :meth:`_transmission_from_law_params` (see :meth:`compute_transmission`).
        bc_law_params, diff_law_params = resolve_bc_diff_law_params(
            params,
            dict(self.config.bc_law_overrides),
            dict(self.config.diff_law_overrides),
        )
        transmission = self._transmission_from_law_params(
            params, wave, ssp_ages_yr, bc_law_params, diff_law_params
        )  # (n_age, n_wave), in [0, 1]

        # ── 2. Apply transmission per age and aggregate ────────────────
        lnu_age_attenuated = lnu_age * transmission
        sed_attenuated = jnp.sum(lnu_age_attenuated, axis=0)
        sed_intrinsic_stellar = jnp.sum(lnu_age, axis=0)

        # ── 2a. Lyman-continuum escape (neb_fesc) ──────────────────────
        # Stellar LyC (λ < 912 Å) is absorbed by the gas that powers the nebular
        # emission; only the escaping fraction ``neb_fesc`` survives. A
        # photoionized backend publishes ``lyc_transmission = where(λ<912,
        # neb_fesc, 1)``. This path rebuilds the stellar SED from the *unmasked*
        # per-age ``lnu_age`` cube, so without applying the mask here the nebular
        # component's fesc mask on ``state.sed_intrinsic`` is bypassed — the LyC
        # leaks into ``sed_dust_attenuated`` and reappears as a phantom
        # ``-stellar_LyC`` in ``non_stellar_other`` (negative flux at fesc<1;
        # #824). Absent (BakedIn / no photoionized nebular) -> no factor.
        #
        # ``sed_intrinsic_stellar`` mirrors the nebular component's *uniform*
        # mask on ``state.sed_intrinsic`` (all ages × neb_fesc below 912) so the
        # ``non_stellar_other`` bookkeeping below stays clean; the below-912
        # region is excluded from the energy-balance integral, so this uniform
        # bookkeeping value never feeds L_ir.
        #
        # ``sed_attenuated`` (the actual stellar output) uses the physical rule:
        #   * default (``lyc_absorb_all=False``) — **young/birth-cloud only**.
        #     ``neb_fesc`` is a birth-cloud escape fraction, so only stars inside
        #     birth clouds (young indicator ``y(a)``) have their LyC reprocessed;
        #     the old/diffuse stellar LyC passes through. Matches bagpipes.
        #     Per-age factor ``1 - y(a)·(1 - lyc_t(λ))`` -> young→neb_fesc,
        #     old→1 below 912; both →1 above 912.
        #   * ``lyc_absorb_all=True`` — all stellar LyC absorbed (FSPS/CIGALE).
        _lyc_t = state.derived.get("lyc_transmission")
        if _lyc_t is not None:
            _lyc_t = jnp.asarray(_lyc_t)
            sed_intrinsic_stellar = sed_intrinsic_stellar * _lyc_t
            if self.config.lyc_absorb_all:
                sed_attenuated = sed_attenuated * _lyc_t
            else:
                # "Which stars are inside their birth cloud" is ONE physical
                # quantity, so it must be ONE function. It was previously spelled
                # three different ways: the logistic in ``two_component_dust``, and
                # base-10 sigmoids here and in the LUT's ``dust_young_indicator``.
                # 10^u = e^(u·ln10), so those two were 2.3× sharper than the screen
                # they were supposed to agree with — the LyC escape fraction was
                # applied to a different set of stars than the birth-cloud dust.
                y_age = _young_indicator(
                    ssp_ages_yr, self.config.t_birth_yr, self.config.transition_width_dex
                )
                lyc_factor = 1.0 - y_age[:, None] * (1.0 - _lyc_t[None, :])  # (n_age, n_wave)
                sed_attenuated = jnp.sum(lnu_age_attenuated * lyc_factor, axis=0)

        # ── 2b. Nebular continuum attenuation (birth-cloud + diffuse) ──────
        # Nebular emission from HII regions is reddened by the same dust as the
        # youngest stars (Charlot & Fall 2000). Photoionized backends (Cue,
        # CloudyGrid) publish a separate ``sed_nebular`` and, via this
        # component's ``optional_inputs``, run *before* dust; their continuum is
        # reddened here with the young-limit transmission (sigmoid weight -> 1,
        # i.e. both screens), matching the emission-LINE treatment
        # (``attenuate_emission`` mode "bc"). BakedIn nebular is already inside
        # ``lnu_age`` (attenuated above) and publishes zeros here -> no-op.
        from tengri.components.dust.attenuation import (
            apply_lyman_cutoff as _lyman_clip,
            resolve_dust_law as _resolve_law,
        )

        _sed_neb = state.derived.get("sed_nebular")
        sed_neb = jnp.zeros_like(wave) if _sed_neb is None else jnp.asarray(_sed_neb)
        # Nebular birth cloud: its own law/params, defaulting to the *stellar*
        # birth cloud (``law_bc`` + the resolved ``bc_law_params``) so the
        # default reddens the continuum exactly like the youngest stars. Setting
        # ``law_neb`` / ``neb_law_overrides`` decouples only the nebular
        # birth-cloud screen; the diffuse ISM screen (``law_diff`` +
        # ``diff_law_params``) is always shared with the stars — HII regions sit
        # in their own clouds behind the same foreground ISM.
        neb_law = self.config.law_neb or self.config.law_bc
        _neb_overrides = dict(self.config.neb_law_overrides)
        neb_bc_params = {
            k: jnp.asarray(_neb_overrides.get(k, v)) for k, v in bc_law_params.items()
        }
        diff_law_kw = {k: jnp.asarray(v) for k, v in diff_law_params.items()}
        k_bc_neb = _resolve_law(neb_law)(wave, **neb_bc_params)
        k_diff_neb = _resolve_law(self.config.law_diff)(wave, **diff_law_kw)
        # Same Lyman-limit clip as the stellar screen, so the nebular continuum
        # below 912 Å is treated consistently when the cutoff is enabled.
        k_bc_neb = _lyman_clip(k_bc_neb, wave, self.config.lyman_cutoff_aa)
        k_diff_neb = _lyman_clip(k_diff_neb, wave, self.config.lyman_cutoff_aa)
        tau_neb = (
            jnp.asarray(params["dust_tau_bc"]) * k_bc_neb
            + jnp.asarray(params["dust_tau_diff"]) * k_diff_neb
        )
        _f_obsc = jnp.asarray(params.get("dust_f_obscuration", 0.0))
        sed_neb_attenuated = sed_neb * (_f_obsc + (1.0 - _f_obsc) * jnp.exp(-tau_neb))

        # ── 3. Energy balance: ∫ (L_nu_intrinsic - L_nu_attenuated) dν ──
        # ν = c/λ. trapezoid(integrand, x=ν) with ν descending returns a
        # negative signed area; abs() recovers the positive erg/s.
        # Mirrors forward/pipeline.py:815.
        #
        # Lyman-continuum exclusion: photons at λ < 912 Å are absorbed by
        # H ionization (→ nebular emission), not by dust grains — they
        # don't contribute to the dust IR re-emission pool. Matches
        # CIGALE ``dustatt_modified_starburst`` (a_vs_ebv clips at 91.2
        # nm, so its energy-balance ∫ stops at 912 Å) without forcing
        # ``calzetti`` / ``leitherer02`` to zero the polynomial there
        # — users querying the curve at any wavelength still get a
        # value, only the dust energy-balance integral excludes those
        # photons.
        nu = C_AA / wave
        # Stellar + nebular absorbed light both feed the dust IR re-emission
        # pool (energy balance): the nebular continuum reddened in step 2b is
        # absorbed by the same grains.
        eb_lut = None
        if isinstance(template_data, dict):
            _dir = template_data.get("dust_ir")
            if isinstance(_dir, dict):
                eb_lut = _dir.get("energy_balance_lut")
        jw = state.derived.get("joint_weights")
        mass_scale = state.derived.get("stellar_mass_scale")
        # FSPS-parity toggle (#961): None disables the canonical LyC mask so
        # all absorbed energy heats dust. The fast-path LUT bakes the same
        # choice at build time (sed_model passes config.eb_include_lyc).
        _eb_cutoff = None if self.config.eb_include_lyc else 912.0
        if eb_lut is not None and jw is not None and mass_scale is not None:
            # Fast path (WavePrecomp): the stellar bolometric absorption comes
            # from a precomputed (tau_bc, tau_diff) LUT contracted with the
            # runtime DSPS weights — no full-wavelength stellar cube. The
            # nebular term is a single-SED integral (cheap), kept exact. Same
            # signed ∫(L_intrinsic - L_attenuated) dν, then abs(); when the
            # full cube is not otherwise needed XLA dead-code-eliminates it.
            from tengri.components.dust.energy_balance_precompute import lut_l_absorbed_stellar
            from tengri.forward.energy_balance import bolometric_absorbed

            stellar_absorbed = lut_l_absorbed_stellar(
                eb_lut,
                jnp.asarray(jw),
                jnp.asarray(mass_scale),
                jnp.asarray(params["dust_tau_bc"]),
                jnp.asarray(params["dust_tau_diff"]),
            )
            neb_absorbed = bolometric_absorbed(
                sed_neb, sed_neb_attenuated, nu, wave=wave, lyman_cutoff_aa=_eb_cutoff
            )
            L_absorbed = jnp.abs(stellar_absorbed + neb_absorbed)
        else:
            from tengri.forward.energy_balance import bolometric_absorbed

            L_absorbed = jnp.abs(
                bolometric_absorbed(
                    sed_intrinsic_stellar + sed_neb,
                    sed_attenuated + sed_neb_attenuated,
                    nu,
                    wave=wave,
                    lyman_cutoff_aa=_eb_cutoff,
                )
            )
        eta_balance = jnp.asarray(params.get("dust_eta_balance", 1.0))
        L_ir = jnp.maximum(L_absorbed * eta_balance, 0.0)

        # ── 4. Combine stellar + nebular SEDs (emission handled by components) ─
        # Preserve any non-stellar contribution that may have been added
        # to ``sed_intrinsic`` by an upstream component (e.g. AGN,
        # Nebular). The stellar contribution at this point equals
        # ``sum(lnu_age, axis=0)`` exactly (Stellar is the only producer
        # of ``lnu_age``), so ``state.sed_intrinsic - sed_intrinsic_stellar``
        # isolates the non-stellar portion. Stellar dust does not attenuate
        # AGN/nebular/radio/xray.
        #
        # IR re-emission is now handled by separate dust emission components
        # in the pipeline (modified_blackbody, dale2014, etc.), not by this
        # attenuation component. Those components read L_ir from state.derived
        # and produce sed_dust_ir, which is summed into the total SED via the
        # orchestrator.
        if state.sed_intrinsic is None:
            non_stellar_pre_dust = jnp.zeros_like(wave)
        else:
            non_stellar_pre_dust = state.sed_intrinsic - sed_intrinsic_stellar
        # The nebular continuum (sed_neb) is part of non_stellar_pre_dust but is
        # reddened by HII-region dust (step 2b). AGN/radio/xray stay unattenuated
        # by stellar dust. Swap the bare nebular for its attenuated form.
        non_stellar_other = non_stellar_pre_dust - sed_neb
        sed_total = non_stellar_other + sed_neb_attenuated + sed_attenuated

        # Per-filter LUTs for two-component attenuation.
        # T(a, λ) factorizes as T_diff(λ) × T_bc(λ)^y(a). For the filter-level
        # path we publish A_diff = exp(-τ_diff·k_diff(λ_eff)) and
        # A_bc = exp(-τ_bc·k_bc(λ_eff)) at each filter pivot, plus their
        # wavelength derivatives via central finite difference. The young
        # indicator ``y(a)`` is exposed for downstream consumers.
        derived_overrides = dict(
            L_ir=L_ir,
            L_absorbed=L_absorbed,
            sed_dust_attenuated=sed_attenuated,
            # Re-publish the nebular continuum as its dust-reddened (observed)
            # form, consistent with ``sed_dust_attenuated`` being the observed
            # stellar SED. Consumers that sum the published per-component SEDs
            # (e.g. the reproduction notebooks:
            # ``sed_dust_attenuated + sed_dust_ir + sed_nebular``) then recover
            # the true observed total. Unattenuated when dust is off / zero-τ.
            sed_nebular=sed_neb_attenuated,
        )
        filter_eff = state.derived.get("filter_eff_waves")
        if filter_eff is not None:
            from tengri.components.dust.attenuation import resolve_dust_law

            tau_bc = jnp.asarray(params["dust_tau_bc"])
            tau_diff = jnp.asarray(params["dust_tau_diff"])
            # Per-component slope (birth cloud may differ from diffuse). The
            # LUT path threads only the slope, matching its existing surface.
            n_slope_bc = jnp.asarray(bc_law_params["n_slope"])
            n_slope_diff = jnp.asarray(diff_law_params["n_slope"])
            law_bc_fn = resolve_dust_law(self.config.law_bc)
            law_diff_fn = resolve_dust_law(self.config.law_diff)
            d_lambda = jnp.asarray(1.0)
            # Evaluate k_bc and k_diff at the filter pivots and ±δλ for
            # the finite-difference slope.
            k_bc_at = law_bc_fn(filter_eff, n_slope=n_slope_bc)
            k_diff_at = law_diff_fn(filter_eff, n_slope=n_slope_diff)
            k_bc_plus = law_bc_fn(filter_eff + d_lambda, n_slope=n_slope_bc)
            k_bc_minus = law_bc_fn(filter_eff - d_lambda, n_slope=n_slope_bc)
            k_diff_plus = law_diff_fn(filter_eff + d_lambda, n_slope=n_slope_diff)
            k_diff_minus = law_diff_fn(filter_eff - d_lambda, n_slope=n_slope_diff)
            k_bc_slope = (k_bc_plus - k_bc_minus) / (2.0 * d_lambda)
            k_diff_slope = (k_diff_plus - k_diff_minus) / (2.0 * d_lambda)
            a_bc = jnp.exp(-tau_bc * k_bc_at)
            a_diff = jnp.exp(-tau_diff * k_diff_at)
            a_bc_slope = -tau_bc * k_bc_slope * a_bc
            a_diff_slope = -tau_diff * k_diff_slope * a_diff
            derived_overrides["dust_bc_attenuation_precomp"] = a_bc
            derived_overrides["dust_bc_attenuation_slope_precomp"] = a_bc_slope
            derived_overrides["dust_diff_attenuation_precomp"] = a_diff
            derived_overrides["dust_diff_attenuation_slope_precomp"] = a_diff_slope
            # Log-derivatives d(ln A)/dλ = −τ·k'(λ_eff), published directly (no
            # division by A) so the two-component Taylor projection (#617) is
            # NaN-safe where A → 0 (e.g. X-ray/UV bands far off the dust curve):
            # there T_a' = T_a·(logslope_diff + y·logslope_bc) with T_a = A_diff·A_bc^y → 0,
            # avoiding the A_bc^(y−1) pole.
            derived_overrides["dust_bc_log_attenuation_slope_precomp"] = -tau_bc * k_bc_slope
            derived_overrides["dust_diff_log_attenuation_slope_precomp"] = -tau_diff * k_diff_slope

            # Sub-band quadrature (#1122). The attenuation is EVALUATED at each
            # sub-band's quadrature node instead of being Taylor-extrapolated away
            # from λ_eff — the extrapolation is what diverges in the rest-UV, where
            # the curve steepens (GALEX FUV +45 % at z=0.05, +215 % at z=1).
            #
            # The law is evaluated live on the (n_age, n_filter, K) node grid, not
            # baked into a table, so ``n_slope`` (and any other shape parameter)
            # stays FREE. Measured cheaper than pre-baking it: the baked form has
            # to stream two extra tensors, while this one is compute-bound and XLA
            # fuses it into the contraction.
            sub_waves = state.derived.get("stellar_subband_waves_rest_precomp")
            if sub_waves is not None:
                a_bc_sub = jnp.exp(-tau_bc * law_bc_fn(sub_waves, n_slope=n_slope_bc))
                a_diff_sub = jnp.exp(-tau_diff * law_diff_fn(sub_waves, n_slope=n_slope_diff))
                derived_overrides["dust_bc_attenuation_subband_precomp"] = a_bc_sub
                derived_overrides["dust_diff_attenuation_subband_precomp"] = a_diff_sub

            # The same screen on the REST band (#1148). ``phot_rest_fnu`` projects at
            # z=0, so its filter samples rest λ_pivot, not rest λ_pivot/(1+z) — a
            # different set of wavelengths, and the galaxy's own dust must be
            # evaluated THERE. The law is analytic, so this is the same expression on
            # a different grid; τ, δ and the bump stay free.
            rb_eff = state.derived.get("filter_restband_eff_waves")
            if rb_eff is not None:
                derived_overrides["dust_bc_restband_attenuation_precomp"] = jnp.exp(
                    -tau_bc * law_bc_fn(rb_eff, n_slope=n_slope_bc)
                )
                derived_overrides["dust_diff_restband_attenuation_precomp"] = jnp.exp(
                    -tau_diff * law_diff_fn(rb_eff, n_slope=n_slope_diff)
                )
            rb_sub_waves = state.derived.get("stellar_restband_subband_waves_precomp")
            if rb_sub_waves is not None:
                derived_overrides["dust_bc_restband_attenuation_subband_precomp"] = jnp.exp(
                    -tau_bc * law_bc_fn(rb_sub_waves, n_slope=n_slope_bc)
                )
                derived_overrides["dust_diff_restband_attenuation_subband_precomp"] = jnp.exp(
                    -tau_diff * law_diff_fn(rb_sub_waves, n_slope=n_slope_diff)
                )

            # IR re-emission is now handled by separate dust emission components.
            # This component no longer computes or publishes photometric
            # dust emission — the emission components handle that via their own
            # precompute paths.

            # Young-star indicator on the SSP age grid: smooth sigmoid
            # transition around t_birth (matches two_component_dust).
            # The LUT must redden exactly the stars the exact screen reddens. This
            # line used to spell the indicator as ``1 / (1 + 10**u)`` while the
            # exact path used the logistic — 2.3x sharper, so the fast path put a
            # different set of stars behind the birth cloud (#1122). One function,
            # one definition.
            y_age = _young_indicator(
                ssp_ages_yr, self.config.t_birth_yr, self.config.transition_width_dex
            )
            derived_overrides["dust_young_indicator"] = y_age

        # SpectrumPrecomp: per-pixel BC + diffuse transmission.
        # T_bc(λ_pix), T_diff(λ_pix) are exact at the pixel — no Taylor slope.
        # The young indicator y(a) is reused to weight the per-age BC layer in
        # ``predict_spectrum_via_precomp``.
        spec_eff = state.derived.get("spec_eff_waves")
        if spec_eff is not None:
            from tengri.components.dust.attenuation import resolve_dust_law

            tau_bc = jnp.asarray(params["dust_tau_bc"])
            tau_diff = jnp.asarray(params["dust_tau_diff"])
            n_slope = jnp.asarray(params.get("dust_slope", -0.7))
            law_bc_fn = resolve_dust_law(self.config.law_bc)
            law_diff_fn = resolve_dust_law(self.config.law_diff)
            t_bc_pix = jnp.exp(-tau_bc * law_bc_fn(spec_eff, n_slope=n_slope))
            t_diff_pix = jnp.exp(-tau_diff * law_diff_fn(spec_eff, n_slope=n_slope))
            derived_overrides["dust_spec_bc_transmission_precomp"] = t_bc_pix
            derived_overrides["dust_spec_diff_transmission_precomp"] = t_diff_pix

            # IR re-emission is now handled by separate dust emission components.

            # Young-star indicator y(a) on the SSP age grid (same sigmoid as
            # the filter branch) — published even when only the spectrum LUT
            # is active.
            if "dust_young_indicator" not in derived_overrides:
                t_birth = self.config.t_birth_yr
                transition = self.config.transition_width_dex
                log_t = jnp.log10(jnp.maximum(ssp_ages_yr, 1.0))
                log_t_birth = jnp.log10(t_birth)
                y_age = 1.0 / (1.0 + 10.0 ** ((log_t - log_t_birth) / transition))
                derived_overrides["dust_young_indicator"] = y_age

        return state.with_(
            sed_intrinsic=sed_total,
            derived=state.derived.with_(**derived_overrides),
        )


# Register in the unified component dispatch table so the grammar type
# ``dust={'type': 'two_component'}`` resolves via _resolve_registry_component
# (the single dispatch seam), not a hardcoded class in build_components (#844).
from tengri.components.sed_model_component import _REGISTRY

_REGISTRY["two_component"] = DustSEDComponent
