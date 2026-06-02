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

import jax.numpy as jnp

from tengri.components.dust.attenuation import two_component_dust
from tengri.components.dust.emission import resolve_emission_model
from tengri.parameters.priors import Fixed, Uniform
from tengri.protocols.component import (
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
    emission_model : str or None
        IR emission template registry key. One of
        ``"modified_blackbody"``, ``"casey2012"``, ``"dale2014"``,
        ``"draine_li2007"``, ``"draine_li2014"``. Default
        ``"modified_blackbody"`` because it has no template-grid
        dependency. Pass ``None`` to disable IR re-emission entirely
        — the component then publishes ``sed_dust_ir`` as zeros and
        omits the energy-balance accumulation.
    t_birth_yr : float
        Birth-cloud dispersal age (sigmoid centre, yr).
        Default 1e7 (10 Myr) per Charlot & Fall (2000).
    transition_width_dex : float
        Sigmoid width (dex) for the BC→diffuse age transition.
    """

    name: str = "dust"
    law_bc: str = "power_law"
    law_diff: str = "power_law"
    emission_model: str | None = "modified_blackbody"
    t_birth_yr: float = 1e7
    transition_width_dex: float = 0.3


@dataclass(frozen=True)
class DustSEDComponentState(SEDComponentState):
    """State for dust component, optionally caching emission templates for JIT threading.

    When ``wave_precomp=True`` is set on the parent SEDModel, this holds
    pre-loaded dust IR emission templates as JAX arrays, so they thread
    through JIT as Parameter ops rather than baking into HLO Constants.
    """

    name: str = "dust"
    dust_emission_templates: Any | None = None


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

    def citations(self) -> tuple[str, ...]:
        """Structurally implements Charlot & Fall (2000) two-component dust;
        per-leaf attenuation laws are config-driven via
        :data:`tengri.citations.associations.DUST_LAW_CITATIONS`."""
        return ("charlot_fall2000",)

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Free parameters this component owns.

        Mirrors the canonical ``dust_*`` priors in
        :mod:`tengri.parameters._param_defs`. Users may override any
        entry as :class:`Fixed` to drop it from the prior.
        """
        return [
            ParamDeclaration(
                "dust_tau_bc",
                Uniform(0.0, 4.0),
                "Birth-cloud V-band optical depth [dimensionless]",
            ),
            ParamDeclaration(
                "dust_tau_diff",
                Uniform(0.0, 3.0),
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
                "dust_T",
                Fixed(35.0),
                "Cold-dust temperature for MBB / Casey 2012 [K]",
            ),
            ParamDeclaration(
                "dust_beta_ir",
                Fixed(1.6),
                "Cold-dust emissivity index for MBB / Casey 2012 [dimensionless]",
            ),
            ParamDeclaration(
                "dust_alpha_dale",
                Fixed(2.0),
                "Dale 2014 template-family alpha [dimensionless, in 0.0625-4.0]",
            ),
            ParamDeclaration(
                "dust_umin",
                Fixed(1.0),
                "DL07/DL14 minimum radiation field U_min [Habing]",
            ),
            ParamDeclaration(
                "dust_qpah",
                Fixed(2.5),
                "DL07/DL14 PAH mass fraction [%]",
            ),
            ParamDeclaration(
                "dust_gamma_dl",
                Fixed(0.01),
                "DL07/DL14 photon-dominated mass fraction [dimensionless]",
            ),
            ParamDeclaration(
                "dust_alpha_dl14",
                Fixed(2.0),
                "DL14 power-law slope [dimensionless, in 1.0-3.0]",
            ),
            ParamDeclaration(
                "dust_alpha_mir",
                Fixed(2.0),
                "Casey 2012 mid-IR power-law slope [dimensionless]",
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
        r"""Optionally pre-load dust IR emission templates for JIT threading.

        When ``approx=WavePrecomp()``, loads the dust IR emission
        template grids into a JAX pytree so they become JIT ``Parameter``
        ops rather than baked-in ``Constant`` ops.

        Parameters
        ----------
        ssp_data : Any | None
            Unused; accepted for Protocol uniformity.
        wave_grid : ndarray | None
            Unused; accepted for Protocol uniformity.
        approx : dict[str, bool] | None
            Approximation flags. When ``approx.get('wave_precomp')`` is
            ``True``, load templates.
        filters : tuple of (wave, trans) pairs | None
            Unused; accepted for Protocol uniformity.

        Returns
        -------
        DustSEDComponentState
            State with optionally-populated ``dust_emission_templates``.
        """
        del ssp_data, wave_grid, filters
        approx = approx or {}

        dust_templates = None
        if approx.get("wave_precomp") and self.config.emission_model is not None:
            # Load the emission template grids for JIT threading
            emission_models_with_templates = {
                "dale2014",
                "draine_li2007",
                "draine_li2014",
                "astrodust",
                "bosa",
            }
            if self.config.emission_model in emission_models_with_templates:
                from tengri.components.dust.emission import (
                    DUST_EMISSION_MODELS,
                    resolve_emission_model,
                )

                try:
                    # Trigger template loading by calling the emission function
                    # with dummy inputs. This populates DUST_EMISSION_MODELS
                    # with the actual template arrays.
                    resolve_emission_model(self.config.emission_model)
                    emission_fn = DUST_EMISSION_MODELS.get(self.config.emission_model)
                    if emission_fn is not None:
                        # Store a reference to the resolved function for apply()
                        dust_templates = emission_fn
                except Exception:
                    # If template loading fails (file not found), gracefully
                    # continue without threading.
                    pass

        return DustSEDComponentState(
            name=self.name,
            dust_emission_templates=dust_templates,
        )

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
        transmission = two_component_dust(
            wavelength=wave,
            age_grid=ssp_ages_yr,
            tau_v1=jnp.asarray(params["dust_tau_bc"]),
            tau_v2=jnp.asarray(params["dust_tau_diff"]),
            law_bc=self.config.law_bc,
            law_diff=self.config.law_diff,
            f_obscuration=jnp.asarray(params.get("dust_f_obscuration", 0.0)),
            t_birth=self.config.t_birth_yr,
            transition_width=self.config.transition_width_dex,
            n_slope=jnp.asarray(params.get("dust_slope", -0.7)),
            dust_bump_strength=jnp.asarray(params.get("dust_bump_strength", 0.0)),
            dust_delta=jnp.asarray(params.get("dust_delta", 0.0)),
            dust_Rv=jnp.asarray(params.get("dust_Rv", 3.1)),
        )  # (n_age, n_wave), in [0, 1]

        # ── 2. Apply transmission per age and aggregate ────────────────
        lnu_age_attenuated = lnu_age * transmission
        sed_attenuated = jnp.sum(lnu_age_attenuated, axis=0)
        sed_intrinsic_stellar = jnp.sum(lnu_age, axis=0)

        # ── 3. Energy balance: ∫ (L_nu_intrinsic - L_nu_attenuated) dν ──
        # ν = c/λ. trapezoid(integrand, x=ν) with ν descending returns a
        # negative signed area; abs() recovers the positive erg/s.
        # Mirrors forward/pipeline.py:815.
        #
        # Lyman-continuum exclusion: photons at λ < 912 Å are absorbed by
        # H ionisation (→ nebular emission), not by dust grains — they
        # don't contribute to the dust IR re-emission pool. Matches
        # CIGALE ``dustatt_modified_starburst`` (a_vs_ebv clips at 91.2
        # nm, so its energy-balance ∫ stops at 912 Å) without forcing
        # ``calzetti`` / ``leitherer02`` to zero the polynomial there
        # — users querying the curve at any wavelength still get a
        # value, only the dust energy-balance integral excludes those
        # photons.
        nu = C_AA / wave
        absorbed_lnu = sed_intrinsic_stellar - sed_attenuated
        # Mask LyC photons out of the L_absorbed integral.
        absorbed_lnu = jnp.where(wave >= 912.0, absorbed_lnu, 0.0)
        L_absorbed = jnp.abs(jnp.trapezoid(absorbed_lnu, nu))
        eta_balance = jnp.asarray(params.get("dust_eta_balance", 1.0))
        L_ir = jnp.maximum(L_absorbed * eta_balance, 0.0)

        # ── 4. IR emission template ────────────────────────────────────
        # When emission_model is None, the user opted out of IR re-emission
        # entirely (`dust_emission=None`). Skip the template call and
        # publish zero — preserves the no-emission behaviour.
        if self.config.emission_model is None:
            sed_ir = jnp.zeros_like(wave)
        else:
            emission_fn = resolve_emission_model(self.config.emission_model)
            z = jnp.asarray(params.get("redshift", 0.0))
            sed_ir = emission_fn(
                wave,
                L_ir,
                dust_T=jnp.asarray(params.get("dust_T", 35.0)),
                dust_beta_ir=jnp.asarray(params.get("dust_beta_ir", 1.6)),
                dust_alpha_dale=jnp.asarray(params.get("dust_alpha_dale", 2.0)),
                dust_umin=jnp.asarray(params.get("dust_umin", 1.0)),
                dust_qpah=jnp.asarray(params.get("dust_qpah", 2.5)),
                dust_gamma_dl=jnp.asarray(params.get("dust_gamma_dl", 0.01)),
                dust_alpha_dl14=jnp.asarray(params.get("dust_alpha_dl14", 2.0)),
                dust_alpha_mir=jnp.asarray(params.get("dust_alpha_mir", 2.0)),
                redshift=z,
            )

        # ── 5. Combine and publish ─────────────────────────────────────
        # Preserve any non-stellar contribution that may have been added
        # to ``sed_intrinsic`` by an upstream component (e.g. AGN,
        # Nebular). The stellar contribution at this point equals
        # ``sum(lnu_age, axis=0)`` exactly (Stellar is the only producer
        # of ``lnu_age``), so ``state.sed_intrinsic - sed_intrinsic_stellar``
        # isolates the non-stellar portion. Stellar dust does not attenuate
        # AGN/nebular/radio/xray.
        if state.sed_intrinsic is None:
            non_stellar_pre_dust = jnp.zeros_like(wave)
        else:
            non_stellar_pre_dust = state.sed_intrinsic - sed_intrinsic_stellar
        sed_total = non_stellar_pre_dust + sed_attenuated + sed_ir

        # Phase 3c-3c-iv-b: per-filter LUTs for two-component attenuation.
        # T(a, λ) factorises as T_diff(λ) × T_bc(λ)^y(a). For the filter-level
        # path we publish A_diff = exp(-τ_diff·k_diff(λ_eff)) and
        # A_bc = exp(-τ_bc·k_bc(λ_eff)) at each filter pivot, plus their
        # wavelength derivatives via central finite difference. The young
        # indicator ``y(a)`` is exposed for downstream consumers.
        derived_overrides = dict(
            L_ir=L_ir,
            L_absorbed=L_absorbed,
            sed_dust_attenuated=sed_attenuated,
            sed_dust_ir=sed_ir,
        )
        filter_eff = state.derived.get("filter_eff_waves")
        if filter_eff is not None:
            from tengri.components.dust.attenuation import resolve_dust_law

            tau_bc = jnp.asarray(params["dust_tau_bc"])
            tau_diff = jnp.asarray(params["dust_tau_diff"])
            n_slope = jnp.asarray(params.get("dust_slope", -0.7))
            law_bc_fn = resolve_dust_law(self.config.law_bc)
            law_diff_fn = resolve_dust_law(self.config.law_diff)
            d_lambda = jnp.asarray(1.0)
            # Evaluate k_bc and k_diff at the filter pivots and ±δλ for
            # the finite-difference slope.
            k_bc_at = law_bc_fn(filter_eff, n_slope=n_slope)
            k_diff_at = law_diff_fn(filter_eff, n_slope=n_slope)
            k_bc_plus = law_bc_fn(filter_eff + d_lambda, n_slope=n_slope)
            k_bc_minus = law_bc_fn(filter_eff - d_lambda, n_slope=n_slope)
            k_diff_plus = law_diff_fn(filter_eff + d_lambda, n_slope=n_slope)
            k_diff_minus = law_diff_fn(filter_eff - d_lambda, n_slope=n_slope)
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

            # IR re-emission on the photometry LUT (#622). The dust IR template
            # is re-emitted (not attenuated), so we publish a rest-frame Lν per
            # filter — ``predict_via_precomp`` sums all ``*_phot_lnu_precomp``
            # families and treats this one as the unattenuated bucket. ``L_ir``
            # is computed on the full SSP grid above (energy balance is exact;
            # only the template's *projection* uses the effective wavelength).
            # Without this the far-IR was ~100% wrong under WavePrecomp.
            if self.config.emission_model is not None:
                # IR re-emission is additive and unattenuated, so it is projected
                # through the *true* filter transmission (the same integral the
                # exact path uses) rather than sampled at the effective
                # wavelength — exact in bands carrying both the stellar continuum
                # and structured dust emission (MIR/PAH). The dense ``sed_ir`` is
                # built on the rest-frame ``wave`` grid above; ``predict_via_precomp``
                # applies cosmology to the summed L_ν. (Sampling the
                # self-normalising emission model at the sparse pivots was the
                # #622 regression that inflated the reddest band ~4×.)
                fw_pad = state.derived.get("phot_filter_waves_padded")
                ft_pad = state.derived.get("phot_filter_trans_padded")
                if fw_pad is not None:
                    from tengri.observation.photometry import lnu_filter_integral_batch

                    derived_overrides["dust_emission_phot_lnu_precomp"] = (
                        lnu_filter_integral_batch(
                            sed_ir, wave, fw_pad, ft_pad, jnp.asarray(params.get("redshift", 0.0))
                        )
                    )
                else:
                    # Fallback (padded curves not published): effective-wavelength
                    # sample of the dense, correctly normalised template.
                    derived_overrides["dust_emission_phot_lnu_precomp"] = jnp.interp(
                        filter_eff, wave, sed_ir
                    )

            # Young-star indicator on the SSP age grid: smooth sigmoid
            # transition around t_birth (matches two_component_dust).
            t_birth = self.config.t_birth_yr
            transition = self.config.transition_width_dex
            log_t = jnp.log10(jnp.maximum(ssp_ages_yr, 1.0))
            log_t_birth = jnp.log10(t_birth)
            # y(a) = 1 / (1 + 10^((log_t - log_t_birth) / transition))
            # → 1 for log_t << log_t_birth (young), 0 for log_t >> log_t_birth (old).
            y_age = 1.0 / (1.0 + 10.0 ** ((log_t - log_t_birth) / transition))
            derived_overrides["dust_young_indicator"] = y_age

        # Phase 5 (SpectrumPrecomp): per-pixel BC + diffuse transmission.
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

            # IR re-emission on the spectrum LUT (#622) — additive, unattenuated,
            # summed by ``predict_spectrum_via_precomp``. Usually negligible in
            # the optical but correct for spectra extending into the IR.
            if self.config.emission_model is not None:
                # Same fix as the filter branch (#622): sample the dense,
                # correctly normalised ``sed_ir`` at the spectral pivots rather
                # than re-evaluating the self-normalising emission model on the
                # sparse ``spec_eff`` grid (which renormalises L_ir over the
                # optical window and corrupts the result).
                derived_overrides["dust_emission_spec_lnu_precomp"] = jnp.interp(
                    spec_eff, wave, sed_ir
                )

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
