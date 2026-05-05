# SPDX-License-Identifier: BSD-3-Clause
"""StellarSEDComponent: Phase II-2.2 implementation.

The body of the migration (merging :mod:`tengri.components.stellar.sfh` and
:mod:`tengri.components.stellar.sps` into a single ``SEDComponent``
adapter) is staged across Phase II-2.1 → II-2.6.

This module implements the **first slice (II-2.2)**:

- ``sfh_model="tsnorm"`` (truncated skew-normal SFH, no GP field)
- ``metallicity_model="delta"`` (single ``met_logzsol`` scalar)
- ``sps_backend="dsps"`` (DSPS native CSP integration)
- ``field=False``

The component publishes the **stable contract** of derived quantities
that downstream Phase II adapters (dust two-component, nebular Cue,
radio, X-ray) read — see ``state.derived`` keys below.

Architectural note: ``ssp_data`` is held on the component instance
itself (constructor field), parallel to how :class:`RadioSEDComponent`,
:class:`IGMSEDComponent`, and :class:`XRaySEDComponent` hold their
``config``. ``precompute()`` returns an empty marker, consistent with
those adapters; the SSP grid is treated as a fixed input baked in at
construction time, not an output of a separate precompute step.

See ``docs/dev/phase_ii_2_stellar_migration.md`` for the migration
plan and the design decisions resolved 2026-05-03.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from tengri.components.stellar.sfh.gp_sfh import make_log_age_grid
from tengri.components.stellar.sfh.mean_sfh import dpl, truncated_skewnormal
from tengri.components.stellar.sps.dsps_wrapper import (
    LSUN_ERG_PER_S,
    SSPData,
    compute_log_z_evolving,
    compute_surviving_mass,
    interpolate_mass_remaining,
)
from tengri.core.component import (
    ParamDeclaration,
    PipelineState,
    SEDComponentConfig,
    SEDComponentState,
)
from tengri.parameters.translate import LOG10_ZSUN
from tengri.utils.physics_constants import C_AA, H_PLANCK

__all__ = [
    "StellarSEDComponent",
    "StellarSEDComponentConfig",
    "StellarSEDComponentState",
]

# Lyman limit — wavelengths below this contribute to the ionising
# photon rate (matches :mod:`tengri.components.nebular.ionizing_spectrum`).
_HI_LIMIT_AA: float = 911.76


@dataclass(frozen=True)
class StellarSEDComponentConfig(SEDComponentConfig):
    """Frozen knobs for :class:`StellarSEDComponent`.

    Parameters
    ----------
    name : str
        Diagnostic identifier.
    sfh_model : str
        Registered SFH name. Phase II-2.2 supports ``"tsnorm"`` only;
        the remaining ``SFH_REGISTRY`` entries land in Phase II-2.3 / II-2.5.
    field : bool
        Stochastic GP field on top of the mean SFH. Phase II-2.2 supports
        ``False`` only; ``True`` lands with II-2.3.
    n_grid : int
        Lookback-time grid resolution for SFH evaluation and the published
        ``state.derived["sfh_grid_lbt_yr"]`` array.
    metallicity_model : str
        Phase II-2.2 supports ``"delta"`` only (single ``met_logzsol``
        parameter, time-constant Z(t)). Other modes land with II-2.4.
    sps_backend : str
        Phase II-2.2 supports ``"dsps"`` only (DSPS native triweight-MDF
        path). ``"dsps_native"`` is an alias for the same backend here.
    use_alpha_grid : bool
        Whether the SSP grid carries an α/Fe axis. Always ``False`` in
        II-2.2 — alpha grids land later.
    lgmet_scatter : float
        Gaussian scatter in log10(Z) (dex) for the DSPS triweight kernel.
        Default 0.2 dex matches Prospector / DSPS convention.
    """

    name: str = "stellar"
    sfh_model: str = "tsnorm"
    field: bool = False
    n_grid: int = 64
    metallicity_model: str = "delta"
    sps_backend: str = "dsps"
    use_alpha_grid: bool = False
    lgmet_scatter: float = 0.2


@dataclass(frozen=True)
class StellarSEDComponentState(SEDComponentState):
    """Marker state. SSP tensors live on the component instance, not here.

    Phase II-2.2 keeps :meth:`StellarSEDComponent.precompute` a no-op
    marker for consistency with :class:`RadioSEDComponent`,
    :class:`IGMSEDComponent`, and :class:`XRaySEDComponent`. The SSP
    grid is held by the component itself as a constructor field; this
    is the **most natural and consistent** plumbing given that
    :class:`tengri.forward.orchestrator.run_components` does not thread
    a precomputed-state argument to ``apply``.
    """

    name: str = "stellar"


@dataclass(frozen=True)
class StellarSEDComponent:
    """SEDComponent adapter for stellar emission.

    Notes
    -----
    **JIT-compatible**: yes — :meth:`apply` is pure JAX. The ``SSPData``
    NamedTuple registers as a JAX pytree, so ``self.ssp_data`` is a
    leaf-set of traced arrays under JIT.

    **Pipeline ordering**: stellar runs **first** in any chain. It
    writes ``state.sed_intrinsic`` from scratch and publishes the full
    set of stellar quantities other components consume.

    Construction
    ------------
    ``ssp_data`` is required at construction time. The component is a
    frozen dataclass; build it once at session start and reuse::

        ssp = load_ssp_data("data/ssp_miles.h5")
        stellar = StellarSEDComponent(ssp_data=ssp)
        result = run_components([stellar, ...], state, params)

    Cross-component publications (``state.derived``)
    ------------------------------------------------
    These keys are the stable contract every downstream adapter relies
    on. See :doc:`/dev/phase_ii_2_stellar_migration` for the full
    discussion.

    - ``log_mstar`` (scalar, dex) — log10(surviving stellar mass / Msun).
      Falls back to ``log_mstar_formed`` when the SSP grid lacks a
      ``ssp_mass_remaining`` table.
    - ``log_mstar_formed`` (scalar, dex) — log10(formed mass / Msun).
    - ``sfr`` (scalar, Msun/yr) — SFR at lookback ≈ 0 (i.e. the youngest
      grid point of ``sfr_history``).
    - ``sfr_10myr`` (scalar, Msun/yr) — time-weighted SFR over the last
      10 Myr of the SFH on the lookback grid.
    - ``sfr_100myr`` (scalar, Msun/yr) — same for 100 Myr.
    - ``L_age`` (ndarray, shape ``(n_age,)``, erg/s) — bolometric L per
      SSP age bin (∫ L_ν dν).
    - ``lnu_age`` (ndarray, shape ``(n_age, n_wave)``, erg/s/Hz) —
      per-age L_nu cube. Memory cost ~3 MB for n_age=140, n_wave=2700.
    - ``nion`` (scalar, photons/s) — ionising photon production rate
      (∫_{λ<911.76 Å} L_ν / (hν) dν, total over all ages).
    - ``sfh_grid_lbt_yr`` (ndarray, shape ``(n_grid,)``, yr) — SFH
      lookback-time grid (log-spaced, 1e5 yr → AGEMAX_YR).
    - ``sfr_history`` (ndarray, shape ``(n_grid,)``, Msun/yr) — SFR on
      the SFH grid.
    - ``log_metallicity_history`` (ndarray, shape ``(n_grid,)``, dex) —
      per-time-bin metallicity (constant for ``metallicity_model="delta"``).
    """

    config: StellarSEDComponentConfig = field(default_factory=StellarSEDComponentConfig)
    ssp_data: SSPData | None = None
    name: str = "stellar"
    parameter_prefix: tuple[str, ...] = ("sfh_", "met_", "chem_")

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Free parameters this component owns.

        Pulled from :data:`tengri.components.stellar.sfh.registry.SFH_REGISTRY`
        for the configured ``sfh_model`` plus a metallicity block keyed
        by ``metallicity_model``. Field parameters are added when
        ``config.field`` is ``True`` (Phase II-2.3+).
        """
        # Lazy-import the registries so this module remains importable even
        # if the SFH registry temporarily fails to build.
        from tengri.components.stellar.sfh.met_registry import MET_REGISTRY
        from tengri.components.stellar.sfh.registry import SFH_REGISTRY

        if self.config.sfh_model not in SFH_REGISTRY:
            raise ValueError(
                f"sfh_model={self.config.sfh_model!r} not in SFH_REGISTRY. "
                f"Available: {list(SFH_REGISTRY.keys())}"
            )
        if self.config.metallicity_model not in MET_REGISTRY:
            raise ValueError(
                f"metallicity_model={self.config.metallicity_model!r} not in MET_REGISTRY. "
                f"Available: {list(MET_REGISTRY.keys())}"
            )

        decls: list[ParamDeclaration] = []
        sfh_spec = SFH_REGISTRY[self.config.sfh_model]
        for pname, pdef in sfh_spec.params.items():
            decls.append(ParamDeclaration(pname, pdef.default, pdef.description))

        met_spec = MET_REGISTRY[self.config.metallicity_model]
        for pname, pdef in getattr(met_spec, "params", {}).items():
            decls.append(ParamDeclaration(pname, pdef.default, pdef.description))

        if self.config.field:
            field_spec = SFH_REGISTRY.get("field")
            if field_spec is not None:
                for pname, pdef in field_spec.params.items():
                    decls.append(ParamDeclaration(pname, pdef.default, pdef.description))

        return decls

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
    ) -> StellarSEDComponentState:
        """No-op marker. SSP grid is held on the component instance.

        Consistent with :class:`RadioSEDComponent`,
        :class:`IGMSEDComponent`, :class:`XRaySEDComponent`, which all
        return empty markers. The legacy ``forward/precompute/`` Protocol
        (filter-preintegrated photometry tables) is unrelated to this
        method and runs separately at ``SEDModel.__init__``.
        """
        del ssp_data, wave_grid
        return StellarSEDComponentState(name=self.name)

    def apply(
        self,
        state: PipelineState,
        params: Mapping[str, jnp.ndarray],
    ) -> PipelineState:
        """Compute stellar SED + publish derived quantities.

        Phase II-2.2 path: ``tsnorm`` SFH + ``delta`` metallicity + DSPS
        native triweight-MDF CSP integration. Other configurations
        raise :class:`NotImplementedError` until later sub-PRs land.

        Parameters
        ----------
        state : PipelineState
            Initial pipeline state. Carries ``wave`` (rest-frame Å); the
            component reads ``redshift`` from ``params`` (allowlist).
        params : mapping
            Receives ``sfh_*``, ``met_*``, ``chem_*`` keys plus the bare
            ``redshift`` from :data:`BARE_NAME_ALLOWLIST`.

        Returns
        -------
        PipelineState
            New state with ``sed_intrinsic`` set and 11 derived keys
            published.
        """
        if self.ssp_data is None:
            raise ValueError(
                "StellarSEDComponent.apply requires ssp_data set on the component. "
                "Pass it at construction: StellarSEDComponent(ssp_data=ssp)."
            )
        if self.config.sfh_model not in ("tsnorm", "dpl"):
            raise NotImplementedError(
                f"sfh_model={self.config.sfh_model!r} not yet implemented "
                f"(Phase II-2.2/II-2.5 supports 'tsnorm' and 'dpl'; "
                f"non-parametric forms 'continuity'/'dirichlet'/'dense_basis' "
                f"are deferred — they need vector-valued parameters with "
                f"separate registry plumbing)."
            )
        if self.config.metallicity_model not in ("delta", "ramp", "chem_evol"):
            raise NotImplementedError(
                f"metallicity_model={self.config.metallicity_model!r} not yet "
                f"implemented (Phase II-2.4 supports 'delta', 'ramp', 'chem_evol'; "
                f"'two_step', 'psb_two_step', 'bins', 'bins_continuity', 'table' "
                f"are deferred — they exist as math primitives in "
                f"components/stellar/sfh/metallicity_history.py but are not "
                f"wired into the legacy CSP forward pass either)."
            )
        # config.field is supported as of Phase II-2.3 (see step 2b below).

        ssp = self.ssp_data
        ssp_ages_yr = (10.0**ssp.ssp_lg_age_gyr) * 1e9
        n_grid = self.config.n_grid

        # ── 1. SFH lookback-time grid ───────────────────────────────────
        # Use the SAME grid construction as the legacy SEDModel path
        # (forward/sed_model.py:467). ``make_log_age_grid`` returns a
        # uniform grid in log10(age/yr) over [6.0, 10.14] (1 Myr →
        # 13.8 Gyr). This is critical for ``field=True`` parity:
        # ``compute_field_gp`` keys on n_grid + d_log_age to build
        # the GP correlation kernel, so both paths must construct
        # the grid identically or the same ``xi`` vector produces
        # different GP realisations. See the Phase II-2.3 finishing
        # commit message + tests/integration/test_orchestrator_vs_legacy.py.
        log_age_grid = make_log_age_grid(n_grid)
        sfh_lbt_grid = 10.0**log_age_grid

        # ── 2. Evaluate mean SFH on grid (parametric models) ────────────
        # Param names + Gyr→yr conversions match
        # ``components/stellar/sfh/registry.py``'s internal_param_map.
        if self.config.sfh_model == "tsnorm":
            sfr_history = truncated_skewnormal(
                sfh_lbt_grid,
                jnp.asarray(params["sfh_tsnorm_log_peak_sfr"]),
                jnp.asarray(params["sfh_tsnorm_peak_lbt_gyr"]) * 1e9,
                jnp.asarray(params["sfh_tsnorm_width_gyr"]) * 1e9,
                jnp.asarray(params["sfh_tsnorm_skew"]),
                jnp.asarray(params["sfh_tsnorm_trunc"]),
            )
        else:  # dpl
            sfr_history = dpl(
                sfh_lbt_grid,
                jnp.asarray(params["sfh_dpl_alpha"]),
                jnp.asarray(params["sfh_dpl_beta"]),
                jnp.asarray(params["sfh_dpl_tau_gyr"]) * 1e9,
                jnp.asarray(params["sfh_dpl_log_peak_sfr"]),
            )

        # ── 2b. GP-field modulation (Phase II-2.3) ──────────────────────
        # Multiplicative log-normal modulation: SFR_total = SFR_mean ×
        # exp(x(t) - K(0)/2), where x(t) is a PSD-governed Gaussian
        # process and K(0)/2 is the lognormal bias correction so the
        # ensemble mean equals SFR_mean. ``compute_field_gp`` lives in
        # the SFH registry next to the prior on ``sfh_field_xi``.
        if self.config.field:
            from tengri.components.stellar.sfh.registry import compute_field_gp

            psd_sigma = jnp.asarray(params["sfh_field_psd_sigma"])
            psd_tau_myr = jnp.asarray(params["sfh_field_psd_tau_myr"])
            xi = jnp.asarray(params.get("sfh_field_xi", jnp.zeros(n_grid)))
            psd_tau_yr = psd_tau_myr * 1e6
            d_log_age = float(log_age_grid[1] - log_age_grid[0])
            gp_x, k0_half = compute_field_gp(
                xi, psd_sigma, psd_tau_yr, n_grid, d_log_age, field_model="drw"
            )
            sfr_history = sfr_history * jnp.exp(gp_x - k0_half)

        # ── 3. Resample to SSP age grid for CSP integration ─────────────
        sfr_on_ssp = jnp.interp(ssp_ages_yr, sfh_lbt_grid, sfr_history)

        # ── 5. Cosmology: t_obs from redshift ───────────────────────────
        # ``age_at_z`` is JIT-compatible (pure JAX under the hood); keep
        # everything as JAX arrays so the whole apply() stays traceable.
        from tengri.utils.cosmology import age_at_z as _age_at_z

        z = jnp.asarray(params.get("redshift", 0.0))
        t_obs_gyr = jnp.asarray(_age_at_z(z)).reshape(())

        # ── 4. Metallicity history Z(t) on SFH grid + per-SSP-age ───────
        # delta: scalar absolute log10(Z), constant in time.
        # ramp: linear interpolation between two endpoints.
        # chem_evol: closed-box gas regulator — Z(t) derived from SFH self-
        # consistently (Phase II-2.4). Mirrors legacy sed_model.py:3578-3592.
        if self.config.metallicity_model == "delta":
            log_z_abs_scalar = jnp.asarray(params["met_logzsol"]) + LOG10_ZSUN
            log_metallicity_history = jnp.full(n_grid, log_z_abs_scalar)
            lgmet_on_ssp_ages = jnp.full_like(ssp_ages_yr, log_z_abs_scalar)
            log_z_for_mr = log_z_abs_scalar
        elif self.config.metallicity_model == "ramp":
            log_z_init_abs = jnp.asarray(params["met_logzsol_0"]) + LOG10_ZSUN
            log_z_final_abs = jnp.asarray(params["met_logzsol_final"]) + LOG10_ZSUN
            # Build the per-age metallicity ramp on both grids (SFH grid for
            # diagnostics + SSP grid for the CSP integral).
            sfh_lg_age_gyr = jnp.log10(jnp.maximum(sfh_lbt_grid, 1.0)) - 9.0
            log_metallicity_history = compute_log_z_evolving(
                sfh_lg_age_gyr, log_z_init_abs, log_z_final_abs, t_obs_gyr
            )
            lgmet_on_ssp_ages = compute_log_z_evolving(
                ssp.ssp_lg_age_gyr, log_z_init_abs, log_z_final_abs, t_obs_gyr
            )
            # For mass-remaining interpolation use the present-day metallicity
            # (newest stars dominate the mass-loss correction).
            log_z_for_mr = log_z_final_abs
        else:  # chem_evol
            from tengri.components.stellar.sfh.chemical_evolution import (
                chem_evol_metallicity_on_ssp_grid,
                closed_box_metallicity,
            )

            yield_y = float(params.get("chem_yield", 0.03))
            eta_outflow = float(params.get("chem_eta_outflow", 0.0))
            f_gas_init = float(params.get("chem_f_gas_init", 0.9))
            return_frac = float(params.get("chem_return_frac", 0.4))

            # Per-age metallicity on the SSP grid — mirrors legacy
            # sed_model.py:3583. Uses log10(age/yr) on both grids; the
            # SSP grid is ssp.ssp_lg_age_gyr + 9.0.
            ssp_log_ages_yr = ssp.ssp_lg_age_gyr + 9.0
            lgmet_on_ssp_ages = chem_evol_metallicity_on_ssp_grid(
                ssp_log_ages_yr,
                log_age_grid,
                sfr_history,
                yield_y=yield_y,
                eta_outflow=eta_outflow,
                f_gas_init=f_gas_init,
                return_frac=return_frac,
            )
            # Z(t) on the SFH grid for diagnostics — closed_box_metallicity
            # returns log10(Z/Zsun); add LOG10_ZSUN for absolute log10(Z).
            log_metallicity_history = (
                closed_box_metallicity(
                    sfh_lbt_grid,
                    sfr_history,
                    yield_y=yield_y,
                    eta_outflow=eta_outflow,
                    f_gas_init=f_gas_init,
                    return_frac=return_frac,
                )
                + LOG10_ZSUN
            )
            # Mass-remaining interpolation: use present-day Z (youngest SSP age).
            log_z_for_mr = lgmet_on_ssp_ages[0]

        # ── 6. CSP integral via DSPS ────────────────────────────────────
        # We call DSPS directly and use ``result.weights`` — the JOINT
        # (n_met, n_age) probability distribution — instead of the
        # separable approximation in compute_dsps_native_weights. The
        # separable form (lgmet_w × age_w) gave the right marginals but
        # the wrong product for non-trivial age-metallicity correlations,
        # over-scaling the CSP SED by orders of magnitude.
        from dsps.sed.stellar_sed import calc_rest_sed_sfh_table_lognormal_mdf

        ssp_age_gyr = ssp_ages_yr / 1e9
        t_cosmic_gyr = jnp.clip(t_obs_gyr - ssp_age_gyr, min=1e-3)
        t_cosmic_asc = t_cosmic_gyr[::-1]
        sfr_asc = sfr_on_ssp[::-1]
        total_mass = jnp.maximum(jnp.trapezoid(sfr_asc, t_cosmic_asc * 1e9), 0.0)

        if self.config.metallicity_model == "delta":
            dsps_result = calc_rest_sed_sfh_table_lognormal_mdf(
                gal_t_table=t_cosmic_asc,
                gal_sfr_table=sfr_asc,
                gal_lgmet=log_z_abs_scalar,
                gal_lgmet_scatter=self.config.lgmet_scatter,
                ssp_lgmet=ssp.ssp_lgmet,
                ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
                ssp_flux=ssp.ssp_flux,
                t_obs=t_obs_gyr,
            )
        else:  # ramp / chem_evol — per-age metallicity table
            from dsps.sed.stellar_sed import calc_rest_sed_sfh_table_met_table

            dsps_result = calc_rest_sed_sfh_table_met_table(
                gal_t_table=t_cosmic_asc,
                gal_sfr_table=sfr_asc,
                gal_lgmet_table=lgmet_on_ssp_ages[::-1],
                gal_lgmet_scatter=self.config.lgmet_scatter,
                ssp_lgmet=ssp.ssp_lgmet,
                ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
                ssp_flux=ssp.ssp_flux,
                t_obs=t_obs_gyr,
            )

        # ``dsps_result.weights`` is the joint (n_met, n_age) probability
        # distribution (sums to 1) over SSP grid points. The age axis is
        # already aligned with tengri's ssp_flux ordering (ascending
        # lookback age) — no flip needed; DSPS handles the cosmic-time
        # bookkeeping internally before storing weights against the
        # SSP grid.
        joint_weights = dsps_result.weights  # (n_met, n_age)
        # Per-age × per-Msun-formed weighted SSP flux (Lsun/Hz/Msun):
        ssp_flux_at_age = jnp.einsum("ma,maw->aw", joint_weights, ssp.ssp_flux)
        # Per-age "mass" for downstream per-age operations (dust BC mask).
        # This is the marginalised age distribution × total_mass.
        age_weights = joint_weights.sum(axis=0) * total_mass  # (n_age,) Msun

        # ── 7. Stellar SED in erg/s/Hz ──────────────────────────────────
        # ``rest_sed`` from DSPS is in Lsun/Hz (mass scaling included).
        # Sum the per-age cube — XLA folds the sum into the same kernel
        # as the einsum and avoids materialising ``rest_sed`` separately.
        lnu_age = total_mass * ssp_flux_at_age * LSUN_ERG_PER_S
        sed_intrinsic = jnp.sum(lnu_age, axis=0)

        # ── 8. Mass quantities ──────────────────────────────────────────
        log_mstar_formed = jnp.log10(jnp.maximum(jnp.sum(age_weights), 1e-30))
        if ssp.ssp_mass_remaining is not None:
            mr_at_met = interpolate_mass_remaining(
                ssp.ssp_mass_remaining, ssp.ssp_lgmet, log_z_for_mr
            )
            mstar_surv = compute_surviving_mass(age_weights, mr_at_met)
            log_mstar = jnp.log10(jnp.maximum(mstar_surv, 1e-30))
        else:
            log_mstar = log_mstar_formed

        # ── 9. SFR averages on the SFH grid ─────────────────────────────
        sfr_now = sfr_history[0]
        sfr_10myr = _time_weighted_sfr(sfr_history, sfh_lbt_grid, 1e7)
        sfr_100myr = _time_weighted_sfr(sfr_history, sfh_lbt_grid, 1e8)

        # ── 10. Bolometric L per SSP age bin ────────────────────────────
        # ν = c/λ ⟹ |dν| = c/λ² dλ. Trapezoid in wavelength with the
        # frequency Jacobian gives ∫ L_ν dν per age.
        wave = ssp.ssp_wave
        nu_jac = C_AA / (wave**2)
        L_age = jnp.trapezoid(lnu_age * nu_jac[None, :], wave, axis=1)

        # ── 11. Ionising photon production rate (λ < 911.76 Å) ──────────
        # photons/s = ∫_{ν > c/λ_HI} L_ν / (hν) dν, summed over all ages.
        # Mirrors components/nebular/ionizing_spectrum.py:299.
        nu = C_AA / wave
        ionizing_mask = wave < _HI_LIMIT_AA
        integrand = sed_intrinsic / (H_PLANCK * nu)
        integrand_masked = jnp.where(ionizing_mask, integrand, 0.0)
        # trapezoid(x=nu) with nu decreasing returns a negative value;
        # take abs to recover the positive photon rate.
        nion = jnp.abs(jnp.trapezoid(integrand_masked, nu))

        # ── 12. Assemble new state ──────────────────────────────────────
        new_derived = dict(state.derived)
        new_derived.update(
            {
                "log_mstar": log_mstar,
                "log_mstar_formed": log_mstar_formed,
                "sfr": sfr_now,
                "sfr_10myr": sfr_10myr,
                "sfr_100myr": sfr_100myr,
                "L_age": L_age,
                "lnu_age": lnu_age,
                "nion": nion,
                "sfh_grid_lbt_yr": sfh_lbt_grid,
                "sfr_history": sfr_history,
                "log_metallicity_history": log_metallicity_history,
                # Published for downstream (dust two-component attenuation
                # needs the SSP age axis to apply the BC/diffuse split).
                "ssp_ages_yr": ssp_ages_yr,
            }
        )
        return state.with_(sed_intrinsic=sed_intrinsic, derived=new_derived)


def _time_weighted_sfr(
    sfr_history: jnp.ndarray,
    sfh_lbt_grid: jnp.ndarray,
    window_yr: float,
) -> jnp.ndarray:
    r"""Time-weighted average of SFR over the last ``window_yr`` years.

    Mirrors the legacy averaging in ``forward/sed_model.py`` (line ~2600):
    ``<SFR>_T = Σ(SFR_i × Δt_i) / Σ(Δt_i)`` for grid points with
    ``lbt ≤ window_yr``. JIT-friendly via :func:`jnp.where` masking on a
    fixed-shape array (no boolean indexing).

    Parameters
    ----------
    sfr_history : ndarray, shape (n_grid,)
        SFR on the lookback grid [Msun/yr].
    sfh_lbt_grid : ndarray, shape (n_grid,)
        Lookback-time grid [yr], ascending.
    window_yr : float
        Window width [yr]. Bins with ``lbt > window_yr`` are excluded.

    Returns
    -------
    scalar jnp.ndarray
        Time-weighted SFR over the window [Msun/yr]. If no grid points
        fall in the window, returns ``sfr_history[0]`` (the present-day
        SFR) as a sensible fallback.
    """
    bin_widths = jnp.gradient(sfh_lbt_grid)
    in_window = sfh_lbt_grid <= window_yr
    weights = jnp.where(in_window, bin_widths, 0.0)
    weighted_sum = jnp.sum(sfr_history * weights)
    weight_total = jnp.sum(weights)
    return jnp.where(weight_total > 0.0, weighted_sum / weight_total, sfr_history[0])


# ─────────────────────────────────────────────────────────────────────
# JAX pytree registration
# ─────────────────────────────────────────────────────────────────────
#
# Register StellarSEDComponent as a JAX pytree so ``self.ssp_data``
# flows through ``jax.jit`` as a TRACED input rather than being baked
# into the XLA graph as a literal constant. The SSP grid is ~8 MB
# (15 × 93 × 5994 doubles); without this registration the cold-compile
# time of any orchestrator chain that contains StellarSEDComponent
# explodes to ~900 ms because XLA inlines the entire grid as constants
# at every call site. With registration cold-compile drops by an
# order of magnitude. (Mirrors the Phase II-2 commit e52bd75 fix to
# the legacy hybrid kernel path.)
#
# ``ssp_data`` is the only data field (it's a JAX-pytree-compatible
# NamedTuple with ndarray leaves). Everything else is structural
# (config, name, parameter_prefix) → meta.

from jax import tree_util as _tree_util

_tree_util.register_dataclass(
    StellarSEDComponent,
    data_fields=("ssp_data",),
    meta_fields=("config", "name", "parameter_prefix"),
)

del _tree_util
