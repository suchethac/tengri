# SPDX-License-Identifier: BSD-3-Clause
"""Build orchestrator-compatible component chains from a single call.

Phase II-2.6 public-API helper. Lets users assemble the
``run_components`` chain from a flat set of keyword arguments without
constructing each :class:`SEDComponent` subclass by hand::

    from tengri.forward.component_factory import build_components
    from tengri.forward.orchestrator import run_components
    from tengri.core.component import PipelineState

    components = build_components(
        ssp_data=ssp,
        sfh_model="tsnorm",
        metallicity_model="ramp",
        dust_law_bc="calzetti",
        dust_emission_model="dale2014",
        agn_model="standard",
        use_radio=True, use_xray=True, use_igm=True,
    )
    state = run_components(components, PipelineState(wave=ssp.ssp_wave), params)

This is the **public-facing** orchestrator entry point — independent
of :class:`tengri.SEDModel` (which keeps its legacy tier-dispatch
path). The two paths coexist; users opt into the orchestrator by
calling this helper.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp

from tengri.components.agn.component import AGNSEDComponent, AGNSEDComponentConfig
from tengri.components.dust.two_component import (
    DustSEDComponent,
    DustSEDComponentConfig,
)
from tengri.components.igm.component import IGMSEDComponent
from tengri.components.nebular.component import (
    NebularSEDComponent,
    NebularSEDComponentConfig,
)
from tengri.components.radio.component import RadioSEDComponent
from tengri.components.stellar import StellarSEDComponent
from tengri.components.stellar.component import StellarSEDComponentConfig
from tengri.components.xray.component import XRaySEDComponent
from tengri.core.component import SEDComponent

__all__ = [
    "build_components",
    "chain_summary",
    "state_to_sed_quantities",
    "state_to_sfh_quantities",
]


def build_components(
    *,
    ssp_data: Any,
    # Stellar
    sfh_model: str = "tsnorm",
    field: bool = False,
    metallicity_model: str = "delta",
    n_grid: int = 64,
    lgmet_scatter: float = 0.2,
    # Nebular
    nebular_backend: str | None = "baked_in",
    nebular_backend_instance: Any | None = None,
    # AGN
    agn_model: str | None = None,
    # Dust two-component
    dust_law_bc: str = "power_law",
    dust_law_diff: str = "power_law",
    dust_emission_model: str = "modified_blackbody",
    use_dust: bool = True,
    # Multiwavelength
    use_radio: bool = False,
    use_xray: bool = False,
    use_igm: bool = False,
) -> list[SEDComponent]:
    r"""Construct an ordered :class:`SEDComponent` list for the orchestrator.

    The component order is the **canonical pipeline order**, which any
    orchestrator-driven prediction should follow:

    1. ``StellarSEDComponent`` — emits the stellar SED, publishes
       ``lnu_age``, ``ssp_ages_yr``, ``log_metallicity_history``,
       ``nion``, etc.
    2. ``NebularSEDComponent`` — adds nebular emission to
       ``sed_intrinsic`` (no-op for the BakedIn backend).
    3. ``AGNSEDComponent`` — adds AGN disc + torus + lines and
       publishes ``L_agn_bol``.
    4. ``DustSEDComponent`` — applies two-component attenuation to
       the per-age cube, integrates absorbed luminosity, adds IR
       re-emission, publishes ``L_ir``.
    5. ``RadioSEDComponent`` — synchrotron, reads ``L_ir``,
       ``log_mstar``, ``L_agn_bol`` with documented fallbacks.
    6. ``XRaySEDComponent`` — XRBs + AGN corona, reads ``sfr``,
       ``log_mstar``, ``L_agn_bol``.
    7. ``IGMSEDComponent`` — multiplies ``sed_observed`` by Inoue+2014
       transmission (no-op if no observed-frame SED yet).

    Parameters
    ----------
    ssp_data : SSPData
        Stellar-population templates, required by stellar.
    sfh_model : str
        Registered SFH model — currently ``"tsnorm"`` or ``"dpl"``.
    field : bool
        Add a stochastic GP field on top of the mean SFH.
    metallicity_model : str
        ``"delta"`` (constant Z) or ``"ramp"`` (linear log10(Z) ramp).
    n_grid : int
        SFH lookback-time grid resolution.
    lgmet_scatter : float
        Gaussian σ in log10(Z) for the DSPS triweight kernel [dex].
    nebular_backend : str | None
        ``"baked_in"`` (default), ``"cloudy_grid"``, ``"cue"``, or
        ``None`` to omit nebular entirely.
    nebular_backend_instance : object | None
        Pre-constructed backend object for ``cloudy_grid`` / ``cue``
        (which need HDF5 / weights paths). Required for those backends.
    agn_model : str | None
        AGN model registry key (``"simple"``, ``"standard"``, …) or
        ``None`` to omit AGN.
    dust_law_bc, dust_law_diff : str
        Birth-cloud / diffuse-ISM attenuation-law registry keys.
    dust_emission_model : str
        IR emission template registry key.
    use_dust : bool
        If ``False`` no dust component is added (no attenuation, no IR).
    use_radio, use_xray, use_igm : bool
        Add the corresponding adapter to the chain.

    Returns
    -------
    list[SEDComponent]
        Ordered component list ready to feed
        :func:`tengri.forward.orchestrator.run_components`.

    Notes
    -----
    **JIT-compatible**: yes — the returned components flow through
    ``jax.jit`` once :class:`tengri.core.PipelineState` is registered
    as a pytree (Phase II-2.2-followup).

    The ``StellarSEDComponent`` carries ``ssp_data`` on its instance
    (the most natural plumbing per Phase II-2.2). All other adapters
    are stateless except their config knobs.
    """
    components: list[SEDComponent] = []

    # 1. Stellar (always required — it publishes the cross-component
    #    inputs that every later adapter reads).
    components.append(
        StellarSEDComponent(
            config=StellarSEDComponentConfig(
                sfh_model=sfh_model,
                field=field,
                metallicity_model=metallicity_model,
                n_grid=n_grid,
                lgmet_scatter=lgmet_scatter,
            ),
            ssp_data=ssp_data,
        )
    )

    # 2. Nebular (optional)
    if nebular_backend is not None:
        components.append(
            NebularSEDComponent(
                config=NebularSEDComponentConfig(backend=nebular_backend),
                backend=nebular_backend_instance,
            )
        )

    # 3. AGN (optional)
    if agn_model is not None:
        components.append(
            AGNSEDComponent(config=AGNSEDComponentConfig(model=agn_model))
        )

    # 4. Dust (optional)
    if use_dust:
        components.append(
            DustSEDComponent(
                config=DustSEDComponentConfig(
                    law_bc=dust_law_bc,
                    law_diff=dust_law_diff,
                    emission_model=dust_emission_model,
                )
            )
        )

    # 5-7. Multiwavelength + IGM (each optional)
    if use_radio:
        components.append(RadioSEDComponent())
    if use_xray:
        components.append(XRaySEDComponent())
    if use_igm:
        components.append(IGMSEDComponent())

    return components


def chain_summary(components: Sequence[SEDComponent]) -> str:
    """Pretty-print a component chain for diagnostics.

    Returns a one-line ``→``-separated string of component names.
    Useful for log lines and error messages where the chain identity
    matters more than the parameter values.
    """
    return " → ".join(c.name for c in components)


# ─────────────────────────────────────────────────────────────────────
# PipelineState → legacy Quantities bridges
# ─────────────────────────────────────────────────────────────────────
#
# Build out the legacy ``SFHQuantities`` / ``SEDQuantities`` NamedTuples
# from a fresh :class:`PipelineState`. Lets users with code that
# expects the legacy types swap in the orchestrator path without
# rewriting downstream call sites. Full :class:`Prediction` parity
# (lines, radio, X-ray, ionising-photon properties) is follow-up.


def state_to_sfh_quantities(state: Any):
    """Convert orchestrator :class:`PipelineState` → :class:`SFHQuantities`.

    Pulls the per-galaxy SFH derived quantities published by
    :class:`StellarSEDComponent` (``log_mstar``, ``log_mstar_formed``,
    ``sfr_10myr``, ``sfr_100myr``, ``sfh_grid_lbt_yr``, ``sfr_history``,
    ``log_metallicity_history``) and packages them in the legacy
    NamedTuple shape so existing code reading
    ``predict_sfh_quantities(...).stellar_mass`` keeps working when
    sourced from the orchestrator path.

    Parameters
    ----------
    state : PipelineState
        Output of :func:`run_components` on a chain that includes
        :class:`StellarSEDComponent`.

    Returns
    -------
    SFHQuantities
        ``stellar_mass``, ``stellar_mass_surviving``, ``sfr_100myr``,
        ``sfr_10myr``, ``ssfr``, ``mass_weighted_age_gyr``,
        ``mass_weighted_metallicity``.

    Notes
    -----
    Mass-weighted age and metallicity are computed from
    ``sfh_grid_lbt_yr`` × ``sfr_history`` (mass per bin) and
    ``log_metallicity_history``. ``ssfr`` uses the **surviving** mass
    in the denominator to match the legacy convention.

    **JIT-compatible**: yes — pure JAX.
    """
    from tengri.forward.prediction import SFHQuantities

    derived = state.derived
    log_mstar = jnp.asarray(derived["log_mstar"])
    log_mstar_formed = jnp.asarray(derived["log_mstar_formed"])
    stellar_mass_surviving = jnp.power(10.0, log_mstar)
    stellar_mass = jnp.power(10.0, log_mstar_formed)

    sfh_lbt = jnp.asarray(derived["sfh_grid_lbt_yr"])
    sfr_history = jnp.asarray(derived["sfr_history"])
    log_z_history = jnp.asarray(derived["log_metallicity_history"])

    # Mass per SFH bin (∫ SFR dt locally) — used as weight for the
    # mass-weighted age and metallicity.
    bin_widths = jnp.gradient(sfh_lbt)
    bin_mass = jnp.maximum(sfr_history * bin_widths, 0.0)
    bin_mass_total = jnp.maximum(jnp.sum(bin_mass), 1e-30)
    mw_age_yr = jnp.sum(sfh_lbt * bin_mass) / bin_mass_total
    mw_age_gyr = mw_age_yr / 1e9
    mw_z = jnp.sum(log_z_history * bin_mass) / bin_mass_total

    sfr_100myr = jnp.asarray(derived["sfr_100myr"])
    sfr_10myr = jnp.asarray(derived["sfr_10myr"])
    ssfr = sfr_100myr / jnp.maximum(stellar_mass_surviving, 1e-30)

    return SFHQuantities(
        stellar_mass=stellar_mass,
        stellar_mass_surviving=stellar_mass_surviving,
        sfr_100myr=sfr_100myr,
        sfr_10myr=sfr_10myr,
        ssfr=ssfr,
        mass_weighted_age_gyr=mw_age_gyr,
        mass_weighted_metallicity=mw_z,
    )


def state_to_sed_quantities(state: Any):
    """Convert orchestrator :class:`PipelineState` → :class:`SEDQuantities`.

    Maps the directly-available SED quantities and computes the
    UV/break diagnostics from ``state.sed_intrinsic`` and
    ``state.wave``. Fields that the orchestrator does not yet publish
    or that require luminosity-weighting infrastructure are returned
    as ``NaN`` — callers that need them should keep using
    :meth:`SEDModel.predict_sed_quantities` until the orchestrator
    bridge is extended.

    Parameters
    ----------
    state : PipelineState
        Output of :func:`run_components` on a chain that includes
        :class:`StellarSEDComponent` (and ideally
        :class:`DustSEDComponent` for ``l_tir`` / ``l_dust_absorbed``).

    Returns
    -------
    SEDQuantities
        ``l_bol``, ``l_tir``, ``l_dust_absorbed`` populated;
        UV-slope / Dn4000 / Balmer-break / M_UV / luminosity-weighted
        quantities returned as ``NaN`` (TODO: port the legacy
        :func:`compute_uv_slope_beta` etc. machinery).

    Notes
    -----
    **JIT-compatible**: yes — pure JAX.
    **Units**: bolometric luminosities returned in Lsun (matches the
    legacy NamedTuple convention).
    """
    from tengri.forward.prediction import SEDQuantities
    from tengri.utils.physics_constants import C_AA, L_SUN
    from tengri.utils.sed_quantities import (
        compute_balmer_break,
        compute_dn4000,
        compute_fuv_flux,
        compute_irx,
        compute_m_uv,
        compute_nuv_flux,
        compute_rest_uv_color,
        compute_uv_slope_beta,
    )

    sed = state.sed_intrinsic
    wave = state.wave
    nu = C_AA / wave

    # Bolometric luminosity in Lsun. Wave is ascending, so nu is
    # descending; ``trapezoid(L_nu, nu)`` returns a negative signed
    # area. ``abs(...)`` recovers the positive luminosity.
    l_bol_erg = jnp.abs(jnp.trapezoid(sed, nu))
    l_bol = l_bol_erg / L_SUN

    # Dust-absorbed and IR re-emission luminosities, available when
    # DustSEDComponent ran in the chain.
    derived = state.derived
    l_ir = jnp.asarray(derived.get("L_ir", 0.0))
    l_absorbed = jnp.asarray(derived.get("L_absorbed", 0.0))
    l_tir = l_ir / L_SUN
    l_dust_absorbed = l_absorbed / L_SUN

    # UV / break diagnostics — pull from the legacy helpers, which
    # take ``(sed, wave)`` arrays directly.
    fuv = compute_fuv_flux(sed, wave)
    nuv = compute_nuv_flux(sed, wave)
    irx = compute_irx(l_tir, fuv * 2.998e15 / 1500.0)  # νLν at 1500 Å in erg/s

    # Pre-dust stellar SED reconstructed from the per-age cube
    # ``lnu_age``, which StellarSEDComponent publishes before
    # DustSEDComponent overwrites ``sed_intrinsic``. If no per-age
    # cube is present (chain has no stellar component), fall back to
    # NaN.
    nan_scalar = jnp.asarray(jnp.nan)
    if "lnu_age" in derived:
        sed_stellar_intrinsic = jnp.sum(jnp.asarray(derived["lnu_age"]), axis=0)
        fuv_intrinsic = compute_fuv_flux(sed_stellar_intrinsic, wave)
        nuv_intrinsic = compute_nuv_flux(sed_stellar_intrinsic, wave)
    else:
        fuv_intrinsic = nan_scalar
        nuv_intrinsic = nan_scalar

    return SEDQuantities(
        l_bol=l_bol,
        l_tir=l_tir,
        l_dust_absorbed=l_dust_absorbed,
        irx=irx,
        uv_slope_beta=compute_uv_slope_beta(sed, wave),
        dn4000=compute_dn4000(sed, wave),
        balmer_break=compute_balmer_break(sed, wave),
        m_uv=compute_m_uv(sed, wave),
        fuv_flux=fuv,
        nuv_flux=nuv,
        fuv_flux_intrinsic=fuv_intrinsic,
        nuv_flux_intrinsic=nuv_intrinsic,
        rest_uv_color=compute_rest_uv_color(sed, wave),
        # Luminosity-weighted age/metallicity require the per-age
        # cube ``lnu_age`` AND the SSP age axis; future bridge step.
        luminosity_weighted_age_gyr=nan_scalar,
        luminosity_weighted_metallicity=nan_scalar,
    )
