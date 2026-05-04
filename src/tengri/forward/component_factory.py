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

__all__ = ["build_components"]


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
