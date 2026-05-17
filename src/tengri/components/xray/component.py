# SPDX-License-Identifier: BSD-3-Clause
"""XRaySEDComponent: SEDComponent adapter around :func:`xray_total`.

Phase II-1 first-cohort exercise (sibling to
:class:`tengri.components.radio.component.RadioSEDComponent`). The
physics in :mod:`tengri.components.xray.xray` is unchanged; this is
a thin wrapper that satisfies :class:`tengri.core.SEDComponent` so
the orchestrator can run X-ray alongside the other Phase II adapters.

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

from tengri.components.xray.xray import xray_total
from tengri.core.component import (
    DerivedKey,
    ParamDeclaration,
    PipelineState,
    SEDComponentConfig,
    SEDComponentState,
)
from tengri.parameters.priors import Fixed

__all__ = ["XRaySEDComponent", "XRaySEDComponentConfig"]


@dataclass(frozen=True)
class XRaySEDComponentConfig(SEDComponentConfig):
    r"""Frozen knobs for :class:`XRaySEDComponent`.

    Attributes
    ----------
    name : str
        Diagnostic identifier. Default ``"xray"``.
    """

    name: str = "xray"


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
    Initialises ``sed_intrinsic`` from zeros if upstream did not.

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

    def declared_parameters(self) -> list[ParamDeclaration]:
        r"""Free parameters this component owns.

        Mirrors the ``xray_*`` entries already in
        :mod:`tengri.parameters._param_defs` so registration via this
        list and via the legacy registry produce the same priors.
        """
        return [
            ParamDeclaration(
                "xray_gamma_hmxb",
                Fixed(2.0),
                "HMXB photon index [dimensionless]",
            ),
            ParamDeclaration(
                "xray_gamma_lmxb",
                Fixed(1.6),
                "LMXB photon index [dimensionless]",
            ),
            ParamDeclaration(
                "xray_gamma_agn",
                Fixed(1.8),
                "AGN corona photon index [dimensionless]",
            ),
            ParamDeclaration(
                "xray_E_cut",
                Fixed(300.0),
                "AGN corona high-energy cutoff [keV]",
            ),
            ParamDeclaration(
                "xray_alpha_ox",
                Fixed(-1.4),
                "alpha_OX = log10(L_2keV / L_2500) for AGN corona [dimensionless]",
            ),
        ]

    def publishes(self) -> tuple[DerivedKey, ...]:
        """Cross-component derived keys this X-ray component publishes.

        See :func:`tengri.forward.orchestrator.validate_pipeline`.

        Notes
        -----
        X-ray reads ``sfr``, ``log_mstar``, and ``L_agn_bol`` opportunistically
        with documented fallbacks. Those reads are NOT declared in
        :meth:`requires` — only hard dependencies belong there.
        """
        return (
            DerivedKey(
                "sed_xray",
                "erg/s/Hz",
                "X-ray luminosity contribution on pipeline wave grid",
            ),
        )

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
    ) -> XRaySEDComponentState:
        r"""No-op precompute. X-ray is a closed-form function of (λ, params)."""
        del ssp_data, wave_grid
        return XRaySEDComponentState(name=self.name)

    def apply(
        self,
        state: PipelineState,
        params: Mapping[str, jnp.ndarray],
    ) -> PipelineState:
        r"""Add X-ray emission to ``state.sed_intrinsic``.

        Parameters
        ----------
        state : PipelineState
            Must carry rest-frame ``wave`` (Å). If ``sed_intrinsic`` is
            ``None`` it is initialised to zeros of the same shape.
        params : mapping
            Receives ``xray_*`` keys plus the bare ``redshift`` from
            the allowlist. Cross-component scalars (``sfr``,
            ``log_mstar``, ``L_agn_bol``) are read from
            ``state.derived`` with documented fallbacks.

        Returns
        -------
        PipelineState
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

        L_xray = xray_total(
            wave,
            sfr=sfr,
            stellar_mass=stellar_mass,
            L_agn_bol=L_agn_bol,
            gamma_hmxb=jnp.asarray(params["xray_gamma_hmxb"]),
            gamma_lmxb=jnp.asarray(params["xray_gamma_lmxb"]),
            gamma_agn=jnp.asarray(params["xray_gamma_agn"]),
            E_cut=jnp.asarray(params["xray_E_cut"]),
            alpha_ox=jnp.asarray(params["xray_alpha_ox"]),
        )

        if state.sed_intrinsic is None:
            new_sed = L_xray
        else:
            new_sed = state.sed_intrinsic + L_xray

        new_derived = dict(state.derived)
        new_derived["sed_xray"] = L_xray
        return state.with_(sed_intrinsic=new_sed, derived=new_derived)
